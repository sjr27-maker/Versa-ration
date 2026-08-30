"""HypothesisGenerator: a speculative, regenerated-each-turn tree of
grounded predictions about the current student — distinct from
HypothesisStore's durable, evidence-backed belief (CLAUDE.md invariant
1). See CLAUDE.md invariant 6 for the append-only `branches` store
this writes to.

`generate()` builds the tree; `resolve()` matches the student's next
real turn against it and closes the generation out. Neither writes
into HypothesisStore: a single-turn branch match is a weak, noisy
signal, and the tree is mostly discarded every turn — writing it
straight into the durable, audited store would inject that noise into
invariant 1's record of *confirmed* belief. What "a pattern repeats
across turns/sessions enough to promote" should mean is left for a
future consolidation step once real match data exists.

Not a Node itself — it has two entrypoints (generate/resolve), which
doesn't fit `SessionLoop._call_node`'s one-`run()`-per-node contract.
`BranchGenerate`/`BranchResolve` below are the two thin Node wrappers
that do, each delegating to a shared `HypothesisGenerator` instance.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import re
from uuid import UUID

from probe.branches import BranchStore
from probe.llm import LLMClient
from probe.models import (
    Branch,
    BranchGeneration,
    BranchSelection,
    BranchStatus,
    CandidateAction,
    EvidenceCheckResult,
    ExplicitRequest,
    Hypothesis,
    OptionProposal,
    PathRequirement,
    ResolutionResult,
)
from probe.options import OptionStore
from probe.reasoning_budget import (
    BranchBudget,
    BranchBudgetConfig,
    compute_branch_budget,
)

# GENERATE:OPTIONS re-ask budget on a rejected mapping (duplicate
# branch id, or a branch id outside the live set) — one corrective
# retry, same shape as Plan's _MAX_PROPOSE_ATTEMPTS.
_MAX_OPTIONS_ATTEMPTS = 2

logger = logging.getLogger(__name__)


def _is_redundant_with_siblings(
    statement: str, sibling_statements: list[str], threshold: float
) -> bool:
    return any(
        difflib.SequenceMatcher(None, statement, sib).ratio() >= threshold
        for sib in sibling_statements
    )


def should_expand_branch(
    plausibility: float,
    statement: str,
    sibling_statements: list[str],
    depth: int,
    branches_so_far: int,
    budget: BranchBudget,
    requires_evidence: str | None = None,
    evidence_satisfied: bool = False,
) -> bool:
    """The "worth expanding to the next layer" filter: (a) its
    requires_evidence, if any, is satisfied, (b) plausible enough,
    (c) distinguishes from siblings rather than restating them, (d) the
    turn's budget still permits it.

    Gate (a) is a hard block, not a plausibility adjustment: a branch
    can be arbitrarily plausible and still not expand, because
    expansion now depends on the student, not just the model's own
    confidence (see Branch.requires_evidence). A branch that fails only
    this gate is not pruned or superseded — it holds at its current
    depth, unresolved, until a click or a typed match satisfies it (see
    BranchStore.set_evidence_satisfied), then expands on a later turn
    exactly as if it had never required evidence.

    No LLM call for (c): a text-similarity heuristic is enough since
    near-duplicate branches from the same generation call tend to be
    near-duplicate text, and an extra call per decision would defeat
    the point of budgeting call count at all. This catches wording-
    level duplication, not semantic duplication — every branch that
    clears it is logged (see generate()) specifically so a human can
    judge from real session data whether survivors are genuinely
    distinct bets or rephrasings of one idea.
    """
    if requires_evidence is not None and not evidence_satisfied:
        return False
    if depth + 1 > budget.max_depth:
        return False
    if branches_so_far >= budget.max_total_branches:
        return False
    if plausibility < budget.expand_plausibility_threshold:
        return False
    return not _is_redundant_with_siblings(
        statement, sibling_statements, budget.redundancy_similarity_threshold
    )


def _clamp01(value: object) -> float:
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def _parse_branch_items(raw: str) -> list[dict]:
    """Shared parser for both a root-wave response (bare JSON list) and
    an expansion response's `children` list. Malformed/missing fields
    are skipped, not fatal — same discipline as Infer/Plan's parsing.

    requires_evidence is optional and nullable: an empty string or
    missing key both normalize to None (a branch that needs nothing
    further), same as a JSON null would."""
    if not isinstance(raw, list):
        return []
    items: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        statement = entry.get("statement")
        predicted = entry.get("predicted_next_turn")
        if not statement or not predicted:
            continue
        requires_evidence = entry.get("requires_evidence")
        items.append(
            {
                "statement": str(statement),
                "plausibility": _clamp01(entry.get("plausibility", 0.0)),
                "predicted_next_turn": str(predicted),
                "requires_evidence": (
                    str(requires_evidence) if requires_evidence else None
                ),
            }
        )
    return items


def _parse_intent_response(raw: str) -> list[dict]:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return _parse_branch_items(parsed)


def _parse_expand_response(raw: str) -> tuple[str, list[dict]]:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return "", []
    if not isinstance(parsed, dict):
        return "", []
    items = _parse_branch_items(parsed.get("children"))
    if not items:
        return "", []
    label = parsed.get("layer_label")
    return (str(label) if label else "unlabeled"), items


def _parse_resolve_response(raw: str, valid_ids: set[UUID]) -> UUID | None:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    raw_id = parsed.get("matched_branch_id")
    if raw_id is None:
        return None
    try:
        candidate = UUID(str(raw_id))
    except ValueError:
        return None
    if candidate not in valid_ids:
        # Hallucinated id — same validation discipline as GroundConcept's
        # rejection of a concept_id not present in the session's graph:
        # reject rather than pass through.
        return None
    return candidate


def _intent_prompt(
    transcript_context: str,
    hypotheses: list[Hypothesis],
    action: CandidateAction,
    budget: BranchBudget,
) -> str:
    lo, hi = budget.root_branch_range
    hyp_listing = (
        "\n".join(
            f"- [{h.layer.value}] {h.statement} (p={h.probability:.2f})"
            for h in hypotheses
        )
        or "(no active hypotheses yet)"
    )
    action_desc = action.action.value
    if action.target_concept:
        action_desc += f" (target concept: {action.target_concept})"
    return (
        "GENERATE:INTENT\n"
        f"session context:\n{transcript_context}\n\n"
        f"current hypotheses about the student:\n{hyp_listing}\n\n"
        f"the tutor is about to take this action: {action_desc}, "
        f"rationale: {action.rationale}\n\n"
        f"Given all of the above, propose between {lo} and {hi} distinct, "
        "plausible intents for why the student sent their last message, "
        "and how they are likely to react once the tutor's planned action "
        "happens — genuinely different bets, not rephrasings of one "
        "idea.\n\n"
        "For each, also state requires_evidence: what would have to be "
        "true about the student — in their own words or actions, not "
        "your own confidence — for this branch to be worth exploring "
        "deeper. Use null when the branch needs nothing further and can "
        "expand on plausibility alone.\n\n"
        'Respond with JSON: [{"statement": "...", "plausibility": 0.0-1.0, '
        '"predicted_next_turn": "a concrete, checkable prediction of what '
        'the student will say or do next if this intent is true", '
        '"requires_evidence": "..." or null}, ...]'
    )


def _expand_prompt(parent: Branch, budget: BranchBudget) -> str:
    lo, hi = budget.children_per_branch_range
    return (
        "GENERATE:EXPAND\n"
        f"Parent (depth {parent.depth}, {parent.depth_label}): {parent.statement}\n\n"
        f"Given this parent holds true, propose between {lo} and {hi} more "
        "specific, plausible follow-ons that condition on it — moving from "
        "intent toward the student's underlying knowledge state, or toward a "
        "concrete predicted action, whichever is more specific than the "
        'parent. Also name a short label for what this new layer is '
        'branching (e.g. "knowledge_gap", "predicted_action", or whatever '
        "fits better here).\n\n"
        "For each child, also state requires_evidence: what would have "
        "to be true about the student for that child to be worth "
        "exploring deeper still, or null if nothing further is "
        "needed.\n\n"
        'Respond with JSON: {"layer_label": "...", "children": '
        '[{"statement": "...", "plausibility": 0.0-1.0, "predicted_next_turn": '
        '"a concrete, checkable prediction of what the student will say or '
        'do next if this branch is true", "requires_evidence": "..." or '
        'null}, ...]}'
    )


def build_branch_path(branches: list[Branch], selected_id: UUID) -> list[Branch]:
    """Root-to-leaf inclusive chain ending at `selected_id`, built from
    an in-memory branch list (a just-generated tree already holds every
    ancestor of anything in it) rather than round-tripping to the DB —
    depth 0's intent, every intermediate layer, and the selected branch
    itself, in that order. `selected_id` need not be an actual leaf:
    SelectBranch can pick any branch in the tree."""
    by_id = {b.id: b for b in branches}
    chain: list[Branch] = []
    current: Branch | None = by_id.get(selected_id)
    while current is not None:
        chain.append(current)
        current = by_id.get(current.parent_id) if current.parent_id else None
    chain.reverse()
    return chain


def _select_prompt(branches: list[Branch]) -> str:
    listing = "\n".join(
        f"- id={b.id} depth={b.depth} parent={b.parent_id} "
        f"plausibility={b.plausibility:.2f} [{b.depth_label}]: {b.statement}"
        for b in branches
    )
    return (
        "SELECT:BRANCH\n"
        "Below is this turn's full generated tree of plausible student "
        "intents, knowledge gaps, and predicted reactions.\n\n"
        f"{listing}\n\n"
        "Select ONE branch to teach toward this turn. The question is "
        "COVERAGE, not likelihood: pick the branch whose path, if taught "
        "to, would also serve the largest share of the OTHER live "
        "branches — not necessarily the single most probable one. A "
        "branch at plausibility 0.6 that covers ground shared by four "
        "siblings beats one at plausibility 0.85 that only serves "
        "itself. State which other branches your choice covers and "
        "why.\n"
        'Respond with JSON: {"selected_branch_id": "<id>", '
        '"rationale": "..."}'
    )


def _parse_select_response(
    raw: str, valid_ids: set[UUID]
) -> tuple[UUID | None, str]:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None, "<unparseable response>"
    if not isinstance(parsed, dict):
        return None, "<response was not a JSON object>"
    rationale = str(parsed.get("rationale") or "")
    raw_id = parsed.get("selected_branch_id")
    if raw_id is None:
        return None, rationale
    try:
        candidate = UUID(str(raw_id))
    except ValueError:
        return None, rationale or "<selected_branch_id not a valid UUID>"
    if candidate not in valid_ids:
        # Hallucinated id — same validation discipline as
        # _parse_resolve_response: reject rather than pass through.
        return None, rationale or "<selected_branch_id not in this generation>"
    return candidate, rationale


def _derive_prompt(
    path: list[Branch],
    student_message: str,
    action_rationale: str,
    explicit_request: ExplicitRequest | None = None,
) -> str:
    listing = "\n".join(
        f"- depth={b.depth} [{b.depth_label}]: {b.statement} "
        f"(predicted future reaction, NOT YET SAID: {b.predicted_next_turn})"
        for b in path
    )
    explicit_request_block = ""
    if explicit_request is not None and explicit_request.present and explicit_request.what:
        explicit_request_block = (
            "\nThe student made an explicit, concrete request this turn "
            f"that scope MUST include, not replace: {explicit_request.what!r}. "
            "You may add framing around it, but the scope you return "
            "must still name and resolve this specific request — do "
            "not substitute a different example or topic.\n"
        )
    return (
        "DERIVE:PATH\n"
        "Below is the full root-to-leaf path this turn's teaching should "
        "be derived from, from the student's root-level intent down to "
        "the most specific selected branch.\n\n"
        f"{listing}\n\n"
        "The student's actual message, verbatim, is the only thing they "
        f"have actually said so far:\n{student_message}\n\n"
        "The tutor is considering teaching this, but has NOT taught it "
        f"yet — the student has no knowledge of it: {action_rationale}\n"
        f"{explicit_request_block}\n"
        "IMPORTANT — read this carefully before answering. Each "
        "\"predicted future reaction\" above is a HYPOTHETICAL guess "
        "about what the student MIGHT say LATER, after being taught "
        "something they have not been taught yet. It has NOT happened. "
        "It is not evidence of the student's current state, only a "
        "forecast. Likewise, the tutor's own not-yet-taught idea above "
        "is not something the student could already believe anything "
        "about — they cannot hold a belief about an analogy or "
        "explanation they have never heard.\n\n"
        "current_belief must be built ONLY from what is actually known: "
        "the student's real message above, the root intent's own "
        "statement, and any ancestor branch statement that describes the "
        "student's EXISTING state (not a predicted reaction, and not the "
        "tutor's own not-yet-taught idea). If none of that actually "
        "supports a specific belief claim, say so plainly — e.g. "
        "\"insufficient evidence to characterize current belief beyond "
        "the root intent\" — rather than manufacturing one. A true "
        "\"not enough evidence\" is a correct, useful answer here, not a "
        "failure to avoid.\n\n"
        "Beyond that: be precise about what the path actually implies "
        "versus what is merely possible — do not invent specifics "
        "(values, signs, conditions) the path does not state.\n"
        'Respond with JSON: {"current_belief": "...", "needed": "what '
        'they must be given to move along this path", "must_not_assume": '
        '["...", ...] (things this path leaves genuinely uncertain that '
        'must NOT be stated as settled), "scope": "the concrete scope of '
        'this turn of teaching — one thing, not a syllabus"}'
    )


def _parse_path_requirement(raw: str) -> PathRequirement:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    must_not_assume = parsed.get("must_not_assume")
    return PathRequirement(
        current_belief=str(parsed.get("current_belief") or ""),
        needed=str(parsed.get("needed") or ""),
        must_not_assume=(
            [str(x) for x in must_not_assume]
            if isinstance(must_not_assume, list)
            else []
        ),
        scope=str(parsed.get("scope") or ""),
    )


def _enforce_explicit_request_scope(
    path_requirement: PathRequirement, explicit_request: ExplicitRequest
) -> PathRequirement:
    """Structural backstop, not just a prompt ask: _derive_prompt above
    already instructs the model to include the explicit request in
    scope, but prompts drift or get overridden by everything else in
    the prompt competing for attention — this guarantees it at the
    code level instead of hoping. Only ever adds to whatever framing
    DerivePath's own scope already produced; never removes it."""
    if not explicit_request.what:
        return path_requirement
    if explicit_request.what.lower() in path_requirement.scope.lower():
        return path_requirement
    combined_scope = (
        f"{explicit_request.what} — {path_requirement.scope}"
        if path_requirement.scope
        else explicit_request.what
    )
    return path_requirement.model_copy(update={"scope": combined_scope})


def _resolve_prompt(actual_turn_text: str, leaves: list[Branch]) -> str:
    listing = "\n".join(f"- {b.id}: {b.predicted_next_turn}" for b in leaves) or (
        "(no live predictions)"
    )
    return (
        "RESOLVE:MATCH\n"
        f"student's actual next message: {actual_turn_text}\n\n"
        f"candidate predictions:\n{listing}\n\n"
        "Does this response clearly match one of these predictions? Do not "
        "force a match — if none of them genuinely fits, say so rather than "
        "picking the closest one; a real \"no match\" is an expected, useful "
        "outcome, not a failure to avoid.\n"
        'Respond with JSON: {"matched_branch_id": "<id>" or null, '
        '"confidence": 0.0-1.0}'
    )


class HypothesisGenerator:
    def __init__(
        self,
        llm: LLMClient,
        branch_store: BranchStore,
        budget_config: BranchBudgetConfig | None = None,
        option_store: OptionStore | None = None,
    ) -> None:
        self._llm = llm
        self._branches = branch_store
        self._budget_config = budget_config
        # Opt-in on top of branch_store's own opt-in: option_store=None
        # (the default) reproduces the tree without the button channel
        # — resolve() simply skips option supersession. See
        # GenerateOptions/CheckEvidence for the rest of this feature.
        self._options = option_store

    async def generate(
        self,
        session_id: UUID,
        turn_index: int,
        transcript_context: str,
        hypotheses: list[Hypothesis],
        action: CandidateAction,
    ) -> BranchGeneration:
        # `hypotheses` is caller-supplied (SessionLoop's own
        # refreshed_hypotheses, already learner-scoped via
        # list_by_learner — see the same fix on Replan's entropy input)
        # rather than fetched again here: generate() now runs after
        # Plan in the turn, by which point the loop already has this
        # exact list in hand, so there is no second DB round-trip and
        # no risk of it drifting from what Plan itself just scored
        # against. `action` is Plan's winning CandidateAction — what
        # the tutor is about to do, the substitute signal for "what's
        # about to happen" now that generation runs before Teach and
        # can no longer condition on Teach's rendered text.
        budget = compute_branch_budget(hypotheses, self._budget_config)
        call_count = 0

        raw = await self._llm.complete(
            _intent_prompt(transcript_context, hypotheses, action, budget)
        )
        call_count += 1
        root_items = _parse_intent_response(raw)

        generation_meta = await self._branches.create_generation(
            session_id, turn_index, len(root_items)
        )

        all_branches: list[Branch] = []
        redundancy_notes: list[str] = []
        wave: list[Branch] = []
        for item in root_items:
            branch = Branch(
                parent_id=None,
                generation_id=generation_meta.id,
                session_id=session_id,
                turn_index=turn_index,
                depth=0,
                depth_label="intent",
                statement=item["statement"],
                predicted_next_turn=item["predicted_next_turn"],
                requires_evidence=item["requires_evidence"],
                plausibility=item["plausibility"],
                is_leaf=True,  # provisional; flipped False below if expanded
            )
            wave.append(branch)
            all_branches.append(branch)

        depth = 0
        while wave:
            to_expand: list[Branch] = []
            for branch in wave:
                # True tree-siblings only: same parent_id (None groups
                # all depth-0 root branches together). Different parents
                # at the same depth are cousins, not siblings — their
                # children shouldn't prune each other just because two
                # unrelated parents happened to produce similar-sounding
                # children.
                siblings = [
                    o.statement
                    for o in wave
                    if o.id != branch.id and o.parent_id == branch.parent_id
                ]
                survives = should_expand_branch(
                    branch.plausibility,
                    branch.statement,
                    siblings,
                    depth,
                    len(all_branches),
                    budget,
                    requires_evidence=branch.requires_evidence,
                    evidence_satisfied=branch.evidence_satisfied,
                )
                if survives:
                    note = (
                        f"branch {branch.id} (depth={depth}, "
                        f"plausibility={branch.plausibility:.2f}) cleared the "
                        f"redundancy check against siblings {siblings!r}"
                    )
                    logger.info("hypothesis_generator: %s", note)
                    redundancy_notes.append(note)
                    to_expand.append(branch)

            next_wave: list[Branch] = []
            # Siblings at this depth are independent of each other — the
            # whole wave's expansion calls fire concurrently rather than
            # one branch at a time. The only place the sequential version
            # short-circuited mid-wave was the max_total_branches ceiling
            # check between parents; that's now checked once before the
            # wave (skip the wave entirely if already at cap) rather than
            # between each parent's call. Results/child-trimming below
            # still enforce the exact same cap on what gets stored — the
            # only behavioral difference is that a wave straddling the
            # cap may fire a few more LLM calls than strictly needed
            # before trimming, never more *branches* than before.
            if to_expand and len(all_branches) < budget.max_total_branches:
                raw_results = await asyncio.gather(
                    *(
                        self._llm.complete(_expand_prompt(parent, budget))
                        for parent in to_expand
                    )
                )
                call_count += len(raw_results)
                for parent, raw in zip(to_expand, raw_results, strict=True):
                    layer_label, child_items = _parse_expand_response(raw)
                    if not child_items:
                        # Expansion produced nothing usable — parent stays
                        # a leaf; it already has its own predicted_next_turn.
                        continue
                    parent.is_leaf = False
                    for item in child_items:
                        if len(all_branches) >= budget.max_total_branches:
                            break
                        child = Branch(
                            parent_id=parent.id,
                            generation_id=generation_meta.id,
                            session_id=session_id,
                            turn_index=turn_index,
                            depth=depth + 1,
                            depth_label=layer_label,
                            statement=item["statement"],
                            predicted_next_turn=item["predicted_next_turn"],
                            requires_evidence=item["requires_evidence"],
                            plausibility=item["plausibility"],
                            is_leaf=True,
                        )
                        next_wave.append(child)
                        all_branches.append(child)

            wave = next_wave
            depth += 1

        await self._branches.add_branches(all_branches)

        return BranchGeneration(
            generation=generation_meta,
            branches=all_branches,
            call_count=call_count,
            redundancy_notes=redundancy_notes,
        )

    async def _mark_matched_chain(
        self, branch_id: UUID, matched_via: str
    ) -> list[UUID]:
        """Marks `branch_id` and its full ancestor chain matched, all
        via the same channel — shared by both the click path and the
        text-match path below so a branch ends up in an identical
        shape regardless of which one resolved it, aside from
        matched_via itself."""
        matched_chain = [branch_id]
        await self._branches.set_matched(branch_id, matched_via)
        for ancestor in await self._branches.get_ancestors(branch_id):
            await self._branches.set_matched(ancestor.id, matched_via)
            matched_chain.append(ancestor.id)
        return matched_chain

    async def resolve(
        self,
        session_id: UUID,
        turn_index: int,
        actual_turn_text: str,
        clicked_branch_id: UUID | None = None,
    ) -> ResolutionResult:
        """`clicked_branch_id` is set only when this turn originated
        from an option click (see SessionLoop.handle_turn) — a click
        is settled evidence, categorically more certain than a text
        match, so it bypasses RESOLVE:MATCH's fuzzy LLM judgment
        entirely rather than letting an uncertain step override a
        known fact. The clicked branch is marked matched directly
        (source="option_click", call_count=0); its non-clicked
        siblings were never tested by this turn at all (the student
        chose one path, they did not reject the others) so they're
        superseded, not marked unmatched — the same treatment a text
        match's own non-matching siblings already get below.
        """
        generation = await self._branches.get_latest_generation(session_id)
        if generation is None:
            return ResolutionResult(
                session_id=session_id,
                turn_index=turn_index,
                matched_branch_id=None,
                status="unmatched",
                call_count=0,
            )
        if self._options is not None:
            # Whatever's still open (never clicked) is done regardless
            # of how this generation's leaf-matching turns out below —
            # an option already `selected` (set at click time, before
            # this ever runs — see SessionLoop.handle_turn) is left
            # untouched by this blanket close-out.
            await self._options.supersede_open_options(generation.id)

        leaves = await self._branches.get_open_leaves(generation.id)

        # A branch whose evidence was satisfied this turn by some
        # OTHER channel than the one resolving this call (e.g. a
        # CheckEvidence match on a different branch than the one this
        # resolve is about) is not "unmatched" and must not be
        # superseded either: it stays open, unpruned. Leaf-prediction
        # matching / a click and evidence satisfaction are independent
        # questions about a branch, so this exclusion is additive to
        # matched_chain, not a replacement for it.
        evidence_satisfied_ids = {b.id for b in leaves if b.evidence_satisfied}

        if clicked_branch_id is not None:
            matched_chain = await self._mark_matched_chain(
                clicked_branch_id, "option_click"
            )
            exclude = matched_chain + list(evidence_satisfied_ids - set(matched_chain))
            await self._branches.supersede_open_branches(generation.id, exclude)
            return ResolutionResult(
                session_id=session_id,
                turn_index=turn_index,
                matched_branch_id=clicked_branch_id,
                matched_chain=matched_chain,
                status="matched",
                source="option_click",
                call_count=0,
            )

        if not leaves:
            return ResolutionResult(
                session_id=session_id,
                turn_index=turn_index,
                matched_branch_id=None,
                status="unmatched",
                call_count=0,
            )

        raw = await self._llm.complete(_resolve_prompt(actual_turn_text, leaves))
        call_count = 1
        matched_id = _parse_resolve_response(raw, {b.id for b in leaves})

        matched_chain = []
        if matched_id is not None:
            matched_chain = await self._mark_matched_chain(matched_id, "text_match")
            status = "matched"
            exclude = matched_chain + list(evidence_satisfied_ids - set(matched_chain))
        else:
            for leaf in leaves:
                if leaf.id in evidence_satisfied_ids:
                    continue
                await self._branches.set_status(leaf.id, BranchStatus.UNMATCHED)
            logger.warning(
                "hypothesis_generator: no leaf branch from generation %s "
                "(session=%s, turn=%d) matched the student's actual next "
                "turn — %d prediction(s) all missed",
                generation.id,
                session_id,
                turn_index,
                len(leaves),
            )
            status = "unmatched"
            exclude = [leaf.id for leaf in leaves if leaf.id not in evidence_satisfied_ids]
            exclude.extend(evidence_satisfied_ids)

        # Close the generation out completely: any branch still `open`
        # (intermediate depths that were never evaluated as leaves, or
        # a leaf whose evidence was satisfied this turn) becomes
        # `superseded`, so nothing from a resolved generation lingers
        # as open *except* what was deliberately excluded above.
        await self._branches.supersede_open_branches(generation.id, exclude)

        return ResolutionResult(
            session_id=session_id,
            turn_index=turn_index,
            matched_branch_id=matched_id,
            matched_chain=matched_chain,
            status=status,
            source="text_match",
            call_count=call_count,
        )


class BranchGenerate:
    """Thin Node wrapper around HypothesisGenerator.generate() — exists
    only so it fits SessionLoop._call_node's node.run(**kwargs) contract
    (CLAUDE.md invariant 2) without changing that contract's shape for
    every other node call site."""

    def __init__(self, generator: HypothesisGenerator) -> None:
        self._generator = generator

    async def run(
        self,
        session_id: UUID,
        turn_index: int,
        transcript_context: str,
        hypotheses: list[Hypothesis],
        action: CandidateAction,
    ) -> BranchGeneration:
        return await self._generator.generate(
            session_id, turn_index, transcript_context, hypotheses, action
        )


class BranchResolve:
    def __init__(self, generator: HypothesisGenerator) -> None:
        self._generator = generator

    async def run(
        self,
        session_id: UUID,
        turn_index: int,
        actual_turn_text: str,
        clicked_branch_id: UUID | None = None,
    ) -> ResolutionResult:
        return await self._generator.resolve(
            session_id, turn_index, actual_turn_text, clicked_branch_id
        )


class SelectBranch:
    """Picks one branch from a just-generated tree for this turn's
    teaching to derive from. Not a HypothesisGenerator method (unlike
    generate()/resolve(), it needs no BranchStore access — it only
    reasons over the branch list already returned by BranchGenerate) —
    a standalone Node like Teach/Plan, fast tier.

    A parse failure or missing selection falls back to the highest-
    plausibility branch, deterministically and without a further LLM
    call — same discipline as Plan's _backfill: DerivePath always needs
    *something* to build a path from, so "nothing selected" is not an
    acceptable terminal outcome the way "no match" is for resolve().
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self.last_call_count: int = 0

    async def run(self, branches: list[Branch]) -> BranchSelection:
        self.last_call_count = 0
        if not branches:
            return BranchSelection(
                selected_branch_id=None, rationale="no branches generated this turn"
            )
        raw = await self._llm.complete(_select_prompt(branches))
        self.last_call_count += 1
        selected_id, rationale = _parse_select_response(
            raw, {b.id for b in branches}
        )
        if selected_id is not None:
            return BranchSelection(selected_branch_id=selected_id, rationale=rationale)
        fallback = max(branches, key=lambda b: b.plausibility)
        return BranchSelection(
            selected_branch_id=fallback.id,
            rationale=(
                "fallback: no valid selection from the model "
                f"({rationale or 'no rationale given'}) — defaulted to the "
                "highest-plausibility branch"
            ),
        )


def _options_prompt(candidates: list[Branch], rejected_reason: str = "") -> str:
    listing = "\n".join(
        f'- id={b.id} plausibility={b.plausibility:.2f}: needs evidence that '
        f'"{b.requires_evidence}" (this bet: "{b.statement}")'
        for b in candidates
    )
    hi = min(4, len(candidates))
    correction = ""
    if rejected_reason:
        correction = (
            f"\nYour previous attempt was rejected: {rejected_reason}. Every "
            "option must map to a DIFFERENT branch id from the list below, "
            "and every branch id used must be one of the ids listed.\n"
        )
    return (
        "GENERATE:OPTIONS\n"
        "Each of the following branches needs a specific piece of "
        f"evidence before it is worth exploring further:\n{listing}\n\n"
        f"Propose between 1 and {hi} clickable options — buttons the "
        "student can select instead of typing. Each option must map to "
        "exactly ONE of the branch ids above and must be a claim that, "
        "if the student affirms it, resolves that branch's stated "
        "evidence requirement.\n\n"
        "Hard rules, each with the reason it exists — this is not "
        "stylistic, violating any of these breaks the mechanism:\n"
        "- Exactly one branch per option. If an option could satisfy "
        "two branches, clicking it would not tell you which one was "
        "true — you would have reintroduced the ambiguity the button "
        "exists to remove.\n"
        "- Exactly one claim per option. No bundling two facts into one "
        "button (e.g. \"I'd check the sign first and then the "
        "magnitude\" is two claims wearing one button; a student who "
        "agrees with half of it produces a corrupt signal).\n"
        "- The option must be answerable about the MATERIAL, not about "
        "the student's own cognition or preference. A student can "
        "reliably tell you which step they would take next; they "
        "cannot reliably tell you how they learn. NEVER generate "
        "something like \"do you prefer diagrams or equations\" — that "
        "is the exact failure mode this must avoid.\n"
        "- The revealing is indirect: write it as a genuine question "
        "about the subject, in the voice of a tutor continuing the "
        "lesson, not a survey question about the student. If a student "
        "could tell they are being profiled by reading it, it is "
        "written wrong.\n"
        "- It should read like the natural next thing a good tutor "
        "would ask — a pause point, not an interruption.\n"
        f"{correction}"
        'Respond with JSON: [{"branch_id": "<id>", "text": "..."}, ...]'
    )


def _parse_options_response(
    raw: str, valid_ids: set[UUID]
) -> list[OptionProposal] | None:
    """None means "reject the whole response, regenerate" — a single
    duplicate or invalid mapping invalidates the batch rather than
    being silently dropped (see GenerateOptions' docstring: no partial
    mapping is acceptable, since every option's legibility depends on
    every OTHER option in the same batch also being clean). An empty
    list is different from a malformed one: the model explicitly
    offering no options is a valid, accepted outcome (same
    "don't force it" discipline as RESOLVE:MATCH's null), not a parse
    failure that should burn a retry."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, list):
        return None
    if not parsed:
        return []
    seen_branch_ids: set[UUID] = set()
    proposals: list[OptionProposal] = []
    for item in parsed:
        if not isinstance(item, dict):
            return None
        text = item.get("text")
        raw_id = item.get("branch_id")
        if not text or raw_id is None:
            return None
        try:
            branch_id = UUID(str(raw_id))
        except ValueError:
            return None
        if branch_id not in valid_ids or branch_id in seen_branch_ids:
            return None
        seen_branch_ids.add(branch_id)
        proposals.append(OptionProposal(branch_id=branch_id, text=str(text)))
    return proposals


def _check_evidence_prompt(actual_turn_text: str, candidates: list[Branch]) -> str:
    listing = "\n".join(
        f"- id={b.id}: {b.requires_evidence}" for b in candidates
    ) or "(nothing pending)"
    return (
        "CHECK:EVIDENCE\n"
        f"student's actual message: {actual_turn_text}\n\n"
        f"pending evidence requirements:\n{listing}\n\n"
        "Does this message clearly establish that one of these "
        "requirements is now true about the student? Do not force a "
        "match — a real \"none of these\" is an expected, useful "
        "outcome, not a failure to avoid.\n"
        'Respond with JSON: {"satisfied_branch_id": "<id>" or null, '
        '"confidence": 0.0-1.0}'
    )


def _parse_evidence_check_response(raw: str, valid_ids: set[UUID]) -> UUID | None:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    raw_id = parsed.get("satisfied_branch_id")
    if raw_id is None:
        return None
    try:
        candidate = UUID(str(raw_id))
    except ValueError:
        return None
    if candidate not in valid_ids:
        return None
    return candidate


class GenerateOptions:
    """Turns live branches with an unsatisfied requires_evidence into
    2-4 clickable options — the second evidence channel, with the
    interpretation step removed (see this module's callers / the
    feature's own design notes for the full rationale). Fast tier.

    Skipped entirely (no LLM call) when no branch needs evidence this
    turn — nothing to ask about. A response with any duplicate or
    invalid branch mapping is rejected wholesale and regenerated once
    (_MAX_OPTIONS_ATTEMPTS); if it still fails, this turn simply shows
    no options rather than a corrupt (ambiguous) mapping — "no options"
    degrades gracefully, an ambiguous one does not.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self.last_call_count: int = 0

    async def run(self, branches: list[Branch]) -> list[OptionProposal]:
        self.last_call_count = 0
        candidates = sorted(
            (b for b in branches if b.requires_evidence and not b.evidence_satisfied),
            key=lambda b: -b.plausibility,
        )
        if not candidates:
            return []
        valid_ids = {b.id for b in candidates}
        rejected_reason = ""
        for _ in range(_MAX_OPTIONS_ATTEMPTS):
            raw = await self._llm.complete(_options_prompt(candidates, rejected_reason))
            self.last_call_count += 1
            proposals = _parse_options_response(raw, valid_ids)
            if proposals is not None:
                return proposals
            rejected_reason = "duplicate branch id, or a branch id not in the live set"
        logger.warning(
            "GenerateOptions: exhausted %d attempt(s) with only invalid "
            "mappings — showing no options this turn rather than an "
            "ambiguous one",
            _MAX_OPTIONS_ATTEMPTS,
        )
        return []


class CheckEvidence:
    """The typed-path counterpart to a button click: does the
    student's typed message establish one of the prior generation's
    still-pending evidence requirements? Unlike a click this still
    requires interpretation — the whole reason options exist is to
    avoid that for the common case — but once judged satisfied, a
    typed match is treated identically to a click from that point on
    (same BranchStore.set_evidence_satisfied call). Fast tier, skipped
    entirely when there is nothing pending to check against.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self.last_call_count: int = 0

    async def run(
        self, actual_turn_text: str, candidates: list[Branch]
    ) -> EvidenceCheckResult:
        self.last_call_count = 0
        if not candidates:
            return EvidenceCheckResult(satisfied_branch_id=None)
        raw = await self._llm.complete(
            _check_evidence_prompt(actual_turn_text, candidates)
        )
        self.last_call_count += 1
        satisfied_id = _parse_evidence_check_response(raw, {b.id for b in candidates})
        return EvidenceCheckResult(satisfied_branch_id=satisfied_id)


class DerivePath:
    """Turns a selected branch's full root-to-leaf path into the
    PathRequirement that scopes Teach — see PathRequirement's docstring
    for why `must_not_assume` is the field that matters most. Fast
    tier, one LLM call. A parse failure degrades to an empty-but-valid
    PathRequirement (all blank/empty fields) rather than raising —
    Teach then simply gets less scoping, the same graceful-degradation
    discipline as a missing topic did before this feature existed.

    `student_message`/`action_rationale` exist specifically so the
    prompt can draw a hard line between what's actually known (the
    student's real words) and what's merely predicted or not-yet-taught
    (a branch's predicted_next_turn, the tutor's own proposed idea) —
    see the exact failure this guards against in _derive_prompt's
    docstring-equivalent comments: a predicted *future* reaction being
    promoted into a stated *current* belief, which Teach then
    unwittingly affirms as something the student already said.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self.last_call_count: int = 0

    async def run(
        self,
        path: list[Branch],
        student_message: str,
        action_rationale: str,
        explicit_request: ExplicitRequest | None = None,
    ) -> PathRequirement:
        self.last_call_count = 0
        raw = await self._llm.complete(
            _derive_prompt(path, student_message, action_rationale, explicit_request)
        )
        self.last_call_count += 1
        result = _parse_path_requirement(raw)
        if explicit_request is not None and explicit_request.present:
            result = _enforce_explicit_request_scope(result, explicit_request)
        return result


# Words too generic to count as "content" for check_current_belief_leak
# — without filtering these, almost any two sentences about the same
# topic would share enough of them to produce false positives.
_BELIEF_CHECK_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "in", "on", "at", "by", "for", "with", "about", "as",
        "that", "this", "these", "those", "it", "its", "and", "or", "but",
        "student", "believes", "belief", "appears", "currently", "their",
        "they", "them", "has", "have", "had", "not", "no", "if", "so",
        "will", "would", "can", "could", "does", "do", "did", "than",
        "into", "from", "which", "what", "how", "when", "where", "who",
    }
)


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) > 3 and w not in _BELIEF_CHECK_STOPWORDS}


def check_current_belief_leak(
    current_belief: str,
    predicted_next_turn: str,
    action_rationale: str,
    student_message: str,
) -> bool:
    """A structural backstop for DerivePath's own prompt instructions,
    not a replacement for them: prompts drift, this catches what one
    misses. True when current_belief shares distinctive vocabulary with
    the selected branch's predicted_next_turn or the proposed action's
    rationale — content that has not actually happened yet — while
    sharing none with what the student actually said. This is a
    heuristic (word overlap, not semantic understanding), same
    "flag for a human to judge" spirit as the redundancy check's own
    wording-not-semantics caveat — it does not prove a leak occurred,
    it flags a pattern worth a human's attention in turn_diagnostics.
    """
    belief_words = _content_words(current_belief)
    if not belief_words:
        return False
    unconfirmed_words = _content_words(predicted_next_turn) | _content_words(
        action_rationale
    )
    message_words = _content_words(student_message)

    leaked = belief_words & unconfirmed_words
    grounded = belief_words & message_words
    return bool(leaked) and not grounded
