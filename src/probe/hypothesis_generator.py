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
from uuid import UUID

from probe.branches import BranchStore
from probe.llm import LLMClient
from probe.models import (
    Branch,
    BranchGeneration,
    BranchSelection,
    BranchStatus,
    CandidateAction,
    Hypothesis,
    PathRequirement,
    ResolutionResult,
)
from probe.reasoning_budget import (
    BranchBudget,
    BranchBudgetConfig,
    compute_branch_budget,
)

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
) -> bool:
    """The one explicit "worth expanding to the next layer" filter:
    (a) plausible enough, (b) distinguishes from siblings rather than
    restating them, (c) the turn's budget still permits it.

    No LLM call for (b): a text-similarity heuristic is enough since
    near-duplicate branches from the same generation call tend to be
    near-duplicate text, and an extra call per decision would defeat
    the point of budgeting call count at all. This catches wording-
    level duplication, not semantic duplication — every branch that
    clears it is logged (see generate()) specifically so a human can
    judge from real session data whether survivors are genuinely
    distinct bets or rephrasings of one idea.
    """
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
    are skipped, not fatal — same discipline as Infer/Plan's parsing."""
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
        items.append(
            {
                "statement": str(statement),
                "plausibility": _clamp01(entry.get("plausibility", 0.0)),
                "predicted_next_turn": str(predicted),
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
        'Respond with JSON: [{"statement": "...", "plausibility": 0.0-1.0, '
        '"predicted_next_turn": "a concrete, checkable prediction of what '
        'the student will say or do next if this intent is true"}, ...]'
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
        'Respond with JSON: {"layer_label": "...", "children": '
        '[{"statement": "...", "plausibility": 0.0-1.0, "predicted_next_turn": '
        '"a concrete, checkable prediction of what the student will say or '
        'do next if this branch is true"}, ...]}'
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


def _derive_prompt(path: list[Branch]) -> str:
    listing = "\n".join(
        f"- depth={b.depth} [{b.depth_label}]: {b.statement} "
        f"(predicted reaction: {b.predicted_next_turn})"
        for b in path
    )
    return (
        "DERIVE:PATH\n"
        "Below is the full root-to-leaf path this turn's teaching should "
        "be derived from, from the student's root-level intent down to "
        "the most specific selected branch.\n\n"
        f"{listing}\n\n"
        "From this path, derive what this turn's teaching should do. Be "
        "precise about what the path actually implies versus what is "
        "merely possible — do not invent specifics (values, signs, "
        "conditions) the path does not state.\n"
        'Respond with JSON: {"current_belief": "what the student appears '
        'to currently believe, based on this path", "needed": "what they '
        'must be given to move along this path", "must_not_assume": '
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
    ) -> None:
        self._llm = llm
        self._branches = branch_store
        self._budget_config = budget_config

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

    async def resolve(
        self, session_id: UUID, turn_index: int, actual_turn_text: str
    ) -> ResolutionResult:
        generation = await self._branches.get_latest_generation(session_id)
        if generation is None:
            return ResolutionResult(
                session_id=session_id,
                turn_index=turn_index,
                matched_branch_id=None,
                status="unmatched",
                call_count=0,
            )
        leaves = await self._branches.get_open_leaves(generation.id)
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

        matched_chain: list[UUID] = []
        if matched_id is not None:
            await self._branches.set_status(matched_id, BranchStatus.MATCHED)
            matched_chain.append(matched_id)
            for ancestor in await self._branches.get_ancestors(matched_id):
                await self._branches.set_status(ancestor.id, BranchStatus.MATCHED)
                matched_chain.append(ancestor.id)
            status = "matched"
            exclude = matched_chain
        else:
            for leaf in leaves:
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
            exclude = [leaf.id for leaf in leaves]

        # Close the generation out completely: any branch still `open`
        # (intermediate depths that were never evaluated as leaves)
        # becomes `superseded`, so nothing from a resolved generation
        # lingers as open.
        await self._branches.supersede_open_branches(generation.id, exclude)

        return ResolutionResult(
            session_id=session_id,
            turn_index=turn_index,
            matched_branch_id=matched_id,
            matched_chain=matched_chain,
            status=status,
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
        self, session_id: UUID, turn_index: int, actual_turn_text: str
    ) -> ResolutionResult:
        return await self._generator.resolve(session_id, turn_index, actual_turn_text)


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


class DerivePath:
    """Turns a selected branch's full root-to-leaf path into the
    PathRequirement that scopes Teach — see PathRequirement's docstring
    for why `must_not_assume` is the field that matters most. Fast
    tier, one LLM call. A parse failure degrades to an empty-but-valid
    PathRequirement (all blank/empty fields) rather than raising —
    Teach then simply gets less scoping, the same graceful-degradation
    discipline as a missing topic did before this feature existed.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self.last_call_count: int = 0

    async def run(self, path: list[Branch]) -> PathRequirement:
        self.last_call_count = 0
        raw = await self._llm.complete(_derive_prompt(path))
        self.last_call_count += 1
        return _parse_path_requirement(raw)
