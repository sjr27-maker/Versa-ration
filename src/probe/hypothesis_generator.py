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

import difflib
import json
import logging
from uuid import UUID

from probe.branches import BranchStore
from probe.llm import LLMClient
from probe.models import Branch, BranchGeneration, BranchStatus, ResolutionResult
from probe.reasoning_budget import (
    BranchBudget,
    BranchBudgetConfig,
    compute_branch_budget,
)
from probe.store import HypothesisStore

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


def _intent_prompt(transcript_context: str, budget: BranchBudget) -> str:
    lo, hi = budget.root_branch_range
    return (
        "GENERATE:INTENT\n"
        f"Given the full session context below, propose between {lo} and {hi} "
        "distinct, plausible intents for why the student sent their last "
        "message — genuinely different bets, not rephrasings of one idea.\n\n"
        f"session context:\n{transcript_context}\n\n"
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
        hypothesis_store: HypothesisStore,
        transcript_context: str,
        learner_id: UUID,
    ) -> BranchGeneration:
        # Learner-scoped, not list_all(): the budget must reflect this
        # session's own learner's uncertainty, not every hypothesis in
        # the database across every learner (see the same fix on
        # Replan's entropy input, loop.py). Note list_by_learner's own
        # documented limitation: a hypothesis with no evidence yet
        # (freshly added, never reweighted) isn't attributable to a
        # learner and won't be counted here either.
        active_hypotheses = await hypothesis_store.list_by_learner(learner_id)
        budget = compute_branch_budget(active_hypotheses, self._budget_config)
        call_count = 0

        raw = await self._llm.complete(_intent_prompt(transcript_context, budget))
        call_count += 1
        root_items = _parse_intent_response(raw)

        generation_meta = await self._branches.create_generation(
            session_id, turn_index, len(root_items)
        )

        all_branches: list[Branch] = []
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
                    logger.info(
                        "hypothesis_generator: branch %s (depth=%d, "
                        "plausibility=%.2f) cleared the redundancy check "
                        "against siblings %r — reviewable to judge distinct "
                        "bets vs. rephrasings",
                        branch.id,
                        depth,
                        branch.plausibility,
                        siblings,
                    )
                    to_expand.append(branch)

            next_wave: list[Branch] = []
            for parent in to_expand:
                if len(all_branches) >= budget.max_total_branches:
                    break
                raw = await self._llm.complete(_expand_prompt(parent, budget))
                call_count += 1
                layer_label, child_items = _parse_expand_response(raw)
                if not child_items:
                    # Expansion produced nothing usable — parent stays a
                    # leaf; it already has its own predicted_next_turn.
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
        hypothesis_store: HypothesisStore,
        transcript_context: str,
        learner_id: UUID,
    ) -> BranchGeneration:
        return await self._generator.generate(
            session_id, turn_index, hypothesis_store, transcript_context, learner_id
        )


class BranchResolve:
    def __init__(self, generator: HypothesisGenerator) -> None:
        self._generator = generator

    async def run(
        self, session_id: UUID, turn_index: int, actual_turn_text: str
    ) -> ResolutionResult:
        return await self._generator.resolve(session_id, turn_index, actual_turn_text)
