"""The session loop: [evidence satisfaction] → [BranchResolve] →
[AttachTopic] → Diagnose → Infer → Update → Replan → Plan →
[BranchGenerate → GenerateOptions → SelectBranch → DerivePath] →
Teach → wait → repeat.

Diagnose runs first each turn, checking the student's response against
what was expected from the *previous* turn's Teach output
(`self._last_teach_message`, threaded forward the same way
`self._generation_width` is) and against this session's linked concept
graph (`session_id` -> `concept_graph_id`, resolved inside Diagnose via
TranscriptStore — a session's graph and learner are both set once at
creation, not threaded through as separate per-turn state).

All node invocations flow through `_call_node`, which records inputs
and outputs to node_calls per CLAUDE.md invariant 2.

Replan runs at the end of each turn and returns a `ReasoningBudget`
(reasoning_budget.py — the single source of truth for the entropy ->
behavior mapping) computed from the just-updated hypothesis
distribution. Three things get threaded from it into *next* turn:
`generation_width` (into Infer and Plan, via `self._generation_width`),
`run_information_value` (into ValueFunctionConfig.enable_information_value
— reusing the Step-3 toggle rather than adding a second switch; ANDed
with whatever the base config already said, so a turn never re-enables
a term an explicit ablation config disabled), and `exploration_target`
(into Plan, so one candidate can be framed at a specific
dormant/background hypothesis instead of only the dominant one).

Node construction is tier-aware (`model_tier_clients`, see llm.py /
model_config.py) but defaults to the pre-tiering behavior — a single
shared `llm` for every node — when omitted, which is what every
existing test still does.

HypothesisGenerator (hypothesis_generator.py) is a separate, parallel
signal — a speculative prediction tree, regenerated every turn,
distinct from the durable Hypothesis/HypothesisStore this loop already
threads through Infer/Update/Replan. It's opt-in via `branch_store`
(`None` reproduces the exact pre-existing turn flow, minus the
branch-derived path — Teach then gets `path_requirement=None` and
falls back to target_concept-only framing). `resolve()` still runs
first each turn (before Diagnose/Infer touch the student's new
message), matching it against the *previous* turn's generation.
`generate()` now runs *before* Teach, right after Plan: it conditions
on the student's message (already in transcript_context via
record_turn), the current hypothesis distribution, and Plan's just-
decided action/target_concept — not on Teach's rendered text, which
doesn't exist yet at this point in the turn. `SelectBranch` then picks
one branch from the tree by *coverage* (how much of the rest of the
live tree its path would also serve), not raw plausibility, and
`DerivePath` turns that branch's full root-to-leaf path into a
`PathRequirement` — what the student appears to believe, what they
need, and critically what must NOT be assumed as settled. Teach
receives that PathRequirement instead of the tree itself: a tree
invites free association, a path constrains. Because generation now
happens before Teach runs, it happens regardless of whether Teach
subsequently fails — see the teach_failed handling below for what that
implies for next turn's resolve().

Some branches carry `requires_evidence` — a claim with an entry
condition, not just a forecast (see should_expand_branch's fourth
gate): they hold at their current depth until evidence_satisfied
flips true. `GenerateOptions` turns live evidence-needing branches
into 2-4 clickable options, an unambiguous evidence channel with the
interpretation step removed. At the top of handle_turn, before
BranchResolve, a click (`selected_option_id`) satisfies its branch
directly with no LLM call; typed text instead gets one `CheckEvidence`
check against the prior generation's still-pending requirements — the
one place this mechanism still accepts interpretation, since a typed
answer isn't a button. Either path calls the same
`BranchStore.set_evidence_satisfied`, so a branch ends up in an
identical state regardless of which channel satisfied it. If neither a
click nor typed text satisfies anything that was on offer,
`options_missed` is set — read as "the options were wrong," not "the
student was uncooperative" — and fed into the next turn's generation
context.

Topic inference (`AttachTopic`, nodes.py) replaces `--topic`: a
session may be created with `concept_graph_id=None` (migration 013),
and its first turn (`turn_index == 0`) runs `AttachTopic` against the
student's message to attach one — best-effort; a failure there is
recorded as a warning and the turn continues with graceful degradation
(Diagnose already handles a None graph gracefully). Any turn *past*
the first with a still-null `concept_graph_id` is a structural
invariant violation, not a transient failure — `SessionMissingTopicError`
propagates out of `handle_turn` uncaught.

Per-node error handling: every node call between (not including) the
topic check and Teach is wrapped so its failure doesn't lose the turn
— caught, recorded as a warning string, and replaced with a safe
neutral fallback for whatever it would have returned. Teach has no
such fallback (its output *is* the turn): a Teach failure returns a
fixed in-band message instead of raising and sets
`turn_diagnostics.teach_failed`. BranchGenerate/SelectBranch/DerivePath
are no longer conditioned on teach_failed at all — they run before
Teach, so Teach's outcome isn't even known yet at that point; a Teach
failure is recorded as a warning alongside a kept (not discarded)
generation instead.

`turn_diagnostics` (diagnostics.py) is written once per turn, opt-in
via `diagnostics_store` (`None` skips recording, same backward-
compatible pattern as `branch_store`) — the persisted form of
everything this module already computes each turn, so the web UI's
Diagnostics panel reads it directly instead of re-deriving anything.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any
from uuid import UUID

from probe.audit import NodeCallStore, TranscriptStore
from probe.branches import BranchStore
from probe.concept_graph import ConceptGraph
from probe.diagnostics import TurnDiagnosticsStore
from probe.grounding import GroundConcept
from probe.hypothesis_generator import (
    BranchGenerate,
    BranchResolve,
    CheckEvidence,
    DerivePath,
    GenerateOptions,
    HypothesisGenerator,
    SelectBranch,
    build_branch_path,
    check_current_belief_leak,
)
from probe.llm import LLMClient, ModelTierClients
from probe.mismatch import MismatchDetector
from probe.models import (
    CandidateAction,
    Option,
    OptionStatus,
    PlanOutput,
    TeachingAction,
    TurnDiagnostics,
)
from probe.nodes import (
    DEFAULT_GENERATION_WIDTH,
    MAX_CALLS_PER_TURN,
    AttachTopic,
    Diagnose,
    Infer,
    Plan,
    Replan,
    SessionMissingTopicError,
    Teach,
    Test,
    Update,
)
from probe.options import OptionStore
from probe.overlay import LearnerOverlay
from probe.reasoning_budget import (
    BranchBudgetConfig,
    ReasoningBudgetConfig,
    compute_reasoning_budget,
)
from probe.revision import WorldModelRevisionStore
from probe.store import HypothesisStore
from probe.value_function import ValueFunction, ValueFunctionConfig

logger = logging.getLogger(__name__)

# Consecutive turns with no grounded concept before it's surfaced as an
# "off_graph_drift" warning. Arbitrary starting point, not measured —
# same honesty as every other placeholder threshold in this codebase;
# revisit once real session data exists. Never triggers a reseed on
# its own — purely a surfaced signal (see module docstring).
_OFF_GRAPH_DRIFT_THRESHOLD = 3

# Deterministic, no-LLM-call fallback for a turn whose Plan failed
# entirely — the same enum-order choice _backfill() already uses when
# the proposer returns too few candidates, just applied to "zero
# candidates scored at all" instead of "some."
_PLAN_FALLBACK_ACTION = TeachingAction.EXPLAIN

_TEACH_FAILURE_MESSAGE = (
    "the tutor failed to respond this turn — see Diagnostics for the "
    "error; try sending your message again"
)

# Diagnose's own "nothing grounded / nothing to say" shape (see
# nodes.py, Diagnose.run()'s initial `result` dict) — reused here as
# the fallback when Diagnose itself raises, so downstream code (the
# off-graph-drift check, the guardrail sum) doesn't need a second shape
# to handle.
def _total_retry_count(tiers: ModelTierClients) -> int:
    """Sum of GeminiLLMClient.retry_count across every distinct client
    in `tiers`, deduplicated by identity. Dedup matters because the
    default (no explicit model_tier_clients) is the *same* LLMClient
    instance for fast/capable/best — summing all three unguarded would
    triple-count. getattr(..., 0) makes this 0 for StubLLMClient (no
    retry mechanism to count) without either client type needing to
    know about the other.

    Called once at the start and once at the end of handle_turn; the
    delta is that turn's retry_count (same before/after snapshot
    pattern as duration_ms's time.monotonic() call)."""
    seen: set[int] = set()
    total = 0
    for client in (tiers.fast, tiers.capable, tiers.best):
        if id(client) in seen:
            continue
        seen.add(id(client))
        total += getattr(client, "retry_count", 0)
    return total


_DIAGNOSE_FALLBACK: dict[str, Any] = {
    "classification": "unknown",
    "matched_expectation": False,
    "notes": "Diagnose failed this turn — see warnings",
    "grounding": None,
    "mismatch": None,
    "action_taken": "none",
    "revision_id": None,
    "reweighted_hypothesis_ids": [],
    "llm_call_count": 0,
}


class SessionLoop:
    def __init__(
        self,
        hypothesis_store: HypothesisStore,
        transcript: TranscriptStore,
        node_calls: NodeCallStore,
        concept_graph: ConceptGraph,
        learner_overlay: LearnerOverlay,
        revision_store: WorldModelRevisionStore,
        llm: LLMClient,
        value_function_config: ValueFunctionConfig | None = None,
        reasoning_budget_config: ReasoningBudgetConfig | None = None,
        model_tier_clients: ModelTierClients | None = None,
        branch_store: BranchStore | None = None,
        branch_budget_config: BranchBudgetConfig | None = None,
        diagnostics_store: TurnDiagnosticsStore | None = None,
        option_store: OptionStore | None = None,
        on_node_start: Callable[[str], None] | None = None,
    ) -> None:
        # Tiering (fast/capable/best -> real Gemini models, see
        # model_config.py) is opt-in via model_tier_clients. Omitting it
        # reproduces the pre-tiering behavior exactly: every node gets
        # the single `llm` argument, which is what every existing test
        # still does and must keep doing unchanged.
        tiers = model_tier_clients or ModelTierClients(fast=llm, capable=llm, best=llm)
        # Kept on self so handle_turn can snapshot _total_retry_count(...)
        # before/after each turn — see that function's docstring.
        self._tiers = tiers
        self._hyp = hypothesis_store
        self._transcript = transcript
        self._node_calls = node_calls
        self._concepts = concept_graph
        self._learner_overlay = learner_overlay
        self._diagnostics = diagnostics_store
        self._on_node_start = on_node_start
        # Kept separately from self.value_function.config so the
        # per-turn run_information_value toggle can be ANDed against
        # the caller's original intent instead of overwriting it —
        # ValueFunctionConfig is mutated in place turn to turn (see
        # handle_turn), so the original has to be remembered elsewhere.
        self._base_enable_information_value = (
            value_function_config.enable_information_value
            if value_function_config is not None
            else True
        )
        # Tier assignment: fast -> Infer, GroundConcept, MismatchDetector,
        # ValueFunction terms, AttachTopic's topic extraction; capable ->
        # Plan's proposer, AttachTopic's seed_graph delegation; best -> Teach.
        self.value_function = ValueFunction(tiers.fast, value_function_config)
        self.infer = Infer(tiers.fast)
        self.plan = Plan(self.value_function, tiers.capable)
        self.teach = Teach(tiers.best)
        self.test = Test()
        self.update = Update()
        self.replan = Replan(reasoning_budget_config)
        self.diagnose = Diagnose(
            mismatch_detector=MismatchDetector(tiers.fast),
            ground_concept=GroundConcept(tiers.fast),
            hypothesis_store=hypothesis_store,
            revision_store=revision_store,
            concept_graph=concept_graph,
            learner_overlay=learner_overlay,
            transcript=transcript,
        )
        self.attach_topic = AttachTopic(
            tiers.fast, tiers.capable, concept_graph, transcript
        )
        self._generation_width: int = DEFAULT_GENERATION_WIDTH
        self._exploration_target = None
        self._last_teach_message: str = ""
        self._consecutive_ungrounded_turns: int = 0
        # Whether the immediately preceding turn presented options the
        # student typed past without satisfying — read by
        # _build_transcript_context to inform this turn's generation
        # (see handle_turn's options_missed handling).
        self._last_options_missed: bool = False

        # HypothesisGenerator is opt-in: branch_store=None (the default,
        # and what every existing test still passes) reproduces the
        # exact pre-existing turn flow — generate()/resolve() are simply
        # never called. Tier: fast, same reasoning as GroundConcept/
        # MismatchDetector (fires every turn, multiplies fast).
        self._branch_store = branch_store
        self._options_store = option_store
        self._prior_generation_id = None
        if branch_store is not None:
            self._hypothesis_generator = HypothesisGenerator(
                tiers.fast, branch_store, branch_budget_config, option_store
            )
            self.branch_generate: BranchGenerate | None = BranchGenerate(
                self._hypothesis_generator
            )
            self.branch_resolve: BranchResolve | None = BranchResolve(
                self._hypothesis_generator
            )
            self.select_branch: SelectBranch | None = SelectBranch(tiers.fast)
            self.derive_path: DerivePath | None = DerivePath(tiers.fast)
        else:
            self.branch_generate = None
            self.branch_resolve = None
            self.select_branch = None
            self.derive_path = None

        # GenerateOptions/CheckEvidence are a further opt-in on top of
        # branch_store: they need real branches to reason over, so
        # option_store without branch_store does nothing. option_store
        # alone (branch_store also set) is what "both channels stay
        # enabled" means — the click path and the typed-evidence-check
        # path both route through this pair.
        if branch_store is not None and option_store is not None:
            self.generate_options: GenerateOptions | None = GenerateOptions(tiers.fast)
            self.check_evidence: CheckEvidence | None = CheckEvidence(tiers.fast)
        else:
            self.generate_options = None
            self.check_evidence = None

    async def run_interactive(
        self, learner_id: UUID, concept_graph_id: UUID | None
    ) -> UUID:
        session_id = await self._transcript.create_session(
            learner_id, concept_graph_id
        )
        print(f"probe: new session {session_id}")
        print("probe: type your message. ctrl-D or empty line + ctrl-C to exit.")
        turn_index = 0
        loop = asyncio.get_running_loop()
        while True:
            try:
                turn_text = await loop.run_in_executor(None, input, "you: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            turn_text = turn_text.strip()
            if not turn_text:
                continue
            message = await self.handle_turn(session_id, turn_index, turn_text)
            print(f"probe: {message}")
            turn_index += 1
        return session_id

    async def handle_turn(
        self,
        session_id: UUID,
        turn_index: int,
        turn_text: str,
        selected_option_id: UUID | None = None,
    ) -> str:
        start = time.monotonic()
        retry_count_start = _total_retry_count(self._tiers)
        warnings: list[str] = []
        node_call_counts: dict[str, int] = {}

        # Evidence satisfaction: a click is unambiguous and needs no
        # LLM call at all; typed text gets one check against the prior
        # generation's still-pending requires_evidence. Both happen
        # before BranchResolve so its own option-supersession (inside
        # resolve()) sees the outcome — an option marked `selected`
        # here is left untouched by that blanket close-out, exactly
        # the way a matched branch's ancestor chain is excluded from
        # supersede_open_branches.
        had_pending_options = False
        satisfied_branch_id: UUID | None = None
        # Set only by an actual click — distinct from satisfied_branch_id
        # (which a typed CheckEvidence match also sets) because
        # BranchResolve needs to know specifically whether *this* turn
        # was a click, not merely whether something got satisfied.
        clicked_branch_id: UUID | None = None
        prior_generation_id = self._prior_generation_id
        if prior_generation_id is not None and self._options_store is not None:
            pending_options = await self._options_store.list_by_generation(
                prior_generation_id
            )
            had_pending_options = any(
                o.status is OptionStatus.OPEN for o in pending_options
            )

        if selected_option_id is not None and self._options_store is not None:
            option = await self._options_store.get(selected_option_id)
            if option is not None:
                await self._options_store.set_status(option.id, OptionStatus.SELECTED)
                await self._branch_store.set_evidence_satisfied(option.branch_id, True)
                satisfied_branch_id = option.branch_id
                clicked_branch_id = option.branch_id
        elif (
            selected_option_id is None
            and self.check_evidence is not None
            and prior_generation_id is not None
        ):
            evidence_candidates = await self._branch_store.list_awaiting_evidence(
                prior_generation_id
            )
            if evidence_candidates:
                result = await self._call_node_or_warn(
                    self.check_evidence,
                    session_id,
                    turn_index,
                    "CheckEvidence",
                    None,
                    warnings,
                    actual_turn_text=turn_text,
                    candidates=evidence_candidates,
                )
                node_call_counts["CheckEvidence"] = self.check_evidence.last_call_count
                if result is not None and result.satisfied_branch_id is not None:
                    satisfied_branch_id = result.satisfied_branch_id
                    await self._branch_store.set_evidence_satisfied(
                        result.satisfied_branch_id, True
                    )

        # Read this as "the options didn't offer what the student
        # actually needed," not as the student being uncooperative —
        # it's a signal about the branch/option set, not the student.
        options_missed = (
            had_pending_options
            and selected_option_id is None
            and satisfied_branch_id is None
        )

        resolution_call_count = 0
        if self.branch_resolve is not None and self._prior_generation_id is not None:
            resolution = await self._call_node_or_warn(
                self.branch_resolve,
                session_id,
                turn_index,
                "BranchResolve",
                None,
                warnings,
                session_id=session_id,
                turn_index=turn_index,
                actual_turn_text=turn_text,
                clicked_branch_id=clicked_branch_id,
            )
            if resolution is not None:
                resolution_call_count = resolution.call_count
                node_call_counts["BranchResolve"] = resolution_call_count
                self._prior_generation_id = None
            # else: BranchResolve failed — leave self._prior_generation_id
            # untouched so next turn's resolve naturally retries against
            # the same still-open generation, rather than silently
            # abandoning it.

        turn_id = await self._transcript.record_turn(
            session_id, turn_index, turn_text
        )
        # Resolved once here for Replan/generate()'s learner-scoped
        # hypothesis reads below (see list_by_learner). Diagnose still
        # resolves its own copy internally — untouched, per this pass's
        # constraint not to change Diagnose's existing logic.
        learner_id = await self._transcript.get_learner_id(session_id)

        # Topic inference: a session created with no topic
        # (concept_graph_id is None, migration 013) gets one attached
        # here, on its first turn only. Best-effort — AttachTopic
        # failing doesn't crash the turn; Diagnose already degrades
        # gracefully against a still-null graph. Any turn past the
        # first with a still-null graph is a structural bug, not a
        # transient failure, so it hard-fails instead.
        inferred_topic: str | None = None
        topic_seeded_new: bool | None = None
        current_graph_id = await self._transcript.get_concept_graph_id(session_id)
        if current_graph_id is None:
            if turn_index == 0:
                try:
                    attachment = await self._call_node(
                        self.attach_topic,
                        session_id,
                        turn_index,
                        message=turn_text,
                        session_id=session_id,
                    )
                    node_call_counts["AttachTopic"] = self.attach_topic.last_call_count
                    inferred_topic = attachment.topic
                    topic_seeded_new = attachment.seeded_new
                    current_graph_id = attachment.concept_graph_id
                    logger.info(
                        "AttachTopic: session %s attached to topic %r "
                        "(concept_graph_id=%s, seeded_new=%s)",
                        session_id,
                        attachment.topic,
                        attachment.concept_graph_id,
                        attachment.seeded_new,
                    )
                except Exception as exc:
                    warnings.append(f"AttachTopic failed: {exc}")
                    logger.warning(
                        "AttachTopic failed on turn 0 for session %s: %s",
                        session_id,
                        exc,
                        exc_info=True,
                    )
            else:
                raise SessionMissingTopicError(
                    f"session {session_id} has no concept_graph_id attached "
                    f"by turn {turn_index} — AttachTopic only ever runs on "
                    "turn 0"
                )

        diagnose_result = await self._call_node_or_warn(
            self.diagnose,
            session_id,
            turn_index,
            "Diagnose",
            dict(_DIAGNOSE_FALLBACK),
            warnings,
            response=turn_text,
            expectation=self._last_teach_message,
            session_id=session_id,
            turn_id=turn_id,
        )
        node_call_counts["Diagnose"] = diagnose_result["llm_call_count"]

        grounding = diagnose_result.get("grounding")
        ungrounded_this_turn = grounding is None or grounding.get("concept_id") is None
        if ungrounded_this_turn:
            self._consecutive_ungrounded_turns += 1
        else:
            self._consecutive_ungrounded_turns = 0
        if self._consecutive_ungrounded_turns >= _OFF_GRAPH_DRIFT_THRESHOLD:
            warnings.append(
                f"off_graph_drift: {self._consecutive_ungrounded_turns} "
                "consecutive turns with no grounded concept"
            )

        # Learner-scoped everywhere now, not list_all(): a turn must
        # never be shown, propose evidence against, or budget reasoning
        # from another learner's hypotheses. list_all() has no learner
        # filter at all — see CLAUDE.md / the entropy-contamination fix
        # this followed. Infer's own hallucination-rejection (nodes.py)
        # is what actually makes "never reweight another learner's
        # hypothesis" true, not just "won't see one" — this scoping is
        # what makes that check meaningful in the first place.
        active_hypotheses = await self._hyp.list_by_learner(learner_id)

        proposals = await self._call_node_or_warn(
            self.infer,
            session_id,
            turn_index,
            "Infer",
            [],
            warnings,
            turn_text=turn_text,
            hypotheses=active_hypotheses,
            generation_width=self._generation_width,
        )
        node_call_counts["Infer"] = self.infer.last_call_count

        await self._call_node_or_warn(
            self.update,
            session_id,
            turn_index,
            "Update",
            [],
            warnings,
            proposals=proposals,
            hypothesis_store=self._hyp,
        )

        refreshed_hypotheses = await self._hyp.list_by_learner(learner_id)

        budget = await self._call_node_or_warn(
            self.replan,
            session_id,
            turn_index,
            "Replan",
            compute_reasoning_budget([]),
            warnings,
            hypotheses=refreshed_hypotheses,
        )
        self._generation_width = budget.generation_width
        self._exploration_target = budget.exploration_target
        # AND, not overwrite: a turn never re-enables a term an explicit
        # ablation config disabled — it can only skip a term the base
        # config already allowed.
        self.value_function.config.enable_information_value = (
            self._base_enable_information_value and budget.run_information_value
        )

        plan_fallback = PlanOutput(
            winner=CandidateAction(
                action=_PLAN_FALLBACK_ACTION,
                target_concept=None,
                rationale="Plan failed this turn — deterministic fallback, no scoring",
            ),
            scores=[],
        )
        concept_state = await self._build_concept_state(
            current_graph_id, learner_id, diagnose_result
        )
        plan_output = await self._call_node_or_warn(
            self.plan,
            session_id,
            turn_index,
            "Plan",
            plan_fallback,
            warnings,
            hypotheses=refreshed_hypotheses,
            concept_state=concept_state,
            generation_width=self._generation_width,
            exploration_target=self._exploration_target,
        )
        plan_scoring_calls = sum(
            s.learning_value_call_count
            + s.information_value_call_count
            + s.cognitive_cost_call_count
            + s.frustration_risk_call_count
            for s in plan_output.scores
        )
        node_call_counts["Plan"] = self.plan.last_generate_call_count + plan_scoring_calls

        # BranchGenerate -> SelectBranch -> DerivePath, all before
        # Teach: the tree now conditions on the student's message
        # (already in transcript_context via record_turn), the current
        # hypotheses, and Plan's just-decided action/target_concept —
        # everything Teach itself would have needed, available without
        # waiting for Teach's rendered text. This also means generation
        # happens regardless of whether Teach subsequently fails (see
        # the teach_failed handling below): nothing about it depends on
        # Teach succeeding, so there is no "skip generation" case left
        # to gate on.
        generation_call_count = 0
        path_requirement = None
        current_belief_unsupported = False
        option_texts: list[str] = []
        if self.branch_generate is not None:
            transcript_context = await self._build_transcript_context(
                session_id, self._last_options_missed
            )
            generation = await self._call_node_or_warn(
                self.branch_generate,
                session_id,
                turn_index,
                "BranchGenerate",
                None,
                warnings,
                session_id=session_id,
                turn_index=turn_index,
                transcript_context=transcript_context,
                hypotheses=refreshed_hypotheses,
                action=plan_output.winner,
            )
            if generation is not None:
                generation_call_count = generation.call_count
                node_call_counts["BranchGenerate"] = generation_call_count
                self._prior_generation_id = generation.generation.id
                for note in generation.redundancy_notes:
                    warnings.append(f"redundancy_check: {note}")

                if generation.branches and self.generate_options is not None:
                    proposals = await self._call_node_or_warn(
                        self.generate_options,
                        session_id,
                        turn_index,
                        "GenerateOptions",
                        [],
                        warnings,
                        branches=generation.branches,
                    )
                    node_call_counts["GenerateOptions"] = (
                        self.generate_options.last_call_count
                    )
                    if proposals:
                        new_options = [
                            Option(
                                branch_id=p.branch_id,
                                generation_id=generation.generation.id,
                                session_id=session_id,
                                turn_index=turn_index,
                                text=p.text,
                            )
                            for p in proposals
                        ]
                        await self._options_store.create_options(new_options)
                        option_texts = [o.text for o in new_options]

                if generation.branches and self.select_branch is not None:
                    selection = await self._call_node_or_warn(
                        self.select_branch,
                        session_id,
                        turn_index,
                        "SelectBranch",
                        None,
                        warnings,
                        branches=generation.branches,
                    )
                    if selection is not None:
                        node_call_counts["SelectBranch"] = (
                            self.select_branch.last_call_count
                        )
                        await self._branch_store.set_selection(
                            generation.generation.id,
                            selection.selected_branch_id,
                            selection.rationale,
                        )
                        if (
                            selection.selected_branch_id is not None
                            and self.derive_path is not None
                        ):
                            path = build_branch_path(
                                generation.branches, selection.selected_branch_id
                            )
                            path_result = await self._call_node_or_warn(
                                self.derive_path,
                                session_id,
                                turn_index,
                                "DerivePath",
                                None,
                                warnings,
                                path=path,
                                student_message=turn_text,
                                action_rationale=plan_output.winner.rationale,
                            )
                            if path_result is not None:
                                node_call_counts["DerivePath"] = (
                                    self.derive_path.last_call_count
                                )
                                path_requirement = path_result
                                await self._branch_store.set_path_requirement(
                                    generation.generation.id, path_requirement
                                )
                                # Structural backstop, not a replacement
                                # for DerivePath's own prompt
                                # instructions: catches a predicted
                                # *future* reaction (or the tutor's own
                                # not-yet-taught idea) being promoted
                                # into a stated *current* belief, which
                                # Teach would otherwise unwittingly
                                # affirm as something the student
                                # already said.
                                selected_branch = path[-1] if path else None
                                if selected_branch is not None and check_current_belief_leak(
                                    path_requirement.current_belief,
                                    selected_branch.predicted_next_turn,
                                    plan_output.winner.rationale,
                                    turn_text,
                                ):
                                    current_belief_unsupported = True
                                    warnings.append(
                                        "current_belief_unsupported: DerivePath's "
                                        "current_belief shares content with the "
                                        "predicted reaction or proposed action "
                                        "rationale but nothing the student "
                                        "actually said — Teach may be about to "
                                        "affirm something as said that wasn't"
                                    )

        # Teach has no fallback — its output *is* the turn. A failure
        # here doesn't crash the turn or the session, but there's no
        # real teaching content to show either way: a fixed in-band
        # message takes its place. Unlike before this reorder,
        # BranchGenerate above already ran regardless — its predictions
        # target the planned action (Plan's decision), not Teach's
        # rendered output, so a Teach failure here doesn't invalidate
        # them. What it does mean: if Teach fails, the student sees
        # _TEACH_FAILURE_MESSAGE, not the planned lesson, so next
        # turn's real response is a reaction to a failure notice, not
        # to what this generation predicted reactions to — flagged
        # below as a warning so match-rate analysis can see why a
        # miss happened, not silently misinterpret it as the mechanism
        # being wrong.
        teach_failed = False
        try:
            message = await self._call_node(
                self.teach,
                session_id,
                turn_index,
                action=plan_output.winner,
                student_message=turn_text,
                path_requirement=path_requirement,
                options=option_texts,
            )
            node_call_counts["Teach"] = self.teach.last_call_count
        except Exception as exc:
            teach_failed = True
            message = _TEACH_FAILURE_MESSAGE
            warnings.append(f"Teach failed: {exc}")
            logger.warning(
                "Teach failed on turn %d for session %s: %s",
                turn_index,
                session_id,
                exc,
                exc_info=True,
            )
            node_call_counts["Teach"] = 0
            if generation_call_count > 0:
                warnings.append(
                    "BranchGenerate ran before Teach failed this turn — its "
                    "predictions target the planned action, not the "
                    "fallback failure message the student actually saw"
                )

        # Monitoring guardrail, not a hard stop (see MAX_CALLS_PER_TURN):
        # the *complete* per-turn LLM-call count, not an undercount of
        # it — every LLM-calling node/term in this turn is instrumented
        # (Diagnose already folds in GroundConcept + MismatchDetector;
        # Infer, Plan's proposer, and Teach each track their own calls;
        # ValueFunction's four LLM-calling terms are on each candidate's
        # ActionScore; BranchResolve/BranchGenerate's call_count is on
        # their own return values; SelectBranch/DerivePath/
        # GenerateOptions/CheckEvidence are each a single tracked call).
        total_call_count = (
            diagnose_result["llm_call_count"]
            + self.infer.last_call_count
            + self.plan.last_generate_call_count
            + plan_scoring_calls
            + node_call_counts.get("Teach", 0)
            + resolution_call_count
            + generation_call_count
            + node_call_counts.get("SelectBranch", 0)
            + node_call_counts.get("DerivePath", 0)
            + node_call_counts.get("GenerateOptions", 0)
            + node_call_counts.get("CheckEvidence", 0)
            + node_call_counts.get("AttachTopic", 0)
        )
        guardrail_fired = total_call_count > MAX_CALLS_PER_TURN
        if guardrail_fired:
            logger.warning(
                "turn %d: LLM calls this turn (%d) exceeded "
                "MAX_CALLS_PER_TURN=%d (%r) — continuing without "
                "truncating reasoning; this is a monitoring guardrail, "
                "not a limit",
                turn_index,
                total_call_count,
                MAX_CALLS_PER_TURN,
                node_call_counts,
            )
            warnings.append(
                f"MAX_CALLS_PER_TURN exceeded: {total_call_count} calls "
                f"(limit {MAX_CALLS_PER_TURN})"
            )

        if options_missed:
            warnings.append(
                "options_missed: the student typed past the prior turn's "
                "options without satisfying any of them — treat this as a "
                "signal the branch/option set was wrong, not that the "
                "student was uncooperative"
            )

        if self._diagnostics is not None:
            await self._diagnostics.record(
                TurnDiagnostics(
                    session_id=session_id,
                    turn_index=turn_index,
                    node_call_counts=node_call_counts,
                    total_call_count=total_call_count,
                    guardrail_fired=guardrail_fired,
                    entropy_bits=budget.entropy_bits,
                    duration_ms=(time.monotonic() - start) * 1000,
                    warnings=warnings,
                    teach_failed=teach_failed,
                    inferred_topic=inferred_topic,
                    topic_seeded_new=topic_seeded_new,
                    retry_count=_total_retry_count(self._tiers) - retry_count_start,
                    options_missed=options_missed,
                    current_belief_unsupported=current_belief_unsupported,
                )
            )

        self._last_teach_message = message
        self._last_options_missed = options_missed
        return message

    async def _build_concept_state(
        self,
        current_graph_id: UUID | None,
        learner_id: UUID,
        diagnose_result: dict,
    ) -> dict:
        """Grounding for Plan's proposer, not curriculum selection: the
        session's topic, the concepts that actually exist in its graph
        (so a target_concept can be a real id instead of invented or
        left null), which one the student's own message was grounded
        in this turn (Diagnose's GroundConcept result), and the
        learner's read-only overlay state for those concepts. This
        does not choose what to teach next — that stays out of scope;
        it only tells the proposer what's actually in front of it.
        """
        if current_graph_id is None:
            return {}
        # Three independent reads (graph metadata, concept list, overlay
        # state) — none depends on another's result, so they're gathered
        # rather than three sequential round trips.
        graph_meta, concepts, overlay = await asyncio.gather(
            self._concepts.get_graph(current_graph_id),
            self._concepts.list_concepts(current_graph_id),
            self._learner_overlay.get_overlay_for_graph(learner_id, current_graph_id),
        )
        grounding = (
            diagnose_result.get("grounding")
            if isinstance(diagnose_result, dict)
            else None
        )
        grounded_concept_id = (
            grounding.get("concept_id") if isinstance(grounding, dict) else None
        )
        return {
            "topic": graph_meta.topic if graph_meta is not None else None,
            "concepts": [{"id": c.id, "name": c.name} for c in concepts],
            "grounded_concept_id": grounded_concept_id,
            "overlay": {
                entry.concept_id: {
                    "state": entry.state.value,
                    "confidence": entry.confidence,
                }
                for entry in overlay
            },
        }

    async def _build_transcript_context(
        self, session_id: UUID, options_missed_last_turn: bool = False
    ) -> str:
        """Full session history for HypothesisGenerator.generate() —
        every student turn recorded so far, including this turn's own
        (record_turn already wrote it earlier in handle_turn). No
        longer appends a rendered Teach message: generation now runs
        *before* Teach, so there is nothing yet to append — the tree
        conditions on Plan's decided action/target_concept instead (see
        handle_turn), which is available at this point in the turn and
        Teach's rendered text is not.

        `options_missed_last_turn` feeds the prior turn's options_missed
        outcome back into this turn's generation prompt: if the student
        went around what was offered, the regenerated tree should be
        able to react to having missed, not repeat the same shape of
        options blind to the fact that they didn't land."""
        turns = await self._transcript.list_turns(session_id)
        lines = [f"student (turn {t.turn_index}): {t.text}" for t in turns]
        if options_missed_last_turn:
            lines.append(
                "[note: the options offered last turn did not match what "
                "the student actually needed -- they answered around them "
                "instead of clicking one. Consider whether the current "
                "branch set is asking the right question.]"
            )
        return "\n".join(lines)

    async def _call_node_or_warn(
        self,
        node: Any,
        session_id: UUID,
        turn_index: int,
        label: str,
        fallback: Any,
        warnings: list[str],
        /,
        **kwargs: Any,
    ) -> Any:
        """Same as `_call_node`, except a raised exception is caught,
        recorded into `warnings`, and swallowed in favor of `fallback`
        — so one node's transient failure doesn't lose the whole turn.
        Not used for Teach, which has no safe fallback (see
        handle_turn's dedicated try/except for that one)."""
        try:
            return await self._call_node(node, session_id, turn_index, **kwargs)
        except Exception as exc:
            warnings.append(f"{label} failed: {exc}")
            logger.warning(
                "%s failed on turn %d for session %s: %s",
                label,
                turn_index,
                session_id,
                exc,
                exc_info=True,
            )
            return fallback

    async def _call_node(
        self,
        node: Any,
        session_id: UUID,
        turn_index: int,
        /,
        **kwargs: Any,
    ) -> Any:
        # session_id/turn_index/node are positional-only so a node's own
        # run() kwargs (Diagnose's `session_id`, in particular) can share
        # a name with them without colliding.
        if self._on_node_start is not None:
            self._on_node_start(type(node).__name__)
        output = await node.run(**kwargs)
        await self._node_calls.record(
            node_name=type(node).__name__,
            session_id=session_id,
            turn_index=turn_index,
            input_json=kwargs,
            output_json=output,
        )
        return output
