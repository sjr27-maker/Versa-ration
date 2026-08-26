"""The session loop: Diagnose → Infer → Update → Replan → Plan → Teach
→ wait → repeat.

Test exists as a node class (see nodes.py) but isn't wired into this
loop yet. That comes with a later step.

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
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import UUID

from probe.audit import NodeCallStore, TranscriptStore
from probe.concept_graph import ConceptGraph
from probe.grounding import GroundConcept
from probe.llm import LLMClient, ModelTierClients
from probe.mismatch import MismatchDetector
from probe.nodes import (
    DEFAULT_GENERATION_WIDTH,
    MAX_CALLS_PER_TURN,
    Diagnose,
    Infer,
    Plan,
    Replan,
    Teach,
    Test,
    Update,
)
from probe.overlay import LearnerOverlay
from probe.reasoning_budget import ReasoningBudgetConfig
from probe.revision import WorldModelRevisionStore
from probe.store import HypothesisStore
from probe.value_function import ValueFunction, ValueFunctionConfig

logger = logging.getLogger(__name__)


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
    ) -> None:
        # Tiering (fast/capable/best -> real Gemini models, see
        # model_config.py) is opt-in via model_tier_clients. Omitting it
        # reproduces the pre-tiering behavior exactly: every node gets
        # the single `llm` argument, which is what every existing test
        # still does and must keep doing unchanged.
        tiers = model_tier_clients or ModelTierClients(fast=llm, capable=llm, best=llm)
        self._hyp = hypothesis_store
        self._transcript = transcript
        self._node_calls = node_calls
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
        # ValueFunction terms; capable -> Plan's proposer (seed_graph is
        # wired separately, from cli.py, since it isn't part of this
        # loop); best -> Teach.
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
        self._generation_width: int = DEFAULT_GENERATION_WIDTH
        self._exploration_target = None
        self._last_teach_message: str = ""

    async def run_interactive(
        self, learner_id: UUID, concept_graph_id: UUID
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
        self, session_id: UUID, turn_index: int, turn_text: str
    ) -> str:
        turn_id = await self._transcript.record_turn(
            session_id, turn_index, turn_text
        )

        diagnose_result = await self._call_node(
            self.diagnose,
            session_id,
            turn_index,
            response=turn_text,
            expectation=self._last_teach_message,
            session_id=session_id,
            turn_id=turn_id,
        )

        active_hypotheses = await self._hyp.list_all()

        proposals = await self._call_node(
            self.infer,
            session_id,
            turn_index,
            turn_text=turn_text,
            hypotheses=active_hypotheses,
            generation_width=self._generation_width,
        )

        await self._call_node(
            self.update,
            session_id,
            turn_index,
            proposals=proposals,
            hypothesis_store=self._hyp,
        )

        refreshed_hypotheses = await self._hyp.list_all()

        budget = await self._call_node(
            self.replan,
            session_id,
            turn_index,
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

        plan_output = await self._call_node(
            self.plan,
            session_id,
            turn_index,
            hypotheses=refreshed_hypotheses,
            concept_state={},
            generation_width=self._generation_width,
            exploration_target=self._exploration_target,
        )

        message = await self._call_node(
            self.teach,
            session_id,
            turn_index,
            action=plan_output.winner,
        )

        # Monitoring guardrail, not a hard stop (see MAX_CALLS_PER_TURN):
        # the *complete* per-turn LLM-call count, not an undercount of
        # it — every LLM-calling node/term in this turn is instrumented
        # (Diagnose already folds in GroundConcept + MismatchDetector;
        # Infer, Plan's proposer, and Teach each track their own calls;
        # ValueFunction's four LLM-calling terms are on each candidate's
        # ActionScore).
        plan_scoring_calls = sum(
            s.learning_value_call_count
            + s.information_value_call_count
            + s.cognitive_cost_call_count
            + s.frustration_risk_call_count
            for s in plan_output.scores
        )
        total_call_count = (
            diagnose_result["llm_call_count"]
            + self.infer.last_call_count
            + self.plan.last_generate_call_count
            + plan_scoring_calls
            + self.teach.last_call_count
        )
        if total_call_count > MAX_CALLS_PER_TURN:
            logger.warning(
                "turn %d: LLM calls this turn (%d) exceeded "
                "MAX_CALLS_PER_TURN=%d (Diagnose=%d, Infer=%d, "
                "Plan.propose=%d, Plan.scoring across %d candidates=%d, "
                "Teach=%d) — continuing without truncating reasoning; "
                "this is a monitoring guardrail, not a limit",
                turn_index,
                total_call_count,
                MAX_CALLS_PER_TURN,
                diagnose_result["llm_call_count"],
                self.infer.last_call_count,
                self.plan.last_generate_call_count,
                len(plan_output.scores),
                plan_scoring_calls,
                self.teach.last_call_count,
            )

        self._last_teach_message = message
        return message

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
        output = await node.run(**kwargs)
        await self._node_calls.record(
            node_name=type(node).__name__,
            session_id=session_id,
            turn_index=turn_index,
            input_json=kwargs,
            output_json=output,
        )
        return output
