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

Replan runs at the end of each turn, computes a `generation_width` from
the just-updated hypothesis distribution, and that width is threaded
into *next* turn's Infer call via `self._generation_width`.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from probe.audit import NodeCallStore, TranscriptStore
from probe.concept_graph import ConceptGraph
from probe.grounding import GroundConcept
from probe.llm import LLMClient
from probe.mismatch import MismatchDetector
from probe.nodes import (
    DEFAULT_GENERATION_WIDTH,
    Diagnose,
    Infer,
    Plan,
    Replan,
    Teach,
    Test,
    Update,
)
from probe.overlay import LearnerOverlay
from probe.revision import WorldModelRevisionStore
from probe.store import HypothesisStore
from probe.value_function import ValueFunction, ValueFunctionConfig


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
    ) -> None:
        self._hyp = hypothesis_store
        self._transcript = transcript
        self._node_calls = node_calls
        self.value_function = ValueFunction(llm, value_function_config)
        self.infer = Infer(llm)
        self.plan = Plan(self.value_function, llm)
        self.teach = Teach(llm)
        self.test = Test()
        self.update = Update()
        self.replan = Replan()
        self.diagnose = Diagnose(
            mismatch_detector=MismatchDetector(llm),
            ground_concept=GroundConcept(llm),
            hypothesis_store=hypothesis_store,
            revision_store=revision_store,
            concept_graph=concept_graph,
            learner_overlay=learner_overlay,
            transcript=transcript,
        )
        self._generation_width: int = DEFAULT_GENERATION_WIDTH
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

        await self._call_node(
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

        self._generation_width = await self._call_node(
            self.replan,
            session_id,
            turn_index,
            hypotheses=refreshed_hypotheses,
        )

        plan_output = await self._call_node(
            self.plan,
            session_id,
            turn_index,
            hypotheses=refreshed_hypotheses,
            concept_state={},
            generation_width=self._generation_width,
        )

        message = await self._call_node(
            self.teach,
            session_id,
            turn_index,
            action=plan_output.winner,
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
