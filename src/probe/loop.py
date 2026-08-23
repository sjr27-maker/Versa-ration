"""The session loop: Infer → Update → Replan → Plan → Teach → wait → repeat.

Test and Diagnose exist as node classes (see nodes.py) but aren't wired
into this loop yet. That comes with a later step.

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
from probe.llm import LLMClient
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
from probe.store import HypothesisStore


class SessionLoop:
    def __init__(
        self,
        hypothesis_store: HypothesisStore,
        transcript: TranscriptStore,
        node_calls: NodeCallStore,
        llm: LLMClient,
    ) -> None:
        self._hyp = hypothesis_store
        self._transcript = transcript
        self._node_calls = node_calls
        self.infer = Infer(llm)
        self.plan = Plan()
        self.teach = Teach(llm)
        self.test = Test()
        self.diagnose = Diagnose()
        self.update = Update()
        self.replan = Replan()
        self._generation_width: int = DEFAULT_GENERATION_WIDTH

    async def run_interactive(self) -> UUID:
        session_id = await self._transcript.create_session()
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
        await self._transcript.record_turn(session_id, turn_index, turn_text)

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

        candidates = await self._call_node(
            self.plan,
            session_id,
            turn_index,
            hypotheses=refreshed_hypotheses,
            concept_state={},
        )

        message = await self._call_node(
            self.teach,
            session_id,
            turn_index,
            action=candidates[0],
        )

        return message

    async def _call_node(
        self,
        node: Any,
        session_id: UUID,
        turn_index: int,
        **kwargs: Any,
    ) -> Any:
        output = await node.run(**kwargs)
        await self._node_calls.record(
            node_name=type(node).__name__,
            session_id=session_id,
            turn_index=turn_index,
            input_json=kwargs,
            output_json=output,
        )
        return output
