"""SessionLoop's retry_count snapshot: _total_retry_count(self._tiers)
is read before and after handle_turn(), and the delta lands on that
turn's TurnDiagnostics row. This tests the snapshot/delta mechanism and
its threading into turn_diagnostics specifically — GeminiLLMClient's
own retry loop is covered separately in test_gemini_llm_client.py.
"""

from __future__ import annotations

import pytest

from probe.diagnostics import TurnDiagnosticsStore
from probe.llm import StubLLMClient
from probe.loop import SessionLoop


class _RetryCountingStub:
    """A LLMClient whose retry_count increments on chosen call numbers
    (1-indexed, across the client's whole lifetime) — lets a test
    control precisely which turn's before/after window contains
    "retries" without needing a real retryable error."""

    def __init__(self, retries_on_call_number: set[int]) -> None:
        self._inner = StubLLMClient()
        self.retry_count = 0
        self._call_number = 0
        self._retries_on_call_number = retries_on_call_number

    async def complete(self, prompt: str) -> str:
        self._call_number += 1
        if self._call_number in self._retries_on_call_number:
            self.retry_count += 1
        return await self._inner.complete(prompt)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_turn_with_no_retries_records_zero(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store,
):
    diagnostics_store = TurnDiagnosticsStore(clean_pool)
    llm = _RetryCountingStub(retries_on_call_number=set())
    loop = SessionLoop(
        hypothesis_store=store,
        transcript=transcript,
        node_calls=node_calls,
        concept_graph=concept_graph,
        learner_overlay=learner_overlay,
        revision_store=revision_store,
        llm=llm,
        diagnostics_store=diagnostics_store,
    )
    session_id = await transcript.create_session(learner_id, concept_graph_id)

    await loop.handle_turn(session_id, 0, "turn zero")

    diag = await diagnostics_store.get_for_turn(session_id, 0)
    assert diag.retry_count == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_retries_during_a_turn_are_attributed_to_that_turns_diagnostics(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store,
):
    diagnostics_store = TurnDiagnosticsStore(clean_pool)
    # Every LLM call in this default (untiered) SessionLoop shares one
    # llm instance across fast/capable/best, so call numbers are global
    # across the whole session, not per-tier.
    llm = _RetryCountingStub(retries_on_call_number={2, 3})
    loop = SessionLoop(
        hypothesis_store=store,
        transcript=transcript,
        node_calls=node_calls,
        concept_graph=concept_graph,
        learner_overlay=learner_overlay,
        revision_store=revision_store,
        llm=llm,
        diagnostics_store=diagnostics_store,
    )
    session_id = await transcript.create_session(learner_id, concept_graph_id)

    # Turn 0 makes several calls (Diagnose/Infer/Plan/Teach); calls #2
    # and #3 land inside it, so its window should see both retries.
    await loop.handle_turn(session_id, 0, "turn zero")
    diag0 = await diagnostics_store.get_for_turn(session_id, 0)
    assert diag0.retry_count == 2
    assert llm.retry_count == 2  # cumulative total, matches this turn's delta

    # Turn 1 makes calls after both retries already happened — its own
    # window should see none of turn 0's retries counted again.
    await loop.handle_turn(session_id, 1, "turn one")
    diag1 = await diagnostics_store.get_for_turn(session_id, 1)
    assert diag1.retry_count == 0
    assert llm.retry_count == 2  # unchanged — no new retries this turn


@pytest.mark.asyncio(loop_scope="session")
async def test_retry_count_defaults_to_zero_against_plain_stub_llm_client(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store,
):
    """StubLLMClient has no retry_count attribute at all —
    getattr(..., 0) in _total_retry_count must not raise, and the
    result must be 0."""
    diagnostics_store = TurnDiagnosticsStore(clean_pool)
    loop = SessionLoop(
        hypothesis_store=store,
        transcript=transcript,
        node_calls=node_calls,
        concept_graph=concept_graph,
        learner_overlay=learner_overlay,
        revision_store=revision_store,
        llm=StubLLMClient(),
        diagnostics_store=diagnostics_store,
    )
    session_id = await transcript.create_session(learner_id, concept_graph_id)

    await loop.handle_turn(session_id, 0, "turn zero")

    diag = await diagnostics_store.get_for_turn(session_id, 0)
    assert diag.retry_count == 0
