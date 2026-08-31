"""run_interactive's turn-count-gated auto-consolidation on exit (see
loop.py's own comment there) — a session below
MemoryConfig.min_turns_for_cli_auto_consolidation must not feed the
thinking-style detector at all; one at or above it must.
"""

import json

import pytest

from probe.ablation import AblationConfig, ReasoningMode
from probe.llm import StubLLMClient
from probe.loop import SessionLoop
from probe.memory import MemoryConfig
from probe.models import ThinkingStyleStatus


def _make_loop(
    store, transcript, node_calls, concept_graph, learner_overlay, revision_store,
    disambiguation_store, learner_fact_store, thinking_style_store, embedding_client,
    llm, min_turns,
):
    return SessionLoop(
        hypothesis_store=store,
        transcript=transcript,
        node_calls=node_calls,
        concept_graph=concept_graph,
        learner_overlay=learner_overlay,
        revision_store=revision_store,
        llm=llm,
        ablation_config=AblationConfig(reasoning_mode=ReasoningMode.DISAMBIGUATE),
        disambiguation_store=disambiguation_store,
        learner_fact_store=learner_fact_store,
        thinking_style_store=thinking_style_store,
        embedding_client=embedding_client,
        memory_config=MemoryConfig(min_turns_for_cli_auto_consolidation=min_turns),
    )


def _scripted_input(monkeypatch, messages: list[str]):
    it = iter(messages)

    def _fake_input(_prompt: str) -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", _fake_input)


@pytest.mark.asyncio(loop_scope="session")
async def test_below_threshold_does_not_auto_consolidate(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, disambiguation_store,
    learner_fact_store, thinking_style_store, embedding_client, monkeypatch,
):
    llm = StubLLMClient(
        canned={
            "ASSESS:BRANCH": json.dumps({"needs_branches": False, "branches": []}),
            "FINAL:ANSWER": "answer",
            "WRITE:FACT": json.dumps({"situation": "s", "resolution": "r"}),
        }
    )
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay, revision_store,
        disambiguation_store, learner_fact_store, thinking_style_store, embedding_client,
        llm, min_turns=2,
    )
    _scripted_input(monkeypatch, ["only one message"])

    await loop.run_interactive(learner_id, None)

    assert await thinking_style_store.list_by_learner(learner_id) == []


@pytest.mark.asyncio(loop_scope="session")
async def test_at_or_above_threshold_auto_consolidates(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, disambiguation_store,
    learner_fact_store, thinking_style_store, embedding_client, monkeypatch,
):
    llm = StubLLMClient(
        canned={
            "ASSESS:BRANCH": json.dumps({"needs_branches": False, "branches": []}),
            "FINAL:ANSWER": "answer",
            "WRITE:FACT": json.dumps({"situation": "s", "resolution": "r"}),
            "SUMMARIZE:PATH": json.dumps({"summary": "a labeled path"}),
        }
    )
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay, revision_store,
        disambiguation_store, learner_fact_store, thinking_style_store, embedding_client,
        llm, min_turns=2,
    )
    _scripted_input(monkeypatch, ["first message", "second message"])

    await loop.run_interactive(learner_id, None)

    candidates = await thinking_style_store.list_by_learner(learner_id)
    assert len(candidates) == 1
    assert candidates[0].path_summary == "a labeled path"
    assert candidates[0].status is ThinkingStyleStatus.CANDIDATE


@pytest.mark.asyncio(loop_scope="session")
async def test_explicit_consolidate_session_ignores_turn_count(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, disambiguation_store,
    learner_fact_store, thinking_style_store, embedding_client,
):
    """The standalone command / web UI button are deliberate,
    unambiguous triggers — no turn-count gate, unlike run_interactive's
    own auto-trigger."""
    llm = StubLLMClient(
        canned={
            "ASSESS:BRANCH": json.dumps({"needs_branches": False, "branches": []}),
            "FINAL:ANSWER": "answer",
            "WRITE:FACT": json.dumps({"situation": "s", "resolution": "r"}),
            "SUMMARIZE:PATH": json.dumps({"summary": "a labeled path"}),
        }
    )
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay, revision_store,
        disambiguation_store, learner_fact_store, thinking_style_store, embedding_client,
        llm, min_turns=1000,  # would never auto-trigger
    )
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    await loop.handle_turn(session_id, 0, "one single turn")

    result = await loop.consolidate_session(session_id)
    assert result is not None
    assert result.path_summary == "a labeled path"
