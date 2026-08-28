"""Topic inference (AttachTopic, replacing --topic): a session created
with concept_graph_id=None gets one attached on its first turn — an
existing exact-topic match is resumed, no match seeds a fresh graph —
and any turn past the first with a still-null graph hard-fails.
"""

import json

import pytest

from probe.llm import StubLLMClient
from probe.loop import SessionLoop
from probe.nodes import SessionMissingTopicError


def _loop(
    store, transcript, node_calls, concept_graph, learner_overlay, revision_store,
    llm, diagnostics_store=None,
):
    return SessionLoop(
        hypothesis_store=store,
        transcript=transcript,
        node_calls=node_calls,
        concept_graph=concept_graph,
        learner_overlay=learner_overlay,
        revision_store=revision_store,
        llm=llm,
        diagnostics_store=diagnostics_store,
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_turn_0_attaches_an_existing_graph_on_exact_topic_match(
    store, transcript, node_calls, clean_pool, learner_id,
    concept_graph, learner_overlay, revision_store,
):
    existing = await concept_graph.create_graph(topic="derivatives")
    session_id = await transcript.create_session(learner_id, concept_graph_id=None)

    llm = StubLLMClient(canned={"TOPIC:INFER": json.dumps({"topic": "derivatives"})})
    loop = _loop(store, transcript, node_calls, concept_graph, learner_overlay, revision_store, llm)

    await loop.handle_turn(session_id, 0, "what is a derivative?")

    assert await transcript.get_concept_graph_id(session_id) == existing.id

    async with clean_pool.acquire() as conn:
        names = {
            r["node_name"]
            for r in await conn.fetch(
                "SELECT node_name FROM node_calls WHERE session_id=$1", session_id
            )
        }
    assert "AttachTopic" in names


@pytest.mark.asyncio(loop_scope="session")
async def test_turn_0_seeds_a_fresh_graph_when_no_topic_match(
    store, transcript, node_calls, clean_pool, learner_id,
    concept_graph, learner_overlay, revision_store,
):
    session_id = await transcript.create_session(learner_id, concept_graph_id=None)
    llm = StubLLMClient(canned={"TOPIC:INFER": json.dumps({"topic": "brand-new-topic"})})
    loop = _loop(store, transcript, node_calls, concept_graph, learner_overlay, revision_store, llm)

    await loop.handle_turn(session_id, 0, "tell me about something new")

    attached_id = await transcript.get_concept_graph_id(session_id)
    assert attached_id is not None
    meta = await concept_graph.get_graph(attached_id)
    assert meta.topic == "brand-new-topic"
    concepts = await concept_graph.list_concepts(attached_id)
    assert concepts  # seed_graph's default stub batch actually landed


@pytest.mark.asyncio(loop_scope="session")
async def test_turn_past_the_first_hard_fails_if_still_null(
    store, transcript, node_calls, clean_pool, learner_id,
    concept_graph, learner_overlay, revision_store,
):
    session_id = await transcript.create_session(learner_id, concept_graph_id=None)
    loop = _loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, StubLLMClient(),
    )

    with pytest.raises(SessionMissingTopicError):
        await loop.handle_turn(session_id, 1, "a later turn, no topic ever attached")


@pytest.mark.asyncio(loop_scope="session")
async def test_inferred_topic_and_seeded_new_are_recorded_in_turn_diagnostics(
    store, transcript, node_calls, clean_pool, learner_id,
    concept_graph, learner_overlay, revision_store, diagnostics_store,
):
    """A wrong topic inference must be visible on turn 0 in
    turn_diagnostics, not something discoverable only via a raw
    node_calls query."""
    session_id = await transcript.create_session(learner_id, concept_graph_id=None)
    llm = StubLLMClient(canned={"TOPIC:INFER": json.dumps({"topic": "brand-new-topic"})})
    loop = _loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, llm, diagnostics_store=diagnostics_store,
    )

    await loop.handle_turn(session_id, 0, "tell me about something new")

    diagnostics = await diagnostics_store.get_for_turn(session_id, 0)
    assert diagnostics.inferred_topic == "brand-new-topic"
    assert diagnostics.topic_seeded_new is True


@pytest.mark.asyncio(loop_scope="session")
async def test_inferred_topic_reflects_resume_not_seed_on_an_exact_match(
    store, transcript, node_calls, clean_pool, learner_id,
    concept_graph, learner_overlay, revision_store, diagnostics_store,
):
    await concept_graph.create_graph(topic="derivatives")
    session_id = await transcript.create_session(learner_id, concept_graph_id=None)
    llm = StubLLMClient(canned={"TOPIC:INFER": json.dumps({"topic": "derivatives"})})
    loop = _loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, llm, diagnostics_store=diagnostics_store,
    )

    await loop.handle_turn(session_id, 0, "what is a derivative?")

    diagnostics = await diagnostics_store.get_for_turn(session_id, 0)
    assert diagnostics.inferred_topic == "derivatives"
    assert diagnostics.topic_seeded_new is False


@pytest.mark.asyncio(loop_scope="session")
async def test_inferred_topic_is_none_on_turns_past_the_first(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, diagnostics_store,
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    loop = _loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, StubLLMClient(), diagnostics_store=diagnostics_store,
    )

    await loop.handle_turn(session_id, 0, "first turn, already has a topic")

    diagnostics = await diagnostics_store.get_for_turn(session_id, 0)
    assert diagnostics.inferred_topic is None  # AttachTopic never ran
    assert diagnostics.topic_seeded_new is None


@pytest.mark.asyncio(loop_scope="session")
async def test_attach_topic_failure_on_turn_0_is_recorded_and_turn_still_completes(
    store, transcript, node_calls, clean_pool, learner_id,
    concept_graph, learner_overlay, revision_store, diagnostics_store,
):
    session_id = await transcript.create_session(learner_id, concept_graph_id=None)
    llm = StubLLMClient(canned={"TOPIC:INFER": "not valid json at all"})
    loop = _loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, llm, diagnostics_store=diagnostics_store,
    )

    message = await loop.handle_turn(session_id, 0, "a first message")

    assert message  # the turn still completed
    assert await transcript.get_concept_graph_id(session_id) is None

    diagnostics = await diagnostics_store.get_for_turn(session_id, 0)
    assert diagnostics is not None
    assert any("AttachTopic failed" in w for w in diagnostics.warnings)
