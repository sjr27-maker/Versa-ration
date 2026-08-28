"""TranscriptStore's additions for topic inference and the Setup
page's resume view: attach_concept_graph_id (set-once), get_turn, and
list_sessions_for_learner.
"""

from uuid import uuid4

import pytest


@pytest.mark.asyncio(loop_scope="session")
async def test_attach_concept_graph_id_sets_a_null_graph(
    transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id=None)
    assert await transcript.get_concept_graph_id(session_id) is None

    await transcript.attach_concept_graph_id(session_id, concept_graph_id)

    assert await transcript.get_concept_graph_id(session_id) == concept_graph_id


@pytest.mark.asyncio(loop_scope="session")
async def test_attach_concept_graph_id_raises_if_already_set(
    transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)

    with pytest.raises(ValueError):
        await transcript.attach_concept_graph_id(session_id, uuid4())


@pytest.mark.asyncio(loop_scope="session")
async def test_attach_concept_graph_id_raises_for_unknown_session(clean_pool):
    from probe.audit import TranscriptStore

    transcript = TranscriptStore(clean_pool)
    with pytest.raises(KeyError):
        await transcript.attach_concept_graph_id(uuid4(), uuid4())


@pytest.mark.asyncio(loop_scope="session")
async def test_get_turn_returns_the_actual_text(
    transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    turn_id = await transcript.record_turn(session_id, 0, "what is a derivative?")

    turn = await transcript.get_turn(turn_id)

    assert turn is not None
    assert turn.text == "what is a derivative?"
    assert turn.session_id == session_id
    assert turn.turn_index == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_get_turn_returns_none_for_unknown_turn(clean_pool):
    from probe.audit import TranscriptStore

    transcript = TranscriptStore(clean_pool)
    assert await transcript.get_turn(uuid4()) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_list_sessions_for_learner_includes_topic_and_turn_count(
    transcript, clean_pool, learner_id, concept_graph, concept_graph_id
):
    session_with_topic = await transcript.create_session(learner_id, concept_graph_id)
    await transcript.record_turn(session_with_topic, 0, "turn 0")
    await transcript.record_turn(session_with_topic, 1, "turn 1")

    session_no_topic = await transcript.create_session(learner_id, concept_graph_id=None)

    summaries = await transcript.list_sessions_for_learner(learner_id)
    by_id = {s.session_id: s for s in summaries}

    graph_meta = await concept_graph.get_graph(concept_graph_id)
    assert by_id[session_with_topic].topic == graph_meta.topic
    assert by_id[session_with_topic].turn_count == 2
    assert by_id[session_no_topic].topic is None
    assert by_id[session_no_topic].turn_count == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_list_sessions_for_learner_most_recent_first(
    transcript, clean_pool, learner_id, concept_graph_id
):
    first = await transcript.create_session(learner_id, concept_graph_id)
    second = await transcript.create_session(learner_id, concept_graph_id)

    summaries = await transcript.list_sessions_for_learner(learner_id)

    assert [s.session_id for s in summaries] == [second, first]


@pytest.mark.asyncio(loop_scope="session")
async def test_list_sessions_for_learner_does_not_include_other_learners(
    transcript, clean_pool, learner_store, concept_graph_id
):
    learner_a = await learner_store.create(label="sessions-list-a")
    learner_b = await learner_store.create(label="sessions-list-b")
    session_a = await transcript.create_session(learner_a.id, concept_graph_id)
    await transcript.create_session(learner_b.id, concept_graph_id)

    summaries = await transcript.list_sessions_for_learner(learner_a.id)

    assert [s.session_id for s in summaries] == [session_a]
