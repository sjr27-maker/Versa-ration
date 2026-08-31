"""TranscriptStore's read helpers for the Setup page's resume view:
get_turn and list_sessions_for_learner.
"""

from uuid import uuid4

import pytest


@pytest.mark.asyncio(loop_scope="session")
async def test_get_turn_returns_the_actual_text(transcript, clean_pool, learner_id):
    session_id = await transcript.create_session(learner_id)
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
async def test_list_sessions_for_learner_includes_turn_count_and_null_topic(
    transcript, clean_pool, learner_id
):
    session = await transcript.create_session(learner_id)
    await transcript.record_turn(session, 0, "turn 0")
    await transcript.record_turn(session, 1, "turn 1")
    empty_session = await transcript.create_session(learner_id)

    summaries = await transcript.list_sessions_for_learner(learner_id)
    by_id = {s.session_id: s for s in summaries}

    assert by_id[session].turn_count == 2
    assert by_id[session].topic is None
    assert by_id[empty_session].turn_count == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_list_sessions_for_learner_most_recent_first(
    transcript, clean_pool, learner_id
):
    first = await transcript.create_session(learner_id)
    second = await transcript.create_session(learner_id)

    summaries = await transcript.list_sessions_for_learner(learner_id)

    assert [s.session_id for s in summaries] == [second, first]


@pytest.mark.asyncio(loop_scope="session")
async def test_list_sessions_for_learner_does_not_include_other_learners(
    transcript, clean_pool, learner_store
):
    learner_a = await learner_store.create(label="sessions-list-a")
    learner_b = await learner_store.create(label="sessions-list-b")
    session_a = await transcript.create_session(learner_a.id)
    await transcript.create_session(learner_b.id)

    summaries = await transcript.list_sessions_for_learner(learner_a.id)

    assert [s.session_id for s in summaries] == [session_a]
