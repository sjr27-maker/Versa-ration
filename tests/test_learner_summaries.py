import pytest


@pytest.mark.asyncio(loop_scope="session")
async def test_list_all_with_session_counts_includes_session_count_and_last_session(
    learner_store, transcript, concept_graph_id
):
    learner = await learner_store.create(label="summary-learner")
    await transcript.create_session(learner.id, concept_graph_id)
    await transcript.create_session(learner.id, concept_graph_id)

    summaries = await learner_store.list_all_with_session_counts()
    by_id = {s.learner.id: s for s in summaries}

    assert by_id[learner.id].session_count == 2
    assert by_id[learner.id].last_session_at is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_list_all_with_session_counts_includes_learners_with_no_sessions(
    learner_store,
):
    learner = await learner_store.create(label="no-sessions-learner")

    summaries = await learner_store.list_all_with_session_counts()
    by_id = {s.learner.id: s for s in summaries}

    assert by_id[learner.id].session_count == 0
    assert by_id[learner.id].last_session_at is None


@pytest.mark.asyncio(loop_scope="session")
async def test_list_all_with_session_counts_orders_most_recent_activity_first(
    learner_store, transcript, concept_graph_id
):
    inactive = await learner_store.create(label="inactive-learner")
    active = await learner_store.create(label="active-learner")
    await transcript.create_session(active.id, concept_graph_id)

    summaries = await learner_store.list_all_with_session_counts()
    ids_in_order = [s.learner.id for s in summaries]

    assert ids_in_order.index(active.id) < ids_in_order.index(inactive.id)
