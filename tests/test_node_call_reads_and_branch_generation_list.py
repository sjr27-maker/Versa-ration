"""Two small read methods the web UI needs: NodeCallStore.get_call_for_turn
/get_latest_call (the read side of invariant 2's audit trail) and
BranchStore.list_by_generation (the full tree, not just open leaves).
"""

from uuid import uuid4

import pytest

from probe.models import Branch, BranchStatus


@pytest.mark.asyncio(loop_scope="session")
async def test_get_call_for_turn_returns_the_matching_row(
    node_calls, transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    await node_calls.record(
        node_name="Plan", session_id=session_id, turn_index=0,
        input_json={"generation_width": 3}, output_json={"winner": "explain"},
    )
    await node_calls.record(
        node_name="Plan", session_id=session_id, turn_index=1,
        input_json={"generation_width": 4}, output_json={"winner": "ask"},
    )

    call = await node_calls.get_call_for_turn(session_id, 0, "Plan")

    assert call is not None
    assert call.turn_index == 0
    assert call.input_json == {"generation_width": 3}
    assert call.output_json == {"winner": "explain"}


@pytest.mark.asyncio(loop_scope="session")
async def test_get_call_for_turn_returns_none_when_absent(
    node_calls, transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    assert await node_calls.get_call_for_turn(session_id, 0, "Plan") is None


@pytest.mark.asyncio(loop_scope="session")
async def test_get_latest_call_ignores_turn_index(
    node_calls, transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    await node_calls.record(
        node_name="BranchResolve", session_id=session_id, turn_index=0,
        input_json={}, output_json={"status": "matched"},
    )
    await node_calls.record(
        node_name="BranchResolve", session_id=session_id, turn_index=2,
        input_json={}, output_json={"status": "unmatched"},
    )

    call = await node_calls.get_latest_call(session_id, "BranchResolve")

    assert call is not None
    assert call.turn_index == 2
    assert call.output_json == {"status": "unmatched"}


@pytest.mark.asyncio(loop_scope="session")
async def test_list_by_generation_returns_every_branch_any_status(
    branch_store, transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    generation = await branch_store.create_generation(session_id, 0, root_count=2)

    open_branch = Branch(
        session_id=session_id, turn_index=0, depth=0, depth_label="intent",
        statement="open", predicted_next_turn="p", plausibility=0.5,
        is_leaf=True, generation_id=generation.id,
    )
    superseded_branch = Branch(
        session_id=session_id, turn_index=0, depth=1, depth_label="gap",
        statement="superseded", predicted_next_turn="p", plausibility=0.5,
        is_leaf=False, status=BranchStatus.SUPERSEDED, generation_id=generation.id,
    )
    await branch_store.add_branches([open_branch, superseded_branch])

    all_branches = await branch_store.list_by_generation(generation.id)

    assert {b.id for b in all_branches} == {open_branch.id, superseded_branch.id}


@pytest.mark.asyncio(loop_scope="session")
async def test_list_by_generation_empty_for_unknown_generation(branch_store):
    assert await branch_store.list_by_generation(uuid4()) == []
