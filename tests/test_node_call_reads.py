"""NodeCallStore's read methods the web UI needs: get_call_for_turn
and get_latest_call (the read side of invariant 2's audit trail)."""

import pytest


@pytest.mark.asyncio(loop_scope="session")
async def test_get_call_for_turn_returns_the_matching_row(
    node_calls, transcript, clean_pool, learner_id
):
    session_id = await transcript.create_session(learner_id)
    await node_calls.record(
        node_name="FinalAnswer", session_id=session_id, turn_index=0,
        input_json={"student_message": "a"}, output_json="answer a",
    )
    await node_calls.record(
        node_name="FinalAnswer", session_id=session_id, turn_index=1,
        input_json={"student_message": "b"}, output_json="answer b",
    )

    call = await node_calls.get_call_for_turn(session_id, 0, "FinalAnswer")

    assert call is not None
    assert call.turn_index == 0
    assert call.input_json == {"student_message": "a"}
    assert call.output_json == "answer a"


@pytest.mark.asyncio(loop_scope="session")
async def test_get_call_for_turn_returns_none_when_absent(
    node_calls, transcript, clean_pool, learner_id
):
    session_id = await transcript.create_session(learner_id)
    assert await node_calls.get_call_for_turn(session_id, 0, "FinalAnswer") is None


@pytest.mark.asyncio(loop_scope="session")
async def test_get_latest_call_ignores_turn_index(
    node_calls, transcript, clean_pool, learner_id
):
    session_id = await transcript.create_session(learner_id)
    await node_calls.record(
        node_name="AssessAndBranch", session_id=session_id, turn_index=0,
        input_json={}, output_json={"needs_branches": False},
    )
    await node_calls.record(
        node_name="AssessAndBranch", session_id=session_id, turn_index=2,
        input_json={}, output_json={"needs_branches": True},
    )

    call = await node_calls.get_latest_call(session_id, "AssessAndBranch")

    assert call is not None
    assert call.turn_index == 2
    assert call.output_json == {"needs_branches": True}
