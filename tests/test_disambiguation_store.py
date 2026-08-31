from uuid import uuid4

import pytest

from probe.models import BranchStatus, DisambiguationBranch, Option, OptionStatus


@pytest.mark.asyncio(loop_scope="session")
async def test_create_turn_roundtrips(disambiguation_store, transcript, learner_id):
    session_id = await transcript.create_session(learner_id)
    turn = await disambiguation_store.create_turn(
        session_id, 0, needs_branches=True, turn_had_direct_answer=False
    )
    assert turn.session_id == session_id
    assert turn.turn_index == 0
    assert turn.needs_branches is True
    assert turn.turn_had_direct_answer is False


@pytest.mark.asyncio(loop_scope="session")
async def test_a_turn_judged_unambiguous_is_still_a_queryable_row(
    disambiguation_store, transcript, learner_id
):
    """See DisambiguationTurn's docstring / CLAUDE.md invariant 9: a
    turn with zero branches is a fact, not a gap."""
    session_id = await transcript.create_session(learner_id)
    turn = await disambiguation_store.create_turn(
        session_id, 0, needs_branches=False, turn_had_direct_answer=True
    )
    reloaded = await disambiguation_store.get_latest_turn(session_id)
    assert reloaded is not None
    assert reloaded.id == turn.id
    assert reloaded.needs_branches is False
    branches = await disambiguation_store.list_branches_for_turn(turn.id)
    assert branches == []


@pytest.mark.asyncio(loop_scope="session")
async def test_add_branches_and_list_for_turn(
    disambiguation_store, transcript, learner_id
):
    session_id = await transcript.create_session(learner_id)
    turn = await disambiguation_store.create_turn(
        session_id, 0, needs_branches=True, turn_had_direct_answer=False
    )
    branches = [
        DisambiguationBranch(
            disambiguation_turn_id=turn.id, session_id=session_id, turn_index=0,
            statement="wants the derivative rule",
        ),
        DisambiguationBranch(
            disambiguation_turn_id=turn.id, session_id=session_id, turn_index=0,
            statement="wants a worked example",
        ),
    ]
    await disambiguation_store.add_branches(branches)

    reloaded = await disambiguation_store.list_branches_for_turn(turn.id)
    assert {b.statement for b in reloaded} == {
        "wants the derivative rule", "wants a worked example",
    }
    assert all(b.status is BranchStatus.OPEN for b in reloaded)


@pytest.mark.asyncio(loop_scope="session")
async def test_mark_matched_and_supersede_open_branches(
    disambiguation_store, transcript, learner_id
):
    session_id = await transcript.create_session(learner_id)
    turn = await disambiguation_store.create_turn(
        session_id, 0, needs_branches=True, turn_had_direct_answer=False
    )
    branches = [
        DisambiguationBranch(
            disambiguation_turn_id=turn.id, session_id=session_id, turn_index=0,
            statement=f"reading {i}",
        )
        for i in range(3)
    ]
    await disambiguation_store.add_branches(branches)

    matched = await disambiguation_store.mark_matched(branches[0].id)
    assert matched.status is BranchStatus.MATCHED

    superseded_count = await disambiguation_store.supersede_open_branches(
        turn.id, exclude_ids=[branches[0].id]
    )
    assert superseded_count == 2

    reloaded = {b.id: b for b in await disambiguation_store.list_branches_for_turn(turn.id)}
    assert reloaded[branches[0].id].status is BranchStatus.MATCHED
    assert reloaded[branches[1].id].status is BranchStatus.SUPERSEDED
    assert reloaded[branches[2].id].status is BranchStatus.SUPERSEDED


@pytest.mark.asyncio(loop_scope="session")
async def test_options_roundtrip_and_click_resolve(
    disambiguation_store, transcript, learner_id
):
    session_id = await transcript.create_session(learner_id)
    turn = await disambiguation_store.create_turn(
        session_id, 0, needs_branches=True, turn_had_direct_answer=False
    )
    branch = DisambiguationBranch(
        disambiguation_turn_id=turn.id, session_id=session_id, turn_index=0,
        statement="wants the derivative rule",
    )
    await disambiguation_store.add_branches([branch])

    option = Option(
        branch_id=branch.id, generation_id=turn.id, session_id=session_id,
        turn_index=0, text="Are you asking about the general power rule?",
    )
    await disambiguation_store.create_options([option])

    listed = await disambiguation_store.list_options_for_turn(turn.id)
    assert len(listed) == 1
    assert listed[0].id == option.id
    assert listed[0].status is OptionStatus.OPEN

    selected = await disambiguation_store.set_option_status(option.id, OptionStatus.SELECTED)
    assert selected.status is OptionStatus.SELECTED

    reloaded_option = await disambiguation_store.get_option(option.id)
    assert reloaded_option.status is OptionStatus.SELECTED


@pytest.mark.asyncio(loop_scope="session")
async def test_supersede_open_options_leaves_selected_untouched(
    disambiguation_store, transcript, learner_id
):
    session_id = await transcript.create_session(learner_id)
    turn = await disambiguation_store.create_turn(
        session_id, 0, needs_branches=True, turn_had_direct_answer=False
    )
    branches = [
        DisambiguationBranch(
            disambiguation_turn_id=turn.id, session_id=session_id, turn_index=0,
            statement=f"reading {i}",
        )
        for i in range(2)
    ]
    await disambiguation_store.add_branches(branches)
    options = [
        Option(branch_id=b.id, generation_id=turn.id, session_id=session_id,
               turn_index=0, text=f"option {i}")
        for i, b in enumerate(branches)
    ]
    await disambiguation_store.create_options(options)
    await disambiguation_store.set_option_status(options[0].id, OptionStatus.SELECTED)

    count = await disambiguation_store.supersede_open_options(turn.id)
    assert count == 1  # only the still-open one

    reloaded = {o.id: o for o in await disambiguation_store.list_options_for_turn(turn.id)}
    assert reloaded[options[0].id].status is OptionStatus.SELECTED
    assert reloaded[options[1].id].status is OptionStatus.SUPERSEDED


@pytest.mark.asyncio(loop_scope="session")
async def test_get_branch_and_get_option_return_none_for_unknown_id(disambiguation_store):
    assert await disambiguation_store.get_branch(uuid4()) is None
    assert await disambiguation_store.get_option(uuid4()) is None
