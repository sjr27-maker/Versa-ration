"""BranchStore's cross-session, learner-scoped aggregates for the
portrait page's "does it actually predict me" section.
"""

from uuid import uuid4

import pytest

from probe.models import Branch, BranchStatus


async def _make_generation(branch_store, session_id, turn_index, branches):
    generation = await branch_store.create_generation(
        session_id, turn_index, root_count=sum(1 for b in branches if b.parent_id is None)
    )
    for b in branches:
        b.generation_id = generation.id
    await branch_store.add_branches(branches)
    return generation


@pytest.mark.asyncio(loop_scope="session")
async def test_match_rate_by_session_for_learner_only_counts_resolved_leaves(
    branch_store, transcript, learner_store, concept_graph_id
):
    learner = await learner_store.create(label="track-record-learner")
    session_id = await transcript.create_session(learner.id, concept_graph_id)

    matched = Branch(
        session_id=session_id, turn_index=0, depth=0, depth_label="intent",
        statement="a", predicted_next_turn="pa", plausibility=0.5,
        is_leaf=True, status=BranchStatus.MATCHED, generation_id=uuid4(),
    )
    unmatched = Branch(
        session_id=session_id, turn_index=0, depth=0, depth_label="intent",
        statement="b", predicted_next_turn="pb", plausibility=0.5,
        is_leaf=True, status=BranchStatus.UNMATCHED, generation_id=uuid4(),
    )
    still_open = Branch(
        session_id=session_id, turn_index=1, depth=0, depth_label="intent",
        statement="c", predicted_next_turn="pc", plausibility=0.5,
        is_leaf=True, status=BranchStatus.OPEN, generation_id=uuid4(),
    )
    await _make_generation(branch_store, session_id, 0, [matched, unmatched])
    await _make_generation(branch_store, session_id, 1, [still_open])

    points = await branch_store.match_rate_by_session_for_learner(learner.id)

    assert len(points) == 1
    point = points[0]
    assert point.session_id == session_id
    assert point.total_resolved == 2  # still_open excluded
    assert point.matched_count == 1
    assert point.match_rate == pytest.approx(0.5)


@pytest.mark.asyncio(loop_scope="session")
async def test_match_rate_orders_sessions_chronologically(
    branch_store, transcript, learner_store, concept_graph_id
):
    learner = await learner_store.create(label="track-record-chrono")
    session_a = await transcript.create_session(learner.id, concept_graph_id)
    session_b = await transcript.create_session(learner.id, concept_graph_id)

    leaf_a = Branch(
        session_id=session_a, turn_index=0, depth=0, depth_label="intent",
        statement="a", predicted_next_turn="pa", plausibility=0.5,
        is_leaf=True, status=BranchStatus.MATCHED, generation_id=uuid4(),
    )
    leaf_b = Branch(
        session_id=session_b, turn_index=0, depth=0, depth_label="intent",
        statement="b", predicted_next_turn="pb", plausibility=0.5,
        is_leaf=True, status=BranchStatus.UNMATCHED, generation_id=uuid4(),
    )
    await _make_generation(branch_store, session_a, 0, [leaf_a])
    await _make_generation(branch_store, session_b, 0, [leaf_b])

    points = await branch_store.match_rate_by_session_for_learner(learner.id)

    assert [p.session_id for p in points] == [session_a, session_b]


@pytest.mark.asyncio(loop_scope="session")
async def test_recurring_root_statements_groups_by_exact_text_across_sessions(
    branch_store, transcript, learner_store, concept_graph_id
):
    learner = await learner_store.create(label="track-record-recurring")
    session_a = await transcript.create_session(learner.id, concept_graph_id)
    session_b = await transcript.create_session(learner.id, concept_graph_id)

    recurring_matched = Branch(
        session_id=session_a, turn_index=0, depth=0, depth_label="intent",
        statement="wants an analogy", predicted_next_turn="p", plausibility=0.5,
        is_leaf=True, status=BranchStatus.MATCHED, generation_id=uuid4(),
    )
    recurring_unmatched = Branch(
        session_id=session_b, turn_index=0, depth=0, depth_label="intent",
        statement="wants an analogy", predicted_next_turn="p", plausibility=0.5,
        is_leaf=True, status=BranchStatus.UNMATCHED, generation_id=uuid4(),
    )
    one_off = Branch(
        session_id=session_a, turn_index=0, depth=0, depth_label="intent",
        statement="one-off bet", predicted_next_turn="p", plausibility=0.5,
        is_leaf=True, status=BranchStatus.MATCHED, generation_id=uuid4(),
    )
    await _make_generation(
        branch_store, session_a, 0, [recurring_matched, one_off]
    )
    await _make_generation(branch_store, session_b, 0, [recurring_unmatched])

    recurring = await branch_store.recurring_root_statements_for_learner(learner.id)

    by_statement = {r.statement: r for r in recurring}
    assert by_statement["wants an analogy"].total_count == 2
    assert by_statement["wants an analogy"].matched_count == 1
    assert by_statement["wants an analogy"].match_rate == pytest.approx(0.5)
    assert by_statement["one-off bet"].total_count == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_recurring_root_statements_excludes_non_root_branches(
    branch_store, transcript, learner_store, concept_graph_id
):
    learner = await learner_store.create(label="track-record-nonroot")
    session_id = await transcript.create_session(learner.id, concept_graph_id)

    root = Branch(
        session_id=session_id, turn_index=0, depth=0, depth_label="intent",
        statement="root statement", predicted_next_turn="p", plausibility=0.5,
        is_leaf=False, status=BranchStatus.MATCHED, generation_id=uuid4(),
    )
    child = Branch(
        parent_id=root.id, session_id=session_id, turn_index=0, depth=1,
        depth_label="knowledge_gap", statement="child statement",
        predicted_next_turn="p", plausibility=0.5, is_leaf=True,
        status=BranchStatus.MATCHED, generation_id=uuid4(),
    )
    await _make_generation(branch_store, session_id, 0, [root, child])

    recurring = await branch_store.recurring_root_statements_for_learner(learner.id)

    statements = {r.statement for r in recurring}
    assert "root statement" in statements
    assert "child statement" not in statements


@pytest.mark.asyncio(loop_scope="session")
async def test_match_rate_excludes_option_click_matches_from_the_rate(
    branch_store, transcript, learner_store, concept_graph_id
):
    """A click confirms the student picked an offered option, not
    that the system predicted them — it must not inflate matched_count
    or total_resolved, and must be reported only via the separate
    option_click_count."""
    learner = await learner_store.create(label="track-record-click-split")
    session_id = await transcript.create_session(learner.id, concept_graph_id)

    text_matched = Branch(
        session_id=session_id, turn_index=0, depth=0, depth_label="intent",
        statement="a", predicted_next_turn="pa", plausibility=0.5, is_leaf=True,
        status=BranchStatus.MATCHED, matched_via="text_match", generation_id=uuid4(),
    )
    click_matched = Branch(
        session_id=session_id, turn_index=0, depth=0, depth_label="intent",
        statement="b", predicted_next_turn="pb", plausibility=0.5, is_leaf=True,
        status=BranchStatus.MATCHED, matched_via="option_click", generation_id=uuid4(),
    )
    unmatched = Branch(
        session_id=session_id, turn_index=0, depth=0, depth_label="intent",
        statement="c", predicted_next_turn="pc", plausibility=0.5, is_leaf=True,
        status=BranchStatus.UNMATCHED, generation_id=uuid4(),
    )
    await _make_generation(
        branch_store, session_id, 0, [text_matched, click_matched, unmatched]
    )

    points = await branch_store.match_rate_by_session_for_learner(learner.id)

    assert len(points) == 1
    point = points[0]
    # Denominator/numerator are text_match-only: click_matched excluded
    # from both, not just the numerator.
    assert point.total_resolved == 2
    assert point.matched_count == 1
    assert point.match_rate == pytest.approx(0.5)
    assert point.option_click_count == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_recurring_root_statements_reports_click_confirmations_separately(
    branch_store, transcript, learner_store, concept_graph_id
):
    learner = await learner_store.create(label="track-record-recurring-click")
    session_a = await transcript.create_session(learner.id, concept_graph_id)
    session_b = await transcript.create_session(learner.id, concept_graph_id)

    text_confirmed = Branch(
        session_id=session_a, turn_index=0, depth=0, depth_label="intent",
        statement="wants an analogy", predicted_next_turn="p", plausibility=0.5,
        is_leaf=True, status=BranchStatus.MATCHED, matched_via="text_match",
        generation_id=uuid4(),
    )
    click_confirmed = Branch(
        session_id=session_b, turn_index=0, depth=0, depth_label="intent",
        statement="wants an analogy", predicted_next_turn="p", plausibility=0.5,
        is_leaf=True, status=BranchStatus.MATCHED, matched_via="option_click",
        generation_id=uuid4(),
    )
    await _make_generation(branch_store, session_a, 0, [text_confirmed])
    await _make_generation(branch_store, session_b, 0, [click_confirmed])

    recurring = await branch_store.recurring_root_statements_for_learner(learner.id)

    row = next(r for r in recurring if r.statement == "wants an analogy")
    assert row.total_count == 2
    assert row.matched_count == 1  # text_match only
    assert row.matched_via_click_count == 1  # tracked separately, never summed in


@pytest.mark.asyncio(loop_scope="session")
async def test_track_record_methods_do_not_cross_contaminate_between_learners(
    branch_store, transcript, learner_store, concept_graph_id
):
    learner_a = await learner_store.create(label="track-record-a")
    learner_b = await learner_store.create(label="track-record-b")
    session_a = await transcript.create_session(learner_a.id, concept_graph_id)
    session_b = await transcript.create_session(learner_b.id, concept_graph_id)

    leaf_a = Branch(
        session_id=session_a, turn_index=0, depth=0, depth_label="intent",
        statement="a's bet", predicted_next_turn="p", plausibility=0.5,
        is_leaf=True, status=BranchStatus.MATCHED, generation_id=uuid4(),
    )
    leaf_b = Branch(
        session_id=session_b, turn_index=0, depth=0, depth_label="intent",
        statement="b's bet", predicted_next_turn="p", plausibility=0.5,
        is_leaf=True, status=BranchStatus.MATCHED, generation_id=uuid4(),
    )
    await _make_generation(branch_store, session_a, 0, [leaf_a])
    await _make_generation(branch_store, session_b, 0, [leaf_b])

    points_a = await branch_store.match_rate_by_session_for_learner(learner_a.id)
    recurring_a = await branch_store.recurring_root_statements_for_learner(learner_a.id)

    assert [p.session_id for p in points_a] == [session_a]
    assert {r.statement for r in recurring_a} == {"a's bet"}
