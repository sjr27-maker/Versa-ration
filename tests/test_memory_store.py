from uuid import uuid4

import pytest

from probe.embeddings import EMBEDDING_DIM
from probe.models import LearnerFact, LearnerFactType, ThinkingStyleStatus


def _vec(*, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> list[float]:
    """A vector with a controlled direction in the first 3 dimensions
    (everything else 0) -- cosine similarity between two of these is
    exactly the cosine similarity of the (x, y, z) parts, regardless
    of EMBEDDING_DIM, since the trailing zeros contribute nothing to
    either dot product or magnitude."""
    return [x, y, z] + [0.0] * (EMBEDDING_DIM - 3)


@pytest.mark.asyncio(loop_scope="session")
async def test_add_and_list_by_learner_roundtrips(
    learner_fact_store, transcript, learner_id
):
    session_id = await transcript.create_session(learner_id)
    turn_id = await transcript.record_turn(session_id, 0, "can you help with derivatives")
    fact = LearnerFact(
        learner_id=learner_id,
        session_id=session_id,
        turn_index=0,
        fact_type=LearnerFactType.DIRECT_ANSWER,
        situation="asked what a derivative is",
        resolution="explained it as instantaneous rate of change",
        embedding=_vec(x=1.0),
        source_turn_id=turn_id,
    )
    await learner_fact_store.add(fact)

    reloaded = await learner_fact_store.list_by_learner(learner_id)
    assert len(reloaded) == 1
    assert reloaded[0].situation == "asked what a derivative is"
    assert reloaded[0].resolution == "explained it as instantaneous rate of change"
    assert reloaded[0].fact_type is LearnerFactType.DIRECT_ANSWER
    assert reloaded[0].embedding == _vec(x=1.0)


@pytest.mark.asyncio(loop_scope="session")
async def test_list_by_session_orders_by_turn_index(
    learner_fact_store, transcript, learner_id
):
    session_id = await transcript.create_session(learner_id)
    for i, situation in enumerate(["first", "second", "third"]):
        turn_id = await transcript.record_turn(session_id, i, situation)
        await learner_fact_store.add(
            LearnerFact(
                learner_id=learner_id, session_id=session_id, turn_index=i,
                fact_type=LearnerFactType.DIRECT_ANSWER, situation=situation,
                resolution="ok", embedding=_vec(x=1.0), source_turn_id=turn_id,
            )
        )
    facts = await learner_fact_store.list_by_session(session_id)
    assert [f.situation for f in facts] == ["first", "second", "third"]


@pytest.mark.asyncio(loop_scope="session")
async def test_search_similar_ranks_by_cosine_similarity_and_is_learner_scoped(
    learner_fact_store, transcript, learner_store, learner_id
):
    """Three facts at controlled angles from the query vector -- the
    nearest one must come back first, with a similarity score that
    actually reflects the angle, and a second learner's fact (even one
    embedded identically to the query) must never appear."""
    other_learner_id = (await learner_store.create()).id
    session_id = await transcript.create_session(learner_id)
    other_session_id = await transcript.create_session(other_learner_id)

    near_turn = await transcript.record_turn(session_id, 0, "near")
    far_turn = await transcript.record_turn(session_id, 1, "far")
    opposite_turn = await transcript.record_turn(session_id, 2, "opposite")
    other_turn = await transcript.record_turn(other_session_id, 0, "other learner")

    await learner_fact_store.add(
        LearnerFact(
            learner_id=learner_id, session_id=session_id, turn_index=0,
            fact_type=LearnerFactType.DIRECT_ANSWER, situation="near match",
            resolution="r", embedding=_vec(x=1.0, y=0.05), source_turn_id=near_turn,
        )
    )
    await learner_fact_store.add(
        LearnerFact(
            learner_id=learner_id, session_id=session_id, turn_index=1,
            fact_type=LearnerFactType.DIRECT_ANSWER, situation="far match",
            resolution="r", embedding=_vec(x=0.2, y=1.0), source_turn_id=far_turn,
        )
    )
    await learner_fact_store.add(
        LearnerFact(
            learner_id=learner_id, session_id=session_id, turn_index=2,
            fact_type=LearnerFactType.DIRECT_ANSWER, situation="opposite",
            resolution="r", embedding=_vec(x=-1.0), source_turn_id=opposite_turn,
        )
    )
    await learner_fact_store.add(
        LearnerFact(
            learner_id=other_learner_id, session_id=other_session_id, turn_index=0,
            fact_type=LearnerFactType.DIRECT_ANSWER, situation="belongs to someone else",
            resolution="r", embedding=_vec(x=1.0), source_turn_id=other_turn,
        )
    )

    results = await learner_fact_store.search_similar(learner_id, _vec(x=1.0), limit=5)
    assert [f.situation for f, _sim in results] == ["near match", "far match", "opposite"]
    assert all(f.learner_id == learner_id for f, _sim in results)
    near_sim = results[0][1]
    far_sim = results[1][1]
    opposite_sim = results[2][1]
    assert near_sim > far_sim > opposite_sim
    assert near_sim == pytest.approx(1.0, abs=0.01)
    assert opposite_sim == pytest.approx(-1.0, abs=0.01)


@pytest.mark.asyncio(loop_scope="session")
async def test_search_similar_returns_empty_for_a_learner_with_no_facts(learner_fact_store):
    assert await learner_fact_store.search_similar(uuid4(), _vec(x=1.0)) == []


@pytest.mark.asyncio(loop_scope="session")
async def test_thinking_style_create_and_confirm(thinking_style_store, learner_id):
    session_a, session_b = uuid4(), uuid4()

    candidate = await thinking_style_store.create_candidate(
        learner_id, session_a, "concrete example before abstract rule", _vec(x=1.0)
    )
    assert candidate.confirmation_count == 1
    assert candidate.session_ids == [session_a]
    assert candidate.status is ThinkingStyleStatus.CANDIDATE

    confirmed = await thinking_style_store.confirm(candidate.id, session_b, promotion_threshold=5)
    assert confirmed.confirmation_count == 2
    assert set(confirmed.session_ids) == {session_a, session_b}
    assert confirmed.status is ThinkingStyleStatus.CANDIDATE  # below threshold still


@pytest.mark.asyncio(loop_scope="session")
async def test_thinking_style_promotes_at_threshold_not_before(thinking_style_store, learner_id):
    session_ids = [uuid4() for _ in range(5)]
    candidate = await thinking_style_store.create_candidate(
        learner_id, session_ids[0], "answers accepted directly, reasons asked after", _vec(y=1.0)
    )
    for sid in session_ids[1:4]:  # confirmations 2, 3, 4 -- still below threshold 5
        candidate = await thinking_style_store.confirm(candidate.id, sid, promotion_threshold=5)
        assert candidate.status is ThinkingStyleStatus.CANDIDATE

    candidate = await thinking_style_store.confirm(
        candidate.id, session_ids[4], promotion_threshold=5
    )
    assert candidate.confirmation_count == 5
    assert candidate.status is ThinkingStyleStatus.CONFIRMED

    reloaded = await thinking_style_store.list_confirmed_for_prompt(learner_id)
    assert len(reloaded) == 1
    assert reloaded[0].id == candidate.id


@pytest.mark.asyncio(loop_scope="session")
async def test_list_confirmed_for_prompt_excludes_below_threshold_candidates(
    thinking_style_store, learner_id
):
    """The exact read path any prompt-building code is allowed to use
    -- a candidate/retired row must be structurally unreachable here."""
    below = await thinking_style_store.create_candidate(
        learner_id, uuid4(), "below threshold", _vec(x=1.0)
    )
    for sid in [uuid4(), uuid4()]:
        below = await thinking_style_store.confirm(below.id, sid, promotion_threshold=5)
    assert below.status is ThinkingStyleStatus.CANDIDATE

    assert await thinking_style_store.list_confirmed_for_prompt(learner_id) == []


@pytest.mark.asyncio(loop_scope="session")
async def test_retire_transitions_status_never_deletes(thinking_style_store, learner_id):
    candidate = await thinking_style_store.create_candidate(
        learner_id, uuid4(), "will be retired", _vec(z=1.0)
    )
    retired = await thinking_style_store.retire(candidate.id)
    assert retired.status is ThinkingStyleStatus.RETIRED

    still_there = await thinking_style_store.get(candidate.id)
    assert still_there is not None
    assert still_there.status is ThinkingStyleStatus.RETIRED

    # Retired candidates are excluded from being re-suggested as a
    # search match (see search_similar's own docstring) but are never
    # gone from the table.
    results = await thinking_style_store.search_similar(learner_id, _vec(z=1.0))
    assert results == []
