"""HypothesisStore's two new audit-trail additions needed for the web
UI's per-turn delta panel: reweight() recording the resulting
probability/confidence on the evidence_ref it creates, and retier()
logging a hypothesis_tier_changes row on a real tier transition.
"""

import pytest

from probe.models import EvidenceRef, Hypothesis, Layer, Polarity, Tier


@pytest.mark.asyncio(loop_scope="session")
async def test_reweight_records_the_resulting_probability_and_confidence(
    store, transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    turn_id = await transcript.record_turn(session_id, 0, "a turn")

    hyp = await store.add(
        Hypothesis(
            layer=Layer.KNOWLEDGE,
            statement="stub",
            probability=0.3,
            confidence=0.3,
            tier=Tier.ACTIVE,
        )
    )
    await store.reweight(
        hyp.id, 0.75, 0.65, EvidenceRef(turn_id=turn_id, polarity=Polarity.SUPPORTING)
    )

    refreshed = await store.get(hyp.id)
    new_ref = refreshed.evidence_refs[-1]

    # Through the normal Hypothesis.evidence_refs path — this is what
    # the web UI's delta panel actually reads, not a raw SQL query.
    assert new_ref.resulting_probability == pytest.approx(0.75)
    assert new_ref.resulting_confidence == pytest.approx(0.65)

    async with clean_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT resulting_probability, resulting_confidence "
            "FROM evidence_refs WHERE id = $1",
            new_ref.id,
        )
    assert row["resulting_probability"] == pytest.approx(0.75)
    assert row["resulting_confidence"] == pytest.approx(0.65)


@pytest.mark.asyncio(loop_scope="session")
async def test_add_time_evidence_has_no_resulting_probability(
    store, transcript, clean_pool, learner_id, concept_graph_id
):
    """Evidence attached at add() time isn't the product of a
    reweight() call — no resulting_probability/confidence is
    fabricated for it, unlike evidence reweight() itself creates."""
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    turn_id = await transcript.record_turn(session_id, 0, "a turn")

    hyp = Hypothesis(
        layer=Layer.KNOWLEDGE,
        statement="stub",
        probability=0.5,
        confidence=0.5,
        tier=Tier.ACTIVE,
        evidence_refs=[EvidenceRef(turn_id=turn_id, polarity=Polarity.SUPPORTING)],
    )
    await store.add(hyp)

    async with clean_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT resulting_probability, resulting_confidence "
            "FROM evidence_refs WHERE hypothesis_id = $1",
            hyp.id,
        )
    assert row["resulting_probability"] is None
    assert row["resulting_confidence"] is None


@pytest.mark.asyncio(loop_scope="session")
async def test_retier_logs_a_tier_change_only_on_a_real_transition(store):
    hyp = await store.add(
        Hypothesis(
            layer=Layer.KNOWLEDGE,
            statement="stub",
            probability=0.5,
            confidence=0.5,
            tier=Tier.DORMANT,
        )
    )

    await store.retier(hyp.id, Tier.ACTIVE)  # real transition
    await store.retier(hyp.id, Tier.ACTIVE)  # no-op, same tier

    changes = await store.list_tier_changes(hyp.id)

    assert len(changes) == 1
    assert changes[0].old_tier is Tier.DORMANT
    assert changes[0].new_tier is Tier.ACTIVE


@pytest.mark.asyncio(loop_scope="session")
async def test_resurrect_is_visible_in_the_tier_change_log(store):
    hyp = await store.add(
        Hypothesis(
            layer=Layer.KNOWLEDGE,
            statement="stub",
            probability=0.5,
            confidence=0.5,
            tier=Tier.ACTIVE,
        )
    )
    await store.retier(hyp.id, Tier.DORMANT)
    await store.resurrect(hyp.id)

    changes = await store.list_tier_changes(hyp.id)

    assert [(c.old_tier, c.new_tier) for c in changes] == [
        (Tier.ACTIVE, Tier.DORMANT),
        (Tier.DORMANT, Tier.ACTIVE),
    ]


@pytest.mark.asyncio(loop_scope="session")
async def test_list_tier_changes_is_empty_for_a_hypothesis_never_retiered(store):
    hyp = await store.add(
        Hypothesis(
            layer=Layer.KNOWLEDGE,
            statement="stub",
            probability=0.5,
            confidence=0.5,
            tier=Tier.ACTIVE,
        )
    )
    assert await store.list_tier_changes(hyp.id) == []
