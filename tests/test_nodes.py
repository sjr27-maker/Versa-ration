from datetime import datetime, timezone
from uuid import uuid4

import pytest

from probe.models import (
    EvidenceRef,
    Hypothesis,
    InferOutput,
    Layer,
    Polarity,
    ProposedEvidence,
    ProposedHypothesis,
    Tier,
)
from probe.nodes import Update


def _hypothesis() -> Hypothesis:
    return Hypothesis(
        layer=Layer.KNOWLEDGE,
        statement="student understands recursion base cases",
        probability=0.4,
        confidence=0.3,
        tier=Tier.ACTIVE,
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_update_applies_reweight_for_each_proposal(
    store, transcript, learner_id, concept_graph_id
):
    hyp_a = _hypothesis()
    hyp_b = _hypothesis()
    await store.add(hyp_a)
    await store.add(hyp_b)

    session_id = await transcript.create_session(learner_id, concept_graph_id)
    turn = await transcript.record_turn(session_id, 0, "shared turn")
    proposals = [
        ProposedEvidence(
            hypothesis_id=hyp_a.id,
            new_probability=0.72,
            new_confidence=0.55,
            evidence_ref=EvidenceRef(
                turn_id=turn,
                polarity=Polarity.SUPPORTING,
                timestamp=datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc),
            ),
        ),
        ProposedEvidence(
            hypothesis_id=hyp_b.id,
            new_probability=0.15,
            new_confidence=0.6,
            evidence_ref=EvidenceRef(
                turn_id=turn,
                polarity=Polarity.CONTRADICTING,
                timestamp=datetime(2026, 8, 24, 12, 1, tzinfo=timezone.utc),
            ),
        ),
    ]

    applied = await Update().run(InferOutput(reweights=proposals), store)

    assert len(applied) == 2

    reloaded_a = await store.get(hyp_a.id)
    assert reloaded_a.probability == pytest.approx(0.72)
    assert reloaded_a.confidence == pytest.approx(0.55)
    assert [e.turn_id for e in reloaded_a.evidence_refs] == [turn]
    assert reloaded_a.counter_evidence_refs == []

    reloaded_b = await store.get(hyp_b.id)
    assert reloaded_b.probability == pytest.approx(0.15)
    assert reloaded_b.confidence == pytest.approx(0.6)
    assert reloaded_b.evidence_refs == []
    assert [e.turn_id for e in reloaded_b.counter_evidence_refs] == [turn]


@pytest.mark.asyncio(loop_scope="session")
async def test_update_appends_rather_than_overwrites_evidence(
    store, transcript, learner_id, concept_graph_id
):
    hyp = _hypothesis()
    await store.add(hyp)

    session_id = await transcript.create_session(learner_id, concept_graph_id)
    first_turn = await transcript.record_turn(session_id, 0, "first")
    await Update().run(
        InferOutput(
            reweights=[
                ProposedEvidence(
                    hypothesis_id=hyp.id,
                    new_probability=0.5,
                    new_confidence=0.4,
                    evidence_ref=EvidenceRef(
                        turn_id=first_turn,
                        polarity=Polarity.SUPPORTING,
                        timestamp=datetime(2026, 8, 24, 10, 0, tzinfo=timezone.utc),
                    ),
                )
            ]
        ),
        store,
    )

    second_turn = await transcript.record_turn(session_id, 1, "second")
    await Update().run(
        InferOutput(
            reweights=[
                ProposedEvidence(
                    hypothesis_id=hyp.id,
                    new_probability=0.8,
                    new_confidence=0.5,
                    evidence_ref=EvidenceRef(
                        turn_id=second_turn,
                        polarity=Polarity.SUPPORTING,
                        timestamp=datetime(2026, 8, 24, 11, 0, tzinfo=timezone.utc),
                    ),
                )
            ]
        ),
        store,
    )

    reloaded = await store.get(hyp.id)
    assert reloaded.probability == pytest.approx(0.8)
    assert reloaded.confidence == pytest.approx(0.5)
    assert [e.turn_id for e in reloaded.evidence_refs] == [first_turn, second_turn]


@pytest.mark.asyncio(loop_scope="session")
async def test_update_with_empty_proposals_is_a_noop(store):
    hyp = _hypothesis()
    await store.add(hyp)

    applied = await Update().run(InferOutput(), store)
    assert applied == []

    reloaded = await store.get(hyp.id)
    assert reloaded.probability == pytest.approx(hyp.probability)
    assert reloaded.confidence == pytest.approx(hyp.confidence)
    assert reloaded.evidence_refs == []


@pytest.mark.asyncio(loop_scope="session")
async def test_update_applies_creates_by_calling_add(
    store, transcript, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    turn = await transcript.record_turn(session_id, 0, "student mentions a new goal")

    create = ProposedHypothesis(
        layer=Layer.GOAL,
        statement="student wants to pass the upcoming calculus exam",
        initial_probability=0.6,
        initial_confidence=0.5,
        evidence_ref=EvidenceRef(turn_id=turn, polarity=Polarity.SUPPORTING),
    )

    applied = await Update().run(InferOutput(creates=[create]), store)

    assert len(applied) == 1
    assert applied[0].turn_id == turn

    active_goals = await store.list_by_learner(learner_id, layer=Layer.GOAL, tier=Tier.ACTIVE)
    assert len(active_goals) == 1
    assert active_goals[0].statement == "student wants to pass the upcoming calculus exam"
    assert active_goals[0].probability == pytest.approx(0.6)
    assert active_goals[0].confidence == pytest.approx(0.5)
    assert active_goals[0].tier is Tier.ACTIVE


@pytest.mark.asyncio(loop_scope="session")
async def test_update_applies_both_reweights_and_creates_in_one_call(
    store, transcript, learner_id, concept_graph_id
):
    hyp = _hypothesis()
    await store.add(hyp)
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    turn = await transcript.record_turn(session_id, 0, "turn")

    reweight = ProposedEvidence(
        hypothesis_id=hyp.id,
        new_probability=0.9,
        new_confidence=0.7,
        evidence_ref=EvidenceRef(turn_id=turn, polarity=Polarity.SUPPORTING),
    )
    create = ProposedHypothesis(
        layer=Layer.MENTAL_MODEL,
        statement="student believes the chain rule multiplies rates",
        initial_probability=0.5,
        initial_confidence=0.4,
        evidence_ref=EvidenceRef(turn_id=turn, polarity=Polarity.SUPPORTING),
    )

    applied = await Update().run(InferOutput(reweights=[reweight], creates=[create]), store)

    assert len(applied) == 2
    reloaded_hyp = await store.get(hyp.id)
    assert reloaded_hyp.probability == pytest.approx(0.9)
    all_learner_hyps = await store.list_by_learner(learner_id)
    assert len(all_learner_hyps) == 2
