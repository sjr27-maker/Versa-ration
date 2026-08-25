import pytest

from probe.models import (
    ConceptNode,
    EvidenceRef,
    Hypothesis,
    Layer,
    OverlayState,
    Polarity,
    RevisionStatus,
    Tier,
    WorldModelRevision,
)
from probe.portrait import build_portrait


async def _hyp_with_evidence(store, transcript, session_id, turn_index, **kwargs):
    hyp = Hypothesis(**kwargs)
    await store.add(hyp)
    turn_id = await transcript.record_turn(session_id, turn_index, "turn")
    await store.reweight(
        hyp.id,
        hyp.probability,
        hyp.confidence,
        EvidenceRef(turn_id=turn_id, polarity=Polarity.SUPPORTING),
    )
    return hyp, turn_id


@pytest.mark.asyncio(loop_scope="session")
async def test_portrait_reports_correct_counts_across_tiers_layers_and_sessions(
    store,
    transcript,
    concept_graph,
    learner_overlay,
    revision_store,
    learner_store,
):
    learner = await learner_store.create(label="fixture-learner")
    other_learner = await learner_store.create(label="other-learner")
    graph_meta = await concept_graph.create_graph(topic="fixture-topic")
    concept_graph_id = graph_meta.id

    session_1 = await transcript.create_session(learner.id, concept_graph_id)
    session_2 = await transcript.create_session(learner.id, concept_graph_id)
    other_session = await transcript.create_session(other_learner.id, concept_graph_id)

    # --- top hypothesis per layer -----------------------------------
    goal_hyp, _ = await _hyp_with_evidence(
        store, transcript, session_1, 0,
        layer=Layer.GOAL, statement="wants to master recursion",
        probability=0.9, confidence=0.7, tier=Tier.ACTIVE,
    )
    # Lower-probability KNOWLEDGE hypothesis, present only to prove the
    # portrait picks the higher one (know_high below), not just any one.
    await _hyp_with_evidence(
        store, transcript, session_1, 1,
        layer=Layer.KNOWLEDGE, statement="knows loops",
        probability=0.6, confidence=0.5, tier=Tier.ACTIVE,
    )
    know_high, know_high_turn = await _hyp_with_evidence(
        store, transcript, session_1, 2,
        layer=Layer.KNOWLEDGE, statement="knows recursion base cases",
        probability=0.8, confidence=0.6, tier=Tier.ACTIVE,
    )
    mental_hyp, _ = await _hyp_with_evidence(
        store, transcript, session_1, 3,
        layer=Layer.MENTAL_MODEL, statement="thinks closures copy by value",
        probability=0.7, confidence=0.4, tier=Tier.ACTIVE,
    )
    cog_hyp, _ = await _hyp_with_evidence(
        store, transcript, session_2, 0,
        layer=Layer.COGNITIVE_STATE, statement="shows signs of fatigue",
        probability=0.5, confidence=0.3, tier=Tier.ACTIVE,
    )
    # No TEACHING-layer hypothesis at all -> top must be None for it.

    # --- non-active tiers, for the boundedness/plateau counts -------
    for i in range(2):
        await _hyp_with_evidence(
            store, transcript, session_2, 10 + i,
            layer=Layer.KNOWLEDGE, statement=f"background {i}",
            probability=0.3, confidence=0.2, tier=Tier.BACKGROUND,
        )
    await _hyp_with_evidence(
        store, transcript, session_2, 20,
        layer=Layer.KNOWLEDGE, statement="dormant one",
        probability=0.2, confidence=0.2, tier=Tier.DORMANT,
    )
    for i in range(3):
        await _hyp_with_evidence(
            store, transcript, session_2, 30 + i,
            layer=Layer.KNOWLEDGE, statement=f"archived {i}",
            probability=0.1, confidence=0.1, tier=Tier.ARCHIVED,
        )

    # A hypothesis with no evidence at all: not attributable to any
    # learner, must not appear in any count.
    await store.add(
        Hypothesis(
            layer=Layer.KNOWLEDGE,
            statement="orphan, no evidence",
            probability=0.99,
            confidence=0.99,
            tier=Tier.ACTIVE,
        )
    )

    # A hypothesis evidenced only from the OTHER learner's session:
    # must not leak into this learner's counts.
    await _hyp_with_evidence(
        store, transcript, other_session, 0,
        layer=Layer.KNOWLEDGE, statement="belongs to someone else",
        probability=0.95, confidence=0.9, tier=Tier.ACTIVE,
    )

    # --- overlay ------------------------------------------------------
    await concept_graph.add_concept(
        ConceptNode(concept_graph_id=concept_graph_id, id="closures", name="Closures")
    )
    await concept_graph.add_concept(
        ConceptNode(concept_graph_id=concept_graph_id, id="loops", name="Loops")
    )
    await learner_overlay.set_state(
        learner.id, concept_graph_id, "closures", OverlayState.PARTIAL, 0.4
    )
    await learner_overlay.set_state(
        learner.id, concept_graph_id, "loops", OverlayState.KNOWN, 0.9
    )

    # --- world-model revisions -----------------------------------------
    pending = await revision_store.propose(
        WorldModelRevision(
            concept_graph_id=concept_graph_id,
            concept_id="closures",
            proposed_change="misconceptions list looks incomplete",
            evidence_refs=[
                EvidenceRef(turn_id=know_high_turn, polarity=Polarity.SUPPORTING)
            ],
            confidence=0.6,
        )
    )
    to_be_approved = await revision_store.propose(
        WorldModelRevision(
            concept_graph_id=concept_graph_id,
            concept_id="loops",
            proposed_change="already resolved, must not show as pending",
            evidence_refs=[
                EvidenceRef(turn_id=know_high_turn, polarity=Polarity.SUPPORTING)
            ],
            confidence=0.5,
        )
    )
    await revision_store.approve(to_be_approved.id, {"name": "Loops (renamed)"})

    other_turn = await transcript.record_turn(other_session, 1, "other turn")
    await revision_store.propose(
        WorldModelRevision(
            concept_graph_id=concept_graph_id,
            concept_id="closures",
            proposed_change="from a different learner entirely",
            evidence_refs=[
                EvidenceRef(turn_id=other_turn, polarity=Polarity.SUPPORTING)
            ],
            confidence=0.5,
        )
    )

    # --- build and assert -----------------------------------------------
    report = await build_portrait(
        learner.id, store, transcript, concept_graph, learner_overlay, revision_store
    )

    assert report.learner_id == learner.id
    assert report.session_count == 2

    by_layer = {top.layer: top.hypothesis for top in report.top_hypotheses}
    assert by_layer[Layer.GOAL].id == goal_hyp.id
    assert by_layer[Layer.KNOWLEDGE].id == know_high.id  # 0.8 beats 0.6
    assert by_layer[Layer.MENTAL_MODEL].id == mental_hyp.id
    assert by_layer[Layer.COGNITIVE_STATE].id == cog_hyp.id
    assert by_layer[Layer.TEACHING] is None

    assert report.tier_counts["active"] == 5  # goal, 2xknowledge, mental, cog
    assert report.tier_counts["background"] == 2
    assert report.tier_counts["dormant"] == 1
    assert report.tier_counts["archived"] == 3

    overlay_by_concept = {e.concept_id: e for e in report.overlay}
    assert set(overlay_by_concept) == {"closures", "loops"}
    assert overlay_by_concept["closures"].concept_name == "Closures"
    assert overlay_by_concept["closures"].entry.state is OverlayState.PARTIAL
    assert overlay_by_concept["closures"].entry.confidence == pytest.approx(0.4)
    assert overlay_by_concept["loops"].entry.state is OverlayState.KNOWN

    assert [r.id for r in report.pending_revisions] == [pending.id]
    assert report.pending_revisions[0].status is RevisionStatus.PENDING


@pytest.mark.asyncio(loop_scope="session")
async def test_portrait_on_a_fresh_learner_has_no_data(
    store, transcript, concept_graph, learner_overlay, revision_store, learner_store
):
    learner = await learner_store.create(label="brand-new")

    report = await build_portrait(
        learner.id, store, transcript, concept_graph, learner_overlay, revision_store
    )

    assert report.session_count == 0
    assert all(top.hypothesis is None for top in report.top_hypotheses)
    assert all(count == 0 for count in report.tier_counts.values())
    assert report.overlay == []
    assert report.pending_revisions == []
