import json

import pytest

from probe.grounding import GroundConcept
from probe.llm import StubLLMClient
from probe.mismatch import MismatchDetector
from probe.models import ConceptNode, Hypothesis, Layer, OverlayState, Tier
from probe.nodes import DIAGNOSE_MISMATCH_THRESHOLD, Diagnose


async def _seed_concept(concept_graph, concept_graph_id, id_="closures") -> str:
    await concept_graph.add_concept(
        ConceptNode(
            concept_graph_id=concept_graph_id,
            id=id_,
            name="Closures",
            common_misconceptions=["closures copy the value at definition time"],
        )
    )
    return id_


async def _seed_linked_hypothesis(store, concept_graph_id, concept_id: str) -> Hypothesis:
    hyp = Hypothesis(
        layer=Layer.MENTAL_MODEL,
        statement="student believes closures copy the value, not the variable",
        probability=0.5,
        confidence=0.4,
        tier=Tier.ACTIVE,
    )
    await store.add(hyp)
    await store.link_concept(hyp.id, concept_graph_id, concept_id)
    return hyp


def _diagnose(
    llm, store, revision_store, concept_graph, learner_overlay, transcript
) -> Diagnose:
    return Diagnose(
        mismatch_detector=MismatchDetector(llm),
        ground_concept=GroundConcept(llm),
        hypothesis_store=store,
        revision_store=revision_store,
        concept_graph=concept_graph,
        learner_overlay=learner_overlay,
        transcript=transcript,
    )


def _grounded_and_mismatched(concept_id: str, suggested_cause: str, confidence: float) -> dict:
    return {
        "GROUND:CONCEPT": json.dumps(
            {"concept_id": concept_id, "confidence": 0.9}
        ),
        "MISMATCH:DETECT": json.dumps(
            {
                "mismatch": True,
                "learner_claim": "closures copy the value",
                "world_claim": "closures capture by reference",
                "confidence": confidence,
                "suggested_cause": suggested_cause,
            }
        ),
    }


@pytest.mark.asyncio(loop_scope="session")
async def test_learner_misconception_routes_to_hypothesis_evidence_not_revision(
    store,
    transcript,
    concept_graph,
    concept_graph_id,
    learner_overlay,
    revision_store,
    learner_id,
):
    concept_id = await _seed_concept(concept_graph, concept_graph_id)
    hyp = await _seed_linked_hypothesis(store, concept_graph_id, concept_id)
    await learner_overlay.set_state(
        learner_id, concept_graph_id, concept_id, OverlayState.PARTIAL, 0.5
    )
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    turn_id = await transcript.record_turn(session_id, 0, "turn zero")

    llm = StubLLMClient(
        canned=_grounded_and_mismatched(concept_id, "learner_misconception", 0.8)
    )
    diagnose = _diagnose(llm, store, revision_store, concept_graph, learner_overlay, transcript)

    result = await diagnose.run(
        response="I think it copies the value",
        expectation="closures capture by reference",
        session_id=session_id,
        turn_id=turn_id,
    )

    assert result["grounding"] == {"concept_id": concept_id, "confidence": pytest.approx(0.9)}
    # 1 grounding call + 1 mismatch-detection call.
    assert result["llm_call_count"] == 2
    assert result["action_taken"] == "hypothesis_reweighted"
    assert result["reweighted_hypothesis_ids"] == [str(hyp.id)]
    assert result["revision_id"] is None
    assert await revision_store.list_pending() == []

    reweighted = await store.get(hyp.id)
    assert reweighted.probability > hyp.probability
    assert reweighted.confidence > hyp.confidence
    assert len(reweighted.evidence_refs) == 1
    assert reweighted.evidence_refs[0].turn_id == turn_id


@pytest.mark.asyncio(loop_scope="session")
async def test_possible_world_model_error_proposes_revision_and_leaves_concept_untouched(
    store,
    transcript,
    concept_graph,
    concept_graph_id,
    learner_overlay,
    revision_store,
    learner_id,
):
    concept_id = await _seed_concept(concept_graph, concept_graph_id)
    hyp = await _seed_linked_hypothesis(store, concept_graph_id, concept_id)
    await learner_overlay.set_state(
        learner_id, concept_graph_id, concept_id, OverlayState.PARTIAL, 0.5
    )
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    turn_id = await transcript.record_turn(session_id, 0, "turn zero")

    llm = StubLLMClient(
        canned=_grounded_and_mismatched(concept_id, "possible_world_model_error", 0.9)
    )
    diagnose = _diagnose(llm, store, revision_store, concept_graph, learner_overlay, transcript)

    concept_before = await concept_graph.get_concept(concept_graph_id, concept_id)

    result = await diagnose.run(
        response="closures capture by reference",
        expectation="closures capture by reference",
        session_id=session_id,
        turn_id=turn_id,
    )

    assert result["action_taken"] == "revision_proposed"
    assert result["revision_id"] is not None
    assert result["reweighted_hypothesis_ids"] == []
    assert result["llm_call_count"] == 2

    # The matched hypothesis was NOT reweighted — cause was world-model,
    # not learner.
    untouched_hyp = await store.get(hyp.id)
    assert untouched_hyp.probability == pytest.approx(hyp.probability)
    assert untouched_hyp.evidence_refs == []

    pending = await revision_store.list_pending(
        concept_graph_id=concept_graph_id, concept_id=concept_id
    )
    assert len(pending) == 1
    revision = pending[0]
    assert str(revision.id) == result["revision_id"]
    assert revision.applied_field_updates is None

    # ConceptNode is untouched until a human calls approve().
    concept_after = await concept_graph.get_concept(concept_graph_id, concept_id)
    assert concept_after == concept_before

    approved = await revision_store.approve(
        revision.id,
        {
            "common_misconceptions": [
                "closures copy the value at definition time",
                "closures capture by reference, confirmed by learner interaction",
            ]
        },
    )
    assert approved.applied_field_updates is not None

    concept_final = await concept_graph.get_concept(concept_graph_id, concept_id)
    assert concept_final.common_misconceptions != concept_before.common_misconceptions


@pytest.mark.asyncio(loop_scope="session")
async def test_low_confidence_world_model_error_takes_no_action(
    store,
    transcript,
    concept_graph,
    concept_graph_id,
    learner_overlay,
    revision_store,
    learner_id,
):
    concept_id = await _seed_concept(concept_graph, concept_graph_id)
    await _seed_linked_hypothesis(store, concept_graph_id, concept_id)
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    turn_id = await transcript.record_turn(session_id, 0, "turn zero")

    llm = StubLLMClient(
        canned=_grounded_and_mismatched(concept_id, "possible_world_model_error", 0.2)
    )
    diagnose = _diagnose(llm, store, revision_store, concept_graph, learner_overlay, transcript)

    result = await diagnose.run(
        response="x",
        expectation="y",
        session_id=session_id,
        turn_id=turn_id,
    )

    assert result["action_taken"] == "none"
    assert result["revision_id"] is None
    # The confidence the threshold silently dropped is still on record —
    # this is exactly what lets a later review see what got skipped.
    assert result["mismatch"] is not None
    assert result["mismatch"]["confidence"] == pytest.approx(0.2)
    assert result["mismatch"]["suggested_cause"] == "possible_world_model_error"
    assert await revision_store.list_pending() == []


@pytest.mark.asyncio(loop_scope="session")
async def test_confidence_at_exactly_threshold_proposes_a_revision(
    store,
    transcript,
    concept_graph,
    concept_graph_id,
    learner_overlay,
    revision_store,
    learner_id,
):
    concept_id = await _seed_concept(concept_graph, concept_graph_id)
    await _seed_linked_hypothesis(store, concept_graph_id, concept_id)
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    turn_id = await transcript.record_turn(session_id, 0, "turn zero")

    llm = StubLLMClient(
        canned=_grounded_and_mismatched(
            concept_id, "possible_world_model_error", DIAGNOSE_MISMATCH_THRESHOLD
        )
    )
    diagnose = _diagnose(llm, store, revision_store, concept_graph, learner_overlay, transcript)

    result = await diagnose.run(
        response="x", expectation="y", session_id=session_id, turn_id=turn_id
    )

    assert result["action_taken"] == "revision_proposed"
    assert result["revision_id"] is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_confidence_just_below_threshold_does_not_propose(
    store,
    transcript,
    concept_graph,
    concept_graph_id,
    learner_overlay,
    revision_store,
    learner_id,
):
    concept_id = await _seed_concept(concept_graph, concept_graph_id)
    await _seed_linked_hypothesis(store, concept_graph_id, concept_id)
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    turn_id = await transcript.record_turn(session_id, 0, "turn zero")

    llm = StubLLMClient(
        canned=_grounded_and_mismatched(
            concept_id,
            "possible_world_model_error",
            DIAGNOSE_MISMATCH_THRESHOLD - 0.01,
        )
    )
    diagnose = _diagnose(llm, store, revision_store, concept_graph, learner_overlay, transcript)

    result = await diagnose.run(
        response="x", expectation="y", session_id=session_id, turn_id=turn_id
    )

    assert result["action_taken"] == "none"
    assert result["revision_id"] is None
    assert result["mismatch"]["confidence"] == pytest.approx(
        DIAGNOSE_MISMATCH_THRESHOLD - 0.01
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_no_mismatch_returns_none_action(
    store,
    transcript,
    concept_graph,
    concept_graph_id,
    learner_overlay,
    revision_store,
    learner_id,
):
    concept_id = await _seed_concept(concept_graph, concept_graph_id)
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    turn_id = await transcript.record_turn(session_id, 0, "turn zero")

    llm = StubLLMClient(
        canned={
            "GROUND:CONCEPT": json.dumps(
                {"concept_id": concept_id, "confidence": 0.9}
            )
        }
    )  # default MISMATCH:DETECT -> {"mismatch": false}
    diagnose = _diagnose(llm, store, revision_store, concept_graph, learner_overlay, transcript)

    result = await diagnose.run(
        response="anything",
        expectation="anything",
        session_id=session_id,
        turn_id=turn_id,
    )

    assert result["action_taken"] == "none"
    assert result["mismatch"] is None
    # No overlay entry and no linked hypothesis were seeded, so
    # MismatchDetector has nothing to compare and returns early without
    # calling the LLM at all — only the grounding call happened.
    assert result["llm_call_count"] == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_grounded_and_comparable_but_llm_reports_no_mismatch(
    store,
    transcript,
    concept_graph,
    concept_graph_id,
    learner_overlay,
    revision_store,
    learner_id,
):
    concept_id = await _seed_concept(concept_graph, concept_graph_id)
    await _seed_linked_hypothesis(store, concept_graph_id, concept_id)
    await learner_overlay.set_state(
        learner_id, concept_graph_id, concept_id, OverlayState.KNOWN, 0.9
    )
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    turn_id = await transcript.record_turn(session_id, 0, "turn zero")

    llm = StubLLMClient(
        canned={
            "GROUND:CONCEPT": json.dumps(
                {"concept_id": concept_id, "confidence": 0.9}
            )
        }
    )  # default MISMATCH:DETECT -> {"mismatch": false}
    diagnose = _diagnose(llm, store, revision_store, concept_graph, learner_overlay, transcript)

    result = await diagnose.run(
        response="closures capture by reference, as expected",
        expectation="anything",
        session_id=session_id,
        turn_id=turn_id,
    )

    assert result["action_taken"] == "none"
    assert result["mismatch"] is None
    # This time there WAS something to compare (overlay + linked
    # hypothesis), so MismatchDetector did call the LLM — it just came
    # back "no mismatch".
    assert result["llm_call_count"] == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_ungrounded_response_skips_mismatch_check_gracefully(
    store,
    transcript,
    concept_graph,
    concept_graph_id,
    learner_overlay,
    revision_store,
    learner_id,
):
    await _seed_concept(concept_graph, concept_graph_id)
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    turn_id = await transcript.record_turn(session_id, 0, "turn zero")

    # Default GROUND:CONCEPT -> {"concept_id": null, "confidence": 0.0}
    llm = StubLLMClient()
    diagnose = _diagnose(llm, store, revision_store, concept_graph, learner_overlay, transcript)

    result = await diagnose.run(
        response="I don't get this",
        expectation="anything",
        session_id=session_id,
        turn_id=turn_id,
    )

    assert result["grounding"] == {"concept_id": None, "confidence": 0.0}
    assert result["action_taken"] == "none"
    assert result["mismatch"] is None
    assert "did not clearly ground" in result["notes"]
    # Grounding failed, so mismatch detection was never invoked.
    assert result["llm_call_count"] == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_hallucinated_concept_id_falls_back_to_ungrounded(
    store,
    transcript,
    concept_graph,
    concept_graph_id,
    learner_overlay,
    revision_store,
    learner_id,
):
    await _seed_concept(concept_graph, concept_graph_id)
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    turn_id = await transcript.record_turn(session_id, 0, "turn zero")

    llm = StubLLMClient(
        canned={
            "GROUND:CONCEPT": json.dumps(
                {"concept_id": "not_a_real_concept", "confidence": 0.95}
            )
        }
    )
    diagnose = _diagnose(llm, store, revision_store, concept_graph, learner_overlay, transcript)

    result = await diagnose.run(
        response="something",
        expectation="anything",
        session_id=session_id,
        turn_id=turn_id,
    )

    # Confidence stays visible even though the id was rejected as
    # unrecognized within this session's graph.
    assert result["grounding"] == {"concept_id": None, "confidence": pytest.approx(0.95)}
    assert result["action_taken"] == "none"


@pytest.mark.asyncio(loop_scope="session")
async def test_grounding_never_matches_a_concept_from_a_different_sessions_graph(
    store,
    transcript,
    concept_graph,
    learner_overlay,
    revision_store,
    learner_id,
):
    graph_a = await concept_graph.create_graph(topic="topic-a")
    graph_b = await concept_graph.create_graph(topic="topic-b")
    await concept_graph.add_concept(
        ConceptNode(concept_graph_id=graph_a.id, id="only_in_a", name="Only In A")
    )
    # graph_b has no concepts at all.

    session_id = await transcript.create_session(learner_id, graph_b.id)
    turn_id = await transcript.record_turn(session_id, 0, "turn zero")

    llm = StubLLMClient(
        canned={
            "GROUND:CONCEPT": json.dumps(
                {"concept_id": "only_in_a", "confidence": 0.9}
            )
        }
    )
    diagnose = _diagnose(llm, store, revision_store, concept_graph, learner_overlay, transcript)

    result = await diagnose.run(
        response="something about only_in_a",
        expectation="",
        session_id=session_id,
        turn_id=turn_id,
    )

    # "only_in_a" isn't in graph_b's candidate list, so GroundConcept
    # rejects it regardless of the LLM's claimed id/confidence.
    assert result["grounding"]["concept_id"] is None
    assert result["action_taken"] == "none"
