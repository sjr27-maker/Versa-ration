"""Learner isolation across two related fixes:

1. Replan's entropy (and, by the same fix, HypothesisGenerator's branch
   budget) must reflect only the current session's own learner —
   HypothesisStore.list_all() has no learner filter at all, so before
   this fix a different learner's hypotheses in the same database would
   silently inflate or deflate a session's reasoning budget. Verified
   via list_by_learner directly and via a full SessionLoop turn.

2. Infer must never be able to reweight or target another learner's
   hypothesis. Scoping Infer's *input* to list_by_learner (this file's
   first fix, and loop.py's) only stops the LLM from being shown
   another learner's hypothesis ids — it doesn't stop a hallucinated or
   copied id from reaching Update.reweight() if nothing checks it.
   Infer.run() now rejects any hypothesis_id outside the candidate set
   it was actually shown (nodes.py) — verified below by adversarially
   crafting an Infer response that names a real hypothesis belonging to
   a *different* learner and confirming it's rejected, not applied.
"""

import json
from uuid import UUID, uuid4

import pytest

from probe.llm import StubLLMClient
from probe.loop import SessionLoop
from probe.models import EvidenceRef, Hypothesis, Layer, Polarity, Tier


async def _seed_high_entropy_hypotheses(
    store, transcript, session_id: UUID, turn_index: int, count: int = 8
) -> None:
    turn_id = await transcript.record_turn(
        session_id, turn_index, "seed turn establishing prior hypotheses"
    )
    for _ in range(count):
        hyp = await store.add(
            Hypothesis(
                layer=Layer.KNOWLEDGE,
                statement="hedge",
                probability=0.5,
                confidence=0.5,
                tier=Tier.ACTIVE,
            )
        )
        await store.reweight(
            hyp.id, 0.5, 0.5, EvidenceRef(turn_id=turn_id, polarity=Polarity.SUPPORTING)
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_list_by_learner_does_not_cross_contaminate_between_learners(
    store, transcript, clean_pool, learner_store, concept_graph_id
):
    learner_a = await learner_store.create(label="learner-a")
    learner_b = await learner_store.create(label="learner-b")
    session_b = await transcript.create_session(learner_b.id, concept_graph_id)

    await _seed_high_entropy_hypotheses(store, transcript, session_b, 0, count=8)
    # learner_a gets nothing.

    a_hyps = await store.list_by_learner(learner_a.id)
    b_hyps = await store.list_by_learner(learner_b.id)

    assert a_hyps == []
    assert len(b_hyps) == 8


@pytest.mark.asyncio(loop_scope="session")
async def test_replan_entropy_is_not_contaminated_by_another_learners_hypotheses(
    store,
    transcript,
    node_calls,
    clean_pool,
    learner_store,
    concept_graph,
    concept_graph_id,
    learner_overlay,
    revision_store,
):
    learner_a = await learner_store.create(label="learner-a-quiet")
    learner_b = await learner_store.create(label="learner-b-noisy")
    session_a = await transcript.create_session(learner_a.id, concept_graph_id)
    session_b = await transcript.create_session(learner_b.id, concept_graph_id)

    # learner_b has 8 high-entropy hypotheses; learner_a has none at all.
    await _seed_high_entropy_hypotheses(store, transcript, session_b, 0, count=8)

    loop_a = SessionLoop(
        hypothesis_store=store,
        transcript=transcript,
        node_calls=node_calls,
        concept_graph=concept_graph,
        learner_overlay=learner_overlay,
        revision_store=revision_store,
        llm=StubLLMClient(),
    )
    await loop_a.handle_turn(session_a, 0, "learner a's first turn")

    async with clean_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT output_json FROM node_calls "
            "WHERE session_id=$1 AND node_name='Replan'",
            session_a,
        )
    entropy_for_a = row["output_json"]["entropy_bits"]

    assert entropy_for_a == 0.0, (
        "learner_a has zero hypotheses of their own — learner_b's 8 "
        "high-entropy hypotheses must not leak into learner_a's Replan"
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_replan_entropy_does_reflect_this_learners_own_hypotheses(
    store,
    transcript,
    node_calls,
    clean_pool,
    learner_store,
    concept_graph,
    concept_graph_id,
    learner_overlay,
    revision_store,
):
    """The flip side of isolation: it isn't just always zero — a
    learner's own hypotheses are still correctly picked up."""
    learner = await learner_store.create(label="learner-with-own-hypotheses")
    session_id = await transcript.create_session(learner.id, concept_graph_id)
    await _seed_high_entropy_hypotheses(store, transcript, session_id, 0, count=8)

    loop = SessionLoop(
        hypothesis_store=store,
        transcript=transcript,
        node_calls=node_calls,
        concept_graph=concept_graph,
        learner_overlay=learner_overlay,
        revision_store=revision_store,
        llm=StubLLMClient(),
    )
    await loop.handle_turn(session_id, 1, "this learner's own turn")

    async with clean_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT output_json FROM node_calls "
            "WHERE session_id=$1 AND node_name='Replan'",
            session_id,
        )
    assert row["output_json"]["entropy_bits"] == pytest.approx(8.0)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_learners_turn_can_never_reweight_another_learners_hypothesis(
    store,
    transcript,
    node_calls,
    clean_pool,
    learner_store,
    concept_graph,
    concept_graph_id,
    learner_overlay,
    revision_store,
    caplog,
):
    import logging

    learner_a = await learner_store.create(label="learner-a-attacker-or-buggy-llm")
    learner_b = await learner_store.create(label="learner-b-victim")
    session_a = await transcript.create_session(learner_a.id, concept_graph_id)
    session_b = await transcript.create_session(learner_b.id, concept_graph_id)

    # A real hypothesis genuinely owned (via evidence) by learner_b.
    seed_turn_b = await transcript.record_turn(session_b, 0, "learner b's own turn")
    victim = await store.add(
        Hypothesis(
            layer=Layer.KNOWLEDGE,
            statement="learner b's own belief",
            probability=0.5,
            confidence=0.5,
            tier=Tier.ACTIVE,
        )
    )
    await store.reweight(
        victim.id, 0.5, 0.5,
        EvidenceRef(turn_id=seed_turn_b, polarity=Polarity.SUPPORTING),
    )

    # A real turn in learner_a's own session, for the crafted evidence_ref.
    seed_turn_a = await transcript.record_turn(session_a, 0, "learner a's own turn")

    # Adversarial (or simply hallucinated) Infer response: targets
    # learner_b's real hypothesis_id from inside learner_a's turn.
    malicious_response = json.dumps(
        [
            {
                "hypothesis_id": str(victim.id),
                "new_probability": 0.99,
                "new_confidence": 0.99,
                "evidence_ref": {
                    "turn_id": str(seed_turn_a),
                    "polarity": "supporting",
                },
            }
        ]
    )
    llm = StubLLMClient(canned={"INFER:": malicious_response})
    loop = SessionLoop(
        hypothesis_store=store,
        transcript=transcript,
        node_calls=node_calls,
        concept_graph=concept_graph,
        learner_overlay=learner_overlay,
        revision_store=revision_store,
        llm=llm,
    )

    with caplog.at_level(logging.WARNING, logger="probe.nodes"):
        await loop.handle_turn(session_a, 1, "learner a's real turn")

    refreshed_victim = await store.get(victim.id)
    assert refreshed_victim.probability == 0.5  # untouched, not 0.99
    assert refreshed_victim.confidence == 0.5
    assert len(refreshed_victim.evidence_refs) == 1  # still just its original evidence

    rejections = [r for r in caplog.records if "not in the" in r.message]
    assert rejections, "the rejection must be logged, not silently dropped"


@pytest.mark.asyncio(loop_scope="session")
async def test_infer_rejects_a_hypothesis_id_never_shown_in_the_candidate_set(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store,
):
    """Same rejection, without needing a second learner: any id outside
    the candidate list handed to Infer.run() is rejected, including a
    fully invented one that doesn't correspond to any real hypothesis
    at all."""
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    fabricated_id = uuid4()
    fabricated_response = json.dumps(
        [
            {
                "hypothesis_id": str(fabricated_id),
                "new_probability": 0.8,
                "new_confidence": 0.8,
                "evidence_ref": {
                    "turn_id": str(await transcript.record_turn(session_id, 0, "seed")),
                    "polarity": "supporting",
                },
            }
        ]
    )
    llm = StubLLMClient(canned={"INFER:": fabricated_response})
    loop = SessionLoop(
        hypothesis_store=store,
        transcript=transcript,
        node_calls=node_calls,
        concept_graph=concept_graph,
        learner_overlay=learner_overlay,
        revision_store=revision_store,
        llm=llm,
    )

    await loop.handle_turn(session_id, 1, "a turn")

    # Nothing was ever created at that id — reweight() was never called.
    assert await store.get(fabricated_id) is None
