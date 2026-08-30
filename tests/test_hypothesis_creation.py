"""The missing hypothesis-creation path: before this fix, Infer could
only reweight an EXISTING hypothesis_id (`HypothesisStore.reweight()`
raises `KeyError` on a missing one), and nothing anywhere ever called
`HypothesisStore.add()` in production code. Every real session (CLI,
web UI, every scripted comparison) therefore ran with the hypothesis
list permanently empty, `compute_reasoning_budget([])` always saw
`entropy_bits=0.0`, and `generation_width`/`run_information_value`/
`exploration_target` were all silently pinned at their zero-entropy
floor forever.

Covers: Infer producing a create-type proposal from a signal-bearing
turn against an empty hypothesis list, and (the test that actually
closes the loop, not just "a row exists") entropy_bits becoming
nonzero both within the SAME turn a hypothesis is created and on a
LATER turn that creates nothing new of its own.
"""

from __future__ import annotations

import json

import pytest

from probe.llm import StubLLMClient
from probe.loop import SessionLoop
from probe.nodes import Infer


def _create_response(statement: str) -> str:
    return json.dumps(
        [
            {
                "kind": "create",
                "layer": "goal",
                "statement": statement,
                "initial_probability": 0.6,
                "initial_confidence": 0.5,
            }
        ]
    )


def _once_then_empty(first_response: str):
    """A canned INFER: response that fires only on the first call to
    Infer this session, then goes quiet -- so a later turn's nonzero
    entropy can only be explained by the earlier turn's hypothesis
    persisting, not by that later turn creating its own."""
    calls = {"n": 0}

    def _respond(_prompt: str) -> str:
        calls["n"] += 1
        return first_response if calls["n"] == 1 else "[]"

    return _respond


@pytest.mark.asyncio
async def test_infer_produces_a_create_proposal_against_an_empty_hypothesis_list():
    llm = StubLLMClient(
        canned={"INFER:": _create_response("student wants to master the chain rule")}
    )
    node = Infer(llm)

    from uuid import uuid4

    output = await node.run(
        turn_text="I really want to get the chain rule down before my exam",
        hypotheses=[],
        turn_id=uuid4(),
    )

    assert output.reweights == []
    assert len(output.creates) == 1
    create = output.creates[0]
    assert create.statement == "student wants to master the chain rule"
    assert create.layer.value == "goal"
    assert create.initial_probability == pytest.approx(0.6)
    assert create.initial_confidence == pytest.approx(0.5)


@pytest.mark.asyncio(loop_scope="session")
async def test_hypothesis_created_this_turn_is_visible_to_this_same_turns_replan(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, diagnostics_store,
):
    llm = StubLLMClient(
        canned={"INFER:": _create_response("student wants to master the chain rule")}
    )
    loop = SessionLoop(
        hypothesis_store=store, transcript=transcript, node_calls=node_calls,
        concept_graph=concept_graph, learner_overlay=learner_overlay,
        revision_store=revision_store, llm=llm, diagnostics_store=diagnostics_store,
    )
    session_id = await transcript.create_session(learner_id, concept_graph_id)

    await loop.handle_turn(session_id, 0, "I really want to master the chain rule")

    active = await store.list_by_learner(learner_id)
    assert len(active) == 1

    diag = await diagnostics_store.get_for_turn(session_id, 0)
    # p=0.6 has nonzero Bernoulli entropy -- if Replan's entropy_bits is
    # still 0.0, the hypothesis Update just created was not actually
    # visible to Replan's list_by_learner read in the SAME turn.
    assert diag.entropy_bits is not None
    assert diag.entropy_bits > 0.0


@pytest.mark.asyncio(loop_scope="session")
async def test_a_later_turn_still_sees_nonzero_entropy_from_an_earlier_turns_hypothesis(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, diagnostics_store,
):
    """The test that actually closes the loop: turn 1 creates nothing
    of its own (canned INFER: goes back to the conservative "[]"
    default), yet its entropy_bits is still nonzero -- provable only by
    turn 0's hypothesis having actually persisted and being read back
    on a completely separate turn, not some same-turn-only artifact."""
    llm = StubLLMClient(
        canned={
            "INFER:": _once_then_empty(
                _create_response("student wants to master the chain rule")
            )
        }
    )
    loop = SessionLoop(
        hypothesis_store=store, transcript=transcript, node_calls=node_calls,
        concept_graph=concept_graph, learner_overlay=learner_overlay,
        revision_store=revision_store, llm=llm, diagnostics_store=diagnostics_store,
    )
    session_id = await transcript.create_session(learner_id, concept_graph_id)

    await loop.handle_turn(session_id, 0, "I really want to master the chain rule")
    await loop.handle_turn(session_id, 1, "what about something unrelated")

    turn1_diag = await diagnostics_store.get_for_turn(session_id, 1)
    assert turn1_diag.entropy_bits is not None
    assert turn1_diag.entropy_bits > 0.0
    # And no second hypothesis was minted on turn 1 -- this really is
    # persistence of turn 0's row, not a second creation.
    assert len(await store.list_by_learner(learner_id)) == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_infer_still_reweights_an_existing_hypothesis_correctly(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, diagnostics_store,
):
    """Additive, not a replacement: an existing hypothesis in the
    candidate list is still reweighted via the (now kind-tagged)
    reweight path, unaffected by create existing alongside it."""
    from probe.models import EvidenceRef, Hypothesis, Layer, Polarity, Tier

    session_id = await transcript.create_session(learner_id, concept_graph_id)
    existing = await store.add(
        Hypothesis(
            layer=Layer.KNOWLEDGE,
            statement="student understands the power rule",
            probability=0.4,
            confidence=0.3,
            tier=Tier.ACTIVE,
        )
    )
    # list_by_learner only surfaces a hypothesis with at least one
    # evidence_ref traceable to this learner's own sessions (see its
    # docstring) -- seed one so Infer is actually shown this id as a
    # valid candidate, same pattern as test_learner_scoped_entropy.py.
    seed_turn = await transcript.record_turn(session_id, 0, "seed turn")
    await store.reweight(
        existing.id, 0.4, 0.3,
        EvidenceRef(turn_id=seed_turn, polarity=Polarity.SUPPORTING),
    )

    def _reweight_response(prompt: str) -> str:
        return json.dumps(
            [
                {
                    "kind": "reweight",
                    "hypothesis_id": str(existing.id),
                    "new_probability": 0.85,
                    "new_confidence": 0.7,
                    "polarity": "supporting",
                }
            ]
        )

    llm = StubLLMClient(canned={"INFER:": _reweight_response})
    loop = SessionLoop(
        hypothesis_store=store, transcript=transcript, node_calls=node_calls,
        concept_graph=concept_graph, learner_overlay=learner_overlay,
        revision_store=revision_store, llm=llm, diagnostics_store=diagnostics_store,
    )

    await loop.handle_turn(session_id, 1, "yes, I understand the power rule now")

    reloaded = await store.get(existing.id)
    assert reloaded.probability == pytest.approx(0.85)
    assert reloaded.confidence == pytest.approx(0.7)
