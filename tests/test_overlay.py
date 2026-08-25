from uuid import uuid4

import pytest

from probe.models import ConceptNode, OverlayState


@pytest.mark.asyncio(loop_scope="session")
async def test_set_get_roundtrip_and_upsert(concept_graph, learner_overlay):
    await concept_graph.add_concept(ConceptNode(id="loops", name="Loops"))
    learner_id = uuid4()

    entry = await learner_overlay.set_state(
        learner_id, "loops", OverlayState.PARTIAL, 0.4
    )
    assert entry.concept_id == "loops"
    assert entry.state is OverlayState.PARTIAL
    assert entry.confidence == pytest.approx(0.4)

    fetched = await learner_overlay.get_state(learner_id, "loops")
    assert fetched is not None
    assert fetched.state is OverlayState.PARTIAL
    assert fetched.confidence == pytest.approx(0.4)

    # Same (learner_id, concept_id) pair: set_state upserts, not appends.
    updated = await learner_overlay.set_state(
        learner_id, "loops", OverlayState.KNOWN, 0.9
    )
    assert updated.state is OverlayState.KNOWN
    assert updated.confidence == pytest.approx(0.9)

    refetched = await learner_overlay.get_state(learner_id, "loops")
    assert refetched.state is OverlayState.KNOWN
    assert refetched.confidence == pytest.approx(0.9)

    full = await learner_overlay.get_full_overlay(learner_id)
    assert list(full.keys()) == ["loops"], "upsert must not duplicate the row"
    assert full["loops"].state is OverlayState.KNOWN
    assert full["loops"].confidence == pytest.approx(0.9)


@pytest.mark.asyncio(loop_scope="session")
async def test_get_state_returns_none_when_absent(learner_overlay):
    assert await learner_overlay.get_state(uuid4(), "nonexistent") is None


@pytest.mark.asyncio(loop_scope="session")
async def test_get_full_overlay_covers_multiple_concepts(
    concept_graph, learner_overlay
):
    await concept_graph.add_concept(ConceptNode(id="stacks", name="Stacks"))
    await concept_graph.add_concept(ConceptNode(id="queues", name="Queues"))
    learner_id = uuid4()

    await learner_overlay.set_state(learner_id, "stacks", OverlayState.KNOWN, 0.8)
    await learner_overlay.set_state(learner_id, "queues", OverlayState.UNKNOWN, 0.1)

    full = await learner_overlay.get_full_overlay(learner_id)
    assert set(full.keys()) == {"stacks", "queues"}
    assert full["stacks"].state is OverlayState.KNOWN
    assert full["queues"].state is OverlayState.UNKNOWN
