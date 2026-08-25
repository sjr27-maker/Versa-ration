from uuid import uuid4

import pytest

from probe.models import ConceptNode, OverlayState


@pytest.mark.asyncio(loop_scope="session")
async def test_set_get_roundtrip_and_upsert(concept_graph, concept_graph_id, learner_overlay):
    await concept_graph.add_concept(
        ConceptNode(concept_graph_id=concept_graph_id, id="loops", name="Loops")
    )
    learner_id = uuid4()

    entry = await learner_overlay.set_state(
        learner_id, concept_graph_id, "loops", OverlayState.PARTIAL, 0.4
    )
    assert entry.concept_id == "loops"
    assert entry.concept_graph_id == concept_graph_id
    assert entry.state is OverlayState.PARTIAL
    assert entry.confidence == pytest.approx(0.4)

    fetched = await learner_overlay.get_state(learner_id, concept_graph_id, "loops")
    assert fetched is not None
    assert fetched.state is OverlayState.PARTIAL
    assert fetched.confidence == pytest.approx(0.4)

    # Same (learner_id, concept_graph_id, concept_id) triple: set_state
    # upserts, not appends.
    updated = await learner_overlay.set_state(
        learner_id, concept_graph_id, "loops", OverlayState.KNOWN, 0.9
    )
    assert updated.state is OverlayState.KNOWN
    assert updated.confidence == pytest.approx(0.9)

    refetched = await learner_overlay.get_state(learner_id, concept_graph_id, "loops")
    assert refetched.state is OverlayState.KNOWN
    assert refetched.confidence == pytest.approx(0.9)

    full = await learner_overlay.get_full_overlay(learner_id)
    assert [e.concept_id for e in full] == ["loops"], "upsert must not duplicate the row"
    assert full[0].state is OverlayState.KNOWN
    assert full[0].confidence == pytest.approx(0.9)


@pytest.mark.asyncio(loop_scope="session")
async def test_get_state_returns_none_when_absent(learner_overlay):
    assert await learner_overlay.get_state(uuid4(), uuid4(), "nonexistent") is None


@pytest.mark.asyncio(loop_scope="session")
async def test_get_full_overlay_covers_multiple_concepts(
    concept_graph, concept_graph_id, learner_overlay
):
    await concept_graph.add_concept(
        ConceptNode(concept_graph_id=concept_graph_id, id="stacks", name="Stacks")
    )
    await concept_graph.add_concept(
        ConceptNode(concept_graph_id=concept_graph_id, id="queues", name="Queues")
    )
    learner_id = uuid4()

    await learner_overlay.set_state(
        learner_id, concept_graph_id, "stacks", OverlayState.KNOWN, 0.8
    )
    await learner_overlay.set_state(
        learner_id, concept_graph_id, "queues", OverlayState.UNKNOWN, 0.1
    )

    full = await learner_overlay.get_full_overlay(learner_id)
    by_concept = {e.concept_id: e for e in full}
    assert set(by_concept) == {"stacks", "queues"}
    assert by_concept["stacks"].state is OverlayState.KNOWN
    assert by_concept["queues"].state is OverlayState.UNKNOWN


@pytest.mark.asyncio(loop_scope="session")
async def test_get_full_overlay_keeps_same_concept_id_from_different_graphs_distinct(
    concept_graph, learner_overlay
):
    graph_a = await concept_graph.create_graph(topic="a")
    graph_b = await concept_graph.create_graph(topic="b")
    await concept_graph.add_concept(
        ConceptNode(concept_graph_id=graph_a.id, id="shared", name="Shared A")
    )
    await concept_graph.add_concept(
        ConceptNode(concept_graph_id=graph_b.id, id="shared", name="Shared B")
    )
    learner_id = uuid4()

    await learner_overlay.set_state(
        learner_id, graph_a.id, "shared", OverlayState.KNOWN, 0.9
    )
    await learner_overlay.set_state(
        learner_id, graph_b.id, "shared", OverlayState.UNKNOWN, 0.1
    )

    full = await learner_overlay.get_full_overlay(learner_id)
    assert len(full) == 2, "same concept_id in two graphs must be two rows, not one"
    by_graph = {e.concept_graph_id: e for e in full}
    assert by_graph[graph_a.id].state is OverlayState.KNOWN
    assert by_graph[graph_b.id].state is OverlayState.UNKNOWN
