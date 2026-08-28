"""LearnerOverlay.get_overlay_for_graph — scoped to one concept graph,
for SessionLoop's concept_state assembly (which only ever needs the
current session's graph, not every graph a learner has touched)."""

import pytest

from probe.models import ConceptNode, OverlayState


@pytest.mark.asyncio(loop_scope="session")
async def test_get_overlay_for_graph_only_returns_that_graphs_entries(
    learner_overlay, concept_graph, learner_id
):
    graph_a = await concept_graph.create_graph(topic="graph-a")
    graph_b = await concept_graph.create_graph(topic="graph-b")
    await concept_graph.add_concept(
        ConceptNode(concept_graph_id=graph_a.id, id="concept-x", name="Concept X")
    )
    await concept_graph.add_concept(
        ConceptNode(concept_graph_id=graph_b.id, id="concept-x", name="Concept X")
    )

    await learner_overlay.set_state(
        learner_id, graph_a.id, "concept-x", OverlayState.KNOWN, 0.9
    )
    await learner_overlay.set_state(
        learner_id, graph_b.id, "concept-x", OverlayState.UNKNOWN, 0.1
    )

    entries = await learner_overlay.get_overlay_for_graph(learner_id, graph_a.id)

    assert len(entries) == 1
    assert entries[0].concept_graph_id == graph_a.id
    assert entries[0].state is OverlayState.KNOWN


@pytest.mark.asyncio(loop_scope="session")
async def test_get_overlay_for_graph_empty_when_nothing_touched(
    learner_overlay, concept_graph, learner_id
):
    graph = await concept_graph.create_graph(topic="untouched-graph")

    assert await learner_overlay.get_overlay_for_graph(learner_id, graph.id) == []
