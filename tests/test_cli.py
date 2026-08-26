"""_resolve_graph's UUID path must fail closed rather than resolve a
concept-graph row that has zero concept_nodes -- that state is only
reachable by calling the low-level ConceptGraph.create_graph() outside
seed_graph/add_batch, which is exactly what these tests do to construct
the fixture, since the CLI itself never creates a graph that way.
"""

from uuid import uuid4

import pytest

from probe.cli import _resolve_graph
from probe.llm import StubLLMClient
from probe.models import ConceptNode


@pytest.mark.asyncio(loop_scope="session")
async def test_resolve_graph_by_uuid_fails_closed_when_graph_has_no_concepts(
    concept_graph, clean_pool, capsys
):
    meta = await concept_graph.create_graph(topic="empty-graph-outside-seed-graph")

    with pytest.raises(SystemExit) as exc_info:
        await _resolve_graph(concept_graph, str(meta.id), StubLLMClient())

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "no concepts" in err
    assert "no concept graph with id" not in err  # distinct from the not-found path


@pytest.mark.asyncio(loop_scope="session")
async def test_resolve_graph_by_uuid_still_errors_clearly_when_graph_does_not_exist(
    concept_graph, clean_pool, capsys
):
    with pytest.raises(SystemExit) as exc_info:
        await _resolve_graph(concept_graph, str(uuid4()), StubLLMClient())

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "no concept graph with id" in err
    assert "no concepts" not in err  # distinct from the empty-graph path


@pytest.mark.asyncio(loop_scope="session")
async def test_resolve_graph_by_uuid_succeeds_when_graph_has_at_least_one_concept(
    concept_graph, clean_pool
):
    concept_graph_id = uuid4()
    await concept_graph.add_batch(
        concept_graph_id,
        "populated-graph",
        [ConceptNode(concept_graph_id=concept_graph_id, id="c1", name="Concept One")],
    )

    meta = await _resolve_graph(concept_graph, str(concept_graph_id), StubLLMClient())

    assert meta.id == concept_graph_id
