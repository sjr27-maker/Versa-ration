import json

import pytest

from probe.concept_graph import ConceptCycleError, ConceptGraph, ConceptValidationError
from probe.llm import StubLLMClient
from probe.seed import SeedGraphError, seed_graph


def _node(id_: str, prerequisites: list[str] | None = None) -> dict:
    return {
        "id": id_,
        "name": id_.title(),
        "prerequisites": prerequisites or [],
        "common_misconceptions": [],
        "representations": ["formal"],
        "diagnostic_questions": [f"what is {id_}?"],
    }


async def _graph_row_count(pool, graph_id) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT count(*) FROM concept_graphs WHERE id = $1", graph_id
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_seed_graph_persists_a_valid_batch(clean_pool):
    graph = ConceptGraph(clean_pool)
    llm = StubLLMClient(
        canned={
            "SEED:CONCEPT_GRAPH": json.dumps(
                [_node("base"), _node("derived", ["base"])]
            )
        }
    )

    graph_id, concepts = await seed_graph(llm, graph, "recursion")

    assert {c.id for c in concepts} == {"base", "derived"}
    persisted_base = await graph.get_concept(graph_id, "base")
    persisted_derived = await graph.get_concept(graph_id, "derived")
    assert persisted_base is not None
    assert persisted_derived is not None
    assert persisted_derived.prerequisites == ["base"]

    meta = await graph.get_graph(graph_id)
    assert meta is not None
    assert meta.topic == "recursion"


@pytest.mark.asyncio(loop_scope="session")
async def test_seed_graph_rejects_prerequisite_not_in_batch(clean_pool):
    graph = ConceptGraph(clean_pool)
    llm = StubLLMClient(
        canned={
            "SEED:CONCEPT_GRAPH": json.dumps(
                [_node("only_one", ["ghost_prerequisite"])]
            )
        }
    )

    with pytest.raises(ConceptValidationError):
        await seed_graph(llm, graph, "recursion")

    # Nothing was written — not the nodes, and not the graph row either
    # (add_batch creates concept_graphs + nodes in one transaction).
    async with clean_pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM concept_nodes") == 0
        assert await conn.fetchval("SELECT count(*) FROM concept_graphs") == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_seed_graph_rejects_cycle_and_inserts_nothing(clean_pool):
    graph = ConceptGraph(clean_pool)
    llm = StubLLMClient(
        canned={
            "SEED:CONCEPT_GRAPH": json.dumps([_node("p", ["q"]), _node("q", ["p"])])
        }
    )

    with pytest.raises(ConceptCycleError):
        await seed_graph(llm, graph, "recursion")

    async with clean_pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM concept_nodes") == 0
        assert await conn.fetchval("SELECT count(*) FROM concept_graphs") == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_seed_graph_rejects_malformed_json(clean_pool):
    graph = ConceptGraph(clean_pool)
    llm = StubLLMClient(canned={"SEED:CONCEPT_GRAPH": "not json at all"})

    with pytest.raises(SeedGraphError):
        await seed_graph(llm, graph, "recursion")


@pytest.mark.asyncio(loop_scope="session")
async def test_seed_graph_rejects_non_list_response(clean_pool):
    graph = ConceptGraph(clean_pool)
    llm = StubLLMClient(canned={"SEED:CONCEPT_GRAPH": json.dumps({"not": "a list"})})

    with pytest.raises(SeedGraphError):
        await seed_graph(llm, graph, "recursion")


@pytest.mark.asyncio(loop_scope="session")
async def test_default_stub_response_seeds_successfully(clean_pool):
    # No canned override: exercises the StubLLMClient default so `probe
    # seed-graph` works out of the box without a real LLM configured.
    graph = ConceptGraph(clean_pool)
    llm = StubLLMClient()

    graph_id, concepts = await seed_graph(llm, graph, "anything")
    assert len(concepts) >= 1
    assert all(c.concept_graph_id == graph_id for c in concepts)


@pytest.mark.asyncio(loop_scope="session")
async def test_reseeding_the_same_topic_creates_an_independent_graph(clean_pool):
    graph = ConceptGraph(clean_pool)
    llm = StubLLMClient(
        canned={"SEED:CONCEPT_GRAPH": json.dumps([_node("base")])}
    )

    graph_id_1, _ = await seed_graph(llm, graph, "recursion")
    graph_id_2, _ = await seed_graph(llm, graph, "recursion")

    assert graph_id_1 != graph_id_2
    # Same concept id "base" in both graphs — no collision, since
    # concept_id is only unique within its own graph.
    assert await graph.get_concept(graph_id_1, "base") is not None
    assert await graph.get_concept(graph_id_2, "base") is not None
