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

    concepts = await seed_graph(llm, graph, "recursion")

    assert {c.id for c in concepts} == {"base", "derived"}
    persisted_base = await graph.get_concept("base")
    persisted_derived = await graph.get_concept("derived")
    assert persisted_base is not None
    assert persisted_derived is not None
    assert persisted_derived.prerequisites == ["base"]


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

    assert await graph.get_concept("only_one") is None


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

    assert await graph.get_concept("p") is None
    assert await graph.get_concept("q") is None


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

    concepts = await seed_graph(llm, graph, "anything")
    assert len(concepts) >= 1
