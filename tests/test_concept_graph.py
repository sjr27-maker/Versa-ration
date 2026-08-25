import pytest

from probe.concept_graph import ConceptCycleError, ConceptValidationError
from probe.models import ConceptNode


def _concept(
    concept_graph_id, id: str, prerequisites: list[str] | None = None
) -> ConceptNode:
    return ConceptNode(
        concept_graph_id=concept_graph_id,
        id=id,
        name=id.replace("_", " ").title(),
        prerequisites=prerequisites or [],
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_add_concept_and_get_roundtrip(concept_graph, concept_graph_id):
    await concept_graph.add_concept(
        ConceptNode(
            concept_graph_id=concept_graph_id,
            id="loops",
            name="Loops",
            common_misconceptions=["off-by-one on the bound"],
            representations=["code", "visual"],
            diagnostic_questions=["what does this loop print?"],
        )
    )

    fetched = await concept_graph.get_concept(concept_graph_id, "loops")
    assert fetched is not None
    assert fetched.concept_graph_id == concept_graph_id
    assert fetched.name == "Loops"
    assert fetched.prerequisites == []
    assert fetched.common_misconceptions == ["off-by-one on the bound"]
    assert fetched.representations == ["code", "visual"]
    assert fetched.diagnostic_questions == ["what does this loop print?"]


@pytest.mark.asyncio(loop_scope="session")
async def test_get_concept_returns_none_when_absent(concept_graph, concept_graph_id):
    assert await concept_graph.get_concept(concept_graph_id, "nonexistent") is None


@pytest.mark.asyncio(loop_scope="session")
async def test_transitive_closure_on_small_dag(concept_graph):
    # a <- b <- c <- d, and d also depends directly on a (diamond shape),
    # so all_prerequisites_of(d) must dedupe a rather than count it twice.
    # add_batch creates its own graph row, so it needs a fresh id, not
    # the concept_graph_id fixture (which already created one).
    from uuid import uuid4

    concept_graph_id = uuid4()
    await concept_graph.add_batch(
        concept_graph_id,
        "dag-topic",
        [
            _concept(concept_graph_id, "a"),
            _concept(concept_graph_id, "b", ["a"]),
            _concept(concept_graph_id, "c", ["b"]),
            _concept(concept_graph_id, "d", ["c", "a"]),
        ],
    )

    assert await concept_graph.prerequisites_of(concept_graph_id, "d") == ["a", "c"]
    assert set(await concept_graph.all_prerequisites_of(concept_graph_id, "d")) == {
        "a",
        "b",
        "c",
    }
    assert await concept_graph.all_prerequisites_of(concept_graph_id, "a") == []
    assert await concept_graph.prerequisites_of(concept_graph_id, "a") == []


@pytest.mark.asyncio(loop_scope="session")
async def test_add_concept_rejects_prerequisite_that_does_not_exist_yet(
    concept_graph, concept_graph_id
):
    import asyncpg

    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await concept_graph.add_concept(
            _concept(concept_graph_id, "orphan", ["ghost"])
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_add_batch_rejects_prerequisite_not_present_in_batch(
    concept_graph, concept_graph_id
):
    with pytest.raises(ConceptValidationError):
        await concept_graph.add_batch(
            concept_graph_id,
            "batch-topic",
            [_concept(concept_graph_id, "only_one", ["ghost"])],
        )
    assert await concept_graph.get_concept(concept_graph_id, "only_one") is None


@pytest.mark.asyncio(loop_scope="session")
async def test_add_batch_rejects_cycle_and_inserts_nothing(
    concept_graph, concept_graph_id
):
    with pytest.raises(ConceptCycleError):
        await concept_graph.add_batch(
            concept_graph_id,
            "cycle-topic",
            [
                _concept(concept_graph_id, "p", ["q"]),
                _concept(concept_graph_id, "q", ["p"]),
            ],
        )
    assert await concept_graph.get_concept(concept_graph_id, "p") is None
    assert await concept_graph.get_concept(concept_graph_id, "q") is None


@pytest.mark.asyncio(loop_scope="session")
async def test_add_batch_rejects_concept_carrying_a_different_graph_id(
    concept_graph, concept_graph_id
):
    from uuid import uuid4

    with pytest.raises(ConceptValidationError):
        await concept_graph.add_batch(
            concept_graph_id,
            "mismatch-topic",
            [_concept(uuid4(), "wrong_graph")],
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_cycle_detection_rejects_a_fixture_with_a_cycle(
    concept_graph, concept_graph_id, clean_pool
):
    # The public API (add_concept requires prerequisites to pre-exist;
    # add_batch validates for cycles up front) can't produce a cyclic
    # graph on its own. Simulate a corrupted/inconsistent graph directly
    # at the SQL layer to exercise the defensive check in
    # all_prerequisites_of.
    await concept_graph.add_concept(_concept(concept_graph_id, "x"))
    await concept_graph.add_concept(_concept(concept_graph_id, "y"))
    async with clean_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO concept_prerequisites "
            "(concept_graph_id, concept_id, prerequisite_id) VALUES ($1, $2, $3)",
            concept_graph_id,
            "x",
            "y",
        )
        await conn.execute(
            "INSERT INTO concept_prerequisites "
            "(concept_graph_id, concept_id, prerequisite_id) VALUES ($1, $2, $3)",
            concept_graph_id,
            "y",
            "x",
        )

    with pytest.raises(ConceptCycleError):
        await concept_graph.all_prerequisites_of(concept_graph_id, "x")


@pytest.mark.asyncio(loop_scope="session")
async def test_two_graphs_do_not_leak_nodes_into_each_others_prerequisites(
    concept_graph,
):
    # Same concept id ("shared") in two different graphs must be two
    # independent nodes, not a collision — and each graph's traversal
    # must never see the other graph's nodes.
    graph_a = await concept_graph.create_graph(topic="topic-a")
    graph_b = await concept_graph.create_graph(topic="topic-b")

    await concept_graph.add_concept(_concept(graph_a.id, "shared"))
    await concept_graph.add_concept(
        _concept(graph_a.id, "only_in_a", ["shared"])
    )
    await concept_graph.add_concept(_concept(graph_b.id, "shared"))

    a_prereqs = await concept_graph.all_prerequisites_of(graph_a.id, "only_in_a")
    assert a_prereqs == ["shared"]

    # graph_b's "shared" has no prerequisites of its own and no edge to
    # graph_a's "only_in_a" — it's a completely separate node.
    b_prereqs = await concept_graph.all_prerequisites_of(graph_b.id, "shared")
    assert b_prereqs == []
    assert await concept_graph.get_concept(graph_b.id, "only_in_a") is None

    b_nodes = await concept_graph.list_concepts(graph_b.id)
    assert {c.id for c in b_nodes} == {"shared"}
    a_nodes = await concept_graph.list_concepts(graph_a.id)
    assert {c.id for c in a_nodes} == {"shared", "only_in_a"}


@pytest.mark.asyncio(loop_scope="session")
async def test_create_get_and_find_graphs_by_topic(concept_graph):
    meta = await concept_graph.create_graph(topic="python closures")
    fetched = await concept_graph.get_graph(meta.id)
    assert fetched is not None
    assert fetched.topic == "python closures"

    matches = await concept_graph.find_graphs_by_topic("python closures")
    assert meta.id in {m.id for m in matches}

    from uuid import uuid4

    assert await concept_graph.get_graph(uuid4()) is None
    assert await concept_graph.find_graphs_by_topic("nonexistent topic") == []


def test_concept_graph_module_has_no_delete():
    import ast
    import re
    from pathlib import Path

    from probe import concept_graph as concept_graph_module

    source = Path(concept_graph_module.__file__).read_text()
    tree = ast.parse(source)

    docstring_nodes: set[int] = set()
    for parent in ast.walk(tree):
        if isinstance(
            parent,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            body = getattr(parent, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring_nodes.add(id(body[0].value))

    delete_kw = re.compile(r"\bDELETE\b", re.IGNORECASE)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstring_nodes
        ):
            assert not delete_kw.search(node.value), (
                f"string literal at line {node.lineno} contains DELETE — "
                "ConceptGraph must be append-only"
            )

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lname = node.name.lower()
            assert not (lname.startswith("delete") or lname.startswith("remove")), (
                f"function {node.name!r} at line {node.lineno} looks like a "
                "removal method — ConceptGraph must be append-only"
            )
