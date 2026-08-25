import pytest

from probe.concept_graph import ConceptCycleError, ConceptValidationError
from probe.models import ConceptNode


def _concept(id: str, prerequisites: list[str] | None = None) -> ConceptNode:
    return ConceptNode(
        id=id,
        name=id.replace("_", " ").title(),
        prerequisites=prerequisites or [],
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_add_concept_and_get_roundtrip(concept_graph):
    await concept_graph.add_concept(
        ConceptNode(
            id="loops",
            name="Loops",
            common_misconceptions=["off-by-one on the bound"],
            representations=["code", "visual"],
            diagnostic_questions=["what does this loop print?"],
        )
    )

    fetched = await concept_graph.get_concept("loops")
    assert fetched is not None
    assert fetched.name == "Loops"
    assert fetched.prerequisites == []
    assert fetched.common_misconceptions == ["off-by-one on the bound"]
    assert fetched.representations == ["code", "visual"]
    assert fetched.diagnostic_questions == ["what does this loop print?"]


@pytest.mark.asyncio(loop_scope="session")
async def test_get_concept_returns_none_when_absent(concept_graph):
    assert await concept_graph.get_concept("nonexistent") is None


@pytest.mark.asyncio(loop_scope="session")
async def test_transitive_closure_on_small_dag(concept_graph):
    # a <- b <- c <- d, and d also depends directly on a (diamond shape),
    # so all_prerequisites_of(d) must dedupe a rather than count it twice.
    await concept_graph.add_batch(
        [
            _concept("a"),
            _concept("b", ["a"]),
            _concept("c", ["b"]),
            _concept("d", ["c", "a"]),
        ]
    )

    assert await concept_graph.prerequisites_of("d") == ["a", "c"]
    assert set(await concept_graph.all_prerequisites_of("d")) == {"a", "b", "c"}
    assert await concept_graph.all_prerequisites_of("a") == []
    assert await concept_graph.prerequisites_of("a") == []


@pytest.mark.asyncio(loop_scope="session")
async def test_add_concept_rejects_prerequisite_that_does_not_exist_yet(
    concept_graph,
):
    import asyncpg

    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await concept_graph.add_concept(_concept("orphan", ["ghost"]))


@pytest.mark.asyncio(loop_scope="session")
async def test_add_batch_rejects_prerequisite_not_present_in_batch(concept_graph):
    with pytest.raises(ConceptValidationError):
        await concept_graph.add_batch([_concept("only_one", ["ghost"])])
    assert await concept_graph.get_concept("only_one") is None


@pytest.mark.asyncio(loop_scope="session")
async def test_add_batch_rejects_cycle_and_inserts_nothing(concept_graph):
    with pytest.raises(ConceptCycleError):
        await concept_graph.add_batch([_concept("p", ["q"]), _concept("q", ["p"])])
    assert await concept_graph.get_concept("p") is None
    assert await concept_graph.get_concept("q") is None


@pytest.mark.asyncio(loop_scope="session")
async def test_cycle_detection_rejects_a_fixture_with_a_cycle(
    concept_graph, clean_pool
):
    # The public API (add_concept requires prerequisites to pre-exist;
    # add_batch validates for cycles up front) can't produce a cyclic
    # graph on its own. Simulate a corrupted/inconsistent graph directly
    # at the SQL layer to exercise the defensive check in
    # all_prerequisites_of.
    await concept_graph.add_batch([_concept("x"), _concept("y")])
    async with clean_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO concept_prerequisites (concept_id, prerequisite_id) "
            "VALUES ($1, $2)",
            "x",
            "y",
        )
        await conn.execute(
            "INSERT INTO concept_prerequisites (concept_id, prerequisite_id) "
            "VALUES ($1, $2)",
            "y",
            "x",
        )

    with pytest.raises(ConceptCycleError):
        await concept_graph.all_prerequisites_of("x")


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
