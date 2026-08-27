import ast
import re
from pathlib import Path

import pytest

import probe.branches as branches_module
from probe.models import Branch, BranchStatus


@pytest.mark.asyncio(loop_scope="session")
async def test_create_generation_and_add_branches_roundtrip(
    branch_store, transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    generation = await branch_store.create_generation(session_id, 0, root_count=2)

    root = Branch(
        parent_id=None,
        generation_id=generation.id,
        session_id=session_id,
        turn_index=0,
        depth=0,
        depth_label="intent",
        statement="wants an analogy",
        predicted_next_turn="will ask for a real-world comparison",
        plausibility=0.6,
        is_leaf=False,
    )
    child = Branch(
        parent_id=root.id,
        generation_id=generation.id,
        session_id=session_id,
        turn_index=0,
        depth=1,
        depth_label="knowledge_gap",
        statement="missing the base definition",
        predicted_next_turn="will ask what the term means",
        plausibility=0.5,
        is_leaf=True,
    )
    await branch_store.add_branches([root, child])

    fetched_root = await branch_store.get(root.id)
    fetched_child = await branch_store.get(child.id)
    assert fetched_root is not None and fetched_root.parent_id is None
    assert fetched_child is not None and fetched_child.parent_id == root.id
    assert fetched_child.depth == 1
    assert fetched_child.status is BranchStatus.OPEN


@pytest.mark.asyncio(loop_scope="session")
async def test_get_open_leaves_only_returns_open_leaf_branches(
    branch_store, transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    generation = await branch_store.create_generation(session_id, 0, root_count=1)

    leaf_open = Branch(
        generation_id=generation.id, session_id=session_id, turn_index=0,
        depth=0, depth_label="intent", statement="a", predicted_next_turn="p-a",
        plausibility=0.5, is_leaf=True,
    )
    non_leaf = Branch(
        generation_id=generation.id, session_id=session_id, turn_index=0,
        depth=0, depth_label="intent", statement="b", predicted_next_turn="p-b",
        plausibility=0.5, is_leaf=False,
    )
    leaf_matched = Branch(
        generation_id=generation.id, session_id=session_id, turn_index=0,
        depth=0, depth_label="intent", statement="c", predicted_next_turn="p-c",
        plausibility=0.5, is_leaf=True, status=BranchStatus.MATCHED,
    )
    await branch_store.add_branches([leaf_open, non_leaf, leaf_matched])

    leaves = await branch_store.get_open_leaves(generation.id)

    assert [leaf.id for leaf in leaves] == [leaf_open.id]


@pytest.mark.asyncio(loop_scope="session")
async def test_get_ancestors_walks_the_full_parent_chain(
    branch_store, transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    generation = await branch_store.create_generation(session_id, 0, root_count=1)

    root = Branch(
        generation_id=generation.id, session_id=session_id, turn_index=0,
        depth=0, depth_label="intent", statement="root", predicted_next_turn="p0",
        plausibility=0.6, is_leaf=False,
    )
    mid = Branch(
        parent_id=root.id, generation_id=generation.id, session_id=session_id,
        turn_index=0, depth=1, depth_label="knowledge_gap", statement="mid",
        predicted_next_turn="p1", plausibility=0.5, is_leaf=False,
    )
    leaf = Branch(
        parent_id=mid.id, generation_id=generation.id, session_id=session_id,
        turn_index=0, depth=2, depth_label="predicted_action", statement="leaf",
        predicted_next_turn="p2", plausibility=0.5, is_leaf=True,
    )
    await branch_store.add_branches([root, mid, leaf])

    ancestors = await branch_store.get_ancestors(leaf.id)

    assert [a.id for a in ancestors] == [mid.id, root.id]
    assert await branch_store.get_ancestors(root.id) == []


@pytest.mark.asyncio(loop_scope="session")
async def test_set_status_transitions_and_supersede_open_branches(
    branch_store, transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    generation = await branch_store.create_generation(session_id, 0, root_count=1)

    matched = Branch(
        generation_id=generation.id, session_id=session_id, turn_index=0,
        depth=0, depth_label="intent", statement="a", predicted_next_turn="p-a",
        plausibility=0.5, is_leaf=True,
    )
    left_open = Branch(
        generation_id=generation.id, session_id=session_id, turn_index=0,
        depth=0, depth_label="intent", statement="b", predicted_next_turn="p-b",
        plausibility=0.5, is_leaf=True,
    )
    await branch_store.add_branches([matched, left_open])

    updated = await branch_store.set_status(matched.id, BranchStatus.MATCHED)
    assert updated.status is BranchStatus.MATCHED

    count = await branch_store.supersede_open_branches(generation.id, [matched.id])
    assert count == 1
    refreshed = await branch_store.get(left_open.id)
    assert refreshed.status is BranchStatus.SUPERSEDED
    # matched branch untouched by the supersede pass
    assert (await branch_store.get(matched.id)).status is BranchStatus.MATCHED


@pytest.mark.asyncio(loop_scope="session")
async def test_get_latest_generation_returns_most_recent_by_turn_index(
    branch_store, transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    await branch_store.create_generation(session_id, 0, root_count=1)
    second = await branch_store.create_generation(session_id, 1, root_count=1)

    latest = await branch_store.get_latest_generation(session_id)

    assert latest is not None
    assert latest.id == second.id


def test_branches_module_has_no_delete():
    """The branch store must be append-only. See CLAUDE.md invariant 6.

    Same AST-based scan as test_hypothesis_store.py's equivalent check:
    walk the module's AST rather than raw text so docstring prose that
    talks *about* the constraint doesn't trip it, and flag actual
    violations (DELETE in string literals, delete/remove-prefixed
    function names).
    """
    source = Path(branches_module.__file__).read_text()
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
                "the branch store must be append-only (see CLAUDE.md)"
            )

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lname = node.name.lower()
            assert not (
                lname.startswith("delete") or lname.startswith("remove")
            ), (
                f"function {node.name!r} at line {node.lineno} looks like a "
                "removal method — the branch store must be append-only "
                "(see CLAUDE.md)"
            )
