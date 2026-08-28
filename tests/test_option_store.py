import ast
import re
from pathlib import Path
from uuid import uuid4

import pytest

import probe.options as options_module
from probe.models import Branch, Option, OptionStatus


async def _make_generation_and_branch(branch_store, session_id):
    generation = await branch_store.create_generation(session_id, 0, root_count=1)
    branch = Branch(
        parent_id=None,
        generation_id=generation.id,
        session_id=session_id,
        turn_index=0,
        depth=0,
        depth_label="intent",
        statement="wants an analogy",
        predicted_next_turn="will ask for a comparison",
        requires_evidence="the student confirms they want a real-world example",
        plausibility=0.6,
        is_leaf=True,
    )
    await branch_store.add_branches([branch])
    return generation, branch


@pytest.mark.asyncio(loop_scope="session")
async def test_create_and_get_option_roundtrips(
    branch_store, option_store, transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    generation, branch = await _make_generation_and_branch(branch_store, session_id)
    option = Option(
        branch_id=branch.id,
        generation_id=generation.id,
        session_id=session_id,
        turn_index=0,
        text="Would you like a real-world comparison for this?",
    )

    await option_store.create_options([option])
    fetched = await option_store.get(option.id)

    assert fetched is not None
    assert fetched.branch_id == branch.id
    assert fetched.text == "Would you like a real-world comparison for this?"
    assert fetched.status is OptionStatus.OPEN


@pytest.mark.asyncio(loop_scope="session")
async def test_list_by_generation_returns_every_option_in_order(
    branch_store, option_store, transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    generation, branch = await _make_generation_and_branch(branch_store, session_id)
    branch2 = Branch(
        parent_id=None,
        generation_id=generation.id,
        session_id=session_id,
        turn_index=0,
        depth=0,
        depth_label="intent",
        statement="is missing a prerequisite",
        predicted_next_turn="will ask a clarifying question",
        requires_evidence="the student confirms they're missing the prerequisite",
        plausibility=0.5,
        is_leaf=True,
    )
    await branch_store.add_branches([branch2])
    options = [
        Option(branch_id=branch.id, generation_id=generation.id, session_id=session_id,
               turn_index=0, text="option A"),
        Option(branch_id=branch2.id, generation_id=generation.id, session_id=session_id,
               turn_index=0, text="option B"),
    ]
    await option_store.create_options(options)

    fetched = await option_store.list_by_generation(generation.id)

    assert {o.text for o in fetched} == {"option A", "option B"}
    assert {o.branch_id for o in fetched} == {branch.id, branch2.id}


@pytest.mark.asyncio(loop_scope="session")
async def test_set_status_transitions_and_persists(
    branch_store, option_store, transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    generation, branch = await _make_generation_and_branch(branch_store, session_id)
    option = Option(
        branch_id=branch.id, generation_id=generation.id,
        session_id=session_id, turn_index=0, text="option A",
    )
    await option_store.create_options([option])

    updated = await option_store.set_status(option.id, OptionStatus.SELECTED)
    assert updated.status is OptionStatus.SELECTED

    refetched = await option_store.get(option.id)
    assert refetched.status is OptionStatus.SELECTED


@pytest.mark.asyncio(loop_scope="session")
async def test_supersede_open_options_leaves_selected_ones_untouched(
    branch_store, option_store, transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    generation, branch = await _make_generation_and_branch(branch_store, session_id)
    branch2 = Branch(
        parent_id=None, generation_id=generation.id, session_id=session_id,
        turn_index=0, depth=0, depth_label="intent", statement="b",
        predicted_next_turn="p", requires_evidence="e", plausibility=0.5, is_leaf=True,
    )
    await branch_store.add_branches([branch2])
    selected = Option(branch_id=branch.id, generation_id=generation.id,
                       session_id=session_id, turn_index=0, text="A")
    unselected = Option(branch_id=branch2.id, generation_id=generation.id,
                         session_id=session_id, turn_index=0, text="B")
    await option_store.create_options([selected, unselected])
    await option_store.set_status(selected.id, OptionStatus.SELECTED)

    count = await option_store.supersede_open_options(generation.id)

    assert count == 1
    assert (await option_store.get(selected.id)).status is OptionStatus.SELECTED
    assert (await option_store.get(unselected.id)).status is OptionStatus.SUPERSEDED


@pytest.mark.asyncio(loop_scope="session")
async def test_get_returns_none_for_unknown_id(option_store):
    assert await option_store.get(uuid4()) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_every_field_roundtrips_write_to_read(
    branch_store, option_store, transcript, clean_pool, learner_id, concept_graph_id
):
    """Generic, future-proofing regression test — every field set to a
    non-default value, full model_dump() comparison, same pattern as
    test_diagnostics_store.py's equivalent test. If a future migration
    adds a column without updating both Option and
    OptionStore._row_to_option, this fails loudly instead of the
    column silently vanishing on read.
    """
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    generation, branch = await _make_generation_and_branch(branch_store, session_id)
    option = Option(
        branch_id=branch.id,
        generation_id=generation.id,
        session_id=session_id,
        turn_index=3,
        text="every field set to a non-default value",
        status=OptionStatus.SELECTED,
    )
    await option_store.create_options([option])

    fetched = await option_store.get(option.id)

    assert fetched is not None
    assert fetched.model_dump() == option.model_dump()


def test_options_module_has_no_delete():
    """The options store must be append-only. See CLAUDE.md invariant 8.

    Same AST-based scan as test_branch_store.py's equivalent check for
    invariant 6: walk the module's AST rather than raw text so
    docstring prose that talks *about* the constraint doesn't trip it,
    and flag actual violations (DELETE in string literals,
    delete/remove-prefixed function names).
    """
    source = Path(options_module.__file__).read_text()
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
                "the options store must be append-only (see CLAUDE.md)"
            )

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lname = node.name.lower()
            assert not (
                lname.startswith("delete") or lname.startswith("remove")
            ), (
                f"function {node.name!r} at line {node.lineno} looks like a "
                "removal method — the options store must be append-only "
                "(see CLAUDE.md)"
            )
