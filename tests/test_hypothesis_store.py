import ast
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest
from pydantic import ValidationError

from probe import store as store_module
from probe.models import ConceptNode, EvidenceRef, Hypothesis, Layer, Polarity, Tier


def _make_hypothesis(**overrides) -> Hypothesis:
    defaults = dict(
        layer=Layer.KNOWLEDGE,
        statement="user prefers terse responses",
        probability=0.6,
        confidence=0.4,
        tier=Tier.ACTIVE,
        conditions=["asked in a work context"],
    )
    defaults.update(overrides)
    return Hypothesis(**defaults)


@pytest.mark.asyncio(loop_scope="session")
async def test_create_and_retrieve(store):
    hyp = _make_hypothesis()
    await store.add(hyp)

    fetched = await store.get(hyp.id)

    assert fetched is not None
    assert fetched.id == hyp.id
    assert fetched.layer is Layer.KNOWLEDGE
    assert fetched.statement == hyp.statement
    assert fetched.probability == pytest.approx(0.6)
    assert fetched.confidence == pytest.approx(0.4)
    assert fetched.tier is Tier.ACTIVE
    assert fetched.conditions == ["asked in a work context"]
    assert fetched.evidence_refs == []
    assert fetched.counter_evidence_refs == []


@pytest.mark.asyncio(loop_scope="session")
async def test_reweight_appends_evidence_rather_than_overwriting(
    store, transcript, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    turn_a = await transcript.record_turn(session_id, 0, "turn a")
    turn_b = await transcript.record_turn(session_id, 1, "turn b")
    turn_c = await transcript.record_turn(session_id, 2, "turn c")
    initial_evidence = EvidenceRef(
        turn_id=turn_a,
        polarity=Polarity.SUPPORTING,
        timestamp=datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc),
    )
    hyp = _make_hypothesis(
        probability=0.5,
        confidence=0.2,
        evidence_refs=[initial_evidence],
    )
    await store.add(hyp)

    second = EvidenceRef(
        turn_id=turn_b,
        polarity=Polarity.SUPPORTING,
        timestamp=datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc),
    )
    after_first_reweight = await store.reweight(hyp.id, 0.7, 0.35, second)
    assert after_first_reweight.probability == pytest.approx(0.7)
    assert after_first_reweight.confidence == pytest.approx(0.35)
    assert [e.turn_id for e in after_first_reweight.evidence_refs] == [
        turn_a,
        turn_b,
    ]

    third = EvidenceRef(
        turn_id=turn_c,
        polarity=Polarity.CONTRADICTING,
        timestamp=datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc),
    )
    after_second_reweight = await store.reweight(hyp.id, 0.55, 0.45, third)
    assert after_second_reweight.probability == pytest.approx(0.55)
    assert after_second_reweight.confidence == pytest.approx(0.45)
    assert [e.turn_id for e in after_second_reweight.evidence_refs] == [
        turn_a,
        turn_b,
    ]
    assert [e.turn_id for e in after_second_reweight.counter_evidence_refs] == [
        turn_c
    ]


@pytest.mark.asyncio(loop_scope="session")
async def test_archived_hypothesis_can_be_resurrected(store):
    hyp = _make_hypothesis(tier=Tier.ACTIVE)
    await store.add(hyp)

    archived = await store.retier(hyp.id, Tier.ARCHIVED)
    assert archived.tier is Tier.ARCHIVED

    assert await store.list_by_layer(hyp.layer, tier=Tier.ACTIVE) == []
    archived_list = await store.list_by_layer(hyp.layer, tier=Tier.ARCHIVED)
    assert [h.id for h in archived_list] == [hyp.id]

    revived = await store.resurrect(hyp.id)
    assert revived.tier is Tier.ACTIVE

    active_list = await store.list_by_layer(hyp.layer, tier=Tier.ACTIVE)
    assert [h.id for h in active_list] == [hyp.id]


@pytest.mark.asyncio(loop_scope="session")
async def test_reweight_rejects_evidence_ref_with_nonexistent_turn_id(store):
    hyp = _make_hypothesis()
    await store.add(hyp)

    orphan_turn_id = uuid4()
    orphan_ref = EvidenceRef(
        turn_id=orphan_turn_id,
        polarity=Polarity.SUPPORTING,
        timestamp=datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc),
    )

    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await store.reweight(hyp.id, 0.7, 0.5, orphan_ref)

    # And the reweight transaction rolled back — probability/confidence
    # unchanged, no partial evidence row appended.
    reloaded = await store.get(hyp.id)
    assert reloaded.probability == pytest.approx(hyp.probability)
    assert reloaded.confidence == pytest.approx(hyp.confidence)
    assert reloaded.evidence_refs == []
    assert reloaded.counter_evidence_refs == []


def test_evidence_refs_rejects_contradicting_polarity():
    contradicting = EvidenceRef(
        turn_id=uuid4(),
        polarity=Polarity.CONTRADICTING,
        timestamp=datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc),
    )
    with pytest.raises(ValidationError, match="evidence_refs must contain only SUPPORTING"):
        _make_hypothesis(evidence_refs=[contradicting])


def test_counter_evidence_refs_rejects_supporting_polarity():
    supporting = EvidenceRef(
        turn_id=uuid4(),
        polarity=Polarity.SUPPORTING,
        timestamp=datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc),
    )
    with pytest.raises(
        ValidationError,
        match="counter_evidence_refs must contain only CONTRADICTING",
    ):
        _make_hypothesis(counter_evidence_refs=[supporting])


@pytest.mark.asyncio(loop_scope="session")
async def test_list_by_concept_filters_via_the_join_table(
    store, concept_graph, concept_graph_id
):
    await concept_graph.add_concept(
        ConceptNode(concept_graph_id=concept_graph_id, id="closures", name="Closures")
    )
    await concept_graph.add_concept(
        ConceptNode(concept_graph_id=concept_graph_id, id="loops", name="Loops")
    )

    linked = _make_hypothesis(layer=Layer.MENTAL_MODEL)
    unlinked = _make_hypothesis(layer=Layer.MENTAL_MODEL)
    other_layer = _make_hypothesis(layer=Layer.KNOWLEDGE)
    await store.add(linked)
    await store.add(unlinked)
    await store.add(other_layer)

    await store.link_concept(linked.id, concept_graph_id, "closures")
    await store.link_concept(other_layer.id, concept_graph_id, "closures")
    await store.link_concept(unlinked.id, concept_graph_id, "loops")

    mental_model_for_closures = await store.list_by_concept(
        concept_graph_id, "closures", layer=Layer.MENTAL_MODEL, tier=Tier.ACTIVE
    )
    assert [h.id for h in mental_model_for_closures] == [linked.id]

    all_for_closures = await store.list_by_concept(concept_graph_id, "closures")
    assert {h.id for h in all_for_closures} == {linked.id, other_layer.id}

    for_loops = await store.list_by_concept(
        concept_graph_id, "loops", layer=Layer.MENTAL_MODEL
    )
    assert [h.id for h in for_loops] == [unlinked.id]


@pytest.mark.asyncio(loop_scope="session")
async def test_link_concept_is_idempotent(store, concept_graph, concept_graph_id):
    await concept_graph.add_concept(
        ConceptNode(concept_graph_id=concept_graph_id, id="closures", name="Closures")
    )
    hyp = _make_hypothesis(layer=Layer.MENTAL_MODEL)
    await store.add(hyp)

    await store.link_concept(hyp.id, concept_graph_id, "closures")
    await store.link_concept(hyp.id, concept_graph_id, "closures")  # must not duplicate

    result = await store.list_by_concept(concept_graph_id, "closures")
    assert [h.id for h in result] == [hyp.id]


def test_store_module_has_no_delete():
    """The store must be append-only. See CLAUDE.md.

    We scan the module's AST rather than raw text so that prose in
    docstrings (which may talk *about* the constraint) doesn't trip the
    check. We flag actual violations: DELETE keywords appearing in string
    literals (i.e. SQL), and function definitions whose names start with
    delete/remove.
    """
    source = Path(store_module.__file__).read_text()
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
                "the hypothesis store must be append-only (see CLAUDE.md)"
            )

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lname = node.name.lower()
            assert not (
                lname.startswith("delete") or lname.startswith("remove")
            ), (
                f"function {node.name!r} at line {node.lineno} looks like a "
                "removal method — the hypothesis store must be append-only "
                "(see CLAUDE.md)"
            )
