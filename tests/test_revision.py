import pytest

from probe.models import (
    ConceptNode,
    EvidenceRef,
    Polarity,
    RevisionStatus,
    WorldModelRevision,
)
from probe.revision import RevisionApplicationError


async def _seed_concept(concept_graph, concept_graph_id, id_="closures"):
    await concept_graph.add_concept(
        ConceptNode(
            concept_graph_id=concept_graph_id,
            id=id_,
            name="Closures",
            common_misconceptions=["closures copy the value at definition time"],
        )
    )
    return id_


@pytest.mark.asyncio(loop_scope="session")
async def test_propose_creates_a_pending_revision_with_evidence(
    concept_graph, concept_graph_id, transcript, revision_store, learner_id
):
    concept_id = await _seed_concept(concept_graph, concept_graph_id)
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    turn_id = await transcript.record_turn(session_id, 0, "turn zero")

    revision = await revision_store.propose(
        WorldModelRevision(
            concept_graph_id=concept_graph_id,
            concept_id=concept_id,
            proposed_change="misconceptions list is missing a common case",
            evidence_refs=[
                EvidenceRef(turn_id=turn_id, polarity=Polarity.SUPPORTING)
            ],
            confidence=0.7,
        )
    )

    assert revision.status is RevisionStatus.PENDING
    assert revision.applied_field_updates is None
    assert len(revision.evidence_refs) == 1

    pending = await revision_store.list_pending()
    assert [r.id for r in pending] == [revision.id]

    pending_for_concept = await revision_store.list_pending(
        concept_graph_id=concept_graph_id, concept_id=concept_id
    )
    assert [r.id for r in pending_for_concept] == [revision.id]

    pending_other = await revision_store.list_pending(
        concept_graph_id=concept_graph_id, concept_id="nonexistent"
    )
    assert pending_other == []

    with pytest.raises(ValueError):
        # concept_id without concept_graph_id is ambiguous across graphs.
        await revision_store.list_pending(concept_id=concept_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_approve_mutates_only_specified_field_and_links_revision(
    concept_graph, concept_graph_id, transcript, revision_store, learner_id
):
    concept_id = await _seed_concept(concept_graph, concept_graph_id)
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    turn_id = await transcript.record_turn(session_id, 0, "turn zero")

    revision = await revision_store.propose(
        WorldModelRevision(
            concept_graph_id=concept_graph_id,
            concept_id=concept_id,
            proposed_change="add the reference-vs-value misconception",
            evidence_refs=[
                EvidenceRef(turn_id=turn_id, polarity=Polarity.SUPPORTING)
            ],
            confidence=0.8,
        )
    )

    before = await concept_graph.get_concept(concept_graph_id, concept_id)
    assert before.representations == []

    updated = await revision_store.approve(
        revision.id,
        {
            "common_misconceptions": [
                "closures copy the value at definition time",
                "closures capture by reference, not by value",
            ]
        },
    )

    assert updated.status is RevisionStatus.APPROVED
    assert updated.resolved_at is not None
    assert updated.applied_field_updates == {
        "common_misconceptions": [
            "closures copy the value at definition time",
            "closures capture by reference, not by value",
        ]
    }

    after = await concept_graph.get_concept(concept_graph_id, concept_id)
    assert after.common_misconceptions == [
        "closures copy the value at definition time",
        "closures capture by reference, not by value",
    ]
    # Only the specified field changed.
    assert after.representations == before.representations
    assert after.name == before.name


@pytest.mark.asyncio(loop_scope="session")
async def test_approve_rejects_unknown_field(
    concept_graph, concept_graph_id, transcript, revision_store, learner_id
):
    concept_id = await _seed_concept(concept_graph, concept_graph_id)
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    turn_id = await transcript.record_turn(session_id, 0, "turn zero")
    revision = await revision_store.propose(
        WorldModelRevision(
            concept_graph_id=concept_graph_id,
            concept_id=concept_id,
            proposed_change="rename this concept",
            evidence_refs=[
                EvidenceRef(turn_id=turn_id, polarity=Polarity.SUPPORTING)
            ],
            confidence=0.5,
        )
    )

    with pytest.raises(RevisionApplicationError):
        await revision_store.approve(revision.id, {"id": "renamed"})

    still_pending = await revision_store.get(revision.id)
    assert still_pending.status is RevisionStatus.PENDING


@pytest.mark.asyncio(loop_scope="session")
async def test_approve_rejects_empty_field_updates(
    concept_graph, concept_graph_id, transcript, revision_store, learner_id
):
    concept_id = await _seed_concept(concept_graph, concept_graph_id)
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    turn_id = await transcript.record_turn(session_id, 0, "turn zero")
    revision = await revision_store.propose(
        WorldModelRevision(
            concept_graph_id=concept_graph_id,
            concept_id=concept_id,
            proposed_change="something vague",
            evidence_refs=[
                EvidenceRef(turn_id=turn_id, polarity=Polarity.SUPPORTING)
            ],
            confidence=0.5,
        )
    )

    with pytest.raises(RevisionApplicationError):
        await revision_store.approve(revision.id, {})


@pytest.mark.asyncio(loop_scope="session")
async def test_reject_leaves_concept_untouched_and_marks_rejected(
    concept_graph, concept_graph_id, transcript, revision_store, learner_id
):
    concept_id = await _seed_concept(concept_graph, concept_graph_id)
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    turn_id = await transcript.record_turn(session_id, 0, "turn zero")

    revision = await revision_store.propose(
        WorldModelRevision(
            concept_graph_id=concept_graph_id,
            concept_id=concept_id,
            proposed_change="this concept definition seems wrong",
            evidence_refs=[
                EvidenceRef(turn_id=turn_id, polarity=Polarity.SUPPORTING)
            ],
            confidence=0.6,
        )
    )

    before = await concept_graph.get_concept(concept_graph_id, concept_id)
    rejected = await revision_store.reject(revision.id)

    assert rejected.status is RevisionStatus.REJECTED
    assert rejected.resolved_at is not None
    assert rejected.applied_field_updates is None

    after = await concept_graph.get_concept(concept_graph_id, concept_id)
    assert after == before

    pending = await revision_store.list_pending()
    assert pending == []


@pytest.mark.asyncio(loop_scope="session")
async def test_approve_on_already_resolved_revision_raises(
    concept_graph, concept_graph_id, transcript, revision_store, learner_id
):
    concept_id = await _seed_concept(concept_graph, concept_graph_id)
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    turn_id = await transcript.record_turn(session_id, 0, "turn zero")
    revision = await revision_store.propose(
        WorldModelRevision(
            concept_graph_id=concept_graph_id,
            concept_id=concept_id,
            proposed_change="x",
            evidence_refs=[
                EvidenceRef(turn_id=turn_id, polarity=Polarity.SUPPORTING)
            ],
            confidence=0.5,
        )
    )
    await revision_store.reject(revision.id)

    with pytest.raises(ValueError):
        await revision_store.approve(revision.id, {"name": "Closures (renamed)"})


def test_revision_store_module_has_no_delete():
    import ast
    import re
    from pathlib import Path

    from probe import revision as revision_module

    source = Path(revision_module.__file__).read_text()
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
                "WorldModelRevisionStore must be append-only"
            )

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lname = node.name.lower()
            assert not (lname.startswith("delete") or lname.startswith("remove")), (
                f"function {node.name!r} at line {node.lineno} looks like a "
                "removal method — WorldModelRevisionStore must be append-only"
            )
