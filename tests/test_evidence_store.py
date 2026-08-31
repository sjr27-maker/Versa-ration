"""EvidenceStore (evidence_records, migration 033) — append-only
(CLAUDE.md invariant 11), roundtrip, and source_type filtering.
"""

import ast
import re
from pathlib import Path

import pytest

import probe.evidence as evidence_module
from probe.models import EvidenceRecord, EvidenceSourceType


def test_evidence_module_has_no_delete():
    """Same AST-based scan as every other store's append-only check:
    walk the AST so docstring prose about the constraint doesn't trip
    it, and flag DELETE in string literals / delete|remove-prefixed
    function names."""
    source = Path(evidence_module.__file__).read_text()
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
                "the evidence store must be append-only (CLAUDE.md invariant 11)"
            )

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lname = node.name.lower()
            assert not (lname.startswith("delete") or lname.startswith("remove")), (
                f"function {node.name!r} at line {node.lineno} looks like a "
                "removal method — the evidence store must be append-only"
            )


def test_evidence_migration_has_no_delete():
    migrations_dir = Path(evidence_module.__file__).resolve().parent / "migrations"
    source = (migrations_dir / "033_evidence_records.sql").read_text()
    code_only = "\n".join(line.split("--", 1)[0] for line in source.splitlines())
    code_only = re.sub(r"\bON\s+DELETE\s+\w+", "", code_only, flags=re.IGNORECASE)
    assert not re.search(r"\bDELETE\b", code_only, re.IGNORECASE), (
        "033_evidence_records.sql contains a DELETE statement — "
        "evidence_records must be append-only (CLAUDE.md invariant 11)"
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_add_and_list_roundtrips(evidence_store, transcript, clean_pool, learner_id):
    session_id = await transcript.create_session(learner_id)
    rec = EvidenceRecord(
        source_type=EvidenceSourceType.STAGED_VERIFICATION,
        part="part_1_within_session",
        title="within-session memory pre-check fired",
        summary="mechanism verified: branching_skipped_by_memory=True on the callback turn",
        body={"matched_fact_id": "abc", "response": "the verbatim response text"},
        learner_id=learner_id,
        session_id=session_id,
    )
    await evidence_store.add(rec)

    fetched = await evidence_store.get(rec.id)
    assert fetched is not None
    assert fetched.source_type is EvidenceSourceType.STAGED_VERIFICATION
    assert fetched.part == "part_1_within_session"
    assert fetched.summary.startswith("mechanism verified")
    assert fetched.body == {"matched_fact_id": "abc", "response": "the verbatim response text"}
    assert fetched.learner_id == learner_id
    assert fetched.session_id == session_id


@pytest.mark.asyncio(loop_scope="session")
async def test_list_all_is_newest_first_and_filters_by_source_type(
    evidence_store, clean_pool
):
    for i in range(3):
        await evidence_store.add(
            EvidenceRecord(
                source_type=EvidenceSourceType.STAGED_VERIFICATION,
                part=f"part_{i}",
                title=f"staged {i}",
                summary="mechanism verified",
                body={"i": i},
            )
        )
    await evidence_store.add(
        EvidenceRecord(
            source_type=EvidenceSourceType.ORGANIC_SESSION,
            part="organic",
            title="organic finding",
            summary="observed in a real session",
            body={},
        )
    )

    all_rows = await evidence_store.list_all()
    assert len(all_rows) == 4
    # newest first
    assert [r.created_at for r in all_rows] == sorted(
        (r.created_at for r in all_rows), reverse=True
    )

    staged = await evidence_store.list_all(EvidenceSourceType.STAGED_VERIFICATION)
    assert len(staged) == 3
    assert all(r.source_type is EvidenceSourceType.STAGED_VERIFICATION for r in staged)

    organic = await evidence_store.list_all(EvidenceSourceType.ORGANIC_SESSION)
    assert [r.title for r in organic] == ["organic finding"]
