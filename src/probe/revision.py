"""WorldModelRevisionStore: claims about the concept graph, reviewed by
a human before anything is applied.

A WorldModelRevision is never auto-applied. It sits `pending` until
`approve()` or `reject()` is called (see `probe review-revisions`).
`approve()` deliberately does not attempt to parse `proposed_change`
(free text) into a structured edit — the caller must supply an explicit
`field_updates` dict naming exactly which ConceptNode fields to set and
to what. That's the human-in-the-loop step: a person reads
`proposed_change`, decides what it actually means as a structured edit,
and confirms it. `approve()` then applies it to `concept_nodes` and
records it as `applied_field_updates` on the revision itself, so the
applied edit and the revision that justified it are one row, not two
things that can drift apart.

No delete method here, no DELETE SQL. Status transitions are UPDATEs,
same as HypothesisStore.reweight/retier — the append-only invariant is
about removal, not mutation.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg

from probe.models import EvidenceRef, Polarity, RevisionStatus, WorldModelRevision

_MUTABLE_STR_FIELDS = frozenset({"name"})
_MUTABLE_LIST_FIELDS = frozenset(
    {"common_misconceptions", "representations", "diagnostic_questions"}
)
_MUTABLE_CONCEPT_FIELDS = _MUTABLE_STR_FIELDS | _MUTABLE_LIST_FIELDS


class RevisionApplicationError(Exception):
    """`field_updates` passed to `approve()` was empty, named a field
    ConceptNode doesn't expose for revision, or had the wrong type.

    `prerequisites` and `id` are deliberately not revisable this way:
    editing prerequisites reopens the same existence/cycle validation
    ConceptGraph.add_batch does for seeding, which this human-review
    path doesn't attempt to replicate.
    """


def _validate_field_updates(field_updates: dict[str, Any]) -> None:
    if not field_updates:
        raise RevisionApplicationError(
            "approve() requires at least one field update — pass the "
            "structured edit a human decided proposed_change actually means"
        )
    for key, value in field_updates.items():
        if key not in _MUTABLE_CONCEPT_FIELDS:
            raise RevisionApplicationError(
                f"{key!r} is not a field ConceptGraph allows revising this "
                f"way (allowed: {sorted(_MUTABLE_CONCEPT_FIELDS)})"
            )
        if key in _MUTABLE_STR_FIELDS and not isinstance(value, str):
            raise RevisionApplicationError(f"{key!r} must be a string")
        if key in _MUTABLE_LIST_FIELDS and not (
            isinstance(value, list) and all(isinstance(v, str) for v in value)
        ):
            raise RevisionApplicationError(f"{key!r} must be a list of strings")


class WorldModelRevisionStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def propose(self, revision: WorldModelRevision) -> WorldModelRevision:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO world_model_revisions (
                        id, concept_graph_id, concept_id, proposed_change,
                        confidence, status, applied_field_updates,
                        created_at, resolved_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    revision.id,
                    revision.concept_graph_id,
                    revision.concept_id,
                    revision.proposed_change,
                    revision.confidence,
                    revision.status.value,
                    revision.applied_field_updates,
                    revision.created_at,
                    revision.resolved_at,
                )
                for ref in revision.evidence_refs:
                    await conn.execute(
                        """
                        INSERT INTO world_model_revision_evidence (
                            id, revision_id, turn_id, polarity, timestamp
                        ) VALUES ($1, $2, $3, $4, $5)
                        """,
                        ref.id,
                        revision.id,
                        ref.turn_id,
                        ref.polarity.value,
                        ref.timestamp,
                    )
        return await self._require(revision.id)

    async def get(self, id: UUID) -> WorldModelRevision | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM world_model_revisions WHERE id = $1", id
            )
            if row is None:
                return None
            ev_rows = await conn.fetch(
                """
                SELECT * FROM world_model_revision_evidence
                WHERE revision_id = $1
                ORDER BY timestamp, id
                """,
                id,
            )
        return self._row_to_revision(row, ev_rows)

    async def list_pending(
        self,
        concept_graph_id: UUID | None = None,
        concept_id: str | None = None,
    ) -> list[WorldModelRevision]:
        """concept_id is only unique within its concept_graph_id, so
        filtering by concept_id alone would be ambiguous — pass both
        together, or neither (list pending across every graph)."""
        if concept_id is not None and concept_graph_id is None:
            raise ValueError(
                "list_pending(concept_id=...) also requires concept_graph_id "
                "— concept_id alone is ambiguous across graphs"
            )
        async with self._pool.acquire() as conn:
            if concept_graph_id is None:
                rows = await conn.fetch(
                    """
                    SELECT * FROM world_model_revisions
                    WHERE status = 'pending'
                    ORDER BY created_at, id
                    """
                )
            elif concept_id is None:
                rows = await conn.fetch(
                    """
                    SELECT * FROM world_model_revisions
                    WHERE status = 'pending' AND concept_graph_id = $1
                    ORDER BY created_at, id
                    """,
                    concept_graph_id,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM world_model_revisions
                    WHERE status = 'pending' AND concept_graph_id = $1
                        AND concept_id = $2
                    ORDER BY created_at, id
                    """,
                    concept_graph_id,
                    concept_id,
                )
            ids = [r["id"] for r in rows]
            ev_by_rev: dict[UUID, list] = {rid: [] for rid in ids}
            if ids:
                ev_rows = await conn.fetch(
                    """
                    SELECT * FROM world_model_revision_evidence
                    WHERE revision_id = ANY($1::uuid[])
                    ORDER BY timestamp, id
                    """,
                    ids,
                )
                for ev in ev_rows:
                    ev_by_rev[ev["revision_id"]].append(ev)
        return [self._row_to_revision(row, ev_by_rev[row["id"]]) for row in rows]

    async def list_by_learner(
        self, learner_id: UUID, status: RevisionStatus | None = None
    ) -> list[WorldModelRevision]:
        """Revisions with at least one evidence_ref traceable to this
        learner's sessions (evidence_ref -> turn -> session ->
        learner_id) — same indirection as
        `HypothesisStore.list_by_learner`, for the same reason: a
        revision has no direct learner_id column.
        """
        conditions = ["s.learner_id = $1"]
        params: list[Any] = [learner_id]
        if status is not None:
            params.append(status.value)
            conditions.append(f"wmr.status = ${len(params)}")
        where = " AND ".join(conditions)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT DISTINCT wmr.* FROM world_model_revisions wmr
                JOIN world_model_revision_evidence wre ON wre.revision_id = wmr.id
                JOIN turns t ON t.id = wre.turn_id
                JOIN sessions s ON s.id = t.session_id
                WHERE {where}
                ORDER BY wmr.created_at, wmr.id
                """,
                *params,
            )
            ids = [r["id"] for r in rows]
            ev_by_rev: dict[UUID, list] = {rid: [] for rid in ids}
            if ids:
                ev_rows = await conn.fetch(
                    """
                    SELECT * FROM world_model_revision_evidence
                    WHERE revision_id = ANY($1::uuid[])
                    ORDER BY timestamp, id
                    """,
                    ids,
                )
                for ev in ev_rows:
                    ev_by_rev[ev["revision_id"]].append(ev)
        return [self._row_to_revision(row, ev_by_rev[row["id"]]) for row in rows]

    async def approve(
        self, id: UUID, field_updates: dict[str, Any]
    ) -> WorldModelRevision:
        _validate_field_updates(field_updates)

        set_clauses = []
        params: list[Any] = []
        for key, value in field_updates.items():
            params.append(value)
            set_clauses.append(f"{key} = ${len(params)}")

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT concept_graph_id, concept_id, status "
                    "FROM world_model_revisions WHERE id = $1 FOR UPDATE",
                    id,
                )
                if row is None:
                    raise KeyError(f"revision {id} not found")
                if row["status"] != RevisionStatus.PENDING.value:
                    raise ValueError(
                        f"revision {id} is not pending "
                        f"(status={row['status']!r})"
                    )
                concept_graph_id = row["concept_graph_id"]
                concept_id = row["concept_id"]

                await conn.execute(
                    f"""
                    UPDATE concept_nodes
                    SET {", ".join(set_clauses)}, updated_at = NOW()
                    WHERE concept_graph_id = ${len(params) + 1}
                        AND id = ${len(params) + 2}
                    """,
                    *params,
                    concept_graph_id,
                    concept_id,
                )
                await conn.execute(
                    """
                    UPDATE world_model_revisions
                    SET status = 'approved',
                        applied_field_updates = $2,
                        resolved_at = NOW()
                    WHERE id = $1
                    """,
                    id,
                    field_updates,
                )
        return await self._require(id)

    async def reject(self, id: UUID) -> WorldModelRevision:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT status FROM world_model_revisions "
                    "WHERE id = $1 FOR UPDATE",
                    id,
                )
                if row is None:
                    raise KeyError(f"revision {id} not found")
                if row["status"] != RevisionStatus.PENDING.value:
                    raise ValueError(
                        f"revision {id} is not pending "
                        f"(status={row['status']!r})"
                    )
                await conn.execute(
                    """
                    UPDATE world_model_revisions
                    SET status = 'rejected', resolved_at = NOW()
                    WHERE id = $1
                    """,
                    id,
                )
        return await self._require(id)

    async def _require(self, id: UUID) -> WorldModelRevision:
        revision = await self.get(id)
        if revision is None:
            raise KeyError(f"revision {id} not found")
        return revision

    def _row_to_revision(self, row, ev_rows) -> WorldModelRevision:
        refs = [
            EvidenceRef(
                id=ev["id"],
                turn_id=ev["turn_id"],
                polarity=Polarity(ev["polarity"]),
                timestamp=ev["timestamp"],
            )
            for ev in ev_rows
        ]
        return WorldModelRevision(
            id=row["id"],
            concept_graph_id=row["concept_graph_id"],
            concept_id=row["concept_id"],
            proposed_change=row["proposed_change"],
            evidence_refs=refs,
            confidence=row["confidence"],
            status=RevisionStatus(row["status"]),
            applied_field_updates=row["applied_field_updates"],
            created_at=row["created_at"],
            resolved_at=row["resolved_at"],
        )
