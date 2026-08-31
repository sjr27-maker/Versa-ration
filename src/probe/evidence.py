"""Append-only store for `evidence_records` (migration 033) — a
durable, labeled log of verification findings.

See CLAUDE.md invariant 11: no delete method, no DELETE SQL. A finding,
once recorded, is a historical fact about what a check saw at that
moment. `source_type` keeps a staged mechanism test (`STAGED_VERIFICATION`)
visibly separate from evidence produced by a real student session
(`ORGANIC_SESSION`), so the Evidence page can never present one as the
other.
"""

from __future__ import annotations

import json
from uuid import UUID

import asyncpg

from probe.models import EvidenceRecord, EvidenceSourceType
from probe.row_mapping import assert_row_consumed


class EvidenceStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def add(self, record: EvidenceRecord) -> EvidenceRecord:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO evidence_records (
                    id, source_type, part, title, summary, body,
                    learner_id, session_id, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                record.id,
                record.source_type.value,
                record.part,
                record.title,
                record.summary,
                json.dumps(record.body),
                record.learner_id,
                record.session_id,
                record.created_at,
            )
        return record

    async def list_all(
        self, source_type: EvidenceSourceType | None = None
    ) -> list[EvidenceRecord]:
        """Every recorded finding, newest first. Pass `source_type` to
        restrict to staged or organic evidence only."""
        async with self._pool.acquire() as conn:
            if source_type is None:
                rows = await conn.fetch(
                    "SELECT * FROM evidence_records ORDER BY created_at DESC"
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM evidence_records WHERE source_type = $1 "
                    "ORDER BY created_at DESC",
                    source_type.value,
                )
        return [self._row_to_record(row) for row in rows]

    async def get(self, record_id: UUID) -> EvidenceRecord | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM evidence_records WHERE id = $1", record_id
            )
        return self._row_to_record(row) if row is not None else None

    def _row_to_record(self, row) -> EvidenceRecord:
        mapped = dict(row)
        body = mapped.get("body")
        if isinstance(body, str):
            mapped["body"] = json.loads(body)
        assert_row_consumed(EvidenceRecord, mapped)
        return EvidenceRecord(**mapped)
