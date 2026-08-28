"""LearnerStore: identity-level records only.

No behavioral logic lives here — what a learner knows, believes, or is
working toward belongs on `Hypothesis` (evidence-backed claims) or
`OverlayEntry` (current concept-mastery state). This is just "which
conversations belong to the same person," created once per learner and
referenced by `sessions.learner_id` (NOT NULL, ON DELETE RESTRICT — a
learner can't be removed out from under sessions that reference it).

No delete method here either, same append-only spirit as the rest of
this schema: once a learner exists, sessions/turns may already
reference it, so removing one would orphan history the same way
deleting a hypothesis or a concept would.
"""

from __future__ import annotations

from uuid import UUID

import asyncpg

from probe.models import Learner, LearnerSummary


class LearnerStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create(
        self,
        label: str | None = None,
        display_name: str | None = None,
        preferred_register: str | None = None,
        timezone: str | None = None,
    ) -> Learner:
        learner = Learner(
            label=label,
            display_name=display_name,
            preferred_register=preferred_register,
            timezone=timezone,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO learners (
                    id, label, display_name, preferred_register, timezone,
                    created_at
                ) VALUES ($1, $2, $3, $4, $5, $6)
                """,
                learner.id,
                learner.label,
                learner.display_name,
                learner.preferred_register,
                learner.timezone,
                learner.created_at,
            )
        return learner

    async def get(self, id: UUID) -> Learner | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM learners WHERE id = $1", id)
        return self._row_to_learner(row) if row is not None else None

    async def get_by_label(self, label: str) -> Learner | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM learners WHERE label = $1", label
            )
        return self._row_to_learner(row) if row is not None else None

    async def list_all_with_session_counts(self) -> list[LearnerSummary]:
        """Every learner plus their session count and most recent
        session's timestamp — for the Setup page's existing-learner
        picker ("label, session count, and last session date")."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    l.*,
                    count(s.id) AS session_count,
                    max(s.created_at) AS last_session_at
                FROM learners l
                LEFT JOIN sessions s ON s.learner_id = l.id
                GROUP BY l.id
                ORDER BY last_session_at DESC NULLS LAST, l.created_at DESC
                """
            )
        return [
            LearnerSummary(
                learner=self._row_to_learner(row),
                session_count=row["session_count"],
                last_session_at=row["last_session_at"],
            )
            for row in rows
        ]

    def _row_to_learner(self, row) -> Learner:
        return Learner(
            id=row["id"],
            label=row["label"],
            display_name=row["display_name"],
            preferred_register=row["preferred_register"],
            timezone=row["timezone"],
            created_at=row["created_at"],
        )
