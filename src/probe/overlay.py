"""LearnerOverlay: current per-learner, per-concept belief state.

Unlike `HypothesisStore`, this is *not* append-only. It tracks current
state, not a claim-with-evidence trail — `set_state` upserts on
(learner_id, concept_graph_id, concept_id) rather than recording
history. This is a deliberate difference in contract, not an oversight:
`OverlayEntry` is not a `Hypothesis` and must not be routed through
`HypothesisStore`.

concept_id is only unique within its concept_graph_id (a deployment can
seed multiple topics), so every method here takes both.
"""

from __future__ import annotations

from uuid import UUID

import asyncpg

from probe.models import OverlayEntry, OverlayState


class LearnerOverlay:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def set_state(
        self,
        learner_id: UUID,
        concept_graph_id: UUID,
        concept_id: str,
        state: OverlayState,
        confidence: float,
    ) -> OverlayEntry:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO learner_overlay (
                    learner_id, concept_graph_id, concept_id, state,
                    confidence, updated_at
                ) VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (learner_id, concept_graph_id, concept_id) DO UPDATE
                    SET state      = EXCLUDED.state,
                        confidence = EXCLUDED.confidence,
                        updated_at = EXCLUDED.updated_at
                RETURNING concept_graph_id, concept_id, state, confidence, updated_at
                """,
                learner_id,
                concept_graph_id,
                concept_id,
                state.value,
                confidence,
            )
        return self._row_to_entry(row)

    async def get_state(
        self, learner_id: UUID, concept_graph_id: UUID, concept_id: str
    ) -> OverlayEntry | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT concept_graph_id, concept_id, state, confidence, updated_at
                FROM learner_overlay
                WHERE learner_id = $1 AND concept_graph_id = $2 AND concept_id = $3
                """,
                learner_id,
                concept_graph_id,
                concept_id,
            )
        return self._row_to_entry(row) if row is not None else None

    async def get_full_overlay(self, learner_id: UUID) -> list[OverlayEntry]:
        """Every overlay entry for this learner, across every graph
        they've touched. Returned as a list, not a dict keyed by
        concept_id: the same concept_id string can exist in more than
        one graph, so it isn't a safe unique key on its own — each
        entry carries its own concept_graph_id."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT concept_graph_id, concept_id, state, confidence, updated_at
                FROM learner_overlay
                WHERE learner_id = $1
                ORDER BY concept_graph_id, concept_id
                """,
                learner_id,
            )
        return [self._row_to_entry(row) for row in rows]

    def _row_to_entry(self, row) -> OverlayEntry:
        return OverlayEntry(
            concept_graph_id=row["concept_graph_id"],
            concept_id=row["concept_id"],
            state=OverlayState(row["state"]),
            confidence=row["confidence"],
            updated_at=row["updated_at"],
        )
