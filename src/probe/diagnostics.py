"""Append-only store for turn_diagnostics — one row per handle_turn()
call, persisting what SessionLoop already computes each turn (call
counts, whether MAX_CALLS_PER_TURN fired, entropy_bits, wall-clock
duration, warnings, whether Teach itself failed) so the web UI's
Diagnostics panel can read it directly instead of re-deriving anything
— see the "zero business logic in the UI" constraint this store exists
to satisfy.

No delete method, no DELETE SQL — a turn's diagnostics are a
historical fact once recorded, same append-only spirit as every other
store in this project.
"""

from __future__ import annotations

from uuid import UUID

import asyncpg

from probe.models import TurnDiagnostics


class TurnDiagnosticsStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def record(self, diagnostics: TurnDiagnostics) -> TurnDiagnostics:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO turn_diagnostics (
                    id, session_id, turn_index, node_call_counts,
                    total_call_count, guardrail_fired, entropy_bits,
                    duration_ms, warnings, teach_failed, inferred_topic,
                    topic_seeded_new, retry_count, options_missed, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """,
                diagnostics.id,
                diagnostics.session_id,
                diagnostics.turn_index,
                diagnostics.node_call_counts,
                diagnostics.total_call_count,
                diagnostics.guardrail_fired,
                diagnostics.entropy_bits,
                diagnostics.duration_ms,
                diagnostics.warnings,
                diagnostics.teach_failed,
                diagnostics.inferred_topic,
                diagnostics.topic_seeded_new,
                diagnostics.retry_count,
                diagnostics.options_missed,
                diagnostics.created_at,
            )
        return diagnostics

    async def get_for_turn(
        self, session_id: UUID, turn_index: int
    ) -> TurnDiagnostics | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM turn_diagnostics WHERE session_id = $1 AND turn_index = $2",
                session_id,
                turn_index,
            )
        return None if row is None else self._row_to_diagnostics(row)

    async def list_for_session(self, session_id: UUID) -> list[TurnDiagnostics]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM turn_diagnostics WHERE session_id = $1 ORDER BY turn_index",
                session_id,
            )
        return [self._row_to_diagnostics(row) for row in rows]

    def _row_to_diagnostics(self, row) -> TurnDiagnostics:
        return TurnDiagnostics(
            id=row["id"],
            session_id=row["session_id"],
            turn_index=row["turn_index"],
            node_call_counts=row["node_call_counts"],
            total_call_count=row["total_call_count"],
            guardrail_fired=row["guardrail_fired"],
            entropy_bits=row["entropy_bits"],
            duration_ms=row["duration_ms"],
            warnings=row["warnings"],
            teach_failed=row["teach_failed"],
            inferred_topic=row["inferred_topic"],
            topic_seeded_new=row["topic_seeded_new"],
            retry_count=row["retry_count"],
            options_missed=row["options_missed"],
            created_at=row["created_at"],
        )
