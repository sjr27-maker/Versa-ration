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

from collections import defaultdict
from uuid import UUID

import asyncpg

from probe.ablation import AblationConfig, AblationCostSummary
from probe.models import TurnDiagnostics
from probe.row_mapping import assert_row_consumed


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
                    duration_ms, warnings, teach_failed, retry_count,
                    memory_match_found, memory_match_confirmed_resolution,
                    branching_skipped_by_memory, matched_fact_id, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
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
                diagnostics.retry_count,
                diagnostics.memory_match_found,
                diagnostics.memory_match_confirmed_resolution,
                diagnostics.branching_skipped_by_memory,
                diagnostics.matched_fact_id,
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

    async def mean_cost_by_config(self) -> list[AblationCostSummary]:
        """"What does this config cost" — mean per-turn wall-clock,
        call count, and retry count across every turn recorded under an
        identical AblationConfig, regardless of session. Grouping
        happens in Python, not SQL: a session's `ablation_config` is
        SQL NULL for "full system" (see TranscriptStore.get_ablation_config),
        and a NULL row must land in the exact same bucket as a session
        that stored the literal full-system config explicitly — hard-
        coding that equivalence as a SQL literal would drift the moment
        AblationConfig gains a field, where resolving both through
        AblationConfig() in Python cannot.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    s.ablation_config,
                    td.duration_ms,
                    td.total_call_count,
                    td.retry_count
                FROM turn_diagnostics td
                JOIN sessions s ON s.id = td.session_id
                """
            )

        buckets: dict[str, list[dict]] = defaultdict(list)
        configs_by_key: dict[str, AblationConfig] = {}
        for row in rows:
            stored = row["ablation_config"]
            config = AblationConfig() if stored is None else AblationConfig(**stored)
            key = config.model_dump_json()
            configs_by_key[key] = config
            buckets[key].append(dict(row))

        summaries: list[AblationCostSummary] = []
        for key, turn_rows in buckets.items():
            n = len(turn_rows)
            summaries.append(
                AblationCostSummary(
                    ablation_config=configs_by_key[key],
                    turn_count=n,
                    mean_duration_ms=sum(r["duration_ms"] for r in turn_rows) / n,
                    mean_call_count=sum(r["total_call_count"] for r in turn_rows) / n,
                    mean_retry_count=sum(r["retry_count"] for r in turn_rows) / n,
                )
            )
        return summaries

    def _row_to_diagnostics(self, row) -> TurnDiagnostics:
        mapped = dict(row)
        assert_row_consumed(TurnDiagnostics, mapped)
        return TurnDiagnostics(**mapped)
