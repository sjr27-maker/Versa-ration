"""Append-only store for options — the clickable-button evidence
channel (see models.py's Option docstring and CLAUDE.md invariant 8).

No delete method, no DELETE SQL — an option's history is a fact once
recorded, same append-only spirit as every other store in this
project. Resolution is a status transition (open -> selected/
superseded) via UPDATE, same pattern as branches.
"""

from __future__ import annotations

from uuid import UUID

import asyncpg

from probe.models import Option, OptionStatus


class OptionStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create_options(self, options: list[Option]) -> list[Option]:
        if not options:
            return []
        async with self._pool.acquire() as conn, conn.transaction():
            for o in options:
                await conn.execute(
                    """
                        INSERT INTO options (
                            id, branch_id, generation_id, session_id,
                            turn_index, text, status, created_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """,
                    o.id,
                    o.branch_id,
                    o.generation_id,
                    o.session_id,
                    o.turn_index,
                    o.text,
                    o.status.value,
                    o.created_at,
                )
        return options

    async def get(self, option_id: UUID) -> Option | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM options WHERE id = $1", option_id)
        return self._row_to_option(row) if row is not None else None

    async def list_by_generation(self, generation_id: UUID) -> list[Option]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM options WHERE generation_id = $1 ORDER BY created_at",
                generation_id,
            )
        return [self._row_to_option(row) for row in rows]

    async def set_status(self, option_id: UUID, status: OptionStatus) -> Option:
        async with self._pool.acquire() as conn:
            updated = await conn.fetchval(
                "UPDATE options SET status = $2 WHERE id = $1 RETURNING id",
                option_id,
                status.value,
            )
        if updated is None:
            raise KeyError(f"option {option_id} not found")
        return await self._require(option_id)

    async def supersede_open_options(self, generation_id: UUID) -> int:
        """Close out a resolved generation's options: whatever is still
        `open` (never clicked) becomes `superseded`. An option already
        `selected` (set at click time, before this ever runs — see
        SessionLoop.handle_turn) is untouched by this blanket update,
        so no exclude list is needed the way branches' matched_chain
        needs one."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE options SET status = 'superseded' "
                "WHERE generation_id = $1 AND status = 'open'",
                generation_id,
            )
        return int(result.split()[-1])

    async def _require(self, option_id: UUID) -> Option:
        option = await self.get(option_id)
        if option is None:
            raise KeyError(f"option {option_id} not found")
        return option

    def _row_to_option(self, row) -> Option:
        return Option(
            id=row["id"],
            branch_id=row["branch_id"],
            generation_id=row["generation_id"],
            session_id=row["session_id"],
            turn_index=row["turn_index"],
            text=row["text"],
            status=OptionStatus(row["status"]),
            created_at=row["created_at"],
        )
