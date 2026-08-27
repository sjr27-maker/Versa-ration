"""Append-only store for HypothesisGenerator's speculative prediction
tree — branch_generations (one row per generate() call) and branches
(the tree itself).

See CLAUDE.md invariant 6: this class has no delete method and its SQL
contains no DELETE. Resolution is expressed as a status transition
(open -> matched/unmatched/superseded) via UPDATE, same pattern as
WorldModelRevisionStore's pending -> approved/rejected.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import asyncpg

from probe.models import Branch, BranchGenerationMeta, BranchStatus


class BranchStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create_generation(
        self, session_id: UUID, turn_index: int, root_count: int
    ) -> BranchGenerationMeta:
        generation_id = uuid4()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO branch_generations (id, session_id, turn_index, root_count)
                VALUES ($1, $2, $3, $4)
                RETURNING id, session_id, turn_index, root_count, created_at
                """,
                generation_id,
                session_id,
                turn_index,
                root_count,
            )
        return BranchGenerationMeta(**dict(row))

    async def add_branches(self, branches: list[Branch]) -> list[Branch]:
        if not branches:
            return []
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for b in branches:
                    await conn.execute(
                        """
                        INSERT INTO branches (
                            id, parent_id, generation_id, session_id, turn_index,
                            depth, depth_label, statement, predicted_next_turn,
                            plausibility, is_leaf, status, created_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                        """,
                        b.id,
                        b.parent_id,
                        b.generation_id,
                        b.session_id,
                        b.turn_index,
                        b.depth,
                        b.depth_label,
                        b.statement,
                        b.predicted_next_turn,
                        b.plausibility,
                        b.is_leaf,
                        b.status.value,
                        b.created_at,
                    )
        return branches

    async def get(self, branch_id: UUID) -> Branch | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM branches WHERE id = $1", branch_id
            )
        return self._row_to_branch(row) if row is not None else None

    async def get_open_leaves(self, generation_id: UUID) -> list[Branch]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM branches
                WHERE generation_id = $1 AND status = 'open' AND is_leaf = TRUE
                ORDER BY created_at, id
                """,
                generation_id,
            )
        return [self._row_to_branch(row) for row in rows]

    async def get_ancestors(self, branch_id: UUID) -> list[Branch]:
        """The chain from `branch_id`'s immediate parent up to the
        root, not including `branch_id` itself. Empty for a root
        (depth 0) branch."""
        ancestors: list[Branch] = []
        current = await self.get(branch_id)
        if current is None:
            raise KeyError(f"branch {branch_id} not found")
        while current.parent_id is not None:
            parent = await self.get(current.parent_id)
            if parent is None:
                raise KeyError(f"branch {current.parent_id} not found")
            ancestors.append(parent)
            current = parent
        return ancestors

    async def set_status(self, branch_id: UUID, status: BranchStatus) -> Branch:
        async with self._pool.acquire() as conn:
            updated = await conn.fetchval(
                "UPDATE branches SET status = $2 WHERE id = $1 RETURNING id",
                branch_id,
                status.value,
            )
        if updated is None:
            raise KeyError(f"branch {branch_id} not found")
        return await self._require(branch_id)

    async def supersede_open_branches(
        self, generation_id: UUID, exclude_ids: list[UUID] | None = None
    ) -> int:
        """Close out a fully-resolved generation: every branch still
        `open` in it (i.e. intermediate depths that were never leaves,
        or leaves resolve() didn't touch) becomes `superseded`, so no
        branch from a resolved generation is left dangling as `open`.
        """
        exclude = exclude_ids or []
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE branches
                SET status = 'superseded'
                WHERE generation_id = $1 AND status = 'open'
                    AND NOT (id = ANY($2::uuid[]))
                """,
                generation_id,
                exclude,
            )
        # asyncpg returns "UPDATE <n>"
        return int(result.split()[-1])

    async def get_latest_generation(
        self, session_id: UUID
    ) -> BranchGenerationMeta | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, session_id, turn_index, root_count, created_at
                FROM branch_generations
                WHERE session_id = $1
                ORDER BY turn_index DESC, created_at DESC
                LIMIT 1
                """,
                session_id,
            )
        return BranchGenerationMeta(**dict(row)) if row is not None else None

    async def _require(self, branch_id: UUID) -> Branch:
        branch = await self.get(branch_id)
        if branch is None:
            raise KeyError(f"branch {branch_id} not found")
        return branch

    def _row_to_branch(self, row) -> Branch:
        return Branch(
            id=row["id"],
            parent_id=row["parent_id"],
            generation_id=row["generation_id"],
            session_id=row["session_id"],
            turn_index=row["turn_index"],
            depth=row["depth"],
            depth_label=row["depth_label"],
            statement=row["statement"],
            predicted_next_turn=row["predicted_next_turn"],
            plausibility=row["plausibility"],
            is_leaf=row["is_leaf"],
            status=BranchStatus(row["status"]),
            created_at=row["created_at"],
        )
