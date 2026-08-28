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

from probe.models import (
    Branch,
    BranchGenerationMeta,
    BranchMatchRatePoint,
    BranchStatus,
    PathRequirement,
    RecurringIntent,
)


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
                            requires_evidence, evidence_satisfied,
                            plausibility, is_leaf, status, matched_via, created_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
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
                        b.requires_evidence,
                        b.evidence_satisfied,
                        b.plausibility,
                        b.is_leaf,
                        b.status.value,
                        b.matched_via,
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

    async def list_by_generation(self, generation_id: UUID) -> list[Branch]:
        """Every branch in one generation, any depth or status — the
        full tree, not just its open leaves (see get_open_leaves for
        that narrower read). For rendering the web UI's branch tree
        panel: depth/parent_id chains, status, everything."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM branches WHERE generation_id = $1 ORDER BY depth, created_at",
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

    async def set_matched(self, branch_id: UUID, matched_via: str) -> Branch:
        """Marks a branch matched AND records which channel resolved
        it, in one UPDATE — the only place status transitions to
        `matched`. `matched_via` ("option_click" | "text_match") is
        what lets aggregate queries below report click-driven and
        text-match-driven confirmations as two separate numbers
        instead of one blended one (see Branch.matched_via)."""
        async with self._pool.acquire() as conn:
            updated = await conn.fetchval(
                "UPDATE branches SET status = 'matched', matched_via = $2 "
                "WHERE id = $1 RETURNING id",
                branch_id,
                matched_via,
            )
        if updated is None:
            raise KeyError(f"branch {branch_id} not found")
        return await self._require(branch_id)

    async def set_evidence_satisfied(
        self, branch_id: UUID, satisfied: bool = True
    ) -> Branch:
        """Flips a branch's evidence_satisfied flag — the one event
        that unblocks should_expand_branch's fourth gate. Set by a
        direct option click (no LLM call, unambiguous) or by
        CheckEvidence judging a typed message to establish it; both
        paths call this same method so the branch ends up in an
        identical state either way."""
        async with self._pool.acquire() as conn:
            updated = await conn.fetchval(
                "UPDATE branches SET evidence_satisfied = $2 WHERE id = $1 RETURNING id",
                branch_id,
                satisfied,
            )
        if updated is None:
            raise KeyError(f"branch {branch_id} not found")
        return await self._require(branch_id)

    async def list_awaiting_evidence(self, generation_id: UUID) -> list[Branch]:
        """Open branches in one generation whose requires_evidence is
        still unsatisfied — the candidate pool for both GenerateOptions
        and CheckEvidence."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM branches
                WHERE generation_id = $1 AND status = 'open'
                    AND requires_evidence IS NOT NULL AND evidence_satisfied = FALSE
                ORDER BY plausibility DESC
                """,
                generation_id,
            )
        return [self._row_to_branch(row) for row in rows]

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
                SELECT * FROM branch_generations
                WHERE session_id = $1
                ORDER BY turn_index DESC, created_at DESC
                LIMIT 1
                """,
                session_id,
            )
        return BranchGenerationMeta(**dict(row)) if row is not None else None

    async def set_selection(
        self, generation_id: UUID, selected_branch_id: UUID | None, rationale: str
    ) -> BranchGenerationMeta:
        """SelectBranch's result, written onto the generation it picked
        from. `selected_branch_id` may itself be None (nothing to
        select from) — the rationale still records why."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE branch_generations
                SET selected_branch_id = $2, selection_rationale = $3
                WHERE id = $1
                RETURNING *
                """,
                generation_id,
                selected_branch_id,
                rationale,
            )
        if row is None:
            raise KeyError(f"branch_generation {generation_id} not found")
        return BranchGenerationMeta(**dict(row))

    async def set_path_requirement(
        self, generation_id: UUID, path_requirement: PathRequirement
    ) -> BranchGenerationMeta:
        """DerivePath's result, written onto the same generation."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE branch_generations
                SET path_requirement = $2
                WHERE id = $1
                RETURNING *
                """,
                generation_id,
                path_requirement.model_dump(mode="json"),
            )
        if row is None:
            raise KeyError(f"branch_generation {generation_id} not found")
        return BranchGenerationMeta(**dict(row))

    async def match_rate_by_session_for_learner(
        self, learner_id: UUID
    ) -> list[BranchMatchRatePoint]:
        """Leaf-branch match rate per session, chronological — "does it
        actually predict me" over time. Only status in
        (matched, unmatched) counts as resolved; a session's most
        recent, still-open generation is excluded until it resolves.

        text_match and option_click are scored separately: a click
        confirms the student chose an offered option, not that the
        system predicted them, so click-resolved matches are excluded
        from total_resolved/matched_count entirely and reported only
        via the standalone option_click_count — never blended into
        the match-rate ratio (see BranchMatchRatePoint's docstring).
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    s.id AS session_id,
                    s.created_at AS session_created_at,
                    count(*) FILTER (
                        WHERE b.status IN ('matched', 'unmatched')
                            AND (b.matched_via IS NULL OR b.matched_via = 'text_match')
                    ) AS total_resolved,
                    count(*) FILTER (
                        WHERE b.status = 'matched'
                            AND (b.matched_via IS NULL OR b.matched_via = 'text_match')
                    ) AS matched_count,
                    count(*) FILTER (
                        WHERE b.status = 'matched' AND b.matched_via = 'option_click'
                    ) AS option_click_count
                FROM branches b
                JOIN sessions s ON s.id = b.session_id
                WHERE s.learner_id = $1 AND b.is_leaf = TRUE
                GROUP BY s.id, s.created_at
                ORDER BY s.created_at
                """,
                learner_id,
            )
        return [
            BranchMatchRatePoint(
                session_id=row["session_id"],
                session_created_at=row["session_created_at"],
                total_resolved=row["total_resolved"],
                matched_count=row["matched_count"],
                option_click_count=row["option_click_count"],
            )
            for row in rows
        ]

    async def recurring_root_statements_for_learner(
        self, learner_id: UUID, limit: int = 10
    ) -> list[RecurringIntent]:
        """Depth-0 (intent) statements grouped by exact text across all
        of a learner's sessions, most-recurring first — which bets
        about this learner keep coming up, and how often they're
        actually confirmed. `matched` (and `matched_via`) propagate up
        from a leaf to its full ancestor chain (see resolve()), so a
        root's own status/matched_via already reflect whether and how
        any descendant of it matched.

        `matched_count` counts text_match confirmations only;
        `matched_via_click_count` is the same click-vs-text split as
        match_rate_by_session_for_learner, kept as its own number —
        never combine the two."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    b.statement,
                    count(*) AS total_count,
                    count(*) FILTER (
                        WHERE b.status = 'matched'
                            AND (b.matched_via IS NULL OR b.matched_via = 'text_match')
                    ) AS matched_count,
                    count(*) FILTER (
                        WHERE b.status = 'matched' AND b.matched_via = 'option_click'
                    ) AS matched_via_click_count
                FROM branches b
                JOIN sessions s ON s.id = b.session_id
                WHERE s.learner_id = $1 AND b.parent_id IS NULL
                GROUP BY b.statement
                ORDER BY total_count DESC, matched_count DESC
                LIMIT $2
                """,
                learner_id,
                limit,
            )
        return [
            RecurringIntent(
                statement=row["statement"],
                total_count=row["total_count"],
                matched_count=row["matched_count"],
                matched_via_click_count=row["matched_via_click_count"],
            )
            for row in rows
        ]

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
            requires_evidence=row["requires_evidence"],
            evidence_satisfied=row["evidence_satisfied"],
            plausibility=row["plausibility"],
            is_leaf=row["is_leaf"],
            status=BranchStatus(row["status"]),
            matched_via=row["matched_via"],
            created_at=row["created_at"],
        )
