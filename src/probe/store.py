from __future__ import annotations

from uuid import UUID

import asyncpg

from probe.models import EvidenceRef, Hypothesis, Layer, Polarity, Tier


class HypothesisStore:
    """Append-only store for hypotheses and their evidence.

    See CLAUDE.md: this class has no delete method and its SQL contains
    no DELETE. Retirement is expressed as a tier transition, and
    reweighting appends evidence rather than mutating it.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def add(self, hypothesis: Hypothesis) -> Hypothesis:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO hypotheses (
                        id, layer, statement, probability, confidence,
                        tier, conditions, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    hypothesis.id,
                    hypothesis.layer.value,
                    hypothesis.statement,
                    hypothesis.probability,
                    hypothesis.confidence,
                    hypothesis.tier.value,
                    hypothesis.conditions,
                    hypothesis.created_at,
                    hypothesis.updated_at,
                )
                for ref in hypothesis.evidence_refs:
                    await self._insert_evidence(conn, hypothesis.id, ref)
                for ref in hypothesis.counter_evidence_refs:
                    await self._insert_evidence(conn, hypothesis.id, ref)
        return await self._require(hypothesis.id)

    async def get(self, id: UUID) -> Hypothesis | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM hypotheses WHERE id = $1", id
            )
            if row is None:
                return None
            ev_rows = await conn.fetch(
                """
                SELECT * FROM evidence_refs
                WHERE hypothesis_id = $1
                ORDER BY timestamp, id
                """,
                id,
            )
        return self._row_to_hypothesis(row, ev_rows)

    async def list_by_layer(
        self, layer: Layer, tier: Tier | None = None
    ) -> list[Hypothesis]:
        async with self._pool.acquire() as conn:
            if tier is None:
                rows = await conn.fetch(
                    """
                    SELECT * FROM hypotheses
                    WHERE layer = $1
                    ORDER BY created_at, id
                    """,
                    layer.value,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM hypotheses
                    WHERE layer = $1 AND tier = $2
                    ORDER BY created_at, id
                    """,
                    layer.value,
                    tier.value,
                )
            ids = [r["id"] for r in rows]
            ev_by_hyp: dict[UUID, list] = {hid: [] for hid in ids}
            if ids:
                ev_rows = await conn.fetch(
                    """
                    SELECT * FROM evidence_refs
                    WHERE hypothesis_id = ANY($1::uuid[])
                    ORDER BY timestamp, id
                    """,
                    ids,
                )
                for ev in ev_rows:
                    ev_by_hyp[ev["hypothesis_id"]].append(ev)
        return [self._row_to_hypothesis(row, ev_by_hyp[row["id"]]) for row in rows]

    async def reweight(
        self,
        id: UUID,
        new_probability: float,
        new_confidence: float,
        evidence_ref: EvidenceRef,
    ) -> Hypothesis:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                updated = await conn.fetchval(
                    """
                    UPDATE hypotheses
                    SET probability = $2,
                        confidence  = $3,
                        updated_at  = NOW()
                    WHERE id = $1
                    RETURNING id
                    """,
                    id,
                    new_probability,
                    new_confidence,
                )
                if updated is None:
                    raise KeyError(f"hypothesis {id} not found")
                await self._insert_evidence(conn, id, evidence_ref)
        return await self._require(id)

    async def retier(self, id: UUID, new_tier: Tier) -> Hypothesis:
        async with self._pool.acquire() as conn:
            updated = await conn.fetchval(
                """
                UPDATE hypotheses
                SET tier = $2, updated_at = NOW()
                WHERE id = $1
                RETURNING id
                """,
                id,
                new_tier.value,
            )
            if updated is None:
                raise KeyError(f"hypothesis {id} not found")
        return await self._require(id)

    async def resurrect(self, id: UUID) -> Hypothesis:
        return await self.retier(id, Tier.ACTIVE)

    async def _insert_evidence(
        self, conn: asyncpg.Connection, hypothesis_id: UUID, ref: EvidenceRef
    ) -> None:
        await conn.execute(
            """
            INSERT INTO evidence_refs (
                id, hypothesis_id, turn_id, polarity, timestamp
            ) VALUES ($1, $2, $3, $4, $5)
            """,
            ref.id,
            hypothesis_id,
            ref.turn_id,
            ref.polarity.value,
            ref.timestamp,
        )

    async def _require(self, id: UUID) -> Hypothesis:
        hyp = await self.get(id)
        if hyp is None:
            raise KeyError(f"hypothesis {id} not found")
        return hyp

    def _row_to_hypothesis(self, row, ev_rows) -> Hypothesis:
        supporting: list[EvidenceRef] = []
        contradicting: list[EvidenceRef] = []
        for ev in ev_rows:
            ref = EvidenceRef(
                id=ev["id"],
                turn_id=ev["turn_id"],
                polarity=Polarity(ev["polarity"]),
                timestamp=ev["timestamp"],
            )
            if ref.polarity is Polarity.SUPPORTING:
                supporting.append(ref)
            else:
                contradicting.append(ref)
        return Hypothesis(
            id=row["id"],
            layer=Layer(row["layer"]),
            statement=row["statement"],
            probability=row["probability"],
            confidence=row["confidence"],
            tier=Tier(row["tier"]),
            conditions=list(row["conditions"]),
            evidence_refs=supporting,
            counter_evidence_refs=contradicting,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
