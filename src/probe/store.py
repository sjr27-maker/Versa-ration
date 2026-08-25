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

    async def list_all(self, tier: Tier | None = None) -> list[Hypothesis]:
        async with self._pool.acquire() as conn:
            if tier is None:
                rows = await conn.fetch(
                    "SELECT * FROM hypotheses ORDER BY created_at, id"
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM hypotheses
                    WHERE tier = $1
                    ORDER BY created_at, id
                    """,
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

    async def link_concept(
        self, hypothesis_id: UUID, concept_graph_id: UUID, concept_id: str
    ) -> None:
        """Assert that a hypothesis pertains to a concept.

        Idempotent — linking the same (hypothesis_id, concept_graph_id,
        concept_id) triple twice is a no-op, not a duplicate row. This
        is the matching convention `MismatchDetector`/`Diagnose` rely on
        to find "the mental_model hypothesis for this concept": a
        separate join table rather than a field on `Hypothesis`, since a
        hypothesis can pertain to zero, one, or several concepts.
        concept_id is only unique within its concept_graph_id.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO hypothesis_concepts (
                    hypothesis_id, concept_graph_id, concept_id
                ) VALUES ($1, $2, $3)
                ON CONFLICT DO NOTHING
                """,
                hypothesis_id,
                concept_graph_id,
                concept_id,
            )

    async def list_by_concept(
        self,
        concept_graph_id: UUID,
        concept_id: str,
        layer: Layer | None = None,
        tier: Tier | None = None,
    ) -> list[Hypothesis]:
        conditions = ["hc.concept_graph_id = $1", "hc.concept_id = $2"]
        params: list = [concept_graph_id, concept_id]
        if layer is not None:
            params.append(layer.value)
            conditions.append(f"h.layer = ${len(params)}")
        if tier is not None:
            params.append(tier.value)
            conditions.append(f"h.tier = ${len(params)}")
        where = " AND ".join(conditions)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT h.* FROM hypotheses h
                JOIN hypothesis_concepts hc ON hc.hypothesis_id = h.id
                WHERE {where}
                ORDER BY h.created_at, h.id
                """,
                *params,
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

    async def list_by_learner(
        self,
        learner_id: UUID,
        layer: Layer | None = None,
        tier: Tier | None = None,
    ) -> list[Hypothesis]:
        """Hypotheses with at least one evidence_ref traceable to this
        learner's sessions (evidence_ref -> turn -> session -> learner_id).

        Hypothesis has no direct learner_id column — same reasoning as
        `hypothesis_concepts` being a join table rather than a field on
        Hypothesis: this avoids a migration on the already-shipped
        model. This is the best signal available today, and it has a
        real limitation: a hypothesis with no evidence yet (freshly
        added, never reweighted) has no learner association and won't
        appear here.
        """
        conditions = ["s.learner_id = $1"]
        params: list = [learner_id]
        if layer is not None:
            params.append(layer.value)
            conditions.append(f"h.layer = ${len(params)}")
        if tier is not None:
            params.append(tier.value)
            conditions.append(f"h.tier = ${len(params)}")
        where = " AND ".join(conditions)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT DISTINCT h.* FROM hypotheses h
                JOIN evidence_refs er ON er.hypothesis_id = h.id
                JOIN turns t ON t.id = er.turn_id
                JOIN sessions s ON s.id = t.session_id
                WHERE {where}
                ORDER BY h.created_at, h.id
                """,
                *params,
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
