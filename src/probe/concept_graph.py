"""ConceptGraph: the world concept graph, backed by Postgres.

Treated as append-only, same spirit as HypothesisStore (CLAUDE.md
invariant 1) though for a different reason: this graph is meant to be
frozen after seeding (`probe seed-graph`), and LearnerOverlay (and any
future consumer) references concept ids directly. There is no delete
method here and no DELETE SQL.

`add_concept` inserts one concept whose prerequisites must already exist
(enforced by the `concept_prerequisites` FK — asyncpg raises
ForeignKeyViolationError on a dangling reference). `add_batch` is for the
seeding path: an LLM proposes several ConceptNodes together that may
reference each other's ids as prerequisites (forward references within
the batch are expected), so it validates prerequisite existence and
acyclicity itself, in Python, before writing anything, and inserts the
whole batch in one transaction.
"""

from __future__ import annotations

import asyncpg

from probe.models import ConceptNode


class ConceptValidationError(Exception):
    """A proposed concept batch failed validation before any insert."""


class ConceptCycleError(ConceptValidationError):
    """A prerequisite graph (proposed batch, or data already on disk)
    contains a cycle."""


def _raise_if_cycle(graph: dict[str, list[str]]) -> None:
    """Raise ConceptCycleError if `graph` (concept_id -> prerequisite
    ids) contains a cycle. Standard three-color DFS so a self-referential
    or mutually-referential batch fails fast instead of recursing forever.
    """
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(node_id: str) -> None:
        visiting.add(node_id)
        for neighbor in graph.get(node_id, []):
            if neighbor in visiting:
                raise ConceptCycleError(
                    f"cycle detected in concept prerequisites: "
                    f"{neighbor!r} depends (directly or transitively) on "
                    f"itself via {node_id!r}"
                )
            if neighbor not in done:
                visit(neighbor)
        visiting.discard(node_id)
        done.add(node_id)

    for node_id in graph:
        if node_id not in done:
            visit(node_id)


class ConceptGraph:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def add_concept(self, concept: ConceptNode) -> ConceptNode:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await self._insert_concept(conn, concept)
                for prereq_id in concept.prerequisites:
                    await self._insert_edge(conn, concept.id, prereq_id)
        return await self._require(concept.id)

    async def add_batch(self, concepts: list[ConceptNode]) -> list[ConceptNode]:
        """Insert a batch of concepts atomically.

        Validates, before touching the database, that every prerequisite
        id referenced by a concept in the batch is itself present in the
        batch, and that the batch's prerequisite edges contain no cycle.
        On either failure nothing is inserted.
        """
        ids = {c.id for c in concepts}
        graph: dict[str, list[str]] = {c.id: c.prerequisites for c in concepts}

        for concept in concepts:
            for prereq_id in concept.prerequisites:
                if prereq_id not in ids:
                    raise ConceptValidationError(
                        f"concept {concept.id!r} lists prerequisite "
                        f"{prereq_id!r}, which is not present in this batch"
                    )
        _raise_if_cycle(graph)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for concept in concepts:
                    await self._insert_concept(conn, concept)
                for concept in concepts:
                    for prereq_id in concept.prerequisites:
                        await self._insert_edge(conn, concept.id, prereq_id)
        return [await self._require(c.id) for c in concepts]

    async def get_concept(self, id: str) -> ConceptNode | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM concept_nodes WHERE id = $1", id
            )
            if row is None:
                return None
            prereq_rows = await conn.fetch(
                """
                SELECT prerequisite_id FROM concept_prerequisites
                WHERE concept_id = $1
                ORDER BY prerequisite_id
                """,
                id,
            )
        return self._row_to_concept(
            row, [r["prerequisite_id"] for r in prereq_rows]
        )

    async def prerequisites_of(self, id: str) -> list[str]:
        """Direct prerequisites only, alphabetical by id."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT prerequisite_id FROM concept_prerequisites
                WHERE concept_id = $1
                ORDER BY prerequisite_id
                """,
                id,
            )
        return [r["prerequisite_id"] for r in rows]

    async def all_prerequisites_of(self, id: str) -> list[str]:
        """Transitive closure of prerequisites, in DFS discovery order.

        Raises ConceptCycleError rather than looping forever if the
        stored edge set contains a cycle. The public write paths
        (add_concept, add_batch) can't produce one on their own — this
        is a defensive check against data that got there some other way.
        """
        graph = await self._edge_map()
        _raise_if_cycle(graph)

        result: list[str] = []
        seen: set[str] = set()

        def visit(node_id: str) -> None:
            for prereq_id in graph.get(node_id, []):
                if prereq_id not in seen:
                    seen.add(prereq_id)
                    result.append(prereq_id)
                    visit(prereq_id)

        visit(id)
        return result

    async def _edge_map(self) -> dict[str, list[str]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT concept_id, prerequisite_id FROM concept_prerequisites"
            )
        graph: dict[str, list[str]] = {}
        for row in rows:
            graph.setdefault(row["concept_id"], []).append(row["prerequisite_id"])
        return graph

    async def _insert_concept(
        self, conn: asyncpg.Connection, concept: ConceptNode
    ) -> None:
        await conn.execute(
            """
            INSERT INTO concept_nodes (
                id, name, common_misconceptions, representations,
                diagnostic_questions, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            concept.id,
            concept.name,
            concept.common_misconceptions,
            concept.representations,
            concept.diagnostic_questions,
            concept.created_at,
            concept.updated_at,
        )

    async def _insert_edge(
        self, conn: asyncpg.Connection, concept_id: str, prerequisite_id: str
    ) -> None:
        await conn.execute(
            """
            INSERT INTO concept_prerequisites (concept_id, prerequisite_id)
            VALUES ($1, $2)
            """,
            concept_id,
            prerequisite_id,
        )

    async def _require(self, id: str) -> ConceptNode:
        concept = await self.get_concept(id)
        if concept is None:
            raise KeyError(f"concept {id!r} not found")
        return concept

    def _row_to_concept(self, row, prereq_ids: list[str]) -> ConceptNode:
        return ConceptNode(
            id=row["id"],
            name=row["name"],
            prerequisites=prereq_ids,
            common_misconceptions=list(row["common_misconceptions"]),
            representations=list(row["representations"]),
            diagnostic_questions=list(row["diagnostic_questions"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
