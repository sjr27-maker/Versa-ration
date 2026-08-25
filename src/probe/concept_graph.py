"""ConceptGraph: the world concept graph, backed by Postgres.

A deployment can seed multiple topics, so nodes are grouped into
`concept_graphs` (see `ConceptGraphMeta`): a concept's `id` (a short
slug) is only unique *within* its graph, not globally — every method
here that addresses a node takes `concept_graph_id` alongside `id`.

Treated as append-only, same spirit as HypothesisStore (CLAUDE.md
invariant 1) though for a different reason: a graph is meant to be
frozen after seeding (`probe seed-graph`), and LearnerOverlay (and
other consumers) reference concept ids directly. There is no delete
method here and no DELETE SQL.

`add_concept` inserts one concept into an *existing* graph, whose
prerequisites must already exist in that same graph (enforced by the
`concept_prerequisites` FK — asyncpg raises ForeignKeyViolationError on
a dangling reference). `add_batch` is for the seeding path: it creates
the graph row and inserts an LLM-proposed batch of ConceptNodes that
may reference each other's ids as prerequisites (forward references
within the batch are expected), validating prerequisite existence and
acyclicity itself, in Python, before writing anything — graph row and
nodes land in one transaction, or none of it does.
"""

from __future__ import annotations

from uuid import UUID

import asyncpg

from probe.models import ConceptGraphMeta, ConceptNode


class ConceptValidationError(Exception):
    """A proposed concept batch failed validation before any insert."""


class ConceptCycleError(ConceptValidationError):
    """A prerequisite graph (proposed batch, or data already on disk)
    contains a cycle."""


def _raise_if_cycle(graph: dict[str, list[str]]) -> None:
    """Raise ConceptCycleError if `graph` (concept_id -> prerequisite
    ids, scoped to one concept_graph_id) contains a cycle. Standard
    three-color DFS so a self-referential or mutually-referential batch
    fails fast instead of recursing forever.
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

    async def create_graph(self, topic: str) -> ConceptGraphMeta:
        """Create an empty graph. Most callers want `add_batch` instead
        (atomic graph-creation + validated node batch, for the
        seed-graph path); this exists for callers that build a graph
        incrementally via `add_concept` (mainly test fixtures)."""
        meta = ConceptGraphMeta(topic=topic)
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO concept_graphs (id, topic, created_at) "
                "VALUES ($1, $2, $3)",
                meta.id,
                meta.topic,
                meta.created_at,
            )
        return meta

    async def get_graph(self, id: UUID) -> ConceptGraphMeta | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM concept_graphs WHERE id = $1", id
            )
        return None if row is None else self._row_to_graph_meta(row)

    async def find_graphs_by_topic(self, topic: str) -> list[ConceptGraphMeta]:
        """Topic is not unique — a topic may have been seeded more than
        once, producing independent graphs. Callers decide what to do
        with more than one match (see `probe chat --topic`)."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM concept_graphs WHERE topic = $1 ORDER BY created_at",
                topic,
            )
        return [self._row_to_graph_meta(row) for row in rows]

    async def add_concept(self, concept: ConceptNode) -> ConceptNode:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await self._insert_concept(conn, concept)
                for prereq_id in concept.prerequisites:
                    await self._insert_edge(
                        conn, concept.concept_graph_id, concept.id, prereq_id
                    )
        return await self._require(concept.concept_graph_id, concept.id)

    async def add_batch(
        self, concept_graph_id: UUID, topic: str, concepts: list[ConceptNode]
    ) -> list[ConceptNode]:
        """Create a new graph for `topic` and insert `concepts` into it
        atomically.

        Validates, before touching the database, that every concept
        carries this same `concept_graph_id`, that every prerequisite id
        referenced by a concept in the batch is itself present in the
        batch, and that the batch's prerequisite edges contain no cycle.
        On any failure nothing is inserted — no orphan `concept_graphs`
        row with zero nodes.
        """
        for concept in concepts:
            if concept.concept_graph_id != concept_graph_id:
                raise ConceptValidationError(
                    f"concept {concept.id!r} has concept_graph_id "
                    f"{concept.concept_graph_id}, expected {concept_graph_id}"
                )

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
                await conn.execute(
                    "INSERT INTO concept_graphs (id, topic) VALUES ($1, $2)",
                    concept_graph_id,
                    topic,
                )
                for concept in concepts:
                    await self._insert_concept(conn, concept)
                for concept in concepts:
                    for prereq_id in concept.prerequisites:
                        await self._insert_edge(
                            conn, concept_graph_id, concept.id, prereq_id
                        )
        return [await self._require(concept_graph_id, c.id) for c in concepts]

    async def get_concept(
        self, concept_graph_id: UUID, id: str
    ) -> ConceptNode | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM concept_nodes WHERE concept_graph_id = $1 AND id = $2",
                concept_graph_id,
                id,
            )
            if row is None:
                return None
            prereq_rows = await conn.fetch(
                """
                SELECT prerequisite_id FROM concept_prerequisites
                WHERE concept_graph_id = $1 AND concept_id = $2
                ORDER BY prerequisite_id
                """,
                concept_graph_id,
                id,
            )
        return self._row_to_concept(
            row, [r["prerequisite_id"] for r in prereq_rows]
        )

    async def list_concepts(self, concept_graph_id: UUID) -> list[ConceptNode]:
        """Every node in one graph, for callers that need the whole
        graph rather than one node (e.g. GroundConcept's candidate
        list)."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM concept_nodes WHERE concept_graph_id = $1 "
                "ORDER BY id",
                concept_graph_id,
            )
            prereq_rows = await conn.fetch(
                """
                SELECT concept_id, prerequisite_id FROM concept_prerequisites
                WHERE concept_graph_id = $1
                ORDER BY concept_id, prerequisite_id
                """,
                concept_graph_id,
            )
        prereqs_by_concept: dict[str, list[str]] = {}
        for row in prereq_rows:
            prereqs_by_concept.setdefault(row["concept_id"], []).append(
                row["prerequisite_id"]
            )
        return [
            self._row_to_concept(row, prereqs_by_concept.get(row["id"], []))
            for row in rows
        ]

    async def prerequisites_of(self, concept_graph_id: UUID, id: str) -> list[str]:
        """Direct prerequisites only, alphabetical by id."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT prerequisite_id FROM concept_prerequisites
                WHERE concept_graph_id = $1 AND concept_id = $2
                ORDER BY prerequisite_id
                """,
                concept_graph_id,
                id,
            )
        return [r["prerequisite_id"] for r in rows]

    async def all_prerequisites_of(
        self, concept_graph_id: UUID, id: str
    ) -> list[str]:
        """Transitive closure of prerequisites, in DFS discovery order.

        Raises ConceptCycleError rather than looping forever if the
        stored edge set contains a cycle. The public write paths
        (add_concept, add_batch) can't produce one on their own — this
        is a defensive check against data that got there some other way.
        """
        graph = await self._edge_map(concept_graph_id)
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

    async def _edge_map(self, concept_graph_id: UUID) -> dict[str, list[str]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT concept_id, prerequisite_id FROM concept_prerequisites "
                "WHERE concept_graph_id = $1",
                concept_graph_id,
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
                concept_graph_id, id, name, common_misconceptions,
                representations, diagnostic_questions, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            concept.concept_graph_id,
            concept.id,
            concept.name,
            concept.common_misconceptions,
            concept.representations,
            concept.diagnostic_questions,
            concept.created_at,
            concept.updated_at,
        )

    async def _insert_edge(
        self,
        conn: asyncpg.Connection,
        concept_graph_id: UUID,
        concept_id: str,
        prerequisite_id: str,
    ) -> None:
        await conn.execute(
            """
            INSERT INTO concept_prerequisites (
                concept_graph_id, concept_id, prerequisite_id
            ) VALUES ($1, $2, $3)
            """,
            concept_graph_id,
            concept_id,
            prerequisite_id,
        )

    async def _require(self, concept_graph_id: UUID, id: str) -> ConceptNode:
        concept = await self.get_concept(concept_graph_id, id)
        if concept is None:
            raise KeyError(f"concept {id!r} not found in graph {concept_graph_id}")
        return concept

    def _row_to_concept(self, row, prereq_ids: list[str]) -> ConceptNode:
        return ConceptNode(
            concept_graph_id=row["concept_graph_id"],
            id=row["id"],
            name=row["name"],
            prerequisites=prereq_ids,
            common_misconceptions=list(row["common_misconceptions"]),
            representations=list(row["representations"]),
            diagnostic_questions=list(row["diagnostic_questions"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_graph_meta(self, row) -> ConceptGraphMeta:
        return ConceptGraphMeta(
            id=row["id"], topic=row["topic"], created_at=row["created_at"]
        )
