"""One-time LLM seeding of a ConceptGraph for a single topic.

This is explicitly not an ingestion pipeline: it runs once per topic,
parses and validates the LLM's proposed batch, and hands it to
`ConceptGraph.add_batch` — which creates the `concept_graphs` row and
inserts the nodes atomically (prerequisite-existence and cycle
validation included). There is no update path here; the graph is meant
to be frozen after this call.
"""

from __future__ import annotations

import json
from uuid import UUID, uuid4

from probe.concept_graph import ConceptGraph
from probe.llm import LLMClient
from probe.models import ConceptNode

MIN_CONCEPTS = 8
MAX_CONCEPTS = 15


class SeedGraphError(Exception):
    """The LLM's proposed batch could not be parsed into ConceptNodes."""


def _seed_prompt(topic: str) -> str:
    return (
        "SEED:CONCEPT_GRAPH\n"
        f"topic={topic}\n"
        f"Propose between {MIN_CONCEPTS} and {MAX_CONCEPTS} ConceptNodes "
        f'that decompose "{topic}" into a teachable prerequisite graph.\n'
        "Each concept id must be a short unique lowercase slug (letters, "
        "digits, underscores only). `prerequisites` must reference only "
        "ids used elsewhere in this same list — do not invent ids that "
        "aren't defined here, and do not create a dependency cycle.\n"
        'Respond with JSON: [{"id": "...", "name": "...", '
        '"prerequisites": ["..."], "common_misconceptions": ["..."], '
        '"representations": ["..."], "diagnostic_questions": ["..."]}, ...]'
    )


async def seed_graph(
    llm: LLMClient, graph: ConceptGraph, topic: str
) -> tuple[UUID, list[ConceptNode]]:
    """Returns (concept_graph_id, concepts)."""
    raw = await llm.complete(_seed_prompt(topic))
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SeedGraphError(
            f"seed-graph response was not valid JSON: {exc}"
        ) from exc
    if not isinstance(parsed, list) or not parsed:
        raise SeedGraphError(
            "seed-graph response was not a non-empty JSON list"
        )

    concept_graph_id = uuid4()
    concepts: list[ConceptNode] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise SeedGraphError(
                f"seed-graph proposed a non-object entry: {item!r}"
            )
        try:
            concepts.append(
                ConceptNode.model_validate(
                    {**item, "concept_graph_id": concept_graph_id}
                )
            )
        except Exception as exc:
            raise SeedGraphError(
                f"seed-graph proposed an invalid concept: {exc}"
            ) from exc

    # add_batch validates prerequisite-existence-within-batch and
    # acyclicity, creates the concept_graphs row, and inserts everything
    # in one transaction — a validation failure there means nothing was
    # written, not even the graph row.
    persisted = await graph.add_batch(concept_graph_id, topic, concepts)
    return concept_graph_id, persisted
