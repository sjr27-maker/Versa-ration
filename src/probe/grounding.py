"""GroundConcept: which concept, if any, does a student response
actually discuss.

A small collaborator invoked from inside Diagnose.run() — like
MismatchDetector, it is not itself a Node and doesn't get its own
node_calls row; its output rides along inside Diagnose's, which is
audited (CLAUDE.md invariant 2).

Explicitly out of scope: choosing what to teach next ("concept
selection"). This only identifies what the current response is about.
"""

from __future__ import annotations

import json

from probe.llm import LLMClient
from probe.models import ConceptGrounding, ConceptNode


def _grounding_prompt(response: str, candidates: list[ConceptNode]) -> str:
    listing = (
        "\n".join(f"- {c.id}: {c.name}" for c in candidates)
        or "(no concepts in this graph)"
    )
    return (
        "GROUND:CONCEPT\n"
        f"student_response={response}\n"
        "Which of the following concepts, if any, does this response "
        "actually discuss? Respond with the concept's id exactly as "
        "listed below, or null if the response is off-topic, purely "
        'affective (e.g. "I don\'t get this"), or doesn\'t clearly '
        "concern any of them.\n"
        f"concepts:\n{listing}\n"
        'Respond with JSON: {"concept_id": "..." or null, '
        '"confidence": 0.0-1.0}'
    )


class GroundConcept:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        # Updated by detect(); read by Diagnose into its
        # llm_call_count, same pattern as ValueFunction's
        # information_value_call_count. Public (not _-prefixed) since,
        # unlike that one, the reader is a different class. Always 1
        # today (detect() never short-circuits before calling the LLM),
        # but tracked rather than assumed so a future retry-on-malformed
        # -response addition (same shape as Plan's candidate proposer)
        # doesn't silently make this count wrong.
        self.last_call_count: int = 0

    async def detect(
        self, response: str, candidates: list[ConceptNode]
    ) -> ConceptGrounding:
        self.last_call_count = 0
        raw = await self._llm.complete(_grounding_prompt(response, candidates))
        self.last_call_count += 1
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return ConceptGrounding(concept_id=None, confidence=0.0)
        if not isinstance(parsed, dict):
            return ConceptGrounding(concept_id=None, confidence=0.0)

        raw_concept_id = parsed.get("concept_id")
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        if raw_concept_id is None:
            return ConceptGrounding(concept_id=None, confidence=confidence)

        valid_ids = {c.id for c in candidates}
        if raw_concept_id not in valid_ids:
            # Hallucinated id: same validation discipline as
            # seed_graph's prerequisite check and Plan's candidate
            # proposer — reject it rather than pass it through, but
            # keep the reported confidence visible (it's informative
            # even when the id itself doesn't survive validation).
            return ConceptGrounding(concept_id=None, confidence=confidence)

        return ConceptGrounding(concept_id=raw_concept_id, confidence=confidence)
