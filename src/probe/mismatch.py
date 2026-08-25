"""MismatchDetector: compares a learner's apparent belief against the
world model (a ConceptNode's definition and listed misconceptions).

Deliberately does not own any stores — like ValueFunction, it's a
stateless collaborator that receives already-fetched data (concept_id,
the ConceptNode, an optional OverlayEntry, and the mental_model
Hypothesis rows already matched to this concept via
HypothesisStore.list_by_concept) and returns a judgment. Diagnose is the
Node that owns the stores and acts on that judgment.

The `suggested_cause` call is a real LLM judgment, not a hardcoded
default: a detected conflict is not automatically the learner's fault.
The prompt explicitly tells the model not to default to blaming the
learner, since the concept definition or its misconception list can
itself be wrong or incomplete.
"""

from __future__ import annotations

import json

from probe.llm import LLMClient
from probe.models import ConceptNode, Hypothesis, MismatchResult, OverlayEntry


def _mismatch_prompt(
    concept_id: str,
    concept: ConceptNode,
    overlay_entry: OverlayEntry | None,
    hypotheses: list[Hypothesis],
) -> str:
    overlay_desc = (
        f"state={overlay_entry.state.value}, confidence={overlay_entry.confidence:.2f}"
        if overlay_entry is not None
        else "no overlay entry on file"
    )
    hyp_listing = (
        "\n".join(
            f"- {h.id} (p={h.probability:.2f}, c={h.confidence:.2f}): {h.statement}"
            for h in hypotheses
        )
        or "(none)"
    )
    misconceptions = (
        "\n".join(f"- {m}" for m in concept.common_misconceptions) or "(none listed)"
    )
    return (
        "MISMATCH:DETECT\n"
        f"concept_id={concept_id}\n"
        f"concept_name={concept.name}\n"
        f"known_misconceptions:\n{misconceptions}\n"
        f"learner_overlay: {overlay_desc}\n"
        f"mental_model hypotheses for this concept:\n{hyp_listing}\n"
        "Decide whether the learner's apparent belief (from the overlay "
        "and/or hypotheses above) conflicts with the concept's definition "
        'or one of its listed misconceptions. If there is no real '
        'conflict, respond with exactly {"mismatch": false}.\n'
        "If there is a conflict, you must also judge whether the most "
        "likely explanation is that the LEARNER holds a misconception, "
        "or that OUR OWN concept definition or misconception list might "
        "itself be wrong or incomplete. The explanation may be wrong — "
        "do not default to blaming the learner. Respond with JSON: "
        '{"mismatch": true, "learner_claim": "...", "world_claim": "...", '
        '"confidence": 0.0-1.0, "suggested_cause": '
        '"learner_misconception" | "possible_world_model_error"}'
    )


class MismatchDetector:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        # Updated by detect(); read by Diagnose into its
        # llm_call_count, same pattern as ValueFunction's
        # information_value_call_count. Public (not _-prefixed) since,
        # unlike that one, the reader is a different class.
        self.last_call_count: int = 0

    async def detect(
        self,
        concept_id: str,
        concept: ConceptNode,
        overlay_entry: OverlayEntry | None,
        hypotheses: list[Hypothesis],
    ) -> MismatchResult | None:
        self.last_call_count = 0
        if overlay_entry is None and not hypotheses:
            # Nothing to compare the world model against.
            return None

        raw = await self._llm.complete(
            _mismatch_prompt(concept_id, concept, overlay_entry, hypotheses)
        )
        self.last_call_count += 1
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict) or not parsed.get("mismatch", False):
            return None
        try:
            return MismatchResult(
                concept_id=concept_id,
                learner_claim=str(parsed.get("learner_claim", "")),
                world_claim=str(parsed.get("world_claim", "")),
                confidence=float(parsed.get("confidence", 0.0)),
                suggested_cause=parsed.get("suggested_cause"),
            )
        except (ValueError, TypeError):
            return None
