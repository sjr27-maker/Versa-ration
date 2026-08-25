"""Node classes for the probe reasoning loop.

Every node exposes a single async `run(...)` method whose signature is
the node's semantic contract. Persistence of inputs/outputs is *not*
each node's responsibility — `SessionLoop._call_node()` intercepts and
records every call. See CLAUDE.md invariant 2.

Most nodes are stubs at this stage; the reasoning logic lands in later
steps. `Update` and `Diagnose` are real because they only depend on
already-built stores/services.
"""

from __future__ import annotations

import json
import math
from typing import Any
from uuid import UUID

from probe.audit import TranscriptStore
from probe.concept_graph import ConceptGraph
from probe.grounding import GroundConcept
from probe.llm import LLMClient
from probe.mismatch import MismatchDetector
from probe.models import (
    ActionScore,
    CandidateAction,
    EvidenceRef,
    Hypothesis,
    Layer,
    PlanOutput,
    Polarity,
    ProposedEvidence,
    SuggestedCause,
    TeachingAction,
    Tier,
    WorldModelRevision,
)
from probe.overlay import LearnerOverlay
from probe.revision import WorldModelRevisionStore
from probe.store import HypothesisStore
from probe.value_function import ValueFunction

DEFAULT_GENERATION_WIDTH = 3
_MIN_WIDTH = 1
_MAX_WIDTH = 8

# One corrective re-ask when the proposer emits actions outside the enum.
_MAX_PROPOSE_ATTEMPTS = 2

# Diagnose: minimum MismatchResult.confidence required to act on
# suggested_cause=possible_world_model_error by proposing a
# WorldModelRevision. Intentionally low — proposals are supervised and
# reversible via review-revisions, so bias toward over-proposing while
# gathering real interaction data; revisit once session data exists to
# set this empirically.
DIAGNOSE_MISMATCH_THRESHOLD = 0.4


class Infer:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(
        self,
        turn_text: str,
        hypotheses: list[Hypothesis],
        generation_width: int = DEFAULT_GENERATION_WIDTH,
    ) -> list[ProposedEvidence]:
        # Real prompt lives in Step 2. For now we ship the plumbing: the
        # LLM is asked for a JSON list of {hypothesis_id, new_probability,
        # new_confidence, polarity} objects, and any parse failure yields
        # an empty proposal set.
        listing = "\n".join(
            f"- {h.id} [{h.layer.value}] {h.statement} "
            f"(p={h.probability:.2f}, c={h.confidence:.2f})"
            for h in hypotheses
        )
        prompt = (
            "INFER: given the following student turn, identify which of the "
            "listed hypotheses it supports or contradicts.\n\n"
            f"Turn: {turn_text}\n\n"
            f"Hypotheses:\n{listing}\n"
        )
        raw = await self._llm.complete(prompt)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        proposals: list[ProposedEvidence] = []
        for item in parsed:
            try:
                proposals.append(ProposedEvidence.model_validate(item))
            except Exception:
                continue
        return proposals


class Plan:
    """Generate N candidate actions, score each, return the winner + scores.

    Candidate generation is LLM-driven: the proposer sees the current
    hypotheses (all five layers) plus `concept_state` and returns the
    `generation_width` actions it judges plausible *for this turn*,
    rather than the value function paying to score all 21. Scoring and
    selection are independent of how candidates were proposed.

    The returned `PlanOutput` carries both the winner (for Teach) and
    the full per-candidate `ActionScore` breakdown. Both survive
    together through the SessionLoop audit choke point (CLAUDE.md
    invariant 2) so ablation runs can inspect every candidate's terms,
    not just the winner's.
    """

    def __init__(self, value_function: ValueFunction, llm: LLMClient) -> None:
        self._vf = value_function
        self._llm = llm

    async def run(
        self,
        hypotheses: list[Hypothesis],
        concept_state: dict,
        generation_width: int,
    ) -> PlanOutput:
        candidates = await self._generate(
            hypotheses, concept_state, generation_width
        )
        scores: list[ActionScore] = []
        for candidate in candidates:
            scores.append(await self._vf.score(candidate, hypotheses, concept_state))
        winner_score = max(scores, key=lambda s: s.total)
        return PlanOutput(winner=winner_score.candidate, scores=scores)

    async def _generate(
        self,
        hypotheses: list[Hypothesis],
        concept_state: dict,
        generation_width: int,
    ) -> list[CandidateAction]:
        """Ask the LLM for `generation_width` distinct plausible actions.

        Invalid actions are dropped, not fatal. If dropping left us short
        we retry once with the rejected values named, so the model can
        correct itself. Only after that do we backfill from enum order —
        the deterministic tie-break applies to the *filler*, never to the
        model's own proposals.
        """
        width = max(_MIN_WIDTH, generation_width)
        proposed: list[CandidateAction] = []
        seen: set[TeachingAction] = set()
        rejected: list[str] = []

        for _ in range(_MAX_PROPOSE_ATTEMPTS):
            raw = await self._llm.complete(
                _propose_prompt(hypotheses, concept_state, width, rejected)
            )
            candidates, rejected = _parse_proposals(raw, concept_state)
            for candidate in candidates:
                if candidate.action in seen:
                    continue
                seen.add(candidate.action)
                proposed.append(candidate)
                if len(proposed) >= width:
                    break
            # Retry only buys something when the model actually emitted
            # garbage. A short-but-clean proposal is a legitimate answer:
            # backfill it instead of badgering the model.
            if len(proposed) >= width or not rejected:
                break

        del proposed[width:]
        proposed.extend(_backfill(width - len(proposed), seen, concept_state))
        return proposed


def _target_concept(concept_state: dict) -> str | None:
    if not isinstance(concept_state, dict):
        return None
    target = concept_state.get("target_concept")
    return None if target is None else str(target)


def _propose_prompt(
    hypotheses: list[Hypothesis],
    concept_state: dict,
    width: int,
    rejected: list[str],
) -> str:
    # Same belief set the value function scores against (ValueFunction.score
    # filters to ACTIVE too), so proposer and scorer never disagree about
    # what's currently in play. All five layers, not just knowledge.
    listing = "\n".join(
        f"- {h.id} [{h.layer.value}] {h.statement} "
        f"(p={h.probability:.2f}, c={h.confidence:.2f})"
        for h in hypotheses
        if h.tier is Tier.ACTIVE
    )
    vocabulary = ", ".join(a.value for a in TeachingAction)
    correction = ""
    if rejected:
        correction = (
            "\nYour previous attempt included values outside the vocabulary: "
            f"{', '.join(rejected)}. Use only the listed action values.\n"
        )
    return (
        "PROPOSE:ACTIONS\n"
        f"Propose exactly {width} distinct teaching actions worth scoring "
        "for this turn. Choose from this vocabulary only:\n"
        f"{vocabulary}\n\n"
        f"concept_state={json.dumps(concept_state, sort_keys=True, default=str)}\n"
        "hypotheses (all layers):\n"
        f"{listing}\n"
        f"{correction}"
        'Respond with JSON: [{"action": "...", "target_concept": "..." or null, '
        '"rationale": "one short sentence"}, ...]'
    )


def _coerce_action(raw: object) -> TeachingAction | None:
    if isinstance(raw, TeachingAction):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        return TeachingAction(raw.strip().lower())
    except ValueError:
        pass
    try:
        return TeachingAction[raw.strip().upper()]
    except KeyError:
        return None


def _parse_proposals(
    raw: str, concept_state: dict
) -> tuple[list[CandidateAction], list[str]]:
    """Split an LLM proposal payload into valid candidates and rejects."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return [], ["<unparseable response>"]
    if not isinstance(parsed, list):
        return [], ["<response was not a JSON list>"]

    default_target = _target_concept(concept_state)
    valid: list[CandidateAction] = []
    rejected: list[str] = []
    for item in parsed:
        if not isinstance(item, dict):
            rejected.append(repr(item))
            continue
        action = _coerce_action(item.get("action"))
        if action is None:
            rejected.append(repr(item.get("action")))
            continue
        target = item.get("target_concept")
        rationale = item.get("rationale")
        valid.append(
            CandidateAction(
                action=action,
                target_concept=default_target if target is None else str(target),
                rationale=str(rationale) if rationale else "",
            )
        )
    return valid, rejected


def _backfill(
    n: int, seen: set[TeachingAction], concept_state: dict
) -> list[CandidateAction]:
    if n <= 0:
        return []
    target = _target_concept(concept_state)
    filler = [a for a in TeachingAction if a not in seen][:n]
    return [
        CandidateAction(
            action=a,
            target_concept=target,
            rationale="backfill: proposer returned fewer candidates than "
            "generation_width",
        )
        for a in filler
    ]


class Teach:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(self, action: CandidateAction) -> str:
        prompt = (
            "TEACH: "
            + json.dumps(
                {
                    "action": action.action.value,
                    "target_concept": action.target_concept,
                    "rationale": action.rationale,
                }
            )
        )
        return await self._llm.complete(prompt)


class Test:
    async def run(self, hypotheses: list[Hypothesis]) -> str:
        # Real diagnostic-question generation lives later.
        return "Can you walk me through how you'd approach this in your own words?"


class Diagnose:
    """Compares a learner's apparent belief against the world model.

    The response-vs-expectation classification is still the Step-2
    stub — a real classifier lives later. What's real here:

    1. GroundConcept determines which concept (if any) `response` is
       actually about, from the full list of nodes in *this session's*
       linked graph (`session_id` -> `concept_graph_id`, via
       `TranscriptStore.get_concept_graph_id` — a session's graph is
       set once at creation, same pattern as its learner). A
       hallucinated id GroundConcept didn't validate, or a grounding
       below `_GROUNDING_CONFIDENCE_THRESHOLD`, skips the rest of this
       method gracefully, same as an unrecognized concept_id always has.
       The grounding's own concept_id/confidence is recorded on the
       output regardless of whether it clears the threshold, so a
       low-confidence-but-still-acted-on grounding stays visible on
       review, not folded silently into the final MismatchResult.
    2. The learner's OverlayEntry and the mental_model hypotheses
       already linked to the grounded concept (via
       `HypothesisStore.list_by_concept` — Hypothesis itself carries no
       concept_id; see `hypothesis_concepts`) go to `MismatchDetector`,
       whose judgment is acted on:

       - `possible_world_model_error` above `DIAGNOSE_MISMATCH_THRESHOLD`
         → propose a `WorldModelRevision`. Never auto-applied — it sits
         pending until a human reviews it (`probe review-revisions`).
         `result["mismatch"]` (and its `confidence`) is persisted
         regardless of whether the threshold is cleared — see
         CLAUDE.md invariant 2 — so a below-threshold mismatch the
         loop declined to act on is still visible on review, not
         silently dropped.
       - `learner_misconception` → append supporting evidence to each
         matched hypothesis via `HypothesisStore.reweight` directly,
         not by calling `Update().run(...)`: CLAUDE.md invariant 2
         forbids invoking a Node's `run()` from a production path
         outside `SessionLoop._call_node`, so this goes straight to the
         same `reweight()` path `Update` itself uses — nothing bypasses
         evidence-append.

    Concept *selection* ("what should we teach next") is explicitly out
    of scope here — this only identifies what the current response is
    about.
    """

    # The revision-proposal threshold lives at module level as
    # DIAGNOSE_MISMATCH_THRESHOLD (it directly controls how often the
    # possible_world_model_error path can fire, so it's named and
    # documented there, not buried here).
    #
    # _GROUNDING_CONFIDENCE_THRESHOLD is a separate, unrelated decision
    # (whether GroundConcept's judgment is trusted enough to run
    # mismatch detection at all) — still an arbitrary midpoint
    # placeholder, not measured or reasoned from anything. Do not
    # assume tuning one of these two thresholds should move the other.
    _GROUNDING_CONFIDENCE_THRESHOLD = 0.5

    def __init__(
        self,
        mismatch_detector: MismatchDetector,
        ground_concept: GroundConcept,
        hypothesis_store: HypothesisStore,
        revision_store: WorldModelRevisionStore,
        concept_graph: ConceptGraph,
        learner_overlay: LearnerOverlay,
        transcript: TranscriptStore,
    ) -> None:
        self._detector = mismatch_detector
        self._grounder = ground_concept
        self._hyp = hypothesis_store
        self._revisions = revision_store
        self._concepts = concept_graph
        self._overlay = learner_overlay
        self._transcript = transcript

    async def run(
        self,
        response: str,
        expectation: str,
        session_id: UUID,
        turn_id: UUID,
    ) -> dict:
        result: dict[str, Any] = {
            "classification": "unknown",
            "matched_expectation": False,
            "notes": "stub response/expectation classifier — no real "
            "classifier yet",
            "grounding": None,
            "mismatch": None,
            "action_taken": "none",
            "revision_id": None,
            "reweighted_hypothesis_ids": [],
            # Real LLM calls this run() made: grounding always counts,
            # mismatch detection only if grounding cleared the
            # threshold. Costs nothing against the stub, but means the
            # per-turn call-count data is already sitting in node_calls
            # once a real LLM is behind this instead of needing
            # instrumentation added retroactively — same reasoning as
            # ActionScore.information_value_call_count.
            "llm_call_count": 0,
        }

        learner_id = await self._transcript.get_learner_id(session_id)
        concept_graph_id = await self._transcript.get_concept_graph_id(session_id)

        candidates = await self._concepts.list_concepts(concept_graph_id)
        grounding = await self._grounder.detect(response, candidates)
        result["llm_call_count"] += self._grounder.last_call_count
        result["grounding"] = grounding.model_dump(mode="json")

        if (
            grounding.concept_id is None
            or grounding.confidence < self._GROUNDING_CONFIDENCE_THRESHOLD
        ):
            result["notes"] += (
                "; response did not clearly ground to a concept in this "
                "session's graph, skipped mismatch check"
            )
            return result
        concept_id = grounding.concept_id

        concept = await self._concepts.get_concept(concept_graph_id, concept_id)
        if concept is None:
            result["notes"] += (
                f"; concept {concept_id!r} not found, skipped mismatch check"
            )
            return result

        overlay_entry = await self._overlay.get_state(
            learner_id, concept_graph_id, concept_id
        )
        hypotheses = await self._hyp.list_by_concept(
            concept_graph_id, concept_id, layer=Layer.MENTAL_MODEL, tier=Tier.ACTIVE
        )

        mismatch = await self._detector.detect(
            concept_id, concept, overlay_entry, hypotheses
        )
        result["llm_call_count"] += self._detector.last_call_count
        if mismatch is None:
            return result
        result["mismatch"] = mismatch.model_dump(mode="json")

        if (
            mismatch.suggested_cause is SuggestedCause.POSSIBLE_WORLD_MODEL_ERROR
            and mismatch.confidence >= DIAGNOSE_MISMATCH_THRESHOLD
        ):
            revision = await self._revisions.propose(
                WorldModelRevision(
                    concept_graph_id=concept_graph_id,
                    concept_id=concept_id,
                    proposed_change=(
                        f"possible error in concept {concept_id!r}: learner "
                        f"claims {mismatch.learner_claim!r}, which conflicts "
                        f"with world_claim {mismatch.world_claim!r}"
                    ),
                    evidence_refs=[
                        EvidenceRef(turn_id=turn_id, polarity=Polarity.SUPPORTING)
                    ],
                    confidence=mismatch.confidence,
                )
            )
            result["action_taken"] = "revision_proposed"
            result["revision_id"] = str(revision.id)
        elif mismatch.suggested_cause is SuggestedCause.LEARNER_MISCONCEPTION:
            reweighted_ids: list[str] = []
            for hyp in hypotheses:
                evidence_ref = EvidenceRef(
                    turn_id=turn_id, polarity=Polarity.SUPPORTING
                )
                new_probability = hyp.probability + mismatch.confidence * (
                    1.0 - hyp.probability
                )
                new_confidence = hyp.confidence + mismatch.confidence * (
                    1.0 - hyp.confidence
                )
                await self._hyp.reweight(
                    hyp.id,
                    min(1.0, new_probability),
                    min(1.0, new_confidence),
                    evidence_ref,
                )
                reweighted_ids.append(str(hyp.id))
            if reweighted_ids:
                result["action_taken"] = "hypothesis_reweighted"
            result["reweighted_hypothesis_ids"] = reweighted_ids

        return result


class Update:
    async def run(
        self,
        proposals: list[ProposedEvidence],
        hypothesis_store: HypothesisStore,
    ) -> list[EvidenceRef]:
        applied: list[EvidenceRef] = []
        for prop in proposals:
            await hypothesis_store.reweight(
                prop.hypothesis_id,
                prop.new_probability,
                prop.new_confidence,
                prop.evidence_ref,
            )
            applied.append(prop.evidence_ref)
        return applied


class Replan:
    """Decides the next turn's generation width from current uncertainty.

    Reads the active-tier hypotheses, computes total Bernoulli entropy
    (bits) across their `probability` values, and returns an integer
    generation_width in [_MIN_WIDTH, _MAX_WIDTH]. Higher entropy → wider
    generation next turn. The width is threaded into next turn's
    `Infer.run(...)` by SessionLoop.

    This is the uncertainty-budget hook from Step 5 of the build plan.
    The exact formula (bit-sum + linear map) is placeholder — the
    contract that matters right now is (a) Replan runs every turn, (b)
    its output feeds the next Infer, (c) it lives in `node_calls`.
    """

    async def run(self, hypotheses: list[Hypothesis]) -> int:
        active = [h for h in hypotheses if h.tier is Tier.ACTIVE]
        if not active:
            return _MIN_WIDTH
        total_entropy_bits = sum(
            _bernoulli_entropy(h.probability) for h in active
        )
        return max(_MIN_WIDTH, min(_MAX_WIDTH, 1 + int(round(total_entropy_bits))))


def _bernoulli_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))
