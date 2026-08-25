"""Node classes for the probe reasoning loop.

Every node exposes a single async `run(...)` method whose signature is
the node's semantic contract. Persistence of inputs/outputs is *not*
each node's responsibility — `SessionLoop._call_node()` intercepts and
records every call. See CLAUDE.md invariant 2.

Most nodes are stubs at this stage; the reasoning logic lands in later
steps. `Update` is real because it only depends on the already-built
HypothesisStore.
"""

from __future__ import annotations

import json
import math

from probe.llm import LLMClient
from probe.models import (
    ActionScore,
    CandidateAction,
    EvidenceRef,
    Hypothesis,
    PlanOutput,
    ProposedEvidence,
    TeachingAction,
    Tier,
)
from probe.store import HypothesisStore
from probe.value_function import ValueFunction

DEFAULT_GENERATION_WIDTH = 3
_MIN_WIDTH = 1
_MAX_WIDTH = 8

# One corrective re-ask when the proposer emits actions outside the enum.
_MAX_PROPOSE_ATTEMPTS = 2


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
    async def run(self, response: str, expectation: str) -> dict:
        # Real classifier lives later.
        return {
            "classification": "unknown",
            "matched_expectation": False,
            "notes": "stub diagnose — no real classifier yet",
        }


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
