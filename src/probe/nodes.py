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

    Candidate generation is deterministic first-N-of-enum for now. When
    the LLM-driven candidate proposer lands, only the `_generate` method
    changes; scoring and selection stay identical.

    The returned `PlanOutput` carries both the winner (for Teach) and
    the full per-candidate `ActionScore` breakdown. Both survive
    together through the SessionLoop audit choke point (CLAUDE.md
    invariant 2) so ablation runs can inspect every candidate's terms,
    not just the winner's.
    """

    def __init__(self, value_function: ValueFunction) -> None:
        self._vf = value_function

    async def run(
        self,
        hypotheses: list[Hypothesis],
        concept_state: dict,
        generation_width: int,
    ) -> PlanOutput:
        candidates = self._generate(generation_width, concept_state)
        scores: list[ActionScore] = []
        for candidate in candidates:
            scores.append(await self._vf.score(candidate, hypotheses, concept_state))
        winner_score = max(scores, key=lambda s: s.total)
        return PlanOutput(winner=winner_score.candidate, scores=scores)

    def _generate(
        self, n: int, concept_state: dict
    ) -> list[CandidateAction]:
        target = None
        if isinstance(concept_state, dict):
            target = concept_state.get("target_concept")
        actions = list(TeachingAction)[: max(1, n)]
        return [
            CandidateAction(
                action=a,
                target_concept=target,
                rationale=f"stub: first-{len(actions)} enum generation",
            )
            for a in actions
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
