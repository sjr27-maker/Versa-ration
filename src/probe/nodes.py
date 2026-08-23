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
    CandidateAction,
    EvidenceRef,
    Hypothesis,
    ProposedEvidence,
    Tier,
)
from probe.store import HypothesisStore

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
    async def run(
        self, hypotheses: list[Hypothesis], concept_state: dict
    ) -> list[CandidateAction]:
        # Step 3 will replace this with the real action space + value fn.
        return [
            CandidateAction(
                kind="teach",
                payload={"concept": "recursion", "approach": "worked-example"},
            ),
            CandidateAction(
                kind="teach",
                payload={"concept": "recursion", "approach": "socratic-question"},
            ),
            CandidateAction(
                kind="teach",
                payload={"concept": "recursion", "approach": "analogy"},
            ),
        ]


class Teach:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(self, action: CandidateAction) -> str:
        prompt = f"TEACH: kind={action.kind} payload={json.dumps(action.payload)}"
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
