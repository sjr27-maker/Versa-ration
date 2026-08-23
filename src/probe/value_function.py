"""The six-term value function.

Each term is an independently-callable async method. Each is gated by a
flag on `ValueFunctionConfig` (CLAUDE.md invariant 3) — a disabled term
returns 0.0 and skips its LLM calls. `score()` aggregates:

    total = learning_value + information_value + long_term_value
          - time_cost - cognitive_cost - frustration_risk

The full six-term breakdown is preserved on `ActionScore` even when
some terms are zero, so ablation runs stay comparable in `node_calls`.
"""

from __future__ import annotations

import json
import math

from pydantic import BaseModel

from probe.llm import LLMClient
from probe.models import (
    ActionScore,
    CandidateAction,
    Hypothesis,
    Layer,
    TeachingAction,
    Tier,
)


class ValueFunctionConfig(BaseModel):
    enable_learning_value: bool = True
    enable_information_value: bool = True
    enable_long_term_value: bool = True
    enable_time_cost: bool = True
    enable_cognitive_cost: bool = True
    enable_frustration_risk: bool = True


# Static time-cost table, in [0, 1]. The numbers represent expected
# turns-to-complete relative to a plain EXPLAIN.
#
#   Fast (single-message transmissions): 0.10-0.20
#   Medium (single-turn question expecting a response): 0.20-0.35
#   Slow (multi-turn interactions where the student does the work): 0.40-0.55
#
# These are placeholders that reflect intuition, not measurements. Once
# we have session data we'll fit real numbers; the shape of the table
# (relative ordering) is what matters right now.
_TIME_COST: dict[TeachingAction, float] = {
    TeachingAction.EXPLAIN: 0.10,
    TeachingAction.RECALL: 0.10,
    TeachingAction.REPHRASE: 0.10,
    TeachingAction.ANALOGY: 0.10,
    TeachingAction.VISUALIZE: 0.15,
    TeachingAction.CONNECT: 0.15,
    TeachingAction.SLOW_DOWN: 0.15,
    TeachingAction.CHANGE_REPRESENTATION: 0.20,
    TeachingAction.CORRECT_MISCONCEPTION: 0.20,
    TeachingAction.EXAMPLE: 0.20,
    TeachingAction.ASK: 0.25,
    TeachingAction.COUNTEREXAMPLE: 0.25,
    TeachingAction.COMPARE: 0.25,
    TeachingAction.DECOMPOSE: 0.30,
    TeachingAction.INCREASE_DIFFICULTY: 0.30,
    TeachingAction.DERIVE: 0.35,
    TeachingAction.APPLY: 0.40,
    TeachingAction.QUIZ: 0.45,
    TeachingAction.CHALLENGE: 0.50,
    TeachingAction.TEACH_BACK: 0.50,
    TeachingAction.SIMULATE: 0.55,
}


def _bernoulli_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _hypothesis_listing(hypotheses: list[Hypothesis]) -> str:
    return "\n".join(
        f"- {h.id} [{h.layer.value}/{h.tier.value}] {h.statement} "
        f"(p={h.probability:.2f}, c={h.confidence:.2f})"
        for h in hypotheses
    )


class ValueFunction:
    def __init__(
        self,
        llm: LLMClient,
        config: ValueFunctionConfig | None = None,
    ) -> None:
        self._llm = llm
        self._config = config or ValueFunctionConfig()
        # Updated by information_value; read by score() into ActionScore.
        self._last_info_call_count: int = 0

    @property
    def config(self) -> ValueFunctionConfig:
        return self._config

    async def learning_value(
        self,
        action: CandidateAction,
        hypotheses: list[Hypothesis],
        concept_state: dict,
    ) -> float:
        prompt = (
            "SCORE:LEARNING_VALUE\n"
            f"action={action.action.value}\n"
            f"target_concept={action.target_concept}\n"
            f"rationale={action.rationale}\n"
            f"concept_state={json.dumps(concept_state, sort_keys=True)}\n"
            "hypotheses:\n"
            f"{_hypothesis_listing(hypotheses)}\n"
            "Estimate expected improvement in concept understanding, 0-1. "
            "Respond with a single float."
        )
        return await self._ask_scalar(prompt)

    async def information_value(
        self, action: CandidateAction, hypotheses: list[Hypothesis]
    ) -> float:
        self._last_info_call_count = 0
        if not hypotheses:
            return 0.0

        h_before = sum(_bernoulli_entropy(h.probability) for h in hypotheses)

        responses_prompt = (
            "SCORE:INFO_RESPONSES\n"
            f"action={action.action.value}\n"
            f"target_concept={action.target_concept}\n"
            "Enumerate 2-4 plausible student responses. "
            'Respond with JSON: [{"response": "...", "probability": 0.5}, ...]'
        )
        raw = await self._llm.complete(responses_prompt)
        self._last_info_call_count += 1
        try:
            responses = json.loads(raw)
        except json.JSONDecodeError:
            return 0.0
        if not isinstance(responses, list) or not responses:
            return 0.0

        expected_h_after = 0.0
        for entry in responses:
            if not isinstance(entry, dict):
                continue
            prob = float(entry.get("probability", 0.0))
            response_text = entry.get("response", "")
            update_prompt = (
                "SCORE:INFO_UPDATE\n"
                f"response={response_text}\n"
                "hypotheses:\n"
                f"{_hypothesis_listing(hypotheses)}\n"
                "For each hypothesis id, estimate its new probability given "
                'this response. Respond with JSON: {"<hyp_id>": <new_prob>}'
            )
            raw_update = await self._llm.complete(update_prompt)
            self._last_info_call_count += 1
            try:
                update = json.loads(raw_update)
            except json.JSONDecodeError:
                update = {}
            if not isinstance(update, dict):
                update = {}
            h_after_given = 0.0
            for h in hypotheses:
                new_p = update.get(str(h.id), h.probability)
                try:
                    new_p = float(new_p)
                except (TypeError, ValueError):
                    new_p = h.probability
                new_p = max(0.0, min(1.0, new_p))
                h_after_given += _bernoulli_entropy(new_p)
            expected_h_after += prob * h_after_given

        return h_before - expected_h_after

    async def long_term_value(
        self, action: CandidateAction, hypotheses: list[Hypothesis]
    ) -> float:
        # TODO: needs resurrection/trajectory history we don't populate
        # yet (would look back across archived hypotheses that came back
        # via resurrect(), and past sessions on the same concept). Returns
        # 0.0 deliberately — a fake random value here would poison the
        # ablation dashboard, which is meant to reveal exactly this
        # kind of gap.
        return 0.0

    def time_cost(self, action: CandidateAction) -> float:
        # Sync, no LLM — pure lookup.
        return _TIME_COST[action.action]

    async def cognitive_cost(
        self, action: CandidateAction, hypotheses: list[Hypothesis]
    ) -> float:
        cog_hyps = [h for h in hypotheses if h.layer is Layer.COGNITIVE_STATE]
        prompt = (
            "SCORE:COGNITIVE_COST\n"
            f"action={action.action.value}\n"
            f"target_concept={action.target_concept}\n"
            "cognitive_state hypotheses:\n"
            f"{_hypothesis_listing(cog_hyps)}\n"
            "Estimate the cognitive load this action imposes, 0-1. "
            "Respond with a single float."
        )
        return await self._ask_scalar(prompt)

    async def frustration_risk(
        self, action: CandidateAction, hypotheses: list[Hypothesis]
    ) -> float:
        cog_hyps = [h for h in hypotheses if h.layer is Layer.COGNITIVE_STATE]
        prompt = (
            "SCORE:FRUSTRATION_RISK\n"
            f"action={action.action.value}\n"
            f"target_concept={action.target_concept}\n"
            f"rationale={action.rationale}\n"
            "cognitive_state hypotheses:\n"
            f"{_hypothesis_listing(cog_hyps)}\n"
            "Estimate risk of frustration, 0-1. "
            "Respond with a single float."
        )
        return await self._ask_scalar(prompt)

    async def score(
        self,
        action: CandidateAction,
        hypotheses: list[Hypothesis],
        concept_state: dict,
    ) -> ActionScore:
        cfg = self._config
        active = [h for h in hypotheses if h.tier is Tier.ACTIVE]

        lv = (
            await self.learning_value(action, active, concept_state)
            if cfg.enable_learning_value
            else 0.0
        )
        iv = (
            await self.information_value(action, active)
            if cfg.enable_information_value
            else 0.0
        )
        iv_calls = (
            self._last_info_call_count if cfg.enable_information_value else 0
        )
        ltv = (
            await self.long_term_value(action, active)
            if cfg.enable_long_term_value
            else 0.0
        )
        tc = self.time_cost(action) if cfg.enable_time_cost else 0.0
        cc = (
            await self.cognitive_cost(action, active)
            if cfg.enable_cognitive_cost
            else 0.0
        )
        fr = (
            await self.frustration_risk(action, active)
            if cfg.enable_frustration_risk
            else 0.0
        )

        total = lv + iv + ltv - tc - cc - fr

        return ActionScore(
            candidate=action,
            learning_value=lv,
            information_value=iv,
            long_term_value=ltv,
            time_cost=tc,
            cognitive_cost=cc,
            frustration_risk=fr,
            total=total,
            information_value_call_count=iv_calls,
        )

    async def _ask_scalar(self, prompt: str) -> float:
        raw = await self._llm.complete(prompt)
        try:
            value = float(raw.strip())
        except (ValueError, AttributeError):
            return 0.0
        return max(0.0, min(1.0, value))
