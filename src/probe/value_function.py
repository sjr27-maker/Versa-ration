"""The six-term value function.

Each term is an independently-callable async method. Each is gated by a
flag on `ValueFunctionConfig` (CLAUDE.md invariant 3) — a disabled term
returns 0.0 and skips its LLM calls. `score()` aggregates:

    total = learning_value + information_value + long_term_value
          - time_cost - cognitive_cost - frustration_risk

The full six-term breakdown is preserved on `ActionScore` even when
some terms are zero, so ablation runs stay comparable in `node_calls`.

Scoring anomalies are named on `ActionScore.flags` rather than silently
folded into `total` — see `FLAG_NEGATIVE_INFORMATION_VALUE`.
"""

from __future__ import annotations

import asyncio
import json
import logging
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

logger = logging.getLogger(__name__)

# Flag names written onto ActionScore.flags. Expected information gain is
# non-negative in theory; a negative one means the response model and the
# posterior model disagree, so it's a modelling bug worth seeing rather
# than a number to quietly absorb into `total`.
FLAG_NEGATIVE_INFORMATION_VALUE = "negative_information_value"


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
        # Updated by their respective methods; read by score() into
        # ActionScore. Tracked per-term (not inferred from which terms
        # are enabled) so a future retry addition to any one of these
        # doesn't silently make an inferred count wrong — same rationale
        # as information_value's counter, now applied to the other three
        # LLM-calling terms too (long_term_value and time_cost never
        # call the LLM, so they have no counter).
        self._last_info_call_count: int = 0
        self._last_learning_value_call_count: int = 0
        self._last_cognitive_cost_call_count: int = 0
        self._last_frustration_risk_call_count: int = 0

    @property
    def config(self) -> ValueFunctionConfig:
        return self._config

    async def learning_value(
        self,
        action: CandidateAction,
        hypotheses: list[Hypothesis],
        concept_state: dict,
    ) -> float:
        value, count = await self._learning_value_impl(action, hypotheses, concept_state)
        self._last_learning_value_call_count = count
        return value

    async def _learning_value_impl(
        self,
        action: CandidateAction,
        hypotheses: list[Hypothesis],
        concept_state: dict,
    ) -> tuple[float, int]:
        # Returns (value, call_count) instead of writing to
        # self._last_learning_value_call_count directly: score() gathers
        # this concurrently across candidates that share one ValueFunction
        # instance, so a shared instance attribute would race across
        # candidates. The public method above still sets the attribute,
        # for a solo/direct call (and the one existing test asserting it).
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
        value = await self._ask_scalar(prompt)
        return value, 1

    async def information_value(
        self, action: CandidateAction, hypotheses: list[Hypothesis]
    ) -> float:
        value, count = await self._information_value_impl(action, hypotheses)
        self._last_info_call_count = count
        return value

    async def _information_value_impl(
        self, action: CandidateAction, hypotheses: list[Hypothesis]
    ) -> tuple[float, int]:
        # Same (value, call_count) convention as _learning_value_impl —
        # see its comment. This one also gathers its own nested per-
        # response SCORE:INFO_UPDATE calls concurrently: each response
        # simulation is independent of the others (they all read the
        # same `hypotheses` snapshot and don't mutate shared state).
        if not hypotheses:
            return 0.0, 0

        h_before = sum(_bernoulli_entropy(h.probability) for h in hypotheses)

        responses_prompt = (
            "SCORE:INFO_RESPONSES\n"
            f"action={action.action.value}\n"
            f"target_concept={action.target_concept}\n"
            "Enumerate 2-4 plausible student responses. "
            'Respond with JSON: [{"response": "...", "probability": 0.5}, ...]'
        )
        raw = await self._llm.complete(responses_prompt)
        call_count = 1
        try:
            responses = json.loads(raw)
        except json.JSONDecodeError:
            return 0.0, call_count
        if not isinstance(responses, list) or not responses:
            return 0.0, call_count

        # Non-dict entries never made an UPDATE call in the original
        # sequential version either (the `continue` skipped straight past
        # it) — filtered out before gathering so call_count matches.
        entries = [e for e in responses if isinstance(e, dict)]

        def _update_prompt(entry: dict) -> str:
            return (
                "SCORE:INFO_UPDATE\n"
                f"response={entry.get('response', '')}\n"
                "hypotheses:\n"
                f"{_hypothesis_listing(hypotheses)}\n"
                "For each hypothesis id, estimate its new probability given "
                'this response. Respond with JSON: {"<hyp_id>": <new_prob>}'
            )

        raw_updates = await asyncio.gather(
            *(self._llm.complete(_update_prompt(entry)) for entry in entries)
        )
        call_count += len(raw_updates)

        expected_h_after = 0.0
        for entry, raw_update in zip(entries, raw_updates, strict=True):
            prob = float(entry.get("probability", 0.0))
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

        gain = h_before - expected_h_after
        if gain < 0.0:
            logger.warning(
                "negative information_value for action=%s "
                "(target_concept=%s): entropy_before=%.6f bits, "
                "expected_entropy_after=%.6f bits, gain=%.6f bits. "
                "Expected information gain should be >= 0; the simulated "
                "responses are pushing hypotheses toward higher entropy.",
                action.action.value,
                action.target_concept,
                h_before,
                expected_h_after,
                gain,
            )
        return gain, call_count

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
        value, count = await self._cognitive_cost_impl(action, hypotheses)
        self._last_cognitive_cost_call_count = count
        return value

    async def _cognitive_cost_impl(
        self, action: CandidateAction, hypotheses: list[Hypothesis]
    ) -> tuple[float, int]:
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
        value = await self._ask_scalar(prompt)
        return value, 1

    async def frustration_risk(
        self, action: CandidateAction, hypotheses: list[Hypothesis]
    ) -> float:
        value, count = await self._frustration_risk_impl(action, hypotheses)
        self._last_frustration_risk_call_count = count
        return value

    async def _frustration_risk_impl(
        self, action: CandidateAction, hypotheses: list[Hypothesis]
    ) -> tuple[float, int]:
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
        value = await self._ask_scalar(prompt)
        return value, 1

    async def score(
        self,
        action: CandidateAction,
        hypotheses: list[Hypothesis],
        concept_state: dict,
    ) -> ActionScore:
        cfg = self._config
        active = [h for h in hypotheses if h.tier is Tier.ACTIVE]

        # The four LLM-calling terms (plus the no-LLM long_term_value
        # placeholder) are independent of each other for a single
        # candidate — gathered concurrently. Each disabled term is a
        # plain 0.0/no-call coroutine so the gather always has a fixed
        # shape regardless of config. Call counts come back as part of
        # each result tuple (via the _impl methods), not through a
        # shared self._last_*_call_count attribute — score() itself is
        # gathered *across candidates* by Plan.run() using one shared
        # ValueFunction instance, so an instance attribute would race
        # between candidates scored at the same time.
        async def _lv() -> tuple[float, int]:
            if not cfg.enable_learning_value:
                return 0.0, 0
            return await self._learning_value_impl(action, active, concept_state)

        async def _iv() -> tuple[float, int]:
            if not cfg.enable_information_value:
                return 0.0, 0
            return await self._information_value_impl(action, active)

        async def _ltv() -> float:
            if not cfg.enable_long_term_value:
                return 0.0
            return await self.long_term_value(action, active)

        async def _cc() -> tuple[float, int]:
            if not cfg.enable_cognitive_cost:
                return 0.0, 0
            return await self._cognitive_cost_impl(action, active)

        async def _fr() -> tuple[float, int]:
            if not cfg.enable_frustration_risk:
                return 0.0, 0
            return await self._frustration_risk_impl(action, active)

        (lv, lv_calls), (iv, iv_calls), ltv, (cc, cc_calls), (fr, fr_calls) = (
            await asyncio.gather(_lv(), _iv(), _ltv(), _cc(), _fr())
        )
        tc = self.time_cost(action) if cfg.enable_time_cost else 0.0

        total = lv + iv + ltv - tc - cc - fr

        flags: list[str] = []
        if cfg.enable_information_value and iv < 0.0:
            flags.append(FLAG_NEGATIVE_INFORMATION_VALUE)

        return ActionScore(
            candidate=action,
            learning_value=lv,
            information_value=iv,
            long_term_value=ltv,
            time_cost=tc,
            cognitive_cost=cc,
            frustration_risk=fr,
            total=total,
            learning_value_call_count=lv_calls,
            information_value_call_count=iv_calls,
            cognitive_cost_call_count=cc_calls,
            frustration_risk_call_count=fr_calls,
            flags=flags,
        )

    async def _ask_scalar(self, prompt: str) -> float:
        raw = await self._llm.complete(prompt)
        try:
            value = float(raw.strip())
        except (ValueError, AttributeError):
            return 0.0
        return max(0.0, min(1.0, value))
