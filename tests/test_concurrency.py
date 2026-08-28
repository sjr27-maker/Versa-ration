"""Verifies the turn-parallelization work (ValueFunction.score() gathering
its terms, Plan.run() gathering across candidates, HypothesisGenerator
gathering a wave's sibling expansions) doesn't corrupt call-count
accounting under real concurrent scheduling.

StubLLMClient's complete() has no `await` inside it, so gathering calls
against it never actually interleaves — every coroutine runs to
completion the instant it's scheduled, in submission order. That would
hide exactly the kind of race a shared-instance-attribute call counter
is vulnerable to (see value_function.py's _learning_value_impl/etc.
comments). _SlowStub wraps a StubLLMClient with a real `await
asyncio.sleep(...)` before each response so concurrent callers actually
interleave, then every test here asserts the returned call counts sum
to the stub's own ground-truth total() — the same style of verification
CLAUDE.md/nodes.py's MAX_CALLS_PER_TURN docstring describes ("28 real
calls, 28 counted") applied to the concurrent path specifically.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from probe.hypothesis_generator import HypothesisGenerator
from probe.llm import StubLLMClient
from probe.models import CandidateAction, Hypothesis, Layer, TeachingAction, Tier
from probe.nodes import Plan
from probe.reasoning_budget import BranchBudgetConfig
from probe.value_function import ValueFunction


class _SlowStub:
    """Wraps a StubLLMClient with a real await point so concurrent
    callers against the same instance genuinely interleave, instead of
    each running to completion synchronously the moment it's scheduled.
    `total()` is the ground truth every test checks returned call counts
    against."""

    def __init__(self, inner: StubLLMClient, delay: float = 0.005) -> None:
        self._inner = inner
        self._delay = delay
        self._count = 0

    async def complete(self, prompt: str) -> str:
        await asyncio.sleep(self._delay)
        self._count += 1
        return await self._inner.complete(prompt)

    def total(self) -> int:
        return self._count


def _hyp(probability: float = 0.5) -> Hypothesis:
    return Hypothesis(
        layer=Layer.KNOWLEDGE,
        statement="stub hypothesis",
        probability=probability,
        confidence=0.5,
        tier=Tier.ACTIVE,
    )


def _action(kind: TeachingAction) -> CandidateAction:
    return CandidateAction(action=kind, target_concept="recursion", rationale="test")


# Two simulated responses -> information_value makes 1 (RESPONSES) + 2
# (one UPDATE per response) = 3 calls per candidate scored.
_TWO_INFO_RESPONSES = json.dumps(
    [
        {"response": "a", "probability": 0.5},
        {"response": "b", "probability": 0.5},
    ]
)
_CANNED = {
    "SCORE:LEARNING_VALUE": "0.5",
    "SCORE:COGNITIVE_COST": "0.3",
    "SCORE:FRUSTRATION_RISK": "0.2",
    "SCORE:INFO_RESPONSES": _TWO_INFO_RESPONSES,
    "SCORE:INFO_UPDATE": "{}",
}


@pytest.mark.asyncio
async def test_concurrent_score_calls_dont_cross_contaminate_call_counts():
    """N candidates scored concurrently against one shared ValueFunction
    — each candidate's own ActionScore call-count fields must reflect
    only its own calls (1 learning_value + 3 information_value +
    1 cognitive_cost + 1 frustration_risk = 6 each), not a mix of
    several candidates' calls landing on the same counter."""
    stub = _SlowStub(StubLLMClient(canned=_CANNED))
    vf = ValueFunction(stub)
    candidates = [
        _action(TeachingAction.EXPLAIN),
        _action(TeachingAction.ASK),
        _action(TeachingAction.QUIZ),
        _action(TeachingAction.EXAMPLE),
    ]

    scores = await asyncio.gather(*(vf.score(c, [_hyp()], {}) for c in candidates))

    for score in scores:
        assert score.learning_value_call_count == 1
        assert score.information_value_call_count == 3
        assert score.cognitive_cost_call_count == 1
        assert score.frustration_risk_call_count == 1

    per_candidate_calls = sum(
        s.learning_value_call_count
        + s.information_value_call_count
        + s.cognitive_cost_call_count
        + s.frustration_risk_call_count
        for s in scores
    )
    assert per_candidate_calls == stub.total() == 4 * 6


@pytest.mark.asyncio
async def test_plan_run_end_to_end_call_count_matches_ground_truth():
    """Same check through the real Plan.run() path (proposer generation
    + concurrent candidate scoring), the exact call SessionLoop makes
    each turn."""
    proposals = json.dumps(
        [
            {"action": "explain", "target_concept": None, "rationale": "r1"},
            {"action": "ask", "target_concept": None, "rationale": "r2"},
            {"action": "quiz", "target_concept": None, "rationale": "r3"},
            {"action": "example", "target_concept": None, "rationale": "r4"},
        ]
    )
    canned = dict(_CANNED, **{"PROPOSE:ACTIONS": proposals})
    stub = _SlowStub(StubLLMClient(canned=canned))
    plan = Plan(ValueFunction(stub), stub)

    output = await plan.run(hypotheses=[_hyp()], concept_state={}, generation_width=4)

    scoring_calls = sum(
        s.learning_value_call_count
        + s.information_value_call_count
        + s.cognitive_cost_call_count
        + s.frustration_risk_call_count
        for s in output.scores
    )
    total_expected = plan.last_generate_call_count + scoring_calls
    assert stub.total() == total_expected
    # 1 proposer call + 4 candidates * 6 scoring calls each.
    assert total_expected == 1 + 4 * 6


# --- HypothesisGenerator: a wave of siblings expanding concurrently ----

_FOUR_DISTINCT_INTENTS = json.dumps(
    [
        {
            "statement": "wants a real-world analogy for the new idea",
            "plausibility": 0.9,
            "predicted_next_turn": "will ask for an example",
        },
        {
            "statement": "is missing an earlier prerequisite concept entirely",
            "plausibility": 0.9,
            "predicted_next_turn": "will ask a clarifying question about basics",
        },
        {
            "statement": "is testing whether the tutor's claim is actually correct",
            "plausibility": 0.9,
            "predicted_next_turn": "will point out a perceived inconsistency",
        },
        {
            "statement": "already knows this and is bored, wants to move faster",
            "plausibility": 0.9,
            "predicted_next_turn": "will ask to skip ahead",
        },
    ]
)
_EXPAND_LEAVES = json.dumps(
    {
        "layer_label": "knowledge_gap",
        "children": [
            {
                "statement": "specific gap A",
                "plausibility": 0.1,
                "predicted_next_turn": "asks about gap A",
            },
            {
                "statement": "specific gap B",
                "plausibility": 0.1,
                "predicted_next_turn": "asks about gap B",
            },
        ],
    }
)


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_wave_expansion_call_count_matches_ground_truth(
    branch_store, store, transcript, learner_id, clean_pool
):
    """4 root branches all clear the redundancy/plausibility gate and
    expand in the same wave — max_depth=1 caps it to exactly one
    concurrent wave of GENERATE:EXPAND calls, so the ground truth is
    exactly 1 (intent) + 4 (one expansion per root branch) = 5."""
    session_id = await transcript.create_session(learner_id, concept_graph_id=None)
    canned = {
        "GENERATE:INTENT": _FOUR_DISTINCT_INTENTS,
        "GENERATE:EXPAND": _EXPAND_LEAVES,
    }
    stub = _SlowStub(StubLLMClient(canned=canned))
    generator = HypothesisGenerator(
        stub, branch_store, BranchBudgetConfig(max_depth=1)
    )

    result = await generator.generate(
        session_id,
        0,
        "student: hello",
        [],
        CandidateAction(action=TeachingAction.EXPLAIN, target_concept=None, rationale="r"),
    )

    assert result.call_count == 5
    assert stub.total() == 5
    assert len(result.branches) == 4 + 4 * 2  # 4 roots + 2 children each
