import json
import logging
from uuid import uuid4

import pytest

from probe.llm import StubLLMClient
from probe.models import (
    ActionScore,
    CandidateAction,
    Hypothesis,
    Layer,
    TeachingAction,
    Tier,
)
from probe.value_function import (
    FLAG_NEGATIVE_INFORMATION_VALUE,
    ValueFunction,
    ValueFunctionConfig,
)


def _hyp(
    layer: Layer = Layer.KNOWLEDGE,
    probability: float = 0.5,
    tier: Tier = Tier.ACTIVE,
) -> Hypothesis:
    return Hypothesis(
        layer=layer,
        statement="stub hypothesis",
        probability=probability,
        confidence=0.5,
        tier=tier,
    )


def _action(kind: TeachingAction = TeachingAction.EXPLAIN) -> CandidateAction:
    return CandidateAction(action=kind, target_concept="recursion", rationale="test")


@pytest.mark.asyncio
async def test_learning_value_returns_stubbed_scalar_in_range():
    vf = ValueFunction(StubLLMClient(canned={"SCORE:LEARNING_VALUE": "0.42"}))
    result = await vf.learning_value(_action(), [_hyp()], {})
    assert result == pytest.approx(0.42)


@pytest.mark.asyncio
async def test_learning_value_clamps_out_of_range_llm_output():
    vf_hi = ValueFunction(StubLLMClient(canned={"SCORE:LEARNING_VALUE": "2.5"}))
    vf_lo = ValueFunction(StubLLMClient(canned={"SCORE:LEARNING_VALUE": "-0.3"}))
    vf_junk = ValueFunction(
        StubLLMClient(canned={"SCORE:LEARNING_VALUE": "not-a-number"})
    )
    assert await vf_hi.learning_value(_action(), [_hyp()], {}) == 1.0
    assert await vf_lo.learning_value(_action(), [_hyp()], {}) == 0.0
    assert await vf_junk.learning_value(_action(), [_hyp()], {}) == 0.0


@pytest.mark.asyncio
async def test_cognitive_cost_returns_scalar_in_range():
    vf = ValueFunction(StubLLMClient(canned={"SCORE:COGNITIVE_COST": "0.7"}))
    result = await vf.cognitive_cost(_action(), [_hyp(layer=Layer.COGNITIVE_STATE)])
    assert result == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_frustration_risk_returns_scalar_in_range():
    vf = ValueFunction(StubLLMClient(canned={"SCORE:FRUSTRATION_RISK": "0.15"}))
    result = await vf.frustration_risk(
        _action(), [_hyp(layer=Layer.COGNITIVE_STATE)]
    )
    assert result == pytest.approx(0.15)


@pytest.mark.asyncio
async def test_long_term_value_is_zero_placeholder():
    vf = ValueFunction(StubLLMClient())
    assert await vf.long_term_value(_action(), [_hyp()]) == 0.0


def test_time_cost_lookup_covers_every_teaching_action():
    vf = ValueFunction(StubLLMClient())
    for kind in TeachingAction:
        cost = vf.time_cost(_action(kind))
        assert 0.0 < cost <= 1.0


def test_time_cost_orders_slow_actions_above_fast_ones():
    vf = ValueFunction(StubLLMClient())
    assert vf.time_cost(_action(TeachingAction.SIMULATE)) > vf.time_cost(
        _action(TeachingAction.EXPLAIN)
    )
    assert vf.time_cost(_action(TeachingAction.CHALLENGE)) > vf.time_cost(
        _action(TeachingAction.RECALL)
    )


@pytest.mark.asyncio
async def test_information_value_is_zero_when_llm_returns_no_responses():
    vf = ValueFunction(StubLLMClient())  # default SCORE:INFO_RESPONSES -> "[]"
    assert await vf.information_value(_action(), [_hyp()]) == 0.0


@pytest.mark.asyncio
async def test_information_value_is_positive_when_expected_entropy_drops():
    # Hypothesis starts at max entropy (p=0.5); the stub simulates one
    # response that would push it near-certain (p=0.99, low entropy).
    hyp = _hyp(probability=0.5)
    canned = {
        "SCORE:INFO_RESPONSES": json.dumps(
            [{"response": "student nails it", "probability": 1.0}]
        ),
        "SCORE:INFO_UPDATE": json.dumps({str(hyp.id): 0.99}),
    }
    vf = ValueFunction(StubLLMClient(canned=canned))
    iv = await vf.information_value(_action(), [hyp])
    assert iv > 0.5  # H(0.5) = 1.0 bit, H(0.99) ≈ 0.08 → gain ≈ 0.92


@pytest.mark.asyncio
async def test_information_value_call_count_tracks_llm_invocations():
    hyp = _hyp(probability=0.5)
    canned = {
        "SCORE:INFO_RESPONSES": json.dumps(
            [
                {"response": "a", "probability": 0.5},
                {"response": "b", "probability": 0.5},
            ]
        ),
        "SCORE:INFO_UPDATE": json.dumps({str(hyp.id): 0.9}),
    }
    vf = ValueFunction(StubLLMClient(canned=canned))
    await vf.information_value(_action(), [hyp])
    # 1 call for INFO_RESPONSES + 1 call per response.
    assert vf._last_info_call_count == 3


@pytest.mark.asyncio
async def test_score_sums_terms_with_correct_signs():
    canned = {
        "SCORE:LEARNING_VALUE": "0.6",
        "SCORE:COGNITIVE_COST": "0.2",
        "SCORE:FRUSTRATION_RISK": "0.1",
        # No info_value with default empty responses; long_term_value = 0.
    }
    vf = ValueFunction(StubLLMClient(canned=canned))
    action = _action(TeachingAction.EXPLAIN)  # time_cost = 0.10
    score = await vf.score(action, [_hyp()], {})

    expected_total = 0.6 + 0.0 + 0.0 - 0.10 - 0.2 - 0.1
    assert score.total == pytest.approx(expected_total)
    assert score.learning_value == pytest.approx(0.6)
    assert score.information_value == pytest.approx(0.0)
    assert score.long_term_value == pytest.approx(0.0)
    assert score.time_cost == pytest.approx(0.10)
    assert score.cognitive_cost == pytest.approx(0.2)
    assert score.frustration_risk == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_score_with_information_value_disabled_differs_from_enabled():
    hyp = _hyp(probability=0.5)
    canned = {
        "SCORE:LEARNING_VALUE": "0.5",
        "SCORE:COGNITIVE_COST": "0.2",
        "SCORE:FRUSTRATION_RISK": "0.1",
        "SCORE:INFO_RESPONSES": json.dumps(
            [{"response": "student nails it", "probability": 1.0}]
        ),
        "SCORE:INFO_UPDATE": json.dumps({str(hyp.id): 0.99}),
    }
    action = _action(TeachingAction.EXPLAIN)

    vf_on = ValueFunction(StubLLMClient(canned=canned))
    score_on = await vf_on.score(action, [hyp], {})

    vf_off = ValueFunction(
        StubLLMClient(canned=canned),
        ValueFunctionConfig(enable_information_value=False),
    )
    score_off = await vf_off.score(action, [hyp], {})

    assert score_on.information_value > 0
    assert score_off.information_value == 0.0
    assert score_on.total != pytest.approx(score_off.total)
    assert score_on.total == pytest.approx(
        score_off.total + score_on.information_value
    )
    # Disabled term also skips its LLM calls.
    assert score_off.information_value_call_count == 0
    assert score_on.information_value_call_count > 0


@pytest.mark.asyncio
async def test_score_result_is_an_action_score_with_full_breakdown():
    vf = ValueFunction(StubLLMClient())
    score = await vf.score(_action(), [_hyp()], {})
    assert isinstance(score, ActionScore)
    # All six terms are present as attributes on the model even when zero.
    for field in (
        "learning_value",
        "information_value",
        "long_term_value",
        "time_cost",
        "cognitive_cost",
        "frustration_risk",
        "total",
        "information_value_call_count",
        "flags",
    ):
        assert hasattr(score, field), field


# --- negative information gain ----------------------------------------


def _negative_gain_llm(hyp: Hypothesis) -> StubLLMClient:
    """A fixture whose simulated response *raises* entropy.

    The hypothesis starts near-certain (p=0.99, H≈0.081 bits) and the one
    simulated response drags it back to maximum uncertainty (p=0.5,
    H=1.0 bit), so expected gain ≈ -0.919 bits.
    """
    return StubLLMClient(
        canned={
            "SCORE:LEARNING_VALUE": "0.5",
            "SCORE:COGNITIVE_COST": "0.2",
            "SCORE:FRUSTRATION_RISK": "0.1",
            "SCORE:INFO_RESPONSES": json.dumps(
                [{"response": "student contradicts themselves", "probability": 1.0}]
            ),
            "SCORE:INFO_UPDATE": json.dumps({str(hyp.id): 0.5}),
        }
    )


@pytest.mark.asyncio
async def test_negative_information_value_logs_a_warning_with_entropies(caplog):
    hyp = _hyp(probability=0.99)
    vf = ValueFunction(_negative_gain_llm(hyp))
    action = _action(TeachingAction.QUIZ)

    with caplog.at_level(logging.WARNING, logger="probe.value_function"):
        iv = await vf.information_value(action, [hyp])

    assert iv < 0
    records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "quiz" in message
    assert "entropy_before" in message and "expected_entropy_after" in message
    # H(0.99) ≈ 0.0808, H(0.5) = 1.0 — both must be legible in the log line.
    assert "0.080793" in message
    assert "1.000000" in message


@pytest.mark.asyncio
async def test_negative_information_value_sets_flag_on_action_score():
    hyp = _hyp(probability=0.99)
    vf = ValueFunction(_negative_gain_llm(hyp))
    score = await vf.score(_action(TeachingAction.QUIZ), [hyp], {})

    assert score.information_value < 0
    assert FLAG_NEGATIVE_INFORMATION_VALUE in score.flags
    # The raw value is still summed into total — the flag surfaces it,
    # it doesn't suppress it.
    assert score.total == pytest.approx(
        score.learning_value
        + score.information_value
        + score.long_term_value
        - score.time_cost
        - score.cognitive_cost
        - score.frustration_risk
    )


@pytest.mark.asyncio
async def test_no_flag_when_information_value_is_non_negative():
    vf = ValueFunction(StubLLMClient())  # default INFO_RESPONSES -> "[]"
    score = await vf.score(_action(), [_hyp()], {})
    assert score.information_value == 0.0
    assert score.flags == []


@pytest.mark.asyncio
async def test_no_negative_flag_when_information_value_is_disabled():
    hyp = _hyp(probability=0.99)
    vf = ValueFunction(
        _negative_gain_llm(hyp),
        ValueFunctionConfig(enable_information_value=False),
    )
    score = await vf.score(_action(TeachingAction.QUIZ), [hyp], {})
    assert score.information_value == 0.0
    assert FLAG_NEGATIVE_INFORMATION_VALUE not in score.flags


def test_teaching_action_enum_has_exactly_21_members():
    assert len(TeachingAction) == 21
    expected = {
        "EXPLAIN",
        "ASK",
        "QUIZ",
        "EXAMPLE",
        "COUNTEREXAMPLE",
        "ANALOGY",
        "VISUALIZE",
        "SIMULATE",
        "DERIVE",
        "DECOMPOSE",
        "COMPARE",
        "REPHRASE",
        "CHALLENGE",
        "RECALL",
        "APPLY",
        "TEACH_BACK",
        "CONNECT",
        "CORRECT_MISCONCEPTION",
        "SLOW_DOWN",
        "INCREASE_DIFFICULTY",
        "CHANGE_REPRESENTATION",
    }
    assert {a.name for a in TeachingAction} == expected


def _dummy_uuid_str():
    return str(uuid4())
