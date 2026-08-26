import pytest

from probe.models import Hypothesis, Layer, Tier
from probe.reasoning_budget import (
    ReasoningBudgetConfig,
    compute_reasoning_budget,
)


def _hyp(
    probability: float,
    tier: Tier = Tier.ACTIVE,
    layer: Layer = Layer.KNOWLEDGE,
    statement: str = "stub hypothesis",
) -> Hypothesis:
    return Hypothesis(
        layer=layer,
        statement=statement,
        probability=probability,
        confidence=0.5,
        tier=tier,
    )


def test_near_zero_entropy_still_reserves_an_exploration_slot():
    # One active hypothesis, nearly certain -> entropy ~= 0.081 bits,
    # which alone would round the raw formula down to base_width (1).
    budget = compute_reasoning_budget([_hyp(0.99)])

    cfg = ReasoningBudgetConfig()
    assert budget.generation_width >= cfg.min_exploration_slots + 1
    assert budget.generation_width == cfg.min_exploration_slots + 1
    assert budget.run_information_value is False


def test_near_max_entropy_produces_wide_width_and_runs_information_value():
    # 8 active hypotheses at p=0.5 (max per-hypothesis entropy) -> 8.0
    # bits combined, comfortably above both the info_value threshold
    # and enough to clamp generation_width at max_width.
    hyps = [_hyp(0.5) for _ in range(8)]
    budget = compute_reasoning_budget(hyps)

    cfg = ReasoningBudgetConfig()
    assert budget.entropy_bits == pytest.approx(8.0)
    assert budget.generation_width == cfg.max_width
    assert budget.run_information_value is True


def test_mid_entropy_width_matches_the_config_derived_value():
    # 3 active hypotheses at p=0.5 -> 3.0 bits combined.
    hyps = [_hyp(0.5) for _ in range(3)]
    budget = compute_reasoning_budget(hyps)

    cfg = ReasoningBudgetConfig()
    expected_raw = cfg.base_width + round(cfg.width_per_bit * 3.0)
    expected = max(
        cfg.min_width, min(cfg.max_width, expected_raw)
    )
    expected = max(expected, cfg.min_exploration_slots + 1)

    assert budget.entropy_bits == pytest.approx(3.0)
    assert budget.generation_width == expected
    assert expected == 4  # pinned: catches an accidental config change too
    assert budget.run_information_value is (3.0 >= cfg.information_value_entropy_threshold)


def test_no_dormant_or_background_hypothesis_leaves_exploration_target_none():
    # Every hypothesis is active-tier — nothing under-examined to point
    # an exploration candidate at. Must not fabricate a target (e.g. by
    # falling back to the lowest-probability ACTIVE one) and must not
    # crash.
    hyps = [_hyp(0.9), _hyp(0.5), _hyp(0.1)]
    budget = compute_reasoning_budget(hyps)

    assert budget.exploration_target is None
    # The width floor still applies even with nothing to point at —
    # the reservation is structural, independent of whether a target
    # was found this turn.
    cfg = ReasoningBudgetConfig()
    assert budget.generation_width >= cfg.min_exploration_slots + 1


def test_exploration_target_picks_lowest_probability_dormant_or_background():
    active = _hyp(0.6, tier=Tier.ACTIVE, statement="active one")
    background_higher = _hyp(0.4, tier=Tier.BACKGROUND, statement="background higher")
    dormant_lowest = _hyp(0.05, tier=Tier.DORMANT, statement="dormant lowest")
    background_lower = _hyp(0.2, tier=Tier.BACKGROUND, statement="background lower")

    budget = compute_reasoning_budget(
        [active, background_higher, dormant_lowest, background_lower]
    )

    assert budget.exploration_target is not None
    assert budget.exploration_target.id == dormant_lowest.id


def test_exploration_target_never_falls_back_to_active_tier():
    # Only active hypotheses exist, including a very-low-probability
    # one. Per the chosen design, this must NOT be picked as a
    # fallback exploration target — an active hypothesis, however
    # uncertain, is still being tracked, not neglected.
    low_prob_active = _hyp(0.02, tier=Tier.ACTIVE, statement="low prob but active")
    budget = compute_reasoning_budget([_hyp(0.9), low_prob_active])

    assert budget.exploration_target is None


def test_empty_hypothesis_list_is_zero_entropy_but_still_floors_width():
    budget = compute_reasoning_budget([])
    cfg = ReasoningBudgetConfig()

    assert budget.entropy_bits == 0.0
    assert budget.generation_width == cfg.min_exploration_slots + 1
    assert budget.run_information_value is False
    assert budget.exploration_target is None


def test_generation_width_never_exceeds_max_width_even_at_extreme_entropy():
    hyps = [_hyp(0.5) for _ in range(50)]
    budget = compute_reasoning_budget(hyps)
    cfg = ReasoningBudgetConfig()
    assert budget.generation_width == cfg.max_width


def test_custom_config_is_actually_used_not_ignored():
    cfg = ReasoningBudgetConfig(
        min_width=1,
        max_width=3,
        base_width=1,
        width_per_bit=1.0,
        information_value_entropy_threshold=10.0,
        min_exploration_slots=2,
    )
    hyps = [_hyp(0.5) for _ in range(8)]  # 8.0 bits, way above threshold normally
    budget = compute_reasoning_budget(hyps, cfg)

    assert budget.generation_width == 3  # clamped by custom max_width
    assert budget.run_information_value is False  # below the custom (higher) threshold
