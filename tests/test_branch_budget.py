import pytest

from probe.models import Hypothesis, Layer, Tier
from probe.reasoning_budget import BranchBudgetConfig, compute_branch_budget


def _hyp(probability: float, tier: Tier = Tier.ACTIVE) -> Hypothesis:
    return Hypothesis(
        layer=Layer.KNOWLEDGE,
        statement="stub",
        probability=probability,
        confidence=0.5,
        tier=tier,
    )


def test_zero_entropy_gives_the_strict_plausibility_threshold():
    budget = compute_branch_budget([])
    cfg = BranchBudgetConfig()

    assert budget.entropy_bits == 0.0
    assert budget.expand_plausibility_threshold == pytest.approx(
        cfg.min_plausibility_at_zero_entropy
    )


def test_max_entropy_gives_the_permissive_plausibility_threshold():
    # 8 active hypotheses at p=0.5 -> 8.0 bits, at cfg.max_entropy_for_scaling.
    hyps = [_hyp(0.5) for _ in range(8)]
    budget = compute_branch_budget(hyps)
    cfg = BranchBudgetConfig()

    assert budget.entropy_bits == pytest.approx(8.0)
    assert budget.expand_plausibility_threshold == pytest.approx(
        cfg.min_plausibility_at_max_entropy
    )


def test_mid_entropy_interpolates_between_the_two_thresholds():
    # Explicit max_entropy_for_scaling so this test's math doesn't
    # silently drift if the module's default is retuned later (it
    # already was once — see reasoning_budget.py's history). 4 active
    # hypotheses at p=0.5 -> 4.0 bits -> halfway to a scaling reference
    # of 8.0 -> t=0.5 -> threshold halfway between the two extremes.
    cfg = BranchBudgetConfig(max_entropy_for_scaling=8.0)
    hyps = [_hyp(0.5) for _ in range(4)]
    budget = compute_branch_budget(hyps, cfg)

    expected = cfg.min_plausibility_at_zero_entropy + 0.5 * (
        cfg.min_plausibility_at_max_entropy - cfg.min_plausibility_at_zero_entropy
    )
    assert budget.entropy_bits == pytest.approx(4.0)
    assert budget.expand_plausibility_threshold == pytest.approx(expected)


def test_entropy_beyond_the_scaling_reference_still_clamps_to_the_permissive_end():
    hyps = [_hyp(0.5) for _ in range(20)]  # way past any reasonable scaling reference
    budget = compute_branch_budget(hyps)
    cfg = BranchBudgetConfig()

    assert budget.expand_plausibility_threshold == pytest.approx(
        cfg.min_plausibility_at_max_entropy
    )


def test_only_active_tier_hypotheses_contribute_entropy():
    hyps = [_hyp(0.5, tier=Tier.DORMANT), _hyp(0.5, tier=Tier.ARCHIVED)]
    budget = compute_branch_budget(hyps)

    assert budget.entropy_bits == 0.0


def test_ceiling_and_depth_and_ranges_pass_through_from_config():
    cfg = BranchBudgetConfig(
        max_total_branches=7,
        max_depth=2,
        root_branch_range=(2, 4),
        children_per_branch_range=(1, 2),
        redundancy_similarity_threshold=0.9,
    )
    budget = compute_branch_budget([], cfg)

    assert budget.max_total_branches == 7
    assert budget.max_depth == 2
    assert budget.root_branch_range == (2, 4)
    assert budget.children_per_branch_range == (1, 2)
    assert budget.redundancy_similarity_threshold == 0.9
