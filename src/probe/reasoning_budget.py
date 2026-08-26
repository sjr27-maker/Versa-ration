"""ReasoningBudget: scales generation width and reasoning depth with
hypothesis-distribution entropy.

Single source of truth for the entropy -> behavior mapping used by
`Replan` (this turn's `generation_width`/`run_information_value`
threaded into `Infer`/`Plan`/`ValueFunction` by `SessionLoop`) and by
`Plan` (the exploration-slot instruction). One config block
(`ReasoningBudgetConfig`), same pattern as `DEFAULT_GENERATION_WIDTH` —
no scattered thresholds elsewhere.

Entropy scope: combined across ALL active-tier hypotheses, every layer
together — not per-layer. Two reasons: (1) this is exactly what
`Replan` already computed before this module existed
(`sum(_bernoulli_entropy(h.probability) for h in active)`), so keeping
it combined is a behavior-preserving refactor into one named,
configurable place rather than a silent change to what Replan already
does; (2) treating each hypothesis as an independent Bernoulli belief,
total system uncertainty is the sum of per-hypothesis entropies — there
is no single principled way to collapse five separate per-layer
entropy numbers into one width decision without inventing an extra
aggregation rule the source docs don't specify either.

All threshold/scaling values below are placeholder heuristics, not
measured or reasoned from real interaction data (same honesty as
`DIAGNOSE_MISMATCH_THRESHOLD` and `_TIME_COST`) — but `generation_width`
defaults reproduce Replan's exact prior formula (`1 + round(entropy)`,
clamped to [1, 8]), so nothing about existing behavior changes by
default; only the exploration-slot floor and the information_value gate
are new.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

from probe.models import Hypothesis, Tier


class ReasoningBudgetConfig(BaseModel):
    # generation_width = clamp(min_width, max_width,
    #     base_width + round(width_per_bit * entropy_bits))
    # then floored again to min_exploration_slots + 1 (see below).
    # Defaults exactly reproduce Replan's original formula.
    min_width: int = 1
    max_width: int = 8
    base_width: int = 1
    width_per_bit: float = 1.0

    # Below this total entropy (bits), run_information_value is False —
    # the expensive multi-call information_value term is skipped and
    # treated as 0 for the turn. At or above it, True.
    information_value_entropy_threshold: float = 1.0

    # generation_width must always leave room for at least this many
    # exploration candidates PLUS at least one candidate targeting the
    # dominant hypothesis — i.e. effective minimum width is
    # min_exploration_slots + 1, regardless of how low entropy is or
    # what min_width says. This is the "don't only narrow onto the top
    # hypothesis" requirement — a named, visible floor, not an
    # incidental effect of rounding.
    min_exploration_slots: int = 1


class ReasoningBudget(BaseModel):
    generation_width: int
    run_information_value: bool
    entropy_bits: float = Field(ge=0.0)
    # The hypothesis Plan should dedicate one candidate to explicitly
    # targeting, instead of the dominant one. None when no dormant or
    # background hypothesis exists to explore — see compute_budget's
    # docstring: this is deliberately never fabricated. Plan is
    # responsible for logging when it's None, not silently proceeding
    # as if exploration happened.
    exploration_target: Hypothesis | None = None


def _bernoulli_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def compute_reasoning_budget(
    hypotheses: list[Hypothesis],
    config: ReasoningBudgetConfig | None = None,
) -> ReasoningBudget:
    """Entropy over active-tier hypotheses -> width/depth/exploration.

    exploration_target is chosen only from DORMANT/BACKGROUND tier
    hypotheses (the lowest-probability one among them) — never a
    fallback to a low-probability ACTIVE hypothesis. An active
    hypothesis, however uncertain, is still something currently being
    tracked; it isn't neglected the way a dormant/background one is, so
    substituting one in would be pretending exploration happened when
    it didn't. If no dormant/background hypothesis exists,
    exploration_target is None — not fabricated.
    """
    cfg = config or ReasoningBudgetConfig()

    active = [h for h in hypotheses if h.tier is Tier.ACTIVE]
    entropy_bits = sum(_bernoulli_entropy(h.probability) for h in active)

    raw_width = cfg.base_width + round(cfg.width_per_bit * entropy_bits)
    width = max(cfg.min_width, min(cfg.max_width, raw_width))
    width = max(width, cfg.min_exploration_slots + 1)

    run_information_value = entropy_bits >= cfg.information_value_entropy_threshold

    exploration_pool = [
        h for h in hypotheses if h.tier in (Tier.DORMANT, Tier.BACKGROUND)
    ]
    exploration_target = (
        min(exploration_pool, key=lambda h: h.probability)
        if exploration_pool
        else None
    )

    return ReasoningBudget(
        generation_width=width,
        run_information_value=run_information_value,
        entropy_bits=entropy_bits,
        exploration_target=exploration_target,
    )
