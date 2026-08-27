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


class BranchBudgetConfig(BaseModel):
    """Entropy -> depth/breadth mapping for HypothesisGenerator's
    prediction tree. A sibling of ReasoningBudgetConfig on the same
    entropy primitive (_bernoulli_entropy, same active-tier-hypotheses
    scope), not a second parallel budgeting system — see
    compute_branch_budget. Kept as its own function/config rather than
    fields bolted onto ReasoningBudget/compute_reasoning_budget so
    Replan's existing, already-tested contract is untouched.
    """

    # Hard ceiling, all depths combined, one generation. Deliberately
    # tighter than the ~66-branch worst case a *fixed* 3-layer 5x3x3
    # tree would produce, since depth here is unbounded in principle
    # and can compound worse than a fixed fan-out if left uncapped.
    max_total_branches: int = 20
    # Depth 0 (intent) is always generated; this is a second,
    # independent structural backstop against runaway depth,
    # regardless of entropy or max_total_branches.
    max_depth: int = 4
    # Fixed, not entropy-scaled — "always 3-5 intents" is an
    # instruction to the layer-0 prompt, not a budget decision.
    root_branch_range: tuple[int, int] = (3, 5)
    children_per_branch_range: tuple[int, int] = (2, 3)

    # difflib.SequenceMatcher ratio against a sibling's statement at or
    # above which a branch is treated as redundant with it (wording-level
    # near-duplicate, not semantic — see should_expand_branch's docstring
    # for why this stays a heuristic instead of an LLM call).
    redundancy_similarity_threshold: float = 0.8

    # How permissive the "worth expanding" plausibility gate is, scaled
    # by this turn's entropy_bits (same computation as ReasoningBudget):
    # more uncertain about the student -> lower bar -> deeper/wider tree.
    #
    # Tuned (2026-08-27, not yet validated against real session data) so
    # a realistic early session actually reaches non-trivial depth: the
    # original 0.75/8.0 pairing meant a brand-new session (0 active
    # hypotheses) needed the strictest threshold in the entire range,
    # and reaching full permissiveness took 10-18 hypotheses depending
    # on confidence — many turns in, for a mechanism whose whole point
    # is early prediction. 0.55 lets a genuinely plausible branch expand
    # even at turn 0; 3.0 means full permissiveness arrives after
    # roughly 3-5 hypotheses instead of 10+. Revisit both once real
    # generate()/resolve() match data exists to calibrate against.
    min_plausibility_at_zero_entropy: float = 0.55
    min_plausibility_at_max_entropy: float = 0.35
    max_entropy_for_scaling: float = 3.0


class BranchBudget(BaseModel):
    max_total_branches: int
    max_depth: int
    root_branch_range: tuple[int, int]
    children_per_branch_range: tuple[int, int]
    redundancy_similarity_threshold: float = Field(ge=0.0, le=1.0)
    expand_plausibility_threshold: float = Field(ge=0.0, le=1.0)
    entropy_bits: float = Field(ge=0.0)


def compute_branch_budget(
    hypotheses: list[Hypothesis],
    config: BranchBudgetConfig | None = None,
) -> BranchBudget:
    """Entropy over active-tier hypotheses -> HypothesisGenerator's
    per-generation depth/breadth budget. Reuses the exact same
    active-tier-hypotheses -> entropy_bits computation
    compute_reasoning_budget uses, so a turn's branch budget and its
    candidate-width budget are always derived from the same signal.
    """
    cfg = config or BranchBudgetConfig()

    active = [h for h in hypotheses if h.tier is Tier.ACTIVE]
    entropy_bits = sum(_bernoulli_entropy(h.probability) for h in active)

    if cfg.max_entropy_for_scaling > 0:
        t = min(1.0, entropy_bits / cfg.max_entropy_for_scaling)
    else:
        t = 0.0
    threshold = cfg.min_plausibility_at_zero_entropy + t * (
        cfg.min_plausibility_at_max_entropy - cfg.min_plausibility_at_zero_entropy
    )

    return BranchBudget(
        max_total_branches=cfg.max_total_branches,
        max_depth=cfg.max_depth,
        root_branch_range=cfg.root_branch_range,
        children_per_branch_range=cfg.children_per_branch_range,
        redundancy_similarity_threshold=cfg.redundancy_similarity_threshold,
        expand_plausibility_threshold=threshold,
        entropy_bits=entropy_bits,
    )
