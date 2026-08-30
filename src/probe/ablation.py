"""AblationConfig — the single switchboard for every subsystem this
codebase can independently turn off, plus the presets that make
picking a comparison point one selection instead of six-to-twelve
toggles.

Why this exists: every layer (portrait, concept grounding, diagnosis,
planning, branch generation, options) has been always-on since it was
built, so nothing has ever been measured against its absence. This
module is the harness that makes "which parts of it are doing
anything" answerable — see loop.py's module docstring for how each
flag is actually wired into a turn, and `AblationConfig.is_full_bypass`
for the true plain-LLM baseline this is all measured against.

Every flag defaults to True (or "entropy" / full width), so
`AblationConfig()` reproduces the exact full-system behavior every
existing test already exercises — passing no ablation_config at all is
indistinguishable from passing `AblationConfig()`.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

from probe.value_function import ValueFunctionConfig

# Mirrors nodes.DEFAULT_GENERATION_WIDTH. Not imported from nodes.py to
# keep this module free of the reasoning-node import graph — it's only
# used as fixed_generation_width's default, which callers in "fixed"
# mode are expected to set explicitly anyway.
_DEFAULT_GENERATION_WIDTH = 3


class ReasoningBudgetMode(str, Enum):
    # Current behavior: generation_width scales with hypothesis-
    # distribution entropy (see reasoning_budget.py).
    ENTROPY = "entropy"
    # generation_width is pinned to fixed_generation_width every turn —
    # no entropy scaling at all. Isolates "does scaling width with
    # uncertainty help" from every other layer.
    FIXED = "fixed"


class ReasoningMode(str, Enum):
    # The existing architecture this whole module otherwise describes:
    # Diagnose/Infer/Update/Replan/Plan, optionally the branch tree.
    FULL = "full"
    # A wholly separate, minimal architecture (see disambiguate.py):
    # AssessAndBranch -> [GenerateOptions] -> FinalAnswer, at most three
    # LLM calls per exchange. Selecting this bypasses every enable_*
    # flag above entirely (SessionLoop._handle_disambiguation_turn is a
    # completely separate code path, same relationship
    # _handle_bypass_turn has to is_full_bypass) — it is not one more
    # thing those flags can turn on or off, since it replaces the branch
    # tree/SelectBranch/DerivePath/Plan/concept-graph machinery outright
    # rather than ablating pieces of it.
    DISAMBIGUATE = "disambiguate"


class AblationConfig(BaseModel):
    """One toggle per subsystem. See loop.py's module docstring for
    exactly what each flag gates turn by turn.

    `enable_options=True` requires `enable_branches=True` — options are
    clickable evidence requests mapped onto branches (see
    GenerateOptions/CheckEvidence in hypothesis_generator.py); an
    option with no branch to map to is meaningless, so this dependency
    is validated at construction, not left to fail confusingly later.
    """

    # Hypotheses read and written (Infer, Update, HypothesisStore).
    enable_portrait: bool = True
    # AttachTopic, concept-graph seeding, GroundConcept, LearnerOverlay.
    enable_concept_graph: bool = True
    # MismatchDetector + world-model revision proposals inside Diagnose.
    enable_diagnose: bool = True
    # Full Plan: proposer + six-term value scoring.
    enable_planner: bool = True
    # Reuse ValueFunctionConfig's own per-term flags verbatim rather
    # than duplicating them — only consulted when enable_planner=True.
    value_function: ValueFunctionConfig = Field(default_factory=ValueFunctionConfig)
    # BranchGenerate, SelectBranch, DerivePath, BranchResolve.
    enable_branches: bool = True
    # GenerateOptions + CheckEvidence (the evidence-extraction channel).
    enable_options: bool = True
    # The mandatory dormant/background exploration slot Replan/Plan
    # otherwise always reserve one candidate for.
    enable_exploration_slot: bool = True
    reasoning_budget_mode: ReasoningBudgetMode = ReasoningBudgetMode.ENTROPY
    # Only consulted when reasoning_budget_mode == FIXED.
    fixed_generation_width: int = _DEFAULT_GENERATION_WIDTH
    # See ReasoningMode.DISAMBIGUATE's docstring: selecting it makes
    # every enable_* flag above irrelevant for this session, the same
    # way is_full_bypass makes them irrelevant, but via a distinct code
    # path rather than by driving every flag to False.
    reasoning_mode: ReasoningMode = ReasoningMode.FULL

    @model_validator(mode="after")
    def _options_require_branches(self) -> AblationConfig:
        if self.enable_options and not self.enable_branches:
            raise ValueError(
                "enable_options=True requires enable_branches=True -- "
                "options are clickable evidence requests mapped onto "
                "branches and cannot exist without them"
            )
        return self

    @property
    def is_full_bypass(self) -> bool:
        """True exactly when every major subsystem is off -- the true
        plain-LLM baseline (see loop.py's _handle_bypass_turn). This is
        a derived condition, not a separate flag: setting every
        enable_* to False by hand has the identical effect as selecting
        the BASELINE preset, on purpose -- there is exactly one way to
        mean "nothing but the raw model," not two that could drift
        apart."""
        return not (
            self.enable_portrait
            or self.enable_concept_graph
            or self.enable_diagnose
            or self.enable_planner
            or self.enable_branches
            or self.enable_options
        )


class AblationPreset(str, Enum):
    BASELINE = "baseline"
    PORTRAIT = "portrait"
    GRAPH = "graph"
    PLANNER = "planner"
    BRANCHES = "branches"
    OPTIONS = "options"


# Each preset adds to the previous -- see module docstring / the task
# this harness was built for. enable_diagnose turns on together with
# enable_concept_graph (GRAPH) rather than getting its own rung: Diagnose's
# MismatchDetector has nothing to ground against without a concept graph,
# so splitting them into separate presets would create a "diagnose without
# a graph" starting point that's meaningless to compare against anything.
# enable_exploration_slot and reasoning_budget_mode aren't part of this
# ladder -- every preset leaves them at their current-behavior default
# (True / "entropy"), since they're secondary knobs, not pipeline stages.
_PRESET_FLAGS: dict[AblationPreset, dict[str, bool]] = {
    AblationPreset.BASELINE: {
        "enable_portrait": False,
        "enable_concept_graph": False,
        "enable_diagnose": False,
        "enable_planner": False,
        "enable_branches": False,
        "enable_options": False,
    },
    AblationPreset.PORTRAIT: {
        "enable_portrait": True,
        "enable_concept_graph": False,
        "enable_diagnose": False,
        "enable_planner": False,
        "enable_branches": False,
        "enable_options": False,
    },
    AblationPreset.GRAPH: {
        "enable_portrait": True,
        "enable_concept_graph": True,
        "enable_diagnose": True,
        "enable_planner": False,
        "enable_branches": False,
        "enable_options": False,
    },
    AblationPreset.PLANNER: {
        "enable_portrait": True,
        "enable_concept_graph": True,
        "enable_diagnose": True,
        "enable_planner": True,
        "enable_branches": False,
        "enable_options": False,
    },
    AblationPreset.BRANCHES: {
        "enable_portrait": True,
        "enable_concept_graph": True,
        "enable_diagnose": True,
        "enable_planner": True,
        "enable_branches": True,
        "enable_options": False,
    },
    AblationPreset.OPTIONS: {
        "enable_portrait": True,
        "enable_concept_graph": True,
        "enable_diagnose": True,
        "enable_planner": True,
        "enable_branches": True,
        "enable_options": True,
    },
}


def build_preset(preset: AblationPreset, **overrides: object) -> AblationConfig:
    """A preset is a starting point, not a locked-in choice: every
    individual toggle remains editable afterward via `overrides`
    (e.g. `build_preset(AblationPreset.GRAPH, enable_diagnose=False)`).
    """
    flags = dict(_PRESET_FLAGS[preset])
    flags.update(overrides)
    return AblationConfig(**flags)


class AblationCostSummary(BaseModel):
    """One row of TurnDiagnosticsStore.mean_cost_by_config() -- the
    number that answers "what does this config cost": mean per-turn
    wall-clock, call count, and retry count across every turn recorded
    under an identical AblationConfig, regardless of which session it
    happened in. Grouped by the full serialized config (not just a
    preset name), since two sessions with hand-edited toggles that
    happen to match are exactly as comparable as two sessions that
    both picked the same preset.

    Lives here rather than in models.py: models.py would need to
    import AblationConfig, and AblationConfig already imports
    ValueFunctionConfig from value_function.py, which itself imports
    from models.py -- keeping this model here avoids that cycle.
    """

    ablation_config: AblationConfig
    turn_count: int
    mean_duration_ms: float
    mean_call_count: float
    mean_retry_count: float
