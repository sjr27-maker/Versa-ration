"""AblationConfig: validation, presets, and is_full_bypass -- no DB,
no LLM, pure construction/logic tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from probe.ablation import (
    AblationConfig,
    AblationPreset,
    ReasoningBudgetMode,
    build_preset,
)


def test_default_config_is_full_system_and_not_a_bypass():
    config = AblationConfig()
    assert config.enable_portrait is True
    assert config.enable_concept_graph is True
    assert config.enable_diagnose is True
    assert config.enable_planner is True
    assert config.enable_branches is True
    assert config.enable_options is True
    assert config.enable_exploration_slot is True
    assert config.reasoning_budget_mode is ReasoningBudgetMode.ENTROPY
    assert config.is_full_bypass is False


def test_enable_options_without_enable_branches_rejected_at_construction():
    with pytest.raises(ValidationError, match="enable_options=True requires enable_branches"):
        AblationConfig(enable_options=True, enable_branches=False)


def test_enable_options_without_enable_branches_rejected_via_preset_override():
    with pytest.raises(ValidationError):
        build_preset(AblationPreset.BASELINE, enable_options=True)


@pytest.mark.parametrize(
    "flag",
    [
        "enable_portrait",
        "enable_concept_graph",
        "enable_diagnose",
        "enable_planner",
        "enable_branches",
    ],
)
def test_disabling_any_single_non_options_flag_is_not_a_full_bypass(flag):
    # enable_branches=False must also carry enable_options=False -- the
    # validator correctly rejects the reverse (options with no branches
    # to map onto), which is a separate thing being tested elsewhere.
    overrides = {flag: False}
    if flag == "enable_branches":
        overrides["enable_options"] = False
    config = AblationConfig(**overrides)
    assert config.is_full_bypass is False


def test_baseline_preset_is_a_full_bypass():
    config = build_preset(AblationPreset.BASELINE)
    assert config.is_full_bypass is True


def test_every_enable_flag_false_by_hand_is_also_a_full_bypass():
    """is_full_bypass is derived, not a separate flag -- hand-setting
    every toggle off must be indistinguishable from picking BASELINE."""
    config = AblationConfig(
        enable_portrait=False,
        enable_concept_graph=False,
        enable_diagnose=False,
        enable_planner=False,
        enable_branches=False,
        enable_options=False,
    )
    assert config.is_full_bypass is True
    assert config == build_preset(AblationPreset.BASELINE)


@pytest.mark.parametrize(
    "preset,expected_true,expected_false",
    [
        (AblationPreset.BASELINE, [], [
            "enable_portrait", "enable_concept_graph", "enable_diagnose",
            "enable_planner", "enable_branches", "enable_options",
        ]),
        (AblationPreset.PORTRAIT, ["enable_portrait"], [
            "enable_concept_graph", "enable_diagnose", "enable_planner",
            "enable_branches", "enable_options",
        ]),
        (AblationPreset.GRAPH, [
            "enable_portrait", "enable_concept_graph", "enable_diagnose",
        ], ["enable_planner", "enable_branches", "enable_options"]),
        (AblationPreset.PLANNER, [
            "enable_portrait", "enable_concept_graph", "enable_diagnose",
            "enable_planner",
        ], ["enable_branches", "enable_options"]),
        (AblationPreset.BRANCHES, [
            "enable_portrait", "enable_concept_graph", "enable_diagnose",
            "enable_planner", "enable_branches",
        ], ["enable_options"]),
    ],
)
def test_preset_flags(preset, expected_true, expected_false):
    config = build_preset(preset)
    for flag in expected_true:
        assert getattr(config, flag) is True, f"{preset}: expected {flag}=True"
    for flag in expected_false:
        assert getattr(config, flag) is False, f"{preset}: expected {flag}=False"


def test_options_preset_is_the_full_system_default():
    assert build_preset(AblationPreset.OPTIONS) == AblationConfig()


def test_preset_is_a_starting_point_overrides_still_apply():
    config = build_preset(AblationPreset.GRAPH, enable_diagnose=False)
    assert config.enable_concept_graph is True
    assert config.enable_diagnose is False


def test_config_round_trips_through_json():
    config = AblationConfig(enable_branches=False, enable_options=False)
    dumped = config.model_dump(mode="json")
    restored = AblationConfig(**dumped)
    assert restored == config
