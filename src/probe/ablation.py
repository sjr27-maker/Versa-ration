"""SessionMode — the two architectures a probe session can run under,
now that the full Diagnose/Infer/Update/Replan/Plan reasoning path and
the original tree-based branch system have been removed.

- `MINIMAL_BRANCH` (the default): the disambiguation flow (see
  disambiguate.py) — AssessAndBranch -> [DisambiguationOptions] ->
  FinalAnswer, at most three LLM calls per exchange, plus the memory
  layer (memory.py).
- `BASELINE`: plain LLM, one call per turn, no reasoning scaffolding
  at all (SessionLoop._handle_bypass_turn / baseline.BaselineTeach).
  The floor `MINIMAL_BRANCH` is measured against.

`AblationConfig` is kept as the persisted, per-session shape (stored in
`sessions.ablation_config`, read by `TranscriptStore.get_ablation_config`)
so the Compare page and `TurnDiagnosticsStore.mean_cost_by_config` can
still group turns by which mode ran. It now carries exactly one field.
A NULL `sessions.ablation_config` still reads back as the default —
`AblationConfig()` — which is now MINIMAL_BRANCH rather than the old
full system.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class SessionMode(str, Enum):
    MINIMAL_BRANCH = "minimal_branch"
    BASELINE = "baseline"


class AblationConfig(BaseModel):
    mode: SessionMode = SessionMode.MINIMAL_BRANCH

    @property
    def is_full_bypass(self) -> bool:
        """Kept under its old name so loop.py's turn-dispatch reads the
        same: True exactly when this session runs the plain-LLM
        BASELINE."""
        return self.mode is SessionMode.BASELINE


class AblationCostSummary(BaseModel):
    """One row of `TurnDiagnosticsStore.mean_cost_by_config()` — mean
    per-turn wall-clock, call count, and retry count across every turn
    recorded under an identical `AblationConfig` (i.e. the same
    `SessionMode`), regardless of which session it happened in.
    """

    ablation_config: AblationConfig
    turn_count: int
    mean_duration_ms: float
    mean_call_count: float
    mean_retry_count: float
