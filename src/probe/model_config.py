"""Tier-to-model mapping for the real (Gemini) LLM client.

Three tiers exist purely as a cost/quality knob over which concrete
Gemini model answers a node's calls — not something nodes know or care
about. Every node still just calls `llm.complete(prompt)`; which tier's
client it was constructed with is decided once, in SessionLoop, never
per-call.

Tier -> node assignment (fixed by agreement, not derived from anything
in this file):
    fast:     AssessAndBranch, DisambiguationOptions, the memory-layer
              judgment nodes (ConfirmFactMatch, WriteLearnerFact,
              SummarizeSessionPath, ConfirmThinkingStyleMatch)
    capable:  (unused since the full reasoning path was removed)
    best:     FinalAnswer, BaselineTeach

Model ids below are pinned to flash-class models across all three
tiers as of 2026-08, verified live against a real key (see the
scratch probe that produced this commit): `gemini-3.1-pro-preview`
(the previous `capable`/`best` pin) is blocked on this key by a
free-tier quota of 0 for Pro-class models — not a stale id, an
entitlement gap. `capable` and `best` are kept as separate fields
rather than collapsed to one flash id specifically so upgrading either
back to a Pro model later, once one is confirmed available on this
key, is a `GEMINI_MODEL_CAPABLE`/`GEMINI_MODEL_BEST` env override, not
a code change. Preview/dated model ids are still the least stable
constant in this codebase — re-verify against
https://ai.google.dev/gemini-api/docs/models (or the same live-probe
approach) before trusting any of this without checking first.
"""

from __future__ import annotations

import os

from pydantic import BaseModel


class ModelTierConfig(BaseModel):
    fast: str = "gemini-3.6-flash"
    capable: str = "gemini-3.5-flash"
    best: str = "gemini-3.5-flash"
    # The memory layer's embedding model (probe.embeddings) — verified
    # live against this key: `text-embedding-004`/`models/embedding-001`
    # both 404 (not available on the v1beta API this SDK version
    # targets), `gemini-embedding-001` works. Not one of the three
    # reasoning tiers above (it never generates text, only vectors), so
    # kept as its own field rather than folded into fast/capable/best.
    embedding: str = "gemini-embedding-001"

    @classmethod
    def from_env(cls) -> ModelTierConfig:
        """Env vars override the defaults without a code change — same
        escape hatch as every other config block in this codebase, here
        specifically because preview model ids drift on their own
        schedule, independent of when this file was last edited."""
        defaults = cls()
        return cls(
            fast=os.getenv("GEMINI_MODEL_FAST", defaults.fast),
            capable=os.getenv("GEMINI_MODEL_CAPABLE", defaults.capable),
            best=os.getenv("GEMINI_MODEL_BEST", defaults.best),
            embedding=os.getenv("GEMINI_MODEL_EMBEDDING", defaults.embedding),
        )
