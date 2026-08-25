"""Read-only learner portrait report — `probe portrait <learner_id>`.

No writes here. This reads across everything that already exists for a
learner (hypotheses, overlay, pending revisions, session count) and
assembles a snapshot. Nothing here is persisted; it's generated fresh
on every call.

Hypotheses and world-model revisions have no direct learner_id column
(see `HypothesisStore.list_by_learner` / `WorldModelRevisionStore.
list_by_learner`), so both are attributed to a learner via their
evidence_refs' turn -> session -> learner_id chain. Anything with zero
evidence isn't attributable and won't appear here.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel

from probe.audit import TranscriptStore
from probe.concept_graph import ConceptGraph
from probe.models import (
    Hypothesis,
    Layer,
    OverlayEntry,
    RevisionStatus,
    Tier,
    WorldModelRevision,
)
from probe.overlay import LearnerOverlay
from probe.revision import WorldModelRevisionStore
from probe.store import HypothesisStore

_ALL_LAYERS = list(Layer)
_ALL_TIERS = list(Tier)


class TopHypothesis(BaseModel):
    layer: Layer
    hypothesis: Hypothesis | None = None


class OverlaySummaryEntry(BaseModel):
    concept_id: str
    concept_name: str | None
    entry: OverlayEntry


class LearnerPortrait(BaseModel):
    learner_id: UUID
    session_count: int
    top_hypotheses: list[TopHypothesis]
    tier_counts: dict[str, int]
    overlay: list[OverlaySummaryEntry]
    pending_revisions: list[WorldModelRevision]


async def build_portrait(
    learner_id: UUID,
    hypothesis_store: HypothesisStore,
    transcript: TranscriptStore,
    concept_graph: ConceptGraph,
    learner_overlay: LearnerOverlay,
    revision_store: WorldModelRevisionStore,
) -> LearnerPortrait:
    session_count = await transcript.count_sessions_for_learner(learner_id)

    top_hypotheses: list[TopHypothesis] = []
    for layer in _ALL_LAYERS:
        active_for_layer = await hypothesis_store.list_by_learner(
            learner_id, layer=layer, tier=Tier.ACTIVE
        )
        top = max(active_for_layer, key=lambda h: h.probability, default=None)
        top_hypotheses.append(TopHypothesis(layer=layer, hypothesis=top))

    tier_counts: dict[str, int] = {}
    for tier in _ALL_TIERS:
        hyps = await hypothesis_store.list_by_learner(learner_id, tier=tier)
        tier_counts[tier.value] = len(hyps)

    full_overlay = await learner_overlay.get_full_overlay(learner_id)
    overlay_summary: list[OverlaySummaryEntry] = []
    for entry in full_overlay:
        concept = await concept_graph.get_concept(
            entry.concept_graph_id, entry.concept_id
        )
        overlay_summary.append(
            OverlaySummaryEntry(
                concept_id=entry.concept_id,
                concept_name=concept.name if concept is not None else None,
                entry=entry,
            )
        )

    pending_revisions = await revision_store.list_by_learner(
        learner_id, status=RevisionStatus.PENDING
    )

    return LearnerPortrait(
        learner_id=learner_id,
        session_count=session_count,
        top_hypotheses=top_hypotheses,
        tier_counts=tier_counts,
        overlay=overlay_summary,
        pending_revisions=pending_revisions,
    )
