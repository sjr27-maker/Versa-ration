from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Layer(str, Enum):
    GOAL = "goal"
    KNOWLEDGE = "knowledge"
    MENTAL_MODEL = "mental_model"
    COGNITIVE_STATE = "cognitive_state"
    TEACHING = "teaching"


class Tier(str, Enum):
    ACTIVE = "active"
    BACKGROUND = "background"
    DORMANT = "dormant"
    ARCHIVED = "archived"


class Polarity(str, Enum):
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"


class EvidenceRef(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    turn_id: UUID
    polarity: Polarity
    timestamp: datetime = Field(default_factory=_utcnow)


class Hypothesis(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    layer: Layer
    statement: str
    probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    tier: Tier
    conditions: list[str] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    counter_evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @field_validator("evidence_refs")
    @classmethod
    def _evidence_refs_must_be_supporting(
        cls, refs: list[EvidenceRef]
    ) -> list[EvidenceRef]:
        for ref in refs:
            if ref.polarity is not Polarity.SUPPORTING:
                raise ValueError(
                    f"evidence_refs must contain only SUPPORTING refs; "
                    f"ref {ref.id} has polarity={ref.polarity.value}"
                )
        return refs

    @field_validator("counter_evidence_refs")
    @classmethod
    def _counter_evidence_refs_must_be_contradicting(
        cls, refs: list[EvidenceRef]
    ) -> list[EvidenceRef]:
        for ref in refs:
            if ref.polarity is not Polarity.CONTRADICTING:
                raise ValueError(
                    f"counter_evidence_refs must contain only CONTRADICTING "
                    f"refs; ref {ref.id} has polarity={ref.polarity.value}"
                )
        return refs


class CandidateAction(BaseModel):
    """A candidate action the loop could take next.

    `kind` differentiates action types (currently only 'teach'). `payload`
    stays loose while the action space is still evolving — the real
    action ontology lands in Step 3.
    """

    id: UUID = Field(default_factory=uuid4)
    kind: str
    payload: dict = Field(default_factory=dict)


class ProposedEvidence(BaseModel):
    """The Infer→Update handoff.

    Infer decides which existing hypothesis a turn bears on, what new
    probability/confidence to assign, and provides the underlying
    EvidenceRef. Update consumes these to call
    `HypothesisStore.reweight(...)`.
    """

    hypothesis_id: UUID
    new_probability: float = Field(ge=0.0, le=1.0)
    new_confidence: float = Field(ge=0.0, le=1.0)
    evidence_ref: EvidenceRef
