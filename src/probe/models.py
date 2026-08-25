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


class TeachingAction(str, Enum):
    """The full teaching-action space. Exactly 21 members.

    Ordering here doubles as Plan's default candidate-generation order
    when it's asked for N < 21. Nothing else in the codebase should
    depend on this ordering — QUIZ (position 3) is not special.
    """

    EXPLAIN = "explain"
    ASK = "ask"
    QUIZ = "quiz"
    EXAMPLE = "example"
    COUNTEREXAMPLE = "counterexample"
    ANALOGY = "analogy"
    VISUALIZE = "visualize"
    SIMULATE = "simulate"
    DERIVE = "derive"
    DECOMPOSE = "decompose"
    COMPARE = "compare"
    REPHRASE = "rephrase"
    CHALLENGE = "challenge"
    RECALL = "recall"
    APPLY = "apply"
    TEACH_BACK = "teach_back"
    CONNECT = "connect"
    CORRECT_MISCONCEPTION = "correct_misconception"
    SLOW_DOWN = "slow_down"
    INCREASE_DIFFICULTY = "increase_difficulty"
    CHANGE_REPRESENTATION = "change_representation"


class CandidateAction(BaseModel):
    """A candidate action the loop could take next."""

    id: UUID = Field(default_factory=uuid4)
    action: TeachingAction
    target_concept: str | None = None
    rationale: str = ""


class ActionScore(BaseModel):
    """Full six-term value-function breakdown for one candidate action.

    Every ablation-relevant term stays visible even when disabled (its
    value will be 0.0). `total` is the raw sum used for ranking; the
    per-term fields are what the ablation dashboard reads. Do not
    collapse this to a single float — CLAUDE.md invariant 3.

    `flags` names anomalies detected while scoring (see
    `probe.value_function` for the flag constants), so `node_calls` can
    be filtered on them without re-deriving the condition from raw
    floats.
    """

    candidate: CandidateAction
    learning_value: float
    information_value: float
    long_term_value: float
    time_cost: float
    cognitive_cost: float
    frustration_risk: float
    total: float
    information_value_call_count: int = 0
    flags: list[str] = Field(default_factory=list)


class PlanOutput(BaseModel):
    """Plan's return value.

    Carries both the winning action (for Teach to render) and the full
    per-candidate score breakdown (for the audit trail). Both survive
    together through `SessionLoop._call_node`'s serialization into
    `node_calls.output_json`.
    """

    winner: CandidateAction
    scores: list[ActionScore]


class ConceptNode(BaseModel):
    """A node in the world concept graph — not learner-specific.

    Seeded once via `probe seed-graph` and frozen afterward. `prerequisites`
    holds concept ids only; `ConceptGraph` resolves them against the
    `concept_prerequisites` edge table, it does not embed nested nodes.
    """

    id: str
    name: str
    prerequisites: list[str] = Field(default_factory=list)
    common_misconceptions: list[str] = Field(default_factory=list)
    representations: list[str] = Field(default_factory=list)
    diagnostic_questions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class OverlayState(str, Enum):
    KNOWN = "known"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    BLOCKED = "blocked"


class OverlayEntry(BaseModel):
    """A learner's current state on one concept.

    Unlike `Hypothesis`, this is current-state tracking, not a claim
    with an evidence trail — `LearnerOverlay.set_state()` upserts. Do
    not route this through `HypothesisStore`.
    """

    concept_id: str
    state: OverlayState
    confidence: float = Field(ge=0.0, le=1.0)
    updated_at: datetime = Field(default_factory=_utcnow)


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
