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


class Learner(BaseModel):
    """Identity-level record only.

    Deliberately thin: label/display_name/preferred_register/timezone
    are the only fields, all optional, none behavioral. What a learner
    knows, believes, or is working toward is never tracked here — that
    belongs on `Hypothesis` (evidence-backed claims) or `OverlayEntry`
    (current concept-mastery state). If a future field would overlap
    with either of those, it doesn't belong on this model.
    """

    id: UUID = Field(default_factory=uuid4)
    label: str | None = None
    display_name: str | None = None
    preferred_register: str | None = None
    timezone: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class EvidenceRef(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    turn_id: UUID
    polarity: Polarity
    timestamp: datetime = Field(default_factory=_utcnow)
    # Populated only for evidence a Hypothesis reweight() actually
    # created (None for add()-time evidence, and always None for a
    # WorldModelRevision's own evidence_refs, which aren't about a
    # probability/confidence at all) — the "after" value needed to
    # show a per-turn delta without recomputing it anywhere else.
    resulting_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    resulting_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


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

    `*_call_count` fields cover every LLM-calling term (learning_value,
    information_value, cognitive_cost, frustration_risk — long_term_value
    and time_cost never call the LLM, so they have none) and are what
    `MAX_CALLS_PER_TURN` sums across candidates to get the real per-turn
    total, not an undercount of it.
    """

    candidate: CandidateAction
    learning_value: float
    information_value: float
    long_term_value: float
    time_cost: float
    cognitive_cost: float
    frustration_risk: float
    total: float
    learning_value_call_count: int = 0
    information_value_call_count: int = 0
    cognitive_cost_call_count: int = 0
    frustration_risk_call_count: int = 0
    flags: list[str] = Field(default_factory=list)


class PlanOutput(BaseModel):
    """Plan's return value.

    Carries both the winning action (for Teach to render) and the full
    per-candidate score breakdown (for the audit trail). Both survive
    together through `SessionLoop._call_node`'s serialization into
    `node_calls.output_json`.

    `argmax_changes_without_information_value` is necessarily computed
    here, not on `ActionScore`: it's a property of the *winner
    selection* across all candidates (would a different one win with
    information_value zeroed out), which a single candidate's own score
    has no visibility into.
    """

    winner: CandidateAction
    scores: list[ActionScore]
    argmax_changes_without_information_value: bool = False


class ConceptGraphMeta(BaseModel):
    """Identity of one seeded concept graph — not its nodes.

    `topic` is deliberately not unique: re-seeding the same topic string
    creates a second, independent graph. `concept_id` (on `ConceptNode`)
    is only unique *within* a graph, not globally — the pair
    (concept_graph_id, id) is the real key everywhere a bare concept_id
    used to be enough.
    """

    id: UUID = Field(default_factory=uuid4)
    topic: str
    created_at: datetime = Field(default_factory=_utcnow)


class ConceptNode(BaseModel):
    """A node in one world concept graph — not learner-specific.

    Seeded once via `probe seed-graph` and frozen afterward. `prerequisites`
    holds concept ids only (scoped to this node's own concept_graph_id);
    `ConceptGraph` resolves them against the `concept_prerequisites` edge
    table, it does not embed nested nodes.
    """

    concept_graph_id: UUID
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

    concept_graph_id: UUID
    concept_id: str
    state: OverlayState
    confidence: float = Field(ge=0.0, le=1.0)
    updated_at: datetime = Field(default_factory=_utcnow)


class ConceptGrounding(BaseModel):
    """GroundConcept's judgment of which concept, if any, a student
    response is actually about.

    Kept as a scalar-plus-confidence pair rather than a bare id so
    downstream mismatch detection can weight a low-confidence grounding
    differently from a clear one, and so a rejected hallucinated id
    (validated against the session's actual graph) still leaves a
    confidence value visible for review.
    """

    concept_id: str | None
    confidence: float = Field(ge=0.0, le=1.0)


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


class SuggestedCause(str, Enum):
    """MismatchDetector's judgment of *why* a mismatch exists.

    This is a real LLM call, not a hardcoded default — a detected
    conflict between a learner's apparent belief and the concept graph
    is not automatically the learner's fault. The concept definition or
    its listed misconceptions can themselves be wrong or incomplete.
    """

    LEARNER_MISCONCEPTION = "learner_misconception"
    POSSIBLE_WORLD_MODEL_ERROR = "possible_world_model_error"


class MismatchResult(BaseModel):
    """A detected conflict between a learner's apparent belief and the
    world model (ConceptNode definition / listed misconceptions)."""

    concept_id: str
    learner_claim: str
    world_claim: str
    confidence: float = Field(ge=0.0, le=1.0)
    suggested_cause: SuggestedCause


class RevisionStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class WorldModelRevision(BaseModel):
    """A claim that the concept graph itself may need editing.

    This is never auto-applied to `ConceptGraph` — it's evidence-backed
    the same way a `Hypothesis` is, but about the world model rather
    than the learner, and it stays `pending` until a human reviews it
    (`probe review-revisions`). `applied_field_updates` is set only by
    `WorldModelRevisionStore.approve()`, and only to the exact structured
    edit a human confirmed — never derived automatically from
    `proposed_change`.
    """

    id: UUID = Field(default_factory=uuid4)
    concept_graph_id: UUID
    concept_id: str
    proposed_change: str
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    status: RevisionStatus = RevisionStatus.PENDING
    applied_field_updates: dict[str, object] | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    resolved_at: datetime | None = None


class NodeCall(BaseModel):
    """One row from node_calls — for read paths (the web UI) that need
    a specific past call's input/output rather than just writing new
    ones (NodeCallStore.record is the only writer, per CLAUDE.md
    invariant 2)."""

    id: UUID
    node_name: str
    session_id: UUID
    turn_index: int
    input_json: dict
    output_json: object
    timestamp: datetime


class TurnRecord(BaseModel):
    """One student turn, as persisted by TranscriptStore.record_turn."""

    id: UUID
    session_id: UUID
    turn_index: int
    text: str
    created_at: datetime


class BranchStatus(str, Enum):
    OPEN = "open"
    MATCHED = "matched"
    UNMATCHED = "unmatched"
    SUPERSEDED = "superseded"


class Branch(BaseModel):
    """One node in HypothesisGenerator's speculative prediction tree.

    Distinct from `Hypothesis`: this is regenerated and mostly discarded
    every turn, never written into `HypothesisStore` directly (see
    CLAUDE.md invariant 6). `depth`/`depth_label` replace a fixed
    intent/knowledge_gap/predicted_action enum because depth is
    situational — a branch keeps expanding only while doing so would
    still distinguish it from siblings and the turn's budget allows
    (see `should_expand_branch`). `predicted_next_turn` is populated on
    every branch at creation time, not just eventual leaves, so
    whichever depth a branch stops at, it stays checkable against the
    student's real next message — `statement` is that layer's own
    semantic content (the intent, the knowledge gap), which is not
    itself always concretely checkable the way a prediction has to be.
    """

    id: UUID = Field(default_factory=uuid4)
    parent_id: UUID | None = None
    generation_id: UUID
    session_id: UUID
    turn_index: int
    depth: int = Field(ge=0)
    depth_label: str
    statement: str
    predicted_next_turn: str
    plausibility: float = Field(ge=0.0, le=1.0)
    is_leaf: bool
    status: BranchStatus = BranchStatus.OPEN
    created_at: datetime = Field(default_factory=_utcnow)


class PathRequirement(BaseModel):
    """DerivePath's output — what Teach is scoped to this turn, derived
    from the selected branch's full root-to-leaf path (every ancestor's
    statement, not just the leaf). Persisted with the generation so the
    web UI can show exactly what Teach was told, and told not, to say.

    `must_not_assume` is the load-bearing field: anything the path
    leaves genuinely uncertain (a sign, a value, a condition never
    stated) that Teach must not present as settled — the mechanism
    meant to prevent a Teach output that quietly assumes an unstated
    quantity (e.g. asserting a charge's sign the problem never gave).
    """

    current_belief: str = ""
    needed: str = ""
    must_not_assume: list[str] = Field(default_factory=list)
    scope: str = ""


class BranchSelection(BaseModel):
    """SelectBranch's output — which branch this turn's teaching should
    derive from, and why. Selection criterion is coverage (how much of
    the rest of the live tree this branch's path would also serve), not
    raw plausibility — see hypothesis_generator._select_prompt.
    `selected_branch_id` is None only when there was nothing to select
    from (an empty generation)."""

    selected_branch_id: UUID | None
    rationale: str = ""


class BranchGenerationMeta(BaseModel):
    """Identity of one full generation event — not its branches.

    `selected_branch_id`/`selection_rationale`/`path_requirement` start
    None (a generation is created before selection runs) and are filled
    in afterward via BranchStore.set_selection()/set_path_requirement().
    Selecting one branch is not a commitment — the tree regenerates
    next turn from what this turn revealed, so an unselected branch is
    deferred, never discarded; nothing about tiering/supersession
    changes because of a selection.
    """

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    turn_index: int
    root_count: int
    selected_branch_id: UUID | None = None
    selection_rationale: str | None = None
    path_requirement: PathRequirement | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class BranchGeneration(BaseModel):
    """HypothesisGenerator.generate()'s return value.

    `call_count` is embedded directly (same precedent as
    `ActionScore`/Diagnose's dict) rather than tracked as a
    side-channel instance attribute, so it survives automatically into
    `node_calls.output_json` and SessionLoop's MAX_CALLS_PER_TURN sum
    can read it straight off the return value.
    """

    generation: BranchGenerationMeta
    branches: list[Branch]
    call_count: int = 0
    # Same facts as the logger.info lines emitted per branch that
    # clears the redundancy check (see should_expand_branch) — kept
    # here too, structured, so SessionLoop can fold them into that
    # turn's turn_diagnostics.warnings for the UI to read directly
    # instead of re-parsing logs.
    redundancy_notes: list[str] = Field(default_factory=list)


class ResolutionResult(BaseModel):
    """HypothesisGenerator.resolve()'s return value."""

    session_id: UUID
    turn_index: int
    matched_branch_id: UUID | None
    matched_chain: list[UUID] = Field(default_factory=list)
    status: str  # "matched" | "unmatched"
    call_count: int = 0


class TopicAttachment(BaseModel):
    """AttachTopic's return value: what topic was inferred and which
    concept_graph_id ended up attached to the session, whether that
    graph was resumed (an existing exact-topic match) or freshly
    seeded."""

    topic: str
    concept_graph_id: UUID
    seeded_new: bool


class HypothesisTierChange(BaseModel):
    """One row of hypothesis_tier_changes — the trace retier()/
    resurrect() leave behind, since the hypotheses table itself only
    ever shows the current tier."""

    id: UUID = Field(default_factory=uuid4)
    hypothesis_id: UUID
    old_tier: Tier
    new_tier: Tier
    changed_at: datetime = Field(default_factory=_utcnow)


class TurnDiagnostics(BaseModel):
    """One row per handle_turn() call — the persisted form of what
    loop.py already computes each turn (call counts, the
    MAX_CALLS_PER_TURN guardrail, entropy_bits, warnings), so the web
    UI's Diagnostics panel can read it directly instead of re-deriving
    it. `teach_failed` is checked separately from `warnings` by
    downstream analysis (branch match rate, portrait stats, call-count
    aggregates) that needs to exclude turns where no real teaching
    happened.

    `inferred_topic`/`topic_seeded_new` are AttachTopic's own result —
    only ever set on turn 0 (the only turn it runs), None every other
    turn. Persisted specifically so a wrong topic inference is visible
    immediately in the UI, not something discoverable only by querying
    node_calls directly.

    `retry_count` is this turn's total across every LLMClient retry
    (GeminiLLMClient.retry_count, snapshotted before/after the turn —
    see loop.py's _total_retry_count) — a rate-limited call and a
    genuinely slow one are otherwise indistinguishable from the outside;
    this makes throttling visible in the UI instead of only in logs.
    Always 0 against StubLLMClient (no retry mechanism to count).
    """

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    turn_index: int
    node_call_counts: dict[str, int] = Field(default_factory=dict)
    total_call_count: int = 0
    guardrail_fired: bool = False
    entropy_bits: float | None = None
    duration_ms: float
    warnings: list[str] = Field(default_factory=list)
    teach_failed: bool = False
    inferred_topic: str | None = None
    topic_seeded_new: bool | None = None
    retry_count: int = 0
    created_at: datetime = Field(default_factory=_utcnow)


class SessionSummary(BaseModel):
    """One row for the Setup page's resume view: a learner's prior
    session, its inferred topic (None if never attached), and how many
    turns it has."""

    session_id: UUID
    concept_graph_id: UUID | None
    topic: str | None
    turn_count: int
    created_at: datetime


class BranchMatchRatePoint(BaseModel):
    """One session's leaf-branch match rate — "does it actually predict
    me," plotted session over session. Only leaves that reached a
    terminal status (matched/unmatched) count; a still-open generation
    (the session's most recent, unresolved one) isn't included."""

    session_id: UUID
    session_created_at: datetime
    total_resolved: int
    matched_count: int

    @property
    def match_rate(self) -> float:
        return self.matched_count / self.total_resolved if self.total_resolved else 0.0


class LearnerSummary(BaseModel):
    """One row for the Setup page's existing-learner picker: a learner
    plus their session count and most recent session's timestamp
    (None if they have no sessions yet)."""

    learner: Learner
    session_count: int
    last_session_at: datetime | None


class RecurringIntent(BaseModel):
    """One depth-0 (root/intent) statement, grouped by exact text
    across all of a learner's sessions — how often it recurs and how
    often it ends up matched. Exact-text grouping only: two
    differently-worded but semantically identical intents are counted
    separately, same documented limitation as the redundancy check's
    wording-not-semantics heuristic."""

    statement: str
    total_count: int
    matched_count: int

    @property
    def match_rate(self) -> float:
        return self.matched_count / self.total_count if self.total_count else 0.0
