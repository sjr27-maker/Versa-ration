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
    """The Infer->Update handoff for an EXISTING hypothesis.

    Infer decides which already-tracked hypothesis a turn bears on and
    what new probability/confidence to assign; `evidence_ref` is built
    by Infer itself (it has `turn_id` from the loop, the LLM does not),
    never trusted from the model's own output -- see Infer.run()'s
    docstring. Update consumes these to call
    `HypothesisStore.reweight(...)`.
    """

    hypothesis_id: UUID
    new_probability: float = Field(ge=0.0, le=1.0)
    new_confidence: float = Field(ge=0.0, le=1.0)
    evidence_ref: EvidenceRef


class ProposedHypothesis(BaseModel):
    """The Infer->Update handoff for a hypothesis that does NOT exist
    yet -- the counterpart to `ProposedEvidence` that was missing
    entirely until this fix (see Infer.run()'s docstring for the full
    story: `reweight()`-only meant Infer had no path to mint a new
    belief, so every real session ever run operated with zero
    hypotheses, forever).

    `evidence_ref` is always SUPPORTING (there is nothing yet to
    contradict) and, like `ProposedEvidence.evidence_ref`, is built by
    Infer from `turn_id`, never read from the model's output. Update
    consumes these to call `HypothesisStore.add(...)`.
    """

    layer: Layer
    statement: str
    initial_probability: float = Field(ge=0.0, le=1.0)
    initial_confidence: float = Field(ge=0.0, le=1.0)
    evidence_ref: EvidenceRef


class InferOutput(BaseModel):
    """Infer's full return value: reweights of hypotheses already in
    the store, and creates for beliefs the turn suggests that nothing
    in the store yet covers. Kept as one object (not two separate
    Infer.run() calls or a plain tuple) so `node_calls` records both
    halves of one turn's inference together, same audit-completeness
    reasoning as `PlanOutput` carrying both the winner and every
    candidate's score.
    """

    reweights: list[ProposedEvidence] = Field(default_factory=list)
    creates: list[ProposedHypothesis] = Field(default_factory=list)


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

    `requires_evidence`/`evidence_satisfied` make a branch a claim with
    an entry condition, not just a forecast: null `requires_evidence`
    means the branch needs nothing further and expands on plausibility
    alone as before; a non-null one blocks expansion (see
    `should_expand_branch`'s fourth gate) until evidence_satisfied
    flips true — via a direct option click (unambiguous, no LLM call)
    or a typed message CheckEvidence judges to establish it (still
    interpreted, the one place this mechanism accepts that trade-off).
    Once satisfied, a branch is expandable next turn exactly as if it
    had never required evidence — the flag never resets.
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
    requires_evidence: str | None = None
    evidence_satisfied: bool = False
    plausibility: float = Field(ge=0.0, le=1.0)
    is_leaf: bool
    status: BranchStatus = BranchStatus.OPEN
    # Which channel resolved a MATCHED branch: "option_click" (an
    # unambiguous click, no interpretation) or "text_match"
    # (RESOLVE:MATCH's LLM judgment). None for anything not currently
    # matched. Propagates up the full ancestor chain alongside status
    # (see HypothesisGenerator.resolve()) so aggregate queries can
    # report the two channels as separate numbers — a click-resolved
    # match is evidence the student picked an offered option, not
    # evidence the system predicted them correctly, and the two must
    # never be blended into one "match rate."
    matched_via: str | None = None
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


class EvidenceCheckResult(BaseModel):
    """CheckEvidence's output. `satisfied_branch_id` is None on a real
    "nothing satisfied" outcome — an expected, useful result (same
    "don't force it" discipline as RESOLVE:MATCH), not represented as
    a bare None return: `node_calls.output_json` is NOT NULL, so every
    node must always return a structured object, even to represent
    "nothing.\""""

    satisfied_branch_id: UUID | None


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


class OptionStatus(str, Enum):
    OPEN = "open"
    SELECTED = "selected"
    SUPERSEDED = "superseded"


class Option(BaseModel):
    """One clickable button generated by GenerateOptions — the second,
    interpretation-free evidence channel: a click is an unambiguous
    fact about which branch's requires_evidence is now true, with no
    guess about what the student meant. `branch_id` is a hard 1:1
    mapping (GenerateOptions rejects and regenerates any response that
    maps two options to the same branch, or an option to no branch in
    the live set at all) — that mapping is what makes a click legible.

    Regenerated every turn from the current branch set and superseded
    the same way branches are (see HypothesisGenerator.resolve()) —
    never carried forward to a later turn.
    """

    id: UUID = Field(default_factory=uuid4)
    branch_id: UUID
    generation_id: UUID
    session_id: UUID
    turn_index: int
    text: str
    status: OptionStatus = OptionStatus.OPEN
    created_at: datetime = Field(default_factory=_utcnow)


class OptionProposal(BaseModel):
    """GenerateOptions' raw per-item output before persistence — not
    itself DB-backed (Option is, once IDs/session/turn context are
    attached in loop.py)."""

    branch_id: UUID
    text: str


class DisambiguationTurn(BaseModel):
    """One AssessAndBranch call's identity — the "how did we think
    about this" log entry disambiguate.py's module docstring describes.
    Written unconditionally, whether or not `needs_branches` fires, so
    a turn where the message was judged unambiguous is still a
    queryable row (`needs_branches=False`, `branch_statements=[]` on
    the branches it owns — see DisambiguationStore.create_turn), not a
    gap in the record.

    `turn_had_direct_answer` is True exactly when FinalAnswer ran
    against this same turn with no branch context — i.e.
    `needs_branches=False` and no click resolution was involved. A
    click-resolution turn (see SessionLoop._handle_disambiguation_turn)
    never creates one of these rows at all: it re-uses the branches
    already generated by an earlier DisambiguationTurn rather than
    running AssessAndBranch again, so there is nothing new here to log
    for it.
    """

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    turn_index: int
    needs_branches: bool
    turn_had_direct_answer: bool
    created_at: datetime = Field(default_factory=_utcnow)


class DisambiguationBranch(BaseModel):
    """One distinct plausible reading of an ambiguous message —
    disambiguate.py's flat counterpart to `Branch`. Deliberately
    missing every field that made `Branch` a tree node: no `parent_id`
    (no expansion, ever — see the module docstring for why depth is
    out of scope for this mode), no `depth`/`depth_label`, no
    `requires_evidence`/`evidence_satisfied` (nothing here is gated on
    a further condition before it's usable), no `predicted_next_turn`
    (nothing here forecasts a future turn to check against — a click
    or a typed message resolves it directly, not a text-match
    judgment).

    `status` reuses `BranchStatus`, but only ever visits OPEN, MATCHED
    (via an option click — this mode has no RESOLVE:MATCH-style text
    matching, so `matched_via` is always "option_click" in spirit even
    though the field itself isn't tracked here), and SUPERSEDED (every
    sibling once one is clicked, or every branch in a generation the
    student typed past instead of clicking — see the module docstring's
    "3b" case). UNMATCHED is never used: there is no prediction to fail
    to match.
    """

    id: UUID = Field(default_factory=uuid4)
    disambiguation_turn_id: UUID
    session_id: UUID
    turn_index: int
    statement: str
    status: BranchStatus = BranchStatus.OPEN
    created_at: datetime = Field(default_factory=_utcnow)


class DisambiguationAssessment(BaseModel):
    """AssessAndBranch's raw output before persistence — not itself
    DB-backed (DisambiguationTurn/DisambiguationBranch are, once IDs/
    session/turn context are attached in disambiguate.py). Empty
    `branch_statements` is the normal, expected shape whenever
    `needs_branches` is False, not a parse failure."""

    needs_branches: bool
    branch_statements: list[str] = Field(default_factory=list)


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
    """HypothesisGenerator.resolve()'s return value.

    `source` distinguishes how a "matched" status was reached:
    "option_click" (the clicked branch was marked matched directly, no
    RESOLVE:MATCH call at all — a click is settled evidence, running
    it through a fuzzy LLM match would add an interpretation layer to
    a signal specifically designed not to need one) vs "text_match"
    (the existing RESOLVE:MATCH judgment against predicted_next_turn).
    Always "text_match" when status is "unmatched" — a click never
    produces an unmatched outcome, it either resolves its branch or
    doesn't run at all. See CLAUDE.md's branch-store invariant and
    Branch.matched_via for why this must never be blended into one
    number downstream.
    """

    session_id: UUID
    turn_index: int
    matched_branch_id: UUID | None
    matched_chain: list[UUID] = Field(default_factory=list)
    status: str  # "matched" | "unmatched"
    source: str = "text_match"  # "option_click" | "text_match"
    call_count: int = 0


class ExplicitRequest(BaseModel):
    """ExtractRequest's output (nodes.py) — whether the student's
    message this turn contains a concrete, answerable request (a
    specific function, problem, example, or question with a definite
    thing to produce), as opposed to an open-ended or exploratory one.

    When `present`, `what` takes precedence over Plan's chosen
    target_concept and DerivePath's scope (see loop.py's handle_turn):
    the pedagogical machinery decides HOW to teach, never WHETHER to
    answer what was actually asked. `what` is None whenever `present`
    is False — there is no partial-request state.
    """

    present: bool
    what: str | None = None


class TeachingArtifact(BaseModel):
    """ExtractTeachingArtifact's output (nodes.py) — the concrete
    function/problem Teach actually worked this turn, and any analogy
    or metaphor it used to explain a concept, each None when absent
    (most turns have at most one of either). Read back out of
    node_calls (NodeCallStore.get_recent_calls, not a second
    persistence path — invariant 2 already guarantees Teach's own
    output is durable) to build the "already used" list Teach's next
    prompt checks against before reaching for a new example or analogy
    — see loop.py's _build_examples_used — and to answer "what did the
    tutor just give the student" when the student refers back to it —
    see nodes.check_prior_reference_unaddressed.
    """

    example: str | None = None
    analogy: str | None = None


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
    # True when options were on offer from the prior turn (open, not
    # yet superseded) and the student typed instead of clicking, and
    # that typed message didn't satisfy any live branch's
    # requires_evidence either. Read this as "the options didn't offer
    # what the student actually needed," not as the student being
    # uncooperative — it's a signal about the branch set, fed into the
    # next turn's generation context (see loop.py's
    # _build_transcript_context) and surfaced prominently in the web
    # UI's Diagnostics panel.
    options_missed: bool = False
    # True when hypothesis_generator.check_current_belief_leak flagged
    # this turn's DerivePath output — current_belief shared distinctive
    # vocabulary with the selected branch's predicted_next_turn or the
    # proposed action's rationale (content that had NOT happened yet)
    # while sharing none with the student's actual message. A
    # structural backstop for DerivePath's own prompt instructions, not
    # proof a leak occurred — see check_current_belief_leak's docstring.
    current_belief_unsupported: bool = False
    # Set only on turn 0 (the only turn BranchGenerate is skipped
    # outright — see loop.py's module docstring), to the reason it was
    # skipped. None on every other turn, including a turn where
    # generation actually ran and failed (that shows up as a warning
    # instead) — so "no branches because none were needed yet" and "no
    # branches because generation broke" are never confused for one
    # another in the UI or in match-rate analysis.
    generation_skipped_reason: str | None = None
    # ExtractRequest's own judgment for this turn (nodes.py) — whether
    # the student's message contained a concrete, answerable request,
    # and what it was. Persisted here (not just in node_calls) so the
    # web UI's Diagnostics panel can show it directly, same reasoning
    # as inferred_topic/topic_seeded_new above.
    explicit_request_present: bool = False
    explicit_request_what: str | None = None
    # True when nodes.check_explicit_request_unaddressed flagged this
    # turn's Teach output — explicit_request_what's distinctive terms
    # either never appeared in the response, or only appeared in a
    # sentence that reads as a deferral ("we can look at this next")
    # rather than a worked answer. A structural backstop for Teach's
    # own must-answer instruction, not proof the request went
    # unanswered — see check_explicit_request_unaddressed's docstring.
    # Always False when explicit_request_present is False.
    explicit_request_unaddressed: bool = False
    # nodes.detect_prior_reference's own judgment for this turn — True
    # when the student's message plausibly points back at something
    # already established this session ("the example you just gave",
    # "going back to X"). Persisted so the UI can distinguish "nothing
    # to check" from "checked and it was fine" the same way
    # explicit_request_present/explicit_request_unaddressed do.
    prior_reference_detected: bool = False
    # True when nodes.check_prior_reference_unaddressed flagged this
    # turn's Teach output — the student referenced prior content but
    # neither the example nor the analogy tracked from the immediately
    # preceding turn's ExtractTeachingArtifact call appears anywhere in
    # the response. A structural backstop for Teach's own
    # recent-history/examples_used prompt blocks, not proof the
    # reference was actually missed — see that function's docstring.
    # Always False when prior_reference_detected is False.
    prior_reference_unaddressed: bool = False
    # The memory layer's own visibility fields (memory.py) — see
    # LearnerFactStore/EmbedAndSearchFacts. `memory_match_found` is
    # true whenever the semantic pre-check's vector search cleared
    # MemoryConfig.fact_similarity_threshold, regardless of what the
    # confirmation call then decided; `memory_match_confirmed_resolution`
    # is true only when that confirmation call said the match actually
    # resolves the current message. `branching_skipped_by_memory` is
    # the literal behavioral consequence a reviewer would search for —
    # AssessAndBranch never ran this turn because of the match — kept
    # as its own field even though it is currently always equal to
    # memory_match_confirmed_resolution, so "the thing that actually
    # happened" is never left implicit. `matched_fact_id` traces which
    # fact caused any of the above, whether or not it was ultimately
    # confirmed.
    memory_match_found: bool = False
    memory_match_confirmed_resolution: bool = False
    branching_skipped_by_memory: bool = False
    matched_fact_id: UUID | None = None
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
    (the session's most recent, unresolved one) isn't included.

    `total_resolved`/`matched_count`/`match_rate` are scoped to
    text_match resolutions ONLY — a click-resolved match is evidence
    the student picked an option the system offered, not evidence the
    system predicted them correctly, so it is excluded from both the
    numerator and denominator here rather than inflating a "prediction
    accuracy" number it isn't evidence for. `option_click_count` is
    reported as its own separate, unrelated number — never combine the
    two into a single rate.
    """

    session_id: UUID
    session_created_at: datetime
    total_resolved: int
    matched_count: int
    option_click_count: int = 0

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
    wording-not-semantics heuristic.

    `matched_count`/`match_rate` count text_match confirmations only;
    `matched_via_click_count` is the same click-vs-text split as
    BranchMatchRatePoint, kept as its own separate number for the same
    reason — a click confirms the student chose that path, not that
    the system predicted it.
    """

    statement: str
    total_count: int
    matched_count: int
    matched_via_click_count: int = 0

    @property
    def match_rate(self) -> float:
        return self.matched_count / self.total_count if self.total_count else 0.0


class LearnerFactType(str, Enum):
    BRANCH_RESOLUTION = "branch_resolution"
    DIRECT_ANSWER = "direct_answer"


class LearnerFact(BaseModel):
    """One row of the memory layer's durable, per-learner record (see
    memory.py's module docstring) — plain English, written every turn
    a real resolution happened (a click resolved an ambiguity, or
    FinalAnswer answered directly), never on a turn that only raised
    options with nothing yet decided.

    `situation`/`resolution` are always in "the student's own terms,"
    not restated jargon — `situation` is what was ambiguous or asked,
    `resolution` is what was chosen (branch_resolution) or answered
    (direct_answer) and, where available, why. `embedding` is over the
    combined situation+resolution text (see memory.WriteLearnerFact),
    so a search on either half of a past fact can still surface it.

    Append-only (CLAUDE.md invariant 10): a fact is never edited or
    superseded once written — even a fact that a later turn's
    confirmation call judges "related but doesn't resolve" this
    message stays exactly as recorded; it was still true of what
    happened at `source_turn_id`.
    """

    id: UUID = Field(default_factory=uuid4)
    learner_id: UUID
    session_id: UUID
    turn_index: int
    fact_type: LearnerFactType
    situation: str
    resolution: str
    embedding: list[float]
    source_turn_id: UUID
    created_at: datetime = Field(default_factory=_utcnow)


class ExtractedFact(BaseModel):
    """WriteLearnerFact's own node output (models.py, not DB-backed) —
    deliberately excludes the embedding vector and every id: a node's
    `output_json` in node_calls should read as a human-checkable fact,
    not carry a 768-float array nobody will ever read there. The full
    `LearnerFact` (with its embedding) is what actually gets persisted
    to `learner_facts`, separately from this return value."""

    situation: str
    resolution: str


class FactSearchResult(BaseModel):
    """EmbedAndSearchFacts' own node output — the nearest learner_facts
    match (if any) and its cosine similarity, deliberately without the
    matched fact's embedding for the same node_calls-readability reason
    as ExtractedFact. `matched_fact_id` is None when the learner simply
    has no facts yet (a new learner, or turn 0) — not an error, the
    expected shape before any fact has ever been written for them."""

    matched_fact_id: UUID | None = None
    situation: str | None = None
    resolution: str | None = None
    similarity: float | None = None


class FactMatchConfirmation(BaseModel):
    """ConfirmFactMatch's output — a strong vector-similarity hit is
    not itself proof the matched fact resolves THIS message (see
    memory.py's module docstring on not asserting a pattern before
    it's earned); this is the structured judgment call that decides
    whether the memory pre-check is actually allowed to skip branching,
    not an inference from prose."""

    resolves: bool


class PathSummary(BaseModel):
    """SummarizeSessionPath's output — a label for the STRUCTURE of one
    session's ordered facts (e.g. "concrete example requested before
    abstract definition, repeatedly"), deliberately abstract enough to
    apply regardless of topic. Not itself a claim about the learner —
    see ThinkingStyleCandidate for what a *pattern* requires before it
    means anything."""

    summary: str


class ThinkingStyleConfirmation(BaseModel):
    """ConfirmThinkingStyleMatch's output — does this session's labeled
    path genuinely share the same order-structure as an existing
    candidate, or only a superficial resemblance? Only a True here
    advances ThinkingStyleCandidate.confirmation_count/session_ids;
    vector similarity alone (what found the candidate to compare
    against in the first place) is never sufficient by itself — see
    memory.py's module docstring."""

    confirms: bool


class ThinkingStyleStatus(str, Enum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    RETIRED = "retired"


class ThinkingStyleCandidate(BaseModel):
    """A hypothesized, cross-session order-structure for one learner —
    e.g. "wants a concrete example before an abstract rule, regardless
    of topic." Starts life as `candidate` (confirmation_count=1,
    session_ids=[the one session it was first labeled from]) and stays
    there, accumulating independent confirmations, until
    `confirmation_count` crosses `MemoryConfig.thinking_style_
    promotion_threshold` — only then does it become `confirmed` and
    only then does it get fed into any live session's prompts (see
    memory.py's module docstring: "nothing below should compromise
    [the payoff] by asserting a pattern before it's actually been
    earned across independent evidence").

    `confirmation_count`/`session_ids` only ever grow via an explicit
    `ConfirmThinkingStyleMatch` call saying yes — never from vector
    similarity alone, which only finds which existing candidate (if
    any) is worth asking that question about.

    Append-only (CLAUDE.md invariant 10): `retired` is a status
    transition via UPDATE, same resurrection-over-deletion principle as
    HypothesisStore's tiers — a candidate that stops matching is not
    erased, it stops being asked about.
    """

    id: UUID = Field(default_factory=uuid4)
    learner_id: UUID
    session_ids: list[UUID]
    path_summary: str
    path_summary_embedding: list[float]
    confirmation_count: int = 1
    status: ThinkingStyleStatus = ThinkingStyleStatus.CANDIDATE
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
