"""Node classes for the probe reasoning loop.

Every node exposes a single async `run(...)` method whose signature is
the node's semantic contract. Persistence of inputs/outputs is *not*
each node's responsibility — `SessionLoop._call_node()` intercepts and
records every call. See CLAUDE.md invariant 2.

Most nodes are stubs at this stage; the reasoning logic lands in later
steps. `Update` and `Diagnose` are real because they only depend on
already-built stores/services.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any
from uuid import UUID

from probe.audit import TranscriptStore
from probe.concept_graph import ConceptGraph
from probe.grounding import GroundConcept
from probe.llm import LLMClient
from probe.mismatch import MismatchDetector
from probe.models import (
    ActionScore,
    CandidateAction,
    EvidenceRef,
    ExplicitRequest,
    Hypothesis,
    InferOutput,
    Layer,
    PathRequirement,
    PlanOutput,
    Polarity,
    ProposedEvidence,
    ProposedHypothesis,
    SuggestedCause,
    TeachingAction,
    TeachingArtifact,
    Tier,
    TopicAttachment,
    WorldModelRevision,
)
from probe.overlay import LearnerOverlay
from probe.reasoning_budget import (
    ReasoningBudget,
    ReasoningBudgetConfig,
    compute_reasoning_budget,
)
from probe.revision import WorldModelRevisionStore
from probe.seed import seed_graph
from probe.store import HypothesisStore
from probe.value_function import ValueFunction

logger = logging.getLogger(__name__)

DEFAULT_GENERATION_WIDTH = 3
_MIN_WIDTH = 1

# One corrective re-ask when the proposer emits actions outside the enum.
_MAX_PROPOSE_ATTEMPTS = 2

# Diagnose: minimum MismatchResult.confidence required to act on
# suggested_cause=possible_world_model_error by proposing a
# WorldModelRevision. Intentionally low — proposals are supervised and
# reversible via review-revisions, so bias toward over-proposing while
# gathering real interaction data; revisit once session data exists to
# set this empirically.
DIAGNOSE_MISMATCH_THRESHOLD = 0.4

# Per-turn LLM-call guardrail, checked in SessionLoop.handle_turn against
# call-count instrumentation on every LLM-calling node/term: Diagnose's
# output_json (llm_call_count, covering GroundConcept + MismatchDetector),
# Infer.last_call_count, Plan.last_generate_call_count, each candidate's
# ActionScore (learning_value/information_value/cognitive_cost/
# frustration_risk_call_count), Teach.last_call_count, and — when
# branch_store is wired in — BranchResolve/BranchGenerate's own
# call_count (HypothesisGenerator's speculative prediction tree; see
# hypothesis_generator.py, opt-in and 0 when not configured). This is a
# complete count, not a floor — verified against a real turn (28 real
# LLM calls, 28 counted) after an earlier version of this guardrail only
# summed Diagnose + information_value and silently undercounted by 4x.
# This is a loud-warning guardrail, not a hard stop: crossing it logs
# and the turn continues — reasoning is never truncated or candidates
# silently dropped to stay under it. 30 is an arbitrary starting point,
# not a measured budget; revisit once real per-turn costs are observed
# against a real client.
MAX_CALLS_PER_TURN = 30


_LAYER_VALUES = {layer.value for layer in Layer}


class Infer:
    """Decides how this turn's message should change what the loop
    believes about the student -- the ONLY place in this codebase that
    creates a new `Hypothesis`, as well as the place that reweights an
    existing one.

    Before this fix, Infer could only emit reweight-shaped proposals
    (`ProposedEvidence`, requiring an already-existing `hypothesis_id`)
    -- there was no path anywhere in production code to mint the FIRST
    hypothesis for a learner. Since `HypothesisStore.reweight()` raises
    on a missing id, and `list_by_learner`/`list_by_concept` return
    only what already exists, every real session (CLI, web UI, every
    scripted comparison) ran with the hypothesis list permanently
    empty: `Infer` always received `hypotheses=[]`, always proposed
    nothing, `compute_reasoning_budget([])` always computed
    `entropy_bits=0.0`, and every downstream consumer of that (Plan's
    `generation_width`, `ValueFunction.enable_information_value`,
    `HypothesisGenerator`'s branch budget) silently ran at its
    zero-entropy floor forever -- a fully valid-looking result at every
    single step, which is exactly why nothing ever raised or logged a
    warning about it.

    The model's own JSON response is decision-only: WHICH existing
    hypothesis to reweight (or what NEW belief, layer/statement/initial
    probability+confidence, to create) and to what value. It is never
    trusted to emit `turn_id` or a full `EvidenceRef` itself -- the
    model has no way to know the turn's UUID, and the pre-fix schema's
    `evidence_ref` field could in fact never be populated by a real
    Gemini response for exactly that reason (confirmed: every existing
    test that exercised a reweight path had to hand-construct a canned
    JSON string containing a synthetic `evidence_ref`, something the
    real structured-output schema never asked the model for). `Infer`
    itself builds every `EvidenceRef` from the `turn_id` it's given.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        # Read by SessionLoop into the MAX_CALLS_PER_TURN accounting.
        # Public (not _-prefixed) since, like GroundConcept/MismatchDetector's
        # last_call_count, the reader is a different class.
        self.last_call_count: int = 0

    async def run(
        self,
        turn_text: str,
        hypotheses: list[Hypothesis],
        turn_id: UUID,
        generation_width: int = DEFAULT_GENERATION_WIDTH,
    ) -> InferOutput:
        self.last_call_count = 0
        listing = (
            "\n".join(
                f"- {h.id} [{h.layer.value}] {h.statement} "
                f"(p={h.probability:.2f}, c={h.confidence:.2f})"
                for h in hypotheses
            )
            if hypotheses
            else "(none tracked yet for this learner)"
        )
        prompt = (
            "INFER: given the following student turn, decide how it should "
            "change what we believe about this student.\n\n"
            f"Turn: {turn_text}\n\n"
            f"Existing tracked hypotheses:\n{listing}\n\n"
            'For each existing hypothesis this turn clearly supports or '
            'contradicts, emit a reweight item: {"kind": "reweight", '
            '"hypothesis_id": "<id from the list above>", '
            '"new_probability": 0.0-1.0, "new_confidence": 0.0-1.0, '
            '"polarity": "supporting" or "contradicting"}.\n'
            "If the turn reveals something worth tracking about the "
            "student that is NOT already covered by any hypothesis "
            "above -- a goal, a piece of knowledge, a mental model, a "
            "cognitive-state observation, or a teaching-relevant fact "
            "-- emit a create item instead: "
            '{"kind": "create", "layer": one of '
            f"{sorted(_LAYER_VALUES)}, "
            '"statement": "the belief, stated about the student", '
            '"initial_probability": 0.0-1.0, "initial_confidence": '
            "0.0-1.0}. Do not create a hypothesis that only restates one "
            "already in the list above.\n"
            "If nothing in the turn bears on any existing hypothesis and "
            "nothing new is worth tracking, return an empty list. Return "
            "a JSON array of these items, nothing else."
        )
        raw = await self._llm.complete(prompt)
        self.last_call_count += 1
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return InferOutput()
        if not isinstance(parsed, list):
            return InferOutput()

        valid_ids = {h.id for h in hypotheses}
        reweights: list[ProposedEvidence] = []
        creates: list[ProposedHypothesis] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            kind = item.get("kind")
            if kind == "reweight":
                reweight = self._parse_reweight(item, valid_ids, turn_id, len(hypotheses))
                if reweight is not None:
                    reweights.append(reweight)
            elif kind == "create":
                create = self._parse_create(item, turn_id)
                if create is not None:
                    creates.append(create)
            # Any other/missing `kind` is silently dropped, same
            # "malformed model output never crashes the turn"
            # discipline as every other parse-and-validate node.
        return InferOutput(reweights=reweights, creates=creates)

    def _parse_reweight(
        self, item: dict, valid_ids: set[UUID], turn_id: UUID, n_candidates: int
    ) -> ProposedEvidence | None:
        try:
            hypothesis_id = UUID(str(item["hypothesis_id"]))
            polarity = Polarity(item.get("polarity", Polarity.SUPPORTING.value))
        except (KeyError, ValueError, TypeError):
            return None
        if hypothesis_id not in valid_ids:
            # Same validation discipline as GroundConcept/Plan: reject
            # a reference outside the candidate set shown this turn
            # rather than pass it through. Not just cosmetic —
            # `hypotheses` is learner-scoped (SessionLoop), so this is
            # what actually stops a hallucinated or copied id from
            # reweighting a *different* learner's hypothesis; the
            # caller can't rely on the LLM simply never seeing one.
            logger.warning(
                "Infer: reweight referenced hypothesis_id %s, not in "
                "the %d candidates shown this turn — rejected rather "
                "than reweighted",
                hypothesis_id,
                n_candidates,
            )
            return None
        try:
            return ProposedEvidence(
                hypothesis_id=hypothesis_id,
                new_probability=item["new_probability"],
                new_confidence=item["new_confidence"],
                evidence_ref=EvidenceRef(turn_id=turn_id, polarity=polarity),
            )
        except Exception:
            return None

    def _parse_create(self, item: dict, turn_id: UUID) -> ProposedHypothesis | None:
        layer_raw = item.get("layer")
        if layer_raw not in _LAYER_VALUES:
            return None
        statement = item.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            return None
        try:
            return ProposedHypothesis(
                layer=Layer(layer_raw),
                statement=statement.strip(),
                initial_probability=item["initial_probability"],
                initial_confidence=item["initial_confidence"],
                # A brand-new hypothesis has nothing yet to contradict —
                # always SUPPORTING, never read from the model's own
                # output (see class docstring: the model decides WHAT,
                # not the evidence bookkeeping).
                evidence_ref=EvidenceRef(turn_id=turn_id, polarity=Polarity.SUPPORTING),
            )
        except Exception:
            return None


def _parse_explicit_request(raw: str) -> ExplicitRequest:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ExplicitRequest(present=False, what=None)
    if not isinstance(parsed, dict):
        return ExplicitRequest(present=False, what=None)
    what = parsed.get("what")
    if not parsed.get("present") or not isinstance(what, str) or not what.strip():
        # No partial-request state (see ExplicitRequest's docstring):
        # a falsy `present`, or a `present=true` with no usable `what`,
        # both collapse to the same "nothing to prioritize" outcome.
        return ExplicitRequest(present=False, what=None)
    return ExplicitRequest(present=True, what=what.strip())


class ExtractRequest:
    """Runs before Plan, every turn: does the student's message contain
    a concrete, answerable request -- a specific function, problem,
    example, or question with a definite thing to produce -- as
    opposed to an open-ended or exploratory one?

    A dedicated node rather than folded into Diagnose or Infer: this
    is not a belief judgment (Diagnose's job) or evidence about a
    hypothesis (Infer's job) -- it's a narrower, structural fact about
    the turn's own text, decoupled from both. Giving it its own
    node_calls row is also what lets Plan/DerivePath/Teach's downstream
    use of it (see loop.py) be audited independently of those other
    concerns, and what lets turn_diagnostics persist it directly for
    the web UI rather than the UI re-deriving it from a Plan or Teach
    row it doesn't otherwise need.

    When `present`, `what` takes precedence over Plan's chosen
    target_concept and DerivePath's scope: the pedagogical machinery
    decides HOW to teach, never WHETHER to answer what was actually
    asked -- see this module's `check_explicit_request_unaddressed`,
    the post-Teach backstop for when that precedence gets lost anyway.

    Fast tier: fires every turn, same tier as GroundConcept/
    MismatchDetector/Infer.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        # Read by SessionLoop into the MAX_CALLS_PER_TURN accounting.
        self.last_call_count: int = 0

    async def run(self, turn_text: str) -> ExplicitRequest:
        self.last_call_count = 0
        prompt = (
            "EXTRACT:REQUEST\n"
            "Does this student message contain a concrete, answerable "
            "request -- a specific function, problem, example, or "
            'question that has a definite thing to produce (e.g. "show '
            'me with sin(x^2)", "what\'s the derivative of 3x^2", "can '
            'you factor x^2-4")? An open-ended or exploratory message '
            '("what is a derivative", "tell me more", "I don\'t get '
            'it") does NOT count -- only a message with a specific, '
            "nameable thing to work out counts.\n\n"
            f"message: {turn_text}\n\n"
            'Respond with JSON: {"present": true or false, "what": '
            '"the specific request, in the student\'s own terms" or '
            "null}"
        )
        raw = await self._llm.complete(prompt)
        self.last_call_count += 1
        return _parse_explicit_request(raw)


class Plan:
    """Generate N candidate actions, score each, return the winner + scores.

    Candidate generation is LLM-driven: the proposer sees the current
    hypotheses (all five layers) plus `concept_state` and returns the
    `generation_width` actions it judges plausible *for this turn*,
    rather than the value function paying to score all 21. Scoring and
    selection are independent of how candidates were proposed.

    The returned `PlanOutput` carries both the winner (for Teach) and
    the full per-candidate `ActionScore` breakdown. Both survive
    together through the SessionLoop audit choke point (CLAUDE.md
    invariant 2) so ablation runs can inspect every candidate's terms,
    not just the winner's.
    """

    def __init__(self, value_function: ValueFunction, llm: LLMClient) -> None:
        self._vf = value_function
        self._llm = llm
        # Set by _generate(); read by SessionLoop into the
        # MAX_CALLS_PER_TURN accounting. Covers only the proposer's own
        # calls (1 or 2, depending on the correction retry) — each
        # candidate's scoring calls live on that candidate's ActionScore
        # instead, since they're per-candidate, not per-Plan-invocation.
        self.last_generate_call_count: int = 0

    async def run(
        self,
        hypotheses: list[Hypothesis],
        concept_state: dict,
        generation_width: int,
        exploration_target: Hypothesis | None = None,
        explicit_request: ExplicitRequest | None = None,
    ) -> PlanOutput:
        if exploration_target is None:
            logger.info(
                "Plan: no exploration_target this turn (no dormant/background "
                "hypothesis to explore) — generation_width=%d spent entirely "
                "on the current top hypothesis, no reservation used",
                generation_width,
            )
        candidates = await self._generate(
            hypotheses, concept_state, generation_width, exploration_target,
            explicit_request,
        )
        # Candidates are independent of each other — scored concurrently.
        # asyncio.gather preserves input order, so `scores` still lines
        # up positionally with `candidates` exactly as the sequential
        # loop did (deterministic tie-breaking in the max() below is
        # unaffected). ValueFunction.score() no longer writes call
        # counts to shared instance attributes (see value_function.py),
        # which is what makes concurrent calls against one shared
        # ValueFunction instance safe.
        scores: list[ActionScore] = list(
            await asyncio.gather(
                *(self._vf.score(c, hypotheses, concept_state) for c in candidates)
            )
        )
        winner_score = max(scores, key=lambda s: s.total)

        # Would a different candidate win with information_value
        # zeroed out? A pure re-max over already-computed per-term
        # floats — no new scoring, no LLM calls.
        winner_without_iv = max(scores, key=lambda s: s.total - s.information_value)
        argmax_changes = winner_without_iv.candidate.id != winner_score.candidate.id

        return PlanOutput(
            winner=winner_score.candidate,
            scores=scores,
            argmax_changes_without_information_value=argmax_changes,
        )

    async def _generate(
        self,
        hypotheses: list[Hypothesis],
        concept_state: dict,
        generation_width: int,
        exploration_target: Hypothesis | None,
        explicit_request: ExplicitRequest | None = None,
    ) -> list[CandidateAction]:
        """Ask the LLM for `generation_width` distinct plausible actions.

        Invalid actions are dropped, not fatal. If dropping left us short
        we retry once with the rejected values named, so the model can
        correct itself. Only after that do we backfill from enum order —
        the deterministic tie-break applies to the *filler*, never to the
        model's own proposals.
        """
        width = max(_MIN_WIDTH, generation_width)
        proposed: list[CandidateAction] = []
        seen: set[TeachingAction] = set()
        rejected: list[str] = []
        self.last_generate_call_count = 0

        for _ in range(_MAX_PROPOSE_ATTEMPTS):
            raw = await self._llm.complete(
                _propose_prompt(
                    hypotheses, concept_state, width, rejected, exploration_target,
                    explicit_request,
                )
            )
            self.last_generate_call_count += 1
            candidates, rejected = _parse_proposals(raw, concept_state)
            for candidate in candidates:
                if candidate.action in seen:
                    continue
                seen.add(candidate.action)
                proposed.append(candidate)
                if len(proposed) >= width:
                    break
            # Retry only buys something when the model actually emitted
            # garbage. A short-but-clean proposal is a legitimate answer:
            # backfill it instead of badgering the model.
            if len(proposed) >= width or not rejected:
                break

        del proposed[width:]
        proposed.extend(_backfill(width - len(proposed), seen, concept_state))
        return proposed


def _target_concept(concept_state: dict) -> str | None:
    """Default target_concept for a candidate the model didn't name
    one for (including backfilled, no-LLM-call filler candidates) —
    the concept the student's own message was actually grounded in
    this turn (GroundConcept, via Diagnose), if any. Not a curriculum
    choice: this is "what were we just talking about," not "what
    should we teach next.\""""
    if not isinstance(concept_state, dict):
        return None
    target = concept_state.get("grounded_concept_id")
    return None if target is None else str(target)


def _valid_concept_ids(concept_state: dict) -> set[str]:
    """The current session's actual concept graph, as seen by the
    proposer — target_concept validation (see _parse_proposals)
    rejects anything outside this set, same discipline as
    GroundConcept rejecting a concept_id outside the session's graph."""
    if not isinstance(concept_state, dict):
        return set()
    concepts = concept_state.get("concepts")
    if not isinstance(concepts, list):
        return set()
    return {c["id"] for c in concepts if isinstance(c, dict) and c.get("id")}


def _propose_prompt(
    hypotheses: list[Hypothesis],
    concept_state: dict,
    width: int,
    rejected: list[str],
    exploration_target: Hypothesis | None = None,
    explicit_request: ExplicitRequest | None = None,
) -> str:
    # Same belief set the value function scores against (ValueFunction.score
    # filters to ACTIVE too), so proposer and scorer never disagree about
    # what's currently in play. All five layers, not just knowledge.
    listing = "\n".join(
        f"- {h.id} [{h.layer.value}] {h.statement} "
        f"(p={h.probability:.2f}, c={h.confidence:.2f})"
        for h in hypotheses
        if h.tier is Tier.ACTIVE
    )
    vocabulary = ", ".join(a.value for a in TeachingAction)
    correction = ""
    if rejected:
        correction = (
            "\nYour previous attempt included values outside the vocabulary: "
            f"{', '.join(rejected)}. Use only the listed action values.\n"
        )
    exploration_instruction = ""
    if exploration_target is not None:
        exploration_instruction = (
            "\nMaintain an exploration budget: exactly one of your "
            f"{width} candidates must explicitly target hypothesis "
            f"{exploration_target.id} [{exploration_target.layer.value}/"
            f"{exploration_target.tier.value}] \"{exploration_target.statement}\" "
            f"(p={exploration_target.probability:.2f}) rather than the "
            "current dominant hypothesis — this is a deliberately "
            "under-examined belief, not the most likely one. Start that "
            'candidate\'s rationale with "[exploration]".\n'
        )
    valid_concept_ids = _valid_concept_ids(concept_state)
    concept_instruction = ""
    if valid_concept_ids:
        concept_instruction = (
            "\nEach candidate's target_concept must be one of the exact "
            "concept ids listed in concept_state.concepts above — pick "
            "the one this turn's action actually addresses. Use null "
            "only when no single concept in that list fits (e.g. a "
            "purely motivational or clarifying action) — never invent "
            "an id that isn't in that list.\n"
        )
    explicit_request_instruction = ""
    if explicit_request is not None and explicit_request.present and explicit_request.what:
        explicit_request_instruction = (
            "\nThe student made an explicit, concrete request this turn "
            f"that MUST be answered, not substituted: {explicit_request.what!r}. "
            "Every candidate you propose must serve THIS request "
            "directly — choose an action and target_concept such that "
            "teaching it means actually resolving what was asked, not a "
            "different example or a related-but-different topic. You "
            "decide HOW to answer it (which action, which framing), "
            "never whether to.\n"
        )
    return (
        "PROPOSE:ACTIONS\n"
        f"Propose exactly {width} distinct teaching actions worth scoring "
        "for this turn. Choose from this vocabulary only:\n"
        f"{vocabulary}\n\n"
        f"concept_state={json.dumps(concept_state, sort_keys=True, default=str)}\n"
        "hypotheses (all layers):\n"
        f"{listing}\n"
        f"{correction}"
        f"{exploration_instruction}"
        f"{concept_instruction}"
        f"{explicit_request_instruction}"
        'Respond with JSON: [{"action": "...", "target_concept": "..." or null, '
        '"rationale": "one short sentence"}, ...]'
    )


def _coerce_action(raw: object) -> TeachingAction | None:
    if isinstance(raw, TeachingAction):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        return TeachingAction(raw.strip().lower())
    except ValueError:
        pass
    try:
        return TeachingAction[raw.strip().upper()]
    except KeyError:
        return None


def _parse_proposals(
    raw: str, concept_state: dict
) -> tuple[list[CandidateAction], list[str]]:
    """Split an LLM proposal payload into valid candidates and rejects."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return [], ["<unparseable response>"]
    if not isinstance(parsed, list):
        return [], ["<response was not a JSON list>"]

    default_target = _target_concept(concept_state)
    valid_concept_ids = _valid_concept_ids(concept_state)
    valid: list[CandidateAction] = []
    rejected: list[str] = []
    for item in parsed:
        if not isinstance(item, dict):
            rejected.append(repr(item))
            continue
        action = _coerce_action(item.get("action"))
        if action is None:
            rejected.append(repr(item.get("action")))
            continue
        target = item.get("target_concept")
        target_concept = default_target if target is None else str(target)
        if (
            target is not None
            and valid_concept_ids
            and target_concept not in valid_concept_ids
        ):
            # Same validation discipline as GroundConcept: reject a
            # concept outside this session's actual graph rather than
            # pass it through — fall back to the grounded default
            # instead of the hallucinated/invented id.
            logger.warning(
                "Plan: proposal named target_concept %r, not in this "
                "session's %d-concept graph — rejected, falling back "
                "to %r",
                target_concept,
                len(valid_concept_ids),
                default_target,
            )
            target_concept = default_target
        rationale = item.get("rationale")
        valid.append(
            CandidateAction(
                action=action,
                target_concept=target_concept,
                rationale=str(rationale) if rationale else "",
            )
        )
    return valid, rejected


def _backfill(
    n: int, seen: set[TeachingAction], concept_state: dict
) -> list[CandidateAction]:
    if n <= 0:
        return []
    target = _target_concept(concept_state)
    filler = [a for a in TeachingAction if a not in seen][:n]
    return [
        CandidateAction(
            action=a,
            target_concept=target,
            rationale="backfill: proposer returned fewer candidates than "
            "generation_width",
        )
        for a in filler
    ]


class Teach:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        # Read by SessionLoop into the MAX_CALLS_PER_TURN accounting.
        self.last_call_count: int = 0

    async def run(
        self,
        action: CandidateAction,
        student_message: str,
        path_requirement: PathRequirement | None = None,
        options: list[str] | None = None,
        explicit_request: ExplicitRequest | None = None,
        recent_history: str = "",
        examples_used: str = "",
    ) -> str:
        """Teach no longer receives the branch tree or a bare topic
        string — a tree invites free association, a path constrains.
        `path_requirement` (DerivePath's output, from the branch this
        turn selected as most representative) is what scopes this
        turn's teaching instead; None only when branch generation is
        disabled or failed upstream, in which case this degrades to
        target_concept-only framing, the same as before this feature
        existed.

        `options` (GenerateOptions' output, just the button text — not
        which branch each maps to; Teach only needs to know what to
        lead into, not the underlying bookkeeping) are rendered as
        buttons by the UI below this response, not restated in it. The
        instruction below is about how the response *ends*, not what
        it contains.

        `explicit_request` (ExtractRequest's output) is stated
        separately from `path_requirement`, deliberately: it is a
        MUST-ANSWER instruction, not scoping guidance — path_requirement
        says how to frame the turn, explicit_request says what has to
        actually get resolved inside it regardless of that framing. See
        `check_explicit_request_unaddressed` for the post-hoc backstop
        when this instruction gets ignored anyway.

        `recent_history` (loop.py's `_build_teach_history` — the last
        few student turns plus Teach's own last few responses, read
        back out of node_calls, NOT the full transcript_context
        HypothesisGenerator uses) exists so Teach can notice it already
        answered something, already stated a constraint, or already
        used a given framing, instead of contradicting or ignoring its
        own recent work.

        `examples_used` (loop.py's `_build_examples_used`, backed by
        ExtractTeachingArtifact's own node_calls) is the structured
        record of concrete examples/analogies already used this
        session — reuse or build on one of these before reaching for a
        new one. See `check_prior_reference_unaddressed` for the
        post-hoc backstop for when the student references something
        from here by name and Teach doesn't.
        """
        self.last_call_count = 0
        focus = (
            f"Focus specifically on the concept {action.target_concept!r}."
            if action.target_concept
            else ""
        )
        path_block = ""
        if path_requirement is not None:
            path_block = (
                "\nWhat the student appears to currently believe (this "
                "is YOUR OWN inference, not a quote — the student did "
                "not necessarily say this): "
                f"{path_requirement.current_belief}\n"
                "What they need from you this turn: "
                f"{path_requirement.needed}\n"
                "The scope of this turn's teaching — stay within this, "
                f"one thing, not a syllabus: {path_requirement.scope}\n"
                "Never use affirmation language — \"exactly\", \"that's "
                "right\", \"yes\", \"as you said\", \"perfect way to "
                "think about it\" — about anything that is not "
                "literally present in the student's own message above. "
                "You may address, build on, or gently correct an "
                "inferred belief, but never credit the student with "
                "having said or confirmed something themselves when "
                "they did not.\n"
            )
            if path_requirement.must_not_assume:
                must_not = "; ".join(path_requirement.must_not_assume)
                path_block += (
                    "Do NOT assume or state as settled: "
                    f"{must_not}. If the answer depends on one of these, "
                    "address that dependency explicitly rather than "
                    "picking a value.\n"
                )
        options_block = ""
        if options:
            option_listing = "\n".join(f"- {o}" for o in options)
            options_block = (
                "\nThe following will appear as clickable buttons right "
                "after your response — they are the real fork this "
                f"conversation is about to take:\n{option_listing}\n"
                "End your response AT that fork, in ONE flowing sentence "
                "or two of ordinary prose — never a list, never bullet "
                "points, never numbered items, and never wording close "
                "enough to either button that it reads as copied. "
                "Paraphrase the underlying difference between the "
                "directions (what each would actually show about the "
                "problem) folded into your own closing sentence, the "
                "same way the rest of your response reads — so the "
                "buttons feel like the obvious next step, not a menu "
                "bolted onto the end. Do not just stop flatly with no "
                "forward motion either, and never call them \"options\" "
                "or say \"choose one.\"\n"
                "Never end by asking how the student feels, what they "
                "prefer, or what kind of learner they are — a question "
                "about the student's own reaction or self-knowledge is "
                "exactly the self-report framing this mechanism exists "
                "to avoid. The close is about what happens next in the "
                "material, not about the student.\n"
            )
        explicit_request_block = ""
        if explicit_request is not None and explicit_request.present and explicit_request.what:
            explicit_request_block = (
                "\nThe student made an explicit, concrete request this "
                "turn that you MUST fully answer within this response: "
                f"{explicit_request.what}\n"
                "Work it through to its actual result — the named "
                "function, problem, or question must be resolved here, "
                "not merely referenced. Deferring it (\"we can apply "
                "this to X next\", \"let's explore this later\", "
                "\"would you like to see...\") is a failure, not a "
                "valid pedagogical choice. You may still use the "
                "guidance above to frame HOW you answer it, but the "
                "request itself must be answered now, not promised.\n"
            )
        history_block = ""
        if recent_history:
            history_block = (
                "\nRecent conversation, for continuity only (do not "
                "repeat this back or restate it — use it to avoid "
                "contradicting or ignoring what you just said):\n"
                f"{recent_history}\n"
            )
        examples_block = ""
        if examples_used:
            examples_block = (
                "\nConcrete examples/analogies already used this "
                f"session:\n{examples_used}\n"
                "Prefer reusing or explicitly building on one of these "
                "over introducing a new one when a prior one still "
                "applies. If the student's message refers to something "
                "from earlier (\"the example you just gave\", \"going "
                "back to X\", \"does that still apply to...\"), you "
                "MUST name and use that specific thing here — do not "
                "restate it generically or replace it with a new "
                "framing or a different analogy.\n"
            )
        prompt = (
            "TEACH: "
            + json.dumps(
                {
                    "action": action.action.value,
                    "target_concept": action.target_concept,
                    "rationale": action.rationale,
                    "student_message": student_message,
                }
            )
            + (f"\n{focus}" if focus else "")
            + explicit_request_block
            + history_block
            + examples_block
            + path_block
            + options_block
            + "\nDo not introduce specific values, signs, conditions, or "
            "givens that appear in neither the student's message nor "
            "what you were told above.\n"
            "Lead with the direct answer or key idea — do not open with "
            "setup or a restatement of the question. If an example "
            "helps, weave it into the explanation inline rather than as "
            "a separate section. Do not partition the response into "
            "steps or add headers unless the content genuinely requires "
            "that structure.\n"
            "Never mention or describe your own fields, arguments, or "
            "internal state (e.g. never say something is \"unspecified\", "
            "\"null\", or refer to target_concept/action/rationale by "
            "name) — just teach, as if you already knew what to say."
        )
        result = await self._llm.complete(prompt)
        self.last_call_count += 1
        return result


class ExtractTeachingArtifact:
    """Runs once, right after Teach succeeds each turn: pulls out the
    concrete function/problem actually worked (if any) and the analogy
    or metaphor used to explain a concept (if any), as short structured
    phrases.

    Teach's rendered text is already the durable record (node_calls
    already has it verbatim, per CLAUDE.md invariant 2) — this node
    exists because "was an analogy like this already used" is far
    cheaper and more reliable to check against two short structured
    fields than by re-reading full rendered prose covered in LaTeX
    every turn. Its own node_calls rows, read back via
    NodeCallStore.get_recent_calls, are what loop.py's
    `_build_examples_used` turns into Teach's "already used" list and
    what `check_prior_reference_unaddressed` checks a backward
    reference against.

    Fast tier, same reasoning as ExtractRequest: a narrow, structural
    fact about text that already exists, not a judgment call.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        # Read by SessionLoop into the MAX_CALLS_PER_TURN accounting.
        self.last_call_count: int = 0

    async def run(self, teach_output: str) -> TeachingArtifact:
        self.last_call_count = 0
        prompt = (
            "EXTRACT:ARTIFACT\n"
            "From this tutor response, identify two things if present:\n"
            "1. The concrete function or problem actually worked (e.g. "
            '"sin(x^2)", "(3x+1)^5") -- not the general topic, the '
            "specific expression.\n"
            "2. Any analogy or metaphor used to explain a concept (e.g. "
            '"nested Russian dolls", "gears turning inside each '
            'other").\n\n'
            f"tutor response: {teach_output}\n\n"
            'Respond with JSON: {"example": "short phrase" or null, '
            '"analogy": "short phrase" or null}. Most responses have '
            "at most one of each -- use null for whichever is absent."
        )
        raw = await self._llm.complete(prompt)
        self.last_call_count += 1
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return TeachingArtifact(example=None, analogy=None)
        if not isinstance(parsed, dict):
            return TeachingArtifact(example=None, analogy=None)
        example = parsed.get("example")
        analogy = parsed.get("analogy")
        return TeachingArtifact(
            example=example.strip() if isinstance(example, str) and example.strip() else None,
            analogy=analogy.strip() if isinstance(analogy, str) and analogy.strip() else None,
        )


# Phrases that read as putting a task off rather than doing it now.
# Deliberately generic (not tied to any one subject/domain) since an
# explicit request can be about anything from a math function to a
# code example — the tell is the deferral language, not the topic.
_DEFERRAL_PHRASES = (
    "we can", "we could", "we'll", "we will", "let's", "let us",
    "you can", "next time", "next turn", "in the future", "later",
    "would you like", "moving forward", "going forward", "afterward",
    "if you'd like", "if you want", "if you'd prefer",
)

# Same discipline as hypothesis_generator._BELIEF_CHECK_STOPWORDS: words
# too generic to count as "the request's own content" — without
# filtering these, almost any sentence would share enough of them with
# the request text to look like a match. Generic task verbs (compute,
# differentiate, solve...) get filtered alongside ordinary stopwords,
# not just connector words: "differentiate sin(x^2)" and "differentiate
# 3x^2" share the verb but ask about different objects, so the verb
# must not be what makes a sentence count as "addressing the request" —
# only the named object should.
_REQUEST_CHECK_STOPWORDS = frozenset(
    {
        "the", "and", "with", "for", "this", "that", "show", "example",
        "can", "you", "what", "how", "does", "using", "use", "let",
        "are", "is", "of", "to", "in", "on", "at", "me", "an", "as",
        "it", "your", "would", "please", "give", "tell", "about",
        "differentiate", "derive", "derivative", "calculate", "compute",
        "find", "solve", "determine", "evaluate", "simplify", "factor",
        "integrate", "expand", "demonstrate", "explain", "prove",
        "verify", "work", "out", "walk", "through",
    }
)


def _request_content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) > 1 and w not in _REQUEST_CHECK_STOPWORDS}


def check_explicit_request_unaddressed(explicit_request_what: str, teach_output: str) -> bool:
    """A structural backstop for Teach's own must-answer instruction,
    same spirit as `hypothesis_generator.check_current_belief_leak`: a
    word-overlap heuristic, not semantic understanding, that flags a
    pattern worth a human's attention in turn_diagnostics rather than
    proving the request went unanswered.

    True when either (a) none of the request's distinctive terms
    appear anywhere in the output at all, or (b) every sentence that
    does mention them also reads as a deferral ("we can look at this
    next") rather than a worked answer — the exact failure this check
    was written after: Plan substituting a different example and Teach
    writing "we can apply this to sin(x^2)" without ever computing it.
    A sentence that mentions the request WITHOUT deferral language is
    treated as evidence it was actually addressed, even if a *different*
    sentence elsewhere also happens to defer — one worked answer is
    enough, regardless of how many hedges surround it.
    """
    request_words = _request_content_words(explicit_request_what)
    if not request_words:
        return False  # nothing distinctive enough in the request to check against

    output_lower = teach_output.lower()
    if not any(word in output_lower for word in request_words):
        return True  # never even mentioned

    sentences = re.split(r"(?<=[.!?])\s+", teach_output)
    mentioning_sentences = [
        s for s in sentences if any(word in s.lower() for word in request_words)
    ]
    if not mentioning_sentences:
        return True  # matched at the word-soup level but not within any real sentence
    return all(
        any(phrase in s.lower() for phrase in _DEFERRAL_PHRASES)
        for s in mentioning_sentences
    )


# Phrases that plausibly point back at something already established
# this session, rather than raising a new topic. Deliberately generic
# (not tied to any one subject/domain), the same discipline as
# _DEFERRAL_PHRASES -- the tell is the backward-pointing language, not
# the topic. "you just gave"/"you gave" cover the exact observed live
# failure: "does that still apply to the chain rule example you just
# gave?"
_REFERENCE_PHRASES = (
    "you just gave", "you gave", "you just showed", "you showed",
    "you just said", "you said", "earlier you", "you mentioned",
    "going back to", "back to the", "the example you", "that example",
    "the one you", "we talked about", "still applies to",
    "does that still apply", "the analogy you", "like you said",
    "as you mentioned", "you just did", "you just used",
)


def detect_prior_reference(student_message: str) -> bool:
    """Heuristic, phrase-based -- same discipline as `_DEFERRAL_PHRASES`:
    flags when the student's message plausibly points back at
    something already established (a prior example, analogy, or
    explanation), without attempting to understand what specifically.
    Gates whether `check_prior_reference_unaddressed` is even
    meaningful this turn."""
    lower = student_message.lower()
    return any(phrase in lower for phrase in _REFERENCE_PHRASES)


def check_prior_reference_unaddressed(
    last_artifact: TeachingArtifact | None, teach_output: str
) -> bool:
    """A structural backstop for Teach's own recent-history/
    examples_used prompt blocks, same spirit as
    `check_explicit_request_unaddressed`: a word-overlap heuristic, not
    semantic understanding, that flags a pattern worth a human's
    attention in turn_diagnostics rather than proving the reference was
    actually missed.

    `last_artifact` is the example/analogy tracked from the
    immediately preceding turn's ExtractTeachingArtifact call (see
    loop.py's `_build_examples_used`) -- "you just gave" can only ever
    mean the turn right before this one. Checked per field, not pooled:
    if the tracked turn had an example, that example's distinctive
    terms must appear somewhere in `teach_output`; separately, if it
    had an analogy, that analogy's terms must appear too. Flags True
    the moment either tracked field goes missing -- reusing one while
    silently dropping the other is still the failure this exists to
    catch, live-observed after the fix first shipped: turn 4 kept the
    "nested dolls" analogy language but swapped in a brand-new function,
    (3x+1)^5, instead of continuing with sin(x^2), the actual thing "the
    chain rule example you just gave" referred to -- a pooled check
    would have (and initially did) wave this through as "addressed"
    because the analogy words alone were present.

    Returns False (nothing to check) when there is no tracked artifact
    from the immediately preceding turn at all, or it tracked neither
    an example nor an analogy -- a student can reference something
    this mechanism has no record of (a turn before this feature
    existed, or a purely conversational aside), which is not evidence
    of a new failure.
    """
    if last_artifact is None:
        return False
    if last_artifact.example is None and last_artifact.analogy is None:
        return False
    output_lower = teach_output.lower()
    for field in (last_artifact.example, last_artifact.analogy):
        if not field:
            continue
        field_words = _request_content_words(field)
        if field_words and not any(word in output_lower for word in field_words):
            return True
    return False


class BaselineTeach:
    """The true plain-LLM baseline (AblationConfig.is_full_bypass, see
    loop.py's `_handle_bypass_turn`): one call, no hypotheses, no
    concept graph, no plan, no branches -- just the student's message
    plus this session's prior turns as context. This is the number
    every other layer is measured against, so it deliberately does not
    reuse `Teach`'s architecture-specific scaffolding (target_concept
    framing, PathRequirement blocks, options blocks) -- those exist to
    carry information (a planned action, a derived path) this baseline
    never has.

    It DOES reuse Teach's plain output-shape instructions (no JSON
    wrapper, no headers/step-partitioning unless the content needs it,
    never close by asking how the student feels or what they prefer):
    those are prompt-writing hygiene, not architecture, and withholding
    them would confound the comparison this node exists to make honest
    -- a baseline that loses to the full system only because nobody
    told it not to leak JSON is not evidence any *layer* is doing
    something, just evidence of an unfair prompt.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        # Read by SessionLoop into the MAX_CALLS_PER_TURN accounting,
        # same convention as every other node.
        self.last_call_count: int = 0

    async def run(self, turn_text: str, prior_turns: list[str]) -> str:
        self.last_call_count = 0
        history = (
            "\n".join(f"student: {t}" for t in prior_turns)
            if prior_turns
            else "(no prior turns)"
        )
        prompt = (
            # The "BASELINE:TEACH" prefix matters, not just for
            # logging: llm.py's GeminiLLMClient dispatches structured-
            # output config by longest matching prefix, and an
            # unrecognized prefix defaults to forcing
            # response_mime_type=application/json at the API level —
            # no prompt text can override that after the fact. TEACH:
            # is registered free-text for the same reason; this prefix
            # must be too.
            "BASELINE:TEACH\n"
            "You are a tutor having a conversation with a student. "
            "Respond directly and helpfully to their latest message.\n\n"
            f"Prior turns in this conversation:\n{history}\n\n"
            f"Student's latest message: {turn_text}\n\n"
            "Lead with the direct answer or key idea — do not open "
            "with setup or a restatement of the question. Do not "
            "partition the response into steps or add headers/numbered "
            "lists unless the content genuinely requires that "
            "structure.\n"
            "Never end by asking how the student feels, what they "
            "prefer, or what kind of learner they are.\n"
            "Respond with plain prose only — never wrap your answer in "
            "JSON or any other structured/markup format."
        )
        result = await self._llm.complete(prompt)
        self.last_call_count += 1
        return result


class Test:
    async def run(self, hypotheses: list[Hypothesis]) -> str:
        # Real diagnostic-question generation lives later.
        return "Can you walk me through how you'd approach this in your own words?"


class Diagnose:
    """Compares a learner's apparent belief against the world model.

    The response-vs-expectation classification is still the Step-2
    stub — a real classifier lives later. What's real here:

    1. GroundConcept determines which concept (if any) `response` is
       actually about, from the full list of nodes in *this session's*
       linked graph (`session_id` -> `concept_graph_id`, via
       `TranscriptStore.get_concept_graph_id` — a session's graph is
       set once at creation, same pattern as its learner). A
       hallucinated id GroundConcept didn't validate, or a grounding
       below `_GROUNDING_CONFIDENCE_THRESHOLD`, skips the rest of this
       method gracefully, same as an unrecognized concept_id always has.
       The grounding's own concept_id/confidence is recorded on the
       output regardless of whether it clears the threshold, so a
       low-confidence-but-still-acted-on grounding stays visible on
       review, not folded silently into the final MismatchResult.
    2. The learner's OverlayEntry and the mental_model hypotheses
       already linked to the grounded concept (via
       `HypothesisStore.list_by_concept` — Hypothesis itself carries no
       concept_id; see `hypothesis_concepts`) go to `MismatchDetector`,
       whose judgment is acted on:

       - `possible_world_model_error` above `DIAGNOSE_MISMATCH_THRESHOLD`
         → propose a `WorldModelRevision`. Never auto-applied — it sits
         pending until a human reviews it (`probe review-revisions`).
         `result["mismatch"]` (and its `confidence`) is persisted
         regardless of whether the threshold is cleared — see
         CLAUDE.md invariant 2 — so a below-threshold mismatch the
         loop declined to act on is still visible on review, not
         silently dropped.
       - `learner_misconception` → append supporting evidence to each
         matched hypothesis via `HypothesisStore.reweight` directly,
         not by calling `Update().run(...)`: CLAUDE.md invariant 2
         forbids invoking a Node's `run()` from a production path
         outside `SessionLoop._call_node`, so this goes straight to the
         same `reweight()` path `Update` itself uses — nothing bypasses
         evidence-append.

    Concept *selection* ("what should we teach next") is explicitly out
    of scope here — this only identifies what the current response is
    about.
    """

    # The revision-proposal threshold lives at module level as
    # DIAGNOSE_MISMATCH_THRESHOLD (it directly controls how often the
    # possible_world_model_error path can fire, so it's named and
    # documented there, not buried here).
    #
    # _GROUNDING_CONFIDENCE_THRESHOLD is a separate, unrelated decision
    # (whether GroundConcept's judgment is trusted enough to run
    # mismatch detection at all) — still an arbitrary midpoint
    # placeholder, not measured or reasoned from anything. Do not
    # assume tuning one of these two thresholds should move the other.
    _GROUNDING_CONFIDENCE_THRESHOLD = 0.5

    def __init__(
        self,
        mismatch_detector: MismatchDetector,
        ground_concept: GroundConcept,
        hypothesis_store: HypothesisStore,
        revision_store: WorldModelRevisionStore,
        concept_graph: ConceptGraph,
        learner_overlay: LearnerOverlay,
        transcript: TranscriptStore,
    ) -> None:
        self._detector = mismatch_detector
        self._grounder = ground_concept
        self._hyp = hypothesis_store
        self._revisions = revision_store
        self._concepts = concept_graph
        self._overlay = learner_overlay
        self._transcript = transcript

    async def run(
        self,
        response: str,
        expectation: str,
        session_id: UUID,
        turn_id: UUID,
        enable_grounding: bool = True,
        enable_mismatch: bool = True,
        enable_hypotheses: bool = True,
    ) -> dict:
        """`enable_grounding`/`enable_mismatch`/`enable_hypotheses` are
        AblationConfig's `enable_concept_graph`/`enable_diagnose`/
        `enable_portrait` respectively (see loop.py's call site) — all
        default True so every existing caller that doesn't pass them
        gets exactly today's behavior.

        `enable_grounding=False` skips GroundConcept entirely (no
        concept to ground against, no LearnerOverlay read) and returns
        immediately — the "no grounding" half of disabling the concept
        graph. `enable_mismatch=False` still grounds (concept grounding
        is a separate concern from mismatch detection) but returns
        right after, before MismatchDetector, ConceptGraph.get_concept,
        or LearnerOverlay are ever touched. `enable_hypotheses=False`
        passes an empty hypothesis list into MismatchDetector instead
        of reading HypothesisStore.list_by_concept — the LEARNER_MISCONCEPTION
        branch below then naturally reweights nothing, since its loop is
        over that same (now empty) list; no separate guard needed there.
        """
        result: dict[str, Any] = {
            "classification": "unknown",
            "matched_expectation": False,
            "notes": "stub response/expectation classifier — no real "
            "classifier yet",
            "grounding": None,
            "mismatch": None,
            "action_taken": "none",
            "revision_id": None,
            "reweighted_hypothesis_ids": [],
            # Real LLM calls this run() made: grounding always counts,
            # mismatch detection only if grounding cleared the
            # threshold. Costs nothing against the stub, but means the
            # per-turn call-count data is already sitting in node_calls
            # once a real LLM is behind this instead of needing
            # instrumentation added retroactively — same reasoning as
            # ActionScore.information_value_call_count.
            "llm_call_count": 0,
        }

        if not enable_grounding:
            result["notes"] += (
                "; concept grounding disabled "
                "(AblationConfig.enable_concept_graph=False) — skipped "
                "grounding and mismatch check entirely"
            )
            return result

        learner_id = await self._transcript.get_learner_id(session_id)
        concept_graph_id = await self._transcript.get_concept_graph_id(session_id)

        candidates = await self._concepts.list_concepts(concept_graph_id)
        grounding = await self._grounder.detect(response, candidates)
        result["llm_call_count"] += self._grounder.last_call_count
        result["grounding"] = grounding.model_dump(mode="json")

        if (
            grounding.concept_id is None
            or grounding.confidence < self._GROUNDING_CONFIDENCE_THRESHOLD
        ):
            result["notes"] += (
                "; response did not clearly ground to a concept in this "
                "session's graph, skipped mismatch check"
            )
            return result
        concept_id = grounding.concept_id

        if not enable_mismatch:
            result["notes"] += (
                "; mismatch/revision detection disabled "
                "(AblationConfig.enable_diagnose=False) — grounded but "
                "skipped MismatchDetector"
            )
            return result

        concept = await self._concepts.get_concept(concept_graph_id, concept_id)
        if concept is None:
            result["notes"] += (
                f"; concept {concept_id!r} not found, skipped mismatch check"
            )
            return result

        overlay_entry = await self._overlay.get_state(
            learner_id, concept_graph_id, concept_id
        )
        hypotheses = (
            await self._hyp.list_by_concept(
                concept_graph_id, concept_id, layer=Layer.MENTAL_MODEL, tier=Tier.ACTIVE
            )
            if enable_hypotheses
            else []
        )

        mismatch = await self._detector.detect(
            concept_id, concept, overlay_entry, hypotheses
        )
        result["llm_call_count"] += self._detector.last_call_count
        if mismatch is None:
            return result
        result["mismatch"] = mismatch.model_dump(mode="json")

        if (
            mismatch.suggested_cause is SuggestedCause.POSSIBLE_WORLD_MODEL_ERROR
            and mismatch.confidence >= DIAGNOSE_MISMATCH_THRESHOLD
        ):
            revision = await self._revisions.propose(
                WorldModelRevision(
                    concept_graph_id=concept_graph_id,
                    concept_id=concept_id,
                    proposed_change=(
                        f"possible error in concept {concept_id!r}: learner "
                        f"claims {mismatch.learner_claim!r}, which conflicts "
                        f"with world_claim {mismatch.world_claim!r}"
                    ),
                    evidence_refs=[
                        EvidenceRef(turn_id=turn_id, polarity=Polarity.SUPPORTING)
                    ],
                    confidence=mismatch.confidence,
                )
            )
            result["action_taken"] = "revision_proposed"
            result["revision_id"] = str(revision.id)
        elif mismatch.suggested_cause is SuggestedCause.LEARNER_MISCONCEPTION:
            # Reweight-only, by deliberate decision, not an oversight
            # left alongside Infer's fix: hypothesis creation stays
            # Infer/Update's sole responsibility. Two independent code
            # paths both minting MENTAL_MODEL hypotheses for the same
            # concept -- this one keyed off a concept-graph mismatch,
            # Infer's off free-text signal in the turn -- would risk
            # duplicate/near-duplicate hypotheses about the same belief
            # with no reconciliation between them. Infer already runs
            # every turn and is the one place already responsible for
            # "does this turn suggest a new belief worth tracking";
            # narrowing that decision to a single call path keeps
            # HypothesisStore's population inspectable from one place.
            #
            # Residual, narrower gap left open by this decision (not
            # fixed here): if `hypotheses` is empty -- no MENTAL_MODEL/
            # ACTIVE hypothesis yet exists for this concept -- a
            # LEARNER_MISCONCEPTION mismatch reweights nothing and
            # `action_taken` stays "none"; the raw `mismatch` fields
            # are still recorded on this turn's result, so the signal
            # isn't invisible, just not durably tracked as a hypothesis
            # yet. That's a smaller, separate gap for Infer's own
            # create path to eventually pick up on a later turn, not
            # something this fix silently papers over.
            reweighted_ids: list[str] = []
            for hyp in hypotheses:
                evidence_ref = EvidenceRef(
                    turn_id=turn_id, polarity=Polarity.SUPPORTING
                )
                new_probability = hyp.probability + mismatch.confidence * (
                    1.0 - hyp.probability
                )
                new_confidence = hyp.confidence + mismatch.confidence * (
                    1.0 - hyp.confidence
                )
                await self._hyp.reweight(
                    hyp.id,
                    min(1.0, new_probability),
                    min(1.0, new_confidence),
                    evidence_ref,
                )
                reweighted_ids.append(str(hyp.id))
            if reweighted_ids:
                result["action_taken"] = "hypothesis_reweighted"
            result["reweighted_hypothesis_ids"] = reweighted_ids

        return result


class Update:
    """Applies Infer's decisions to `HypothesisStore` — the literal
    missing link before this fix: `Infer` could decide a new belief was
    worth tracking, but nothing ever called `HypothesisStore.add()` to
    make it real. `reweight()` for existing hypotheses is unchanged;
    `add()` for `InferOutput.creates` is additive, not a replacement.
    """

    async def run(
        self,
        proposals: InferOutput,
        hypothesis_store: HypothesisStore,
    ) -> list[EvidenceRef]:
        applied: list[EvidenceRef] = []
        for prop in proposals.reweights:
            await hypothesis_store.reweight(
                prop.hypothesis_id,
                prop.new_probability,
                prop.new_confidence,
                prop.evidence_ref,
            )
            applied.append(prop.evidence_ref)
        for create in proposals.creates:
            # tier=ACTIVE, not DORMANT/BACKGROUND: a hypothesis Infer
            # just decided is worth tracking from this turn's evidence
            # needs to be visible to THIS turn's still-to-run Replan
            # call (see SessionLoop.handle_turn's refreshed_hypotheses
            # read right after Update) — starting it anywhere entropy
            # computation ignores would silently reproduce the exact
            # bug this fix exists to close.
            await hypothesis_store.add(
                Hypothesis(
                    layer=create.layer,
                    statement=create.statement,
                    probability=create.initial_probability,
                    confidence=create.initial_confidence,
                    tier=Tier.ACTIVE,
                    evidence_refs=[create.evidence_ref],
                )
            )
            applied.append(create.evidence_ref)
        return applied


class Replan:
    """Decides the next turn's reasoning budget from current uncertainty.

    Delegates entirely to `compute_reasoning_budget` (reasoning_budget.py)
    — the single source of truth for the entropy -> generation_width /
    run_information_value / exploration_target mapping. Replan itself
    holds no formula; it's just the audited call site (CLAUDE.md
    invariant 2) that feeds the result into next turn's `Infer`/`Plan`/
    `ValueFunction` via SessionLoop.
    """

    def __init__(self, config: ReasoningBudgetConfig | None = None) -> None:
        self._config = config

    async def run(self, hypotheses: list[Hypothesis]) -> ReasoningBudget:
        return compute_reasoning_budget(hypotheses, self._config)


class SessionMissingTopicError(Exception):
    """Raised by SessionLoop when a turn past the first still has no
    concept_graph_id attached — AttachTopic only ever runs on turn 0;
    if it never succeeded (or was never attempted), every later turn
    hard-fails here rather than silently teaching against no graph at
    all. No retry: the session needs a fresh topic-bearing turn 0, not
    something a later turn can fix on its own.
    """


def _parse_topic(raw: str) -> str:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(
            f"TOPIC:INFER response was not valid JSON: {raw!r}"
        ) from exc
    topic = parsed.get("topic") if isinstance(parsed, dict) else None
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError(f"TOPIC:INFER response missing a usable topic: {raw!r}")
    return topic.strip()


class AttachTopic:
    """Turn 0's topic inference: extracts a topic from the student's
    first message, resolves it against existing concept graphs
    (exact-match, same as `probe chat --topic`), and attaches whichever
    graph_id results to the session — seeding a fresh one via the
    existing `seed_graph()` when no exact match exists.

    Two tiers, since this delegates two different kinds of calls:
    `llm` (fast) for the topic extraction itself, `seed_llm` (capable)
    only for `seed_graph()` on the seed-fresh path — same tier `probe
    chat --topic`'s seeding already uses.

    Raises (ValueError, from a malformed topic response; whatever
    seed_graph/SeedGraphError raises on the seed-fresh path) rather
    than fabricating a fallback topic — a nonsense topic, once
    attached, would be stuck for the session's lifetime (set-once).
    SessionLoop's turn-0 handling is what decides what a failure here
    means for the turn, not this node.

    Concept selection ("what to teach next") is still out of scope —
    like GroundConcept, this only identifies a topic, not a curriculum.
    """

    def __init__(
        self,
        llm: LLMClient,
        seed_llm: LLMClient,
        concept_graph: ConceptGraph,
        transcript: TranscriptStore,
    ) -> None:
        self._llm = llm
        self._seed_llm = seed_llm
        self._concepts = concept_graph
        self._transcript = transcript
        # Read by SessionLoop into the MAX_CALLS_PER_TURN accounting:
        # 1 for the topic-extraction call, +1 more if the seed-fresh
        # path ran (seed_graph makes exactly one call itself).
        self.last_call_count: int = 0

    async def run(self, message: str, session_id: UUID) -> TopicAttachment:
        self.last_call_count = 0
        prompt = (
            "TOPIC:INFER\n"
            "What subject/topic is being discussed in this student "
            "message? Respond with a short, canonical topic label (a "
            "few words, consistent phrasing so the same subject "
            "matches on future sessions).\n\n"
            f"message: {message}\n"
        )
        raw = await self._llm.complete(prompt)
        self.last_call_count += 1
        topic = _parse_topic(raw)

        matches = await self._concepts.find_graphs_by_topic(topic)
        if matches:
            # find_graphs_by_topic already orders by created_at —
            # resume the most recently seeded graph for this topic.
            graph_meta = matches[-1]
            seeded_new = False
        else:
            concept_graph_id, _concepts = await seed_graph(
                self._seed_llm, self._concepts, topic
            )
            self.last_call_count += 1
            graph_meta = await self._concepts.get_graph(concept_graph_id)
            assert graph_meta is not None  # just inserted, in the same call
            seeded_new = True

        await self._transcript.attach_concept_graph_id(session_id, graph_meta.id)

        return TopicAttachment(
            topic=topic, concept_graph_id=graph_meta.id, seeded_new=seeded_new
        )
