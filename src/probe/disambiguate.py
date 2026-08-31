"""Minimal three-call disambiguation flow — a new, separate reasoning
mode (`AblationConfig.reasoning_mode = ReasoningMode.DISAMBIGUATE`,
wired in via `SessionLoop._handle_disambiguation_turn`) that replaces
the branch tree / SelectBranch / DerivePath / Plan / concept-graph
machinery outright for a session run this way. Not an ablation of the
full system — a deliberately shallow, separate architecture, measured
against it, exactly the way `AblationConfig.is_full_bypass`'s
plain-LLM baseline is.

At most three LLM calls to fully resolve one exchange with the
student, spread across up to two `handle_turn` invocations — never
more than two calls in a single one:

1. `AssessAndBranch` — is the student's message genuinely ambiguous or
   under-specified? If not: proceed straight to `FinalAnswer` with no
   branch context, in the SAME turn (2 calls total: assess, then
   answer). If so: propose 2-4 distinct plausible readings of what the
   student means, wants, or is asking. Every call is persisted
   (`DisambiguationTurn` + `DisambiguationBranch` rows) unconditionally,
   whether or not `needs_branches` fires — a turn judged unambiguous is
   a queryable row with zero branches, not a gap in the record.
2. `GenerateOptions` — one clickable option per branch, generated only
   when `needs_branches` fired. Options are shown to the student
   INSTEAD of an answer this turn: no `FinalAnswer` call happens yet.
   This turn costs 2 calls (assess + options) and produces no answer.
3. `FinalAnswer` — runs on a LATER turn, once the student has resolved
   which reading they meant:
   - a click: that option's branch is marked matched, every sibling
     branch (and its option) in the same generation is superseded, and
     `FinalAnswer` runs immediately with the matched branch's statement
     as context. `AssessAndBranch` does NOT run this turn — the
     reading is already known. 1 call, one turn.
   - typed text instead of a click: no attempt is made to match the
     typed text against the old branches (that generation's branches
     are superseded, unmatched, exactly as if nothing had been
     offered) — this is a deliberate choice, not a missing feature; a
     free-text match here would just be RESOLVE:MATCH's interpretation
     layer smuggled back into a mode built specifically to avoid it.
     Instead the typed message is treated as this turn's own new
     input to `AssessAndBranch`, called fresh, with a note that the
     student typed past offered options threaded into its prompt (see
     `_TYPED_PAST_NOTE_TEMPLATE`) — their own words become one more
     input to what they meant, not a discarded signal. This turn then
     costs 2 (unambiguous: assess + answer) or 2 (ambiguous again:
     assess + options, no answer yet) calls by the same logic as any
     other fresh message.

So a full branching exchange costs 3 calls total (1 to assess + 1 to
raise options, then 1 more once resolved by a click) — matching this
feature's "3 (branching)" framing exactly, at the level of one full
exchange with the student. A fully direct exchange costs 2 (assess,
then answer), not the "1 (direct)" the originating spec named.

This is a deliberate, disclosed departure from that literal number,
not an oversight: `AssessAndBranch` must run on every fresh message to
even know a message is unambiguous in the first place — there is no
way to reach "1 call for the direct case" without either (a) skipping
the ambiguity check for some messages on a cheaper heuristic, which
reintroduces exactly the guessing this feature exists to replace with
a real judgment call, or (b) merging the ambiguity judgment and the
actual pedagogical answer into one fast-tier call, which breaks the
judgment/response tier split this codebase uses everywhere else
(ExtractRequest+Teach, GroundConcept+MismatchDetector, detect_prior_
reference+Teach — a narrow structural check is always a separate,
cheaper call from the response it gates) and would make `FinalAnswer`'s
"best tier: this is what the student actually sees" reasoning
meaningless. Two calls for a direct exchange, never more, is the
actual guarantee this module makes.

`FinalAnswer` is given the exact same compact recent-history window
`AssessAndBranch` was given this turn (see loop.py's
`_handle_disambiguation_turn`, which computes it once and threads it
into both calls). This is conversation continuity, not the portrait/
concept-graph/planner "scaffolding" this module's design deliberately
excludes — see `FinalAnswer`'s own docstring for the live failure
(an off-topic answer to a message containing a bare, context-dependent
reference — "sketch **that**") that motivated adding it: `AssessAndBranch`
correctly used its own history to judge the message unambiguous, but
`FinalAnswer` was never given that same history to resolve what the
message was actually asking about.

REUSE, not rebuild: `LLMClient`/tiering (fast for `AssessAndBranch`/
`GenerateOptions`, same tier as `ExtractRequest`/hypothesis_generator's
own `GenerateOptions` — a narrow structural judgment, not the final
response; best for `FinalAnswer`, same tier as `Teach`/`BaselineTeach`
— it IS the response); `Option`/`OptionStatus`/`OptionProposal`
(models.py, unchanged) for the click-resolve channel; `node_calls`
audit logging (CLAUDE.md invariant 2 — every one of the 1-3 calls
below is a real node, routed through `SessionLoop._call_node`).

Not literally reused: the `branches`/`options` SQL tables themselves.
`disambiguation_branches`/`disambiguation_turns`/`disambiguation_options`
(migration 029) are new, parallel tables with the reshaped, flat shape
described above — see that migration's own comment for why
`options.branch_id`'s FK to `branches(id)` makes literally sharing the
existing tables impossible, and why the existing tables must not be
altered to accommodate this new, separate mode (the full tree-based
system still depends on every column they already have).

Append-only, same discipline as branches.py/options.py (CLAUDE.md
invariant 9 for this store specifically): no delete method, no DELETE
SQL, resolution is a status transition (open -> matched/superseded)
via UPDATE only.
"""

from __future__ import annotations

import difflib
import json
import logging
from uuid import UUID, uuid4

import asyncpg

from probe.llm import LLMClient
from probe.models import (
    DisambiguationAssessment,
    DisambiguationBranch,
    DisambiguationTurn,
    Option,
    OptionProposal,
    OptionStatus,
)
from probe.row_mapping import assert_row_consumed

logger = logging.getLogger(__name__)

# One corrective retry on a rejected AssessAndBranch response (a
# duplicate reading, or a branch count outside 2-4 when
# needs_branches=True) — same shape as GenerateOptions'
# _MAX_OPTIONS_ATTEMPTS/Plan's _MAX_PROPOSE_ATTEMPTS elsewhere in this
# codebase.
_MAX_ASSESS_ATTEMPTS = 2

# One corrective retry on a rejected GenerateOptions response (a
# duplicate branch mapping, or a mapping outside the live branch set)
# — identical constant to hypothesis_generator._MAX_OPTIONS_ATTEMPTS,
# duplicated rather than imported: this module is a deliberately
# separate mode (see this module's docstring), not a consumer of the
# tree-based system's internals.
_MAX_OPTIONS_ATTEMPTS = 2

# Two readings are "the same reading restated" above this similarity —
# identical value to reasoning_budget.BranchBudgetConfig's
# redundancy_similarity_threshold default, reused for consistency
# rather than inventing a second arbitrary threshold for the same kind
# of judgment.
_REDUNDANCY_THRESHOLD = 0.8

_MIN_BRANCHES = 2
_MAX_BRANCHES = 4

_TYPED_PAST_NOTE_TEMPLATE = (
    "\nThe student was previously offered these distinct readings of "
    "an earlier ambiguous message, and typed past all of them instead "
    "of picking one:\n{prior_readings}\n"
    "Do not try to match their new message against this old list —  "
    "treat their own words below as the primary signal of what they "
    "meant, with the list above as context for what didn't land.\n"
)


def _is_duplicate_reading(statement: str, others: list[str]) -> bool:
    return any(
        difflib.SequenceMatcher(None, statement, o).ratio() >= _REDUNDANCY_THRESHOLD
        for o in others
    )


def _assess_prompt(
    message: str,
    recent_history: str = "",
    typed_past_note: str = "",
    rejected_reason: str = "",
    thinking_style_hint: str = "",
) -> str:
    history_block = (
        f"\nRecent conversation, for context:\n{recent_history}\n" if recent_history else ""
    )
    thinking_style_block = ""
    if thinking_style_hint:
        thinking_style_block = (
            "\nAcross many prior sessions, this student has confirmed "
            f"pattern(s) in how they move through material: {thinking_style_hint}\n"
            "This describes an abstract order of engagement, not a "
            "fact about the current topic — use it only to judge "
            "whether a fork you're about to raise is one this student "
            "would find genuinely ambiguous, not to assume anything "
            "about the current subject matter itself.\n"
        )
    correction = ""
    if rejected_reason:
        correction = (
            f"\nYour previous attempt was rejected: {rejected_reason}. Propose "
            f"between {_MIN_BRANCHES} and {_MAX_BRANCHES} genuinely distinct "
            "readings, no two of which restate the same idea.\n"
        )
    return (
        "ASSESS:BRANCH\n"
        f"{history_block}"
        f"{thinking_style_block}"
        f"{typed_past_note}"
        f"\nStudent's message: {message}\n\n"
        "Is this message genuinely ambiguous or under-specified -- could "
        "it reasonably mean more than one distinct thing the student "
        "wants? A concrete question, a named problem, or an unambiguous "
        "follow-up is NOT ambiguous, even if it is terse -- only flag it "
        "when there is a real fork in what they might mean.\n\n"
        f"If it is NOT ambiguous, return needs_branches=false and an "
        f"empty branches list. If it IS ambiguous, propose between "
        f"{_MIN_BRANCHES} and {_MAX_BRANCHES} distinct, plausible "
        "readings of what the student means, wants, or is asking -- "
        "genuinely different bets, not rephrasings of one idea.\n"
        f"{correction}"
        'Respond with JSON: {"needs_branches": true or false, "branches": '
        '[{"statement": "..."}, ...]}'
    )


def _parse_assessment(raw: str) -> DisambiguationAssessment | None:
    """None means "reject the whole response, regenerate" — same
    all-or-nothing discipline as hypothesis_generator's
    _parse_options_response: a malformed response or a branch count
    outside 2-4 invalidates the batch rather than being silently
    padded or truncated into something the model didn't actually say.
    """
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    needs_branches = parsed.get("needs_branches")
    if not isinstance(needs_branches, bool):
        return None
    if not needs_branches:
        return DisambiguationAssessment(needs_branches=False, branch_statements=[])

    raw_branches = parsed.get("branches")
    if not isinstance(raw_branches, list):
        return None
    statements: list[str] = []
    for item in raw_branches:
        statement = item.get("statement") if isinstance(item, dict) else None
        if not statement or not isinstance(statement, str):
            return None
        if _is_duplicate_reading(statement, statements):
            return None
        statements.append(statement)
    if not (_MIN_BRANCHES <= len(statements) <= _MAX_BRANCHES):
        return None
    return DisambiguationAssessment(needs_branches=True, branch_statements=statements)


class AssessAndBranch:
    """Step 1: is this message ambiguous, and if so, what are the
    distinct plausible readings? See module docstring for the full
    flow and why a rejected response (duplicate reading, bad count)
    gets one corrective retry before degrading to "not ambiguous"
    rather than crashing or showing a broken branch set.

    Fast tier: a narrow structural judgment about the turn's own text,
    same reasoning as ExtractRequest/hypothesis_generator's own
    GenerateOptions.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        # Read by SessionLoop into the MAX_CALLS_PER_TURN accounting.
        self.last_call_count: int = 0

    async def run(
        self,
        message: str,
        recent_history: str = "",
        typed_past_note: str = "",
        thinking_style_hint: str = "",
    ) -> DisambiguationAssessment:
        self.last_call_count = 0
        rejected_reason = ""
        for _ in range(_MAX_ASSESS_ATTEMPTS):
            raw = await self._llm.complete(
                _assess_prompt(
                    message, recent_history, typed_past_note, rejected_reason,
                    thinking_style_hint,
                )
            )
            self.last_call_count += 1
            assessment = _parse_assessment(raw)
            if assessment is not None:
                return assessment
            rejected_reason = (
                f"malformed response, or a duplicate reading, or a branch "
                f"count outside {_MIN_BRANCHES}-{_MAX_BRANCHES}"
            )
        logger.warning(
            "AssessAndBranch: exhausted %d attempt(s) with only invalid "
            "responses -- treating this turn as unambiguous rather than "
            "showing a broken branch set",
            _MAX_ASSESS_ATTEMPTS,
        )
        return DisambiguationAssessment(needs_branches=False, branch_statements=[])


def _options_prompt(candidates: list[DisambiguationBranch], rejected_reason: str = "") -> str:
    listing = "\n".join(f"- id={b.id}: {b.statement}" for b in candidates)
    hi = min(_MAX_BRANCHES, len(candidates))
    correction = ""
    if rejected_reason:
        correction = (
            f"\nYour previous attempt was rejected: {rejected_reason}. Every "
            "option must map to a DIFFERENT branch id from the list below, "
            "and every branch id used must be one of the ids listed.\n"
        )
    return (
        "DISAMBIGUATE:OPTIONS\n"
        f"The student's last message could plausibly mean any of these "
        f"distinct things:\n{listing}\n\n"
        f"Propose exactly one clickable option per reading -- between "
        f"{_MIN_BRANCHES} and {hi} options total. Each option must map "
        "to exactly ONE of the branch ids above and must be phrased as "
        "the natural next thing a tutor would say to confirm that "
        "specific reading -- a genuine continuation of the "
        "conversation, not a survey question about the student and not "
        'a bare restatement like "did you mean X."\n\n'
        "Hard rules:\n"
        "- Exactly one branch per option, exactly one claim per "
        "option -- no bundling two readings into one button.\n"
        "- Write it as a real question or statement about the subject, "
        "not a menu item or a label.\n"
        f"{correction}"
        'Respond with JSON: [{"branch_id": "<id>", "text": "..."}, ...]'
    )


def _parse_options_response(
    raw: str, valid_ids: set[UUID]
) -> list[OptionProposal] | None:
    """Same all-or-nothing discipline as
    hypothesis_generator._parse_options_response: a single duplicate or
    invalid mapping invalidates the whole batch."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, list):
        return None
    if not parsed:
        return []
    seen_branch_ids: set[UUID] = set()
    proposals: list[OptionProposal] = []
    for item in parsed:
        if not isinstance(item, dict):
            return None
        text = item.get("text")
        raw_id = item.get("branch_id")
        if not text or raw_id is None:
            return None
        try:
            branch_id = UUID(str(raw_id))
        except ValueError:
            return None
        if branch_id not in valid_ids or branch_id in seen_branch_ids:
            return None
        seen_branch_ids.add(branch_id)
        proposals.append(OptionProposal(branch_id=branch_id, text=str(text)))
    return proposals


class DisambiguationOptions:
    """Step 2: one clickable option per branch. Only ever called when
    `AssessAndBranch` produced at least one branch this turn.

    A response with any duplicate or invalid branch mapping is
    rejected wholesale and regenerated once (_MAX_OPTIONS_ATTEMPTS); if
    it still fails, this returns an empty list rather than an ambiguous
    mapping — `SessionLoop._handle_disambiguation_turn` treats that the
    same as "nothing to click," falling back to answering the original
    message directly rather than showing broken buttons or dropping the
    turn.

    Same role and validation discipline as
    hypothesis_generator.GenerateOptions, deliberately given a
    different class name (not just a different module) so
    `node_calls.node_name` never collides between the two modes — the
    tree-based system filters candidates by `requires_evidence` and
    operates on `Branch`, not `DisambiguationBranch`; keeping the audit
    trail unambiguous between them matters more here than reusing a
    name.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self.last_call_count: int = 0

    async def run(self, branches: list[DisambiguationBranch]) -> list[OptionProposal]:
        self.last_call_count = 0
        if not branches:
            return []
        valid_ids = {b.id for b in branches}
        rejected_reason = ""
        for _ in range(_MAX_OPTIONS_ATTEMPTS):
            raw = await self._llm.complete(_options_prompt(branches, rejected_reason))
            self.last_call_count += 1
            proposals = _parse_options_response(raw, valid_ids)
            if proposals is not None:
                return proposals
            rejected_reason = "duplicate branch id, or a branch id not in the live set"
        logger.warning(
            "GenerateOptions (disambiguate): exhausted %d attempt(s) with "
            "only invalid mappings -- showing no options this turn",
            _MAX_OPTIONS_ATTEMPTS,
        )
        return []


class FinalAnswer:
    """Step 4 (there is no step 3 that calls the LLM — see module
    docstring, 3a/3b are branch-resolution logic, not a call): the one
    call that actually answers the student. Has no fallback — its
    output IS the turn — mirroring Teach's own discipline; a failure
    here is `SessionLoop._handle_disambiguation_turn`'s concern, not
    this class's (same split as Teach/loop.py).

    `branch_context` is the selected reading's statement, only present
    when this turn resolved a click; None means either the original
    message was never ambiguous, or nothing was resolved to select
    from.

    `recent_history` is the exact same compact window
    `AssessAndBranch` was given this turn (see loop.py's
    `_handle_disambiguation_turn`, which computes it once and threads
    it into both calls) — NOT new scaffolding in the sense the "no
    scaffolding" design goal meant (no portrait, no concept graph, no
    planner): it is conversation continuity, the same category of
    input Teach's own `recent_history` already carries in the full
    system. Withholding it from FinalAnswer while giving it to
    AssessAndBranch was a live-confirmed bug, not a deliberate
    asymmetry: a message like "can you sketch **that**?" is judged
    unambiguous by AssessAndBranch (there is only one plausible
    *reading* of the request), but "that" is a bare reference that
    still requires the conversation so far to resolve — and
    FinalAnswer, given nothing but the bare message, is free to
    hallucinate an entirely unrelated topic to sketch. Confirmed live:
    exactly this happened (a math conversation about the chain rule
    produced a cell-membrane biology diagram) before this field
    existed.

    `memory_context` is set only on a memory-skip turn (see
    memory.py's module docstring / loop.py's `_handle_disambiguation_turn`):
    a *past* turn's fact — possibly from an earlier session entirely —
    that a `ConfirmFactMatch` call already judged resolves this
    message, so `AssessAndBranch` never ran at all this turn.
    Deliberately a separate parameter from `branch_context`, not reused
    for it: `branch_context` is this turn's own branch selection;
    `memory_context` is a durable fact from possibly a different
    session, and the two must never be conflated in the prompt (a
    student's current message being treated as if it just went through
    branching when it didn't would misrepresent what actually happened
    this turn).

    Best tier: this is what the student actually sees, same tier as
    Teach/BaselineTeach.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self.last_call_count: int = 0

    async def run(
        self,
        student_message: str,
        branch_context: str | None = None,
        recent_history: str = "",
        memory_context: str | None = None,
    ) -> str:
        self.last_call_count = 0
        context_block = ""
        if branch_context:
            context_block = (
                "\nThe student's earlier message was ambiguous; they "
                f"confirmed they meant this specific reading: {branch_context!r}. "
                "Answer accordingly -- do not re-ask which they meant.\n"
            )
        memory_block = ""
        if memory_context:
            memory_block = (
                "\nThis student previously established the following, "
                "and it directly applies to their current message -- "
                f"use it, do not ask them to re-establish it: {memory_context}\n"
            )
        history_block = ""
        if recent_history:
            history_block = (
                "\nRecent conversation, for continuity only (do not "
                "repeat this back or restate it — use it to correctly "
                "resolve any reference in the student's message below, "
                "e.g. \"that\", \"it\", or \"the one you mentioned\"):\n"
                f"{recent_history}\n"
            )
        prompt = (
            "FINAL:ANSWER\n"
            "You are a tutor having a conversation with a student. "
            "Respond directly and helpfully to their latest message.\n"
            f"{context_block}"
            f"{memory_block}"
            f"{history_block}"
            f"\nStudent's message: {student_message}\n\n"
            "Lead with the direct answer or key idea -- do not open "
            "with setup or a restatement of the question. Do not "
            "partition the response into steps or add headers/numbered "
            "lists unless the content genuinely requires that "
            "structure.\n"
            "Never end by asking how the student feels, what they "
            "prefer, or what kind of learner they are.\n"
            "Respond with plain prose only -- never wrap your answer in "
            "JSON or any other structured/markup format."
        )
        result = await self._llm.complete(prompt)
        self.last_call_count += 1
        return result


class DisambiguationStore:
    """Append-only store for this mode's own tables (migration 029).
    See CLAUDE.md invariant 9: no delete/remove method, no DELETE SQL.
    Resolution is a status transition (open -> matched/superseded) via
    UPDATE, same pattern as BranchStore/OptionStore.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create_turn(
        self, session_id: UUID, turn_index: int, needs_branches: bool, turn_had_direct_answer: bool
    ) -> DisambiguationTurn:
        turn_id = uuid4()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO disambiguation_turns
                    (id, session_id, turn_index, needs_branches, turn_had_direct_answer)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id, session_id, turn_index, needs_branches,
                          turn_had_direct_answer, created_at
                """,
                turn_id,
                session_id,
                turn_index,
                needs_branches,
                turn_had_direct_answer,
            )
        return self._row_to_turn(row)

    async def add_branches(
        self, branches: list[DisambiguationBranch]
    ) -> list[DisambiguationBranch]:
        if not branches:
            return []
        async with self._pool.acquire() as conn, conn.transaction():
            for b in branches:
                await conn.execute(
                    """
                    INSERT INTO disambiguation_branches
                        (id, disambiguation_turn_id, session_id, turn_index,
                         statement, status, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    b.id,
                    b.disambiguation_turn_id,
                    b.session_id,
                    b.turn_index,
                    b.statement,
                    b.status.value,
                    b.created_at,
                )
        return branches

    async def get_branch(self, branch_id: UUID) -> DisambiguationBranch | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM disambiguation_branches WHERE id = $1", branch_id
            )
        return self._row_to_branch(row) if row is not None else None

    async def list_branches_for_turn(
        self, disambiguation_turn_id: UUID
    ) -> list[DisambiguationBranch]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM disambiguation_branches "
                "WHERE disambiguation_turn_id = $1 ORDER BY created_at",
                disambiguation_turn_id,
            )
        return [self._row_to_branch(row) for row in rows]

    async def mark_matched(self, branch_id: UUID) -> DisambiguationBranch:
        async with self._pool.acquire() as conn:
            updated = await conn.fetchval(
                "UPDATE disambiguation_branches SET status = 'matched' "
                "WHERE id = $1 RETURNING id",
                branch_id,
            )
        if updated is None:
            raise KeyError(f"disambiguation branch {branch_id} not found")
        return await self._require_branch(branch_id)

    async def supersede_open_branches(
        self, disambiguation_turn_id: UUID, exclude_ids: list[UUID] | None = None
    ) -> int:
        exclude = exclude_ids or []
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE disambiguation_branches
                SET status = 'superseded'
                WHERE disambiguation_turn_id = $1 AND status = 'open'
                    AND NOT (id = ANY($2::uuid[]))
                """,
                disambiguation_turn_id,
                exclude,
            )
        return int(result.split()[-1])

    async def get_latest_turn(self, session_id: UUID) -> DisambiguationTurn | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM disambiguation_turns
                WHERE session_id = $1
                ORDER BY turn_index DESC, created_at DESC
                LIMIT 1
                """,
                session_id,
            )
        return self._row_to_turn(row) if row is not None else None

    async def create_options(self, options: list[Option]) -> list[Option]:
        if not options:
            return []
        async with self._pool.acquire() as conn, conn.transaction():
            for o in options:
                await conn.execute(
                    """
                    INSERT INTO disambiguation_options
                        (id, branch_id, generation_id, session_id,
                         turn_index, text, status, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    o.id,
                    o.branch_id,
                    o.generation_id,
                    o.session_id,
                    o.turn_index,
                    o.text,
                    o.status.value,
                    o.created_at,
                )
        return options

    async def get_option(self, option_id: UUID) -> Option | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM disambiguation_options WHERE id = $1", option_id
            )
        return self._row_to_option(row) if row is not None else None

    async def list_options_for_turn(self, generation_id: UUID) -> list[Option]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM disambiguation_options "
                "WHERE generation_id = $1 ORDER BY created_at",
                generation_id,
            )
        return [self._row_to_option(row) for row in rows]

    async def set_option_status(self, option_id: UUID, status: OptionStatus) -> Option:
        async with self._pool.acquire() as conn:
            updated = await conn.fetchval(
                "UPDATE disambiguation_options SET status = $2 WHERE id = $1 RETURNING id",
                option_id,
                status.value,
            )
        if updated is None:
            raise KeyError(f"disambiguation option {option_id} not found")
        return await self._require_option(option_id)

    async def supersede_open_options(self, generation_id: UUID) -> int:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE disambiguation_options SET status = 'superseded' "
                "WHERE generation_id = $1 AND status = 'open'",
                generation_id,
            )
        return int(result.split()[-1])

    async def _require_branch(self, branch_id: UUID) -> DisambiguationBranch:
        branch = await self.get_branch(branch_id)
        if branch is None:
            raise KeyError(f"disambiguation branch {branch_id} not found")
        return branch

    async def _require_option(self, option_id: UUID) -> Option:
        option = await self.get_option(option_id)
        if option is None:
            raise KeyError(f"disambiguation option {option_id} not found")
        return option

    def _row_to_turn(self, row) -> DisambiguationTurn:
        mapped = dict(row)
        assert_row_consumed(DisambiguationTurn, mapped)
        return DisambiguationTurn(**mapped)

    def _row_to_branch(self, row) -> DisambiguationBranch:
        mapped = dict(row)
        assert_row_consumed(DisambiguationBranch, mapped)
        return DisambiguationBranch(**mapped)

    def _row_to_option(self, row) -> Option:
        mapped = dict(row)
        assert_row_consumed(Option, mapped)
        return Option(**mapped)


def build_typed_past_note(prior_branches: list[DisambiguationBranch]) -> str:
    """Threads a prior turn's ignored branch set into this turn's
    AssessAndBranch call (see module docstring's "3b" case) — the
    field on the *input*, not a schema change to DisambiguationBranch
    itself, per this feature's explicit instruction that the typed-past
    signal is threaded forward via next turn's input, not stored on the
    branch."""
    if not prior_branches:
        return ""
    listing = "\n".join(f"- {b.statement}" for b in prior_branches)
    return _TYPED_PAST_NOTE_TEMPLATE.format(prior_readings=listing)
