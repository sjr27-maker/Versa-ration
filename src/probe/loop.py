"""The session loop, after the full Diagnose/Infer/Update/Replan/Plan
reasoning path and the tree-based branch system were removed. Two
architectures remain, chosen once per session by
`ablation_config.mode` (ablation.py):

- `SessionMode.MINIMAL_BRANCH` (the default) -> `_handle_disambiguation_turn`:
  the minimal three-call flow (disambiguate.py) — AssessAndBranch ->
  [DisambiguationOptions] -> FinalAnswer, at most three LLM calls per
  exchange — wrapped by the memory layer (memory.py): a semantic
  pre-check that can skip branching entirely when a past fact for this
  learner already resolves the message, and a per-turn fact write.
- `SessionMode.BASELINE` (`ablation_config.is_full_bypass`) ->
  `_handle_bypass_turn`: one plain-LLM call per turn, no scaffolding at
  all. The floor MINIMAL_BRANCH is measured against.

All node invocations flow through `_call_node`, which records inputs
and outputs to `node_calls` per CLAUDE.md invariant 2. `turn_diagnostics`
(diagnostics.py) is written once per turn, opt-in via `diagnostics_store`.

`consolidate_session` (memory.py steps 6-8) is background-only — see its
docstring — driven by `probe consolidate-session`, `run_interactive`'s
turn-count-gated auto-trigger on exit, and the web UI's explicit
"End session & consolidate" button, never by `handle_turn` itself.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any
from uuid import UUID

from probe.ablation import AblationConfig
from probe.audit import NodeCallStore, TranscriptStore
from probe.baseline import MAX_CALLS_PER_TURN, BaselineTeach
from probe.diagnostics import TurnDiagnosticsStore
from probe.disambiguate import (
    AssessAndBranch,
    DisambiguationOptions,
    DisambiguationStore,
    FinalAnswer,
    build_typed_past_note,
)
from probe.embeddings import EmbeddingClient
from probe.llm import LLMClient, ModelTierClients
from probe.memory import (
    ConfirmFactMatch,
    ConfirmThinkingStyleMatch,
    EmbedAndSearchFacts,
    LearnerFactStore,
    MemoryConfig,
    SummarizeSessionPath,
    ThinkingStyleStore,
    WriteLearnerFact,
)
from probe.models import (
    BranchStatus,
    DisambiguationAssessment,
    DisambiguationBranch,
    ExtractedFact,
    FactMatchConfirmation,
    FactSearchResult,
    LearnerFactType,
    Option,
    OptionStatus,
    TurnDiagnostics,
)

logger = logging.getLogger(__name__)

_TEACH_FAILURE_MESSAGE = (
    "the tutor failed to respond this turn — see Diagnostics for the "
    "error; try sending your message again"
)

# The compact recent-history window AssessAndBranch / FinalAnswer share
# (see _build_disambiguation_history) — deliberately small: it exists
# for conversation continuity (resolving a bare "that"/"it"), not to
# re-derive the whole session.
_HISTORY_TURNS = 3


def _total_retry_count(tiers: ModelTierClients) -> int:
    """Sum of GeminiLLMClient.retry_count across every distinct client
    in `tiers`, deduplicated by identity. Dedup matters because the
    default (no explicit model_tier_clients) is the *same* LLMClient
    instance for fast/capable/best. getattr(..., 0) makes this 0 for
    StubLLMClient without either client type needing to know about the
    other. Snapshotted before/after each turn; the delta is that turn's
    retry_count."""
    seen: set[int] = set()
    total = 0
    for client in (tiers.fast, tiers.capable, tiers.best):
        if id(client) in seen:
            continue
        seen.add(id(client))
        total += getattr(client, "retry_count", 0)
    return total


class SessionLoop:
    def __init__(
        self,
        transcript: TranscriptStore,
        node_calls: NodeCallStore,
        llm: LLMClient,
        model_tier_clients: ModelTierClients | None = None,
        diagnostics_store: TurnDiagnosticsStore | None = None,
        on_node_start: Callable[[str], None] | None = None,
        ablation_config: AblationConfig | None = None,
        disambiguation_store: DisambiguationStore | None = None,
        learner_fact_store: LearnerFactStore | None = None,
        thinking_style_store: ThinkingStyleStore | None = None,
        embedding_client: EmbeddingClient | None = None,
        memory_config: MemoryConfig | None = None,
    ) -> None:
        # Tiering (fast/capable/best -> real Gemini models, see
        # model_config.py) is opt-in via model_tier_clients. Omitting it
        # reproduces the pre-tiering behavior exactly: every node gets
        # the single `llm` argument.
        tiers = model_tier_clients or ModelTierClients(fast=llm, capable=llm, best=llm)
        self._tiers = tiers
        self._transcript = transcript
        self._node_calls = node_calls
        self._diagnostics = diagnostics_store
        self._on_node_start = on_node_start
        self._ablation = ablation_config or AblationConfig()

        # The plain-LLM BASELINE — same tier as FinalAnswer, since
        # that's what it's compared against. Cheap to construct
        # unconditionally (no store dependency at all).
        self.baseline_teach = BaselineTeach(tiers.best)

        # disambiguate.py's minimal three-call flow. Fast tier for the
        # two judgment calls (a narrow structural judgment, not the
        # final response); best tier for FinalAnswer, since it is the
        # response the student actually sees.
        self._disambiguation = disambiguation_store
        self.assess_and_branch = AssessAndBranch(tiers.fast)
        self.disambiguation_options = DisambiguationOptions(tiers.fast)
        self.final_answer = FinalAnswer(tiers.best)
        # The most recent AssessAndBranch-generating turn's id, if its
        # branches are still unresolved. None whenever the last turn was
        # a click resolution, a direct answer, or has already been
        # superseded by a later typed-past turn.
        self._prior_disambiguation_turn_id: UUID | None = None

        # The memory layer (memory.py) — additive on top of
        # minimal_branch, never required by it. `_memory_enabled` gates
        # the per-turn semantic pre-check / fact-writing (needs both a
        # place to search/write facts and something to embed with);
        # `_thinking_styles` (the cross-session layer) is independently
        # optional on top of that — consolidate_session needs it, but
        # nothing per-turn does except reading already-`confirmed`
        # candidates for AssessAndBranch's prompt (see
        # `_build_thinking_style_hint`).
        self._learner_facts = learner_fact_store
        self._thinking_styles = thinking_style_store
        self._embedding_client = embedding_client
        self._memory_config = memory_config or MemoryConfig()
        self._memory_enabled = (
            learner_fact_store is not None and embedding_client is not None
        )
        if self._memory_enabled:
            self.embed_and_search_facts = EmbedAndSearchFacts(
                embedding_client, learner_fact_store, self._memory_config
            )
            self.confirm_fact_match = ConfirmFactMatch(tiers.fast)
            self.write_learner_fact = WriteLearnerFact(
                tiers.fast, embedding_client, learner_fact_store
            )
        else:
            self.embed_and_search_facts = None
            self.confirm_fact_match = None
            self.write_learner_fact = None
        # Background-only (see consolidate_session) — needs the fact
        # store, the thinking-style store, and an embedding client.
        if (
            learner_fact_store is not None
            and thinking_style_store is not None
            and embedding_client is not None
        ):
            self.summarize_session_path = SummarizeSessionPath(tiers.fast)
            self.confirm_thinking_style_match = ConfirmThinkingStyleMatch(tiers.fast)
        else:
            self.summarize_session_path = None
            self.confirm_thinking_style_match = None

    async def run_interactive(self, learner_id: UUID) -> UUID:
        # ablation_config is fixed for a session's lifetime (set-once,
        # see TranscriptStore.set_ablation_config) — pass this loop's
        # own config through explicitly so the persisted row always
        # matches what actually ran.
        session_id = await self._transcript.create_session(
            learner_id, ablation_config=self._ablation
        )
        print(f"probe: new session {session_id}")
        print("probe: type your message. ctrl-D or empty line + ctrl-C to exit.")
        turn_index = 0
        loop = asyncio.get_running_loop()
        while True:
            try:
                turn_text = await loop.run_in_executor(None, input, "you: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            turn_text = turn_text.strip()
            if not turn_text:
                continue
            message = await self.handle_turn(session_id, turn_index, turn_text)
            print(f"probe: {message}")
            turn_index += 1

        # Auto-consolidation on interactive exit (memory.py steps 6-8),
        # gated by turn count: a session below
        # MemoryConfig.min_turns_for_cli_auto_consolidation is not
        # eligible at all. No-ops harmlessly (via consolidate_session's
        # own guards) for a BASELINE session (nothing in learner_facts
        # to consolidate) or when the memory layer isn't configured.
        if turn_index >= self._memory_config.min_turns_for_cli_auto_consolidation:
            result = await self.consolidate_session(session_id)
            if result is not None:
                print(
                    f"probe: consolidated this session's path — "
                    f"thinking-style candidate {result.id} now has "
                    f"confirmation_count={result.confirmation_count} "
                    f"(status={result.status.value})"
                )
        return session_id

    async def handle_turn(
        self,
        session_id: UUID,
        turn_index: int,
        turn_text: str,
        selected_option_id: UUID | None = None,
    ) -> str:
        if self._ablation.is_full_bypass:
            # SessionMode.BASELINE — plain LLM, one call, no scaffolding.
            return await self._handle_bypass_turn(session_id, turn_index, turn_text)
        return await self._handle_disambiguation_turn(
            session_id, turn_index, turn_text, selected_option_id
        )

    async def _handle_bypass_turn(
        self, session_id: UUID, turn_index: int, turn_text: str
    ) -> str:
        """SessionMode.BASELINE's turn: student message -> one LLM call
        (with prior turns in this session as context) -> response. Every
        store touched here is the transcript (turns + the one node_calls
        row for the call itself, plus turn_diagnostics if configured)
        and nothing else. This is what makes the comparison honest —
        MINIMAL_BRANCH's cost is measured against this floor, so this
        floor must not itself carry any of the loop's overhead.
        """
        start = time.monotonic()
        retry_count_start = _total_retry_count(self._tiers)

        await self._transcript.record_turn(session_id, turn_index, turn_text)
        prior_turns = [
            t.text
            for t in await self._transcript.list_turns(session_id)
            if t.turn_index < turn_index
        ]

        teach_failed = False
        warnings: list[str] = []
        try:
            message = await self._call_node(
                self.baseline_teach,
                session_id,
                turn_index,
                turn_text=turn_text,
                prior_turns=prior_turns,
            )
            call_count = self.baseline_teach.last_call_count
        except Exception as exc:
            teach_failed = True
            message = _TEACH_FAILURE_MESSAGE
            call_count = 0
            warnings.append(f"BaselineTeach failed: {exc}")
            logger.warning(
                "BaselineTeach failed on turn %d for session %s: %s",
                turn_index,
                session_id,
                exc,
                exc_info=True,
            )

        if self._diagnostics is not None:
            await self._diagnostics.record(
                TurnDiagnostics(
                    session_id=session_id,
                    turn_index=turn_index,
                    node_call_counts={"BaselineTeach": call_count},
                    total_call_count=call_count,
                    guardrail_fired=False,
                    entropy_bits=None,
                    duration_ms=(time.monotonic() - start) * 1000,
                    warnings=warnings,
                    teach_failed=teach_failed,
                    retry_count=_total_retry_count(self._tiers) - retry_count_start,
                )
            )
        return message

    async def _handle_disambiguation_turn(
        self,
        session_id: UUID,
        turn_index: int,
        turn_text: str,
        selected_option_id: UUID | None = None,
    ) -> str:
        """SessionMode.MINIMAL_BRANCH's turn — see disambiguate.py's
        module docstring for the full flow."""
        assert self._disambiguation is not None, (
            "SessionMode.MINIMAL_BRANCH requires disambiguation_store"
        )
        start = time.monotonic()
        retry_count_start = _total_retry_count(self._tiers)
        warnings: list[str] = []
        node_call_counts: dict[str, int] = {}

        turn_id = await self._transcript.record_turn(session_id, turn_index, turn_text)
        learner_id = await self._transcript.get_learner_id(session_id)

        # Same compact recent-history window AssessAndBranch gets --
        # computed once, up front, so every call site into FinalAnswer
        # below threads the identical context AssessAndBranch itself
        # judged against.
        recent_history = await self._build_disambiguation_history(session_id, turn_index)

        # 3a: a click resolves immediately -- no AssessAndBranch this
        # turn, the reading is already known, and no memory pre-check
        # applies either. An unrecognized/stale option id falls through
        # to the normal flow below.
        if selected_option_id is not None:
            option = await self._disambiguation.get_option(selected_option_id)
            if option is not None:
                await self._disambiguation.set_option_status(
                    option.id, OptionStatus.SELECTED
                )
                branch = await self._disambiguation.mark_matched(option.branch_id)
                sibling_branches = await self._disambiguation.list_branches_for_turn(
                    branch.disambiguation_turn_id
                )
                await self._disambiguation.supersede_open_branches(
                    branch.disambiguation_turn_id, exclude_ids=[branch.id]
                )
                await self._disambiguation.supersede_open_options(option.generation_id)
                self._prior_disambiguation_turn_id = None
                message, teach_failed = await self._finish_turn_with_fact(
                    session_id, turn_index, turn_text, turn_id, learner_id,
                    branch_context=branch.statement,
                    memory_context=None,
                    recent_history=recent_history,
                    fact_type=LearnerFactType.BRANCH_RESOLUTION,
                    branch_statements=[b.statement for b in sibling_branches],
                    node_call_counts=node_call_counts,
                    warnings=warnings,
                )
                await self._record_disambiguation_diagnostics(
                    session_id, turn_index, node_call_counts, warnings,
                    teach_failed, start, retry_count_start,
                )
                return message

        # 3b: nothing was clicked. If the immediately preceding
        # AssessAndBranch turn still has unresolved (open) branches, the
        # student typed past them -- no attempt is made to match the
        # typed text against that old set; it is superseded outright and
        # threaded into THIS turn's own fresh AssessAndBranch call as
        # context instead.
        typed_past_note = ""
        prior_turn_id = self._prior_disambiguation_turn_id
        self._prior_disambiguation_turn_id = None
        if prior_turn_id is not None:
            prior_branches = await self._disambiguation.list_branches_for_turn(
                prior_turn_id
            )
            open_branches = [b for b in prior_branches if b.status is BranchStatus.OPEN]
            if open_branches:
                typed_past_note = build_typed_past_note(open_branches)
                await self._disambiguation.supersede_open_branches(prior_turn_id)
                await self._disambiguation.supersede_open_options(prior_turn_id)
                warnings.append(
                    "disambiguation_typed_past: student typed past the "
                    "prior turn's options -- superseded, threaded as "
                    "context into this turn's AssessAndBranch instead of "
                    "being matched against"
                )

        # Semantic pre-check (memory.py steps 3-4): does a past fact for
        # THIS learner -- possibly from an earlier session -- already
        # resolve this exact message? Vector similarity alone never
        # decides this; only a confirmed "yes" is allowed to skip
        # AssessAndBranch entirely.
        memory_match_found = False
        memory_match_confirmed = False
        matched_fact_id: UUID | None = None
        memory_context: str | None = None
        if self._memory_enabled:
            search_result = await self._call_node_or_warn(
                self.embed_and_search_facts,
                session_id,
                turn_index,
                "EmbedAndSearchFacts",
                FactSearchResult(),
                warnings,
                learner_id=learner_id,
                message=turn_text,
            )
            node_call_counts["EmbedAndSearchFacts"] = (
                self.embed_and_search_facts.last_call_count
            )
            if search_result.matched_fact_id is not None:
                memory_match_found = True
                matched_fact_id = search_result.matched_fact_id
                confirmation = await self._call_node_or_warn(
                    self.confirm_fact_match,
                    session_id,
                    turn_index,
                    "ConfirmFactMatch",
                    FactMatchConfirmation(resolves=False),
                    warnings,
                    matched_situation=search_result.situation,
                    matched_resolution=search_result.resolution,
                    current_message=turn_text,
                )
                node_call_counts["ConfirmFactMatch"] = (
                    self.confirm_fact_match.last_call_count
                )
                if confirmation.resolves:
                    memory_match_confirmed = True
                    memory_context = (
                        f"{search_result.situation} -- {search_result.resolution}"
                    )

        if memory_match_confirmed:
            # AssessAndBranch never runs this turn -- no DisambiguationTurn
            # row is created for it either (there is nothing it would
            # record beyond what turn_diagnostics already makes visible).
            message, teach_failed = await self._finish_turn_with_fact(
                session_id, turn_index, turn_text, turn_id, learner_id,
                branch_context=None,
                memory_context=memory_context,
                recent_history=recent_history,
                fact_type=LearnerFactType.DIRECT_ANSWER,
                branch_statements=None,
                node_call_counts=node_call_counts,
                warnings=warnings,
            )
            await self._record_disambiguation_diagnostics(
                session_id, turn_index, node_call_counts, warnings,
                teach_failed, start, retry_count_start,
                memory_match_found=memory_match_found,
                memory_match_confirmed_resolution=memory_match_confirmed,
                branching_skipped_by_memory=True,
                matched_fact_id=matched_fact_id,
            )
            return message

        thinking_style_hint = await self._build_thinking_style_hint(learner_id)
        assessment = await self._call_node_or_warn(
            self.assess_and_branch,
            session_id,
            turn_index,
            "AssessAndBranch",
            DisambiguationAssessment(needs_branches=False, branch_statements=[]),
            warnings,
            message=turn_text,
            recent_history=recent_history,
            typed_past_note=typed_past_note,
            thinking_style_hint=thinking_style_hint,
        )
        node_call_counts["AssessAndBranch"] = self.assess_and_branch.last_call_count

        if not assessment.needs_branches:
            # Persisted unconditionally -- a turn judged unambiguous is a
            # queryable row, not a gap.
            await self._disambiguation.create_turn(
                session_id, turn_index, needs_branches=False, turn_had_direct_answer=True
            )
            message, teach_failed = await self._finish_turn_with_fact(
                session_id, turn_index, turn_text, turn_id, learner_id,
                branch_context=None,
                memory_context=None,
                recent_history=recent_history,
                fact_type=LearnerFactType.DIRECT_ANSWER,
                branch_statements=None,
                node_call_counts=node_call_counts,
                warnings=warnings,
            )
            await self._record_disambiguation_diagnostics(
                session_id, turn_index, node_call_counts, warnings,
                teach_failed, start, retry_count_start,
                memory_match_found=memory_match_found,
                matched_fact_id=matched_fact_id,
            )
            return message

        disamb_turn = await self._disambiguation.create_turn(
            session_id, turn_index, needs_branches=True, turn_had_direct_answer=False
        )
        branches = [
            DisambiguationBranch(
                disambiguation_turn_id=disamb_turn.id,
                session_id=session_id,
                turn_index=turn_index,
                statement=statement,
            )
            for statement in assessment.branch_statements
        ]
        await self._disambiguation.add_branches(branches)

        proposals = await self._call_node_or_warn(
            self.disambiguation_options,
            session_id,
            turn_index,
            "DisambiguationOptions",
            [],
            warnings,
            branches=branches,
        )
        node_call_counts["DisambiguationOptions"] = (
            self.disambiguation_options.last_call_count
        )

        if not proposals:
            # DisambiguationOptions produced nothing usable -- graceful
            # degrade: answer directly rather than show broken/empty
            # buttons or drop the turn.
            warnings.append(
                "disambiguation_options_empty: the message was judged "
                "ambiguous but no usable options were generated -- "
                "answering directly instead of showing nothing"
            )
            message, teach_failed = await self._finish_turn_with_fact(
                session_id, turn_index, turn_text, turn_id, learner_id,
                branch_context=None,
                memory_context=None,
                recent_history=recent_history,
                fact_type=LearnerFactType.DIRECT_ANSWER,
                branch_statements=None,
                node_call_counts=node_call_counts,
                warnings=warnings,
            )
            await self._record_disambiguation_diagnostics(
                session_id, turn_index, node_call_counts, warnings,
                teach_failed, start, retry_count_start,
                memory_match_found=memory_match_found,
                matched_fact_id=matched_fact_id,
            )
            return message

        new_options = [
            Option(
                branch_id=p.branch_id,
                generation_id=disamb_turn.id,
                session_id=session_id,
                turn_index=turn_index,
                text=p.text,
            )
            for p in proposals
        ]
        await self._disambiguation.create_options(new_options)
        self._prior_disambiguation_turn_id = disamb_turn.id

        # No FinalAnswer this turn -- options are shown INSTEAD of an
        # answer. Nothing was resolved yet, so no fact is written
        # either. The option texts are read by the caller (web UI/CLI)
        # from DisambiguationStore.list_options_for_turn.
        message = "Which of these did you mean?"
        await self._record_disambiguation_diagnostics(
            session_id, turn_index, node_call_counts, warnings,
            teach_failed=False, start=start, retry_count_start=retry_count_start,
            memory_match_found=memory_match_found,
            matched_fact_id=matched_fact_id,
        )
        return message

    async def _finish_turn_with_fact(
        self,
        session_id: UUID,
        turn_index: int,
        turn_text: str,
        turn_id: UUID,
        learner_id: UUID,
        branch_context: str | None,
        memory_context: str | None,
        recent_history: str,
        fact_type: LearnerFactType,
        branch_statements: list[str] | None,
        node_call_counts: dict[str, int],
        warnings: list[str],
    ) -> tuple[str, bool]:
        """Every path through `_handle_disambiguation_turn` that
        actually resolves something (a click, a memory-confirmed
        shortcut, an unambiguous direct answer, or the empty-options
        fallback) ends here: run FinalAnswer, then — memory.py step 5 —
        write exactly one fact recording what just happened, unless
        FinalAnswer itself failed (nothing real to record then)."""
        message, teach_failed = await self._run_final_answer(
            session_id, turn_index, turn_text, branch_context, recent_history,
            node_call_counts, warnings, memory_context=memory_context,
        )
        if not teach_failed and self._memory_enabled:
            await self._call_node_or_warn(
                self.write_learner_fact,
                session_id,
                turn_index,
                "WriteLearnerFact",
                ExtractedFact(situation="", resolution=""),
                warnings,
                fact_type=fact_type,
                learner_id=learner_id,
                session_id=session_id,
                turn_index=turn_index,
                source_turn_id=turn_id,
                student_message=turn_text,
                tutor_message=message,
                branch_statements=branch_statements,
            )
            node_call_counts["WriteLearnerFact"] = self.write_learner_fact.last_call_count
        return message, teach_failed

    async def _run_final_answer(
        self,
        session_id: UUID,
        turn_index: int,
        turn_text: str,
        branch_context: str | None,
        recent_history: str,
        node_call_counts: dict[str, int],
        warnings: list[str],
        memory_context: str | None = None,
    ) -> tuple[str, bool]:
        """FinalAnswer has no fallback -- its output IS the turn. A
        failure here degrades to a fixed in-band message and is the
        caller's concern, not this method's.

        `recent_history` is the same window `AssessAndBranch` was given
        this turn — FinalAnswer must not judge a poorer context than
        AssessAndBranch already reasoned against."""
        try:
            message = await self._call_node(
                self.final_answer,
                session_id,
                turn_index,
                student_message=turn_text,
                branch_context=branch_context,
                recent_history=recent_history,
                memory_context=memory_context,
            )
            node_call_counts["FinalAnswer"] = self.final_answer.last_call_count
            return message, False
        except Exception as exc:
            warnings.append(f"FinalAnswer failed: {exc}")
            logger.warning(
                "FinalAnswer failed on turn %d for session %s: %s",
                turn_index,
                session_id,
                exc,
                exc_info=True,
            )
            node_call_counts["FinalAnswer"] = 0
            return _TEACH_FAILURE_MESSAGE, True

    async def _build_thinking_style_hint(self, learner_id: UUID) -> str:
        """Step 8's "only once promoted does it get fed into future
        sessions' prompts" — the one place `AssessAndBranch`'s prompt
        input is built from `thinking_style_candidates`, and the only
        read it is allowed to use (`list_confirmed_for_prompt`
        structurally excludes anything not yet `confirmed`). Empty
        string whenever this layer is off or nothing has been promoted
        yet for this learner."""
        if self._thinking_styles is None:
            return ""
        confirmed = await self._thinking_styles.list_confirmed_for_prompt(learner_id)
        if not confirmed:
            return ""
        return "; ".join(c.path_summary for c in confirmed)

    async def consolidate_session(self, session_id: UUID):
        """Steps 6-8 of memory.py's flow — background-only, by design
        never called from `handle_turn`: labeling and comparing an
        ENTIRE session's order-structure only means something once the
        session actually has one, and doing it mid-turn would mean
        re-doing the same expensive comparison every single turn for no
        new information most of the time. Callers: cli.py's standalone
        `probe consolidate-session` command, `run_interactive`'s own
        turn-count-gated auto-trigger on exit, and the web UI's explicit
        "End session & consolidate" button — never SessionLoop itself.

        Returns None when there is nothing to consolidate (the
        thinking-style layer isn't configured, or this session wrote no
        facts at all). Otherwise returns the `ThinkingStyleCandidate`
        this session ended up confirming (which may have just been
        promoted to `confirmed`) or newly created.
        """
        if (
            self.summarize_session_path is None
            or self.confirm_thinking_style_match is None
            or self._learner_facts is None
            or self._thinking_styles is None
            or self._embedding_client is None
        ):
            return None

        facts = await self._learner_facts.list_by_session(session_id)
        if not facts:
            return None

        learner_id = await self._transcript.get_learner_id(session_id)
        turns = await self._transcript.list_turns(session_id)
        last_turn_index = max((t.turn_index for t in turns), default=0)

        path_summary = await self._call_node(
            self.summarize_session_path, session_id, last_turn_index, facts=facts,
        )
        embedding = await self._embedding_client.embed(path_summary.summary)

        nearest = await self._thinking_styles.search_similar(learner_id, embedding, limit=1)
        if nearest:
            candidate, similarity = nearest[0]
            if similarity >= self._memory_config.thinking_style_similarity_threshold:
                confirmation = await self._call_node(
                    self.confirm_thinking_style_match,
                    session_id,
                    last_turn_index,
                    existing_path_summary=candidate.path_summary,
                    new_path_summary=path_summary.summary,
                )
                if confirmation.confirms:
                    return await self._thinking_styles.confirm(
                        candidate.id,
                        session_id,
                        promotion_threshold=self._memory_config.thinking_style_promotion_threshold,
                    )

        # No existing candidate was even worth asking about, or the
        # confirmation call said the resemblance was only superficial --
        # either way, this session's own labeled path becomes a brand
        # new candidate (confirmation_count=1).
        return await self._thinking_styles.create_candidate(
            learner_id, session_id, path_summary.summary, embedding,
        )

    async def _build_disambiguation_history(
        self, session_id: UUID, turn_index: int
    ) -> str:
        """Compact recent-history input shared by `AssessAndBranch` and
        `FinalAnswer` (computed once per turn and threaded into both) —
        the last few student turns paired with FinalAnswer's own
        responses to them, read back out of node_calls (invariant 2
        already guarantees FinalAnswer's rendered output is durable)."""
        if turn_index == 0:
            return ""
        turns = [
            t
            for t in await self._transcript.list_turns(session_id)
            if t.turn_index < turn_index
        ][-_HISTORY_TURNS:]
        answer_calls = await self._node_calls.get_recent_calls(
            session_id, "FinalAnswer", turn_index, _HISTORY_TURNS
        )
        answers_by_turn = {c.turn_index: c.output_json for c in answer_calls}
        lines: list[str] = []
        for t in turns:
            lines.append(f"turn {t.turn_index} student: {t.text}")
            answer = answers_by_turn.get(t.turn_index)
            if answer:
                lines.append(f"turn {t.turn_index} tutor: {answer}")
        return "\n".join(lines)

    async def _record_disambiguation_diagnostics(
        self,
        session_id: UUID,
        turn_index: int,
        node_call_counts: dict[str, int],
        warnings: list[str],
        teach_failed: bool,
        start: float,
        retry_count_start: int,
        memory_match_found: bool = False,
        memory_match_confirmed_resolution: bool = False,
        branching_skipped_by_memory: bool = False,
        matched_fact_id: UUID | None = None,
    ) -> None:
        if self._diagnostics is None:
            return
        total_call_count = sum(node_call_counts.values())
        guardrail_fired = total_call_count > MAX_CALLS_PER_TURN
        if guardrail_fired:
            warnings.append(
                f"MAX_CALLS_PER_TURN exceeded: {total_call_count} calls "
                f"(limit {MAX_CALLS_PER_TURN})"
            )
        if branching_skipped_by_memory:
            # Must be visible and auditable per turn, never a silent
            # shortcut (see memory.py's module docstring).
            warnings.append(
                f"branching_skipped_by_memory: fact {matched_fact_id} was "
                "confirmed to resolve this message -- AssessAndBranch was "
                "never called this turn"
            )
        await self._diagnostics.record(
            TurnDiagnostics(
                session_id=session_id,
                turn_index=turn_index,
                node_call_counts=node_call_counts,
                total_call_count=total_call_count,
                guardrail_fired=guardrail_fired,
                entropy_bits=None,
                duration_ms=(time.monotonic() - start) * 1000,
                warnings=warnings,
                teach_failed=teach_failed,
                retry_count=_total_retry_count(self._tiers) - retry_count_start,
                memory_match_found=memory_match_found,
                memory_match_confirmed_resolution=memory_match_confirmed_resolution,
                branching_skipped_by_memory=branching_skipped_by_memory,
                matched_fact_id=matched_fact_id,
            )
        )

    async def _call_node_or_warn(
        self,
        node: Any,
        session_id: UUID,
        turn_index: int,
        label: str,
        fallback: Any,
        warnings: list[str],
        /,
        **kwargs: Any,
    ) -> Any:
        """Same as `_call_node`, except a raised exception is caught,
        recorded into `warnings`, and swallowed in favor of `fallback`
        — so one node's transient failure doesn't lose the whole turn.
        Not used for FinalAnswer, which has no safe fallback (see
        `_run_final_answer`'s dedicated try/except)."""
        try:
            return await self._call_node(node, session_id, turn_index, **kwargs)
        except Exception as exc:
            warnings.append(f"{label} failed: {exc}")
            logger.warning(
                "%s failed on turn %d for session %s: %s",
                label,
                turn_index,
                session_id,
                exc,
                exc_info=True,
            )
            return fallback

    async def _call_node(
        self,
        node: Any,
        session_id: UUID,
        turn_index: int,
        /,
        **kwargs: Any,
    ) -> Any:
        # session_id/turn_index/node are positional-only so a node's own
        # run() kwargs can share a name with them without colliding.
        if self._on_node_start is not None:
            self._on_node_start(type(node).__name__)
        output = await node.run(**kwargs)
        await self._node_calls.record(
            node_name=type(node).__name__,
            session_id=session_id,
            turn_index=turn_index,
            input_json=kwargs,
            output_json=output,
        )
        return output
