"""SessionLoop wiring for ReasoningMode.DISAMBIGUATE — see
disambiguate.py's module docstring for the full flow this exercises at
the loop level (node-level parsing/rejection is covered by
test_disambiguate_nodes.py; raw store CRUD by test_disambiguation_store.py).
"""

import json

import pytest

from probe.ablation import AblationConfig, ReasoningMode
from probe.llm import StubLLMClient
from probe.loop import SessionLoop
from probe.models import BranchStatus, OptionStatus


def _make_loop(
    store, transcript, node_calls, concept_graph, learner_overlay, revision_store,
    disambiguation_store, llm=None, diagnostics_store=None,
):
    return SessionLoop(
        hypothesis_store=store,
        transcript=transcript,
        node_calls=node_calls,
        concept_graph=concept_graph,
        learner_overlay=learner_overlay,
        revision_store=revision_store,
        llm=llm or StubLLMClient(),
        diagnostics_store=diagnostics_store,
        ablation_config=AblationConfig(reasoning_mode=ReasoningMode.DISAMBIGUATE),
        disambiguation_store=disambiguation_store,
    )


_NOT_AMBIGUOUS = json.dumps({"needs_branches": False, "branches": []})

_TWO_BRANCHES = json.dumps(
    {
        "needs_branches": True,
        "branches": [
            {"statement": "wants the power rule explained"},
            {"statement": "wants a worked numeric example"},
        ],
    }
)


@pytest.mark.asyncio(loop_scope="session")
async def test_direct_message_needs_no_branches_and_costs_two_calls(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, disambiguation_store,
    diagnostics_store,
):
    """AssessAndBranch always gates first (see disambiguate.py's module
    docstring for why this is 2 calls, not the originating spec's
    literal "1"), then FinalAnswer answers directly with no
    scaffolding — the "direct" path this mode still guarantees never
    exceeds."""
    llm = StubLLMClient(
        canned={"ASSESS:BRANCH": _NOT_AMBIGUOUS, "FINAL:ANSWER": "the direct answer"}
    )
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, disambiguation_store, llm=llm, diagnostics_store=diagnostics_store,
    )

    message = await loop.handle_turn(session_id, 0, "what is the derivative of x^2?")

    assert message == "the direct answer"

    async with clean_pool.acquire() as conn:
        names = [
            r["node_name"]
            for r in await conn.fetch(
                "SELECT node_name FROM node_calls WHERE session_id=$1 ORDER BY seq",
                session_id,
            )
        ]
        branch_count = await conn.fetchval("SELECT count(*) FROM disambiguation_branches")
    assert names == ["AssessAndBranch", "FinalAnswer"]
    assert branch_count == 0

    diag = await diagnostics_store.get_for_turn(session_id, 0)
    assert diag.total_call_count == 2
    assert diag.node_call_counts == {"AssessAndBranch": 1, "FinalAnswer": 1}

    turn = await disambiguation_store.get_latest_turn(session_id)
    assert turn.needs_branches is False
    assert turn.turn_had_direct_answer is True


@pytest.mark.asyncio(loop_scope="session")
async def test_every_turns_branches_persist_regardless_of_needs_branches(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, disambiguation_store,
):
    """See CLAUDE.md invariant 9 / DisambiguationTurn's docstring: the
    full history is queryable, including turns that needed no
    branches at all."""
    llm = StubLLMClient(
        canned={"ASSESS:BRANCH": _NOT_AMBIGUOUS, "FINAL:ANSWER": "answer"}
    )
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, disambiguation_store, llm=llm,
    )

    await loop.handle_turn(session_id, 0, "first message")
    await loop.handle_turn(session_id, 1, "second message")

    turn0 = await disambiguation_store.get_latest_turn(session_id)
    assert turn0.turn_index == 1  # get_latest_turn orders by turn_index desc

    async with clean_pool.acquire() as conn:
        turn_rows = await conn.fetch(
            "SELECT turn_index, needs_branches FROM disambiguation_turns "
            "WHERE session_id=$1 ORDER BY turn_index",
            session_id,
        )
    assert [r["turn_index"] for r in turn_rows] == [0, 1]
    assert all(r["needs_branches"] is False for r in turn_rows)


@pytest.mark.asyncio(loop_scope="session")
async def test_ambiguous_message_persists_branches_and_options_with_no_answer_yet(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, disambiguation_store,
    diagnostics_store,
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)

    # AssessAndBranch runs first and persists branches synchronously
    # inside handle_turn, before DisambiguationOptions is ever called —
    # so the real branch ids can be looked up from the DB at
    # DISAMBIGUATE:OPTIONS call time, rather than needing to be known
    # in advance.
    async def _lookup_and_respond() -> str:
        latest = await disambiguation_store.get_latest_turn(session_id)
        branches = await disambiguation_store.list_branches_for_turn(latest.id)
        return json.dumps(
            [
                {"branch_id": str(branches[0].id), "text": "the power rule?"},
                {"branch_id": str(branches[1].id), "text": "a worked example?"},
            ]
        )

    class _AsyncCannedLLM(StubLLMClient):
        async def complete(self, prompt: str) -> str:
            if prompt.startswith("DISAMBIGUATE:OPTIONS"):
                self.prompts.append(prompt)
                return await _lookup_and_respond()
            return await super().complete(prompt)

    llm = _AsyncCannedLLM(canned={"ASSESS:BRANCH": _TWO_BRANCHES})
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, disambiguation_store, llm=llm, diagnostics_store=diagnostics_store,
    )

    message = await loop.handle_turn(session_id, 0, "can you help with derivatives?")
    assert message  # a short "which did you mean" prompt, not a real answer

    disamb_turn = await disambiguation_store.get_latest_turn(session_id)
    assert disamb_turn.needs_branches is True
    assert disamb_turn.turn_had_direct_answer is False

    branches = await disambiguation_store.list_branches_for_turn(disamb_turn.id)
    assert len(branches) == 2
    assert all(b.status is BranchStatus.OPEN for b in branches)

    options = await disambiguation_store.list_options_for_turn(disamb_turn.id)
    assert len(options) == 2
    assert {o.branch_id for o in options} == {b.id for b in branches}
    assert all(o.status is OptionStatus.OPEN for o in options)

    async with clean_pool.acquire() as conn:
        names = [
            r["node_name"]
            for r in await conn.fetch(
                "SELECT node_name FROM node_calls WHERE session_id=$1 ORDER BY seq",
                session_id,
            )
        ]
    assert names == ["AssessAndBranch", "DisambiguationOptions"]
    assert "FinalAnswer" not in names

    diag = await diagnostics_store.get_for_turn(session_id, 0)
    assert diag.total_call_count == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_final_answer_gets_the_same_recent_history_assess_and_branch_got(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, disambiguation_store,
    diagnostics_store,
):
    """Regression test for the live-confirmed turn 9 failure: a bare,
    context-dependent reference ("sketch that") judged unambiguous by
    AssessAndBranch (correctly, using its own recent_history) must not
    be answered by FinalAnswer with zero context -- that produced a
    completely unrelated topic before FinalAnswer received the same
    window. Reproduces the exact shape: turns 0-2 establish a topic,
    turn 3's bare reference must resolve using turns 1-2's own tutor
    text, which only reaches FinalAnswer via recent_history."""
    llm = StubLLMClient(canned={"ASSESS:BRANCH": _NOT_AMBIGUOUS})

    def _final_answer(prompt: str) -> str:
        return "resolved using history" if "chain rule" in prompt else "no history seen"

    llm.canned["FINAL:ANSWER"] = _final_answer

    session_id = await transcript.create_session(learner_id, concept_graph_id)
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, disambiguation_store, llm=llm, diagnostics_store=diagnostics_store,
    )

    await loop.handle_turn(session_id, 0, "can you explain the chain rule?")
    await loop.handle_turn(session_id, 1, "okay that makes sense")
    message = await loop.handle_turn(session_id, 2, "can you sketch that?")

    # FinalAnswer's own prompt must have contained "chain rule" -- only
    # reachable via recent_history, since turn 2's raw message ("sketch
    # that") never names it.
    assert message == "resolved using history"

    assess_call = await node_calls.get_call_for_turn(session_id, 2, "AssessAndBranch")
    final_call = await node_calls.get_call_for_turn(session_id, 2, "FinalAnswer")
    assert assess_call.input_json["recent_history"] == final_call.input_json["recent_history"]
    assert "chain rule" in final_call.input_json["recent_history"]


@pytest.mark.asyncio(loop_scope="session")
async def test_click_resolves_to_final_answer_using_that_branchs_content(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, disambiguation_store,
    diagnostics_store,
):
    """3a: a click skips AssessAndBranch entirely and answers using the
    matched branch's statement as context — 1 call, one turn."""
    turn_id = None

    async def _options_after_branches(prompt: str) -> str:
        nonlocal turn_id
        latest = await disambiguation_store.get_latest_turn(session_id)
        turn_id = latest.id
        branches = await disambiguation_store.list_branches_for_turn(latest.id)
        return json.dumps(
            [
                {"branch_id": str(branches[0].id), "text": "the power rule?"},
                {"branch_id": str(branches[1].id), "text": "a worked example?"},
            ]
        )

    class _AsyncCannedLLM(StubLLMClient):
        async def complete(self, prompt: str) -> str:
            if prompt.startswith("DISAMBIGUATE:OPTIONS"):
                self.prompts.append(prompt)
                return await _options_after_branches(prompt)
            return await super().complete(prompt)

    llm = _AsyncCannedLLM(
        canned={"ASSESS:BRANCH": _TWO_BRANCHES, "FINAL:ANSWER": "answer using the branch"}
    )
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, disambiguation_store, llm=llm, diagnostics_store=diagnostics_store,
    )

    await loop.handle_turn(session_id, 0, "can you help with derivatives?")
    branches = await disambiguation_store.list_branches_for_turn(turn_id)
    options = await disambiguation_store.list_options_for_turn(turn_id)
    clicked_option = options[0]
    clicked_branch = next(b for b in branches if b.id == clicked_option.branch_id)

    message = await loop.handle_turn(
        session_id, 1, "", selected_option_id=clicked_option.id
    )

    assert message == "answer using the branch"

    final_answer_call = await node_calls.get_call_for_turn(session_id, 1, "FinalAnswer")
    assert final_answer_call is not None
    assert final_answer_call.input_json["branch_context"] == clicked_branch.statement

    # The reading was already known -- AssessAndBranch does not run
    # again this turn.
    assert await node_calls.get_call_for_turn(session_id, 1, "AssessAndBranch") is None
    assert await node_calls.get_call_for_turn(session_id, 1, "DisambiguationOptions") is None

    reloaded_clicked = await disambiguation_store.get_branch(clicked_branch.id)
    reloaded_other = await disambiguation_store.get_branch(
        next(b.id for b in branches if b.id != clicked_branch.id)
    )
    assert reloaded_clicked.status is BranchStatus.MATCHED
    assert reloaded_other.status is BranchStatus.SUPERSEDED

    diag = await diagnostics_store.get_for_turn(session_id, 1)
    assert diag.total_call_count == 1
    assert diag.node_call_counts == {"FinalAnswer": 1}


@pytest.mark.asyncio(loop_scope="session")
async def test_typed_past_options_supersedes_them_and_threads_context_into_next_assess(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, disambiguation_store,
    diagnostics_store,
):
    """3b: typing past the options, instead of clicking, does not try
    to match the typed text against the old branches — they are
    superseded outright, and their statements are threaded into this
    turn's own AssessAndBranch call as context (see
    build_typed_past_note)."""
    turn_id_holder: dict[str, object] = {}

    async def _options_after_branches(prompt: str) -> str:
        latest = await disambiguation_store.get_latest_turn(session_id)
        turn_id_holder["id"] = latest.id
        branches = await disambiguation_store.list_branches_for_turn(latest.id)
        return json.dumps(
            [
                {"branch_id": str(branches[0].id), "text": "the power rule?"},
                {"branch_id": str(branches[1].id), "text": "a worked example?"},
            ]
        )

    class _AsyncCannedLLM(StubLLMClient):
        async def complete(self, prompt: str) -> str:
            if prompt.startswith("DISAMBIGUATE:OPTIONS"):
                self.prompts.append(prompt)
                return await _options_after_branches(prompt)
            return await super().complete(prompt)

    session_id = await transcript.create_session(learner_id, concept_graph_id)
    llm = _AsyncCannedLLM(
        canned={"ASSESS:BRANCH": _TWO_BRANCHES, "FINAL:ANSWER": "direct after typing past"}
    )
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, disambiguation_store, llm=llm, diagnostics_store=diagnostics_store,
    )

    await loop.handle_turn(session_id, 0, "can you help with derivatives?")
    first_turn_id = turn_id_holder["id"]

    # Second call to ASSESS:BRANCH (triggered by typing past) returns
    # not-ambiguous, so this turn resolves directly -- but we assert on
    # the prompt it was actually given first.
    llm.canned["ASSESS:BRANCH"] = _NOT_AMBIGUOUS

    message = await loop.handle_turn(session_id, 1, "just tell me the power rule then")

    assert message == "direct after typing past"

    # The old branches are superseded, not matched/unmatched -- no
    # attempt was made to resolve them against the typed text.
    old_branches = await disambiguation_store.list_branches_for_turn(first_turn_id)
    assert all(b.status is BranchStatus.SUPERSEDED for b in old_branches)
    old_options = await disambiguation_store.list_options_for_turn(first_turn_id)
    assert all(o.status is OptionStatus.SUPERSEDED for o in old_options)

    # The typed-past note, naming the old readings, was in the prompt
    # for this turn's AssessAndBranch call.
    assess_prompts = [p for p in llm.prompts if p.startswith("ASSESS:BRANCH")]
    second_assess_prompt = assess_prompts[-1]
    assert "typed past all of them" in second_assess_prompt
    for b in old_branches:
        assert b.statement in second_assess_prompt

    diag = await diagnostics_store.get_for_turn(session_id, 1)
    assert any("disambiguation_typed_past" in w for w in diag.warnings)
