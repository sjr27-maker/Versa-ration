"""SessionLoop wiring for the memory layer (memory.py), on top of
SessionMode.MINIMAL_BRANCH — raw store CRUD is covered by
test_memory_store.py; this exercises the loop-level wiring.
"""

import json
from uuid import uuid4

import pytest

from probe.embeddings import EMBEDDING_DIM
from probe.llm import StubLLMClient
from probe.loop import SessionLoop
from probe.memory import MemoryConfig
from probe.models import LearnerFact, LearnerFactType

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


def _vec(x: float = 0.0) -> list[float]:
    return [x] + [0.0] * (EMBEDDING_DIM - 1)


def _make_loop(
    transcript, node_calls, disambiguation_store, llm=None, diagnostics_store=None,
    learner_fact_store=None, thinking_style_store=None, embedding_client=None,
):
    return SessionLoop(
        transcript=transcript,
        node_calls=node_calls,
        llm=llm or StubLLMClient(),
        diagnostics_store=diagnostics_store,
        disambiguation_store=disambiguation_store,
        learner_fact_store=learner_fact_store,
        thinking_style_store=thinking_style_store,
        embedding_client=embedding_client,
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_fact_written_for_a_direct_answer_turn(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    diagnostics_store, learner_fact_store, embedding_client,
):
    llm = StubLLMClient(
        canned={
            "ASSESS:BRANCH": _NOT_AMBIGUOUS,
            "FINAL:ANSWER": "the direct answer",
            "WRITE:FACT": json.dumps(
                {"situation": "asked a direct question", "resolution": "answered it directly"}
            ),
        }
    )
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, llm=llm,
        diagnostics_store=diagnostics_store,
        learner_fact_store=learner_fact_store, embedding_client=embedding_client,
    )

    await loop.handle_turn(session_id, 0, "what is a derivative?")

    facts = await learner_fact_store.list_by_learner(learner_id)
    assert len(facts) == 1
    assert facts[0].fact_type is LearnerFactType.DIRECT_ANSWER
    assert facts[0].situation == "asked a direct question"
    assert facts[0].resolution == "answered it directly"

    write_call = await node_calls.get_call_for_turn(session_id, 0, "WriteLearnerFact")
    assert write_call is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_fact_written_for_a_branch_resolution_turn(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    diagnostics_store, learner_fact_store, embedding_client,
):
    async def _options_after_branches(prompt: str) -> str:
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
                return await _options_after_branches(prompt)
            return await super().complete(prompt)

    session_id = await transcript.create_session(learner_id)
    llm = _AsyncCannedLLM(
        canned={
            "ASSESS:BRANCH": _TWO_BRANCHES,
            "FINAL:ANSWER": "answer using the branch",
            "WRITE:FACT": json.dumps(
                {"situation": "was unsure which reading was meant",
                 "resolution": "confirmed the power rule reading"}
            ),
        }
    )
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, llm=llm,
        diagnostics_store=diagnostics_store,
        learner_fact_store=learner_fact_store, embedding_client=embedding_client,
    )

    await loop.handle_turn(session_id, 0, "can you help with derivatives?")
    disamb_turn = await disambiguation_store.get_latest_turn(session_id)
    options = await disambiguation_store.list_options_for_turn(disamb_turn.id)

    await loop.handle_turn(session_id, 1, "", selected_option_id=options[0].id)

    facts = await learner_fact_store.list_by_learner(learner_id)
    assert len(facts) == 1
    assert facts[0].fact_type is LearnerFactType.BRANCH_RESOLUTION
    assert facts[0].situation == "was unsure which reading was meant"
    assert facts[0].resolution == "confirmed the power rule reading"

    write_call = await node_calls.get_call_for_turn(session_id, 1, "WriteLearnerFact")
    assert write_call is not None
    assert set(write_call.input_json["branch_statements"]) == {
        "wants the power rule explained", "wants a worked numeric example",
    }


@pytest.mark.asyncio(loop_scope="session")
async def test_options_only_turn_writes_no_fact(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    diagnostics_store, learner_fact_store, embedding_client,
):
    """A turn that only raises options (nothing resolved yet) must not
    write a fact -- see WriteLearnerFact's docstring."""
    session_id = await transcript.create_session(learner_id)

    async def _options_after_branches(prompt: str) -> str:
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
                return await _options_after_branches(prompt)
            return await super().complete(prompt)

    llm = _AsyncCannedLLM(canned={"ASSESS:BRANCH": _TWO_BRANCHES})
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, llm=llm,
        diagnostics_store=diagnostics_store,
        learner_fact_store=learner_fact_store, embedding_client=embedding_client,
    )
    await loop.handle_turn(session_id, 0, "can you help with derivatives?")
    assert await learner_fact_store.list_by_learner(learner_id) == []


async def _seed_matching_fact(
    learner_fact_store, transcript, embedding_client, learner_id, message_text: str,
):
    seed_session_id = await transcript.create_session(learner_id)
    turn_id = await transcript.record_turn(seed_session_id, 0, "an earlier message")
    shared_vector = _vec(1.0)
    fact = await learner_fact_store.add(
        LearnerFact(
            learner_id=learner_id, session_id=seed_session_id, turn_index=0,
            fact_type=LearnerFactType.DIRECT_ANSWER,
            situation="confused about which rule applies to (something)^n",
            resolution="clarified it is the general power rule, not the product rule",
            embedding=shared_vector, source_turn_id=turn_id,
        )
    )
    embedding_client.canned[message_text] = shared_vector
    return fact


@pytest.mark.asyncio(loop_scope="session")
async def test_memory_pre_check_skips_branching_when_confirmed(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    diagnostics_store, learner_fact_store, embedding_client,
):
    message = "which rule applies here again?"
    fact = await _seed_matching_fact(
        learner_fact_store, transcript, embedding_client, learner_id, message
    )
    llm = StubLLMClient(
        canned={
            "ASSESS:BRANCH": _TWO_BRANCHES,  # a trap: must never be called
            "CONFIRM:FACT_MATCH": json.dumps({"resolves": True}),
            "FINAL:ANSWER": "answer using the remembered fact",
            "WRITE:FACT": json.dumps({"situation": "s", "resolution": "r"}),
        }
    )
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, llm=llm,
        diagnostics_store=diagnostics_store,
        learner_fact_store=learner_fact_store, embedding_client=embedding_client,
    )

    result = await loop.handle_turn(session_id, 0, message)
    assert result == "answer using the remembered fact"

    assert await node_calls.get_call_for_turn(session_id, 0, "AssessAndBranch") is None
    assert await node_calls.get_call_for_turn(session_id, 0, "DisambiguationOptions") is None
    assert await disambiguation_store.get_latest_turn(session_id) is None

    final_call = await node_calls.get_call_for_turn(session_id, 0, "FinalAnswer")
    assert "confused about which rule applies" in final_call.input_json["memory_context"]

    diag = await diagnostics_store.get_for_turn(session_id, 0)
    assert diag.memory_match_found is True
    assert diag.memory_match_confirmed_resolution is True
    assert diag.branching_skipped_by_memory is True
    assert diag.matched_fact_id == fact.id
    assert any("branching_skipped_by_memory" in w for w in diag.warnings)


@pytest.mark.asyncio(loop_scope="session")
async def test_memory_pre_check_proceeds_normally_when_not_confirmed(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    diagnostics_store, learner_fact_store, embedding_client,
):
    message = "which rule applies here again?"
    await _seed_matching_fact(
        learner_fact_store, transcript, embedding_client, learner_id, message
    )
    llm = StubLLMClient(
        canned={
            "ASSESS:BRANCH": _NOT_AMBIGUOUS,
            "CONFIRM:FACT_MATCH": json.dumps({"resolves": False}),
            "FINAL:ANSWER": "a fresh direct answer",
            "WRITE:FACT": json.dumps({"situation": "s", "resolution": "r"}),
        }
    )
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, llm=llm,
        diagnostics_store=diagnostics_store,
        learner_fact_store=learner_fact_store, embedding_client=embedding_client,
    )

    result = await loop.handle_turn(session_id, 0, message)
    assert result == "a fresh direct answer"

    assert await node_calls.get_call_for_turn(session_id, 0, "AssessAndBranch") is not None

    diag = await diagnostics_store.get_for_turn(session_id, 0)
    assert diag.memory_match_found is True
    assert diag.memory_match_confirmed_resolution is False
    assert diag.branching_skipped_by_memory is False


@pytest.mark.asyncio(loop_scope="session")
async def test_no_memory_stores_configured_behaves_exactly_as_before(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    diagnostics_store,
):
    """Backward compatibility: omitting learner_fact_store/embedding_client
    must not touch learner_facts or crash."""
    llm = StubLLMClient(canned={"ASSESS:BRANCH": _NOT_AMBIGUOUS, "FINAL:ANSWER": "answer"})
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, llm=llm,
        diagnostics_store=diagnostics_store,
    )
    message = await loop.handle_turn(session_id, 0, "what is a derivative?")
    assert message == "answer"
    diag = await diagnostics_store.get_for_turn(session_id, 0)
    assert diag.memory_match_found is False
    assert diag.branching_skipped_by_memory is False


@pytest.mark.asyncio(loop_scope="session")
async def test_thinking_style_only_increments_on_explicit_confirmation(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    learner_fact_store, thinking_style_store, embedding_client,
):
    """consolidate_session must never grow confirmation_count/session_ids
    off similarity alone -- only CONFIRM:THINKING_STYLE saying yes."""
    session_a = await transcript.create_session(learner_id)
    turn_a = await transcript.record_turn(session_a, 0, "turn a")
    await learner_fact_store.add(
        LearnerFact(
            learner_id=learner_id, session_id=session_a, turn_index=0,
            fact_type=LearnerFactType.DIRECT_ANSWER, situation="s", resolution="r",
            embedding=_vec(1.0), source_turn_id=turn_a,
        )
    )
    shared_summary_vector = _vec(1.0)
    embedding_client.canned["same structure, worded differently"] = shared_summary_vector
    existing = await thinking_style_store.create_candidate(
        learner_id, uuid4(), "concrete example before abstract rule", shared_summary_vector
    )

    llm = StubLLMClient(
        canned={
            "SUMMARIZE:PATH": json.dumps({"summary": "same structure, worded differently"}),
            "CONFIRM:THINKING_STYLE": json.dumps({"confirms": False}),
        }
    )
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, llm=llm,
        learner_fact_store=learner_fact_store,
        thinking_style_store=thinking_style_store, embedding_client=embedding_client,
    )

    result = await loop.consolidate_session(session_a)

    unchanged = await thinking_style_store.get(existing.id)
    assert unchanged.confirmation_count == 1
    assert unchanged.session_ids == existing.session_ids
    assert result.id != existing.id
    assert result.confirmation_count == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_thinking_style_confirms_and_grows_when_llm_agrees(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    learner_fact_store, thinking_style_store, embedding_client,
):
    session_a = await transcript.create_session(learner_id)
    turn_a = await transcript.record_turn(session_a, 0, "turn a")
    await learner_fact_store.add(
        LearnerFact(
            learner_id=learner_id, session_id=session_a, turn_index=0,
            fact_type=LearnerFactType.DIRECT_ANSWER, situation="s", resolution="r",
            embedding=_vec(1.0), source_turn_id=turn_a,
        )
    )
    shared_summary_vector = _vec(1.0)
    embedding_client.canned["same structure, worded differently"] = shared_summary_vector
    existing = await thinking_style_store.create_candidate(
        learner_id, uuid4(), "concrete example before abstract rule", shared_summary_vector
    )

    llm = StubLLMClient(
        canned={
            "SUMMARIZE:PATH": json.dumps({"summary": "same structure, worded differently"}),
            "CONFIRM:THINKING_STYLE": json.dumps({"confirms": True}),
        }
    )
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, llm=llm,
        learner_fact_store=learner_fact_store,
        thinking_style_store=thinking_style_store, embedding_client=embedding_client,
    )

    result = await loop.consolidate_session(session_a)
    assert result.id == existing.id
    assert result.confirmation_count == 2
    assert session_a in result.session_ids


@pytest.mark.asyncio(loop_scope="session")
async def test_candidate_below_threshold_never_appears_in_a_live_prompt(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    diagnostics_store, thinking_style_store,
):
    below_threshold = await thinking_style_store.create_candidate(
        learner_id, uuid4(), "wants a concrete example before any abstract rule", _vec(1.0)
    )
    for sid in [uuid4(), uuid4()]:  # count=3, still below default 5
        below_threshold = await thinking_style_store.confirm(
            below_threshold.id, sid,
            promotion_threshold=MemoryConfig().thinking_style_promotion_threshold,
        )
    assert below_threshold.confirmation_count == 3

    llm = StubLLMClient(canned={"ASSESS:BRANCH": _NOT_AMBIGUOUS, "FINAL:ANSWER": "answer"})
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, llm=llm,
        diagnostics_store=diagnostics_store,
        thinking_style_store=thinking_style_store,
    )

    await loop.handle_turn(session_id, 0, "hello")

    assess_call = await node_calls.get_call_for_turn(session_id, 0, "AssessAndBranch")
    assert assess_call.input_json["thinking_style_hint"] == ""
    assert "concrete example" not in json.dumps(assess_call.input_json)


@pytest.mark.asyncio(loop_scope="session")
async def test_promoted_candidate_appears_in_the_live_prompt(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    diagnostics_store, thinking_style_store,
):
    promoted = await thinking_style_store.create_candidate(
        learner_id, uuid4(), "wants a concrete example before any abstract rule", _vec(1.0)
    )
    for sid in [uuid4(), uuid4(), uuid4(), uuid4()]:  # count reaches 5
        promoted = await thinking_style_store.confirm(
            promoted.id, sid,
            promotion_threshold=MemoryConfig().thinking_style_promotion_threshold,
        )
    assert promoted.confirmation_count == 5

    llm = StubLLMClient(canned={"ASSESS:BRANCH": _NOT_AMBIGUOUS, "FINAL:ANSWER": "answer"})
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, llm=llm,
        diagnostics_store=diagnostics_store,
        thinking_style_store=thinking_style_store,
    )

    await loop.handle_turn(session_id, 0, "hello")

    assess_call = await node_calls.get_call_for_turn(session_id, 0, "AssessAndBranch")
    assert "concrete example before any abstract rule" in assess_call.input_json["thinking_style_hint"]
