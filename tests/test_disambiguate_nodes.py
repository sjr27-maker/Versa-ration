"""Pure node-level tests for disambiguate.py, no DB — StubLLMClient
only. Store-level persistence is covered by test_disambiguation_store.py;
full-turn wiring (click resolution, typed-past threading) is covered by
test_disambiguation_loop_wiring.py.
"""

import json

import pytest

from probe.disambiguate import AssessAndBranch, DisambiguationOptions, FinalAnswer
from probe.llm import StubLLMClient
from probe.models import DisambiguationBranch


@pytest.mark.asyncio(loop_scope="session")
async def test_unambiguous_message_needs_no_branches_and_costs_one_call():
    llm = StubLLMClient(
        canned={
            "ASSESS:BRANCH": json.dumps({"needs_branches": False, "branches": []}),
        }
    )
    node = AssessAndBranch(llm)
    result = await node.run("what is the derivative of x^2?")
    assert result.needs_branches is False
    assert result.branch_statements == []
    assert node.last_call_count == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_ambiguous_message_produces_two_to_four_distinct_branches():
    llm = StubLLMClient(
        canned={
            "ASSESS:BRANCH": json.dumps(
                {
                    "needs_branches": True,
                    "branches": [
                        {"statement": "wants the power rule explained"},
                        {"statement": "wants a worked numeric example"},
                        {"statement": "wants to know why the rule works"},
                    ],
                }
            ),
        }
    )
    node = AssessAndBranch(llm)
    result = await node.run("can you help with derivatives?")
    assert result.needs_branches is True
    assert len(result.branch_statements) == 3
    assert len(set(result.branch_statements)) == 3
    assert node.last_call_count == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_duplicate_reading_is_rejected_and_regenerated():
    attempts = {"n": 0}

    def _respond(_prompt: str) -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return json.dumps(
                {
                    "needs_branches": True,
                    "branches": [
                        {"statement": "wants the derivative of x squared"},
                        {"statement": "wants the derivative of x squared"},
                    ],
                }
            )
        return json.dumps(
            {
                "needs_branches": True,
                "branches": [
                    {"statement": "wants the derivative of x squared"},
                    {"statement": "wants to check their own attempt at it"},
                ],
            }
        )

    llm = StubLLMClient(canned={"ASSESS:BRANCH": _respond})
    node = AssessAndBranch(llm)
    result = await node.run("about x^2")
    assert result.needs_branches is True
    assert len(result.branch_statements) == 2
    assert node.last_call_count == 2  # rejected once, retried once


@pytest.mark.asyncio(loop_scope="session")
async def test_exhausted_retries_degrade_to_not_ambiguous_rather_than_crash():
    llm = StubLLMClient(canned={"ASSESS:BRANCH": "not json at all"})
    node = AssessAndBranch(llm)
    result = await node.run("anything")
    assert result.needs_branches is False
    assert result.branch_statements == []
    assert node.last_call_count == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_branch_count_outside_range_is_rejected():
    llm = StubLLMClient(
        canned={
            "ASSESS:BRANCH": json.dumps(
                {"needs_branches": True, "branches": [{"statement": "only one reading"}]}
            )
        }
    )
    node = AssessAndBranch(llm)
    result = await node.run("anything")
    # A single retry with the same malformed shape exhausts attempts and
    # degrades to not-ambiguous, same as any other unparseable response.
    assert result.needs_branches is False
    assert node.last_call_count == 2


def _branch(statement: str) -> DisambiguationBranch:
    import uuid

    return DisambiguationBranch(
        disambiguation_turn_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        turn_index=0,
        statement=statement,
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_one_option_per_branch():
    branches = [_branch("reading a"), _branch("reading b")]

    def _respond(_prompt: str) -> str:
        return json.dumps(
            [
                {"branch_id": str(branches[0].id), "text": "Is it about reading a?"},
                {"branch_id": str(branches[1].id), "text": "Is it about reading b?"},
            ]
        )

    llm = StubLLMClient(canned={"DISAMBIGUATE:OPTIONS": _respond})
    node = DisambiguationOptions(llm)
    proposals = await node.run(branches)
    assert len(proposals) == 2
    assert {p.branch_id for p in proposals} == {branches[0].id, branches[1].id}


@pytest.mark.asyncio(loop_scope="session")
async def test_duplicate_branch_mapping_is_rejected_and_retried():
    branches = [_branch("reading a"), _branch("reading b")]
    attempts = {"n": 0}

    def _respond(_prompt: str) -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            # Both options map to the same branch -- invalid.
            return json.dumps(
                [
                    {"branch_id": str(branches[0].id), "text": "one"},
                    {"branch_id": str(branches[0].id), "text": "two"},
                ]
            )
        return json.dumps(
            [
                {"branch_id": str(branches[0].id), "text": "one"},
                {"branch_id": str(branches[1].id), "text": "two"},
            ]
        )

    llm = StubLLMClient(canned={"DISAMBIGUATE:OPTIONS": _respond})
    node = DisambiguationOptions(llm)
    proposals = await node.run(branches)
    assert len(proposals) == 2
    assert node.last_call_count == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_exhausted_option_retries_return_empty_not_a_crash():
    branches = [_branch("reading a"), _branch("reading b")]
    llm = StubLLMClient(canned={"DISAMBIGUATE:OPTIONS": "garbage"})
    node = DisambiguationOptions(llm)
    proposals = await node.run(branches)
    assert proposals == []
    assert node.last_call_count == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_no_branches_means_no_call_at_all():
    llm = StubLLMClient(canned={"DISAMBIGUATE:OPTIONS": "should never be called"})
    node = DisambiguationOptions(llm)
    proposals = await node.run([])
    assert proposals == []
    assert node.last_call_count == 0
    assert llm.prompts == []


@pytest.mark.asyncio(loop_scope="session")
async def test_final_answer_uses_branch_context_when_given():
    llm = StubLLMClient(canned={"FINAL:ANSWER": "the answer"})
    node = FinalAnswer(llm)
    result = await node.run("help me", branch_context="wants the power rule explained")
    assert result == "the answer"
    assert "wants the power rule explained" in llm.prompts[-1]
    assert node.last_call_count == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_final_answer_direct_call_has_no_branch_scaffolding():
    llm = StubLLMClient(canned={"FINAL:ANSWER": "direct answer"})
    node = FinalAnswer(llm)
    result = await node.run("what's the derivative of x^2?")
    assert result == "direct answer"
    assert "confirmed they meant" not in llm.prompts[-1]


@pytest.mark.asyncio(loop_scope="session")
async def test_final_answer_receives_recent_history_to_resolve_references():
    """Regression test for the live-confirmed failure: a bare reference
    ("sketch that") answered with zero conversation context produced a
    completely unrelated topic. FinalAnswer must actually see
    recent_history when given it, not just accept the parameter."""
    llm = StubLLMClient(canned={"FINAL:ANSWER": "answer grounded in history"})
    node = FinalAnswer(llm)
    history = "turn 8 student: what about (something)^n?\nturn 8 tutor: bring n down..."
    result = await node.run(
        "can you sketch that?", branch_context=None, recent_history=history
    )
    assert result == "answer grounded in history"
    assert history in llm.prompts[-1]


@pytest.mark.asyncio(loop_scope="session")
async def test_final_answer_with_no_recent_history_omits_the_history_block():
    llm = StubLLMClient(canned={"FINAL:ANSWER": "answer"})
    node = FinalAnswer(llm)
    await node.run("what's the derivative of x^2?", recent_history="")
    assert "Recent conversation" not in llm.prompts[-1]
