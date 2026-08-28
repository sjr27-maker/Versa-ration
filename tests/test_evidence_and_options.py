"""Evidence requirement (Branch.requires_evidence/evidence_satisfied,
should_expand_branch's fourth gate) and the options mechanism
(GenerateOptions, CheckEvidence) that satisfies it — the unambiguous
click channel plus its interpreted typed-text counterpart.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from probe.hypothesis_generator import (
    CheckEvidence,
    GenerateOptions,
    should_expand_branch,
)
from probe.llm import StubLLMClient
from probe.models import Branch
from probe.reasoning_budget import BranchBudgetConfig, compute_branch_budget


def _branch(
    generation_id, session_id, *, requires_evidence=None, evidence_satisfied=False,
    plausibility=0.9,
) -> Branch:
    return Branch(
        parent_id=None,
        generation_id=generation_id,
        session_id=session_id,
        turn_index=0,
        depth=0,
        depth_label="intent",
        statement="a plausible bet",
        predicted_next_turn="predicts something",
        requires_evidence=requires_evidence,
        evidence_satisfied=evidence_satisfied,
        plausibility=plausibility,
        is_leaf=True,
    )


# --- should_expand_branch's fourth gate --------------------------------


def test_branch_with_unsatisfied_requires_evidence_does_not_expand():
    budget = compute_branch_budget([], BranchBudgetConfig())
    survives = should_expand_branch(
        plausibility=0.95,  # would otherwise clear every other gate easily
        statement="a plausible bet",
        sibling_statements=[],
        depth=0,
        branches_so_far=0,
        budget=budget,
        requires_evidence="the student confirms they want to see the derivation",
        evidence_satisfied=False,
    )
    assert survives is False


def test_branch_with_satisfied_requires_evidence_expands_normally():
    budget = compute_branch_budget([], BranchBudgetConfig())
    survives = should_expand_branch(
        plausibility=0.95,
        statement="a plausible bet",
        sibling_statements=[],
        depth=0,
        branches_so_far=0,
        budget=budget,
        requires_evidence="the student confirms they want to see the derivation",
        evidence_satisfied=True,
    )
    assert survives is True


def test_branch_with_no_requires_evidence_is_unaffected_by_the_new_gate():
    budget = compute_branch_budget([], BranchBudgetConfig())
    survives = should_expand_branch(
        plausibility=0.95,
        statement="a plausible bet",
        sibling_statements=[],
        depth=0,
        branches_so_far=0,
        budget=budget,
    )
    assert survives is True


# --- GenerateOptions -----------------------------------------------------


@pytest.mark.asyncio
async def test_one_branch_per_option_in_a_clean_mapping():
    generation_id, session_id = uuid4(), uuid4()
    b1 = _branch(generation_id, session_id, requires_evidence="wants an example")
    b2 = _branch(generation_id, session_id, requires_evidence="missing a prerequisite")
    canned = json.dumps(
        [
            {"branch_id": str(b1.id), "text": "Want to see a worked example first?"},
            {"branch_id": str(b2.id), "text": "Does the setup so far make sense?"},
        ]
    )
    node = GenerateOptions(StubLLMClient(canned={"GENERATE:OPTIONS": canned}))

    proposals = await node.run([b1, b2])

    assert {p.branch_id for p in proposals} == {b1.id, b2.id}
    assert len({p.branch_id for p in proposals}) == len(proposals)  # no duplicates


@pytest.mark.asyncio
async def test_duplicate_branch_mapping_is_rejected_and_regenerated():
    generation_id, session_id = uuid4(), uuid4()
    b1 = _branch(generation_id, session_id, requires_evidence="wants an example")
    b2 = _branch(generation_id, session_id, requires_evidence="missing a prerequisite")

    bad = json.dumps(
        [
            {"branch_id": str(b1.id), "text": "first claim"},
            {"branch_id": str(b1.id), "text": "second claim, same branch"},
        ]
    )
    good = json.dumps([{"branch_id": str(b1.id), "text": "a clean single claim"}])
    calls = {"n": 0}

    def _dispatch(_prompt: str) -> str:
        calls["n"] += 1
        return bad if calls["n"] == 1 else good

    node = GenerateOptions(StubLLMClient(canned={"GENERATE:OPTIONS": _dispatch}))

    proposals = await node.run([b1, b2])

    assert calls["n"] == 2  # rejected once, regenerated once
    assert len(proposals) == 1
    assert proposals[0].text == "a clean single claim"


@pytest.mark.asyncio
async def test_exhausted_retries_on_bad_mappings_yields_no_options_not_a_corrupt_one():
    generation_id, session_id = uuid4(), uuid4()
    b1 = _branch(generation_id, session_id, requires_evidence="wants an example")
    always_duplicate = json.dumps(
        [
            {"branch_id": str(b1.id), "text": "a"},
            {"branch_id": str(b1.id), "text": "b"},
        ]
    )
    node = GenerateOptions(StubLLMClient(canned={"GENERATE:OPTIONS": always_duplicate}))

    proposals = await node.run([b1])

    assert proposals == []


@pytest.mark.asyncio
async def test_no_candidates_skips_the_llm_call_entirely():
    generation_id, session_id = uuid4(), uuid4()
    satisfied = _branch(
        generation_id, session_id, requires_evidence="x", evidence_satisfied=True
    )
    no_requirement = _branch(generation_id, session_id, requires_evidence=None)
    llm = StubLLMClient()
    node = GenerateOptions(llm)

    proposals = await node.run([satisfied, no_requirement])

    assert proposals == []
    assert llm.prompts == []  # never called -- nothing to ask about


@pytest.mark.asyncio
async def test_an_option_referencing_an_id_outside_the_live_set_is_rejected():
    generation_id, session_id = uuid4(), uuid4()
    b1 = _branch(generation_id, session_id, requires_evidence="wants an example")
    stranger_id = uuid4()
    bad = json.dumps([{"branch_id": str(stranger_id), "text": "not a live branch"}])
    good = json.dumps([{"branch_id": str(b1.id), "text": "a real one"}])
    calls = {"n": 0}

    def _dispatch(_prompt: str) -> str:
        calls["n"] += 1
        return bad if calls["n"] == 1 else good

    node = GenerateOptions(StubLLMClient(canned={"GENERATE:OPTIONS": _dispatch}))

    proposals = await node.run([b1])

    assert calls["n"] == 2
    assert proposals[0].branch_id == b1.id


# --- CheckEvidence ---------------------------------------------------------


@pytest.mark.asyncio
async def test_check_evidence_matches_a_typed_message_to_the_right_branch():
    generation_id, session_id = uuid4(), uuid4()
    b1 = _branch(generation_id, session_id, requires_evidence="wants an example")
    canned = json.dumps({"satisfied_branch_id": str(b1.id), "confidence": 0.9})
    node = CheckEvidence(StubLLMClient(canned={"CHECK:EVIDENCE": canned}))

    result = await node.run("yeah can you show me an example", [b1])

    assert result.satisfied_branch_id == b1.id


@pytest.mark.asyncio
async def test_check_evidence_rejects_an_id_outside_the_candidate_set():
    generation_id, session_id = uuid4(), uuid4()
    b1 = _branch(generation_id, session_id, requires_evidence="wants an example")
    stranger_id = uuid4()
    canned = json.dumps({"satisfied_branch_id": str(stranger_id), "confidence": 0.9})
    node = CheckEvidence(StubLLMClient(canned={"CHECK:EVIDENCE": canned}))

    result = await node.run("something", [b1])

    assert result.satisfied_branch_id is None


@pytest.mark.asyncio
async def test_check_evidence_with_no_candidates_skips_the_llm_call():
    llm = StubLLMClient()
    node = CheckEvidence(llm)

    result = await node.run("anything", [])

    assert result.satisfied_branch_id is None
    assert llm.prompts == []


@pytest.mark.asyncio
async def test_check_evidence_default_stub_finds_nothing():
    generation_id, session_id = uuid4(), uuid4()
    b1 = _branch(generation_id, session_id, requires_evidence="wants an example")
    node = CheckEvidence(StubLLMClient())  # default CHECK:EVIDENCE -> null

    result = await node.run("totally unrelated text", [b1])

    assert result.satisfied_branch_id is None
