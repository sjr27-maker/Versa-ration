"""SelectBranch and DerivePath — the two new stages between
BranchGenerate and Teach. SelectBranch's criterion is coverage, not
probability (see hypothesis_generator._select_prompt); these tests
prove the mechanism actually passes a real coverage-based judgment
through rather than silently defaulting to the highest-plausibility
branch regardless of what the model says.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from probe.hypothesis_generator import DerivePath, SelectBranch, build_branch_path
from probe.llm import StubLLMClient
from probe.models import Branch


def _branch(
    statement: str,
    plausibility: float,
    generation_id,
    session_id,
    parent_id=None,
    depth: int = 0,
    depth_label: str = "intent",
) -> Branch:
    return Branch(
        parent_id=parent_id,
        generation_id=generation_id,
        session_id=session_id,
        turn_index=0,
        depth=depth,
        depth_label=depth_label,
        statement=statement,
        predicted_next_turn=f"predicts: {statement}",
        plausibility=plausibility,
        is_leaf=True,
    )


@pytest.mark.asyncio
async def test_selection_picks_lower_probability_branch_with_better_coverage():
    """4 siblings: one at p=0.85 that only serves itself, one at p=0.6
    that a real model judges to cover the other three's shared ground.
    The canned response names the p=0.6 branch explicitly — this
    proves SelectBranch passes that judgment through as-is, not that
    it independently reimplements "pick the max plausibility" and
    happens to agree."""
    generation_id, session_id = uuid4(), uuid4()
    high_solo = _branch("wants a fully worked numeric example", 0.85, generation_id, session_id)
    coverage_winner = _branch(
        "is unsure how the general method applies here", 0.6, generation_id, session_id
    )
    sibling_a = _branch("doesn't see how to start applying the method", 0.5, generation_id, session_id)
    sibling_b = _branch("applied the method but got confused partway", 0.45, generation_id, session_id)
    branches = [high_solo, coverage_winner, sibling_a, sibling_b]

    canned_response = json.dumps(
        {
            "selected_branch_id": str(coverage_winner.id),
            "rationale": (
                "covers the general-method confusion shared by the two "
                "other siblings, unlike the solo worked-example request"
            ),
        }
    )
    node = SelectBranch(StubLLMClient(canned={"SELECT:BRANCH": canned_response}))

    result = await node.run(branches)

    assert result.selected_branch_id == coverage_winner.id
    assert result.selected_branch_id != high_solo.id  # not just the highest-plausibility one
    assert "covers" in result.rationale


@pytest.mark.asyncio
async def test_selection_falls_back_to_highest_plausibility_on_unparseable_response():
    generation_id, session_id = uuid4(), uuid4()
    low = _branch("a", 0.3, generation_id, session_id)
    high = _branch("b", 0.9, generation_id, session_id)
    node = SelectBranch(StubLLMClient(canned={"SELECT:BRANCH": "not json"}))

    result = await node.run([low, high])

    assert result.selected_branch_id == high.id
    assert "fallback" in result.rationale.lower()


@pytest.mark.asyncio
async def test_selection_with_no_branches_selects_nothing():
    node = SelectBranch(StubLLMClient())
    result = await node.run([])
    assert result.selected_branch_id is None


@pytest.mark.asyncio
async def test_derive_path_produces_non_empty_must_not_assume_for_unstated_quantity():
    """A path whose leaf implies the student needs to reason about a
    charge's sign, which was never given in the problem — the exact
    shape of the charge-sign failure this mechanism exists to prevent.
    The canned response is a realistic derivation naming that gap; this
    proves it survives parsing intact, not that DerivePath invents it."""
    generation_id, session_id = uuid4(), uuid4()
    root = _branch(
        "is working through a Coulomb's law problem involving a charge Q",
        0.7,
        generation_id,
        session_id,
    )
    leaf = _branch(
        "will ask whether the force on Q is attractive or repulsive",
        0.6,
        generation_id,
        session_id,
        parent_id=root.id,
        depth=1,
        depth_label="predicted_action",
    )
    path = [root, leaf]

    canned_response = json.dumps(
        {
            "current_belief": "the student understands Coulomb's law's magnitude formula",
            "needed": "how to determine attraction vs repulsion from the force direction",
            "must_not_assume": [
                "the sign of charge Q (positive or negative) — never stated in the problem"
            ],
            "scope": "explain how relative charge sign determines attraction vs repulsion",
        }
    )
    node = DerivePath(StubLLMClient(canned={"DERIVE:PATH": canned_response}))

    result = await node.run(
        path,
        student_message="I don't know if the force is attractive or repulsive",
        action_rationale="explain force direction via Coulomb's law",
    )

    assert result.must_not_assume  # non-empty
    assert "sign" in result.must_not_assume[0].lower()
    assert result.current_belief
    assert result.scope


@pytest.mark.asyncio
async def test_derive_path_degrades_gracefully_on_unparseable_response():
    generation_id, session_id = uuid4(), uuid4()
    branch = _branch("a", 0.5, generation_id, session_id)
    node = DerivePath(StubLLMClient(canned={"DERIVE:PATH": "not json"}))

    result = await node.run([branch], student_message="message", action_rationale="r")

    assert result.must_not_assume == []
    assert result.current_belief == ""
    assert result.scope == ""


def test_build_branch_path_is_root_to_leaf_inclusive():
    generation_id, session_id = uuid4(), uuid4()
    root = _branch("root intent", 0.8, generation_id, session_id)
    mid = _branch(
        "knowledge gap", 0.6, generation_id, session_id,
        parent_id=root.id, depth=1, depth_label="knowledge_gap",
    )
    leaf = _branch(
        "predicted action", 0.5, generation_id, session_id,
        parent_id=mid.id, depth=2, depth_label="predicted_action",
    )
    branches = [leaf, root, mid]  # deliberately out of order

    path = build_branch_path(branches, leaf.id)

    assert [b.id for b in path] == [root.id, mid.id, leaf.id]


def test_build_branch_path_for_a_root_branch_is_just_itself():
    generation_id, session_id = uuid4(), uuid4()
    root = _branch("root intent", 0.8, generation_id, session_id)

    path = build_branch_path([root], root.id)

    assert path == [root]


# --- Regression: predicted reaction promoted into a stated belief ------
#
# Real failure: the selected root branch's predicted_next_turn was "That
# whiteboard idea makes sense, so are the roommates acting like
# processors sharing the same memory?" -- a HYPOTHETICAL future reaction
# to an analogy Plan proposed but had not yet taught. DerivePath turned
# this into current_belief: "The student believes the whiteboard analogy
# translates to roommates acting as processors..." -- stated as settled
# fact. Teach then affirmed it ("Exactly, that's a perfect way to
# visualize it...") as if the student had said it, when they never had.


@pytest.mark.asyncio
async def test_derive_prompt_marks_predicted_reaction_as_hypothetical_not_happened():
    generation_id, session_id = uuid4(), uuid4()
    root = _branch(
        "wants a high-level overview of parallel computing", 0.5,
        generation_id, session_id,
    )
    root.predicted_next_turn = (
        "That whiteboard idea makes sense, so are the roommates acting "
        "like processors sharing the same memory?"
    )
    llm = StubLLMClient()
    node = DerivePath(llm)

    await node.run(
        [root],
        student_message="what is parallel computing",
        action_rationale=(
            "An analogy of roommates sharing a kitchen whiteboard helps "
            "explain shared memory architecture."
        ),
    )

    prompt = llm.prompts[-1]
    assert "predicted future reaction, NOT YET SAID" in prompt
    assert "has NOT happened" in prompt
    assert "hypothetical" in prompt.lower()


@pytest.mark.asyncio
async def test_derive_prompt_warns_against_crediting_untaught_action_content():
    generation_id, session_id = uuid4(), uuid4()
    root = _branch("wants an overview", 0.5, generation_id, session_id)
    llm = StubLLMClient()
    node = DerivePath(llm)

    await node.run(
        [root],
        student_message="what is parallel computing",
        action_rationale="roommates sharing a kitchen whiteboard analogy",
    )

    prompt = llm.prompts[-1]
    assert "has NOT taught it yet" in prompt
    assert "roommates sharing a kitchen whiteboard analogy" in prompt
    assert "cannot hold a belief" in prompt.lower() or "cannot already believe" in prompt.lower()


@pytest.mark.asyncio
async def test_derive_prompt_offers_insufficient_evidence_as_a_legitimate_answer():
    generation_id, session_id = uuid4(), uuid4()
    root = _branch("wants an overview", 0.5, generation_id, session_id)
    llm = StubLLMClient()
    node = DerivePath(llm)

    await node.run([root], student_message="what is parallel computing", action_rationale="r")

    prompt = llm.prompts[-1]
    assert "insufficient evidence to characterize current belief" in prompt
    assert "not a failure to avoid" in prompt.lower()


@pytest.mark.asyncio
async def test_derive_prompt_includes_the_students_actual_message_verbatim():
    generation_id, session_id = uuid4(), uuid4()
    root = _branch("wants an overview", 0.5, generation_id, session_id)
    llm = StubLLMClient()
    node = DerivePath(llm)

    await node.run(
        [root],
        student_message="what is parallel computing",
        action_rationale="r",
    )

    prompt = llm.prompts[-1]
    assert "what is parallel computing" in prompt
