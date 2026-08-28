"""Teach's prompt: must carry the derived PathRequirement (what scopes
this turn's teaching) and must NOT carry a branch tree — a tree invites
free association, a path constrains. Teach never receives the tree at
all (its signature has no such parameter); this is really a proof that
PathRequirement's content reaches the prompt and that nothing tree-
shaped (statement/plausibility/depth_label — the tree's own
vocabulary) leaks in some other way.
"""

from __future__ import annotations

import pytest

from probe.llm import StubLLMClient
from probe.models import CandidateAction, PathRequirement, TeachingAction
from probe.nodes import Teach


def _action(target_concept: str | None = "derivatives") -> CandidateAction:
    return CandidateAction(
        action=TeachingAction.EXPLAIN, target_concept=target_concept, rationale="r"
    )


@pytest.mark.asyncio
async def test_prompt_contains_path_requirement_fields():
    path_requirement = PathRequirement(
        current_belief="thinks a derivative is just a static slope value",
        needed="the idea of a limit as the interval shrinks to zero",
        must_not_assume=["the function is linear"],
        scope="connect average rate of change to the limiting instantaneous rate",
    )
    llm = StubLLMClient()
    teach = Teach(llm)

    await teach.run(_action(), "what is a derivative", path_requirement)

    prompt = llm.prompts[-1]
    assert "thinks a derivative is just a static slope value" in prompt
    assert "the idea of a limit as the interval shrinks to zero" in prompt
    assert "the function is linear" in prompt
    assert "connect average rate of change" in prompt
    assert "what is a derivative" in prompt  # student's own message


@pytest.mark.asyncio
async def test_prompt_contains_no_tree_vocabulary():
    """Nothing shaped like a branch (depth_label, plausibility, a
    sibling statement) appears — Teach only ever sees the single
    derived path summary, never the tree it came from."""
    path_requirement = PathRequirement(
        current_belief="belief X",
        needed="needed Y",
        must_not_assume=["constraint Z"],
        scope="scope W",
    )
    llm = StubLLMClient()
    teach = Teach(llm)

    await teach.run(_action(), "message", path_requirement)

    prompt = llm.prompts[-1]
    for tree_word in ("plausibility", "depth_label", "predicted_next_turn", "is_leaf"):
        assert tree_word not in prompt


@pytest.mark.asyncio
async def test_missing_path_requirement_degrades_to_target_concept_only():
    llm = StubLLMClient()
    teach = Teach(llm)

    await teach.run(_action(target_concept="limits"), "message", None)

    prompt = llm.prompts[-1]
    assert "limits" in prompt
    assert "currently believe" not in prompt
    assert "must not assume" not in prompt.lower()


@pytest.mark.asyncio
async def test_must_not_assume_instruction_is_explicit_when_present():
    path_requirement = PathRequirement(
        current_belief="b", needed="n",
        must_not_assume=["the sign of charge Q"], scope="s",
    )
    llm = StubLLMClient()
    teach = Teach(llm)

    await teach.run(_action(target_concept=None), "message", path_requirement)

    prompt = llm.prompts[-1]
    assert "Do NOT assume" in prompt
    assert "the sign of charge Q" in prompt
