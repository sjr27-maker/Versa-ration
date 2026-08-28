"""Plan's proposer must pick target_concept from the session's actual
concept graph (concept_state.concepts), not invent one — and reject an
off-graph id the same way GroundConcept rejects a hallucinated
concept_id, rather than pass it through. This is grounding ("what is
this turn about"), not curriculum selection ("what should we teach
next") — concept_state never decides what to teach, only what's
actually available to talk about.
"""

import json

import pytest

from probe.llm import StubLLMClient
from probe.nodes import Plan
from probe.value_function import ValueFunction

_CALCULUS_CONCEPT_STATE = {
    "topic": "Calculus",
    "concepts": [
        {"id": "limits", "name": "Limits"},
        {"id": "derivatives", "name": "Derivatives"},
    ],
    "grounded_concept_id": "derivatives",
    "overlay": {},
}


def _plan(llm: StubLLMClient) -> Plan:
    return Plan(ValueFunction(llm), llm)


@pytest.mark.asyncio
async def test_plan_accepts_a_real_target_concept_from_the_graph():
    canned = json.dumps(
        [{"action": "explain", "target_concept": "limits", "rationale": "because"}]
    )
    llm = StubLLMClient(canned={"PROPOSE:ACTIONS": canned})
    plan = _plan(llm)

    result = await plan.run(
        hypotheses=[], concept_state=_CALCULUS_CONCEPT_STATE, generation_width=1
    )

    assert result.winner.target_concept == "limits"


@pytest.mark.asyncio
async def test_plan_rejects_an_off_graph_target_concept_and_falls_back_to_grounded(
    caplog,
):
    import logging

    canned = json.dumps(
        [
            {
                "action": "explain",
                "target_concept": "quantum_entanglement",
                "rationale": "because",
            }
        ]
    )
    llm = StubLLMClient(canned={"PROPOSE:ACTIONS": canned})
    plan = _plan(llm)

    with caplog.at_level(logging.WARNING, logger="probe.nodes"):
        result = await plan.run(
            hypotheses=[], concept_state=_CALCULUS_CONCEPT_STATE, generation_width=1
        )

    assert result.winner.target_concept != "quantum_entanglement"
    # Falls back to grounded_concept_id, not None and not the invented id.
    assert result.winner.target_concept == "derivatives"
    assert any(
        "quantum_entanglement" in r.message and "not in this session" in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_plan_falls_back_to_none_when_off_graph_and_nothing_grounded():
    concept_state = {
        "topic": "Calculus",
        "concepts": [{"id": "limits", "name": "Limits"}],
        "grounded_concept_id": None,
        "overlay": {},
    }
    canned = json.dumps(
        [{"action": "explain", "target_concept": "made_up_id", "rationale": "because"}]
    )
    llm = StubLLMClient(canned={"PROPOSE:ACTIONS": canned})
    plan = _plan(llm)

    result = await plan.run(
        hypotheses=[], concept_state=concept_state, generation_width=1
    )

    assert result.winner.target_concept is None


@pytest.mark.asyncio
async def test_plan_with_empty_concept_state_does_not_validate_at_all():
    """No graph attached (concept_state={}, e.g. before AttachTopic
    runs) — the same behavior as before this feature existed. Nothing
    to validate against, so nothing gets rejected."""
    canned = json.dumps(
        [{"action": "explain", "target_concept": "anything", "rationale": "because"}]
    )
    llm = StubLLMClient(canned={"PROPOSE:ACTIONS": canned})
    plan = _plan(llm)

    result = await plan.run(hypotheses=[], concept_state={}, generation_width=1)

    assert result.winner.target_concept == "anything"


@pytest.mark.asyncio
async def test_backfilled_candidates_default_to_grounded_concept_not_none():
    """A candidate the proposer didn't name a target for (including
    enum-order backfill filler) defaults to grounded_concept_id — what
    the student was actually just talking about — not a blank None."""
    canned = json.dumps(
        [{"action": "explain", "target_concept": None, "rationale": "because"}]
    )
    llm = StubLLMClient(canned={"PROPOSE:ACTIONS": canned})
    plan = _plan(llm)

    result = await plan.run(
        hypotheses=[], concept_state=_CALCULUS_CONCEPT_STATE, generation_width=3
    )

    assert all(s.candidate.target_concept == "derivatives" for s in result.scores)
