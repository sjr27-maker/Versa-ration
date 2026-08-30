"""SessionLoop wiring for check_current_belief_leak: when DerivePath's
current_belief leaks content from the predicted reaction or the
proposed action's rationale that the student never actually said,
turn_diagnostics.current_belief_unsupported is set and a warning is
recorded — the structural backstop firing end to end, not just the
unit-level heuristic in isolation.
"""

from __future__ import annotations

import json

import pytest

from probe.diagnostics import TurnDiagnosticsStore
from probe.llm import StubLLMClient
from probe.loop import SessionLoop

_LEAKED_CONTENT = "roommates sharing a kitchen whiteboard"

_INTENT_WITH_LEAKY_PREDICTION = json.dumps(
    [
        {
            "statement": "wants a high-level overview of parallel computing",
            "plausibility": 0.9,
            "predicted_next_turn": (
                f"That {_LEAKED_CONTENT} idea makes sense, so are they "
                "acting like processors sharing memory?"
            ),
        }
    ]
)

_PROPOSE_ANALOGY_ACTION = json.dumps(
    [
        {
            "action": "analogy",
            "target_concept": None,
            "rationale": f"An analogy of {_LEAKED_CONTENT} explains shared memory.",
        }
    ]
)

_DERIVE_PATH_THAT_LEAKS = json.dumps(
    {
        "current_belief": f"The student believes the {_LEAKED_CONTENT} analogy applies here.",
        "needed": "confirmation of the analogy",
        "must_not_assume": [],
        "scope": "shared memory architecture",
    }
)

_DERIVE_PATH_GROUNDED = json.dumps(
    {
        "current_belief": "The student is asking a broad definitional question.",
        "needed": "a direct definition",
        "must_not_assume": [],
        "scope": "what parallel computing is",
    }
)


def _make_loop(store, transcript, node_calls, concept_graph, learner_overlay,
               revision_store, branch_store, diagnostics_store, llm):
    return SessionLoop(
        hypothesis_store=store,
        transcript=transcript,
        node_calls=node_calls,
        concept_graph=concept_graph,
        learner_overlay=learner_overlay,
        revision_store=revision_store,
        llm=llm,
        branch_store=branch_store,
        diagnostics_store=diagnostics_store,
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_leaked_current_belief_sets_the_diagnostics_flag(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, branch_store,
):
    diagnostics_store = TurnDiagnosticsStore(clean_pool)
    llm = StubLLMClient(
        canned={
            "GENERATE:INTENT": _INTENT_WITH_LEAKY_PREDICTION,
            "PROPOSE:ACTIONS": _PROPOSE_ANALOGY_ACTION,
            "DERIVE:PATH": _DERIVE_PATH_THAT_LEAKS,
        }
    )
    loop = _make_loop(store, transcript, node_calls, concept_graph, learner_overlay,
                       revision_store, branch_store, diagnostics_store, llm)
    session_id = await transcript.create_session(learner_id, concept_graph_id)

    # Turn 0 never generates (see loop.py) — the generation this test
    # is about starts on turn 1.
    await loop.handle_turn(session_id, 0, "hello")
    await loop.handle_turn(session_id, 1, "what is parallel computing")

    diag = await diagnostics_store.get_for_turn(session_id, 1)
    assert diag.current_belief_unsupported is True
    assert any("current_belief_unsupported" in w for w in diag.warnings)


@pytest.mark.asyncio(loop_scope="session")
async def test_grounded_current_belief_does_not_set_the_flag(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, branch_store,
):
    diagnostics_store = TurnDiagnosticsStore(clean_pool)
    llm = StubLLMClient(
        canned={
            "GENERATE:INTENT": _INTENT_WITH_LEAKY_PREDICTION,
            "PROPOSE:ACTIONS": _PROPOSE_ANALOGY_ACTION,
            "DERIVE:PATH": _DERIVE_PATH_GROUNDED,
        }
    )
    loop = _make_loop(store, transcript, node_calls, concept_graph, learner_overlay,
                       revision_store, branch_store, diagnostics_store, llm)
    session_id = await transcript.create_session(learner_id, concept_graph_id)

    # Turn 0 never generates (see loop.py) — the generation this test
    # is about starts on turn 1.
    await loop.handle_turn(session_id, 0, "hello")
    await loop.handle_turn(session_id, 1, "what is parallel computing")

    diag = await diagnostics_store.get_for_turn(session_id, 1)
    assert diag.current_belief_unsupported is False
    assert not any("current_belief_unsupported" in w for w in diag.warnings)
