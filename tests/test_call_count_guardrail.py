"""MAX_CALLS_PER_TURN is a loud-warning guardrail, not a limit: a turn
that trips it must still complete normally (nothing truncated, no
candidates dropped) while logging the actual count. The count itself
is the complete per-turn total (Diagnose, Infer, Plan's proposer,
Plan's per-candidate ValueFunction terms, and Teach are all
individually instrumented), not an undercount of it — see
MAX_CALLS_PER_TURN's docstring in nodes.py for what changed and why.
"""

import json
import logging

import pytest

from probe.llm import StubLLMClient
from probe.loop import SessionLoop
from probe.models import EvidenceRef, Hypothesis, Layer, Polarity, Tier
from probe.nodes import MAX_CALLS_PER_TURN

# Three simulated responses per information_value call (instead of the
# stub default's empty list) pushes each of the 8 max-width candidates
# to 1 + 3 = 4 information_value calls apiece -- 32 known-instrumented
# calls this turn, comfortably over MAX_CALLS_PER_TURN=30.
_CANNED_INFO_RESPONSES = json.dumps(
    [
        {"response": "a", "probability": 0.3},
        {"response": "b", "probability": 0.3},
        {"response": "c", "probability": 0.4},
    ]
)


@pytest.mark.asyncio(loop_scope="session")
async def test_turn_exceeding_max_calls_per_turn_logs_a_warning_and_still_completes(
    store,
    transcript,
    node_calls,
    clean_pool,
    learner_id,
    concept_graph_id,
    concept_graph,
    learner_overlay,
    revision_store,
    caplog,
):
    # Replan's entropy is learner-scoped (list_by_learner), so each
    # hypothesis needs evidence tying it to this learner's own session
    # before Replan will count it — a bare store.add() with no evidence
    # isn't attributable to anyone.
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    seed_turn_id = await transcript.record_turn(
        session_id, 0, "seed turn establishing prior hypotheses"
    )
    for _ in range(8):
        hyp = await store.add(
            Hypothesis(
                layer=Layer.KNOWLEDGE,
                statement="hedge",
                probability=0.5,
                confidence=0.5,
                tier=Tier.ACTIVE,
            )
        )
        await store.reweight(
            hyp.id,
            0.5,
            0.5,
            EvidenceRef(turn_id=seed_turn_id, polarity=Polarity.SUPPORTING),
        )
    llm = StubLLMClient(canned={"SCORE:INFO_RESPONSES": _CANNED_INFO_RESPONSES})
    loop = SessionLoop(
        hypothesis_store=store,
        transcript=transcript,
        node_calls=node_calls,
        concept_graph=concept_graph,
        learner_overlay=learner_overlay,
        revision_store=revision_store,
        llm=llm,
    )

    with caplog.at_level(logging.WARNING, logger="probe.loop"):
        message = await loop.handle_turn(session_id, 1, "turn one")

    assert message  # the turn completed; nothing was truncated to dodge the ceiling
    warnings = [r.getMessage() for r in caplog.records if "MAX_CALLS_PER_TURN" in r.message]
    assert warnings, "expected a guardrail warning to be logged"
    assert f"MAX_CALLS_PER_TURN={MAX_CALLS_PER_TURN}" in warnings[0]


@pytest.mark.asyncio(loop_scope="session")
async def test_ordinary_turn_stays_under_the_guardrail_without_warning(
    store,
    transcript,
    node_calls,
    clean_pool,
    learner_id,
    concept_graph_id,
    concept_graph,
    learner_overlay,
    revision_store,
    caplog,
):
    loop = SessionLoop(
        hypothesis_store=store,
        transcript=transcript,
        node_calls=node_calls,
        concept_graph=concept_graph,
        learner_overlay=learner_overlay,
        revision_store=revision_store,
        llm=StubLLMClient(),
    )
    session_id = await transcript.create_session(learner_id, concept_graph_id)

    with caplog.at_level(logging.WARNING, logger="probe.loop"):
        await loop.handle_turn(session_id, 0, "turn zero")

    assert not [r for r in caplog.records if "MAX_CALLS_PER_TURN" in r.message]
