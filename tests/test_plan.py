import json

import pytest

from probe.llm import StubLLMClient
from probe.loop import SessionLoop
from probe.models import (
    ActionScore,
    CandidateAction,
    Hypothesis,
    Layer,
    PlanOutput,
    TeachingAction,
    Tier,
)
from probe.nodes import Plan
from probe.value_function import ValueFunction


def _plan(llm: StubLLMClient) -> Plan:
    return Plan(ValueFunction(llm), llm)


def _proposal(action: str, rationale: str = "because") -> dict:
    return {"action": action, "target_concept": "recursion", "rationale": rationale}


@pytest.mark.asyncio
async def test_plan_returns_plan_output_with_winning_candidate():
    plan = _plan(StubLLMClient())
    result = await plan.run(hypotheses=[], concept_state={}, generation_width=3)

    assert isinstance(result, PlanOutput)
    assert isinstance(result.winner, CandidateAction)
    assert len(result.scores) == 3
    assert all(isinstance(s, ActionScore) for s in result.scores)
    winner_total = max(s.total for s in result.scores)
    assert result.winner.id == next(
        s.candidate.id for s in result.scores if s.total == winner_total
    )


@pytest.mark.asyncio
async def test_plan_generates_exactly_generation_width_candidates():
    plan = _plan(StubLLMClient())
    for n in (1, 3, 5):
        result = await plan.run(hypotheses=[], concept_state={}, generation_width=n)
        assert len(result.scores) == n


@pytest.mark.asyncio(loop_scope="session")
async def test_plan_audit_row_contains_all_candidate_breakdowns(
    store,
    transcript,
    node_calls,
    clean_pool,
    learner_id,
    concept_graph_id,
    concept_graph,
    learner_overlay,
    revision_store,
):
    # Seed a hypothesis so information_value has something to work with,
    # though with default stub INFO_RESPONSES="[]" it stays at 0.
    await store.add(
        Hypothesis(
            layer=Layer.KNOWLEDGE,
            statement="student is following along",
            probability=0.5,
            confidence=0.5,
            tier=Tier.ACTIVE,
        )
    )
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
    await loop.handle_turn(session_id, 0, "turn zero")

    async with clean_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT output_json
            FROM node_calls
            WHERE session_id = $1 AND node_name = $2 AND turn_index = 0
            """,
            session_id,
            "Plan",
        )

    assert row is not None
    plan_output = row["output_json"]
    assert "winner" in plan_output
    assert "scores" in plan_output
    assert len(plan_output["scores"]) >= 1
    for score in plan_output["scores"]:
        for field in (
            "learning_value",
            "information_value",
            "long_term_value",
            "time_cost",
            "cognitive_cost",
            "frustration_risk",
            "total",
            "information_value_call_count",
            "flags",
        ):
            assert field in score, f"missing {field} in audit row"
        assert score["candidate"]["action"] in {a.value for a in TeachingAction}


@pytest.mark.asyncio(loop_scope="session")
async def test_plan_winner_is_teach_targets_matching_action(
    store,
    transcript,
    node_calls,
    clean_pool,
    learner_id,
    concept_graph_id,
    concept_graph,
    learner_overlay,
    revision_store,
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
    await loop.handle_turn(session_id, 0, "hello")

    async with clean_pool.acquire() as conn:
        plan_row = await conn.fetchrow(
            "SELECT output_json FROM node_calls WHERE session_id=$1 AND node_name='Plan'",
            session_id,
        )
        teach_row = await conn.fetchrow(
            "SELECT input_json FROM node_calls WHERE session_id=$1 AND node_name='Teach'",
            session_id,
        )

    assert plan_row["output_json"]["winner"]["id"] == teach_row["input_json"]["action"]["id"]


# --- LLM-driven candidate proposal ------------------------------------


def _hyp(layer: Layer, statement: str, tier: Tier = Tier.ACTIVE) -> Hypothesis:
    return Hypothesis(
        layer=layer,
        statement=statement,
        probability=0.6,
        confidence=0.5,
        tier=tier,
    )


MISCONCEPTION_HYP = _hyp(
    Layer.MENTAL_MODEL,
    "student holds the misconception that recursion needs an explicit stack",
)
PLAIN_HYP = _hyp(Layer.KNOWLEDGE, "student can trace a simple loop")


def _layer_sensitive_proposer(prompt: str) -> str:
    """Stands in for a real proposer that reads the hypothesis listing.

    Keys off the hypothesis statement, not the substring "misconception"
    — that also appears in the prompt's own action vocabulary.
    """
    if MISCONCEPTION_HYP.statement in prompt:
        return json.dumps(
            [
                _proposal("correct_misconception", "name the wrong model directly"),
                _proposal("counterexample", "break the stack assumption"),
                _proposal("compare", "contrast with the real call stack"),
            ]
        )
    return json.dumps(
        [
            _proposal("recall", "surface what they already have"),
            _proposal("analogy", "bridge from loops"),
            _proposal("example", "concrete worked case"),
        ]
    )


@pytest.mark.asyncio
async def test_proposer_sees_all_layers_of_active_hypotheses():
    llm = StubLLMClient()
    await _plan(llm).run(
        hypotheses=[MISCONCEPTION_HYP, PLAIN_HYP],
        concept_state={"target_concept": "recursion"},
        generation_width=2,
    )
    propose_prompts = [p for p in llm.prompts if p.startswith("PROPOSE:ACTIONS")]
    assert len(propose_prompts) == 1
    prompt = propose_prompts[0]
    assert MISCONCEPTION_HYP.statement in prompt
    assert PLAIN_HYP.statement in prompt
    assert Layer.MENTAL_MODEL.value in prompt
    assert "recursion" in prompt


@pytest.mark.asyncio
async def test_candidates_vary_with_the_hypothesis_fixture():
    llm = StubLLMClient(canned={"PROPOSE:ACTIONS": _layer_sensitive_proposer})

    misconception_run = await _plan(llm).run(
        hypotheses=[MISCONCEPTION_HYP],
        concept_state={"target_concept": "recursion"},
        generation_width=3,
    )
    plain_run = await _plan(llm).run(
        hypotheses=[PLAIN_HYP],
        concept_state={"target_concept": "recursion"},
        generation_width=3,
    )

    misconception_actions = {s.candidate.action for s in misconception_run.scores}
    plain_actions = {s.candidate.action for s in plain_run.scores}

    assert TeachingAction.CORRECT_MISCONCEPTION in misconception_actions
    assert TeachingAction.CORRECT_MISCONCEPTION not in plain_actions
    assert misconception_actions != plain_actions
    # And neither is the old deterministic first-N-of-enum output.
    first_three = set(list(TeachingAction)[:3])
    assert misconception_actions != first_three
    assert plain_actions != first_three


@pytest.mark.asyncio
async def test_proposed_candidates_carry_target_concept_and_rationale():
    llm = StubLLMClient(canned={"PROPOSE:ACTIONS": _layer_sensitive_proposer})
    result = await _plan(llm).run(
        hypotheses=[MISCONCEPTION_HYP],
        concept_state={"target_concept": "recursion"},
        generation_width=3,
    )
    for score in result.scores:
        assert isinstance(score.candidate, CandidateAction)
        assert score.candidate.target_concept == "recursion"
        assert score.candidate.rationale


@pytest.mark.asyncio
async def test_invalid_actions_are_dropped_and_retried_not_fatal():
    attempts: list[str] = []

    def proposer(prompt: str) -> str:
        attempts.append(prompt)
        if len(attempts) == 1:
            return json.dumps(
                [
                    _proposal("socratic_spiral"),  # not in the enum
                    _proposal("explain"),
                    "not even an object",
                ]
            )
        return json.dumps([_proposal("quiz"), _proposal("analogy")])

    llm = StubLLMClient(canned={"PROPOSE:ACTIONS": proposer})
    result = await _plan(llm).run(
        hypotheses=[PLAIN_HYP], concept_state={}, generation_width=3
    )

    assert len(attempts) == 2
    assert "socratic_spiral" in attempts[1]  # rejects named back to the model
    actions = [s.candidate.action for s in result.scores]
    assert actions == [
        TeachingAction.EXPLAIN,
        TeachingAction.QUIZ,
        TeachingAction.ANALOGY,
    ]


@pytest.mark.asyncio
async def test_malformed_proposer_output_falls_back_to_backfill():
    llm = StubLLMClient(canned={"PROPOSE:ACTIONS": "not json at all"})
    result = await _plan(llm).run(
        hypotheses=[PLAIN_HYP], concept_state={}, generation_width=3
    )
    assert len(result.scores) == 3
    assert [s.candidate.action for s in result.scores] == list(TeachingAction)[:3]


@pytest.mark.asyncio
async def test_short_proposal_backfills_without_duplicates():
    llm = StubLLMClient(
        canned={"PROPOSE:ACTIONS": json.dumps([_proposal("quiz"), _proposal("quiz")])}
    )
    result = await _plan(llm).run(
        hypotheses=[PLAIN_HYP], concept_state={}, generation_width=4
    )
    actions = [s.candidate.action for s in result.scores]
    assert len(actions) == 4
    assert len(set(actions)) == 4, "backfill must not pad duplicates"
    assert actions[0] is TeachingAction.QUIZ
    # Backfill is enum-order over what wasn't already proposed.
    assert actions[1:] == [
        TeachingAction.EXPLAIN,
        TeachingAction.ASK,
        TeachingAction.EXAMPLE,
    ]


@pytest.mark.asyncio
async def test_clean_short_proposal_does_not_trigger_a_retry():
    llm = StubLLMClient(
        canned={"PROPOSE:ACTIONS": json.dumps([_proposal("quiz")])}
    )
    await _plan(llm).run(hypotheses=[PLAIN_HYP], concept_state={}, generation_width=3)
    assert len([p for p in llm.prompts if p.startswith("PROPOSE:ACTIONS")]) == 1


@pytest.mark.asyncio
async def test_over_long_proposal_is_truncated_to_generation_width():
    llm = StubLLMClient(
        canned={
            "PROPOSE:ACTIONS": json.dumps(
                [_proposal(a.value) for a in list(TeachingAction)[:6]]
            )
        }
    )
    result = await _plan(llm).run(
        hypotheses=[PLAIN_HYP], concept_state={}, generation_width=2
    )
    assert [s.candidate.action for s in result.scores] == [
        TeachingAction.EXPLAIN,
        TeachingAction.ASK,
    ]
