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


@pytest.mark.asyncio
async def test_plan_returns_plan_output_with_winning_candidate():
    plan = Plan(ValueFunction(StubLLMClient()))
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
    plan = Plan(ValueFunction(StubLLMClient()))
    for n in (1, 3, 5):
        result = await plan.run(hypotheses=[], concept_state={}, generation_width=n)
        assert len(result.scores) == n


@pytest.mark.asyncio(loop_scope="session")
async def test_plan_audit_row_contains_all_candidate_breakdowns(
    store, transcript, node_calls, clean_pool
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
        llm=StubLLMClient(),
    )
    session_id = await transcript.create_session()
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
        ):
            assert field in score, f"missing {field} in audit row"
        assert score["candidate"]["action"] in {a.value for a in TeachingAction}


@pytest.mark.asyncio(loop_scope="session")
async def test_plan_winner_is_teach_targets_matching_action(
    store, transcript, node_calls, clean_pool
):
    loop = SessionLoop(
        hypothesis_store=store,
        transcript=transcript,
        node_calls=node_calls,
        llm=StubLLMClient(),
    )
    session_id = await transcript.create_session()
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
