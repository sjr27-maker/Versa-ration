import pytest

from probe.llm import StubLLMClient
from probe.loop import SessionLoop
from probe.models import Hypothesis, Layer, Tier
from probe.nodes import DEFAULT_GENERATION_WIDTH, Replan
from probe.reasoning_budget import ReasoningBudgetConfig


@pytest.mark.asyncio(loop_scope="session")
async def test_replan_records_a_node_call_every_turn(
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

    for i in range(3):
        await loop.handle_turn(session_id, i, f"turn {i} from student")

    async with clean_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT turn_index, output_json
            FROM node_calls
            WHERE session_id = $1 AND node_name = $2
            ORDER BY turn_index
            """,
            session_id,
            "Replan",
        )

    assert [r["turn_index"] for r in rows] == [0, 1, 2]
    for row in rows:
        assert isinstance(row["output_json"], dict)
        for field in (
            "generation_width",
            "run_information_value",
            "entropy_bits",
            "exploration_target",
        ):
            assert field in row["output_json"]


@pytest.mark.asyncio(loop_scope="session")
async def test_replan_output_threads_into_next_turn_infer_input(
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
    # Seed a hypothesis at p=0.5 so Replan's entropy computation is
    # non-trivial and > DEFAULT_GENERATION_WIDTH on turn 0.
    for _ in range(6):
        await store.add(
            Hypothesis(
                layer=Layer.KNOWLEDGE,
                statement="hedge",
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
    await loop.handle_turn(session_id, 1, "turn one")

    async with clean_pool.acquire() as conn:
        infer_widths = await conn.fetch(
            """
            SELECT turn_index, (input_json->>'generation_width')::int AS width
            FROM node_calls
            WHERE session_id = $1 AND node_name = $2
            ORDER BY turn_index
            """,
            session_id,
            "Infer",
        )
        replan_rows = await conn.fetch(
            """
            SELECT turn_index, output_json
            FROM node_calls
            WHERE session_id = $1 AND node_name = $2
            ORDER BY turn_index
            """,
            session_id,
            "Replan",
        )

    assert infer_widths[0]["width"] == DEFAULT_GENERATION_WIDTH
    assert infer_widths[1]["width"] == replan_rows[0]["output_json"]["generation_width"]


def test_replan_returns_exploration_floor_when_no_active_hypotheses():
    import asyncio

    result = asyncio.run(Replan().run([]))
    # No hypotheses -> zero entropy, but generation_width still floors
    # to min_exploration_slots + 1, not down to 1.
    assert result.generation_width == ReasoningBudgetConfig().min_exploration_slots + 1
    assert result.run_information_value is False
    assert result.exploration_target is None


# --- run_information_value gating ValueFunctionConfig -------------------


async def _plan_info_call_counts(clean_pool, session_id) -> list[int]:
    async with clean_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT output_json FROM node_calls WHERE session_id=$1 "
            "AND node_name='Plan' ORDER BY timestamp DESC LIMIT 1",
            session_id,
        )
    return [s["information_value_call_count"] for s in row["output_json"]["scores"]]


@pytest.mark.asyncio(loop_scope="session")
async def test_low_entropy_turn_disables_information_value_for_that_turns_plan(
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
    await store.add(
        Hypothesis(
            layer=Layer.KNOWLEDGE,
            statement="near certain",
            probability=0.99,
            confidence=0.9,
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

    counts = await _plan_info_call_counts(clean_pool, session_id)
    assert counts, "expected at least one scored candidate"
    assert all(c == 0 for c in counts), (
        "low-entropy turn should skip information_value entirely, not just "
        "compute a small value"
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_high_entropy_turn_runs_information_value_for_that_turns_plan(
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
    for _ in range(8):
        await store.add(
            Hypothesis(
                layer=Layer.KNOWLEDGE,
                statement="hedge",
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

    counts = await _plan_info_call_counts(clean_pool, session_id)
    assert counts
    assert all(c >= 1 for c in counts)


@pytest.mark.asyncio(loop_scope="session")
async def test_ablation_disabled_information_value_is_never_reenabled_by_budget(
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
    from probe.value_function import ValueFunctionConfig

    # High entropy: run_information_value would normally be True, but
    # the caller explicitly disabled the term for this whole ablation
    # run — Replan's per-turn suggestion must never override that.
    for _ in range(8):
        await store.add(
            Hypothesis(
                layer=Layer.KNOWLEDGE,
                statement="hedge",
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
        value_function_config=ValueFunctionConfig(enable_information_value=False),
    )
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    await loop.handle_turn(session_id, 0, "turn zero")

    counts = await _plan_info_call_counts(clean_pool, session_id)
    assert counts
    assert all(c == 0 for c in counts)
