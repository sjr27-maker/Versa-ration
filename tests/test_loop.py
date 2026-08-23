import pytest

from probe.llm import StubLLMClient
from probe.loop import SessionLoop
from probe.models import Hypothesis, Layer, Tier
from probe.nodes import DEFAULT_GENERATION_WIDTH, Replan


@pytest.mark.asyncio(loop_scope="session")
async def test_replan_records_a_node_call_every_turn(
    store, transcript, node_calls, clean_pool
):
    loop = SessionLoop(
        hypothesis_store=store,
        transcript=transcript,
        node_calls=node_calls,
        llm=StubLLMClient(),
    )
    session_id = await transcript.create_session()

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
        assert isinstance(row["output_json"], int)


@pytest.mark.asyncio(loop_scope="session")
async def test_replan_output_threads_into_next_turn_infer_input(
    store, transcript, node_calls, clean_pool
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
        llm=StubLLMClient(),
    )
    session_id = await transcript.create_session()

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
    assert infer_widths[1]["width"] == replan_rows[0]["output_json"]


def test_replan_returns_min_width_when_no_active_hypotheses():
    import asyncio

    result = asyncio.run(Replan().run([]))
    assert result == 1
