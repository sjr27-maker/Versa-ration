"""SessionLoop's HypothesisGenerator wiring: backward compatibility
when branch_store is omitted (the default, and what every pre-existing
SessionLoop test still does), turn ordering (generate() now runs
before Teach, right after Plan, with SelectBranch/DerivePath between
them; resolve() still runs first, before Diagnose, on the following
turn), and that MAX_CALLS_PER_TURN's guardrail sum actually includes
BranchGenerate/BranchResolve/SelectBranch/DerivePath's call counts,
not just the original five nodes.
"""

import logging

import pytest

from probe.llm import StubLLMClient
from probe.loop import SessionLoop


@pytest.mark.asyncio(loop_scope="session")
async def test_branch_store_none_reproduces_pre_existing_turn_flow(
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
    assert loop.branch_generate is None
    assert loop.branch_resolve is None

    session_id = await transcript.create_session(learner_id, concept_graph_id)
    message = await loop.handle_turn(session_id, 0, "hello")
    assert message

    async with clean_pool.acquire() as conn:
        branch_rows = await conn.fetchval("SELECT count(*) FROM branches")
        gen_rows = await conn.fetchval("SELECT count(*) FROM branch_generations")
        rows = await conn.fetch(
            "SELECT node_name FROM node_calls WHERE session_id=$1", session_id
        )
    assert branch_rows == 0
    assert gen_rows == 0
    assert {r["node_name"] for r in rows} == {
        "Diagnose",
        "Infer",
        "Update",
        "Replan",
        "Plan",
        "Teach",
    }


@pytest.mark.asyncio(loop_scope="session")
async def test_generate_runs_before_teach_and_resolve_runs_before_diagnose_next_turn(
    store,
    transcript,
    node_calls,
    clean_pool,
    learner_id,
    concept_graph_id,
    concept_graph,
    learner_overlay,
    revision_store,
    branch_store,
):
    loop = SessionLoop(
        hypothesis_store=store,
        transcript=transcript,
        node_calls=node_calls,
        concept_graph=concept_graph,
        learner_overlay=learner_overlay,
        revision_store=revision_store,
        llm=StubLLMClient(),
        branch_store=branch_store,
    )
    session_id = await transcript.create_session(learner_id, concept_graph_id)

    await loop.handle_turn(session_id, 0, "first turn")
    await loop.handle_turn(session_id, 1, "second turn")

    async with clean_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT turn_index, node_name FROM node_calls "
            "WHERE session_id=$1 ORDER BY turn_index, timestamp",
            session_id,
        )
        gen_count = await conn.fetchval(
            "SELECT count(*) FROM branch_generations WHERE session_id=$1",
            session_id,
        )

    turn0_names = [r["node_name"] for r in rows if r["turn_index"] == 0]
    turn1_names = [r["node_name"] for r in rows if r["turn_index"] == 1]

    assert turn0_names == [
        "Diagnose", "Infer", "Update", "Replan", "Plan",
        "BranchGenerate", "SelectBranch", "DerivePath", "Teach",
    ]
    # No prior generation on turn 0, so no BranchResolve call at all that turn.
    assert "BranchResolve" not in turn0_names
    assert turn1_names[0] == "BranchResolve"  # resolves turn 0's generation first
    assert turn1_names[1] == "Diagnose"
    assert turn1_names[-1] == "Teach"  # generation now precedes Teach, not after it
    assert gen_count == 2  # one generation per turn


@pytest.mark.asyncio(loop_scope="session")
async def test_max_calls_per_turn_guardrail_sums_in_branch_calls(
    store,
    transcript,
    node_calls,
    clean_pool,
    learner_id,
    concept_graph_id,
    concept_graph,
    learner_overlay,
    revision_store,
    branch_store,
    caplog,
    monkeypatch,
):
    import probe.loop as loop_module

    # Force the guardrail to trip on any turn, so we can inspect exactly
    # what it summed without needing to engineer a naturally-huge turn.
    monkeypatch.setattr(loop_module, "MAX_CALLS_PER_TURN", 0)

    loop = SessionLoop(
        hypothesis_store=store,
        transcript=transcript,
        node_calls=node_calls,
        concept_graph=concept_graph,
        learner_overlay=learner_overlay,
        revision_store=revision_store,
        llm=StubLLMClient(),
        branch_store=branch_store,
    )
    session_id = await transcript.create_session(learner_id, concept_graph_id)

    with caplog.at_level(logging.WARNING, logger="probe.loop"):
        await loop.handle_turn(session_id, 0, "first turn")

    async with clean_pool.acquire() as conn:
        gen_row = await conn.fetchrow(
            "SELECT output_json FROM node_calls "
            "WHERE session_id=$1 AND node_name='BranchGenerate'",
            session_id,
        )
    branch_generate_calls = gen_row["output_json"]["call_count"]
    assert branch_generate_calls >= 1  # at least the root GENERATE:INTENT call

    warnings = [r.getMessage() for r in caplog.records if "MAX_CALLS_PER_TURN" in r.message]
    assert warnings
    # The per-node breakdown is now a real dict (also what gets persisted
    # to turn_diagnostics.node_call_counts), not a hand-formatted string.
    assert f"'BranchGenerate': {branch_generate_calls}" in warnings[0]
    # BranchResolve never ran on turn 0 (no prior generation), so it's
    # simply absent from the breakdown rather than present at 0.
    assert "BranchResolve" not in warnings[0]
