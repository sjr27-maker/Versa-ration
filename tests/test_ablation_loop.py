"""SessionLoop's ablation wiring: BASELINE is a genuine bypass (exactly
one LLM call, zero reasoning-store rows beyond it), and each individual
AblationConfig toggle actually stops its subsystem from running — the
node is absent from node_calls (or makes zero LLM calls, for Diagnose's
partial-disable case) and no corresponding rows land in its store, not
merely "the output still looks plausible."
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from probe.ablation import AblationConfig, ReasoningBudgetMode
from probe.llm import StubLLMClient
from probe.loop import _TEACH_FAILURE_MESSAGE, SessionLoop
from probe.models import ConceptNode

_GROUND_DERIVATIVES = json.dumps({"concept_id": "derivatives", "confidence": 0.9})


async def _seed_graph(concept_graph, graph_id):
    await concept_graph.add_batch(
        graph_id,
        "Calculus",
        [ConceptNode(concept_graph_id=graph_id, id="derivatives", name="Derivatives")],
    )


def _make_loop(
    store, transcript, node_calls, concept_graph, learner_overlay, revision_store,
    ablation_config, llm=None, branch_store=None, option_store=None,
    diagnostics_store=None,
):
    return SessionLoop(
        hypothesis_store=store,
        transcript=transcript,
        node_calls=node_calls,
        concept_graph=concept_graph,
        learner_overlay=learner_overlay,
        revision_store=revision_store,
        llm=llm or StubLLMClient(),
        branch_store=branch_store,
        option_store=option_store,
        diagnostics_store=diagnostics_store,
        ablation_config=ablation_config,
    )


# --- BASELINE: the genuine bypass ------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_baseline_makes_exactly_one_llm_call_and_touches_no_reasoning_stores(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph,
    learner_overlay, revision_store, branch_store, option_store, diagnostics_store,
):
    config = AblationConfig(
        enable_portrait=False,
        enable_concept_graph=False,
        enable_diagnose=False,
        enable_planner=False,
        enable_branches=False,
        enable_options=False,
    )
    assert config.is_full_bypass is True
    session_id = await transcript.create_session(
        learner_id, concept_graph_id=None, ablation_config=config
    )
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, config, branch_store=branch_store, option_store=option_store,
        diagnostics_store=diagnostics_store,
    )

    message = await loop.handle_turn(session_id, 0, "what is a derivative?")

    assert message

    async with clean_pool.acquire() as conn:
        node_call_names = [
            r["node_name"]
            for r in await conn.fetch(
                "SELECT node_name FROM node_calls WHERE session_id=$1", session_id
            )
        ]
        hyp_count = await conn.fetchval("SELECT count(*) FROM hypotheses")
        branch_count = await conn.fetchval("SELECT count(*) FROM branches")
        option_count = await conn.fetchval("SELECT count(*) FROM options")
        overlay_count = await conn.fetchval("SELECT count(*) FROM learner_overlay")
        revision_count = await conn.fetchval("SELECT count(*) FROM world_model_revisions")

    assert node_call_names == ["BaselineTeach"]
    assert hyp_count == 0
    assert branch_count == 0
    assert option_count == 0
    assert overlay_count == 0
    assert revision_count == 0

    diag = await diagnostics_store.get_for_turn(session_id, 0)
    assert diag.total_call_count == 1
    assert diag.node_call_counts == {"BaselineTeach": 1}


@pytest.mark.asyncio(loop_scope="session")
async def test_baseline_carries_prior_turns_as_context(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph,
    learner_overlay, revision_store,
):
    config = AblationConfig(
        enable_portrait=False, enable_concept_graph=False, enable_diagnose=False,
        enable_planner=False, enable_branches=False, enable_options=False,
    )
    session_id = await transcript.create_session(
        learner_id, concept_graph_id=None, ablation_config=config
    )
    llm = StubLLMClient()
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, config, llm=llm,
    )

    await loop.handle_turn(session_id, 0, "what is a derivative?")
    await loop.handle_turn(session_id, 1, "and what about an integral?")

    second_prompt = llm.prompts[-1]
    assert "what is a derivative?" in second_prompt
    assert "and what about an integral?" in second_prompt


@pytest.mark.asyncio(loop_scope="session")
async def test_hand_toggled_full_bypass_behaves_identically_to_baseline_preset(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph,
    learner_overlay, revision_store,
):
    """is_full_bypass is derived, not a separate switch (see
    AblationConfig) -- setting every flag off by hand must reach
    _handle_bypass_turn exactly the way selecting BASELINE does."""
    config = AblationConfig(
        enable_portrait=False, enable_concept_graph=False, enable_diagnose=False,
        enable_planner=False, enable_branches=False, enable_options=False,
    )
    session_id = await transcript.create_session(
        learner_id, concept_graph_id=None, ablation_config=config
    )
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, config,
    )

    await loop.handle_turn(session_id, 0, "hello")

    async with clean_pool.acquire() as conn:
        names = [
            r["node_name"]
            for r in await conn.fetch(
                "SELECT node_name FROM node_calls WHERE session_id=$1", session_id
            )
        ]
    assert names == ["BaselineTeach"]


@pytest.mark.asyncio(loop_scope="session")
async def test_baseline_teach_failure_returns_the_fixed_message_and_writes_no_node_call(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph,
    learner_overlay, revision_store, diagnostics_store,
):
    """BaselineTeach is BASELINE's only call -- unlike the full path,
    where a Teach failure still leaves other nodes' rows as evidence
    the turn ran, a failure here has nothing to fall back on. Must
    degrade exactly like every other node's failure in this loop
    (fixed message, teach_failed=True), not crash the turn."""
    config = AblationConfig(
        enable_portrait=False, enable_concept_graph=False, enable_diagnose=False,
        enable_planner=False, enable_branches=False, enable_options=False,
    )

    def _raise(_prompt: str) -> str:
        raise RuntimeError("simulated transport failure")

    session_id = await transcript.create_session(
        learner_id, concept_graph_id=None, ablation_config=config
    )
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, config,
        llm=StubLLMClient(canned={"BASELINE:TEACH": _raise}),
        diagnostics_store=diagnostics_store,
    )

    message = await loop.handle_turn(session_id, 0, "hello")

    assert message == _TEACH_FAILURE_MESSAGE

    async with clean_pool.acquire() as conn:
        node_call_count = await conn.fetchval(
            "SELECT count(*) FROM node_calls WHERE session_id=$1", session_id
        )
    assert node_call_count == 0  # the one call failed -- nothing to record

    diag = await diagnostics_store.get_for_turn(session_id, 0)
    assert diag.teach_failed is True
    assert diag.total_call_count == 0
    assert any("BaselineTeach failed" in w for w in diag.warnings)


# --- Individual toggles ------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_enable_portrait_false_skips_infer_and_update_and_touches_no_hypotheses(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, diagnostics_store,
):
    config = AblationConfig(enable_portrait=False)
    session_id = await transcript.create_session(
        learner_id, concept_graph_id, ablation_config=config
    )
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, config, diagnostics_store=diagnostics_store,
    )

    await loop.handle_turn(session_id, 0, "a turn")

    async with clean_pool.acquire() as conn:
        names = {
            r["node_name"]
            for r in await conn.fetch(
                "SELECT node_name FROM node_calls WHERE session_id=$1", session_id
            )
        }
        hyp_count = await conn.fetchval("SELECT count(*) FROM hypotheses")

    assert "Infer" not in names
    assert "Update" not in names
    assert "Diagnose" in names  # the turn actually ran, not merely short-circuited
    assert "Plan" in names
    assert hyp_count == 0

    diag = await diagnostics_store.get_for_turn(session_id, 0)
    assert "Infer" not in diag.node_call_counts
    assert "Update" not in diag.node_call_counts


@pytest.mark.asyncio(loop_scope="session")
async def test_enable_concept_graph_false_skips_grounding_and_bypasses_the_hard_fail(
    store, transcript, node_calls, clean_pool, learner_id,
    concept_graph, learner_overlay, revision_store, diagnostics_store,
):
    config = AblationConfig(enable_concept_graph=False)
    session_id = await transcript.create_session(
        learner_id, concept_graph_id=None, ablation_config=config
    )
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, config, diagnostics_store=diagnostics_store,
    )

    await loop.handle_turn(session_id, 0, "first")
    # Would normally raise SessionMissingTopicError past turn 0 with a
    # still-null concept_graph_id -- deliberately bypassed when this
    # configuration disables the concept graph outright.
    await loop.handle_turn(session_id, 1, "second")

    assert await transcript.get_concept_graph_id(session_id) is None

    async with clean_pool.acquire() as conn:
        names = {
            r["node_name"]
            for r in await conn.fetch(
                "SELECT node_name FROM node_calls WHERE session_id=$1", session_id
            )
        }
    assert "AttachTopic" not in names
    assert "Diagnose" in names  # Diagnose still runs -- just makes zero calls

    diag0 = await diagnostics_store.get_for_turn(session_id, 0)
    diag1 = await diagnostics_store.get_for_turn(session_id, 1)
    assert diag0.node_call_counts.get("Diagnose", 0) == 0
    assert diag1.node_call_counts.get("Diagnose", 0) == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_enable_diagnose_false_grounds_but_skips_mismatch_and_revisions(
    store, transcript, node_calls, clean_pool, learner_id,
    concept_graph, learner_overlay, revision_store, diagnostics_store,
):
    graph_id = uuid4()
    await _seed_graph(concept_graph, graph_id)
    config = AblationConfig(enable_diagnose=False)
    session_id = await transcript.create_session(
        learner_id, graph_id, ablation_config=config
    )
    llm = StubLLMClient(canned={"GROUND:CONCEPT": _GROUND_DERIVATIVES})
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, config, llm=llm, diagnostics_store=diagnostics_store,
    )

    await loop.handle_turn(session_id, 0, "a derivative is a slope")

    diag = await diagnostics_store.get_for_turn(session_id, 0)
    # Grounding happened (1 call) but mismatch detection never fired
    # (would be 2+ calls if it had) -- confirms Diagnose still ran and
    # still grounded, just skipped everything past that point.
    assert diag.node_call_counts.get("Diagnose") == 1

    diagnose_call = await node_calls.get_call_for_turn(session_id, 0, "Diagnose")
    assert diagnose_call.output_json["grounding"]["concept_id"] == "derivatives"
    assert diagnose_call.output_json["mismatch"] is None
    assert diagnose_call.output_json["action_taken"] == "none"

    async with clean_pool.acquire() as conn:
        revision_count = await conn.fetchval("SELECT count(*) FROM world_model_revisions")
    assert revision_count == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_enable_planner_false_uses_fixed_action_with_no_plan_call(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, diagnostics_store,
):
    config = AblationConfig(enable_planner=False)
    session_id = await transcript.create_session(
        learner_id, concept_graph_id, ablation_config=config
    )
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, config, diagnostics_store=diagnostics_store,
    )

    await loop.handle_turn(session_id, 0, "what is a derivative?")

    plan_call = await node_calls.get_call_for_turn(session_id, 0, "Plan")
    assert plan_call is None

    teach_call = await node_calls.get_call_for_turn(session_id, 0, "Teach")
    assert teach_call.input_json["action"]["action"] == "explain"
    assert teach_call.input_json["action"]["rationale"] == "answer the student's question"

    diag = await diagnostics_store.get_for_turn(session_id, 0)
    assert "Plan" not in diag.node_call_counts


@pytest.mark.asyncio(loop_scope="session")
async def test_enable_branches_false_never_generates_a_branch(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, branch_store, option_store,
    diagnostics_store,
):
    config = AblationConfig(enable_branches=False, enable_options=False)
    session_id = await transcript.create_session(
        learner_id, concept_graph_id, ablation_config=config
    )
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, config, branch_store=branch_store, option_store=option_store,
        diagnostics_store=diagnostics_store,
    )
    assert loop.branch_generate is None

    await loop.handle_turn(session_id, 0, "first")  # turn 0 would skip anyway
    await loop.handle_turn(session_id, 1, "second")  # would normally generate

    async with clean_pool.acquire() as conn:
        names_t1 = {
            r["node_name"]
            for r in await conn.fetch(
                "SELECT node_name FROM node_calls WHERE session_id=$1 AND turn_index=1",
                session_id,
            )
        }
        branch_count = await conn.fetchval("SELECT count(*) FROM branches")

    assert "BranchGenerate" not in names_t1
    assert branch_count == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_enable_options_false_generates_branches_but_never_options(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, branch_store, option_store,
    diagnostics_store,
):
    config = AblationConfig(enable_options=False)
    session_id = await transcript.create_session(
        learner_id, concept_graph_id, ablation_config=config
    )
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, config, branch_store=branch_store, option_store=option_store,
        diagnostics_store=diagnostics_store,
    )
    assert loop.branch_generate is not None
    assert loop.generate_options is None

    await loop.handle_turn(session_id, 0, "first")  # turn 0 never generates
    await loop.handle_turn(session_id, 1, "second")

    async with clean_pool.acquire() as conn:
        names_t1 = {
            r["node_name"]
            for r in await conn.fetch(
                "SELECT node_name FROM node_calls WHERE session_id=$1 AND turn_index=1",
                session_id,
            )
        }
        option_count = await conn.fetchval("SELECT count(*) FROM options")

    assert "BranchGenerate" in names_t1
    assert "GenerateOptions" not in names_t1
    assert option_count == 0


# --- Secondary knobs -----------------------------------------------------


def test_enable_exploration_slot_false_zeroes_the_reasoning_budget_floor():
    from probe.reasoning_budget import ReasoningBudgetConfig

    loop_on = SessionLoop(
        hypothesis_store=object(), transcript=object(), node_calls=object(),
        concept_graph=object(), learner_overlay=object(), revision_store=object(),
        llm=StubLLMClient(),
        ablation_config=AblationConfig(enable_exploration_slot=True),
    )
    loop_off = SessionLoop(
        hypothesis_store=object(), transcript=object(), node_calls=object(),
        concept_graph=object(), learner_overlay=object(), revision_store=object(),
        llm=StubLLMClient(),
        ablation_config=AblationConfig(enable_exploration_slot=False),
    )
    assert loop_on.replan._config.min_exploration_slots == (
        ReasoningBudgetConfig().min_exploration_slots
    )
    assert loop_off.replan._config.min_exploration_slots == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_reasoning_budget_mode_fixed_pins_generation_width(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store,
):
    # With zero active hypotheses, ENTROPY mode would floor generation_width
    # to min_exploration_slots(1) + 1 == 2 -- 5 is unambiguous evidence
    # FIXED mode overrode it, not a coincidence of the entropy formula.
    config = AblationConfig(
        reasoning_budget_mode=ReasoningBudgetMode.FIXED, fixed_generation_width=5
    )
    session_id = await transcript.create_session(
        learner_id, concept_graph_id, ablation_config=config
    )
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, config,
    )

    await loop.handle_turn(session_id, 0, "a turn")

    plan_call = await node_calls.get_call_for_turn(session_id, 0, "Plan")
    assert plan_call.input_json["generation_width"] == 5
