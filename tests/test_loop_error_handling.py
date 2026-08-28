"""A failing node must not lose the turn — Diagnose/Infer/Replan/Plan/
BranchResolve/BranchGenerate/SelectBranch/DerivePath failures are
caught, recorded as a warning, and replaced with a safe fallback so
the turn still completes. Teach is the one exception: it has no
fallback, so its failure returns a fixed in-band message and sets
turn_diagnostics.teach_failed. BranchGenerate now runs *before* Teach
(see loop.py's reorder), so it is no longer conditioned on
teach_failed at all — a Teach failure doesn't discard that turn's
already-generated tree, it's just flagged with an extra warning (see
test_teach_failure_still_generates_and_keeps_the_branch_tree).
"""


import pytest

from probe.llm import StubLLMClient
from probe.loop import _TEACH_FAILURE_MESSAGE, SessionLoop


def _raise(_prompt: str) -> str:
    raise RuntimeError("simulated LLM failure")


def _make_loop(
    store, transcript, node_calls, concept_graph, learner_overlay, revision_store,
    llm, diagnostics_store, branch_store=None,
):
    return SessionLoop(
        hypothesis_store=store,
        transcript=transcript,
        node_calls=node_calls,
        concept_graph=concept_graph,
        learner_overlay=learner_overlay,
        revision_store=revision_store,
        llm=llm,
        diagnostics_store=diagnostics_store,
        branch_store=branch_store,
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_diagnose_failure_does_not_lose_the_turn(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, diagnostics_store,
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    llm = StubLLMClient(canned={"GROUND:CONCEPT": _raise})
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, llm, diagnostics_store,
    )

    message = await loop.handle_turn(session_id, 0, "a turn")

    assert message
    diagnostics = await diagnostics_store.get_for_turn(session_id, 0)
    assert any("Diagnose failed" in w for w in diagnostics.warnings)
    assert diagnostics.teach_failed is False


@pytest.mark.asyncio(loop_scope="session")
async def test_infer_failure_does_not_lose_the_turn(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, diagnostics_store,
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    llm = StubLLMClient(canned={"INFER:": _raise})
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, llm, diagnostics_store,
    )

    message = await loop.handle_turn(session_id, 0, "a turn")

    assert message
    diagnostics = await diagnostics_store.get_for_turn(session_id, 0)
    assert any("Infer failed" in w for w in diagnostics.warnings)


@pytest.mark.asyncio(loop_scope="session")
async def test_replan_failure_does_not_lose_the_turn(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, diagnostics_store,
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, StubLLMClient(), diagnostics_store,
    )

    async def _raise_async(*_args, **_kwargs):
        raise RuntimeError("simulated Replan failure")

    loop.replan.run = _raise_async

    message = await loop.handle_turn(session_id, 0, "a turn")

    assert message
    diagnostics = await diagnostics_store.get_for_turn(session_id, 0)
    assert any("Replan failed" in w for w in diagnostics.warnings)


@pytest.mark.asyncio(loop_scope="session")
async def test_plan_failure_falls_back_to_a_deterministic_action(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, diagnostics_store,
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    llm = StubLLMClient(canned={"PROPOSE:ACTIONS": _raise})
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, llm, diagnostics_store,
    )

    message = await loop.handle_turn(session_id, 0, "a turn")

    assert message  # Teach still ran, off the deterministic fallback action
    diagnostics = await diagnostics_store.get_for_turn(session_id, 0)
    assert any("Plan failed" in w for w in diagnostics.warnings)


@pytest.mark.asyncio(loop_scope="session")
async def test_teach_failure_returns_the_fixed_message_and_sets_teach_failed(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, diagnostics_store,
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    llm = StubLLMClient(canned={"TEACH:": _raise})
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, llm, diagnostics_store,
    )

    message = await loop.handle_turn(session_id, 0, "a turn")

    assert message == _TEACH_FAILURE_MESSAGE
    diagnostics = await diagnostics_store.get_for_turn(session_id, 0)
    assert diagnostics.teach_failed is True
    assert any("Teach failed" in w for w in diagnostics.warnings)


@pytest.mark.asyncio(loop_scope="session")
async def test_teach_failure_still_generates_and_keeps_the_branch_tree(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, diagnostics_store, branch_store,
):
    """BranchGenerate/SelectBranch/DerivePath now run before Teach, so
    they've already happened by the time Teach fails — nothing about
    them depends on Teach's success. The generation is kept (not
    discarded), since its predictions target Plan's planned action, not
    Teach's rendered text; a warning flags the failure for downstream
    match-rate analysis instead."""
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    llm = StubLLMClient(canned={"TEACH:": _raise})
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, llm, diagnostics_store, branch_store=branch_store,
    )

    await loop.handle_turn(session_id, 0, "a turn")

    async with clean_pool.acquire() as conn:
        names = {
            r["node_name"]
            for r in await conn.fetch(
                "SELECT node_name FROM node_calls WHERE session_id=$1", session_id
            )
        }
    assert "BranchGenerate" in names
    generation = await branch_store.get_latest_generation(session_id)
    assert generation is not None

    diagnostics = await diagnostics_store.get_for_turn(session_id, 0)
    assert diagnostics.teach_failed is True
    assert any(
        "BranchGenerate ran before Teach failed" in w for w in diagnostics.warnings
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_branch_resolve_failure_leaves_the_prior_generations_branches_untouched(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, diagnostics_store, branch_store,
):
    """Turn 1 still runs its own (successful) BranchGenerate regardless
    of whether turn 1's BranchResolve failed, so _prior_generation_id
    legitimately ends up pointing at turn 1's new generation either
    way — that's not what a failed resolve should be judged on. What
    actually matters: resolve() never got to run its status-mutation
    logic, so turn 0's branches must still be exactly as open as they
    were, not silently (and wrongly) marked matched/unmatched."""
    loop = _make_loop(
        store, transcript, node_calls, concept_graph, learner_overlay,
        revision_store, StubLLMClient(), diagnostics_store, branch_store=branch_store,
    )
    session_id = await transcript.create_session(learner_id, concept_graph_id)

    await loop.handle_turn(session_id, 0, "first turn")
    turn_0_generation = await branch_store.get_latest_generation(session_id)
    assert turn_0_generation is not None
    leaves_before = await branch_store.get_open_leaves(turn_0_generation.id)
    assert leaves_before  # something exists to (fail to) resolve

    loop.branch_resolve.run = _raise_async_factory()
    await loop.handle_turn(session_id, 1, "second turn")

    leaves_after = await branch_store.get_open_leaves(turn_0_generation.id)
    assert {b.id for b in leaves_after} == {b.id for b in leaves_before}


def _raise_async_factory():
    async def _raise_async(*_args, **_kwargs):
        raise RuntimeError("simulated BranchResolve failure")

    return _raise_async
