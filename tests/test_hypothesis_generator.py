import json
import logging

import pytest

from probe.hypothesis_generator import HypothesisGenerator
from probe.llm import StubLLMClient
from probe.models import BranchStatus
from probe.reasoning_budget import BranchBudgetConfig

STRONG_STATEMENT = "strong bet: the student wants to connect this to something familiar"
WEAK_STATEMENT = "weak bet: something vague and unlikely"

_INTENT_RESPONSE = json.dumps(
    [
        {
            "statement": STRONG_STATEMENT,
            "plausibility": 0.9,
            "predicted_next_turn": "will ask for an analogy",
        },
        {
            "statement": WEAK_STATEMENT,
            "plausibility": 0.1,
            "predicted_next_turn": "will say something incoherent",
        },
    ]
)

# Exactly one child per expansion call, always plausible enough to keep
# expanding — this is what lets the strong branch's depth be driven
# purely by max_depth in these tests, deterministically.
_EXPAND_RESPONSE = json.dumps(
    {
        "layer_label": "deeper",
        "children": [
            {
                "statement": "one more specific bet",
                "plausibility": 0.9,
                "predicted_next_turn": "will do something specific",
            }
        ],
    }
)


@pytest.mark.asyncio(loop_scope="session")
async def test_generate_produces_a_valid_tree_with_correct_parent_chains(
    branch_store, store, transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    llm = StubLLMClient(
        canned={"GENERATE:INTENT": _INTENT_RESPONSE, "GENERATE:EXPAND": _EXPAND_RESPONSE}
    )
    gen = HypothesisGenerator(
        llm, branch_store, BranchBudgetConfig(max_depth=3, max_total_branches=20)
    )

    result = await gen.generate(session_id, 0, store, "student context so far", learner_id)

    assert result.generation.root_count == 2
    by_id = {b.id: b for b in result.branches}
    for b in result.branches:
        if b.parent_id is not None:
            assert b.parent_id in by_id, "every parent_id must point at a branch in this generation"
            assert by_id[b.parent_id].depth == b.depth - 1


@pytest.mark.asyncio(loop_scope="session")
async def test_strong_branch_expands_deeper_than_weak_branch_same_generation(
    branch_store, store, transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    llm = StubLLMClient(
        canned={"GENERATE:INTENT": _INTENT_RESPONSE, "GENERATE:EXPAND": _EXPAND_RESPONSE}
    )
    gen = HypothesisGenerator(
        llm, branch_store, BranchBudgetConfig(max_depth=3, max_total_branches=20)
    )

    result = await gen.generate(session_id, 0, store, "student context so far", learner_id)

    strong_root = next(b for b in result.branches if b.statement == STRONG_STATEMENT)
    weak_root = next(b for b in result.branches if b.statement == WEAK_STATEMENT)

    assert weak_root.depth == 0
    assert weak_root.is_leaf is True  # too implausible to clear the expansion filter
    assert strong_root.is_leaf is False  # plausible enough, kept expanding

    depths_reached = sorted(b.depth for b in result.branches)
    assert depths_reached == [0, 0, 1, 2, 3]  # weak stops at 0; strong's chain reaches max_depth
    assert result.call_count == 4  # 1 intent call + one expand call per depth (0->1->2->3)


@pytest.mark.asyncio(loop_scope="session")
async def test_every_leaf_carries_a_concretely_checkable_prediction(
    branch_store, store, transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    llm = StubLLMClient(
        canned={"GENERATE:INTENT": _INTENT_RESPONSE, "GENERATE:EXPAND": _EXPAND_RESPONSE}
    )
    gen = HypothesisGenerator(
        llm, branch_store, BranchBudgetConfig(max_depth=3, max_total_branches=20)
    )

    result = await gen.generate(session_id, 0, store, "student context so far", learner_id)

    leaves = [b for b in result.branches if b.is_leaf]
    assert leaves, "expected at least one leaf branch"
    for leaf in leaves:
        assert leaf.predicted_next_turn.strip() != ""


@pytest.mark.asyncio(loop_scope="session")
async def test_branches_clearing_redundancy_check_are_logged_with_siblings(
    branch_store, store, transcript, clean_pool, learner_id, concept_graph_id, caplog
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    llm = StubLLMClient(
        canned={"GENERATE:INTENT": _INTENT_RESPONSE, "GENERATE:EXPAND": _EXPAND_RESPONSE}
    )
    gen = HypothesisGenerator(
        llm, branch_store, BranchBudgetConfig(max_depth=1, max_total_branches=20)
    )

    with caplog.at_level(logging.INFO, logger="probe.hypothesis_generator"):
        await gen.generate(session_id, 0, store, "student context so far", learner_id)

    info_records = [
        r for r in caplog.records if "cleared the redundancy check" in r.message
    ]
    assert info_records, "expected a logged line for the branch that survived expansion"


@pytest.mark.asyncio(loop_scope="session")
async def test_resolve_matches_a_leaf_and_propagates_matched_status_up_the_chain(
    branch_store, store, transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    gen_llm = StubLLMClient(
        canned={"GENERATE:INTENT": _INTENT_RESPONSE, "GENERATE:EXPAND": _EXPAND_RESPONSE}
    )
    generator = HypothesisGenerator(
        gen_llm, branch_store, BranchBudgetConfig(max_depth=2, max_total_branches=20)
    )
    generation = await generator.generate(session_id, 0, store, "student context so far", learner_id)

    leaves = [b for b in generation.branches if b.is_leaf]
    target_leaf = max(leaves, key=lambda b: b.depth)
    assert target_leaf.depth > 0, "need a leaf with ancestors to prove propagation"

    match_response = json.dumps(
        {"matched_branch_id": str(target_leaf.id), "confidence": 0.9}
    )
    resolve_llm = StubLLMClient(canned={"RESOLVE:MATCH": match_response})
    resolver = HypothesisGenerator(resolve_llm, branch_store)

    result = await resolver.resolve(session_id, 1, "the student's real next message")

    assert result.status == "matched"
    assert result.matched_branch_id == target_leaf.id
    assert result.call_count == 1
    assert (await branch_store.get(target_leaf.id)).status is BranchStatus.MATCHED
    ancestors = await branch_store.get_ancestors(target_leaf.id)
    assert ancestors
    for ancestor in ancestors:
        assert (await branch_store.get(ancestor.id)).status is BranchStatus.MATCHED


@pytest.mark.asyncio(loop_scope="session")
async def test_resolve_no_match_marks_unmatched_and_logs_explicitly(
    branch_store, store, transcript, clean_pool, learner_id, concept_graph_id, caplog
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    gen_llm = StubLLMClient(
        canned={"GENERATE:INTENT": _INTENT_RESPONSE, "GENERATE:EXPAND": _EXPAND_RESPONSE}
    )
    generator = HypothesisGenerator(
        gen_llm, branch_store, BranchBudgetConfig(max_depth=1, max_total_branches=20)
    )
    generation = await generator.generate(session_id, 0, store, "student context so far", learner_id)
    leaves_before = [b for b in generation.branches if b.is_leaf]
    assert leaves_before

    # StubLLMClient's RESOLVE:MATCH default is {"matched_branch_id": null, ...}.
    resolver = HypothesisGenerator(StubLLMClient(), branch_store)

    with caplog.at_level(logging.WARNING, logger="probe.hypothesis_generator"):
        result = await resolver.resolve(session_id, 1, "a totally unrelated response")

    assert result.status == "unmatched"
    assert result.matched_branch_id is None
    warnings = [r for r in caplog.records if "no leaf branch" in r.message]
    assert warnings, "a no-match must be logged explicitly, not silent"
    for leaf in leaves_before:
        assert (await branch_store.get(leaf.id)).status is BranchStatus.UNMATCHED


@pytest.mark.asyncio(loop_scope="session")
async def test_resolve_supersedes_intermediate_open_branches_and_never_deletes(
    branch_store, store, transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    gen_llm = StubLLMClient(
        canned={"GENERATE:INTENT": _INTENT_RESPONSE, "GENERATE:EXPAND": _EXPAND_RESPONSE}
    )
    generator = HypothesisGenerator(
        gen_llm, branch_store, BranchBudgetConfig(max_depth=2, max_total_branches=20)
    )
    generation = await generator.generate(session_id, 0, store, "student context so far", learner_id)
    intermediate = [b for b in generation.branches if not b.is_leaf]
    assert intermediate, "expected at least one non-leaf (intermediate) branch"

    resolver = HypothesisGenerator(StubLLMClient(), branch_store)
    await resolver.resolve(session_id, 1, "unrelated")

    for branch in intermediate:
        refreshed = await branch_store.get(branch.id)
        assert refreshed is not None, "resolve() must never delete a branch"
        assert refreshed.status is BranchStatus.SUPERSEDED

    async with clean_pool.acquire() as conn:
        row_count = await conn.fetchval(
            "SELECT count(*) FROM branches WHERE generation_id = $1",
            generation.generation.id,
        )
    assert row_count == len(generation.branches), "row count must be unchanged — append-only"


@pytest.mark.asyncio(loop_scope="session")
async def test_resolve_with_no_prior_generation_is_unmatched_and_free(
    branch_store, transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    resolver = HypothesisGenerator(StubLLMClient(), branch_store)

    result = await resolver.resolve(session_id, 0, "first ever student message")

    assert result.status == "unmatched"
    assert result.matched_branch_id is None
    assert result.call_count == 0  # nothing to resolve against — no LLM call made
