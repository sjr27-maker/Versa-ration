"""SessionLoop wiring for the evidence/options mechanism: a click
marks its branch matched directly (source="option_click"), skipping
BranchResolve's LLM matching entirely and propagating up the full
ancestor chain, exactly as a text match does but without the
interpretation step; a typed answer that CheckEvidence judges to
satisfy a requirement behaves identically to a click for evidence
purposes (but still goes through BranchResolve's own LLM prediction
matching independently, since evidence-satisfaction and leaf-prediction
matching are different questions); a typed answer that satisfies
nothing sets options_missed and that note reaches the next turn's
generation prompt.
"""

from __future__ import annotations

import json
import re
from uuid import uuid4

import pytest

from probe.diagnostics import TurnDiagnosticsStore
from probe.llm import StubLLMClient
from probe.loop import SessionLoop
from probe.models import BranchStatus


def _one_evidence_branch_response(requirement: str) -> str:
    return json.dumps(
        [
            {
                "statement": "wants to see a worked example before proceeding",
                "plausibility": 0.9,
                "predicted_next_turn": "will ask for a worked example",
                "requires_evidence": requirement,
            }
        ]
    )


def _options_echoing_branch_id(text: str):
    def _dispatch(prompt: str) -> str:
        match = re.search(r"id=([0-9a-fA-F-]{36})", prompt)
        assert match, f"expected a branch id in the GENERATE:OPTIONS prompt: {prompt}"
        return json.dumps([{"branch_id": match.group(1), "text": text}])

    return _dispatch


def _two_evidence_branches_response(req_a: str, req_b: str) -> str:
    return json.dumps(
        [
            {
                "statement": "wants a worked example",
                "plausibility": 0.9,
                "predicted_next_turn": "will ask for a worked example",
                "requires_evidence": req_a,
            },
            {
                "statement": "wants a geometric picture",
                "plausibility": 0.7,
                "predicted_next_turn": "will ask for a diagram",
                "requires_evidence": req_b,
            },
        ]
    )


def _options_echoing_both_branch_ids(prompt: str) -> str:
    ids = re.findall(r"id=([0-9a-fA-F-]{36})", prompt)
    assert len(ids) == 2, f"expected 2 branch ids in the prompt: {prompt}"
    return json.dumps(
        [
            {"branch_id": ids[0], "text": "Want to see a worked example?"},
            {"branch_id": ids[1], "text": "Want to see a geometric picture?"},
        ]
    )


def _check_evidence_matching_first_candidate(prompt: str) -> str:
    match = re.search(r"id=([0-9a-fA-F-]{36})", prompt)
    assert match, f"expected a branch id in the CHECK:EVIDENCE prompt: {prompt}"
    return json.dumps({"satisfied_branch_id": match.group(1), "confidence": 0.9})


def _make_loop(store, transcript, node_calls, concept_graph, learner_overlay,
               revision_store, branch_store, option_store, diagnostics_store, llm):
    return SessionLoop(
        hypothesis_store=store,
        transcript=transcript,
        node_calls=node_calls,
        concept_graph=concept_graph,
        learner_overlay=learner_overlay,
        revision_store=revision_store,
        llm=llm,
        branch_store=branch_store,
        option_store=option_store,
        diagnostics_store=diagnostics_store,
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_click_marks_its_branch_matched_via_option_click(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, branch_store, option_store,
):
    """A click is settled evidence, categorically more certain than a
    text match -- it bypasses RESOLVE:MATCH's LLM judgment entirely
    and marks its branch matched directly (source="option_click"),
    rather than being routed through the same fuzzy matching a typed
    message gets."""
    diagnostics_store = TurnDiagnosticsStore(clean_pool)
    requirement = "the student confirms they want to see a worked example"
    llm = StubLLMClient(
        canned={
            "GENERATE:INTENT": _one_evidence_branch_response(requirement),
            "GENERATE:OPTIONS": _options_echoing_branch_id(
                "Want to see a worked example first?"
            ),
        }
    )
    loop = _make_loop(store, transcript, node_calls, concept_graph, learner_overlay,
                       revision_store, branch_store, option_store, diagnostics_store, llm)
    session_id = await transcript.create_session(learner_id, concept_graph_id)

    await loop.handle_turn(session_id, 0, "I think I understand the idea generally")

    generation0 = await branch_store.get_latest_generation(session_id)
    branches0 = await branch_store.list_by_generation(generation0.id)
    assert len(branches0) == 1
    branch = branches0[0]
    assert branch.requires_evidence == requirement
    assert branch.evidence_satisfied is False

    options0 = await option_store.list_by_generation(generation0.id)
    assert len(options0) == 1
    option = options0[0]
    assert option.branch_id == branch.id

    await loop.handle_turn(session_id, 1, option.text, selected_option_id=option.id)

    refreshed_branch = await branch_store.get(branch.id)
    assert refreshed_branch.evidence_satisfied is True
    assert refreshed_branch.status is BranchStatus.MATCHED
    assert refreshed_branch.matched_via == "option_click"

    refreshed_option = await option_store.get(option.id)
    assert refreshed_option.status.value == "selected"

    resolve_call = await node_calls.get_latest_call(session_id, "BranchResolve")
    assert resolve_call.output_json["call_count"] == 0  # no LLM call at all
    assert resolve_call.output_json["source"] == "option_click"
    assert resolve_call.output_json["status"] == "matched"


@pytest.mark.asyncio(loop_scope="session")
async def test_click_propagates_matched_up_the_ancestor_chain(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, branch_store, option_store,
):
    """The clicked branch's full ancestor chain is marked matched too,
    via the same channel -- exactly as a text match already does."""
    diagnostics_store = TurnDiagnosticsStore(clean_pool)
    root_id = str(uuid4())
    requirement = "the student confirms they want to see a worked example"
    intent_response = json.dumps(
        [
            {
                "statement": "root intent",
                "plausibility": 0.9,
                "predicted_next_turn": "will ask for an example",
                "requires_evidence": None,
            }
        ]
    )
    expand_response = json.dumps(
        {
            "layer_label": "knowledge_gap",
            "children": [
                {
                    "statement": "specific gap needing an example",
                    "plausibility": 0.9,
                    "predicted_next_turn": "will ask for an example",
                    "requires_evidence": requirement,
                }
            ],
        }
    )
    llm = StubLLMClient(
        canned={
            "GENERATE:INTENT": intent_response,
            "GENERATE:EXPAND": expand_response,
            "GENERATE:OPTIONS": _options_echoing_branch_id("Want an example?"),
        }
    )
    loop = _make_loop(store, transcript, node_calls, concept_graph, learner_overlay,
                       revision_store, branch_store, option_store, diagnostics_store, llm)
    session_id = await transcript.create_session(learner_id, concept_graph_id)

    await loop.handle_turn(session_id, 0, "I think I get it")

    generation0 = await branch_store.get_latest_generation(session_id)
    branches0 = await branch_store.list_by_generation(generation0.id)
    root = next(b for b in branches0 if b.parent_id is None)
    child = next(b for b in branches0 if b.parent_id == root.id)
    assert child.requires_evidence == requirement

    options0 = await option_store.list_by_generation(generation0.id)
    option = next(o for o in options0 if o.branch_id == child.id)

    await loop.handle_turn(session_id, 1, option.text, selected_option_id=option.id)

    refreshed_root = await branch_store.get(root.id)
    refreshed_child = await branch_store.get(child.id)
    assert refreshed_child.status is BranchStatus.MATCHED
    assert refreshed_child.matched_via == "option_click"
    assert refreshed_root.status is BranchStatus.MATCHED
    assert refreshed_root.matched_via == "option_click"


@pytest.mark.asyncio(loop_scope="session")
async def test_clicking_one_option_supersedes_non_clicked_siblings_not_unmatched(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, branch_store, option_store,
):
    """The student chose one path; they didn't reject the others. A
    sibling branch that was never tested must not count against
    prediction accuracy the way a genuine miss would -- it becomes
    superseded, the same treatment a text match's own non-matching
    siblings already get, not unmatched."""
    diagnostics_store = TurnDiagnosticsStore(clean_pool)
    req_a = "the student confirms they want a worked example"
    req_b = "the student confirms they want a geometric picture"
    llm = StubLLMClient(
        canned={
            "GENERATE:INTENT": _two_evidence_branches_response(req_a, req_b),
            "GENERATE:OPTIONS": _options_echoing_both_branch_ids,
        }
    )
    loop = _make_loop(store, transcript, node_calls, concept_graph, learner_overlay,
                       revision_store, branch_store, option_store, diagnostics_store, llm)
    session_id = await transcript.create_session(learner_id, concept_graph_id)

    await loop.handle_turn(session_id, 0, "I think I get the idea")

    generation0 = await branch_store.get_latest_generation(session_id)
    branches0 = await branch_store.list_by_generation(generation0.id)
    assert len(branches0) == 2
    clicked_branch = next(b for b in branches0 if b.requires_evidence == req_a)
    sibling_branch = next(b for b in branches0 if b.requires_evidence == req_b)

    options0 = await option_store.list_by_generation(generation0.id)
    clicked_option = next(o for o in options0 if o.branch_id == clicked_branch.id)

    await loop.handle_turn(
        session_id, 1, clicked_option.text, selected_option_id=clicked_option.id
    )

    refreshed_clicked = await branch_store.get(clicked_branch.id)
    refreshed_sibling = await branch_store.get(sibling_branch.id)
    assert refreshed_clicked.status is BranchStatus.MATCHED
    assert refreshed_sibling.status is BranchStatus.SUPERSEDED
    assert refreshed_sibling.status is not BranchStatus.UNMATCHED


@pytest.mark.asyncio(loop_scope="session")
async def test_a_typed_match_satisfies_the_branch_identically_to_a_click(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, branch_store, option_store,
):
    diagnostics_store = TurnDiagnosticsStore(clean_pool)
    requirement = "the student confirms they want to see a worked example"
    llm = StubLLMClient(
        canned={
            "GENERATE:INTENT": _one_evidence_branch_response(requirement),
            "GENERATE:OPTIONS": _options_echoing_branch_id(
                "Want to see a worked example first?"
            ),
            "CHECK:EVIDENCE": _check_evidence_matching_first_candidate,
        }
    )
    loop = _make_loop(store, transcript, node_calls, concept_graph, learner_overlay,
                       revision_store, branch_store, option_store, diagnostics_store, llm)
    session_id = await transcript.create_session(learner_id, concept_graph_id)

    await loop.handle_turn(session_id, 0, "I think I understand the idea generally")
    generation0 = await branch_store.get_latest_generation(session_id)
    branch = (await branch_store.list_by_generation(generation0.id))[0]

    # Typed, not clicked -- no selected_option_id.
    await loop.handle_turn(session_id, 1, "yes please show me a worked example")

    refreshed_branch = await branch_store.get(branch.id)
    assert refreshed_branch.evidence_satisfied is True
    assert refreshed_branch.status is BranchStatus.OPEN

    diag = await diagnostics_store.get_for_turn(session_id, 1)
    assert diag.options_missed is False
    assert diag.node_call_counts.get("CheckEvidence") == 1
    assert not any("CheckEvidence failed" in w for w in diag.warnings)

    options0 = await option_store.list_by_generation(generation0.id)
    # Never clicked -- superseded by the same resolve() call, not left
    # dangling open just because a different channel satisfied the branch.
    assert options0[0].status.value == "superseded"


@pytest.mark.asyncio(loop_scope="session")
async def test_typed_text_satisfying_nothing_sets_options_missed_and_informs_next_generation(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, branch_store, option_store,
):
    diagnostics_store = TurnDiagnosticsStore(clean_pool)
    requirement = "the student confirms they want to see a worked example"
    llm = StubLLMClient(
        canned={
            "GENERATE:INTENT": _one_evidence_branch_response(requirement),
            "GENERATE:OPTIONS": _options_echoing_branch_id(
                "Want to see a worked example first?"
            ),
            # Default CHECK:EVIDENCE -> satisfied_branch_id: null.
        }
    )
    loop = _make_loop(store, transcript, node_calls, concept_graph, learner_overlay,
                       revision_store, branch_store, option_store, diagnostics_store, llm)
    session_id = await transcript.create_session(learner_id, concept_graph_id)

    await loop.handle_turn(session_id, 0, "I think I understand the idea generally")
    await loop.handle_turn(session_id, 1, "totally unrelated tangent about something else")

    diag1 = await diagnostics_store.get_for_turn(session_id, 1)
    assert diag1.options_missed is True
    assert any("options_missed" in w for w in diag1.warnings)
    # A genuine "nothing satisfied" LLM judgment, not a swallowed error
    # masquerading as one — CheckEvidence must actually have completed
    # (regression guard: it used to return a bare None on this exact
    # path, which node_calls' NOT NULL output_json column rejected,
    # silently caught by _call_node_or_warn and misreported as this
    # same options_missed=True outcome for the wrong reason).
    assert diag1.node_call_counts.get("CheckEvidence") == 1
    assert not any("CheckEvidence failed" in w for w in diag1.warnings)

    # Turn 2's own GENERATE:INTENT prompt must carry the note forward.
    await loop.handle_turn(session_id, 2, "another message")
    intent_prompts = [p for p in llm.prompts if p.startswith("GENERATE:INTENT")]
    assert len(intent_prompts) >= 2
    assert "options offered last turn did not match" in intent_prompts[-1]
