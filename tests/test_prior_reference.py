"""Teach's working-memory fix: a compact recent-history input plus a
tracked list of examples/analogies used this session, so Teach can
notice its own recent work instead of contradicting or ignoring it —
and, specifically, so that when the student references something from
earlier ("the example you just gave"), Teach names and uses that
specific thing rather than introducing an unrelated new one. Covers
ExtractTeachingArtifact's own extraction, Teach's prompt blocks, the
post-Teach check_prior_reference_unaddressed backstop, and a full
loop-level 2-turn integration test.
"""

from __future__ import annotations

import json

import pytest

from probe.llm import StubLLMClient
from probe.loop import SessionLoop
from probe.models import CandidateAction, TeachingAction, TeachingArtifact
from probe.nodes import (
    ExtractTeachingArtifact,
    Teach,
    check_prior_reference_unaddressed,
    detect_prior_reference,
)

# --- ExtractTeachingArtifact --------------------------------------------


@pytest.mark.asyncio
async def test_extract_teaching_artifact_captures_example_and_analogy():
    llm = StubLLMClient(
        canned={
            "EXTRACT:ARTIFACT": json.dumps(
                {"example": "sin(x^2)", "analogy": "nested Russian dolls"}
            )
        }
    )
    node = ExtractTeachingArtifact(llm)

    result = await node.run("some rendered teach output")

    assert result.example == "sin(x^2)"
    assert result.analogy == "nested Russian dolls"
    assert node.last_call_count == 1


@pytest.mark.asyncio
async def test_extract_teaching_artifact_defaults_to_none_none():
    """No canned override -> StubLLMClient's conservative default,
    same convention as EXTRACT:REQUEST/MISMATCH:DETECT."""
    node = ExtractTeachingArtifact(StubLLMClient())

    result = await node.run("some rendered teach output")

    assert result.example is None
    assert result.analogy is None


@pytest.mark.asyncio
async def test_extract_teaching_artifact_treats_blank_strings_as_none():
    llm = StubLLMClient(
        canned={"EXTRACT:ARTIFACT": json.dumps({"example": "   ", "analogy": None})}
    )
    node = ExtractTeachingArtifact(llm)

    result = await node.run("some rendered teach output")

    assert result.example is None
    assert result.analogy is None


# --- detect_prior_reference ----------------------------------------------


def test_detect_prior_reference_fires_on_the_observed_phrasing():
    assert detect_prior_reference(
        "Does that still apply to the chain rule example you just gave?"
    ) is True


def test_detect_prior_reference_quiet_on_a_fresh_question():
    assert detect_prior_reference("What is the derivative of x^3?") is False


# --- check_prior_reference_unaddressed ------------------------------------


def test_check_fires_when_a_new_unrelated_analogy_replaces_the_named_one():
    """The exact observed live failure: the student asked whether an
    idea still applies to "the chain rule example you just gave"
    (sin(x^2), worked the prior turn) and Teach introduced an unrelated
    gears analogy instead of naming it."""
    last_artifact = TeachingArtifact(example="sin(x^2)", analogy=None)
    output = (
        "Think of gears turning inside one another -- as the outer "
        "gear turns, it drives the inner gear, and that's the same "
        "layering idea at work here."
    )
    assert check_prior_reference_unaddressed(last_artifact, output) is True


def test_check_stays_quiet_when_the_named_thing_is_actually_used():
    last_artifact = TeachingArtifact(example="sin(x^2)", analogy=None)
    output = (
        "Yes -- the same idea applies to sin(x^2): its derivative was "
        "2x cos(x^2), and that chain-rule logic carries over directly."
    )
    assert check_prior_reference_unaddressed(last_artifact, output) is False


def test_check_fires_when_analogy_is_reused_but_the_example_is_swapped():
    """The real live failure caught after the first version of this fix
    shipped: turn 4 kept the "nested dolls" analogy language but
    silently worked a brand-new function, (3x+1)^5, instead of
    continuing with sin(x^2) -- the actual thing "the chain rule
    example you just gave" referred to. A pooled (either-field)
    word-overlap check would wave this through as addressed since the
    analogy words alone are present; each tracked field must be
    checked independently."""
    last_artifact = TeachingArtifact(example="sin(x^2)", analogy="nested Russian dolls")
    output = (
        "The same nesting idea applies to a function like (3x + 1)^5, "
        "where we again have an outer doll wrapping around an inner "
        "doll nested inside."
    )
    assert check_prior_reference_unaddressed(last_artifact, output) is True


def test_check_stays_quiet_when_both_tracked_fields_are_present():
    last_artifact = TeachingArtifact(example="sin(x^2)", analogy="nested Russian dolls")
    output = (
        "Yes -- the same nested-doll idea applies to sin(x^2): its "
        "derivative was 2x cos(x^2)."
    )
    assert check_prior_reference_unaddressed(last_artifact, output) is False


def test_check_returns_false_when_no_prior_artifact_is_tracked():
    assert check_prior_reference_unaddressed(None, "some response") is False


def test_check_returns_false_when_tracked_artifact_has_no_content():
    last_artifact = TeachingArtifact(example=None, analogy=None)
    assert check_prior_reference_unaddressed(last_artifact, "some response") is False


# --- Teach's prompt --------------------------------------------------------


@pytest.mark.asyncio
async def test_teach_prompt_includes_recent_history_and_examples_used():
    llm = StubLLMClient()
    teach = Teach(llm)

    await teach.run(
        action=CandidateAction(action=TeachingAction.EXPLAIN, target_concept="chain_rule"),
        student_message="Does that still apply to the example you just gave?",
        recent_history="turn 0 student: What about the chain rule?\n"
        "turn 0 you (tutor): The chain rule handles nested functions.",
        examples_used="turn 0 — example: sin(x^2)",
    )

    prompt = llm.prompts[-1]
    assert "turn 0 you (tutor): The chain rule handles nested functions." in prompt
    assert "example: sin(x^2)" in prompt
    assert "MUST name and use that specific thing" in prompt


@pytest.mark.asyncio
async def test_teach_prompt_omits_history_blocks_when_absent():
    llm = StubLLMClient()
    teach = Teach(llm)

    await teach.run(
        action=CandidateAction(action=TeachingAction.EXPLAIN, target_concept="derivative"),
        student_message="what is a derivative?",
    )

    prompt = llm.prompts[-1]
    assert "Recent conversation" not in prompt
    assert "already used this session" not in prompt


# --- Loop-level integration -------------------------------------------------


def _teach_sequence(turn0_text: str, turn1_text: str):
    calls = {"n": 0}

    def _respond(_prompt: str) -> str:
        calls["n"] += 1
        return turn0_text if calls["n"] == 1 else turn1_text

    return _respond


_TURN0_TEACH_OUTPUT = (
    "For sin(x^2), the outer function is sine and the inner is x^2. "
    "The derivative works out to 2x cos(x^2)."
)


@pytest.mark.asyncio(loop_scope="session")
async def test_loop_flags_when_turn2_ignores_the_referenced_example(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, diagnostics_store,
):
    llm = StubLLMClient(
        canned={
            "TEACH:": _teach_sequence(
                _TURN0_TEACH_OUTPUT,
                "Think of gears turning inside one another -- the same "
                "layering idea applies here.",
            ),
            "EXTRACT:ARTIFACT": json.dumps({"example": "sin(x^2)", "analogy": None}),
        }
    )
    loop = SessionLoop(
        hypothesis_store=store, transcript=transcript, node_calls=node_calls,
        concept_graph=concept_graph, learner_overlay=learner_overlay,
        revision_store=revision_store, llm=llm, diagnostics_store=diagnostics_store,
    )
    session_id = await transcript.create_session(learner_id, concept_graph_id)

    await loop.handle_turn(session_id, 0, "Can you show me the chain rule with sin(x^2)?")
    await loop.handle_turn(
        session_id, 1, "Does that still apply to the chain rule example you just gave?"
    )

    diag = await diagnostics_store.get_for_turn(session_id, 1)
    assert diag.prior_reference_detected is True
    assert diag.prior_reference_unaddressed is True
    assert any("prior_reference_unaddressed" in w for w in diag.warnings)

    artifact_call = await node_calls.get_call_for_turn(
        session_id, 0, "ExtractTeachingArtifact"
    )
    assert artifact_call is not None
    assert artifact_call.output_json["example"] == "sin(x^2)"


@pytest.mark.asyncio(loop_scope="session")
async def test_loop_does_not_flag_when_turn2_names_the_referenced_example(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, diagnostics_store,
):
    llm = StubLLMClient(
        canned={
            "TEACH:": _teach_sequence(
                _TURN0_TEACH_OUTPUT,
                "Yes -- the same idea applies to sin(x^2): its derivative "
                "was 2x cos(x^2), so that chain-rule logic carries over.",
            ),
            "EXTRACT:ARTIFACT": json.dumps({"example": "sin(x^2)", "analogy": None}),
        }
    )
    loop = SessionLoop(
        hypothesis_store=store, transcript=transcript, node_calls=node_calls,
        concept_graph=concept_graph, learner_overlay=learner_overlay,
        revision_store=revision_store, llm=llm, diagnostics_store=diagnostics_store,
    )
    session_id = await transcript.create_session(learner_id, concept_graph_id)

    await loop.handle_turn(session_id, 0, "Can you show me the chain rule with sin(x^2)?")
    await loop.handle_turn(
        session_id, 1, "Does that still apply to the chain rule example you just gave?"
    )

    diag = await diagnostics_store.get_for_turn(session_id, 1)
    assert diag.prior_reference_detected is True
    assert diag.prior_reference_unaddressed is False
    assert not any("prior_reference_unaddressed" in w for w in diag.warnings)


@pytest.mark.asyncio(loop_scope="session")
async def test_loop_skips_the_check_entirely_when_no_backward_reference(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, diagnostics_store,
):
    loop = SessionLoop(
        hypothesis_store=store, transcript=transcript, node_calls=node_calls,
        concept_graph=concept_graph, learner_overlay=learner_overlay,
        revision_store=revision_store, llm=StubLLMClient(),
        diagnostics_store=diagnostics_store,
    )
    session_id = await transcript.create_session(learner_id, concept_graph_id)

    await loop.handle_turn(session_id, 0, "what is a derivative?")

    diag = await diagnostics_store.get_for_turn(session_id, 0)
    assert diag.prior_reference_detected is False
    assert diag.prior_reference_unaddressed is False
