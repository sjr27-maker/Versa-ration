"""ExtractRequest's precedence fix: a student's concrete, answerable
request (a specific function, problem, example, or question) must take
precedence over Plan's chosen target_concept and DerivePath's scope —
the pedagogical machinery decides HOW to teach, never WHETHER to
answer what was asked. Covers each stage independently (extraction,
Plan's prompt, DerivePath's structural enforcement, Teach's prompt)
plus the post-Teach check_explicit_request_unaddressed backstop and a
full loop-level integration test.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from probe.hypothesis_generator import DerivePath
from probe.llm import StubLLMClient
from probe.loop import SessionLoop
from probe.models import Branch, CandidateAction, ExplicitRequest, TeachingAction
from probe.nodes import ExtractRequest, Plan, Teach, check_explicit_request_unaddressed
from probe.value_function import ValueFunction

# --- ExtractRequest ---------------------------------------------------


@pytest.mark.asyncio
async def test_extract_request_captures_a_named_function():
    llm = StubLLMClient(
        canned={
            "EXTRACT:REQUEST": json.dumps(
                {"present": True, "what": "show me with sin(x^2) as an example"}
            )
        }
    )
    node = ExtractRequest(llm)

    result = await node.run("Can you show me with sin(x^2) as an example?")

    assert result.present is True
    assert result.what == "show me with sin(x^2) as an example"
    assert node.last_call_count == 1


@pytest.mark.asyncio
async def test_extract_request_defaults_to_absent():
    """No canned override -> StubLLMClient's conservative default
    (present=false), same convention as MISMATCH:DETECT/GROUND:CONCEPT."""
    node = ExtractRequest(StubLLMClient())

    result = await node.run("what is a derivative?")

    assert result.present is False
    assert result.what is None


@pytest.mark.asyncio
async def test_extract_request_treats_present_true_with_no_what_as_absent():
    """No partial-request state (see ExplicitRequest's docstring): a
    malformed present=true with a blank/missing `what` collapses to
    the same "nothing to prioritize" outcome as present=false."""
    llm = StubLLMClient(
        canned={"EXTRACT:REQUEST": json.dumps({"present": True, "what": "   "})}
    )
    node = ExtractRequest(llm)

    result = await node.run("some message")

    assert result.present is False
    assert result.what is None


# --- Plan's proposer prompt --------------------------------------------


@pytest.mark.asyncio
async def test_plan_proposer_prompt_contains_the_explicit_request():
    llm = StubLLMClient()
    plan = Plan(ValueFunction(llm), llm)
    explicit_request = ExplicitRequest(present=True, what="differentiate sin(x^2)")

    await plan.run(
        hypotheses=[],
        concept_state={},
        generation_width=2,
        explicit_request=explicit_request,
    )

    prompt = next(p for p in llm.prompts if p.startswith("PROPOSE:ACTIONS"))
    assert "differentiate sin(x^2)" in prompt
    assert "MUST be answered, not substituted" in prompt


@pytest.mark.asyncio
async def test_plan_proposer_prompt_omits_the_block_when_absent():
    llm = StubLLMClient()
    plan = Plan(ValueFunction(llm), llm)

    await plan.run(
        hypotheses=[],
        concept_state={},
        generation_width=2,
        explicit_request=ExplicitRequest(present=False, what=None),
    )

    prompt = next(p for p in llm.prompts if p.startswith("PROPOSE:ACTIONS"))
    assert "MUST be answered" not in prompt


# --- DerivePath's scope --------------------------------------------------


def _branch(**overrides) -> Branch:
    defaults = {
        "generation_id": uuid4(),
        "session_id": uuid4(),
        "turn_index": 0,
        "depth": 0,
        "depth_label": "intent",
        "statement": "wants to see chain rule applied",
        "predicted_next_turn": "will ask for a worked example",
        "plausibility": 0.8,
        "is_leaf": True,
    }
    defaults.update(overrides)
    return Branch(**defaults)


@pytest.mark.asyncio
async def test_derive_path_prompt_contains_the_explicit_request():
    llm = StubLLMClient()
    derive_path = DerivePath(llm)
    explicit_request = ExplicitRequest(present=True, what="differentiate sin(x^2)")

    await derive_path.run(
        path=[_branch()],
        student_message="Can you show me with sin(x^2)?",
        action_rationale="teach the chain rule",
        explicit_request=explicit_request,
    )

    prompt = llm.prompts[-1]
    assert "differentiate sin(x^2)" in prompt
    assert "scope MUST include, not replace" in prompt


@pytest.mark.asyncio
async def test_derive_path_scope_structurally_includes_explicit_request_even_if_model_omits_it():
    """The prompt asks the model to include it, but prompts get
    ignored -- this is the structural backstop, not the prompt ask:
    even when DERIVE:PATH's own response has a scope that never
    mentions the request, the returned PathRequirement.scope must."""
    llm = StubLLMClient(
        canned={
            "DERIVE:PATH": json.dumps(
                {
                    "current_belief": "",
                    "needed": "",
                    "must_not_assume": [],
                    "scope": "the chain rule in general",
                }
            )
        }
    )
    derive_path = DerivePath(llm)
    explicit_request = ExplicitRequest(present=True, what="differentiate sin(x^2)")

    result = await derive_path.run(
        path=[_branch()],
        student_message="Can you show me with sin(x^2)?",
        action_rationale="teach the chain rule",
        explicit_request=explicit_request,
    )

    assert "differentiate sin(x^2)" in result.scope
    assert "the chain rule in general" in result.scope  # framing kept, not replaced


@pytest.mark.asyncio
async def test_derive_path_scope_not_duplicated_when_model_already_included_it():
    llm = StubLLMClient(
        canned={
            "DERIVE:PATH": json.dumps(
                {
                    "current_belief": "",
                    "needed": "",
                    "must_not_assume": [],
                    "scope": "differentiate sin(x^2) using the chain rule",
                }
            )
        }
    )
    derive_path = DerivePath(llm)
    explicit_request = ExplicitRequest(present=True, what="differentiate sin(x^2)")

    result = await derive_path.run(
        path=[_branch()],
        student_message="Can you show me with sin(x^2)?",
        action_rationale="teach the chain rule",
        explicit_request=explicit_request,
    )

    assert result.scope == "differentiate sin(x^2) using the chain rule"


@pytest.mark.asyncio
async def test_derive_path_scope_unmodified_when_no_explicit_request():
    llm = StubLLMClient(
        canned={
            "DERIVE:PATH": json.dumps(
                {
                    "current_belief": "",
                    "needed": "",
                    "must_not_assume": [],
                    "scope": "the chain rule in general",
                }
            )
        }
    )
    derive_path = DerivePath(llm)

    result = await derive_path.run(
        path=[_branch()],
        student_message="tell me more",
        action_rationale="teach the chain rule",
        explicit_request=ExplicitRequest(present=False, what=None),
    )

    assert result.scope == "the chain rule in general"


# --- Teach's prompt --------------------------------------------------------


@pytest.mark.asyncio
async def test_teach_prompt_states_explicit_request_separately_from_path_requirement():
    from probe.models import PathRequirement

    llm = StubLLMClient()
    teach = Teach(llm)
    explicit_request = ExplicitRequest(present=True, what="differentiate sin(x^2)")
    path_requirement = PathRequirement(
        current_belief="understands the chain rule conceptually",
        needed="a worked example",
        must_not_assume=[],
        scope="chain rule mechanics",
    )

    await teach.run(
        action=CandidateAction(action=TeachingAction.EXAMPLE, target_concept="chain_rule"),
        student_message="Can you show me with sin(x^2)?",
        path_requirement=path_requirement,
        explicit_request=explicit_request,
    )

    prompt = llm.prompts[-1]
    assert "differentiate sin(x^2)" in prompt
    assert "MUST fully answer within this response" in prompt
    assert "Deferring it" in prompt
    # Separate from path_requirement's own scope field -- both appear,
    # neither replaces the other.
    assert "chain rule mechanics" in prompt


@pytest.mark.asyncio
async def test_teach_prompt_omits_explicit_request_block_when_absent():
    llm = StubLLMClient()
    teach = Teach(llm)

    await teach.run(
        action=CandidateAction(action=TeachingAction.EXPLAIN, target_concept="derivative"),
        student_message="what is a derivative?",
        explicit_request=ExplicitRequest(present=False, what=None),
    )

    prompt = llm.prompts[-1]
    assert "MUST fully answer" not in prompt


# --- check_explicit_request_unaddressed ------------------------------------


def test_check_fires_when_output_only_defers():
    """The exact observed live failure: Plan substituted (3x+1)^2 and
    Teach wrote this sentence instead of ever computing sin(x^2)."""
    output = (
        "To understand how to differentiate a composite function, we "
        "look at (3x+1)^2 as an example, taking the derivative of the "
        "outer layer first. We can apply this exact same perspective "
        "to sin(x^2) by identifying its own inner and outer layers."
    )
    assert check_explicit_request_unaddressed("differentiate sin(x^2)", output) is True


def test_check_fires_when_never_mentioned_at_all():
    output = "Let's talk about the chain rule using nesting dolls as an analogy."
    assert check_explicit_request_unaddressed("differentiate sin(x^2)", output) is True


def test_check_stays_quiet_when_actually_worked():
    output = (
        "The derivative of sin(something) is cos(something), so keeping "
        "the inner part the same gives us cos(x^2). Next, the derivative "
        "of the inner function, x^2, is 2x. Multiplying these together "
        "gives the final derivative of 2x cos(x^2)."
    )
    assert check_explicit_request_unaddressed("differentiate sin(x^2)", output) is False


def test_check_stays_quiet_when_worked_despite_a_later_deferral_sentence():
    """One worked sentence is enough, regardless of how many hedges
    surround it elsewhere in the same response."""
    output = (
        "The derivative of sin(x^2) is 2x cos(x^2), found via the chain "
        "rule. We can explore more trig examples like this next time."
    )
    assert check_explicit_request_unaddressed("differentiate sin(x^2)", output) is False


def test_check_returns_false_when_request_has_no_distinctive_words():
    """Nothing to check against -- same "don't force it" discipline as
    the rest of this heuristic family (RESOLVE:MATCH, etc.)."""
    assert check_explicit_request_unaddressed("this and that", "some response") is False


# --- Loop-level integration -------------------------------------------------


def _plan_targeting_chain_rule() -> str:
    return json.dumps(
        [{"action": "example", "target_concept": None, "rationale": "show the chain rule"}]
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_loop_persists_explicit_request_and_flags_when_teach_defers(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, diagnostics_store,
):
    llm = StubLLMClient(
        canned={
            "EXTRACT:REQUEST": json.dumps(
                {"present": True, "what": "differentiate sin(x^2)"}
            ),
            "PROPOSE:ACTIONS": _plan_targeting_chain_rule(),
            "TEACH:": (
                "We can apply this exact same perspective to sin(x^2) "
                "by identifying its own inner and outer layers."
            ),
        }
    )
    loop = SessionLoop(
        hypothesis_store=store, transcript=transcript, node_calls=node_calls,
        concept_graph=concept_graph, learner_overlay=learner_overlay,
        revision_store=revision_store, llm=llm, diagnostics_store=diagnostics_store,
    )
    session_id = await transcript.create_session(learner_id, concept_graph_id)

    await loop.handle_turn(session_id, 0, "Can you show me with sin(x^2)?")

    diag = await diagnostics_store.get_for_turn(session_id, 0)
    assert diag.explicit_request_present is True
    assert diag.explicit_request_what == "differentiate sin(x^2)"
    assert diag.explicit_request_unaddressed is True
    assert any("explicit_request_unaddressed" in w for w in diag.warnings)

    extract_call = await node_calls.get_call_for_turn(session_id, 0, "ExtractRequest")
    assert extract_call is not None
    assert extract_call.output_json["present"] is True


@pytest.mark.asyncio(loop_scope="session")
async def test_loop_does_not_flag_when_teach_actually_answers(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph_id,
    concept_graph, learner_overlay, revision_store, diagnostics_store,
):
    llm = StubLLMClient(
        canned={
            "EXTRACT:REQUEST": json.dumps(
                {"present": True, "what": "differentiate sin(x^2)"}
            ),
            "PROPOSE:ACTIONS": _plan_targeting_chain_rule(),
            "TEACH:": (
                "The derivative of sin(x^2) is 2x cos(x^2), found by "
                "multiplying the outer and inner derivatives together."
            ),
        }
    )
    loop = SessionLoop(
        hypothesis_store=store, transcript=transcript, node_calls=node_calls,
        concept_graph=concept_graph, learner_overlay=learner_overlay,
        revision_store=revision_store, llm=llm, diagnostics_store=diagnostics_store,
    )
    session_id = await transcript.create_session(learner_id, concept_graph_id)

    await loop.handle_turn(session_id, 0, "Can you show me with sin(x^2)?")

    diag = await diagnostics_store.get_for_turn(session_id, 0)
    assert diag.explicit_request_present is True
    assert diag.explicit_request_unaddressed is False
    assert not any("explicit_request_unaddressed" in w for w in diag.warnings)


@pytest.mark.asyncio(loop_scope="session")
async def test_loop_skips_the_check_entirely_when_no_explicit_request(
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
    assert diag.explicit_request_present is False
    assert diag.explicit_request_what is None
    assert diag.explicit_request_unaddressed is False
