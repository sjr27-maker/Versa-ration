import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from probe.llm import StubLLMClient
from probe.mismatch import MismatchDetector
from probe.models import (
    ConceptNode,
    Hypothesis,
    Layer,
    OverlayEntry,
    OverlayState,
    SuggestedCause,
    Tier,
)

_GRAPH_ID = uuid4()


def _concept() -> ConceptNode:
    return ConceptNode(
        concept_graph_id=_GRAPH_ID,
        id="closures",
        name="Closures",
        common_misconceptions=[
            "a closure copies the variable's value at definition time"
        ],
    )


def _overlay() -> OverlayEntry:
    return OverlayEntry(
        concept_graph_id=_GRAPH_ID,
        concept_id="closures",
        state=OverlayState.PARTIAL,
        confidence=0.6,
        updated_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )


def _hypothesis() -> Hypothesis:
    return Hypothesis(
        layer=Layer.MENTAL_MODEL,
        statement="student believes closures copy the value, not the variable",
        probability=0.6,
        confidence=0.5,
        tier=Tier.ACTIVE,
    )


@pytest.mark.asyncio
async def test_detects_a_mismatch_from_overlay_and_hypothesis_fixture():
    llm = StubLLMClient(
        canned={
            "MISMATCH:DETECT": json.dumps(
                {
                    "mismatch": True,
                    "learner_claim": "closures copy the value at definition time",
                    "world_claim": "closures capture the variable by reference",
                    "confidence": 0.8,
                    "suggested_cause": "learner_misconception",
                }
            )
        }
    )
    detector = MismatchDetector(llm)

    result = await detector.detect(
        "closures", _concept(), _overlay(), [_hypothesis()]
    )

    assert result is not None
    assert result.concept_id == "closures"
    assert result.suggested_cause is SuggestedCause.LEARNER_MISCONCEPTION
    assert result.confidence == pytest.approx(0.8)
    assert "copy" in result.learner_claim


@pytest.mark.asyncio
async def test_prompt_does_not_hardcode_blame_and_carries_context():
    llm = StubLLMClient()
    detector = MismatchDetector(llm)

    await detector.detect("closures", _concept(), _overlay(), [_hypothesis()])

    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]
    assert "do not default to blaming the learner" in prompt.lower()
    assert "learner_misconception" in prompt
    assert "possible_world_model_error" in prompt
    assert _concept().common_misconceptions[0] in prompt
    assert _hypothesis().statement in prompt


@pytest.mark.asyncio
async def test_returns_none_when_llm_reports_no_mismatch():
    llm = StubLLMClient(canned={"MISMATCH:DETECT": json.dumps({"mismatch": False})})
    detector = MismatchDetector(llm)

    result = await detector.detect(
        "closures", _concept(), _overlay(), [_hypothesis()]
    )
    assert result is None


@pytest.mark.asyncio
async def test_returns_none_with_nothing_to_compare():
    detector = MismatchDetector(StubLLMClient())
    result = await detector.detect("closures", _concept(), None, [])
    assert result is None
    # Early return before ever calling the LLM.
    assert detector.last_call_count == 0


@pytest.mark.asyncio
async def test_last_call_count_is_one_when_the_llm_is_actually_asked():
    detector = MismatchDetector(StubLLMClient())
    await detector.detect("closures", _concept(), _overlay(), [_hypothesis()])
    assert detector.last_call_count == 1


@pytest.mark.asyncio
async def test_returns_none_on_malformed_llm_response():
    llm = StubLLMClient(canned={"MISMATCH:DETECT": "not json"})
    detector = MismatchDetector(llm)
    result = await detector.detect(
        "closures", _concept(), _overlay(), [_hypothesis()]
    )
    assert result is None


@pytest.mark.asyncio
async def test_returns_none_on_invalid_suggested_cause():
    llm = StubLLMClient(
        canned={
            "MISMATCH:DETECT": json.dumps(
                {
                    "mismatch": True,
                    "learner_claim": "x",
                    "world_claim": "y",
                    "confidence": 0.5,
                    "suggested_cause": "not_a_real_enum_value",
                }
            )
        }
    )
    detector = MismatchDetector(llm)
    result = await detector.detect(
        "closures", _concept(), _overlay(), [_hypothesis()]
    )
    assert result is None


@pytest.mark.asyncio
async def test_can_suggest_possible_world_model_error():
    llm = StubLLMClient(
        canned={
            "MISMATCH:DETECT": json.dumps(
                {
                    "mismatch": True,
                    "learner_claim": "closures capture by reference always",
                    "world_claim": (
                        "concept definition says closures copy by value "
                        "(possibly stale/incorrect)"
                    ),
                    "confidence": 0.7,
                    "suggested_cause": "possible_world_model_error",
                }
            )
        }
    )
    detector = MismatchDetector(llm)

    result = await detector.detect(
        "closures", _concept(), _overlay(), [_hypothesis()]
    )
    assert result is not None
    assert result.suggested_cause is SuggestedCause.POSSIBLE_WORLD_MODEL_ERROR
