import json
from uuid import uuid4

import pytest

from probe.grounding import GroundConcept
from probe.llm import StubLLMClient
from probe.models import ConceptNode

_GRAPH_ID = uuid4()


def _candidates() -> list[ConceptNode]:
    return [
        ConceptNode(concept_graph_id=_GRAPH_ID, id="closures", name="Closures"),
        ConceptNode(concept_graph_id=_GRAPH_ID, id="loops", name="Loops"),
    ]


@pytest.mark.asyncio
async def test_identifies_a_concept_the_response_clearly_discusses():
    llm = StubLLMClient(
        canned={
            "GROUND:CONCEPT": json.dumps(
                {"concept_id": "closures", "confidence": 0.9}
            )
        }
    )
    grounder = GroundConcept(llm)

    result = await grounder.detect(
        "I think closures capture the variable by reference", _candidates()
    )

    assert result.concept_id == "closures"
    assert result.confidence == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_returns_none_for_off_topic_or_affective_response():
    llm = StubLLMClient(
        canned={"GROUND:CONCEPT": json.dumps({"concept_id": None, "confidence": 0.1})}
    )
    grounder = GroundConcept(llm)

    result = await grounder.detect("I don't get this", _candidates())

    assert result.concept_id is None
    assert result.confidence == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_rejects_hallucinated_concept_id_not_in_graph():
    # Same validation discipline as seed_graph's prerequisite check and
    # Plan's candidate proposer: an id the LLM invents that isn't in the
    # candidate list is rejected, not passed through.
    llm = StubLLMClient(
        canned={
            "GROUND:CONCEPT": json.dumps(
                {"concept_id": "quantum_entanglement", "confidence": 0.85}
            )
        }
    )
    grounder = GroundConcept(llm)

    result = await grounder.detect("some response", _candidates())

    assert result.concept_id is None
    # Confidence stays visible even though the id was rejected — useful
    # for reviewing what the model tried to hallucinate and how sure it
    # claimed to be.
    assert result.confidence == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_returns_none_confidence_zero_on_malformed_response():
    llm = StubLLMClient(canned={"GROUND:CONCEPT": "not json"})
    grounder = GroundConcept(llm)

    result = await grounder.detect("anything", _candidates())

    assert result.concept_id is None
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_returns_none_when_response_is_not_a_json_object():
    llm = StubLLMClient(canned={"GROUND:CONCEPT": json.dumps(["not", "an", "object"])})
    grounder = GroundConcept(llm)

    result = await grounder.detect("anything", _candidates())

    assert result.concept_id is None
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_default_stub_response_grounds_to_nothing():
    grounder = GroundConcept(StubLLMClient())
    result = await grounder.detect("anything", _candidates())
    assert result.concept_id is None
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_last_call_count_is_one_every_time_detect_runs():
    grounder = GroundConcept(StubLLMClient())
    assert grounder.last_call_count == 0
    await grounder.detect("anything", _candidates())
    assert grounder.last_call_count == 1


@pytest.mark.asyncio
async def test_prompt_lists_every_candidates_id_and_name():
    llm = StubLLMClient()
    grounder = GroundConcept(llm)

    await grounder.detect("what about closures?", _candidates())

    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]
    assert "closures" in prompt and "Closures" in prompt
    assert "loops" in prompt and "Loops" in prompt


@pytest.mark.asyncio
async def test_confidence_is_clamped_into_zero_one():
    llm = StubLLMClient(
        canned={
            "GROUND:CONCEPT": json.dumps({"concept_id": "closures", "confidence": 5.0})
        }
    )
    grounder = GroundConcept(llm)

    result = await grounder.detect("closures", _candidates())

    assert result.confidence == 1.0
