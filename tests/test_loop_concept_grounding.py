"""End-to-end: a session with a real seeded Calculus graph produces a
Plan whose target_concept is a real node from that graph, and a Teach
output that is actually about that concept — asserted against the
graph's own stored concept name, not just "the turn didn't crash".
"""

import json

import pytest

from probe.llm import StubLLMClient
from probe.loop import SessionLoop
from probe.models import ConceptNode

_PROPOSE_DERIVATIVES = json.dumps(
    [
        {
            "action": "explain",
            "target_concept": "derivatives",
            "rationale": "the student asked about it",
        }
    ]
)


def _teach_echoes_target_concept(prompt: str) -> str:
    """Stands in for a real model that actually stays within what it
    was given — reads target_concept back out of its own prompt so the
    test can assert the OUTPUT is about the right concept, not just
    that the right concept was passed in."""
    payload = json.loads(prompt[len("TEACH: "):].split("\n", 1)[0])
    if payload.get("target_concept") == "derivatives":
        return "Let's talk about Derivatives, the rate of change of a function."
    return "[stub teach: no target_concept match]"


async def _seed_calculus_graph(concept_graph, graph_id):
    await concept_graph.add_batch(
        graph_id,
        "Calculus",
        [
            ConceptNode(concept_graph_id=graph_id, id="limits", name="Limits"),
            ConceptNode(concept_graph_id=graph_id, id="derivatives", name="Derivatives"),
        ],
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_plan_and_teach_ground_in_the_seeded_graphs_real_concepts(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph,
    learner_overlay, revision_store,
):
    from uuid import uuid4

    graph_id = uuid4()
    await _seed_calculus_graph(concept_graph, graph_id)
    session_id = await transcript.create_session(learner_id, graph_id)

    llm = StubLLMClient(
        canned={
            "PROPOSE:ACTIONS": _PROPOSE_DERIVATIVES,
            "TEACH:": _teach_echoes_target_concept,
        }
    )
    loop = SessionLoop(
        hypothesis_store=store,
        transcript=transcript,
        node_calls=node_calls,
        concept_graph=concept_graph,
        learner_overlay=learner_overlay,
        revision_store=revision_store,
        llm=llm,
    )

    message = await loop.handle_turn(session_id, 0, "I think a derivative is a slope")

    # Plan's winner names a real concept id from the actual seeded graph.
    async with clean_pool.acquire() as conn:
        plan_row = await conn.fetchrow(
            "SELECT output_json, input_json FROM node_calls "
            "WHERE session_id=$1 AND node_name='Plan'",
            session_id,
        )
    real_concept_ids = {c.id for c in await concept_graph.list_concepts(graph_id)}
    winner_target = plan_row["output_json"]["winner"]["target_concept"]
    assert winner_target in real_concept_ids
    assert winner_target == "derivatives"

    # concept_state actually carried the real graph's topic/concepts.
    assert plan_row["input_json"]["concept_state"]["topic"] == "Calculus"
    assert {c["id"] for c in plan_row["input_json"]["concept_state"]["concepts"]} == (
        real_concept_ids
    )

    # Teach's output is actually about that concept — checked against
    # the graph's own stored name, not just the raw id string.
    concept = await concept_graph.get_concept(graph_id, "derivatives")
    assert concept.name in message


@pytest.mark.asyncio(loop_scope="session")
async def test_teach_never_leaks_target_concept_is_null_language(
    store, transcript, node_calls, clean_pool, learner_id, concept_graph,
    learner_overlay, revision_store,
):
    """Regression for the exact leak found in the live smoke test:
    Teach's prompt now explicitly forbids narrating its own missing
    fields. This doesn't prove a real model will comply, but proves
    the instruction is actually present in what gets sent."""
    from uuid import uuid4

    graph_id = uuid4()
    await _seed_calculus_graph(concept_graph, graph_id)
    session_id = await transcript.create_session(learner_id, graph_id)

    llm = StubLLMClient()  # default PROPOSE:ACTIONS -> target_concept: None
    loop = SessionLoop(
        hypothesis_store=store,
        transcript=transcript,
        node_calls=node_calls,
        concept_graph=concept_graph,
        learner_overlay=learner_overlay,
        revision_store=revision_store,
        llm=llm,
    )

    await loop.handle_turn(session_id, 0, "hello")

    teach_prompt = next(p for p in llm.prompts if p.startswith("TEACH:"))
    assert "never mention or describe your own fields" in teach_prompt.lower()
    # Teach no longer receives topic (dropped in favor of PathRequirement
    # scoping) — what it does always receive is the student's own
    # message, which must actually be passed through.
    assert "hello" in teach_prompt
