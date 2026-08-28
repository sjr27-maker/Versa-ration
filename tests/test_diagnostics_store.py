import pytest

from probe.diagnostics import TurnDiagnosticsStore
from probe.models import TurnDiagnostics


@pytest.mark.asyncio(loop_scope="session")
async def test_record_and_get_for_turn_roundtrips(
    transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    store = TurnDiagnosticsStore(clean_pool)

    diagnostics = TurnDiagnostics(
        session_id=session_id,
        turn_index=0,
        node_call_counts={"Diagnose": 1, "Infer": 1, "Plan": 7, "Teach": 1},
        total_call_count=10,
        guardrail_fired=False,
        entropy_bits=1.5,
        duration_ms=123.4,
        warnings=["off_graph_drift: 3 consecutive turns"],
        teach_failed=False,
        inferred_topic="derivatives",
        topic_seeded_new=True,
        retry_count=2,
    )
    await store.record(diagnostics)

    fetched = await store.get_for_turn(session_id, 0)

    assert fetched is not None
    assert fetched.node_call_counts == {"Diagnose": 1, "Infer": 1, "Plan": 7, "Teach": 1}
    assert fetched.total_call_count == 10
    assert fetched.guardrail_fired is False
    assert fetched.entropy_bits == pytest.approx(1.5)
    assert fetched.duration_ms == pytest.approx(123.4)
    assert fetched.warnings == ["off_graph_drift: 3 consecutive turns"]
    assert fetched.teach_failed is False
    assert fetched.inferred_topic == "derivatives"
    assert fetched.topic_seeded_new is True
    assert fetched.retry_count == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_retry_count_defaults_to_zero_when_not_specified(
    transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    store = TurnDiagnosticsStore(clean_pool)

    await store.record(
        TurnDiagnostics(
            session_id=session_id,
            turn_index=0,
            node_call_counts={},
            total_call_count=0,
            guardrail_fired=False,
            entropy_bits=None,
            duration_ms=1.0,
            warnings=[],
            teach_failed=False,
        )
    )

    fetched = await store.get_for_turn(session_id, 0)
    assert fetched.retry_count == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_inferred_topic_defaults_to_none_for_turns_past_the_first(
    transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    store = TurnDiagnosticsStore(clean_pool)

    await store.record(
        TurnDiagnostics(
            session_id=session_id,
            turn_index=1,
            node_call_counts={},
            total_call_count=0,
            guardrail_fired=False,
            entropy_bits=None,
            duration_ms=1.0,
            warnings=[],
            teach_failed=False,
        )
    )

    fetched = await store.get_for_turn(session_id, 1)

    assert fetched.inferred_topic is None
    assert fetched.topic_seeded_new is None


@pytest.mark.asyncio(loop_scope="session")
async def test_get_for_turn_returns_none_when_absent(
    transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    store = TurnDiagnosticsStore(clean_pool)

    assert await store.get_for_turn(session_id, 0) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_list_for_session_is_chronological(
    transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    store = TurnDiagnosticsStore(clean_pool)

    for i in range(3):
        await store.record(
            TurnDiagnostics(
                session_id=session_id,
                turn_index=i,
                node_call_counts={},
                total_call_count=0,
                guardrail_fired=False,
                entropy_bits=None,
                duration_ms=1.0,
                warnings=[],
                teach_failed=False,
            )
        )

    rows = await store.list_for_session(session_id)

    assert [r.turn_index for r in rows] == [0, 1, 2]


@pytest.mark.asyncio(loop_scope="session")
async def test_teach_failed_flag_and_empty_warnings_roundtrip(
    transcript, clean_pool, learner_id, concept_graph_id
):
    session_id = await transcript.create_session(learner_id, concept_graph_id)
    store = TurnDiagnosticsStore(clean_pool)

    await store.record(
        TurnDiagnostics(
            session_id=session_id,
            turn_index=0,
            node_call_counts={},
            total_call_count=0,
            guardrail_fired=False,
            entropy_bits=None,
            duration_ms=5.0,
            warnings=[],
            teach_failed=True,
        )
    )

    fetched = await store.get_for_turn(session_id, 0)

    assert fetched.teach_failed is True
    assert fetched.warnings == []
    assert fetched.entropy_bits is None
