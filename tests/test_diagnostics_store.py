import pytest

from probe.diagnostics import TurnDiagnosticsStore
from probe.models import TurnDiagnostics


@pytest.mark.asyncio(loop_scope="session")
async def test_record_and_get_for_turn_roundtrips(transcript, clean_pool, learner_id):
    session_id = await transcript.create_session(learner_id)
    store = TurnDiagnosticsStore(clean_pool)

    diagnostics = TurnDiagnostics(
        session_id=session_id,
        turn_index=0,
        node_call_counts={"AssessAndBranch": 1, "FinalAnswer": 1},
        total_call_count=2,
        guardrail_fired=False,
        entropy_bits=None,
        duration_ms=123.4,
        warnings=["disambiguation_typed_past: ..."],
        teach_failed=False,
        retry_count=2,
    )
    await store.record(diagnostics)

    fetched = await store.get_for_turn(session_id, 0)

    assert fetched is not None
    assert fetched.node_call_counts == {"AssessAndBranch": 1, "FinalAnswer": 1}
    assert fetched.total_call_count == 2
    assert fetched.guardrail_fired is False
    assert fetched.duration_ms == pytest.approx(123.4)
    assert fetched.warnings == ["disambiguation_typed_past: ..."]
    assert fetched.teach_failed is False
    assert fetched.retry_count == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_every_field_roundtrips_write_to_read(transcript, clean_pool, learner_id):
    """Generic, future-proofing regression test: every field is set to a
    value that differs from its own default, then the fetched object's
    full model_dump() is compared against the original's. If a future
    migration adds a column without updating both the TurnDiagnostics
    model AND TurnDiagnosticsStore's INSERT/_row_to_diagnostics, this
    test fails instead of the gap surfacing later as a webui
    AttributeError.
    """
    session_id = await transcript.create_session(learner_id)
    store = TurnDiagnosticsStore(clean_pool)

    diagnostics = TurnDiagnostics(
        session_id=session_id,
        turn_index=0,
        node_call_counts={"AssessAndBranch": 1, "FinalAnswer": 2},
        total_call_count=17,
        guardrail_fired=True,
        entropy_bits=2.75,
        duration_ms=456.7,
        warnings=["warning one", "warning two"],
        teach_failed=True,
        retry_count=3,
        memory_match_found=True,
        memory_match_confirmed_resolution=True,
        branching_skipped_by_memory=True,
        # matched_fact_id left None: it FKs to learner_facts, and this
        # store-level test writes no fact. Its own round-trip is covered
        # by test_memory_loop_wiring's end-to-end skip test.
    )
    await store.record(diagnostics)

    fetched = await store.get_for_turn(session_id, 0)

    assert fetched is not None
    assert fetched.model_dump() == diagnostics.model_dump()


@pytest.mark.asyncio(loop_scope="session")
async def test_retry_count_defaults_to_zero_when_not_specified(
    transcript, clean_pool, learner_id
):
    session_id = await transcript.create_session(learner_id)
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
async def test_memory_fields_default_to_false_none(transcript, clean_pool, learner_id):
    session_id = await transcript.create_session(learner_id)
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

    assert fetched.memory_match_found is False
    assert fetched.memory_match_confirmed_resolution is False
    assert fetched.branching_skipped_by_memory is False
    assert fetched.matched_fact_id is None


@pytest.mark.asyncio(loop_scope="session")
async def test_get_for_turn_returns_none_when_absent(transcript, clean_pool, learner_id):
    session_id = await transcript.create_session(learner_id)
    store = TurnDiagnosticsStore(clean_pool)

    assert await store.get_for_turn(session_id, 0) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_list_for_session_is_chronological(transcript, clean_pool, learner_id):
    session_id = await transcript.create_session(learner_id)
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
    transcript, clean_pool, learner_id
):
    session_id = await transcript.create_session(learner_id)
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
