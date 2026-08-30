"""TurnDiagnosticsStore.mean_cost_by_config -- the number that answers
"what does this config cost," grouped across sessions by identical
AblationConfig, with a NULL (never-set) session config bucketed
together with an explicit full-system one rather than split out."""

from __future__ import annotations

import pytest

from probe.ablation import AblationConfig, AblationPreset, build_preset
from probe.models import TurnDiagnostics


async def _record(diagnostics_store, session_id, turn_index, duration_ms, calls, retries=0):
    await diagnostics_store.record(
        TurnDiagnostics(
            session_id=session_id,
            turn_index=turn_index,
            node_call_counts={},
            total_call_count=calls,
            guardrail_fired=False,
            entropy_bits=None,
            duration_ms=duration_ms,
            warnings=[],
            teach_failed=False,
            retry_count=retries,
        )
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_groups_by_config_and_averages_correctly(
    transcript, diagnostics_store, clean_pool, learner_id, concept_graph_id
):
    baseline_config = build_preset(AblationPreset.BASELINE)
    full_config = AblationConfig()

    baseline_session = await transcript.create_session(
        learner_id, concept_graph_id, ablation_config=baseline_config
    )
    full_session = await transcript.create_session(
        learner_id, concept_graph_id, ablation_config=full_config
    )

    await _record(diagnostics_store, baseline_session, 0, duration_ms=1000.0, calls=1)
    await _record(diagnostics_store, baseline_session, 1, duration_ms=2000.0, calls=1)
    await _record(diagnostics_store, full_session, 0, duration_ms=30000.0, calls=10)

    summaries = await diagnostics_store.mean_cost_by_config()
    by_config = {s.ablation_config.model_dump_json(): s for s in summaries}

    baseline_summary = by_config[baseline_config.model_dump_json()]
    assert baseline_summary.turn_count == 2
    assert baseline_summary.mean_duration_ms == pytest.approx(1500.0)
    assert baseline_summary.mean_call_count == pytest.approx(1.0)

    full_summary = by_config[full_config.model_dump_json()]
    assert full_summary.turn_count == 1
    assert full_summary.mean_duration_ms == pytest.approx(30000.0)
    assert full_summary.mean_call_count == pytest.approx(10.0)


@pytest.mark.asyncio(loop_scope="session")
async def test_null_session_config_buckets_with_explicit_full_config(
    transcript, diagnostics_store, clean_pool, learner_id, concept_graph_id
):
    null_config_session = await transcript.create_session(learner_id, concept_graph_id)
    explicit_full_session = await transcript.create_session(
        learner_id, concept_graph_id, ablation_config=AblationConfig()
    )

    await _record(diagnostics_store, null_config_session, 0, duration_ms=100.0, calls=5)
    await _record(diagnostics_store, explicit_full_session, 0, duration_ms=300.0, calls=15)

    summaries = await diagnostics_store.mean_cost_by_config()

    matching = [s for s in summaries if s.ablation_config == AblationConfig()]
    assert len(matching) == 1
    assert matching[0].turn_count == 2
    assert matching[0].mean_duration_ms == pytest.approx(200.0)
    assert matching[0].mean_call_count == pytest.approx(10.0)


@pytest.mark.asyncio(loop_scope="session")
async def test_empty_when_nothing_recorded(diagnostics_store, clean_pool):
    assert await diagnostics_store.mean_cost_by_config() == []
