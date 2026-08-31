"""TranscriptStore's ablation_config persistence (migration 025):
round-tripping on the session row, NULL's default interpretation
(now SessionMode.MINIMAL_BRANCH), and the set-once enforcement (config
is fixed at session creation and must not change once turns exist)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from probe.ablation import AblationConfig, SessionMode


@pytest.mark.asyncio(loop_scope="session")
async def test_ablation_config_round_trips_on_create_session(
    transcript, clean_pool, learner_id
):
    config = AblationConfig(mode=SessionMode.BASELINE)
    session_id = await transcript.create_session(learner_id, ablation_config=config)

    fetched = await transcript.get_ablation_config(session_id)

    assert fetched == config


@pytest.mark.asyncio(loop_scope="session")
async def test_no_ablation_config_at_creation_defaults_to_minimal_branch(
    transcript, clean_pool, learner_id
):
    session_id = await transcript.create_session(learner_id)

    fetched = await transcript.get_ablation_config(session_id)

    assert fetched == AblationConfig()
    assert fetched.mode is SessionMode.MINIMAL_BRANCH


@pytest.mark.asyncio(loop_scope="session")
async def test_get_ablation_config_raises_for_unknown_session(clean_pool, transcript):
    with pytest.raises(KeyError):
        await transcript.get_ablation_config(uuid4())


@pytest.mark.asyncio(loop_scope="session")
async def test_set_ablation_config_on_a_fresh_session_succeeds(
    transcript, clean_pool, learner_id
):
    session_id = await transcript.create_session(learner_id)
    new_config = AblationConfig(mode=SessionMode.BASELINE)

    await transcript.set_ablation_config(session_id, new_config)

    assert await transcript.get_ablation_config(session_id) == new_config


@pytest.mark.asyncio(loop_scope="session")
async def test_set_ablation_config_raises_once_a_turn_exists(
    transcript, clean_pool, learner_id
):
    session_id = await transcript.create_session(learner_id)
    await transcript.record_turn(session_id, 0, "first turn")

    with pytest.raises(ValueError, match="cannot change mid-session"):
        await transcript.set_ablation_config(
            session_id, AblationConfig(mode=SessionMode.BASELINE)
        )

    # Unchanged -- the attempted (and rejected) write left no trace.
    assert await transcript.get_ablation_config(session_id) == AblationConfig()


@pytest.mark.asyncio(loop_scope="session")
async def test_set_ablation_config_raises_for_unknown_session(clean_pool, transcript):
    with pytest.raises(KeyError):
        await transcript.set_ablation_config(uuid4(), AblationConfig())
