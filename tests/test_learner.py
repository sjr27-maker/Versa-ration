from uuid import uuid4

import asyncpg
import pytest

from probe.cli import _resolve_learner


@pytest.mark.asyncio(loop_scope="session")
async def test_resolve_by_new_label_creates_a_learner(learner_store):
    learner = await _resolve_learner(learner_store, "ada")

    assert learner.label == "ada"
    fetched = await learner_store.get_by_label("ada")
    assert fetched is not None
    assert fetched.id == learner.id


@pytest.mark.asyncio(loop_scope="session")
async def test_resolve_by_label_round_trips_to_the_same_learner(learner_store):
    first = await _resolve_learner(learner_store, "grace")
    second = await _resolve_learner(learner_store, "grace")

    assert first.id == second.id


@pytest.mark.asyncio(loop_scope="session")
async def test_resolve_by_existing_uuid_returns_that_learner(learner_store):
    created = await learner_store.create(label="alan")

    resolved = await _resolve_learner(learner_store, str(created.id))

    assert resolved.id == created.id
    assert resolved.label == "alan"


@pytest.mark.asyncio(loop_scope="session")
async def test_resolve_by_unknown_uuid_exits_rather_than_guessing(learner_store):
    with pytest.raises(SystemExit):
        await _resolve_learner(learner_store, str(uuid4()))


@pytest.mark.asyncio(loop_scope="session")
async def test_session_rejected_without_a_valid_learner_id(transcript, concept_graph_id):
    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await transcript.create_session(uuid4(), concept_graph_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_session_rejected_without_a_valid_concept_graph_id(transcript, learner_id):
    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await transcript.create_session(learner_id, uuid4())
