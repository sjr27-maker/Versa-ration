import os
from pathlib import Path

import pytest
import pytest_asyncio

from probe.audit import NodeCallStore, TranscriptStore
from probe.branches import BranchStore
from probe.concept_graph import ConceptGraph
from probe.db import create_pool
from probe.diagnostics import TurnDiagnosticsStore
from probe.disambiguate import DisambiguationStore
from probe.embeddings import StubEmbeddingClient
from probe.learner import LearnerStore
from probe.memory import LearnerFactStore, ThinkingStyleStore
from probe.options import OptionStore
from probe.overlay import LearnerOverlay
from probe.revision import WorldModelRevisionStore
from probe.store import HypothesisStore

DATABASE_URL = os.getenv(
    "PROBE_TEST_DATABASE_URL",
    "postgresql://probe:probe@localhost:5434/probe",
)

MIGRATIONS_DIR = (
    Path(__file__).resolve().parent.parent / "src" / "probe" / "migrations"
)
MIGRATIONS = sorted(MIGRATIONS_DIR.glob("*.sql"))


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def pool():
    pool = await create_pool(DATABASE_URL, min_size=1, max_size=4)
    async with pool.acquire() as conn:
        # Fresh schema for the test session. DROP is DDL cleanup here in
        # tests only — the stores themselves must never remove rows
        # (CLAUDE.md invariants 1 and 2).
        await conn.execute("DROP TABLE IF EXISTS node_calls CASCADE")
        await conn.execute("DROP TABLE IF EXISTS turn_diagnostics CASCADE")
        await conn.execute("DROP TABLE IF EXISTS hypothesis_tier_changes CASCADE")
        await conn.execute("DROP TABLE IF EXISTS learner_facts CASCADE")
        await conn.execute("DROP TABLE IF EXISTS thinking_style_candidates CASCADE")
        await conn.execute("DROP TABLE IF EXISTS disambiguation_options CASCADE")
        await conn.execute("DROP TABLE IF EXISTS disambiguation_branches CASCADE")
        await conn.execute("DROP TABLE IF EXISTS disambiguation_turns CASCADE")
        await conn.execute("DROP TABLE IF EXISTS options CASCADE")
        await conn.execute("DROP TABLE IF EXISTS branches CASCADE")
        await conn.execute("DROP TABLE IF EXISTS branch_generations CASCADE")
        await conn.execute("DROP TABLE IF EXISTS world_model_revision_evidence CASCADE")
        await conn.execute("DROP TABLE IF EXISTS world_model_revisions CASCADE")
        await conn.execute("DROP TABLE IF EXISTS hypothesis_concepts CASCADE")
        await conn.execute("DROP TABLE IF EXISTS evidence_refs CASCADE")
        await conn.execute("DROP TABLE IF EXISTS turns CASCADE")
        await conn.execute("DROP TABLE IF EXISTS sessions CASCADE")
        await conn.execute("DROP TABLE IF EXISTS learners CASCADE")
        await conn.execute("DROP TABLE IF EXISTS hypotheses CASCADE")
        await conn.execute("DROP TABLE IF EXISTS learner_overlay CASCADE")
        await conn.execute("DROP TABLE IF EXISTS concept_prerequisites CASCADE")
        await conn.execute("DROP TABLE IF EXISTS concept_nodes CASCADE")
        await conn.execute("DROP TABLE IF EXISTS concept_graphs CASCADE")
        await conn.execute("DROP TYPE IF EXISTS learner_fact_type")
        await conn.execute("DROP TYPE IF EXISTS thinking_style_status")
        await conn.execute("DROP TYPE IF EXISTS option_status")
        await conn.execute("DROP TYPE IF EXISTS branch_status")
        await conn.execute("DROP TYPE IF EXISTS revision_status")
        await conn.execute("DROP TYPE IF EXISTS overlay_state")
        await conn.execute("DROP TYPE IF EXISTS evidence_polarity")
        await conn.execute("DROP TYPE IF EXISTS hypothesis_tier")
        await conn.execute("DROP TYPE IF EXISTS hypothesis_layer")
        for migration in MIGRATIONS:
            await conn.execute(migration.read_text())
        # On a genuinely first-ever bootstrap (extension didn't exist
        # yet when this exact connection was created — see db.py's
        # _init_connection), the vector codec silently failed to
        # register at connection-init time and this connection would
        # otherwise sit in the pool "poisoned" (returning raw
        # '[0.1,...]' strings instead of Python lists) for its whole
        # lifetime. The migration just above this line is guaranteed
        # to have created the extension, so it's always safe to
        # register it now, before this connection goes back to the pool.
        from pgvector.asyncpg import register_vector

        await register_vector(conn)
    try:
        yield pool
    finally:
        await pool.close()


@pytest_asyncio.fixture(loop_scope="session")
async def clean_pool(pool):
    async with pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE node_calls, turn_diagnostics, hypothesis_tier_changes, "
            "turns, sessions, learners, evidence_refs, "
            "hypotheses, learner_overlay, concept_prerequisites, concept_nodes, "
            "concept_graphs, hypothesis_concepts, world_model_revisions, "
            "world_model_revision_evidence, branches, branch_generations, "
            "options, disambiguation_options, disambiguation_branches, "
            "disambiguation_turns, learner_facts, thinking_style_candidates "
            "RESTART IDENTITY CASCADE"
        )
    return pool


@pytest_asyncio.fixture(loop_scope="session")
async def store(clean_pool):
    return HypothesisStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def transcript(clean_pool):
    return TranscriptStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def node_calls(clean_pool):
    return NodeCallStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def concept_graph(clean_pool):
    return ConceptGraph(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def learner_overlay(clean_pool):
    return LearnerOverlay(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def revision_store(clean_pool):
    return WorldModelRevisionStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def branch_store(clean_pool):
    return BranchStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def option_store(clean_pool):
    return OptionStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def diagnostics_store(clean_pool):
    return TurnDiagnosticsStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def disambiguation_store(clean_pool):
    return DisambiguationStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def learner_fact_store(clean_pool):
    return LearnerFactStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def thinking_style_store(clean_pool):
    return ThinkingStyleStore(clean_pool)


@pytest.fixture
def embedding_client():
    """A fresh StubEmbeddingClient per test — unlike the DB-backed
    fixtures above, this holds no shared state worth reusing across
    tests (just a `canned` dict and a `texts` log)."""
    return StubEmbeddingClient()


@pytest_asyncio.fixture(loop_scope="session")
async def learner_store(clean_pool):
    return LearnerStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def learner_id(learner_store):
    """A fresh learner per test, for tests that just need *a* valid
    learner_id to satisfy sessions.learner_id's FK and don't care about
    learner identity itself."""
    learner = await learner_store.create()
    return learner.id


@pytest_asyncio.fixture(loop_scope="session")
async def concept_graph_id(concept_graph):
    """A fresh empty concept graph per test, for tests that just need
    *a* valid concept_graph_id to satisfy sessions.concept_graph_id's
    FK (or to add concepts into) and don't care about the topic."""
    meta = await concept_graph.create_graph(topic="test-topic")
    return meta.id
