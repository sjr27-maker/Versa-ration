import os
from pathlib import Path

import pytest
import pytest_asyncio

from probe.audit import NodeCallStore, TranscriptStore
from probe.db import create_pool
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
        await conn.execute("DROP TABLE IF EXISTS node_calls")
        await conn.execute("DROP TABLE IF EXISTS turns")
        await conn.execute("DROP TABLE IF EXISTS sessions")
        await conn.execute("DROP TABLE IF EXISTS evidence_refs")
        await conn.execute("DROP TABLE IF EXISTS hypotheses")
        await conn.execute("DROP TYPE IF EXISTS evidence_polarity")
        await conn.execute("DROP TYPE IF EXISTS hypothesis_tier")
        await conn.execute("DROP TYPE IF EXISTS hypothesis_layer")
        for migration in MIGRATIONS:
            await conn.execute(migration.read_text())
    try:
        yield pool
    finally:
        await pool.close()


@pytest_asyncio.fixture(loop_scope="session")
async def clean_pool(pool):
    async with pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE node_calls, turns, sessions, evidence_refs, hypotheses "
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
