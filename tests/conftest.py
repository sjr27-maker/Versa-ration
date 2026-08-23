import os
from pathlib import Path

import asyncpg
import pytest
import pytest_asyncio

from probe.store import HypothesisStore

DATABASE_URL = os.getenv(
    "PROBE_TEST_DATABASE_URL",
    "postgresql://probe:probe@localhost:5434/probe",
)

MIGRATION = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "probe"
    / "migrations"
    / "001_initial.sql"
)


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def pool():
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=4)
    async with pool.acquire() as conn:
        # Fresh schema for the test session. DROP is DDL cleanup here in
        # tests only — the store itself must never remove hypotheses (see
        # CLAUDE.md).
        await conn.execute("DROP TABLE IF EXISTS evidence_refs")
        await conn.execute("DROP TABLE IF EXISTS hypotheses")
        await conn.execute("DROP TYPE IF EXISTS evidence_polarity")
        await conn.execute("DROP TYPE IF EXISTS hypothesis_tier")
        await conn.execute("DROP TYPE IF EXISTS hypothesis_layer")
        await conn.execute(MIGRATION.read_text())
    try:
        yield pool
    finally:
        await pool.close()


@pytest_asyncio.fixture(loop_scope="session")
async def store(pool):
    async with pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE evidence_refs, hypotheses RESTART IDENTITY CASCADE"
        )
    return HypothesisStore(pool)
