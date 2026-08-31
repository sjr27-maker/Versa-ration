"""The standalone `probe migrate` runner (probe/migrate.py).

Runs against a throwaway database created for this module only — the
shared test schema (conftest's session `pool`) is built by
drop-and-replay and has no ledger, which is exactly the state
`baseline()` exists to adopt.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from probe import migrate
from probe.db import create_pool
from tests.conftest import DATABASE_URL

_TEST_DB = "probe_migrate_pytest"


@pytest_asyncio.fixture(loop_scope="session")
async def fresh_db_url(pool):
    """A brand-new empty database, dropped afterward. Uses the session
    `pool` (against the default DB) only to run CREATE/DROP DATABASE."""
    base = DATABASE_URL.rsplit("/", 1)[0]
    url = f"{base}/{_TEST_DB}"
    async with pool.acquire() as conn:
        await conn.execute(f'DROP DATABASE IF EXISTS "{_TEST_DB}" WITH (FORCE)')
        await conn.execute(f'CREATE DATABASE "{_TEST_DB}"')
    try:
        yield url
    finally:
        async with pool.acquire() as conn:
            await conn.execute(f'DROP DATABASE IF EXISTS "{_TEST_DB}" WITH (FORCE)')


@pytest.mark.asyncio(loop_scope="session")
async def test_apply_all_on_a_fresh_database_then_idempotent(fresh_db_url):
    target = await create_pool(fresh_db_url, min_size=1, max_size=2)
    try:
        async with target.acquire() as conn:
            applied = await migrate.apply_all(conn)
            assert applied == [p.name for p in migrate.migration_files()]
            assert len(applied) >= 32

            # ledger records exactly what ran
            recorded = {
                r["filename"]
                for r in await conn.fetch("SELECT filename FROM schema_migrations")
            }
            assert recorded == set(applied)

            # end state: the retired subsystems are gone, the kept
            # tables are present
            tables = {
                r["tablename"]
                for r in await conn.fetch(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
                )
            }
            assert "hypotheses" not in tables
            assert "concept_graphs" not in tables
            assert "branches" not in tables
            assert {"disambiguation_turns", "learner_facts", "turn_diagnostics"} <= tables

            # second run is a no-op
            assert await migrate.apply_all(conn) == []
    finally:
        await target.close()


@pytest.mark.asyncio(loop_scope="session")
async def test_status_reports_applied_and_pending(fresh_db_url):
    target = await create_pool(fresh_db_url, min_size=1, max_size=2)
    try:
        async with target.acquire() as conn:
            applied, pending = await migrate.status(conn)
            assert applied == []
            assert pending == [p.name for p in migrate.migration_files()]

            await migrate.apply_all(conn)

            applied, pending = await migrate.status(conn)
            assert pending == []
            assert applied == [p.name for p in migrate.migration_files()]
    finally:
        await target.close()


@pytest.mark.asyncio(loop_scope="session")
async def test_baseline_adopts_an_existing_schema_without_running_sql(fresh_db_url):
    target = await create_pool(fresh_db_url, min_size=1, max_size=2)
    try:
        async with target.acquire() as conn:
            # Build the schema the way the test fixture does: raw replay,
            # no ledger.
            for path in migrate.migration_files():
                await conn.execute(path.read_text())

            stamped = await migrate.baseline(conn)
            assert stamped == [p.name for p in migrate.migration_files()]

            # Now apply_all must do nothing (if it tried to re-run
            # 001_initial it would raise "type already exists").
            assert await migrate.apply_all(conn) == []

            # And baseline again is a no-op.
            assert await migrate.baseline(conn) == []
    finally:
        await target.close()
