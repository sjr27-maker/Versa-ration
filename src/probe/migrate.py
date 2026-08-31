"""Standalone migration runner — the deploy-time path (Cloud SQL etc.).

Applies `src/probe/migrations/*.sql` to a database in filename order,
exactly once each, tracked in a `schema_migrations` ledger table.
Idempotent: re-running applies nothing new. Each migration and its
ledger row commit together in one transaction, so a mid-run failure
leaves a consistent applied prefix — fix the cause and re-run to
continue.

The pytest suite does NOT use this: `tests/conftest.py` drops and
replays the whole schema every session and needs no ledger. This
module is only for real, persistent databases.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import asyncpg

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

_LEDGER_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename    TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def migration_files() -> list[Path]:
    """Every migration file, in the order they must be applied
    (lexicographic — the `NNN_` prefix makes that correct)."""
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


async def _ensure_ledger(conn: asyncpg.Connection) -> set[str]:
    await conn.execute(_LEDGER_DDL)
    rows = await conn.fetch("SELECT filename FROM schema_migrations")
    return {r["filename"] for r in rows}


async def status(conn: asyncpg.Connection) -> tuple[list[str], list[str]]:
    """`(applied, pending)` filenames, each in apply order. Creates the
    ledger table if absent (harmless — an empty ledger just means every
    migration is pending)."""
    applied = await _ensure_ledger(conn)
    names = [p.name for p in migration_files()]
    return (
        [n for n in names if n in applied],
        [n for n in names if n not in applied],
    )


async def apply_all(
    conn: asyncpg.Connection,
    *,
    on_apply: Callable[[str], None] | None = None,
) -> list[str]:
    """Apply every pending migration, in order. Returns the filenames
    applied by this call (empty if the database was already up to
    date)."""
    applied = await _ensure_ledger(conn)
    done: list[str] = []
    for path in migration_files():
        if path.name in applied:
            continue
        sql = path.read_text()
        async with conn.transaction():
            await conn.execute(sql)
            await conn.execute(
                "INSERT INTO schema_migrations (filename) VALUES ($1)", path.name
            )
        done.append(path.name)
        if on_apply is not None:
            on_apply(path.name)
    return done


async def baseline(conn: asyncpg.Connection) -> list[str]:
    """Record every not-yet-recorded migration as applied WITHOUT
    running its SQL — for adopting a database that already has the full
    schema but no `schema_migrations` ledger (a hand-migrated database,
    or one built by the test fixture's drop-and-replay). Returns the
    filenames stamped."""
    applied = await _ensure_ledger(conn)
    stamped: list[str] = []
    async with conn.transaction():
        for path in migration_files():
            if path.name in applied:
                continue
            await conn.execute(
                "INSERT INTO schema_migrations (filename) VALUES ($1)", path.name
            )
            stamped.append(path.name)
    return stamped
