"""Pool + connection helpers.

The JSONB codec here ensures every consumer of `node_calls.input_json` /
`output_json` receives Python values (dicts, lists, ints, strings)
rather than raw JSON text — otherwise every reader has to remember to
`json.loads(...)` and it's easy to forget.

The `vector` codec (pgvector's own `register_vector`) does the same
for `learner_facts.embedding`/`thinking_style_candidates.
path_summary_embedding` — a plain Python `list[float]` in and out,
never a hand-parsed `'[0.1,0.2,...]'` string. Registration is best-
effort: on a genuinely fresh database, the very first connection this
pool ever creates can happen before migration 030's own
`CREATE EXTENSION IF NOT EXISTS vector` has run (pool creation itself
acquires a connection to run migrations through), so the `vector` type
doesn't exist in `pg_type` yet and `register_vector` raises
`ValueError` — skipped here, not fatal, since nothing will actually
read/write a vector column until after that migration has applied.
Every connection created after that point (including ones already
open in the pool, since `init` fires per-connection, not once) does
register successfully.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg
from pgvector.asyncpg import register_vector


async def _init_connection(conn: asyncpg.Connection) -> None:
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )
    try:
        await register_vector(conn)
    except ValueError:
        pass


async def create_pool(dsn: str, **kwargs: Any) -> asyncpg.Pool:
    return await asyncpg.create_pool(dsn, init=_init_connection, **kwargs)
