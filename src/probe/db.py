"""Pool + connection helpers.

The JSONB codec here ensures every consumer of `node_calls.input_json` /
`output_json` receives Python values (dicts, lists, ints, strings)
rather than raw JSON text — otherwise every reader has to remember to
`json.loads(...)` and it's easy to forget.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg


async def _init_connection(conn: asyncpg.Connection) -> None:
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )


async def create_pool(dsn: str, **kwargs: Any) -> asyncpg.Pool:
    return await asyncpg.create_pool(dsn, init=_init_connection, **kwargs)
