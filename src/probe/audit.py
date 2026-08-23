"""Persistence for sessions, turns, and node_calls.

`TranscriptStore` owns sessions + raw student turns. `NodeCallStore`
owns the audit trail required by CLAUDE.md invariant 2 — every node
call gets a row.

Neither store deletes. Sessions and turns are append-only just like
hypotheses; the audit trail is by definition history and must not be
edited.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import asyncpg
from pydantic import BaseModel


def to_jsonable(value: Any) -> Any:
    """Convert a value into a form json.dumps() can handle.

    Pydantic models → `.model_dump(mode="json")`. UUIDs and datetimes
    become strings. Lists/dicts recurse. Objects that don't fit are
    represented by their class name so we still record *that* they were
    passed, without pretending we captured their state (e.g., a
    `HypothesisStore` dependency handed to `Update.run`).
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    return {"__unserializable__": type(value).__name__}


class TranscriptStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create_session(self, session_id: UUID | None = None) -> UUID:
        session_id = session_id or uuid4()
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO sessions (id) VALUES ($1)", session_id
            )
        return session_id

    async def record_turn(
        self, session_id: UUID, turn_index: int, text: str
    ) -> UUID:
        turn_id = uuid4()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO turns (id, session_id, turn_index, text)
                VALUES ($1, $2, $3, $4)
                """,
                turn_id,
                session_id,
                turn_index,
                text,
            )
        return turn_id


class NodeCallStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def record(
        self,
        node_name: str,
        session_id: UUID,
        turn_index: int,
        input_json: dict,
        output_json: Any,
    ) -> UUID:
        call_id = uuid4()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO node_calls (
                    id, node_name, session_id, turn_index,
                    input_json, output_json, timestamp
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                call_id,
                node_name,
                session_id,
                turn_index,
                to_jsonable(input_json),
                to_jsonable(output_json),
                datetime.now(timezone.utc),
            )
        return call_id
