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

from probe.models import TurnRecord


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

    async def create_session(
        self,
        learner_id: UUID,
        concept_graph_id: UUID,
        session_id: UUID | None = None,
    ) -> UUID:
        session_id = session_id or uuid4()
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO sessions (id, learner_id, concept_graph_id) "
                "VALUES ($1, $2, $3)",
                session_id,
                learner_id,
                concept_graph_id,
            )
        return session_id

    async def get_learner_id(self, session_id: UUID) -> UUID:
        """The learner a session belongs to — set once at creation, not
        per-turn state. This is how a node (e.g. Diagnose) gets
        learner_id: looked up from the session row it already has
        session_id for, not threaded through as separate call state.
        """
        async with self._pool.acquire() as conn:
            learner_id = await conn.fetchval(
                "SELECT learner_id FROM sessions WHERE id = $1", session_id
            )
        if learner_id is None:
            raise KeyError(f"session {session_id} not found")
        return learner_id

    async def get_concept_graph_id(self, session_id: UUID) -> UUID:
        """The concept graph this session is teaching — set once at
        creation, same pattern as `get_learner_id`. This is how
        Diagnose/GroundConcept get "this session's linked graph"
        without it being threaded as separate call state.
        """
        async with self._pool.acquire() as conn:
            graph_id = await conn.fetchval(
                "SELECT concept_graph_id FROM sessions WHERE id = $1", session_id
            )
        if graph_id is None:
            raise KeyError(f"session {session_id} not found")
        return graph_id

    async def count_sessions_for_learner(self, learner_id: UUID) -> int:
        async with self._pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT count(*) FROM sessions WHERE learner_id = $1", learner_id
            )
        return count

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

    async def list_turns(self, session_id: UUID) -> list[TurnRecord]:
        """All of a session's student turns, chronological.

        Used by HypothesisGenerator's transcript_context — full session
        history, not just the most recent message.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, session_id, turn_index, text, created_at
                FROM turns
                WHERE session_id = $1
                ORDER BY turn_index
                """,
                session_id,
            )
        return [TurnRecord(**dict(row)) for row in rows]


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
