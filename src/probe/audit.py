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

from probe.models import NodeCall, SessionSummary, TurnRecord


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
        concept_graph_id: UUID | None = None,
        session_id: UUID | None = None,
    ) -> UUID:
        # concept_graph_id is nullable as of migration 013: a session
        # created with no topic yet (web UI's "no topic input" setup)
        # gets one attached by AttachTopic on its first turn instead —
        # SessionLoop.handle_turn hard-fails on any turn past the
        # first if it's still null by then.
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

    async def get_concept_graph_id(self, session_id: UUID) -> UUID | None:
        """The concept graph this session is teaching — set once, by
        AttachTopic on the session's first turn, same lookup pattern as
        `get_learner_id`. This is how Diagnose/GroundConcept get "this
        session's linked graph" without it being threaded as separate
        call state.

        Returns None for a session that hasn't had a topic attached
        yet (nullable as of migration 013) — distinct from the session
        not existing at all, which still raises KeyError. Diagnose
        already degrades gracefully against a None graph (empty
        candidate list, same as its existing "no concepts found" path)
        so this doesn't require any change on that side.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT concept_graph_id FROM sessions WHERE id = $1", session_id
            )
        if row is None:
            raise KeyError(f"session {session_id} not found")
        return row["concept_graph_id"]

    async def attach_concept_graph_id(
        self, session_id: UUID, concept_graph_id: UUID
    ) -> None:
        """Set once, by AttachTopic on a session's first turn. Raises
        if the session already has one — a session's graph, once
        attached, is set for its lifetime (same "set once at creation"
        spirit as learner_id, just attached a turn later instead of at
        INSERT time)."""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT concept_graph_id FROM sessions WHERE id = $1 FOR UPDATE",
                    session_id,
                )
                if row is None:
                    raise KeyError(f"session {session_id} not found")
                if row["concept_graph_id"] is not None:
                    raise ValueError(
                        f"session {session_id} already has concept_graph_id "
                        f"{row['concept_graph_id']} attached"
                    )
                await conn.execute(
                    "UPDATE sessions SET concept_graph_id = $2 WHERE id = $1",
                    session_id,
                    concept_graph_id,
                )

    async def get_turn(self, turn_id: UUID) -> TurnRecord | None:
        """One turn by id — for showing an evidence ref's actual source
        text, not just its turn_id."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, session_id, turn_index, text, created_at "
                "FROM turns WHERE id = $1",
                turn_id,
            )
        return None if row is None else TurnRecord(**dict(row))

    async def list_sessions_for_learner(
        self, learner_id: UUID
    ) -> list[SessionSummary]:
        """A learner's sessions, most recent first, each with its
        inferred topic (None if a topic was never attached) and turn
        count — for the Setup page's read-only resume view."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    s.id AS session_id,
                    s.concept_graph_id,
                    cg.topic AS topic,
                    s.created_at,
                    count(t.id) AS turn_count
                FROM sessions s
                LEFT JOIN concept_graphs cg ON cg.id = s.concept_graph_id
                LEFT JOIN turns t ON t.session_id = s.id
                WHERE s.learner_id = $1
                GROUP BY s.id, s.concept_graph_id, cg.topic, s.created_at
                ORDER BY s.created_at DESC
                """,
                learner_id,
            )
        return [
            SessionSummary(
                session_id=row["session_id"],
                concept_graph_id=row["concept_graph_id"],
                topic=row["topic"],
                turn_count=row["turn_count"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

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

    async def get_call_for_turn(
        self, session_id: UUID, turn_index: int, node_name: str
    ) -> NodeCall | None:
        """The read side of invariant 2's audit trail: one node's
        recorded call for one specific turn — e.g. the web UI's
        Decision trace panel reading that turn's Plan row directly,
        rather than re-deriving anything from raw scores itself."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM node_calls
                WHERE session_id = $1 AND turn_index = $2 AND node_name = $3
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                session_id,
                turn_index,
                node_name,
            )
        return None if row is None else NodeCall(**dict(row))

    async def get_latest_call(
        self, session_id: UUID, node_name: str
    ) -> NodeCall | None:
        """The most recent call to `node_name` in this session,
        regardless of turn — e.g. the web UI's branch panel reading
        the previous turn's BranchResolve without needing to know its
        exact turn_index."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM node_calls
                WHERE session_id = $1 AND node_name = $2
                ORDER BY turn_index DESC, timestamp DESC
                LIMIT 1
                """,
                session_id,
                node_name,
            )
        return None if row is None else NodeCall(**dict(row))
