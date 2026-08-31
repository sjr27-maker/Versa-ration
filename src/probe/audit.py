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

from probe.ablation import AblationConfig
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
        session_id: UUID | None = None,
        ablation_config: AblationConfig | None = None,
    ) -> UUID:
        # ablation_config is fixed for the session's lifetime (see
        # set_ablation_config's raise-if-turns-exist guard below) — None
        # here is stored as SQL NULL rather than a serialized
        # AblationConfig() so get_ablation_config's default
        # interpretation stays the single source of truth for what NULL
        # means, not duplicated into every INSERT.
        session_id = session_id or uuid4()
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO sessions (id, learner_id, ablation_config) "
                "VALUES ($1, $2, $3)",
                session_id,
                learner_id,
                ablation_config.model_dump(mode="json") if ablation_config else None,
            )
        return session_id

    async def get_ablation_config(self, session_id: UUID) -> AblationConfig:
        """The AblationConfig a session was created with — NULL (never
        set, or created before migration 025) is interpreted as
        `AblationConfig()`'s default, which is now `SessionMode.
        MINIMAL_BRANCH`."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT ablation_config FROM sessions WHERE id = $1", session_id
            )
        if row is None:
            raise KeyError(f"session {session_id} not found")
        stored = row["ablation_config"]
        return AblationConfig() if stored is None else AblationConfig(**stored)

    async def set_ablation_config(
        self, session_id: UUID, ablation_config: AblationConfig
    ) -> None:
        """Config is fixed at session creation — this exists only to
        raise clearly if something attempts to change it after turns
        have happened, not as a normal write path. Switching config
        mid-session would make every turn after the switch
        uninterpretable (which layer produced that turn's behavior?)
        and let hypotheses accumulated under one config silently carry
        into another."""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT id FROM sessions WHERE id = $1 FOR UPDATE", session_id
                )
                if row is None:
                    raise KeyError(f"session {session_id} not found")
                turn_count = await conn.fetchval(
                    "SELECT count(*) FROM turns WHERE session_id = $1", session_id
                )
                if turn_count > 0:
                    raise ValueError(
                        f"session {session_id} already has {turn_count} turn(s) "
                        "-- ablation_config cannot change mid-session"
                    )
                await conn.execute(
                    "UPDATE sessions SET ablation_config = $2 WHERE id = $1",
                    session_id,
                    ablation_config.model_dump(mode="json"),
                )

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
        """A learner's sessions, most recent first, each with its turn
        count — for the Setup page's read-only resume view.
        `SessionSummary.topic` is always None now that sessions have no
        concept graph attached."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    s.id AS session_id,
                    s.created_at,
                    count(t.id) AS turn_count
                FROM sessions s
                LEFT JOIN turns t ON t.session_id = s.id
                WHERE s.learner_id = $1
                GROUP BY s.id, s.created_at
                ORDER BY s.created_at DESC
                """,
                learner_id,
            )
        return [
            SessionSummary(
                session_id=row["session_id"],
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
                ORDER BY seq DESC
                LIMIT 1
                """,
                session_id,
                turn_index,
                node_name,
            )
        return None if row is None else NodeCall(**dict(row))

    async def get_recent_calls(
        self, session_id: UUID, node_name: str, before_turn_index: int, limit: int
    ) -> list[NodeCall]:
        """The last `limit` calls to `node_name` strictly before
        `before_turn_index`, chronological (oldest first) — e.g.
        Teach's own compact recent-history input (loop.py's
        `_build_teach_history`) and the examples/analogies list built
        from ExtractTeachingArtifact's calls (`_build_examples_used`).
        Reads a node's own past output straight back out of the audit
        trail invariant 2 already requires, rather than a second
        persistence path for the same text."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM node_calls
                WHERE session_id = $1 AND node_name = $2 AND turn_index < $3
                ORDER BY turn_index DESC, seq DESC
                LIMIT $4
                """,
                session_id,
                node_name,
                before_turn_index,
                limit,
            )
        return list(reversed([NodeCall(**dict(row)) for row in rows]))

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
                ORDER BY turn_index DESC, seq DESC
                LIMIT 1
                """,
                session_id,
                node_name,
            )
        return None if row is None else NodeCall(**dict(row))
