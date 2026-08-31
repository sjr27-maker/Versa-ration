"""Starlette app + a small JSON/SSE API over ``SessionLoop`` — the calm
single-page UI in ``probe/static/`` is the only client. Launched by
``probe serve`` (see ``cli.py``).

No business logic here (this and the static client are the whole web UI). Every
route wires an existing store or ``SessionLoop`` method onto an HTTP
shape and computes nothing a store doesn't already return. The single
piece of genuinely new glue is per-turn node-progress streaming — and
even that is just forwarding ``SessionLoop``'s existing
``on_node_start`` callback fires to the browser as Server-Sent Events
so the generation overlay can name the node that is currently running.

Unlike the Streamlit bridge this module does not need a background
thread: Starlette is natively async, so one asyncpg pool lives on the
app's own event loop and every handler is a coroutine.

State: one ``SessionLoop`` per active session, held in-process in
``_SESSIONS`` (the same lifetime model as the Streamlit page's
``st.session_state["loop"]``). A server restart drops those in-memory
loops; ``GET /api/session/{id}`` rebuilds one from the durable record
(transcript + disambiguation + diagnostics), so a browser reload — or a
restart — never loses the conversation, only the choice of real-vs-stub
client, which a cold rebuild defaults back to stub.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from probe.ablation import AblationConfig
from probe.audit import NodeCallStore, TranscriptStore
from probe.baseline import MAX_CALLS_PER_TURN as _MAX_CALLS_PER_TURN
from probe.db import create_pool
from probe.diagnostics import TurnDiagnosticsStore
from probe.disambiguate import DisambiguationStore
from probe.evidence import EvidenceStore
from probe.embeddings import (
    EmbeddingClient,
    StubEmbeddingClient,
    build_embedding_client,
)
from probe.learner import LearnerStore
from probe.llm import ModelTierClients, StubLLMClient, build_tier_clients
from probe.loop import SessionLoop
from probe.memory import LearnerFactStore, ThinkingStyleStore
from probe.models import Learner, OptionStatus

_STATIC_DIR = Path(__file__).resolve().parent / "static"


# ─────────────────────────── client wiring ───────────────────────────


def _database_url() -> str:
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set (check .env)")
    return url


def _tier_clients(use_stub: bool) -> ModelTierClients:
    if use_stub:
        stub = StubLLMClient()
        return ModelTierClients(fast=stub, capable=stub, best=stub)
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set (check .env) — start the session with "
            "'stub LLM' on to run without a real key"
        )
    return build_tier_clients(api_key)


def _embedding_client(use_stub: bool) -> EmbeddingClient:
    if use_stub:
        return StubEmbeddingClient()
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set (check .env)")
    return build_embedding_client(api_key)


# ─────────────────────────── session state ───────────────────────────


@dataclass
class _Session:
    """One live session's loop + the bookkeeping the Streamlit page
    keeps in ``st.session_state``. ``queue`` is set for the duration of
    one streaming turn: ``on_node`` forwards ``SessionLoop``'s
    ``on_node_start`` fires into it, and the SSE generator drains it."""

    loop: SessionLoop
    session_id: UUID
    learner: Learner
    use_stub: bool
    turn_index: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    queue: asyncio.Queue[str] | None = None

    def on_node(self, name: str) -> None:
        q = self.queue
        if q is not None:
            q.put_nowait(name)


_SESSIONS: dict[str, _Session] = {}


class _AppState:
    pool = None


_state = _AppState()


def _stores() -> dict:
    pool = _state.pool
    return {
        "transcript": TranscriptStore(pool),
        "node_calls": NodeCallStore(pool),
        "diagnostics": TurnDiagnosticsStore(pool),
        "learners": LearnerStore(pool),
        "disambiguation": DisambiguationStore(pool),
        "learner_facts": LearnerFactStore(pool),
        "thinking_styles": ThinkingStyleStore(pool),
        "evidence": EvidenceStore(pool),
    }


def _build_loop(use_stub: bool, on_node) -> SessionLoop:
    pool = _state.pool
    tiers = _tier_clients(use_stub)
    return SessionLoop(
        transcript=TranscriptStore(pool),
        node_calls=NodeCallStore(pool),
        llm=tiers.fast,
        model_tier_clients=tiers,
        diagnostics_store=TurnDiagnosticsStore(pool),
        on_node_start=on_node,
        disambiguation_store=DisambiguationStore(pool),
        learner_fact_store=LearnerFactStore(pool),
        thinking_style_store=ThinkingStyleStore(pool),
        embedding_client=_embedding_client(use_stub),
    )


async def _resolve_learner(store: LearnerStore, spec: str) -> Learner:
    """``learner`` accepts an existing UUID or a label — a UUID must
    already exist, a label resumes the matching learner or creates one.
    Same rule as ``cli._resolve_learner``, minus the ``sys.exit``."""
    spec = spec.strip()
    try:
        learner_id = UUID(spec)
    except ValueError:
        learner_id = None
    if learner_id is not None:
        learner = await store.get(learner_id)
        if learner is None:
            raise LookupError(f"no learner with id {spec}")
        return learner
    existing = await store.get_by_label(spec)
    return existing if existing is not None else await store.create(label=spec)


# ─────────────────────────── read helpers ────────────────────────────


async def _tutor_message_for_turn(
    node_calls: NodeCallStore, session_id: UUID, turn_index: int
) -> str | None:
    """A turn's tutor line comes from that turn's own FinalAnswer /
    BaselineTeach ``node_calls`` row — a 'show options' turn has
    neither, and returns None."""
    for node_name in ("FinalAnswer", "BaselineTeach"):
        call = await node_calls.get_call_for_turn(session_id, turn_index, node_name)
        if call is not None:
            return str(call.output_json)
    return None


async def _rebuild_history(stores: dict, session_id: UUID) -> list[dict]:
    history: list[dict] = []
    for turn in await stores["transcript"].list_turns(session_id):
        history.append({"role": "student", "text": turn.text})
        tutor = await _tutor_message_for_turn(
            stores["node_calls"], session_id, turn.turn_index
        )
        if tutor is not None:
            history.append({"role": "tutor", "text": tutor})
    return history


async def _pending_options(stores: dict, session_id: UUID) -> list[dict]:
    latest = await stores["disambiguation"].get_latest_turn(session_id)
    if latest is None:
        return []
    options = await stores["disambiguation"].list_options_for_turn(latest.id)
    return [
        {"id": str(o.id), "text": o.text}
        for o in options
        if o.status is OptionStatus.OPEN
    ]


async def _inspect_payload(stores: dict, session_id: UUID) -> dict:
    """The repurposed inspector: the live architecture's real per-turn
    record — the latest disambiguation turn's branch statements and the
    options it offered, plus that turn's ``turn_diagnostics`` row."""
    disamb = stores["disambiguation"]
    latest = await disamb.get_latest_turn(session_id)
    branches: list[dict] = []
    options: list[dict] = []
    branch_turn_index: int | None = None
    branched = False
    if latest is not None:
        branch_turn_index = latest.turn_index
        branched = latest.needs_branches
        turn_options = await disamb.list_options_for_turn(latest.id)
        option_by_branch = {o.branch_id: o for o in turn_options}
        for b in await disamb.list_branches_for_turn(latest.id):
            opt = option_by_branch.get(b.id)
            branches.append(
                {
                    "statement": b.statement,
                    "status": b.status.value,
                    "option": opt.text if opt is not None else None,
                }
            )
        for o in turn_options:
            options.append({"text": o.text, "status": o.status.value})

    # TRACE always reflects the most recent turn's diagnostics — a
    # click-resolution turn writes a turn_diagnostics row but no new
    # disambiguation_turn, so keying trace off the latest disambiguation
    # turn's index would show a stale (pre-click) call breakdown.
    all_diag = await stores["diagnostics"].list_for_session(session_id)
    diag_row = all_diag[-1] if all_diag else None

    diagnostics = None
    if diag_row is not None:
        diagnostics = {
            "turn_index": diag_row.turn_index,
            "node_call_counts": diag_row.node_call_counts,
            "total_call_count": diag_row.total_call_count,
            "guardrail_fired": diag_row.guardrail_fired,
            "guardrail_limit": _MAX_CALLS_PER_TURN,
            "duration_ms": round(diag_row.duration_ms, 1),
            "warnings": diag_row.warnings,
            "teach_failed": diag_row.teach_failed,
            "retry_count": diag_row.retry_count,
            "branching_skipped_by_memory": diag_row.branching_skipped_by_memory,
            "memory_match_found": diag_row.memory_match_found,
        }
    return {
        "turn_index": diag_row.turn_index if diag_row is not None else branch_turn_index,
        "branch_turn_index": branch_turn_index,
        "branched": branched,
        "branches": branches,
        "options": options,
        "diagnostics": diagnostics,
    }


# ───────────────────────────── routes ────────────────────────────────


async def _index(_request: Request) -> Response:
    return FileResponse(_STATIC_DIR / "index.html")


async def _create_session(request: Request) -> Response:
    body = await request.json()
    learner_spec = (body.get("learner") or "").strip()
    if not learner_spec:
        return JSONResponse({"error": "learner is required"}, status_code=400)
    use_stub = bool(body.get("stub", True))

    stores = _stores()
    try:
        learner = await _resolve_learner(stores["learners"], learner_spec)
    except LookupError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)

    try:
        session = _Session(
            loop=None,  # type: ignore[arg-type]
            session_id=UUID(int=0),
            learner=learner,
            use_stub=use_stub,
        )
        session.loop = _build_loop(use_stub, session.on_node)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    session.session_id = await stores["transcript"].create_session(
        learner.id, ablation_config=AblationConfig()
    )
    _SESSIONS[str(session.session_id)] = session

    return JSONResponse(
        {
            "session_id": str(session.session_id),
            "learner": {"id": str(learner.id), "label": learner.label},
            "stub": use_stub,
            "turn_index": 0,
            "turns": [],
            "pending_options": [],
        }
    )


async def _get_or_rebuild(session_id: str) -> _Session | None:
    if session_id in _SESSIONS:
        return _SESSIONS[session_id]
    try:
        sid = UUID(session_id)
    except ValueError:
        return None
    stores = _stores()
    try:
        learner_id = await stores["transcript"].get_learner_id(sid)
    except KeyError:
        return None  # no such session row
    learner = await stores["learners"].get(learner_id)
    if learner is None:
        return None
    turns = await stores["transcript"].list_turns(sid)
    # A cold rebuild can't recover the real-vs-stub choice — default to
    # stub so a reload never starts spending against the real API
    # without the user re-picking it.
    session = _Session(
        loop=None,  # type: ignore[arg-type]
        session_id=sid,
        learner=learner,
        use_stub=True,
        turn_index=len(turns),
    )
    session.loop = _build_loop(True, session.on_node)
    # Recover the one piece of loop state a click doesn't reconstruct
    # from the store: if the latest disambiguation turn still has open
    # branches, a *typed* next message should be treated as typing past
    # them (loop.py step 3b). A click resolves fine regardless.
    latest = await stores["disambiguation"].get_latest_turn(sid)
    if latest is not None and latest.needs_branches:
        open_branches = [
            b
            for b in await stores["disambiguation"].list_branches_for_turn(latest.id)
            if b.status.value == "open"
        ]
        if open_branches:
            session.loop._prior_disambiguation_turn_id = latest.id
    _SESSIONS[session_id] = session
    return session


async def _get_session(request: Request) -> Response:
    session_id = request.path_params["session_id"]
    session = await _get_or_rebuild(session_id)
    if session is None:
        return JSONResponse({"error": "no such session"}, status_code=404)
    stores = _stores()
    return JSONResponse(
        {
            "session_id": session_id,
            "learner": {
                "id": str(session.learner.id),
                "label": session.learner.label,
            },
            "stub": session.use_stub,
            "turn_index": session.turn_index,
            "turns": await _rebuild_history(stores, session.session_id),
            "pending_options": await _pending_options(stores, session.session_id),
        }
    )


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


async def _run_turn(request: Request) -> Response:
    session_id = request.path_params["session_id"]
    session = await _get_or_rebuild(session_id)
    if session is None:
        return JSONResponse({"error": "no such session"}, status_code=404)

    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)
    option_id_raw = body.get("option_id")
    try:
        option_id = UUID(option_id_raw) if option_id_raw else None
    except ValueError:
        return JSONResponse({"error": "bad option_id"}, status_code=400)

    stores = _stores()

    async def stream():
        # One turn at a time per session — the client disables input
        # while streaming, but a stray second POST must still not
        # interleave two handle_turn calls on the same loop.
        async with session.lock:
            turn_index = session.turn_index
            queue: asyncio.Queue[str] = asyncio.Queue()
            session.queue = queue
            yield _sse({"phase": "start", "turn_index": turn_index, "query": text})
            # handle_turn is deliberately NOT cancelled if the client
            # disconnects mid-stream: a turn is atomic (it persists
            # branches / options / the answer / diagnostics), same as the
            # CLI and Streamlit treat it. Only the SSE forwarding stops.
            task = asyncio.create_task(
                session.loop.handle_turn(
                    session.session_id, turn_index, text, option_id
                )
            )
            drain: asyncio.Task | None = None
            try:
                while True:
                    drain = asyncio.create_task(queue.get())
                    done, _pending = await asyncio.wait(
                        {task, drain},
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=15,
                    )
                    if drain in done:
                        yield _sse({"phase": "node", "node": drain.result()})
                    else:
                        drain.cancel()
                    if task in done:
                        break
                while not queue.empty():
                    yield _sse({"phase": "node", "node": queue.get_nowait()})
                message = task.result()
            except Exception as exc:  # noqa: BLE001 - surfaced to the client
                yield _sse(
                    {"phase": "error", "error": f"{type(exc).__name__}: {exc}"}
                )
                return
            finally:
                if drain is not None and not drain.done():
                    drain.cancel()
                session.queue = None

            session.turn_index = turn_index + 1
            pending = await _pending_options(stores, session.session_id)
            inspect = await _inspect_payload(stores, session.session_id)
            # A turn "branched" exactly when it showed options instead of
            # an answer — i.e. there are now open options awaiting a click.
            yield _sse(
                {
                    "phase": "done",
                    "turn_index": turn_index,
                    "next_turn_index": session.turn_index,
                    "message": message,
                    "branched": bool(pending),
                    "pending_options": pending,
                    "diagnostics": inspect["diagnostics"],
                    "inspect": inspect,
                }
            )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _inspect(request: Request) -> Response:
    session_id = request.path_params["session_id"]
    session = await _get_or_rebuild(session_id)
    if session is None:
        return JSONResponse({"error": "no such session"}, status_code=404)
    return JSONResponse(await _inspect_payload(_stores(), session.session_id))


async def _learner_sessions(request: Request) -> Response:
    """This learner's prior sessions, newest first, each with its turn
    count — the calm UI's 'Resume a prior session' list. Straight
    passthrough of TranscriptStore.list_sessions_for_learner."""
    try:
        learner_id = UUID(request.path_params["learner_id"])
    except ValueError:
        return JSONResponse({"error": "bad learner id"}, status_code=400)
    rows = await _stores()["transcript"].list_sessions_for_learner(learner_id)
    return JSONResponse(
        {
            "sessions": [
                {
                    "session_id": str(s.session_id),
                    "turn_count": s.turn_count,
                    "created_at": s.created_at.isoformat(),
                }
                for s in rows
            ]
        }
    )


async def _learners(request: Request) -> Response:
    if request.method == "POST":
        return await _learner_create(request)
    return await _learners_list(request)


async def _learners_list(_request: Request) -> Response:
    """Every learner with their session count and most recent session —
    the Learners panel's picker. LearnerStore.list_all_with_session_counts
    passthrough."""
    rows = await _stores()["learners"].list_all_with_session_counts()
    return JSONResponse(
        {
            "learners": [
                {
                    "id": str(s.learner.id),
                    "label": s.learner.label,
                    "display_name": s.learner.display_name,
                    "session_count": s.session_count,
                    "last_session_at": (
                        s.last_session_at.isoformat()
                        if s.last_session_at is not None
                        else None
                    ),
                }
                for s in rows
            ]
        }
    )


async def _learner_create(request: Request) -> Response:
    """Create a learner from the Learners panel's 'new' form —
    LearnerStore.create passthrough (label / display_name only, the
    model is deliberately thin)."""
    body = await request.json()
    label = (body.get("label") or "").strip() or None
    display_name = (body.get("display_name") or "").strip() or None
    if label is None and display_name is None:
        return JSONResponse(
            {"error": "give the learner a label or a display name"},
            status_code=400,
        )
    learner = await _stores()["learners"].create(
        label=label, display_name=display_name
    )
    return JSONResponse(
        {
            "id": str(learner.id),
            "label": learner.label,
            "display_name": learner.display_name,
        }
    )


async def _learner_facts(request: Request) -> Response:
    """The Story view: every learner_facts row for this learner, in
    order, across every session (memory.py step 9).
    LearnerFactStore.list_by_learner passthrough — no computation, the
    plain-language framing is the client's."""
    try:
        learner_id = UUID(request.path_params["learner_id"])
    except ValueError:
        return JSONResponse({"error": "bad learner id"}, status_code=400)
    facts = await _stores()["learner_facts"].list_by_learner(learner_id)
    return JSONResponse(
        {
            "facts": [
                {
                    "session_id": str(f.session_id),
                    "turn_index": f.turn_index,
                    "fact_type": f.fact_type.value,
                    "situation": f.situation,
                    "resolution": f.resolution,
                    "created_at": f.created_at.isoformat(),
                }
                for f in facts
            ]
        }
    )


async def _compare(request: Request) -> Response:
    """Two sessions side by side: their mode, per-turn cost, and
    transcripts aligned turn by turn. Every value
    is a store read; nothing here judges which session did better."""
    stores = _stores()
    try:
        a_id = UUID(request.query_params["a"])
        b_id = UUID(request.query_params["b"])
    except (KeyError, ValueError):
        return JSONResponse(
            {"error": "pass ?a=<session_id>&b=<session_id>"}, status_code=400
        )

    async def _side(sid: UUID) -> dict | None:
        try:
            learner_id = await stores["transcript"].get_learner_id(sid)
        except KeyError:
            return None
        learner = await stores["learners"].get(learner_id)
        config = await stores["transcript"].get_ablation_config(sid)
        diags = await stores["diagnostics"].list_for_session(sid)
        diag_by_turn = {d.turn_index: d for d in diags}
        turns = await stores["transcript"].list_turns(sid)
        rows = []
        for t in turns:
            d = diag_by_turn.get(t.turn_index)
            rows.append(
                {
                    "turn_index": t.turn_index,
                    "student": t.text,
                    "tutor": await _tutor_message_for_turn(
                        stores["node_calls"], sid, t.turn_index
                    ),
                    "duration_ms": round(d.duration_ms) if d else None,
                    "calls": d.total_call_count if d else None,
                    "retries": d.retry_count if d else None,
                }
            )
        n = len(diags)
        return {
            "session_id": str(sid),
            "learner": (learner.label or str(learner.id)) if learner else None,
            "mode": config.mode.value,
            "turns": rows,
            "mean_ms": round(sum(d.duration_ms for d in diags) / n) if n else None,
            "mean_calls": (
                round(sum(d.total_call_count for d in diags) / n, 1) if n else None
            ),
        }

    a = await _side(a_id)
    b = await _side(b_id)
    if a is None or b is None:
        return JSONResponse({"error": "no such session"}, status_code=404)
    return JSONResponse(
        {"a": a, "b": b, "mode_differs": a["mode"] != b["mode"]}
    )


async def _evidence(request: Request) -> Response:
    """Every recorded verification finding, newest first — the Evidence
    panel. `?source_type=staged_verification` (or `organic_session`)
    filters. `EvidenceStore.list_all` passthrough; the source_type
    label is carried verbatim so a staged mechanism test is never shown
    as anything else."""
    from probe.models import EvidenceSourceType

    raw = request.query_params.get("source_type")
    source_type = None
    if raw is not None:
        try:
            source_type = EvidenceSourceType(raw)
        except ValueError:
            return JSONResponse(
                {"error": f"unknown source_type {raw!r}"}, status_code=400
            )
    rows = await _stores()["evidence"].list_all(source_type)
    return JSONResponse(
        {
            "records": [
                {
                    "id": str(r.id),
                    "source_type": r.source_type.value,
                    "part": r.part,
                    "title": r.title,
                    "summary": r.summary,
                    "body": r.body,
                    "learner_id": str(r.learner_id) if r.learner_id else None,
                    "session_id": str(r.session_id) if r.session_id else None,
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ]
        }
    )


async def _consolidate(request: Request) -> Response:
    session_id = request.path_params["session_id"]
    session = await _get_or_rebuild(session_id)
    if session is None:
        return JSONResponse({"error": "no such session"}, status_code=404)
    result = await session.loop.consolidate_session(session.session_id)
    if result is None:
        return JSONResponse({"consolidated": False})
    return JSONResponse(
        {
            "consolidated": True,
            "candidate_id": str(result.id),
            "path_summary": result.path_summary,
            "confirmation_count": result.confirmation_count,
            "status": result.status.value,
        }
    )


# ─────────────────────────── app factory ─────────────────────────────


@contextlib.asynccontextmanager
async def _lifespan(_app: Starlette):
    _state.pool = await create_pool(_database_url(), min_size=1, max_size=8)
    try:
        yield
    finally:
        if _state.pool is not None:
            await _state.pool.close()
            _state.pool = None


def create_app() -> Starlette:
    routes = [
        Route("/", _index),
        Route("/api/session", _create_session, methods=["POST"]),
        Route("/api/session/{session_id}", _get_session, methods=["GET"]),
        Route(
            "/api/session/{session_id}/turn", _run_turn, methods=["POST"]
        ),
        Route("/api/session/{session_id}/inspect", _inspect, methods=["GET"]),
        Route(
            "/api/session/{session_id}/consolidate",
            _consolidate,
            methods=["POST"],
        ),
        Route("/api/learners", _learners, methods=["GET", "POST"]),
        Route(
            "/api/learners/{learner_id}/sessions",
            _learner_sessions,
            methods=["GET"],
        ),
        Route(
            "/api/learners/{learner_id}/facts", _learner_facts, methods=["GET"]
        ),
        Route("/api/compare", _compare, methods=["GET"]),
        Route("/api/evidence", _evidence, methods=["GET"]),
        Mount(
            "/static",
            app=StaticFiles(directory=str(_STATIC_DIR)),
            name="static",
        ),
    ]
    return Starlette(routes=routes, lifespan=_lifespan)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, log_level="info")
