from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv

from probe.audit import NodeCallStore, TranscriptStore
from probe.db import create_pool
from probe.diagnostics import TurnDiagnosticsStore
from probe.disambiguate import DisambiguationStore
from probe.embeddings import (
    EmbeddingClient,
    StubEmbeddingClient,
    build_embedding_client,
)
from probe.learner import LearnerStore
from probe.llm import ModelTierClients, StubLLMClient, build_tier_clients
from probe.loop import SessionLoop
from probe.memory import LearnerFactStore, ThinkingStyleStore
from probe.models import Learner
from probe import migrate as _migrate


def _database_url() -> str:
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        print("error: DATABASE_URL not set (check .env)", file=sys.stderr)
        sys.exit(2)
    return url


def _require_gemini_api_key() -> str:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print(
            "error: GEMINI_API_KEY not set (check .env) — pass --stub to "
            "run against StubLLMClient instead of the real Gemini API",
            file=sys.stderr,
        )
        sys.exit(2)
    return key


def _build_tier_clients(use_stub: bool) -> ModelTierClients:
    if use_stub:
        stub = StubLLMClient()
        return ModelTierClients(fast=stub, capable=stub, best=stub)
    return build_tier_clients(_require_gemini_api_key())


def _build_embedding_client(use_stub: bool) -> EmbeddingClient:
    if use_stub:
        return StubEmbeddingClient()
    return build_embedding_client(_require_gemini_api_key())


async def _resolve_learner(store: LearnerStore, spec: str) -> Learner:
    """--learner accepts either an existing learner's UUID or a label.

    A UUID must already exist (there's no "create by guessing an id").
    A label resumes the matching learner if one exists, else creates a
    new one — this is the session's identity, resolved once here, not
    per-turn state.
    """
    try:
        learner_id = UUID(spec)
    except ValueError:
        learner_id = None
    if learner_id is not None:
        learner = await store.get(learner_id)
        if learner is None:
            print(f"error: no learner with id {spec}", file=sys.stderr)
            sys.exit(2)
        return learner

    learner = await store.get_by_label(spec)
    if learner is not None:
        return learner
    return await store.create(label=spec)


def _build_loop(pool, tiers: ModelTierClients, embedding_client: EmbeddingClient) -> SessionLoop:
    return SessionLoop(
        transcript=TranscriptStore(pool),
        node_calls=NodeCallStore(pool),
        llm=tiers.fast,
        model_tier_clients=tiers,
        diagnostics_store=TurnDiagnosticsStore(pool),
        disambiguation_store=DisambiguationStore(pool),
        learner_fact_store=LearnerFactStore(pool),
        thinking_style_store=ThinkingStyleStore(pool),
        embedding_client=embedding_client,
    )


async def _chat(learner_spec: str, use_stub: bool) -> None:
    tiers = _build_tier_clients(use_stub)
    embedding_client = _build_embedding_client(use_stub)
    pool = await create_pool(_database_url(), min_size=1, max_size=4)
    try:
        learner = await _resolve_learner(LearnerStore(pool), learner_spec)
        label_suffix = f" (label={learner.label!r})" if learner.label else ""
        print(f"probe: learner {learner.id}{label_suffix}")
        print("probe: minimal_branch mode — no concept graph")
        loop = _build_loop(pool, tiers, embedding_client)
        await loop.run_interactive(learner.id)
    finally:
        await pool.close()


async def _consolidate_session(session_id_str: str, use_stub: bool) -> None:
    try:
        session_id = UUID(session_id_str)
    except ValueError:
        print(f"error: {session_id_str!r} is not a valid session id", file=sys.stderr)
        sys.exit(2)

    tiers = _build_tier_clients(use_stub)
    embedding_client = _build_embedding_client(use_stub)
    pool = await create_pool(_database_url(), min_size=1, max_size=4)
    try:
        loop = _build_loop(pool, tiers, embedding_client)
        # Deliberate, unambiguous trigger — no turn-count gate (unlike
        # run_interactive's own auto-consolidate on exit): this command
        # exists specifically to consolidate a session on demand,
        # regardless of how many turns it has.
        result = await loop.consolidate_session(session_id)
        if result is None:
            print(
                "probe: nothing to consolidate — this session wrote no "
                "learner_facts (a BASELINE session, or it never resolved "
                "anything)"
            )
            return
        print(
            f"probe: thinking-style candidate {result.id}\n"
            f"  path_summary: {result.path_summary}\n"
            f"  confirmation_count={result.confirmation_count} "
            f"status={result.status.value}\n"
            f"  session_ids: {[str(s) for s in result.session_ids]}"
        )
    finally:
        await pool.close()


async def _run_migrations(status_only: bool, do_baseline: bool) -> None:
    pool = await create_pool(_database_url(), min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            if status_only:
                applied, pending = await _migrate.status(conn)
                print(f"migrations: {len(applied)} applied, {len(pending)} pending")
                for name in applied:
                    print(f"  [x] {name}")
                for name in pending:
                    print(f"  [ ] {name}")
                return
            if do_baseline:
                stamped = await _migrate.baseline(conn)
                if stamped:
                    print(
                        f"probe migrate: recorded {len(stamped)} migration(s) as "
                        f"already-applied without running them "
                        f"({stamped[0]} .. {stamped[-1]})"
                    )
                else:
                    print("probe migrate: nothing to baseline - ledger already complete")
                return
            applied = await _migrate.apply_all(
                conn, on_apply=lambda name: print(f"  applied {name}")
            )
            if applied:
                print(f"probe migrate: applied {len(applied)} migration(s)")
            else:
                print("probe migrate: database already up to date")
    finally:
        await pool.close()


def _web() -> None:
    """`probe web` — one command to launch the Streamlit UI. Shells out
    to `streamlit run` rather than importing streamlit here, so this
    module (and every other CLI command) has no dependency on the web
    UI even existing; `probe web` is the only path that touches it.

    Binds to the port Cloud Run (or any PaaS) injects via `PORT`, on
    0.0.0.0 so the container's published port is reachable. Falls back
    to Streamlit's own default (8501) locally, where PORT is unset, so
    `probe web` on a workstation is unchanged."""
    import subprocess

    app_path = Path(__file__).resolve().parent / "webui" / "app.py"
    port = os.environ.get("PORT", "8501")
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", port,
            "--server.address", "0.0.0.0",
        ],
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="probe")
    subparsers = parser.add_subparsers(dest="command")
    chat_parser = subparsers.add_parser(
        "chat", help="start an interactive minimal_branch session loop"
    )
    chat_parser.add_argument(
        "--learner",
        required=True,
        help="learner label (resumes if it exists, creates if not) "
        "or an existing learner's UUID",
    )
    chat_parser.add_argument(
        "--stub",
        action="store_true",
        help="use StubLLMClient/StubEmbeddingClient instead of the "
        "real Gemini API (no GEMINI_API_KEY needed, no cost)",
    )
    consolidate_parser = subparsers.add_parser(
        "consolidate-session",
        help="background step 6-8 of the memory layer (memory.py) for "
        "one session on demand: label its facts' order-structure and "
        "compare against this learner's thinking_style_candidates",
    )
    consolidate_parser.add_argument("session_id", help="session id (UUID) to consolidate")
    consolidate_parser.add_argument(
        "--stub",
        action="store_true",
        help="use StubLLMClient/StubEmbeddingClient instead of the "
        "real Gemini API (no GEMINI_API_KEY needed, no cost)",
    )
    migrate_parser = subparsers.add_parser(
        "migrate",
        help="apply pending SQL migrations to DATABASE_URL, in order, "
        "once each (idempotent; tracked in a schema_migrations table)",
    )
    migrate_parser.add_argument(
        "--status",
        action="store_true",
        help="show applied/pending migrations and exit without changing anything",
    )
    migrate_parser.add_argument(
        "--baseline",
        action="store_true",
        help="record every migration as already-applied WITHOUT running "
        "it - for a database that already has the full schema but no "
        "schema_migrations ledger (e.g. a hand-migrated dev DB)",
    )
    subparsers.add_parser(
        "web",
        help="launch the local Streamlit web UI",
    )
    args = parser.parse_args()

    if args.command == "chat":
        asyncio.run(_chat(args.learner, args.stub))
    elif args.command == "consolidate-session":
        asyncio.run(_consolidate_session(args.session_id, args.stub))
    elif args.command == "migrate":
        asyncio.run(_run_migrations(args.status, args.baseline))
    elif args.command == "web":
        _web()
    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
