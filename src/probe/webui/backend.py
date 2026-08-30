"""Shared async/DB bridge for the Streamlit UI. Every page imports
from here — no page constructs its own pool or event loop, and no
page contains business logic: this module only wires existing stores
together, it doesn't compute anything they don't already return.

Streamlit reruns the whole script on every interaction; a fresh
asyncpg pool per rerun would be slow, and asyncpg pools are bound to
the event loop that created them. Instead: one background thread runs
one persistent asyncio event loop for the life of the Streamlit
process, with one pool bound to it, cached via `st.cache_resource` so
every rerun reuses the same one. `run_async()` is how a page's sync
script code calls into it.

Progress display (`run_turn_with_progress`) can't have `SessionLoop`'s
`on_node_start` callback touch Streamlit elements directly — that
callback fires on the background loop's thread, not Streamlit's script
thread, and Streamlit's APIs aren't safe to call off that thread. It
instead writes to a plain dict (safe from any thread — ordinary
attribute/item assignment is GIL-atomic); the *main* thread polls that
dict on a short timeout loop and is the only thing that ever calls
`st.*`.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
from collections.abc import Callable, Coroutine
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from probe.audit import NodeCallStore, TranscriptStore
from probe.branches import BranchStore
from probe.concept_graph import ConceptGraph
from probe.db import create_pool
from probe.diagnostics import TurnDiagnosticsStore
from probe.disambiguate import DisambiguationStore
from probe.learner import LearnerStore
from probe.llm import ModelTierClients, StubLLMClient, build_tier_clients
from probe.options import OptionStore
from probe.overlay import LearnerOverlay
from probe.revision import WorldModelRevisionStore
from probe.store import HypothesisStore


class _LoopThread:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()


@st.cache_resource
def _loop_thread() -> _LoopThread:
    return _LoopThread()


def run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    future = asyncio.run_coroutine_threadsafe(coro, _loop_thread().loop)
    return future.result()


def run_turn_with_progress[T](
    coro: Coroutine[Any, Any, T], progress: dict, placeholder: Any
) -> T:
    """Same as run_async, but polls `progress["node"]` (written by
    SessionLoop's on_node_start callback) every 200ms and renders it
    into `placeholder` — an in-progress indicator that names which
    node is currently running, not just a static spinner."""
    future = asyncio.run_coroutine_threadsafe(coro, _loop_thread().loop)
    while True:
        try:
            return future.result(timeout=0.2)
        except concurrent.futures.TimeoutError:
            node = progress.get("node")
            if node:
                placeholder.markdown(f"⏳ running: **{node}**")


def _database_url() -> str:
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set (check .env)")
    return url


@st.cache_resource
def get_pool():
    return run_async(create_pool(_database_url(), min_size=1, max_size=8))


class Stores:
    """One instance per Streamlit script run — cheap, since each store
    is just a thin wrapper around the one shared pool."""

    def __init__(self) -> None:
        pool = get_pool()
        self.hypotheses = HypothesisStore(pool)
        self.transcript = TranscriptStore(pool)
        self.node_calls = NodeCallStore(pool)
        self.concept_graph = ConceptGraph(pool)
        self.learner_overlay = LearnerOverlay(pool)
        self.revisions = WorldModelRevisionStore(pool)
        self.branches = BranchStore(pool)
        self.options = OptionStore(pool)
        self.diagnostics = TurnDiagnosticsStore(pool)
        self.learners = LearnerStore(pool)
        self.disambiguation = DisambiguationStore(pool)


def get_stores() -> Stores:
    return Stores()


def get_tier_clients(use_stub: bool) -> ModelTierClients:
    if use_stub:
        stub = StubLLMClient()
        return ModelTierClients(fast=stub, capable=stub, best=stub)
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set (check .env) — enable 'use stub' in "
            "Setup to run without a real key"
        )
    return build_tier_clients(api_key)


def make_progress_tracker() -> tuple[dict, Callable[[str], None]]:
    """A plain dict + a callback that writes to it — the callback is
    what SessionLoop's on_node_start gets, and it's safe to call from
    any thread since it's just a dict item assignment."""
    progress: dict = {"node": None}

    def on_node_start(name: str) -> None:
        progress["node"] = name

    return progress, on_node_start
