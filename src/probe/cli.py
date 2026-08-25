from __future__ import annotations

import argparse
import asyncio
import os
import sys

import asyncpg
from dotenv import load_dotenv

from probe.audit import NodeCallStore, TranscriptStore
from probe.concept_graph import ConceptGraph, ConceptValidationError
from probe.db import create_pool
from probe.llm import StubLLMClient
from probe.loop import SessionLoop
from probe.seed import SeedGraphError, seed_graph
from probe.store import HypothesisStore


def _database_url() -> str:
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        print("error: DATABASE_URL not set (check .env)", file=sys.stderr)
        sys.exit(2)
    return url


async def _chat() -> None:
    pool = await create_pool(_database_url(), min_size=1, max_size=4)
    try:
        loop = SessionLoop(
            hypothesis_store=HypothesisStore(pool),
            transcript=TranscriptStore(pool),
            node_calls=NodeCallStore(pool),
            llm=StubLLMClient(),
        )
        await loop.run_interactive()
    finally:
        await pool.close()


async def _seed_graph(topic: str) -> None:
    pool = await create_pool(_database_url(), min_size=1, max_size=4)
    try:
        graph = ConceptGraph(pool)
        try:
            concepts = await seed_graph(StubLLMClient(), graph, topic)
        except (SeedGraphError, ConceptValidationError) as exc:
            print(f"error: seed-graph rejected the proposed batch: {exc}", file=sys.stderr)
            sys.exit(1)
        except asyncpg.UniqueViolationError as exc:
            print(
                "error: one or more proposed concept ids already exist in "
                f"the graph ({exc}). seed-graph is a one-time, frozen "
                "operation — it does not update an existing graph.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"probe: seeded {len(concepts)} concepts for topic {topic!r}")
        for concept in concepts:
            print(f"  - {concept.id}: {concept.name}"
                  f"  prerequisites={concept.prerequisites}")
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="probe")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("chat", help="start an interactive session loop")
    seed_parser = subparsers.add_parser(
        "seed-graph",
        help="one-time LLM seed of a concept graph for a topic (frozen after creation)",
    )
    seed_parser.add_argument(
        "topic", nargs="+", help="topic to seed, e.g. python closures"
    )
    args = parser.parse_args()

    if args.command == "chat":
        asyncio.run(_chat())
    elif args.command == "seed-graph":
        asyncio.run(_seed_graph(" ".join(args.topic)))
    else:
        parser.print_help()
        sys.exit(1)
