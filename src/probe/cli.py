from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

from probe.audit import NodeCallStore, TranscriptStore
from probe.db import create_pool
from probe.llm import StubLLMClient
from probe.loop import SessionLoop
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


def main() -> None:
    parser = argparse.ArgumentParser(prog="probe")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("chat", help="start an interactive session loop")
    args = parser.parse_args()

    if args.command == "chat":
        asyncio.run(_chat())
    else:
        parser.print_help()
        sys.exit(1)
