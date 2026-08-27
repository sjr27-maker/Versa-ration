from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from uuid import UUID

from dotenv import load_dotenv

from probe.audit import NodeCallStore, TranscriptStore
from probe.branches import BranchStore
from probe.concept_graph import ConceptGraph, ConceptValidationError
from probe.db import create_pool
from probe.learner import LearnerStore
from probe.llm import LLMClient, ModelTierClients, StubLLMClient, build_tier_clients
from probe.loop import SessionLoop
from probe.models import ConceptGraphMeta, Learner
from probe.overlay import LearnerOverlay
from probe.portrait import LearnerPortrait, build_portrait
from probe.revision import RevisionApplicationError, WorldModelRevisionStore
from probe.seed import SeedGraphError, seed_graph
from probe.store import HypothesisStore


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


async def _do_seed_graph(
    graph: ConceptGraph, topic: str, llm: LLMClient
) -> ConceptGraphMeta:
    try:
        concept_graph_id, concepts = await seed_graph(llm, graph, topic)
    except (SeedGraphError, ConceptValidationError) as exc:
        print(f"error: seed-graph rejected the proposed batch: {exc}", file=sys.stderr)
        sys.exit(1)
    print(
        f"probe: seeded {len(concepts)} concepts for topic {topic!r} "
        f"(graph {concept_graph_id})"
    )
    for concept in concepts:
        print(f"  - {concept.id}: {concept.name}"
              f"  prerequisites={concept.prerequisites}")
    meta = await graph.get_graph(concept_graph_id)
    assert meta is not None  # just inserted, in the same call
    return meta


async def _resolve_graph(
    graph: ConceptGraph, spec: str, llm: LLMClient
) -> ConceptGraphMeta:
    """--topic accepts either an existing graph's UUID or a topic label.

    A UUID must already exist, and must have at least one concept node
    — a graph row with zero concepts only exists if something created
    it outside seed_graph/add_batch, and starting a chat against it
    would silently run GroundConcept against nothing. A label may
    match zero, one, or several existing graphs (topic isn't unique,
    per migration 009) — with any matches, ask which to resume rather
    than silently picking one; with none, there's nothing to resume,
    so seed a fresh graph (which always has at least one concept,
    since seed_graph rejects an empty proposed batch).
    """
    try:
        graph_id = UUID(spec)
    except ValueError:
        graph_id = None
    if graph_id is not None:
        meta = await graph.get_graph(graph_id)
        if meta is None:
            print(f"error: no concept graph with id {spec}", file=sys.stderr)
            sys.exit(2)
        concepts = await graph.list_concepts(graph_id)
        if not concepts:
            # Distinct from "not found": the row exists but has zero
            # concept_nodes, which only happens via a low-level
            # create_graph() call outside seed_graph/add_batch's
            # atomic (row + nodes together) path — every graph the CLI
            # itself creates has at least one concept. Fail closed
            # rather than start a session GroundConcept can never
            # ground anything against.
            print(
                f"error: concept graph {spec} exists but has no concepts "
                "— was it created outside seed-graph?",
                file=sys.stderr,
            )
            sys.exit(2)
        return meta

    matches = await graph.find_graphs_by_topic(spec)
    if matches:
        print(f"found {len(matches)} existing graph(s) for topic {spec!r}:")
        for i, m in enumerate(matches, start=1):
            print(f"  [{i}] {m.id}  (created {m.created_at})")
        print("  [n] seed a fresh graph instead")
        choice = input("resume which? ").strip().lower()
        if choice != "n":
            try:
                return matches[int(choice) - 1]
            except (ValueError, IndexError):
                print("error: invalid choice", file=sys.stderr)
                sys.exit(2)

    return await _do_seed_graph(graph, spec, llm)


async def _chat(learner_spec: str, topic_spec: str, use_stub: bool) -> None:
    tiers = _build_tier_clients(use_stub)
    pool = await create_pool(_database_url(), min_size=1, max_size=4)
    try:
        learner = await _resolve_learner(LearnerStore(pool), learner_spec)
        graph_meta = await _resolve_graph(ConceptGraph(pool), topic_spec, tiers.capable)
        label_suffix = f" (label={learner.label!r})" if learner.label else ""
        print(f"probe: learner {learner.id}{label_suffix}")
        print(f"probe: concept graph {graph_meta.id} (topic={graph_meta.topic!r})")

        loop = SessionLoop(
            hypothesis_store=HypothesisStore(pool),
            transcript=TranscriptStore(pool),
            node_calls=NodeCallStore(pool),
            concept_graph=ConceptGraph(pool),
            learner_overlay=LearnerOverlay(pool),
            revision_store=WorldModelRevisionStore(pool),
            llm=tiers.fast,
            model_tier_clients=tiers,
            branch_store=BranchStore(pool),
        )
        await loop.run_interactive(learner.id, graph_meta.id)
    finally:
        await pool.close()


async def _seed_graph(topic: str, use_stub: bool) -> None:
    tiers = _build_tier_clients(use_stub)
    pool = await create_pool(_database_url(), min_size=1, max_size=4)
    try:
        await _do_seed_graph(ConceptGraph(pool), topic, tiers.capable)
    finally:
        await pool.close()


def _prompt_field_updates() -> dict:
    print(
        "  enter the structured edit as a JSON object, e.g. "
        '{"common_misconceptions": ["..."]} (blank to skip approval):'
    )
    raw = input("  field_updates> ").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print("  could not parse that as JSON — skipping approval")
        return {}
    if not isinstance(parsed, dict):
        print("  field_updates must be a JSON object — skipping approval")
        return {}
    return parsed


async def _review_revisions() -> None:
    pool = await create_pool(_database_url(), min_size=1, max_size=4)
    try:
        revisions = WorldModelRevisionStore(pool)
        pending = await revisions.list_pending()
        if not pending:
            print("probe: no pending world-model revisions")
            return

        for revision in pending:
            print(f"\nrevision {revision.id} — concept {revision.concept_id!r}")
            print(f"  proposed_change: {revision.proposed_change}")
            print(f"  confidence: {revision.confidence:.2f}")
            if revision.evidence_refs:
                print("  evidence:")
                for ref in revision.evidence_refs:
                    print(
                        f"    - turn {ref.turn_id} ({ref.polarity.value}) "
                        f"at {ref.timestamp}"
                    )
            else:
                print("  evidence: (none)")

            decision = input("  [a]pprove / [r]eject / [s]kip? ").strip().lower()
            if decision == "a":
                field_updates = _prompt_field_updates()
                if not field_updates:
                    print("  no field updates entered — leaving pending")
                    continue
                try:
                    updated = await revisions.approve(revision.id, field_updates)
                except RevisionApplicationError as exc:
                    print(f"  error: {exc} — leaving pending")
                    continue
                print(f"  approved. applied_field_updates={updated.applied_field_updates}")
            elif decision == "r":
                await revisions.reject(revision.id)
                print("  rejected.")
            else:
                print("  skipped.")
    finally:
        await pool.close()


def _print_portrait(learner: Learner, report: LearnerPortrait) -> None:
    print(f"probe portrait — learner {learner.id}")
    if learner.label:
        print(f"  label: {learner.label}")
    print(f"  sessions so far: {report.session_count}")

    print("\ntop hypothesis per layer:")
    for top in report.top_hypotheses:
        if top.hypothesis is None:
            print(f"  {top.layer.value}: (none active)")
            continue
        h = top.hypothesis
        turns = [str(ref.turn_id) for ref in h.evidence_refs] or ["(none)"]
        print(
            f"  {top.layer.value}: {h.statement}\n"
            f"    p={h.probability:.2f} c={h.confidence:.2f} "
            f"evidence turns: {', '.join(turns)}"
        )

    print("\nhypothesis tier counts (boundedness/plateau check):")
    sessions = report.session_count
    for tier_name, count in report.tier_counts.items():
        ratio = f"{count / sessions:.2f}" if sessions else "n/a"
        print(f"  {tier_name}: {count}  ({ratio} per session)")

    print("\nlearner overlay:")
    if not report.overlay:
        print("  (no concepts touched yet)")
    for entry in report.overlay:
        name = entry.concept_name or "(unknown concept)"
        print(
            f"  {entry.concept_id} ({name}): {entry.entry.state.value} "
            f"(confidence={entry.entry.confidence:.2f})"
        )

    print("\npending world-model revisions from this learner:")
    if not report.pending_revisions:
        print("  (none)")
    for revision in report.pending_revisions:
        print(
            f"  {revision.id} [{revision.status.value}] "
            f"concept={revision.concept_id!r}: {revision.proposed_change}"
        )


async def _portrait(learner_id_str: str) -> None:
    try:
        learner_id = UUID(learner_id_str)
    except ValueError:
        print(f"error: {learner_id_str!r} is not a valid learner id", file=sys.stderr)
        sys.exit(2)

    pool = await create_pool(_database_url(), min_size=1, max_size=4)
    try:
        learners = LearnerStore(pool)
        learner = await learners.get(learner_id)
        if learner is None:
            print(f"error: no learner with id {learner_id}", file=sys.stderr)
            sys.exit(1)

        report = await build_portrait(
            learner_id,
            HypothesisStore(pool),
            TranscriptStore(pool),
            ConceptGraph(pool),
            LearnerOverlay(pool),
            WorldModelRevisionStore(pool),
        )
        _print_portrait(learner, report)
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="probe")
    subparsers = parser.add_subparsers(dest="command")
    chat_parser = subparsers.add_parser(
        "chat", help="start an interactive session loop"
    )
    chat_parser.add_argument(
        "--learner",
        required=True,
        help="learner label (resumes if it exists, creates if not) "
        "or an existing learner's UUID",
    )
    chat_parser.add_argument(
        "--topic",
        required=True,
        help="topic label (resumes/asks if a matching graph exists, "
        "seeds fresh if not) or an existing concept graph's UUID",
    )
    chat_parser.add_argument(
        "--stub",
        action="store_true",
        help="use StubLLMClient instead of the real Gemini API (no "
        "GEMINI_API_KEY needed, no cost)",
    )
    seed_parser = subparsers.add_parser(
        "seed-graph",
        help="one-time LLM seed of a concept graph for a topic (frozen after creation)",
    )
    seed_parser.add_argument(
        "topic", nargs="+", help="topic to seed, e.g. python closures"
    )
    seed_parser.add_argument(
        "--stub",
        action="store_true",
        help="use StubLLMClient instead of the real Gemini API (no "
        "GEMINI_API_KEY needed, no cost)",
    )
    subparsers.add_parser(
        "review-revisions",
        help="interactively approve or reject pending world-model revisions",
    )
    portrait_parser = subparsers.add_parser(
        "portrait",
        help="read-only report of what probe has learned about a learner",
    )
    portrait_parser.add_argument("learner_id", help="learner id (UUID)")
    args = parser.parse_args()

    if args.command == "chat":
        asyncio.run(_chat(args.learner, args.topic, args.stub))
    elif args.command == "seed-graph":
        asyncio.run(_seed_graph(" ".join(args.topic), args.stub))
    elif args.command == "review-revisions":
        asyncio.run(_review_revisions())
    elif args.command == "portrait":
        asyncio.run(_portrait(args.learner_id))
    else:
        parser.print_help()
        sys.exit(1)
