from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from collections.abc import Callable
from typing import NamedTuple, Protocol

import httpx
from google import genai
from google.genai import errors, types

from probe.model_config import ModelTierConfig

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Async LLM interface. Nodes depend on this, not on a specific provider."""

    async def complete(self, prompt: str) -> str: ...


# A canned entry is either a fixed string or a function of the prompt, so
# tests can stub prompt-sensitive behaviour (e.g. a proposer that reacts
# to which hypotheses appear in the listing) without a real LLM.
CannedResponse = str | Callable[[str], str]


# Two generic openers, so a width-3 plan still needs one backfill and the
# backfill path stays exercised by the default configuration.
_DEFAULT_PROPOSALS = json.dumps(
    [
        {
            "action": "explain",
            "target_concept": None,
            "rationale": "stub proposer: lead with a direct explanation",
        },
        {
            "action": "ask",
            "target_concept": None,
            "rationale": "stub proposer: check what the student already has",
        },
    ]
)


_DEFAULT_CONCEPT_BATCH = json.dumps(
    [
        {
            "id": "stub_base",
            "name": "Stub Base Concept",
            "prerequisites": [],
            "common_misconceptions": [],
            "representations": ["formal"],
            "diagnostic_questions": ["what is the stub base concept?"],
        },
        {
            "id": "stub_derived",
            "name": "Stub Derived Concept",
            "prerequisites": ["stub_base"],
            "common_misconceptions": [],
            "representations": ["formal"],
            "diagnostic_questions": ["how does stub derived build on stub base?"],
        },
    ]
)


_DEFAULT_INTENT_BRANCHES = json.dumps(
    [
        {
            "statement": "wants to relate the new idea to something familiar",
            "plausibility": 0.6,
            "predicted_next_turn": "will ask for a real-world analogy",
        },
        {
            "statement": "is missing a prerequisite concept",
            "plausibility": 0.5,
            "predicted_next_turn": "will ask a clarifying question about an earlier concept",
        },
        {
            "statement": "is testing the tutor's explanation for correctness",
            "plausibility": 0.3,
            "predicted_next_turn": "will point out a perceived inconsistency",
        },
    ]
)


_DEFAULT_EXPAND_BRANCHES = json.dumps(
    {
        "layer_label": "knowledge_gap",
        "children": [
            {
                "statement": "may be missing the underlying definition",
                "plausibility": 0.5,
                "predicted_next_turn": "will ask what the term actually means",
            },
            {
                "statement": "may be conflating this with a related but different concept",
                "plausibility": 0.4,
                "predicted_next_turn": "will describe the related concept instead",
            },
        ],
    }
)


_DEFAULT_PATH_REQUIREMENT = json.dumps(
    {
        "current_belief": "the student's current understanding is not yet clear",
        "needed": "a direct explanation of the target concept",
        "must_not_assume": [],
        "scope": "one concrete idea, not a full syllabus",
    }
)


_DEFAULT_RESPONSES: dict[str, str] = {
    # Existing loop nodes.
    "INFER:": "[]",
    "TEACH:": "[stub teach]",
    "PROPOSE:ACTIONS": _DEFAULT_PROPOSALS,
    "SEED:CONCEPT_GRAPH": _DEFAULT_CONCEPT_BATCH,
    # Conservative default: no mismatch, so MismatchDetector doesn't
    # propose revisions or reweight hypotheses unless a test opts in.
    "MISMATCH:DETECT": json.dumps({"mismatch": False}),
    # Conservative default: no grounding, so GroundConcept doesn't act
    # on a fabricated concept unless a test opts in.
    "GROUND:CONCEPT": json.dumps({"concept_id": None, "confidence": 0.0}),
    # ValueFunction stubs — conservative middle values so score() is well-
    # defined even without canned test overrides. Individual tests provide
    # explicit canned responses when they want specific values.
    "SCORE:LEARNING_VALUE": "0.5",
    "SCORE:COGNITIVE_COST": "0.3",
    "SCORE:FRUSTRATION_RISK": "0.2",
    "SCORE:INFO_RESPONSES": "[]",
    "SCORE:INFO_UPDATE": "{}",
    # HypothesisGenerator's speculative prediction tree.
    "GENERATE:INTENT": _DEFAULT_INTENT_BRANCHES,
    "GENERATE:EXPAND": _DEFAULT_EXPAND_BRANCHES,
    # Conservative default: no match, so resolve() doesn't fabricate a
    # match unless a test opts in — same convention as MISMATCH/GROUND.
    "RESOLVE:MATCH": json.dumps({"matched_branch_id": None, "confidence": 0.0}),
    # Conservative default: no selection, so SelectBranch falls back to
    # its own deterministic highest-plausibility choice rather than a
    # stub-fabricated id — same convention as RESOLVE:MATCH.
    "SELECT:BRANCH": json.dumps({"selected_branch_id": None, "rationale": ""}),
    "DERIVE:PATH": _DEFAULT_PATH_REQUIREMENT,
    # AttachTopic's topic-extraction call.
    "TOPIC:INFER": json.dumps({"topic": "stub topic"}),
}


class StubLLMClient:
    """Deterministic stub used until the real client lands.

    Callers may pass a `canned` dict of {prompt_prefix: response}; the
    longest matching prefix wins. A canned value may be a callable taking
    the full prompt, for stubs that need to vary with prompt content.
    Anything unmatched falls through to `_DEFAULT_RESPONSES` and finally
    to a generic placeholder.

    Every prompt is appended to `self.prompts` so tests can assert on
    what a node actually asked for.
    """

    def __init__(self, canned: dict[str, CannedResponse] | None = None) -> None:
        self.canned = canned or {}
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        for prefix in sorted(self.canned, key=len, reverse=True):
            if prompt.startswith(prefix):
                response = self.canned[prefix]
                return response(prompt) if callable(response) else response
        for prefix, response in _DEFAULT_RESPONSES.items():
            if prompt.startswith(prefix):
                return response
        return "[stub llm response]"


class LLMTransportError(Exception):
    """Raised when GeminiLLMClient exhausts its retries against a
    transport-level failure (timeout, rate limit, 5xx).

    A malformed or empty response body is deliberately NOT this: that's
    a content issue, and content-level recovery already exists per-node
    (Plan's rejected-value re-ask, GroundConcept/MismatchDetector's
    graceful None on a JSONDecodeError, Infer's per-item skip). This
    client returns whatever text a successful HTTP round-trip produced
    — including "" for a blocked/empty candidate — and lets that
    existing logic reject it exactly as it already rejects a canned
    garbage StubLLMClient response. Client-level and node-level retries
    must not compound: this client only ever retries the HTTP
    round-trip itself, never the model's content.
    """


# Prompt-prefix -> Gemini structured-output config, mirroring the same
# longest-prefix dispatch StubLLMClient uses to decide what each node's
# prompt expects back. Every node either json.loads()s the response or
# float()s it — TEACH: is the one exception (its output is the message
# shown to the student, not something downstream parses), so it alone
# gets no JSON mode. SCORE:INFO_UPDATE gets JSON mode without a fixed
# schema: its keys are hypothesis UUIDs decided per-call, which a static
# schema can't name in advance.
_NUMBER_SCHEMA = {"type": "NUMBER"}
_JSON_ONLY = object()  # sentinel: JSON mode, no fixed schema
_FREE_TEXT = object()  # sentinel: no JSON mode at all

_SCHEMA_BY_PREFIX: dict[str, object] = {
    # Best-effort schema for the current INFER: prompt. Note the prompt
    # itself (nodes.py, Infer.run) never actually asks the model for an
    # evidence_ref/turn_id — it's a still-stub prompt (see its
    # docstring) — so a real model's replies validate against
    # ProposedEvidence about as often as StubLLMClient's "[]" default
    # does today: rarely to never. That's a pre-existing gap in the
    # prompt, not something this client should paper over by inventing
    # a turn_id the model was never told.
    "INFER:": {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "hypothesis_id": {"type": "STRING"},
                "new_probability": {"type": "NUMBER"},
                "new_confidence": {"type": "NUMBER"},
                "polarity": {
                    "type": "STRING",
                    "enum": ["supporting", "contradicting"],
                },
            },
            "required": ["hypothesis_id", "new_probability", "new_confidence"],
        },
    },
    "PROPOSE:ACTIONS": {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING"},
                "target_concept": {"type": "STRING", "nullable": True},
                "rationale": {"type": "STRING"},
            },
            "required": ["action"],
        },
    },
    "SEED:CONCEPT_GRAPH": {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "id": {"type": "STRING"},
                "name": {"type": "STRING"},
                "prerequisites": {"type": "ARRAY", "items": {"type": "STRING"}},
                "common_misconceptions": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                },
                "representations": {"type": "ARRAY", "items": {"type": "STRING"}},
                "diagnostic_questions": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                },
            },
            "required": ["id", "name"],
        },
    },
    "MISMATCH:DETECT": {
        "type": "OBJECT",
        "properties": {
            "mismatch": {"type": "BOOLEAN"},
            "learner_claim": {"type": "STRING"},
            "world_claim": {"type": "STRING"},
            "confidence": {"type": "NUMBER"},
            "suggested_cause": {
                "type": "STRING",
                "enum": ["learner_misconception", "possible_world_model_error"],
            },
        },
        "required": ["mismatch"],
    },
    "GROUND:CONCEPT": {
        "type": "OBJECT",
        "properties": {
            "concept_id": {"type": "STRING", "nullable": True},
            "confidence": {"type": "NUMBER"},
        },
        "required": ["confidence"],
    },
    "SCORE:LEARNING_VALUE": _NUMBER_SCHEMA,
    "SCORE:COGNITIVE_COST": _NUMBER_SCHEMA,
    "SCORE:FRUSTRATION_RISK": _NUMBER_SCHEMA,
    "SCORE:INFO_RESPONSES": {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "response": {"type": "STRING"},
                "probability": {"type": "NUMBER"},
            },
            "required": ["response", "probability"],
        },
    },
    "SCORE:INFO_UPDATE": _JSON_ONLY,
    "TEACH:": _FREE_TEXT,
    "GENERATE:INTENT": {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "statement": {"type": "STRING"},
                "plausibility": {"type": "NUMBER"},
                "predicted_next_turn": {"type": "STRING"},
            },
            "required": ["statement", "plausibility", "predicted_next_turn"],
        },
    },
    "GENERATE:EXPAND": {
        "type": "OBJECT",
        "properties": {
            "layer_label": {"type": "STRING"},
            "children": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "statement": {"type": "STRING"},
                        "plausibility": {"type": "NUMBER"},
                        "predicted_next_turn": {"type": "STRING"},
                    },
                    "required": ["statement", "plausibility", "predicted_next_turn"],
                },
            },
        },
        "required": ["layer_label", "children"],
    },
    "RESOLVE:MATCH": {
        "type": "OBJECT",
        "properties": {
            "matched_branch_id": {"type": "STRING", "nullable": True},
            "confidence": {"type": "NUMBER"},
        },
        "required": ["confidence"],
    },
    "SELECT:BRANCH": {
        "type": "OBJECT",
        "properties": {
            "selected_branch_id": {"type": "STRING", "nullable": True},
            "rationale": {"type": "STRING"},
        },
        "required": ["rationale"],
    },
    "DERIVE:PATH": {
        "type": "OBJECT",
        "properties": {
            "current_belief": {"type": "STRING"},
            "needed": {"type": "STRING"},
            "must_not_assume": {"type": "ARRAY", "items": {"type": "STRING"}},
            "scope": {"type": "STRING"},
        },
        "required": ["current_belief", "needed", "must_not_assume", "scope"],
    },
    "TOPIC:INFER": {
        "type": "OBJECT",
        "properties": {"topic": {"type": "STRING"}},
        "required": ["topic"],
    },
}


def _response_config_kwargs(prompt: str) -> dict:
    """GenerateContentConfig kwargs for one prompt, by longest matching
    known prefix. An unrecognized prefix defaults to plain JSON mode
    (no fixed schema) rather than free text: every node in this
    codebase except Teach either json.loads()s or float()s its
    response, so JSON-shaped is the safer default for anything new."""
    for prefix in sorted(_SCHEMA_BY_PREFIX, key=len, reverse=True):
        if prompt.startswith(prefix):
            schema = _SCHEMA_BY_PREFIX[prefix]
            if schema is _FREE_TEXT:
                return {}
            if schema is _JSON_ONLY:
                return {"response_mime_type": "application/json"}
            return {
                "response_mime_type": "application/json",
                "response_json_schema": schema,
            }
    return {"response_mime_type": "application/json"}


# Exceptions the SDK itself would have retried transparently (timeout,
# connection failure) that don't carry an HTTP status code the way
# errors.APIError does — always retryable, same as the SDK's own
# _HTTPX_TRANSIENT_EXC scope.
_TRANSIENT_HTTPX_EXC: tuple[type[Exception], ...] = (
    httpx.TimeoutException,
    httpx.ConnectError,
)

# How much of a prompt to log on a retry — enough to identify which
# node/call this was (every prompt starts with a fixed prefix like
# "TEACH:", "SCORE:LEARNING_VALUE", "GENERATE:EXPAND") without dumping
# the whole prompt (which can include full hypothesis listings) into logs.
_RETRY_LOG_PROMPT_CHARS = 80


class GeminiLLMClient:
    """Real LLMClient backed by the Gemini API, implementing the exact
    same Protocol as StubLLMClient — no node changes needed to use it.

    Structured-output mode (responseSchema) is used for every call
    whose result a node will JSON-parse (see `_SCHEMA_BY_PREFIX`
    above); TEACH: is exempted since its output is free-text shown
    directly to the student.

    Retry/timeout/rate-limit handling lives entirely at this boundary —
    but as its own explicit loop, not the SDK's built-in
    `HttpRetryOptions` (build_tier_clients disables that at the
    genai.Client level, attempts=1). The SDK's retry is opaque tenacity
    machinery with no callback hook: it logs a generic "retrying in N
    seconds" line to its own internal logger with no way to attach
    which node/model/prompt the retry belongs to. Owning the loop here
    means a throttled call and a genuinely slow call are distinguishable
    — every attempt logs model, prompt prefix, attempt number, the HTTP
    status that triggered it, and the backoff chosen — and `retry_count`
    gives SessionLoop a running total it can snapshot per turn for
    turn_diagnostics (see loop.py's _total_retry_count).

    Retries only transport-level failure modes (timeout, 429, 5xx) and
    never touches a 200 response whose body a node's own validation
    later rejects. See `LLMTransportError` for why those two layers are
    kept separate.
    """

    def __init__(
        self,
        client: genai.Client,
        model: str,
        *,
        max_attempts: int = 3,
        retryable_status_codes: tuple[int, ...] = (408, 429, 500, 502, 503, 504),
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exp_base: float = 2.0,
        jitter: float = 1.0,
    ) -> None:
        self._client = client
        self._model = model
        self._max_attempts = max_attempts
        self._retryable_status_codes = retryable_status_codes
        self._initial_delay = initial_delay
        self._max_delay = max_delay
        self._exp_base = exp_base
        self._jitter = jitter
        # Cumulative for this client's whole lifetime, not reset per
        # call — SessionLoop reads the delta between a snapshot taken
        # at the start and end of a turn to get that turn's retry
        # count, the same before/after pattern duration_ms already uses
        # with time.monotonic().
        self.retry_count: int = 0

    async def complete(self, prompt: str) -> str:
        config_kwargs = _response_config_kwargs(prompt)
        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        prompt_prefix = prompt[:_RETRY_LOG_PROMPT_CHARS]
        delay = self._initial_delay

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )
                text = getattr(response, "text", None)
                # A blocked/empty candidate (safety filter, no valid
                # part) is a content issue, not a transport one —
                # return "" and let the calling node's existing
                # parse-failure path handle it, same as it already
                # handles a StubLLMClient canned garbage response.
                return text if text is not None else ""
            except (errors.ServerError, errors.ClientError, *_TRANSIENT_HTTPX_EXC) as exc:
                status = getattr(exc, "code", None)
                retryable = (
                    status in self._retryable_status_codes
                    if status is not None
                    else isinstance(exc, _TRANSIENT_HTTPX_EXC)
                )
                if not retryable or attempt >= self._max_attempts:
                    # Wrapped in one exception type so anything
                    # upstream that wants to catch this has a single
                    # name to catch, not the SDK's internal error
                    # hierarchy.
                    raise LLMTransportError(
                        f"Gemini request failed after {attempt} attempt(s) "
                        f"(model={self._model}): {exc}"
                    ) from exc
                self.retry_count += 1
                backoff = min(self._max_delay, delay) + random.uniform(
                    0, self._jitter
                )
                logger.warning(
                    "GeminiLLMClient retry: model=%s prompt_prefix=%r "
                    "attempt=%d/%d http_status=%s backoff=%.2fs",
                    self._model,
                    prompt_prefix,
                    attempt,
                    self._max_attempts,
                    status,
                    backoff,
                )
                await asyncio.sleep(backoff)
                delay = min(self._max_delay, delay * self._exp_base)
        # Unreachable: the loop above always returns on success or
        # raises once attempts are exhausted.
        raise AssertionError("unreachable")


class _PooledGeminiLLMClient:
    """Round-robins `complete()` calls across a small pool of independent
    `genai.Client` connections for one tier, so N concurrent calls to the
    same tier don't serialize behind a single HTTP connection. Each call
    still goes through the exact same GeminiLLMClient.complete() path
    (same retry policy, same structured-output config) — only which
    underlying connection carries it differs, so behavior/content is
    identical to the unpooled client.

    The round-robin counter is a plain int mutated between awaits, which
    is safe under asyncio's single-threaded cooperative scheduling (no
    `await` occurs between reading and incrementing it), not because of
    any lock.
    """

    def __init__(
        self, clients: list[genai.Client], model: str, *, max_attempts: int = 3
    ) -> None:
        self._delegates = [
            GeminiLLMClient(c, model, max_attempts=max_attempts) for c in clients
        ]
        self._next = 0

    async def complete(self, prompt: str) -> str:
        delegate = self._delegates[self._next % len(self._delegates)]
        self._next += 1
        return await delegate.complete(prompt)

    @property
    def retry_count(self) -> int:
        # Sum across every connection in the pool — each GeminiLLMClient
        # delegate tracks its own retries; this is the pool-wide total
        # SessionLoop reads for turn_diagnostics.
        return sum(d.retry_count for d in self._delegates)


class ModelTierClients(NamedTuple):
    """One LLMClient per tier. Passed into SessionLoop so each node is
    constructed with the tier its agreed assignment calls for (see
    model_config.py). Never constructed with mixed provider types in
    practice, but nothing here requires that — a test can hand
    SessionLoop three StubLLMClients, or three of the same instance for
    the current default (fully backward-compatible) behavior."""

    fast: LLMClient
    capable: LLMClient
    best: LLMClient


def build_tier_clients(
    api_key: str,
    tier_config: ModelTierConfig | None = None,
    *,
    timeout_seconds: float = 30.0,
    max_attempts: int = 3,
    pool_size: int | None = None,
) -> ModelTierClients:
    """Construct the three tiered LLMClients used by a real (non-stub)
    run. Each tier gets its own pool of `pool_size` independent
    genai.Client connections (each with the same retry/timeout policy),
    round-robinned by _PooledGeminiLLMClient — so a turn's fan-out of
    concurrent calls to the same tier (e.g. Plan scoring several
    candidates at once) doesn't queue behind one shared HTTP connection.
    pool_size=1 skips the round-robin wrapper entirely and returns the
    plain unpooled GeminiLLMClient, unchanged from before this existed.

    Defaults to 4, overridable via GEMINI_CLIENT_POOL_SIZE — same
    env-override escape hatch as ModelTierConfig.from_env(), here
    because the right pool size depends on the API key's actual rate
    limit, not something this codebase can pin correctly in advance.
    """
    cfg = tier_config or ModelTierConfig.from_env()
    if pool_size is None:
        pool_size = int(os.getenv("GEMINI_CLIENT_POOL_SIZE", "4"))

    def _make_connection() -> genai.Client:
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout_seconds * 1000),
                # SDK-level retry disabled (attempts=1) — GeminiLLMClient
                # now runs its own retry loop instead, so every attempt
                # can be logged with the model/prompt/status/backoff
                # this boundary needs (see GeminiLLMClient's docstring
                # for why the SDK's own tenacity retry can't do this).
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    def _tier_client(model: str) -> LLMClient:
        if pool_size <= 1:
            return GeminiLLMClient(_make_connection(), model, max_attempts=max_attempts)
        connections = [_make_connection() for _ in range(pool_size)]
        return _PooledGeminiLLMClient(connections, model, max_attempts=max_attempts)

    return ModelTierClients(
        fast=_tier_client(cfg.fast),
        capable=_tier_client(cfg.capable),
        best=_tier_client(cfg.best),
    )
