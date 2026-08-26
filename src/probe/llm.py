from __future__ import annotations

import json
from collections.abc import Callable
from typing import NamedTuple, Protocol

from google import genai
from google.genai import errors, types

from probe.model_config import ModelTierConfig


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


class GeminiLLMClient:
    """Real LLMClient backed by the Gemini API, implementing the exact
    same Protocol as StubLLMClient — no node changes needed to use it.

    Structured-output mode (responseSchema) is used for every call
    whose result a node will JSON-parse (see `_SCHEMA_BY_PREFIX`
    above); TEACH: is exempted since its output is free-text shown
    directly to the student.

    Retry/timeout/rate-limit handling lives entirely at this boundary,
    via the SDK's own `HttpRetryOptions` — it retries only the
    transport-level failure modes (timeout, 429, 5xx) and never touches
    a 200 response whose body a node's own validation later rejects.
    See `LLMTransportError` for why those two layers are kept separate.
    """

    def __init__(self, client: genai.Client, model: str) -> None:
        self._client = client
        self._model = model

    async def complete(self, prompt: str) -> str:
        config_kwargs = _response_config_kwargs(prompt)
        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )
        except (errors.ServerError, errors.ClientError) as exc:
            # The SDK already retried transport-level failures per
            # HttpRetryOptions (see build_tier_clients); getting here
            # means it gave up. Wrapped in one exception type so
            # anything upstream that wants to catch this has a single
            # name to catch, not the SDK's internal error hierarchy.
            raise LLMTransportError(
                f"Gemini request failed after retries: {exc}"
            ) from exc
        text = getattr(response, "text", None)
        # A blocked/empty candidate (safety filter, no valid part) is a
        # content issue, not a transport one — return "" and let the
        # calling node's existing parse-failure path handle it, same as
        # it already handles a StubLLMClient canned garbage response.
        return text if text is not None else ""


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
) -> ModelTierClients:
    """Construct the three tiered GeminiLLMClients used by a real
    (non-stub) run. All three share one underlying genai.Client (and
    thus one retry/timeout policy, one connection pool) and differ only
    in which model name each wraps."""
    cfg = tier_config or ModelTierConfig.from_env()
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=int(timeout_seconds * 1000),
            retry_options=types.HttpRetryOptions(
                attempts=max_attempts,
                http_status_codes=[408, 429, 500, 502, 503, 504],
            ),
        ),
    )
    return ModelTierClients(
        fast=GeminiLLMClient(client, cfg.fast),
        capable=GeminiLLMClient(client, cfg.capable),
        best=GeminiLLMClient(client, cfg.best),
    )
