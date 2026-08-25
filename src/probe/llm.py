from __future__ import annotations

import json
from collections.abc import Callable
from typing import Protocol


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


_DEFAULT_RESPONSES: dict[str, str] = {
    # Existing loop nodes.
    "INFER:": "[]",
    "TEACH:": "[stub teach]",
    "PROPOSE:ACTIONS": _DEFAULT_PROPOSALS,
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
