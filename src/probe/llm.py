from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Async LLM interface. Nodes depend on this, not on a specific provider."""

    async def complete(self, prompt: str) -> str: ...


_DEFAULT_RESPONSES: dict[str, str] = {
    # Existing loop nodes.
    "INFER:": "[]",
    "TEACH:": "[stub teach]",
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
    longest matching prefix wins. Anything unmatched falls through to
    `_DEFAULT_RESPONSES` and finally to a generic placeholder.
    """

    def __init__(self, canned: dict[str, str] | None = None) -> None:
        self.canned = canned or {}

    async def complete(self, prompt: str) -> str:
        for prefix in sorted(self.canned, key=len, reverse=True):
            if prompt.startswith(prefix):
                return self.canned[prefix]
        for prefix, response in _DEFAULT_RESPONSES.items():
            if prompt.startswith(prefix):
                return response
        return "[stub llm response]"
