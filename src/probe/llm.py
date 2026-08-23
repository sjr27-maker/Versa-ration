from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Async LLM interface. Nodes depend on this, not on a specific provider."""

    async def complete(self, prompt: str) -> str: ...


class StubLLMClient:
    """Deterministic stub used until the real client lands.

    Infer's prompt gets an empty JSON array back (no evidence extracted).
    Teach's prompt gets a canned string that echoes the payload so we
    can see the loop end-to-end without paying for tokens.
    """

    async def complete(self, prompt: str) -> str:
        if prompt.startswith("INFER:"):
            return "[]"
        if prompt.startswith("TEACH:"):
            return f"[stub teach] {prompt[len('TEACH:'):].strip()[:200]}"
        return "[stub llm response]"
