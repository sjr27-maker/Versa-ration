"""Embedding client — the memory layer's own LLMClient-equivalent
(probe.llm.LLMClient), but for vectors instead of text.

`gemini-embedding-001` is a Matryoshka-trained model: requesting a
smaller `output_dimensionality` still yields a usable embedding (just
truncated to fewer leading dimensions, not a degraded/different
model), and doing so keeps embeddings inside pgvector's actual index
limits — verified live: this pgvector build (0.8.6, see migration 030)
rejects an HNSW index on a `vector(3072)` column outright ("column
cannot have more than 2000 dimensions for hnsw index"), while a
`vector(768)` column indexes cleanly. 768 is EMBEDDING_DIM below, not
a config value: it's baked into the migration's column/index
definitions, so changing it means a new migration, not a runtime flag.
"""

from __future__ import annotations

import hashlib
import random
from typing import Protocol

from google import genai
from google.genai import types

# Baked into migration 030's `vector(768)` columns and HNSW indexes —
# see module docstring for why 768, not the model's native 3072.
EMBEDDING_DIM = 768


# gemini-embedding-001 is an ASYMMETRIC retrieval model: the query and
# the stored document must be embedded with matching task_types for
# cosine similarity to land where it should. Measured on real data
# (scratchpad diagnostic, 2026-09-01): a genuine paraphrase match rose
# from 0.76 -> 0.80 just by switching an untyped embed pair to
# RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT. `None` reproduces the old
# untyped behaviour exactly (and is what StubEmbeddingClient ignores).
TASK_QUERY = "RETRIEVAL_QUERY"
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_SIMILARITY = "SEMANTIC_SIMILARITY"


class EmbeddingClient(Protocol):
    """Async embedding interface. Nodes/stores depend on this, not on
    a specific provider — same split as LLMClient/GeminiLLMClient.

    `task_type` (see TASK_* above) is the retrieval role of `text`:
    the memory layer's search query passes TASK_QUERY, a fact being
    stored passes TASK_DOCUMENT, and a symmetric candidate-vs-candidate
    compare passes TASK_SIMILARITY. `None` = untyped (back-compat)."""

    async def embed(self, text: str, *, task_type: str | None = None) -> list[float]: ...


class StubEmbeddingClient:
    """Deterministic stub used in tests — no real API key, no cost,
    and (unlike a real embedding model) fully predictable similarity:
    the same text always embeds to the same vector, and two different
    default-hashed texts land far apart (effectively orthogonal,
    similarity near 0) unless a test explicitly wants two specific
    texts to read as similar.

    `canned` maps exact text -> a fixed vector, for tests that need
    two different strings to embed close together (e.g. "the SAME
    situation, worded differently" should still match) — same
    precedent as StubLLMClient's `canned` dict, just keyed by exact
    text rather than prompt prefix (there's no meaningful "prefix"
    for a single opaque string to embed).
    """

    def __init__(self, canned: dict[str, list[float]] | None = None) -> None:
        self.canned = canned or {}
        self.texts: list[str] = []

    async def embed(self, text: str, *, task_type: str | None = None) -> list[float]:
        # task_type is deliberately ignored: the stub's whole point is a
        # deterministic text -> vector map, so the same text must embed
        # identically regardless of retrieval role.
        self.texts.append(text)
        if text in self.canned:
            return self.canned[text]
        # A text's own hash seeds a deterministic RNG -- identical text
        # always reproduces the identical vector, and unrelated text
        # produces effectively uncorrelated (low-cosine-similarity)
        # vectors, without needing a real embedding model.
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        rng = random.Random(seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(EMBEDDING_DIM)]


class GeminiEmbeddingClient:
    """Real EmbeddingClient backed by the Gemini API. No retry/backoff
    loop of its own (unlike GeminiLLMClient) -- embedding calls are a
    much smaller part of this codebase's total call volume for now;
    revisit if live use shows embedding calls actually failing under
    load."""

    def __init__(self, client: genai.Client, model: str) -> None:
        self._client = client
        self._model = model

    async def embed(self, text: str, *, task_type: str | None = None) -> list[float]:
        config = types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIM,
            task_type=task_type,  # None -> the API's untyped default
        )
        response = await self._client.aio.models.embed_content(
            model=self._model,
            contents=text,
            config=config,
        )
        return list(response.embeddings[0].values)


def build_embedding_client(api_key: str, model: str | None = None) -> EmbeddingClient:
    """Mirrors llm.build_tier_clients' construction pattern, scaled
    down to the one client the memory layer needs -- no pooling, since
    embedding calls aren't expected to fan out concurrently the way
    Plan's candidate scoring does."""
    from probe.model_config import ModelTierConfig

    resolved_model = model or ModelTierConfig.from_env().embedding
    client = genai.Client(api_key=api_key)
    return GeminiEmbeddingClient(client, resolved_model)
