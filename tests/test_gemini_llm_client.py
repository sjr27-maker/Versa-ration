"""GeminiLLMClient's own retry loop (replacing the SDK's opaque
internal tenacity retry — see its docstring for why). Exercised against
a fake genai.Client double, not the real API: GeminiLLMClient only ever
calls `self._client.aio.models.generate_content(...)`, so a bare object
shaped the same way is enough to drive every retry/backoff/give-up path
deterministically and for free.
"""

from __future__ import annotations

import logging

import pytest
from google.genai import errors

from probe.llm import GeminiLLMClient, LLMTransportError, _PooledGeminiLLMClient


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModels:
    """outcomes: a list consumed in order — each entry is either an
    Exception instance (raised) or a string (wrapped as a successful
    response's .text)."""

    def __init__(self, outcomes: list) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    async def generate_content(self, model, contents, config):
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeResponse(outcome)


class _FakeAio:
    def __init__(self, models: _FakeModels) -> None:
        self.models = models


class _FakeGenaiClient:
    def __init__(self, outcomes: list) -> None:
        self.models = _FakeModels(outcomes)
        self.aio = _FakeAio(self.models)


def _rate_limited() -> errors.ClientError:
    return errors.ClientError(429, {"error": {"message": "rate limited"}})


def _bad_request() -> errors.ClientError:
    return errors.ClientError(400, {"error": {"message": "bad request"}})


def _fast_client(outcomes: list, **kwargs) -> tuple[GeminiLLMClient, _FakeGenaiClient]:
    fake = _FakeGenaiClient(outcomes)
    kwargs.setdefault("initial_delay", 0.0)
    kwargs.setdefault("max_delay", 0.0)
    kwargs.setdefault("jitter", 0.0)
    return GeminiLLMClient(fake, "test-model", **kwargs), fake


@pytest.mark.asyncio
async def test_succeeds_immediately_with_no_retries():
    client, fake = _fast_client(["hello"])
    result = await client.complete("TEACH: hi")
    assert result == "hello"
    assert fake.models.calls == 1
    assert client.retry_count == 0


@pytest.mark.asyncio
async def test_retries_a_429_then_succeeds_and_counts_the_retry():
    client, fake = _fast_client([_rate_limited(), "hello after retry"])
    result = await client.complete("TEACH: hi")
    assert result == "hello after retry"
    assert fake.models.calls == 2
    assert client.retry_count == 1


@pytest.mark.asyncio
async def test_logs_model_prompt_prefix_attempt_status_and_backoff(caplog):
    client, _fake = _fast_client([_rate_limited(), "ok"])
    with caplog.at_level(logging.WARNING, logger="probe.llm"):
        await client.complete("SCORE:LEARNING_VALUE\naction=explain\n" + "x" * 200)

    records = [r for r in caplog.records if "retry" in r.getMessage().lower()]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "model=test-model" in message
    assert "SCORE:LEARNING_VALUE" in message
    # Prompt is truncated in the log, not dumped in full.
    assert "x" * 200 not in message
    assert "attempt=1/3" in message
    assert "http_status=429" in message
    assert "backoff=" in message


@pytest.mark.asyncio
async def test_gives_up_after_max_attempts_and_wraps_in_llm_transport_error():
    client, fake = _fast_client(
        [_rate_limited(), _rate_limited(), _rate_limited()], max_attempts=3
    )
    with pytest.raises(LLMTransportError):
        await client.complete("TEACH: hi")
    assert fake.models.calls == 3
    assert client.retry_count == 2  # 2 retries before the 3rd, final failure


@pytest.mark.asyncio
async def test_non_retryable_status_fails_immediately_without_retrying():
    client, fake = _fast_client([_bad_request(), "unreachable"])
    with pytest.raises(LLMTransportError):
        await client.complete("TEACH: hi")
    assert fake.models.calls == 1  # never tried the second (would-be) attempt
    assert client.retry_count == 0


@pytest.mark.asyncio
async def test_pooled_client_sums_retry_count_across_its_delegates():
    fakes = [_FakeGenaiClient([_rate_limited(), "a"]), _FakeGenaiClient(["b"])]
    pooled = _PooledGeminiLLMClient(fakes, "test-model", max_attempts=3)
    for delegate in pooled._delegates:
        delegate._initial_delay = 0.0
        delegate._max_delay = 0.0
        delegate._jitter = 0.0

    await pooled.complete("first")  # round-robins to delegate 0 (1 retry)
    await pooled.complete("second")  # round-robins to delegate 1 (0 retries)

    assert pooled.retry_count == 1
