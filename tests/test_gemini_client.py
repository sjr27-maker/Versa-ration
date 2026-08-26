"""GeminiLLMClient tests run entirely offline: a fake stand-in for
google.genai.Client's `aio.models.generate_content` captures what
GeminiLLMClient asked for, so these assert the client's own dispatch
and error-wrapping logic without ever calling the real API.
"""

from types import SimpleNamespace

import pytest
from google.genai import errors

from probe.llm import GeminiLLMClient, LLMTransportError


class _FakeModels:
    def __init__(self, *, text: str = '{"ok": true}', raise_exc: Exception | None = None):
        self.calls: list[dict] = []
        self._text = text
        self._raise_exc = raise_exc

    async def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self._raise_exc is not None:
            raise self._raise_exc
        return SimpleNamespace(text=self._text)


class _FakeClient:
    def __init__(self, **kwargs):
        self.models = _FakeModels(**kwargs)
        self.aio = SimpleNamespace(models=self.models)


@pytest.mark.asyncio
async def test_teach_prompt_gets_no_json_mode_config():
    fake = _FakeClient(text="a plain-text teaching message")
    client = GeminiLLMClient(fake, "some-model")

    result = await client.complete("TEACH: {\"action\": \"explain\"}")

    assert result == "a plain-text teaching message"
    assert fake.models.calls[0]["config"] is None


@pytest.mark.asyncio
async def test_json_shaped_prompt_gets_a_response_schema():
    fake = _FakeClient(text='{"concept_id": null, "confidence": 0.0}')
    client = GeminiLLMClient(fake, "some-model")

    await client.complete("GROUND:CONCEPT\nstudent_response=ok\nconcepts:\n")

    config = fake.models.calls[0]["config"]
    assert config is not None
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema is not None
    assert config.response_json_schema["type"] == "OBJECT"


@pytest.mark.asyncio
async def test_scalar_score_prompt_gets_a_number_schema():
    fake = _FakeClient(text="0.42")
    client = GeminiLLMClient(fake, "some-model")

    result = await client.complete("SCORE:LEARNING_VALUE\naction=explain\n")

    assert result == "0.42"
    config = fake.models.calls[0]["config"]
    assert config.response_json_schema == {"type": "NUMBER"}


@pytest.mark.asyncio
async def test_info_update_prompt_gets_json_mode_without_a_fixed_schema():
    fake = _FakeClient(text="{}")
    client = GeminiLLMClient(fake, "some-model")

    await client.complete("SCORE:INFO_UPDATE\nresponse=x\n")

    config = fake.models.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema is None


@pytest.mark.asyncio
async def test_empty_candidate_returns_empty_string_not_an_exception():
    fake = _FakeClient(text=None)
    client = GeminiLLMClient(fake, "some-model")

    result = await client.complete("GROUND:CONCEPT\n")

    assert result == ""


@pytest.mark.asyncio
async def test_transport_failure_after_retries_raises_llm_transport_error():
    fake = _FakeClient(raise_exc=errors.ServerError(503, {"error": {"message": "unavailable"}}))
    client = GeminiLLMClient(fake, "some-model")

    with pytest.raises(LLMTransportError):
        await client.complete("TEACH: hi")


@pytest.mark.asyncio
async def test_client_error_after_retries_also_raises_llm_transport_error():
    fake = _FakeClient(raise_exc=errors.ClientError(429, {"error": {"message": "rate limited"}}))
    client = GeminiLLMClient(fake, "some-model")

    with pytest.raises(LLMTransportError):
        await client.complete("TEACH: hi")
