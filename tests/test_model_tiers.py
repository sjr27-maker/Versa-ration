"""ModelTierConfig's env-override escape hatch, and SessionLoop's tier
wiring (default single-client backward compatibility, and the actual
fast/capable/best assignment when model_tier_clients is supplied).

The wiring tests construct SessionLoop directly with no database at
all: its constructor only stores references to the store objects it's
given, never calls a method on them, so a bare `object()` placeholder
is enough to exercise the tier assignment in isolation.
"""

from unittest.mock import Mock

from probe.llm import ModelTierClients, StubLLMClient
from probe.loop import SessionLoop
from probe.model_config import ModelTierConfig


def test_model_tier_config_defaults_when_no_env_vars_set(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL_FAST", raising=False)
    monkeypatch.delenv("GEMINI_MODEL_CAPABLE", raising=False)
    monkeypatch.delenv("GEMINI_MODEL_BEST", raising=False)

    cfg = ModelTierConfig.from_env()

    assert cfg == ModelTierConfig()


def test_model_tier_config_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL_FAST", "custom-fast")
    monkeypatch.setenv("GEMINI_MODEL_CAPABLE", "custom-capable")
    monkeypatch.setenv("GEMINI_MODEL_BEST", "custom-best")

    cfg = ModelTierConfig.from_env()

    assert cfg.fast == "custom-fast"
    assert cfg.capable == "custom-capable"
    assert cfg.best == "custom-best"


def _make_loop(**kwargs) -> SessionLoop:
    return SessionLoop(
        hypothesis_store=Mock(),
        transcript=Mock(),
        node_calls=Mock(),
        concept_graph=Mock(),
        learner_overlay=Mock(),
        revision_store=Mock(),
        **kwargs,
    )


def test_omitting_model_tier_clients_gives_every_node_the_same_llm():
    llm = StubLLMClient()
    loop = _make_loop(llm=llm)

    assert loop.infer._llm is llm
    assert loop.teach._llm is llm
    assert loop.plan._llm is llm
    assert loop.value_function._llm is llm
    assert loop.diagnose._detector._llm is llm
    assert loop.diagnose._grounder._llm is llm


def test_model_tier_clients_assigns_fast_capable_best_per_the_agreed_tiers():
    fast = StubLLMClient()
    capable = StubLLMClient()
    best = StubLLMClient()
    tiers = ModelTierClients(fast=fast, capable=capable, best=best)

    # llm= is still required (used as the fallback when
    # model_tier_clients is omitted) but must be ignored once tiers are
    # supplied explicitly.
    loop = _make_loop(llm=StubLLMClient(), model_tier_clients=tiers)

    assert loop.infer._llm is fast
    assert loop.value_function._llm is fast
    assert loop.diagnose._detector._llm is fast
    assert loop.diagnose._grounder._llm is fast
    assert loop.plan._llm is capable
    assert loop.teach._llm is best
