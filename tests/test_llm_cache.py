import time
from types import SimpleNamespace

import litellm
import pytest

from src.config import AppConfig, LLMConfig, ProviderConfig
from src.llm.client import LLMGatewayError, StatsTracker, chat


def make_config(tmp_path, fallback_chain, max_retries=1, backoff_base_s=0.0):
    return AppConfig(
        seed=42,
        cache_dir=tmp_path / "cache",
        llm=LLMConfig(
            fallback_chain=fallback_chain,
            temperature=0.0,
            seed=42,
            max_tokens=64,
            max_retries=max_retries,
            backoff_base_s=backoff_base_s,
        ),
    )


def fake_response(content, prompt_tokens=5, completion_tokens=3):
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        get=lambda key, default=None: usage if key == "usage" else default,
    )


def test_chat_second_call_is_cache_hit(monkeypatch, tmp_path):
    call_count = {"n": 0}

    def fake_completion(**kwargs):
        call_count["n"] += 1
        return fake_response("hello world")

    monkeypatch.setattr(litellm, "completion", fake_completion)

    provider = ProviderConfig(provider="ollama", model="ollama/test-model", api_base="http://x")
    cfg = make_config(tmp_path, [provider])
    tracker = StatsTracker()

    r1 = chat("ping", config=cfg, tracker=tracker)
    r2 = chat("ping", config=cfg, tracker=tracker)

    assert r1.cached is False
    assert r2.cached is True
    assert r1.text == r2.text == "hello world"
    assert call_count["n"] == 1

    stats = tracker.summary()
    assert stats.calls == 2
    assert stats.cache_hits == 1
    assert stats.prompt_tokens == 5  # cache hit contributes 0 tokens, not a replay of the original


def test_chat_skips_provider_missing_required_key(monkeypatch, tmp_path):
    monkeypatch.delenv("FAKE_KEY_NOT_SET", raising=False)

    def fake_completion(**kwargs):
        assert kwargs["model"] == "ollama/test-model"
        return fake_response("from-fallback")

    monkeypatch.setattr(litellm, "completion", fake_completion)

    needs_key = ProviderConfig(
        provider="needs-key", model="needs/key-model", api_key_env="FAKE_KEY_NOT_SET"
    )
    ollama = ProviderConfig(provider="ollama", model="ollama/test-model")
    cfg = make_config(tmp_path, [needs_key, ollama])

    r = chat("ping", config=cfg)
    assert r.model == "ollama/test-model"


def test_chat_falls_back_after_retries_exhausted(monkeypatch, tmp_path):
    call_log = []

    def flaky_completion(**kwargs):
        call_log.append(kwargs["model"])
        if kwargs["model"] == "flaky/model":
            raise litellm.RateLimitError(
                message="rate limited", llm_provider="flaky", model="flaky/model"
            )
        return fake_response("from-secondary")

    monkeypatch.setattr(litellm, "completion", flaky_completion)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    flaky = ProviderConfig(provider="flaky", model="flaky/model")
    ollama = ProviderConfig(provider="ollama", model="ollama/test-model")
    cfg = make_config(tmp_path, [flaky, ollama], max_retries=1)

    r = chat("ping", config=cfg)
    assert r.model == "ollama/test-model"
    assert call_log.count("flaky/model") == 2  # initial attempt + 1 retry


def test_chat_raises_when_all_providers_fail(monkeypatch, tmp_path):
    def always_fails(**kwargs):
        raise litellm.APIConnectionError(
            message="down", llm_provider=kwargs["model"], model=kwargs["model"]
        )

    monkeypatch.setattr(litellm, "completion", always_fails)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    provider = ProviderConfig(provider="ollama", model="ollama/test-model")
    cfg = make_config(tmp_path, [provider], max_retries=0)

    with pytest.raises(LLMGatewayError):
        chat("ping", config=cfg)
