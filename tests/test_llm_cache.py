import time
from types import SimpleNamespace

import litellm
import pytest

from src.config import AppConfig, LLMConfig, ProviderConfig
from src.llm.client import LLMContextBudgetError, LLMGatewayError, StatsTracker, chat


def make_config(tmp_path, fallback_chain, max_retries=1, backoff_base_s=0.0, **llm_kwargs):
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
            **llm_kwargs,
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


def test_context_budget_guard_raises_before_any_call(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_completion(**kwargs):
        calls["n"] += 1
        return fake_response("should not be reached")

    monkeypatch.setattr(litellm, "completion", fake_completion)

    provider = ProviderConfig(provider="ollama", model="ollama/test-model")
    cfg = make_config(tmp_path, [provider], num_ctx=100, context_completion_reserve=20)

    with pytest.raises(LLMContextBudgetError):
        chat("word " * 500, config=cfg)  # ~500 tokens > 100 - 20 budget
    assert calls["n"] == 0  # hard stop before the provider is touched


def test_context_budget_guard_allows_prompt_within_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(litellm, "completion", lambda **kw: fake_response("ok"))
    provider = ProviderConfig(provider="ollama", model="ollama/test-model")
    cfg = make_config(tmp_path, [provider], num_ctx=8192, context_completion_reserve=512)
    r = chat("a short prompt", config=cfg)
    assert r.text == "ok"


def test_num_ctx_is_part_of_cache_key(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_completion(**kwargs):
        calls["n"] += 1
        return fake_response(f"call-{calls['n']}")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    provider = ProviderConfig(provider="ollama", model="ollama/test-model")

    cfg_a = make_config(tmp_path, [provider], num_ctx=4096)
    cfg_b = make_config(tmp_path, [provider], num_ctx=8192)
    chat("ping", config=cfg_a)
    chat("ping", config=cfg_b)  # different num_ctx -> different cache key -> real call
    assert calls["n"] == 2


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


# --------------------------------------------------------------------------
# logprobs (Phase 5, src/calibration): chat(logprobs=True) routing, extraction,
# cache-key isolation, and cache round-trip.
# --------------------------------------------------------------------------


def fake_response_with_logprobs(content, top_logprobs=None, prompt_tokens=5, completion_tokens=1):
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    lp_content = SimpleNamespace(
        token=content,
        logprob=-0.001,
        top_logprobs=[
            SimpleNamespace(token=t, logprob=lp) for t, lp in (top_logprobs or [(content, -0.001)])
        ],
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                logprobs=SimpleNamespace(content=[lp_content]),
            )
        ],
        get=lambda key, default=None: usage if key == "usage" else default,
    )


def test_chat_logprobs_false_by_default_leaves_logprobs_field_none(monkeypatch, tmp_path):
    monkeypatch.setattr(litellm, "completion", lambda **kw: fake_response("hi"))
    provider = ProviderConfig(provider="ollama", model="ollama/test-model")
    cfg = make_config(tmp_path, [provider])

    resp = chat("ping", config=cfg)
    assert resp.logprobs is None


def test_chat_logprobs_true_extracts_first_token_and_alternatives(monkeypatch, tmp_path):
    captured_kwargs = {}

    def fake_completion(**kwargs):
        captured_kwargs.update(kwargs)
        return fake_response_with_logprobs("B", top_logprobs=[("B", -0.001), ("A", -5.0)])

    monkeypatch.setattr(litellm, "completion", fake_completion)
    provider = ProviderConfig(provider="ollama", model="ollama/test-model")
    cfg = make_config(tmp_path, [provider])

    resp = chat("pick one", config=cfg, logprobs=True, top_logprobs=6)

    assert resp.logprobs == [
        {
            "token": "B",
            "logprob": -0.001,
            "top_logprobs": [{"token": "B", "logprob": -0.001}, {"token": "A", "logprob": -5.0}],
        }
    ]


def test_chat_logprobs_true_on_ollama_routes_through_openai_passthrough(monkeypatch, tmp_path):
    """AUDIT-adjacent fix: LiteLLM's native ollama/ provider path rejects
    logprobs. A logprobs=True call against an ollama provider must be routed
    through the openai/ provider type at <api_base>/v1 instead."""
    captured_kwargs = {}

    def fake_completion(**kwargs):
        captured_kwargs.update(kwargs)
        return fake_response_with_logprobs("A")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    provider = ProviderConfig(
        provider="ollama", model="ollama/qwen2.5:7b-instruct", api_base="http://localhost:11434"
    )
    cfg = make_config(tmp_path, [provider])

    chat("pick one", config=cfg, logprobs=True, top_logprobs=5)

    assert captured_kwargs["model"] == "openai/qwen2.5:7b-instruct"
    assert captured_kwargs["api_base"] == "http://localhost:11434/v1"
    assert captured_kwargs["api_key"] == "ollama"
    assert captured_kwargs["logprobs"] is True
    assert captured_kwargs["top_logprobs"] == 5


def test_chat_logprobs_true_on_non_ollama_provider_passes_through_unrouted(monkeypatch, tmp_path):
    captured_kwargs = {}

    def fake_completion(**kwargs):
        captured_kwargs.update(kwargs)
        return fake_response_with_logprobs("A")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    provider = ProviderConfig(provider="groq", model="groq/llama-3.1-8b-instant")
    cfg = make_config(tmp_path, [provider])

    chat("pick one", config=cfg, logprobs=True, top_logprobs=4)

    assert captured_kwargs["model"] == "groq/llama-3.1-8b-instant"
    assert "api_key" not in captured_kwargs  # only the ollama passthrough forces one
    assert captured_kwargs["logprobs"] is True


def test_chat_logprobs_true_and_false_get_different_cache_keys(monkeypatch, tmp_path):
    """A plain (logprobs=False, the default) call must compute the SAME cache
    key it always has -- adding this feature must not invalidate any existing
    .cache/llm/ entry."""
    monkeypatch.setattr(litellm, "completion", lambda **kw: fake_response_with_logprobs("A"))
    provider = ProviderConfig(provider="ollama", model="ollama/test-model")
    cfg = make_config(tmp_path, [provider])

    r_plain = chat("pick one", config=cfg, use_cache=False)
    r_lp = chat("pick one", config=cfg, use_cache=False, logprobs=True)
    assert r_plain.prompt_hash != r_lp.prompt_hash


def test_chat_logprobs_cache_hit_replays_logprobs_field(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_completion(**kwargs):
        calls["n"] += 1
        return fake_response_with_logprobs("B", top_logprobs=[("B", -0.01), ("A", -6.0)])

    monkeypatch.setattr(litellm, "completion", fake_completion)
    provider = ProviderConfig(provider="ollama", model="ollama/test-model")
    cfg = make_config(tmp_path, [provider])

    r1 = chat("pick one", config=cfg, logprobs=True)
    r2 = chat("pick one", config=cfg, logprobs=True)

    assert calls["n"] == 1  # second call was a cache hit, no real call made
    assert r2.cached is True
    assert r2.logprobs == r1.logprobs


def test_chat_logprobs_missing_from_response_is_none_not_an_error(monkeypatch, tmp_path):
    """A provider that silently ignores the logprobs request (returns no
    logprobs on the choice) must not crash -- just report None."""
    monkeypatch.setattr(litellm, "completion", lambda **kw: fake_response("A"))
    provider = ProviderConfig(provider="ollama", model="ollama/test-model")
    cfg = make_config(tmp_path, [provider])

    resp = chat("pick one", config=cfg, logprobs=True)
    assert resp.logprobs is None
