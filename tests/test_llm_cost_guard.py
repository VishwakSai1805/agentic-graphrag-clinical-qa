"""The §0.6 free-tier cost guard and the provider/model cache namespace
(ADR-047, src/llm/client.py). No network: litellm.completion is monkeypatched.
"""

from __future__ import annotations

import json

import litellm
import pytest

from src.config import AppConfig, LLMConfig, ProviderConfig
from src.llm.client import LLMCostGuardError, _cache_dir_for, _model_slug, chat
from tests.test_llm_cache import fake_response

GEMINI = ProviderConfig(
    provider="gemini", model="gemini/gemini-3.8-flash", api_key_env="GEMINI_API_KEY"
)
OLLAMA = ProviderConfig(provider="ollama", model="ollama/qwen2.5:7b-instruct")
PAID = ProviderConfig(provider="openai", model="openai/gpt-4o", api_key_env="OPENAI_API_KEY")


def make_config(tmp_path, chain, **llm_kwargs):
    return AppConfig(
        seed=42,
        cache_dir=tmp_path / "cache",
        llm=LLMConfig(fallback_chain=chain, max_retries=0, backoff_base_s=0.0, **llm_kwargs),
    )


def test_guard_is_off_by_default_so_existing_configs_are_unaffected(monkeypatch, tmp_path):
    monkeypatch.setattr(litellm, "completion", lambda **kw: fake_response("A"))
    cfg = make_config(tmp_path, [OLLAMA])
    assert chat("q", config=cfg).text == "A"


def test_guard_hard_fails_on_a_model_outside_the_allowlist(monkeypatch, tmp_path):
    called = {"n": 0}

    def boom(**kw):  # pragma: no cover - must never run
        called["n"] += 1
        return fake_response("should not happen")

    monkeypatch.setattr(litellm, "completion", boom)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = make_config(
        tmp_path,
        [PAID],
        free_tier_only=True,
        free_tier_allowed=["gemini:gemini/gemini-3.8-flash"],
    )
    with pytest.raises(LLMCostGuardError, match="not in"):
        chat("q", config=cfg)
    assert called["n"] == 0, "the guard must raise BEFORE any provider call"


def test_guard_does_not_fall_through_to_a_billable_provider(monkeypatch, tmp_path):
    """The whole point: a disallowed provider is a hard stop, never a skip.

    If the guard merely `continue`d, this chain would quietly reach the paid
    provider the moment Gemini rate-limited -- mid-run, unattended.
    """
    monkeypatch.setattr(
        litellm, "completion", lambda **kw: (_ for _ in ()).throw(litellm.RateLimitError(
            "429", llm_provider="gemini", model="gemini/gemini-3.8-flash"
        ))
    )
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = make_config(
        tmp_path,
        [GEMINI, PAID],
        free_tier_only=True,
        free_tier_allowed=["gemini:gemini/gemini-3.8-flash"],
    )
    with pytest.raises(LLMCostGuardError, match="openai:openai/gpt-4o"):
        chat("q", config=cfg)


def test_guard_allows_an_allowlisted_model(monkeypatch, tmp_path):
    monkeypatch.setattr(litellm, "completion", lambda **kw: fake_response("B"))
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    cfg = make_config(
        tmp_path,
        [GEMINI],
        free_tier_only=True,
        free_tier_allowed=["gemini:gemini/gemini-3.8-flash"],
    )
    assert chat("q", config=cfg).text == "B"


def test_guard_fires_before_the_cache_read(monkeypatch, tmp_path):
    """A misconfigured chain must surface immediately, not hide behind cache
    hits until the first miss."""
    monkeypatch.setattr(litellm, "completion", lambda **kw: fake_response("cached me"))
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    allowed = dict(free_tier_only=True, free_tier_allowed=["gemini:gemini/gemini-3.8-flash"])
    cfg = make_config(tmp_path, [GEMINI], **allowed)
    assert chat("q", config=cfg).cached is False
    assert chat("q", config=cfg).cached is True  # warm

    tightened = make_config(tmp_path, [GEMINI], free_tier_only=True, free_tier_allowed=[])
    with pytest.raises(LLMCostGuardError):
        chat("q", config=tightened)


# --------------------------------------------------------------------------
# cache namespace
# --------------------------------------------------------------------------


def test_namespace_off_by_default_keeps_the_existing_cache_layout(tmp_path):
    cfg = make_config(tmp_path, [OLLAMA])
    assert _cache_dir_for(cfg, OLLAMA) == cfg.cache_dir


def test_namespace_separates_providers_on_disk(tmp_path):
    cfg = make_config(tmp_path, [OLLAMA], cache_namespace_by_provider_model=True)
    gem = _cache_dir_for(cfg, GEMINI)
    olm = _cache_dir_for(cfg, OLLAMA)
    assert gem != olm
    assert gem.name == "gemini__gemini-gemini-3.8-flash"
    assert gem.parent == cfg.cache_dir


def test_model_slug_is_filesystem_safe():
    assert _model_slug("gemini/gemini-3.8-flash") == "gemini-gemini-3.8-flash"
    assert "/" not in _model_slug("ollama/qwen2.5:7b-instruct")


def test_gemini_replies_land_outside_the_ollama_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(litellm, "completion", lambda **kw: fake_response("gem"))
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    cfg = make_config(
        tmp_path,
        [GEMINI],
        cache_namespace_by_provider_model=True,
        free_tier_only=True,
        free_tier_allowed=["gemini:gemini/gemini-3.8-flash"],
    )
    chat("shared prompt", config=cfg)
    written = list((cfg.cache_dir / "gemini__gemini-gemini-3.8-flash").rglob("*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text())["text"] == "gem"
    # nothing landed at the un-namespaced root
    assert not list(cfg.cache_dir.glob("*/*.json"))
