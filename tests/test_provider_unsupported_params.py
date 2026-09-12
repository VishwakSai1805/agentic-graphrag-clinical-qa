"""Per-provider param exclusions (ADR-047, src/llm/client.py).

Gemini has no `seed`; Ollama does. The drop must be scoped to the provider
that declares it -- never a global `litellm.drop_params`, which would discard
unsupported params for every provider and could mask a real bug in a run whose
determinism matters.
"""

from __future__ import annotations

import litellm
import pytest

from src.config import AppConfig, LLMConfig, ProviderConfig
from src.llm.client import _supported_params, chat
from tests.test_llm_cache import fake_response

GEMINI = ProviderConfig(
    provider="gemini",
    model="gemini/gemini-3.8-flash",
    api_key_env="GEMINI_API_KEY",
    unsupported_params=["seed"],
)
OLLAMA = ProviderConfig(provider="ollama", model="ollama/qwen2.5:7b-instruct")


def make_config(tmp_path, chain, **llm_kwargs):
    return AppConfig(
        seed=42,
        cache_dir=tmp_path / "cache",
        llm=LLMConfig(fallback_chain=chain, max_retries=0, backoff_base_s=0.0, **llm_kwargs),
    )


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        return fake_response("ok")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    return seen


def test_seed_is_omitted_for_a_provider_that_declares_it_unsupported(
    captured, tmp_path, monkeypatch
):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    cfg = make_config(tmp_path, [GEMINI])
    chat("q", config=cfg, seed=42)
    assert "seed" not in captured, "Gemini rejects `seed` (UnsupportedParamsError)"
    assert captured["model"] == "gemini/gemini-3.8-flash"


def test_seed_is_still_sent_to_ollama(captured, tmp_path):
    cfg = make_config(tmp_path, [OLLAMA])
    chat("q", config=cfg, seed=42)
    assert captured["seed"] == 42, "the drop must not leak to other providers"


def test_temperature_zero_survives_the_drop(captured, tmp_path, monkeypatch):
    """`seed` goes, `temperature` stays -- it is the only determinism lever the
    Gemini API leaves us."""
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    cfg = make_config(tmp_path, [GEMINI], temperature=0.0)
    chat("q", config=cfg)
    assert captured["temperature"] == 0.0


def test_dropped_params_are_excluded_from_the_cache_key(tmp_path):
    params = {"temperature": 0.0, "seed": 42, "max_tokens": 64}
    assert _supported_params(GEMINI, params) == {"temperature": 0.0, "max_tokens": 64}


def test_a_provider_with_no_exclusions_keeps_every_param(tmp_path):
    """No committed cache entry may be invalidated by this feature."""
    params = {"temperature": 0.0, "seed": 42, "max_tokens": 64}
    assert _supported_params(OLLAMA, params) is params


def test_no_global_drop_params_flag_is_set():
    """A module-level `litellm.drop_params = True` would make every provider
    silently tolerant of bad params. Importing the client must not set it."""
    import src.llm.client  # noqa: F401

    assert getattr(litellm, "drop_params", False) is False
