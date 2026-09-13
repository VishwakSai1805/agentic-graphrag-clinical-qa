"""The ADR-048 defect and the guard that makes it impossible to write.

The failure being guarded against is the most dangerous one in this system,
because it does not look like a failure: five "Gemini" cells ran to completion,
exited 0, wrote well-formed metrics.json files, and recorded a config.yaml
naming `gemini/gemini-3.8-flash` -- while `ollama/qwen2.5:7b-instruct` had
answered every question from cache. Nothing downstream could tell those runs
from real results.

`test_reproduces_the_defect_*` reproduces the original mechanism end to end, so
the regression is pinned by a demonstration rather than by an assertion about
one field.
"""

from __future__ import annotations

import litellm
import pytest

from src.config import AppConfig, LLMConfig, ProviderConfig
from src.llm.client import (
    CallStats,
    LLMProvenanceError,
    StatsTracker,
    assert_generator_provenance,
    chat,
    set_active_config,
)
from tests.test_llm_cache import fake_response

QWEN = ProviderConfig(provider="ollama", model="ollama/qwen2.5:7b-instruct")
GEMINI = ProviderConfig(
    provider="gemini",
    model="gemini/gemini-3.8-flash",
    api_key_env="GEMINI_API_KEY",
    unsupported_params=["seed"],
)


def make_config(tmp_path, chain, **llm_kwargs):
    return AppConfig(
        seed=42,
        cache_dir=tmp_path / "cache",
        llm=LLMConfig(fallback_chain=chain, max_retries=0, backoff_base_s=0.0, **llm_kwargs),
    )


@pytest.fixture(autouse=True)
def _reset_active_config():
    """set_active_config is a module global -- never leak it between tests."""
    yield
    set_active_config(None)


# --------------------------------------------------------------------------
# the defect itself
# --------------------------------------------------------------------------


def test_reproduces_the_defect_a_gemini_run_served_from_the_qwen_cache(
    monkeypatch, tmp_path
):
    """Without set_active_config, chat() ignores the resolved config entirely."""
    monkeypatch.setattr(litellm, "completion", lambda **kw: fake_response("7B answer"))
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    qwen_cfg = make_config(tmp_path, [QWEN])
    gemini_cfg = make_config(tmp_path, [GEMINI])

    # a prior 7B run warms the shared cache
    chat("MedQA question", config=qwen_cfg)

    # the "Gemini" run: config resolved, but nothing installs it, and no feature
    # call site passes config= -- so chat() falls back to the module default.
    set_active_config(qwen_cfg)  # stands in for the hardcoded configs/base.yaml
    tracker = StatsTracker()
    resp = chat("MedQA question", tracker=tracker)

    assert resp.cached is True, "replayed the 7B answer"
    assert resp.model == "ollama/qwen2.5:7b-instruct"
    stats = tracker.summary()
    assert stats.cache_hits == stats.calls == 1

    # exactly the situation that produced five invalid run dirs -- now refused
    with pytest.raises(LLMProvenanceError, match="actually served the run"):
        assert_generator_provenance(gemini_cfg, stats, run_label="V3-gemini")


def test_set_active_config_is_the_fix(monkeypatch, tmp_path):
    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return fake_response("gemini answer")

    monkeypatch.setattr(litellm, "completion", capture)
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    qwen_cfg = make_config(tmp_path, [QWEN])
    gemini_cfg = make_config(tmp_path, [GEMINI], cache_namespace_by_provider_model=True)
    chat("MedQA question", config=qwen_cfg)  # warm the 7B cache

    set_active_config(gemini_cfg)
    tracker = StatsTracker()
    resp = chat("MedQA question", tracker=tracker)  # no config= -- as feature code calls it

    assert resp.cached is False, "must not replay the 7B entry"
    assert resp.model == "gemini/gemini-3.8-flash"
    assert seen["model"] == "gemini/gemini-3.8-flash"
    assert_generator_provenance(gemini_cfg, tracker.summary(), run_label="V3-gemini")


# --------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------


def _stats(models, calls, hits):
    return CallStats(calls=calls, cache_hits=hits, models_used=models)


def test_guard_rejects_a_model_the_config_does_not_name(tmp_path):
    cfg = make_config(tmp_path, [GEMINI])
    stats = _stats({"ollama/qwen2.5:7b-instruct": 200}, 200, 200)
    with pytest.raises(LLMProvenanceError) as e:
        assert_generator_provenance(cfg, stats, run_label="V3-gemini")
    assert "gemini/gemini-3.8-flash" in str(e.value)
    assert "ollama/qwen2.5:7b-instruct" in str(e.value)


def test_guard_rejects_a_fully_cached_first_run_against_a_hosted_provider(tmp_path):
    """Second, independent check: once the config genuinely reaches chat(), the
    model-identity check reads the same field on both sides and cannot fail. A
    hosted first run must therefore also prove it made real calls."""
    cfg = make_config(tmp_path, [GEMINI])
    stats = _stats({"gemini/gemini-3.8-flash": 200}, 200, 200)
    with pytest.raises(LLMProvenanceError, match="from cache on a run not marked as a resume"):
        assert_generator_provenance(cfg, stats, run_label="V3-gemini")


def test_guard_allows_a_warm_cache_when_the_run_is_a_declared_resume(tmp_path):
    cfg = make_config(tmp_path, [GEMINI])
    stats = _stats({"gemini/gemini-3.8-flash": 200}, 200, 200)
    block = assert_generator_provenance(
        cfg, stats, run_label="V3-gemini", allow_warm_cache=True
    )
    assert block["allow_warm_cache"] is True
    assert block["cache_hit_rate"] == 1.0


def test_guard_does_not_penalise_a_local_provider_for_cache_hits(tmp_path):
    """The 7B grid replays from cache by design; only hosted runs are checked."""
    cfg = make_config(tmp_path, [QWEN])
    stats = _stats({"ollama/qwen2.5:7b-instruct": 500}, 500, 500)
    block = assert_generator_provenance(cfg, stats, run_label="V3")
    assert block["hosted_provider"] is False


def test_guard_passes_a_genuine_hosted_run_and_reports_provenance(tmp_path):
    cfg = make_config(tmp_path, [GEMINI])
    stats = _stats({"gemini/gemini-3.8-flash": 200}, 200, 3)
    block = assert_generator_provenance(cfg, stats, run_label="V3-gemini")
    assert block["configured_model"] == "gemini/gemini-3.8-flash"
    assert block["models_used"] == {"gemini/gemini-3.8-flash": 200}
    assert block["cache_hit_rate"] == pytest.approx(0.015)
    assert block["hosted_provider"] is True


def test_guard_is_a_noop_when_the_run_made_no_llm_calls(tmp_path):
    cfg = make_config(tmp_path, [GEMINI])
    block = assert_generator_provenance(cfg, CallStats(), run_label="V0-gemini")
    assert "nothing to verify" in block["note"]
