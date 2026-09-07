"""Unit tests for scripts/raw_logprob_baseline.py's pure logic: extracting a
per-option raw probability vector from a chat() response's logprobs. LLM is
always mocked -- no provider, no GPU, no real Ollama call."""

from __future__ import annotations

import math

import pytest

from scripts.raw_logprob_baseline import _raw_probs_for_question
from src.kg.dataset_questions import Question
from src.llm.client import ChatResponse, StatsTracker, Usage


def _question(qid="train_00001"):
    return Question(
        question_id=qid,
        dataset="medqa",
        split="train",
        question="A 45 year old man presents with chest pain. Best next step?",
        options={"A": "aspirin", "B": "ibuprofen", "C": "morphine", "D": "observation"},
    )


def _fake_chat_response(letter_logprobs: dict[str, float]) -> ChatResponse:
    top = [{"token": tok, "logprob": lp} for tok, lp in letter_logprobs.items()]
    return ChatResponse(
        text=max(letter_logprobs, key=letter_logprobs.get),
        model="test-model",
        usage=Usage(prompt_tokens=10, completion_tokens=1, total_tokens=11, cost_usd=0.0),
        cached=False,
        prompt_hash="deadbeef",
        latency_ms=1.0,
        logprobs=[{"token": "B", "logprob": -0.01, "top_logprobs": top}],
    )


def test_raw_probs_sum_to_one(monkeypatch):
    import scripts.raw_logprob_baseline as mod

    monkeypatch.setattr(
        mod,
        "chat",
        lambda *a, **kw: _fake_chat_response({"A": -5.0, "B": -0.01, "C": -6.0, "D": -7.0}),
    )
    q = _question()
    result = _raw_probs_for_question(q, q.options, "evidence text", 42, StatsTracker())
    assert sum(result["raw_probs"].values()) == pytest.approx(1.0)


def test_raw_probs_highest_logprob_option_gets_highest_probability(monkeypatch):
    import scripts.raw_logprob_baseline as mod

    monkeypatch.setattr(
        mod,
        "chat",
        lambda *a, **kw: _fake_chat_response({"A": -5.0, "B": -0.01, "C": -6.0, "D": -7.0}),
    )
    q = _question()
    result = _raw_probs_for_question(q, q.options, "evidence text", 42, StatsTracker())
    assert max(result["raw_probs"], key=result["raw_probs"].get) == "B"
    assert result["n_letters_matched_in_top_logprobs"] == 4


def test_raw_probs_missing_letter_gets_a_sentinel_not_a_crash(monkeypatch):
    """top_logprobs only returned 2 of the 4 valid letters -- the other two
    must still get a (low) probability, not be dropped or raise."""
    import scripts.raw_logprob_baseline as mod

    monkeypatch.setattr(
        mod, "chat", lambda *a, **kw: _fake_chat_response({"B": -0.01, "E": -10.0})
    )
    q = _question()
    result = _raw_probs_for_question(q, q.options, "evidence text", 42, StatsTracker())
    assert set(result["raw_probs"]) == {"A", "B", "C", "D"}
    assert sum(result["raw_probs"].values()) == pytest.approx(1.0)
    assert result["n_letters_matched_in_top_logprobs"] == 1  # only "B" matched
    # unmatched letters (A, C, D) got the sentinel and so are all equal
    assert result["raw_probs"]["A"] == pytest.approx(result["raw_probs"]["C"])
    assert result["raw_probs"]["B"] > result["raw_probs"]["A"]


def test_raw_probs_no_logprobs_at_all_falls_back_to_uniform_sentinel(monkeypatch):
    """A provider that returns no logprobs whatsoever (resp.logprobs is None)
    must not crash -- every letter gets the same fallback and a uniform
    distribution results."""
    import scripts.raw_logprob_baseline as mod

    def fake_chat(*a, **kw):
        return ChatResponse(
            text="B",
            model="test-model",
            usage=Usage(prompt_tokens=10, completion_tokens=1, total_tokens=11, cost_usd=0.0),
            cached=False,
            prompt_hash="deadbeef",
            latency_ms=1.0,
            logprobs=None,
        )

    monkeypatch.setattr(mod, "chat", fake_chat)
    q = _question()
    result = _raw_probs_for_question(q, q.options, "evidence text", 42, StatsTracker())
    assert result["n_letters_matched_in_top_logprobs"] == 0
    probs = list(result["raw_probs"].values())
    assert all(p == pytest.approx(probs[0]) for p in probs)


def test_raw_probs_case_insensitive_letter_matching(monkeypatch):
    import scripts.raw_logprob_baseline as mod

    monkeypatch.setattr(
        mod, "chat", lambda *a, **kw: _fake_chat_response({"b": -0.01, "a": -5.0})
    )
    q = _question()
    result = _raw_probs_for_question(q, q.options, "evidence text", 42, StatsTracker())
    assert result["n_letters_matched_in_top_logprobs"] == 2


def test_raw_probs_all_finite(monkeypatch):
    import scripts.raw_logprob_baseline as mod

    monkeypatch.setattr(
        mod, "chat", lambda *a, **kw: _fake_chat_response({"A": -0.001, "B": -50.0})
    )
    q = _question()
    result = _raw_probs_for_question(q, q.options, "evidence text", 42, StatsTracker())
    assert all(math.isfinite(p) for p in result["raw_probs"].values())


def test_raw_probs_passes_tracker_through_to_chat(monkeypatch):
    """The StatsTracker passed in must reach chat() so llm_calls_total is
    accurate -- regression test for a real bug caught during development."""
    import scripts.raw_logprob_baseline as mod

    captured = {}

    def fake_chat(*a, **kw):
        captured["tracker"] = kw.get("tracker")
        return _fake_chat_response({"A": -0.01, "B": -5.0})

    monkeypatch.setattr(mod, "chat", fake_chat)
    tracker = StatsTracker()
    q = _question()
    _raw_probs_for_question(q, q.options, "evidence text", 42, tracker)
    assert captured["tracker"] is tracker
