"""`--question-ids` pinning on src/eval/harness.py (ADR-047).

The §0.6 Gemini check pairs against the Phase 7 grid runs question by question,
which is only sound if the 200 questions ARE 200 of the grid's 500. A seeded
sample at a smaller limit is not -- see `test_a_smaller_seeded_limit_is_not_a_prefix`,
which pins the real behaviour that forced this flag to exist.
"""

from __future__ import annotations

import json
import types

import pytest

from src.eval.harness import run_eval

CONFIG = "configs/base.yaml"


def _questions(n=6):
    from src.kg.dataset_questions import Question

    return [
        Question(
            question_id=f"test_{i:05d}",
            dataset="medqa",
            split="test",
            question=f"Question {i}?",
            options={"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
            answer="alpha",
        )
        for i in range(n)
    ]


@pytest.fixture
def mocked(monkeypatch):
    from src.kg import dataset_questions

    monkeypatch.setattr(dataset_questions, "load_question_sample", lambda *a, **k: _questions())
    monkeypatch.setattr(
        "src.eval.baselines.chat",
        lambda prompt, **kw: types.SimpleNamespace(
            text='{"answer": "A", "rationale": "Alpha fits the stem."}'
        ),
    )


def _run(tmp_path, **kw):
    return run_eval(
        CONFIG,
        dataset="medqa",
        split="test",
        limit=None,
        variant="V0",
        nli_scorer=lambda premise, hypothesis: 0.95,
        results_dir=tmp_path,
        **kw,
    )


def _pred_ids(run_dir):
    return [
        json.loads(line)["question_id"]
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_pinned_ids_are_run_in_the_given_order(tmp_path, mocked):
    pinned = ["test_00004", "test_00001", "test_00003"]
    run_dir = _run(tmp_path, question_ids=pinned)
    assert _pred_ids(run_dir) == pinned


def test_limit_is_recorded_as_the_pinned_count(tmp_path, mocked):
    pinned = ["test_00000", "test_00002"]
    run_dir = _run(tmp_path, question_ids=pinned)
    meta = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert meta["limit"] == 2
    assert meta["metrics"]["n_evaluated"] == 2


def test_an_unknown_pinned_id_fails_loudly(tmp_path, mocked):
    with pytest.raises(ValueError, match="absent from medqa/test"):
        _run(tmp_path, question_ids=["test_00000", "test_99999"])


def test_question_ids_and_exclude_limit_are_mutually_exclusive(tmp_path, mocked):
    with pytest.raises(ValueError, match="mutually exclusive"):
        _run(tmp_path, question_ids=["test_00000"], exclude_limit=2)


def test_variant_suffix_is_recorded_but_never_dispatched_on(tmp_path, mocked):
    """A `-gemini` cell must run V0's code and record `V0-gemini`, so
    scripts/report.py's canonical_variant can never map it to the V0 grid row."""
    run_dir = _run(tmp_path, question_ids=["test_00000"], variant_suffix="-gemini")
    meta = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert meta["variant"] == "V0-gemini"
    assert "V0-gemini" in run_dir.name

    from scripts.report import canonical_variant

    assert canonical_variant("V0-gemini") is None
    assert canonical_variant("V0") == "V0"


def test_llm_model_and_api_version_land_in_the_run_config(tmp_path, mocked):
    run_dir = _run(
        tmp_path,
        question_ids=["test_00000"],
        llm_model="gemini/gemini-3.8-flash",
        llm_api_version="v1beta",
    )
    import yaml

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["llm"]["fallback_chain"][0]["model"] == "gemini/gemini-3.8-flash"
    assert cfg["llm"]["api_version"] == "v1beta"
    assert cfg["llm"]["model_resolved_utc"], "the run date is part of the provenance"


def test_a_smaller_seeded_limit_is_not_a_prefix_of_a_larger_one():
    """The real behaviour that forced --question-ids to exist (ADR-047).

    Pure RNG, no dataset file: `random.sample` at a different k draws a
    different sequence, so `--limit 200` cannot be assumed to be the first 200
    of `--limit 500`.
    """
    import random

    n = 1273  # the real MedQA test split size
    small = sorted(random.Random(42).sample(range(n), 200))
    large = sorted(random.Random(42).sample(range(n), 500))
    assert small != large[:200]
    assert not set(small).issubset(set(large))
