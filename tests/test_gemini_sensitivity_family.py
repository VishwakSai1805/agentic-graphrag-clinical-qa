"""scripts/analysis/gemini_sensitivity_family.py -- the §0.6 check's family
composition and guards (ADR-047). Synthetic runs only; no real results read.
"""

from __future__ import annotations

import json

import pytest

from scripts.analysis import gemini_sensitivity_family as fam


def test_family_size_is_fixed_at_eight():
    assert sum(len(m) for *_, m in fam.FAMILY_SPEC) == fam.FAMILY_M == 8


def test_family_carries_no_halluc_ranking_or_ece_cell():
    """B3 has no claim-level NLI data, a V0 pair is ungrounded by construction,
    and V0/V3/B3 are single_prediction -- so only acc and macro-F1 are testable."""
    for _, _, _, metrics in fam.FAMILY_SPEC:
        assert set(metrics) == {"accuracy", "macro_f1"}


def test_confirmatory_is_v3_vs_v5_accuracy_and_sits_outside_the_family():
    assert fam.CONFIRMATORY == ("V3", "V5", "accuracy")
    assert not any(p == "confirmatory" for p, *_ in fam.FAMILY_SPEC)


def test_the_four_cells_are_v0_v3_b3_v5_only():
    cells = {s for _, a, b, _ in fam.FAMILY_SPEC for s in (a, b)}
    assert cells == {"V0", "V3", "B3", "V5"}
    assert "V1" not in cells and "V2" not in cells and "V4-conflict" not in cells


def test_chance_row_is_the_analytic_four_option_expectation():
    assert fam.CHANCE_4WAY["p_at_1"] == 0.25
    assert fam.CHANCE_4WAY["p_at_2"] == 0.5
    assert fam.CHANCE_4WAY["mrr"] == pytest.approx(0.5208333, abs=1e-6)


def _write_run(run_dir, variant, preds, n, extra_metrics=None):
    run_dir.mkdir(parents=True)
    (run_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(p) for p in preds) + "\n", encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "variant": f"{variant}-gemini",
                "dataset": "medqa",
                "split": "test",
                "seed": 42,
                "metrics": {"n_evaluated": n, **(extra_metrics or {})},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "config.yaml").write_text(
        "llm:\n"
        "  api_version: v1beta\n"
        "  model_resolved_utc: '2026-09-12T00:00:00Z'\n"
        "  fallback_chain:\n"
        "  - model: gemini/gemini-3.8-flash\n",
        encoding="utf-8",
    )


def test_validate_rejects_a_run_without_the_gemini_variant_suffix(tmp_path):
    ids = [f"test_{i:05d}" for i in range(4)]
    preds = [{"question_id": q, "gold_option": "A", "pred_option": "A"} for q in ids]
    d = tmp_path / "bad"
    _write_run(d, "V3", preds, len(ids))
    # overwrite the variant with the unsuffixed name
    m = json.loads((d / "metrics.json").read_text())
    m["variant"] = "V3"
    (d / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(SystemExit, match="V3-gemini"):
        fam.validate_gemini_run(d, "V3", len(ids))


def test_validate_requires_a_run_date_in_the_config(tmp_path):
    ids = [f"test_{i:05d}" for i in range(4)]
    preds = [{"question_id": q, "gold_option": "A", "pred_option": "A"} for q in ids]
    d = tmp_path / "nodate"
    _write_run(d, "V3", preds, len(ids))
    (d / "config.yaml").write_text(
        "llm:\n  fallback_chain:\n  - model: gemini/gemini-3.8-flash\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="model_resolved_utc"):
        fam.validate_gemini_run(d, "V3", len(ids))


def test_run_family_refuses_a_run_that_misses_a_pinned_id(tmp_path, monkeypatch):
    ids = [f"test_{i:05d}" for i in range(8)]

    def preds_for(n_correct, drop=0):
        return [
            {"question_id": q, "gold_option": "ABCD"[i % 4],
             "pred_option": "ABCD"[i % 4] if i < n_correct else "X"}
            for i, q in enumerate(ids[: len(ids) - drop])
        ]

    dirs = {}
    for name, n_correct, drop in (("V0", 4, 0), ("V3", 6, 0), ("B3", 3, 0), ("V5", 2, 1)):
        d = tmp_path / name
        _write_run(d, name, preds_for(n_correct, drop), len(ids) - drop)
        dirs[name] = d

    monkeypatch.setattr(fam, "PAIRED_7B_RUNS", {k: f"missing_{k}" for k in dirs})
    with pytest.raises(SystemExit, match="missing 1 pinned ids"):
        fam.run_family(dirs, ids, tmp_path / "out.json")
