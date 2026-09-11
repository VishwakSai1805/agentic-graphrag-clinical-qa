"""scripts/analysis/posthoc_boundary_family.py -- the post-hoc tier-boundary
family (ADR-046). Synthetic runs only; no real results are read."""

from __future__ import annotations

import json

import pytest

from scripts.analysis import posthoc_boundary_family as fam


def test_family_size_is_fixed_at_seven_with_no_v0_halluc_cell():
    assert sum(len(m) for *_, m in fam.FAMILY_SPEC) == fam.FAMILY_M == 7
    for _, a, b, metrics in fam.FAMILY_SPEC:
        if "V0" in (a, b):
            assert "halluc_rate" not in metrics  # V0 is ungrounded by construction


@pytest.mark.parametrize(
    "res, metric, expect",
    [
        ({"holm_reject": True, "ci_lo": 0.1, "ci_hi": 0.2}, "accuracy", "A significantly better"),
        ({"holm_reject": True, "ci_lo": -0.2, "ci_hi": -0.1}, "accuracy", "B significantly better"),
        # A's halluc-rate is significantly HIGHER -> B is the better side
        ({"holm_reject": True, "ci_lo": 0.1, "ci_hi": 0.2}, "halluc_rate", "B significantly"),
        ({"holm_reject": False, "ci_lo": 0.01, "ci_hi": 0.2}, "accuracy", "not significant after"),
        ({"holm_reject": False, "ci_lo": -0.01, "ci_hi": 0.2}, "accuracy", "tie"),
    ],
)
def test_verdict_is_decided_by_holm_and_respects_metric_direction(res, metric, expect):
    assert fam._verdict(res, "A", "B", metric).startswith(expect)


def _write(run_dir, preds, per_claim=None):
    run_dir.mkdir(parents=True)
    (run_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(p) for p in preds) + "\n", encoding="utf-8"
    )
    metrics = {"hallucination": {"per_claim": per_claim}} if per_claim is not None else {}
    (run_dir / "metrics.json").write_text(json.dumps({"metrics": metrics}), encoding="utf-8")


def test_run_family_tests_all_seven_cells_and_labels_itself_post_hoc(tmp_path):
    n = 40
    gold = ["ABCD"[i % 4] for i in range(n)]

    def preds(n_correct):
        return [
            {"question_id": f"q{i}", "gold_option": g, "pred_option": g if i < n_correct else "X"}
            for i, g in enumerate(gold)
        ]

    def claims(n_entailed):
        return [
            {"question_id": f"q{i}", "predicted_option": True, "entailed": i < n_entailed}
            for i in range(n)
        ]

    dirs = {name: tmp_path / name for name in ("V0", "V3", "V4-conflict", "B3")}
    _write(dirs["V0"], preds(38))
    _write(dirs["V3"], preds(36), claims(10))
    _write(dirs["V4-conflict"], preds(10), claims(20))
    _write(dirs["B3"], preds(35))

    out = fam.run_family(dirs, tmp_path / "out.json")
    assert out["family_size"] == 7 and len(out["results"]) == 7
    assert out["family"].startswith("POST-HOC")
    x1 = out["results"]["X1:accuracy"]
    assert (x1["point_a"], x1["point_b"]) == (0.9, 0.25)
    assert x1["holm_reject"] and x1["verdict"].startswith("V3 significantly better")
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8")) == out
