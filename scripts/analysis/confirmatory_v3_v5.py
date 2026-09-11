"""The single pre-registered confirmatory test (docs/04_EVALUATION.md §1.0):
V3 vs V5, accuracy, MedQA test n=500 seed 42, paired bootstrap 10k resamples
seed 42, two-sided, alpha=0.05. One test, one metric.

    uv run python -m scripts.analysis.confirmatory_v3_v5
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.metrics import paired_bootstrap_ci, paired_bootstrap_p  # noqa: E402

V3_RUN_DIR = REPO_ROOT / "results" / "medqa_V3_41f7e14-dirty_20260908T230633Z"
V5_RUN_DIR = REPO_ROOT / "results" / "medqa_V5_41f7e14-dirty_20260909T103452Z"
OUT_PATH = REPO_ROOT / "results" / "phase7_grid" / "confirmatory_v3_v5.json"

N_RESAMPLES = 10_000
SEED = 42
ALPHA = 0.05


def _load_predictions(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "predictions.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return {r["question_id"]: r for r in rows}


def main() -> None:
    v3 = _load_predictions(V3_RUN_DIR)
    v5 = _load_predictions(V5_RUN_DIR)
    common = sorted(set(v3) & set(v5))
    if len(common) != len(v3) or len(common) != len(v5):
        raise SystemExit(
            f"[confirmatory_v3_v5] question_id sets differ: V3 n={len(v3)}, "
            f"V5 n={len(v5)}, common={len(common)} -- refusing to pair a "
            "mismatched set"
        )

    diffs = []
    for qid in common:
        a = v3[qid]
        b = v5[qid]
        gold = a["gold_option"]
        assert gold == b["gold_option"], f"gold mismatch on {qid}"
        acc_v3 = 1.0 if a["pred_option"] == gold else 0.0
        acc_v5 = 1.0 if b["pred_option"] == gold else 0.0
        diffs.append(acc_v3 - acc_v5)

    mean_diff, ci_lo, ci_hi = paired_bootstrap_ci(
        diffs, n_resamples=N_RESAMPLES, seed=SEED, alpha=ALPHA
    )
    p = paired_bootstrap_p(diffs, n_resamples=N_RESAMPLES, seed=SEED)

    n_correct_v3 = sum(1 for qid in common if v3[qid]["pred_option"] == v3[qid]["gold_option"])
    n_correct_v5 = sum(1 for qid in common if v5[qid]["pred_option"] == v5[qid]["gold_option"])
    point_v3 = n_correct_v3 / len(common)
    point_v5 = n_correct_v5 / len(common)

    if ci_lo > 0:
        verdict = "V3 significantly > V5"
    elif ci_hi < 0:
        verdict = "V5 significantly > V3"
    else:
        verdict = (
            "tie (95% CI includes 0) -- reported as a tie, not evidence either system is better"
        )

    out = {
        "test": "confirmatory (docs/04_EVALUATION.md §1.0, pre-registered 2026-09-07)",
        "comparison": "V3 vs V5",
        "metric": "accuracy",
        "dataset": "medqa",
        "split": "test",
        "n": len(common),
        "seed": SEED,
        "n_resamples": N_RESAMPLES,
        "alpha": ALPHA,
        "two_sided": True,
        "accuracy_v3": point_v3,
        "accuracy_v5": point_v5,
        "point_diff_v3_minus_v5": point_v3 - point_v5,
        "mean_diff_v3_minus_v5": mean_diff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p,
        "significant": p <= ALPHA,
        "verdict": verdict,
        "v3_run_id": v3[common[0]]["question_id"] and str(V3_RUN_DIR.name),
        "v5_run_id": str(V5_RUN_DIR.name),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[confirmatory_v3_v5] wrote {OUT_PATH}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
