"""Descriptive, per-variant marginal bootstrap confidence intervals for the
paper's Table 1 (docs/04_EVALUATION.md §5.0b, ADR-051 in docs/05_DECISIONS.md).

These replace externally-computed Wilson intervals that broke CLAUDE.md §2's
traceability rule (every reported number must come from a committed
results/ artifact). They are DESCRIPTIVE ONLY: an uncertainty display around
one variant's own accuracy (and, for the three ranking variants, P@2/MRR),
never a comparison. They are not part of the §1.0 pre-registered family, the
unplanned exploratory family, or the post-hoc tier-boundary family, and must
never be Holm-corrected alongside them.

    uv run python -m scripts.analysis.marginal_cis
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.metrics import precision_at_k, reciprocal_rank_option  # noqa: E402
from src.retrieval.metrics import marginal_bootstrap_ci  # noqa: E402

N_RESAMPLES = 10_000
SEED = 42
ALPHA = 0.05

# Exact run dirs cited by paper/tables/results_grid.md -- hardcoded by full
# run_id, never globbed/"latest", so a stale or invalid run under the same
# variant letter (several exist for V2/V4/V5) can never be silently picked up.
RUN_IDS: dict[str, str] = {
    "V0": "medqa_V0_b24015a-dirty_20260911T080323Z",
    "V1": "medqa_V1_41f7e14_20260908T212655Z",
    "V2": "medqa_V2_b24015a-dirty_20260911T104513Z",
    "V3": "medqa_V3_41f7e14-dirty_20260908T230633Z",
    "B3": "medqa_B3_41f7e14-dirty_20260909T023658Z",
    "V4-conflict": "medqa_V4-conflict_41f7e14-dirty_20260909T220306Z",
    "V5": "medqa_V5_41f7e14-dirty_20260909T103452Z",
    "V5-refit": "medqa_V5-refit_b24015a_20260911T074318Z",
}

# Only these three emit a real multi-item ranking; the rest emit a 1-item
# list, where P@2/MRR are undefined (src/eval/harness.py line ~356,
# docs/04_EVALUATION.md §5.0a's note on exploratory-family pair b).
RANKING_VARIANTS = ("V4-conflict", "V5", "V5-refit")

OUT_JSON = REPO_ROOT / "results" / "phase7_marginal_cis.json"
OUT_MD = REPO_ROOT / "results" / "phase7_marginal_cis.md"


def _load_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "predictions.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _load_published_metrics(run_dir: Path) -> dict:
    path = run_dir / "metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["metrics"] if "metrics" in data else data


def _ci_block(point: float, lo: float, hi: float, n: int) -> dict:
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "n": n}


def main() -> None:
    variants: dict[str, dict] = {}
    md_rows: list[str] = []

    for label, run_id in RUN_IDS.items():
        run_dir = REPO_ROOT / "results" / run_id
        rows = _load_rows(run_dir)
        n = len(rows)
        published = _load_published_metrics(run_dir)

        correct = [1.0 if r["pred_option"] == r["gold_option"] else 0.0 for r in rows]
        acc_point, acc_lo, acc_hi = marginal_bootstrap_ci(
            correct, n_resamples=N_RESAMPLES, seed=SEED, alpha=ALPHA
        )
        if abs(acc_point - published["accuracy"]) > 1e-9:
            raise SystemExit(
                f"[marginal_cis] {label}: recomputed accuracy {acc_point} != "
                f"published {published['accuracy']} in {run_dir / 'metrics.json'} "
                "-- refusing to tabulate a mismatch"
            )

        entry = {
            "run_id": run_id,
            "n": n,
            "accuracy": _ci_block(acc_point, acc_lo, acc_hi, n),
        }
        md_rows.append(
            f"| {label} | accuracy | {acc_point:.3f} | {acc_lo:.3f} | {acc_hi:.3f} | {n} |"
        )

        if label in RANKING_VARIANTS:
            p2_vec = [precision_at_k(r["ranked"], r["gold_option"], 2) for r in rows]
            rr_vec = [reciprocal_rank_option(r["ranked"], r["gold_option"]) for r in rows]

            p2_point, p2_lo, p2_hi = marginal_bootstrap_ci(
                p2_vec, n_resamples=N_RESAMPLES, seed=SEED, alpha=ALPHA
            )
            mrr_point, mrr_lo, mrr_hi = marginal_bootstrap_ci(
                rr_vec, n_resamples=N_RESAMPLES, seed=SEED, alpha=ALPHA
            )

            if abs(p2_point - published["p_at_2"]) > 1e-9:
                raise SystemExit(
                    f"[marginal_cis] {label}: recomputed P@2 {p2_point} != "
                    f"published {published['p_at_2']} -- refusing to tabulate a mismatch"
                )
            if abs(mrr_point - published["mrr"]) > 1e-9:
                raise SystemExit(
                    f"[marginal_cis] {label}: recomputed MRR {mrr_point} != "
                    f"published {published['mrr']} -- refusing to tabulate a mismatch"
                )

            entry["p_at_2"] = _ci_block(p2_point, p2_lo, p2_hi, n)
            entry["mrr"] = _ci_block(mrr_point, mrr_lo, mrr_hi, n)
            md_rows.append(
                f"| {label} | p_at_2 | {p2_point:.3f} | {p2_lo:.3f} | {p2_hi:.3f} | {n} |"
            )
            md_rows.append(
                f"| {label} | mrr | {mrr_point:.3f} | {mrr_lo:.3f} | {mrr_hi:.3f} | {n} |"
            )

        variants[label] = entry

    out = {
        "kind": "descriptive_marginal_bootstrap_ci",
        "note": (
            "Descriptive only -- not part of any multiplicity-corrected family "
            "(docs/04_EVALUATION.md §1.0 pre-registered family, the unplanned "
            "exploratory family, or the post-hoc tier-boundary family). See "
            "docs/04_EVALUATION.md §5.0b and ADR-051 in docs/05_DECISIONS.md."
        ),
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "alpha": ALPHA,
        "variants": variants,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    header = (
        "# Phase 7 marginal (per-variant) bootstrap confidence intervals\n\n"
        "Descriptive only -- see docs/04_EVALUATION.md §5.0b and ADR-051. "
        "Generated by `scripts/analysis/marginal_cis.py`, do not edit by hand.\n\n"
        "| variant | metric | point | ci_lo | ci_hi | n |\n"
        "|---|---|---|---|---|---|\n"
    )
    md_text = header + "\n".join(md_rows) + "\n"
    OUT_MD.write_text(md_text, encoding="utf-8")

    print(f"[marginal_cis] wrote {OUT_JSON}")
    print(f"[marginal_cis] wrote {OUT_MD}")
    print()
    print(md_text)


if __name__ == "__main__":
    main()
