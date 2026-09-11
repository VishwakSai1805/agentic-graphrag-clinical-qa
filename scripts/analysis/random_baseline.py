"""Chance-level reference row for every ranking table (`docs/04_EVALUATION.md`
§5 / §5.0a, `paper/tables/results_grid.*`, ADR-040).

Analytic expectations for a uniformly random ranking of the k=4 MedQA
options, one of which is gold:

    P@1 = 1/k,   P@2 = 2/k,   MRR = (1/k) * sum_{r=1..k} 1/r = H_k / k
    accuracy of a uniform random single guess = 1/k

For k=4: 0.25, 0.5, 25/48. They are expectations, not measurements -- a
seeded Monte-Carlo simulation checks the arithmetic, and `scripts/report.py`
reads the analytic values from this script's JSON so the row is never typed
by hand.

**Post-hoc comparison vs chance (NOT pre-registered).** Not in the m=35
family, not in the unplanned V0/V2 family. It exists only because
`docs/04` §5.0a states that the ranking variants rank "barely above chance",
and CLAUDE.md §2 forbids a comparison claim without a bootstrap. Chance is an
exact expectation, not a sample, so this is a one-sample test: the
per-question excess `metric_q - chance` is resampled
(`paired_bootstrap_ci` / `paired_bootstrap_p`, 10k, seed 42) and Holm-
corrected within this display set only.

    uv run python -m scripts.analysis.random_baseline
"""

from __future__ import annotations

import json
import random
import sys
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.metrics import (  # noqa: E402
    holm_bonferroni,
    paired_bootstrap_ci,
    paired_bootstrap_p,
)

OUT_PATH = REPO_ROOT / "results" / "phase7_grid" / "random_baseline.json"
K = 4
N_RESAMPLES = 10_000
SEED = 42
ALPHA = 0.05
MC_TRIALS = 200_000

RANKING_RUNS = {
    "V5": REPO_ROOT / "results" / "medqa_V5_41f7e14-dirty_20260909T103452Z",
    "V4-conflict": REPO_ROOT / "results" / "medqa_V4-conflict_41f7e14-dirty_20260909T220306Z",
    "post-hoc-binary": REPO_ROOT / "results" / "phase7_posthoc_binary_audit",
}
METRICS = ("p_at_1", "p_at_2", "mrr")


def analytic(k: int) -> dict[str, float]:
    mrr = sum(Fraction(1, r) for r in range(1, k + 1)) / k
    return {
        "accuracy": float(Fraction(1, k)),
        "p_at_1": float(Fraction(1, k)),
        "p_at_2": float(Fraction(min(2, k), k)),
        "mrr": float(mrr),
        "mrr_exact_fraction": f"{mrr.numerator}/{mrr.denominator}",
    }


def monte_carlo(k: int, trials: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    p1 = p2 = rr = 0.0
    for _ in range(trials):
        ranked = list(range(k))
        rng.shuffle(ranked)
        pos = ranked.index(0)  # option 0 is gold, by symmetry
        p1 += pos < 1
        p2 += pos < 2
        rr += 1.0 / (pos + 1)
    return {"p_at_1": p1 / trials, "p_at_2": p2 / trials, "mrr": rr / trials}


def _per_question(row: dict, metric: str) -> float:
    ranked, gold = row["ranked"], row["gold_option"]
    if metric == "p_at_1":
        return 1.0 if gold in ranked[:1] else 0.0
    if metric == "p_at_2":
        return 1.0 if gold in ranked[:2] else 0.0
    return (1.0 / (ranked.index(gold) + 1)) if gold in ranked else 0.0


def main() -> None:
    chance = analytic(K)
    mc = monte_carlo(K, MC_TRIALS, SEED)
    for m in METRICS:
        if abs(mc[m] - chance[m]) > 0.01:
            raise SystemExit(f"[random_baseline] Monte-Carlo {m}={mc[m]} vs analytic {chance[m]}")

    comparisons: dict[str, dict] = {}
    p_values: dict[str, float] = {}
    for name, run_dir in RANKING_RUNS.items():
        lines = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        bad = [r["question_id"] for r in rows if len(r["ranked"]) != K]
        if bad:
            raise SystemExit(
                f"[random_baseline] {name}: {len(bad)} rows are not full {K}-option rankings "
                "-- the chance values do not apply to them"
            )
        for m in METRICS:
            excess = [_per_question(r, m) - chance[m] for r in rows]
            mean_excess, lo, hi = paired_bootstrap_ci(
                excess, n_resamples=N_RESAMPLES, seed=SEED, alpha=ALPHA
            )
            p = paired_bootstrap_p(excess, n_resamples=N_RESAMPLES, seed=SEED)
            key = f"{name}:{m}"
            comparisons[key] = {
                "variant": name,
                "metric": m,
                "n": len(rows),
                "observed": chance[m] + mean_excess,
                "chance": chance[m],
                "excess_over_chance": mean_excess,
                "ci_lo": lo,
                "ci_hi": hi,
                "p_value": p,
            }
            p_values[key] = p

    holm = holm_bonferroni(p_values, alpha=ALPHA)
    for key, h in holm.items():
        c = comparisons[key]
        c["holm_rank"] = h["rank"]
        c["holm_reject"] = h["reject"]
        if not h["reject"]:
            c["verdict"] = "not distinguishable from chance (Holm, this display set)"
        elif c["excess_over_chance"] > 0:
            c["verdict"] = "above chance (Holm, this display set)"
        else:
            c["verdict"] = "BELOW chance (Holm, this display set)"

    out = {
        "k_options": K,
        "analytic": chance,
        "monte_carlo_check": {"trials": MC_TRIALS, "seed": SEED, **mc},
        "vs_chance_post_hoc": {
            "status": (
                "post-hoc descriptive display: NOT pre-registered, NOT in the m=35 "
                "exploratory family or the unplanned V0/V2 family; one-sample bootstrap of "
                "the per-question excess over the exact chance expectation, Holm-corrected "
                "within this display set only"
            ),
            "n_resamples": N_RESAMPLES,
            "seed": SEED,
            "alpha": ALPHA,
            "display_set_size": len(p_values),
            "comparisons": comparisons,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[random_baseline] wrote {OUT_PATH}")
    print(json.dumps({k: v for k, v in out.items() if k != "vs_chance_post_hoc"}, indent=2))
    for key, c in comparisons.items():
        print(
            f"  {key:26s} obs={c['observed']:.3f} chance={c['chance']:.3f} "
            f"excess={c['excess_over_chance']:+.3f} [{c['ci_lo']:+.3f},{c['ci_hi']:+.3f}] "
            f"p={c['p_value']:.4f} -> {c['verdict']}"
        )


if __name__ == "__main__":
    main()
