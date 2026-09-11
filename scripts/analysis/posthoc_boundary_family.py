"""POST-HOC family: the tier-boundary pairs no earlier family tested (ADR-046).

Identified on 2026-09-11 AFTER the Phase 7 grid's two-tier structure was
visible (`docs/04_EVALUATION.md` §5.0a, ADR-045): every tested pair inside
the retrieval tier {V0, V1, V2, V3, B3} ties, every tested pair inside the
agent tier {V4-conflict, V5, V5-refit} ties, and the step between them was
established only by cross-tier pairs against V5 / V5-refit. The boundary
step itself -- V3 -> V4-conflict, where the drop sits on the nested component
chain -- appears in neither the m=35 family nor the unplanned U1-U8 family.

    X1  V3 vs V4-conflict   acc, macro_f1, halluc_rate   the boundary step itself
    X2  V0 vs V4-conflict   acc, macro_f1                zero-shot vs the agent tier's best cell
    X3  V0 vs B3            acc, macro_f1                zero-shot vs the binary verifier

m_X = 7, fixed here before any p-value in this family was computed. No
halluc-rate cell for a V0 pair (V0 cites nothing: every claim is ungrounded
by construction, `docs/04` §1.0b); B3 has no claim-level NLI data anyway.

Because these pairs were chosen after their point estimates were visible,
they are POST-HOC: Holm-corrected WITHIN THIS FAMILY ONLY, never pooled with
the m=35 or the U1-U8 family, never used to reinterpret either, and never
cited as pre-registered support. They check that the central claim does not
rest on inference from adjacent pairs alone.

Same method as the other two families (helpers reused from
`scripts/analysis/exploratory_family.py`): paired bootstrap, 10k resamples,
seed 42, two-sided; accuracy by per-question difference, Macro-F1 /
halluc-rate by joint question resampling. V3 / V4-conflict / B3 are the
grid's pinned run dirs; the V0 run is passed by exact name and validated
exactly as the unplanned family validates it.

    uv run python -m scripts.analysis.posthoc_boundary_family --v0 medqa_V0_<sha>_<ts>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.exploratory_family import (  # noqa: E402
    ALPHA,
    N_RESAMPLES,
    SEED,
    _bootstrap_aggregate_diff,
    _halluc_rate_predicted,
    _load_per_claim,
    _load_predictions,
    _macro_f1_of,
    _rank_metric_test,
)
from scripts.analysis.exploratory_family import RUN_DIRS as GRID_RUN_DIRS  # noqa: E402
from scripts.analysis.unplanned_v0_v2_family import validate_run  # noqa: E402
from src.retrieval.metrics import holm_bonferroni  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
OUT_PATH = RESULTS_DIR / "phase7_grid" / "posthoc_boundary_family.json"

FAMILY_SPEC: list[tuple[str, str, str, list[str]]] = [
    ("X1", "V3", "V4-conflict", ["accuracy", "macro_f1", "halluc_rate"]),
    ("X2", "V0", "V4-conflict", ["accuracy", "macro_f1"]),
    ("X3", "V0", "B3", ["accuracy", "macro_f1"]),
]
FAMILY_M = 7
COMPARATORS = ("V3", "V4-conflict", "B3")


def _verdict(res: dict, side_a: str, side_b: str) -> str:
    """Family-level verdict: Holm decides "significant", the raw CI never does."""
    if res["holm_reject"]:
        winner = side_a if res["ci_lo"] > 0 else side_b
        return f"{winner} significantly better (Holm, post-hoc family)"
    if res["ci_lo"] > 0 or res["ci_hi"] < 0:
        return "not significant after Holm (unadjusted 95% CI excludes 0)"
    return "tie (95% CI includes 0)"


def run_family(run_dirs: dict[str, Path], out_path: Path) -> dict:
    preds = {name: _load_predictions(d) for name, d in run_dirs.items()}
    results: dict[str, dict] = {}
    p_values: dict[str, float] = {}
    for pair, side_a, side_b, metrics in FAMILY_SPEC:
        pa, pb = preds[side_a], preds[side_b]
        common = sorted(set(pa) & set(pb))
        if len(common) != len(pa) or len(common) != len(pb):
            raise SystemExit(
                f"[posthoc_boundary_family] {pair}: question_id mismatch "
                f"{side_a} n={len(pa)} vs {side_b} n={len(pb)}, common={len(common)}"
            )
        if any(pa[q]["gold_option"] != pb[q]["gold_option"] for q in common):
            raise SystemExit(f"[posthoc_boundary_family] {pair}: gold labels disagree")

        for metric in metrics:
            if metric == "accuracy":
                res = _rank_metric_test(pa, pb, common, "accuracy")
            elif metric == "macro_f1":
                stat_a = {q: (pa[q]["pred_option"], pa[q]["gold_option"]) for q in common}
                stat_b = {q: (pb[q]["pred_option"], pb[q]["gold_option"]) for q in common}
                res = _bootstrap_aggregate_diff(common, stat_a, stat_b, _macro_f1_of)
            elif metric == "halluc_rate":
                claims_a = _load_per_claim(run_dirs[side_a])
                claims_b = _load_per_claim(run_dirs[side_b])
                if claims_a is None or claims_b is None:
                    raise SystemExit(f"[posthoc_boundary_family] {pair}: no hallucination data")
                res = _bootstrap_aggregate_diff(
                    common,
                    {q: claims_a.get(q, []) for q in common},
                    {q: claims_b.get(q, []) for q in common},
                    _halluc_rate_predicted,
                )
            else:
                raise ValueError(metric)
            res.update(pair=pair, side_a=side_a, side_b=side_b, metric=metric, n=len(common))
            key = f"{pair}:{metric}"
            results[key] = res
            p_values[key] = res["p_value"]

    if len(p_values) != FAMILY_M:
        raise SystemExit(f"[posthoc_boundary_family] tested {len(p_values)}, expected {FAMILY_M}")
    for key, h in holm_bonferroni(p_values, alpha=ALPHA).items():
        r = results[key]
        r.update(holm_rank=h["rank"], holm_threshold=h["threshold"], holm_reject=h["reject"])
        r["verdict"] = _verdict(r, r["side_a"], r["side_b"])

    out = {
        "family": "POST-HOC -- tier-boundary pairs untested by any earlier family (ADR-046)",
        "status": (
            "Identified 2026-09-11 AFTER the grid's two-tier structure was visible "
            "(docs/04 §5.0a, ADR-045); the family (m=7) was fixed before any of its "
            "p-values was computed, but its pairs were chosen with the point estimates "
            "known. POST-HOC: Holm-corrected within this family only; never pooled with, "
            "or used to reinterpret, the pre-registered m=35 family or the unplanned "
            "U1-U8 family; never cited as pre-registered support."
        ),
        "family_size": FAMILY_M,
        "run_dirs": {k: v.name for k, v in sorted(run_dirs.items())},
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "alpha": ALPHA,
        "multiplicity_correction": (
            "Holm-Bonferroni within this family (src.retrieval.metrics.holm_bonferroni)"
        ),
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--v0", required=True, help="V0 run dir name under results/")
    args = ap.parse_args()

    run_dirs = {name: GRID_RUN_DIRS[name] for name in COMPARATORS}
    v0 = RESULTS_DIR / args.v0
    validate_run(v0, "V0")
    run_dirs["V0"] = v0

    out = run_family(run_dirs, OUT_PATH)
    print(f"[posthoc_boundary_family] wrote {OUT_PATH}")
    for key, r in sorted(out["results"].items(), key=lambda kv: kv[1]["holm_rank"]):
        print(
            f"  {key:16s} A={r['point_a']:.3f} B={r['point_b']:.3f} "
            f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p={r['p_value']:.4f} "
            f"rank={r['holm_rank']} reject={r['holm_reject']}  {r['verdict']}"
        )


if __name__ == "__main__":
    main()
