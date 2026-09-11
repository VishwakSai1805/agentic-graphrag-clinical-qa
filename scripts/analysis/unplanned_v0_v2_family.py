"""Unplanned exploratory family: the V0 / V2 grid rows (`docs/04_EVALUATION.md`
§1.0b, ADR-040) and V5-refit (ADR-043).

V0 (zero-shot, no retrieval) and V2 (`{dense, lexical}`, no KG) are
pre-specified grid rows (`docs/04` §0.3) that never ran; V5-refit is the
disclosed post-hoc correction of V5's scoring rule. None of their
comparisons was in the pre-registered m=35 family. U1-U5 were registered on
2026-09-10 and U6-U8 on 2026-09-11, each before any run of this family
existed. The family is Holm-corrected WITHIN ITSELF -- never merged into the
m=35 family (merging would move that family's thresholds after its outcomes
are known):

    U1  V0 vs V1        acc, macro_f1               does retrieval help at all (dense)
    U2  V0 vs V3        acc, macro_f1               ... (the tri-modal stack)
    U3  V0 vs V5        acc, macro_f1               does the full system beat the bare LLM
    U4  V2 vs V3        acc, macro_f1, halluc_rate  C1: KG's value over {dense, lexical}
    U5  V1 vs V2        acc, macro_f1, halluc_rate  lexical's value over dense
    U6  V5-refit vs V5  acc, macro_f1               does correcting V5's scoring rule change it
    U7  V5-refit vs V4-conflict  acc, macro_f1      ... vs the uncalibrated (s+ - s-) margin
    U8  V5-refit vs V3  acc, macro_f1               ... vs tri-modal retrieval alone

m_U = 18. Fixed contingencies (`docs/04` §1.0b): no P@K / MRR / ECE cell
anywhere (V0-V3 are all `single_prediction` -- the pair-b lesson, ADR-040;
U6-U8 are registered on accuracy and Macro-F1 only); no halluc-rate cell for
any V0 pair (V0 retrieves and cites nothing, so every claim is ungrounded by
construction); a variant whose run is not supplied takes its pairs out of the
family (V0: -6; V2: -6; V5-refit: -6), recorded in
`pairs_dropped_by_contingency`.

Each new run is validated before use and the script refuses to proceed
otherwise: variant, dataset medqa, split test, limit 500, n_evaluated 500,
seed 42, not superseded, hallucination scored (V2), and code identical to the
grid commit 41f7e14 (`scripts.report.code_matches_launch_commit`); for
V5-refit also derived from the grid V5 run and carrying the frozen spec's
digest. The comparators V1 / V3 / V5 / V4-conflict are the grid's pinned run
dirs.

Same method as the m=35 family (`scripts/analysis/exploratory_family.py`,
whose helpers this reuses): paired bootstrap, 10k resamples, seed 42,
two-sided; accuracy by per-question difference, Macro-F1 / halluc-rate by
joint question resampling.

    uv run python -m scripts.analysis.unplanned_v0_v2_family \\
        --v0 medqa_V0_<sha>_<ts> --v2 medqa_V2_<sha>_<ts> \\
        --v5-refit medqa_V5-refit_<sha>_<ts>
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
from scripts.report import code_matches_launch_commit  # noqa: E402
from src.retrieval.metrics import holm_bonferroni  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
OUT_PATH = RESULTS_DIR / "phase7_grid" / "unplanned_v0_v2_family.json"

FAMILY_SPEC: list[tuple[str, str, str, list[str]]] = [
    ("U1", "V0", "V1", ["accuracy", "macro_f1"]),
    ("U2", "V0", "V3", ["accuracy", "macro_f1"]),
    ("U3", "V0", "V5", ["accuracy", "macro_f1"]),
    ("U4", "V2", "V3", ["accuracy", "macro_f1", "halluc_rate"]),
    ("U5", "V1", "V2", ["accuracy", "macro_f1", "halluc_rate"]),
    ("U6", "V5-refit", "V5", ["accuracy", "macro_f1"]),
    ("U7", "V5-refit", "V4-conflict", ["accuracy", "macro_f1"]),
    ("U8", "V5-refit", "V3", ["accuracy", "macro_f1"]),
]
REGISTERED_M = 18
COMPARATORS = ("V1", "V3", "V5", "V4-conflict")


def validate_run(run_dir: Path, variant: str) -> None:
    m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    problems = []
    expect = {"variant": variant, "dataset": "medqa", "split": "test", "limit": 500, "seed": 42}
    for k, v in expect.items():
        if m.get(k) != v:
            problems.append(f"{k}={m.get(k)!r} (expected {v!r})")
    if m["metrics"].get("n_evaluated") != 500:
        problems.append(f"n_evaluated={m['metrics'].get('n_evaluated')!r} (expected 500)")
    if m.get("superseded"):
        problems.append("run is marked superseded")
    ok, why = code_matches_launch_commit(m)
    if not ok:
        problems.append(f"code provenance: {why}")
    if variant == "V2" and not m["metrics"].get("hallucination"):
        problems.append(
            "hallucination not scored -- run "
            "`uv run python -m scripts.analysis.rescore_hallucination "
            f"results/{run_dir.name}` first"
        )
    if variant == "V5-refit":
        from scripts.analysis.v5_refit import load_spec

        if m.get("derived_from") != GRID_RUN_DIRS["V5"].name:
            problems.append(f"derived_from={m.get('derived_from')!r} is not the grid V5 run")
        if m.get("refit_spec_sha256") != load_spec()["spec_sha256"]:
            problems.append("refit_spec_sha256 does not match the frozen spec")
    if problems:
        raise SystemExit(
            f"[unplanned_v0_v2_family] {variant} run {run_dir.name} is not a valid grid cell: "
            + "; ".join(problems)
        )


def run_family(run_dirs: dict[str, Path], out_path: Path) -> dict:
    """`run_dirs`: the comparators V1/V3/V5 plus whichever of V0/V2 exist."""
    present = set(run_dirs)
    spec = [s for s in FAMILY_SPEC if s[1] in present and s[2] in present]
    dropped = [s[0] for s in FAMILY_SPEC if s not in spec]
    m_expected = REGISTERED_M - sum(len(s[3]) for s in FAMILY_SPEC if s[0] in dropped)

    preds = {name: _load_predictions(d) for name, d in run_dirs.items()}
    per_claim = {name: _load_per_claim(d) for name, d in run_dirs.items() if name != "V0"}

    results: dict[str, dict] = {}
    p_values: dict[str, float] = {}
    for pair, side_a, side_b, metrics in spec:
        pa, pb = preds[side_a], preds[side_b]
        common = sorted(set(pa) & set(pb))
        if len(common) != len(pa) or len(common) != len(pb):
            raise SystemExit(
                f"[unplanned_v0_v2_family] {pair}: question_id mismatch "
                f"{side_a} n={len(pa)} vs {side_b} n={len(pb)}, common={len(common)}"
            )
        if any(pa[q]["gold_option"] != pb[q]["gold_option"] for q in common):
            raise SystemExit(f"[unplanned_v0_v2_family] {pair}: gold labels disagree")

        for metric in metrics:
            if metric == "accuracy":
                res = _rank_metric_test(pa, pb, common, "accuracy")
            elif metric == "macro_f1":
                stat_a = {q: (pa[q]["pred_option"], pa[q]["gold_option"]) for q in common}
                stat_b = {q: (pb[q]["pred_option"], pb[q]["gold_option"]) for q in common}
                res = _bootstrap_aggregate_diff(common, stat_a, stat_b, _macro_f1_of)
            elif metric == "halluc_rate":
                claims_a, claims_b = per_claim[side_a], per_claim[side_b]
                if claims_a is None or claims_b is None:
                    raise SystemExit(f"[unplanned_v0_v2_family] {pair}: missing hallucination data")
                stat_a = {q: claims_a.get(q, []) for q in common}
                stat_b = {q: claims_b.get(q, []) for q in common}
                res = _bootstrap_aggregate_diff(common, stat_a, stat_b, _halluc_rate_predicted)
            else:
                raise ValueError(metric)

            res.update(pair=pair, side_a=side_a, side_b=side_b, metric=metric, n=len(common))
            if res["ci_lo"] > 0:
                res["verdict"] = f"{side_a} significantly > {side_b}"
            elif res["ci_hi"] < 0:
                res["verdict"] = f"{side_b} significantly > {side_a}"
            else:
                res["verdict"] = "tie (95% CI includes 0)"
            key = f"{pair}:{metric}"
            results[key] = res
            p_values[key] = res["p_value"]

    if len(p_values) != m_expected:
        raise SystemExit(f"[unplanned_v0_v2_family] tested {len(p_values)}, expected {m_expected}")
    holm = holm_bonferroni(p_values, alpha=ALPHA)
    for key, h in holm.items():
        results[key].update(
            holm_rank=h["rank"], holm_threshold=h["threshold"], holm_reject=h["reject"]
        )

    out = {
        "family": (
            "UNPLANNED exploratory -- V0/V2 grid rows and V5-refit (docs/04 §1.0b, "
            "ADR-040, ADR-043)"
        ),
        "status": (
            "V0 and V2 are pre-specified grid rows (docs/04 §0.3); V5-refit is a disclosed "
            "post-hoc correction of V5's scoring rule. None of their TESTS was pre-registered. "
            "U1-U5 registered 2026-09-10, U6-U8 2026-09-11, each before any run of this family "
            "existed. Holm-corrected within this family only; never merged into, pooled with, "
            "or used to reinterpret the m=35 pre-registered family."
        ),
        "registered_family_size": REGISTERED_M,
        "tested_family_size": len(p_values),
        "pairs_dropped_by_contingency": dropped,
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
    ap.add_argument("--v0", help="V0 run dir name under results/ (omit -> U1-U3 leave the family)")
    ap.add_argument("--v2", help="V2 run dir name under results/ (omit -> U4-U5 leave the family)")
    ap.add_argument(
        "--v5-refit", help="V5-refit run dir name under results/ (omit -> U6-U8 leave the family)"
    )
    args = ap.parse_args()
    if not (args.v0 or args.v2 or args.v5_refit):
        ap.error("pass at least one of --v0 / --v2 / --v5-refit")

    run_dirs = {name: GRID_RUN_DIRS[name] for name in COMPARATORS}
    for variant, name in (("V0", args.v0), ("V2", args.v2), ("V5-refit", args.v5_refit)):
        if name:
            d = RESULTS_DIR / name
            validate_run(d, variant)
            run_dirs[variant] = d

    out = run_family(run_dirs, OUT_PATH)
    print(f"[unplanned_v0_v2_family] wrote {OUT_PATH}")
    print(
        f"[unplanned_v0_v2_family] m_registered={REGISTERED_M} "
        f"m_tested={out['tested_family_size']} "
        f"dropped={out['pairs_dropped_by_contingency']}"
    )
    for key, r in sorted(out["results"].items(), key=lambda kv: kv[1]["holm_rank"]):
        print(
            f"  {key:16s} A={r['point_a']:.3f} B={r['point_b']:.3f} "
            f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p={r['p_value']:.4f} "
            f"rank={r['holm_rank']} reject={r['holm_reject']}  {r['verdict']}"
        )


if __name__ == "__main__":
    main()
