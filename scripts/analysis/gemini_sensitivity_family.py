"""The §0.6 stronger-model sensitivity check (docs/04 §1.0d, ADR-047).

ONE confirmatory test -- V3-gemini vs V5-gemini, accuracy, n=200, paired
bootstrap 10k, seed 42, two-sided, alpha=0.05 -- the direct replication of the
MedQA confirmatory test (§1.0). Plus an exploratory family:

    G1  V0-gemini vs V3-gemini   acc, macro_f1
    G2  V0-gemini vs V5-gemini   acc, macro_f1
    G3  V3-gemini vs B3-gemini   acc, macro_f1
    G4  V5-gemini vs B3-gemini   acc, macro_f1

m_G = 8, fixed here before any p-value in this family exists. Holm-Bonferroni
WITHIN THIS FAMILY ONLY -- never pooled with, and never used to reinterpret or
re-threshold, the pre-registered m=35 family, the unplanned U1-U8 family or the
post-hoc m_X=7 family.

WHAT THIS MEASURES: model dependence, not MedQA. Same code, same prompts, same
retrieval config, same scoring rule, same agent graph, same 200 questions --
only the generator differs. It is not independent evidence about any system's
MedQA accuracy and is never a headline row.

Descriptive only, never tested and never Holm-corrected: B3's
verifier_decision_rate and n_supported distribution, V5's ECE and ranking
metrics against the analytic chance row, and per-question 7B-vs-Gemini
agreement. The three registered predictions (P1/P2/P3, §1.0d) are scored
against the 7B grid runs restricted to the same pinned 200 ids.

    uv run python -m scripts.analysis.gemini_sensitivity_family \
        --v0 medqa_V0-gemini_<sha>_<ts> --v3 medqa_V3-gemini_<sha>_<ts> \
        --b3 medqa_B3-gemini_<sha>_<ts> --v5 medqa_V5-gemini_<sha>_<ts>
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
    _load_predictions,
    _macro_f1_of,
    _rank_metric_test,
)
from scripts.freeze_gemini_question_ids import PAIRED_7B_RUNS  # noqa: E402
from src.retrieval.metrics import holm_bonferroni  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
PINNED_IDS_PATH = RESULTS_DIR / "phase7_gemini" / "question_ids_200.json"
OUT_PATH = RESULTS_DIR / "phase7_gemini" / "gemini_sensitivity_family.json"

CONFIRMATORY = ("V3", "V5", "accuracy")
FAMILY_SPEC: list[tuple[str, str, str, list[str]]] = [
    ("G1", "V0", "V3", ["accuracy", "macro_f1"]),
    ("G2", "V0", "V5", ["accuracy", "macro_f1"]),
    ("G3", "V3", "B3", ["accuracy", "macro_f1"]),
    ("G4", "V5", "B3", ["accuracy", "macro_f1"]),
]
FAMILY_M = 8

# Analytic expectations of a uniformly random ranking of 4 options -- the chance
# row every ranking metric is displayed against (ADR-040 Decision 2).
CHANCE_4WAY = {
    "p_at_1": 0.25,
    "p_at_2": 0.5,
    "mrr": (1 + 1 / 2 + 1 / 3 + 1 / 4) / 4,
}

# P3's reference point: the 7B B3 grid cell's verifier_decision_rate. 0.186 at
# n=500, 0.185 on the pinned 200; §1.0d states the prediction against "the 7B's
# 0.19". ADR-033: this figure is prompt-sensitive (0.05-0.35 across phrasings),
# so P3 is a claim about THIS prompt, held fixed across both generators.
P3_FLOOR_7B = 0.185
P3_CEILING = 0.50
P1_MIN_GAP = 0.15


def load_pinned_ids() -> list[str]:
    doc = json.loads(PINNED_IDS_PATH.read_text(encoding="utf-8"))
    return doc["question_ids"]


def validate_gemini_run(run_dir: Path, variant: str, n_expected: int) -> dict:
    """A Gemini cell must be n=200 on the pinned set, carry the recorded
    `-gemini` variant, and name the model that served it. Fails loudly."""
    m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    problems = []
    if m.get("variant") != f"{variant}-gemini":
        problems.append(f"variant={m.get('variant')!r} (expected {variant + '-gemini'!r})")
    for k, v in {"dataset": "medqa", "split": "test", "seed": 42}.items():
        if m.get(k) != v:
            problems.append(f"{k}={m.get(k)!r} (expected {v!r})")
    if m["metrics"].get("n_evaluated") != n_expected:
        problems.append(f"n_evaluated={m['metrics'].get('n_evaluated')!r} (expected {n_expected})")
    if m.get("superseded"):
        problems.append("run is marked superseded")

    cfg_path = run_dir / "config.yaml"
    model = api_version = resolved_utc = None
    if cfg_path.exists():
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        chain = (cfg.get("llm") or {}).get("fallback_chain") or []
        model = chain[0].get("model") if chain else None
        api_version = (cfg.get("llm") or {}).get("api_version")
        resolved_utc = (cfg.get("llm") or {}).get("model_resolved_utc")
        if not model or "gemini" not in str(model):
            problems.append(f"config.yaml model={model!r} is not a Gemini model")
        if not resolved_utc:
            problems.append("config.yaml has no llm.model_resolved_utc (run date provenance)")
    else:
        problems.append("config.yaml missing")

    if problems:
        raise SystemExit(
            f"[gemini_sensitivity_family] {variant} run {run_dir.name} is not a valid "
            "§1.0d cell: " + "; ".join(problems)
        )
    return {"model": model, "api_version": api_version, "model_resolved_utc": resolved_utc}


def _acc(preds: dict[str, dict], ids: list[str]) -> float:
    return sum(1 for q in ids if preds[q]["pred_option"] == preds[q]["gold_option"]) / len(ids)


def _pred_gold(rows: dict[str, dict], ids: list[str]) -> dict[str, tuple]:
    """qid -> (pred, gold), the per-question payload `_macro_f1_of` aggregates."""
    return {q: (rows[q]["pred_option"], rows[q]["gold_option"]) for q in ids}


def _metrics_of(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]


def _paired_7b(cell: str, ids: list[str]) -> dict[str, dict] | None:
    """The 7B grid run for `cell`, restricted to the pinned ids -- or None when
    predictions.jsonl is absent (it is gitignored)."""
    path = RESULTS_DIR / PAIRED_7B_RUNS[cell] / "predictions.jsonl"
    if not path.exists():
        return None
    rows = _load_predictions(RESULTS_DIR / PAIRED_7B_RUNS[cell])
    missing = [q for q in ids if q not in rows]
    if missing:
        raise SystemExit(
            f"[gemini_sensitivity_family] 7B {cell} run is missing {len(missing)} pinned ids"
        )
    return {q: rows[q] for q in ids}


def _verdict(res: dict, side_a: str, side_b: str) -> str:
    if res["holm_reject"]:
        return f"{side_a if res['ci_lo'] > 0 else side_b} significantly better (Holm, m_G=8)"
    if res["ci_lo"] > 0 or res["ci_hi"] < 0:
        return "not significant after Holm (unadjusted 95% CI excludes 0)"
    return "tie (95% CI includes 0)"


def run_family(run_dirs: dict[str, Path], ids: list[str], out_path: Path) -> dict:
    preds = {name: _load_predictions(d) for name, d in run_dirs.items()}
    for name, p in preds.items():
        missing = [q for q in ids if q not in p]
        if missing:
            raise SystemExit(
                f"[gemini_sensitivity_family] {name} is missing {len(missing)} pinned ids "
                f"(first: {missing[:3]}) -- the run did not cover the frozen question set"
            )
        preds[name] = {q: p[q] for q in ids}

    provenance = {
        name: validate_gemini_run(d, name, len(ids)) for name, d in sorted(run_dirs.items())
    }

    # --- the single confirmatory test ------------------------------------
    a, b, metric = CONFIRMATORY
    conf = _rank_metric_test(preds[a], preds[b], ids, metric)
    conf.update(
        test="confirmatory (docs/04 §1.0d) -- the §1.0 MedQA confirmatory test, replicated",
        comparison=f"{a}-gemini vs {b}-gemini",
        metric=metric,
        n=len(ids),
        significant=conf["p_value"] <= ALPHA,
        verdict=(
            f"{a} significantly > {b}"
            if conf["ci_lo"] > 0
            else f"{b} significantly > {a}"
            if conf["ci_hi"] < 0
            else "tie (95% CI includes 0) -- reported as a tie, never as a win"
        ),
        note="Not Holm-corrected: it is the single confirmatory test, outside the m_G=8 family.",
    )

    # --- exploratory family, Holm within these 8 only --------------------
    results: dict[str, dict] = {}
    p_values: dict[str, float] = {}
    for pair, side_a, side_b, metrics in FAMILY_SPEC:
        for m in metrics:
            if m == "accuracy":
                res = _rank_metric_test(preds[side_a], preds[side_b], ids, "accuracy")
            elif m == "macro_f1":
                res = _bootstrap_aggregate_diff(
                    ids,
                    _pred_gold(preds[side_a], ids),
                    _pred_gold(preds[side_b], ids),
                    _macro_f1_of,
                )
            else:
                raise ValueError(m)
            res.update(pair=pair, side_a=side_a, side_b=side_b, metric=m, n=len(ids))
            key = f"{pair}:{m}"
            results[key] = res
            p_values[key] = res["p_value"]

    if len(p_values) != FAMILY_M:
        raise SystemExit(
            f"[gemini_sensitivity_family] tested {len(p_values)}, expected m_G={FAMILY_M}"
        )
    for key, h in holm_bonferroni(p_values, alpha=ALPHA).items():
        r = results[key]
        r.update(holm_rank=h["rank"], holm_threshold=h["threshold"], holm_reject=h["reject"])
        r["verdict"] = _verdict(r, r["side_a"], r["side_b"])

    # --- descriptive only: never tested, never Holm-corrected ------------
    b3_metrics = _metrics_of(run_dirs["B3"])
    v5_metrics = _metrics_of(run_dirs["V5"])
    agreement = {}
    for cell in sorted(run_dirs):
        sevenb = _paired_7b(cell, ids)
        if sevenb is None:
            agreement[cell] = {"note": "7B predictions.jsonl absent (gitignored)"}
            continue
        same = sum(1 for q in ids if sevenb[q]["pred_option"] == preds[cell][q]["pred_option"])
        both = sum(
            1
            for q in ids
            if sevenb[q]["pred_option"] == sevenb[q]["gold_option"]
            and preds[cell][q]["pred_option"] == preds[cell][q]["gold_option"]
        )
        agreement[cell] = {
            "same_prediction_rate": same / len(ids),
            "both_correct_rate": both / len(ids),
            "accuracy_7b": _acc(sevenb, ids),
            "accuracy_gemini": _acc(preds[cell], ids),
        }

    descriptive = {
        "note": (
            "DESCRIPTIVE ONLY -- displayed, never tested, never Holm-corrected, "
            "never cited as a significance result."
        ),
        "b3_verifier_decision_rate": b3_metrics.get("verifier_decision_rate"),
        "b3_n_supported_distribution": b3_metrics.get("n_supported_distribution"),
        "v5_ece": v5_metrics.get("ece"),
        "v5_ranking_vs_chance": {
            k: {"v5": v5_metrics.get(k), "chance": CHANCE_4WAY[k]} for k in CHANCE_4WAY
        },
        "per_question_7b_vs_gemini_agreement": agreement,
    }

    # --- the three registered predictions (§1.0d) ------------------------
    acc_g = {c: _acc(preds[c], ids) for c in preds}
    acc_7b = {}
    for c in sorted(run_dirs):
        s = _paired_7b(c, ids)
        acc_7b[c] = _acc(s, ids) if s else None

    p1_gap = acc_g["V3"] - acc_g["V5"]
    p2_rows = {
        c: (None if acc_7b[c] is None else acc_g[c] > acc_7b[c]) for c in sorted(run_dirs)
    }
    b3_rate = b3_metrics.get("verifier_decision_rate")
    predictions = {
        "P1": {
            "text": (
                "the two-tier structure replicates -- V3 beats V5 on accuracy "
                "by >=0.15, significant."
            ),
            "gap_v3_minus_v5": p1_gap,
            "significant": conf["significant"] and conf["ci_lo"] > 0,
            "held": bool(p1_gap >= P1_MIN_GAP and conf["significant"] and conf["ci_lo"] > 0),
            "on_failure": (
                "the V3>V5 step is a property of the 7B generator, not the architecture; "
                "§5.0a's two-tier claim would have to be restated as model-specific."
            ),
        },
        "P2": {
            "text": "every cell scores above its Qwen2.5-7B counterpart on the same 200 questions.",
            "accuracy_gemini": acc_g,
            "accuracy_7b_same_200": acc_7b,
            "per_cell": p2_rows,
            "held": (
                None
                if any(v is None for v in p2_rows.values())
                else all(p2_rows.values())
            ),
            "on_failure": (
                "the pipeline does not simply improve with a stronger generator; a cell that "
                "gets worse indicates the scaffold is tuned to 7B-specific behaviour."
            ),
        },
        "P3": {
            "text": "B3's verifier_decision_rate rises above the 7B's 0.19 but stays below 0.50.",
            "verifier_decision_rate_gemini": b3_rate,
            "verifier_decision_rate_7b_same_200": P3_FLOOR_7B,
            "held": (
                None if b3_rate is None else bool(P3_FLOOR_7B < b3_rate < P3_CEILING)
            ),
            "on_failure": (
                "below 0.19: the binary bottleneck is not a weak-model artifact at all. "
                "above 0.50: it largely is one, and C2's verifier-collapse finding must be "
                "restated as a small-model phenomenon."
            ),
        },
    }

    out = {
        "check": "docs/04_EVALUATION.md §1.0d -- §0.6 stronger-model sensitivity check (ADR-047)",
        "measures": (
            "MODEL DEPENDENCE, not MedQA. Same code, prompts, retrieval config, scoring rule "
            "and agent graph; same 200 questions; only the generator differs. Not independent "
            "evidence about any system's MedQA accuracy. Robustness only, never a headline row."
        ),
        "multiplicity": (
            "One confirmatory test (V3 vs V5 accuracy, uncorrected). Exploratory family m_G=8, "
            "Holm-Bonferroni WITHIN THIS FAMILY ONLY -- never pooled with, or used to "
            "reinterpret or re-threshold, the m=35, U1-U8 or m_X=7 families."
        ),
        "n": len(ids),
        "question_set": {
            "path": str(PINNED_IDS_PATH.relative_to(REPO_ROOT)),
            "sha256": json.loads(PINNED_IDS_PATH.read_text(encoding="utf-8"))["sha256"],
            "selection": "first 200 of the MedQA test-500 seeded order (seed 42), pinned by id",
        },
        "run_dirs": {k: v.name for k, v in sorted(run_dirs.items())},
        "generator_provenance": provenance,
        "paired_7b_runs": PAIRED_7B_RUNS,
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "alpha": ALPHA,
        "two_sided": True,
        "confirmatory": conf,
        "family_size": FAMILY_M,
        "exploratory": results,
        "descriptive": descriptive,
        "registered_predictions": predictions,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for cell in ("v0", "v3", "b3", "v5"):
        ap.add_argument(f"--{cell}", required=True, help=f"{cell.upper()}-gemini run dir name")
    args = ap.parse_args()

    ids = load_pinned_ids()
    run_dirs = {
        "V0": RESULTS_DIR / args.v0,
        "V3": RESULTS_DIR / args.v3,
        "B3": RESULTS_DIR / args.b3,
        "V5": RESULTS_DIR / args.v5,
    }
    out = run_family(run_dirs, ids, OUT_PATH)

    print(f"[gemini_sensitivity_family] wrote {OUT_PATH}")
    c = out["confirmatory"]
    print(
        f"  CONFIRMATORY V3 vs V5 acc: {c['point_a']:.3f} vs {c['point_b']:.3f} "
        f"[{c['ci_lo']:+.3f},{c['ci_hi']:+.3f}] p={c['p_value']:.4f}  {c['verdict']}"
    )
    for key, r in sorted(out["exploratory"].items(), key=lambda kv: kv[1]["holm_rank"]):
        print(
            f"  {key:14s} A={r['point_a']:.3f} B={r['point_b']:.3f} "
            f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p={r['p_value']:.4f} "
            f"rank={r['holm_rank']} reject={r['holm_reject']}  {r['verdict']}"
        )
    for name, p in out["registered_predictions"].items():
        print(f"  {name}: held={p['held']}")


if __name__ == "__main__":
    main()
