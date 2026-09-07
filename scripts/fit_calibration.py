"""Phase 5 exit test: fit calibration weights on a disjoint dev slice, report
ECE on a HELD-OUT half of it (not the fitting half -- fitting+evaluating on
the same n=20 the way an earlier draft plan proposed would make the exit test
circular, since C3 is the one contribution most likely to come out positive;
see the ADR this phase adds), and emit a full ranked-output row with
per-option citations for every question.

Reads a completed V4 agent run's `predictions.jsonl` (one `SupportMatrix` per
line -- `src.pipeline._run_agents_eval`'s output, run with `--exclude-limit`
so it is disjoint from Phase 3/4's dev-20). Splits the 120 questions by sorted
`question_id` into the first half (fit) and second half (held-out) --
deterministic, no extra randomness, and the exact split
`scripts/raw_logprob_baseline.py` uses independently over the SAME question
set, so the two scripts' subsets line up question-for-question.

Usage:
    uv run python -m scripts.fit_calibration \\
        --run-dir results/medqa_V4_<sha>_<ts> \\
        --raw-baseline-json results/phase5_calibration/medqa_raw_logprob_baseline.json \\
        --write-v5-config
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.agents.state import SupportMatrix  # noqa: E402
from src.calibration.citations import cite_all  # noqa: E402
from src.calibration.scoring import extract_option_features  # noqa: E402
from src.calibration.softmax_rank import (  # noqa: E402
    bootstrap_weight_uncertainty,
    choose_abstention_threshold,
    coverage_risk_curve,
    fit_weights,
    rank_options,
)
from src.eval.metrics import ece, paired_bootstrap_ece_diff  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "phase5_calibration"
V5_CONFIG_PATH = REPO_ROOT / "configs" / "v5.yaml"


def load_matrices(run_dir: Path) -> list[SupportMatrix]:
    path = run_dir / "predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- point --run-dir at a completed "
            "`python -m src.pipeline` run directory"
        )
    matrices = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                matrices.append(SupportMatrix.model_validate(json.loads(line)))
    return matrices


def split_fit_held_out(
    matrices: list[SupportMatrix],
) -> tuple[list[SupportMatrix], list[SupportMatrix]]:
    """First half (sorted by question_id) = fit, second half = held-out.
    Deterministic; matches scripts/raw_logprob_baseline.py's split exactly
    over the same question set."""
    ordered = sorted(matrices, key=lambda m: m.question_id)
    mid = len(ordered) // 2
    return ordered[:mid], ordered[mid:]


def _rank1_accuracy(matrices: list[SupportMatrix], weights) -> float | None:
    scored = 0
    correct = 0
    for m in matrices:
        if not m.gold_option:
            continue
        feats = extract_option_features(m)
        ranked = rank_options(feats, weights)
        scored += 1
        if ranked[0][0] == m.gold_option:
            correct += 1
    return correct / scored if scored else None


def _probs_and_gold(matrices: list[SupportMatrix], weights) -> tuple[list[list[float]], list[int]]:
    """Per-question probability vector (option order = sorted option ids, to
    match rank_options's stable pairing) and the gold option's index in that
    order. Questions with no gold_option are skipped (nothing to score)."""
    probs, gold_idx = [], []
    for m in matrices:
        if not m.gold_option:
            continue
        feats = extract_option_features(m)
        ranked = dict(rank_options(feats, weights))
        option_ids = sorted(ranked)
        probs.append([ranked[oid] for oid in option_ids])
        gold_idx.append(option_ids.index(m.gold_option))
    return probs, gold_idx


def paired_calibrated_vs_raw(
    matrices: list[SupportMatrix], weights, raw_by_qid: dict[str, dict]
) -> tuple[list[list[float]], list[list[float]], list[int]]:
    """Builds (calibrated_probs, raw_probs, gold_idx), all three aligned
    question-for-question AND option-for-option, joined explicitly by
    `question_id` (never by list position -- the two scripts that produce
    these numbers run independently and must not be assumed to iterate in
    the same order). A question missing from `raw_by_qid`, or whose gold
    option isn't in either side's letter set, is skipped and reported --
    never silently dropped without a count."""
    calibrated_probs, raw_probs, gold_idx = [], [], []
    n_missing_raw = 0
    n_gold_mismatch = 0
    for m in matrices:
        if not m.gold_option:
            continue
        raw_row = raw_by_qid.get(m.question_id)
        if raw_row is None:
            n_missing_raw += 1
            continue
        feats = extract_option_features(m)
        ranked = dict(rank_options(feats, weights))
        option_ids = sorted(ranked)
        if m.gold_option not in option_ids or m.gold_option not in raw_row["raw_probs"]:
            n_gold_mismatch += 1
            continue
        calibrated_probs.append([ranked[oid] for oid in option_ids])
        raw_probs.append([raw_row["raw_probs"][oid] for oid in option_ids])
        gold_idx.append(option_ids.index(m.gold_option))
    if n_missing_raw or n_gold_mismatch:
        print(
            f"[fit_calibration] WARNING: {n_missing_raw} question(s) missing from the raw "
            f"baseline, {n_gold_mismatch} gold-option mismatch(es) -- excluded from the "
            "paired ECE comparison, not silently coerced"
        )
    return calibrated_probs, raw_probs, gold_idx


def build_ranked_output(matrices: list[SupportMatrix], weights) -> list[dict]:
    """[(question_id, [(option_id, p_i, citations), ...] sorted desc by p_i)]
    -- the ranked-output row docs/00_RESEARCH_SPEC.md §4 point 5 specifies,
    with per-option citations for EVERY question (the Phase 5 exit test's
    second half)."""
    rows = []
    for m in matrices:
        feats = extract_option_features(m)
        ranked = rank_options(feats, weights)
        citations = cite_all(m)
        rows.append(
            {
                "question_id": m.question_id,
                "gold_option": m.gold_option,
                "ranked": [
                    {"option_id": oid, "p": p, "citations": citations[oid]} for oid, p in ranked
                ],
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="a completed V4 run directory under results/")
    ap.add_argument(
        "--raw-baseline-json",
        default=None,
        help="scripts/raw_logprob_baseline.py's output; if given, the exit-test "
        "comparison (calibrated ECE < raw-logprob ECE) is computed and asserted here",
    )
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    ap.add_argument(
        "--n-ece-bootstrap",
        type=int,
        default=10_000,
        help="paired bootstrap resamples for the ECE difference exit-test comparison "
        "(project standard, docs/04_EVALUATION.md SS1 / ADR-012)",
    )
    ap.add_argument("--ece-bootstrap-seed", type=int, default=42)
    ap.add_argument("--target-risk", type=float, default=0.10)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--write-v5-config",
        action="store_true",
        help="write the fitted weights into configs/v5.yaml",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    matrices = load_matrices(run_dir)
    print(f"[fit_calibration] loaded {len(matrices)} SupportMatrix rows from {run_dir}")

    fit_set, held_out_set = split_fit_held_out(matrices)
    print(f"[fit_calibration] fit n={len(fit_set)}  held_out n={len(held_out_set)}")
    assert not ({m.question_id for m in fit_set} & {m.question_id for m in held_out_set}), (
        "fit/held-out split overlapped -- this would make the exit test circular"
    )

    fit_examples = [
        (extract_option_features(m), m.gold_option) for m in fit_set if m.gold_option
    ]
    weights = fit_weights(fit_examples)
    print(f"[fit_calibration] fitted weights: {weights}")

    fit_probs, fit_gold = _probs_and_gold(fit_set, weights)
    held_out_probs, held_out_gold = _probs_and_gold(held_out_set, weights)
    ece_fit = ece(fit_probs, fit_gold, n_bins=5, binning="equal_mass")
    ece_held_out = ece(held_out_probs, held_out_gold, n_bins=5, binning="equal_mass")

    print(f"[fit_calibration] ECE fit-set (in-sample):  {ece_fit['ece']}")
    print(f"[fit_calibration] ECE held-out (out-of-sample): {ece_held_out['ece']}")
    ece_gap = (
        (ece_held_out["ece"] - ece_fit["ece"])
        if ece_fit["ece"] is not None and ece_held_out["ece"] is not None
        else None
    )
    if ece_gap is not None:
        print(
            f"[fit_calibration] fit-vs-held-out ECE gap: {ece_gap:+.4f} "
            "(itself informative -- see the ADR)"
        )

    print("[fit_calibration] bootstrapping weight uncertainty on the fit set "
          f"({args.n_bootstrap} resamples)...")
    boot = bootstrap_weight_uncertainty(
        fit_examples, n_resamples=args.n_bootstrap, seed=args.bootstrap_seed
    )
    for name, stats in boot.items():
        print(f"    {name}: mean={stats['mean']:+.4f} std={stats['std']:.4f} "
              f"range=[{stats['min']:+.4f}, {stats['max']:+.4f}]")

    curve = coverage_risk_curve(
        [(extract_option_features(m), m.gold_option) for m in held_out_set if m.gold_option],
        weights,
    )
    tau = choose_abstention_threshold(curve, target_risk=args.target_risk)
    print(f"[fit_calibration] abstention tau (target_risk={args.target_risk}): {tau:.4f}")

    rank1_fit = _rank1_accuracy(fit_set, weights)
    rank1_held_out = _rank1_accuracy(held_out_set, weights)
    print(
        f"[fit_calibration] rank-1 accuracy: fit={rank1_fit}  held_out={rank1_held_out} "
        "(a number, NOT a B2 comparison -- the AUDIT SS4.3 budget asymmetry is unfixed)"
    )

    ranked_output_fit = build_ranked_output(fit_set, weights)
    ranked_output_held_out = build_ranked_output(held_out_set, weights)
    n_missing_citations_struct = sum(
        1
        for row in ranked_output_fit + ranked_output_held_out
        if len(row["ranked"]) == 0
    )
    assert n_missing_citations_struct == 0, (
        "every question must have a ranked row with per-option citations, even if a "
        "given option's own citation list is empty"
    )

    exit_test = None
    if args.raw_baseline_json:
        raw = json.loads(Path(args.raw_baseline_json).read_text(encoding="utf-8"))
        raw_by_qid = {row["question_id"]: row for row in raw["per_question"]}
        calibrated_probs, raw_probs, joined_gold_idx = paired_calibrated_vs_raw(
            held_out_set, weights, raw_by_qid
        )
        print(
            f"[fit_calibration] paired ECE comparison over {len(joined_gold_idx)} "
            "held-out questions (joined by question_id)"
        )
        boot_ece = paired_bootstrap_ece_diff(
            calibrated_probs,
            raw_probs,
            joined_gold_idx,
            n_bins=5,
            binning="equal_mass",
            n_resamples=args.n_ece_bootstrap,
            seed=args.ece_bootstrap_seed,
        )
        # a_significantly_better means the CALIBRATED side (probs_a) has the
        # lower (better) ECE -- see paired_bootstrap_ece_diff's verdict rule.
        passed = boot_ece["verdict"] == "a_significantly_better"
        if boot_ece["verdict"] == "tie":
            verdict_text = (
                "TIE (95% CI on the ECE difference includes 0 -- calibration is not "
                "worse; improvement not demonstrated at this sample size, per ADR-012)"
            )
        elif passed:
            verdict_text = "PASS (calibrated ECE significantly lower, CI entirely below 0)"
        else:
            verdict_text = "FAIL (raw-logprob ECE significantly lower -- calibration is worse)"
        exit_test = {
            "n_paired": len(joined_gold_idx),
            "calibrated_ece_held_out_point": boot_ece["point_ece_a"],
            "raw_logprob_ece_held_out_point": boot_ece["point_ece_b"],
            "diff_point_estimate": boot_ece["point_estimate_diff"],
            "diff_bootstrap_mean": boot_ece["mean_diff"],
            "diff_ci_lo": boot_ece["ci_lo"],
            "diff_ci_hi": boot_ece["ci_hi"],
            "n_resamples": boot_ece["n_resamples"],
            "seed": boot_ece["seed"],
            "verdict": boot_ece["verdict"],
            "passed": passed,
        }
        print(
            f"[fit_calibration] EXIT TEST: diff (calibrated - raw) = "
            f"{boot_ece['point_estimate_diff']:+.4f}, 95% CI "
            f"[{boot_ece['ci_lo']:+.4f}, {boot_ece['ci_hi']:+.4f}] "
            f"({boot_ece['n_resamples']} resamples, seed={boot_ece['seed']}) -> {verdict_text}"
        )

    out = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_dir": str(run_dir),
        "n_fit": len(fit_set),
        "n_held_out": len(held_out_set),
        "weights": weights.model_dump(),
        "ece_fit": ece_fit,
        "ece_held_out": ece_held_out,
        "ece_fit_vs_held_out_gap": ece_gap,
        "gamma_bootstrap": boot["gamma"],
        "weight_bootstrap": boot,
        "coverage_risk_curve": curve,
        "abstention_tau": tau,
        "abstention_target_risk": args.target_risk,
        "rank1_accuracy_fit": rank1_fit,
        "rank1_accuracy_held_out": rank1_held_out,
        "ranked_output_fit": ranked_output_fit,
        "ranked_output_held_out": ranked_output_held_out,
        "exit_test": exit_test,
    }
    out_path = Path(args.out) if args.out else RESULTS_DIR / "medqa_dev_calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[fit_calibration] wrote {out_path}")

    if args.write_v5_config:
        _write_v5_config(weights, tau, run_dir)
        print(f"[fit_calibration] wrote {V5_CONFIG_PATH}")


def _write_v5_config(weights, tau: float, run_dir: Path) -> None:
    content = f"""# Phase 7 V5 variant: agents + calibration (Phase 5). Inherits everything
# from configs/agents.yaml (which itself extends base.yaml) via `_extends` --
# only the calibration block differs. See docs/05_DECISIONS.md's Phase 5 ADR
# for the fit procedure and docs/04_EVALUATION.md for the reported numbers.
_extends: agents.yaml

calibration:
  enabled: true
  alpha: {weights.alpha}
  beta: {weights.beta}
  gamma: {weights.gamma}
  delta: {weights.delta}
  temperature: {weights.temperature}
  abstention_tau: {tau}
  n_ece_bins: 15
  ece_binning: equal_width
  fit_dataset: medqa
  fit_split: train
  fit_limit: 60
  fit_seed: 42
  # Fitted from {run_dir} (predictions.jsonl), a 120-question MedQA train
  # slice disjoint from Phase 3/4's dev-20 (--exclude-limit 20, seed 42).
  # See scripts/fit_calibration.py and results/phase5_calibration/medqa_dev_calibration.json.
"""
    V5_CONFIG_PATH.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
