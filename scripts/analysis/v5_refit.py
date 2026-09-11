"""V5-refit -- the disclosed post-hoc correction of V5's calibrated ranking
(ADR-043). Its tests are registered in `docs/04_EVALUATION.md` §1.0b's
unplanned family (U6-U8). V5 (`configs/v5.yaml`) stays the pre-registered
headline; V5-refit is reported beside it and replaces nothing.

ADR-041 found three defects in V5's scoring rule and ADR-042 one in its
inputs. V5-refit changes exactly these four things and nothing else:

  1. Sign. alpha (SUPPORT count), beta (CONTRADICT count, entering as
     -beta*s-) and gamma (KG SUPPORT count) are constrained >= 0, so every
     evidential term enters in its evidential direction. V5's fit put alpha
     at -0.0139. delta (a retrieval-quality prior over the cells consulted,
     in either direction) has no a-priori sign and stays free; T as before.
  2. No-signal rerank. `mean_rerank_score` is 0.0 for an option with no
     SUPPORT/CONTRADICT cell (`src/calibration/scoring.py`). A MedCPT score
     of 0 reads as "very low quality", not "no signal", against a train
     median near 5.9. V5-refit centres the feature at a reference r_ref
     frozen from the train fit half (the median over options with >= 1
     SUPPORT/CONTRADICT cell) and gives a no-signal option the reference
     itself: centred value 0, so delta*rerank contributes nothing for it.
  3. Ties. `rank_options` breaks exact probability ties by list order, i.e.
     by option letter (A first). V5-refit shuffles the options with a
     per-question seeded RNG (key "v5-refit-tiebreak:<seed>:<question_id>")
     before the same stable sort: uniform over tied options, reproducible,
     independent of letter and of question order.
  4. Feature source (ADR-042). s+ / s- are counted from the SupportMatrix's
     cells, not read from `option_support`, which is stale on questions
     whose audit failed mid-question (status agent_error). The two agree on
     every question of the train slice.

Everything else is V5's protocol, unchanged: the same 120-question MedQA
train slice (ADR-028) split by sorted question_id exactly as
`scripts/fit_calibration.py` does; the same 60-question fit half; the same
objective (mean gold-option NLL + the same L2 on alpha..delta); the same Adam
(lr, betas, eps, 500 iterations, init). Only the parameter bounds differ,
enforced by the same per-step projection. The held-out 60 set the
abstention threshold (V5's procedure) and give a descriptive pre-test check;
they choose nothing.

Lives under scripts/, not src/calibration/: src/ must stay byte-identical to
the grid commit 41f7e14 so the V0/V2 cells, and this variant's own launch,
keep their pin. It imports src.calibration unchanged. Port into
src/calibration/ after the V0/V2/V5-refit runs, with a replay test against
this run's stored rankings.

  fit    TRAIN ONLY -> `results/phase7_v5_refit/refit_spec.json`, the frozen
         spec, committed before any test evaluation. Refuses any input whose
         question ids are not all `train_*`.
  apply  TEST. Re-scores the grid V5 run's saved SupportMatrices under the
         frozen spec -> `results/medqa_V5-refit_<sha7>_<ts>/`. Zero LLM calls,
         zero GPU: the agent stage is the grid's own 41f7e14 output, so
         V5-refit vs V5 isolates the scoring rule exactly. Launch only via
         `scripts/run_phase7_v5_refit.ps1` (code pin + committed-spec check).

    uv run python -m scripts.analysis.v5_refit fit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.exploratory_family import RUN_DIRS as GRID_RUN_DIRS  # noqa: E402
from scripts.analysis.spos_consistency import cell_counts  # noqa: E402
from src.agents.state import SupportMatrix  # noqa: E402
from src.calibration import softmax_rank  # noqa: E402
from src.calibration.citations import cite_all  # noqa: E402
from src.calibration.scoring import (  # noqa: E402
    CalibrationWeights,
    OptionFeatures,
    compute_logit,
    extract_option_features,
)
from src.calibration.softmax_rank import choose_abstention_threshold, rank_options  # noqa: E402

VARIANT = "V5-refit"
SPEC_VERSION = 1
FIT_RUN_DIR = REPO_ROOT / "results" / "medqa_V4_a714fe0-dirty_20260907T121703Z"
PHASE5_JSON = REPO_ROOT / "results" / "phase5_calibration" / "medqa_dev_calibration.json"
SPEC_PATH = REPO_ROOT / "results" / "phase7_v5_refit" / "refit_spec.json"
V5_CONFIG_PATH = REPO_ROOT / "configs" / "v5.yaml"
V5_RUN_DIR = GRID_RUN_DIRS["V5"]
RESULTS_DIR = REPO_ROOT / "results"

TIE_BREAK_SEED = 42
N_BOOTSTRAP = 200  # Phase 5's weight-bootstrap convention (scripts/fit_calibration.py)
BOOTSTRAP_SEED = 42
TARGET_RISK = 0.10  # V5's abstention target (scripts/fit_calibration.py default)
CONVERGENCE_ITERS = 5000  # descriptive convergence check only; the fit uses V5's 500

PARAM_NAMES = ("alpha", "beta", "gamma", "delta", "log_t")
EVIDENTIAL = ("alpha", "beta", "gamma")
# V5's bounds with the evidential weights' lower bound raised to 0.
REFIT_BOUNDS = {
    **softmax_rank._BOUNDS,
    **{name: (0.0, softmax_rank._BOUNDS[name][1]) for name in EVIDENTIAL},
}


# ---- the rule --------------------------------------------------------------


def rerank_reference(matrices: list[SupportMatrix]) -> float:
    """Median raw `mean_rerank_score` over options with >= 1 SUPPORT or
    CONTRADICT cell -- exactly the options that feature is defined for."""
    values = []
    for m in matrices:
        counts = cell_counts(m)
        for f in extract_option_features(m):
            if sum(counts[f.option_id]) > 0:
                values.append(f.mean_rerank_score)
    return statistics.median(values)


def refit_features(matrix: SupportMatrix, r_ref: float) -> list[OptionFeatures]:
    """V5's features with (a) s+ / s- from the cell list and (b) rerank
    centred at `r_ref`, a no-signal option taking the reference (centred 0)."""
    counts = cell_counts(matrix)
    out = []
    for f in extract_option_features(matrix):
        s_pos, s_neg = counts[f.option_id]
        centred = f.mean_rerank_score - r_ref if s_pos + s_neg > 0 else 0.0
        out.append(
            f.model_copy(update={"s_pos": s_pos, "s_neg": s_neg, "mean_rerank_score": centred})
        )
    return out


def tie_break_order(
    features: list[OptionFeatures], question_id: str, seed: int = TIE_BREAK_SEED
) -> list[OptionFeatures]:
    """Options in a per-question seeded random order. `random.Random(str)`
    seeds from a SHA-512 of the string: stable across processes and
    platforms, independent of PYTHONHASHSEED."""
    order = sorted(features, key=lambda f: f.option_id)
    random.Random(f"v5-refit-tiebreak:{seed}:{question_id}").shuffle(order)
    return order


def rank_refit(
    features: list[OptionFeatures],
    weights: CalibrationWeights,
    question_id: str,
    seed: int = TIE_BREAK_SEED,
) -> list[tuple[str, float]]:
    """`rank_options` over the seeded order. Its sort is stable, so options
    with an exactly equal probability keep the shuffled order."""
    return rank_options(tie_break_order(features, question_id, seed), weights)


def fit_weights_bounded(
    examples: list[softmax_rank.FitExample],
    bounds: dict[str, tuple[float, float]],
    init: CalibrationWeights | None = None,
    max_iters: int = softmax_rank._MAX_ITERS,
) -> CalibrationWeights:
    """`src.calibration.softmax_rank.fit_weights`, step for step, except that
    the per-step projection uses `bounds`. Under that module's own `_BOUNDS`
    it reproduces `fit_weights` exactly (tests/test_v5_refit.py)."""
    if not examples:
        raise ValueError("fit_weights_bounded: examples must be non-empty")
    init = init or CalibrationWeights()
    params = [init.alpha, init.beta, init.gamma, init.delta, math.log(init.temperature)]
    for value, name in zip(params, PARAM_NAMES, strict=True):
        lo, hi = bounds[name]
        if not lo <= value <= hi:
            raise ValueError(f"init {name}={value} outside bounds {bounds[name]}")

    m = [0.0] * 5
    v = [0.0] * 5
    b1, b2 = softmax_rank._ADAM_BETA1, softmax_rank._ADAM_BETA2
    for t in range(1, max_iters + 1):
        _loss, grad = softmax_rank._loss_and_grad(params, examples)
        for i in range(5):
            m[i] = b1 * m[i] + (1 - b1) * grad[i]
            v[i] = b2 * v[i] + (1 - b2) * grad[i] ** 2
            m_hat = m[i] / (1 - b1**t)
            v_hat = v[i] / (1 - b2**t)
            params[i] -= softmax_rank._ADAM_LR * m_hat / (math.sqrt(v_hat) + softmax_rank._ADAM_EPS)
            params[i] = softmax_rank._clip(params[i], bounds[PARAM_NAMES[i]])

    alpha, beta, gamma, delta, log_t = params
    return CalibrationWeights(
        alpha=alpha, beta=beta, gamma=gamma, delta=delta, temperature=math.exp(log_t)
    )


def _params(w: CalibrationWeights) -> list[float]:
    return [w.alpha, w.beta, w.gamma, w.delta, math.log(w.temperature)]


def objective(examples: list[softmax_rank.FitExample], w: CalibrationWeights) -> float:
    """The fit objective (mean NLL + L2), as `fit_weights` minimises it."""
    return softmax_rank._loss_and_grad(_params(w), examples)[0]


def spec_digest(spec: dict) -> str:
    """SHA-256 over the parts of the spec that determine a prediction."""
    frozen = {k: spec[k] for k in ("spec_version", "variant", "rule", "weights", "abstention_tau")}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode("utf-8")).hexdigest()


def assert_train_only(matrices: list[SupportMatrix]) -> None:
    bad = [m.question_id for m in matrices if not m.question_id.startswith("train_")]
    if bad:
        raise SystemExit(
            f"[v5_refit] fit input holds {len(bad)} non-train question(s) (e.g. {bad[:3]}) -- "
            "the refit must never see a test question"
        )


# ---- train-side diagnostics (descriptive; choose nothing) ------------------


def _v5_rank(feats, weights, _qid):
    return rank_options(feats, weights)


def rule_diagnostics(matrices: list[SupportMatrix], feats_fn, weights, rank_fn) -> dict:
    """Rank-1 accuracy, tie-aware expected accuracy (1/k when gold is among
    k exactly-tied top options), mean gold NLL, and the tie profile."""
    from src.eval.metrics import ece

    n = correct = n_tied = 0
    tie_aware = nll = 0.0
    picks: Counter = Counter()
    probs, gold_idx = [], []
    for m in matrices:
        if not m.gold_option:
            continue
        feats = feats_fn(m)
        ranked = rank_fn(feats, weights, m.question_id)
        z = {f.option_id: compute_logit(f, weights) for f in feats}
        top = max(z.values())
        tied = [o for o, val in z.items() if val == top]
        n += 1
        correct += ranked[0][0] == m.gold_option
        tie_aware += (1 / len(tied)) if m.gold_option in tied else 0.0
        if len(tied) > 1:
            n_tied += 1
            picks[ranked[0][0]] += 1
        p = dict(ranked)
        nll += -math.log(max(p[m.gold_option], 1e-12))
        oids = sorted(p)
        probs.append([p[o] for o in oids])
        gold_idx.append(oids.index(m.gold_option))
    return {
        "n": n,
        "accuracy": correct / n,
        "tie_aware_expected_accuracy": tie_aware / n,
        "mean_gold_nll": nll / n,
        "ece_equal_mass_5_bins": ece(probs, gold_idx, n_bins=5, binning="equal_mass")["ece"],
        "n_top_exactly_tied": n_tied,
        "tied_top_pick_by_letter": dict(sorted(picks.items())),
    }


def coverage_risk_refit(
    examples: list[tuple[SupportMatrix, list[OptionFeatures]]], weights, n_thresholds=20
):
    """`softmax_rank.coverage_risk_curve`, with V5-refit's tie-break deciding
    the top option (the threshold grid and risk definition are identical)."""
    per_q = []
    for m, feats in examples:
        top_id, top_p = rank_refit(feats, weights, m.question_id)[0]
        per_q.append((top_p, top_id == m.gold_option))
    p1 = [p for p, _ in per_q]
    lo, hi = min(p1), max(p1)
    taus = (
        [lo] if lo == hi else [lo + (hi - lo) * i / (n_thresholds - 1) for i in range(n_thresholds)]
    )
    n = len(per_q)
    curve = []
    for tau in taus:
        covered = [c for p, c in per_q if p >= tau]
        k = len(covered)
        curve.append(
            {
                "tau": tau,
                "coverage": k / n,
                "risk": (1.0 - sum(covered) / k) if k else 0.0,
                "n_covered": k,
            }
        )
    return curve


def bootstrap_refit(examples, bounds, n_resamples=N_BOOTSTRAP, seed=BOOTSTRAP_SEED) -> dict:
    """`softmax_rank.bootstrap_weight_uncertainty`'s resampling, under
    `bounds`, plus how often each evidential weight sits on its 0 bound."""
    rng = random.Random(seed)
    n = len(examples)
    fits = []
    for _ in range(n_resamples):
        resample = [examples[rng.randrange(n)] for _ in range(n)]
        fits.append(fit_weights_bounded(resample, bounds))
    out: dict = {}
    for name in ("alpha", "beta", "gamma", "delta", "temperature"):
        vals = [getattr(w, name) for w in fits]
        mean = sum(vals) / len(vals)
        out[name] = {
            "mean": mean,
            "std": math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals)),
            "min": min(vals),
            "max": max(vals),
        }
        if name in EVIDENTIAL:
            out[name]["fraction_at_zero_bound"] = sum(1 for x in vals if x <= 0.0) / len(vals)
    out["n_resamples"] = n_resamples
    out["seed"] = seed
    return out


# ---- fit (train only) ------------------------------------------------------


def cmd_fit(_args) -> None:
    from src.config import load_config
    from src.eval.results import git_provenance

    matrices = load_matrices_train()
    fit_set, held_out = split_like_v5(matrices)

    # 1. V5's own fit reproduces bit-for-bit from this split and this code.
    v5_cal = load_config(V5_CONFIG_PATH).calibration
    v5_w = CalibrationWeights(
        alpha=v5_cal.alpha,
        beta=v5_cal.beta,
        gamma=v5_cal.gamma,
        delta=v5_cal.delta,
        temperature=v5_cal.temperature,
    )
    v5_ex = [(extract_option_features(m), m.gold_option) for m in fit_set if m.gold_option]
    repro = softmax_rank.fit_weights(v5_ex)
    repro_bounded = fit_weights_bounded(v5_ex, softmax_rank._BOUNDS)
    for name in ("alpha", "beta", "gamma", "delta", "temperature"):
        if getattr(repro, name) != getattr(v5_w, name) or getattr(repro_bounded, name) != getattr(
            v5_w, name
        ):
            raise SystemExit(
                f"[v5_refit] V5's fit does not reproduce on {name} -- refusing to refit"
            )
    print(f"[v5_refit] V5's weights reproduce exactly from the fit half: {v5_w}")

    # 2. The refit.
    r_ref = rerank_reference(fit_set)
    fit_ex = [(refit_features(m, r_ref), m.gold_option) for m in fit_set if m.gold_option]
    weights = fit_weights_bounded(fit_ex, REFIT_BOUNDS)
    binding = {name: getattr(weights, name) <= REFIT_BOUNDS[name][0] for name in EVIDENTIAL}
    print(f"[v5_refit] r_ref={r_ref:.6f}  refit weights: {weights}  at 0 bound: {binding}")

    ho_pairs = [(m, refit_features(m, r_ref)) for m in held_out if m.gold_option]
    curve = coverage_risk_refit(ho_pairs, weights)
    tau = choose_abstention_threshold(curve, target_risk=TARGET_RISK)

    # 3. Descriptive diagnostics -- reported, never used to choose.
    unconstrained = fit_weights_bounded(fit_ex, softmax_rank._BOUNDS)
    long_fit = fit_weights_bounded(fit_ex, REFIT_BOUNDS, max_iters=CONVERGENCE_ITERS)
    print(f"[v5_refit] bootstrapping ({N_BOOTSTRAP} resamples) ...", flush=True)
    boot = bootstrap_refit(fit_ex, REFIT_BOUNDS)

    def refit_fn(m):
        return refit_features(m, r_ref)

    diagnostics = {}
    for split_name, part in (("fit_half", fit_set), ("held_out_half", held_out)):
        diagnostics[split_name] = {
            "V5_rule": rule_diagnostics(part, extract_option_features, v5_w, _v5_rank),
            "V5_refit_rule": rule_diagnostics(part, refit_fn, weights, rank_refit),
        }

    spec = {
        "spec_version": SPEC_VERSION,
        "variant": VARIANT,
        "adr": "ADR-043",
        "status": (
            "frozen on the train slice before any test-set evaluation of V5-refit; "
            "apply only via scripts/run_phase7_v5_refit.ps1"
        ),
        "rule": {
            "logit": "z = alpha*s_pos - beta*s_neg + gamma*kg_support_count + delta*rerank_c",
            "s_pos_s_neg_source": "counted from SupportMatrix.cells (not option_support; ADR-042)",
            "rerank_c": (
                "mean MedCPT rerank score over the option's SUPPORT+CONTRADICT cells minus "
                "rerank_reference; 0.0 (= the reference) for an option with no such cell"
            ),
            "rerank_reference": r_ref,
            "rerank_reference_definition": (
                "median raw mean_rerank_score over the fit half's options with >= 1 "
                "SUPPORT/CONTRADICT cell"
            ),
            "probabilities": "softmax(z / temperature)",
            "bounds": {k: list(v) for k, v in REFIT_BOUNDS.items()},
            "optimizer": (
                "src.calibration.softmax_rank.fit_weights verbatim (Adam, lr/betas/eps/"
                "500 iterations/init/L2 unchanged), projection onto `bounds` each step"
            ),
            "tie_break": {
                "method": "seeded uniform shuffle, then stable sort on p",
                "seed": TIE_BREAK_SEED,
                "rng_key": "v5-refit-tiebreak:<seed>:<question_id>",
            },
            "abstention": (
                f"choose_abstention_threshold on the held-out half, target_risk={TARGET_RISK}"
            ),
        },
        "weights": weights.model_dump(),
        "abstention_tau": tau,
        "evidential_weight_at_zero_bound": binding,
        "fit_source_run": FIT_RUN_DIR.name,
        "fit_question_ids": [m.question_id for m in fit_set],
        "held_out_question_ids": [m.question_id for m in held_out],
        "v5_reproduction": {
            "weights_reproduced_exactly": True,
            "v5_weights": v5_w.model_dump(),
            "note": "softmax_rank.fit_weights and fit_weights_bounded(_BOUNDS) both reproduce "
            "configs/v5.yaml bit-for-bit from the fit half",
        },
        "descriptive_not_used": {
            "unconstrained_equals_refit": unconstrained == weights,
            "effective_coefficients_w_over_T": {
                name: {
                    k: getattr(w, k) / w.temperature for k in ("alpha", "beta", "gamma", "delta")
                }
                for name, w in (
                    ("V5", v5_w),
                    ("refit", weights),
                    ("refit_5000_iterations", long_fit),
                    ("unconstrained_refit_same_features", unconstrained),
                )
            },
            "effective_coefficients_note": (
                "softmax(z/T) depends on (alpha..delta)/T only. The L2 penalty acts on "
                "alpha..delta but not T, so the weights' overall scale and T trade off: 500 "
                "and 5000 iterations can differ in scale while agreeing in w/T"
            ),
            "unconstrained_refit_same_features": unconstrained.model_dump(),
            "unconstrained_objective": objective(fit_ex, unconstrained),
            "refit_objective": objective(fit_ex, weights),
            "refit_5000_iterations": long_fit.model_dump(),
            "refit_5000_iterations_objective": objective(fit_ex, long_fit),
            "weight_bootstrap_under_refit_bounds": boot,
            "coverage_risk_curve_held_out": curve,
            "rule_diagnostics": diagnostics,
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git": git_provenance(),
    }
    spec["spec_sha256"] = spec_digest(spec)
    SPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPEC_PATH.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"[v5_refit] wrote {SPEC_PATH}  spec_sha256={spec['spec_sha256']}")
    print(f"[v5_refit] abstention tau={tau:.6f}")
    print(f"[v5_refit] unconstrained (same features): {unconstrained}")
    print(f"[v5_refit] 5000-iter refit: {long_fit}")
    for k in ("alpha", "beta", "gamma"):
        print(f"[v5_refit] bootstrap {k}: {boot[k]}")
    for split_name, d in diagnostics.items():
        for rule, r in d.items():
            print(f"[v5_refit] {split_name:14s} {rule:14s} {r}")


def load_matrices_train() -> list[SupportMatrix]:
    from scripts.fit_calibration import load_matrices

    matrices = load_matrices(FIT_RUN_DIR)
    assert_train_only(matrices)
    return matrices


def split_like_v5(matrices):
    """scripts/fit_calibration.py's split, checked against the split V5 was
    actually fit on (the Phase 5 output JSON)."""
    from scripts.fit_calibration import split_fit_held_out

    fit_set, held_out = split_fit_held_out(matrices)
    p5 = json.loads(PHASE5_JSON.read_text(encoding="utf-8"))
    if [m.question_id for m in fit_set] != [r["question_id"] for r in p5["ranked_output_fit"]]:
        raise SystemExit("[v5_refit] fit half differs from the one V5 was fit on")
    if [m.question_id for m in held_out] != [
        r["question_id"] for r in p5["ranked_output_held_out"]
    ]:
        raise SystemExit("[v5_refit] held-out half differs from V5's")
    return fit_set, held_out


# ---- apply (test) ----------------------------------------------------------


def load_spec(path: Path | None = None) -> dict:
    path = path or SPEC_PATH
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec_digest(spec) != spec.get("spec_sha256"):
        raise SystemExit(f"[v5_refit] {path}: spec_sha256 does not match its content")
    return spec


def rescore_rows(rows: list[dict], spec: dict) -> list:
    """V5 prediction rows -> V5-refit `QuestionRecord`s. Pure: no I/O."""
    from src.eval.harness import QuestionRecord

    weights = CalibrationWeights(**spec["weights"])
    r_ref = spec["rule"]["rerank_reference"]
    seed = spec["rule"]["tie_break"]["seed"]
    tau = spec["abstention_tau"]
    out = []
    for rec in rows:
        matrix = SupportMatrix(**rec["agent_matrix"])
        ranked_pairs = rank_refit(refit_features(matrix, r_ref), weights, matrix.question_id, seed)
        ranked = [oid for oid, _ in ranked_pairs]
        pv = [p for _, p in ranked_pairs]
        pred = ranked[0]
        citations = cite_all(matrix)
        claim_records = [
            {**cr, "predicted_option": cr["option_id"] == pred} for cr in rec["claim_records"]
        ]
        out.append(
            QuestionRecord(
                question_id=matrix.question_id,
                gold_option=rec["gold_option"],
                pred_option=pred,
                ranked=ranked,
                p_vector=pv,
                ranking_source="calibrated_softmax",
                abstained=pv[0] < tau,
                citations=citations,
                invalid_citations=list(rec["invalid_citations"]),
                grounded=len(citations.get(pred, [])) > 0,
                rationale_claims=[cr["claim"] for cr in claim_records if cr["predicted_option"]],
                claim_records=claim_records,
                status=rec["status"],
                n_llm_calls=rec["n_llm_calls"],
                agent_matrix=rec["agent_matrix"],
            )
        )
    return out


def reflag_hallucination(v5_hall: dict, pred_by_q: dict[str, str | None]) -> dict:
    """V5's own per-claim NLI judgements, re-aggregated under V5-refit's
    predicted option. Entailment depends only on (claim, cited evidence),
    which re-scoring does not change -- zero new NLI calls."""
    from src.eval.nli import _aggregate

    per_claim = [
        {**row, "predicted_option": row["option_id"] == pred_by_q.get(row["question_id"])}
        for row in v5_hall["per_claim"]
    ]
    pred = _aggregate([r for r in per_claim if r["predicted_option"]])
    allc = _aggregate(per_claim)
    return {
        "primary": "hallucination_rate_predicted_option",
        "hallucination_rate_predicted_option": pred["hallucination_rate"],
        "n_claims_predicted_option": pred["n_claims"],
        "n_ungrounded_predicted_option": pred["n_ungrounded"],
        "n_uncited_predicted_option": pred["n_uncited"],
        "hallucination_rate_all_support_cells": allc["hallucination_rate"],
        "n_claims_all_support_cells": allc["n_claims"],
        "n_ungrounded_all_support_cells": allc["n_ungrounded"],
        "n_uncited_all_support_cells": allc["n_uncited"],
        "threshold": v5_hall["threshold"],
        "nli_model": v5_hall["nli_model"],
        "per_claim": per_claim,
        "note": "re-aggregated from V5's per_claim NLI judgements under V5-refit's predicted "
        "option (same claims, same evidence); zero new NLI calls",
    }


def _validate_v5_run(v5: dict) -> None:
    from scripts.report import code_matches_launch_commit

    expect = {"variant": "V5", "dataset": "medqa", "split": "test", "limit": 500, "seed": 42}
    problems = [f"{k}={v5.get(k)!r}" for k, val in expect.items() if v5.get(k) != val]
    if v5["metrics"].get("n_evaluated") != 500:
        problems.append(f"n_evaluated={v5['metrics'].get('n_evaluated')!r}")
    if v5.get("superseded"):
        problems.append("superseded")
    ok, why = code_matches_launch_commit(v5)
    if not ok:
        problems.append(why)
    if problems:
        raise SystemExit(f"[v5_refit] grid V5 run is not a valid input: {problems}")


def cmd_apply(_args) -> None:
    from src.eval.harness import score_run
    from src.eval.results import RunResult, git_provenance, make_run_id

    spec = load_spec()
    v5 = json.loads((V5_RUN_DIR / "metrics.json").read_text(encoding="utf-8"))
    _validate_v5_run(v5)
    rows = [
        json.loads(line)
        for line in (V5_RUN_DIR / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    qids = [r["question_id"] for r in rows]
    if not all(q.startswith("test_") for q in qids) or len(set(qids)) != 500:
        raise SystemExit("[v5_refit] V5 rows are not 500 distinct test questions")
    if set(qids) & (set(spec["fit_question_ids"]) | set(spec["held_out_question_ids"])):
        raise SystemExit("[v5_refit] test/train overlap")

    # upstream check: V5's own rule reproduces V5's stored rankings exactly
    v5_cfg = yaml.safe_load((V5_RUN_DIR / "config.yaml").read_text(encoding="utf-8"))
    cal = v5_cfg["calibration"]
    v5_w = CalibrationWeights(
        **{k: cal[k] for k in ("alpha", "beta", "gamma", "delta", "temperature")}
    )
    for r in rows:
        replay = [
            o
            for o, _ in rank_options(
                extract_option_features(SupportMatrix(**r["agent_matrix"])), v5_w
            )
        ]
        if replay != r["ranked"]:
            raise SystemExit(f"[v5_refit] {r['question_id']}: V5's stored ranking does not replay")

    records = rescore_rows(rows, spec)
    metrics = score_run(
        records,
        dataset="medqa",
        n_ece_bins=cal["n_ece_bins"],
        ece_binning=cal["ece_binning"],
        compute_hallucination=False,
    )
    pred_by_q = {r.question_id: r.pred_option for r in records}
    metrics["hallucination"] = reflag_hallucination(v5["metrics"]["hallucination"], pred_by_q)
    metrics["hallucination_rate"] = metrics["hallucination"]["hallucination_rate_predicted_option"]
    metrics["hallucination_pending"] = False
    metrics["contamination_assertion"] = v5["metrics"]["contamination_assertion"]
    metrics["llm_calls_total"] = 0
    metrics["llm_calls_note"] = "derived from V5's saved SupportMatrices: zero LLM calls"
    metrics["n_pred_option_changed_vs_v5"] = sum(
        1 for r, rec in zip(rows, records, strict=True) if r["pred_option"] != rec.pred_option
    )
    weights = CalibrationWeights(**spec["weights"])
    tied = Counter()
    n_tied = 0
    for rec in records:
        feats = refit_features(SupportMatrix(**rec.agent_matrix), spec["rule"]["rerank_reference"])
        z = [compute_logit(f, weights) for f in feats]
        if sum(1 for x in z if x == max(z)) > 1:
            n_tied += 1
            tied[rec.pred_option] += 1
    metrics["n_top_exactly_tied"] = n_tied
    metrics["tied_top_pick_by_letter"] = dict(sorted(tied.items()))

    prov = git_provenance()
    run_id = make_run_id("medqa", VARIANT, sha7=prov["git_sha7"])
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg_doc = {
        **v5_cfg,
        "calibration": {**cal, **spec["weights"], "abstention_tau": spec["abstention_tau"]},
        "v5_refit": {
            "spec_path": str(SPEC_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "spec_sha256": spec["spec_sha256"],
            "rule": spec["rule"],
            "derived_from_run": v5["run_id"],
        },
    }
    cfg_text = yaml.safe_dump(cfg_doc, sort_keys=True)
    (run_dir / "config.yaml").write_text(cfg_text, encoding="utf-8")
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.model_dump(mode="json"), sort_keys=True) + "\n")
    result = RunResult(
        run_id=run_id,
        dataset="medqa",
        variant=VARIANT,
        split=v5["split"],
        limit=v5["limit"],
        seed=v5["seed"],
        **prov,
        config_hash=hashlib.sha256(cfg_text.encode("utf-8")).hexdigest(),
        metrics=metrics,
    ).model_dump(mode="json")
    result.update(
        derived_from=v5["run_id"],
        upstream_git_sha7=v5["git_sha7"],
        refit_spec=cfg_doc["v5_refit"]["spec_path"],
        refit_spec_sha256=spec["spec_sha256"],
        adr="ADR-043",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    m = metrics
    print(f"[v5_refit] wrote {run_dir}")
    print(
        f"  n={m['n_evaluated']} acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} "
        f"P@1={m['p_at_1']:.3f} P@2={m['p_at_2']:.3f} MRR={m['mrr']:.3f} ECE={m['ece']['ece']:.4f}"
    )
    print(
        f"  halluc(pred)={m['hallucination_rate']:.3f} grounded={m['grounded_rate']:.3f} "
        f"pred changed vs V5={m['n_pred_option_changed_vs_v5']} ties={n_tied} {dict(tied)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit", help="TRAIN ONLY: fit and freeze the spec")
    sub.add_parser("apply", help="TEST: re-score V5's matrices (launch via the .ps1 only)")
    args = ap.parse_args()
    {"fit": cmd_fit, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    main()
