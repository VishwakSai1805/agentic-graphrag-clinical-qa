"""Decomposition of V5's calibrated score z into its four terms -- which
term actually decides V5's ranking? Read-only over V5's `predictions.jsonl`
and `configs/v5.yaml`: no new runs, no GPU, no LLM calls
(`docs/04_EVALUATION.md` §5.0a, ADR-041).

V5 ranks options by `softmax(z / T)` with (`src/calibration/scoring.py:
compute_logit`)

    z_i = alpha*s_pos_i - beta*s_neg_i + gamma*kg_support_count_i
          + delta*mean_rerank_score_i

and this script splits z into those four additive terms
(`alpha_s_pos`, `neg_beta_s_neg`, `gamma_kg`, `delta_rerank`):

  (i)   the fitted weights (configs/v5.yaml, cross-checked field-by-field
        against the V5 run's own resolved `config.yaml`);
  (ii)  each term's variance contribution to z, as Cov(t_k, z) / Var(z) --
        an exact additive decomposition (the four shares sum to 1). Reported
        pooled over all 2000 option rows AND within-question (every term
        demeaned per question). The within-question figure is the one that
        governs ranking: softmax is shift-invariant per question, so only
        between-option differences inside a question can move p;
  (iii) how often argmax z differs from argmax alpha*s_pos alone -- plus the
        sign-free argmax s_pos (alpha is fitted NEGATIVE, so argmax
        alpha*s_pos is argmin s_pos), leave-one-term-out argmax changes, and
        single-term tie-aware accuracies;
  (iv)  on the questions where a zero-SUPPORT option beat a supported one
        (`v5_diagnostics.json` (c): 68), which term supplied the winning
        margin over the best-scoring supported option.

Tie-breaking: V5's `rank_options` sorts by p descending with Python's stable
sort over `matrix.options` order, so an exact tie goes to the earlier option
letter. Every argmax here uses that same rule (`_argmax_first`); a tie-aware
set version (`_argmax_set`) is reported wherever ties are common.

Before any statistic is computed, V5's stored `ranked` and `p_vector` are
reproduced from the matrices + weights for every question (exact order,
|dp| < 1e-12) -- the decomposition refuses to run on a scoring path that does
not reproduce the run.

    uv run python -m scripts.analysis.calibration_decomposition
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from src.agents.state import SupportMatrix  # noqa: E402
from src.calibration.scoring import (  # noqa: E402
    CalibrationWeights,
    compute_logit,
    extract_option_features,
)
from src.calibration.softmax_rank import rank_options  # noqa: E402
from src.config import load_config  # noqa: E402

V5_RUN_ID = "medqa_V5_41f7e14-dirty_20260909T103452Z"
V5_RUN_DIR = REPO_ROOT / "results" / V5_RUN_ID
V5_CONFIG = REPO_ROOT / "configs" / "v5.yaml"
V5_DIAGNOSTICS = REPO_ROOT / "results" / "phase7_grid" / "v5_diagnostics.json"
OUT_PATH = REPO_ROOT / "results" / "phase7_grid" / "calibration_decomposition.json"

TERMS = ("alpha_s_pos", "neg_beta_s_neg", "gamma_kg", "delta_rerank")
WEIGHT_NAMES = ("alpha", "beta", "gamma", "delta", "temperature")


def _weights() -> CalibrationWeights:
    cal = load_config(V5_CONFIG).calibration
    w = CalibrationWeights(
        alpha=cal.alpha,
        beta=cal.beta,
        gamma=cal.gamma,
        delta=cal.delta,
        temperature=cal.temperature,
    )
    run_cfg = yaml.safe_load((V5_RUN_DIR / "config.yaml").read_text(encoding="utf-8"))
    run_cal = run_cfg["calibration"]
    for name in WEIGHT_NAMES:
        if run_cal[name] != getattr(w, name):
            raise SystemExit(
                f"[calibration_decomposition] {name}: configs/v5.yaml={getattr(w, name)!r} "
                f"but {V5_RUN_ID}/config.yaml={run_cal[name]!r} -- refusing to decompose "
                "with weights the run did not use"
            )
    return w


def _terms(f, w: CalibrationWeights) -> dict[str, float]:
    return {
        "alpha_s_pos": w.alpha * f.s_pos,
        "neg_beta_s_neg": -w.beta * f.s_neg,
        "gamma_kg": w.gamma * f.kg_support_count,
        "delta_rerank": w.delta * f.mean_rerank_score,
    }


def _load_questions(w: CalibrationWeights) -> list[dict]:
    lines = (V5_RUN_DIR / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    questions = []
    for r in rows:
        feats = extract_option_features(SupportMatrix(**r["agent_matrix"]))
        ranked = rank_options(feats, w)
        if [o for o, _ in ranked] != r["ranked"] or any(
            abs(p - q) > 1e-12 for (_, p), q in zip(ranked, r["p_vector"], strict=True)
        ):
            raise SystemExit(
                f"[calibration_decomposition] {r['question_id']}: recomputed ranking "
                f"{ranked} does not reproduce the stored ranked={r['ranked']} "
                f"p_vector={r['p_vector']}"
            )
        terms = {f.option_id: _terms(f, w) for f in feats}
        z = {f.option_id: compute_logit(f, w) for f in feats}
        for oid in z:
            if abs(sum(terms[oid][k] for k in TERMS) - z[oid]) > 1e-12:
                raise SystemExit(
                    f"[calibration_decomposition] {r['question_id']}/{oid}: terms != z"
                )
        questions.append(
            {
                "question_id": r["question_id"],
                "gold": r["gold_option"],
                "pred": r["pred_option"],
                "status": r["status"],
                "order": [f.option_id for f in feats],
                "feats": {f.option_id: f for f in feats},
                "terms": terms,
                "z": z,
            }
        )
    for q in questions:
        if _argmax_first(q["z"], q["order"]) != q["pred"]:
            raise SystemExit(f"[calibration_decomposition] {q['question_id']}: argmax z != pred")
    return questions


def _argmax_first(vals: dict[str, float], order: list[str]) -> str:
    """V5's own tie-break: the earliest option (in matrix.options order) at the max."""
    best = max(vals[o] for o in order)
    return next(o for o in order if vals[o] == best)


def _argmax_set(vals: dict[str, float], order: list[str]) -> set[str]:
    best = max(vals[o] for o in order)
    return {o for o in order if vals[o] == best}


def _cov(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    return sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / n


def _demean(vals: list[float]) -> list[float]:
    m = sum(vals) / len(vals)
    return [v - m for v in vals]


# --------------------------------------------------------------------------
# (ii) variance decomposition
# --------------------------------------------------------------------------


def variance_decomposition(questions: list[dict], *, within: bool) -> dict:
    cols: dict[str, list[float]] = {k: [] for k in TERMS}
    zs: list[float] = []
    ranges: dict[str, list[float]] = {k: [] for k in TERMS}
    for q in questions:
        for k in TERMS:
            vals = [q["terms"][o][k] for o in q["order"]]
            ranges[k].append(max(vals) - min(vals))
            cols[k].extend(_demean(vals) if within else vals)
        zv = [q["z"][o] for o in q["order"]]
        zs.extend(_demean(zv) if within else zv)

    var_z = _cov(zs, zs)
    terms_out = {}
    for k in TERMS:
        var_k = _cov(cols[k], cols[k])
        cov_kz = _cov(cols[k], zs)
        terms_out[k] = {
            "var_term": var_k,
            "sd_term": math.sqrt(var_k),
            "cov_with_z": cov_kz,
            "share_of_var_z": cov_kz / var_z,
            "mean_within_question_range": sum(ranges[k]) / len(ranges[k]),
        }
    return {
        "scope": (
            "within-question (each term demeaned per question -- the ranking-relevant "
            "decomposition, softmax is shift-invariant per question)"
            if within
            else "pooled over all option rows"
        ),
        "n_option_rows": len(zs),
        "var_z": var_z,
        "terms": terms_out,
        "share_sum_check": sum(t["share_of_var_z"] for t in terms_out.values()),
        "covariance_matrix": {
            k1: {k2: _cov(cols[k1], cols[k2]) for k2 in TERMS} for k1 in TERMS
        },
    }


# --------------------------------------------------------------------------
# (iii) argmax comparisons
# --------------------------------------------------------------------------


def _tie_aware_accuracy(questions: list[dict], score_fn) -> float:
    """Expected accuracy under a uniformly random tie-break: 1/|S| if gold is in
    the arg-max set S, else 0 -- strips the letter-order tie-break out."""
    total = 0.0
    for q in questions:
        s = _argmax_set(score_fn(q), q["order"])
        total += (1.0 / len(s)) if q["gold"] in s else 0.0
    return total / len(questions)


def _first_accuracy(questions: list[dict], score_fn) -> float:
    return sum(1 for q in questions if _argmax_first(score_fn(q), q["order"]) == q["gold"]) / len(
        questions
    )


def _vs_argmax_z(questions: list[dict], score_fn) -> dict:
    n = len(questions)
    differs = sum(1 for q in questions if _argmax_first(score_fn(q), q["order"]) != q["pred"])
    pred_not_in_set = sum(
        1 for q in questions if q["pred"] not in _argmax_set(score_fn(q), q["order"])
    )
    all_tied = sum(
        1 for q in questions if len(_argmax_set(score_fn(q), q["order"])) == len(q["order"])
    )
    return {
        "argmax_differs_from_argmax_z_same_tiebreak": differs,
        "argmax_differs_rate": differs / n,
        "v5_pred_not_among_tied_maximisers": pred_not_in_set,
        "v5_pred_not_among_tied_maximisers_rate": pred_not_in_set / n,
        "questions_where_all_options_tie_on_this_score": all_tied,
        "accuracy_letter_order_tiebreak": _first_accuracy(questions, score_fn),
        "accuracy_tie_aware": _tie_aware_accuracy(questions, score_fn),
    }


def argmax_comparisons(questions: list[dict]) -> dict:
    n = len(questions)

    def term_only(k):
        return lambda q: {o: q["terms"][o][k] for o in q["order"]}

    def z_without(k):
        return lambda q: {o: sum(q["terms"][o][j] for j in TERMS if j != k) for o in q["order"]}

    def z_score(q):
        return q["z"]

    def s_pos_score(q):
        return {o: float(q["feats"][o].s_pos) for o in q["order"]}

    tied = [q for q in questions if len(_argmax_set(q["z"], q["order"])) > 1]
    tied_at_zero = [q for q in tied if max(q["z"].values()) == 0.0]
    pred_no_signal = [
        q
        for q in questions
        if q["feats"][q["pred"]].s_pos == 0 and q["feats"][q["pred"]].s_neg == 0
    ]

    return {
        "n_questions": n,
        "requested_alpha_s_pos_alone": {
            "note": (
                "argmax of alpha*s_pos alone. alpha is fitted NEGATIVE, so this is "
                "argmin s_pos -- the option with the FEWEST SUPPORT cells."
            ),
            **_vs_argmax_z(questions, term_only("alpha_s_pos")),
        },
        "s_pos_alone_sign_free": {
            "note": (
                "argmax s_pos: the option with the MOST SUPPORT cells (the evidential direction)"
            ),
            **_vs_argmax_z(questions, s_pos_score),
        },
        "each_term_alone": {k: _vs_argmax_z(questions, term_only(k)) for k in TERMS},
        "leave_one_term_out": {
            k: {
                "pred_changes_when_term_removed": sum(
                    1
                    for q in questions
                    if _argmax_first(z_without(k)(q), q["order"]) != q["pred"]
                ),
                "accuracy_without_term_letter_order_tiebreak": _first_accuracy(
                    questions, z_without(k)
                ),
            }
            for k in TERMS
        },
        "full_z": {
            "accuracy_letter_order_tiebreak": _first_accuracy(questions, z_score),
            "accuracy_tie_aware": _tie_aware_accuracy(questions, z_score),
        },
        "top_z_ties": {
            "n_questions_top_z_tied": len(tied),
            "of_which_tied_max_is_exactly_zero": len(tied_at_zero),
            "tied_pred_letter_counts": dict(sorted(Counter(q["pred"] for q in tied).items())),
            "tied_gold_letter_counts": dict(sorted(Counter(q["gold"] for q in tied).items())),
            "tied_accuracy_letter_order_tiebreak": (
                sum(1 for q in tied if q["pred"] == q["gold"]) / len(tied) if tied else None
            ),
            "tied_accuracy_tie_aware": (_tie_aware_accuracy(tied, z_score) if tied else None),
        },
        "pred_option_has_no_support_and_no_contradict_cells": len(pred_no_signal),
    }


# --------------------------------------------------------------------------
# (iv) zero-support-beats-supported: which term won
# --------------------------------------------------------------------------


def zero_support_wins(questions: list[dict]) -> dict:
    cases = [
        q
        for q in questions
        if q["feats"][q["pred"]].s_pos == 0 and any(f.s_pos > 0 for f in q["feats"].values())
    ]
    expected = json.loads(V5_DIAGNOSTICS.read_text(encoding="utf-8"))[
        "diagnostic_c_grounded_rate_explanation"
    ]["of_those_where_some_OTHER_option_did_have_SUPPORT"]
    if len(cases) != expected:
        raise SystemExit(
            f"[calibration_decomposition] {len(cases)} zero-support wins, but "
            f"v5_diagnostics.json reports {expected} -- definitions have drifted"
        )

    per_case = []
    winning = Counter()
    decisive = Counter()
    n_no_single_decisive = 0
    delta_sums = {k: 0.0 for k in TERMS}
    for q in cases:
        p = q["pred"]
        supported_order = [o for o in q["order"] if q["feats"][o].s_pos > 0]
        s_star = _argmax_first(q["z"], supported_order)
        deltas = {k: q["terms"][p][k] - q["terms"][s_star][k] for k in TERMS}
        margin = q["z"][p] - q["z"][s_star]
        p_first = q["order"].index(p) < q["order"].index(s_star)
        if margin > 0:
            win = max(TERMS, key=lambda k: deltas[k])
        else:  # exact tie on z: P won only because it precedes S* in letter order
            win = "tie_break_letter_order"
        winning[win] += 1
        dec = []
        for k in TERMS:
            m_without = sum(deltas[j] for j in TERMS if j != k)
            if m_without < 0 or (m_without == 0 and not p_first):
                dec.append(k)
        for k in dec:
            decisive[k] += 1
        if not dec:
            n_no_single_decisive += 1
        for k in TERMS:
            delta_sums[k] += deltas[k]
        fp, fs = q["feats"][p], q["feats"][s_star]
        per_case.append(
            {
                "question_id": q["question_id"],
                "status": q["status"],
                "gold": q["gold"],
                "pred_zero_support": p,
                "best_supported": s_star,
                "pred_features": fp.model_dump(),
                "best_supported_features": fs.model_dump(),
                "z_pred": q["z"][p],
                "z_best_supported": q["z"][s_star],
                "margin": margin,
                "term_deltas_pred_minus_supported": deltas,
                "winning_term": win,
                "decisive_terms": dec,
            }
        )

    n = len(cases)
    return {
        "definition": (
            "pred option has s_pos == 0 while >= 1 other option has s_pos > 0 (identical to "
            "v5_diagnostics.json (c) 'of_those_where_some_OTHER_option_did_have_SUPPORT'). "
            "Compared against S* = the supported option with the highest z (V5's own "
            "tie-break) -- the hardest supported competitor P had to beat. "
            "winning_term = the term with the largest positive (P - S*) contribution to the "
            "margin; decisive_terms = terms whose removal alone flips P vs S*."
        ),
        "n_cases": n,
        "winning_term_counts": dict(winning.most_common()),
        "decisive_term_counts": {k: decisive.get(k, 0) for k in TERMS},
        "cases_with_no_single_decisive_term": n_no_single_decisive,
        "mean_term_delta_pred_minus_supported": {k: delta_sums[k] / n for k in TERMS},
        "pred_has_no_non_neutral_cells_z_exactly_zero": sum(
            1 for c in per_case if c["pred_features"]["s_neg"] == 0 and c["z_pred"] == 0.0
        ),
        "pred_has_contradict_cells": sum(1 for c in per_case if c["pred_features"]["s_neg"] > 0),
        "best_supported_z_negative": sum(1 for c in per_case if c["z_best_supported"] < 0),
        "best_supported_has_contradict_cells": sum(
            1 for c in per_case if c["best_supported_features"]["s_neg"] > 0
        ),
        "best_supported_s_pos_counts": dict(
            sorted(Counter(c["best_supported_features"]["s_pos"] for c in per_case).items())
        ),
        "pred_is_gold": sum(1 for c in per_case if c["pred_zero_support"] == c["gold"]),
        "best_supported_is_gold": sum(1 for c in per_case if c["best_supported"] == c["gold"]),
        "status_counts": dict(Counter(c["status"] for c in per_case)),
        "cases": per_case,
    }


def main() -> None:
    w = _weights()
    questions = _load_questions(w)
    out = {
        "run_id": V5_RUN_ID,
        "n": len(questions),
        "reproduction_check": (
            "V5's stored ranked + p_vector reproduced from agent_matrix + weights on "
            f"{len(questions)}/{len(questions)} questions (exact order, |dp| < 1e-12)"
        ),
        "i_fitted_weights": {
            **{name: getattr(w, name) for name in WEIGHT_NAMES},
            "source": "configs/v5.yaml, equal field-by-field to the V5 run's config.yaml",
            "logit": (
                "z = alpha*s_pos - beta*s_neg + gamma*kg_support_count "
                "+ delta*mean_rerank_score"
            ),
        },
        "feature_ranges": {
            "mean_rerank_score_nonzero_rows": _rerank_summary(questions),
        },
        "ii_variance_decomposition": {
            "within_question": variance_decomposition(questions, within=True),
            "pooled": variance_decomposition(questions, within=False),
        },
        "iii_argmax_comparisons": argmax_comparisons(questions),
        "iv_zero_support_wins": zero_support_wins(questions),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[calibration_decomposition] wrote {OUT_PATH}")
    summary = {k: v for k, v in out.items() if k != "iv_zero_support_wins"}
    summary["iv_zero_support_wins"] = {
        k: v for k, v in out["iv_zero_support_wins"].items() if k != "cases"
    }
    print(json.dumps(summary, indent=2))


def _rerank_summary(questions: list[dict]) -> dict:
    """mean_rerank_score over option rows that HAVE non-neutral cells (rows with
    none carry the 0.0 'no signal' default, src/calibration/scoring.py)."""
    vals = sorted(
        q["feats"][o].mean_rerank_score
        for q in questions
        for o in q["order"]
        if q["feats"][o].s_pos + q["feats"][o].s_neg > 0
    )
    n = len(vals)
    return {
        "n_option_rows_with_non_neutral_cells": n,
        "n_option_rows_without": sum(len(q["order"]) for q in questions) - n,
        "fraction_negative": sum(1 for v in vals if v < 0) / n,
        "min": vals[0],
        "median": vals[n // 2],
        "max": vals[-1],
        "mean": sum(vals) / n,
    }


if __name__ == "__main__":
    main()
