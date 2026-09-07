"""Phase 5: temperature-scaled softmax ranking, weight fitting, and the
abstention coverage-risk curve.

`p = softmax(z / T)` (docs/01_ARCHITECTURE.md). Weights (alpha, beta, gamma,
delta) and temperature T are fit jointly on a dev split by minimizing mean
negative log-likelihood of the gold option -- a single fit does double duty
as both the z_i weighting and the calibration temperature, rather than the
two-stage "fit weights, then fit T on top" procedure common in the
classification-calibration literature (Guo et al. 2017), because here the
"model" being calibrated (z_i itself) doesn't exist independently of this
fit -- there is no pre-existing classifier logit to calibrate after the fact.

Fitting uses a hand-rolled Adam gradient descent on the analytic NLL gradient
below, NOT `scipy.optimize`: `scipy.optimize` (any method, confirmed down to a
bare `import scipy.optimize`) segfaults in this project's environment -- a
pre-existing binary-compatibility problem in the installed scipy/numpy/BLAS
stack, unrelated to this code, and not something to risk fixing by touching
shared native libraries while a multi-hour agent run shares the machine.
5 parameters and an analytic gradient make a dependency-free optimizer both
safe and transparent; see `_grad_and_loss` for the derivation.
"""

from __future__ import annotations

import math
import random

from src.calibration.scoring import CalibrationWeights, OptionFeatures, compute_logit

# Fitting example: this question's per-option features, and the gold option's id.
FitExample = tuple[list[OptionFeatures], str]

# Bounds keep the optimizer well-behaved on small, near-separable fits (a few
# integer-valued features and n in the tens can otherwise drive alpha/beta
# toward +-infinity in unconstrained logistic-regression-style fits).
_BOUNDS = {
    "alpha": (-10.0, 10.0),
    "beta": (-10.0, 10.0),
    "gamma": (-10.0, 10.0),
    "delta": (-10.0, 10.0),
    "log_t": (math.log(0.05), math.log(20.0)),
}
# Light L2 penalty on alpha/beta/gamma/delta only (never on temperature,
# which is a scale parameter, not a coefficient to shrink toward 0).
_L2_LAMBDA = 0.01
_ADAM_LR = 0.05
_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.999
_ADAM_EPS = 1e-8
_MAX_ITERS = 500


def softmax(logits: list[float], temperature: float) -> list[float]:
    """Numerically stable softmax(z / T). `temperature` must be > 0."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    scaled = [z / temperature for z in logits]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def rank_options(
    features: list[OptionFeatures], weights: CalibrationWeights
) -> list[tuple[str, float]]:
    """[(option_id, p_i), ...] sorted descending by p_i. `features` order
    does not matter; the return order is the ranking."""
    logits = [compute_logit(f, weights) for f in features]
    probs = softmax(logits, weights.temperature)
    paired = list(zip((f.option_id for f in features), probs, strict=True))
    return sorted(paired, key=lambda kv: kv[1], reverse=True)


def _loss_and_grad(params: list[float], examples: list[FitExample]) -> tuple[float, list[float]]:
    """Mean NLL of the gold option + light L2 penalty, and its analytic
    gradient w.r.t. (alpha, beta, gamma, delta, log_t).

    Derivation: let u_i = z_i / T, p = softmax(u), L = -log(p_gold) =
    -u_gold + logsumexp(u). Then dL/du_i = p_i - 1[i==gold]. Since
    dz_i/dalpha = s_pos_i (etc.) and du_i/dz_i = 1/T:
        dL/dalpha = sum_i (p_i - 1[i==gold]) * s_pos_i / T   (analogously beta/gamma/delta)
    And since u_i = z_i * exp(-log_t), du_i/d(log_t) = -u_i:
        dL/d(log_t) = -sum_i (p_i - 1[i==gold]) * u_i
    """
    alpha, beta, gamma, delta, log_t = params
    temperature = math.exp(log_t)
    weights = CalibrationWeights(
        alpha=alpha, beta=beta, gamma=gamma, delta=delta, temperature=temperature
    )

    total_loss = 0.0
    grad = [0.0, 0.0, 0.0, 0.0, 0.0]  # d/dalpha, d/dbeta, d/dgamma, d/ddelta, d/d(log_t)
    n_used = 0
    for features, gold_id in examples:
        logits = [compute_logit(f, weights) for f in features]
        us = [z / temperature for z in logits]
        probs = softmax(logits, temperature)
        gold_pos = next((i for i, f in enumerate(features) if f.option_id == gold_id), None)
        if gold_pos is None:
            continue  # gold option not in this question's feature list -- skip, don't crash
        n_used += 1
        total_loss += -math.log(max(probs[gold_pos], 1e-12))
        for i, f in enumerate(features):
            indicator = 1.0 if i == gold_pos else 0.0
            dL_du_i = probs[i] - indicator
            grad[0] += dL_du_i * f.s_pos / temperature
            grad[1] += dL_du_i * (-f.s_neg) / temperature
            grad[2] += dL_du_i * f.kg_support_count / temperature
            grad[3] += dL_du_i * f.mean_rerank_score / temperature
            grad[4] += -dL_du_i * us[i]

    denom = max(n_used, 1)
    reg = _L2_LAMBDA * (alpha**2 + beta**2 + gamma**2 + delta**2)
    loss = total_loss / denom + reg
    grad = [g / denom for g in grad]
    grad[0] += 2 * _L2_LAMBDA * alpha
    grad[1] += 2 * _L2_LAMBDA * beta
    grad[2] += 2 * _L2_LAMBDA * gamma
    grad[3] += 2 * _L2_LAMBDA * delta
    return loss, grad


def _clip(value: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))


def fit_weights(
    examples: list[FitExample],
    init: CalibrationWeights | None = None,
    max_iters: int = _MAX_ITERS,
) -> CalibrationWeights:
    """Fits alpha, beta, gamma, delta, temperature jointly by minimizing mean
    NLL of the gold option (plus a light L2 penalty on alpha/beta/gamma/delta
    -- see module docstring) via a hand-rolled Adam gradient descent on the
    analytic gradient in `_loss_and_grad` (not `scipy.optimize` -- see the
    module docstring for why). Raises ValueError on empty `examples`.

    `init` seeds the optimizer; the module default (alpha=beta=1, gamma=delta=0,
    T=1) is "sanity ranking" (the Phase 4 pre-calibration s+-s- argmax) with no
    KG/rerank contribution and no temperature scaling.
    """
    if not examples:
        raise ValueError("fit_weights: examples must be non-empty")
    init = init or CalibrationWeights()
    params = [init.alpha, init.beta, init.gamma, init.delta, math.log(init.temperature)]
    param_names = ("alpha", "beta", "gamma", "delta", "log_t")

    m = [0.0] * 5
    v = [0.0] * 5
    for t in range(1, max_iters + 1):
        _loss, grad = _loss_and_grad(params, examples)
        for i in range(5):
            m[i] = _ADAM_BETA1 * m[i] + (1 - _ADAM_BETA1) * grad[i]
            v[i] = _ADAM_BETA2 * v[i] + (1 - _ADAM_BETA2) * grad[i] ** 2
            m_hat = m[i] / (1 - _ADAM_BETA1**t)
            v_hat = v[i] / (1 - _ADAM_BETA2**t)
            params[i] -= _ADAM_LR * m_hat / (math.sqrt(v_hat) + _ADAM_EPS)
            params[i] = _clip(params[i], _BOUNDS[param_names[i]])

    alpha, beta, gamma, delta, log_t = params
    return CalibrationWeights(
        alpha=alpha, beta=beta, gamma=gamma, delta=delta, temperature=math.exp(log_t)
    )


def bootstrap_weight_uncertainty(
    examples: list[FitExample],
    n_resamples: int = 200,
    seed: int = 42,
    init: CalibrationWeights | None = None,
) -> dict[str, dict]:
    """Resamples `examples` with replacement `n_resamples` times, refitting
    weights each time, and reports the spread per parameter -- the project's
    established bootstrap-for-uncertainty convention
    (`src.retrieval.metrics.paired_bootstrap_ci`), applied to weight-fitting
    uncertainty instead of a metric difference.

    Purpose (docs/05_DECISIONS.md ADR this phase adds): with `kg_support_count`
    near-zero for almost every option (§5.5 finding 2: 1/108 audit cells cite
    KG), gamma's likelihood surface is nearly flat -- the POINT ESTIMATE alone
    cannot show that. A wide or degenerate bootstrap spread on gamma IS the
    evidence that it is unidentifiable from this data, independent confirmation
    of the KG-invisibility finding from a different direction than §5.5's raw
    count. Report this spread whenever gamma is discussed, not just the point
    estimate.

    Returns `{"alpha": {...}, "beta": {...}, "gamma": {...}, "delta": {...},
    "temperature": {...}}`, each with `{"mean", "std", "min", "max"}` over the
    n_resamples refits (temperature reported directly, not in log-space).
    """
    if not examples:
        raise ValueError("bootstrap_weight_uncertainty: examples must be non-empty")
    n = len(examples)
    rng = random.Random(seed)
    fits: list[CalibrationWeights] = []
    for _ in range(n_resamples):
        resample = [examples[rng.randrange(n)] for _ in range(n)]
        fits.append(fit_weights(resample, init=init))

    out: dict[str, dict] = {}
    for name in ("alpha", "beta", "gamma", "delta", "temperature"):
        values = [getattr(w, name) for w in fits]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        out[name] = {"mean": mean, "std": math.sqrt(var), "min": min(values), "max": max(values)}
    return out


def coverage_risk_curve(
    examples: list[FitExample], weights: CalibrationWeights, n_thresholds: int = 20
) -> list[dict]:
    """For each of `n_thresholds` candidate abstention thresholds tau (evenly
    spaced over the observed p1 range), reports coverage (fraction of
    questions NOT abstained, i.e. p1 >= tau) and risk (error rate among the
    non-abstained questions; 0.0 when coverage is 0 -- no answered questions
    to be wrong about, not undefined).

    Returns `[{"tau", "coverage", "risk", "n_covered"}, ...]`, tau ascending.
    """
    if not examples:
        raise ValueError("coverage_risk_curve: examples must be non-empty")
    per_question = []
    for features, gold_id in examples:
        ranked = rank_options(features, weights)
        top_id, top_p = ranked[0]
        per_question.append((top_p, top_id == gold_id))

    p1_values = [p for p, _ in per_question]
    lo, hi = min(p1_values), max(p1_values)
    if lo == hi:
        taus = [lo]
    else:
        taus = [lo + (hi - lo) * i / (n_thresholds - 1) for i in range(n_thresholds)]

    n = len(per_question)
    curve = []
    for tau in taus:
        covered = [correct for p1, correct in per_question if p1 >= tau]
        n_covered = len(covered)
        coverage = n_covered / n
        risk = (1.0 - sum(covered) / n_covered) if n_covered else 0.0
        curve.append({"tau": tau, "coverage": coverage, "risk": risk, "n_covered": n_covered})
    return curve


def choose_abstention_threshold(curve: list[dict], target_risk: float = 0.10) -> float:
    """The largest tau in `curve` whose risk is <= target_risk, preferring
    higher coverage among ties. Falls back to the tau with the lowest risk
    (regardless of target) if no tau meets the target -- an explicit,
    reported fallback, not a silent one. `target_risk` is a chosen default
    (docs/04_EVALUATION.md specifies no target); callers needing a different
    target pass one."""
    if not curve:
        raise ValueError("choose_abstention_threshold: curve must be non-empty")
    meeting = [row for row in curve if row["risk"] <= target_risk]
    if meeting:
        return max(meeting, key=lambda row: row["coverage"])["tau"]
    return min(curve, key=lambda row: row["risk"])["tau"]


def fit_temperature_only(
    logits: list[list[float]],
    gold_idx: list[int],
    t_min: float = 0.05,
    t_max: float = 2000.0,
    n_grid: int = 6000,
) -> float:
    """Fits a SINGLE scalar temperature T minimizing mean NLL of the gold
    option under `softmax(logit / T)`, via a log-spaced grid search over
    `[t_min, t_max]`.

    This is the classical two-stage temperature-scaling procedure (Guo et al.
    2017: fix the logits, fit only T) -- used for the temperature-scaled
    raw-logprob-baseline ablation (docs/05_DECISIONS.md's Phase 5 ADR), which
    isolates how much of any ECE gap is "temperature scaling alone" vs. "the
    evidential features add something beyond temperature scaling." A grid
    search, not gradient descent, because a single scalar parameter does not
    need more machinery and a grid search cannot diverge or get stuck.

    **`t_max` matters more than it looks.** When the logits carry little
    information correlated with the gold label (as raw single-shot LLM
    confidence on this project's MedQA slice turned out to), the NLL surface
    flattens toward `log(n_options)` (the uniform-prediction NLL) as T grows,
    with no interior minimum -- the fit-optimal T can be genuinely large (this
    project measured ~92 against a naive `t_max=20` default that silently
    saturated the search at its boundary and produced a materially wrong,
    over-confident "optimal" T -- caught only by manually widening the range
    and checking the NLL kept improving past the old bound). **Callers MUST
    check whether the returned T sits at or near `t_max` and, if so, re-run
    with a wider range before trusting the result** -- this function does not
    self-check that, since it has no way to know what "near the boundary"
    should mean for a given caller's tolerance.

    `logits` need not come from `compute_logit` -- e.g. `log(raw_probability)`
    per option is a valid logit vector for this purpose: softmax is
    shift-invariant, and `log(p_i) = u_i - logsumexp(u)` is `u_i` shifted by a
    PER-QUESTION CONSTANT, so `softmax(log(p)/T) == softmax(u/T)` for any T
    -- temperature-scaling recovered log-probabilities gives the exact answer
    you'd get from the true (unobserved) original logits. Verified to
    floating-point precision in `tests/test_calibration_softmax_rank.py`.

    Raises ValueError on empty input.
    """
    if not logits or len(logits) != len(gold_idx):
        raise ValueError("fit_temperature_only: logits/gold_idx must be non-empty and same length")
    log_lo, log_hi = math.log(t_min), math.log(t_max)
    best_t, best_loss = t_min, math.inf
    for i in range(n_grid):
        log_t = log_lo + (log_hi - log_lo) * i / (n_grid - 1)
        t = math.exp(log_t)
        total = 0.0
        for logit_row, gold in zip(logits, gold_idx, strict=True):
            probs = softmax(logit_row, t)
            total += -math.log(max(probs[gold], 1e-12))
        mean_nll = total / len(logits)
        if mean_nll < best_loss:
            best_loss, best_t = mean_nll, t
    return best_t
