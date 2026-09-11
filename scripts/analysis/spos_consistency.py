"""ADR-042: is `option_support.s_pos` consistent with the SupportMatrix's own
cell list, and what does any disagreement do to the grid's support-based
numbers? Read-only over already-written runs -- no GPU, no LLM calls.

ADR-041 noted, without investigating, that on 2/500 V5 questions the
predicted option's `option_support.s_pos` is 0 while the matrix holds SUPPORT
cells for it. `s_pos` is supposed to be exactly the count of that option's
SUPPORT cells (`src/agents/graph.py: _aggregate_option_support`), and
`s_neg` its CONTRADICT count. This script checks that identity on every
option of every question in every agent run the paper's numbers come from:

  - the grid V5 run and the grid V4-conflict run (test, n=500), and
  - the Phase 5 fit run (train, n=120) that V5's weights were fit on.

For each disagreement it records the question's status and the stored vs
cell-derived counts. For the grid runs it then re-ranks every question with
the counts taken from the cell list -- each variant's own rule otherwise
unchanged (V5: configs/v5.yaml weights, softmax, letter tie-break; V4-conflict:
(s+ - s-) margin, letter tie-break; post-hoc-binary: V5 weights with s- = 0)
-- and reports stored vs corrected headline metrics and the ADR-041
support-based counts. Descriptive: this sizes a defect, it chooses nothing.

    uv run python -m scripts.analysis.spos_consistency
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.exploratory_family import RUN_DIRS as GRID_RUN_DIRS  # noqa: E402
from scripts.analysis.posthoc_binary import V5_WEIGHTS  # noqa: E402
from src.agents.state import SupportMatrix  # noqa: E402
from src.calibration.scoring import (  # noqa: E402
    OptionFeatures,
    compute_logit,
    extract_option_features,
)
from src.calibration.softmax_rank import rank_options  # noqa: E402
from src.eval.metrics import accuracy, macro_f1, mean_mrr, mean_precision_at_k  # noqa: E402

FIT_RUN_DIR = REPO_ROOT / "results" / "medqa_V4_a714fe0-dirty_20260907T121703Z"
OUT_PATH = REPO_ROOT / "results" / "phase7_grid" / "spos_consistency.json"
MEDQA_LETTERS = ("A", "B", "C", "D")


class _AV:
    def __init__(self, p, g):
        self.pred_option = p
        self.gold_option = g


def _load_rows(run_dir: Path) -> list[dict]:
    lines = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _matrix(row: dict) -> SupportMatrix:
    """Harness runs nest the matrix under `agent_matrix`; the Phase 5 fit run
    (written by `src.pipeline`) stores it at top level."""
    return SupportMatrix(**(row["agent_matrix"] if "agent_matrix" in row else row))


def cell_counts(matrix: SupportMatrix) -> dict[str, tuple[int, int]]:
    """option_id -> (SUPPORT cells, CONTRADICT cells), from the cell list."""
    counts = {oid: [0, 0] for oid in matrix.options}
    for c in matrix.cells:
        if c.option_id not in counts:
            continue
        if c.label == "SUPPORT":
            counts[c.option_id][0] += 1
        elif c.label == "CONTRADICT":
            counts[c.option_id][1] += 1
    return {oid: (v[0], v[1]) for oid, v in counts.items()}


def mismatches(matrix: SupportMatrix) -> list[dict]:
    """Options whose stored (s_pos, s_neg) differ from the cell-derived counts.
    An option absent from `option_support` is stored as (None, None) -- which
    `extract_option_features` reads as (0, 0)."""
    stored = {os.option_id: os for os in matrix.option_support}
    out = []
    for oid, (c_pos, c_neg) in cell_counts(matrix).items():
        os = stored.get(oid)
        s_pos = os.s_pos if os else None
        s_neg = os.s_neg if os else None
        if (s_pos or 0, s_neg or 0) != (c_pos, c_neg):
            out.append(
                {
                    "option_id": oid,
                    "stored_s_pos": s_pos,
                    "stored_s_neg": s_neg,
                    "cell_s_pos": c_pos,
                    "cell_s_neg": c_neg,
                }
            )
    return out


def corrected_features(matrix: SupportMatrix) -> list[OptionFeatures]:
    """`extract_option_features`, with s_pos / s_neg taken from the cell list.
    kg_support_count and mean_rerank_score are already cell-derived there."""
    counts = cell_counts(matrix)
    return [
        f.model_copy(update={"s_pos": counts[f.option_id][0], "s_neg": counts[f.option_id][1]})
        for f in extract_option_features(matrix)
    ]


def audit_run(run_dir: Path) -> dict:
    rows = _load_rows(run_dir)
    per_q = []
    statuses_all = Counter()
    for row in rows:
        m = _matrix(row)
        statuses_all[m.status] += 1
        mis = mismatches(m)
        if mis:
            per_q.append(
                {
                    "question_id": m.question_id,
                    "status": m.status,
                    "n_option_support_rows": len(m.option_support),
                    "pred_option": row.get("pred_option"),
                    "gold_option": m.gold_option,
                    "options_disagreeing": mis,
                    "last_note": m.notes[-1][:160] if m.notes else "",
                }
            )
    n_options = sum(len(_matrix(r).options) for r in rows)
    return {
        "run_dir": run_dir.name,
        "n_questions": len(rows),
        "n_options": n_options,
        "status_counts": dict(statuses_all),
        "n_questions_disagreeing": len(per_q),
        "n_options_disagreeing": sum(len(q["options_disagreeing"]) for q in per_q),
        "disagreeing_status_counts": dict(Counter(q["status"] for q in per_q)),
        "questions": per_q,
    }


# ---- re-ranking under each variant's own rule -------------------------------


def _rank_v5(feats: list[OptionFeatures], s_neg_zero: bool = False) -> list[str]:
    if s_neg_zero:
        feats = [f.model_copy(update={"s_neg": 0}) for f in feats]
    return [oid for oid, _ in rank_options(feats, V5_WEIGHTS)]


def _rank_margin(feats: list[OptionFeatures]) -> list[str]:
    # identical to src/eval/harness.py _agent_record's uncalibrated path
    margins = sorted(
        ((f.option_id, f.s_pos - f.s_neg) for f in feats), key=lambda kv: kv[1], reverse=True
    )
    return [oid for oid, _ in margins]


def _headline(rankings: list[list[str]], golds: list[str]) -> dict:
    preds = [r[0] if r else None for r in rankings]
    views = [_AV(p, g) for p, g in zip(preds, golds, strict=True)]
    return {
        "accuracy": accuracy(views),
        "macro_f1": macro_f1(views, list(MEDQA_LETTERS)),
        "p_at_1": mean_precision_at_k(rankings, golds, 1),
        "p_at_2": mean_precision_at_k(rankings, golds, 2),
        "mrr": mean_mrr(rankings, golds),
    }


def _support_counts(rows: list[dict], feats_fn, rank_fn) -> dict:
    """The ADR-041 / v5_diagnostics support-based counts under a feature source."""
    n_pred_no_signal = n_zero_support_win = n_top_tied = 0
    for row in rows:
        m = _matrix(row)
        feats = feats_fn(m)
        by_id = {f.option_id: f for f in feats}
        pred = rank_fn(feats)[0]
        pf = by_id[pred]
        n_pred_no_signal += pf.s_pos == 0 and pf.s_neg == 0
        n_zero_support_win += pf.s_pos == 0 and any(f.s_pos > 0 for f in feats)
        z = [compute_logit(f, V5_WEIGHTS) for f in feats]
        n_top_tied += sum(1 for v in z if v == max(z)) > 1
    return {
        "pred_option_has_no_signal": n_pred_no_signal,
        "zero_support_wins": n_zero_support_win,
        "v5_z_top_exactly_tied": n_top_tied,
    }


def rerank_impact(name: str, run_dir: Path, rank_fn, stored_dir: Path | None = None) -> dict:
    """`run_dir` supplies the SupportMatrices; `stored_dir` (default `run_dir`)
    the stored rankings -- they differ for post-hoc-binary, which is derived
    from V5's matrices but writes its own rankings."""
    rows = _load_rows(run_dir)
    golds = [r["gold_option"] or "" for r in rows]
    stored_by_q = {r["question_id"]: r["ranked"] for r in _load_rows(stored_dir or run_dir)}
    stored_rankings = [stored_by_q[r["question_id"]] for r in rows]
    # the stored ranking must be exactly reproducible from stored features first
    replay = [rank_fn(extract_option_features(_matrix(r))) for r in rows]
    n_replay_mismatch = sum(1 for a, b in zip(replay, stored_rankings, strict=True) if a != b)
    if n_replay_mismatch:
        raise SystemExit(
            f"[spos_consistency] {name}: {n_replay_mismatch} stored rankings not reproduced"
        )
    corrected = [rank_fn(corrected_features(_matrix(r))) for r in rows]
    changed = [
        {
            "question_id": r["question_id"],
            "gold_option": r["gold_option"],
            "stored_ranked": s,
            "corrected_ranked": c,
            "stored_correct": s[0] == r["gold_option"],
            "corrected_correct": c[0] == r["gold_option"],
        }
        for r, s, c in zip(rows, stored_rankings, corrected, strict=True)
        if s != c
    ]
    stored_m = _headline(stored_rankings, golds)
    corr_m = _headline(corrected, golds)
    return {
        "variant": name,
        "run_dir": run_dir.name,
        "stored_rankings_reproduced": f"{len(rows)}/{len(rows)}",
        "n_rankings_changed": len(changed),
        "n_pred_option_changed": sum(
            1 for c in changed if c["stored_ranked"][0] != c["corrected_ranked"][0]
        ),
        "changed": changed,
        "stored": stored_m,
        "corrected": corr_m,
        "delta_corrected_minus_stored": {k: corr_m[k] - stored_m[k] for k in stored_m},
    }


def main() -> None:
    v5_dir = GRID_RUN_DIRS["V5"]
    v4c_dir = GRID_RUN_DIRS["V4-conflict"]
    posthoc_dir = GRID_RUN_DIRS["post-hoc-binary"]

    runs = {
        "V5": audit_run(v5_dir),
        "V4-conflict": audit_run(v4c_dir),
        "phase5_fit_run_train": audit_run(FIT_RUN_DIR),
    }

    impact = [
        rerank_impact("V5", v5_dir, _rank_v5),
        rerank_impact("V4-conflict", v4c_dir, _rank_margin),
    ]
    # post-hoc-binary: V5's matrices, its own stored rankings
    ph = rerank_impact(
        "post-hoc-binary", v5_dir, lambda f: _rank_v5(f, s_neg_zero=True), stored_dir=posthoc_dir
    )
    ph["run_dir"] = posthoc_dir.name
    impact.append(ph)

    v5_rows = _load_rows(v5_dir)
    support_counts = {
        "stored_features": _support_counts(v5_rows, extract_option_features, _rank_v5),
        "cell_derived_features": _support_counts(v5_rows, corrected_features, _rank_v5),
    }

    out = {
        "adr": "ADR-042",
        "identity_checked": (
            "option_support[o].s_pos == #SUPPORT cells of o and option_support[o].s_neg == "
            "#CONTRADICT cells of o, for every option of every question"
        ),
        "runs": runs,
        "rerank_impact_on_grid": impact,
        "v5_support_based_counts": support_counts,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[spos_consistency] wrote {OUT_PATH}")
    for name, r in runs.items():
        print(
            f"  {name:22s} n={r['n_questions']} disagreeing q={r['n_questions_disagreeing']} "
            f"options={r['n_options_disagreeing']} by status={r['disagreeing_status_counts']} "
            f"(all statuses {r['status_counts']})"
        )
    for r in impact:
        d = r["delta_corrected_minus_stored"]
        print(
            f"  {r['variant']:16s} rankings changed={r['n_rankings_changed']} "
            f"pred changed={r['n_pred_option_changed']}  "
            + " ".join(f"d{k}={v:+.4f}" for k, v in d.items())
        )
    print(f"  V5 support-based counts: {support_counts}")


if __name__ == "__main__":
    main()
