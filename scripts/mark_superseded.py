"""Stamp `superseded: true` + a reason into an already-written run's metrics.json.

AUDIT 2026-09-06 §6.2. Two committed Phase 3 batches share the SHA `d5e3af5`:
the ~0954-1006Z batch was produced with Ollama `num_ctx=4096`, which silently
truncated the evidence block (B2/B3 accuracy 0.15, 12/20 `parse_error`, 283-284
invalid citations); the ~1011-1021Z batch is the same code at `num_ctx=8192` and
is the batch `docs/04_EVALUATION.md` §5.4 reports. Nothing in `results/` marked
the first batch invalid, so `docs/04`'s `medqa_B{0,1,2,3}_<sha>_<ts>/` wildcard
gave any reader -- or Phase 8's `scripts/report.py` -- a coin-flip chance of
tabulating the corrupt run.

This script exists so that fix does not become a hand-edit of `results/`
(CLAUDE.md: "Don't touch `results/` by hand. Ever."). It rewrites only the
three `superseded*` keys defined on `src.eval.results.RunResult`, leaves every
measured value untouched, and is idempotent.

A superseded run is NOT deleted: the num_ctx=4096 batch is the evidence for the
context-window sensitivity reported in `docs/04_EVALUATION.md` §5.4, and
`docs/05_DECISIONS.md` ADR-021's no-op perturbation row reads B0 out of it.
Superseded means "never tabulate this as a headline result", not "discard".

Usage:
    uv run python -m scripts.mark_superseded --preset phase3-num-ctx
    uv run python -m scripts.mark_superseded --run-id <id> --reason "..." --superseded-by <id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
NEWLINE = chr(10)

_NUM_CTX_REASON = (
    "Produced with Ollama num_ctx=4096, which truncated the evidence block mid-prompt. "
    "Superseded by the num_ctx=8192 re-run of the same code and seed. Retained (not "
    "deleted) as the context-window sensitivity comparator cited in "
    "docs/04_EVALUATION.md §5.4 and docs/05_DECISIONS.md ADR-021; never tabulate it as "
    "a headline result."
)

# (superseded run_id, superseding run_id)
_PHASE3_NUM_CTX: list[tuple[str, str]] = [
    ("medqa_B0_d5e3af5_20260906T095449Z", "medqa_B0_d5e3af5_20260906T101125Z"),
    ("medqa_B1_d5e3af5_20260906T095549Z", "medqa_B1_d5e3af5_20260906T101242Z"),
    ("medqa_B2_d5e3af5_20260906T095935Z", "medqa_B2_d5e3af5_20260906T101648Z"),
    ("medqa_B3_d5e3af5_20260906T100551Z", "medqa_B3_d5e3af5_20260906T102134Z"),
]


_TEST_SPLIT_PROBE_REASON = (
    "Run on the MedQA TEST split, which Phase 7 reports on (AUDIT 2026-09-06 §3.1). "
    "Every Phase 2 KG decision it originally supported (ADR-015/016/017/018) has been "
    "re-derived from the train-500 probe named in superseded_by. Retained as provenance "
    "and disclosed in docs/00_RESEARCH_SPEC.md §7; must NOT be cited as the basis of any "
    "hyper-parameter or keep/drop decision, and must not be reported as a Phase 2 result."
)

# (superseded probe file, superseding probe file)
_PHASE2_TEST_SPLIT: list[tuple[str, str]] = [
    ("medqa_n200.json", "medqa_train_n500.json"),
    ("medqa_n200_kg_rerank_variant.json", "medqa_train_n500_kg_rerank_variant.json"),
    ("medqa_n200_kg_rerank_ablation.json", "medqa_train_n500_kg_rerank_ablation.json"),
    ("medqa_n500.json", "medqa_train_n500.json"),
]


# 2026-09-10 launch instruction: "results/ also holds nine older run dirs
# from commits d5e3af5, b1e355e, 1388904, a714fe0, aa73199, d7d3c21, a394291
# ... mark the nine older dirs superseded." These are exactly the pre-grid
# dirs whose `variant` field collides with a current or future Phase 7 §5
# grid row name (V4, V4-conflict, V5, B3) plus the shadowed older B2eq
# duplicate -- the set capable of being mistaken for a grid result by
# name-based selection. `superseded` here means "never auto-tabulate as a
# grid headline", not "invalid" -- several of these remain the cited,
# authoritative provenance for their own phase (§5.4/§5.5/§5.6/§5.7), exactly
# as the phase3-num-ctx preset above already established for this project.
# Two (medqa_V5_a394291_..., medqa_B2eq_aa73199-dirty_...) were already
# stamped superseded in an earlier session for independent, still-correct
# reasons; only their `superseded_by` is filled in here now that a real
# replacement exists (V5) or already existed (B2eq).
_PHASE7_PRE_GRID: list[tuple[str, str, str | None]] = [
    (
        "medqa_V4-conflict_d7d3c21-dirty_20260908T074028Z",
        "Pre-grid V4-conflict run (commit d7d3c21, ADR-035 prep/wiring). "
        "Superseded by the real Phase 7 grid cell at n=500 test, commit "
        "41f7e14.",
        "medqa_V4-conflict_41f7e14-dirty_20260909T220306Z",
    ),
    (
        "medqa_V5_aa73199-dirty_20260907T170604Z",
        "Phase 6 harness exit-test dev-20 run (commit aa73199). Remains the "
        "cited artifact for docs/04_EVALUATION.md §5.7's hallucination-rate "
        "control and ADR-031 -- retained, not invalid -- but must never be "
        "auto-tabulated as the V5 grid row now that the real n=500 test-set "
        "cell exists.",
        "medqa_V5_41f7e14-dirty_20260909T103452Z",
    ),
    (
        "medqa_B3_d5e3af5_20260906T102134Z",
        "Phase 3 dev-20 baseline plumbing check (commit d5e3af5, num_ctx=8192 "
        "post-fix batch). Remains the cited artifact for "
        "docs/04_EVALUATION.md §5.4's B3/verifier_decision_rate discussion -- "
        "retained, not invalid -- but must never be auto-tabulated as the B3 "
        "grid row now that the real n=500 test-set cell exists (report.py's "
        "latest-by-timestamp selection already prefers the real cell "
        "regardless; this stamp makes that explicit rather than incidental).",
        "medqa_B3_41f7e14-dirty_20260909T023658Z",
    ),
    (
        "medqa_V4_1388904_20260906T115020Z",
        "Phase 4 multi-agent consensus dev-20 plumbing check (commit "
        "1388904). Remains the cited artifact for docs/04_EVALUATION.md "
        "§5.5's Findings 1-3 -- retained, not invalid -- but must never be "
        "auto-tabulated as the V4 grid row (still TBD -- no Phase 7 V4 cell "
        "has run) if scripts/report.py's variant matching would otherwise "
        "pick it up.",
        None,
    ),
    (
        "medqa_V4_a714fe0-dirty_20260907T121703Z",
        "Phase 5 calibration fit-set run (commit a714fe0, disjoint 120-q "
        "MedQA train slice). Remains the cited artifact for "
        "docs/04_EVALUATION.md §5.6 and configs/v5.yaml's fit provenance "
        "comment -- retained, not invalid -- but must never be "
        "auto-tabulated as the V4 grid row (still TBD).",
        None,
    ),
    (
        "medqa_V4_b1e355e_20260906T111821Z",
        "Earlier duplicate V4 attempt (commit b1e355e), superseded in "
        "practice by the same-day medqa_V4_1388904_... run that "
        "docs/04_EVALUATION.md §5.5 actually cites. Never used as evidence "
        "anywhere; stamped so it cannot be silently selected as the V4 grid "
        "row.",
        "medqa_V4_1388904_20260906T115020Z",
    ),
    (
        "medqa_V4_b1e355e_20260906T112332Z",
        "Earlier duplicate V4 attempt (commit b1e355e), superseded in "
        "practice by the same-day medqa_V4_1388904_... run that "
        "docs/04_EVALUATION.md §5.5 actually cites. Never used as evidence "
        "anywhere; stamped so it cannot be silently selected as the V4 grid "
        "row.",
        "medqa_V4_1388904_20260906T115020Z",
    ),
]

# Already-superseded dirs from this same pre-grid set that only need their
# `superseded_by` filled in now that a real replacement exists.
_PHASE7_PRE_GRID_BACKFILL_SUPERSEDED_BY: list[tuple[str, str]] = [
    ("medqa_V5_a394291_20260908T074445Z", "medqa_V5_41f7e14-dirty_20260909T103452Z"),
]


def stamp_probe(filename: str, reason: str, superseded_by: str | None) -> bool:
    """Stamp a `results/phase2_kg_probe/*.json` artifact. These are probe
    outputs, not run directories, so they have no `metrics.json` -- the same
    three `superseded*` keys are written at the top level instead."""
    path = RESULTS_DIR / "phase2_kg_probe" / filename
    if not path.exists():
        print(f"[mark_superseded] ERROR: {path} not found", file=sys.stderr)
        sys.exit(2)
    doc = json.loads(path.read_text(encoding="utf-8"))
    before = (doc.get("superseded"), doc.get("superseded_reason"), doc.get("superseded_by"))
    doc["superseded"] = True
    doc["superseded_reason"] = reason
    doc["superseded_by"] = superseded_by
    if before == (doc["superseded"], doc["superseded_reason"], doc["superseded_by"]):
        print(f"[mark_superseded] {filename}: already stamped, unchanged")
        return False
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + NEWLINE, encoding="utf-8")
    print(f"[mark_superseded] {filename}: superseded=true (by {superseded_by})")
    return True


def stamp(run_id: str, reason: str, superseded_by: str | None) -> bool:
    """Rewrite one run's metrics.json in place. Returns True if it changed."""
    path = RESULTS_DIR / run_id / "metrics.json"
    if not path.exists():
        print(f"[mark_superseded] ERROR: {path} not found", file=sys.stderr)
        sys.exit(2)
    doc = json.loads(path.read_text(encoding="utf-8"))
    before = (doc.get("superseded"), doc.get("superseded_reason"), doc.get("superseded_by"))
    doc["superseded"] = True
    doc["superseded_reason"] = reason
    doc["superseded_by"] = superseded_by
    after = (doc["superseded"], doc["superseded_reason"], doc["superseded_by"])
    if before == after:
        print(f"[mark_superseded] {run_id}: already stamped, unchanged")
        return False
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[mark_superseded] {run_id}: superseded=true (by {superseded_by})")
    return True


def backfill_superseded_by(run_id: str, superseded_by: str) -> bool:
    """For a run already stamped `superseded: true` with a correct reason:
    fill in `superseded_by` without touching the existing reason. Errors if
    the run is not already superseded (use `stamp` for that case)."""
    path = RESULTS_DIR / run_id / "metrics.json"
    if not path.exists():
        print(f"[mark_superseded] ERROR: {path} not found", file=sys.stderr)
        sys.exit(2)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not doc.get("superseded"):
        print(
            f"[mark_superseded] ERROR: {run_id} is not superseded -- use stamp(), not backfill",
            file=sys.stderr,
        )
        sys.exit(2)
    if doc.get("superseded_by") == superseded_by:
        print(f"[mark_superseded] {run_id}: superseded_by already {superseded_by!r}, unchanged")
        return False
    doc["superseded_by"] = superseded_by
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[mark_superseded] {run_id}: superseded_by backfilled to {superseded_by!r}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--preset",
        choices=["phase3-num-ctx", "phase2-kg-probe-test-split", "phase7-pre-grid"],
        help="stamp a known-invalid batch. phase3-num-ctx: the four pre-num_ctx-fix B runs. "
        "phase2-kg-probe-test-split: the four test-split Phase 2 KG probe artifacts "
        "superseded by the train-500 re-run (AUDIT §3.1). phase7-pre-grid: the nine "
        "older MedQA run dirs whose variant name collides with a Phase 7 §5 grid row "
        "(2026-09-10 launch instruction).",
    )
    ap.add_argument("--run-id", help="single run to stamp")
    ap.add_argument("--reason", help="required with --run-id")
    ap.add_argument("--superseded-by", default=None, help="run_id that replaces it")
    args = ap.parse_args()

    if args.preset == "phase3-num-ctx":
        changed = sum(stamp(rid, _NUM_CTX_REASON, by) for rid, by in _PHASE3_NUM_CTX)
        print(f"[mark_superseded] preset phase3-num-ctx: {changed} file(s) changed")
        return
    if args.preset == "phase2-kg-probe-test-split":
        changed = sum(
            stamp_probe(f, _TEST_SPLIT_PROBE_REASON, by) for f, by in _PHASE2_TEST_SPLIT
        )
        print(f"[mark_superseded] preset phase2-kg-probe-test-split: {changed} file(s) changed")
        return
    if args.preset == "phase7-pre-grid":
        changed = sum(stamp(rid, reason, by) for rid, reason, by in _PHASE7_PRE_GRID)
        changed += sum(
            backfill_superseded_by(rid, by)
            for rid, by in _PHASE7_PRE_GRID_BACKFILL_SUPERSEDED_BY
        )
        print(f"[mark_superseded] preset phase7-pre-grid: {changed} file(s) changed")
        return
    if not args.run_id or not args.reason:
        ap.error("give --preset, or both --run-id and --reason")
    stamp(args.run_id, args.reason, args.superseded_by)


if __name__ == "__main__":
    main()
