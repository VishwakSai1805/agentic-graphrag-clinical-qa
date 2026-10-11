"""Phase 7: regenerate the results tables from `results/` -- zero hand-typed
numbers (PROJECT_RULES.md §7, `docs/04_EVALUATION.md` §5).

    uv run python -m scripts.report

Scans every `results/*/metrics.json` PLUS the derived
`results/phase7_posthoc_binary_audit/metrics.json`, skips any marked
`superseded` or not grid-eligible, maps each run to a `docs/04` §5 grid row,
and writes:

    paper/tables/results_grid.md    -- markdown, for reading / PR review
    paper/tables/results_grid.tex   -- LaTeX booktabs, for the paper

Every cell is either a value copied from a `metrics.json` field, an analytic
chance value read from `results/phase7_grid/random_baseline.json`
(`scripts/analysis/random_baseline.py` -- the Random row), or the literal
`TBD` (no eligible run yet) / `--` (metric not defined for that
variant/dataset). Re-running the script on an unchanged `results/` produces
byte-identical files.

**Eligibility (ADR-040).** The grid is n=500, seed 42, on an evaluation split
(`docs/04` §0.3). A run may fill a grid or diagnostics row only if
`limit == 500`, `seed == 42` and its split is not a dev/fit split
(`DEV_SPLITS`). A dev-slice number never appears in these tables, footnote or
not. The earlier dev-slice fallback -- Phase 3 `B0`/`B1`/`B2` mapped into
V0/V1/V3 when no real run existed (ADR-029) -- is removed: it put a dev-20 V0
number into a table of n=500 test results.

**Provenance guard (launch instruction 2026-09-10, extended ADR-040).** Every
selected MedQA row must come from code identical to Phase 7's launch commit
`41f7e14` (`code_matches_launch_commit`): its `git_sha7` is `41f7e14`
(`-dirty` allowed), or a later commit whose `RUN_AFFECTING_PATHS` are
byte-identical to `41f7e14` (`git diff --quiet`, e.g. the V0/V2 cells launched
after docs-only commits); and none of its `git_dirty_files` is a
run-affecting path (ADR-038's manual check, now executed). Anything else
raises -- never silently tabulated.

Row mapping: a `metrics.json` `variant` field that matches a `docs/04` §5 row
name (`V0`..`V5`, `V4-audit`, `V4-conflict`, `B3`) maps to its own row;
`post-hoc-binary` (ADR-036 Constraint B, derived from V5) is its own
diagnostics row, never merged into the V5 row; nothing else is tabulated.
`select_cell_runs` picks the LATEST eligible non-superseded run per row by
`run_id` timestamp.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = REPO_ROOT / "paper" / "tables"

PHASE7_LAUNCH_COMMIT_SHORT = "41f7e14"
# Tracked paths whose content determines what a harness run computes. Docs,
# scripts/ (report + analysis) and results/ do not.
RUN_AFFECTING_PATHS = ("src", "configs", "pyproject.toml", "uv.lock")

GRID_N = 500
GRID_SEED = 42
DEV_SPLITS = frozenset({"train", "dev"})

RANDOM_ROW = "random"
RANDOM_DATASET_LABEL = "any 4-option"
# metrics a uniformly random ranking has an analytic value for
RANDOM_METRICS = frozenset({"accuracy", "p_at_1", "p_at_2", "mrr"})

# (row key, dataset) rows of the §5 grid, in reporting order.
GRID_ROWS: list[tuple[str, str]] = [
    (RANDOM_ROW, RANDOM_DATASET_LABEL),
    ("V0", "medqa"),
    ("V1", "medqa"),
    ("V2", "medqa"),
    ("V3", "medqa"),
    ("V4", "medqa"),
    ("V4-audit", "medqa"),
    ("V4-conflict", "medqa"),
    ("V5", "medqa"),
    ("V5-refit", "medqa"),
    ("B3", "medqa"),
    ("V3", "medmcqa"),
    ("V5", "medmcqa"),
    ("V3", "pubmedqa"),
    ("V5", "pubmedqa"),
]

VARIANT_LABELS = {
    RANDOM_ROW: "Random ranking (chance)",
    "V0": "V0 zero-shot",
    "V1": "V1 dense RAG",
    "V2": "V2 hybrid",
    "V3": "V3 +KG",
    "V4": "V4 +agents",
    "V4-audit": "V4-audit (binary audit)",
    "V4-conflict": "V4-conflict (no Conflict agent)",
    "V5": "V5 full",
    "V5-refit": "V5-refit (post-hoc refit, ADR-043)",
    "B3": "B3 binary verifier",
}
# `variant` strings a real run can carry (the Random row is analytic, never a run)
RUN_VARIANTS = frozenset(VARIANT_LABELS) - {RANDOM_ROW}

DIAGNOSTIC_VARIANTS = {"post-hoc-binary": "post-hoc-binary from V5, ADR-036 Constraint B"}

# metric key in metrics.json -> column header
METRIC_COLUMNS: list[tuple[str, str]] = [
    ("accuracy", "Acc"),
    ("macro_f1", "Macro-F1"),
    ("hallucination_rate", "Halluc."),
    ("grounded_rate", "Grounded"),
    ("p_at_1", "P@1"),
    ("p_at_2", "P@2"),
    ("mrr", "MRR"),
    ("ece", "ECE"),
]

# Cells the grid marks with a dash rather than a number: the row's variant
# genuinely does not produce that kind of output (not "no run yet").
DASH_CELLS = {
    ("pubmedqa", "p_at_1"),
    ("pubmedqa", "p_at_2"),
    ("pubmedqa", "mrr"),
    ("B3", "p_at_1"),
    ("B3", "p_at_2"),
    ("B3", "mrr"),
    ("B3", "ece"),
}


def canonical_variant(variant: str) -> str | None:
    """`metrics.json` `variant` field -> §5 grid row key, or None if the run
    is not a grid row (a Phase 3 B0/B1/B2 baseline, B2eq, anything else)."""
    return variant if variant in RUN_VARIANTS else None


def is_grid_eligible(run: dict) -> bool:
    """n=500, seed 42, not a dev/fit split -- the only runs any row may show."""
    split = run.get("split")
    return (
        run.get("limit") == GRID_N
        and run.get("seed") == GRID_SEED
        and split is not None
        and split not in DEV_SPLITS
    )


def _run_timestamp(run_id: str) -> str:
    """Trailing `<utc_ts>` component of a run_id (sortable as a plain string)."""
    return run_id.rsplit("_", 1)[-1]


def _is_run_affecting(path: str) -> bool:
    p = path.replace("\\", "/").strip().rstrip("/")
    return any(p == r or p.startswith(r + "/") for r in RUN_AFFECTING_PATHS)


def code_matches_launch_commit(run: dict) -> tuple[bool, str]:
    """(ok, reason): was `run` produced by code identical to the Phase 7
    launch commit on every run-affecting path?"""
    dirty_hits = [f for f in (run.get("git_dirty_files") or []) if _is_run_affecting(f)]
    if dirty_hits:
        return False, f"run-affecting paths were dirty at run time: {dirty_hits}"
    sha = str(run.get("git_sha7") or "").removesuffix("-dirty")
    if sha.startswith(PHASE7_LAUNCH_COMMIT_SHORT):
        return True, f"launch commit {PHASE7_LAUNCH_COMMIT_SHORT}"
    if not sha or sha == "nogit":
        return False, "no git_sha7 recorded"
    proc = subprocess.run(
        ["git", "diff", "--quiet", PHASE7_LAUNCH_COMMIT_SHORT, sha, "--", *RUN_AFFECTING_PATHS],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    if proc.returncode == 0:
        return True, f"{sha} is identical to {PHASE7_LAUNCH_COMMIT_SHORT} on {RUN_AFFECTING_PATHS}"
    return False, (
        f"{sha} is not identical to {PHASE7_LAUNCH_COMMIT_SHORT} on {RUN_AFFECTING_PATHS} "
        f"(git diff exit {proc.returncode})"
    )


def load_runs() -> list[dict]:
    """Every non-superseded `results/*/metrics.json`, as parsed dicts (the
    derived post-hoc-binary analysis included -- it lives at the same depth)."""
    runs: list[dict] = []
    for path in sorted(RESULTS_DIR.glob("*/metrics.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:  # pragma: no cover - unreadable file
            print(f"[report] skipping unreadable {path}: {e}", file=sys.stderr)
            continue
        if not isinstance(obj, dict) or "metrics" not in obj:
            continue
        if obj.get("superseded"):
            continue
        runs.append(obj)
    return runs


def _assert_phase7_provenance(selected: dict[tuple[str, str], dict]) -> None:
    """Every MedQA row filled from a real run must pass
    `code_matches_launch_commit`. Fails loudly rather than silently
    tabulating a run from other code."""
    for (variant, dataset), run in selected.items():
        if dataset != "medqa":
            continue
        ok, why = code_matches_launch_commit(run)
        if not ok:
            raise AssertionError(
                f"[report] {variant}/{dataset} run {run['run_id']!r} "
                f"(git_sha7={run.get('git_sha7')!r}): {why} -- refusing to tabulate "
                f"it as a Phase 7 result (launch commit {PHASE7_LAUNCH_COMMIT_SHORT})"
            )


def _latest_per_key(runs: list[dict], key_fn) -> dict[tuple[str, str], dict]:
    chosen: dict[tuple[str, str], dict] = {}
    for run in runs:
        if not is_grid_eligible(run):
            continue
        key = key_fn(run)
        if key is None:
            continue
        cur = chosen.get(key)
        if cur is None or _run_timestamp(run["run_id"]) > _run_timestamp(cur["run_id"]):
            chosen[key] = run
    return chosen


def select_cell_runs(runs: list[dict]) -> dict[tuple[str, str], dict]:
    """(row key, dataset) -> the chosen run (latest eligible non-superseded)."""

    def key_fn(run: dict) -> tuple[str, str] | None:
        v = canonical_variant(str(run.get("variant", "")))
        return None if v is None else (v, str(run.get("dataset", "")))

    chosen = _latest_per_key(runs, key_fn)
    _assert_phase7_provenance(chosen)
    return chosen


def _fmt_metric(value: object) -> str:
    if isinstance(value, dict):  # ece() returns a dict; the number is under "ece"
        value = value.get("ece")
    if value is None:
        return "TBD"
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return str(value)


def cell_value(run: dict | None, row_key: str, dataset: str, metric_key: str) -> str:
    if (dataset, metric_key) in DASH_CELLS or (row_key, metric_key) in DASH_CELLS:
        return "--"
    if run is None:
        return "TBD"
    return _fmt_metric(run["metrics"].get(metric_key))


def _random_row() -> list[str]:
    """Analytic chance row: expectations of a uniformly random ranking of 4
    options, read from `scripts/analysis/random_baseline.py`'s JSON."""
    path = RESULTS_DIR / "phase7_grid" / "random_baseline.json"
    analytic = json.loads(path.read_text(encoding="utf-8"))["analytic"] if path.exists() else None
    cells = []
    for key, _ in METRIC_COLUMNS:
        if key not in RANDOM_METRICS:
            cells.append("--")
        elif analytic is None:
            cells.append("TBD")
        else:
            cells.append(_fmt_metric(analytic[key]))
    source = "phase7_grid/random_baseline.json (analytic)" if analytic is not None else "TBD"
    return [VARIANT_LABELS[RANDOM_ROW], RANDOM_DATASET_LABEL, *cells, source]


def build_grid(chosen: dict[tuple[str, str], dict]) -> list[list[str]]:
    header = ["Variant", "Dataset", *[h for _, h in METRIC_COLUMNS], "run_id"]
    rows = [header]
    for row_key, dataset in GRID_ROWS:
        if row_key == RANDOM_ROW:
            rows.append(_random_row())
            continue
        run = chosen.get((row_key, dataset))
        row = [VARIANT_LABELS[row_key], dataset]
        row += [cell_value(run, row_key, dataset, key) for key, _ in METRIC_COLUMNS]
        # only name a run_id if it actually carries a headline number
        has_headline = run is not None and run["metrics"].get("accuracy") is not None
        row.append(run["run_id"] if (run is not None and has_headline) else "TBD")
        rows.append(row)
    return rows


def _diagnostics(runs: list[dict]) -> list[list[str]]:
    """post-hoc-binary rows -- traceable, eligible, but not grid variants."""
    header = ["variant", "dataset", "Acc", "Macro-F1", "note", "run_id"]
    out = [header]

    def key_fn(run: dict) -> tuple[str, str] | None:
        var = str(run.get("variant", ""))
        return (var, str(run.get("dataset", ""))) if var in DIAGNOSTIC_VARIANTS else None

    seen = _latest_per_key(runs, key_fn)
    _assert_phase7_provenance(seen)
    for (var, dataset), run in sorted(seen.items()):
        m = run["metrics"]
        out.append(
            [
                var,
                dataset,
                _fmt_metric(m.get("accuracy")),
                _fmt_metric(m.get("macro_f1")),
                DIAGNOSTIC_VARIANTS[var],
                run["run_id"],
            ]
        )
    return out


def _filled_row_keys(chosen: dict[tuple[str, str], dict]) -> set[str]:
    """MedQA row keys that carry a real headline number."""
    return {
        v
        for (v, d), run in chosen.items()
        if d == "medqa" and run["metrics"].get("accuracy") is not None
    }


def _status_notes(filled: set[str]) -> list[str]:
    """Per-row status prose, derived from which rows actually have a run --
    so the text can never claim a row is "not run" while it shows a number."""
    if {"V0", "V2"} <= filled:
        v0_v2 = (
            "V0 (zero-shot) and V2 (`{dense,lexical}`) ran at n=500 via "
            "`scripts/run_phase7_v0_v2.ps1` (ADR-040, ADR-044). Their tests are the "
            "UNPLANNED family's U1-U5 (`docs/04` §1.0b), never merged into the m=35 family."
        )
    else:
        missing = " and ".join(v for v in ("V0", "V2") if v not in filled)
        v0_v2 = (
            f"{missing} {'has' if ' and ' not in missing else 'have'} no n=500 test run yet: "
            "prepared, not run, in `scripts/run_phase7_v0_v2.ps1` (ADR-040)."
        )
    refit_status = (
        "Run by `scripts/run_phase7_v5_refit.ps1`."
        if "V5-refit" in filled
        else "Prepared, not run, in `scripts/run_phase7_v5_refit.ps1`."
    )
    return [
        v0_v2,
        "",
        "V5 is the pre-registered result and the headline. V5-refit is its disclosed "
        "post-hoc correction (ADR-043): V5's own test SupportMatrices re-scored with "
        "weights refit on the Phase 5 train slice (evidential weights >= 0, a neutral "
        "no-signal rerank value, a seeded tie-break). It is reported beside V5 and "
        "replaces nothing; its tests are the unplanned family's U6-U8 (`docs/04` §1.0b). "
        + refit_status,
        "",
        "`Grounded` is `grounded_rate`, and it has two operationalisations -- compare "
        "within a group, not across. Single-prediction rows (V0-V3, B3): the answer "
        "carries >=1 valid citation, i.e. a cited id present in the retrieved bundle "
        "(`src/eval/baselines.py`). Agent rows (V4-conflict, V5, V5-refit): the predicted "
        "option has >=1 SUPPORT audit cell (`src/eval/harness.py`, "
        "`src/calibration/citations.py`), a stricter bar. V0 retrieves nothing, so its "
        "Grounded = 0 and Halluc. = 1 by construction, not by measurement. Grounded is "
        "displayed, never significance-tested.",
        "",
    ]


def render_markdown(
    grid: list[list[str]], diag: list[list[str]], filled: set[str] | None = None
) -> str:
    lines = [
        "# Results grid (generated by `scripts/report.py` -- do not edit by hand)",
        "",
        "Every number here is copied from a `results/<run_id>/metrics.json` field "
        "(or `results/phase7_posthoc_binary_audit/metrics.json` for the derived "
        "post-hoc-binary row), except the Random row, which is the analytic "
        "expectation of a uniformly random ranking of 4 options read from "
        "`results/phase7_grid/random_baseline.json` (P@1 = 1/4, P@2 = 2/4, "
        "MRR = 25/48) -- a chance floor, not a run. `TBD` = no eligible run yet; "
        "`--` = metric not defined for that variant/dataset. Only n=500, seed-42, "
        "non-dev-split runs are eligible for any row: a dev-slice number never "
        "appears here (ADR-040). Every MedQA row is asserted to come from code "
        "identical to Phase 7's launch commit `41f7e14` -- a run from other code "
        "raises instead of silently tabulating.",
        "",
    ]
    lines += _status_notes(filled or set())
    lines += _md_table(grid)
    lines += ["", "## Diagnostics (not grid variants)", ""]
    lines += _md_table(diag)
    lines.append("")
    return "\n".join(lines)


def _md_table(rows: list[list[str]]) -> list[str]:
    header, *body = rows
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return out


def render_latex(grid: list[list[str]]) -> str:
    header, *body = grid
    ncols = len(header)
    esc = lambda s: s.replace("_", r"\_").replace("&", r"\&")  # noqa: E731
    lines = [
        "% generated by scripts/report.py -- do not edit by hand",
        r"\begin{tabular}{" + "l" * ncols + "}",
        r"\toprule",
        " & ".join(esc(h) for h in header) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(esc(c) for c in r) + r" \\" for r in body]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def main() -> int:
    runs = load_runs()
    chosen = select_cell_runs(runs)
    grid = build_grid(chosen)
    diag = _diagnostics(runs)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    md_path = TABLES_DIR / "results_grid.md"
    tex_path = TABLES_DIR / "results_grid.tex"
    md_path.write_text(render_markdown(grid, diag, _filled_row_keys(chosen)), encoding="utf-8")
    tex_path.write_text(render_latex(grid), encoding="utf-8")

    run_rows = [(v, d) for v, d in GRID_ROWS if v != RANDOM_ROW]
    filled = sum(
        1
        for v, d in run_rows
        if (v, d) in chosen and chosen[(v, d)]["metrics"].get("accuracy") is not None
    )
    print(f"[report] {len(runs)} live runs -> {filled}/{len(run_rows)} grid cells filled")
    print(f"[report] wrote {md_path}")
    print(f"[report] wrote {tex_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
