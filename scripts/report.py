"""Phase 6: regenerate the results tables from `results/` -- zero hand-typed
numbers (CLAUDE.md §7, `docs/04_EVALUATION.md` §5).

    uv run python -m scripts.report

Scans every `results/*/metrics.json`, skips any marked `superseded`, maps each
run to a `docs/04` §5 grid cell, and writes:

    paper/tables/results_grid.md    -- markdown, for reading / PR review
    paper/tables/results_grid.tex   -- LaTeX booktabs, for the paper

Every cell is either a value copied from a `metrics.json` field or the literal
`TBD` (no run yet) / `--` (metric not defined for that dataset). Re-running the
script on an unchanged `results/` produces byte-identical files.

Variant mapping (ADR-029):
  * `V0..V5` runs (the Phase 6 harness) map to their own row.
  * The committed Phase 3 baselines map by modality set, NOT by name:
      B0 -> V0, B1 -> V1, **B2 -> V3** (B2's bundle is {dense,lexical,kg}).
    V2 = {dense,lexical} has never been run -> its row stays `TBD`.
  * `B2eq` (equal-budget B2) and `B3` (binary verifier) are NOT grid variants
    and are listed in a separate diagnostics section, never in the grid.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = REPO_ROOT / "paper" / "tables"

# (variant, dataset) rows of the §5 grid, in reporting order.
GRID_ROWS: list[tuple[str, str]] = [
    ("V0", "medqa"),
    ("V1", "medqa"),
    ("V2", "medqa"),
    ("V3", "medqa"),
    ("V4", "medqa"),
    ("V5", "medqa"),
    ("V3", "medmcqa"),
    ("V5", "medmcqa"),
    ("V3", "pubmedqa"),
    ("V5", "pubmedqa"),
]

VARIANT_LABELS = {
    "V0": "V0 zero-shot",
    "V1": "V1 dense RAG",
    "V2": "V2 hybrid",
    "V3": "V3 +KG",
    "V4": "V4 +agents",
    "V5": "V5 full",
}

# metric key in metrics.json -> column header
METRIC_COLUMNS: list[tuple[str, str]] = [
    ("accuracy", "Acc"),
    ("macro_f1", "Macro-F1"),
    ("hallucination_rate", "Halluc."),
    ("p_at_1", "P@1"),
    ("p_at_2", "P@2"),
    ("mrr", "MRR"),
    ("ece", "ECE"),
]

# Cells the grid marks with a dash rather than a number.
DASH_CELLS = {("pubmedqa", "p_at_1"), ("pubmedqa", "p_at_2"), ("pubmedqa", "mrr")}

# Phase 3 baseline name -> V-id. Applied for MedQA ONLY: the committed
# `medqa_B2_*` runs are the documented {dense,lexical,kg} == V3 mapping
# (ADR-029). The `pubmedqa_B*` runs are dev-20 smoke runs, never grid cells,
# so PubMedQA rows stay TBD until a real harness V3/V5 run exists.
_BASELINE_TO_VARIANT = {"B0": "V0", "B1": "V1", "B2": "V3", "V4": "V4"}


def canonical_variant(variant: str, dataset: str) -> str | None:
    """`metrics.json` `variant` field -> §5 grid V-id, or None if the run is
    not a grid variant (B2eq, B3, a non-MedQA baseline, anything unrecognised)."""
    if variant in VARIANT_LABELS:
        return variant
    if dataset == "medqa":
        return _BASELINE_TO_VARIANT.get(variant)
    return None


def _run_timestamp(run_id: str) -> str:
    """Trailing `<utc_ts>` component of a run_id (sortable as a plain string)."""
    return run_id.rsplit("_", 1)[-1]


def load_runs() -> list[dict]:
    """Every non-superseded `results/*/metrics.json`, as parsed dicts."""
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


def select_cell_runs(runs: list[dict]) -> dict[tuple[str, str], dict]:
    """(V-id, dataset) -> the chosen run (latest non-superseded by timestamp)."""
    chosen: dict[tuple[str, str], dict] = {}
    for run in runs:
        dataset = str(run.get("dataset", ""))
        v = canonical_variant(str(run.get("variant", "")), dataset)
        if v is None:
            continue
        key = (v, dataset)
        cur = chosen.get(key)
        if cur is None or _run_timestamp(run["run_id"]) > _run_timestamp(cur["run_id"]):
            chosen[key] = run
    return chosen


def _fmt_metric(value: object) -> str:
    if isinstance(value, dict):  # ece() returns a dict; the number is under "ece"
        value = value.get("ece")
    if value is None:
        return "TBD"
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return str(value)


def cell_value(run: dict | None, dataset: str, metric_key: str) -> str:
    if (dataset, metric_key) in DASH_CELLS:
        return "--"
    if run is None:
        return "TBD"
    return _fmt_metric(run["metrics"].get(metric_key))


def build_grid(chosen: dict[tuple[str, str], dict]) -> list[list[str]]:
    header = ["Variant", "Dataset", *[h for _, h in METRIC_COLUMNS], "run_id"]
    rows = [header]
    for variant, dataset in GRID_ROWS:
        run = chosen.get((variant, dataset))
        row = [VARIANT_LABELS[variant], dataset]
        row += [cell_value(run, dataset, key) for key, _ in METRIC_COLUMNS]
        # only name a run_id if it actually carries a headline number -- a run
        # that predates the Phase 6 harness (e.g. the old V4 pipeline run, which
        # writes accuracy_rank1_by_support, not accuracy) is not a grid result.
        has_headline = run is not None and run["metrics"].get("accuracy") is not None
        row.append(run["run_id"] if (run is not None and has_headline) else "TBD")
        rows.append(row)
    return rows


def _diagnostics(runs: list[dict]) -> list[list[str]]:
    """B2eq / B3 rows -- traceable, but not grid variants."""
    header = ["variant", "dataset", "Acc", "Macro-F1", "note", "run_id"]
    out = [header]
    keep = {"B2eq": "B2 equal-budget (§5.5)", "B3": "binary verifier (§5.4)"}
    seen: dict[tuple[str, str], dict] = {}
    for run in runs:
        var = str(run.get("variant", ""))
        if var not in keep:
            continue
        key = (var, str(run.get("dataset", "")))
        cur = seen.get(key)
        if cur is None or _run_timestamp(run["run_id"]) > _run_timestamp(cur["run_id"]):
            seen[key] = run
    for (var, dataset), run in sorted(seen.items()):
        m = run["metrics"]
        out.append(
            [
                var,
                dataset,
                _fmt_metric(m.get("accuracy")),
                _fmt_metric(m.get("macro_f1")),
                keep[var],
                run["run_id"],
            ]
        )
    return out


def render_markdown(grid: list[list[str]], diag: list[list[str]]) -> str:
    lines = [
        "# Results grid (generated by `scripts/report.py` -- do not edit by hand)",
        "",
        "Every number here is copied from a `results/<run_id>/metrics.json` field.",
        "`TBD` = no run yet; `--` = metric not defined for that dataset.",
        "",
        "V2 (`{dense,lexical}`) has never been run. The committed `medqa_B2_*` runs "
        "use `{dense,lexical,kg}` and are mapped to the **V3** row (ADR-029).",
        "",
    ]
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
    md_path.write_text(render_markdown(grid, diag), encoding="utf-8")
    tex_path.write_text(render_latex(grid), encoding="utf-8")

    filled = sum(
        1
        for v, d in GRID_ROWS
        if (v, d) in chosen and chosen[(v, d)]["metrics"].get("accuracy") is not None
    )
    print(f"[report] {len(runs)} live runs -> {filled}/{len(GRID_ROWS)} grid cells filled")
    print(f"[report] wrote {md_path}")
    print(f"[report] wrote {tex_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
