"""Phase 6/7: scripts/report.py -- table generation from results/ fixtures.

No real runs: a synthetic results/ tree is built in tmp_path. Fixture runs
default to grid-eligible provenance (test split, n=500, seed 42, launch
commit) so each test varies only the field it is about.
"""

from __future__ import annotations

import json

import pytest

from scripts import report


def _write_run(
    root,
    run_id,
    dataset,
    variant,
    metrics,
    *,
    superseded=False,
    split="test",
    limit=500,
    seed=42,
    git_sha7="41f7e14",
    git_dirty_files=None,
):
    d = root / run_id
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "dataset": dataset,
                "variant": variant,
                "split": split,
                "limit": limit,
                "seed": seed,
                "git_sha7": git_sha7,
                "git_dirty_files": git_dirty_files or [],
                "superseded": superseded,
                "metrics": metrics,
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def results_tree(tmp_path, monkeypatch):
    root = tmp_path / "results"
    root.mkdir()
    tables = tmp_path / "paper" / "tables"
    monkeypatch.setattr(report, "RESULTS_DIR", root)
    monkeypatch.setattr(report, "TABLES_DIR", tables)
    return root


def _grid_rows(root):
    grid = report.build_grid(report.select_cell_runs(report.load_runs()))
    header, *rows = grid
    return header, {(r[0], r[1]): r for r in rows}


def test_v5_refit_fills_its_own_row_and_never_replaces_v5(results_tree):
    # ADR-043: the post-hoc correction sits beside the pre-registered V5 row
    _write_run(
        results_tree, "medqa_V5_41f7e14_20260909T103452Z", "medqa", "V5", {"accuracy": 0.316}
    )
    _write_run(
        results_tree, "medqa_V5-refit_41f7e14_20260912T000000Z", "medqa", "V5-refit",
        {"accuracy": 0.3},
    )
    header, rows = _grid_rows(results_tree)
    acc = header.index("Acc")
    assert rows[(report.VARIANT_LABELS["V5"], "medqa")][acc] == "0.316"
    assert rows[(report.VARIANT_LABELS["V5-refit"], "medqa")][acc] == "0.300"


def test_dev_slice_runs_never_fill_a_grid_row(results_tree):
    # the Phase 3 dev-20 baselines that used to fall back into V0/V1/V3
    for b in ("B0", "B1", "B2"):
        _write_run(
            results_tree, f"medqa_{b}_d5e3af5_20260906T101125Z", "medqa", b,
            {"accuracy": 0.55, "macro_f1": 0.52}, split="train", limit=20, git_sha7="d5e3af5",
        )
    # and a real harness V0 run, but on a dev slice
    _write_run(
        results_tree, "medqa_V0_x_20260101T000000Z", "medqa", "V0",
        {"accuracy": 0.55}, split="train", limit=20,
    )
    chosen = report.select_cell_runs(report.load_runs())
    assert chosen == {}
    header, rows = _grid_rows(results_tree)
    v0 = rows[("V0 zero-shot", "medqa")]
    assert v0[header.index("Acc")] == "TBD"
    assert v0[-1] == "TBD"


@pytest.mark.parametrize(
    "override",
    [{"limit": 3}, {"seed": 43}, {"split": "dev"}, {"split": None}],
)
def test_only_n500_seed42_eval_split_runs_are_eligible(results_tree, override):
    _write_run(
        results_tree, "medqa_V5_x_20260101T000000Z", "medqa", "V5", {"accuracy": 0.3}, **override
    )
    assert report.select_cell_runs(report.load_runs()) == {}


def test_latest_non_superseded_run_wins_a_cell(results_tree):
    _write_run(
        results_tree, "medqa_V5_a_20260101T000000Z", "medqa", "V5", {"accuracy": 0.1}
    )
    _write_run(
        results_tree, "medqa_V5_b_20260202T000000Z", "medqa", "V5", {"accuracy": 0.9}
    )
    _write_run(
        results_tree,
        "medqa_V5_c_20260303T000000Z",
        "medqa",
        "V5",
        {"accuracy": 0.5},
        superseded=True,
    )
    chosen = report.select_cell_runs(report.load_runs())
    assert chosen[("V5", "medqa")]["metrics"]["accuracy"] == 0.9  # latest, non-superseded


def test_grid_marks_missing_cells_tbd_and_pubmedqa_ranking_dashed(results_tree):
    _write_run(
        results_tree,
        "medqa_V0_x_20260101T000000Z",
        "medqa",
        "V0",
        {"accuracy": 0.55, "macro_f1": 0.52},
    )
    header, rows = _grid_rows(results_tree)
    v0 = rows[("V0 zero-shot", "medqa")]
    assert v0[header.index("Acc")] == "0.550"
    assert v0[header.index("P@1")] == "TBD"  # no ranking metric for V0
    v3_pqa = rows[("V3 +KG", "pubmedqa")]
    assert v3_pqa[header.index("Acc")] == "TBD"  # no run
    assert v3_pqa[header.index("P@1")] == "--"  # pubmedqa: ranking not defined


def test_random_row_is_read_from_the_analytic_json(results_tree):
    header, rows = _grid_rows(results_tree)
    rnd = rows[("Random ranking (chance)", "any 4-option")]
    assert rnd[header.index("P@1")] == "TBD"  # no JSON yet -> never a typed number

    (results_tree / "phase7_grid").mkdir()
    (results_tree / "phase7_grid" / "random_baseline.json").write_text(
        json.dumps(
            {"analytic": {"accuracy": 0.25, "p_at_1": 0.25, "p_at_2": 0.5, "mrr": 25 / 48}}
        ),
        encoding="utf-8",
    )
    header, rows = _grid_rows(results_tree)
    rnd = rows[("Random ranking (chance)", "any 4-option")]
    assert [rnd[header.index(h)] for h in ("Acc", "P@1", "P@2", "MRR")] == [
        "0.250", "0.250", "0.500", "0.521",
    ]
    assert [rnd[header.index(h)] for h in ("Macro-F1", "Halluc.", "Grounded", "ECE")] == [
        "--", "--", "--", "--",
    ]


def test_grounded_rate_is_copied_into_its_column(results_tree):
    _write_run(
        results_tree, "medqa_V0_41f7e14_20260911T000000Z", "medqa", "V0",
        {"accuracy": 0.58, "grounded_rate": 0.0, "hallucination_rate": 1.0},
    )
    header, rows = _grid_rows(results_tree)
    v0 = rows[("V0 zero-shot", "medqa")]
    assert v0[header.index("Grounded")] == "0.000"  # a real 0.0, not a missing TBD
    assert rows[("V2 hybrid", "medqa")][header.index("Grounded")] == "TBD"  # no run


@pytest.mark.parametrize(
    "filled, expect, reject",
    [
        (set(), ["V0 and V2 have no n=500 test run yet", "Prepared, not run"], ["ran at n=500"]),
        ({"V0"}, ["V2 has no n=500 test run yet"], ["ran at n=500"]),
        ({"V0", "V2", "V5-refit"}, ["ran at n=500", "Run by"], ["no n=500 test run", "Prepared"]),
    ],
)
def test_status_notes_follow_the_filled_rows(filled, expect, reject):
    # the prose must never call a row "not run" while the table shows its number
    text = "\n".join(report._status_notes(filled))
    for s in expect:
        assert s in text
    for s in reject:
        assert s not in text


def test_ece_dict_is_unwrapped_to_its_number(results_tree):
    _write_run(
        results_tree,
        "medqa_V5_x_20260101T000000Z",
        "medqa",
        "V5",
        {"accuracy": 0.4, "ece": {"ece": 0.0724, "n_bins": 15}},
    )
    header, rows = _grid_rows(results_tree)
    assert rows[("V5 full", "medqa")][header.index("ECE")] == "0.072"


def test_provenance_guard_rejects_a_run_from_other_code(results_tree):
    _write_run(
        results_tree, "medqa_V2_0000000_20260911T000000Z", "medqa", "V2",
        {"accuracy": 0.5}, git_sha7="0000000",
    )
    with pytest.raises(AssertionError, match="refusing to tabulate"):
        report.select_cell_runs(report.load_runs())


def test_provenance_guard_rejects_a_dirty_run_affecting_path(results_tree):
    _write_run(
        results_tree, "medqa_V2_41f7e14-dirty_20260911T000000Z", "medqa", "V2",
        {"accuracy": 0.5}, git_sha7="41f7e14-dirty",
        git_dirty_files=["results/medqa_V1_x/", "src/eval/harness.py"],
    )
    with pytest.raises(AssertionError, match="dirty"):
        report.select_cell_runs(report.load_runs())


def test_dirty_results_only_run_passes_the_guard(results_tree):
    _write_run(
        results_tree, "medqa_V2_41f7e14-dirty_20260911T000000Z", "medqa", "V2",
        {"accuracy": 0.5}, git_sha7="41f7e14-dirty",
        git_dirty_files=["results/medqa_V1_x/", "docs/04_EVALUATION.md", "scripts/report.py"],
    )
    assert ("V2", "medqa") in report.select_cell_runs(report.load_runs())


def test_main_writes_both_files_and_is_idempotent(results_tree):
    _write_run(
        results_tree,
        "medqa_V5_x_20260101T000000Z",
        "medqa",
        "V5",
        {"accuracy": 0.4, "macro_f1": 0.39, "hallucination_rate": 0.1,
         "p_at_1": 0.4, "p_at_2": 0.7, "mrr": 0.6, "ece": {"ece": 0.07}},
    )
    _write_run(
        results_tree, "phase7_posthoc_binary_audit", "medqa", "post-hoc-binary",
        {"accuracy": 0.2, "macro_f1": 0.19},
    )
    _write_run(  # dev-20 diagnostic: must not appear next to n=500 rows
        results_tree, "medqa_B2eq_y_20260101T000000Z", "medqa", "B2eq",
        {"accuracy": 0.6, "macro_f1": 0.59}, split="train", limit=20,
    )
    assert report.main() == 0
    md = report.TABLES_DIR / "results_grid.md"
    tex = report.TABLES_DIR / "results_grid.tex"
    first_md, first_tex = md.read_text(encoding="utf-8"), tex.read_text(encoding="utf-8")
    assert "V5 full" in first_md
    assert "post-hoc-binary" in first_md  # diagnostics section
    assert "B2eq" not in first_md
    assert "0.400" in first_md
    assert r"\toprule" in first_tex
    assert report.main() == 0
    assert md.read_text(encoding="utf-8") == first_md  # byte-identical on re-run
    assert tex.read_text(encoding="utf-8") == first_tex


def test_pubmedqa_baseline_does_not_leak_into_v_grid(results_tree):
    _write_run(
        results_tree, "pubmedqa_B2_z_20260101T000000Z", "pubmedqa", "B2",
        {"accuracy": 0.3},
    )
    chosen = report.select_cell_runs(report.load_runs())
    assert ("V3", "pubmedqa") not in chosen  # a B-baseline is never a grid cell
