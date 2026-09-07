"""Phase 6: scripts/report.py -- table generation from results/ fixtures.

No real runs: a synthetic results/ tree is built in tmp_path.
"""

from __future__ import annotations

import json

import pytest

from scripts import report


def _write_run(root, run_id, dataset, variant, metrics, *, superseded=False):
    d = root / run_id
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "dataset": dataset,
                "variant": variant,
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


def test_b2_medqa_maps_to_v3_row_not_v2(results_tree):
    _write_run(
        results_tree,
        "medqa_B2_abc1234_20260906T101648Z",
        "medqa",
        "B2",
        {"accuracy": 0.65, "macro_f1": 0.61},
    )
    chosen = report.select_cell_runs(report.load_runs())
    assert ("V3", "medqa") in chosen
    assert ("V2", "medqa") not in chosen  # V2 has never been run


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
    grid = report.build_grid(report.select_cell_runs(report.load_runs()))
    header, *rows = grid
    by_label = {(r[0], r[1]): r for r in rows}
    v0 = by_label[("V0 zero-shot", "medqa")]
    assert v0[header.index("Acc")] == "0.550"
    assert v0[header.index("P@1")] == "TBD"  # no ranking metric for V0
    v3_pqa = by_label[("V3 +KG", "pubmedqa")]
    assert v3_pqa[header.index("Acc")] == "TBD"  # no run
    assert v3_pqa[header.index("P@1")] == "--"  # pubmedqa: ranking not defined


def test_ece_dict_is_unwrapped_to_its_number(results_tree):
    _write_run(
        results_tree,
        "medqa_V5_x_20260101T000000Z",
        "medqa",
        "V5",
        {"accuracy": 0.4, "ece": {"ece": 0.0724, "n_bins": 15}},
    )
    grid = report.build_grid(report.select_cell_runs(report.load_runs()))
    header, *rows = grid
    v5 = next(r for r in rows if r[0] == "V5 full" and r[1] == "medqa")
    assert v5[header.index("ECE")] == "0.072"


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
        results_tree, "medqa_B2eq_y_20260101T000000Z", "medqa", "B2eq",
        {"accuracy": 0.6, "macro_f1": 0.59},
    )
    assert report.main() == 0
    md = report.TABLES_DIR / "results_grid.md"
    tex = report.TABLES_DIR / "results_grid.tex"
    first_md, first_tex = md.read_text(encoding="utf-8"), tex.read_text(encoding="utf-8")
    assert "V5 full" in first_md
    assert "B2eq" in first_md  # diagnostics section
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
    assert ("V3", "pubmedqa") not in chosen  # a pubmedqa smoke run is not a grid cell
