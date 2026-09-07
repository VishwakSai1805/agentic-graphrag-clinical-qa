"""Unit tests for scripts/fit_calibration.py's pure logic: loading, the
fit/held-out split, and the ranked-output builder. No LLM, no GPU."""

from __future__ import annotations

import json

import pytest

from scripts.fit_calibration import (
    build_ranked_output,
    load_matrices,
    paired_calibrated_vs_raw,
    split_fit_held_out,
)
from src.agents.state import OptionSupport, SupportCell, SupportMatrix
from src.calibration.scoring import CalibrationWeights


def _matrix(qid, gold="A", cells=None, option_support=None):
    return SupportMatrix(
        question_id=qid,
        dataset="medqa",
        query="q?",
        options={"A": "a", "B": "b", "C": "c", "D": "d"},
        gold_option=gold,
        cells=cells or [],
        option_support=option_support or [OptionSupport(option_id=o) for o in "ABCD"],
    )


def test_load_matrices_reads_predictions_jsonl(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    m = _matrix("train_00001")
    (run_dir / "predictions.jsonl").write_text(
        json.dumps(m.model_dump(mode="json")) + "\n", encoding="utf-8"
    )
    loaded = load_matrices(run_dir)
    assert len(loaded) == 1
    assert loaded[0].question_id == "train_00001"


def test_load_matrices_skips_blank_lines(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    m = _matrix("train_00001")
    (run_dir / "predictions.jsonl").write_text(
        json.dumps(m.model_dump(mode="json")) + "\n\n\n", encoding="utf-8"
    )
    assert len(load_matrices(run_dir)) == 1


def test_load_matrices_raises_clear_error_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="predictions.jsonl"):
        load_matrices(tmp_path / "does-not-exist")


def test_split_fit_held_out_is_disjoint_and_covers_everything():
    matrices = [_matrix(f"train_{i:05d}") for i in range(10)]
    fit, held_out = split_fit_held_out(matrices)
    fit_ids = {m.question_id for m in fit}
    held_out_ids = {m.question_id for m in held_out}
    assert fit_ids.isdisjoint(held_out_ids)
    assert fit_ids | held_out_ids == {m.question_id for m in matrices}


def test_split_fit_held_out_is_first_half_by_sorted_id():
    # Deliberately out-of-order input -- the split must sort first.
    matrices = [_matrix("train_00003"), _matrix("train_00001"), _matrix("train_00002")]
    fit, held_out = split_fit_held_out(matrices)
    assert [m.question_id for m in fit] == ["train_00001"]
    assert [m.question_id for m in held_out] == ["train_00002", "train_00003"]


def test_split_fit_held_out_even_split_for_even_n():
    matrices = [_matrix(f"train_{i:05d}") for i in range(120)]
    fit, held_out = split_fit_held_out(matrices)
    assert len(fit) == 60
    assert len(held_out) == 60


def test_build_ranked_output_every_question_has_a_row_for_every_option():
    cells = [SupportCell(option_id="A", claim_idx=0, evidence_id="chunk::1", label="SUPPORT")]
    m = _matrix(
        "train_00001",
        gold="A",
        cells=cells,
        option_support=[OptionSupport(option_id=o, s_pos=1 if o == "A" else 0) for o in "ABCD"],
    )
    rows = build_ranked_output([m], CalibrationWeights())
    assert len(rows) == 1
    assert len(rows[0]["ranked"]) == 4  # one entry per option, always


def test_build_ranked_output_includes_citations_key_for_every_option():
    cells = [SupportCell(option_id="A", claim_idx=0, evidence_id="chunk::1", label="SUPPORT")]
    m = _matrix("train_00001", cells=cells)
    rows = build_ranked_output([m], CalibrationWeights())
    for entry in rows[0]["ranked"]:
        assert "citations" in entry
        assert isinstance(entry["citations"], list)


def test_build_ranked_output_sorted_descending_by_probability():
    cells = [
        SupportCell(option_id="A", claim_idx=0, evidence_id="chunk::1", label="SUPPORT"),
        SupportCell(option_id="A", claim_idx=1, evidence_id="chunk::2", label="SUPPORT"),
    ]
    m = _matrix(
        "train_00001",
        cells=cells,
        option_support=[OptionSupport(option_id=o, s_pos=2 if o == "A" else 0) for o in "ABCD"],
    )
    rows = build_ranked_output([m], CalibrationWeights())
    ps = [entry["p"] for entry in rows[0]["ranked"]]
    assert ps == sorted(ps, reverse=True)
    assert rows[0]["ranked"][0]["option_id"] == "A"


# --------------------------------------------------------------------------
# paired_calibrated_vs_raw (joins by question_id, never by position)
# --------------------------------------------------------------------------


def _raw_row(qid, raw_probs):
    return {"question_id": qid, "raw_probs": raw_probs}


def test_paired_calibrated_vs_raw_aligns_by_question_id():
    m1 = _matrix("train_00001", gold="A")
    m2 = _matrix("train_00002", gold="B")
    # deliberately reversed order in raw_by_qid vs. the matrices list
    raw_by_qid = {
        "train_00002": _raw_row("train_00002", {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}),
        "train_00001": _raw_row("train_00001", {"A": 0.4, "B": 0.2, "C": 0.2, "D": 0.2}),
    }
    calibrated, raw, gold_idx = paired_calibrated_vs_raw([m1, m2], CalibrationWeights(), raw_by_qid)
    assert len(calibrated) == len(raw) == len(gold_idx) == 2
    # question order follows the matrices list, not raw_by_qid's dict order
    assert gold_idx[0] == sorted(m1.options).index("A")
    assert gold_idx[1] == sorted(m2.options).index("B")


def test_paired_calibrated_vs_raw_skips_a_question_missing_from_raw(capsys):
    m1 = _matrix("train_00001", gold="A")
    m2 = _matrix("train_00002", gold="B")
    raw_by_qid = {"train_00001": _raw_row("train_00001", {"A": 0.4, "B": 0.2, "C": 0.2, "D": 0.2})}
    calibrated, raw, gold_idx = paired_calibrated_vs_raw([m1, m2], CalibrationWeights(), raw_by_qid)
    assert len(calibrated) == 1  # only train_00001 survived
    captured = capsys.readouterr()
    assert "1 question(s) missing" in captured.out


def test_paired_calibrated_vs_raw_skips_on_gold_mismatch(capsys):
    m1 = _matrix("train_00001", gold="A")
    # raw side has no "A" key at all -- a malformed/incomplete raw row
    raw_by_qid = {"train_00001": _raw_row("train_00001", {"B": 0.5, "C": 0.3, "D": 0.2})}
    calibrated, raw, gold_idx = paired_calibrated_vs_raw([m1], CalibrationWeights(), raw_by_qid)
    assert len(calibrated) == 0
    captured = capsys.readouterr()
    assert "gold-option mismatch" in captured.out


def test_paired_calibrated_vs_raw_probability_vectors_use_same_option_order():
    m1 = _matrix("train_00001", gold="C")
    raw_by_qid = {"train_00001": _raw_row("train_00001", {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4})}
    calibrated, raw, gold_idx = paired_calibrated_vs_raw([m1], CalibrationWeights(), raw_by_qid)
    # raw vector must be in sorted-option-id order: A, B, C, D
    assert raw[0] == [0.1, 0.2, 0.3, 0.4]
    assert gold_idx[0] == 2  # "C" is index 2 in sorted ["A","B","C","D"]


def test_paired_calibrated_vs_raw_empty_matrices_returns_empty():
    calibrated, raw, gold_idx = paired_calibrated_vs_raw([], CalibrationWeights(), {})
    assert calibrated == raw == gold_idx == []


# --------------------------------------------------------------------------
# End-to-end: temperature-scaled raw-logprob ablation (main(), via CLI)
# --------------------------------------------------------------------------


def _write_run(tmp_path, matrices):
    run_dir = tmp_path / "fake_run"
    run_dir.mkdir()
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for m in matrices:
            f.write(json.dumps(m.model_dump(mode="json")) + "\n")
    return run_dir


def _write_raw_baseline(tmp_path, matrices, seed=0):
    import random

    rng = random.Random(seed)
    rows = []
    for m in matrices:
        letters = sorted(m.options)
        vals = [max(0.01, rng.random()) for _ in letters]
        total = sum(vals)
        raw_probs = {letters[i]: vals[i] / total for i in range(len(letters))}
        rows.append(
            {
                "question_id": m.question_id,
                "gold_letter": m.gold_option,
                "letters": letters,
                "raw_probs": raw_probs,
            }
        )
    path = tmp_path / "raw_baseline.json"
    path.write_text(json.dumps({"per_question": rows}), encoding="utf-8")
    return path


def test_main_writes_temperature_scaled_ablation_block(tmp_path, monkeypatch):
    import scripts.fit_calibration as mod

    matrices = [
        _matrix(
            f"train_{i:05d}",
            gold="A",
            cells=[
                SupportCell(
                    option_id="A", claim_idx=0, evidence_id=f"chunk::{i}#0", label="SUPPORT"
                )
            ],
            option_support=[
                OptionSupport(option_id=o, s_pos=1 if o == "A" else 0) for o in "ABCD"
            ],
        )
        for i in range(20)
    ]
    run_dir = _write_run(tmp_path, matrices)
    raw_path = _write_raw_baseline(tmp_path, matrices)
    out_path = tmp_path / "out.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "fit_calibration",
            "--run-dir",
            str(run_dir),
            "--raw-baseline-json",
            str(raw_path),
            "--n-bootstrap",
            "5",
            "--n-ece-bootstrap",
            "50",
            "--out",
            str(out_path),
        ],
    )
    mod.main()

    out = json.loads(out_path.read_text(encoding="utf-8"))
    ablation = out["exit_test"]["temperature_scaled_raw_ablation"]
    assert "fitted_temperature" in ablation
    assert 0.05 - 1e-6 <= ablation["fitted_temperature"] <= 100_000.0
    assert "tscaled_raw_ece_held_out_point" in ablation
    for key in ("calibrated_vs_tscaled", "raw_vs_tscaled"):
        assert set(("point_estimate_diff", "ci_lo", "ci_hi", "verdict")) <= set(ablation[key])


def test_main_without_raw_baseline_json_has_no_exit_test(tmp_path, monkeypatch):
    import scripts.fit_calibration as mod

    matrices = [_matrix(f"train_{i:05d}") for i in range(20)]
    run_dir = _write_run(tmp_path, matrices)
    out_path = tmp_path / "out.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "fit_calibration",
            "--run-dir",
            str(run_dir),
            "--n-bootstrap",
            "5",
            "--out",
            str(out_path),
        ],
    )
    mod.main()

    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert out["exit_test"] is None


def test_main_auto_widens_temperature_search_when_saturated(tmp_path, monkeypatch, capsys):
    """Regression test for a real bug: a too-narrow --temp-search-max silently
    saturates the grid search, reporting a materially wrong 'optimal'
    temperature. main() must detect saturation and retry with a wider range,
    warning about it, rather than trusting the boundary value."""
    import scripts.fit_calibration as mod

    # Construct raw probs where the truly NLL-optimal T is well above a
    # deliberately tiny --temp-search-max (2.0), forcing saturation.
    matrices = [
        _matrix(f"train_{i:05d}", gold="A" if i % 2 == 0 else "B") for i in range(20)
    ]
    run_dir = _write_run(tmp_path, matrices)

    rows = []
    for m in matrices:
        letters = sorted(m.options)
        # Sharp, mostly-wrong raw confidence -- needs a large T to fix, well
        # past a --temp-search-max of 2.0.
        vals = [0.94 if letters[i] == "C" else 0.02 for i in range(len(letters))]
        raw_probs = {letters[i]: vals[i] for i in range(len(letters))}
        rows.append(
            {
                "question_id": m.question_id,
                "gold_letter": m.gold_option,
                "letters": letters,
                "raw_probs": raw_probs,
            }
        )
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps({"per_question": rows}), encoding="utf-8")
    out_path = tmp_path / "out.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "fit_calibration",
            "--run-dir",
            str(run_dir),
            "--raw-baseline-json",
            str(raw_path),
            "--n-bootstrap",
            "5",
            "--n-ece-bootstrap",
            "50",
            "--temp-search-max",
            "2.0",
            "--temp-search-grid",
            "200",
            "--out",
            str(out_path),
        ],
    )
    mod.main()

    captured = capsys.readouterr()
    assert "saturated the search bound" in captured.out
    out = json.loads(out_path.read_text(encoding="utf-8"))
    fitted_t = out["exit_test"]["temperature_scaled_raw_ablation"]["fitted_temperature"]
    assert fitted_t > 2.0  # the retry actually found something past the original bound
