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
