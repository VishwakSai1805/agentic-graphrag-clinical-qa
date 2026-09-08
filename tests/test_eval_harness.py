"""Phase 6: src/eval/harness.py -- variant table, scoring, and one end-to-end
run with the LLM + NLI model fully mocked.
"""

from __future__ import annotations

import json
import types

import pytest

from src.agents.state import OptionSupport, SupportMatrix
from src.config import load_config
from src.eval.harness import (
    VARIANTS,
    QuestionRecord,
    resolve_variant,
    run_eval,
    score_run,
)

CONFIG = "configs/base.yaml"
AGENTS_CONFIG = "configs/agents.yaml"
V5_CONFIG = "configs/v5.yaml"


# --------------------------------------------------------------------------
# variant table
# --------------------------------------------------------------------------


def test_variant_modality_sets_are_the_documented_ones():
    assert VARIANTS["V0"].baseline == "B0"
    assert VARIANTS["V1"].modalities == frozenset({"dense"})
    assert VARIANTS["V2"].modalities == frozenset({"dense", "lexical"})
    assert VARIANTS["V3"].modalities == frozenset({"dense", "lexical", "kg"})
    assert VARIANTS["V4"].kind == "agents" and not VARIANTS["V4"].calibration
    assert VARIANTS["V5"].kind == "agents" and VARIANTS["V5"].calibration


def test_v2_and_v3_differ_only_by_kg():
    assert VARIANTS["V3"].modalities - VARIANTS["V2"].modalities == {"kg"}


def test_v4_conflict_variant_is_an_uncalibrated_agent_cell():
    spec = VARIANTS["V4-conflict"]
    assert spec.kind == "agents" and spec.calibration is False


def test_resolve_variant_from_config():
    assert resolve_variant(load_config(V5_CONFIG)) == "V5"
    assert resolve_variant(load_config(AGENTS_CONFIG)) == "V4"
    with pytest.raises(ValueError, match="pass --variant"):
        resolve_variant(load_config(CONFIG))


# --------------------------------------------------------------------------
# score_run
# --------------------------------------------------------------------------


def _fake_nli(entail: float):
    return lambda premise, hypothesis: entail


def _baseline_record(qid, pred, gold, *, claim="a claim here", entailed_cite=True):
    from src.eval.nli import ClaimRecord

    return QuestionRecord(
        question_id=qid,
        gold_option=gold,
        pred_option=pred,
        ranked=[pred] if pred else [],
        p_vector=None,
        ranking_source="single_prediction",
        citations={pred: ["chunk::x#1"]} if pred else {},
        invalid_citations=[],
        grounded=True,
        rationale_claims=[claim],
        claim_records=[
            ClaimRecord(
                question_id=qid,
                option_id=pred,
                claim=claim,
                cited_evidence_texts=["evidence"] if entailed_cite else [],
                predicted_option=True,
            )
        ],
        status="ok",
        n_llm_calls=1,
    )


def test_score_run_baseline_nulls_ranking_metrics():
    recs = [
        _baseline_record("q1", "A", "A"),
        _baseline_record("q2", "B", "A"),
    ]
    m = score_run(recs, dataset="medqa", nli_scorer=_fake_nli(0.9))
    assert m["n_evaluated"] == 2
    assert m["accuracy"] == 0.5
    assert m["p_at_1"] is None and m["p_at_2"] is None and m["mrr"] is None
    assert m["ece"] is None
    assert m["hallucination_rate"] == 0.0  # both claims entailed
    assert m["hallucination"]["primary"] == "hallucination_rate_predicted_option"


def test_score_run_baseline_counts_ungrounded_claims():
    recs = [_baseline_record("q1", "A", "A", entailed_cite=False)]
    m = score_run(recs, dataset="medqa", nli_scorer=_fake_nli(0.9))
    assert m["hallucination_rate"] == 1.0  # uncited -> ungrounded


def _agent_matrix(qid, gold, s_pos: dict[str, int]):
    return SupportMatrix(
        question_id=qid,
        dataset="medqa",
        query="q?",
        options={k: k.lower() for k in s_pos},
        gold_option=gold,
        llm_calls=6,
        rounds=1,
        option_support=[
            OptionSupport(option_id=k, s_pos=v, s_neg=0) for k, v in s_pos.items()
        ],
    ).model_dump(mode="json")


def _agent_record(qid, gold, ranked, *, p_vector=None, s_pos=None):
    src = "calibrated_softmax" if p_vector else "uncalibrated_support_margin"
    return QuestionRecord(
        question_id=qid,
        gold_option=gold,
        pred_option=ranked[0],
        ranked=ranked,
        p_vector=p_vector,
        ranking_source=src,
        citations={r: [] for r in ranked},
        invalid_citations=[],
        grounded=False,
        rationale_claims=[],
        claim_records=[],
        status="ok",
        n_llm_calls=6,
        agent_matrix=_agent_matrix(qid, gold, s_pos or {r: 0 for r in ranked}),
    )


def test_score_run_v4_ranking_from_support_margin_no_ece():
    recs = [
        _agent_record("q1", "A", ["A", "B", "C", "D"]),  # gold rank 1
        _agent_record("q2", "C", ["A", "C", "B", "D"]),  # gold rank 2
    ]
    m = score_run(recs, dataset="medqa", nli_scorer=_fake_nli(0.5))
    assert m["ranking_source"] == "uncalibrated_support_margin"
    assert m["p_at_1"] == 0.5  # gold rank-1 on q1 only
    assert m["p_at_2"] == 1.0  # gold in top-2 on both
    assert m["mrr"] == pytest.approx((1.0 + 0.5) / 2)
    assert m["ece"] is None  # V4 has no probability vector
    assert m["agent_kg_hit_rate"] == 0.0


def test_score_run_v5_computes_ece_from_p_vector():
    recs = [
        _agent_record("q1", "A", ["A", "B"], p_vector=[0.9, 0.1]),
        _agent_record("q2", "A", ["A", "B"], p_vector=[0.9, 0.1]),
        _agent_record("q3", "B", ["A", "B"], p_vector=[0.9, 0.1]),  # confident + wrong
    ]
    m = score_run(recs, dataset="medqa", nli_scorer=_fake_nli(0.5))
    assert m["ranking_source"] == "calibrated_softmax"
    assert isinstance(m["ece"], dict)
    assert m["ece"]["ece"] is not None
    assert "abstention_rate" in m
    assert m["p_at_1"] == pytest.approx(2 / 3)


def test_score_run_pubmedqa_uses_three_class_labels():
    recs = [_baseline_record("q1", "C", "C"), _baseline_record("q2", "A", "B")]
    m = score_run(recs, dataset="pubmedqa", nli_scorer=_fake_nli(0.9))
    # macro_f1 computed over {A,B,C}, never a phantom D
    assert 0.0 <= m["macro_f1"] <= 1.0


# --------------------------------------------------------------------------
# run_eval end-to-end (V0, LLM + NLI mocked, tmp results dir)
# --------------------------------------------------------------------------


def test_run_eval_v0_end_to_end(tmp_path, monkeypatch):
    from src.kg import dataset_questions
    from src.kg.dataset_questions import Question

    qs = [
        Question(
            question_id=f"train_{i:05d}",
            dataset="medqa",
            split="train",
            question=f"Question {i}?",
            options={"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
            answer="alpha",
        )
        for i in range(4)
    ]
    monkeypatch.setattr(dataset_questions, "load_question_sample", lambda *a, **k: qs)

    def fake_chat(prompt, **kwargs):
        return types.SimpleNamespace(
            text='{"answer": "A", "rationale": "Alpha is correct here. It fits the stem."}'
        )

    monkeypatch.setattr("src.eval.baselines.chat", fake_chat)

    run_dir = run_eval(
        CONFIG,
        dataset="medqa",
        split="train",
        limit=4,
        variant="V0",
        nli_scorer=lambda premise, hypothesis: 0.95,
        results_dir=tmp_path,
    )
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    assert metrics["n_evaluated"] == 4
    assert metrics["accuracy"] == 1.0  # gold is always "A"
    assert metrics["p_at_1"] is None  # single-prediction variant
    # V0 zero-shot cites nothing (ungrounded by construction, like grounded_rate
    # == 0.0) -> every claim is ungrounded -> hallucination_rate == 1.0
    assert metrics["hallucination_rate"] == 1.0
    assert metrics["grounded_rate"] == 0.0
    assert (run_dir / "predictions.jsonl").exists()
    preds = [
        json.loads(x)
        for x in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(preds) == 4
    assert preds[0]["rationale_claims"]  # rationale was captured and split
