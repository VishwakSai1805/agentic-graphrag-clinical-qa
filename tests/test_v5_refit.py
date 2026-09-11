"""scripts/analysis/v5_refit.py -- the V5-refit rule (ADR-043). Synthetic
matrices only; nothing here reads a results/ run."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from scripts.analysis import v5_refit
from src.agents.state import SupportCell, SupportMatrix
from src.calibration import softmax_rank
from src.calibration.scoring import CalibrationWeights, OptionFeatures


def _cell(oid, label, rerank, *, ev="chunk::d#1", cidx=0):
    return SupportCell(
        option_id=oid,
        claim_idx=cidx,
        evidence_id=ev,
        label=label,
        rationale="" if label == "NEUTRAL" else "because",
        modality="passage",
        rerank_score=rerank,
    )


def _matrix(qid, cells, gold="A"):
    return SupportMatrix(
        question_id=qid,
        dataset="medqa",
        query="q",
        options={k: k.lower() for k in "ABCD"},
        gold_option=gold,
        cells=cells,
        option_support=[],  # empty: the stale agent_error shape (ADR-042)
    )


def _feat(oid, s_pos=0, s_neg=0, rr=0.0):
    return OptionFeatures(
        option_id=oid,
        s_pos=s_pos,
        s_neg=s_neg,
        s_neu=0,
        kg_support_count=0,
        mean_rerank_score=rr,
        resolved=True,
    )


def _examples(n=40, seed=0, support_hurts=False):
    """Synthetic fit examples. With `support_hurts`, the gold option carries
    fewer SUPPORT cells than its distractors, so an unconstrained fit wants
    alpha < 0."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        gold = "ABCD"[rng.randrange(4)]
        feats = []
        for oid in "ABCD":
            extra = int((oid == gold) != support_hurts)
            feats.append(
                _feat(
                    oid,
                    s_pos=rng.randrange(2) + extra,
                    s_neg=rng.randrange(3),
                    rr=rng.gauss(0.0, 2.0),
                )
            )
        out.append((feats, gold))
    return out


def test_refit_features_counts_cells_and_centres_rerank():
    m = _matrix(
        "train_1",
        [
            _cell("A", "SUPPORT", 8.0),
            _cell("A", "SUPPORT", 6.0, ev="chunk::d#2"),
            _cell("B", "CONTRADICT", 4.0),
            _cell("C", "NEUTRAL", 9.0),
        ],
    )
    f = {x.option_id: x for x in v5_refit.refit_features(m, r_ref=6.0)}
    # counted from cells even though option_support is empty
    assert (f["A"].s_pos, f["A"].s_neg, f["A"].mean_rerank_score) == (2, 0, 1.0)
    assert (f["B"].s_pos, f["B"].s_neg, f["B"].mean_rerank_score) == (0, 1, -2.0)
    # NEUTRAL-only and cell-less options carry no signal: the reference itself
    assert f["C"].mean_rerank_score == 0.0
    assert f["D"].mean_rerank_score == 0.0


def test_rerank_reference_is_median_over_signal_options_only():
    ms = [
        _matrix("train_1", [_cell("A", "SUPPORT", 2.0), _cell("B", "CONTRADICT", 10.0)]),
        _matrix("train_2", [_cell("C", "SUPPORT", 4.0), _cell("C", "SUPPORT", 6.0, ev="x")]),
    ]
    # option means 2.0, 10.0, 5.0 -> median 5.0; the five cell-less options are excluded
    assert v5_refit.rerank_reference(ms) == 5.0


def test_refit_bounds_only_raise_the_evidential_lower_bounds():
    for name in ("alpha", "beta", "gamma"):
        assert v5_refit.REFIT_BOUNDS[name] == (0.0, softmax_rank._BOUNDS[name][1])
    for name in ("delta", "log_t"):
        assert v5_refit.REFIT_BOUNDS[name] == softmax_rank._BOUNDS[name]


def test_bounded_fit_reproduces_fit_weights_under_v5_bounds():
    ex = _examples()
    assert v5_refit.fit_weights_bounded(ex, softmax_rank._BOUNDS, max_iters=60) == (
        softmax_rank.fit_weights(ex, max_iters=60)
    )


def test_evidential_weights_respect_zero_bound_where_unconstrained_goes_negative():
    ex = _examples(support_hurts=True)
    free = v5_refit.fit_weights_bounded(ex, softmax_rank._BOUNDS, max_iters=200)
    assert free.alpha < 0  # the fixture really does push alpha negative
    refit = v5_refit.fit_weights_bounded(ex, v5_refit.REFIT_BOUNDS, max_iters=200)
    assert refit.alpha >= 0.0 and refit.beta >= 0.0 and refit.gamma >= 0.0


def test_init_outside_bounds_is_refused():
    with pytest.raises(ValueError):
        v5_refit.fit_weights_bounded(
            _examples(), v5_refit.REFIT_BOUNDS, init=CalibrationWeights(alpha=-1.0)
        )


def test_tie_break_is_deterministic_order_independent_and_letter_free():
    w = CalibrationWeights(alpha=1.0, beta=1.0, temperature=1.0)
    feats = [_feat(o) for o in "ABCD"]  # every option at z = 0
    first = v5_refit.rank_refit(feats, w, "test_00001")
    assert first == v5_refit.rank_refit(list(reversed(feats)), w, "test_00001")
    tops = Counter(v5_refit.rank_refit(feats, w, f"test_{i:05d}")[0][0] for i in range(4000))
    assert set(tops) == set("ABCD")
    assert all(850 < c < 1150 for c in tops.values())
    # V5's rule: always the earliest letter
    assert softmax_rank.rank_options(feats, w)[0][0] == "A"


def test_tie_break_never_reorders_untied_options():
    w = CalibrationWeights(alpha=1.0, beta=1.0, temperature=1.0)
    feats = [_feat("A"), _feat("B", s_pos=2), _feat("C", s_pos=1), _feat("D")]
    for i in range(50):
        ranked = [o for o, _ in v5_refit.rank_refit(feats, w, f"q{i}")]
        assert ranked[:2] == ["B", "C"]
        assert set(ranked[2:]) == {"A", "D"}


def test_spec_digest_covers_prediction_fields_only():
    spec = {
        "spec_version": 1,
        "variant": "V5-refit",
        "rule": {"rerank_reference": 5.0},
        "weights": {"alpha": 0.1},
        "abstention_tau": 0.5,
        "generated_at_utc": "t0",
    }
    d = v5_refit.spec_digest(spec)
    assert v5_refit.spec_digest({**spec, "generated_at_utc": "t1"}) == d
    assert v5_refit.spec_digest({**spec, "weights": {"alpha": 0.2}}) != d
    assert v5_refit.spec_digest({**spec, "rule": {"rerank_reference": 5.1}}) != d


def test_fit_refuses_any_non_train_question():
    with pytest.raises(SystemExit):
        v5_refit.assert_train_only([_matrix("train_1", []), _matrix("test_00001", [])])
    v5_refit.assert_train_only([_matrix("train_1", [])])


def _v5_row(matrix):
    return {
        "question_id": matrix.question_id,
        "gold_option": matrix.gold_option,
        "agent_matrix": matrix.model_dump(mode="json"),
        "claim_records": [
            {
                "question_id": matrix.question_id,
                "option_id": "A",
                "claim": "ca",
                "cited_evidence_texts": ["t"],
                "predicted_option": True,
            },
            {
                "question_id": matrix.question_id,
                "option_id": "B",
                "claim": "cb",
                "cited_evidence_texts": ["t"],
                "predicted_option": False,
            },
        ],
        "invalid_citations": [],
        "status": "ok",
        "n_llm_calls": 5,
    }


def test_rescore_rows_reranks_and_reflags_the_predicted_option():
    m = _matrix(
        "test_00001",
        [
            _cell("A", "CONTRADICT", 6.0, ev="chunk::a#1"),
            _cell("A", "CONTRADICT", 6.0, ev="chunk::a#2"),
            _cell("B", "SUPPORT", 6.0, ev="chunk::b#1"),
        ],
        gold="B",
    )
    spec = {
        "weights": CalibrationWeights(alpha=1.0, beta=1.0, temperature=1.0).model_dump(),
        "rule": {"rerank_reference": 6.0, "tie_break": {"seed": 42}},
        "abstention_tau": 0.0,
    }
    (rec,) = v5_refit.rescore_rows([_v5_row(m)], spec)
    assert rec.pred_option == "B"
    assert rec.ranked[-1] == "A"
    assert abs(sum(rec.p_vector) - 1.0) < 1e-12
    assert rec.ranking_source == "calibrated_softmax"
    assert rec.grounded and rec.citations["B"] == ["chunk::b#1"]
    flags = {cr.option_id: cr.predicted_option for cr in rec.claim_records}
    assert flags == {"A": False, "B": True}
    assert rec.rationale_claims == ["cb"]


def test_reflag_hallucination_rescopes_to_the_new_prediction():
    v5_hall = {
        "threshold": 0.5,
        "nli_model": "m",
        "per_claim": [
            {
                "question_id": "q",
                "option_id": "A",
                "claim": "ca",
                "predicted_option": True,
                "cited": True,
                "max_entail_score": 0.1,
                "entailed": False,
            },
            {
                "question_id": "q",
                "option_id": "B",
                "claim": "cb",
                "predicted_option": False,
                "cited": True,
                "max_entail_score": 0.9,
                "entailed": True,
            },
        ],
    }
    h = v5_refit.reflag_hallucination(v5_hall, {"q": "B"})
    assert h["hallucination_rate_predicted_option"] == 0.0
    assert h["n_claims_predicted_option"] == 1
    assert h["hallucination_rate_all_support_cells"] == 0.5


def test_apply_end_to_end_on_a_synthetic_v5_run(tmp_path, monkeypatch):
    """The whole `apply` path on a synthetic 500-question V5 run dir -- never
    the real one -- then the unplanned family's own V5-refit validation."""
    import json

    import yaml

    from scripts import report
    from scripts.analysis import unplanned_v0_v2_family as fam
    from src.calibration.scoring import extract_option_features

    v5_w = CalibrationWeights(alpha=-0.0139, beta=0.11, gamma=0.29, delta=0.0077, temperature=0.21)
    rng = random.Random(7)
    results = tmp_path / "results"
    v5_dir = results / v5_refit.V5_RUN_DIR.name
    v5_dir.mkdir(parents=True)
    rows, per_claim = [], []
    for i in range(500):
        qid = f"test_{i:05d}"
        cells = [
            _cell(
                o,
                rng.choice(["SUPPORT", "CONTRADICT", "NEUTRAL"]),
                rng.uniform(0, 12),
                ev=f"chunk::{qid}{o}#{k}",
                cidx=k,
            )
            for o in "ABCD"
            for k in range(rng.randrange(3))
        ]
        m = _matrix(qid, cells, gold="ABCD"[rng.randrange(4)])
        m.status = "ok"
        ranked = [o for o, _ in softmax_rank.rank_options(extract_option_features(m), v5_w)]
        claims = [
            {
                "question_id": qid,
                "option_id": c.option_id,
                "claim": c.rationale,
                "cited_evidence_texts": ["t"],
                "predicted_option": c.option_id == ranked[0],
            }
            for c in cells
            if c.label == "SUPPORT"
        ]
        rows.append(
            {
                "question_id": qid,
                "gold_option": m.gold_option,
                "pred_option": ranked[0],
                "ranked": ranked,
                "agent_matrix": m.model_dump(mode="json"),
                "claim_records": claims,
                "invalid_citations": [],
                "status": "ok",
                "n_llm_calls": 5,
            }
        )
        per_claim += [
            {
                "question_id": qid,
                "option_id": c["option_id"],
                "claim": c["claim"],
                "predicted_option": c["predicted_option"],
                "cited": True,
                "max_entail_score": 0.9,
                "entailed": bool(rng.randrange(2)),
            }
            for c in claims
        ]
    (v5_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    cal = {
        **v5_w.model_dump(),
        "enabled": True,
        "abstention_tau": 0.5,
        "n_ece_bins": 15,
        "ece_binning": "equal_width",
    }
    (v5_dir / "config.yaml").write_text(
        yaml.safe_dump({"seed": 42, "calibration": cal}), encoding="utf-8"
    )
    (v5_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": v5_dir.name,
                "variant": "V5",
                "dataset": "medqa",
                "split": "test",
                "limit": 500,
                "seed": 42,
                "git_sha7": "41f7e14",
                "git_dirty_files": [],
                "superseded": False,
                "metrics": {
                    "n_evaluated": 500,
                    "contamination_assertion": {"checked": True},
                    "hallucination": {"threshold": 0.5, "nli_model": "m", "per_claim": per_claim},
                },
            }
        ),
        encoding="utf-8",
    )
    spec = {
        "spec_version": 1,
        "variant": "V5-refit",
        "rule": {"rerank_reference": 6.0, "tie_break": {"seed": 42}},
        "weights": CalibrationWeights(
            alpha=0.006, beta=0.074, gamma=0.27, delta=0.0135, temperature=0.185
        ).model_dump(),
        "abstention_tau": 0.47,
        "fit_question_ids": ["train_00001"],
        "held_out_question_ids": ["train_00002"],
    }
    spec["spec_sha256"] = v5_refit.spec_digest(spec)
    spec_path = tmp_path / "refit_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    monkeypatch.setattr(v5_refit, "SPEC_PATH", spec_path)
    monkeypatch.setattr(v5_refit, "V5_RUN_DIR", v5_dir)
    monkeypatch.setattr(v5_refit, "RESULTS_DIR", results)
    monkeypatch.setattr(v5_refit, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(report, "code_matches_launch_commit", lambda run: (True, "test"))
    monkeypatch.setattr(fam, "code_matches_launch_commit", lambda run: (True, "test"))

    v5_refit.cmd_apply(None)

    (run_dir,) = results.glob("medqa_V5-refit_*")
    doc = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    preds = [json.loads(x) for x in (run_dir / "predictions.jsonl").read_text().splitlines()]
    met = doc["metrics"]
    assert doc["variant"] == "V5-refit" and doc["derived_from"] == v5_dir.name
    assert doc["refit_spec_sha256"] == spec["spec_sha256"]
    assert met["n_evaluated"] == len(preds) == 500
    assert met["ranking_source"] == "calibrated_softmax"
    assert met["accuracy"] == sum(p["pred_option"] == p["gold_option"] for p in preds) / 500
    assert met["p_at_2"] is not None and met["ece"]["n"] == 500
    assert met["llm_calls_total"] == 0 and met["hallucination_pending"] is False
    assert (
        yaml.safe_load((run_dir / "config.yaml").read_text())["v5_refit"]["spec_sha256"]
        == (spec["spec_sha256"])
    )
    fam.validate_run(run_dir, "V5-refit")  # raises SystemExit if not a valid family cell
