"""Phase 6: src/eval/nli.py -- claim splitting + hallucination scoring.

The NLI model is never loaded here: every test injects a fake
`scorer(premise, hypothesis) -> float`.
"""

from __future__ import annotations

from src.eval.nli import ClaimRecord, score_hallucination, split_claims

# --------------------------------------------------------------------------
# split_claims
# --------------------------------------------------------------------------


def test_split_claims_basic_sentences():
    assert split_claims("Aspirin inhibits COX-1. It reduces platelet aggregation.") == [
        "Aspirin inhibits COX-1.",
        "It reduces platelet aggregation.",
    ]


def test_split_claims_empty_and_blank():
    assert split_claims("") == []
    assert split_claims("   ") == []
    assert split_claims(None) == []  # type: ignore[arg-type]


def test_split_claims_does_not_break_on_decimal():
    claims = split_claims("The dose is 5.5 mg daily. This is standard.")
    assert claims == ["The dose is 5.5 mg daily.", "This is standard."]


def test_split_claims_keeps_abbreviations_glued():
    claims = split_claims("Use a beta-blocker, e.g. metoprolol, first. Then reassess.")
    assert claims == [
        "Use a beta-blocker, e.g. metoprolol, first.",
        "Then reassess.",
    ]


def test_split_claims_et_al_not_a_boundary():
    claims = split_claims("Smith et al. showed benefit. The trial was small.")
    assert claims == ["Smith et al. showed benefit.", "The trial was small."]


def test_split_claims_drops_bare_list_markers():
    # "1." / "2." are not letters+len>=3 -> dropped, the real clauses survive
    claims = split_claims("1. First point here. 2. Second point here.")
    assert claims == ["First point here.", "Second point here."]


def test_split_claims_single_sentence_no_trailing_period():
    assert split_claims("Metformin is first line") == ["Metformin is first line"]


# --------------------------------------------------------------------------
# score_hallucination
# --------------------------------------------------------------------------


def _fake_scorer(entail_map: dict[str, float]):
    """scorer keyed on the claim text; premise ignored."""

    def _scorer(premise: str, hypothesis: str) -> float:
        return entail_map.get(hypothesis, 0.0)

    return _scorer


def test_all_claims_entailed_zero_hallucination():
    recs = [
        ClaimRecord(
            question_id="q1",
            claim="c1",
            cited_evidence_texts=["ev"],
            predicted_option=True,
        ),
        ClaimRecord(
            question_id="q1",
            claim="c2",
            cited_evidence_texts=["ev"],
            predicted_option=True,
        ),
    ]
    out = score_hallucination(recs, scorer=_fake_scorer({"c1": 0.9, "c2": 0.8}))
    assert out["hallucination_rate_predicted_option"] == 0.0
    assert out["n_claims_predicted_option"] == 2
    assert out["primary"] == "hallucination_rate_predicted_option"


def test_uncited_claim_is_ungrounded_even_if_a_scorer_would_entail_it():
    recs = [
        ClaimRecord(question_id="q1", claim="c1", cited_evidence_texts=[], predicted_option=True)
    ]
    out = score_hallucination(recs, scorer=_fake_scorer({"c1": 0.99}))
    assert out["hallucination_rate_predicted_option"] == 1.0
    assert out["n_uncited_predicted_option"] == 1


def test_below_threshold_is_ungrounded():
    recs = [
        ClaimRecord(
            question_id="q1", claim="c1", cited_evidence_texts=["ev"], predicted_option=True
        )
    ]
    out = score_hallucination(recs, scorer=_fake_scorer({"c1": 0.49}), threshold=0.5)
    assert out["hallucination_rate_predicted_option"] == 1.0


def test_dual_denominators_differ_when_other_options_have_cells():
    recs = [
        # predicted option: 1 grounded claim
        ClaimRecord(
            question_id="q1", option_id="A", claim="good",
            cited_evidence_texts=["ev"], predicted_option=True,
        ),
        # a rejected option: 1 ungrounded claim
        ClaimRecord(
            question_id="q1", option_id="B", claim="bad",
            cited_evidence_texts=["ev"], predicted_option=False,
        ),
    ]
    out = score_hallucination(recs, scorer=_fake_scorer({"good": 0.9, "bad": 0.1}))
    assert out["hallucination_rate_predicted_option"] == 0.0
    assert out["n_claims_predicted_option"] == 1
    assert out["hallucination_rate_all_support_cells"] == 0.5
    assert out["n_claims_all_support_cells"] == 2


def test_max_over_multiple_cited_evidences():
    recs = [
        ClaimRecord(
            question_id="q1", claim="c1",
            cited_evidence_texts=["weak", "strong"], predicted_option=True,
        )
    ]

    def scorer(premise: str, hypothesis: str) -> float:
        return {"weak": 0.2, "strong": 0.95}[premise]

    out = score_hallucination(recs, scorer=scorer)
    assert out["hallucination_rate_predicted_option"] == 0.0
    assert out["per_claim"][0]["max_entail_score"] == 0.95


def test_empty_records_is_zero_rate_with_zero_claims():
    out = score_hallucination([], scorer=_fake_scorer({}))
    assert out["hallucination_rate_predicted_option"] == 0.0
    assert out["n_claims_predicted_option"] == 0
    assert out["n_claims_all_support_cells"] == 0
