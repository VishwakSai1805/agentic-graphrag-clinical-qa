"""Unit tests for src/kg/categorize.py (ADR-015's rule-based MedQA category
heuristic -- keyword-bucket logic, no model, so this is a pure-function test."""

from __future__ import annotations

from src.kg.categorize import CATEGORIES, categorize_medqa_question


def test_categories_include_other_fallback():
    assert "other" in CATEGORIES


def test_pharmacology_keyword_hit():
    q = "A patient is started on a beta-blocker. What is its mechanism of action?"
    assert categorize_medqa_question(q) == "pharmacology"


def test_treatment_management_keyword_hit():
    q = "Which of the following is the best next step in management of this patient?"
    assert categorize_medqa_question(q) == "treatment_management"


def test_pathology_diagnosis_keyword_hit():
    q = "A biopsy shows features consistent with a diagnosis of carcinoma."
    assert categorize_medqa_question(q) == "pathology_diagnosis"


def test_anatomy_keyword_hit():
    q = "Which nerve innervates the muscle responsible for this patient's finding?"
    assert categorize_medqa_question(q) == "anatomy"


def test_physiology_keyword_hit():
    q = "Which hormone regulates the feedback mechanism described in this scenario?"
    assert categorize_medqa_question(q) == "physiology"


def test_no_keyword_hits_falls_back_to_other():
    q = "A 45-year-old person visits a place and something happens to them."
    assert categorize_medqa_question(q) == "other"


def test_case_insensitive_matching():
    q = "WHICH DRUG causes this ADVERSE EFFECT?"
    assert categorize_medqa_question(q) == "pharmacology"


def test_tie_broken_by_fixed_priority_order():
    """Both pharmacology ("drug") and physiology ("hormone") keywords appear
    exactly once each -- pharmacology wins per _CATEGORY_PRIORITY."""
    q = "This drug affects a hormone."
    assert categorize_medqa_question(q) == "pharmacology"
