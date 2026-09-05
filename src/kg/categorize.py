"""ADR-015 -- rule-based MedQA question-category heuristic.

MedQA-USMLE (`GBaker/MedQA-USMLE-4-options`) ships no gold subject/category
label, unlike MedMCQA's 21 real `subject_name` tags (`Question.subject_name`
in `src/kg/dataset_questions.py`). `scripts/kg_probe.py` needs *some* way to
break the KG-hit-rate result down by question type (ADR-014's closing note:
disease/phenotype-heavy questions have a structurally better crosswalk
coverage than drug/gene-heavy ones, and that confound needs a category axis
to be visible instead of hidden inside one aggregate number).

This is a plain keyword-bucket heuristic, not a classifier and not a gold
label -- explicitly imperfect, and reported as such everywhere its output is
used. No ML, no training data, no model file: a category -> list[str]
keyword dict anyone can read and audit directly, which is the whole point
(an opaque classifier would be worse for a paper that needs to say exactly
why a question landed in a bucket).

Method: lowercase substring-count each category's keyword list against the
question stem; the category with the highest total keyword-occurrence count
wins. Ties (including the common "0 hits everywhere") are broken by the
fixed priority order in `_CATEGORY_PRIORITY` below, ending in "other" -- a
question that matches literally nothing falls to "other" rather than being
force-fit into a category it has no keyword evidence for. Category
distribution over a real sample must always be reported alongside `n`
(`scripts/kg_probe.py` does this) so a near-empty bucket is visible.
"""

from __future__ import annotations

_KEYWORDS: dict[str, list[str]] = {
    "pharmacology": [
        "drug",
        "dose",
        "dosing",
        "dosage",
        "mg/kg",
        "mechanism of action",
        "medication",
        "antibiotic",
        "administered",
        "administration of",
        "prescribed",
        "therapy with",
        "inhibitor",
        "agonist",
        "antagonist",
        "receptor blocker",
        "beta-blocker",
        "side effect",
        "adverse effect",
        "toxicity",
        "overdose",
        "pharmacokinetics",
        "pharmacodynamics",
        "contraindicated",
        "drug interaction",
        "half-life",
        "route of administration",
    ],
    "treatment_management": [
        "next step in management",
        "best next step",
        "most appropriate next step",
        "next best step",
        "treatment",
        "management",
        "surgery",
        "surgical",
        "operative",
        "procedure",
        "initiate",
        "should be started",
        "recommended next step",
        "intervention",
        "resuscitation",
        "vaccination",
        "screening",
        "follow-up",
    ],
    "pathology_diagnosis": [
        "diagnosis",
        "diagnostic",
        "biopsy",
        "histology shows",
        "histological",
        "microscopy",
        "pathology",
        "most likely diagnosis",
        "most likely cause",
        "carcinoma",
        "tumor",
        "neoplasm",
        "malignancy",
        "infection",
        "syndrome",
        "disease process",
        "findings are consistent with",
        "confirms the diagnosis",
        "biopsy shows",
        "autopsy",
    ],
    "physiology": [
        "mechanism",
        "physiology",
        "physiological",
        "hormone",
        "regulates",
        "regulation of",
        "feedback",
        "reflex",
        "action potential",
        "membrane potential",
        "cardiac output",
        "glomerular filtration",
        "lab values",
        "laboratory values",
        "serum",
        "concentration of",
        "enzyme activity",
        "homeostasis",
        "signaling pathway",
        "gene expression",
    ],
    "anatomy": [
        "nerve",
        "artery",
        "vein",
        "muscle",
        "bone",
        "ligament",
        "foramen",
        "innervates",
        "innervation",
        "supplies blood",
        "blood supply",
        "anatomical",
        "anatomy",
        "which structure",
        "branch of the",
        "vertebra",
        "cranial nerve",
        "lymph node",
    ],
}

# Priority order used only to break ties (including the all-zero tie, which
# falls through to "other" since "other" has no keyword list to match on).
_CATEGORY_PRIORITY = [
    "pharmacology",
    "treatment_management",
    "pathology_diagnosis",
    "physiology",
    "anatomy",
]

CATEGORIES = [*_CATEGORY_PRIORITY, "other"]


def categorize_medqa_question(question_text: str) -> str:
    """Returns one of `CATEGORIES` via keyword-bucket heuristics (see module
    docstring). Deliberately simple and inspectable -- not a gold label."""
    text = question_text.lower()
    scores = {
        category: sum(text.count(kw) for kw in keywords)
        for category, keywords in _KEYWORDS.items()
    }
    best_category = max(_CATEGORY_PRIORITY, key=lambda c: (scores[c], -_CATEGORY_PRIORITY.index(c)))
    if scores[best_category] == 0:
        return "other"
    return best_category
