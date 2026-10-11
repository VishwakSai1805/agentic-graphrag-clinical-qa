"""Verbalises a PrimeKG `Triple` into a citable `Evidence(modality="kg")`
object -- see docs/01_ARCHITECTURE.md "Evidence object" and PROJECT_RULES.md
("Verbalise triples into natural clinical phrasing for the LLM, but keep the
structured triple in Evidence.source_meta so citations stay
machine-checkable").

One simple template per PrimeKG relation type (the 7-item allowlist in
docs/03_DATASETS.md / configs/base.yaml: kg.edge_allowlist). Deliberately
plain, faithful renderings, not fluent prose -- the goal is an unambiguous
statement of what the triple says, not a persuasive one.
"""

from __future__ import annotations

from src.kg.traversal import Triple
from src.schema import Evidence

_TEMPLATES: dict[str, str] = {
    "indication": "{head} is indicated for {tail}.",
    "contraindication": "{head} is contraindicated in {tail}.",
    # "off-label use" (hyphen + space) is PrimeKG's real kg.csv relation
    # string, not "off_label_use" -- see src/config.py: KGConfig and the ADR
    # this phase adds. Using the underscored key here would silently fall
    # through to _FALLBACK_TEMPLATE for every real off-label-use triple.
    "off-label use": "{head} is used off-label for {tail}.",
    "disease_phenotype_positive": "{tail} is a phenotype associated with {head}.",
    "disease_protein": "{tail} is a protein/gene associated with {head}.",
    "drug_effect": "{head} has the effect {tail}.",
    "disease_disease": "{head} is related to the disease {tail}.",
}

_FALLBACK_TEMPLATE = "{head} has relation '{relation}' to {tail}."


def verbalize_triple(triple: Triple) -> Evidence:
    """Renders `triple` as natural clinical phrasing and packages it as an
    `Evidence(modality="kg")`. The structured triple (head/relation/tail/
    source_db) is preserved verbatim in `source_meta` so a downstream
    citation check can verify the rendered text against the graph fact it
    claims to come from, per PROJECT_RULES.md.
    """
    template = _TEMPLATES.get(triple.relation, _FALLBACK_TEMPLATE)
    text = template.format(head=triple.head, relation=triple.relation, tail=triple.tail)
    return Evidence(
        id=f"kg::{triple.head}|{triple.relation}|{triple.tail}",
        modality="kg",
        text=text,
        source_meta={
            "source_db": triple.source_db,
            "relation": triple.relation,
            "head": triple.head,
            "tail": triple.tail,
        },
        scores={},
    )
