"""Shared evidence schema — retrieval (Phase 1) and KG (Phase 2) both emit this."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Evidence(BaseModel):
    """A single retrievable unit, citable by id in a generated answer.

    id is "chunk::<doc_id>#<span>" for dense/lexical hits, or
    "kg::<head>|<rel>|<tail>" for KG triples (Phase 2). See
    docs/01_ARCHITECTURE.md "Evidence object".
    """

    id: str
    modality: Literal["dense", "lexical", "kg"]
    text: str
    source_meta: dict
    scores: dict[str, float]


class LinkedEntity(BaseModel):
    """One scispaCy + UmlsEntityLinker hit (src/kg/linker.py), with the
    result of crosswalking its UMLS CUI to a PrimeKG node id (src/kg/queries.py
    `Entity.id`, i.e. PrimeKG's `node_index`).

    kg_node_id is None when the CUI could not be mapped to any PrimeKG node --
    this is expected and logged (docs/03_DATASETS.md: "log unmapped CUIs,
    they explain KG-hit-rate gaps"), not an error. match_method records how a
    non-None kg_node_id was found: "mondo_xref" / "hpo_xref" (real ontology
    cross-reference, disease/phenotype only -- see ADR entry this phase adds)
    or "name_match" (normalized node-name string match, the only route
    available for drug/gene node types without a licensed UMLS Metathesaurus
    install providing DrugBank/NCBI-gene xrefs).
    """

    text: str
    cui: str
    score: float
    kg_node_id: str | None = None
    kg_node_type: str | None = None
    match_method: str | None = None
