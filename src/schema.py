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
