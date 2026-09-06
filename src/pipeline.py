"""Phase 3: tri-modal EvidenceBundle assembly.

`TriModalPipeline.build_bundle` runs the retrieval half of
`docs/01_ARCHITECTURE.md` stage 1 -- dense (MedCPT+FAISS) + lexical (BM25)
fused with RRF (k=60, ADR-002), cross-encoder reranked to `top_k_rerank`, plus
KG triples read from the precomputed cache (`get_subgraph_cached`, ADR-007 --
no live Neo4j at eval time). The agent network (Phase 4) and calibration
(Phase 5) consume the `EvidenceBundle` this produces; the Phase 3 baselines
(`src/eval/baselines.py`) consume it directly with no agents.

`modalities` selects which retrieval arms feed the bundle, so one pipeline
object serves every baseline:
  B1 dense-only  -> {"dense"}
  B2 hybrid      -> {"dense", "lexical", "kg"}
  B3 verifier    -> {"dense", "lexical", "kg"}
(B0 is zero-shot and never calls the pipeline.)

The corpus text map (`chunk_id -> text/doc_id`) is loaded once, lazily, on the
first `build_bundle` call and kept on the instance. `data/processed/corpus.jsonl`
is ~162 MB / ~113k chunks; an in-memory dict is fine for a 20-question dev
slice but the Phase 6 harness will need a real chunk store (sqlite / mmap) --
see docs/02_ROADMAP.md Phase 6. Not solved here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel

from src.config import AppConfig
from src.kg.traversal import get_subgraph_cached
from src.retrieval.dense import DenseIndex
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.lexical import BM25Index
from src.retrieval.rerank import CrossEncoderReranker
from src.schema import Evidence

logger = logging.getLogger(__name__)

_ALL_MODALITIES = frozenset({"dense", "lexical", "kg"})


class EvidenceBundle(BaseModel):
    """The tri-modal evidence set for one question. `passages` are dense/
    lexical chunk hits (RRF-fused, cross-encoder reranked to `top_k_rerank`);
    `triples` are verbalised KG `Evidence(modality="kg")` objects from the
    precomputed cache. Either list may be empty (KG cache miss, or a
    dense-only bundle) -- an empty list is recorded, never an error.
    """

    query: str
    options: dict[str, str]
    passages: list[Evidence]
    triples: list[Evidence]
    kg_cache_hit: bool

    def evidence(self) -> list[Evidence]:
        """All citable evidence, passages first then triples."""
        return [*self.passages, *self.triples]

    def valid_citation_ids(self) -> set[str]:
        """The set a generated citation id must be in to count as grounded
        (docs/01_ARCHITECTURE.md "Evidence object"; CLAUDE.md grounding rule).
        """
        return {e.id for e in self.evidence()}


class TriModalPipeline:
    def __init__(
        self,
        cfg: AppConfig,
        *,
        dense: DenseIndex,
        bm25: BM25Index,
        reranker: CrossEncoderReranker,
        corpus_path: Path,
        kg_cache_dir: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.dense = dense
        self.bm25 = bm25
        self.reranker = reranker
        self.corpus_path = Path(corpus_path)
        self.kg_cache_dir = Path(kg_cache_dir) if kg_cache_dir else Path(cfg.kg.cache_dir)
        self._chunk_meta: dict[str, dict] | None = None

    # -- corpus text map (lazy, once per process) --------------------------

    def _load_chunk_meta(self) -> dict[str, dict]:
        if self._chunk_meta is None:
            if not self.corpus_path.exists():
                raise FileNotFoundError(
                    f"{self.corpus_path} not found -- run scripts.build_corpus first."
                )
            meta: dict[str, dict] = {}
            with self.corpus_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    meta[row["chunk_id"]] = {
                        "doc_id": str(row["doc_id"]),
                        "text": row["text"],
                        "source": row.get("source", ""),
                        "title": (row.get("meta") or {}).get("title", ""),
                    }
            self._chunk_meta = meta
            logger.info("loaded %d chunk metas from %s", len(meta), self.corpus_path)
        return self._chunk_meta

    # -- retrieval --------------------------------------------------------

    def _passages(self, query: str, modalities: frozenset[str]) -> list[Evidence]:
        rcfg = self.cfg.retrieval
        want_dense = "dense" in modalities
        want_lexical = "lexical" in modalities
        if not (want_dense or want_lexical):
            return []

        dense_hits = self.dense.search(query, rcfg.top_k_dense) if want_dense else []
        bm25_hits = self.bm25.search(query, rcfg.top_k_bm25) if want_lexical else []
        dense_ids = [cid for cid, _ in dense_hits]
        bm25_ids = [cid for cid, _ in bm25_hits]
        dense_score = {cid: s for cid, s in dense_hits}
        bm25_score = {cid: s for cid, s in bm25_hits}

        rankings = [r for r in (dense_ids, bm25_ids) if r]
        if len(rankings) == 1:
            fused_ids = rankings[0]
            rrf_score: dict[str, float] = {}
        else:
            fused = reciprocal_rank_fusion(rankings, k=rcfg.fusion.rrf_k)
            fused_ids = [cid for cid, _ in fused]
            rrf_score = {cid: s for cid, s in fused}

        meta = self._load_chunk_meta()
        candidates: list[Evidence] = []
        for cid in fused_ids:
            m = meta.get(cid)
            if m is None:
                logger.warning("chunk_id %s in a ranking but not in corpus map", cid)
                continue
            scores: dict[str, float] = {}
            if cid in dense_score:
                scores["dense"] = dense_score[cid]
            if cid in bm25_score:
                scores["bm25"] = bm25_score[cid]
            if cid in rrf_score:
                scores["rrf"] = rrf_score[cid]
            modality = "dense" if cid in dense_score else "lexical"
            candidates.append(
                Evidence(
                    id=cid,
                    modality=modality,
                    text=m["text"],
                    source_meta={
                        "doc_id": m["doc_id"],
                        "source": m["source"],
                        "title": m["title"],
                    },
                    scores=scores,
                )
            )

        if not candidates:
            return []
        return self.reranker.rerank(query, candidates, rcfg.top_k_rerank)

    def _triples(
        self, dataset: str, question_id: str, modalities: frozenset[str]
    ) -> tuple[list[Evidence], bool]:
        if "kg" not in modalities:
            return [], False
        entry = get_subgraph_cached(dataset, question_id, self.kg_cache_dir)
        if entry is None:
            logger.info("KG cache miss: %s/%s under %s", dataset, question_id, self.kg_cache_dir)
            return [], False
        return list(entry.evidence), True

    # -- public ----------------------------------------------------------

    def build_bundle(
        self,
        query: str,
        options: dict[str, str],
        *,
        dataset: str,
        question_id: str,
        modalities: frozenset[str] = _ALL_MODALITIES,
    ) -> EvidenceBundle:
        modalities = frozenset(modalities)
        passages = self._passages(query, modalities)
        triples, kg_hit = self._triples(dataset, question_id, modalities)
        return EvidenceBundle(
            query=query,
            options=options,
            passages=passages,
            triples=triples,
            kg_cache_hit=kg_hit,
        )
