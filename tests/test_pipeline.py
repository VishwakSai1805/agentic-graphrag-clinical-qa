"""Unit tests for src/pipeline.py -- TriModalPipeline.build_bundle.

Retrieval models are faked (no FAISS / no transformers). The KG cache path is
exercised against a tmp cache dir written in the real KGCacheEntry format.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import load_config
from src.pipeline import EvidenceBundle, TriModalPipeline
from src.schema import Evidence

CONFIG = load_config(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")


class FakeDense:
    def __init__(self, hits: list[tuple[str, float]]):
        self._hits = hits

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self._hits[:k]


class FakeBM25:
    def __init__(self, hits: list[tuple[str, float]]):
        self._hits = hits

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self._hits[:k]


class FakeReranker:
    """Identity rerank: keeps input order, truncates to top_n, tags scores."""

    def rerank(self, query: str, candidates: list[Evidence], top_n: int) -> list[Evidence]:
        out = []
        for i, c in enumerate(candidates[:top_n]):
            s = dict(c.scores)
            s["rerank"] = float(top_n - i)
            out.append(c.model_copy(update={"scores": s}))
        return out


def _corpus(tmp_path: Path) -> Path:
    rows = [
        {
            "chunk_id": "chunk::D1#0",
            "doc_id": "D1",
            "source": "textbook",
            "text": "alpha text",
            "meta": {"title": "T1"},
        },
        {
            "chunk_id": "chunk::D2#0",
            "doc_id": "D2",
            "source": "pubmed_abstract",
            "text": "beta text",
            "meta": {"title": "T2"},
        },
        {
            "chunk_id": "chunk::D3#0",
            "doc_id": "D3",
            "source": "textbook",
            "text": "gamma text",
            "meta": {"title": "T3"},
        },
    ]
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def _kg_cache(tmp_path: Path, dataset: str, qid: str, evidence: list[dict]) -> Path:
    d = tmp_path / "kg_cache" / dataset
    d.mkdir(parents=True)
    (d / f"{qid}.json").write_text(
        json.dumps(
            {
                "question_id": qid,
                "linked_entities": [],
                "triples": [],
                "evidence": evidence,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path / "kg_cache"


def _mk_pipeline(tmp_path, dense_hits, bm25_hits, kg_cache_dir=None):
    return TriModalPipeline(
        CONFIG,
        dense=FakeDense(dense_hits),
        bm25=FakeBM25(bm25_hits),
        reranker=FakeReranker(),
        corpus_path=_corpus(tmp_path),
        kg_cache_dir=kg_cache_dir or (tmp_path / "nonexistent_kg_cache"),
    )


def test_dense_only_bundle_no_kg(tmp_path):
    p = _mk_pipeline(tmp_path, [("chunk::D1#0", 0.9), ("chunk::D2#0", 0.5)], [])
    b = p.build_bundle(
        "q", {"A": "x"}, dataset="medqa", question_id="train_00000", modalities=frozenset({"dense"})
    )
    assert isinstance(b, EvidenceBundle)
    assert [e.id for e in b.passages] == ["chunk::D1#0", "chunk::D2#0"]
    assert all(e.modality == "dense" for e in b.passages)
    assert b.triples == []
    assert b.kg_cache_hit is False


def test_hybrid_fuses_dense_and_bm25(tmp_path):
    p = _mk_pipeline(
        tmp_path,
        [("chunk::D1#0", 0.9), ("chunk::D3#0", 0.4)],
        [("chunk::D2#0", 2.0), ("chunk::D1#0", 1.0)],
    )
    b = p.build_bundle(
        "q",
        {"A": "x"},
        dataset="medqa",
        question_id="train_00000",
        modalities=frozenset({"dense", "lexical"}),
    )
    ids = [e.id for e in b.passages]
    assert set(ids) == {"chunk::D1#0", "chunk::D2#0", "chunk::D3#0"}
    # D1 is rank1 in dense and rank2 in bm25 -> highest RRF, must come first
    assert ids[0] == "chunk::D1#0"
    d1 = next(e for e in b.passages if e.id == "chunk::D1#0")
    assert "dense" in d1.scores and "bm25" in d1.scores and "rrf" in d1.scores


def test_kg_cache_hit(tmp_path):
    ev = [
        {
            "id": "kg::Aspirin|indication|pain",
            "modality": "kg",
            "text": "Aspirin is indicated for pain.",
            "source_meta": {"relation": "indication"},
            "scores": {},
        },
    ]
    cache_dir = _kg_cache(tmp_path, "medqa", "train_00001", ev)
    p = _mk_pipeline(tmp_path, [("chunk::D1#0", 0.9)], [], kg_cache_dir=cache_dir)
    b = p.build_bundle("q", {"A": "x"}, dataset="medqa", question_id="train_00001")
    assert b.kg_cache_hit is True
    assert [e.id for e in b.triples] == ["kg::Aspirin|indication|pain"]
    assert "kg::Aspirin|indication|pain" in b.valid_citation_ids()


def test_kg_cache_miss_is_not_an_error(tmp_path):
    p = _mk_pipeline(tmp_path, [("chunk::D1#0", 0.9)], [])
    b = p.build_bundle("q", {"A": "x"}, dataset="medqa", question_id="train_99999")
    assert b.kg_cache_hit is False
    assert b.triples == []


def test_modalities_none_gives_empty_passages(tmp_path):
    p = _mk_pipeline(tmp_path, [("chunk::D1#0", 0.9)], [("chunk::D2#0", 1.0)])
    b = p.build_bundle(
        "q", {"A": "x"}, dataset="medqa", question_id="train_00000", modalities=frozenset({"kg"})
    )
    assert b.passages == []


def test_rerank_truncates_to_top_k_rerank(tmp_path):
    many = [(f"chunk::D{i}#0", 1.0 / (i + 1)) for i in range(3)]
    p = _mk_pipeline(tmp_path, many, [])
    p.cfg.retrieval.top_k_rerank = 2
    b = p.build_bundle(
        "q", {"A": "x"}, dataset="medqa", question_id="train_00000", modalities=frozenset({"dense"})
    )
    assert len(b.passages) == 2


def test_unknown_chunk_id_skipped(tmp_path):
    p = _mk_pipeline(tmp_path, [("chunk::MISSING#0", 0.9), ("chunk::D1#0", 0.5)], [])
    b = p.build_bundle(
        "q", {"A": "x"}, dataset="medqa", question_id="train_00000", modalities=frozenset({"dense"})
    )
    assert [e.id for e in b.passages] == ["chunk::D1#0"]


def test_valid_citation_ids_covers_passages_and_triples(tmp_path):
    ev = [{"id": "kg::a|r|b", "modality": "kg", "text": "a r b", "source_meta": {}, "scores": {}}]
    cache_dir = _kg_cache(tmp_path, "medqa", "train_00002", ev)
    p = _mk_pipeline(tmp_path, [("chunk::D1#0", 0.9)], [], kg_cache_dir=cache_dir)
    b = p.build_bundle("q", {"A": "x"}, dataset="medqa", question_id="train_00002")
    assert b.valid_citation_ids() == {"chunk::D1#0", "kg::a|r|b"}


def test_missing_corpus_raises(tmp_path):
    p = TriModalPipeline(
        CONFIG,
        dense=FakeDense([("chunk::D1#0", 0.9)]),
        bm25=FakeBM25([]),
        reranker=FakeReranker(),
        corpus_path=tmp_path / "nope.jsonl",
        kg_cache_dir=tmp_path / "kg",
    )
    with pytest.raises(FileNotFoundError):
        p.build_bundle(
            "q",
            {"A": "x"},
            dataset="medqa",
            question_id="train_00000",
            modalities=frozenset({"dense"}),
        )
