"""YAML -> pydantic config loading, with a content hash for run provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel


class ProviderConfig(BaseModel):
    provider: str
    model: str
    api_base: str | None = None
    api_key_env: str | None = None
    timeout_s: float = 60.0


class LLMConfig(BaseModel):
    fallback_chain: list[ProviderConfig]
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int | None = 512
    max_retries: int = 3
    backoff_base_s: float = 1.0


class EvalConfig(BaseModel):
    """Eval-run concurrency/keep-alive knobs. See docs/04_EVALUATION.md §0.4-0.5, ADR-006."""

    concurrency: int = 4
    ollama_keep_alive: int = -1


class DenseRetrievalConfig(BaseModel):
    """MedCPT dual-encoder (ADR-010) + FAISS knobs. `model_type="bge"` selects
    the symmetric BAAI/bge-large-en-v1.5 fallback instead (config-selectable,
    not necessarily wired end-to-end)."""

    model_type: str = "medcpt"  # "medcpt" | "bge"
    query_model: str = "ncbi/MedCPT-Query-Encoder"
    article_model: str = "ncbi/MedCPT-Article-Encoder"
    bge_model: str = "BAAI/bge-large-en-v1.5"
    bge_query_prefix: str = "Represent this sentence for searching relevant passages: "
    device: str | None = None
    batch_size: int = 64
    max_query_length: int = 64
    max_article_length: int = 512
    nlist: int = 100
    nprobe: int = 16
    checkpoint_every_batches: int = 25


class BM25RetrievalConfig(BaseModel):
    k1: float = 1.5
    b: float = 0.75


class FusionConfig(BaseModel):
    """RRF constant. ADR-002: fixed at 60 unless a new ADR says otherwise."""

    rrf_k: int = 60


class RerankConfig(BaseModel):
    model: str = "ncbi/MedCPT-Cross-Encoder"
    max_length: int = 512


class CorpusConfig(BaseModel):
    """Fixed-seed subset size of pqa_artificial used to build the PubMed-
    abstract half of the retrieval corpus. See docs/02_ROADMAP.md Phase 1."""

    pubmed_subset_size: int = 20000
    pubmed_seed: int = 42


class RetrievalConfig(BaseModel):
    """Tri-modal retrieval knobs: chunking, dense, BM25, fusion, rerank.
    See docs/01_ARCHITECTURE.md and docs/02_ROADMAP.md Phase 1."""

    chunk_tokens: int = 256
    chunk_overlap: int = 32
    tokenizer_model: str = "ncbi/MedCPT-Article-Encoder"
    top_k_dense: int = 50
    top_k_bm25: int = 50
    top_k_rerank: int = 12
    dense: DenseRetrievalConfig = DenseRetrievalConfig()
    bm25: BM25RetrievalConfig = BM25RetrievalConfig()
    fusion: FusionConfig = FusionConfig()
    rerank: RerankConfig = RerankConfig()
    corpus: CorpusConfig = CorpusConfig()


class AppConfig(BaseModel):
    seed: int = 42
    llm: LLMConfig
    eval: EvalConfig = EvalConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    cache_dir: Path = Path(".cache/llm")


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return AppConfig.model_validate(raw)


def config_hash(cfg: AppConfig) -> str:
    """Stable sha256 hex digest of the resolved config, for run_id / provenance."""
    canonical = json.dumps(cfg.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
