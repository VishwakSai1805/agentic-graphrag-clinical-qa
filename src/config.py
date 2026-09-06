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
    # Ollama context window (llama.cpp `num_ctx`). Sent only on the ollama
    # provider path (src/llm/client.py). qwen2.5:7b trains at 32k; 8192 fits a
    # 12-passage + 40-triple tri-modal EvidenceBundle prompt with headroom.
    # It is part of the LLM cache key -- a different window is a different call.
    num_ctx: int = 8192
    # Tokens held back for the completion. A prompt that would leave less than
    # this raises LLMContextBudgetError -- a recorded hard stop, never a silent
    # tail truncation (which would drop the KG-triple block). See CLAUDE.md
    # budget-guard rule.
    context_completion_reserve: int = 512


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


class KGConfig(BaseModel):
    """PrimeKG / Neo4j knobs (Phase 2). See docs/03_DATASETS.md "Knowledge
    graph"/"Entity linking" and ADR-007 (docs/05_DECISIONS.md).

    Only scripts/load_primekg.py and scripts/precompute_kg_cache.py (and ad
    hoc debugging) open a live Neo4j connection using neo4j_uri/neo4j_user/
    neo4j_password_env. All eval-time code reads cache_dir only -- see
    src/kg/traversal.py: get_subgraph_cached.
    """

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    # Name of the env var holding the password -- never the password itself.
    neo4j_password_env: str = "NEO4J_PASSWORD"
    # scispaCy UmlsEntityLinker threshold, tuned on 50 hand-checked items
    # (docs/03_DATASETS.md "Entity linking"; tuning recorded in an ADR when
    # changed).
    linker_threshold: float = 0.85
    # Edge-type allowlist for traversal, docs/03_DATASETS.md "Knowledge graph".
    # Bounded traversal only -- an edge type not in this list is never
    # followed, regardless of what's in the loaded graph.
    # NOTE: "off-label use" (hyphen + space) is PrimeKG's real on-disk
    # `relation` column value in kg.csv, verified against the actual
    # downloaded dump -- docs/03_DATASETS.md's "off_label_use" (underscore)
    # does not occur anywhere in the real data (0 matching rows). Using the
    # underscored form here would silently load zero off-label-use edges.
    # See the ADR this phase adds to docs/05_DECISIONS.md.
    edge_allowlist: list[str] = [
        "indication",
        "contraindication",
        "off-label use",
        "disease_phenotype_positive",
        "disease_protein",
        "drug_effect",
        "disease_disease",
    ]
    max_hops: int = 2
    # Per-node degree cap to avoid hub explosion (e.g. aspirin) -- see
    # docs/03_DATASETS.md.
    degree_cap: int = 200
    # Per-question TOTAL triple cap, added this phase after a real hub-
    # explosion finding: degree_cap alone bounds edges per *node* per hop,
    # not the total size of a multi-seed 2-hop traversal. A real MedQA
    # question with 8 resolved seed entities produced 15,212 triples at
    # degree_cap=200/max_hops=2 (several broad disease/anatomy seed nodes
    # each independently near-saturating the per-node cap, compounding over
    # 2 hops) -- CLAUDE.md's "unbounded traversal on a hub node is a bug"
    # rule, just triggered by seed-set compounding rather than one classic
    # hub node. See src/kg/traversal.py: traverse_live's max_triples
    # docstring and the ADR this phase adds. 40 is a practical ceiling given
    # the project's actual downstream use (EvidenceBundle targets ~12 items
    # total across ALL modalities, docs/04_EVALUATION.md §0.1) -- 40 KG
    # triples/question is already generous headroom relative to that, not a
    # tight squeeze.
    max_triples_per_question: int = 40
    cache_dir: Path = Path("data/processed/kg_cache")
    # Ablation flag (Phase 2 follow-up, docs/05_DECISIONS.md ADR-018): when
    # False (default -- MUST stay the default so this doesn't silently change
    # Phase 3's behavior or invalidate the already-committed canonical
    # data/processed/kg_cache/ files), triple selection ahead of the
    # max_triples_per_question cap is the original deterministic
    # alphabetical-sort-then-truncate behavior baked into
    # `src.kg.traversal.traverse_live`. When True,
    # `scripts/precompute_kg_cache.py` instead gathers a larger candidate pool
    # (`rerank_pool_size`) from the live traversal and uses
    # `src.kg.rerank.select_triples_reranked` (the project's existing MedCPT
    # cross-encoder from `src/retrieval/rerank.py`, reused as-is) to pick the
    # final `max_triples_per_question` triples by relevance to the question
    # text, instead of alphabetical order. This is a measured on/off
    # ablation, not a silent default change -- see the ADR for the real
    # naive-vs-reranked comparison numbers.
    kg_rerank: bool = False
    # Candidate pool size handed to the live traversal before kg_rerank's
    # selection step narrows it down to max_triples_per_question. Only
    # consulted when kg_rerank=True. Bounded (not unlimited) per CLAUDE.md's
    # traversal-boundedness rule -- the pool is a wider candidate set for the
    # reranker to choose from, not a return to unbounded traversal.
    kg_rerank_pool_size: int = 150


class AppConfig(BaseModel):
    seed: int = 42
    llm: LLMConfig
    eval: EvalConfig = EvalConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    kg: KGConfig = KGConfig()
    cache_dir: Path = Path(".cache/llm")


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return AppConfig.model_validate(raw)


def config_hash(cfg: AppConfig) -> str:
    """Stable sha256 hex digest of the resolved config, for run_id / provenance."""
    canonical = json.dumps(cfg.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
