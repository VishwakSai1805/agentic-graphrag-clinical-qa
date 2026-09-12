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

    # --- §0.6 Gemini sensitivity check (ADR-047) ---------------------------
    # Cost guard. When True, EVERY provider the fallback chain would call must
    # appear in `free_tier_allowed` as an exact "<provider>:<model>" string, or
    # `chat()` raises LLMCostGuardError BEFORE any network call. This is a hard
    # stop, never a skip-and-fall-through: falling through to the next provider
    # is exactly the failure mode the guard exists to prevent (the default chain
    # ends at a billable cloud provider).
    #
    # It deliberately does NOT consult `litellm.model_cost`. Google AI Studio's
    # free tier is a property of the API KEY's quota, not of the model id --
    # litellm prices `gemini/gemini-3.8-flash` at 7.5e-07/token input and would
    # report a nonzero `completion_cost` for a call served entirely free. A
    # price-based guard would therefore reject a legitimately free call and, more
    # dangerously, could be argued into accepting a paid one. An explicit
    # allowlist is the only check that means what it says.
    free_tier_only: bool = False
    free_tier_allowed: list[str] = []
    # Provenance for a run served by a hosted endpoint (ADR-047). A model
    # endpoint can change behaviour behind a fixed name, so "gemini-3.8-flash"
    # alone does not identify what generated a run -- the API version and the
    # date it served are part of the identity. Stamped at launch by
    # `src.eval.harness --llm-api-version`, so they land in the resolved
    # results/<run_id>/config.yaml rather than only in a launcher log.
    api_version: str | None = None
    model_resolved_utc: str | None = None
    # Cache namespacing. `_cache_key` already mixes the model id into the hash,
    # so a Gemini reply can never be READ as an Ollama reply. This flag adds a
    # directory-level namespace (`.cache/llm/<provider>__<model>/`) so the two
    # generators' caches are also separable on disk -- inspectable, measurable
    # and deletable independently. Default False: flipping it on globally would
    # orphan every existing `.cache/llm/` entry.
    cache_namespace_by_provider_model: bool = False


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


class AgentsConfig(BaseModel):
    """Phase 4 multi-agent consensus network. See docs/01_ARCHITECTURE.md
    "Agent contracts" / "Budget guards" and ADR-022/023 (docs/05_DECISIONS.md).

    `enabled=False` in base.yaml -- only configs/agents.yaml (and Phase 7's
    V4/V5 variants) turn the network on. All eval-time code that runs agents
    reads these knobs; nothing here opens a network connection.
    """

    enabled: bool = False
    # Conflict is CLAIM-level (ADR-025): it fires for a (option, claim_idx)
    # when that claim has >= epsilon SUPPORT cells AND >= epsilon CONTRADICT
    # cells, from different evidence ids. (Mass = integer cell count for
    # Phase 4; a fractional mass over the calibrated tuple is Phase 5's job.)
    epsilon: int = 1
    # Phase 7 V4−conflict ablation cell (ADR-035): False routes the graph
    # audit -> END, skipping the Conflict node entirely. ADR-025 already
    # measured `conflict_fire_rate = 0.00` on the dev-20, so V4−conflict is
    # expected to be numerically ~identical to V4/V5's agent portion -- a null
    # here is the reportable finding, not a bug.
    run_conflict: bool = True
    max_rounds: int = 2
    max_llm_calls_per_question: int = 12
    max_evidence: int = 12
    # [n_passages, n_triples] reserved slots. The KG count is a FLOOR, not a
    # fixed ratio: unfilled KG slots backfill with passages so the bundle
    # always reaches max_evidence when that much evidence exists (ADR-023).
    # Ablatable in Phase 7; the realised split is stored per question.
    evidence_split: list[int] = [8, 4]
    # Rank the KG slot with the MedCPT cross-encoder (reuse of the ADR-018
    # mechanism at agent-selection granularity -- its own value is a Phase 7
    # evidence_split ablation question, not inherited from ADR-018's n=200 tie).
    kg_slot_rerank: bool = True
    # Hypothesis agent: deterministic by default (ADR-001 wants replayability;
    # claim decomposition of given MCQ options gains little from diversity).
    # Sampling is an ablation flag, not the default.
    hypothesis_sample: bool = False
    hypothesis_temperature: float = 0.0
    # 3-tier source precedence for the Conflict agent (ADR-022): the
    # conclusion/background split and "recency" from docs/01 are unimplementable
    # on this corpus (no section tags, no dates) and are dropped, recorded.
    source_precedence: list[str] = ["kg", "textbook", "pubmed_abstract"]
    prompt_versions: dict[str, str] = {
        "hypothesis": "v1",
        "audit": "v1",
        "conflict": "v2",  # v2: claim-level, semantic-check-first (ADR-025)
    }


class CalibrationConfig(BaseModel):
    """Phase 5 evidential calibration & ranking. See docs/01_ARCHITECTURE.md
    "3. MULTI-CHOICE EVIDENTIAL CALIBRATION & RANKING" and the ADR this phase
    adds to docs/05_DECISIONS.md for the feature definitions and fit
    procedure. `enabled=False` in base.yaml/agents.yaml -- only Phase 7's V5
    variant (configs/v5.yaml, per agents.yaml's own header comment) turns
    calibration on.

    z_i = alpha*s_pos_i - beta*s_neg_i + gamma*kg_support_count_i +
    delta*mean_rerank_score_i; p = softmax(z / temperature).
    """

    enabled: bool = False
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.0
    delta: float = 0.0
    temperature: float = 1.0
    # Abstain (report no ranked answer) when the top-1 probability is below
    # this threshold. 0.0 = never abstain (the coverage-risk curve's t=0 end).
    abstention_tau: float = 0.0
    # ECE bin count/strategy for THIS config's own eval runs (docs/04
    # §4 default is 15 equal-width bins for the n=500 grid; a dev-scale fit
    # run uses fewer, equal-mass bins -- see the ADR and src.eval.metrics.ece).
    n_ece_bins: int = 15
    ece_binning: str = "equal_width"
    # The dev slice these weights were fit on -- recorded here (not just in
    # the fitting script's own output) so a resolved config.yaml alone shows
    # provenance for a V5 run's weights, per CLAUDE.md's reproducibility rule.
    fit_dataset: str = "medqa"
    fit_split: str = "train"
    fit_limit: int = 60
    fit_seed: int = 42


class AppConfig(BaseModel):
    seed: int = 42
    llm: LLMConfig
    eval: EvalConfig = EvalConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    kg: KGConfig = KGConfig()
    agents: AgentsConfig = AgentsConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    cache_dir: Path = Path(".cache/llm")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`. Dicts merge key-wise;
    every other type (including lists) is replaced wholesale by `override`."""
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path: str | Path) -> AppConfig:
    """Load a YAML config into an `AppConfig`.

    A config may carry a top-level `_extends: <path>` (relative to its own
    directory, or absolute) to inherit from another YAML -- the parent is
    loaded, then this file is deep-merged over it (dicts key-wise, lists
    replaced). One level of `_extends` chaining is supported recursively.
    The fully resolved config is what `results/<run_id>/config.yaml` records,
    so provenance never depends on the extends indirection.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    parent_ref = raw.pop("_extends", None)
    if parent_ref is not None:
        parent_path = Path(parent_ref)
        if not parent_path.is_absolute():
            parent_path = (path.parent / parent_path).resolve()
        parent_raw = _resolve_raw(parent_path)
        raw = _deep_merge(parent_raw, raw)
    return AppConfig.model_validate(raw)


def _resolve_raw(path: Path) -> dict:
    """Return a config file's raw dict with any `_extends` chain already merged in."""
    raw = yaml.safe_load(path.read_text()) or {}
    parent_ref = raw.pop("_extends", None)
    if parent_ref is None:
        return raw
    parent_path = Path(parent_ref)
    if not parent_path.is_absolute():
        parent_path = (path.parent / parent_path).resolve()
    return _deep_merge(_resolve_raw(parent_path), raw)


def config_hash(cfg: AppConfig) -> str:
    """Stable sha256 hex digest of the resolved config, for run_id / provenance."""
    canonical = json.dumps(cfg.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
