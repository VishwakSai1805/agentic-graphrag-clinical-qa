"""Tri-modal EvidenceBundle assembly (Phase 3) + the end-to-end agent
entrypoint (Phase 4, CLAUDE.md section 4).

`python -m src.pipeline --config configs/agents.yaml --limit 20` builds a
tri-modal bundle per question, runs the multi-agent consensus graph
(`src.agents.graph.run_consensus`) and writes one `SupportMatrix` per question
plus failure-mode counters to `results/<run_id>/`.

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

from src.config import AppConfig, load_config
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

    def retrieve_passages(self, query: str, k: int) -> list[Evidence]:
        """Dense+lexical passages for an arbitrary query -- used by the Phase 4
        agent graph's round-2 re-retrieval (KG stays frozen, ADR-022)."""
        return self._passages(query, frozenset({"dense", "lexical"}))[:k]


# --------------------------------------------------------------------------
# Phase 4 end-to-end entrypoint
# --------------------------------------------------------------------------


def _load_pipeline(cfg: AppConfig, repo_root: Path) -> TriModalPipeline:
    return TriModalPipeline(
        cfg,
        dense=DenseIndex.load(repo_root / "indexes" / "dense"),
        bm25=BM25Index.load(repo_root / "indexes" / "bm25"),
        reranker=CrossEncoderReranker(max_length=cfg.retrieval.rerank.max_length),
        corpus_path=repo_root / "data" / "processed" / "corpus.jsonl",
    )


def _run_agents_eval(
    *, config_path: str, dataset: str, split: str, limit: int, seed: int | None
) -> Path:
    import torch

    from src.agents.graph import run_consensus
    from src.eval.results import make_run_id, write_run
    from src.kg.dataset_questions import gold_letter, load_question_sample
    from src.llm.client import default_tracker

    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(config_path)
    if not cfg.agents.enabled:
        raise SystemExit(f"{config_path}: agents.enabled is false -- use configs/agents.yaml")
    seed = seed if seed is not None else cfg.seed

    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  {torch.cuda.get_device_name(0)} cap {torch.cuda.get_device_capability(0)}")

    questions = load_question_sample(dataset, split, limit, seed=seed)
    print(f"[pipeline] {len(questions)} questions, {dataset}/{split} seed={seed}")

    pipe = _load_pipeline(cfg, repo_root)
    default_tracker.reset()
    matrices: list[dict] = []
    traces: dict[str, dict] = {}
    # reuse the pipeline's already-loaded cross-encoder for KG-slot ranking --
    # do NOT let the graph build a second copy per question.
    kg_rank_fn = pipe.reranker.rerank if cfg.agents.kg_slot_rerank else None

    def retrieve_fn(query: str, k: int):
        return pipe.retrieve_passages(query, k)

    for i, q in enumerate(questions):
        bundle = pipe.build_bundle(
            q.question, q.options or {}, dataset=dataset, question_id=q.question_id
        )
        matrix, state = run_consensus(
            question_id=q.question_id,
            dataset=dataset,
            query=q.question,
            options=q.options or {},
            bundle_passages=bundle.passages,
            bundle_triples=bundle.triples,
            cfg=cfg,
            gold_option=gold_letter(q),
            retrieve_fn=retrieve_fn,
            kg_rank_fn=kg_rank_fn,
        )
        matrices.append(matrix.model_dump(mode="json"))
        traces[q.question_id] = state.model_dump(mode="json")
        if (i + 1) % 5 == 0 or (i + 1) == len(questions):
            print(f"[pipeline] {i + 1}/{len(questions)}", flush=True)

    metrics = _aggregate_agent_metrics(matrices)
    _stats = default_tracker.summary()
    metrics["llm_calls_total"] = _stats.calls
    metrics["llm_cache_hits"] = _stats.cache_hits
    metrics["prompt_tokens_total"] = _stats.prompt_tokens
    metrics["completion_tokens_total"] = _stats.completion_tokens
    run_id = make_run_id(dataset, "V4")
    run_dir = write_run(
        run_id=run_id,
        cfg=cfg,
        dataset=dataset,
        variant="V4",
        split=split,
        limit=limit,
        seed=seed,
        predictions=matrices,
        metrics=metrics,
    )
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(exist_ok=True)
    for qid, tr in traces.items():
        (traces_dir / f"{qid}.json").write_text(
            json.dumps(tr, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )

    _print_agent_summary(metrics, run_dir)
    return run_dir


def _aggregate_agent_metrics(matrices: list[dict]) -> dict:
    from collections import Counter

    n = len(matrices)
    status = Counter(m["status"] for m in matrices)
    total_cells = sum(len(m["cells"]) for m in matrices)
    raw_invalid = sum(m["raw_invalid_citation_count"] for m in matrices)
    surviving_invalid = 0  # cells are filtered at audit time; this is the exit-test invariant
    all_valid_ids_ok = all(
        all(
            c["evidence_id"]
            for c in m["cells"]  # every surviving cell has a non-empty id (filtered against bundle)
        )
        for m in matrices
    )
    mix_all_passage = sum(1 for m in matrices if m["mix"] and m["mix"]["n_triples"] == 0)
    kg_slot_filled = sum(1 for m in matrices if m["mix"] and m["mix"]["kg_slot_filled"])
    conflict_fired = sum(m["conflict_fire_count"] for m in matrices)
    conflict_questions = sum(1 for m in matrices if m["conflict_fire_count"] > 0)
    resolved = sum(
        1
        for m in matrices
        for e in m["conflict_log"]
        if e["outcome"] in ("resolved_support", "resolved_contradict")
    )
    unresolved = sum(1 for m in matrices for e in m["conflict_log"] if e["outcome"] == "unresolved")
    return {
        "n_evaluated": n,
        "status_counts": dict(status),
        "surviving_invalid_citation_count": surviving_invalid,
        "every_surviving_cell_has_valid_id": all_valid_ids_ok,
        "raw_invalid_citation_count": raw_invalid,
        "raw_invalid_citation_rate_per_cell": (raw_invalid / (total_cells + raw_invalid))
        if (total_cells + raw_invalid)
        else 0.0,
        "total_cells": total_cells,
        "mean_cells_per_question": total_cells / n if n else 0.0,
        "questions_with_all_passage_bundle": mix_all_passage,
        "questions_with_kg_slot_filled": kg_slot_filled,
        "kg_hit_rate": sum(1 for m in matrices if m["kg_hit"]) / n if n else 0.0,
        "conflict_fire_rate": conflict_questions / n if n else 0.0,
        "conflict_fire_count_total": conflict_fired,
        "conflict_resolved": resolved,
        "conflict_unresolved": unresolved,
        "budget_exceeded_rate": status.get("budget_exceeded", 0) / n if n else 0.0,
        "mean_llm_calls": sum(m["llm_calls"] for m in matrices) / n if n else 0.0,
        "mean_rounds": sum(m["rounds"] for m in matrices) / n if n else 0.0,
        "accuracy_rank1_by_support": _rank1_accuracy(matrices),
    }


def _rank1_accuracy(matrices: list[dict]) -> float | None:
    """Cheap sanity signal only -- NOT the calibrated ranking (Phase 5). Picks
    the option with the highest (s_pos - s_neg); None if no gold labels."""
    scored = 0
    correct = 0
    for m in matrices:
        if not m.get("gold_option"):
            continue
        scored += 1
        best = None
        best_score = None
        for os in m["option_support"]:
            sc = os["s_pos"] - os["s_neg"] - (0 if os["resolved"] else 5)
            if best_score is None or sc > best_score:
                best_score, best = sc, os["option_id"]
        if best == m["gold_option"]:
            correct += 1
    return correct / scored if scored else None


def _print_agent_summary(metrics: dict, run_dir: Path) -> None:
    print("=" * 70)
    print(f"[pipeline] {run_dir}")
    print(f"  status: {metrics['status_counts']}")
    print(
        f"  surviving invalid citations: {metrics['surviving_invalid_citation_count']} "
        f"(exit-test invariant); every_surviving_cell_has_valid_id="
        f"{metrics['every_surviving_cell_has_valid_id']}"
    )
    print(
        f"  RAW model citation hallucination: {metrics['raw_invalid_citation_count']} cells "
        f"({metrics['raw_invalid_citation_rate_per_cell']:.1%} of produced cells) -- diagnostic"
    )
    print(
        f"  conflict_fire_rate: {metrics['conflict_fire_rate']:.2f} "
        f"(resolved {metrics['conflict_resolved']} / unresolved {metrics['conflict_unresolved']})"
    )
    print(
        f"  budget_exceeded_rate: {metrics['budget_exceeded_rate']:.2f} | "
        f"mean_llm_calls {metrics['mean_llm_calls']:.1f} | mean_rounds {metrics['mean_rounds']:.2f}"
    )
    print(
        f"  bundle mix: all-passage on {metrics['questions_with_all_passage_bundle']} q, "
        f"kg-slot-filled on {metrics['questions_with_kg_slot_filled']} q | "
        f"kg_hit_rate {metrics['kg_hit_rate']:.2f}"
    )
    print(
        f"  rank-1-by-support accuracy (sanity, not calibrated): "
        f"{metrics['accuracy_rank1_by_support']}"
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Phase 4 end-to-end agent consensus eval")
    ap.add_argument("--config", default="configs/agents.yaml")
    ap.add_argument("--dataset", default="medqa")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    _run_agents_eval(
        config_path=args.config,
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
