"""Phase 1 retrieval quality eval -- three dataset modes with different
pass/fail semantics (see docs/05_DECISIONS.md for the ADR covering why).

Usage:
    uv run python -m scripts.eval_retrieval --dataset nfcorpus  --limit 323
    uv run python -m scripts.eval_retrieval --dataset pubmedqa --limit 200
    uv run python -m scripts.eval_retrieval --dataset medqa    --limit 200

--dataset nfcorpus (PRIMARY, GATING for fusion quality): NFCorpus (BEIR) is
real, human-judged, graded-relevance IR with no single-modality saturation
-- unlike PubMedQA, where a question closely paraphrases its own source
abstract and any working modality saturates near ceiling. Requires
`scripts.build_nfcorpus` to have been run first (separate small index under
indexes/nfcorpus/, not mixed into the main medical corpus/index).

Judged by a PRE-REGISTERED, statistical-significance rule (ADR-012), not raw
point-estimate inequality -- at ~300 queries a small point-estimate gap can
be pure sampling noise (see docs/04_EVALUATION.md §1's paired-bootstrap
methodology, applied here via src.retrieval.metrics.paired_bootstrap_ci):
  - RRF significantly BETTER than the best single modality (95% CI of the
    per-query nDCG@10 difference lies entirely above 0) -> PASS, fusion
    demonstrably adds value.
  - Statistical TIE (CI includes 0) -> PASS, reported as "fusion is not
    harmful; dominance not demonstrated at this sample size." This certifies
    retrieval-plumbing correctness, not fusion value -- the fusion-value
    question is Phase 7's job (H1, docs/00_RESEARCH_SPEC.md). A tie is never
    printed as a win for RRF.
  - RRF significantly WORSE (CI lies entirely below 0) -> FAIL. This is a
    real fusion bug, not a benchmark-saturation artifact, and this case is
    not softened.

--dataset pubmedqa (regression-guard floor, NOT a dominance comparison):
each PubMedQA question closely paraphrases its own source abstract, so
dense/bm25/rrf all saturate near ceiling here (observed ~0.98-0.995
Recall@10) and RRF is *not* expected to beat both other modalities on this
benchmark -- that was the original (retracted) gate. The only thing gated
now is an infra-correctness floor: all three modalities must clear
Recall@10 >= 0.9, a regression guard against something being badly broken,
not a fusion-quality comparison. Also prints (never gates):
  (a) Recall@1 for all three modalities.
  (b) A "hard subset" cut -- among sampled queries, those where dense's own
      top-1 chunk's doc_id != gold pubid -- with bm25/rrf Recall@10 and MRR
      restricted to just that subset (does fusion recover what dense alone
      misses?).

--dataset medqa (SECONDARY, diagnostic only -- unchanged, still never
gates): MedQA has no gold supporting passage, so "relevant" is approximated
as "retrieved chunk text contains the gold answer-option string"
(case-insensitive substring) -- a metric structurally biased toward BM25
(surface string matching).

Unconditionally prints torch.cuda.is_available() (and device name/capability
if true) at the start of main(), so a silent CPU fallback can never hide.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Iterable
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402
from src.retrieval.dense import DenseIndex  # noqa: E402
from src.retrieval.fusion import reciprocal_rank_fusion  # noqa: E402
from src.retrieval.lexical import BM25Index  # noqa: E402
from src.retrieval.metrics import (  # noqa: E402
    mean_ndcg_at_k,
    mean_recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    paired_bootstrap_ci,
)

CORPUS_PATH = REPO_ROOT / "data" / "processed" / "corpus.jsonl"
DENSE_DIR = REPO_ROOT / "indexes" / "dense"
BM25_DIR = REPO_ROOT / "indexes" / "bm25"
PUBMED_ARTIFICIAL_PATH = REPO_ROOT / "data" / "raw" / "pubmedqa" / "pqa_artificial_train.jsonl"
MEDQA_TEST_PATH = REPO_ROOT / "data" / "raw" / "medqa_usmle" / "test.jsonl"
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"

NFCORPUS_RAW_DIR = REPO_ROOT / "data" / "raw" / "nfcorpus"
NFCORPUS_QUERIES_PATH = NFCORPUS_RAW_DIR / "queries.jsonl"
NFCORPUS_QRELS_TEST_PATH = NFCORPUS_RAW_DIR / "qrels_test.tsv"
NFCORPUS_CORPUS_PATH = REPO_ROOT / "data" / "processed" / "nfcorpus_corpus.jsonl"
NFCORPUS_DENSE_DIR = REPO_ROOT / "indexes" / "nfcorpus" / "dense"
NFCORPUS_BM25_DIR = REPO_ROOT / "indexes" / "nfcorpus" / "bm25"

PUBMEDQA_RECALL_FLOOR = 0.9


def print_gpu_status() -> None:
    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available() = {available}")
    if available:
        print(f"torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
        print(f"torch.cuda.get_device_capability(0) = {torch.cuda.get_device_capability(0)}")


def load_chunk_index(corpus_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Returns (chunk_id -> doc_id, chunk_id -> text) built once from corpus.jsonl."""
    chunk_to_doc: dict[str, str] = {}
    chunk_to_text: dict[str, str] = {}
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            chunk_to_doc[row["chunk_id"]] = str(row["doc_id"])
            chunk_to_text[row["chunk_id"]] = row["text"]
    return chunk_to_doc, chunk_to_text


def indexed_pubmed_doc_ids(corpus_path: Path) -> set[str]:
    ids: set[str] = set()
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("source") == "pubmed_abstract":
                ids.add(str(row["doc_id"]))
    return ids


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def chunk_rankings_to_doc_rankings(
    chunk_rankings: list[list[str]], chunk_to_doc: dict[str, str], dedupe: bool = False
) -> list[list[str]]:
    """Maps each chunk_id in each ranking to its doc_id (position-preserving).
    dedupe=True collapses repeated doc_ids to their first occurrence -- use
    this for document-level metrics (e.g. NFCorpus nDCG); leave False to keep
    exact chunk-level ranking semantics (PubMedQA/MedQA, matching the
    original spec of "a retrieved chunk counts as relevant iff...").
    """
    out = []
    for ranking in chunk_rankings:
        doc_ranking = [chunk_to_doc.get(cid, cid) for cid in ranking]
        out.append(dedupe_preserve_order(doc_ranking) if dedupe else doc_ranking)
    return out


def print_metrics_table(results: dict[str, dict[str, float]]) -> None:
    if not results:
        return
    metric_names = list(next(iter(results.values())).keys())
    header = f"{'modality':<10}" + "".join(f"{m:>14}" for m in metric_names)
    print(header)
    for name, metrics in results.items():
        row = f"{name:<10}" + "".join(f"{metrics[m]:>14.4f}" for m in metric_names)
        print(row)


def retrieve_rankings(
    dense: DenseIndex, bm25: BM25Index, query: str, top_k_dense: int, top_k_bm25: int, rrf_k: int
) -> tuple[list[str], list[str], list[str]]:
    dense_hits = [cid for cid, _score in dense.search(query, top_k_dense)]
    bm25_hits = [cid for cid, _score in bm25.search(query, top_k_bm25)]
    fused = [cid for cid, _score in reciprocal_rank_fusion([dense_hits, bm25_hits], k=rrf_k)]
    return dense_hits, bm25_hits, fused


# --------------------------------------------------------------------------
# NFCorpus: PRIMARY, GATING fusion sanity-check (nDCG@10, graded relevance)
# --------------------------------------------------------------------------


def load_nfcorpus_queries_and_qrels() -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    queries: dict[str, str] = {}
    with NFCORPUS_QUERIES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            queries[row["query_id"]] = row["text"]

    qrels: dict[str, dict[str, float]] = {}
    with NFCORPUS_QRELS_TEST_PATH.open("r", encoding="utf-8") as f:
        next(f)  # header: query-id  corpus-id  score
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, cid, score = line.split("\t")
            qrels.setdefault(qid, {})[cid] = float(score)

    return queries, qrels


def eval_nfcorpus(cfg, limit: int, k: int, seed: int) -> bool:
    print("=" * 70)
    print("NFCorpus (BEIR) fusion sanity-check eval (PRIMARY, GATING)")
    print("=" * 70)

    if not NFCORPUS_DENSE_DIR.exists() or not NFCORPUS_BM25_DIR.exists():
        print(
            "[eval_retrieval] ERROR: NFCorpus indexes not found under "
            f"{NFCORPUS_DENSE_DIR.parent} -- run `python -m scripts.build_nfcorpus` first.",
            file=sys.stderr,
        )
        sys.exit(2)

    queries, qrels = load_nfcorpus_queries_and_qrels()
    eval_qids = sorted(qrels.keys())
    print(f"[eval_retrieval] {len(eval_qids)} NFCorpus test queries with qrels available")

    rng = random.Random(seed)
    sample_qids = (
        rng.sample(eval_qids, limit) if limit < len(eval_qids) else eval_qids
    )
    print(f"[eval_retrieval] evaluating {len(sample_qids)} queries (seed={seed})")

    dense = DenseIndex.load(NFCORPUS_DENSE_DIR)
    bm25 = BM25Index.load(NFCORPUS_BM25_DIR)
    chunk_to_doc, _chunk_to_text = load_chunk_index(NFCORPUS_CORPUS_PATH)

    top_k_dense = cfg.retrieval.top_k_dense
    top_k_bm25 = cfg.retrieval.top_k_bm25
    rrf_k = cfg.retrieval.fusion.rrf_k

    dense_chunk_rankings: list[list[str]] = []
    bm25_chunk_rankings: list[list[str]] = []
    rrf_chunk_rankings: list[list[str]] = []
    relevance_maps: list[dict[str, float]] = []

    for i, qid in enumerate(sample_qids):
        query = queries[qid]
        dense_hits, bm25_hits, fused = retrieve_rankings(
            dense, bm25, query, top_k_dense, top_k_bm25, rrf_k
        )
        dense_chunk_rankings.append(dense_hits)
        bm25_chunk_rankings.append(bm25_hits)
        rrf_chunk_rankings.append(fused)
        relevance_maps.append(qrels[qid])

        if (i + 1) % 50 == 0 or i + 1 == len(sample_qids):
            print(f"[eval_retrieval] retrieved for {i + 1}/{len(sample_qids)} queries", flush=True)

    # NFCorpus qrels are document-level; dedupe chunk rankings to unique docs
    # (first-occurrence order) before scoring, since multiple chunks of the
    # same doc appearing at different ranks would otherwise double-count.
    dense_doc_rankings = chunk_rankings_to_doc_rankings(
        dense_chunk_rankings, chunk_to_doc, dedupe=True
    )
    bm25_doc_rankings = chunk_rankings_to_doc_rankings(
        bm25_chunk_rankings, chunk_to_doc, dedupe=True
    )
    rrf_doc_rankings = chunk_rankings_to_doc_rankings(rrf_chunk_rankings, chunk_to_doc, dedupe=True)

    # Per-query nDCG@k (not just the aggregate mean) -- required for the
    # paired bootstrap significance test the gate is actually judged on.
    per_query_dense = [
        ndcg_at_k(r, rel, k) for r, rel in zip(dense_doc_rankings, relevance_maps, strict=True)
    ]
    per_query_bm25 = [
        ndcg_at_k(r, rel, k) for r, rel in zip(bm25_doc_rankings, relevance_maps, strict=True)
    ]
    per_query_rrf = [
        ndcg_at_k(r, rel, k) for r, rel in zip(rrf_doc_rankings, relevance_maps, strict=True)
    ]

    ndcg_col = f"nDCG@{k}"
    results = {
        "dense": {ndcg_col: mean_ndcg_at_k(dense_doc_rankings, relevance_maps, k)},
        "bm25": {ndcg_col: mean_ndcg_at_k(bm25_doc_rankings, relevance_maps, k)},
        "rrf": {ndcg_col: mean_ndcg_at_k(rrf_doc_rankings, relevance_maps, k)},
    }
    print()
    print_metrics_table(results)
    print()

    dense_mean = results["dense"][ndcg_col]
    bm25_mean = results["bm25"][ndcg_col]
    per_query_best_single = per_query_dense if dense_mean >= bm25_mean else per_query_bm25
    best_single_name = "dense" if dense_mean >= bm25_mean else "bm25"
    best_single_mean = max(dense_mean, bm25_mean)

    diffs = [r - b for r, b in zip(per_query_rrf, per_query_best_single, strict=True)]
    mean_diff, ci_lo, ci_hi = paired_bootstrap_ci(diffs, n_resamples=10_000, seed=seed)
    print(
        f"[significance] RRF vs best single modality ({best_single_name}, "
        f"mean {ndcg_col}={best_single_mean:.4f}): mean diff={mean_diff:+.4f}, "
        f"95% bootstrap CI=[{ci_lo:+.4f}, {ci_hi:+.4f}] (n={len(diffs)}, 10k resamples, "
        f"see scripts/analysis/nfcorpus_significance.py for a standalone reproduction)"
    )

    if ci_lo > 0:
        gate_pass = True
        print(
            f"GATE PASSED (SIGNIFICANT WIN): RRF {ndcg_col} is significantly higher than "
            f"{best_single_name} -- fusion demonstrably adds value on this run."
        )
    elif ci_hi < 0:
        gate_pass = False
        print(
            f"GATE FAILED (SIGNIFICANT LOSS): RRF {ndcg_col} is significantly lower than "
            f"{best_single_name} on this real run. This is NFCorpus, not PubMedQA -- no "
            "saturation confound here, so this is a genuine fusion bug to investigate, "
            "not a benchmark artifact to explain away.",
            file=sys.stderr,
        )
    else:
        gate_pass = True
        print(
            f"GATE PASSED (STATISTICAL TIE): RRF {ndcg_col} is not significantly different "
            f"from {best_single_name} (95% CI includes 0) -- fusion is not harmful, but "
            "dominance is NOT demonstrated at this sample size. This certifies "
            "retrieval-plumbing correctness, not fusion value; the fusion-value question is "
            "Phase 7's job (H1, docs/00_RESEARCH_SPEC.md). Do not read this as a win for RRF."
        )
    return gate_pass


# --------------------------------------------------------------------------
# PubMedQA: regression-guard floor + non-gating diagnostics
# --------------------------------------------------------------------------


def eval_pubmedqa(cfg, limit: int, k: int, seed: int) -> bool:
    print("=" * 70)
    print("PubMedQA retrieval eval (regression-guard floor, NOT a dominance gate)")
    print("=" * 70)

    indexed_ids = indexed_pubmed_doc_ids(CORPUS_PATH)
    print(f"[eval_retrieval] {len(indexed_ids)} distinct pubmed doc_ids indexed in the corpus")

    with PUBMED_ARTIFICIAL_PATH.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    candidates = []
    for line in lines:
        row = json.loads(line)
        pid = str(row["pubid"])
        if pid in indexed_ids:
            candidates.append(row)

    if not candidates:
        print(
            "[eval_retrieval] ERROR: zero pqa_artificial rows have a pubid indexed in the "
            "corpus -- run scripts.build_corpus first.",
            file=sys.stderr,
        )
        sys.exit(2)

    rng = random.Random(seed)
    sample = rng.sample(candidates, min(limit, len(candidates)))
    print(
        f"[eval_retrieval] sampled {len(sample)} eval questions "
        f"(seed={seed}) from {len(candidates)} indexable candidates"
    )

    dense = DenseIndex.load(DENSE_DIR)
    bm25 = BM25Index.load(BM25_DIR)
    chunk_to_doc, _chunk_to_text = load_chunk_index(CORPUS_PATH)

    top_k_dense = cfg.retrieval.top_k_dense
    top_k_bm25 = cfg.retrieval.top_k_bm25
    rrf_k = cfg.retrieval.fusion.rrf_k

    dense_chunk_rankings: list[list[str]] = []
    bm25_chunk_rankings: list[list[str]] = []
    rrf_chunk_rankings: list[list[str]] = []
    gold_doc_ids: list[str] = []

    for i, row in enumerate(sample):
        query = row["question"]
        gold_doc_ids.append(str(row["pubid"]))

        dense_hits, bm25_hits, fused = retrieve_rankings(
            dense, bm25, query, top_k_dense, top_k_bm25, rrf_k
        )
        dense_chunk_rankings.append(dense_hits)
        bm25_chunk_rankings.append(bm25_hits)
        rrf_chunk_rankings.append(fused)

        if (i + 1) % 50 == 0 or i + 1 == len(sample):
            print(f"[eval_retrieval] retrieved for {i + 1}/{len(sample)} queries", flush=True)

    # Chunk-level rankings on purpose (not deduped to doc): "a retrieved
    # chunk counts as relevant iff chunk.doc_id == str(pubid)" per the
    # original spec -- but mapped to doc ids so the shared metrics.py
    # functions (which compare "is this id in the relevant set") can be
    # reused as-is instead of a bespoke predicate.
    dense_doc_rankings = chunk_rankings_to_doc_rankings(dense_chunk_rankings, chunk_to_doc)
    bm25_doc_rankings = chunk_rankings_to_doc_rankings(bm25_chunk_rankings, chunk_to_doc)
    rrf_doc_rankings = chunk_rankings_to_doc_rankings(rrf_chunk_rankings, chunk_to_doc)
    relevant_sets: list[set[str]] = [{gold} for gold in gold_doc_ids]

    def modality_metrics(doc_rankings: list[list[str]]) -> dict[str, float]:
        return {
            f"Recall@{k}": mean_recall_at_k(doc_rankings, relevant_sets, k),
            "MRR": mean_reciprocal_rank(doc_rankings, relevant_sets),
            "Recall@1": mean_recall_at_k(doc_rankings, relevant_sets, 1),
        }

    results = {
        "dense": modality_metrics(dense_doc_rankings),
        "bm25": modality_metrics(bm25_doc_rankings),
        "rrf": modality_metrics(rrf_doc_rankings),
    }

    print()
    print_metrics_table(results)
    print()

    recall_col = f"Recall@{k}"
    modality_names = ("dense", "bm25", "rrf")
    floor_pass = all(results[name][recall_col] >= PUBMEDQA_RECALL_FLOOR for name in modality_names)

    if floor_pass:
        print(
            f"FLOOR CHECK PASSED: dense/bm25/rrf all have {recall_col} >= "
            f"{PUBMEDQA_RECALL_FLOOR} (regression guard, not a dominance comparison)."
        )
    else:
        failing = [
            name for name in modality_names if results[name][recall_col] < PUBMEDQA_RECALL_FLOOR
        ]
        print(
            f"FLOOR CHECK FAILED: {failing} fell below {recall_col} >= "
            f"{PUBMEDQA_RECALL_FLOOR} -- this is a regression guard, a failure here means "
            "something is broken, not just 'fusion doesn't dominate'.",
            file=sys.stderr,
        )
    print(
        "NOTE: PubMedQA questions closely paraphrase their own source abstract, so all "
        "three modalities are expected to saturate near ceiling here; RRF is not expected "
        "to (and is not required to) dominate dense/bm25 on this benchmark. See "
        "docs/05_DECISIONS.md."
    )

    # --- diagnostic (a): Recall@1 already included in the table above ---

    # --- diagnostic (b): hard subset where dense's own top-1 chunk is wrong ---
    hard_idx = [
        i
        for i in range(len(sample))
        if (dense_chunk_rankings[i][0] if dense_chunk_rankings[i] else None) is None
        or chunk_to_doc.get(dense_chunk_rankings[i][0]) != gold_doc_ids[i]
    ]
    print()
    print(
        f"[diagnostic] hard subset (dense top-1 chunk's doc != gold): "
        f"{len(hard_idx)}/{len(sample)} sampled queries"
    )
    if hard_idx:
        hard_bm25_rankings = [bm25_doc_rankings[i] for i in hard_idx]
        hard_rrf_rankings = [rrf_doc_rankings[i] for i in hard_idx]
        hard_relevant = [relevant_sets[i] for i in hard_idx]
        hard_results = {
            "bm25": {
                f"Recall@{k}": mean_recall_at_k(hard_bm25_rankings, hard_relevant, k),
                "MRR": mean_reciprocal_rank(hard_bm25_rankings, hard_relevant),
            },
            "rrf": {
                f"Recall@{k}": mean_recall_at_k(hard_rrf_rankings, hard_relevant, k),
                "MRR": mean_reciprocal_rank(hard_rrf_rankings, hard_relevant),
            },
        }
        print_metrics_table(hard_results)
    else:
        print("[diagnostic] hard subset is empty -- dense got top-1 right on every sampled query.")

    return floor_pass


# --------------------------------------------------------------------------
# MedQA: SECONDARY, diagnostic only -- unchanged semantics, still never gates
# --------------------------------------------------------------------------


def eval_medqa(cfg, limit: int, k: int, seed: int) -> None:
    print("=" * 70)
    print("MedQA retrieval eval (SECONDARY, DIAGNOSTIC ONLY -- ADR-011)")
    print("=" * 70)
    print(
        "NOTE: this metric is lexically biased toward BM25 (surface string "
        "matching) -- diagnostic only, not the pass/fail criterion. See "
        "docs/05_DECISIONS.md ADR-011."
    )

    with MEDQA_TEST_PATH.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    rng = random.Random(seed)
    sample_lines = rng.sample(lines, min(limit, len(lines)))
    sample = [json.loads(line) for line in sample_lines]
    print(f"[eval_retrieval] sampled {len(sample)} MedQA test questions (seed={seed})")

    dense = DenseIndex.load(DENSE_DIR)
    bm25 = BM25Index.load(BM25_DIR)
    _chunk_to_doc, chunk_to_text = load_chunk_index(CORPUS_PATH)

    top_k_dense = cfg.retrieval.top_k_dense
    top_k_bm25 = cfg.retrieval.top_k_bm25
    rrf_k = cfg.retrieval.fusion.rrf_k

    dense_rankings: list[list[str]] = []
    bm25_rankings: list[list[str]] = []
    rrf_rankings: list[list[str]] = []
    relevant_ids_list: list[set[str]] = []

    for i, row in enumerate(sample):
        query = row["question"]
        gold_text = row["options"][row["answer_idx"]]

        dense_hits, bm25_hits, fused = retrieve_rankings(
            dense, bm25, query, top_k_dense, top_k_bm25, rrf_k
        )
        dense_rankings.append(dense_hits)
        bm25_rankings.append(bm25_hits)
        rrf_rankings.append(fused)

        # Only chunks that actually appear in one of this query's rankings
        # can matter to its Recall/MRR -- cheap substring check over just
        # those, instead of the whole corpus.
        candidate_cids = set(dense_hits) | set(bm25_hits) | set(fused)
        relevant_ids = {
            cid
            for cid in candidate_cids
            if gold_text and gold_text.lower() in chunk_to_text.get(cid, "").lower()
        }
        relevant_ids_list.append(relevant_ids)

        if (i + 1) % 50 == 0 or i + 1 == len(sample):
            print(f"[eval_retrieval] retrieved for {i + 1}/{len(sample)} queries", flush=True)

    results = {
        "dense": {
            f"Recall@{k}": mean_recall_at_k(dense_rankings, relevant_ids_list, k),
            "MRR": mean_reciprocal_rank(dense_rankings, relevant_ids_list),
        },
        "bm25": {
            f"Recall@{k}": mean_recall_at_k(bm25_rankings, relevant_ids_list, k),
            "MRR": mean_reciprocal_rank(bm25_rankings, relevant_ids_list),
        },
        "rrf": {
            f"Recall@{k}": mean_recall_at_k(rrf_rankings, relevant_ids_list, k),
            "MRR": mean_reciprocal_rank(rrf_rankings, relevant_ids_list),
        },
    }

    print()
    print_metrics_table(results)
    print()
    print("(medqa mode always exits 0 -- diagnostic only, not a gate. See ADR-011.)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["nfcorpus", "pubmedqa", "medqa"], required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--k", type=int, default=10, help="Recall@k / nDCG@k cutoff")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print_gpu_status()

    if args.dataset in ("pubmedqa", "medqa") and not CORPUS_PATH.exists():
        print(
            f"[eval_retrieval] ERROR: {CORPUS_PATH} does not exist -- run "
            "scripts.build_corpus first.",
            file=sys.stderr,
        )
        sys.exit(2)

    cfg = load_config(CONFIG_PATH)

    if args.dataset == "nfcorpus":
        gate_pass = eval_nfcorpus(cfg, args.limit, args.k, args.seed)
        sys.exit(0 if gate_pass else 1)
    elif args.dataset == "pubmedqa":
        floor_pass = eval_pubmedqa(cfg, args.limit, args.k, args.seed)
        sys.exit(0 if floor_pass else 1)
    else:
        eval_medqa(cfg, args.limit, args.k, args.seed)
        sys.exit(0)


if __name__ == "__main__":
    main()
