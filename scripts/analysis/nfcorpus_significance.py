"""Paired bootstrap significance test: RRF vs. dense (and vs. BM25) nDCG@10
on the NFCorpus (BEIR) fusion sanity-check gate in `scripts/eval_retrieval.py`.

Why this exists: `eval_retrieval.py --dataset nfcorpus` compares RRF's raw
*point-estimate* nDCG@10 against dense-only and BM25-only. At 323 test
queries, a small point-estimate gap can be pure sampling noise rather than a
real fusion defect. This script answers "is the observed gap statistically
distinguishable from zero?" using the same paired-bootstrap methodology
(10k resamples) that `docs/04_EVALUATION.md` §1 already mandates for the
project's significance testing generally, so the NFCorpus gate is judged
consistently with everything else rather than by an arbitrary raw
inequality.

Decision rule this script's output is judged against (pre-registered before
reading results, see ADR-012 in docs/05_DECISIONS.md):
  - RRF significantly BETTER than the best single modality (p < 0.05)
    -> gate passes, fusion adds value.
  - Statistical TIE (95% CI of the mean difference includes 0)
    -> gate passes as "fusion is not harmful; dominance not demonstrated at
       this sample size." This certifies retrieval-plumbing correctness,
       not fusion value -- the fusion-value question belongs to Phase 7's
       downstream-accuracy ablation (H1 in docs/00_RESEARCH_SPEC.md). A tie
       must be recorded as a tie, not described as a win for RRF.
  - RRF significantly WORSE -> gate fails, investigate as a real bug.

Run: `uv run python -m scripts.analysis.nfcorpus_significance`
Requires `scripts.build_nfcorpus` to have been run first (reads the same
`indexes/nfcorpus/{dense,bm25}` this repo's eval_retrieval.py reads).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

RESULTS_PATH = REPO_ROOT / "results" / "phase1_nfcorpus" / "metrics.json"

from src.config import load_config  # noqa: E402
from src.retrieval.dense import DenseIndex  # noqa: E402
from src.retrieval.fusion import reciprocal_rank_fusion  # noqa: E402
from src.retrieval.lexical import BM25Index  # noqa: E402
from src.retrieval.metrics import ndcg_at_k, paired_bootstrap_ci  # noqa: E402

NFCORPUS_RAW_DIR = REPO_ROOT / "data" / "raw" / "nfcorpus"
NFCORPUS_CORPUS_PATH = REPO_ROOT / "data" / "processed" / "nfcorpus_corpus.jsonl"
NFCORPUS_DENSE_DIR = REPO_ROOT / "indexes" / "nfcorpus" / "dense"
NFCORPUS_BM25_DIR = REPO_ROOT / "indexes" / "nfcorpus" / "bm25"
K = 10
N_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42


def _load_chunk_to_doc() -> dict[str, str]:
    chunk_to_doc: dict[str, str] = {}
    with NFCORPUS_CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                chunk_to_doc[row["chunk_id"]] = str(row["doc_id"])
    return chunk_to_doc


def _load_queries() -> dict[str, str]:
    queries: dict[str, str] = {}
    with (NFCORPUS_RAW_DIR / "queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                queries[row["query_id"]] = row["text"]
    return queries


def _load_qrels() -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = {}
    with (NFCORPUS_RAW_DIR / "qrels_test.tsv").open(encoding="utf-8") as f:
        next(f)
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, cid, score = line.split("\t")
            qrels.setdefault(qid, {})[cid] = float(score)
    return qrels


def _dedupe_to_docs(ranking: list[str], chunk_to_doc: dict[str, str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for cid in ranking:
        doc = chunk_to_doc.get(cid, cid)
        if doc not in seen:
            seen.add(doc)
            out.append(doc)
    return out


def main() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    chunk_to_doc = _load_chunk_to_doc()
    queries = _load_queries()
    qrels = _load_qrels()
    eval_qids = sorted(qrels.keys())
    print(f"n queries: {len(eval_qids)}")

    dense = DenseIndex.load(NFCORPUS_DENSE_DIR)
    bm25 = BM25Index.load(NFCORPUS_BM25_DIR)
    top_k_dense = cfg.retrieval.top_k_dense
    top_k_bm25 = cfg.retrieval.top_k_bm25
    rrf_k = cfg.retrieval.fusion.rrf_k

    dense_ndcg: list[float] = []
    bm25_ndcg: list[float] = []
    rrf_ndcg: list[float] = []
    for i, qid in enumerate(eval_qids):
        query = queries[qid]
        dense_hits = [cid for cid, _ in dense.search(query, top_k_dense)]
        bm25_hits = [cid for cid, _ in bm25.search(query, top_k_bm25)]
        fused = [cid for cid, _ in reciprocal_rank_fusion([dense_hits, bm25_hits], k=rrf_k)]

        relevance = qrels[qid]
        dense_ndcg.append(ndcg_at_k(_dedupe_to_docs(dense_hits, chunk_to_doc), relevance, K))
        bm25_ndcg.append(ndcg_at_k(_dedupe_to_docs(bm25_hits, chunk_to_doc), relevance, K))
        rrf_ndcg.append(ndcg_at_k(_dedupe_to_docs(fused, chunk_to_doc), relevance, K))

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(eval_qids)}", flush=True)

    print()
    print(f"mean dense nDCG@{K} = {sum(dense_ndcg) / len(dense_ndcg):.4f}")
    print(f"mean bm25  nDCG@{K} = {sum(bm25_ndcg) / len(bm25_ndcg):.4f}")
    print(f"mean rrf   nDCG@{K} = {sum(rrf_ndcg) / len(rrf_ndcg):.4f}")
    print()

    pairs = [
        ("rrf_vs_dense", rrf_ndcg, dense_ndcg),
        ("rrf_vs_bm25", rrf_ndcg, bm25_ndcg),
        ("dense_vs_bm25", dense_ndcg, bm25_ndcg),
    ]
    comparisons: dict[str, dict] = {}
    for name, a, b in pairs:
        diffs = [x - y for x, y in zip(a, b, strict=True)]
        mean_diff, lo, hi = paired_bootstrap_ci(diffs, n_resamples=N_RESAMPLES, seed=BOOTSTRAP_SEED)
        significant = lo > 0 or hi < 0
        verdict = "SIGNIFICANT" if significant else "not significant (CI includes 0)"
        print(f"{name}: mean diff = {mean_diff:+.4f}, 95% CI = [{lo:+.4f}, {hi:+.4f}] -> {verdict}")
        comparisons[name] = {
            "mean_diff": mean_diff,
            "ci_lo": lo,
            "ci_hi": hi,
            "significant": significant,
        }

    # AUDIT 2026-09-06 §1.1: this script used to print to stdout only, with no
    # persisted artifact -- the entire Phase 1 gate existed only in a
    # gitignored log. Write the real numbers PROJECT_RULES.md §2 requires.
    result = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "n_evaluated": len(eval_qids),
        "k": K,
        "n_resamples": N_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "mean_ndcg": {
            "dense": sum(dense_ndcg) / len(dense_ndcg),
            "bm25": sum(bm25_ndcg) / len(bm25_ndcg),
            "rrf": sum(rrf_ndcg) / len(rrf_ndcg),
        },
        "comparisons": comparisons,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
