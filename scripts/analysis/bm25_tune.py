"""Pre-Phase-7 fix: BM25 `k1`/`b` tuning on the NFCorpus dev slice.

Parked in Phase 1 (`configs/base.yaml` ships the `rank_bm25` defaults k1=1.5,
b=0.75). The roadmap requires: retune on a fixed dev slice, report Recall@10 /
MRR before/after, record the chosen values in an ADR and `configs/base.yaml`.

NFCorpus (BEIR) is the project's retrieval-quality dev slice (ADR-012) — real
human-judged graded relevance, 323 test queries with qrels. This is CPU-only
(no dense encoder, no GPU): it rebuilds `BM25Okapi` in memory for each
(k1, b) grid point over `data/processed/nfcorpus_corpus.jsonl` and never
touches `indexes/`.

    uv run python -m scripts.analysis.bm25_tune

Judged the project way (PROJECT_RULES.md / ADR-012): the best grid cell is only
adopted if its per-query nDCG@10 gain over the current default survives a
paired bootstrap (10k, seed 42). Because "best" is selected on the same
NFCorpus sample it is then tested on, that CI is optimistic — so the bar is
deliberately: significant **and** practically meaningful (Δ nDCG@10 ≳ 0.01).
Otherwise the default stays and this is recorded as a checked non-change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from rank_bm25 import BM25Okapi  # noqa: E402

from scripts.eval_retrieval import (  # noqa: E402
    NFCORPUS_CORPUS_PATH,
    chunk_rankings_to_doc_rankings,
    load_chunk_index,
    load_nfcorpus_queries_and_qrels,
)
from src.retrieval.lexical import _tokenize  # noqa: E402
from src.retrieval.metrics import (  # noqa: E402
    mean_ndcg_at_k,
    mean_recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    paired_bootstrap_ci,
    paired_bootstrap_p,
)

OUT_DIR = REPO_ROOT / "results" / "phase1_bm25_tune"
DEFAULT_K1, DEFAULT_B = 1.5, 0.75
TOP_K = 50
NDCG_K = 10

K1_GRID = [0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]
B_GRID = [0.3, 0.5, 0.65, 0.75, 0.9, 1.0]
ADOPT_MIN_DELTA = 0.01  # practical-significance floor on Δ nDCG@10


def _build_and_rank(tokenized_corpus, chunk_ids, queries, eval_qids, k1, b):
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
    rankings: list[list[str]] = []
    for qid in eval_qids:
        scores = bm25.get_scores(_tokenize(queries[qid]))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]
        rankings.append([chunk_ids[i] for i in top])
    return rankings


def main() -> None:
    if not NFCORPUS_CORPUS_PATH.exists():
        raise SystemExit(
            f"{NFCORPUS_CORPUS_PATH} missing — run `python -m scripts.build_nfcorpus` first"
        )
    queries, qrels = load_nfcorpus_queries_and_qrels()
    eval_qids = sorted(qrels.keys())
    chunk_to_doc, chunk_to_text = load_chunk_index(NFCORPUS_CORPUS_PATH)
    chunk_ids = list(chunk_to_text)
    tokenized_corpus = [_tokenize(chunk_to_text[cid]) for cid in chunk_ids]
    relevance_maps = [qrels[qid] for qid in eval_qids]
    print(
        f"[bm25_tune] {len(chunk_ids)} NFCorpus chunks, {len(eval_qids)} queries with qrels; "
        f"grid = {len(K1_GRID)}x{len(B_GRID)}"
    )

    def score(k1, b):
        chunk_rankings = _build_and_rank(
            tokenized_corpus, chunk_ids, queries, eval_qids, k1, b
        )
        doc_rankings = chunk_rankings_to_doc_rankings(chunk_rankings, chunk_to_doc, dedupe=True)
        per_q_ndcg = [
            ndcg_at_k(r, rel, NDCG_K)
            for r, rel in zip(doc_rankings, relevance_maps, strict=True)
        ]
        relset = [set(m) for m in relevance_maps]
        return {
            "k1": k1,
            "b": b,
            f"ndcg@{NDCG_K}": mean_ndcg_at_k(doc_rankings, relevance_maps, NDCG_K),
            f"recall@{NDCG_K}": mean_recall_at_k(doc_rankings, relset, NDCG_K),
            "mrr": mean_reciprocal_rank(doc_rankings, relset),
            "_per_q_ndcg": per_q_ndcg,
        }

    default = score(DEFAULT_K1, DEFAULT_B)
    grid = []
    for k1 in K1_GRID:
        for b in B_GRID:
            row = default if (k1, b) == (DEFAULT_K1, DEFAULT_B) else score(k1, b)
            grid.append(row)
            print(
                f"  k1={k1:<4} b={b:<4}  nDCG@{NDCG_K}={row[f'ndcg@{NDCG_K}']:.4f}  "
                f"R@{NDCG_K}={row[f'recall@{NDCG_K}']:.4f}  MRR={row['mrr']:.4f}"
            )

    best = max(grid, key=lambda r: r[f"ndcg@{NDCG_K}"])
    diffs = [
        bq - dq for bq, dq in zip(best["_per_q_ndcg"], default["_per_q_ndcg"], strict=True)
    ]
    mean_diff, ci_lo, ci_hi = paired_bootstrap_ci(diffs, n_resamples=10_000, seed=42)
    p = paired_bootstrap_p(diffs, n_resamples=10_000, seed=42)
    delta_point = best[f"ndcg@{NDCG_K}"] - default[f"ndcg@{NDCG_K}"]
    significant = ci_lo > 0
    adopt = significant and delta_point >= ADOPT_MIN_DELTA

    print("\n" + "=" * 70)
    print(f"default  k1={DEFAULT_K1} b={DEFAULT_B}: nDCG@{NDCG_K}={default[f'ndcg@{NDCG_K}']:.4f} "
          f"R@{NDCG_K}={default[f'recall@{NDCG_K}']:.4f} MRR={default['mrr']:.4f}")
    print(f"best     k1={best['k1']} b={best['b']}: nDCG@{NDCG_K}={best[f'ndcg@{NDCG_K}']:.4f} "
          f"R@{NDCG_K}={best[f'recall@{NDCG_K}']:.4f} MRR={best['mrr']:.4f}")
    print(f"Δ nDCG@{NDCG_K} (best − default): point {delta_point:+.4f}, "
          f"paired-bootstrap mean {mean_diff:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}], p={p:.4f}")
    print(
        "selection caveat: 'best' chosen on the same query sample it is tested on "
        "-> CI is optimistic"
    )
    if adopt:
        print(f"DECISION: ADOPT k1={best['k1']}, b={best['b']}")
    else:
        print("DECISION: KEEP DEFAULT (k1=1.5, b=0.75)")
        if significant:
            print(
                f"  (gain is CI-significant but only {delta_point:+.4f} nDCG@{NDCG_K} "
                f"< {ADOPT_MIN_DELTA} -> not worth a config change)"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "dev_slice": "nfcorpus_test",
        "n_queries": len(eval_qids),
        "metric_primary": f"ndcg@{NDCG_K}",
        "top_k": TOP_K,
        "default": {k: v for k, v in default.items() if not k.startswith("_")},
        "best": {k: v for k, v in best.items() if not k.startswith("_")},
        "grid": [{k: v for k, v in r.items() if not k.startswith("_")} for r in grid],
        "paired_bootstrap_best_vs_default": {
            "on": f"per_query_ndcg@{NDCG_K}",
            "mean_diff": mean_diff,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p_value": p,
            "n_resamples": 10_000,
            "seed": 42,
            "significant_ci": significant,
            "selection_caveat": "best selected on the same sample it is tested on; CI optimistic",
        },
        "adopt_min_delta_ndcg": ADOPT_MIN_DELTA,
        "decision": "adopt" if adopt else "keep_default",
    }
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\n[bm25_tune] wrote {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
