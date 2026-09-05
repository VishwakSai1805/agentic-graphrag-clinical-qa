"""Fetch NFCorpus (BEIR) and build a small, separate dense+BM25 index for it.

NFCorpus is a fusion *sanity-check* benchmark, not part of the project's
medical retrieval corpus: it is real, human-judged, graded-relevance IR with
no single-modality saturation (unlike PubMedQA, where questions closely
paraphrase their own source abstract -- see docs/05_DECISIONS.md). It exists
to test whether RRF fusion actually adds value when no modality dominates.

Sources (verified against the actual HF dataset schemas, not assumed):
  - `BeIR/nfcorpus`, config "corpus": 3,633 rows, fields `_id`, `title`, `text`.
  - `BeIR/nfcorpus`, config "queries": 3,237 rows (all splits merged), fields
    `_id`, `title` (always empty), `text`.
  - `BeIR/nfcorpus-qrels`, file `test.tsv`: tab-separated
    `query-id  corpus-id  score` graded relevance judgments (score in {1, 2}
    observed), 323 distinct test query ids.

Cached under data/raw/nfcorpus/ (jsonl for corpus/queries, tsv for qrels),
following the same raw-data convention as the other datasets. The chunked,
indexable corpus lives at data/processed/nfcorpus_corpus.jsonl (same schema
as the main corpus.jsonl, built with the identical chunker). Indexes live
under indexes/nfcorpus/{dense,bm25}/ -- deliberately NOT mixed into
indexes/dense or indexes/bm25, which are the main medical corpus's indexes.

Reuses DenseIndex, BM25Index, and chunk_text as-is (no forking).

Usage:
    uv run python -m scripts.build_nfcorpus [--force]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402
from src.retrieval.chunking import chunk_text, load_default_tokenizer  # noqa: E402
from src.retrieval.dense import DenseIndex  # noqa: E402
from src.retrieval.lexical import BM25Index  # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "raw" / "nfcorpus"
CORPUS_RAW_PATH = RAW_DIR / "corpus.jsonl"
QUERIES_RAW_PATH = RAW_DIR / "queries.jsonl"
QRELS_TEST_PATH = RAW_DIR / "qrels_test.tsv"

PROCESSED_CORPUS_PATH = REPO_ROOT / "data" / "processed" / "nfcorpus_corpus.jsonl"
DENSE_DIR = REPO_ROOT / "indexes" / "nfcorpus" / "dense"
BM25_DIR = REPO_ROOT / "indexes" / "nfcorpus" / "bm25"
MANIFEST_PATH = REPO_ROOT / "data" / "MANIFEST.json"
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"


def fetch_raw(force: bool = False) -> tuple[int, int, int, int]:
    """Downloads and caches NFCorpus corpus/queries/qrels under data/raw/nfcorpus/.
    Returns (n_corpus_docs, n_queries, n_qrels_test_queries, n_qrels_test_judgments).
    Idempotent: skips any file that already exists unless force=True.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if CORPUS_RAW_PATH.exists() and not force:
        print(f"[build_nfcorpus] {CORPUS_RAW_PATH} exists; skipping corpus fetch.")
        n_corpus = sum(1 for _ in CORPUS_RAW_PATH.open("r", encoding="utf-8"))
    else:
        from datasets import load_dataset

        print("[build_nfcorpus] downloading BeIR/nfcorpus (corpus)...")
        corpus_ds = load_dataset("BeIR/nfcorpus", "corpus")["corpus"]
        with CORPUS_RAW_PATH.open("w", encoding="utf-8") as f:
            for row in corpus_ds:
                f.write(
                    json.dumps(
                        {"doc_id": row["_id"], "title": row["title"], "text": row["text"]}
                    )
                    + "\n"
                )
        n_corpus = len(corpus_ds)
        print(f"[build_nfcorpus] wrote {n_corpus} corpus docs to {CORPUS_RAW_PATH}")

    if QUERIES_RAW_PATH.exists() and not force:
        print(f"[build_nfcorpus] {QUERIES_RAW_PATH} exists; skipping queries fetch.")
        n_queries = sum(1 for _ in QUERIES_RAW_PATH.open("r", encoding="utf-8"))
    else:
        from datasets import load_dataset

        print("[build_nfcorpus] downloading BeIR/nfcorpus (queries)...")
        queries_ds = load_dataset("BeIR/nfcorpus", "queries")["queries"]
        with QUERIES_RAW_PATH.open("w", encoding="utf-8") as f:
            for row in queries_ds:
                f.write(json.dumps({"query_id": row["_id"], "text": row["text"]}) + "\n")
        n_queries = len(queries_ds)
        print(f"[build_nfcorpus] wrote {n_queries} queries to {QUERIES_RAW_PATH}")

    if QRELS_TEST_PATH.exists() and not force:
        print(f"[build_nfcorpus] {QRELS_TEST_PATH} exists; skipping qrels fetch.")
    else:
        from huggingface_hub import hf_hub_download

        print("[build_nfcorpus] downloading BeIR/nfcorpus-qrels test.tsv...")
        downloaded = hf_hub_download(
            repo_id="BeIR/nfcorpus-qrels", filename="test.tsv", repo_type="dataset"
        )
        shutil.copyfile(downloaded, QRELS_TEST_PATH)
        print(f"[build_nfcorpus] wrote qrels to {QRELS_TEST_PATH}")

    test_qids: set[str] = set()
    n_judgments = 0
    with QRELS_TEST_PATH.open("r", encoding="utf-8") as f:
        next(f)  # header: query-id  corpus-id  score
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, _cid, _score = line.split("\t")
            test_qids.add(qid)
            n_judgments += 1

    return n_corpus, n_queries, len(test_qids), n_judgments


def build_processed_corpus(chunk_tokens: int, chunk_overlap: int, tokenizer_model: str) -> int:
    """Chunks data/raw/nfcorpus/corpus.jsonl with the identical project
    chunker and writes data/processed/nfcorpus_corpus.jsonl in the same
    schema as the main corpus.jsonl (doc_id, chunk_id, source, text, meta).
    """
    tokenizer = load_default_tokenizer(tokenizer_model)
    PROCESSED_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)

    n_chunks = 0
    with (
        CORPUS_RAW_PATH.open("r", encoding="utf-8") as fin,
        PROCESSED_CORPUS_PATH.open("w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            doc_id = row["doc_id"]
            chunks = chunk_text(
                row["text"],
                chunk_tokens=chunk_tokens,
                chunk_overlap=chunk_overlap,
                tokenizer=tokenizer,
            )
            for c in chunks:
                out_row = {
                    "doc_id": doc_id,
                    "chunk_id": f"chunk::{doc_id}#{c.index}",
                    "source": "nfcorpus",
                    "text": c.text,
                    "meta": {"title": row.get("title", ""), "n_tokens": c.n_tokens},
                }
                fout.write(json.dumps(out_row) + "\n")
                n_chunks += 1

    return n_chunks


def update_manifest(
    manifest_path: Path,
    *,
    n_corpus_docs: int,
    n_chunks: int,
    n_queries_total: int,
    n_test_queries: int,
    n_test_judgments: int,
    chunk_tokens: int,
    chunk_overlap: int,
    dense_manifest: dict,
    bm25_manifest: dict,
    build_wall_clock_s: dict[str, float],
) -> None:
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["nfcorpus_sanity_benchmark"] = {
        "purpose": "fusion sanity-check benchmark, NOT part of the medical retrieval corpus",
        "source": {
            "corpus": "BeIR/nfcorpus (config=corpus)",
            "queries": "BeIR/nfcorpus (config=queries)",
            "qrels": "BeIR/nfcorpus-qrels (test.tsv)",
        },
        "n_corpus_docs": n_corpus_docs,
        "n_chunks": n_chunks,
        "n_queries_total": n_queries_total,
        "n_test_queries": n_test_queries,
        "n_test_judgments": n_test_judgments,
        "chunk_tokens": chunk_tokens,
        "chunk_overlap": chunk_overlap,
        "dense_index": dense_manifest,
        "bm25_index": bm25_manifest,
        "build_wall_clock_s": build_wall_clock_s,
        "built_at_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild everything from scratch")
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    rc = cfg.retrieval

    build_wall_clock: dict[str, float] = {}

    t0 = time.time()
    n_corpus_docs, n_queries_total, n_test_queries, n_test_judgments = fetch_raw(force=args.force)
    build_wall_clock["fetch_raw_s"] = time.time() - t0
    print(
        f"[build_nfcorpus] corpus={n_corpus_docs} docs, queries={n_queries_total} total, "
        f"test queries={n_test_queries}, test judgments={n_test_judgments}"
    )

    if PROCESSED_CORPUS_PATH.exists() and not args.force:
        print(
            f"[build_nfcorpus] {PROCESSED_CORPUS_PATH} exists; skipping chunking "
            "(pass --force to rebuild)."
        )
        n_chunks = sum(1 for _ in PROCESSED_CORPUS_PATH.open("r", encoding="utf-8"))
    else:
        t0 = time.time()
        n_chunks = build_processed_corpus(rc.chunk_tokens, rc.chunk_overlap, rc.tokenizer_model)
        build_wall_clock["chunk_s"] = time.time() - t0
        print(f"[build_nfcorpus] chunked into {n_chunks} chunks -> {PROCESSED_CORPUS_PATH}")

    print("[build_nfcorpus] building dense (FAISS + MedCPT) index...")
    t0 = time.time()
    DenseIndex.build(
        corpus_path=PROCESSED_CORPUS_PATH,
        out_dir=DENSE_DIR,
        model_type=rc.dense.model_type,
        query_model=rc.dense.query_model,
        article_model=rc.dense.article_model,
        bge_model=rc.dense.bge_model,
        device=rc.dense.device,
        batch_size=rc.dense.batch_size,
        max_query_length=rc.dense.max_query_length,
        max_article_length=rc.dense.max_article_length,
        nlist=rc.dense.nlist,
        nprobe=rc.dense.nprobe,
        checkpoint_every=rc.dense.checkpoint_every_batches,
        force=args.force,
    )
    build_wall_clock["dense_build_s"] = time.time() - t0
    dense_manifest = json.loads((DENSE_DIR / "manifest.json").read_text(encoding="utf-8"))

    print("[build_nfcorpus] building BM25 index...")
    t0 = time.time()
    BM25Index.build(
        corpus_path=PROCESSED_CORPUS_PATH,
        out_dir=BM25_DIR,
        k1=rc.bm25.k1,
        b=rc.bm25.b,
        force=args.force,
    )
    build_wall_clock["bm25_build_s"] = time.time() - t0
    bm25_manifest = json.loads((BM25_DIR / "manifest.json").read_text(encoding="utf-8"))

    update_manifest(
        MANIFEST_PATH,
        n_corpus_docs=n_corpus_docs,
        n_chunks=n_chunks,
        n_queries_total=n_queries_total,
        n_test_queries=n_test_queries,
        n_test_judgments=n_test_judgments,
        chunk_tokens=rc.chunk_tokens,
        chunk_overlap=rc.chunk_overlap,
        dense_manifest=dense_manifest,
        bm25_manifest=bm25_manifest,
        build_wall_clock_s=build_wall_clock,
    )
    print(f"[build_nfcorpus] merged nfcorpus build record into {MANIFEST_PATH}")
    print(f"[build_nfcorpus] done. total wall-clock: {build_wall_clock}")


if __name__ == "__main__":
    main()
