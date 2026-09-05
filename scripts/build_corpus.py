"""Build the retrieval corpus (PubMed abstracts subset + 18 USMLE textbooks),
chunk it, dedupe, and build the dense (FAISS/MedCPT) and BM25 indexes.

Usage:
    uv run python -m scripts.build_corpus [--force] [--pubmed-limit N]

Idempotent / resumable (CLAUDE.md: "a 2-hour index build that dies at 90%
must restart from a checkpoint"):
  - If data/processed/corpus.jsonl already exists and --force is not passed,
    the corpus-build step (sampling, chunking, dedupe) is skipped entirely.
  - The dense index build (the actual multi-hour step) checkpoints embedding
    progress every N batches (see src/retrieval/dense.py) and resumes from
    the last checkpoint on rerun.
  - The BM25 index build is skipped if its manifest already exists (it is
    fast enough that no finer-grained checkpointing is needed).

After building, asserts no PubMedQA-test-blocklisted pubid made it into the
corpus (src/eval/contamination.py) -- hard failure, not a warning, per
CLAUDE.md. Then merges a "corpus" key into data/MANIFEST.json (preserving the
existing "datasets" key written by scripts/fetch_datasets.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402
from src.eval.contamination import assert_no_contamination, blocklisted_pmids  # noqa: E402
from src.retrieval.chunking import chunk_text, load_default_tokenizer  # noqa: E402
from src.retrieval.dense import DenseIndex  # noqa: E402
from src.retrieval.lexical import BM25Index  # noqa: E402

PUBMED_ARTIFICIAL_PATH = REPO_ROOT / "data" / "raw" / "pubmedqa" / "pqa_artificial_train.jsonl"
TEXTBOOKS_DIR = REPO_ROOT / "data" / "raw" / "textbooks" / "en"
CORPUS_PATH = REPO_ROOT / "data" / "processed" / "corpus.jsonl"
DENSE_DIR = REPO_ROOT / "indexes" / "dense"
BM25_DIR = REPO_ROOT / "indexes" / "bm25"
MANIFEST_PATH = REPO_ROOT / "data" / "MANIFEST.json"
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"


def sample_pubmed_docs(path: Path, subset_size: int, seed: int) -> list[dict]:
    """Fixed-seed subset of pqa_artificial (211k rows, disjoint pubid pool
    from the blocklisted pqa_labeled test split). Concatenates each row's
    context.contexts list into one text blob per doc; doc_id = str(pubid).
    """
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    rng = random.Random(seed)
    n = len(lines)
    chosen_idx = rng.sample(range(n), min(subset_size, n))

    docs = []
    for i in chosen_idx:
        row = json.loads(lines[i])
        contexts = (row.get("context") or {}).get("contexts") or []
        text = " ".join(contexts).strip()
        if not text:
            continue
        docs.append(
            {
                "doc_id": str(row["pubid"]),
                "source": "pubmed_abstract",
                "title": row.get("question", "") or "",
                "text": text,
                "meta": {
                    "final_decision": row.get("final_decision"),
                    "n_contexts": len(contexts),
                },
            }
        )
    return docs


def load_textbook_docs(dir_path: Path) -> list[dict]:
    """All 18 USMLE textbooks, in full (ADR-009 -- not subsampled). One
    normalized "document" per book; chunking splits each into many chunks.
    """
    docs = []
    for fp in sorted(dir_path.glob("*.txt")):
        book = fp.stem
        text = fp.read_text(encoding="utf-8", errors="ignore")
        docs.append(
            {
                "doc_id": f"textbook::{book}",
                "source": "textbook",
                "title": book.replace("_", " "),
                "text": text,
                "meta": {"book": book},
            }
        )
    return docs


def build_corpus_file(
    corpus_path: Path,
    docs: list[dict],
    tokenizer,
    chunk_tokens: int,
    chunk_overlap: int,
) -> tuple[int, dict[str, int]]:
    """Chunks every doc with the identical chunker, dedupes exact-text-match
    chunks globally, and writes data/processed/corpus.jsonl.
    """
    seen_hashes: set[str] = set()
    n_chunks = 0
    n_deduped = 0
    per_source_counts: dict[str, int] = defaultdict(int)
    t0 = time.time()

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("w", encoding="utf-8") as out:
        for doc_i, doc in enumerate(docs):
            chunks = chunk_text(
                doc["text"],
                chunk_tokens=chunk_tokens,
                chunk_overlap=chunk_overlap,
                tokenizer=tokenizer,
            )
            for c in chunks:
                h = hashlib.md5(c.text.encode("utf-8")).hexdigest()
                if h in seen_hashes:
                    n_deduped += 1
                    continue
                seen_hashes.add(h)
                row = {
                    "doc_id": doc["doc_id"],
                    "chunk_id": f"chunk::{doc['doc_id']}#{c.index}",
                    "source": doc["source"],
                    "text": c.text,
                    "meta": {**doc["meta"], "title": doc["title"], "n_tokens": c.n_tokens},
                }
                out.write(json.dumps(row) + "\n")
                n_chunks += 1
                per_source_counts[doc["source"]] += 1

            if (doc_i + 1) % 250 == 0 or doc_i + 1 == len(docs):
                elapsed = time.time() - t0
                print(
                    f"[build_corpus] chunked {doc_i + 1}/{len(docs)} docs, "
                    f"{n_chunks} chunks written, {n_deduped} exact-dup chunks "
                    f"dropped ({elapsed:.1f}s elapsed)",
                    flush=True,
                )

    return n_chunks, dict(per_source_counts)


def count_existing_corpus(corpus_path: Path) -> tuple[int, dict[str, int]]:
    n_chunks = 0
    per_source_counts: dict[str, int] = defaultdict(int)
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_chunks += 1
            per_source_counts[row.get("source", "unknown")] += 1
    return n_chunks, dict(per_source_counts)


def update_manifest(
    manifest_path: Path,
    *,
    n_chunks: int,
    per_source_counts: dict[str, int],
    n_pubmed_docs: int,
    n_textbook_docs: int,
    chunk_tokens: int,
    chunk_overlap: int,
    pubmed_subset_size: int,
    pubmed_seed: int,
    dense_manifest: dict,
    bm25_manifest: dict,
    build_wall_clock_s: dict[str, float],
) -> None:
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["corpus"] = {
        "n_chunks_total": n_chunks,
        "n_chunks_per_source": per_source_counts,
        "n_pubmed_docs": n_pubmed_docs,
        "n_textbook_docs": n_textbook_docs,
        "chunk_tokens": chunk_tokens,
        "chunk_overlap": chunk_overlap,
        "pubmed_subset_size": pubmed_subset_size,
        "pubmed_seed": pubmed_seed,
        "dense_index": dense_manifest,
        "bm25_index": bm25_manifest,
        "build_wall_clock_s": build_wall_clock_s,
        "built_at_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="rebuild corpus + indexes from scratch"
    )
    parser.add_argument(
        "--pubmed-limit",
        type=int,
        default=None,
        help="override retrieval.corpus.pubmed_subset_size from configs/base.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    rc = cfg.retrieval

    pubmed_subset_size = args.pubmed_limit or rc.corpus.pubmed_subset_size
    pubmed_seed = rc.corpus.pubmed_seed

    build_wall_clock: dict[str, float] = {}
    n_pubmed_docs = 0
    n_textbook_docs = 0

    if CORPUS_PATH.exists() and not args.force:
        print(
            f"[build_corpus] {CORPUS_PATH} already exists; skipping corpus build "
            "(pass --force to rebuild from scratch)."
        )
        n_chunks, per_source_counts = count_existing_corpus(CORPUS_PATH)
    else:
        t0 = time.time()
        print(
            f"[build_corpus] sampling {pubmed_subset_size} pqa_artificial docs "
            f"(seed={pubmed_seed})..."
        )
        pubmed_docs = sample_pubmed_docs(PUBMED_ARTIFICIAL_PATH, pubmed_subset_size, pubmed_seed)
        n_pubmed_docs = len(pubmed_docs)
        print(f"[build_corpus] sampled {n_pubmed_docs} non-empty pubmed docs")

        print(f"[build_corpus] loading textbooks from {TEXTBOOKS_DIR}...")
        textbook_docs = load_textbook_docs(TEXTBOOKS_DIR)
        n_textbook_docs = len(textbook_docs)
        print(f"[build_corpus] loaded {n_textbook_docs} textbooks")

        all_docs = pubmed_docs + textbook_docs
        print(f"[build_corpus] loading tokenizer ({rc.tokenizer_model})...")
        tokenizer = load_default_tokenizer(rc.tokenizer_model)

        print(
            f"[build_corpus] chunking {len(all_docs)} docs "
            f"(chunk_tokens={rc.chunk_tokens}, chunk_overlap={rc.chunk_overlap})..."
        )
        n_chunks, per_source_counts = build_corpus_file(
            CORPUS_PATH, all_docs, tokenizer, rc.chunk_tokens, rc.chunk_overlap
        )
        corpus_build_s = time.time() - t0
        build_wall_clock["corpus_build_s"] = corpus_build_s
        print(
            f"[build_corpus] wrote {n_chunks} chunks to {CORPUS_PATH} "
            f"in {corpus_build_s:.1f}s ({per_source_counts})"
        )

    print("[build_corpus] building dense (FAISS + MedCPT) index...")
    t0 = time.time()
    dense_index = DenseIndex.build(
        corpus_path=CORPUS_PATH,
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

    print("[build_corpus] building BM25 index...")
    t0 = time.time()
    BM25Index.build(
        corpus_path=CORPUS_PATH,
        out_dir=BM25_DIR,
        k1=rc.bm25.k1,
        b=rc.bm25.b,
        force=args.force,
    )
    build_wall_clock["bm25_build_s"] = time.time() - t0
    bm25_manifest = json.loads((BM25_DIR / "manifest.json").read_text(encoding="utf-8"))

    print("[build_corpus] asserting no PubMedQA-test contamination in the corpus...")
    assert_no_contamination(CORPUS_PATH, blocklisted_pmids("pubmedqa"))
    print("[build_corpus] contamination check passed: no blocklisted pubid in the corpus.")

    update_manifest(
        MANIFEST_PATH,
        n_chunks=n_chunks,
        per_source_counts=per_source_counts,
        n_pubmed_docs=n_pubmed_docs,
        n_textbook_docs=n_textbook_docs,
        chunk_tokens=rc.chunk_tokens,
        chunk_overlap=rc.chunk_overlap,
        pubmed_subset_size=pubmed_subset_size,
        pubmed_seed=pubmed_seed,
        dense_manifest=dense_manifest,
        bm25_manifest=bm25_manifest,
        build_wall_clock_s=build_wall_clock,
    )
    print(f"[build_corpus] merged corpus build record into {MANIFEST_PATH}")
    print(f"[build_corpus] done. total wall-clock: {build_wall_clock}")

    del dense_index  # keep the reference alive above for clarity; nothing else to do with it here


if __name__ == "__main__":
    main()
