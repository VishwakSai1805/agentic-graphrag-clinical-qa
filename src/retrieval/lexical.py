"""Lexical retrieval: BM25 (`rank_bm25`) over the same corpus.jsonl chunks
dense.py indexes. dev-scale choice per CLAUDE.md §3 (Pyserini at real scale).

Index artifacts (pickled BM25Okapi + chunk_ids + manifest) live under
`indexes/<name>/`, gitignored, always rebuildable from
`data/processed/corpus.jsonl` -- never hand-edited.
"""

from __future__ import annotations

import json
import pickle
import re
from datetime import UTC, datetime
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _read_corpus_rows(corpus_path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(corpus_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class BM25Index:
    def __init__(self, bm25: BM25Okapi, chunk_ids: list[str]) -> None:
        self.bm25 = bm25
        self.chunk_ids = chunk_ids

    @classmethod
    def build(
        cls,
        corpus_path: Path,
        out_dir: Path,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        force: bool = False,
    ) -> BM25Index:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifest.json"

        if manifest_path.exists() and not force:
            print(
                f"[bm25] {manifest_path} exists; skipping BM25 build "
                "(pass force=True / --force to rebuild)."
            )
            return cls.load(out_dir)

        rows = _read_corpus_rows(corpus_path)
        chunk_ids = [r["chunk_id"] for r in rows]
        tokenized = [_tokenize(r["text"]) for r in rows]

        print(f"[bm25] indexing {len(chunk_ids)} chunks (k1={k1}, b={b})...")
        bm25 = BM25Okapi(tokenized, k1=k1, b=b)

        with (out_dir / "bm25.pkl").open("wb") as f:
            pickle.dump(bm25, f)
        (out_dir / "chunk_ids.json").write_text(json.dumps(chunk_ids), encoding="utf-8")

        manifest = {
            "k1": k1,
            "b": b,
            "n_chunks": len(chunk_ids),
            "corpus_path": str(corpus_path),
            "built_at_utc": datetime.now(UTC).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[bm25] done, manifest at {manifest_path}")
        return cls(bm25, chunk_ids)

    @classmethod
    def load(cls, out_dir: Path) -> BM25Index:
        out_dir = Path(out_dir)
        with (out_dir / "bm25.pkl").open("rb") as f:
            bm25 = pickle.load(f)  # noqa: S301 -- our own build artifact, not untrusted input
        chunk_ids = json.loads((out_dir / "chunk_ids.json").read_text(encoding="utf-8"))
        return cls(bm25, chunk_ids)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(_tokenize(query))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_idx]
