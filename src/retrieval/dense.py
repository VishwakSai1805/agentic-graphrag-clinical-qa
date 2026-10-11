"""Dense retrieval: MedCPT dual-encoder embeddings + FAISS index.

ADR-010: MedCPT is an *asymmetric dual encoder*, not one model for both sides.
Verified directly against the model cards (huggingface.co/ncbi/MedCPT-Query-Encoder,
.../MedCPT-Article-Encoder), not assumed:
  - `ncbi/MedCPT-Query-Encoder` embeds queries: `AutoModel`, tokenize with
    `max_length=64`, representation = `last_hidden_state[:, 0, :]` ([CLS] pooling,
    NOT mean pooling).
  - `ncbi/MedCPT-Article-Encoder` embeds documents as `[title, text]` sentence
    pairs: `AutoModel`, tokenize the pair with `max_length=512`, same [CLS]
    pooling. Both live in the same embedding space (inner product comparable).

Fallback (config-selectable, not necessarily wired end-to-end): `BAAI/bge-large-en-v1.5`,
a *symmetric* encoder -- same model both sides. Verified against its model
card: query side gets the fixed instruction prefix
"Represent this sentence for searching relevant passages: " (passages get no
prefix), embeddings L2-normalised and compared by inner product (== cosine).

Indexes are build artifacts (PROJECT_RULES.md: never hand-edited, always gitignored,
always rebuildable) written under `indexes/<name>/` with a `manifest.json`
recording the embedding model(s), chunk params, and corpus doc/chunk counts.

Embedding is checkpointed every `checkpoint_every` batches
(`embeddings.npy` + `embed_checkpoint.json` in out_dir) so an interrupted
multi-hour build resumes from the last checkpoint instead of restarting from
zero, per PROJECT_RULES.md's "a 2-hour index build that dies at 90% must restart
from a checkpoint."
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MEDCPT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
MEDCPT_ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
BGE_MODEL = "BAAI/bge-large-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _read_corpus_rows(corpus_path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(corpus_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pick_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _cls_pool(last_hidden_state: torch.Tensor) -> torch.Tensor:
    return last_hidden_state[:, 0, :]


class DenseIndex:
    def __init__(
        self,
        faiss_index: faiss.Index,
        chunk_ids: list[str],
        *,
        model_type: str,
        query_model_name: str,
        device: str,
        nprobe: int,
        max_query_length: int,
        bge_query_prefix: str = BGE_QUERY_PREFIX,
    ) -> None:
        self.faiss_index = faiss_index
        self.chunk_ids = chunk_ids
        self.model_type = model_type
        self.device = device
        self.max_query_length = max_query_length
        self.bge_query_prefix = bge_query_prefix

        if hasattr(self.faiss_index, "nprobe"):
            self.faiss_index.nprobe = nprobe

        self._tokenizer = AutoTokenizer.from_pretrained(query_model_name)
        self._model = AutoModel.from_pretrained(query_model_name).to(device).eval()

    def _embed_query(self, query: str) -> np.ndarray:
        text = self.bge_query_prefix + query if self.model_type == "bge" else query
        with torch.no_grad():
            encoded = self._tokenizer(
                [text],
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=self.max_query_length,
            ).to(self.device)
            out = _cls_pool(self._model(**encoded).last_hidden_state)
            if self.model_type == "bge":
                out = torch.nn.functional.normalize(out, p=2, dim=1)
        return out.cpu().numpy().astype("float32")

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Returns up to k (chunk_id, score) pairs, best first. Score is raw
        FAISS inner-product similarity -- not comparable across modalities
        (ADR-002); only used for within-modality ranking before RRF fusion.
        """
        vec = self._embed_query(query)
        scores, idxs = self.faiss_index.search(vec, k)
        results: list[tuple[str, float]] = []
        for idx, score in zip(idxs[0], scores[0], strict=True):
            if idx == -1:
                continue
            results.append((self.chunk_ids[int(idx)], float(score)))
        return results

    @classmethod
    def build(
        cls,
        corpus_path: Path,
        out_dir: Path,
        *,
        model_type: str = "medcpt",
        query_model: str = MEDCPT_QUERY_MODEL,
        article_model: str = MEDCPT_ARTICLE_MODEL,
        bge_model: str = BGE_MODEL,
        device: str | None = None,
        batch_size: int = 64,
        max_query_length: int = 64,
        max_article_length: int = 512,
        nlist: int = 100,
        nprobe: int = 16,
        checkpoint_every: int = 25,
        print_every: int = 10,
        force: bool = False,
    ) -> DenseIndex:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifest.json"

        if manifest_path.exists() and not force:
            print(
                f"[dense] {manifest_path} exists; skipping dense index build "
                "(pass force=True / --force to rebuild)."
            )
            return cls.load(out_dir, device=device, nprobe=nprobe)

        device = _pick_device(device)
        rows = _read_corpus_rows(corpus_path)
        n = len(rows)
        if n == 0:
            raise ValueError(f"corpus at {corpus_path} has zero rows -- cannot build an index")
        chunk_ids = [r["chunk_id"] for r in rows]

        article_model_name = bge_model if model_type == "bge" else article_model
        q_model_name = bge_model if model_type == "bge" else query_model

        print(f"[dense] model_type={model_type} article_model={article_model_name} device={device}")
        tokenizer = AutoTokenizer.from_pretrained(article_model_name)
        model = AutoModel.from_pretrained(article_model_name).to(device).eval()
        dim = model.config.hidden_size

        emb_path = out_dir / "embeddings.npy"
        ckpt_path = out_dir / "embed_checkpoint.json"

        done = 0
        embeddings: np.ndarray
        if emb_path.exists() and ckpt_path.exists() and not force:
            ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
            if ckpt.get("n_total") == n:
                embeddings = np.load(emb_path)
                done = int(ckpt["n_done"])
                print(f"[dense] resuming embedding from checkpoint: {done}/{n} chunks already done")
            else:
                print("[dense] checkpoint corpus size mismatch -- starting embedding from scratch")
                embeddings = np.zeros((n, dim), dtype="float32")
        else:
            embeddings = np.zeros((n, dim), dtype="float32")

        t0 = time.time()
        n_batches_done = 0
        for start in range(done, n, batch_size):
            end = min(start + batch_size, n)
            batch_rows = rows[start:end]
            if model_type == "bge":
                inputs: list = [r["text"] for r in batch_rows]
            else:
                inputs = [[(r.get("meta") or {}).get("title") or "", r["text"]] for r in batch_rows]

            with torch.no_grad():
                encoded = tokenizer(
                    inputs,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=max_article_length,
                ).to(device)
                out = _cls_pool(model(**encoded).last_hidden_state)
                if model_type == "bge":
                    out = torch.nn.functional.normalize(out, p=2, dim=1)
            embeddings[start:end] = out.cpu().numpy().astype("float32")
            done = end
            n_batches_done += 1

            if n_batches_done % print_every == 0 or done == n:
                elapsed = time.time() - t0
                rate = (n_batches_done * batch_size) / max(elapsed, 1e-9)
                print(
                    f"[dense] embedded {done}/{n} chunks "
                    f"({elapsed:.1f}s elapsed, ~{rate:.1f} chunks/s)",
                    flush=True,
                )
            if n_batches_done % checkpoint_every == 0:
                np.save(emb_path, embeddings)
                ckpt_path.write_text(json.dumps({"n_done": done, "n_total": n}), encoding="utf-8")

        np.save(emb_path, embeddings)
        ckpt_path.write_text(json.dumps({"n_done": done, "n_total": n}), encoding="utf-8")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        print(f"[dense] building FAISS IVF-Flat index over {n} vectors (dim={dim})...")
        eff_nlist = max(1, min(nlist, n // 40)) if n >= 40 else 1
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, eff_nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        eff_nprobe = min(nprobe, eff_nlist)
        index.nprobe = eff_nprobe
        faiss.write_index(index, str(out_dir / "faiss.index"))
        (out_dir / "chunk_ids.json").write_text(json.dumps(chunk_ids), encoding="utf-8")

        manifest = {
            "model_type": model_type,
            "query_model": q_model_name,
            "article_model": article_model_name,
            "dim": dim,
            "n_chunks": n,
            "index_type": "IndexIVFFlat",
            "metric": "inner_product",
            "nlist": eff_nlist,
            "nprobe": eff_nprobe,
            "batch_size": batch_size,
            "max_query_length": max_query_length,
            "max_article_length": max_article_length,
            "corpus_path": str(corpus_path),
            "built_at_utc": datetime.now(UTC).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[dense] done, manifest at {manifest_path}")

        return cls.load(out_dir, device=device, nprobe=nprobe)

    @classmethod
    def load(
        cls, out_dir: Path, device: str | None = None, nprobe: int | None = None
    ) -> DenseIndex:
        out_dir = Path(out_dir)
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        chunk_ids = json.loads((out_dir / "chunk_ids.json").read_text(encoding="utf-8"))
        index = faiss.read_index(str(out_dir / "faiss.index"))
        resolved_device = _pick_device(device)
        return cls(
            index,
            chunk_ids,
            model_type=manifest["model_type"],
            query_model_name=manifest["query_model"],
            device=resolved_device,
            nprobe=nprobe if nprobe is not None else manifest.get("nprobe", 16),
            max_query_length=manifest.get("max_query_length", 64),
        )
