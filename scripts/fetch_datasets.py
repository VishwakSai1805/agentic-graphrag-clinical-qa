"""Fetch and cache every Phase 1 dataset under data/raw/, idempotently.

Sources:
  - PubMedQA (pqa_labeled, pqa_artificial) via HF `datasets`
  - MedQA-USMLE 4-options via HF `datasets`
  - MedMCQA via HF `datasets`
  - MedQA textbook corpus (18 USMLE textbooks) via the HF mirror
    `MedRAG/textbooks` — the original jind11/MedQA release only ships these
    via an unscriptable Google Drive link, so this is the primary path, not
    a fallback. If the HF mirror(s) are unreachable, a best-effort attempt
    at the GitHub-linked Google Drive archive is made; if that also fails,
    this script HALTS with a nonzero exit rather than silently proceeding
    on an abstracts-only corpus (the textbooks are what USMLE-style
    questions are actually answered from — see docs/02_ROADMAP.md Phase 1
    and the ADR in docs/05_DECISIONS.md).

Writes/merges dataset revisions and example counts into data/MANIFEST.json
under the "datasets" key. Re-running is safe: already-downloaded files are
skipped unless --force is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download

DATA_RAW = Path("data/raw")
MANIFEST_PATH = Path("data/MANIFEST.json")

TEXTBOOK_HF_MIRRORS = ["MedRAG/textbooks"]
# Referenced from https://github.com/jind11/MedQA's README as the only
# source for the raw textbook release; kept only as a last-resort fallback
# since large Google Drive files require confirm-token handling and are
# not a stable scripted download path.
TEXTBOOK_GDRIVE_FILE_ID = "1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw"


def _load_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _hf_revision(repo_id: str, repo_type: str = "dataset") -> str:
    return HfApi().dataset_info(repo_id).sha if repo_type == "dataset" else ""


def fetch_pubmedqa(force: bool) -> dict[str, Any]:
    out_dir = DATA_RAW / "pubmedqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for config in ("pqa_labeled", "pqa_artificial"):
        ds = load_dataset("qiaojin/PubMedQA", config)
        for split, data in ds.items():
            path = out_dir / f"{config}_{split}.jsonl"
            if path.exists() and not force:
                counts[f"{config}_{split}"] = len(data)
                continue
            data.to_json(str(path))
            counts[f"{config}_{split}"] = len(data)
    return {
        "hf_id": "qiaojin/PubMedQA",
        "revision": _hf_revision("qiaojin/PubMedQA"),
        "splits": counts,
    }


def fetch_medqa_usmle(force: bool) -> dict[str, Any]:
    out_dir = DATA_RAW / "medqa_usmle"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("GBaker/MedQA-USMLE-4-options")
    counts: dict[str, int] = {}
    for split, data in ds.items():
        path = out_dir / f"{split}.jsonl"
        if path.exists() and not force:
            counts[split] = len(data)
            continue
        data.to_json(str(path))
        counts[split] = len(data)
    return {
        "hf_id": "GBaker/MedQA-USMLE-4-options",
        "revision": _hf_revision("GBaker/MedQA-USMLE-4-options"),
        "splits": counts,
    }


def fetch_medmcqa(force: bool) -> dict[str, Any]:
    out_dir = DATA_RAW / "medmcqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("openlifescienceai/medmcqa")
    counts: dict[str, int] = {}
    for split, data in ds.items():
        path = out_dir / f"{split}.jsonl"
        if path.exists() and not force:
            counts[split] = len(data)
            continue
        data.to_json(str(path))
        counts[split] = len(data)
    return {
        "hf_id": "openlifescienceai/medmcqa",
        "revision": _hf_revision("openlifescienceai/medmcqa"),
        "splits": counts,
    }


def _reconstruct_book_text(chunks_jsonl: Path) -> str:
    """MedRAG's chunk/*.jsonl is pre-chunked ({"id": "<Title>_<n>", "content": ...}).
    Reconstruct one raw-text file per book, ordered by chunk index, so every
    corpus source (PubMed abstracts, textbooks) goes through the *same*
    ~256 tok / 32 overlap chunker in scripts/build_corpus.py rather than
    mixing chunking policies across sources."""
    lines = chunks_jsonl.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line]
    rows.sort(key=lambda r: int(r["id"].rsplit("_", 1)[-1]))
    return "\n\n".join(r["content"] for r in rows)


def _try_hf_textbook_mirrors(out_dir: Path, force: bool) -> dict[str, Any] | None:
    en_dir = out_dir / "en"
    for repo_id in TEXTBOOK_HF_MIRRORS:
        try:
            HfApi().dataset_info(repo_id)  # cheap reachability check first
        except Exception as e:  # noqa: BLE001 - report and try next mirror
            print(f"  [textbooks] mirror {repo_id} unreachable: {e}", file=sys.stderr)
            continue
        try:
            local_dir = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=["chunk/*.jsonl"],
                local_dir=str(out_dir),
                force_download=force,
            )
            chunk_files = sorted(Path(local_dir, "chunk").glob("*.jsonl"))
            if not chunk_files:
                print(f"  [textbooks] mirror {repo_id} had no chunk/*.jsonl files", file=sys.stderr)
                continue
            en_dir.mkdir(parents=True, exist_ok=True)
            for cf in chunk_files:
                (en_dir / f"{cf.stem}.txt").write_text(_reconstruct_book_text(cf), encoding="utf-8")
            return {
                "hf_id": repo_id,
                "revision": _hf_revision(repo_id),
                "n_books": len(chunk_files),
                "books": [b.stem for b in chunk_files],
            }
        except Exception as e:  # noqa: BLE001 - report and try next mirror
            print(f"  [textbooks] mirror {repo_id} download failed: {e}", file=sys.stderr)
            continue
    return None


def _try_github_gdrive_fallback(out_dir: Path, force: bool) -> dict[str, Any] | None:
    """jind11/MedQA's README links a Google Drive zip, not a scripted API —
    large-file Drive downloads need a virus-scan confirm token and no stable
    API, so this is a best-effort fallback, not a guaranteed path. Extracts
    just the English textbooks (data_clean/textbooks/en/*.txt) on success."""
    import zipfile

    en_dir = out_dir / "en"
    existing = sorted(en_dir.glob("*.txt")) if en_dir.exists() else []
    if existing and not force:
        return {
            "source": "jind11/MedQA (Google Drive)",
            "n_books": len(existing),
            "books": [b.stem for b in existing],
        }

    url = (
        "https://drive.usercontent.google.com/download"
        f"?id={TEXTBOOK_GDRIVE_FILE_ID}&export=download&confirm=t"
    )
    archive = out_dir / "medqa_data_clean.zip"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, archive)  # noqa: S310 - one-shot fallback, not user input
        # a real archive is hundreds of MB; anything tiny is an HTML warning page
        if archive.stat().st_size < 10_000_000:
            archive.unlink(missing_ok=True)
            return None
        en_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            books = [
                i
                for i in z.infolist()
                if i.filename.startswith("data_clean/textbooks/en/") and i.filename.endswith(".txt")
            ]
            if not books:
                return None
            for info in books:
                (en_dir / Path(info.filename).name).write_bytes(z.read(info))
        return {
            "source": "jind11/MedQA (Google Drive)",
            "n_books": len(books),
            "books": [Path(b.filename).stem for b in books],
        }
    except Exception as e:  # noqa: BLE001
        print(f"  [textbooks] GitHub/Google Drive fallback failed: {e}", file=sys.stderr)
        return None


def fetch_medqa_textbooks(force: bool) -> dict[str, Any]:
    out_dir = DATA_RAW / "textbooks"
    en_dir = out_dir / "en"
    print("Fetching MedQA textbook corpus (18 USMLE textbooks)...")

    existing = sorted(en_dir.glob("*.txt")) if en_dir.exists() else []
    if len(existing) >= 18 and not force:
        print(f"  already present: {len(existing)} books in {en_dir}")
        return {"source": "cached", "n_books": len(existing), "books": [b.stem for b in existing]}

    result = _try_hf_textbook_mirrors(out_dir, force)
    if result is not None:
        print(f"  ok: {result['n_books']} books from {result['hf_id']}")
        return result

    print("  HF mirrors exhausted, trying jind11/MedQA GitHub->Google Drive source...")
    result = _try_github_gdrive_fallback(out_dir, force)
    if result is not None:
        print(f"  ok: {result['n_books']} books from {result['source']}")
        return result

    print(
        "\nFATAL: could not fetch the MedQA textbook corpus from any source "
        "(HF mirrors or the jind11/MedQA GitHub->Google Drive link).\n"
        "Per the Phase 1 decision, this is NOT a fast-follow — the pipeline "
        "does not proceed on an abstracts-only corpus. Stopping.\n",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-download even if files already exist"
    )
    args = parser.parse_args()

    manifest = _load_manifest()
    manifest.setdefault("datasets", {})

    print("Fetching PubMedQA (pqa_labeled + pqa_artificial)...")
    manifest["datasets"]["pubmedqa"] = fetch_pubmedqa(args.force)

    print("Fetching MedQA-USMLE-4-options...")
    manifest["datasets"]["medqa_usmle"] = fetch_medqa_usmle(args.force)

    print("Fetching MedMCQA...")
    manifest["datasets"]["medmcqa"] = fetch_medmcqa(args.force)

    manifest["datasets"]["medqa_textbooks"] = fetch_medqa_textbooks(args.force)

    _save_manifest(manifest)
    print(f"\nManifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
