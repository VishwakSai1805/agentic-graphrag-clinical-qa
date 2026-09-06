"""Checksum manifest for the precomputed KG caches (ADR-007 artifacts).

Why this exists: `data/processed/kg_cache/` and
`data/processed/kg_cache_kg_rerank/` are gitignored (`.gitignore`: `data/*`).
ADR-018 called them "already-committed canonical" -- they were never
committed. Committing ~1700 small JSON files is the wrong fix (churns the
repo, and the real provenance is the seed + Neo4j graph, not the bytes). This
script records, into the tracked `data/MANIFEST.json`, one aggregate SHA-256
per cache tree plus the exact command that regenerates it, so drift is
detectable and the cache is reproducible. See ADR (this phase) correcting
ADR-018.

    uv run python -m scripts.kg_cache_manifest --write     # refresh the manifest block
    uv run python -m scripts.kg_cache_manifest --check      # verify caches (exit 1 on drift)

`aggregate_sha256` = SHA-256 over the newline-joined sorted list of
`<relpath>\\t<sha256-of-file>` lines, relpath POSIX-style and relative to the
cache tree root. Order-independent, rename-sensitive, content-sensitive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "MANIFEST.json"

# tree relpath -> {regeneration command, provenance}. One entry per cache tree
# that Phase 2 (and its follow-up) actually produced -- see
# docs/02_ROADMAP.md Phase 2 and ADR-005 for the split/limit/seed rationale.
_PC = "uv run python -m scripts.precompute_kg_cache"
CACHE_TREES: dict[str, dict[str, str]] = {
    "data/processed/kg_cache/medqa": {
        "regen": f"{_PC} --dataset medqa --split test --limit 200  "
        f"&&  {_PC} --dataset medqa --split train --limit 20",
        "note": "Phase 2 exit slice (200 MedQA test, seed 42) + Phase 3 dev slice "
        "(20 MedQA train, seed 42).",
    },
    "data/processed/kg_cache/pubmedqa": {
        "regen": f"{_PC} --dataset pubmedqa --split test --limit 500  "
        f"&&  {_PC} --dataset pubmedqa --split dev --limit 500",
        "note": "Phase 2 follow-up: 500 test + 500 dev, seed 42 (ADR-005 eval scope).",
    },
    "data/processed/kg_cache/medmcqa": {
        "regen": f"{_PC} --dataset medmcqa --split validation --limit 500",
        "note": "Phase 2 follow-up: 500-question seeded validation subsample, seed 42 (ADR-005).",
    },
    "data/processed/kg_cache_kg_rerank/medqa": {
        "regen": f"{_PC} --dataset medqa --split test --limit 200 --kg-rerank "
        "--cache-dir data/processed/kg_cache_kg_rerank",
        "note": "ADR-018 kg_rerank ablation variant, 200 MedQA test, seed 42. Off by default.",
    },
}


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def aggregate_sha256(tree_root: Path) -> tuple[str, int]:
    """Returns (aggregate hex digest, n_files) for every *.json under tree_root."""
    files = sorted(tree_root.rglob("*.json"))
    lines = []
    for f in files:
        rel = f.relative_to(tree_root).as_posix()
        lines.append(f"{rel}\t{_file_sha256(f)}")
    digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    return digest, len(files)


def compute_block() -> dict:
    trees: dict[str, dict] = {}
    for rel, info in CACHE_TREES.items():
        root = REPO_ROOT / rel
        if not root.exists():
            trees[rel] = {"present": False, **info}
            continue
        digest, n = aggregate_sha256(root)
        trees[rel] = {
            "present": True,
            "n_files": n,
            "aggregate_sha256": digest,
            **info,
        }
    return {
        "_comment": (
            "Aggregate SHA-256 per precomputed KG cache tree (ADR-007). Trees are "
            "gitignored; regenerate with the 'regen' command (needs Neo4j up -- "
            "docker/compose.yaml + scripts/load_primekg.py). See ADR correcting "
            "ADR-018's 'already-committed canonical' claim."
        ),
        "trees": trees,
    }


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def write_manifest() -> None:
    manifest = _load_manifest()
    manifest["kg_cache"] = compute_block()
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[kg_cache_manifest] wrote kg_cache block into {MANIFEST_PATH}")
    for rel, t in manifest["kg_cache"]["trees"].items():
        if t.get("present"):
            print(f"  {rel}: {t['n_files']} files, sha256={t['aggregate_sha256'][:16]}...")
        else:
            print(f"  {rel}: NOT PRESENT on disk")


def check_manifest() -> int:
    manifest = _load_manifest()
    recorded = manifest.get("kg_cache", {}).get("trees")
    if not recorded:
        print(
            "[kg_cache_manifest] no kg_cache block in MANIFEST.json -- run --write first.",
            file=sys.stderr,
        )
        return 1

    live = compute_block()["trees"]
    drift = 0
    for rel, rec in recorded.items():
        cur = live.get(rel, {})
        if not rec.get("present"):
            continue
        if not cur.get("present"):
            print(f"DRIFT {rel}: recorded as present, missing on disk")
            drift += 1
            continue
        if cur["aggregate_sha256"] != rec["aggregate_sha256"]:
            print(
                f"DRIFT {rel}: sha256 {cur['aggregate_sha256'][:16]}... != "
                f"recorded {rec['aggregate_sha256'][:16]}... "
                f"(n_files {cur['n_files']} vs {rec['n_files']})"
            )
            drift += 1
        else:
            print(f"OK    {rel}: {cur['n_files']} files match")
    if drift:
        print(f"[kg_cache_manifest] {drift} tree(s) drifted from the manifest.", file=sys.stderr)
        return 1
    print("[kg_cache_manifest] all recorded caches match on disk.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="(re)compute and write the manifest block")
    g.add_argument("--check", action="store_true", help="verify on-disk caches match the manifest")
    args = ap.parse_args()
    if args.write:
        write_manifest()
    else:
        sys.exit(check_manifest())


if __name__ == "__main__":
    main()
