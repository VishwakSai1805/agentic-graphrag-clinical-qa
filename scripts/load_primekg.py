"""Downloads PrimeKG and loads its nodes + allowlisted edges into Neo4j 5.

Source verification (see the ADR this phase adds to docs/05_DECISIONS.md for
the full transcript -- summarized here, not fabricated):
  PrimeKG's authors publish it themselves on Harvard Dataverse:
  https://doi.org/10.7910/DVN/IXA7BM (Chandak, Huang, Zitnik, "PrimeKG",
  Harvard Dataverse V2, CC0 1.0). Confirmed via the Dataverse search API
  (https://dataverse.harvard.edu/api/search?q=PrimeKG) and the dataset's own
  metadata API, then verified live with real HTTP GET requests (not just a
  HEAD) against the two files this script downloads:
    - nodes.csv (Dataverse file id 6180617): downloaded with
      `?format=original` (Dataverse re-ingests uploaded CSVs into an internal
      tab-delimited "ingest" format by default; `format=original` recovers
      the authors' original bytes). Locally verified: MD5
      7f9ab4109c54049e819ecd14e15a6038, matching the checksum the Dataverse
      API itself reports for this file's original upload -- an exact match,
      not an assumption.
    - kg.csv (Dataverse file id 6180620, ~982MB, README.txt on the same
      record: "This is the Precision Medicine knowledge graph. Primary key:
      (x_index, y_index)"): downloaded as-is (never tab-converted by
      Dataverse, per its own metadata). Locally verified: MD5
      aac8191d4fbc5bf09cdf8c3c78b4e75f, matching the Dataverse-reported
      checksum for this file.
  No official PrimeKG mirror on Hugging Face was found (a HF dataset search
  for "primekg" turns up only unrelated third-party derivative datasets, not
  a mirror of the authors' own release) -- the Dataverse record above is
  used directly.

kg.csv columns: relation, display_relation, x_index, x_id, x_type, x_name,
x_source, y_index, y_id, y_type, y_name, y_source. Only rows whose `relation`
is in KGConfig.edge_allowlist (configs/base.yaml: kg.edge_allowlist, from
docs/03_DATASETS.md) are loaded -- this is a config value, not a hardcoded
list in this script.

Neo4j schema and all Cypher: src/kg/queries.py.

No torch/CUDA anywhere in this script -- loading CSV rows into Neo4j is not a
GPU task (CLAUDE.md's unconditional CUDA-status-print rule is scoped to
GPU-relevant scripts; this one has no such path).

Usage:
    uv run python -m scripts.load_primekg [--force-download] [--force-load]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

load_dotenv()  # NEO4J_PASSWORD etc. from a local .env, same convention as src/llm/client.py

from src.config import load_config  # noqa: E402
from src.kg import queries  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
MANIFEST_PATH = REPO_ROOT / "data" / "MANIFEST.json"
RAW_DIR = REPO_ROOT / "data" / "raw" / "primekg"

DATAVERSE_DOI = "doi:10.7910/DVN/IXA7BM"
DATAVERSE_SOURCE_URL = "https://doi.org/10.7910/DVN/IXA7BM"
NODES_FILE_ID = 6180617
NODES_URL = f"https://dataverse.harvard.edu/api/access/datafile/{NODES_FILE_ID}?format=original"
NODES_MD5_EXPECTED = "7f9ab4109c54049e819ecd14e15a6038"
KG_FILE_ID = 6180620
KG_URL = f"https://dataverse.harvard.edu/api/access/datafile/{KG_FILE_ID}"
KG_MD5_EXPECTED = "aac8191d4fbc5bf09cdf8c3c78b4e75f"

NODES_PATH = RAW_DIR / "nodes.csv"
KG_PATH = RAW_DIR / "kg.csv"

NODE_BATCH_SIZE = 5000
EDGE_BATCH_SIZE = 5000


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, expected_md5: str, *, force: bool) -> str:
    if dest.exists() and not force:
        digest = _md5(dest)
        if digest == expected_md5:
            print(f"[load_primekg] {dest} already present and MD5-verified, skipping download.")
            return digest
        print(
            f"[load_primekg] {dest} exists but MD5 mismatch "
            f"({digest} != {expected_md5}), redownloading."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[load_primekg] downloading {url} -> {dest} ...")
    t0 = time.time()
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        n_bytes = 0
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                n_bytes += len(chunk)
    print(f"[load_primekg] downloaded {n_bytes} bytes in {time.time() - t0:.1f}s")

    digest = _md5(dest)
    if digest != expected_md5:
        raise RuntimeError(
            f"MD5 mismatch for {dest}: got {digest}, expected {expected_md5}. "
            "Not proceeding with a load from an unverified file."
        )
    print(f"[load_primekg] MD5 verified: {digest}")
    return digest


def iter_node_rows(nodes_path: Path):
    with nodes_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {
                "id": row["node_index"].strip(),
                "node_type": row["node_type"].strip(),
                "name": row["node_name"].strip(),
                "source": row["node_source"].strip(),
            }


def iter_allowlisted_edge_rows(kg_path: Path, edge_allowlist: set[str]):
    with kg_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            relation = row["relation"].strip()
            if relation not in edge_allowlist:
                continue
            yield {
                "x_id": row["x_index"].strip(),
                "y_id": row["y_index"].strip(),
                "relation": relation,
                "display_relation": row["display_relation"].strip(),
            }


def _batched(iterable, size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_into_neo4j(
    *,
    uri: str,
    user: str,
    password: str,
    nodes_path: Path,
    kg_path: Path,
    edge_allowlist: list[str],
    force_load: bool,
) -> dict:
    from neo4j import GraphDatabase

    allowlist_set = set(edge_allowlist)
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            session.run(queries.CONSTRAINT_ENTITY_ID)

            if force_load:
                print("[load_primekg] --force-load: clearing existing :Entity/:KG_EDGE data...")
                session.run("MATCH (n:Entity) DETACH DELETE n")

            print(f"[load_primekg] loading nodes from {nodes_path} ...")
            t0 = time.time()
            n_nodes = 0
            for batch in _batched(iter_node_rows(nodes_path), NODE_BATCH_SIZE):
                session.run(queries.UPSERT_NODES, queries.upsert_nodes_params(batch))
                n_nodes += len(batch)
                if n_nodes % (NODE_BATCH_SIZE * 5) == 0:
                    print(f"[load_primekg] ... {n_nodes} nodes upserted", flush=True)
            print(f"[load_primekg] {n_nodes} nodes upserted in {time.time() - t0:.1f}s")

            print(
                f"[load_primekg] loading edges from {kg_path} filtered to "
                f"allowlist {sorted(allowlist_set)} ..."
            )
            t0 = time.time()
            n_edges = 0
            relation_counts_seen: Counter[str] = Counter()
            edge_row_iter = iter_allowlisted_edge_rows(kg_path, allowlist_set)
            for batch in _batched(edge_row_iter, EDGE_BATCH_SIZE):
                session.run(queries.UPSERT_EDGES, queries.upsert_edges_params(batch))
                n_edges += len(batch)
                for row in batch:
                    relation_counts_seen[row["relation"]] += 1
                if n_edges % (EDGE_BATCH_SIZE * 10) == 0:
                    print(f"[load_primekg] ... {n_edges} allowlisted edges upserted", flush=True)
            print(f"[load_primekg] {n_edges} allowlisted edges upserted in {time.time() - t0:.1f}s")

            node_count_record = session.run(queries.COUNT_NODES).single()
            edge_count_record = session.run(queries.COUNT_EDGES).single()
            assert node_count_record is not None and edge_count_record is not None
            real_node_count = node_count_record["n"]
            real_edge_count = edge_count_record["n"]
            real_edges_by_relation = {
                r["relation"]: r["n"] for r in session.run(queries.COUNT_EDGES_BY_RELATION)
            }
            real_nodes_by_type = {
                r["node_type"]: r["n"] for r in session.run(queries.COUNT_NODES_BY_TYPE)
            }
    finally:
        driver.close()

    return {
        "n_node_rows_seen_in_csv": n_nodes,
        "n_edge_rows_seen_in_csv_after_allowlist": n_edges,
        "edge_rows_seen_by_relation": dict(relation_counts_seen),
        "real_node_count_in_neo4j": real_node_count,
        "real_edge_count_in_neo4j": real_edge_count,
        "real_edges_by_relation_in_neo4j": real_edges_by_relation,
        "real_nodes_by_type_in_neo4j": real_nodes_by_type,
    }


def update_manifest(manifest_path: Path, record: dict) -> None:
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["primekg"] = record
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-download", action="store_true", help="redownload even if MD5-verified copies exist"
    )
    parser.add_argument(
        "--force-load", action="store_true", help="clear and reload Neo4j even if data exists"
    )
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    kg_cfg = cfg.kg

    password = os.environ.get(kg_cfg.neo4j_password_env)
    if not password:
        print(
            f"[load_primekg] ERROR: env var {kg_cfg.neo4j_password_env} is not set. "
            "Set it (see .env.example: NEO4J_PASSWORD) before loading -- refusing to "
            "guess or fall back to a default password.",
            file=sys.stderr,
        )
        sys.exit(1)

    nodes_md5 = _download(NODES_URL, NODES_PATH, NODES_MD5_EXPECTED, force=args.force_download)
    kg_md5 = _download(KG_URL, KG_PATH, KG_MD5_EXPECTED, force=args.force_download)

    print(f"[load_primekg] loading into Neo4j at {kg_cfg.neo4j_uri} ...")
    load_stats = load_into_neo4j(
        uri=kg_cfg.neo4j_uri,
        user=kg_cfg.neo4j_user,
        password=password,
        nodes_path=NODES_PATH,
        kg_path=KG_PATH,
        edge_allowlist=kg_cfg.edge_allowlist,
        force_load=args.force_load,
    )

    record = {
        "source": {
            "dataverse_doi": DATAVERSE_DOI,
            "dataverse_url": DATAVERSE_SOURCE_URL,
            "citation": "Chandak, Payal; Huang, Kexin; Zitnik, Marinka (2022), "
            '"PrimeKG", Harvard Dataverse, V2, doi:10.7910/DVN/IXA7BM',
            "license": "CC0 1.0",
            "files": {
                "nodes.csv": {
                    "dataverse_file_id": NODES_FILE_ID,
                    "url": NODES_URL,
                    "md5": nodes_md5,
                },
                "kg.csv": {"dataverse_file_id": KG_FILE_ID, "url": KG_URL, "md5": kg_md5},
            },
        },
        "edge_allowlist": kg_cfg.edge_allowlist,
        "neo4j_uri": kg_cfg.neo4j_uri,
        **load_stats,
        "loaded_at_utc": datetime.now(UTC).isoformat(),
    }
    update_manifest(MANIFEST_PATH, record)
    print(f"[load_primekg] merged 'primekg' record into {MANIFEST_PATH}")
    print(f"[load_primekg] real counts in Neo4j: {load_stats['real_node_count_in_neo4j']} nodes, "
          f"{load_stats['real_edge_count_in_neo4j']} edges "
          f"({load_stats['real_edges_by_relation_in_neo4j']})")


if __name__ == "__main__":
    main()
