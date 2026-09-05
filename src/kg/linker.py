"""Entity linking: scispaCy `en_core_sci_lg` + `UmlsEntityLinker`, plus a
CUI -> PrimeKG node id crosswalk.

CPU-only module: scispaCy's `en_core_sci_lg` pipeline and the UMLS candidate
generator both run on CPU here (no `torch`/CUDA path in this file at all --
there is nothing to gate on `torch.cuda.is_available()`, unlike
src/retrieval/dense.py). If GPU support is ever wired in (spaCy can use a
transformer-backed pipeline on GPU via `spacy.require_gpu()`), that must add
the same unconditional CUDA-status print CLAUDE.md requires of GPU-relevant
scripts -- it is deliberately absent now because this path is CPU-only.

CUI -> PrimeKG crosswalk, honestly partial (see docs/03_DATASETS.md "Entity
linking" and the ADR this phase adds):
  - disease (PrimeKG node_type "disease", source MONDO/MONDO_grouped): exact
    via MONDO's own published UMLS cross-reference table
    (mondo_exactmatch_umls.sssom.tsv, monarch-initiative/mondo on GitHub --
    real ontology xrefs, curated, not guessed).
  - effect/phenotype (source HPO): exact via HPO's own `hp.obo` inline
    `xref: UMLS:Cxxxxxxx` annotations (obophenotype/human-phenotype-ontology).
  - drug (DrugBank) and gene/protein (NCBI Gene): NO open, free, unlicensed
    CUI crosswalk exists for these without a full licensed UMLS Metathesaurus
    install (RxNorm/DrugBank/NCBI-gene cross-references live inside
    MRCONSO.RRF, which requires a UMLS license this project does not hold).
    Falls back to normalized-name string matching against PrimeKG node names,
    scoped to the node type(s) implied by the candidate's UMLS semantic types
    (TUIs). This is a real limitation, not a shortcut hidden from the paper --
    every non-id-xref hit is tagged `match_method="name_match"` and every miss
    is logged so the KG-hit-rate gap is attributable.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import requests
import spacy
from scispacy.linking import EntityLinker  # noqa: F401 -- registers "scispacy_linker"

from src.schema import LinkedEntity

logger = logging.getLogger(__name__)

MODEL_NAME = "en_core_sci_lg"

MONDO_UMLS_XREF_URL = (
    "https://raw.githubusercontent.com/monarch-initiative/mondo/master/"
    "src/ontology/mappings/mondo_exactmatch_umls.sssom.tsv"
)
HPO_OBO_URL = (
    "https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/"
    "master/hp.obo"
)

# UMLS semantic type (TUI) -> PrimeKG node_type bucket, used only to scope the
# name-match fallback search space for drug/gene hits (and as a second-chance
# route for disease/phenotype misses on the id-xref path). Coarse and
# intentionally not exhaustive -- narrowing the search space wrong just means
# a fallback miss, not a wrong answer, because name_match still requires an
# exact normalized-name collision.
TUI_TO_NODE_TYPE: dict[str, str] = {
    "T047": "disease",  # Disease or Syndrome
    "T191": "disease",  # Neoplastic Process
    "T048": "disease",  # Mental or Behavioral Dysfunction
    "T019": "disease",  # Congenital Abnormality
    "T046": "disease",  # Pathologic Function
    "T037": "disease",  # Injury or Poisoning
    "T121": "drug",  # Pharmacologic Substance
    "T195": "drug",  # Antibiotic
    "T200": "drug",  # Clinical Drug
    "T109": "drug",  # Organic Chemical (many drugs are typed this way too)
    "T184": "effect/phenotype",  # Sign or Symptom
    "T033": "effect/phenotype",  # Finding
    "T028": "gene/protein",  # Gene or Genome
    "T116": "gene/protein",  # Amino Acid, Peptide, or Protein
    "T192": "gene/protein",  # Receptor
    "T126": "gene/protein",  # Enzyme
}


def _normalize_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def _mondo_curie_from_primekg_id(node_id: str) -> list[str]:
    """PrimeKG disease node_id is either a bare numeric MONDO local id
    ("8019") or, for MONDO_grouped nodes, several underscore-joined local ids
    ("13924_12592_..."). MONDO CURIEs are "MONDO:" + the local id zero-padded
    to 7 digits -- verified directly against mondo_exactmatch_umls.sssom.tsv
    (e.g. PrimeKG id "8019" -> "MONDO:0008019", which is exactly the row
    labelled "mullerian aplasia and hyperandrogenism" in that file)."""
    curies = []
    for part in node_id.split("_"):
        part = part.strip()
        if part.isdigit():
            curies.append(f"MONDO:{int(part):07d}")
    return curies


def _hpo_curie_from_primekg_id(node_id: str) -> str | None:
    if node_id.strip().isdigit():
        return f"HP:{int(node_id):07d}"
    return None


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading crosswalk source %s -> %s", url, dest)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


class PrimeKGCrosswalk:
    """CUI -> PrimeKG `Entity.id` (node_index) lookup, built from the raw
    PrimeKG node table (scripts/load_primekg.py's download) plus MONDO/HPO's
    own UMLS cross-reference files. Construction does real file I/O (and, on
    first use, real network downloads of the two small ontology xref files)
    but never opens a Neo4j connection -- this class only reads flat files.
    """

    def __init__(
        self,
        nodes_path: Path,
        crosswalk_dir: Path,
        *,
        download_if_missing: bool = True,
    ) -> None:
        self.nodes_path = Path(nodes_path)
        self.crosswalk_dir = Path(crosswalk_dir)
        self._by_type_name: dict[str, dict[str, tuple[str, str]]] = {}
        self._mondo_to_node: dict[str, str] = {}
        self._hpo_to_node: dict[str, str] = {}
        self._cui_to_mondo: dict[str, list[str]] = {}
        self._cui_to_hpo: dict[str, list[str]] = {}

        self._load_nodes()
        self._load_mondo_xref(download_if_missing=download_if_missing)
        self._load_hpo_xref(download_if_missing=download_if_missing)

    def _load_nodes(self) -> None:
        import csv

        if not self.nodes_path.exists():
            raise FileNotFoundError(
                f"PrimeKG node table not found at {self.nodes_path} -- run "
                "scripts/load_primekg.py first."
            )
        with self.nodes_path.open("r", encoding="utf-8", newline="") as f:
            # original comma-delimited nodes.csv, see scripts/load_primekg.py
            reader = csv.DictReader(f)
            for row in reader:
                node_index = row["node_index"].strip()
                node_id = row["node_id"].strip().strip('"')
                node_type = row["node_type"].strip().strip('"')
                node_name = row["node_name"].strip().strip('"')

                norm = _normalize_name(node_name)
                self._by_type_name.setdefault(node_type, {}).setdefault(
                    norm, (node_index, node_name)
                )

                if node_type == "disease":
                    for curie in _mondo_curie_from_primekg_id(node_id):
                        self._mondo_to_node.setdefault(curie, node_index)
                elif node_type == "effect/phenotype":
                    hpo_curie = _hpo_curie_from_primekg_id(node_id)
                    if hpo_curie:
                        self._hpo_to_node.setdefault(hpo_curie, node_index)

    def _load_mondo_xref(self, *, download_if_missing: bool) -> None:
        path = self.crosswalk_dir / "mondo_exactmatch_umls.sssom.tsv"
        if not path.exists():
            if not download_if_missing:
                logger.warning("mondo xref file missing at %s, skipping disease id-xref", path)
                return
            _download(MONDO_UMLS_XREF_URL, path)
        with path.open("r", encoding="utf-8") as f:
            header = None
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                fields = line.rstrip("\n").split("\t")
                if header is None:
                    header = fields
                    continue
                row = dict(zip(header, fields, strict=False))
                subject_id = row.get("subject_id", "")
                object_id = row.get("object_id", "")
                if subject_id.startswith("MONDO:") and object_id.startswith("UMLS:"):
                    cui = object_id.split(":", 1)[1]
                    self._cui_to_mondo.setdefault(cui, []).append(subject_id)

    def _load_hpo_xref(self, *, download_if_missing: bool) -> None:
        path = self.crosswalk_dir / "hp.obo"
        if not path.exists():
            if not download_if_missing:
                logger.warning("hp.obo missing at %s, skipping phenotype id-xref", path)
                return
            _download(HPO_OBO_URL, path)
        current_id: str | None = None
        xref_re = re.compile(r"^xref:\s*UMLS:(C\d+)")
        id_re = re.compile(r"^id:\s*(HP:\d+)")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "[Term]":
                    current_id = None
                    continue
                m_id = id_re.match(line)
                if m_id:
                    current_id = m_id.group(1)
                    continue
                m_xref = xref_re.match(line)
                if m_xref and current_id:
                    self._cui_to_hpo.setdefault(m_xref.group(1), []).append(current_id)

    def lookup(
        self, cui: str, canonical_name: str, tuis: list[str]
    ) -> tuple[str | None, str | None, str | None]:
        """Returns (kg_node_id, kg_node_type, match_method) or (None, None, None)."""
        for mondo_curie in self._cui_to_mondo.get(cui, []):
            node_id = self._mondo_to_node.get(mondo_curie)
            if node_id:
                return node_id, "disease", "mondo_xref"
        for hpo_curie in self._cui_to_hpo.get(cui, []):
            node_id = self._hpo_to_node.get(hpo_curie)
            if node_id:
                return node_id, "effect/phenotype", "hpo_xref"

        candidate_types = [TUI_TO_NODE_TYPE[t] for t in tuis if t in TUI_TO_NODE_TYPE]
        candidate_types = candidate_types or list(self._by_type_name)
        norm = _normalize_name(canonical_name)
        for node_type in candidate_types:
            hit = self._by_type_name.get(node_type, {}).get(norm)
            if hit:
                return hit[0], node_type, "name_match"
        return None, None, None


_nlp: spacy.language.Language | None = None


def _get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        logger.info("loading %s + scispacy UMLS linker (CPU-only)", MODEL_NAME)
        nlp = spacy.load(MODEL_NAME)
        nlp.add_pipe(
            "scispacy_linker",
            config={"resolve_abbreviations": True, "linker_name": "umls"},
        )
        _nlp = nlp
    return _nlp


def link_entities(
    text: str,
    threshold: float = 0.85,
    crosswalk: PrimeKGCrosswalk | None = None,
) -> list[LinkedEntity]:
    """Runs scispaCy NER + UMLS linking over `text`, keeping only candidates
    with UMLS linker score >= threshold (default 0.85, tuned on 50
    hand-checked items -- docs/03_DATASETS.md). If `crosswalk` is given, each
    kept entity is also mapped to a PrimeKG node id (see PrimeKGCrosswalk);
    unmapped CUIs are logged at INFO, never silently dropped.
    """
    nlp = _get_nlp()
    linker = nlp.get_pipe("scispacy_linker")
    doc = nlp(text)

    results: list[LinkedEntity] = []
    seen: set[tuple[str, str]] = set()
    for ent in doc.ents:
        for cui, score in ent._.kb_ents:
            if score < threshold:
                continue
            key = (ent.text, cui)
            if key in seen:
                continue
            seen.add(key)

            kg_node_id = kg_node_type = match_method = None
            if crosswalk is not None:
                umls_ent = linker.kb.cui_to_entity[cui]  # type: ignore[attr-defined]
                kg_node_id, kg_node_type, match_method = crosswalk.lookup(
                    cui, umls_ent.canonical_name, list(umls_ent.types)
                )
                if kg_node_id is None:
                    logger.info(
                        "kg_unmapped_cui cui=%s text=%r canonical_name=%r tuis=%s",
                        cui,
                        ent.text,
                        umls_ent.canonical_name,
                        umls_ent.types,
                    )

            results.append(
                LinkedEntity(
                    text=ent.text,
                    cui=cui,
                    score=float(score),
                    kg_node_id=kg_node_id,
                    kg_node_type=kg_node_type,
                    match_method=match_method,
                )
            )
    return results
