# 03 — Datasets & Corpora

## Benchmarks

| Dataset | HF id | Task | Split used | Role |
|---|---|---|---|---|
| PubMedQA | `qiaojin/PubMedQA` (`pqa_labeled`, 1k) | yes/no/maybe over abstract | 500 test (standard), 500 dev | Baseline parity vs. MEGA-RAG-class binary systems |
| MedQA-USMLE | `GBaker/MedQA-USMLE-4-options` | 4-option MCQ | official test; 500 from train as dev | **Primary novelty benchmark** — multi-option clinical reasoning |
| MedMCQA | `openlifescienceai/medmcqa` | 4-option MCQ, 21 subjects | official validation (test labels withheld) | Diagnostic breadth across specialties |

Notes:
- PubMedQA `pqa_artificial` (211k) may be used **only** to fit calibration
  weights, never for reporting.
- MedMCQA test labels are not public → report on validation and say so.
- Record per-subject accuracy for MedMCQA (it is the breadth claim).

## Retrieval corpora

| Corpus | Source | Use |
|---|---|---|
| PubMed abstracts subset | MedCPT/BioASQ dumps or PubMed baseline | dense + BM25 index |
| MedQA textbooks (18 books) | shipped with MedQA | dense + BM25, essential for USMLE-style items |
| StatPearls | open access | optional breadth |

**Contamination rule:** when evaluating PubMedQA, the abstract that *is* the
question's context must not also be retrievable as an independent corpus doc.
Build a blocklist of test PMIDs and assert it at index-build time.

## Knowledge graph

- **PrimeKG** — 129k nodes / 4M edges, 10 node types (disease, drug, gene,
  phenotype, pathway…), harmonised from 20 sources. Loaded into Neo4j 5.
- Edge-type allowlist for traversal (start with): `indication`, `contraindication`,
  `off_label_use`, `disease_phenotype_positive`, `disease_protein`,
  `drug_effect`, `disease_disease`.
- Degree cap 200 per node to avoid hub explosion (e.g. `aspirin`).
- Optional later: UMLS RRF subset for synonymy; only if licence is in place.

## Entity linking

scispaCy `en_core_sci_lg` + `UmlsEntityLinker` (threshold 0.85, tuned on 50
hand-checked items — record the tuning in `docs/05_DECISIONS.md`). Map CUI →
PrimeKG node id via PrimeKG's own crosswalk; log unmapped CUIs, they explain
KG-hit-rate gaps.

## Storage

Everything under `data/` is gitignored. `scripts/fetch_datasets.py` must be able
to rebuild `data/` from scratch on a clean machine. Record dataset revision
hashes in `data/MANIFEST.json`.
