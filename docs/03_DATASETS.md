# 03 — Datasets & Corpora

## Benchmarks

| Dataset | HF id | Task | Split used | Role |
|---|---|---|---|---|
| PubMedQA | `qiaojin/PubMedQA` (`pqa_labeled`, 1k) | yes/no/maybe over abstract | 500 test (standard), 500 dev | Baseline parity vs. MEGA-RAG-class binary systems |
| MedQA-USMLE | `GBaker/MedQA-USMLE-4-options` | 4-option MCQ | official test; 500 from train as dev | **Primary novelty benchmark** — multi-option clinical reasoning |
| MedMCQA | `openlifescienceai/medmcqa` | 4-option MCQ, 21 subjects | 500-question fixed seeded subsample of official validation (test labels withheld) | Diagnostic breadth across specialties |

Notes:
- **Calibration weights are fitted on the `pqa_labeled` DEV half (500 items),
  not on `pqa_artificial`** — corrected 2026-09-07, ADR-026 (AUDIT 2026-09-06
  §3.3). This doc previously reserved `pqa_artificial` for calibration
  fitting, but `configs/base.yaml` *also* builds the retrieval corpus from a
  20,000-doc `pqa_artificial` subset (`data/MANIFEST.json: n_pubmed_docs =
  20000`), so the calibration items' own source abstracts were indexed and
  retrievable while the temperature was fitted on them. `pqa_labeled` draws
  from a pubid pool disjoint from `pqa_artificial`; that disjointness is
  *executed*, not assumed, by
  `src.eval.contamination.assert_calibration_fit_set_uncontaminated`, which
  Phase 5 must call before fitting (verified 2026-09-07 against the built
  corpus: 0 of the 500 fit PMIDs are indexed).
- `pqa_artificial` (211k) is a **retrieval-corpus source only**, never a
  reporting set and no longer a fitting set.
- The `pqa_labeled` TEST half (500) remains the reported set and stays
  blocklisted from the corpus (`blocklisted_pmids("pubmedqa")`). Test and dev
  come from the same `pubmedqa_split()` call, so the fit set is exactly the
  complement of the reported set.
- MedMCQA test labels are not public → report on validation and say so.
- MedMCQA's full validation split (~4.2k questions) is cut to a 500-question
  fixed seeded subsample (seed 42) for scale parity with PubMedQA and MedQA
  under the Phase 7 grid budget — see `docs/04_EVALUATION.md` §0.3 and
  ADR-005. Report per-subject accuracy *within that subsample*, and note the
  subsample size explicitly wherever breadth is claimed in the paper — it is
  not the full validation set.
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

- **PrimeKG** — 129k nodes / ~4M edges (full graph, all relation types), 10
  node types (disease, drug, gene, phenotype, pathway…), harmonised from 20
  sources. Loaded into Neo4j 5. This project only loads a 7-item
  edge-type-allowlisted subset, not the full ~4M — see ADR-014
  (`docs/05_DECISIONS.md`) for the real raw `kg.csv` row count (8,100,498
  directional rows, i.e. ~4.05M edges since PrimeKG stores each undirected
  fact as two rows) versus the 740,674-row post-allowlist count actually in
  Neo4j; 740,674 is a deliberate filter result, not an incomplete load.
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
