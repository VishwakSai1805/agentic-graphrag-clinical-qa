# Next session — start here

**Done:** Phases 0–6, all committed to `master`. Phase 6 harness `ca8f8df`
(sha recorded `79ee599`); pre-Phase-7 fixes `82ea3d9` (hallucination control),
`cf3e757` (BM25), `26141cd` (B3 prompt-sensitivity + §6.8 bootstrap index),
`f469618` (§6.5 KG cache stem-binding). 459 tests pass, ruff clean.

**The MedQA grid can launch.** Every pre-Phase-7 gate is closed (see below).
Command: `python -m src.eval.harness --config <cfg> --dataset medqa --split test
--limit 500 --variant <V>` per variant — sequential, no concurrency layer
(cut, ADR-029). Cut grid (`docs/SCHEDULE.md`): **V1, V3, V5, V4−audit,
V4−conflict on MedQA test-500, seed 42.** PubMedQA/MedMCQA are cut.

---

## Phase 6 (ADR-029) — minimal harness, run → score → write

- `src/eval/harness.py` — `run_eval(config, dataset, split, limit, variant, …)`,
  sequential. `VARIANTS` table is the one source of truth for V0–V5:
  **V2 = `{dense,lexical}` has never been run; committed `medqa_B2_*` = V3.**
- `src/eval/nli.py` — `split_claims` (pure regex, no LLM) + `score_hallucination`
  vs *cited* evidence, **both denominators** (predicted option = primary).
- `src/eval/metrics.py` — `precision_at_k` / `reciprocal_rank_option` / `mean_*`.
- `scripts/report.py` — `results/*/metrics.json` → `paper/tables/results_grid.{md,tex}`,
  superseded skipped, every cell a value or `TBD`/`--`, idempotent.
- **CUT:** concurrency semaphore, resumability, the 1/2/4 benchmark, RAGAS.
- Exit test passed on the dev-20: `results/medqa_V5_aa73199-dirty_20260907T170604Z/`
  (acc 0.30, P@1 0.30, P@2 0.55, MRR 0.57, ECE 0.080, halluc 0.68 predicted /
  0.72 all-cells, `abstention_rate` 1.00 — every calibrated p1 below
  `configs/v5.yaml: abstention_tau`, a real Phase 5 near-degenerate-fit
  property, ADR-028).

## Pre-Phase-7 fixes — ALL CLOSED

| Item | Outcome |
|---|---|
| **§6.7 multiplicity** (ADR-030) | Pre-registered in `docs/04` §1.0 **before any grid number**: confirmatory = **V3 vs V5, accuracy, α=0.05 two-sided**; a null is the reported finding; all else exploratory, Holm across a fixed family of 30. |
| **§4.1 B3 prompt-sensitivity** (ADR-033) | B3 `verifier_decision_rate` = 0.05 (strict) / 0.30 (neutral) / 0.35 (permissive) on the dev-20. **"1/20 = 0.05" is retired** — it was a strict-prompt artifact. §5.4 + `docs/00` §3 C2 now quote the range; C2's audit-discrimination claim leans on the 108 audit cells + the Phase 7 V4−audit cell, not on B3. Shipped `_VERIFY_PROMPT` unchanged (no prompt selection). |
| **§4.3 equal-budget B2** | Run. B2 at 12-item budget = 0.60 vs unrestricted 0.65 — inside ADR-021's ±0.05 floor, so budget asymmetry does **not** explain B2 > agents. §5.5 Finding 3 rewritten: **V4 0.35 vs B2-equal-budget 0.60 is now unconfounded** — the agent layer underperforms plain hybrid RAG at equal evidence, reported as a finding. |
| **§4.6 B3 row** | Annotated: B3's answer vector = B2's on 19/20 by construction (`llm_cache_hits: 19`). |
| **BM25 `k1`/`b`** (ADR-032) | 8×6 grid on NFCorpus. Default (1.5, 0.75) kept — best cell's +0.0064 nDCG@10 is below the 0.01 floor and BM25 is flat across the whole grid. A checked non-change. |
| **§6.8 `paired_bootstrap_ci`** | p-value already added (ADR-027). Upper index now symmetric (`n-1-lo_idx`). Shift ≤1 sample, flips no verdict. |
| **§6.5 KG cache id-binding** (ADR-034) | `KGCacheEntry.question_stem` asserted on read via `get_subgraph_cached(expected_stem=)`, threaded through the pipeline. All 3633 existing cache files backfilled + integrity-checked → **0 drift**. `data/MANIFEST.json` kg_cache block refreshed (was stale: 220 vs 1123 medqa files). |
| **§1.1/§1.2, §5.4 PubMedQA wiring, §3.5, §1.10, §1.14, §6.10** | Done (overnight run + this session). |
| **§1.5 hallucination control** (ADR-031) | `hallucination_rate` at NLI 0.5 is **metric-strict**: B2 0.741 vs V5 0.682, both 0 uncited → it is a *relative* cross-variant comparison, never an absolute "X% hallucinated". Phase 7 adds a 0.3/0.5/0.7 threshold sweep on V3/V5. |
| **§4.2 PubMedQA degenerate predictor** (ADR-030) | Word-framed prompt tried (3 versions); all still degenerate at 7B scale — a **model-capability ceiling**, disclosed. PubMedQA stays cut from the grid. |

## Still open (NOT blocking the grid)

- **Docs-half MINOR sweep** (§1.11–1.13, §2.7–2.9, §3.6, §4.4, §4.5, §6.9) —
  cosmetic doc/number corrections + `models_used` in call-stats (§4.5, small
  code). §3.6 is effectively done (`medqa_overlap_check` runs). §4.4 is
  addressed by the harness `VARIANTS` table + `write_run`'s recorded `variant`.
- **§2.2** — ADR-024 `Status: accepted` while asserting `conflict_fire_rate =
  0.55` numbers ADR-025 retracts. One-line status fix.
- **KG cache per-phase provenance** (ADR-034) — the manifest checksums are
  fresh but the `regen`/`note` for the post-Phase-2 cache additions
  (train-500 etc.) is not fully reconstructed. Needs a Neo4j-up content
  re-verify, separate from the grid.
- **C2 reframing** — decide (before writing the C2 paper section) whether C2
  is claimed around conflict *resolution* or audit *discrimination*; the
  Phase 7 V4−audit / V4−conflict cells are the evidence, run them first.
- **§0.6 Gemini Flash sensitivity check** — the model-strength axis of the B3
  robustness check + the V3-vs-V5 200-question robustness row.

---

## Where the contributions stand (`docs/00_RESEARCH_SPEC.md` §3 is authoritative)

- **C1** — diagnostic/negative, pending Phase 7 V2-vs-V3. 1/108 (dev-20) and
  12/626 (120-slice) audit cells cite KG; gamma unidentifiable (ADR-028).
- **C2** — audit-discrimination supported; the B3 "1/20" prop is **retired**
  (ADR-033, prompt-fragile); conflict *resolution* unverified
  (`conflict_fire_rate` ≈ 0 every time measured). C2's real evidence is the
  Phase 7 leave-one-agent-out cells.
- **C3** — restated (ADR-028): temperature scaling alone matches the evidential
  features' ECE at n=60; features' added value not established; calibrated
  rank-1 (0.267) below the uncalibrated rule (0.35). Phase 7 grid at n=500
  re-tests.
- **C4** — pending Phase 7 (MedQA only per SCHEDULE).
- **C5** — SUPPORTED: the KG-hit-rate arithmetic artifact.

---

## Delivery schedule

`docs/SCHEDULE.md`, hard deadline Sat 2026-09-12. Wed 9 = launch the grid
overnight. If it slips past Wed, cut n to 300, never a cell.

## Standing rules

- A result that undercuts a contribution is reported, not engineered around
  (ADR-012/018/025/027/028/031/033).
- Every A-vs-B numeric claim gets a paired-bootstrap CI, never a point estimate
  alone — including ECE (`paired_bootstrap_ece_diff`).
- The confirmatory test is **V3 vs V5, accuracy, two-sided** — fixed in
  `docs/04` §1.0, not re-openable after grid numbers exist. A null there is a
  reported finding.
- Fitting set and evaluation set disjoint, always.
- Never select a hyper-parameter / threshold / ablation decision by reading the
  split Phase 7 reports on (ADR-027).
- `hallucination_rate` is a relative comparison at a fixed NLI threshold, never
  an absolute claim (ADR-031).
