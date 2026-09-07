# 02 — Roadmap

Execute in order. One phase per Claude Code session where possible. Each phase
has an **exit test** — the phase is not done until that command passes and its
output is real.

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done (add run_id/commit)

---

## Phase 0 — Scaffold & environment  `[x]` (8d457e5, 2026-09-04)
- `pyproject.toml` (uv), `Makefile`, `.gitignore`, `.env.example`
- `src/llm/client.py`: LiteLLM gateway + on-disk cache + token/cost accounting
- `configs/base.yaml`; config loader with hash
- `tests/test_llm_cache.py` (mocked)
- **Exit test:** `pytest -q` green; `python -c "from src.llm.client import chat; print(chat('ping'))"` returns and the second call is a cache hit.
- Deviation from the original phase text, see ADR-004: `client.py` is an
  ordered fallback chain (Ollama primary, Groq/Gemini free-tier overflow)
  rather than a single provider, to satisfy the zero-API-spend constraint.
  Exit test verified live against Ollama alone, no API key present.

## Phase 1 — Corpora & lexical/dense indexes  `[x]` (74457b0, 2026-09-05)
- `scripts/fetch_datasets.py` — PubMedQA (pqa_labeled/pqa_artificial), MedQA-USMLE 4-opt, MedMCQA via HF
  `datasets`; MedQA textbook corpus (18 USMLE textbooks) via the HF mirror `MedRAG/textbooks` — this is
  the primary content source for USMLE-style questions, not a fast-follow (see ADR-009). If no HF
  mirror and the jind11/MedQA GitHub→Google Drive fallback both fail, the script halts rather than
  proceeding on an abstracts-only corpus.
- `scripts/build_corpus.py` — chunk (≈256 tok, 32 overlap), dedupe, `data/processed/corpus.jsonl`,
  `data/MANIFEST.json` (dataset revisions + corpus doc/chunk counts). Real build: 20,000 pubmed-abstract
  docs (fixed-seed subset of `pqa_artificial`) + 18 full textbooks → 113,465 chunks.
- `scripts/build_nfcorpus.py` — separate small NFCorpus (BEIR) index under `indexes/nfcorpus/`, the
  fusion-quality sanity-check corpus (see ADR-012) — not part of the project's actual retrieval corpus.
- `src/retrieval/dense.py` — MedCPT **dual encoder**: `MedCPT-Query-Encoder` for queries,
  `MedCPT-Article-Encoder` for indexed documents (see ADR-010, correcting CLAUDE.md §3) + FAISS;
  `lexical.py` (BM25), `fusion.py` (RRF), `rerank.py` (MedCPT cross-encoder), `metrics.py` (Recall@k,
  MRR, nDCG@k, `paired_bootstrap_ci` — all pure, tested functions).
- `src/eval/contamination.py` — assert no test-item abstract in the index (passed on the real build).
- **Exit test (methodology corrected twice — see ADR-011, then ADR-012 which supersedes ADR-011's
  gating role for PubMedQA):**
  `python -m scripts.eval_retrieval --dataset nfcorpus --limit 323` is the **primary, gating** run —
  NFCorpus (BEIR) is real, human-judged, graded-relevance IR with no single-modality saturation, unlike
  PubMedQA (see ADR-012 for why PubMedQA was retracted as the gate). Judged by a **pre-registered
  statistical-significance rule**, not raw point-estimate inequality: RRF significantly better than the
  best single modality → pass (fusion adds value); statistical tie (95% bootstrap CI includes 0) → pass
  (plumbing correct, dominance not demonstrated — Phase 7's job); RRF significantly worse → fail, real
  bug. **Real result: statistical tie** (rrf−dense mean diff −0.0034, 95% CI [−0.0171, +0.0103]; RRF
  significantly beats bm25, +0.0398 CI [+0.0242,+0.0564]) — gate **PASSED**.
  `python -m scripts.eval_retrieval --dataset pubmedqa --limit 200` is now an **infra-correctness floor**
  (all three modalities Recall@10 ≥ 0.9, a regression guard) plus non-gating diagnostics (Recall@1,
  hard-subset cut) — real result: floor passed comfortably (0.98–0.995 across all three).
  `python -m scripts.eval_retrieval --dataset medqa --limit 200` remains a **secondary diagnostic only**,
  labelled `lexically biased` in its own output, never gating (ADR-011).
  All three modes print `torch.cuda.is_available()` and device capability unconditionally (real run:
  RTX 5070 Ti, capability (12,0), CUDA confirmed working — no silent CPU fallback).

## Phase 2 — Knowledge graph layer  `[x]` (82f7824, follow-up corrections 36071ef, 2026-09-05/06)
- `docker/compose.yaml` Neo4j 5 + `scripts/load_primekg.py`
- `src/kg/linker.py` (scispaCy UMLS linking, threshold tuned on 50 hand-checked items)
- `src/kg/traversal.py` — 1–2 hop typed subgraph, edge-type allowlist, degree cap
- `src/kg/queries.py` — every Cypher string lives here
- Triple verbaliser → `Evidence(modality="kg")`
- `scripts/precompute_kg_cache.py` — runs linker + traversal once per question
  across every dataset/split that will be evaluated, writes
  `data/processed/kg_cache/<dataset>/<question_id>.json` (linked CUIs,
  subgraph triples, verbalised evidence). See ADR-007: this is what lets
  Neo4j be shut down for all of Phase 3–7's eval runs on 16GB RAM, and makes
  KG retrieval deterministic/replayable independent of Neo4j uptime.
- `src/kg/traversal.py` gains a cache-first read path: eval-time code reads
  the precomputed file and never opens a live Neo4j connection; only
  `precompute_kg_cache.py` and ad-hoc debugging talk to Neo4j directly.
- **Amended scope (see the strategic framing at the top of this phase's
  session and ADR-014–017):** the roadmap's original exit test is a routine
  plumbing check on its literal reading, but the NFCorpus fusion result
  (ADR-012) makes C1 depend almost entirely on the KG modality, so this
  phase also delivers (a) MedQA KG-hit rate broken down by question category,
  not just an aggregate, and (b) a marginal-value diagnostic — what fraction
  of retrieved KG triples are both *not already stated* in the real 12-item
  dense `EvidenceBundle` and *actually relevant* to the question
  (`novel_and_relevant_rate`), since presence of a linked CUI is not evidence
  of value on its own.
- **Exit test (literal):** `python -m scripts.kg_probe --dataset medqa --limit 200` reports KG-hit rate ≥ 60% and dumps 10 sample subgraphs for manual inspection; then `docker compose down` (Neo4j stopped) and re-running the same 200 questions from the cache file reproduces identical KG-hit rate and subgraphs byte-for-byte.
  **Literal result: PASSED.** Real run on n=200 MedQA test: `aggregate_hit_rate = 0.960` (≥0.60 floor); cache-reproducibility confirmed byte-for-byte (only the provenance timestamp differs) after a full `docker compose down`/container-removal cycle. Full artifact: `results/phase2_kg_probe/medqa_n200.json`; table: `docs/04_EVALUATION.md` §5.3.
  **Substantive result (the one that matters for C1, per ADR-016 — do not read the literal PASS above as a coverage or value claim):** the 96% hit rate is an arithmetic artifact of high entity fan-out (`entity_level_mapping_rate = 0.1122`; `naive_expected_hit_rate_under_independence = 0.9998` — a hit is near-guaranteed by having ~74 linked entities/question, regardless of real coverage). The defensible C1 evidence figure is `overall_novel_and_relevant_rate = 0.4063` (final, seeded-random-subset measurement — see below) — real but middling, bottlenecked by retrieval precision (only `overall_relevant_rate = 0.4734` of kept triples are even on-topic; hub-node noise from ADR-015's 40-triple cap, saturated on 192/200 questions) more than by redundancy with dense passages (`overall_novel_given_relevant_rate = 0.8581` — when a triple *is* relevant, it usually is novel). **The KG modality's marginal value over dense retrieval is not yet established; this is the honest position Phase 3 inherits, not a result to soften.**
  **Follow-up corrections, same phase, after initial close-out (ADR-014, ADR-016, ADR-018):** (1) PrimeKG's raw edge count (8,100,498 directional rows, ~4.05M unique undirected edges) is now stated alongside the 740,674 post-allowlist figure so it reads as a deliberate filter, not an incomplete load. (2) The novelty/relevance scoring's triple subset selection was alphabetically biased; fixed to a seeded random sample — real effect on the headline figure was small (0.4078→0.4063). (3) `precompute_kg_cache.py` has now also been run for real on PubMedQA (500 test + 500 dev) and MedMCQA (500-question seeded validation subsample), matching ADR-005's evaluation scope exactly (not full splits — verified on disk: 1000 pubmedqa files split 500/500, 500 medmcqa files all `validation_*`). (4) An ablatable `kg_rerank` flag (MedCPT cross-encoder triple selection) was implemented and tested via a real paired-bootstrap significance check (ADR-018): a narrow significant win on `relevant_rate`, but a **tie** on the primary `novel_and_relevant_rate` metric at n=200 — ships off by default, not adopted as a result. **(All four of these Phase 2 measurements were made on the MedQA TEST split — AUDIT 2026-09-06 §3.1. Re-derived on a train-500 slice on 2026-09-07, ADR-027: ADR-015/016/017 upheld, ADR-018's primary-metric tie flipped to a significant win that survives Holm-Bonferroni correction across the 2-metric family (AUDIT §6.8). The default stays off anyway — a named pre-registration deviation from ADR-018's own rule, recorded in ADR-027 with its reasons. The train artifacts `results/phase2_kg_probe/medqa_train_n500*.json` are now authoritative; the test-split ones are stamped `superseded: true`.)**
  **Still deferred, stated plainly:** full-scope novelty scoring (all ~38 triples/question × 12 passages, currently a disclosed 10-triple subset); a paraphrase-aware relevance check (currently exact-substring); the CUI→PrimeKG crosswalk still covers drug/gene entities via name-matching only, no licensed UMLS Metathesaurus (ADR-014); ~~`kg_rerank`'s tied primary metric has not been re-tested at larger n~~ (re-tested 2026-09-07 on train-500, ADR-027: the tie flipped to a significant win that survives Holm correction across the 2-metric family; default stays off as a named pre-registration deviation, settled by the Phase 7 grid row).

## Phase 3 — Tri-modal bundle + baseline RAG  `[x]` (b10c385, 2026-09-06)
- `src/pipeline.py` — `EvidenceBundle` (query, options, passages[], triples[],
  kg_cache_hit) + `TriModalPipeline.build_bundle` (dense/BM25 → RRF k=60 →
  MedCPT cross-encoder rerank to top-12; KG triples from `get_subgraph_cached`,
  no live Neo4j — ADR-007). `modalities` arg selects which arms feed the bundle.
- `src/eval/baselines.py` — B0 zero-shot, B1 dense-only, B2 hybrid (no agents),
  B3 MEGA-RAG-style binary verifier (per-option SUPPORTED/NOT_SUPPORTED;
  decides iff exactly one SUPPORTED, else falls back to B2-style generation).
  Reports `verifier_decision_rate` — see ADR-020.
- `src/eval/metrics.py` (accuracy, macro_f1 — Phase 6 extends), `src/eval/results.py`
  (`run_id`, `results/<run_id>/{config.yaml,predictions.jsonl,metrics.json}`).
- `scripts/run_baselines.py` — exit-test CLI. `scripts/kg_cache_manifest.py` +
  `data/MANIFEST.json: kg_cache` — checksum-tracks the gitignored KG caches
  (ADR-019, correcting ADR-018's "already-committed" claim).
- `llm.num_ctx` (8192) + hard `LLMContextBudgetError` guard in `src/llm/client.py`
  — a 4096-token window was silently truncating the KG-triple block; ADR-021.
- Dev slice = MedQA **train** split (seed 42), test stays unseen till Phase 7 (ADR-020).
- **Exit test:** `python -m scripts.run_baselines --dataset medqa --split train --limit 20 --baselines B0,B1,B2,B3` — all four score end-to-end, one `results/<run_id>/` each. Roadmap wording says "200-item"; run on 20 per CLAUDE.md §5. Real numbers: `docs/04_EVALUATION.md` §5.4 (B0 .55 / B1 .60 / B2 .65 / B3 .65 acc, grounded 0.95; n=20, plumbing check, no comparison claim).

## Phase 4 — Multi-agent consensus network  `[x]` (e75de9c, ADR-025 fix 30700f9, 2026-09-06)
- `src/agents/graph.py` LangGraph state machine + checkpointer
- `hypothesis.py`, `audit.py`, `conflict.py` with strict JSON schemas (pydantic) and repair-on-parse-failure
- Budget guards + failure-mode instrumentation from `docs/01_ARCHITECTURE.md`
- `tests/` with a mocked LLM covering: valid path, malformed JSON, invalid citation, conflict, budget trip
- **`max_evidence=12` bundle trim — modality-reserved, never a single cross-modal
  ranking (design constraint set before Phase 4, same silent-elimination risk
  class as the Phase 3 truncation bug, ADR-021).** The pipeline emits 12
  passages + up to 40 KG triples; MedCPT's cross-encoder is a query–*article*
  model, so a short verbalised triple systematically scores below a full
  passage — a single top-12 cross-modal cut would eliminate the KG modality
  entirely and make the agent layer "tri-modal in name, dense-only in fact".
  Instead: rank *within* each modality and reserve slots via a config value
  `agents.evidence_split` (start 8 passages + 4 triples). Rank the KG slot with
  the ADR-018 `kg_rerank` cross-encoder (the comparison it was actually built
  for). Log the realised modality mix per question so an all-passage bundle is
  visible immediately. `evidence_split` is ablatable — the right ratio is a
  Phase 7 empirical question, not a baked-in guess.
- **Exit test:** `python -m src.pipeline --config configs/agents.yaml --limit 20` produces SupportMatrices with zero invalid citations on the dev slice.
  **Result: PASSED** (`results/medqa_V4_1388904_20260906T115020Z/`, MedQA train
  dev-20, seed 42, post the ADR-025 claim-level conflict fix). 20 SupportMatrices
  + traces; 0 surviving cells cite a non-bundle id; RAW citation hallucination 6
  cells (5.3%, filtered at audit). status {ok 19, insufficient_evidence 1};
  `budget_exceeded` 0/20; mean 5.8 calls/Q, 1.05 rounds. Ran on 20 per CLAUDE.md
  §5, not the literal 200. **Findings** (`docs/04_EVALUATION.md` §5.5): (1) under
  a correct claim-level conflict definition the Conflict agent fires 0× on this
  slice — genuine evidence contradictions are rare here, so C2 reframes around
  the Audit layer; (2) 1/108 audit cells cite KG — the KG modality is nearly
  invisible to the agents; (3) uncalibrated `rank-1-by-support` 0.35 < B2 0.65
  — the gap Phase 5 calibration must close. Deviations: `_extends:` config merge
  (ADR-022); modality-reserved evidence trim + backfill (ADR-023); 3-tier
  precedence, recency dropped, Hypothesis deterministic by default (ADR-024);
  conflict is claim-level, semantic-check-first (ADR-025, amends docs/01);
  CLI in `src/pipeline.py: main()` (CLAUDE.md §4). ADR-022…ADR-025.

## Phase 5 — Evidential calibration & ranking  `[x]` DONE 2026-09-07 (a0060cf)
- `src/calibration/scoring.py` (logit assembly), `softmax_rank.py` (temperature
  scaling + fitting — hand-rolled analytic-gradient Adam, not `scipy.optimize`,
  which segfaults on bare import in this environment), `citations.py`. ECE
  itself lives in `src/eval/metrics.py` per CLAUDE.md's file layout, not
  `softmax_rank.py` — `ece()` (equal-width 15 bins for the Phase 7 grid,
  equal-mass 5 bins for dev-scale runs) and `paired_bootstrap_ece_diff()`
  (the ADR-012-style significance check for an ECE comparison).
- Fit α,β,γ,δ,T on a **dev** split (MedQA train, 60 questions), disjoint by
  construction from Phase 3/4's dev-20 (`load_question_sample_excluding`) —
  **and disjoint from the 60-question held-out set the exit test is measured
  on**, so the result is not circular. See ADR-028.
- Abstention threshold τ + coverage/risk curve — computed on the held-out set
  (`tau=0.5458` at a 10% target risk).
- **Exit test result (real, not simulated):** calibrated ECE 0.0724 vs.
  raw-logprob-baseline ECE 0.5418 on the same 60 held-out questions
  (real per-token logprobs via `src.llm.client.chat(logprobs=True)`, same
  EvidenceBundle as the agent pipeline saw); paired bootstrap (10k resamples,
  seed 42) on the difference: **95% CI [−0.5426, −0.2826], entirely below
  0 — calibrated significantly better, not a marginal call.** Ranked output
  carries per-option citations for all 120 questions × 4 options. **PASSED.**
  `results/phase5_calibration/medqa_dev_calibration.json`,
  `results/phase5_calibration/medqa_raw_logprob_baseline.json`,
  `results/medqa_V4_a714fe0-dirty_20260907T121703Z/`.
- **Not established by this result:** any comparison to B2/B3 (§4.3's budget
  asymmetry is still unfixed) or to Phase 4's uncalibrated rank-1-by-support
  (0.267 calibrated vs 0.35 uncalibrated — reported, not explained away; small-n
  fit on a disjoint slice, not a controlled ablation of the same questions).
- **Gamma (KG-evidence weight) is unidentifiable, confirmed from a second
  direction.** 12/626 cells (1.9%) on this slice are KG-modality SUPPORT
  cells, reproducing §5.5 Finding 2 (1/108) at n=120. A 200-resample bootstrap
  gives gamma std=0.234 (>3x alpha's 0.072) with an interval spanning
  [−0.40, +0.66] — consistent with zero. See ADR-028.

## Phase 6 — Evaluation harness  `[ ]`
- `src/eval/harness.py` (parallel, resumable, cached), `metrics.py` (Acc, Macro-F1, P@K, MRR, ECE), `ragas_eval.py`, `nli.py`
- Harness issues LLM calls through an `asyncio.Semaphore` sized by
  `configs/base.yaml: eval.concurrency` (default 4, matching
  `OLLAMA_NUM_PARALLEL=4` — see `docs/04_EVALUATION.md` §0.4 and ADR-006).
- Benchmark concurrency = 1, 2, 4 on a fixed dev slice; write the measured
  questions/hour for each into `results/` — this real number replaces the
  planning-estimate 3× speedup assumption in `docs/04_EVALUATION.md` §0.2
  before Phase 7 is scheduled.
- `scripts/report.py` → markdown + LaTeX tables into `paper/tables/`
- **Exit test:** one command reproduces a full dataset run from a config and regenerates identical tables; the concurrency benchmark table (1/2/4) is written to `results/` with no `TBD`.

## Pre-Phase-7 fixes — MUST complete before any grid run  `[ ]`
Items parked earlier plus the AUDIT 2026-09-06 findings deferred on 2026-09-07,
tracked here so they are not silently dropped. None affects Phases 3–6; all gate
the Phase 7 ablation grid.
- **BM25 parameter tuning** (`k1`/`b`) — parked in Phase 1; retune on a fixed
  dev slice, report Recall@10/MRR before/after, record the chosen values in an
  ADR and `configs/base.yaml`.
- ~~**`kg_rerank` re-test at n=500**~~ — **DONE 2026-09-07 (ADR-027).** Re-run on
  a 500-question MedQA **train** slice (not test — the original n=200 was
  test-split, AUDIT §3.1). The primary metric's tie **flipped to a significant
  win**: `novel_and_relevant_rate_for_question` +0.0367, 95% CI
  [+0.0044, +0.0681], n_paired=477, two-sided p=0.0256. Holm-Bonferroni
  correction across this 2-metric family was run the same day, not deferred —
  **both metrics survive** (`src.retrieval.metrics.paired_bootstrap_p` +
  `holm_bonferroni`, AUDIT §6.8). `kg_rerank` stays off by default and becomes
  its own Phase 7 grid row — **not** because ADR-018's pre-registered rule
  said keep-off (it said adopt on a cleared verdict, and the verdict cleared:
  this is recorded in ADR-027 as a named deviation from that rule, with
  reasons, not as compliance with it). The grid-wide multiplicity discipline
  for the full Phase 7 grid (~9 metrics × ~10 variant-pairs) is still §6.7,
  deferred below. Artifact:
  `results/phase2_kg_probe/medqa_train_n500_kg_rerank_ablation.json`.
- **B3 verifier-collapse robustness check** — Phase 3's `verifier_decision_rate
  = 0.05` (`n_supported = 0` on 15/20) is evidence for C2's binary bottleneck,
  but could instead be a 7B model with a mis-calibrated SUPPORTED threshold.
  The Phase 6 Gemini Flash sensitivity check (§0.6, ADR-008) must record
  `verifier_decision_rate` + the `n_supported` distribution for B3 alongside
  accuracy. If a stronger model still collapses to near-zero decisions the
  argument is robust; if it does not, that must be known before the claim is
  in the paper.
- **C2 reframing — conflict resolution vs. audit discrimination.** Phase 4
  (ADR-025, `docs/04` §5.5) found the Conflict agent fires 0× under a correct
  claim-level definition on the dev-20, and 1/108 audit cells cite KG. Before
  the paper, decide whether C2 ("multi-agent evidence consensus") is claimed
  around conflict *resolution* (needs the Phase 7 grid's V4−conflict cell to
  show it matters — likely won't) or around the Audit layer's
  SUPPORT/CONTRADICT/NEUTRAL discrimination + grounding (the V4−audit cell).
  The grid's leave-one-agent-out cells are the evidence; run them before
  writing the C2 section, not after.

### Deferred from AUDIT 2026-09-06 (triaged 2026-09-07)
The 2026-09-07 session fixed the leakage (§3.1), the four integrity blockers
(§2.1, §3.2, §6.1, §6.2), the calibration/corpus overlap (§3.3), the §3
contribution claims (§5.1–5.6) and ADR-021 (§6.6). These were explicitly
deferred to this batch and **must be closed before the grid runs**:

- **§1.1 + §1.2 — write real `metrics.json` for Phase 1.** `BLOCKER`. The
  entire Phase 1 gate exists only in gitignored logs: NFCorpus nDCG@10
  (0.3362 / 0.2929 / 0.3327) and all three CIs live in `logs/eval_nfcorpus.log`;
  the PubMedQA infra-floor Recall@10/MRR and the MedQA diagnostic row live in
  `logs/eval_pubmedqa_v2.log`. From a clean clone, Phase 1 is unverifiable.
  Make `scripts/analysis/nfcorpus_significance.py` write
  `results/phase1_nfcorpus/metrics.json` (means, all three CIs, `n_resamples`,
  `seed`) and `scripts/eval_retrieval.py` write
  `results/phase1_retrieval_<dataset>/metrics.json` per mode; then re-point
  `docs/04_EVALUATION.md` §5.2 at those paths. Until this is done, every Phase 1
  number in the docs violates CLAUDE.md §2.
- **§4.1 — B3 prompt-sensitivity check.** `MAJOR`. `verifier_decision_rate`
  currently rests on a single verifier phrasing while B2 gets a neutral
  instruction, which leaves the project's designated motivating result open to
  a "strawman by prompt construction" objection. Run B3 on the dev-20 with at
  least two further phrasings (one neutral, one permissive) and report
  `verifier_decision_rate` and the `n_supported` distribution for all three.
  Note the bar this must now clear: §5.4 already shows the rate is invariant at
  0.05 across a context-window change that inverts the failure direction, so
  the open question is specifically wording, not stability in general. Fold
  into the existing *B3 verifier-collapse robustness check* item above — same
  claim, two different sensitivity axes.
- **§4.3 — equal-budget B2 comparator.** `MAJOR`. B2/B3 receive up to 12
  reranked passages **plus** all cached KG triples (up to 40) ≈ up to 52
  evidence items; V4 receives `max_evidence: 12` (8 passages + 4 triples).
  `docs/04_EVALUATION.md` §5.5 Finding 3 reports V4's rank-1-by-support 0.35
  "vs B2's 0.65 on the same slice" and attributes the gap to the absence of
  calibration with the budget asymmetry unstated. Add a 12-item-trimmed-bundle
  B2 as the correct comparator, and state the asymmetry at §5.5 in the
  meantime. **Phase 5 must not read that gap until this exists.**
- **§6.7 — multiple-comparison discipline.** `MAJOR`. The grid is 8 MedQA
  variants + 2 datasets × 2 variants at ~9 metrics each; ADR-012 already ran 3
  comparisons and ADR-018 ran 2, all at an uncorrected 95% CI, and nothing
  mentions family-wise error. **Pre-register in `docs/04_EVALUATION.md` §1,
  before the grid runs, the single confirmatory comparison (V3 vs V5), declare
  every other comparison exploratory, and specify Holm correction for the
  exploratory family.** Pre-registration after seeing grid numbers is worthless,
  so this one is ordered first in this batch.
- **All AUDIT MINORs** (§1.10–1.14, §2.7–2.9, §3.5, §3.6, §4.4–4.6, §6.8–6.10).
  Mostly one-line doc/number corrections — quote 5.75 not 5.8; the
  `-ling-wanli` NLI model id in CLAUDE.md §3; `off-label use` in
  `docs/03_DATASETS.md`; ADR-018's status line; per-variant `configs/B*.yaml`;
  `models_used` in the call-stats; the B3 "19/20 identical to B2 by
  construction" annotation; `src/eval/harness.py` marked not-yet-implemented.
  Two are not cosmetic and should be done with the code, not the docs pass:
  **§6.8** (`paired_bootstrap_ci` returns no p-value while §1 mandates them,
  and its upper percentile index is off by one — this affects every CI the
  project has reported) and **§6.5** (the KG cache is keyed on a positional
  index into a raw HF file, so any re-fetch that reorders rows silently
  re-associates every cached subgraph with the wrong question).

Also still open from the audit and **not** in the 2026-09-07 work order —
flagged here so they are not lost: §2.2 (ADR-024 is `Status: accepted` while
still asserting the `conflict_fire_rate = 0.55` numbers ADR-025 retracts — a
BLOCKER, and a one-line status fix), §1.3–1.9, §2.3–2.6, §3.4, §5.5 (H4 cannot
be measured — no p-vector is persisted), §6.3, §6.4.

## Phase 7 — Full runs & ablation grid  `[ ]`

| Variant | Dense | BM25 | KG | Agents | Calibration |
|---|---|---|---|---|---|
| V0 baseline LLM | – | – | – | – | – |
| V1 | ✓ | – | – | – | – |
| V2 | ✓ | ✓ | – | – | – |
| V3 | ✓ | ✓ | ✓ | – | – |
| V4 | ✓ | ✓ | ✓ | ✓ | – |
| V5 (full) | ✓ | ✓ | ✓ | ✓ | ✓ |
| V4−audit, V4−conflict | leave-one-agent-out | | | | |

**Grid scale (ADR-005, `docs/04_EVALUATION.md` §0.3 — the ceiling estimate in
§0.2 is what forced this cut):**
- Full 8-variant ablation (row above, including the two leave-one-out cells) —
  **MedQA only**, 500-question fixed seeded subset (seed 42).
- PubMedQA and MedMCQA — **V3 and V5 only**, 500-question fixed seeded subsets
  each (PubMedQA: official test; MedMCQA: seeded subsample of validation, see
  `docs/03_DATASETS.md`).
- 3 seeds (42/43/44) — **V3 and V5 on MedQA only**, the pair the paired
  bootstrap significance test compares. All other cells single-seed.
- Run with `eval.concurrency=4` and `OLLAMA_KEEP_ALIVE=-1`; expect the grid to
  span multiple background-job sessions (§0.2 estimates ~26h at 3×
  concurrency speedup, to be replaced by the Phase 6 measured number).
- **Sensitivity check (ADR-008):** V3 vs V5, 200 MedQA questions, via free
  Gemini Flash tier — reported alongside the grid as a robustness check, not
  a headline result (`docs/04_EVALUATION.md` §0.6/§5.1).

- **Exit test:** `results/` contains every cell of the cut grid above (not the
  original full cross-product); `docs/04_EVALUATION.md` §5 and §5.1 have no
  `TBD`.

## Phase 8 — Analysis & paper artifacts  `[ ]`
- Error taxonomy from `conflict_log` + failure-mode counters (100 sampled errors, hand-labelled)
- Figures: architecture, reliability diagram, ablation bars, coverage–risk curve
- `paper/` LaTeX with all tables generated by `scripts/report.py`
- **Exit test:** `make paper` builds the PDF from scratch with zero hand-typed numbers.
