# 00 — Research Specification

## 1. Problem

Medical LLMs hallucinate; in clinical decision support this is a safety-critical
failure. Retrieval-augmented mitigation frameworks (MEGA-RAG and successors)
add a verification module that cross-checks a generated answer against medical
knowledge sources. Those verifiers — DPAG/DISC-style — are **binary-constrained**:
they emit a yes/no support judgement over a single claim.

## 2. Gap

Clinical reasoning is rarely binary. Differential diagnosis is a *ranking*
problem over competing hypotheses, and board-style assessment (USMLE, NEET-PG)
is 4–5 option multiple choice. A binary verifier:

- cannot express relative support across mutually exclusive options,
- collapses "no evidence" and "contradicting evidence" into one signal,
- produces no calibrated confidence, so downstream triage cannot use it,
- gives no ranked output a clinician can inspect.

## 3. Contribution

*Rewritten 2026-09-07 against AUDIT 2026-09-06 §5.1–5.6. The previous version
stated C1–C4 flatly as achieved results; at the time, three of the four had no
supporting run and one had three runs against it. Each claim below now carries
its evidential status and the artifact that settles it. **A claim marked
PENDING may not appear in paper text as an achieved result until the run named
in it exists.** Status is part of the claim, not a caveat appended to it.*

**C1 — Tri-modal evidence: a diagnostic (currently negative) contribution.**
`PENDING the Phase 7 V2-vs-V3 cell.` Joint dense + lexical + knowledge-graph
retrieval, with KG triples treated as first-class evidence objects rather than
context text. **No run yet shows the KG arm improving any downstream metric,
and three measurements point against it:** RRF fusion is a statistical tie with
dense alone on NFCorpus (ADR-012 — a *tie*, reported as a tie); the KG arm's
own quality figure is middling (`overall_novel_and_relevant_rate ≈ 0.41`,
`entity_level_mapping_rate ≈ 0.11`); and on the Phase 4 dev-20 the KG slot was
filled on 17/20 questions but only **1 of 108** audit cells cited a KG triple —
the modality is delivered to the agents and then almost never used. Until the
V2 (no KG) vs V3 (KG) cell of the Phase 7 grid exists, C1's honest form is a
*diagnostic* result — "here is what it costs to wire a biomedical KG into a
clinical RAG pipeline, and here is the measured value it returns" — with the
current answer being *not much*. A negative result, reported with its
mechanism, is a publishable contribution; an unsupported positive claim is not.

**C2 — Multi-agent audit discrimination.** `SUPPORTED: audit. UNVERIFIED:
conflict resolution.` A bounded agent network that audits each option's claims
against retrieved evidence and labels them SUPPORT / CONTRADICT / NEUTRAL,
replacing the single binary verifier. The *discrimination* half is supported:
on the dev-20 the audit layer produced 108 non-neutral cells (78 SUPPORT / 30
CONTRADICT) giving a graded signal on 61 of 80 options, where the B3 binary
verifier isolated exactly one option on 1/20 questions under **both** context
conditions tested (`docs/04_EVALUATION.md` §5.4). The *conflict-resolution*
half is *not* supported and must not be claimed: under the corrected
claim-level definition (ADR-025) the Conflict agent has **never executed on
real data** — `conflict_fire_rate = 0.00`, `conflict_llm_calls = 0`,
`claim_level_conflicts_evaluated = 0`. Not one claim on the dev-20 had genuinely
opposing evidence. Whether that is a property of MedQA, of the evidence bundle
size, or of the definition itself is open, and is the first thing Phase 7 must
answer at n=500.

**C3 — Evidential calibration for multi-choice.** `RESTATED — mechanism works,
the evidential contribution is not established (ADR-028, Phase 5).` Per-option
evidence scoring → temperature-scaled softmax → ranked differential with
per-option citation sets, evaluated by ECE. There is no calibration
*guarantee* — the word stays removed; a temperature fitted on a dev split is
an empirical fit, not a bound. Fit on a 60-question MedQA train slice
(disjoint from Phase 3/4's dev-20), ECE measured on a disjoint 60-question
held-out slice: 0.0724, significantly better than a real raw-logprob baseline
at temperature 1 (0.5418, 95% CI [−0.5426, −0.2826]). **A follow-up ablation
shows that gap is almost entirely a temperature-scaling effect: a single
temperature fit on the same raw logprobs, with zero evidential features,
reaches ECE 0.0725 — a statistical tie with the full calibrated pipeline
(95% CI [−0.1344, +0.0887]).** The claim is therefore restated: temperature
scaling of the raw signal suffices to match the evidential features'
calibration quality at this sample size; the features' (`s_pos`, `s_neg`,
`kg_support_count`, `mean_rerank_score`) additional contribution beyond
temperature scaling is not established, and must not be claimed until a
larger-n test (Phase 7) shows otherwise. Separately, calibrated rank-1
accuracy (0.267) is below Phase 4's uncalibrated `rank-1-by-support` (0.35,
different question set) — a better-calibrated but less-accurate model, stated
plainly. `kg_support_count`'s weight (gamma) is itself unidentifiable
(bootstrap std 3x alpha's, interval spanning zero) — independent confirmation
that the KG modality is nearly invisible to this pipeline, consistent with C1.
Fit set: a MedQA train slice, not `pqa_labeled` dev — ADR-026's PubMedQA
fitting-set discipline applies once/if a PubMedQA calibration pipeline exists
(C4 is still pending), not to this MedQA-only Phase 5 run.

**C4 — Empirical study.** `PENDING Phase 7.` Baseline parity on PubMedQA
(binary), and the MedQA-USMLE / MedMCQA multi-choice comparison, with an
ablation grid isolating each modality and each agent. **Every row of the §5
grid in `docs/04_EVALUATION.md` is currently `TBD`**, and PubMedQA has never
run through any baseline — `scripts/run_baselines.py --dataset` accepts only
`{medqa, medmcqa}`, so "baseline parity on PubMedQA" has no code path yet. The
Phase 3/4 numbers that do exist are n=20 single-seed dev-slice plumbing checks,
explicitly not grid results, and sit inside a **measured ±0.05 noise floor**
(ADR-021's no-op perturbation row). Nothing here is a comparison claim yet.

**C5 — The question-level KG-hit-rate is an arithmetic artifact of entity
fan-out.** `SUPPORTED — fully traceable, and currently the best-evidenced
result in the project.` The headline-friendly "PrimeKG covers 96% of MedQA
questions" (`aggregate_hit_rate ≈ 0.96`) is very nearly a *guarantee* of the
measurement's own shape rather than a finding about coverage: with ~74 linked
entity mentions per question and a true per-entity resolution rate of only
`entity_level_mapping_rate ≈ 0.11`, the probability that *at least one* mention
resolves is ~1 by construction — `naive_expected_hit_rate_under_independence ≈
0.9998`, which is *higher* than the rate actually observed. A question-level
"≥1 entity mapped" hit definition therefore cannot distinguish a well-covered
question from a barely-covered one, and any KG coverage number reported that way
— by this project or by the prior work it compares against — is measuring entity
fan-out, not coverage. The honest per-entity figure is 11%.

This is a methodological finding about how KG-augmented RAG systems report
coverage, it is reproducible from a single committed artifact
(`results/phase2_kg_probe/*.json`, which reports all three quantities plus an
explicit `hit_definition` string precisely so the permissive rate cannot be
misread), and it is the reason C1 above is stated as a diagnostic rather than a
win. It was absent from this section entirely until 2026-09-07 (AUDIT §5.6) —
the project's strongest supported result was the one it was not claiming.

## 4. Formal setup

Query `q` with option set `O = {o_1..o_k}` (k ∈ {2,4,5}; k=3 for PubMedQA
yes/no/maybe).

1. Retrieval returns evidence set `E = E_dense ∪ E_bm25 ∪ E_kg`, each item
   `e = (id, text|triple, source, score)`.
2. For each option `o_i`, the agent network produces an evidence subset
   `E_i ⊆ E` and a support tuple `(s_i^+, s_i^-, s_i^0)` — supporting,
   contradicting, neutral mass.
3. Option logit `z_i = α·s_i^+ − β·s_i^- + γ·kg_i + δ·rerank_i`, weights fit on
   a dev split (not test).
4. `p = softmax(z / T)`, temperature `T` fit on dev for calibration.
5. Output: ranked `[(o_i, p_i, cite(E_i))]`.

## 5. Hypotheses

- **H1** Tri-modal retrieval beats dense-only on MedQA accuracy (KG traversal
  supplies drug–disease relations absent from top-k passages).
- **H2** The audit + conflict agents reduce ungrounded assertion rate vs. a
  single-pass RAG generator at equal retrieval budget.
- **H3** Evidential calibration yields lower ECE than raw LLM option
  log-probabilities, at equal or better accuracy.
- **H4** Ranking quality (MRR, P@2) degrades gracefully when the gold option is
  not rank-1, i.e. the true diagnosis stays in the top-2 differential.

## 6. Out of scope

Fine-tuning a medical LLM; multimodal (imaging) inputs; real patient data (all
datasets are public and de-identified); clinical deployment claims.

## 7. Threats to validity (address explicitly in the paper)

- **The Phase 2 KG probe originally ran on the MedQA *test* split.** Disclosed
  in full: the first probe (`results/phase2_kg_probe/medqa_n200.json`, and the
  later `medqa_n500.json`) was run over MedQA test questions, and the Phase 2
  KG hyper-parameters and the `kg_rerank` keep/drop decision (ADR-015, ADR-016,
  ADR-017, ADR-018) were originally selected by reading those test-split
  outputs — while Phase 7 reports on a 500-question seeded subset of that same
  test split. A parameter was chosen by looking at a quantity it would later be
  judged on. Remediated 2026-09-07 (AUDIT 2026-09-06 §3.1): the probe was
  re-run on a 500-question MedQA **train** slice for both `kg_rerank` variants,
  every one of those four decisions was re-derived from the train numbers, and
  the test-split probe outputs are retained but marked superseded so they cannot
  be cited as the basis of a decision. The original test-split selection is
  disclosed here rather than erased, because a reader cannot verify that the
  re-derivation reached the same conclusion unless they know it was re-derived.
- Retrieval corpus overlap with benchmark test items → `src/eval/contamination.py`.
- Calibration fitting set vs. retrieval corpus (ADR-026): calibration is fitted
  on `pqa_labeled` dev, verified absent from the indexed corpus. The earlier
  plan fitted on `pqa_artificial`, 20k documents of which *are* indexed.
- LLM version drift → pin model ids and record them in every `config.yaml`.
- PrimeKG coverage gaps for exam-style questions → report KG-hit rate per dataset.
- Agent cost/latency → report tokens and wall-clock per question alongside accuracy.
