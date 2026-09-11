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

*Updated 2026-09-11 against the completed Phase 7 MedQA grid
(`docs/04_EVALUATION.md` §5/§5.0a). That grid is seven test n=500 cells (V0,
V1, V2, V3, B3, V4−conflict, V5, all on code identical to `41f7e14`), plus
V5-refit (ADR-043), the post-hoc-binary derived analysis, the one
pre-registered confirmatory test, the Holm-corrected m=31-of-35 exploratory
family (ADR-040), and the separately Holm-corrected unplanned family U1–U8
(m_U = 18, `docs/04` §1.0b). Every project number cited below traces to
`results/phase7_grid/{confirmatory_v3_v5,exploratory_family,unplanned_v0_v2_family,v5_diagnostics}.json`,
`results/phase7_posthoc_binary_audit/metrics.json`, or a grid run's
`metrics.json` via `paper/tables/results_grid.md`. MEGA-RAG's figures are
quoted from the cited paper, verified against it on 2026-09-11. Earlier
revisions: 2026-09-10 (the five-cell grid) and, below that, 2026-09-07
against AUDIT 2026-09-06 §5.1–5.6. The previous version
stated C1–C4 flatly as achieved results; at the time, three of the four had no
supporting run and one had three runs against it. Each claim below now carries
its evidential status and the artifact that settles it. **A claim marked
PENDING may not appear in paper text as an achieved result until the run named
in it exists.** Status is part of the claim, not a caveat appended to it.*

**Headline (MedQA test, n=500): a negative result shaped as one step, not a
slope.** Along the nested component chain the accuracy point estimates fall
at every step: V0 zero-shot 0.582 → V1 +dense 0.566 → V2 +lexical 0.556 →
V3 +KG 0.546 → V4−conflict +Hypothesis and Audit agents 0.358 → V5 +Conflict
agent, +calibration 0.316. The off-chain variants are B3 (V3's retrieval
plus the binary verifier) at 0.538 and V5-refit (V5 re-scored) at 0.292.
**Zero-shot with no retrieval has the highest accuracy point estimate in the
grid.** The significance tests resolve that monotone sequence into two tiers
(`docs/04` §5.0a):

- **Retrieval tier, {V0, V1, V2, V3, B3}: no tested difference survives
  correction.** That includes **V0 vs V1, a tie** (+0.016, 95% CI [−0.008,
  +0.042], p = 0.2308, U1). V0 vs V3 (+0.036) clears an unadjusted CI but
  not Holm (U2).
- **Agent tier, {V4−conflict, V5, V5-refit}: no tested difference survives
  correction either** (pair d, U6, U7).
- **Between the tiers: a 0.22–0.27 accuracy drop, significant in every
  cross-tier test run.** V0 > V5 (U3), V1 > V5 (pair e), V3 > V5 (the
  confirmatory test), B3 > V5 (pair h), V3 > V5-refit (U8).

**What the data support.** Retrieval does not improve accuracy over
zero-shot. **"Retrieval hurts" is not supported:** the one direct test is a
tie. The graded audit-and-rank stack, this project's own contribution,
significantly hurts accuracy against zero-shot and against every
retrieval-tier variant it was tested against, and correcting its scoring
rule does not change that. C1–C3 each now have n=500 test evidence against
them. The delivered part of C4 *is* that evidence. Each claim below names
its cells.

**Relation to prior art: the anomaly in MEGA-RAG's own Table 2.** MEGA-RAG
(Frontiers in Public Health, 2025; PMC12540348), Table 2, HealthQuestDB (890
public-health QA pairs, LLaMA3-70B behind every LLM row): the plain **LLM
scores accuracy 0.6674, and LLM+RAG scores 0.6281** (F1 0.7643 vs 0.6739).
Adding standard retrieval lowered both below the bare model. The paper
reports no significance test for any Table 2 comparison. Its §5.2 describes
the two baselines' error profiles: the vanilla LLM's inflated false
positives, and standard RAG's "overcorrection, excluding many
borderline-relevant passages and increasing false negatives". It never
states that retrieval made the model *less accurate*, and never asks why.
That inversion runs the same direction as this project's V0 > V1. What this
work adds, stated only as strongly as the data allow:

1. **It re-measures the inversion with a paired significance test, and
   finds a tie.** The direction replicates on MedQA at n = 500, but the gap
   cannot be told apart from zero. The supportable statement is "standard
   RAG does not beat the bare LLM", not "standard RAG hurts". MEGA-RAG's own
   0.039 gap cannot be re-tested here: its per-question predictions are not
   available, and the two results differ in dataset, task format and
   generator.
2. **It locates where a real, significant loss does occur, and gives its
   mechanism.** The loss is not at the retrieval step. It is at the graded
   audit-and-rank layer: 0.22–0.27 accuracy. ADR-041 explains it: V5's
   calibrated score is functionally a "fewest CONTRADICT cells" rule, and
   that rule cannot separate the gold option from options the audit never
   examined. In MEGA-RAG's Table 2 the full system recovers above the plain
   LLM (0.7913). This project's full system falls far below it. The two
   results diverge at exactly the layer this project added.
3. **It does not explain MEGA-RAG's retrieval-step inversion.** No mechanism
   for a retrieval-step loss was measured, because at this n there is no
   retrieval-step loss to explain. The paper may say the anomaly went
   unremarked, was re-tested, and came out at most a null. **It may not say
   this work "reproduces and explains" it.**

**C1 — Tri-modal evidence: a diagnostic (negative) contribution.**
`AGAINST at n=500. The isolation C1 always needed, V2 ({dense,lexical}) vs V3
(+KG), ran as U4 and is a tie on accuracy, macro-F1 and halluc-rate.` Read
from the KG's side, U4's accuracy CI puts the KG's effect between −3.0 and
+1.0 points (V2 − V3 = +0.010, 95% CI [−0.010, +0.030], p = 0.3634). Adding
lexical to dense ties too (U5). Joint dense + lexical + knowledge-graph retrieval,
with KG triples treated as first-class evidence objects rather than context
text. **No run shows the KG arm improving any downstream metric, and five
measurements point against it, the last being the exact isolation (U4,
above):** RRF fusion is a statistical tie
with dense alone on NFCorpus (ADR-012 — a *tie*, reported as a tie); the KG
arm's own quality figure is middling (`overall_novel_and_relevant_rate ≈
0.41`, `entity_level_mapping_rate ≈ 0.11`); on the Phase 4 dev-20 the KG slot
was filled on 17/20 questions but only **1 of 108** audit cells cited a KG
triple; and **the Phase 7 exploratory family's pair a (V1 dense-only vs V3
dense+lexical+KG, MedQA test n=500) is a statistical tie on accuracy (+0.020,
95% CI includes 0, p=0.125), macro-F1 (p=0.110), and halluc-rate (p=0.349)**.
Pair a changes two modalities at once; U4 then isolates the KG and finds the
same tie. At n=500, adding retrieval modalities on top of dense-only produces
no detectable accuracy gain, and dense-only produces none over zero-shot
(U1, headline above). C1's honest form stays a *diagnostic* result: "here is
what it costs to wire a biomedical KG into a clinical RAG pipeline, and here
is the measured value it returns". The measured answer is *nothing
detectable*, now from the exact isolation at n=500. A negative result, reported
with its mechanism, is a publishable contribution; an unsupported positive
claim is not.

**C2 — Multi-agent audit discrimination.** `SUPPORTED: audit. UNVERIFIED:
conflict resolution.` A bounded agent network that audits each option's claims
against retrieved evidence and labels them SUPPORT / CONTRADICT / NEUTRAL,
replacing the single binary verifier. The *discrimination* half is supported:
on the dev-20 the audit layer produced 108 non-neutral cells (78 SUPPORT / 30
CONTRADICT) giving a graded signal on 61 of 80 options. The B3 binary verifier
comparison is **weaker than earlier stated and must be quoted with its range**
(ADR-033, AUDIT §4.1): B3's `verifier_decision_rate` is prompt-sensitive —
**0.05** under the shipped strict phrasing ("SUPPORTED only if the evidence
positively supports it"), but **0.30–0.35** under neutral / permissive
phrasings on the same dev-20. The prompt-invariant part of the claim is
narrower: under *every* phrasing tested the binary verifier isolates a single
option on a minority of questions (best case 7/20) and returns multiple- or
zero-SUPPORTED on the rest — a per-option yes/no verifier cannot rank
mutually-exclusive options — but the strong "1/20" figure is an artifact of
the strict instruction and is retired (`docs/04_EVALUATION.md` §5.4). The
*conflict-resolution*
half is *not* supported and must not be claimed: under the corrected
claim-level definition (ADR-025) the Conflict agent has **never executed on
real data** — `conflict_fire_rate = 0.00`, `conflict_llm_calls = 0`,
`claim_level_conflicts_evaluated = 0`. Not one claim on the dev-20 had genuinely
opposing evidence. **Phase 7 answers this at n=500: pair d (V4−conflict vs V5,
the Conflict agent's isolated effect, MedQA test) is a statistical tie on
every metric tested (accuracy p=0.067, macro-F1 p=0.169, halluc-rate p=0.889,
P@1/P@2/MRR all p≥0.31, none surviving Holm correction) — disabling the
Conflict agent entirely changes nothing detectable at n=500.** This is not a
property of an unlucky dev-20; it replicates at 25× the sample size. The
Conflict agent, as specified, has no measured contribution on this benchmark.

**Both halves now carry a second, sharper piece of evidence — and it complicates
the picture rather than resolving it favorably.** `docs/04` §5.0a's
post-hoc-binary derived analysis (ADR-036 Constraint B: collapse the graded
SUPPORT/CONTRADICT/NEUTRAL audit label to binary SUPPORTED/NOT_SUPPORTED,
force `s⁻=0`, re-rank with V5's fitted weights, zero new LLM calls) tests
whether the *graded* label itself carries value, independent of the prompt
that produces it. **It does, in the narrow sense that removing it hurts:**
post-hoc-binary (accuracy 0.218) is significantly *worse* than both
V4−conflict (0.358, pair i, p<0.0001) and V5 (0.316, pair j, p<0.0001) on
accuracy, macro-F1, and every ranking metric tested — halluc-rate is a tie in
both (pairs i/j, p=0.61/0.61). **So within the agent framework, the graded
audit label measurably beats a binary collapse of the same evidence** — a real,
positive, C2-relevant finding. **But this has to be read against the
much larger fact that the entire graded-audit agent framework (V4−conflict,
V5) loses badly to plain retrieval and even to the binary verifier B3 it was
built to outperform**: V1 (dense-only) beats V5 by +0.250 accuracy (pair e,
p<0.0001), and **B3 itself — the very binary verifier C2 is framed against —
significantly beats V5 by +0.222 accuracy (pair h, p<0.0001)**. C2's honest
statement is now: *the graded label is better than a binary collapse of the
same audit evidence, but the audit-and-rank mechanism this project built,
graded or not, is substantially worse than either doing no agentic
verification at all (V1/V3) or using the cruder binary verifier it was meant
to replace (B3).* This is not the C2 the project set out to establish, and it
is stated as such rather than softened. **The unplanned family adds the
bluntest comparison: the agent stack is significantly worse than no
retrieval at all.** V0 zero-shot beats V5 by +0.266 accuracy (U3,
p < 0.0001, Holm-significant), a larger gap than any retrieval baseline's.

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

**Phase 7's promised larger-n test (n=500 MedQA test, pair d: V4−conflict vs
V5, i.e. calibration's isolated effect holding the agent stack fixed) is now
in: a statistical tie on accuracy (p=0.067), macro-F1 (p=0.169), and every
ranking metric (P@1/P@2/MRR, all p≥0.31), none surviving Holm correction.**
This closes the open question from the paragraph above in the same direction
Phase 5 already pointed: **calibration does not measurably move accuracy or
ranking quality at n=500, for better or worse.** Combined with Phase 5's
temperature-scaling-alone result, the honest position is now stronger than
"not established" — at both the fit-time ECE check and the n=500 downstream
accuracy/ranking check, the evidential calibration layer's contribution over
a simpler alternative (temperature scaling, or no calibration at all) has
twice failed to clear significance. C3 is a **negative result on the
calibration layer's own terms**: the mechanism runs correctly and is
well-calibrated (ECE 0.0724 at fit time — that part is real), but nothing
measured shows it earning its added complexity over simpler alternatives.

**The disclosed scoring-rule correction does not change this (V5-refit,
ADR-043).** V5-refit keeps the evidential weights ≥ 0, imputes a neutral
no-signal rerank value, uses a seeded tie-break, and was fit on train and
frozen before test. It **ties V5** (0.292 vs 0.316, U6, p = 0.2572, the
outcome registered in advance) and **loses significantly to V3** (−0.254,
U8). Its ranking (P@1 0.292 / P@2 0.530 / MRR 0.551, displayed, not tested
against chance) sits next to the chance row's 0.250 / 0.500 / 0.521. The
near-chance ranking is not an artifact of V5's negative α.

**C4 — Empirical study.** `PARTIALLY DELIVERED — MedQA test-set grid complete
(seven cells, 2026-09-11); MedMCQA and PubMedQA still PENDING.` Baseline parity on
PubMedQA (binary), and the MedQA-USMLE / MedMCQA multi-choice comparison, with
an ablation grid isolating each modality and each agent. **The MedQA rows of
the `docs/04_EVALUATION.md` §5 grid are now filled from seven real n=500
test-set runs (V0, V1, V2, V3, B3, V5, V4−conflict, code identical to
`41f7e14`), plus V5-refit beside V5. V4 and V4−audit remain `TBD` (no run),
and every MedMCQA/PubMedQA row remains `TBD` (never run: C4's
empirical-study scope is still only one-third delivered).** The delivered
third's headline is the two-tier step above. Retrieval does not beat
zero-shot, and the agent stack sits 0.22–0.27 accuracy below both. Every
adjacent step of the component chain that was tested is a tie. The one step
where the drop sits, V3 → V4−conflict, was never tested directly. The drop
is established by the cross-tier pairs listed in the headline.
The one pre-registered confirmatory comparison is done: **V3 significantly
beats V5 on accuracy** (0.546 vs 0.316, paired bootstrap 10k resamples seed
42, 95% CI [+0.172, +0.288], p<0.0001, `docs/04` §5.0a) — not the null the
pre-registration flagged as a live possibility, a large, significant *loss*
for the full stack relative to tri-modal retrieval alone. The Holm-corrected
exploratory family (m=31 of 35 pre-registered — two B3 halluc-rate cells are
not computable, a discovered data gap, and pair b's P@2/MRR are not
applicable because V3 emits a 1-item list, ADR-040 — neither a chosen
exclusion) corroborates
this from every angle tested: V1 beats V5 by +0.250 accuracy (pair e), B3
beats V5 by +0.222 accuracy (pair h), and V5's own diagnostics
(`docs/04` §5.0a) show the accuracy floor is already ~0.36 before calibration
is even applied (the raw uncalibrated-margin ranking on V5's own audit
evidence), present uniformly across the run (459 `ok`-status questions score
0.312, statistically indistinguishable from the all-500 figure of 0.316), and
partly explained by a ranking mechanism that occasionally (68/500 questions)
promotes a zero-evidence option over one the audit layer actually supported.
**C4's honest headline, for the portion delivered:** on MedQA at n=500, this
project's full pipeline underperforms its own simpler baselines, significantly
and by a wide margin, at every tier tested (dense-only, tri-modal, and the
binary verifier). The Phase 3/4 n=20 dev-slice numbers that motivated earlier,
softer framings remain useful provenance (they sit inside a **measured ±0.05
noise floor**, ADR-021) but are now superseded as the project's headline
evidence by the real n=500 test-set grid above.

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
- **One NLI model judges two nominally independent results (AUDIT §3.5).**
  `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` is both the
  entailment judge behind the Phase 2 KG `novel_and_relevant_rate` (C1's
  evidence figure) and the Phase 6 hallucination-rate judge (§3 / §5.7). A
  systematic bias in that one model — e.g. under-entailing biomedical claims
  that need an inferential step — propagates, correlated, into both. Phase 6's
  §5.7 control (B2 vs V5 at the same threshold) partly addresses this for the
  hallucination number by showing it is metric-driven not pipeline-driven; the
  KG side has no equivalent control yet. State the shared-judge dependence
  wherever either number is used, and do not present the two as independent
  corroboration of the KG modality's (lack of) value.
