# 04 — Evaluation Protocol

## 0. Eval-scale design (read before Phase 6/7)

These are **planning estimates** to size the grid, not measured results — every
number here is a design input, distinct from `results/*/metrics.json`. Phase 6
must replace the throughput/latency figures with real benchmark numbers before
Phase 7 is scheduled.

### 0.1 Per-question latency budget

Agent calls (V4/V5) carry an `EvidenceBundle` of ~12 items ≈ 3–4k tokens of
context. At a measured local-Ollama prefill rate of 958 tok/s, that's
`3500 / 958 ≈ 3.7s` prefill *before generation even starts*, per call — the
original 37s/question estimate for V4/V5 did not budget this. Revised budget:

| Variant class | LLM calls/question | Budget/question | Basis |
|---|---|---|---|
| V0–V3 (no agent network) | 1 | ~10s | single call: ~4s prefill + short completion |
| V4, V5, V4−audit, V4−conflict | avg ~5–6, hard cap 12 (`max_llm_calls_per_question`) | **~60s** | ~4s prefill + generation, summed over Hypothesis→Audit→Conflict plus up to 2 re-retrieval rounds |

A question that hits the 12-call budget guard can exceed 60s — that's an
expected `status: budget_exceeded` outcome (see `docs/01_ARCHITECTURE.md`
failure modes), not a bug, and is counted separately rather than padding the
per-question average.

### 0.2 Grid ceiling (re-derived from the 60s/question figure)

Serial (concurrency=1) wall-clock for the full post-cut grid (§0.3), using
10s/question for non-agent variants and 60s/question for agent variants:

| Slice | Question-runs | Est. serial time |
|---|---|---|
| MedQA, 8-variant single-seed (V0,V1,V2,V3 @10s + V4,V5,V4−audit,V4−conflict @60s) × 500 | 4,000 | ~38.9h |
| MedQA, extra 2 seeds × {V3,V5} × 500 | 2,000 | ~19.4h |
| PubMedQA, {V3,V5} × 500 | 1,000 | ~9.7h |
| MedMCQA, {V3,V5} × 500 (subsample, see `docs/03_DATASETS.md`) | 1,000 | ~9.7h |
| **Total, concurrency=1** | 8,000 | **~77.8h** |

This is why item 3 below (concurrency) and item 2 (grid cut) are both
necessary — neither alone brings this into a runnable range on a single 16GB
machine. At `OLLAMA_NUM_PARALLEL=4`, expect *sublinear* speedup (RAM-bound, not
purely compute-bound): budget for **3×**, not 4×, until Phase 6 measures the
real number → **~26h** effective, run as background jobs across sessions, not
in one sitting. The original (pre-cut) 8-variant × 3-dataset × 3-seed grid
would have been roughly 3× this — not runnable on this hardware, which is the
concrete justification for the cut in §0.3.

### 0.3 Grid cut (post-decision, supersedes the implied "everything × everything" reading of Phase 7)

- **Full 8-variant ablation** (V0…V5, V4−audit, V4−conflict) — **MedQA only**,
  500-question fixed seeded subset (seed 42), stated as such in the paper.
- **PubMedQA and MedMCQA** — **V3 and V5 only** (the tri-modal-no-agent and
  full-system endpoints). No intermediate ablation cells on these datasets.
- **Multi-seed (3 seeds: 42, 43, 44)** — **V3 and V5 on MedQA only**, since
  that is the pair the paired-bootstrap significance test (§1) compares. All
  other cells are single-seed.
- MedMCQA uses a fixed 500-question seeded subsample of the official
  validation split (not the full ~4.2k), for parity with the other two
  datasets' scale — see `docs/03_DATASETS.md` and ADR-005.

### 0.4 Concurrency

- Server: `OLLAMA_NUM_PARALLEL=4` (see `.env.example`) so a single Ollama
  process can serve overlapping requests instead of queuing them.
- **The harness concurrency layer and the 1/2/4 benchmark were CUT** (ADR-029,
  `docs/SCHEDULE.md`). `src/eval/harness.py` is a sequential loop — no
  `asyncio.Semaphore`, no `eval.concurrency` wiring. The §0.2 "3×" figure is
  therefore **not** replaced by a measured number; the Phase 7 grid runs
  sequentially, which is acceptable at n=500 within the deadline. The LLM cache
  (`.cache/llm/`) absorbs re-runs of a crashed grid job.
- `configs/base.yaml: eval.concurrency` remains as a knob for a future
  parallel harness but is currently read by nothing.

### 0.5 Keep-alive

Eval runs set `OLLAMA_KEEP_ALIVE=-1` (never idle-unload) — see
`.env.example` and ADR-006. Without it, the default ~5-minute unload timeout
means the first request after any harness pause (a slow question, NLI-model
load between the consensus and scoring phases) pays a full model-load penalty.

### 0.6 Stronger-model sensitivity check (robustness, not a headline result)

V3 vs V5 on **200 MedQA questions** (fixed seeded subset, disjoint from or a
subset of the 500-question dev/test slice — pin exactly which in the run
config) via the **free Gemini Flash tier**, to show the V3→V5 relative gain is
not an artifact of the 7B local model. Reported in the paper as a robustness
check alongside the main grid, not as a headline number, and not subject to
the local-hardware throughput assumptions above (it's API-latency- and
free-tier-rate-limit-bound instead — treat as a background job, not a timed
benchmark). See ADR-008.

**Also record for B3 in this check (added post-Phase-3):** `verifier_decision_rate`
and the `n_supported` distribution, not just accuracy. Phase 3's near-zero
verifier decision rate (§5.4) is evidence for C2's binary bottleneck only if
it is not merely an artifact of the 7B model's SUPPORTED-threshold calibration
— a stronger model must be checked. Tracked under the roadmap's pre-Phase-7
fixes.

## 1. Correctness
- **Accuracy** — exact option match (rank-1 prediction).
- **Macro-F1** — over option positions / classes; the reported headline for
  PubMedQA (class-imbalanced yes/no/maybe).
- Paired bootstrap (10k resamples) vs. the nearest ablation; report p-values.

### 1.0 Pre-registered analysis plan (AUDIT 2026-09-06 §6.7)

**Registered 2026-09-07, before any Phase 7 grid run exists.** Committed at the
Phase 6 checkpoint (commit `ca8f8df`, `docs/05_DECISIONS.md` ADR-030); the grid
launches Wednesday 2026-09-09. Multiple comparison discipline is fixed *here and
now* precisely
because deciding it after seeing grid numbers is worthless: ADR-012 already ran
3 uncorrected comparisons and ADR-018 ran 2, and the grid multiplies that.

**The single confirmatory comparison: V3 vs V5, accuracy, on MedQA (n=500,
seed 42).**
- One test. The metric is **accuracy, singular** — not accuracy *and* Macro-F1
  *and* ECE (that would smuggle a family back into the confirmatory slot).
- **α = 0.05, two-sided.** Two-sided is not negotiable: Phase 4 (`§5.5`
  Finding 3: the uncalibrated agent stack at 0.35 vs equal-budget hybrid RAG at
  0.60) and Phase 5 (ADR-028: temperature scaling alone matches the evidential
  features) both point at V5 possibly *not* beating V3, so the direction may not
  be assumed.
- Method: `src.retrieval.metrics.paired_bootstrap_ci` +
  `paired_bootstrap_p` (10k resamples, seed 42), paired on question id.
- **What V3 = V5 means, written before the number is seen:** if the confirmatory
  test shows no significant accuracy difference, *that is the reported finding* —
  the full tri-modal + multi-agent + evidential-calibration stack does not beat
  tri-modal retrieval alone on MedQA at this scale. Given the fusion tie
  (ADR-012), the Conflict agent firing zero times (ADR-025), and temperature
  scaling matching the evidential features (ADR-028), a null here is a **live
  outcome the paper reports as a result, not a disappointment to re-frame**. The
  contribution in that case is the negative result plus its mechanism (C1/C3 are
  already written as diagnostic, not triumphant — `docs/00` §3).

**Everything else is exploratory** and reported with Holm–Bonferroni correction
across the pre-registered exploratory family below (family-wise α = 0.05,
`src.retrieval.metrics.holm_bonferroni`). **Recomputed 2026-09-08 (ADR-035,
ADR-036), before the grid ran.** The first grid slot runs 5 MedQA cells —
**V1, V3, V5, V4−conflict, B3** (n=500 test, seed 42). **V4−audit is deferred**
to a later slot (it needs the binary-audit implementation, ADR-036) and its
pairs are *not* registered here — they are re-registered before that run. One
zero-GPU derived analysis, **post-hoc-binary** (V5's SupportMatrices with
`SUPPORT`→`SUPPORTED`, `CONTRADICT`∪`NEUTRAL`→`NOT_SUPPORTED`, `s⁻=0`,
re-ranked — ADR-036 Constraint B), joins the family now.

| # | Pair | Metrics tested | count |
|---|---|---|---|
| a | V1 vs V3 | Acc, Macro-F1, halluc-rate | 3 |
| b | V3 vs V5 | Macro-F1, halluc-rate, P@1, P@2, MRR | 5 |
| d | V4−conflict vs V5 | Acc, Macro-F1, halluc-rate, P@1, P@2, MRR | 6 |
| e | V1 vs V5 | Acc, Macro-F1, halluc-rate | 3 |
| g | **V3 vs B3** | Acc, Macro-F1, halluc-rate | 3 |
| h | **V5 vs B3** | Acc, Macro-F1, halluc-rate | 3 |
| i | post-hoc-binary vs V4−conflict | Acc, Macro-F1, halluc-rate, P@1, P@2, MRR | 6 |
| j | post-hoc-binary vs V5 | Acc, Macro-F1, halluc-rate, P@1, P@2, MRR | 6 |

**Family size m = 35.** Holm–Bonferroni over all 35. Fixed contingencies (so
`m` is never chosen after seeing a number):
- **b** has no ECE cell — V3 has no probability vector (this is why b is 5, not
  6); the V5-vs-raw-logprob ECE comparison already lives in §5.6, outside this
  family.
- **g/h** carry no P@K/MRR/ECE — B3 emits one option, not a ranking.
- **i** isolates the graded label cleanly (both sides are *uncalibrated* agent
  rankings — `(s⁺−s⁻)` for V4−conflict, `s⁺`-only for post-hoc-binary). **j**
  is the comparison ADR-036 names but it **confounds** "lost the graded label"
  with "lost calibration" (post-hoc-binary is uncalibrated, V5 is calibrated) —
  reported with that caveat stated.
- If **V4−conflict is dropped for budget** (ADR-035 — the Conflict agent fires
  0×, ADR-025), pairs **d and i leave the family** (−12) → **m = 23**. This one
  substitution is pre-authorised; no other cell may be dropped, and B3 and the
  endpoints (V1/V3/V5) are never dropped.
- P@K/MRR appear only where both sides produce a ranking. V4−conflict and
  post-hoc-binary rankings are uncalibrated — flagged as such wherever reported
  (ADR-029).

**Deferred, to be re-registered *before* the V4−audit slot runs** (ADR-036):
V4−audit vs V5, V4−audit vs V4−conflict, and V4−audit(prompt) vs post-hoc-binary
(the "does the binary *prompt* change the auditor's reasoning" test).

Any comparison **not** in this table that is run after seeing grid numbers is
reported as unplanned, excluded from the Holm family, and flagged as post-hoc.

The halluc-rate tests in the family use the 0.5 entailment threshold; ADR-031
(§5.7) established that threshold is metric-strict, so alongside the family
tests the V3/V5 runs also report halluc-rate at 0.3 / 0.5 / 0.7 as a
robustness display — not a paired test, not in the Holm family, there to show
the V3-vs-V5 *ordering* on halluc-rate does not flip with the threshold.

**Post-grid accounting (added 2026-09-10, ADR-040 — an audit of the
registration above, not a change to it).** Pre-registered family size
**m = 35**; actually tested **m = 31**. Four cells are untested, each for a
reason fixed by the data's structure rather than by any outcome:
- **g / h halluc-rate (2) — not computable.** B3 has no claim-level NLI data
  (§5 table note §).
- **b P@2 and b MRR (2) — not applicable.** V3 is `ranking_source:
  single_prediction`: a 1-item list on all 500 questions. So its P@2 =
  P(gold in a 1-item list) = accuracy = 0.546, and its MRR = 1 × accuracy + 0
  = 0.546. Against V5's 4-item ranking (P@2 0.582, MRR 0.571) the comparison
  scores list length, not ranking quality. The contingency rule above ("P@K/MRR
  appear only where both sides produce a ranking") already excludes both
  cells; the b row contradicted the rule it sits under.

Holm is re-run over the 31. **No significance verdict changes under either
m.** The same 18 cells are rejected at m = 35 (the two not-computable cells
entered as p = 1.0 placeholders, the most conservative placement), at m = 33
(the previous reading, which tested the not-applicable cells) and at m = 31.
All 18 rejections have bootstrap p = 0 — none of 10,000 resamples crossed
zero; the smallest attainable non-zero p is 0.0002 — which clears even the
strictest threshold either family size imposes (Bonferroni 0.05/35 =
0.00143). The smallest p not rejected is 0.0476 (d MRR), above every threshold
it faces under both m. Source: `results/phase7_grid/exploratory_family.json` →
`holm_sensitivity`. b P@1 stays: top-1 is well-defined for a single
prediction. But it is the same per-question comparison as the confirmatory
accuracy test, so it adds no independent evidence (dropping it as well,
m = 30, leaves the other 17 rejections unchanged).

### 1.0b Unplanned exploratory family — V0, V2 and V5-refit (V0/V2 registered 2026-09-10, ADR-040; V5-refit added 2026-09-11, ADR-043 — each before any run of this family exists)

V0 (zero-shot, no retrieval) and V2 (`{dense, lexical}`, no KG) are
**pre-specified grid rows** — §0.3's full 8-variant MedQA ablation — that the
`docs/SCHEDULE.md` cut left unrun. **Their tests were never pre-registered**:
the m=35 family above was registered without them. Their comparisons therefore
form a **separate, explicitly unplanned exploratory family**, registered here
before either run exists, Holm-corrected **within itself**, and **never merged
into the m=35 family** (merging would move that family's thresholds after its
outcomes are known). The distinction is the point: the *rows* were planned;
the *tests* were not. They answer two questions the paper cannot omit — does
retrieval help at all (V0), and does the KG add anything over `{dense,
lexical}` (V2 vs V3, the isolation C1 has always needed and never had; the
m=35 family's pair a, V1 vs V3, changes two modalities at once).

**V5-refit (added 2026-09-11, ADR-043).** V5-refit is the disclosed
post-hoc correction of V5's scoring rule: α, β, γ ≥ 0; a neutral no-signal
rerank value; a seeded tie-break; s⁺ / s⁻ counted from the cell list
(ADR-042). It was refit on the Phase 5 train slice and frozen in
`results/phase7_v5_refit/refit_spec.json` before any test evaluation. Its
three comparisons join this family, not the m=35 one. U6 asks whether
correcting the rule changes V5; U7 and U8 place it against the uncalibrated
margin and against retrieval alone. Adding them now moves no threshold after
an outcome is known, because none of V0, V2 or V5-refit has run. V5 stays
the pre-registered headline, and V5-refit is reported beside it, never in
its place.

| # | Pair | Metrics tested | count |
|---|---|---|---|
| U1 | V0 vs V1 | Acc, Macro-F1 | 2 |
| U2 | V0 vs V3 | Acc, Macro-F1 | 2 |
| U3 | V0 vs V5 | Acc, Macro-F1 | 2 |
| U4 | **V2 vs V3** | Acc, Macro-F1, halluc-rate | 3 |
| U5 | V1 vs V2 | Acc, Macro-F1, halluc-rate | 3 |
| U6 | **V5-refit vs V5** | Acc, Macro-F1 | 2 |
| U7 | V5-refit vs V4−conflict | Acc, Macro-F1 | 2 |
| U8 | V5-refit vs V3 | Acc, Macro-F1 | 2 |

**m_U = 18**, Holm–Bonferroni at family-wise α = 0.05 within these 18 only;
same method as the m=35 family (paired bootstrap, 10k resamples, seed 42,
two-sided; Macro-F1 and halluc-rate by joint question resampling). Fixed
contingencies, so m_U is never chosen after seeing a number:
- **No P@K / MRR / ECE cell anywhere.** V0–V3 are all `single_prediction`
  (the pair-b lesson above). U6–U8 are registered on accuracy and Macro-F1
  only; V5-refit's P@K, MRR, ECE and halluc-rate are displayed, not tested.
- **No halluc-rate cell for any V0 pair.** V0 retrieves and cites nothing, so
  every claim is ungrounded by construction.
- **If V0, V2 or V5-refit does not produce a valid run** (variant, split
  test, n = 500, seed 42, code identical to `41f7e14`, hallucination scored;
  for V5-refit, also derived from the grid V5 run and carrying the frozen
  spec's digest — all checked by the analysis script), that variant's pairs
  leave the family (−6 each). Pre-authorised; no other cell may be dropped.
- Results are reported under the "unplanned" label, never pooled with, and
  never used to reinterpret, the m=35 family.

Runs, in this order: `scripts/run_phase7_v5_refit.ps1` (re-scores V5's saved
SupportMatrices; zero LLM calls; refuses to launch unless the spec, the rule
code, ADR-043 and this table are committed); then
`scripts/run_phase7_v0_v2.ps1 -Only V0` and `-Only V2`. All are pinned to
`41f7e14`'s code, and `-DryRun` runs the checks only. Analysis: `uv run python -m
scripts.analysis.unplanned_v0_v2_family --v0 <V0 run dir> --v2 <V2 run dir>
--v5-refit <V5-refit run dir>` → `results/phase7_grid/unplanned_v0_v2_family.json`.

### 1.0c POST-HOC tier-boundary family (identified 2026-09-11 after the grid's two-tier structure was visible; ADR-046)

**This family is post-hoc.** Its pairs were chosen after every Phase 7 point
estimate, and the two-tier structure in §5.0a, were already known. It is
reported separately. It is never pooled with the m=35 family or the U1–U8
family, never used to reinterpret either, and **never cited as
pre-registered support**. It exists because §5.0a's central claim, one
significant step at the retrieval/agent boundary, otherwise rests on
inference. The boundary step on the component chain, V3 → V4−conflict,
appears in no earlier family. The drop was established only by cross-tier
pairs against V5 / V5-refit.

| # | Pair | Metrics tested | count |
|---|---|---|---|
| X1 | **V3 vs V4−conflict** | Acc, Macro-F1, halluc-rate | 3 |
| X2 | V0 vs V4−conflict | Acc, Macro-F1 | 2 |
| X3 | V0 vs B3 | Acc, Macro-F1 | 2 |

**m_X = 7**, fixed, together with the script
(`scripts/analysis/posthoc_boundary_family.py`), and committed before any of
its p-values was computed. Holm–Bonferroni within these 7 only; same method
as §1.0 / §1.0b (paired bootstrap, 10k resamples, seed 42, two-sided). There
is no halluc-rate cell for a V0 pair (by construction, §1.0b), and B3 has no
claim-level NLI data. Holm decides "significant". An unadjusted CI that
excludes zero is reported as "not significant after Holm", never as a
finding. Not in the family, and still untested: V0 vs V2, and B3 vs
V4−conflict. They were not added after the fact.
Output: `results/phase7_grid/posthoc_boundary_family.json`.

### 1.0d Stronger-model sensitivity check — the §0.6 deferred item (registered 2026-09-12, ADR-047, before any Gemini API call)

**What this measures, stated first because it is the thing most easily
overclaimed: model dependence, not independent evidence about MedQA.** Every
cell re-runs the *same code* on the *same 200 questions* with the *same
prompts, retrieval config, scoring rule and agent graph* — only the generator
changes. A result here says "the Phase 7 finding does / does not survive a
stronger generator". It is **not** a second, independent measurement of any
system's MedQA accuracy, it does not add evidence for or against any
pre-registered Phase 7 claim, and it is **never pooled with, or used to
re-threshold, the m=35, U1–U8 or m_X=7 families.** Reported as a robustness
check (§0.6, ADR-008), never as a headline row.

**Model.** `gemini/gemini-3.8-flash` via LiteLLM, Google AI Studio free tier
(`GEMINI_API_KEY`). Fallbacks, in order, if the free tier does not serve it:
`gemini/gemini-2.5-flash`, then `gemini/gemini-3.5-flash-lite`. The **exact
model id, API version and UTC run date** are stamped into each run's own
`results/<run_id>/config.yaml` (`llm.fallback_chain[0].model`,
`llm.api_version`, `llm.model_resolved_utc`) — a hosted endpoint can change
behaviour behind a fixed name, so the date is part of the provenance, not
decoration.

**Questions — the first 200 of the MedQA test-500 in seeded order (seed 42),
pinned by explicit id.** Frozen in
`results/phase7_gemini/question_ids_200.json` with a sha256 the launcher
re-verifies before it runs. These are the same questions the 7B cells ran, so
every comparison is paired within question.

> **A trap this check had to route around, recorded because it would have
> silently invalidated the pairing.** `--limit 200` does **not** give the first
> 200 of the 500. `load_question_sample` draws
> `random.Random(42).sample(range(n), limit)`, and `random.sample` at a
> different `k` is a different draw, not a prefix — measured, the two sets
> overlap on only **190 of 200** ids, with 10 questions lying outside the 500
> entirely and never run by any 7B cell. `load_question_sample_excluding`'s
> docstring already warned about this from the other direction. The check
> therefore selects by resolved `question_id` (`--question-ids`), never by
> re-sampling.

**Cells: V0, V3, B3, V5**, recorded as `V0-gemini` / `V3-gemini` /
`B3-gemini` / `V5-gemini`. Not V1/V2 (the retrieval tier already ties, §5.0a),
not V4−conflict (ties with V5). The `-gemini` suffix plus n=200 keeps these
runs out of the §5 grid table twice over: `scripts/report.py`'s
`is_grid_eligible` requires n=500, and `canonical_variant` does not recognise
a suffixed name.

**The single confirmatory test: V3-gemini vs V5-gemini, accuracy, n=200.**
One test, one metric. Paired bootstrap, 10k resamples, seed 42, two-sided,
α = 0.05 — the direct replication of the MedQA confirmatory test (§1.0).

**Exploratory family**, Holm–Bonferroni **within this family only**:

| # | Pair | Metrics tested | count |
|---|---|---|---|
| G1 | V0-gemini vs V3-gemini | Acc, Macro-F1 | 2 |
| G2 | V0-gemini vs V5-gemini | Acc, Macro-F1 | 2 |
| G3 | V3-gemini vs B3-gemini | Acc, Macro-F1 | 2 |
| G4 | V5-gemini vs B3-gemini | Acc, Macro-F1 | 2 |

**m_G = 8**, fixed here and in
`scripts/analysis/gemini_sensitivity_family.py` before any of its p-values
exists. No halluc-rate cell (B3 has no claim-level NLI data; a V0 pair is
ungrounded by construction — §1.0b), no P@K/MRR/ECE cell (V0/V3/B3 are
`single_prediction`). If a cell fails to produce a valid run, its pairs leave
the family (−2 each); pre-authorised, nothing else may be dropped.

**Descriptive only — displayed, never tested, never Holm-corrected:** B3's
`verifier_decision_rate` and `n_supported` distribution; V5's ECE and ranking
metrics against the analytic chance row; per-question 7B-vs-Gemini agreement;
and the V3-gemini repeat-run agreement below.

**Limitation — these runs are not seed-deterministic (ADR-047).** Every Phase 7
grid cell is seeded and bit-reproducible. These are not: the Gemini API exposes
no `seed` parameter (litellm raises `UnsupportedParamsError`), so the only
determinism lever available is `temperature`, pinned explicitly to **0.0** —
the value the 7B cells used — in `configs/gemini_base.yaml` and recorded in
each run's own `config.yaml`. Greedy decoding is not a reproducibility
guarantee, and a hosted endpoint can change behind a fixed name. **These runs
are never described as reproducible**, and the check is therefore a robustness
signal, not a replication. Also in `docs/00` §7.

**Registered now, descriptive only: the V3-gemini repeat.** After the four
cells complete, V3-gemini is run a second time (`configs/gemini_v3_repeat.yaml`,
variant `V3-gemini-rep2`) and the **per-question agreement between the two
runs** is reported — turning "not deterministic" into a number. V3 is the
cheapest cell (~1 call/question). This is **never tested, never
Holm-corrected, not part of m_G = 8, and never a claim**; it bounds how much of
any Gemini-vs-7B difference could be run-to-run noise. The repeat config's only
change is a separate `cache_dir`: on the shared cache the second run would
replay the first run's replies and report a meaningless 1.000, so
`gemini_sensitivity_family.py` refuses a repeat whose `llm_cache_hits` is
nonzero.

**Pre-registered predictions.** Recorded verbatim, each with what a failure
would mean:

- **P1: the two-tier structure replicates — V3 beats V5 on accuracy by ≥0.15,
  significant.** *Failure would mean* the V3 > V5 step is a property of the 7B
  generator rather than of the architecture, which would undercut §5.0a's
  central two-tier claim and force it to be restated as model-specific.
- **P2: every cell scores above its Qwen2.5-7B counterpart on the same 200
  questions.** *Failure would mean* the pipeline does not simply improve with a
  stronger generator — a cell that gets *worse* would indicate the scaffold is
  tuned to 7B-specific behaviour rather than being generator-agnostic.
- **P3: B3's `verifier_decision_rate` rises above the 7B's 0.19 but stays
  below 0.50.** *Failure below 0.19* would mean the binary bottleneck is not a
  weak-model artifact at all; *failure above 0.50* would mean it largely is
  one, and the C2 verifier-collapse finding — the paper's strongest result —
  would have to be restated as a small-model phenomenon.

**The 7B numbers these are judged against, on the same pinned 200** (computed
from the committed grid runs' `predictions.jsonl`, not the n=500 headlines):
V0 0.540, V3 0.505, B3 0.505, V5 0.325; V3 − V5 = **+0.180**; B3
`verifier_decision_rate` = **0.185** (0.186 at n=500). These are the paired
baselines, and P1/P2/P3 are scored against them.

Analysis: `uv run python -m scripts.analysis.gemini_sensitivity_family` →
`results/phase7_gemini/gemini_sensitivity_family.json`.

## 2. Factual fidelity
- **RAGAS faithfulness** — fraction of generated claims entailed by the retrieved
  context. Pin the RAGAS version; run with a fixed judge model recorded in config.
- **Context precision / recall** — retrieval quality independent of the generator.

## 3. Hallucination rate
- Split the answer rationale into atomic claims (`src/eval/nli.py:split_claims`
  — regex, decimals/abbreviations guarded, no LLM).
- For each claim, run NLI (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`)
  against its **cited** evidence.
- `hallucination_rate = ungrounded_claims / total_claims`, where ungrounded =
  no cited evidence reaches ENTAILMENT ≥ 0.5. Reported under **both**
  denominators (predicted option / all SUPPORT cells; ADR-029).
- **Relative comparison only (ADR-031).** The control in §5.7 shows this
  criterion marks ~70% of clinical rationale claims not-entailed *regardless of
  pipeline* (B2-equal-budget 0.741 vs V5 0.682 on the dev-20, both 0 uncited) —
  it is metric-strict. Use `hallucination_rate` to compare variants at a fixed
  threshold, never as an absolute "X% hallucinated" statement. Phase 7 adds a
  threshold sweep (0.3 / 0.5 / 0.7) on the V3/V5 runs to show the variant
  *ordering* is threshold-stable.
- Also report `invalid_citation_count` (cited id not in the bundle) — a strictly
  worse, unambiguous failure than a weak entailment.

## 4. Diagnostic ranking quality
- **Precision@K** (K = 1, 2) over the ranked option list.
- **MRR** of the gold option.
- **ECE** (15 bins) + reliability diagram over `p_1`.
- **Coverage–risk curve** from the abstention threshold τ.

## 5. Results table (fill only from `results/` — no hand-typed numbers)

**Filled 2026-09-10 from the Phase 7 grid, extended 2026-09-11 with V0, V2
and V5-refit, `python -m scripts.report` → `paper/tables/results_grid.md` /
`.tex`** (the generated table also carries a `Grounded` column; see its
two-definition note before comparing across groups). Grid per §0.3 and the
`docs/SCHEDULE.md` cut: **MedQA only**, seven cells ran —
**V0, V1, V2, V3, B3, V5, V4−conflict** (n=500 test slice, seed 42, all on
code identical to commit `41f7e14` on every run-affecting path; the five
original run dirs are pinned by exact name in `scripts/analysis/*`, never by
glob, the V0/V2/V5-refit dirs are passed by exact name to
`scripts/analysis/unplanned_v0_v2_family.py`, which validates each, and
`scripts/report.py` takes the latest *eligible* run per row and asserts its
code provenance). V0 and V2 ran via `scripts/run_phase7_v0_v2.ps1`; V2's
launcher attempt exited 1 on a transient CUDA OOM and the tabulated V2 run is
a clean retry (ADR-044). **V5-refit**, the disclosed post-hoc correction of
V5's scoring rule (ADR-043), has its own row beside V5 and never replaces
it: V5 is the pre-registered result and the headline. **V4−audit remains
deferred** (ADR-036, needs the binary-audit prompt implementation). **The V4
row stays `TBD`: no n=500 test run exists.** The Phase 3 dev-20 B0 number that previously filled the V0
row under a dagger is gone: a dev-20 number does not appear in a table of
n=500 test results, footnote or not, and `scripts/report.py` now admits only
n=500, seed-42, non-dev-split runs to any row (ADR-040). The **Random** row is
not a run. It is the analytic expectation of a uniformly random ranking of the
4 options (P@1 = 1/4, P@2 = 2/4, MRR = 25/48;
`results/phase7_grid/random_baseline.json`) — the floor every P@K / MRR
number in this table must be read against (§5.0a). Faithfulness (RAGAS, §2)
was never wired up — every cell in that column is `TBD`, not measured, not
implied by any other metric here. B3 is scored on accuracy / Macro-F1 only in
this grid — see the discovered-gap note below; it has no ranking (P@K / MRR /
ECE are `–`).

| Variant | Dataset | Acc | Macro-F1 | Faithfulness | Halluc. rate | P@1 | P@2 | MRR | ECE | run_id |
|---|---|---|---|---|---|---|---|---|---|---|
| **Random ranking (chance)** | any 4-option | 0.250 | – | – | – | 0.250 | 0.500 | 0.521 | – | `phase7_grid/random_baseline.json` (analytic) |
| V0 zero-shot | MedQA | 0.582 | 0.579 | – | 1.000¶ | TBD‡ | TBD‡ | TBD‡ | TBD | `medqa_V0_b24015a-dirty_20260911T080323Z` |
| V1 dense RAG | MedQA | 0.566 | 0.565 | TBD | 0.893 | TBD‡ | TBD‡ | TBD‡ | TBD | `medqa_V1_41f7e14_20260908T212655Z` |
| V2 hybrid | MedQA | 0.556 | 0.555 | TBD | 0.870 | TBD‡ | TBD‡ | TBD‡ | TBD | `medqa_V2_b24015a-dirty_20260911T104513Z` |
| V3 +KG | MedQA | 0.546 | 0.545 | TBD | 0.883 | TBD‡ | TBD‡ | TBD‡ | TBD | `medqa_V3_41f7e14-dirty_20260908T230633Z` |
| V4 +agents | MedQA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| V4−audit | MedQA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| V4−conflict | MedQA | 0.358 | 0.342 | TBD | 0.725 | 0.358 | 0.606 | 0.598 | TBD | `medqa_V4-conflict_41f7e14-dirty_20260909T220306Z` |
| V5 full | MedQA | **0.316** | 0.311 | TBD | 0.728 | 0.316 | 0.582 | 0.571 | 0.068 | `medqa_V5_41f7e14-dirty_20260909T103452Z` |
| V5-refit (post-hoc refit, ADR-043) | MedQA | 0.292 | 0.290 | TBD | 0.736 | 0.292 | 0.530 | 0.551 | 0.054 | `medqa_V5-refit_b24015a_20260911T074318Z` |
| B3 binary verifier | MedQA | 0.538 | 0.537 | – | TBD§ | – | – | – | – | `medqa_B3_41f7e14-dirty_20260909T023658Z` |
| V3 +KG | MedMCQA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| V5 full | MedMCQA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| V3 +KG | PubMedQA | TBD | TBD | TBD | TBD | – | – | – | TBD | TBD |
| V5 full | PubMedQA | TBD | TBD | TBD | TBD | – | – | – | TBD | TBD |

¶ **By construction, not a measurement.** V0 retrieves nothing and cites
nothing (`src/eval/baselines.py`, the CLAUDE.md grounding carve-out for the
zero-shot floor), so every claim it makes is ungrounded under §3's criterion
and its `grounded_rate` is 0.000. No V0 pair tests halluc-rate (§1.0b).

‡ V0, V1, V2 and V3 are `ranking_source: single_prediction` — `src/eval/harness.py`'s
`score_run` (`_RANKED_SOURCES = ("calibrated_softmax",
"uncalibrated_support_margin")`) deliberately leaves P@1/P@2/MRR/ECE null for
this ranking source (a single predicted option is not a ranking). This is an
aggregate-table convention, not a data gap. The same fact governs the
exploratory family: pair b tests V3 vs V5 on **P@1 only** (top-1 is
well-defined for a single prediction). b's P@2 and MRR are **not
applicable** — V3's 1-item list makes both equal its accuracy (0.546) — see
§1.0's post-grid accounting and §5.0a.

§ **Discovered gap, not a Phase 7 run failure.** B3 (`scripts/run_baselines.py`)
never computes `claim_records` / resolves cited-evidence text, so it has no
claim-level NLI data at all — confirmed from its `predictions.jsonl` schema
(raw `rationale` + citation IDs, no resolved evidence text) and its
`metrics.json` (no `hallucination` key). Recomputing it now would mean
re-running retrieval to resolve citation IDs back to text, out of scope for a
read-only analysis pass. See §5.0a's exploratory-family note (pairs g/h drop
their halluc-rate sub-test for the same reason).

Also report, per variant: mean LLM calls/question, tokens/question, wall-clock,
KG-hit rate, and the failure-mode counters from `docs/01_ARCHITECTURE.md`.
**Wall-clock is only available for the grid cells a runner itself
timed** (V1 50.7 min, V3 99.7 min, B3 207.7 min — `results/phase7_grid/manifest.json`;
V0 5.1 min — `results/phase7_grid/v0_v2_launch.json`);
V5 and V4−conflict were run manually after the runner failed to launch them
(ADR-039), and the tabulated V2 run is a retry launched outside the runner
(ADR-044), so none of the three has an independently captured duration — see
§5.0a. V5-refit makes zero LLM calls (it re-scores V5's saved
SupportMatrices).

### 5.0a Phase 7 grid: confirmatory test, exploratory family, and V5 diagnostics

**Grid provenance and manifest.** `results/phase7_grid/manifest.json`
(rebuilt by `scripts/analysis/rebuild_phase7_manifest.py`, never by hand)
pins the five run dirs by exact name and records that V5 and V4−conflict's
runner attempts failed silently (`exit 2`, no log) and were completed
manually from the same launch commit — investigated in **ADR-039**. All five
`-dirty` git SHA stamps are resolved in **ADR-038**: every dirty run's own
`git_dirty_files` list names only untracked `results/` output from earlier
cells in the same grid, never a tracked source path — commit `41f7e14` is
authoritative for all five cells.

**Confirmatory test (the one pre-registered comparison, §1.0).** V3 vs V5,
accuracy, MedQA test n=500, paired bootstrap 10k resamples seed 42, two-sided
(`scripts/analysis/confirmatory_v3_v5.py` →
`results/phase7_grid/confirmatory_v3_v5.json`):

| | V3 | V5 | diff (V3−V5) | 95% CI | p | verdict |
|---|---|---|---|---|---|---|
| accuracy | 0.546 | 0.316 | **+0.230** | **[+0.172, +0.288]** | <0.0001 | **V3 significantly > V5** |

**This is not the null the §1.0 pre-registration flagged as a live
possibility — it is a large, significant loss.** The full tri-modal +
multi-agent + evidential-calibration stack (V5) scores 23 accuracy points
*below* tri-modal retrieval alone (V3) on MedQA test at this scale, with a CI
that excludes zero by a wide margin. Per the pre-registration's own framing
(§1.0), a null here would have been reported as "the stack does not beat
retrieval alone"; the actual result is stronger and less comfortable than
that — the stack actively *regresses* accuracy relative to the tri-modal
baseline it is built on top of. Read together with Phase 4 §5.5 Finding 3
(uncalibrated agent stack 0.35 vs. equal-budget hybrid RAG 0.60 on the dev-20)
and Phase 5 (temperature scaling alone matches the evidential features'
ECE), this is the same direction of finding, now confirmed at n=500 test
with a real significance test rather than a raw point estimate.

**Exploratory family (Holm-Bonferroni, m=31 of 35 pre-registered).**
`scripts/analysis/exploratory_family.py` →
`results/phase7_grid/exploratory_family.json`, 10k resamples, seed 42,
two-sided, α=0.05. Four pre-registered cells are untested, each for a reason
fixed by the data's structure before any p-value in the family was read
(§1.0 post-grid accounting, ADR-040):
- **Not computable (2).** Pairs g and h drop their halluc-rate sub-test: B3
  has no claim-level NLI data (see the §5 table's `§` note). A
  data-availability finding disclosed up front, not a post-hoc exclusion.
- **Not applicable (2).** Pair b's P@2 and MRR. V3 emits a 1-item list on
  500/500 questions, so V3's P@2 = MRR = accuracy = 0.546 — P@2 asks whether
  gold is in a list with no second slot, and MRR is 1 × accuracy + 0. V5's
  4-item ranking earns rank-2 credit V3 cannot (V5 P@2 0.582, MRR 0.571), so
  the previous table's "V3 − V5 = 0.546 − 0.582 on P@2, 0.546 − 0.571 on MRR"
  measured list length, not ranking quality. Excluded from Holm exactly as
  the B3 halluc-rate cells are; §1.0's own rule ("P@K/MRR appear only where
  both sides produce a ranking") already excluded them.

The 31 are Holm-corrected together (not per-pair). **No verdict changes**:
the reject set — the same 18 cells — is identical at m = 35, 33 and 31
(`holm_sensitivity`, §1.0).

All numbers below are machine-generated from `results/phase7_grid/exploratory_family.json` (`scripts/analysis/exploratory_family.py`) — verified against that file, not re-typed by estimate.

| pair | comparison | metric | point diff | 95% CI | p | Holm rank | Holm reject |
|---|---|---|---|---|---|---|---|
| a | V1 vs V3 | accuracy | +0.020 | tie | 0.1250 | 23 | no |
| a | V1 vs V3 | macro-F1 | +0.020 | tie | 0.1096 | 22 | no |
| a | V1 vs V3 | halluc-rate | +0.010 | tie | 0.3494 | 28 | no |
| b | V3 vs V5 | macro-F1 | +0.234 | [+0.177,+0.290] | <0.0001 | 1 | **yes** |
| b | V3 vs V5 | halluc-rate | +0.155 | [+0.110,+0.202] | <0.0001 | 2 | **yes** |
| b | V3 vs V5 | P@1 | +0.230 | [+0.172,+0.288] | <0.0001 | 3 | **yes** (= the confirmatory accuracy test; not independent) |
| b | V3 vs V5 | P@2 | not applicable — V3 P@2 = V3 acc = 0.546 (1-item list) vs V5 0.582 (4-item) | — | — | — | — |
| b | V3 vs V5 | MRR | not applicable — V3 MRR = V3 acc = 0.546 (1-item list) vs V5 0.571 (4-item) | — | — | — | — |
| d | V4−conflict vs V5 | accuracy | +0.042 | tie | 0.0666 | 20 | no |
| d | V4−conflict vs V5 | macro-F1 | +0.031 | tie | 0.1694 | 24 | no |
| d | V4−conflict vs V5 | halluc-rate | −0.002 | tie | 0.8888 | 31 | no |
| d | V4−conflict vs V5 | P@1 | +0.042 | tie | 0.0666 | 21 | no |
| d | V4−conflict vs V5 | P@2 | +0.024 | tie | 0.3096 | 27 | no |
| d | V4−conflict vs V5 | MRR | +0.027 | [+0.000,+0.054] | 0.0476 | 19 | no (Holm step-down already stopped at rank 19) |
| e | V1 vs V5 | accuracy | +0.250 | [+0.192,+0.306] | <0.0001 | 4 | **yes** |
| e | V1 vs V5 | macro-F1 | +0.254 | [+0.196,+0.309] | <0.0001 | 5 | **yes** |
| e | V1 vs V5 | halluc-rate | +0.165 | [+0.120,+0.211] | <0.0001 | 6 | **yes** |
| g | V3 vs B3 | accuracy | +0.008 | tie | 0.2588 | 26 | no |
| g | V3 vs B3 | macro-F1 | +0.008 | tie | 0.1916 | 25 | no |
| g | V3 vs B3 | halluc-rate | not computable | — | — | — | — |
| h | V5 vs B3 | accuracy | −0.222 | [−0.278,−0.166] | <0.0001 | 7 | **yes** (B3 > V5) |
| h | V5 vs B3 | macro-F1 | −0.226 | [−0.282,−0.169] | <0.0001 | 8 | **yes** (B3 > V5) |
| h | V5 vs B3 | halluc-rate | not computable | — | — | — | — |
| i | post-hoc-binary vs V4−conflict | accuracy | −0.140 | [−0.196,−0.086] | <0.0001 | 9 | **yes** (V4−conflict > phb) |
| i | post-hoc-binary vs V4−conflict | macro-F1 | −0.137 | [−0.192,−0.081] | <0.0001 | 10 | **yes** |
| i | post-hoc-binary vs V4−conflict | halluc-rate | +0.013 | tie | 0.6068 | 29 | no |
| i | post-hoc-binary vs V4−conflict | P@1 | −0.140 | [−0.196,−0.086] | <0.0001 | 11 | **yes** |
| i | post-hoc-binary vs V4−conflict | P@2 | −0.148 | [−0.212,−0.084] | <0.0001 | 12 | **yes** |
| i | post-hoc-binary vs V4−conflict | MRR | −0.104 | [−0.142,−0.066] | <0.0001 | 13 | **yes** |
| j | post-hoc-binary vs V5 | accuracy | −0.098 | [−0.140,−0.056] | <0.0001 | 14 | **yes** (V5 > phb) |
| j | post-hoc-binary vs V5 | macro-F1 | −0.106 | [−0.148,−0.064] | <0.0001 | 15 | **yes** |
| j | post-hoc-binary vs V5 | halluc-rate | +0.011 | tie | 0.6106 | 30 | no |
| j | post-hoc-binary vs V5 | P@1 | −0.098 | [−0.140,−0.056] | <0.0001 | 16 | **yes** |
| j | post-hoc-binary vs V5 | P@2 | −0.124 | [−0.176,−0.072] | <0.0001 | 17 | **yes** |
| j | post-hoc-binary vs V5 | MRR | −0.077 | [−0.105,−0.049] | <0.0001 | 18 | **yes** |

**Reading, stated plainly.** V5 loses to every real baseline it should beat
to support C1–C3 (V1, V3, B3 all significantly beat V5 on accuracy/macro-F1;
pair **a**, V1 vs V3, is itself a tie — the KG modality adds nothing
detectable even at the retrieval-only level, consistent with C1's existing
diagnostic framing). Pair **d** (V4−conflict vs V5, i.e. the calibration
layer's own effect, holding the agent stack fixed) is a **tie on accuracy**
(uncorrected p=0.067, already short of α=0.05) and its only nominally-below-α
cell (MRR, p=0.048) does not survive Holm correction — **calibration does not
measurably help OR hurt accuracy/ranking here; it is a wash**, distinct from
and more precise than diagnostic (a) below. Pairs **i** and **j**
(post-hoc-binary vs. the two real agent variants) show post-hoc-binary
significantly *worse* on every rank-sensitive metric — dropping the
CONTRADICT/NEUTRAL-vs-SUPPORT distinction (forcing s⁻=0) costs accuracy, not
just calibration quality — but **halluc-rate is a tie in both i and j**, so
the binary collapse's damage is concentrated in ranking quality, not in
grounding.

**Diagnostics on V5's collapse** (`scripts/analysis/v5_diagnostics.py` →
`results/phase7_grid/v5_diagnostics.json`, read-only over `predictions.jsonl`,
no new runs):

**(a) "Hypothesis agent's own stated answer" accuracy — not computable, and
the real substitute is precise.** The Hypothesis agent does not propose or
rank a candidate answer, in the architecture spec or in the implementation:
its output is `[{option_id, claims, required_evidence_types}]`
(`src/agents/schemas.py: HypothesisOut`; `docs/01_ARCHITECTURE.md
#HypothesisAgent: "Out: [{option_id, claims:[str], required_evidence_types:[str]}]"`)
— pure claim decomposition per option, no answer signal exists anywhere in
the persisted state to measure. Reported as not computable rather than
substituting a fabricated proxy under that name. **The closest real,
available diagnostic** — V5's own SupportMatrices re-ranked by the raw
uncalibrated (s⁺−s⁻) margin instead of the calibrated softmax, i.e. the same
Hypothesis+Audit evidence V5 used, minus only the calibration/temperature
step — scores **0.358**, exactly V4−conflict's own accuracy from an
independent run using the identical uncalibrated-margin mechanism. This
precisely corroborates pair **d** above: the raw audit-evidence floor on this
grid is ~0.36 regardless of which run produced it, and calibration
(0.358→0.316) is a real but statistically-tied-at-this-n further loss on top
of an already-mediocre floor — **the defect is not concentrated in
calibration; it is already present in the audit evidence itself**, with
calibration adding a further (untested-significant) decrement.

**(b) ok-only (459) vs. all-500 accuracy — the collapse is not a status
artifact.** Accuracy on the 459 `ok`-status questions alone is **0.312**,
statistically indistinguishable from the all-500 figure (**0.316**) — the
37 `insufficient_evidence` and 4 `agent_error` rows are not concentrating the
failure; if anything the 41 non-`ok` rows score slightly *higher* (0.366, but
n=41, noisy). **V5's collapse is present throughout the `ok` majority of the
run, not an artifact of the questions that failed to complete normally.**

**(c) grounded_rate = 0.756 vs. V1/V3's 0.98–1.00 — fully explained, and
100% coincides with abstention.** `grounded` means the predicted option has
≥1 SUPPORT cell (`src/calibration/citations.py: cite()` draws only from
SUPPORT cells). Of the 122/500 ungrounded rows, **all 122 have
`pred_option_s_pos == 0`** (tautological given the definition, confirmed
directly) and **all 122 are also flagged `abstained`** (p1 < τ=0.546) — V5's
own confidence gate already flags every ungrounded prediction as low-
confidence, so grounding loss is not silent. Of those 122, **68 occurred on
questions where some OTHER option did have real SUPPORT evidence** — the
calibrated softmax (which also weighs `kg_support_count` and
`mean_rerank_score`, not `s⁺` alone) picked a zero-evidence option over one
the audit layer actually supported on 68/500 (13.6%) questions. Breakdown by
status: 83 of the 122 are `ok`, 37 are `insufficient_evidence` (where zero
support is expected), 2 are `agent_error`.

**Against chance — the direct test of C3's headline claim.** C3's headline
claim is *probabilistically ranked differential diagnoses*. On a 4-option
MCQ, a uniformly random ranking scores **P@1 0.250, P@2 0.500, MRR 0.521**
(analytic expectations; `scripts/analysis/random_baseline.py` →
`results/phase7_grid/random_baseline.json`, checked by a 200,000-trial seeded
simulation; the Random row of the §5 table). **V5 (P@1 0.316 / P@2 0.582 /
MRR 0.571) and V4−conflict (0.358 / 0.606 / 0.598) rank barely above
chance.** V5's full calibrated ranking beats a random shuffle of the four
options by +0.066 P@1, +0.082 P@2 and +0.051 MRR; V4−conflict by +0.108,
+0.106 and +0.077. The margins are real, not noise: a one-sample bootstrap of
each per-question excess over the exact chance value (10k, seed 42) rejects
chance for all six, Holm-corrected within the 9-cell display set (e.g. V5 MRR
excess 95% CI [+0.024, +0.077], P@1 [+0.026, +0.106]). So the finding is
"barely above chance", not "at chance". The post-hoc-binary ranking (0.218 /
0.458 / 0.494) is not distinguishable from chance, with all three point
estimates below it. **C3's headline claim — probabilistically ranked
differential diagnoses — therefore does not hold on MedQA at n=500: the
system does emit a probabilistically ranked differential with citations, but
that ranking is barely better than shuffling the four options at random.**
These chance comparisons are post-hoc — not pre-registered, outside both the
m=35 family and the unplanned V0/V2 family — and are shown only so that
"barely above chance" rests on a test rather than a raw inequality (CLAUDE.md
§2). (d) below gives the mechanism.

**(d) Decomposition of V5's calibrated score — what actually decides the
ranking** (`scripts/analysis/calibration_decomposition.py` →
`results/phase7_grid/calibration_decomposition.json`, ADR-041). Read-only.
The script first reproduces V5's stored ranking and p-vector from the saved
SupportMatrices and fitted weights on 500/500 questions, then splits
z = α·s⁺ − β·s⁻ + γ·kg + δ·rerank (p = softmax(z/T)) into its four terms.

*(i) Fitted weights* (`configs/v5.yaml`, equal field-by-field to the run's
`config.yaml`): **α = −0.0139**, β = 0.1098, γ = 0.2949, δ = 0.00772,
T = 0.2145. **α is negative: every SUPPORT cell lowers an option's score.**
Its sign was never identified — Phase 5's 200-resample bootstrap of the same
60-question fit already spanned zero ([−0.2777, +0.1934], §5.6). MedCPT rerank
scores on option rows with any SUPPORT/CONTRADICT cell are mostly positive
(median 6.16; 6.4% negative), so δ·rerank *raises* any option the audit
judged, in either direction, above one it never judged (which carries the 0.0
no-signal default).

*(ii)–(iii) What each term does to the ranking.* Within-question figures —
the ranking-relevant scope, since softmax is shift-invariant per question;
variance shares are Cov(term, z)/Var(z) and sum to 1:

| term | share of within-question Var(z) | within-question SD | V5 predictions changed if the term is removed | term alone: accuracy, ties broken at random |
|---|---|---|---|---|
| −β·s⁻ (contradictions) | 0.521 | 0.0510 | 225 / 500 | 0.316 |
| γ·kg (KG support) | 0.527 | 0.0497 | 20 / 500 | 0.2455 |
| α·s⁺ (support) | −0.050 | 0.0090 | 61 / 500 | 0.211 |
| δ·rerank (passage score) | 0.002 | 0.0169 | 186 / 500 | 0.259 |

Variance share and decision influence diverge, and decision influence is the
honest read. γ·kg's 53% share lives entirely in the 21 questions where any
option has KG support — the term is tied across all four options on the
other 479 — so removing it changes only 20 predictions. δ·rerank carries 0.2%
of the variance yet removing it changes 186: it is a continuous tiebreaker
among options the integer count terms leave tied.

*argmax z vs argmax α·s⁺ alone* (the requested comparison): they differ on
**341/500** questions under V5's own tie-break (V5's pick lies outside α·s⁺'s
tied maximisers on 288/500). Because α < 0, argmax α·s⁺ is the
**least**-supported option (accuracy 0.224). Against the sign-free
*most*-supported option (argmax s⁺), V5 differs on 259/500 and lies outside
the most-supported set on 183. By contrast, V5's pick lies inside the
fewest-contradiction set (argmax −β·s⁻) on **493/500**.

*(iv) The 68 zero-support wins* — which term supplied the winning margin over
the best-scoring supported option: **α·s⁺ 38, −β·s⁻ 26, δ·rerank 4,
γ·kg 0**. Decisive on their own (removing that term alone flips the outcome):
α·s⁺ 28, −β·s⁻ 20, δ·rerank 0, γ·kg 0; the other 20 needed two terms
jointly. The zero-support winner had s⁺ = s⁻ = 0 and z = 0 exactly on 63/68;
the best supported rival's z was negative on 68/68; and δ·rerank's mean
contribution to the winner's margin was −0.015 — on these questions the
reranker worked *against* the zero-support winner, not for it.

**Verdict on the passage-reranker hypothesis: half right.** δ·rerank does
dominate α·s⁺ — within-question SD 0.0169 vs 0.0090, and 186 vs 61
predictions changed on removal. But the evidential score is **not**
functionally a passage reranker: δ·rerank alone ranks at chance (0.259),
carries 0.2% of the ranking-relevant variance, and supplied the margin in
only 4 of the 68 zero-support wins. **The dominant term is −β·s⁻: V5 is
functionally a "fewest CONTRADICT cells" rule.** That term alone, with ties
broken at random, has expected accuracy 0.316 — V5's own figure. Its ties go
to δ·rerank and then to option-letter order: the top of z is exactly tied on
**170/500** questions (92 of them at z = 0), and the letter tie-break picks A
on 120 of those 170 against a gold split of A 38 / B 47 / C 48 / D 37 — on
those 170 questions option position, not evidence, breaks the tie.

**The mechanism behind the near-chance ranking, stated plainly.** The
calibrated score rewards the *absence of contradiction*, not the *presence of
support*. Positive evidence enters with a negative weight and a negative
variance share. An option the audit never judged scores exactly z = 0; a
supported option falls below that whenever its support penalty plus any
contradictions outweigh its reranker bonus — true of the best supported
option in all 68 zero-support wins. V5's top-ranked option has s⁺ = s⁻ = 0 on
**118/500** questions. "No contradiction found" cannot separate the gold
option from distractors the audit never examined, so the ranked differential
sits barely above chance.

*Not a license to retune.* The what-if accuracies above (0.330 with α·s⁺
removed; argmax s⁺ 0.348 with the letter tie-break, 0.332 with random ties)
are descriptive, computed on the **test** set, and not significance-tested.
None may be used to choose a replacement scoring rule — that would be
test-set tuning. Any refit (α constrained ≥ 0, a neutral no-signal imputation
for `mean_rerank_score`, a principled tie-break) is fit on a train slice and
re-registered before it touches test (ADR-041).

**Done that way: V5-refit (ADR-043).** V5-refit is fit on the Phase 5 train
slice and frozen in `results/phase7_v5_refit/refit_spec.json` before any
test evaluation; its tests are §1.0b's U6–U8. On train, once the no-signal
rerank value is neutral, the unconstrained fit already puts α > 0 (0.0059),
so no bound is active: V5's negative α was caused by the 0.0 no-signal
default. But α is still weakly identified (on its 0 bound in 0.495 of 200
bootstrap resamples), a CONTRADICT cell still outweighs a SUPPORT cell about
twelve-fold, and V5-refit is no more accurate than V5 on either train half.
The expectation registered before the test run is a near-chance ranking and
a tie on U6. **Both held** (U6 tie, ranking 0.292 / 0.530 / 0.551 against
chance 0.250 / 0.500 / 0.521). See the unplanned family results below.

**Also resolved (ADR-042):** the 2/500 questions whose `option_support`
disagrees with their cells are `agent_error` questions whose audit failed
mid-question, leaving `option_support` empty. This is one code path, not a
systematic s⁺ bug: the identity holds on the other 498 and on all 120 train
questions. With cell-derived counts, V5's accuracy is 0.318 (one prediction
changes), each count above moves by at most one, and no significance verdict
can change (`results/phase7_grid/spos_consistency.json`).

**Unplanned family results — U1–U8 (§1.0b; added 2026-09-11).**
`scripts/analysis/unplanned_v0_v2_family.py` →
`results/phase7_grid/unplanned_v0_v2_family.json`. V0, V2 and V5-refit all
produced valid runs, so no pair left the family: **m_U = 18 tested of 18
registered**. Paired bootstrap, 10k resamples, seed 42, two-sided,
Holm–Bonferroni within these 18 only. Never pooled with, and never used to
reinterpret, the m=35 family. The rows are generated from the JSON (diff =
A − B; "Holm thr." is the step-down threshold at that rank):

| pair | A vs B | metric | A | B | diff | 95% CI | p | Holm rank | Holm thr. | Holm reject |
|---|---|---|---|---|---|---|---|---|---|---|
| U1 | V0 vs V1 | accuracy | 0.582 | 0.566 | +0.016 | [-0.008,+0.042] | 0.2308 | 11 | 0.00625 | no |
| U1 | V0 vs V1 | macro-F1 | 0.579 | 0.565 | +0.015 | [-0.010,+0.040] | 0.2464 | 12 | 0.00714 | no |
| U2 | V0 vs V3 | accuracy | 0.582 | 0.546 | +0.036 | [+0.006,+0.066] | 0.0192 | 6 | 0.00385 | no |
| U2 | V0 vs V3 | macro-F1 | 0.579 | 0.545 | +0.034 | [+0.005,+0.064] | 0.0244 | 8 | 0.00455 | no |
| U3 | V0 vs V5 | accuracy | 0.582 | 0.316 | +0.266 | [+0.210,+0.324] | <0.0001 | 1 | 0.00278 | **yes** |
| U3 | V0 vs V5 | macro-F1 | 0.579 | 0.311 | +0.268 | [+0.211,+0.325] | <0.0001 | 2 | 0.00294 | **yes** |
| U4 | V2 vs V3 | accuracy | 0.556 | 0.546 | +0.010 | [-0.010,+0.030] | 0.3634 | 16 | 0.01667 | no |
| U4 | V2 vs V3 | macro-F1 | 0.555 | 0.545 | +0.010 | [-0.010,+0.029] | 0.3278 | 15 | 0.01250 | no |
| U4 | V2 vs V3 | halluc-rate | 0.870 | 0.883 | -0.013 | [-0.031,+0.004] | 0.1352 | 10 | 0.00556 | no |
| U5 | V1 vs V2 | accuracy | 0.566 | 0.556 | +0.010 | [-0.012,+0.034] | 0.4424 | 18 | 0.05000 | no |
| U5 | V1 vs V2 | macro-F1 | 0.565 | 0.555 | +0.010 | [-0.013,+0.033] | 0.3898 | 17 | 0.02500 | no |
| U5 | V1 vs V2 | halluc-rate | 0.893 | 0.870 | +0.023 | [+0.003,+0.043] | 0.0232 | 7 | 0.00417 | no |
| U6 | V5-refit vs V5 | accuracy | 0.292 | 0.316 | -0.024 | [-0.064,+0.016] | 0.2572 | 13 | 0.00833 | no |
| U6 | V5-refit vs V5 | macro-F1 | 0.290 | 0.311 | -0.021 | [-0.060,+0.018] | 0.2874 | 14 | 0.01000 | no |
| U7 | V5-refit vs V4-conflict | accuracy | 0.292 | 0.358 | -0.066 | [-0.116,-0.016] | 0.0114 | 5 | 0.00357 | no |
| U7 | V5-refit vs V4-conflict | macro-F1 | 0.290 | 0.342 | -0.051 | [-0.101,-0.004] | 0.0358 | 9 | 0.00500 | no |
| U8 | V5-refit vs V3 | accuracy | 0.292 | 0.546 | -0.254 | [-0.312,-0.198] | <0.0001 | 3 | 0.00313 | **yes** |
| U8 | V5-refit vs V3 | macro-F1 | 0.290 | 0.545 | -0.255 | [-0.313,-0.198] | <0.0001 | 4 | 0.00333 | **yes** |

**Four of 18 cells survive Holm, all four across the retrieval / agent
boundary** (U3, U8). The step-down stops at rank 5 (U7 accuracy, p = 0.0114
against 0.00357). Five more cells have an unadjusted 95% CI that excludes zero
but do not survive Holm: U2 accuracy and macro-F1 (V0 over V3), U7 accuracy
and macro-F1 (V4−conflict over V5-refit), and U5 halluc-rate (V1 above V2).
**Those five are not significant.** The JSON's per-cell `verdict` string
reads the unadjusted CI and so says "significantly" for them. The
family-level answer is `holm_reject`, which is `false` for all five.

**U1, the question this family exists to answer: V0 vs V1 is a tie.**
Zero-shot scores +0.016 above dense RAG (0.582 vs 0.566), 95% CI [−0.008,
+0.042], p = 0.2308, with macro-F1 the same. **The data support "retrieval
does not improve accuracy over zero-shot". They do not support "retrieval
hurts accuracy".** Read from dense retrieval's side, the CI puts its effect
between −4.2 and +0.8 accuracy points: at best, under one point of gain.
Against the tri-modal stack (U2, V0 vs V3, +0.036) the unadjusted CI
excludes zero, but the cell fails Holm. **"Tri-modal retrieval hurts" is not
established either.** U2 is the family's closest miss on a retrieval-only
pair, and it stays a miss.

**U4, C1's isolation finally measured: the KG adds nothing detectable over
{dense, lexical}.** V2 vs V3 ties on accuracy (+0.010 to V2, CI [−0.010,
+0.030], p = 0.3634), macro-F1 and halluc-rate. Read from the KG's side,
adding it moves accuracy by between −3.0 and +1.0 points. **U5** (dense vs
{dense, lexical}) ties on all three metrics after Holm.

**U6: V5-refit ties V5, as registered before the run (ADR-043).** Correcting
the scoring rule moves accuracy 0.316 → 0.292, CI [−0.064, +0.016],
p = 0.2572. **U8:** V3 beats V5-refit by +0.254, significant, so the
corrected rule does not rescue the agent stack. **U7:** V4−conflict's
uncalibrated margin sits 0.066 above V5-refit, significant unadjusted but not
after Holm. V5-refit's P@1 / P@2 / MRR (0.292 / 0.530 / 0.551) and ECE
(0.054) are displayed, not tested.

**The accuracy ordering across the grid, and its significance structure.**
Along the nested component chain, where each variant adds components to the
one before it, the accuracy point estimates fall at every step:

V0 0.582 → V1 (+dense) 0.566 → V2 (+lexical) 0.556 → V3 (+KG) 0.546 →
V4−conflict (+Hypothesis and Audit agents, uncalibrated margin) 0.358 → V5
(+Conflict agent, +calibration) 0.316.

The two off-chain variants sit where the chain predicts: B3 (V3's retrieval
plus the binary verifier) at 0.538, and V5-refit (V5 re-scored, not a further
component) at 0.292. **The ordering is monotone in point estimates. Its
significance structure is not a graded curve. It is two tiers separated by
one large step:**

- **Retrieval tier: {V0, V1, V2, V3, B3}, accuracy 0.538–0.582.** Every
  tested pair inside the tier ties after correction: U1 V0–V1, U5 V1–V2, U4
  V2–V3, pair a V1–V3, pair g V3–B3, and U2 V0–V3 (unadjusted only). At
  n = 500 no adjacent within-tier step can be told apart from zero. (The
  post-hoc X3, V0 vs B3, below, is the one within-tier exception.)
- **Agent tier: {V4−conflict, V5, V5-refit}, accuracy 0.292–0.358.** Every
  tested pair inside the tier ties after correction: pair d V4−conflict–V5,
  U6 V5–V5-refit, and U7 V4−conflict–V5-refit (unadjusted only).
- **Between the tiers, a significant drop of 0.22–0.27 accuracy.** Every
  cross-tier pair tested is Holm-significant in its own family: V0 > V5 (U3,
  +0.266), V1 > V5 (pair e, +0.250), V3 > V5 (confirmatory, +0.230),
  B3 > V5 (pair h, +0.222) and V3 > V5-refit (U8, +0.254).

*(Superseded in part by the post-hoc family directly below.)* When the
paragraphs above were written, no test placed V4−conflict directly against a
retrieval-tier variant, and its lower-tier placement rested on inference
through V5. §1.0c's post-hoc family now tests the boundary step directly.

**Post-hoc tier-boundary family results: X1–X3 (§1.0c, ADR-046).**
**POST-HOC. The pairs were identified after the tier structure above was
visible.** The family (m_X = 7) was committed before its p-values were
computed (`d53b40f`). It is Holm-corrected within these 7 only, never pooled
with the m=35 or U1–U8 families, and **never cited as pre-registered
support.** `scripts/analysis/posthoc_boundary_family.py` →
`results/phase7_grid/posthoc_boundary_family.json`. Rows are generated from
the JSON (diff = A − B):

| pair | A vs B | metric | A | B | diff | 95% CI | p | Holm rank | Holm thr. | Holm reject |
|---|---|---|---|---|---|---|---|---|---|---|
| X1 | V3 vs V4-conflict | accuracy | 0.546 | 0.358 | +0.188 | [+0.132,+0.244] | <0.0001 | 1 | 0.00714 | **yes** |
| X1 | V3 vs V4-conflict | macro-F1 | 0.545 | 0.342 | +0.204 | [+0.149,+0.258] | <0.0001 | 2 | 0.00833 | **yes** |
| X1 | V3 vs V4-conflict | halluc-rate | 0.883 | 0.725 | +0.158 | [+0.119,+0.198] | <0.0001 | 3 | 0.01000 | **yes** (V4−conflict lower) |
| X2 | V0 vs V4-conflict | accuracy | 0.582 | 0.358 | +0.224 | [+0.170,+0.280] | <0.0001 | 4 | 0.01250 | **yes** |
| X2 | V0 vs V4-conflict | macro-F1 | 0.579 | 0.342 | +0.238 | [+0.184,+0.292] | <0.0001 | 5 | 0.01667 | **yes** |
| X3 | V0 vs B3 | accuracy | 0.582 | 0.538 | +0.044 | [+0.014,+0.076] | 0.0054 | 6 | 0.02500 | **yes** |
| X3 | V0 vs B3 | macro-F1 | 0.579 | 0.537 | +0.042 | [+0.011,+0.073] | 0.0058 | 7 | 0.05000 | **yes** |

All 7 cells reject. What each pair adds, with its label attached:

- **X1: the boundary step, tested directly (post-hoc), is significant.** V3
  scores +0.188 accuracy above V4−conflict (95% CI [+0.132, +0.244]). The
  step no longer rests on inference through V5. As in pair b,
  V4−conflict's halluc-rate is significantly *lower* (0.725 vs 0.883,
  relative-only per §3 / ADR-031). The accuracy loss and the halluc-rate
  reduction arrive at the same boundary.
- **X2:** zero-shot beats the agent tier's best cell by +0.224 (post-hoc).
- **X3 qualifies the retrieval tier: post-hoc, zero-shot significantly
  beats B3** (+0.044, CI [+0.014, +0.076], p = 0.0054). B3 is V3's
  tri-modal retrieval plus the binary verifier. This is the first
  within-tier pair to survive correction in any family, so "every pair
  inside the retrieval tier ties" no longer holds without a qualifier. Read
  it at its real weight. It is post-hoc. Its correction is m = 7, looser
  than U1–U8's m = 18: p = 0.0054 would not clear m = 18's rank-1
  threshold of 0.00278, and that comparison is stated only to calibrate the
  result, never to re-threshold either family. It sits at the extremes of
  the tier (its highest and lowest cells). It joins U2 (V0 vs V3, +0.036,
  unadjusted only) in pointing the same direction.

**Revised statement of the structure.** One large step at the
retrieval/agent boundary: 0.19–0.27 accuracy, significant in every
cross-tier test run, and now tested at the boundary step itself (X1,
post-hoc). Inside the retrieval tier there is a small decline. Adjacent
pairs cannot resolve it (U1, U5, U4, a, g all tie). It becomes detectable
only between the tier's extremes, and only post-hoc (X3) or unadjusted
(U2). **Stated plainly: nearly all the measurable accuracy loss arrives with
the graded audit-and-rank stack.** Within retrieval, "retrieval does not
help" is established. "Retrieval hurts" still has no support from any
planned or unplanned test. It has one post-hoc signal: zero-shot over the
retrieval + binary-verifier baseline B3, by 0.044. That signal is a reason
for a pre-registered test in the next dataset (MedMCQA), not a claim for
this one. Zero-shot has the highest point estimate in the grid. It is
significantly better than the whole agent tier, and, post-hoc, than B3.

**Grounding vs. accuracy: a trade-off, stated as one.** Accuracy and
`grounded_rate` per cell, from each run's `metrics.json` (also the `Grounded`
column of `paper/tables/results_grid.md`):

| variant | Acc | grounded_rate | what "grounded" means for this row |
|---|---|---|---|
| V0 | 0.582 | 0.000 | nothing to cite: no retrieval, by construction |
| V1 | 0.566 | 1.000 | ≥1 valid citation (cited id is in the retrieved bundle) |
| V2 | 0.556 | 0.998 | ≥1 valid citation |
| V3 | 0.546 | 0.984 | ≥1 valid citation |
| B3 | 0.538 | 0.972 | ≥1 valid citation |
| V4−conflict | 0.358 | 0.856 | predicted option has ≥1 SUPPORT audit cell |
| V5 | 0.316 | 0.756 | predicted option has ≥1 SUPPORT audit cell |
| V5-refit | 0.292 | 0.652 | predicted option has ≥1 SUPPORT audit cell |

- **V0 has the highest accuracy point estimate and grounds nothing.** Its
  answers carry no citation a reader could check. Under CLAUDE.md §2 an
  uncited answer is ungrounded, so V0 fails the project's grounding
  requirement by construction, however accurate it is.
- **V1, V2 and V3 ground 0.98–1.00 of answers at slightly lower accuracy.**
  Against V0 the accuracy gap is a tie for V1 (U1) and not Holm-significant
  for V3 (U2). At this n, near-complete grounding costs an accuracy decrement
  the data cannot distinguish from zero. That is the size of the trade-off.
- **V5 grounds 0.756 at roughly half V0's accuracy** (0.316 vs 0.582, U3
  significant). Its grounding bar is stricter than V1–V3's: a SUPPORT audit
  cell, not merely a valid citation. So 0.756 vs 0.98–1.00 is not a
  like-for-like comparison, and `grounded_rate` is significance-tested
  nowhere. But there is no trade-off inside the agent stack. Grounding and
  accuracy fall together (V4−conflict 0.856 / 0.358 → V5 0.756 / 0.316 →
  V5-refit 0.652 / 0.292): lower on both. §5.0a(c) shows every ungrounded V5
  prediction is also abstained.
- **This does not soften the accuracy finding.** The trade-off exists only
  at the V0 / retrieval boundary. It is no argument for the agent stack,
  which loses significantly on accuracy to every retrieval-tier variant it
  was tested against, while grounding fewer answers even under its own
  stricter criterion. For completeness, the agent stack's halluc-rate is
  lower (pair b V3 − V5 +0.155, pair e V1 − V5 +0.165, both significant).
  That metric is relative-only (§3, ADR-031) and is computed over the
  predicted option's claims. It is reported beside the accuracy result, not
  netted against it.

**Wall-clock note.** V5 and V4−conflict's durations are not independently
recoverable: `write_run()` writes `config.yaml`/`predictions.jsonl`/
`metrics.json` together at completion, so file mtimes and the `run_id`
timestamp both record completion time only, not a start time (confirmed
against V1's known 50.7-minute runner-timed duration). Only the three
runner-timed cells (V1, V3, B3) have a real wall-clock figure.

### 5.1 Stronger-model sensitivity check (§0.6, robustness only — not a headline row)

**Registered 2026-09-12, not yet run.** The analysis plan, question set, cost
guard and launcher are committed (§1.0d, ADR-047); no Gemini API call has been
made and no run exists under `results/`. Four cells, not two: V0 and B3 join
V3 and V5 so the check covers the retrieval floor and the binary-verifier
baseline as well as the confirmatory pair.

Every number below stays `TBD` until
`results/phase7_gemini/gemini_sensitivity_family.json` exists. **Robustness
only — never a headline row, never pooled with the Phase 7 families** (§1.0d).

| Variant | Model | Dataset (n) | Acc | 7B counterpart (same 200) | run_id |
|---|---|---|---|---|---|
| V0-gemini zero-shot | TBD (stamped at launch) | MedQA (200) | TBD | 0.540 | TBD |
| V3-gemini +KG | TBD (stamped at launch) | MedQA (200) | TBD | 0.505 | TBD |
| B3-gemini binary verifier | TBD (stamped at launch) | MedQA (200) | TBD | 0.505 | TBD |
| V5-gemini full | TBD (stamped at launch) | MedQA (200) | TBD | 0.325 | TBD |

The 7B column is the committed grid runs restricted to the same pinned 200
question ids — the paired baseline P2 is scored against, not the n=500
headline figures (0.582 / 0.546 / 0.538 / 0.316).

### 5.2 Phase 1 retrieval-infra validation (real numbers, NOT part of the ablation grid)

These are retrieval-quality checks that validate the tri-modal plumbing
itself (Phase 1 exit test), not QA-accuracy results for any V-variant — they
don't belong in §5's table schema and have no `run_id` (Phase 6's harness
doesn't exist yet). Reproduce via `scripts/eval_retrieval.py` and
`scripts/analysis/nfcorpus_significance.py`. See ADR-012 for full context.

**Traceability fix (2026-09-08, AUDIT 2026-09-06 §1.1/§1.2):** both scripts
previously printed to stdout only, so these numbers existed solely in a
gitignored log — a CLAUDE.md §2 violation. Both now write a committed
artifact: `scripts/analysis/nfcorpus_significance.py` →
`results/phase1_nfcorpus/metrics.json` (means, all three pairwise CIs,
`n_resamples`, `seed`); `scripts/eval_retrieval.py` →
`results/phase1_retrieval_<dataset>/metrics.json` per mode (nfcorpus /
pubmedqa / medqa). **The numbers below are the last values recorded before
this fix and are not yet backed by either artifact** — regeneration is
queued in `scripts/run_overnight.ps1`; once it completes, re-point this
table's citations at the two paths above and treat any number that drifts
from the table below as the current, authoritative one.

| Benchmark | Metric | dense | bm25 | rrf | Verdict |
|---|---|---|---|---|---|
| NFCorpus (BEIR), n=323, gating | nDCG@10 | 0.3362 | 0.2929 | 0.3327 | **PASS** — statistical tie vs. dense (95% CI [-0.0171, +0.0103]); RRF significantly beats bm25 (CI [+0.0242, +0.0564]) |
| PubMedQA (`pqa_artificial`), n=200, infra floor | Recall@10 / MRR | 0.9950 / 0.9925 | 0.9800 / 0.9501 | 0.9900 / 0.9780 | **PASS** — all ≥ 0.9 floor (saturated by design, see ADR-012; not a fusion comparison) |

**Note recorded per ADR-012, not to be quietly revised later:** dense-only
retrieval is a real, evidenced competitor at this point — it saturates
PubMedQA and statistically ties (does not lose to) fused RRF on NFCorpus
while significantly beating BM25 alone. If V1 (dense-only) matches or beats
hybrid/agent variants on MedQA downstream accuracy in the §5 grid, report it
as such.

### 5.3 Phase 2 KG-layer validation (real numbers, NOT part of the ablation grid)

**Split correction (2026-09-07, ADR-027 / AUDIT 2026-09-06 §3.1).** The numbers
in this section were originally measured on the MedQA **test** split — the split
Phase 7 reports on — and the Phase 2 design decisions were made by reading them.
Every figure below has been **re-measured on a 500-question MedQA *train* slice**
and is the train figure. The superseded test-split artifacts
(`medqa_n200.json`, `medqa_n500.json` and the two `kg_rerank` files) are retained
in `results/phase2_kg_probe/` carrying `superseded: true`, and must not be cited.

Reproduce via:

```bash
python -m scripts.kg_probe --dataset medqa --split train --limit 500 \
    --out results/phase2_kg_probe/medqa_train_n500.json
```

reading real PrimeKG data cached under `data/processed/kg_cache/medqa/`
(Neo4j not required — cache-first per ADR-007). Full artifact:
`results/phase2_kg_probe/medqa_train_n500.json` (n=500, 0 cache misses, seed 42).
See ADR-014, ADR-015, ADR-016, ADR-017 and ADR-027 for complete context; this
table is the numeric summary only.

Numbers below use the final, seeded-random triple-selection measurement
(superseding an earlier alphabetical-selection pass whose bias ADR-016
documents and quantifies).

| Metric | Value (train n=500) | Reading |
|---|---|---|
| `aggregate_hit_rate` (question-level) | 0.9540 | Clears the roadmap's ≥0.60 floor, but see the next two rows before citing this as coverage |
| `entity_level_mapping_rate` | 0.1158 | The real per-entity PrimeKG resolution rate — read together with the row above, never alone |
| `naive_expected_hit_rate_under_independence` | 0.9999 | At 75.17 linked entities/question and 11.58% per-entity mapping, a question-level hit is arithmetically near-certain — and this expectation *exceeds* the 0.9540 observed. The question-level rate is that artifact, not evidence PrimeKG covers 95% of MedQA content. This is contribution **C5** (`docs/00_RESEARCH_SPEC.md` §3) |
| `overall_novel_relation_rate` (vs. real 12-item `EvidenceBundle`) | 0.9019 | Fraction of scored KG triples not entailed by top-12 dense passages (corrected from an initial, wrong top-3 measurement — see ADR-016) |
| `overall_relevant_rate` | 0.4744 | Fraction of scored triples mentioning an entity the question/options are about |
| **`overall_novel_and_relevant_rate`** | **0.4237** | **The actual C1 evidence figure** — on-topic *and* not already in dense passages. Middling, not strong; bottlenecked by retrieval precision (only ~47% relevant), not by redundancy with dense retrieval |
| `overall_novel_given_relevant_rate` | 0.8931 | Conditional on relevance, KG evidence is usually novel — the encouraging half of the finding |
| `unmapped_cui.unmapped_rate` | 0.8842 | 33,231/37,584 linked entities have no PrimeKG crosswalk hit, over 6,852 distinct unmapped CUIs (ADR-014: disease/phenotype via real ontology xrefs, drug/gene via name-match only) |
| Per-question triples (post ADR-015 cap) | median 40.0, mean 38.1 (n=500) | 476/500 questions (95.2%) saturate the 40-triple cap — the cap is the normal case, not a backstop. 23 questions yield zero triples |
| PrimeKG raw edges vs. loaded | 8,100,498 raw rows (~4.05M unique undirected edges) → 740,674 loaded (9.1%) | Deliberate 7-item allowlist filter, not an incomplete load — see ADR-014 (corpus/index provenance, traced to `data/MANIFEST.json`) |

Per-category breakdown (`n` shown per this phase's stated discipline;
`MIN_CATEGORY_N_FOR_RATE=15` — at n=500 every category clears it, including
`anatomy`, whose rates were suppressed at n=8 on the old test slice — see
ADR-017):

| Category | n | hit_rate | novel_and_relevant_rate |
|---|---|---|---|
| pharmacology | 127 | 0.976 | 0.483 |
| pathology_diagnosis | 116 | 0.991 | 0.392 |
| treatment_management | 101 | 0.931 | 0.413 |
| other | 92 | 0.913 | 0.365 |
| physiology | 41 | 0.976 | 0.425 |
| anatomy | 23 | 0.870 | 0.530 |

**KG triple reranking ablation (`kg_rerank`, ADR-018 → re-decided in ADR-027) —
the verdict flipped at n=500.** Reusing the Phase 1 MedCPT cross-encoder
(`src/retrieval/rerank.py`) to select which triples survive the ADR-015
40-triple cap (instead of raw traversal order) was tested as an off-by-default
config flag, with a real paired-bootstrap significance check (10k resamples,
seed 42, paired by `question_id`, n_paired=477, n_excluded=23 zero-triple
questions excluded identically in both variants) per CLAUDE.md's "never claim A
beats B from a raw mean alone" rule. Artifact:
`results/phase2_kg_probe/medqa_train_n500_kg_rerank_ablation.json`.

| Metric | naive | kg_rerank | Paired 95% CI | p (two-sided) | Holm survives (m=2) | Verdict (train n=500) | was (test n=200) |
|---|---|---|---|---|---|---|---|
| `relevant_rate_for_question` (secondary) | 0.4744 | 0.5296 | [+0.0203, +0.0889] | 0.0026 | **yes** (rank 1, threshold 0.025) | **significant win** | marginal win |
| `novel_and_relevant_rate_for_question` (**primary**) | 0.4237 | 0.4604 | **[+0.0044, +0.0681]** | 0.0256 | **yes** (rank 2, threshold 0.050) | **significant win** | **tie** |

At n=200 the primary metric was a tie and was reported as one. At n=500 on a
clean split it clears significance, exactly as ADR-018 anticipated when it
pre-committed to re-deciding at larger n. The primary CI's lower bound is
+0.0044 — one-tenth of the point estimate — so this is applied and reported as
a Holm-Bonferroni-corrected result for the 2-metric family
(`src.retrieval.metrics.paired_bootstrap_p` + `holm_bonferroni`, AUDIT §6.8),
not deferred to the grid-wide multiplicity discipline still pending for Phase 7
(§6.7). **Both metrics survive the correction.** This is not a coincidence to
skate past: with exactly 2 comparisons, Holm only tightens the *smaller*
p-value's threshold (secondary, tested at α/2 = 0.025); the *larger* p-value
(primary) is tested at the same threshold as the uncorrected case, α = 0.05,
because it is last in the step-down chain — so the primary's significance here
is genuinely robust to this correction, not narrowly surviving it. It remains a
*marginal* win in effect size (CI lower bound close to zero) even though it is
not marginal in significance. See ADR-027 for the full computation and for the
**pre-registration deviation** this ADR records: ADR-018's rule said adopt on a
cleared verdict, and this decision does not adopt — `kg_rerank` remains **off
by default**, because the metric that moved is an intermediate evidence-quality
proxy and no run yet shows it moving a downstream metric. The Phase 7
`kg_rerank` grid row is what actually settles whether that caution was
warranted.

**The honest position going into Phase 3, per ADR-016, ADR-018 and ADR-027:**
Phase 2's infrastructure is real and correct (verified PrimeKG source, real Neo4j
load, cache-first reproducibility confirmed byte-for-byte). But **the KG
modality's marginal value over dense retrieval is not yet established** — the
widely-visible ~95% hit rate is an arithmetic artifact of entity fan-out, and the
defensible C1 evidence figure (`novel_and_relevant_rate = 0.4237`) is real but
middling, limited primarily by retrieval precision (hub-node noise, ADR-015) and
partial entity crosswalk coverage (ADR-014), not by redundancy with dense
passages. Reranking measurably improves triple *selection* without yet being
shown to improve anything downstream. This is not a result to soften before
Phase 3.

### 5.4 Phase 3 baseline plumbing check (dev slice, n=20 — NOT a grid result)

Real numbers from `scripts/run_baselines.py --dataset medqa --split train
--limit 20` (seed 42). The four runs the table below reports are, exactly
(AUDIT 2026-09-06 §6.2 — no wildcard timestamp, because two committed batches
share the SHA `d5e3af5` and only one is valid):

- `results/medqa_B0_d5e3af5_20260906T101125Z/`
- `results/medqa_B1_d5e3af5_20260906T101242Z/`
- `results/medqa_B2_d5e3af5_20260906T101648Z/`
- `results/medqa_B3_d5e3af5_20260906T102134Z/`

The earlier `~0954–1006Z` batch at the same SHA (`…T095449Z`, `…T095549Z`,
`…T095935Z`, `…T100551Z`) is the pre-`num_ctx`-fix batch. It is **not**
deleted — it is the context-window sensitivity comparator used below and in
ADR-021 — but each of its four `metrics.json` files now carries
`superseded: true` with a `superseded_reason` and a `superseded_by` run id,
stamped by `scripts/mark_superseded.py --preset phase3-num-ctx`, so no reader
and no report script can tabulate it as a headline result by mistake.

Caveat on the recorded SHA: all eight runs were produced from an uncommitted
worktree, so `d5e3af5` does not identify the code that produced them (AUDIT
§6.1 — `src/eval/baselines.py` does not exist at that SHA). `src/eval/results.py`
now stamps `<sha7>-dirty` plus a `git_diff_sha256` for any run made from a dirty
tree; the eight runs above predate that fix and their SHA must be read as
"approximately, the tree around `d5e3af5`", not as a checkout target.

This is the Phase 3 exit test: it certifies that query → tri-modal
`EvidenceBundle` → answer → grounded citations runs end-to-end for all four
baselines. It is **not** a QA-accuracy result — n=20, a dev (train-split)
slice, single-seed, no significance test, no "A beats B" claim. The §5 grid
(Phase 7, 500-question seeded test slice) is the reportable table.

| Baseline | Acc | Macro-F1† | grounded_rate | invalid_citation_count | status |
|---|---|---|---|---|---|
| B0 zero-shot        | 0.55 | 0.52 | 0.00‡ | 0 | 20 ok |
| B1 dense-only RAG   | 0.60 | 0.57 | 0.95 | 1 | 20 ok |
| B2 hybrid (no agents)| 0.65 | 0.61 | 0.95 | 3 | 20 ok |
| B3 binary verifier§  | 0.65 | 0.61 | 0.95 | 3 | 20 ok |

† position-bias check, not a skill metric on 4-option MCQ (ADR-020).
‡ B0 runs no retrieval — ungrounded by construction, not a failure (ADR-020).
§ B3's answer vector is **identical to B2's on 19/20 questions by construction**
(AUDIT §4.6): the verifier decides on only 1/20 (strict prompt), and on the
other 19 it issues the byte-identical B2 generation call — `llm_cache_hits: 19`
in `medqa_B3_d5e3af5_20260906T102134Z/metrics.json`. B3's 0.65 is B2's 0.65
plus one verifier-decided question; the row is not an independent result and
the diagnostic value is `verifier_decision_rate` (below), not accuracy.

**B3 diagnostic — the `verifier_decision_rate` is robust to the context window
but NOT to verifier phrasing (ADR-033, AUDIT §4.1).** The earlier framing
("invariant at 1/20 = 0.05") was premature and is retired.

*Context-window axis (holds).* At the shipped strict verify prompt, the rate is
0.05 under `num_ctx=4096` and 0.05 under `num_ctx=8192`; only the *direction* of
failure changes:

| Run | `num_ctx` | `n_supported` distribution | exactly-one | `verifier_decision_rate` |
|---|---|---|---|---|
| `medqa_B3_d5e3af5_20260906T100551Z` (superseded) | 4096 | `{0: 7, 1: 1, 3: 2, 4: 10}` | 1/20 | **0.05** (over-admits — 10/20 all-four SUPPORTED) |
| `medqa_B3_d5e3af5_20260906T102134Z` (reported) | 8192 | `{0: 15, 1: 1, 2: 4}` | 1/20 | **0.05** (over-rejects — 15/20 zero SUPPORTED) |

*Prompt-wording axis (does not hold).* Re-run on the same dev-20 with two
further verify phrasings (`scripts/analysis/b3_prompt_sensitivity.py`,
`results/phase3_b3_prompt_sensitivity/metrics.json`):

| verify phrasing | `verifier_decision_rate` | `n_supported` distribution |
|---|---|---|
| strict ("SUPPORTED only if the evidence positively supports it" — shipped) | **0.05** | `{0: 15, 1: 1, 2: 4}` |
| neutral ("is option X supported as the correct answer?") | **0.30** | `{0: 8, 1: 6, 2: 3, 3: 1, 4: 2}` |
| permissive ("could option X be correct? SUPPORTED if the evidence is consistent with it") | **0.35** | `{0: 6, 1: 7, 2: 4, 3: 1, 4: 2}` |

Spread 0.30. **The 0.05 is substantially an artifact of the strict instruction**
— it makes the verifier reject nearly everything. Under a fair (neutral or
permissive) instruction the rate is 0.30–0.35. The AUDIT §4.1
"strawman-by-prompt-construction" objection is therefore *confirmed* for the
strong form of the claim.

**What survives.** Even at the *most permissive* phrasing the binary verifier
isolates exactly one option on only 7/20 questions and returns multiple- or
zero-SUPPORTED on the other 13 — a per-option yes/no head cannot rank
mutually-exclusive options, and no phrasing tested gets it above 0.35. That is a
real limitation of binary verification and it motivates C2, but the honest
figure is "isolates a single option on ≤ 35% of questions", **not** "1/20". C2's
audit-discrimination evidence does not depend on this comparison (it rests on
the 108 non-neutral audit cells and the leave-one-agent-out V4−audit grid cell,
`§5.5` / Phase 7); the B3 number is a motivating illustration, quoted with its
0.05–0.35 range, not a headline result. To be re-measured at n=500 in Phase 7.

n=20 single-seed remains a limit on all of the above (0.35 = 7/20 carries a wide
interval).

**Recorded per ADR-021:** the 4096-token batch was Ollama's default context,
which silently truncated the KG-triple block (appended last) and collapsed
B2/B3 to Acc 0.15 with 12/20 parse errors. The main table above is post-fix
(`llm.num_ctx = 8192` + a hard `LLMContextBudgetError` guard). ADR-021 also
records the no-op perturbation floor measured from the same pair of batches:
B0, which does no retrieval and is therefore *unaffected* by the context fix,
still flips one of 20 answers between them — so a 1/20 (0.05) accuracy delta
across these two batches is inside the noise floor and is not read as an effect.

**PubMedQA B3 — the binary verifier fails to isolate an option even on a 3-way
task (ADR-030).** PubMedQA is yes/no/maybe — the exact format MEGA-RAG-class
binary verifiers were built for — and B3 still cannot use it. On the dev-20
(`results/pubmedqa_B3_aa73199-dirty_20260907T165909Z/`, the word-framed prompt
ADR-030 keeps): `verifier_decision_rate = 0.15`, `n_supported = {0: 9, 1: 3,
2: 7, 3: 1}` — it isolates a single SUPPORTED answer on **3/20** three-way
questions and falls back to generation on 17/20. Under the earlier
A/B/C-lettered prompt the rate was **0.00** (`n_supported = {0: 12, 2: 6,
3: 2}`, superseded run `…T163512Z`) — so word framing *helps* the verifier a
little. Both PubMedQA numbers use the shipped *strict* verify instruction; per
ADR-033 that instruction depresses the rate on MedQA (0.05 strict vs 0.30–0.35
neutral/permissive), so the PubMedQA 0.15 is almost certainly a strict-prompt
floor too and should be read as "≤ ~0.35 under a fair instruction", not as a
hard 0.15. The prompt-invariant point stands: a per-option yes/no verifier does
not rank three competing answers any better than four, and no phrasing gets it
near reliable single-option isolation. (The PubMedQA baselines are otherwise a
degenerate one-class predictor at this model scale — see ADR-030; PubMedQA is
not in the Phase 7 grid, `docs/SCHEDULE.md`.)

### 5.5 Phase 4 multi-agent consensus plumbing check (dev slice, n=20 — NOT a grid result)

`python -m src.pipeline --config configs/agents.yaml --limit 20` (MedQA train
dev-20, seed 42), `results/medqa_V4_1388904_20260906T115020Z/`. Certifies that
query → tri-modal bundle → modality-reserved 12-item evidence set →
Hypothesis→Audit→Conflict → `SupportMatrix` with only real citations runs
end-to-end. **Not** a QA-accuracy result. Numbers below are post the ADR-025
claim-level conflict fix.

| Invariant / counter | Value |
|---|---|
| SupportMatrices emitted | 20/20 (+ per-node traces) |
| surviving cells citing a non-bundle id | **0** (exit-test invariant) |
| RAW model citation hallucination (dropped at audit) | 6 cells = 5.3% of produced cells |
| status | `ok` 19 · `insufficient_evidence` 1 |
| `budget_exceeded` rate | 0.00 |
| mean LLM calls / question | 5.75 (cap 12) · mean rounds 1.05 |
| bundle mix | KG slot filled 17/20 · all-passage 3/20 |
| audit cells | 108 (78 SUPPORT, 30 CONTRADICT); 61/80 options get a signal |

**Finding 1 — the Conflict agent never fires under a correct definition.**
With conflict redefined as claim-level (ADR-025: same `claim_idx`, SUPPORT and
CONTRADICT from different evidence ids), `conflict_fire_rate = 0.00` on the
dev-20 — **not one claim** had genuinely opposing evidence. The 11
`unresolved_conflict` statuses in the pre-fix run were entirely artifacts of
the wrong option-level definition (which counted "claim 1 supported, claim 3
contradicted" as a conflict). **Genuine evidence contradictions are rare on
this corpus**, so C2's contribution must be reframed around what the **Audit**
layer does (SUPPORT/CONTRADICT/NEUTRAL discrimination against retrieved
evidence) rather than around conflict *resolution*. This is the honest
position Phase 5 inherits.

**Finding 2 — the KG modality is nearly invisible to the agents.** Of 108
audit cells, **1 cites a KG triple**; 107 cite passages. Even with the KG slot
filled on 17/20 questions, the Audit agent audits claims against passage text
and effectively ignores the verbalised triples. Read with ADR-016
(`novel_and_relevant_rate ≈ 0.41` in retrieval) this puts the KG modality's
actual contribution to the agent pipeline close to nil on this slice — a
Phase 7 `evidence_split` / KG-verbalisation question, recorded not softened.

**Finding 3 — the agent layer underperforms plain hybrid RAG at equal evidence
budget.** `rank-1-by-support` (crude `s⁺ − s⁻` argmax, no calibration,
`resolved` not penalised per ADR-025) = **0.35** on the dev-20. The original
comparison was against B2 at 0.65, but B2 there saw up to ~52 evidence items
(12 reranked passages + all cached KG triples) against the agents' 12 — a
budget confound. That confound is now removed: `scripts/run_baselines.py
--equal-budget` (AUDIT §4.3) trims B2's bundle to the *same* 12-item,
`[8, 4]`-reserved split V4/V5 use, and
`results/medqa_B2eq_aa73199-dirty_20260907T162956Z/` scores **B2 = 0.600**
(macro-F1 0.594) on the identical dev-20. B2's drop from 0.650 → 0.600 under
the trim is **0.05 — exactly ADR-021's measured ±0.05 no-op noise floor** — so
the evidence-budget asymmetry does *not* explain B2's advantage over the
agents.

The unconfounded comparison therefore stands: on the same 20 questions at the
same 12-item budget, **plain hybrid RAG (one LLM call) reaches 0.60 and the
Hypothesis→Audit→Conflict agent stack reaches 0.35**. The gold option carries
the (tied-)max `s⁺` on only 10/20 questions and the Audit agent is
SUPPORT-heavy (78 vs 30). Calibration (Phase 5) fits α,β,γ,δ,T to turn this
count vector into a better logit, but a raw-argmax gap this large against an
equal-budget baseline is a **real finding to report** — the multi-agent layer,
as built, costs ~5.8 LLM calls/question to land below a single-call hybrid RAG
baseline — not a caveat to defer to Phase 7. Phase 7's V2/V3-vs-V4/V5 cells at
n=500 settle whether calibration closes it or whether the agent layer is a net
negative on MedQA.

### 5.6 Phase 5 evidential calibration (disjoint fit/held-out, n=60/60 — NOT a grid result)

`python -m src.pipeline --config configs/agents.yaml --dataset medqa --split
train --limit 120 --exclude-limit 20 --seed 42` → `results/medqa_V4_a714fe0-dirty_20260907T121703Z/`
(a fresh 120-question MedQA train slice, disjoint by construction from Phase
3/4's dev-20 — `load_question_sample_excluding`, joined by `question_id`, not
index arithmetic). Split by sorted `question_id` into fit (first 60) and
held-out (second 60); weights fit on the fit half only, every ECE number below
is measured on the held-out half. See ADR-028 for the full feature
definitions, split rationale, and gamma-identifiability discussion.

**The literal exit test** (`docs/02_ROADMAP.md`: "ECE on dev improves vs. raw
LLM option log-probs; ranked output includes per-option citations for every
question") **passes on its literal reading — and a follow-up ablation shows
that reading is not the one to build a claim on:**

| | value |
|---|---|
| Calibrated ECE, held-out (5 equal-mass bins, n=60) | **0.0724** |
| Raw-logprob ECE (temperature 1), held-out (same 60 questions, same bins) | 0.5418 |
| Paired bootstrap mean diff (calibrated − raw), 10k resamples seed 42 | −0.4204 |
| 95% CI | **[−0.5426, −0.2826]** |
| Verdict | calibrated significantly better than raw-at-T=1 — CI entirely below 0 |

The raw-logprob comparator is a real per-token-logprob readout
(`scripts/raw_logprob_baseline.py`, `src.llm.client.chat(logprobs=True)` —
Ollama's native provider rejects `logprobs`; routed through LiteLLM's
`openai/` provider type at the same server instead, confirmed empirically),
given the **same EvidenceBundle** the agent pipeline saw. Ranked output
(`results/phase5_calibration/medqa_dev_calibration.json`) carries all 120
questions × 4 options with per-option citations structurally present for every
row (63.1% of option-rows carry ≥1 real citation; the rest are legitimately
ungrounded per CLAUDE.md's carve-out, not missing data).

**But this comparison does not isolate what C3 claims, and a direct ablation
shows the gap above is almost entirely a temperature effect, not an
evidential-calibration effect.** A single scalar temperature T was fit on the
fit-60's already-saved raw logprobs (no new LLM calls — `log(raw_probability)`
is a valid stand-in for the true logit here, since softmax is shift-invariant)
and applied, frozen, to the held-out-60:

| comparison (held-out, n=60) | point ECE | diff | 95% CI | verdict |
|---|---|---|---|---|
| Calibrated (0.0724) vs. **T-scaled raw** (0.0725, T=92.03) | −0.0001 | −0.0001 | **[−0.1344, +0.0887]** | **tie** |
| Raw-at-T=1 (0.5418) vs. T-scaled raw (0.0725) | +0.4694 | +0.4694 | [+0.2168, +0.5563] | T-scaling significantly better |

**Temperature scaling alone — zero evidential features, zero agent
reasoning — reaches the same ECE as the full calibrated pipeline.** Nearly all
of the 0.54→0.07 improvement above is a temperature-scaling effect. **C3 is
restated accordingly: at this sample size, temperature scaling of the raw
signal suffices to match the evidential features' ECE; the evidential
features' additional contribution over temperature scaling alone is not
established.** This is not a null result to bury — it is exactly the finding
a careful ablation should surface, and it directly answers the "the gain is
just temperature scaling" objection a reviewer would otherwise have to raise.
It leaves open (Phase 7, larger n, or a different axis such as abstention or
citation quality) whether the features earn their place; it does not, on this
evidence, establish that they already have. Fitting `fit_temperature_only`
also caught and fixed a real methodological bug worth naming: an initial
narrow search bound (`t_max=20`, borrowed from the evidential fit's own
temperature bound) silently saturated, reporting a spuriously low T=20 whose
ECE (0.095) looked like a real gap vs. calibrated — corrected by widening the
search and confirming the true optimum (T≈92) before reporting anything
(see ADR-028 Decision 5; `fit_temperature_only` and `scripts/fit_calibration.py`
now detect and guard against this).

**Calibration improved confidence quality while accuracy went down — stated
plainly, not softened into a caveat.** Phase 4's uncalibrated
`rank-1-by-support` (raw `s+ − s−` argmax, §5.5 Finding 3) was **0.35** on its
own dev-20. This phase's fitted, temperature-scaled model gets **0.267**
rank-1 accuracy on both the fit-60 and held-out-60. These come from different
question sets, not a controlled before/after on the same items — a real
caveat, not a reason to soften the number. A model that is better-calibrated
(ECE 0.07) and less accurate than the cruder rule it replaces is a genuinely
limited contribution, and that is the honest headline alongside the ECE
result, not a footnote under a "not established" note. The §4.3 budget
asymmetry (V4/V5 get 12 evidence items; B2/B3 get up to ~52) separately still
forbids any comparison to B2 (AUDIT 2026-09-06, deferred) — a different
limitation from this one.

**Gamma identifiability, checked from two independent directions and
consistent.** On this 120-question slice, KG-modality SUPPORT cells are 12 of
626 total cells (1.9%) — reproducing §5.5 Finding 2's 1/108 almost exactly at
larger n. A 200-resample bootstrap of the fit gives:

| param | mean | std | range |
|---|---|---|---|
| alpha | −0.0169 | 0.0718 | [−0.2777, +0.1934] |
| beta | +0.1139 | 0.0867 | [−0.2836, +0.3304] |
| **gamma** | **+0.2315** | **0.2340** | **[−0.4031, +0.6642]** |
| delta | +0.0071 | 0.0205 | [−0.0438, +0.1334] |

Gamma's std is >3× alpha's despite a comparable point estimate, and its
interval straddles zero — the signature of an unidentifiable parameter, not a
real moderate effect. This is independent confirmation of the KG-invisibility
finding from a different measurement than the raw cell count: not only does
the audit layer almost never cite KG evidence, but even a downstream optimizer
handed the chance to weight it cannot find a stable weight, because there is
almost no variance in the feature to constrain one. Report gamma's interval
alongside its point estimate; the point estimate alone reads as a real effect
it is not.

### 5.7 Phase 6 harness — hallucination-rate control (dev-20, NOT a grid result)

The Phase 6 harness exit test (`src.eval.harness --config configs/v5.yaml
--dataset medqa --split train --limit 20`,
`results/medqa_V5_aa73199-dirty_20260907T170604Z/`) reported
`hallucination_rate_predicted_option = 0.682` (n=22 SUPPORT-cell claims for the
predicted option; `hallucination_rate_all_support_cells = 0.718`, n=78).
`hallucination_rate` here is: split each SUPPORT cell's rationale into claims
(`src.eval.nli.split_claims` — regex, decimals/abbreviations guarded), NLI each
claim against **its cited evidence text**, ungrounded = no cited evidence
reaches `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` entailment
**≥ 0.5** (`docs/04` §3).

**0.682 needs a control** — a 0.5 DeBERTa-NLI entailment threshold is strict,
and a clinical claim that is genuinely supported *after one step of inference*
scores as not-entailed. So run the identical scorer / threshold / splitter over
a **non-agent** pipeline's rationales on the same 20 questions
(`scripts/analysis/hallucination_control.py`, artifact
`results/phase6_hallucination_control/control.json`):

| pipeline (dev-20, same scorer, threshold 0.5, same `split_claims`) | halluc-rate (predicted option) | n claims | uncited |
|---|---|---|---|
| **B2 equal-budget** (hybrid RAG, one LLM call, 12-item bundle) | **0.741** | 58 | 0 |
| **V5** (agents + calibration) | **0.682** | 22 | 0 |
| V5 − B2 | **−0.060** | — | — |

**The number reflects the NLI criterion's strictness, not the agent layer.**
Both pipelines land near 0.7, **both with zero uncited claims** — so this is
not about missing citations, it is the NLI model declining to score
cited-but-inferential clinical claims as entailed at ≥ 0.5. Example from the B2
run (`train_00409`, cited to the correct passage): *"This protein is crucial
for the activation of B cells by T cells… its deficiency leads to low levels of
IgG, IgM, and IgA"* scores 0.545 (just entailed); the sibling claim *"the
patient has a defect in the activation-induced cytidine deaminase (AID)
protein"* scores 0.331 against the same passage. If anything V5 is marginally
*lower* than plain hybrid RAG (−0.060, n=20 single-seed — no significance
claimed).

**Reading (ADR-031):** at threshold 0.5, DeBERTa NLI marks ~70% of clinical
rationale claims as not-entailed by their cited evidence **regardless of
pipeline**. `hallucination_rate` is therefore usable as a *relative* comparison
across variants at a fixed threshold (its role in the Phase 7 exploratory
family, `§1.0`), **not** as an absolute "70% of claims are hallucinated"
statement — that framing would be measuring the entailment threshold. The
threshold and splitter are stated here and in ADR-031 so the number is
auditable; Phase 7 additionally reports `invalid_citation_count` (a cited id
not in the bundle — a strictly worse and unambiguous failure) alongside it.

## 6. Run discipline
- `results/<run_id>/config.yaml` — full resolved config incl. model ids + seed
- `results/<run_id>/predictions.jsonl` — one row per question: options, p vector,
  citations, status, agent trace ref
- `results/<run_id>/metrics.json` — everything in §1–4
- Only `metrics.json` and `config.yaml` are committed; predictions are large.

## 7. Sanity checks before believing any result
1. Shuffle option order → accuracy should not move more than noise (no position bias).
2. Empty-retrieval control → accuracy should drop toward V0.
3. Gold-evidence oracle → upper bound; if V5 ≈ oracle, retrieval is not the bottleneck.
4. Contamination assertion passed (logged in `metrics.json`).
