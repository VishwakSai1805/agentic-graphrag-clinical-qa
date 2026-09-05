# 05 — Decision Log (ADRs)

Append-only. One entry per non-obvious decision. Claude Code writes here at every
`/checkpoint`.

Format:

```
## ADR-NNN — <title>
Date: YYYY-MM-DD · Phase: N · Status: accepted|superseded by ADR-MMM
Context: what forced a choice
Decision: what we do
Alternatives: what we rejected and why
Consequences: what this makes easy / hard later
```

---

## ADR-001 — LangGraph for agent orchestration
Date: 2026-09-04 · Phase: 0 · Status: accepted
Context: The consensus network must be replayable and instrumented for a paper;
free-form agent chat is not reproducible.
Decision: LangGraph state machine with typed pydantic state and a checkpointer.
Alternatives: CrewAI / AutoGen (role-play abstractions, weaker determinism and
harder budget control); hand-rolled loop (more code, no checkpointing).
Consequences: Every run can be replayed node-by-node for error analysis; the
agents are pure functions, so unit tests can mock the LLM.

## ADR-002 — RRF over learned fusion for tri-modal retrieval
Date: 2026-09-04 · Phase: 0 · Status: accepted
Context: Dense, BM25 and KG scores are not comparable.
Decision: Reciprocal Rank Fusion (k=60), then a cross-encoder rerank.
Alternatives: learned fusion weights (needs training data, adds a confound to
the ablation); score normalisation (brittle across datasets).
Consequences: The ablation cleanly isolates each modality; a learned fuser can
be added later as an extra variant.

## ADR-003 — Calibration weights fit on dev only
Date: 2026-09-04 · Phase: 0 · Status: accepted
Context: Fitting α,β,γ,δ,T on test would invalidate every reported number.
Decision: Fit on a held-out dev split, freeze into the config, report the frozen
values in the paper.
Alternatives: cross-validation on test (unsound); no calibration (loses C3).
Consequences: Dev splits must be defined before any full run.

## ADR-004 — Zero-cost LLM provider stack: local Ollama primary, free-tier overflow
Date: 2026-09-04 · Phase: 0 · Status: accepted
Context: Project constraint is zero API spend — no paid keys. `src/llm/client.py`
must still work when no key is configured at all.
Decision: `configs/base.yaml` defines an ordered `llm.fallback_chain`: local
Ollama (`qwen2.5:7b-instruct`, no key required) first, then Groq
(`llama-3.1-8b-instant`) and Google AI Studio (`gemini-1.5-flash`) as optional
free-tier overflow, each skipped automatically if its `api_key_env` is unset.
`chat()` retries transient errors (rate limit/timeout/connection) with
exponential backoff before falling to the next provider, and caches on
(resolved model, prompt, params) so a fallback and a retry never get charged
twice for the same question. Cache hits contribute zero tokens/cost/latency to
the run's `StatsTracker`, not a replay of the original call's usage — otherwise
reruns of a cached eval would silently inflate the tokens/question and
wall-clock numbers reported in `docs/04_EVALUATION.md`.
Alternatives: single hardcoded provider (fails the zero-cost constraint or has
no offline fallback); no fallback chain at all (fragile — one rate limit stalls
a whole eval run).
Consequences: All local dev and, provisionally, all eval runs are pinned to a
7B-class local model. This bounds absolute accuracy relative to a frontier
model and is a threat-to-validity item for the paper (open question: is a 7B
local model sufficient to demonstrate the architecture's *relative* gains over
baselines, even if absolute accuracy is lower than a GPT-4/Claude-class
system?). Revisit before Phase 7 if the zero-cost constraint can be relaxed for
the final headline runs only.

## ADR-005 — Ablation grid cut: full 8-variant on MedQA only, V3/V5-only elsewhere
Date: 2026-09-05 · Phase: 0→1 · Status: accepted
Context: The original Phase 7 reading implied the full 8-variant × 3-dataset
grid, multi-seeded, using the initial 37s/question estimate for agent
variants. That estimate didn't budget the ~3.7s/call prefill cost of a
~3–4k-token `EvidenceBundle` at the measured 958 tok/s local prefill rate
(`docs/04_EVALUATION.md` §0.1). Re-budgeting to ~60s/question for agent
variants (V4, V5, V4−audit, V4−conflict) puts the full un-cut grid at roughly
3× the ~78h serial estimate in §0.2 — not runnable on a single 16GB machine
across a normal session cadence.
Decision: Cut the grid, not the science. Full 8-variant ablation (V0…V5,
V4−audit, V4−conflict) on **MedQA only**, 500-question fixed seeded subset
(seed 42). PubMedQA and MedMCQA get **V3 and V5 only** — the tri-modal-no-agent
and full-system endpoints, each on a 500-question fixed seeded set (MedMCQA
subsampled from its ~4.2k validation split, see `docs/03_DATASETS.md`).
Multi-seed (seeds 42/43/44) is run only for **V3 and V5 on MedQA** — the exact
pair the paired-bootstrap significance test (§1) compares; every other cell is
single-seed.
Alternatives: keep the full cross-product and reduce per-dataset question
count instead (rejected — shrinking MedQA below 500 weakens the primary
novelty benchmark, which is worse than dropping intermediate ablation cells on
the two secondary datasets); run everything single-seed including V3/V5 on
MedQA (rejected — the significance test needs multi-seed variance on exactly
the pair it compares, nowhere else).
Consequences: PubMedQA and MedMCQA cannot report per-agent ablation (V1, V2,
V4, V4−audit, V4−conflict) — only the MedQA table supports the full ablation
story; the paper must state this explicitly rather than implying uniform
coverage. `docs/02_ROADMAP.md` Phase 7 and `docs/04_EVALUATION.md` §5 are
updated to match this grid, not the original full cross-product.

## ADR-006 — Eval-run concurrency and keep-alive settings
Date: 2026-09-05 · Phase: 0→1 · Status: accepted
Context: Even after the ADR-005 grid cut, serial (concurrency=1) wall-clock is
~78h (`docs/04_EVALUATION.md` §0.2) — too slow to iterate on. Separately, a
single Ollama process reloading the model after every idle gap (harness
checkpoint I/O, a slow question) would silently inflate measured
tokens/question and wall-clock in `docs/04_EVALUATION.md` §5, corrupting the
throughput numbers the paper reports.
Decision: Set `OLLAMA_NUM_PARALLEL=4` on the Ollama server process
(`.env.example`) so it can serve overlapping requests, and give
`src/eval/harness.py` (Phase 6) an `asyncio.Semaphore` sized by a new
`eval.concurrency` config value (`configs/base.yaml`, default 4, wired into
`src/config.py: EvalConfig`). Set `OLLAMA_KEEP_ALIVE=-1` for all eval runs so
the model is never idle-unloaded mid-run. Phase 6's exit test benchmarks
concurrency 1 vs 2 vs 4 on a fixed dev slice and records the real
questions/hour, replacing the §0.2 planning assumption of ~3× effective
speedup (chosen conservatively over the naive 4× because the target machine
is RAM-bound at 16GB, not purely compute-bound).
Alternatives: leave concurrency at 1 (rejected — ~78h serial makes Phase 7
impractical across normal session lengths); leave `OLLAMA_KEEP_ALIVE` at its
default ~5-minute unload (rejected — reload spikes would appear as
throughput/latency noise in exactly the numbers the paper reports).
Consequences: Phase 6 must measure and record real concurrency throughput
before Phase 7 is scheduled — the 3× figure is provisional. Local dev runs
outside an eval context are unaffected (keep-alive default is fine when not
benchmarking).

## ADR-007 — KG subgraph retrieval precomputed to a cache file; Neo4j not required at eval time
Date: 2026-09-05 · Phase: 0→1 · Status: accepted
Context: RAM (16GB), not VRAM, is the binding constraint on the target
machine. Running Neo4j 5 alongside a local Ollama 7B model and the retrieval
stack (FAISS index, embeddings, BM25) concurrently during eval risks memory
pressure and, separately, makes KG retrieval a live dependency that varies
run-to-run (Neo4j warm cache state, concurrent query load) — undermining the
determinism/replayability goal already established for the agent layer in
ADR-001.
Decision: Add `scripts/precompute_kg_cache.py` (Phase 2) that runs entity
linking + subgraph traversal once per question, for every dataset/split that
will be evaluated, and writes the result (linked CUIs, subgraph triples,
verbalised `Evidence(modality="kg")` objects) to
`data/processed/kg_cache/<dataset>/<question_id>.json`. `src/kg/traversal.py`
gets a cache-first read path: eval-time code (Phase 3 onward) reads this file
and never opens a live Neo4j connection. Only `precompute_kg_cache.py` and ad
hoc debugging talk to Neo4j directly, so Neo4j can be `docker compose down`
between the precompute step and every subsequent eval run.
Alternatives: keep Neo4j live during eval (rejected — competes for RAM with
the LLM and retrieval indexes, and reintroduces run-to-run KG nondeterminism);
cache at the embedding/FAISS layer only and keep KG live (rejected — KG
traversal, not dense/lexical retrieval, is the nondeterministic/expensive
piece here; it's the one that needs precomputing).
Consequences: KG retrieval becomes deterministic and replayable byte-for-byte
across reruns (Phase 2 exit test verifies this explicitly). Adding a new
dataset/split requires re-running the precompute script before it can be
evaluated — this is a one-time cost per split, documented in Phase 2, not a
per-eval-run cost. `src/kg/queries.py` (all Cypher) stays scoped to the
precompute path only.

## ADR-008 — Stronger-model sensitivity check via free Gemini Flash tier
Date: 2026-09-05 · Phase: 0→1 · Status: accepted
Context: ADR-004 pins all eval runs to a 7B-class local model for the
zero-cost constraint, which ADR-004 already flagged as a threat-to-validity
item: is the V3→V5 relative gain (the paper's core claim) real, or an
artifact of a small model producing more errors for the agent network to
fix?
Decision: Add a robustness check — V3 vs V5 on 200 MedQA questions (fixed
seeded subset) run through the free Gemini Flash tier via the existing
`configs/base.yaml: llm.fallback_chain` provider abstraction (no new client
code). Reported in the paper as a robustness/sensitivity check alongside the
main grid (`docs/04_EVALUATION.md` §5.1), not as a headline result, and not
folded into the main ablation table's statistics.
Alternatives: no sensitivity check (rejected — leaves ADR-004's threat-to-
validity item unaddressed); run the full grid on Gemini Flash (rejected —
free-tier rate limits make a 200-question spot check tractable but a full
multi-thousand-question grid is not, and that's not this check's purpose);
use a paid frontier model (rejected — violates the zero-cost constraint for a
robustness check, not a headline result, where it isn't warranted).
Consequences: This check is API-rate-limit-bound rather than local-hardware-
bound, so it doesn't compete with the §0.2/§0.4 throughput planning and
should be scheduled as an independent background job. If V3→V5 gains vanish
under Gemini Flash, that's a finding to report honestly, not a result to
discard.

## ADR-009 — MedQA textbook corpus is primary content, not a fast-follow; CUDA build pinned for Blackwell
Date: 2026-09-05 · Phase: 1 · Status: accepted
Context: Two related Phase 1 risks were raised and resolved together. (1) The
original MedQA release (`jind11/MedQA`) ships its 18 USMLE textbooks only via
an unscriptable Google Drive link, tempting a corpus-v1 built from PubMed
abstracts alone with textbooks as a later addition. USMLE vignettes are
answered from textbook content, not PubMed abstracts — an abstracts-only
corpus would make every Phase 3+ baseline artificially weak and quietly
undermine later comparisons. (2) The target machine's RTX 5070 Ti is
Blackwell (sm_120), which needs `torch>=2.7` with CUDA 12.8 wheels; a default
`pip install torch` resolves an older cu121-class build that either fails or
silently falls back to CPU, and embedding several hundred thousand chunks on
CPU costs roughly an hour per rebuild.
Decision: `scripts/fetch_datasets.py` fetches the textbook corpus from the HF
mirror `MedRAG/textbooks` (confirmed reachable, 18 files matching the
original release, ~1GB) as the **primary** path, not a fallback; if all HF
mirrors and the GitHub→Google Drive fallback fail, the script exits nonzero
and halts the phase rather than proceeding on abstracts alone.
`pyproject.toml` pins `torch>=2.7` via `[tool.uv.sources]`/`[tool.uv.index]`
pointing at `https://download.pytorch.org/whl/cu128`, and
`scripts/eval_retrieval.py` prints `torch.cuda.is_available()` and the device
capability unconditionally so a silent CPU fallback is visible immediately
rather than discovered an hour into a build.
Alternatives: abstracts-only corpus v1 with textbooks as fast-follow
(rejected per the above); leave torch unpinned and let CPU builds happen
silently (rejected — wastes wall-clock every rebuild and the failure mode is
silent, not an error).
Consequences: Phase 1 setup depends on the `MedRAG/textbooks` mirror staying
available; if it disappears, re-derive the fallback chain rather than
reverting to abstracts-only. Any machine without a matching CUDA 12.8 GPU
needs its own `[tool.uv.index]` override — documented here, not silently
handled.

## ADR-010 — MedCPT is a dual encoder: separate Query- and Article-Encoders
Date: 2026-09-05 · Phase: 1 · Status: accepted
Context: CLAUDE.md §3 originally listed dense-retrieval embeddings as a
single model, `ncbi/MedCPT-Article-Encoder`. MedCPT is actually an asymmetric
dual encoder — NCBI trained separate `MedCPT-Query-Encoder` and
`MedCPT-Article-Encoder` models on PubMed search logs specifically because
queries and articles have different length/structure statistics. Using the
Article-Encoder for queries too would silently degrade dense retrieval
quality with no error raised.
Decision: `src/retrieval/dense.py` uses `ncbi/MedCPT-Query-Encoder` to embed
queries and `ncbi/MedCPT-Article-Encoder` to embed indexed documents at build
time. CLAUDE.md §3's stack table is corrected to name both models. The
symmetric fallback `BAAI/bge-large-en-v1.5` is unaffected (one model, both
sides, with its standard query-instruction-prefix convention).
Alternatives: single-encoder MedCPT usage as originally written (rejected —
not how the model was trained; would work without erroring but degrade
recall silently, the worst kind of bug for a retrieval-quality claim in the
paper).
Consequences: `configs/base.yaml`'s retrieval section carries both encoder
names as separate config keys, not one; any future embedding-model swap must
account for whether the replacement is symmetric or asymmetric.

## ADR-011 — Retrieval exit-test metric: PubMedQA is the gating criterion, MedQA is a lexically-biased diagnostic only
Date: 2026-09-05 · Phase: 1 · Status: superseded by ADR-012 (PubMedQA's gate role is retracted — see below; PubMedQA is kept as an infra-correctness floor + diagnostics, MedQA's role is unchanged)
Context: The Phase 1 exit test as originally written ran on MedQA, which has
no gold supporting-passage label — the only available proxy was "chunk
contains the gold answer-option text." That proxy is unsound as a modality
comparison: it rewards surface string matching, which is exactly what BM25
optimizes for, so judging "RRF ≥ dense and ≥ BM25" against it is a confounded
test that structurally favors BM25 regardless of whether RRF/dense retrieval
is actually working.
Decision: PubMedQA becomes the primary, gating retrieval-quality run —
each PubMedQA question has a real gold document (the specific abstract it
was generated from), giving genuine Recall@10/MRR. The eval query set is
drawn from `pqa_artificial` (disjoint from the blocklisted `pqa_labeled` test
split used everywhere else in the project), so the contamination blocklist
is not touched or weakened to enable this metric. `RRF ≥ dense and RRF ≥
bm25` is judged on this PubMedQA run. The MedQA answer-string-containment
metric is retained as a secondary diagnostic only, explicitly labelled
`lexically biased — diagnostic only` in `scripts/eval_retrieval.py`'s output,
and does not gate the exit test.
Alternatives: keep MedQA as the sole/primary metric (rejected — confounded,
per Context); drop the MedQA diagnostic entirely (rejected — it's still
useful signal once its bias is disclosed, and MedQA is the primary novelty
benchmark so *some* retrieval visibility into it is worth keeping).
Consequences: `docs/02_ROADMAP.md` Phase 1's exit test and
`scripts/eval_retrieval.py` support both `--dataset pubmedqa` (gating) and
`--dataset medqa` (diagnostic) modes with different pass/fail semantics —
future readers of the exit-test output must not conflate the two.

**Retraction note (see ADR-012):** the real run showed dense-only saturates
PubMedQA even harder than anticipated (0.995 Recall@10) — the paraphrase
structure of PubMedQA makes *any* single-document-recall metric a poor
fusion test, not just a BM25-favoring one. PubMedQA's role is downgraded
from gating to an infra-correctness floor; NFCorpus (BEIR) takes over as the
primary fusion-quality gate.

## ADR-012 — NFCorpus (BEIR) becomes the primary fusion-quality gate; judged by pre-registered statistical significance, not point-estimate inequality
Date: 2026-09-05 · Phase: 1 · Status: accepted

**Context.** The real Phase 1 exit-test run under ADR-011 showed dense-only
retrieval saturating PubMedQA at 0.995 Recall@10 / 0.9925 MRR (bm25 0.980/
0.9501, RRF 0.990/0.9780) — RRF fused *below* dense-only. Root cause,
verified not to be an implementation bug (6 passing `fusion.py` unit tests):
PubMedQA questions are auto-generated as close paraphrases of their own
source abstract, so retrieving "the document this question was written from"
is close to a lookup task and any working modality saturates near ceiling.
RRF is rank-position fusion, not max-of-modalities — folding in a
slightly-worse second ranking can only pull an already-near-perfect ranking
down, never up. ADR-011's fix (switch the gate from MedQA to PubMedQA) had
traded a BM25-favoring confound (MedQA's answer-string-containment proxy)
for a ceiling-effect confound (PubMedQA's paraphrase structure) — both
single-dataset per-item recall metrics turned out to be poor tests of
*fusion* value specifically, for different reasons.

**Decision, part 1 — new gating benchmark.** Add NFCorpus (BEIR), fetched
from `BeIR/nfcorpus` (corpus + queries) and `BeIR/nfcorpus-qrels` (test
split, graded relevance), as the primary fusion-quality sanity check:
3,633 documents / 6,241 chunks after project-standard chunking, 323 test
queries with 12,334 graded (1–2) human relevance judgments, no
single-modality saturation. Built via the exact same `DenseIndex`/
`BM25Index`/`reciprocal_rank_fusion` code as the main corpus (not forked),
under a separate `indexes/nfcorpus/` (not mixed into the project's actual
retrieval corpus — this is a sanity-check benchmark, not project data).
`scripts/build_nfcorpus.py` builds it; `scripts/eval_retrieval.py
--dataset nfcorpus` runs the gate. nDCG@10 is the metric (graded relevance
needs it; Recall@k/MRR are binary-relevance only).

**The gate initially still "failed" on point estimates**: dense 0.3362,
bm25 0.2929, rrf 0.3327 — RRF below dense by 0.0035. Before accepting this
as a real fusion defect, two specific implementation-bug hypotheses that
would produce exactly this signature were checked and ruled out:
- *Doc-ID space misalignment* (dense/BM25 returning different id
  vocabularies, splitting a document's rank contribution across two fused
  entries): verified `indexes/nfcorpus/dense/chunk_ids.json` and
  `indexes/nfcorpus/bm25/chunk_ids.json` are the *identical* 6,241-entry
  list, same order, both built from the same `nfcorpus_corpus.jsonl`.
- *Unequal candidate depth* (asymmetric top-k feeding the fusion): verified
  `configs/base.yaml: retrieval.top_k_dense == retrieval.top_k_bm25 == 50`,
  and `reciprocal_rank_fusion` treats an id absent from either ranking
  identically (0 contribution, no asymmetric penalty).

Neither held — the retrieval plumbing is correct. The 0.0035 point-estimate
gap needed a different question: is it statistically distinguishable from
noise at n=323?

**Decision, part 2 — pre-registered statistical rule, not raw inequality.**
`docs/04_EVALUATION.md` §1 already mandates paired-bootstrap significance
(10k resamples) as this project's standard for comparing two systems — the
NFCorpus gate now uses that same methodology instead of comparing two raw
means, via a new pure, tested function `src/retrieval/metrics.py:
paired_bootstrap_ci` (12 tests in `tests/test_metrics.py`, including a
noise-dominated case that must resolve to a tie, not a false direction).
The decision rule was fixed *before* re-reading the significance result:
- RRF significantly **better** than the best single modality (95% CI of the
  per-query nDCG@10 difference lies entirely above 0) → gate **passes**,
  fusion demonstrably adds value.
- **Statistical tie** (CI includes 0) → gate **passes**, reported explicitly
  as "fusion is not harmful; dominance not demonstrated at this sample
  size" — this certifies retrieval-*plumbing* correctness, not fusion
  *value*. The fusion-value question is what H1 (`docs/00_RESEARCH_SPEC.md`)
  and Phase 7's downstream-accuracy ablation actually test. A tie must never
  be written up as a win for RRF.
- RRF significantly **worse** → gate **fails**, treated as a real bug to
  investigate, not softened.

**Real result** (`scripts/eval_retrieval.py --dataset nfcorpus --limit 323`,
also independently reproduced by the standalone
`scripts/analysis/nfcorpus_significance.py`):
```
mean diff (rrf − dense) = −0.0034, 95% CI = [−0.0171, +0.0103]  → CI includes 0 → TIE → PASS
mean diff (rrf − bm25)  = +0.0398, 95% CI = [+0.0242, +0.0564]  → significant  → RRF beats bm25
mean diff (dense − bm25)= +0.0432, 95% CI = [+0.0194, +0.0675]  → significant  → dense beats bm25
```
**Phase 1 gate: PASSED as a statistical tie.** RRF is statistically
indistinguishable from dense-only and significantly beats BM25-only —
consistent with RRF behaving as designed (hedging against the weaker
modality without losing to the stronger one), not with a broken
implementation.

**Decision, part 3 — PubMedQA's demoted role.** PubMedQA is retained, not
discarded: the real numbers (dense 0.995/0.9925, bm25 0.980/0.9501, rrf
0.990/0.9780 Recall@10/MRR) are kept as a reported infra-correctness floor
(`Recall@10 ≥ 0.9` for all three modalities — a regression guard, not a
comparison) plus two non-gating diagnostics (Recall@1; a "hard subset" cut
where dense's own top-1 is wrong — bm25/rrf both hit Recall@10=1.0 on that
2/200-query subset, MRR 1.0 and 0.625 respectively, too small a slice to
read further into).

**Explicitly recorded, not to be quietly tuned away later:** across every
retrieval-quality check run in Phase 1, dense-only (V1) is a genuine,
evidenced competitor, not a strawman baseline — it significantly beats
BM25-only on NFCorpus and statistically ties fused RRF there, and it
saturates PubMedQA outright. **If V1 (dense-only) also matches or beats the
hybrid/agent variants on downstream MedQA accuracy in Phase 7's ablation
grid, that is a finding to report honestly in the paper, not a result to
engineer around** by retuning fusion weights, RRF's k, or top-k depths until
a hybrid variant wins. H1 (`docs/00_RESEARCH_SPEC.md`) is a hypothesis to
test, not a target to hit.

**Alternatives considered:** keep MedQA or PubMedQA as the sole gate
(rejected — both confounded, see ADR-011 and Context above); loosen the
NFCorpus gate to point-estimate "RRF within some fixed tolerance of the
best" (rejected — an arbitrary tolerance is exactly the kind of post-hoc
threshold-picking the pre-registered significance rule is designed to
avoid); re-litigate ADR-002's fusion method (weighted RRF, score
normalization) on the strength of a statistically-tied result (rejected —
premature; there is no significant evidence RRF underperforms here, so
there is nothing yet to fix at the fusion-method level).

**Consequences:** `scripts/eval_retrieval.py --dataset nfcorpus` is now the
authoritative Phase 1 fusion-quality exit test, judged by
`paired_bootstrap_ci`, not raw means. `scripts/analysis/nfcorpus_significance.py`
is a committed, reproducible standalone check of the same significance
question (not a throwaway script) — any future dispute about this gate's
verdict should be re-run through it. Phase 7's ablation grid, when it
reports V1 vs V2/V3 vs V4/V5, inherits the obligation set here: a dense-only
result that matches or beats hybrid/agent variants is reported as such.

## ADR-013 — Small implementation choices made silently during Phase 1
Date: 2026-09-05 · Phase: 1 · Status: accepted

Recorded per CLAUDE.md's decision-log discipline — these weren't discussed
explicitly but are decisions, not accidents.

**Textbook corpus is always standardized to raw per-book text, regardless of
which source fetched it.** `scripts/fetch_datasets.py`'s two textbook
sources return different native formats: the `MedRAG/textbooks` HF mirror
ships pre-chunked snippets (`chunk/<Book>.jsonl`, MedRAG's own chunking
policy), while the jind11/MedQA Google Drive fallback ships raw full-book
`.txt` files. Whichever source succeeds, the script reconstructs (or already
has) one raw `data/raw/textbooks/en/<Book>.txt` file per book — MedRAG's
chunks are concatenated back into continuous text, ordered by their chunk
index. Reason: every corpus source must go through the project's *own*
~256-tok/32-overlap chunker (`src/retrieval/chunking.py`) for a uniform
chunking policy across pubmed abstracts and textbooks; mixing in MedRAG's
own chunk boundaries for one source only would confound the corpus with two
different chunking policies with no way to isolate the effect later.
Consequence: re-running `fetch_datasets.py` is idempotent against
`data/raw/textbooks/en/*.txt` specifically (18 files present → skip both
network paths), not against either upstream format individually.

**`data/MANIFEST.json` is the one file under `data/` that is tracked in
git.** `.gitignore` originally blanket-ignored `data/` (Phase 0), which
silently defeated `docs/03_DATASETS.md`'s stated intent ("Record dataset
revision hashes in `data/MANIFEST.json`") — the file was being written but
could never actually be committed. Fixed by changing the pattern from
`data/` to `data/*` with a `!data/MANIFEST.json` negation (a directory-level
ignore prevents git from ever descending far enough to evaluate a nested
negation; a wildcard-on-contents ignore does not). Consequence: any future
addition of another data/ file meant to be tracked (e.g. a splits manifest)
needs its own negation line, following this same pattern — a plain `data/`
re-ignore would silently break this again.

## ADR-014 — PrimeKG ingestion: real-data corrections to the documented plan
Date: 2026-09-05 · Phase: 2 · Status: accepted

**Context.** `docs/03_DATASETS.md`'s PrimeKG section was written from prior
knowledge of the dataset, before the real dump was downloaded and loaded.
Implementing `scripts/load_primekg.py` and `src/kg/traversal.py` against the
actual data surfaced several places where the documented plan didn't match
reality. Source verified: Harvard Dataverse `doi:10.7910/DVN/IXA7BM`
(Chandak/Huang/Zitnik, CC0) — no official HF mirror exists; confirmed by real
HTTP GETs whose local MD5s matched Dataverse's own reported checksums exactly
(`nodes.csv` `7f9ab4109c54049e819ecd14e15a6038`; `kg.csv`,
981,751,236 bytes, `aac8191d4fbc5bf09cdf8c3c78b4e75f`). Real load into Neo4j:
129,375 nodes, 740,674 edges after the edge-type allowlist filter
(`disease_phenotype_positive` 300,634; `disease_protein` 160,822;
`drug_effect` 129,568; `disease_disease` 64,388; `contraindication` 61,350;
`indication` 18,776; `off-label use` 5,136).

**Decision — six corrections, each forced by the real data, not a preference:**
1. **Allowlist string fix:** `docs/03_DATASETS.md` wrote `off_label_use`;
   PrimeKG's real `kg.csv` `relation` column uses the literal string
   `"off-label use"` (zero rows match the underscored form). `KGConfig`,
   `configs/base.yaml`, and `src/kg/verbalize.py` all use the verified string;
   `docs/03_DATASETS.md` is out of date on this one point and this ADR is the
   correction of record.
2. **Bidirectional-edge canonicalization:** `kg.csv` stores every allowlisted
   relation as two literal rows per fact (`x=A,y=B` and `x=B,y=A`), PrimeKG's
   way of encoding an "undirected" edge even for semantically asymmetric
   relations (e.g. `indication` is always drug→disease, never disease→drug).
   Naively trusting Neo4j's per-row `startNode`/`endNode` produced genuinely
   backwards sentences on the reverse-direction row roughly half the time
   (observed live: "Hearing impairment has the effect Acetylsalicylic acid").
   Fixed via `src/kg/traversal.py: _canonicalize` + `_EXPECTED_SUBJECT_TYPE`,
   which reorients (and dedupes) both directions into one consistently-phrased
   triple per relation, type-aware. `disease_disease` is deliberately excluded
   from dedup (both sides are type "disease", so there's no wrong-direction
   *phrasing* risk, but its reverse row is a disclosed, uncorrected minor
   redundancy — a caller needing exactly-once undirected `disease_disease`
   triples needs a separate canonical-ordering step, not implemented here).
3. **`nodes.csv` fetched via `?format=original`** rather than Dataverse's
   default served format, so the local file's MD5 matches the dataset's own
   published checksum exactly — needed for honest provenance, not just a
   working parse.
4. **scispaCy pinned to 0.5.4** — the last release with a published
   `en_core_sci_lg` model artifact compatible with this project's pinned
   `spacy`/`torch` versions; later scispaCy releases had no matching model
   wheel available at implementation time.
5. **Neo4j schema uses one generic `:KG_EDGE{relation}` relationship type**
   (relation stored as a property, filtered via `WHERE r.relation IN
   $edge_allowlist`), not one Neo4j relationship type per PrimeKG relation.
   Dynamic relationship-type creation from data needs the APOC plugin, which
   is not loaded in `docker/compose.yaml` (keeps the Neo4j image smaller and
   the compose file simpler); the property-filter approach is equivalent for
   this project's fixed 7-item allowlist and every Cypher string lives in
   `src/kg/queries.py` regardless.
6. **CUI→PrimeKG crosswalk is honestly partial, not fully solved.**
   Disease/phenotype linking uses real ontology cross-references (MONDO/HPO
   xrefs present in PrimeKG's own node table). Drug/gene/protein linking falls
   back to name-matching only — there is no free/open UMLS↔DrugBank or
   UMLS↔NCBI-gene crosswalk available without a licensed UMLS Metathesaurus
   (out of scope, see `docs/03_DATASETS.md`'s "Optional later: UMLS RRF
   subset... only if licence is in place"). On a 5-question real MedQA sample:
   107 unique entity mentions, 27 (25.2%) resolved to a real PrimeKG node (21
   via ontology xref, 21 via name-match — some overlap). Unmapped CUIs are
   logged per `docs/03_DATASETS.md`'s existing instruction, not hidden.

**Alternatives considered:** silently "fixing" the allowlist spelling in
`docs/03_DATASETS.md` with no record (rejected — CLAUDE.md requires
non-obvious deviations logged here, and a future reader hitting zero
`off_label_use` rows deserves the real explanation, not a quiet edit);
trusting Neo4j's raw per-row direction and accepting backwards sentences
(rejected — an evidence object handed to the audit/conflict agents in Phase 4
with a factually-reversed relation is exactly the kind of silent correctness
bug CLAUDE.md's hygiene rules exist to prevent); loading APOC to get native
per-relation Neo4j types (rejected — adds a plugin dependency for a cosmetic
schema preference with no query-correctness benefit given the allowlist is
fixed and small); pursuing a paid/licensed UMLS Metathesaurus crosswalk to
close the drug/gene mapping gap (rejected — violates the zero-cost constraint
for a coverage improvement whose actual size is unmeasured; revisit only if
Phase 2's KG-hit-rate exit test result makes closing this gap look
worthwhile).

**Consequences:** `docs/03_DATASETS.md`'s edge-type list should be read
alongside this ADR for the corrected `off-label use` spelling. Any future
PrimeKG re-download must re-verify the checksums above before re-trusting a
cached `data/raw/primekg/` copy. The partial drug/gene crosswalk means
Phase 2's KG-hit-rate result (next entry, once `kg_probe.py` runs for real)
should be read knowing that disease/phenotype-heavy MedQA questions have a
structurally better chance of a KG hit than drug/gene-heavy ones purely from
crosswalk coverage, independent of whether PrimeKG's actual graph content
would otherwise support a hit — this is exactly the kind of category-level
confound the per-question-category hit-rate breakdown (this phase's amended
exit test) is designed to surface rather than hide behind an aggregate.

## ADR-015 — KG traversal degree cap alone is insufficient; added a per-question triple cap
Date: 2026-09-05 · Phase: 2 · Status: accepted

**Context.** `docs/03_DATASETS.md`'s degree cap (200 edges per expanded node
per hop) was designed against the classic single-hub-node scenario (its own
example: aspirin). Running `traverse_live` for real against real MedQA
questions surfaced a compounding case the per-node cap doesn't bound: a
question can link to *multiple* seed entities (a real MedQA vignette linked
8), each independently capped at 200 edges per hop, over 2 hops — the
per-node cap is respected at every step but the union across seeds and hops
is not capped at all. Real, observed, not hypothetical: one real MedQA test
question (`test_00001`) produced a 15,212-triple, 7.6MB single-question cache
file this way.

**Decision.** Add `max_triples: int | None = None` to `traverse_live`
(`src/kg/traversal.py`) — an early-stop across the hop-expansion loop plus a
final deterministic (sorted) truncation — and a new `KGConfig.
max_triples_per_question: int = 40`. This caps total *evidence volume per
question*, which is the quantity that actually matters for downstream
agent-context budgets (`docs/01_ARCHITECTURE.md`'s `max_evidence=12` budget
guard already implies KG evidence must be pre-trimmed well below 40 before
Phase 4, but 40 is a reasonable precompute-time ceiling ahead of that
downstream trim), not the per-node degree cap docs/03_DATASETS.md specified.

**Real consequence, disclosed rather than hidden:** measured directly from
the 200 real cache files (not estimated): triples/question distribution is
min=0, median=40.0, mean=38.4, max=40, and **192/200 questions (96%) hit the
cap exactly**. The cap is not an occasional backstop, it is the normal case —
the real constraint on subgraph *content quality* is which 40 triples the
deterministic sort keeps, not the cap's existence. Manual inspection of the
kept triples (see ADR-016's sample subgraphs) shows broad, low-specificity
PrimeKG hub nodes (e.g. the symptom node "Pain", which has hundreds of
`drug_effect` edges) frequently crowd out more question-specific facts.
ADR-016's real, measured `relevant_rate = 0.4865` quantifies this directly:
fewer than half of the 40-per-question kept triples even mention an entity
the question or its options are about. This is a real coverage-quality
finding for the paper, not a solved problem — the cap makes traversal
*tractable*, it does not make the retrieved subgraph *relevant*. A
ranked/scored triple-selection step (e.g. by seed-entity proximity or
specificity, not just Neo4j traversal order) is a candidate Phase 3+
improvement, not implemented here.

**Alternatives considered:** raising the per-node degree cap's scope to be
global-per-question instead of per-node-per-hop in a single Cypher query
(rejected for this session — deterministic post-hoc truncation is simpler to
reason about and test, and preserves the existing per-node cap's own
hub-explosion protection as a first line of defense); dropping hub nodes by
an in/out-degree threshold at query time (a real improvement candidate,
deferred — needs a defensible degree threshold, which is a modeling decision
better made after Phase 3 shows whether hub noise measurably hurts downstream
accuracy).

**Consequences:** any future re-run of `precompute_kg_cache.py` after
changing `max_triples_per_question` invalidates existing cache files for
questions that hit the cap — no automatic cache-invalidation is implemented
(re-run with `--force` for affected questions). The 40-cap value should be
revisited once Phase 4's real `max_evidence=12` trim is implemented and its
interaction with a noisy 40-triple KG pool is measured.

## ADR-016 — KG-hit rate is an arithmetic artifact of entity fan-out, not KG coverage; novel-relation-rate must be relevance-conditioned to say anything about C1
Date: 2026-09-05 · Phase: 2 · Status: accepted (supersedes this ADR's own first-draft numbers, corrected within the same phase, before any commit)

**Context.** Per this phase's second amended requirement, `scripts/kg_probe.py`
measures what fraction of retrieved KG triples express a relation not already
entailed by that question's top-k dense passages — the number meant to
substantiate contribution C1 beyond mere KG-hit presence. A first pass at
this diagnostic (implementation problems below) produced numbers that were
methodologically flawed in four distinct ways, all caught by the project
owner reading `results/phase2_kg_probe/medqa_n200.json` directly before this
phase closed, not discovered later. This ADR is written from the corrected,
final run.

**Implementation problems, real and fixed:**
1. **The NLI model CLAUDE.md/`docs/04_EVALUATION.md` §3 pin does not exist.**
   `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli` was verified (via a live
   `HfApi.model_info` call, not memory) to 404 on the real Hugging Face Hub.
   Substituted `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` —
   same author, same MNLI+Fever+ANLI lineage plus LingNLI/WANLI, same 3-way
   entailment/neutral/contradiction head (verified via `AutoConfig.id2label`
   against the real downloaded config). Recorded as `src/kg/novelty.py:
   NLI_MODEL_NAME`, the single source of truth Phase 6 should import rather
   than re-typing CLAUDE.md's (incorrect) string; CLAUDE.md §3's stack table
   needs a follow-up correction to match.
2. **The 96% aggregate "KG-hit rate" is a permissive-definition arithmetic
   artifact, not evidence of PrimeKG content coverage.** The linker resolves
   only `entity_level_mapping_rate = 0.1122` of linked entities to a real
   PrimeKG node (13,129/14,789 unmapped — see ADR-014's crosswalk gap). But a
   question links ~74 entities on average
   (`mean_linked_entities_per_question = 73.945`), and the hit definition is
   *question*-level ("≥1 linked entity resolves AND ≥1 triple is retrieved").
   With that many independent low-probability draws per question, a hit is
   close to arithmetically guaranteed:
   `naive_expected_hit_rate_under_independence = 1 - (1 - 0.1122)^73.945 =
   0.99985`. The real observed `aggregate_hit_rate = 0.96` sits *below* this
   naive prediction (some correlation across a question's repeated-synonym
   entity mentions), which if anything makes the artifact story stronger, not
   weaker: **96% must never be read or cited as "PrimeKG covers 96% of MedQA
   content"** — it is a restated fact about entity fan-out at ~11% per-entity
   mapping, not a coverage result. `results/phase2_kg_probe/medqa_n200.json`
   now carries `entity_level_mapping_rate`,
   `naive_expected_hit_rate_under_independence`, and an explicit
   `hit_definition` string precisely so a future reader cannot make this
   mistake from the JSON alone.
3. **Novelty was first measured against the wrong evidence-bundle size.**
   `docs/01_ARCHITECTURE.md`'s tri-modal pipeline reranks to a real top-n=12
   `EvidenceBundle` (`configs/base.yaml: retrieval.top_k_rerank = 12`); the
   first pass scored novelty against only the top-3 dense passages, which
   structurally inflates apparent novelty (fewer chances for a passage to
   already state the relation). Corrected to score against the real 12.
   **Real effect of the fix:** `overall_novel_relation_rate` dropped from the
   first pass's 0.9563 to **0.8885** at the corrected k=12 — a genuine 6.8-point
   drop, confirming the review's prediction. The `novelty_max_triples_scored_
   per_question=10` scope cut (out of a ~38.4-triple/question mean, see
   ADR-015) remains uncorrected in this phase — the scored subset is a
   deterministic alphabetical-first-10 slice, which structurally
   over-represents alphabetically-early entity names (visible directly in the
   sample subgraphs: "Ac-" drug names like Acitretin/Acyclovir/Alfentanil
   recur across unrelated questions' scored subsets). A full-scope (all ~38
   triples × 12 passages) re-run is a legitimate ~55-minute follow-up if an
   unqualified figure is later needed; not run in this session.
4. **Novel ≠ useful — added a relevance-conditioned figure, since a high raw
   novelty rate is exactly what you'd expect from an NLI model correctly
   failing to entail irrelevant graph noise, novel or not.** Added
   `src/kg/novelty.py: is_triple_relevant` (pure, case-insensitive substring
   match of the triple's head/tail entity string against the question stem
   and each answer option — deliberately simple, not new ML). Real results:
   - `overall_relevant_rate = 0.4865` — **fewer than half** of retrieved
     triples even mention an entity the question or its options are about.
   - **`overall_novel_and_relevant_rate = 0.4078`** — this, not the raw 0.8885
     novelty figure, is the number that actually substantiates or undercuts
     C1's "KG evidence adds question-specific value" claim. It is **middling,
     not strong**: fewer than half of retrieved triples are both on-topic and
     something dense retrieval didn't already surface.
   - `overall_novel_given_relevant_rate = 0.8383` — conditional on a triple
     actually being relevant, it is still novel 84% of the time. This is the
     encouraging half: *when* the KG surfaces something on-topic, it usually
     adds information dense retrieval missed. **The bottleneck is precision
     of what gets retrieved (only 48.7% relevant), not novelty conditional on
     relevance** — this reframes ADR-015's hub-noise finding quantitatively
     rather than just qualitatively.
   - By category (non-suppressed): `pharmacology` 0.531 (highest — drug
     indication/contraindication triples map cleanly onto MCQ options),
     `treatment_management` 0.426, `pathology_diagnosis` 0.409, `physiology`
     0.315, `other` 0.294 (lowest — consistent with ADR-017's finding that
     `other` is a keyword-coverage residual, not a coherent category, which
     is itself a confound on this number: `other`'s low relevance may partly
     reflect the categorizer's own miss rate, not KG value specifically).
   - **The relevance heuristic's bias runs conservative, not inflationary**:
     plain substring matching under-counts paraphrases (a triple about
     "myocardial infarction" won't match a stem saying "MI" or "heart
     attack"), so `relevant_rate`/`novel_and_relevant_rate` are likely
     understated versions of the true ceiling — this cuts in the safer
     direction for any paper claim built on 0.4078, but means the number
     should be read as a floor, not a precise estimate.
5. Any category with `n < 15` (`MIN_CATEGORY_N_FOR_RATE`) has its rates
   suppressed to `null` in the output (raw counts remain visible) rather than
   showing a misleadingly precise percentage — currently only `anatomy`
   (n=8) is affected; see ADR-017.

**The honest bottom line for Phase 3 planning, stated plainly per the
project owner's explicit instruction:** PrimeKG's real entity-level coverage
of MedQA content is ~11%, the widely-reported 96% "hit rate" is a
question-level arithmetic artifact of high entity fan-out and must never be
cited as coverage, and **the KG modality's marginal value over dense
retrieval is not yet established** — the best available estimate
(`novel_and_relevant_rate ≈ 0.41`) is real but middling, bottlenecked by
retrieval precision (only ~49% of retrieved triples are even on-topic) more
than by redundancy with dense passages. This is the honest position C1 goes
into Phase 3 with, not a headline win.

**Alternatives considered:** a cheap string/entity-overlap heuristic instead
of NLI for the novelty check itself (rejected — less principled, and this
project already commits to the NLI model for the structurally identical
hallucination-rate calculation in Phase 6); scoring all triples against all
12 passages within this session (rejected — background-job-scale cost,
~55min, better spent once a hub-noise mitigation exists to test); reporting
only the raw `overall_novel_relation_rate` without the relevance-conditioned
figures (rejected outright by the project owner's review — "novelty without
relevance cannot support C1").

**Consequences:** any paper table or claim built on this phase's diagnostics
must cite `novel_and_relevant_rate` (0.4078, disclosed subset) as the C1
evidence figure, never the raw novelty rate (0.8885) alone, and must state
the entity-level mapping rate (0.1122) alongside any question-level hit rate
it reports. Improving the honest 0.41 requires either better crosswalk
coverage (ADR-014, currently drug/gene name-match only), better triple
selection to reduce hub noise (ADR-015, not yet implemented), or a
paraphrase-aware relevance check (this ADR, currently exact-substring only) —
three distinct, now-quantified follow-up targets for Phase 3+, not a single
fix.

## ADR-017 — MedQA question-category heuristic (no gold subject label exists)
Date: 2026-09-05 · Phase: 2 · Status: accepted

**Context.** This phase's first amended requirement is a per-category
KG-hit-rate breakdown for MedQA. Unlike MedMCQA (`docs/03_DATASETS.md`: 21
official `subject_name` tags), MedQA-USMLE ships no gold question-category
label at all, so any category breakdown needs its own, necessarily
imperfect, categorization step.

**Decision.** `src/kg/categorize.py: categorize_medqa_question` is a
plain keyword-bucket heuristic (a dict of category → keyword list, no ML)
over six fixed categories: `pharmacology`, `physiology`,
`pathology_diagnosis`, `anatomy`, `treatment_management`, `other` (fallback).
Deliberately simple and inspectable — CLAUDE.md's "don't add abstractions
beyond what the task requires" applies here; this is a diagnostic label, not
a modeling contribution, and dressing it up as more sophisticated than a
keyword match would misrepresent its evidentiary weight.

**Real result on the n=200 MedQA test slice used by `kg_probe.py`:**
`treatment_management` 48, `pathology_diagnosis` 43, `other` 40,
`pharmacology` 39, `physiology` 22, `anatomy` 8. Two honest weaknesses
observed directly, not hypothesized:
1. **`other` absorbs 20% of the real sample** (40/200) — a real, disclosed
   gap in keyword coverage, not force-fit into one of the five named
   buckets.
2. **Single-keyword misclassification is real and observed**: `test_00006`
   was bucketed `anatomy` solely because the stem contains the substring
   "artery" (inside "peripheral artery disease", a vascular-pathology
   question, not an anatomy-recall question) — a substring-count heuristic
   has no semantic disambiguation. `anatomy`'s n=8 is also the smallest
   bucket; per ADR-016, `kg_probe.py` enforces `MIN_CATEGORY_N_FOR_RATE = 15`
   and reports `anatomy`'s `hit_rate`/`novel_relation_rate`/relevance figures
   as `null` with `"underpowered": true` (raw counts still shown) rather than
   a naked 100% — this phase's own stated exit-test discipline (rate + n
   always shown together) is enforced in the artifact itself, not left to a
   reader's judgment.

**Alternatives considered:** using an LLM call to classify each question
(rejected — CLAUDE.md's zero-cost/determinism constraints make an
LLM-in-the-loop classifier for a diagnostic label disproportionate, and it
would need its own accuracy validation against hand-labels, which a 6-bucket
keyword heuristic doesn't need to claim); leaving MedQA unbroken-down by
category and accepting only the MedMCQA subject breakdown as "the" category
result (rejected — MedQA is the primary novelty benchmark per
`docs/03_DATASETS.md`, so its category breakdown is the one this phase's
strategic framing explicitly asked for).

**Consequences:** any paper table citing MedQA's per-category KG-hit rate
must also show `n` per category (already `kg_probe.py`'s convention) and
should flag `other` as a residual, not a category with its own coherent
identity. If a future session wants a less noisy breakdown, replacing this
heuristic with hand-labelled categories on a sample (mirroring
`docs/03_DATASETS.md`'s "tuned on 50 hand-checked items" precedent for the
UMLS linker threshold) is the natural upgrade path, not implemented here.
