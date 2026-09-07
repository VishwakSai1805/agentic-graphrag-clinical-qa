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
981,751,236 bytes, `aac8191d4fbc5bf09cdf8c3c78b4e75f`).

**Real raw edge count vs. the post-allowlist loaded count — stated explicitly
so neither number is mistaken for the other (follow-up correction, see the
2026-09-05 Phase 2 follow-up session):** the real, downloaded `kg.csv` file
contains **8,100,498 raw data rows** (verified by parsing the file with
Python's `csv` module and counting rows after the header, not `wc -l` alone —
the file does end in a newline so the two agree here, but the exact figure is
from a real parse, not an estimate). Of those 8,100,498 raw rows, **740,674
(9.1%) were loaded into Neo4j** — this is the count *after* deliberately
filtering to the project's fixed 7-item `KGConfig.edge_allowlist`
(`docs/03_DATASETS.md`), **not** a sign of an incomplete or failed load.
PrimeKG's authors' own published edge count for the *full* graph (all ~30
relation types they define, not just this project's 7-item allowlist) is
commonly cited as ~4M edges (`docs/03_DATASETS.md`'s "129k nodes / 4M edges"
line) because `kg.csv` stores every relation as **two** literal directional
rows per undirected fact (point 2 below) — 8,100,498 raw rows / 2 ≈ 4,050,249
unique undirected edges, consistent with that ~4M figure to within rounding.
So, reading the two headline numbers together: **8,100,498 raw directional
rows in the downloaded file → ~4.05M unique PrimeKG edges across all relation
types → 740,674 directional rows survive this project's 7-item allowlist
filter and are what's actually loaded into Neo4j.** The per-relation
breakdown of that final 740,674 (verified via `COUNT_EDGES_BY_RELATION`, and
independently reproduced byte-for-byte after a full volume wipe + reload in
the same follow-up session that added this paragraph):
`disease_phenotype_positive` 300,634; `disease_protein` 160,822;
`drug_effect` 129,568; `disease_disease` 64,388; `contraindication` 61,350;
`indication` 18,776; `off-label use` 5,136. Real load into Neo4j: 129,375
nodes, 740,674 edges.

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
Date: 2026-09-05 · Phase: 2 · Status: accepted - evidential basis SUPERSEDED by ADR-027 (2026-09-07): this ADR was decided on MedQA TEST-split probe output (AUDIT 2026-09-06 §3.1); re-derived on train-500 and UPHELD

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
Date: 2026-09-05 · Phase: 2 · Status: accepted (supersedes this ADR's own first-draft numbers, corrected within the same phase, before any commit) - evidential basis SUPERSEDED by ADR-027 (2026-09-07): this ADR was decided on MedQA TEST-split probe output (AUDIT 2026-09-06 §3.1); re-derived on train-500 and UPHELD

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
   first pass's 0.9563 to 0.8885 at the corrected k=12 — a genuine 6.8-point
   drop, confirming the review's prediction. The `novelty_max_triples_scored_
   per_question=10` scope cut (out of a ~38.4-triple/question mean, see
   ADR-015) remains a disclosed cut — full-scope (all ~38 triples × 12
   passages) scoring is a legitimate ~55-minute follow-up if an unqualified
   figure is later needed, not run in this phase.
   **Follow-up fix, same phase, before commit:** the scored subset was
   originally a deterministic alphabetical-first-10 slice, which structurally
   over-represented alphabetically-early entity names (visible directly in
   the sample subgraphs: "Ac-" drug names like Acitretin/Acyclovir/Alfentanil
   recurring across unrelated questions' scored subsets) — flagged by the
   project owner as biasing `novel_and_relevant_rate` in an unknown direction.
   Fixed: `scripts/kg_probe.py`'s selection now uses a **seeded random
   sample** of up to 10 triples per question, seeded per-question via
   `sha256(f"{seed}:{question_id}")` (reproducible independent of loop order,
   not just seeded once globally). **Real measured effect of the bias itself,
   now that it's fixed:** the alphabetical bias was real but small —
   `overall_novel_relation_rate` moved 0.8885→0.9016, `overall_relevant_rate`
   0.4865→0.4734, and critically **`overall_novel_and_relevant_rate` moved
   only 0.4078→0.4063** (Δ=−0.0016). The headline C1 figure barely moved; the
   *fix* mattered for methodological defensibility, not because the original
   number was substantially wrong.
4. **Novel ≠ useful — added a relevance-conditioned figure, since a high raw
   novelty rate is exactly what you'd expect from an NLI model correctly
   failing to entail irrelevant graph noise, novel or not.** Added
   `src/kg/novelty.py: is_triple_relevant` (pure, case-insensitive substring
   match of the triple's head/tail entity string against the question stem
   and each answer option — deliberately simple, not new ML). Real results:
   - `overall_relevant_rate = 0.4734` (seeded-random subset, superseding the
     alphabetical-subset 0.4865 above) — **fewer than half** of retrieved
     triples even mention an entity the question or its options are about.
   - **`overall_novel_and_relevant_rate = 0.4063`** — this, not the raw
     novelty figure, is the number that actually substantiates or undercuts
     C1's "KG evidence adds question-specific value" claim. It is **middling,
     not strong**: fewer than half of retrieved triples are both on-topic and
     something dense retrieval didn't already surface.
   - `overall_novel_given_relevant_rate = 0.8581` — conditional on a triple
     actually being relevant, it is still novel about 86% of the time. This
     is the encouraging half: *when* the KG surfaces something on-topic, it
     usually adds information dense retrieval missed. **The bottleneck is
     precision of what gets retrieved (only ~47% relevant), not novelty
     conditional on relevance** — this reframes ADR-015's hub-noise finding
     quantitatively rather than just qualitatively. See ADR-018 for a real,
     significance-tested attempt to fix this precision bottleneck via
     cross-encoder reranking.
   - By category (non-suppressed, seeded-random subset): `pharmacology`
     highest, `other` lowest — consistent with ADR-017's finding that `other`
     is a keyword-coverage residual, not a coherent category, which is itself
     a confound on this number: `other`'s low relevance may partly reflect
     the categorizer's own miss rate, not KG value specifically.
   - **The relevance heuristic's bias runs conservative, not inflationary**:
     plain substring matching under-counts paraphrases (a triple about
     "myocardial infarction" won't match a stem saying "MI" or "heart
     attack"), so `relevant_rate`/`novel_and_relevant_rate` are likely
     understated versions of the true ceiling — this cuts in the safer
     direction for any paper claim built on 0.4063, but means the number
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
(`novel_and_relevant_rate ≈ 0.41`, 0.4063 at the final seeded-random
measurement) is real but middling, bottlenecked by retrieval precision (only
~47% of retrieved triples are even on-topic) more than by redundancy with
dense passages. This is the honest position C1 goes into Phase 3 with, not a
headline win. See ADR-018: a real attempt to fix the precision bottleneck via
cross-encoder reranking gives a marginal, partially-significant improvement,
not a solved problem.

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
must cite `novel_and_relevant_rate` (0.4063, seeded-random disclosed subset)
as the C1 evidence figure, never the raw novelty rate alone, and must state
the entity-level mapping rate (0.1122) alongside any question-level hit rate
it reports. Improving the honest ~0.41 requires either better crosswalk
coverage (ADR-014, currently drug/gene name-match only), better triple
selection to reduce hub noise (ADR-015; see ADR-018 for a first real attempt
via cross-encoder reranking), or a paraphrase-aware relevance check (this
ADR, currently exact-substring only) — three distinct, now-quantified
follow-up targets for Phase 3+, not a single fix.

## ADR-018 — KG triple reranking (kg_rerank): a marginal, only partially significant fix to the retrieval-precision bottleneck; ships off by default
Date: 2026-09-06 · Phase: 2 · Status: accepted - evidential basis SUPERSEDED by ADR-027 (2026-09-07): this ADR was decided on MedQA TEST-split probe output (AUDIT 2026-09-06 §3.1); re-derived on train-500 and its primary-metric TIE VERDICT FLIPPED to a significant win (default stays off; see ADR-027)

**Context.** ADR-016 identified retrieval *precision*, not redundancy with
dense passages, as the real bottleneck on `novel_and_relevant_rate`: only
`overall_relevant_rate ≈ 0.47` of the 40 triples kept per question (ADR-015's
cap, saturated on 96% of questions) are even on-topic, because selection
into that cap was by raw Neo4j traversal order / a deterministic sort, not
by relevance to the question. The project already has a MedCPT cross-encoder
reranker (`src/retrieval/rerank.py`) used for dense+BM25 passages — reusing
it to score verbalized KG triples against the question, and selecting the
top-40 by that score instead of traversal order, is the natural fix to test.

**Decision.** Added `KGConfig.kg_rerank: bool = False` (default **off** —
this must not silently change Phase 3's default behavior or the
already-committed canonical `data/processed/kg_cache/` files) and
`KGConfig.kg_rerank_pool_size: int = 150` — when `kg_rerank=True`, live
traversal gathers this many candidates (wider than the final
`max_triples_per_question=40`) for the reranker to choose among; the pool
stays subject to ADR-015's degree cap and hop limit, so this is a larger
*pre-selection* pool for the reranker, not a return to unbounded traversal.
Also added `src/kg/rerank.py: select_triples_reranked`, which verbalizes each candidate
triple and scores it against the question text using the existing
`CrossEncoderReranker.rerank` (reused directly, not reimplemented), keeping
the top-`max_triples_per_question` by score. Wired into
`scripts/precompute_kg_cache.py` via `--kg-rerank`/`--cache-dir`, writing to
a separate `data/processed/kg_cache_kg_rerank/` location so the canonical
cache is untouched. `scripts/kg_rerank_ablation.py` compares the two
variants on the same 200 real MedQA test questions.

**Per the project owner's explicit instruction, this is reported as a
measured ablation result, not folded in as a silent improvement — including
running it through the exact same paired-bootstrap significance discipline
ADR-012 already established for this project, since CLAUDE.md forbids any
"A beats B" claim from a raw mean inequality alone.** Real point estimates
(n=200 MedQA test, naive traversal-order vs. kg_rerank):

| metric | naive | kg_rerank | raw Δ |
|---|---|---|---|
| `overall_relevant_rate` | 0.4734 | 0.5276 | +0.0542 |
| `overall_novel_and_relevant_rate` | 0.4063 | 0.4443 | +0.0380 |

Both deltas look like wins on point estimates alone. Real paired-bootstrap
check (10k resamples, seed 42, `src/retrieval/metrics.py:
paired_bootstrap_ci`, paired by `question_id`, n=192 after excluding 8
questions with zero scoreable triples in *both* variants — not a
variant-specific exclusion bias):

| metric | paired mean diff | 95% CI | verdict (ADR-012 rule) |
|---|---|---|---|
| `relevant_rate_for_question` | +0.0542 | **[+0.0026, +0.1052]** | **significant win** (but the lower bound sits close to zero — a marginal, not robust, win) |
| `novel_and_relevant_rate_for_question` (primary, per ADR-016) | +0.0380 | **[−0.0125, +0.0875]** | **tie** (CI includes 0) |

**Real, honest verdict, exactly the ADR-012 scenario repeating:** the
primary C1 evidence metric (`novel_and_relevant_rate`) does **not** survive
significance testing at n=200 — reranking's positive point estimate on this
metric is a tie, not a demonstrated win, and must not be reported as
"reranking measurably helps" (an earlier draft of this finding said exactly
that, from the point estimate alone, before the significance check ran).
The secondary, more direct precision metric (`relevant_rate`) is a real but
narrow significant win — MedCPT reranking does plausibly select more
on-topic triples, but the effect is not yet large enough to also carry the
noisier, doubly-conditioned primary metric past significance at this sample
size. n=200 (192 paired) may simply be underpowered for an effect this size,
given the CI width relative to the point estimate.

**Decision on shipping:** `kg_rerank` ships as a real, tested,
config-gated **off-by-default** option — implemented and available for
Phase 3+ to opt into and re-test at larger n (e.g. the full 500-question
MedQA slice, or as part of the Phase 7 ablation grid, where it would earn
its own V-variant-style row rather than a Phase 2 side-diagnostic), but not
adopted as this phase's default because the evidence for it is not yet
strong enough to justify changing already-committed baseline behavior.

**Alternatives considered:** shipping `kg_rerank=True` as the new default on
the strength of the raw point estimate (rejected outright — exactly the
methodology CLAUDE.md's significance rule and ADR-012 exist to prevent);
declaring the feature a failure and dropping it because the primary metric
tied (rejected — the secondary metric's real, if narrow, significant win is
genuine signal that the underlying mechanism is doing something real; a tie
on an underpowered n=200 sample is not evidence of zero effect); running a
larger n immediately to resolve the tie (deferred — better spent as part of
Phase 3+'s planned larger-scale runs than as more Phase 2 wall-clock).

**Consequences:** any future claim that `kg_rerank` improves KG evidence
quality must cite the paired-bootstrap verdict above (tie on the primary
metric), not the raw point-estimate table, until a larger-n re-test changes
that verdict. The feature, its cache location, and its ablation script are
real and committed, so re-testing at scale requires no new implementation —
only a bigger run and a fresh significance check.

## ADR-017 — MedQA question-category heuristic (no gold subject label exists)
Date: 2026-09-05 · Phase: 2 · Status: accepted - evidential basis SUPERSEDED by ADR-027 (2026-09-07): this ADR was decided on MedQA TEST-split probe output (AUDIT 2026-09-06 §3.1); re-derived on train-500 and UPHELD

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

## ADR-019 — Phase 2's KG caches were never committed; ADR-018's "already-committed canonical" is corrected of record; a checksum manifest replaces committing the files
Date: 2026-09-06 · Phase: 3 · Status: accepted (corrects a factual claim in ADR-018)

**Context.** ADR-018 states, twice, that `data/processed/kg_cache/` holds the
"already-committed canonical" KG cache. This is false. `.gitignore` is
`data/*` with a single negation `!data/MANIFEST.json` (ADR-013); nothing under
`data/processed/` has ever been tracked. Phase 3 found the entire Phase 2 KG
cache (200 MedQA test + 1000 PubMedQA + 500 MedMCQA question files, plus the
200-file `kg_cache_kg_rerank/` variant) and the PrimeKG raw inputs
(`nodes.csv`, crosswalk) existing in exactly one place: the untracked
`.claude/worktrees/phase2-kg-layer/` worktree directory, one `git worktree
remove` from deletion with no other copy. Phase 7's reproducibility depends on
these artifacts; "canonical" for a file that is gitignored and un-backed-up is
not a defensible position.

**Decision.**
1. Relocate the caches and the required PrimeKG inputs (`nodes.csv` —
   MD5 `7f9ab4109c54049e819ecd14e15a6038`, matches ADR-014; `crosswalk/`) into
   the main working tree under `data/processed/` and `data/raw/primekg/`. They
   stay gitignored — ~1700 small JSON files would churn the repo on every
   re-precompute and the real provenance is (seed, split, PrimeKG graph),
   not the bytes.
2. Add `scripts/kg_cache_manifest.py` and a `kg_cache` block in the tracked
   `data/MANIFEST.json` recording, per cache tree: `n_files`, an aggregate
   SHA-256 (SHA-256 over the sorted `<relpath>\t<file-sha256>` lines), and the
   exact `precompute_kg_cache.py` command that regenerates it. `--check`
   exits non-zero on drift; `tests/test_kg_cache_manifest.py` covers the
   digest (stable / content-sensitive / rename-sensitive / order-independent).
3. `kg.csv` (982 MB) is *not* relocated — it is only needed to load Neo4j and
   is re-fetchable from Dataverse via `scripts/load_primekg.py` (ADR-014).
4. `.claude/worktrees/` added to `.gitignore` (a stray `git add .` would
   otherwise pull a whole nested checkout in). The Phase 2 worktree is kept
   for now as the only remaining copy of `kg.csv`; `git worktree remove` is a
   documented housekeeping step once a full re-precompute from the relocated
   inputs has been verified.

**Alternatives considered:** commit the ~1700 cache files (rejected — repo
churn, and it does not actually make them more reproducible); commit nothing
and rely on the ADR-007 exit test (rejected — that verifies reproducibility
given the graph, but gives no way to detect that an on-disk cache has drifted
from what a run used); a git-lfs pointer set (rejected — adds an LFS
dependency for artifacts that are cheap to regenerate).

**Consequences:** `scripts/kg_cache_manifest.py --check` should be run before
any Phase 7 grid run that consumes the cache, and re-`--write`n whenever a
split is (re)precomputed. A future reader who sees ADR-018's "already-committed
canonical" phrasing should read it as "canonical, and now checksum-tracked in
`data/MANIFEST.json`" — the caches were never in git and are not now.

## ADR-020 — Phase 3 dev slice is MedQA train, not test; baseline set, B3 spec, and small decisions
Date: 2026-09-06 · Phase: 3 · Status: accepted

**Context.** Phase 3's roadmap exit test says "200-item MedQA **dev** slice".
`docs/03_DATASETS.md` defines MedQA's dev split as "500 from train", with the
official test split reserved. Phase 2 only ever precomputed a KG cache for
MedQA **test** (200 q, seed 42). Running Phase 3's KG-using baselines (B2, B3)
on a roadmap-faithful dev slice therefore needed a fresh precompute.

**Correction (2026-09-07, AUDIT 2026-09-06 §2.6 / ADR-027):** the sentence
below originally read "the test-set hygiene held since Phase 1 is worth more
than the few minutes saved" as an unqualified project-wide claim. That was
false: it held for the *retrieval corpus and Phase 3/4 baselines* (this ADR's
actual subject), but the Phase 2 KG probe had already run on the MedQA test
split at the time this ADR was written (`medqa_n200.json`,
`per_question[*].question_id = "test_*"`), and ADR-015/016/017/018 were
decided from it — see ADR-027 for the full remediation. The claim below is
read as scoped to Phase 3/4 baseline plumbing, not as a project-wide
guarantee.

**Decision.**
- **Dev slice = MedQA train split, `load_question_sample("medqa", "train",
  limit, seed=42)`.** The Phase 3 exit test runs on 20 of these (CLAUDE.md §5
  "20-question dev slice"); a KG cache for the 20 was precomputed
  (`--split train --limit 20`, Neo4j already loaded, 88 s). The official test
  split stays untouched until Phase 7. The alternative (reuse the existing
  test-slice cache) was rejected by the project owner: test-set hygiene for
  the *retrieval corpus and baselines* is worth more than the few minutes
  saved. (This did not hold for the KG layer at the time — see the correction
  above and ADR-027.)
- **Baseline → retrieval mapping.** B0 zero-shot (no retrieval, ungrounded by
  construction — a documented carve-out from CLAUDE.md's "empty citation set
  is a bug" rule, since B0 is the ungrounded floor). B1 dense-only
  (`modalities={"dense"}`). B2 hybrid, no agents (`{"dense","lexical","kg"}`,
  one generation call). B3 as below. B2 maps to Phase 7's V3 (tri-modal, no
  agents); there is no separate "dense+bm25 only" Phase 3 baseline — that is
  V2, produced by the Phase 7 grid, not Phase 3.
- **B3 = MEGA-RAG-style binary verifier reimplementation.** Tri-modal
  retrieve (same bundle as B2). For each option, one `temperature=0` call
  returning `{verdict: SUPPORTED|NOT_SUPPORTED, evidence_id}`. If exactly one
  option is SUPPORTED → predict it (`verifier_decided=True`). Otherwise (0 or
  ≥2 SUPPORTED) → fall back to one B2-style generation call
  (`verifier_decided=False`). B3 gets the identical model, temperature,
  retrieval budget, evidence block and prompt care as B2 — it is the "binary
  bottleneck" C2 argues against, reimplemented in good faith, not a strawman.
  `scripts/run_baselines.py` reports `verifier_decision_rate` (fraction where
  the verifier decided vs. fell back) and the `n_supported` distribution:
  a high fallback rate is *direct empirical evidence for the binary
  bottleneck*, and per the project owner may be the most useful number B3
  produces.
- **Small decisions, recorded per CLAUDE.md discipline.** (a) No separate
  `configs/baseline.yaml` — all four baselines resolve the same
  `configs/base.yaml`; the only thing that varies (which modalities feed the
  bundle) is a code-level property of the baseline, not a config knob.
  `run_id`'s `<variant>` field (`B0`…`B3`) carries the distinction.
  (b) `Evidence.id` for passages is the corpus `chunk_id` verbatim (already
  `chunk::<doc_id>#<span>`, e.g. `chunk::textbook::Pathology_Robbins#618`) —
  no synthesis, uniform with Phase 2's `kg::` ids for citation validation.
  (c) `contamination_assertion` for MedQA is a recorded no-op: MedQA questions
  are not generated from indexed corpus docs, so the PubMedQA test-PMID
  blocklist (`src/eval/contamination.py`) does not apply; MedQA textbooks are
  in the corpus by design (ADR-009). Logged in every `metrics.json`, not
  skipped.
  (d) `src/eval/metrics.py` ships only `accuracy` + `macro_f1` now; Phase 6
  extends it (P@K, MRR, ECE, RAGAS/NLI). `macro_f1` for 4-option MedQA is
  labelled a **position-bias detector, not a skill metric** — a model that
  always answers "A" scores ~0.25 accuracy but macro-F1 collapses; Accuracy
  is the headline.

**Consequences:** Phase 3's baseline numbers are a plumbing/parity check on
n=20, never a comparison result — `scripts/run_baselines.py` makes no
"A beats B" claim and n=20 forbids one (CLAUDE.md significance rule). Phase 7
re-runs all baselines fresh on the seeded 500-question test slice; the Phase 3
`results/` dirs are development artifacts, not paper inputs. A dev KG cache
for the full 500-question MedQA train slice is still outstanding (tracked
under `data/MANIFEST.json: kg_cache` regen commands and Phase 6's needs).

## ADR-021 — `llm.num_ctx` + a hard context-budget guard: a silent tail-truncation was corrupting the hybrid baselines
Date: 2026-09-06 · Phase: 3 · Status: accepted · amended 2026-09-07 (AUDIT
2026-09-06 §6.6: no-op perturbation row added; the B1 accuracy delta is
retracted as an effect — the decision itself stands on the B2/B3 categorical
evidence)

**Context.** The first Phase 3 exit run used Ollama's default context window
(4096 tokens, confirmed via `/api/ps`). The tri-modal prompt is question +
options + 12 reranked passages (~256 tok each) + up to 40 verbalised KG
triples, and `_format_evidence` appends the KG triples **last**. At 4096
tokens llama.cpp silently truncates the tail — i.e. it drops exactly the KG
modality that B2/B3 exist to measure, and the JSON-format instruction with it.
This is not a "context-optimal vs not" tuning question; it is a correctness
bug of the same class as the Phase 0 cache-hit token double-count: silent, and
it corrupts numbers rather than crashing. Left unfixed it would propagate
through Phases 4–7 as "dense+lexical mislabelled as tri-modal".

**Real measured effect** (`scripts/run_baselines.py`, MedQA train dev-20,
seed 42 — before = num_ctx 4096, after = num_ctx 8192; n=20, no comparison
claim, plumbing only):

| baseline | acc before | acc after | grounded before | grounded after | parse_error before | parse_error after |
|---|---|---|---|---|---|---|
| B0 (no retrieval) | 0.55 | 0.55 | 0.00 (by construction) | 0.00 | 0 | 0 |
| B1 dense-only     | 0.55 | 0.60 | 0.00 | 0.95 | 2 | 0 |
| B2 hybrid         | **0.15** | **0.65** | 0.00 | 0.95 | **12** | 0 |
| B3 binary verifier| **0.15** | **0.65** | 0.00 | 0.95 | **12** | 0 |

**Amendment 2026-09-07 (AUDIT 2026-09-06 §6.6) — the no-op perturbation row,
and what it does to the B1 reading above.**

The table above pairs an *intervention* (num_ctx 4096 -> 8192) with a *delta*
and, for B1, invited that delta to be read as the fix's effect. It has no
control. B0 is that control: B0 does no retrieval, so its prompt is question +
options only, nowhere near 4096 tokens, and the intervention cannot touch it.
Measured on the same two batches, from `predictions.jsonl`:

| baseline | intervention reaches the prompt? | answers changed | net acc | reading |
|---|---|---|---|---|
| **B0 (no-op control)** | **no** — nothing to truncate | **1/20** (`train_06912` A→B) | 0.55 → 0.55 | pure run-to-run nondeterminism |
| B1 dense-only | yes | 5/20 | 0.55 → 0.60 (net +1) | see below |

B0's flip changed an answer without changing accuracy only because both A and
B are wrong (gold D). A single flip that happens to cross the gold option is
worth exactly 5 accuracy points at n=20. So the noise floor for an accuracy
delta between these two batches is **at least ±0.05, measured, under an
intervention that provably does nothing.**

B1's `0.55 → 0.60` is exactly that size. It is also not one question changing
its mind: B1 changed **5 of 20** answers between the batches, netting +1.
Two are genuine truncation recoveries (`train_00520`, `train_01424`:
`parse_error` → `ok`, and *both are still wrong*). The other three
(`train_01679` B→C, gold B — lost; `train_04012` D→A, gold A — gained;
`train_08279` B→C, gold C — gained) parsed cleanly in *both* runs, so
truncation never touched them; the model simply answered differently. The
headline `+0.05` is the net of that churn, and every question it actually
gained was gained on an input the fix did not repair.

**Therefore: the B1 0.55 → 0.60 delta is inside the noise floor and is no
longer read as an effect of the num_ctx fix, here or in
`docs/04_EVALUATION.md` §5.4.** The evidence that ADR-021 fixed a real bug is
*not* that delta and never needed to be — it is the categorical, far-outside-
noise B2/B3 signal: `parse_error` 12/20 → 0/20 and `grounded_rate`
0.00 → 0.95, changes no plausible 1-flip noise process produces. B1's
`parse_error` 2 → 0 belongs to that same categorical evidence; B1's accuracy
number does not. Any future n=20 comparison in this project must clear this
measured ±0.05 floor before it is called an effect, or be reported as a tie
(CLAUDE.md: "A tie is reported as a tie, never as a win").

The B2/B3 collapse under truncation (0.15) is entirely an artifact: 12/20
prompts were cut mid-output so the model never emitted a parseable answer.
The grounded-rate jump (0.00 → 0.95) is a second, independent bug fixed in
the same round — `_format_evidence` had numbered each item `N. [id] text`, so
the model cited `"[chunk::…]"` (brackets included) or the bare number `"12"`,
neither of which matched a bundle id; now items render as `[id] text` and
`_normalize_cite` strips surrounding brackets/quotes.

**Decision.**
1. `LLMConfig.num_ctx` (default **8192**) and
   `LLMConfig.context_completion_reserve` (default 512), surfaced in
   `configs/base.yaml`. qwen2.5:7b trains at 32k, so 8192 is safe and fits the
   full bundle with headroom; VRAM cost of the larger KV cache is a few
   hundred MB on a 12 GB card.
2. `src/llm/client.py` sends `num_ctx` in the Ollama options **only** on the
   `provider == "ollama"` path (Groq/Gemini reject unknown params).
3. `num_ctx` is part of the LLM cache key (`params` dict): a call under a
   different context window is a genuinely different call and must not replay
   a smaller-window cache entry. (This is why the re-run actually
   re-executed rather than serving the truncated responses.)
4. Hard guard: before any cache read or network call, `chat()` token-counts
   `system + prompt` (via `litellm.token_counter`, model-real tokenizer where
   available) and raises `LLMContextBudgetError` if it exceeds
   `num_ctx - context_completion_reserve`. Per CLAUDE.md's budget-guard rule
   this is a **recorded hard stop, never a silent truncation**. Feature code
   that batches calls (`src/eval/baselines.py`) catches it and records the
   item as `status="context_exceeded"` (new `Prediction.status` member), which
   `accuracy`/`macro_f1` score as wrong and `run_baselines.py` surfaces in
   `status_counts`.

**Alternatives considered:** raise `num_ctx` without the guard (rejected —
the guard is cheap insurance against the same failure re-appearing when the
Phase 4 agent bundle or a longer question pushes past 8192; a silent failure
here is exactly what bit us); pre-truncate the bundle to fit (rejected for
Phase 3 — trimming evidence is the Phase 4 agent layer's job with
`max_evidence=12`, and doing it implicitly in the gateway would hide the
same information loss the guard exists to make loud); keep `num_ctx` out of
the cache key and clear `.cache/llm` on the re-run (rejected — `num_ctx`
provably changes output, so it belongs in the key).

**Consequences:** the B2/B3 context-truncation escalation is resolved before
Phase 4, as the project owner required. Phase 4's agent prompts carry an
`EvidenceBundle` of similar size and now inherit both the raised window and
the guard. `docs/04_EVALUATION.md` §0.1's per-call prefill budget (~3.7 s at
958 tok/s for a ~3.5k-token bundle) is unchanged — 8192 is the *ceiling*, not
the typical prompt size.

## ADR-022 — Config inheritance via `_extends:`; deep-merge, resolved config recorded
Date: 2026-09-06 · Phase: 4 · Status: accepted

**Context.** `load_config` was a flat `yaml.safe_load` -> `model_validate`.
Phase 4 needs `configs/agents.yaml` (agents on); Phase 5 needs a calibration
variant; Phase 7 needs 8 variant configs. Full copies of `base.yaml` per
variant drift silently -- a retrieval knob changes in base, seven copies keep
the stale value, and a V4/V5 grid run then uses stale retrieval config with
nothing flagging it.

**Decision.** A config file may carry a top-level `_extends: <path>` (relative
to its own dir, or absolute). `load_config` loads the parent, then deep-merges
the child over it: dicts merge key-wise, every other type (lists included) is
replaced wholesale. `_extends` chains recursively. `configs/agents.yaml` is now
~15 lines (just the `agents:` block + `_extends: base.yaml`). The fully
resolved config is still what `results/<run_id>/config.yaml` records and what
`config_hash` hashes, so run provenance never depends on the indirection.

**Alternatives:** full-copy variant files + a test asserting shared keys match
base (rejected -- the test is a patch over a design smell, and a forgotten
test lets drift through); a real templating layer like Hydra (rejected --
heavyweight for a handful of 15-line overlays).

**Consequences:** Phase 5 and Phase 7 variant configs all use `_extends:
base.yaml` (or `_extends: agents.yaml` for V5). List fields are replace-only,
so overriding one element of `llm.fallback_chain` means restating the whole
list -- documented in `load_config`'s docstring.

## ADR-023 — Agent evidence bundle: modality-reserved slots with passage backfill; MixReport is a stored variable
Date: 2026-09-06 · Phase: 4 · Status: accepted (implements the design constraint recorded in docs/02_ROADMAP.md Phase 4)

**Context.** `docs/01_ARCHITECTURE.md` caps the agent bundle at
`max_evidence=12`. The Phase 3 pipeline emits 12 passages + up to 40 KG
triples. A single top-12 cross-modal ranking would eliminate KG almost
entirely: MedCPT's cross-encoder is a query-*article* model, so a short
verbalised triple systematically scores below a full passage -- the agent
layer would be "tri-modal in name, dense-only in fact", the same silent-
elimination class as the Phase 3 context-truncation bug (ADR-021).

**Decision.** `src/agents/evidence_select.py`: rank *within* each modality and
reserve slots via `agents.evidence_split = [n_passages, n_triples]` (default
`[8, 4]`). The KG count is a **floor, not a fixed ratio** -- unfilled KG slots
backfill with passages so the bundle reaches `max_evidence` whenever that much
evidence exists (a zero-triple question must not hand the agents 8 items when
B2 got 12; that confound is worse than the ratio wobble). Rank the KG slot
with the MedCPT cross-encoder (`agents.kg_slot_rerank=True`; reuse of ADR-018's
mechanism at agent-selection granularity -- its value here is a Phase 7
ablation question, not inherited from ADR-018's n=200 tie). The realised split
is returned in `MixReport` (requested split, n_passages, n_triples,
kg_available, `kg_slot_filled`, `backfilled_passages`) and stored on every
`SupportMatrix` -- a per-question **analysis variable**, so Phase 7 can report
the `evidence_split` ablation both overall and conditioned on
`kg_slot_filled`, not just as a log line.

**Real Phase-4 dev-20 result:** kg-slot filled on 17/20 questions, all-passage
bundle on 3/20 (the zero/low-triple questions). `evidence_split` ablation
deferred to Phase 7.

**Alternatives:** strict 8/4 with a <12 bundle when KG underfills (rejected --
starves the agent variants vs B2 for a reason unrelated to agent quality);
dynamic per-question split by a learned gate (rejected -- adds a confound to
the ablation, ADR-002's spirit).

## ADR-024 — Phase 4 agent graph: deterministic Hypothesis, source-precedence-only Conflict, and the conflict-resolver-never-resolves finding
Date: 2026-09-06 · Phase: 4 · Status: partly superseded by ADR-025 (the §3
`conflict_fire_rate = 0.55` / resolved-0/unresolved-18 numbers below are
retracted — they were an artifact of the wrong option-level conflict
definition; ADR-025's claim-level `conflict_fire_rate = 0.00` is the correct
figure. Everything else in this ADR — the deterministic Hypothesis agent, the
source-precedence-only Conflict tiebreak mechanism, and the graph structure —
stands)

**1. Hypothesis is deterministic by default.** `docs/01`/ADR-001 allow the
Hypothesis agent to sample. It decomposes *given* MCQ options into claims --
diversity buys little there, and determinism buys the node-level replayability
ADR-001 explicitly wanted. `agents.hypothesis_sample=False`,
`hypothesis_temperature=0.0` by default; sampling is an ablation flag. This
removes the "cache-cleared rerun produces different matrices" risk entirely.

**2. The Conflict agent is a source-precedence resolver, NOT a semantic one.**
`docs/01` specifies "source precedence & recency" with a 4-tier order
(KG-curated > guideline/textbook > abstract conclusion > abstract background).
On this corpus that is unimplementable: PubMed chunks carry no
conclusion/background section tags and no publication dates (verified against
`data/processed/corpus.jsonl`). So the implemented resolver is 3-tier --
`kg > textbook > pubmed_abstract` from `Evidence.source_meta` -- and "recency"
is dropped. The winner is computed **deterministically** from `source_rank`
(lower rank wins; tied rank -> unresolved, no LLM call); the LLM is invoked
only when precedence gives a clear winner, and only to judge whether the
contradiction is genuine (same clinical point) or spurious -- it may confirm
the precedence winner or downgrade to UNRESOLVED, never flip to the losing
side.

**3. The finding, elevated per the B3 lesson (ADR-020 / docs/04 §5.4).**
On the Phase 4 MedQA train dev-20 (seed 42):
`conflict_fire_rate = 0.55` (11/20 questions had an option with both SUPPORT
and CONTRADICT cells), but **resolved 0 / unresolved 18** -- every one of the
18 option-level conflicts came back "tied source precedence (rank 1)", and the
**Conflict LLM ran zero times**. Mechanism: nearly all audited evidence is
textbook-tier passages (rank 1); KG triples (rank 0) rarely land on both sides
of a single option's conflict. So the source-precedence resolver has no
differential precedence to act on, and the deterministic fallback always ties.
**C2's "explicitly resolves contradictions" currently rests on a component
that fires but resolves nothing on this corpus.** This is recorded, not
softened. Candidate fixes for Phase 7+: a real evidence-tier signal (PrimeKG
edge curation tier; textbook vs guideline vs review distinction; abstract
section classifier), or a semantic conflict-resolution step that does not
depend on source tiers at all. Not attempted in Phase 4.

**4. Small decisions.** (a) `status="agent_error"` added for a schema
validation failure surviving one repair -- a recorded hard stop distinct from
`budget_exceeded`. (b) Call granularity (the architecture doesn't specify it):
Hypothesis 1 batched call, Audit 1 call/option, Conflict ≤1 call/conflicted
option. Round-1 worst case 1+4+4 = 9 ≤ 12. (c) Round-2 re-retrieval was
initially gated on `n_options + conflict_ceiling` free calls, which made it
unreachable with 4 options at the 12-call budget -- fixed to gate on the
re-audit cost only (`n_options`), since a no-signal round rarely leads to
conflict. Real dev-20: re-retrieval fired on 1/20, `budget_exceeded` on 0/20
(well under the 20% "raise the cap" threshold from the Phase 4 approval).
(d) LangGraph checkpointing is `InMemorySaver` + a full per-node JSON trace
dump per question (`results/<run_id>/traces/`, gitignored); `SqliteSaver` for
true node-replay-at-scale is deferred to the Phase 6 harness.

## ADR-025 — Conflict is a claim-level property; semantic check first, source precedence as tiebreaker
Date: 2026-09-06 · Phase: 4 (pre-Phase-5 correction) · Status: accepted · Amends docs/01_ARCHITECTURE.md

**Context.** The Phase 4 exit run showed the Conflict agent firing on 55% of
questions, resolving 0, all 18 "tied source precedence", the Conflict LLM
running 0 times, and `rank-1-by-support` at 0.35 vs B2's 0.65 on the same dev
slice -- the agent layer was *below* its own no-agent baseline. Two spec-level
bugs, not implementation bugs:

1. **`docs/01` defined conflict at the option level** ("an option has both
   SUPPORT and CONTRADICT mass above ε"). But an option with claim 1 supported
   and claim 3 contradicted is *partially supported*, not contradictory -- and
   that is already fully captured by `(s⁺, s⁻)`. Treating it as a "conflict"
   invented conflicts everywhere and then, finding no differential precedence,
   marked options `unresolved` and suppressed their logits.
2. **Precedence was consulted before the LLM.** The deterministic
   `min(source_rank)` comparison ran first; on this corpus nearly all evidence
   is textbook-tier (rank 1), so it tied and returned `unresolved` without the
   "Conflict Resolution agent" ever executing. The agent had never resolved
   anything.

**Decision (amends `docs/01`):**
- **Conflict is claim-level.** It exists for a `(option_id, claim_idx)` iff
  that claim carries both SUPPORT and CONTRADICT cells *from different evidence
  ids*. `src/agents/graph.py: _claim_level_conflicts`.
- **Semantic check first, precedence second.** On a claim-level conflict the
  Conflict LLM (`conflict.v2` prompt, temperature=0) judges only whether the
  two evidence items *genuinely contradict on the same clinical point*
  (`ConflictOut{genuine_contradiction: bool}` -- no `winner` field). Only if
  genuine does `min(source_rank)` pick the winner in code; only if precedence
  also ties is the claim `unresolved`. A non-genuine "conflict" is recorded as
  `spurious` and changes nothing.
- **`unresolved` no longer suppresses a logit on a weak rule.** The Phase 4
  pre-calibration `rank-1-by-support` sanity metric scores `s⁺ − s⁻` with no
  `resolved` penalty. Whether `unresolved` should down-weight an option at all
  is Phase 5's calibration decision, made against dev ECE, not a Phase 4
  guess. `OptionSupport.resolved` / `unresolved_reason` are still recorded for
  Phase 5 to use.
- `docs/01`'s "abstract conclusion > background" precedence split and "recency"
  stay dropped (ADR-024): unimplementable on this corpus.

**Real result after the fix** (`results/medqa_V4_<sha>_<ts>`, same MedQA train
dev-20, seed 42): see `docs/04_EVALUATION.md` §5.5 -- filled from the re-run.

**Alternatives considered:** keep option-level conflict but raise ε (rejected
-- ε tuning cannot fix a definition that conflates "partially supported" with
"contradictory"); keep precedence-first but add more tiers (rejected -- the
tiers don't exist in the data; the LLM judgement is the part that was missing);
drop the Conflict agent entirely and fold everything into Audit (considered --
if genuine claim-level contradictions turn out to be rare on this corpus,
`docs/00` C2 should be reframed around the Audit layer's SUPPORT/CONTRADICT/
NEUTRAL discrimination rather than the Conflict resolver; the re-run's
`conflict_fire_rate` is the number that decides this).

**Consequences:** `conflict.v1.md` / the v1 `ConflictOut{winner}` schema are
retired (v1 file kept for provenance). Any future claim that the Conflict
agent contributes to accuracy must cite the claim-level `conflict_fire_rate`
and the resolved/unresolved/spurious split, not an option-level fire rate.

## ADR-026 — Calibration weights are fitted on `pqa_labeled` dev, not `pqa_artificial`
Date: 2026-09-07 · Phase: 5 (pre-requisite) · Status: accepted

**Context.** AUDIT 2026-09-06 §3.3. `docs/03_DATASETS.md` reserved
`pqa_artificial` (211k) *only* for fitting calibration weights.
`configs/base.yaml` independently builds the retrieval corpus from a
20,000-document `pqa_artificial` subset (`base.yaml: pubmed_subset_size:
20000`, `pubmed_seed: 42`; `data/MANIFEST.json: n_pubmed_docs = 20000`,
`n_chunks_per_source.pubmed_abstract = 32683`). Nothing connected the two
statements, so the fitting set and a fifth of the retrieval corpus were drawn
from the same pool.

Why that is not a cosmetic overlap: a calibration temperature maps evidence
scores to probabilities. If the fitting items' own source abstracts are
indexed, retrieval on the fitting set is near-perfect — this is precisely the
`Recall@10 = 0.995` saturation ADR-012 recorded on `pqa_artificial` — so the
temperature is fitted against an evidence-score distribution that no reported
test question will ever see. The result is a systematically over-confident
calibration on the test set, and ECE (C3's headline metric) would be measuring
the leak rather than the method. Phase 5 was unsound until this was resolved.

`src/eval/contamination.py` could not catch it: `blocklisted_pmids` blocks only
the `pqa_labeled` TEST PMIDs and raises for every other dataset, so nothing in
the codebase held an opinion about the fitting set at all.

**Decision.** Fit calibration on the **`pqa_labeled` DEV half (500 items)**.

1. `calibration_fit_pmids()` returns `pubmedqa_split()[1]` — the dev half. The
   fit set is therefore *by construction* the exact complement of the reported
   test half, from the same single-source-of-truth split function.
2. `assert_calibration_fit_set_uncontaminated(corpus_path)` hard-raises
   `ContaminationError` if any fit PMID is indexed. **Phase 5 must call this
   before fitting** — per CLAUDE.md this is an assertion in code, not a comment.
3. `pqa_artificial` is reclassified as a retrieval-corpus source only: never a
   reporting set, and no longer a fitting set.

**Verified, not assumed.** `pqa_labeled` draws from a pubid pool disjoint from
`pqa_artificial`, which until now was only a code comment in
`scripts/build_corpus.py`. Executed against the built corpus on 2026-09-07:
20,000 distinct PubMed doc_ids indexed, of which **0 of the 500 dev
(calibration-fit) PMIDs and 0 of the 500 test PMIDs** appear. The overlap the
audit identified is therefore removable with no index rebuild.

**Alternatives considered.**
- *Extend the blocklist to the exact `pqa_artificial` calibration items and
  rebuild the corpus* (the audit's other option) — rejected. It costs a full
  ~17-minute GPU index rebuild, shrinks the corpus, and leaves calibration
  fitted on a set drawn from the same distribution as 20k indexed documents
  (near-duplicate abstracts by other authors remain retrievable even once the
  exact PMIDs are excluded). Switching pools removes the leak at the source
  instead of patching its most visible symptom.
- *Fit on MedQA train* — rejected for now: calibration is fitted per dataset,
  and PubMedQA is the dataset whose parity claim (C4) depends on it.

**Consequences.** The Phase 5 fitting set is 500 items rather than 211k. That
is a real cost — a 500-item fit supports a single global temperature, not a
richly parameterised per-feature calibrator — and it must be stated as a
limitation when C3 is reported, not quietly absorbed. It does not change the
reported test set, the corpus, or any index. `docs/03_DATASETS.md` is corrected
of record; `configs/base.yaml`'s corpus block is unchanged and correct as it
stands.

## ADR-027 — Phase 2 KG decisions re-derived on a MedQA *train* slice; ADR-018's verdict flips
Date: 2026-09-07 · Phase: 2 (remediation) · Status: accepted · supersedes the
test-split basis of ADR-015, ADR-016, ADR-017, ADR-018

**Context.** AUDIT 2026-09-06 §3.1, the audit's most serious finding. The Phase 2
KG probe ran on the MedQA **test** split (`results/phase2_kg_probe/medqa_n200.json`,
`per_question[*].question_id = "test_*"`; the overnight `medqa_n500.json`
extended the same mistake). Every Phase 2 design decision was then made by
reading test-split outputs — ADR-015's 40-triple cap (motivated by observing
`test_00001`), ADR-016's relevance/novelty correction, ADR-017's category
heuristic (tuned by inspecting `test_00006`), and ADR-018's `kg_rerank`
adopt/keep-off verdict. Phase 7 evaluates on a 500-question seeded subset of
that same test split, and `load_question_sample` draws both from one file with
`random.Random(42).sample`, so the sets overlap (~79 questions in expectation
between the 200 and the 500). Parameters were selected by looking at a quantity
they will later be judged on. ADR-020's claim that "test-set hygiene held since
Phase 1" was false for the KG layer.

**Remediation performed (2026-09-07).**
1. Precomputed the `kg_rerank` cache for a 500-question MedQA **train** slice
   (seed 42) into `data/processed/kg_cache_kg_rerank/medqa/train_*.json` (500
   files; the naive train-500 cache already existed).
2. Re-ran `scripts/kg_probe.py` on train-500 for **both** variants →
   `results/phase2_kg_probe/medqa_train_n500.json` and
   `…_kg_rerank_variant.json` (0 cache misses in either).
3. Re-ran `scripts/kg_rerank_ablation.py` on those two →
   `…_kg_rerank_ablation.json`.
4. Stamped the four test-split probe artifacts `superseded: true` via
   `scripts/mark_superseded.py --preset phase2-kg-probe-test-split` (retained,
   not deleted — they are the provenance for this ADR).
5. Disclosed the original test-split selection in `docs/00_RESEARCH_SPEC.md` §7.

**The measurements barely moved. The selection was still illegitimate.** Both
things are true and both matter:

| metric | TEST n=200 (superseded) | TRAIN n=500 (authoritative) |
|---|---|---|
| `aggregate_hit_rate` | 0.9600 | **0.9540** |
| `entity_level_mapping_rate` | 0.1122 | **0.1158** |
| `naive_expected_hit_rate_under_independence` | 0.9998 | **0.9999** |
| `mean_linked_entities_per_question` | 73.95 | **75.17** |
| `overall_novel_relation_rate` | 0.9016 | **0.9019** |
| `overall_relevant_rate` | 0.4734 | **0.4744** |
| `overall_novel_and_relevant_rate` | 0.4062 | **0.4237** |
| `overall_novel_given_relevant_rate` | 0.8581 | **0.8931** |

That the numbers reproduce is *not* a retroactive excuse for having selected on
test — a parameter chosen by looking at the reporting split is contaminated
whether or not the reading happened to be stable, because the stability is only
knowable after the fact, as here. It does mean the Phase 2 *findings* survive
intact and no downstream work has to be discarded.

**Decision, re-derived per ADR, from the train numbers only.**

- **ADR-015 (40-triple per-question cap) — UPHELD.** The cap's premise is
  saturation, and it reproduces: on train-500, **476/500 (95.2%)** questions hit
  the 40-triple cap (test: 96.0%), median 40, mean 38.1, and 23 questions yield
  zero triples. Triple volume is still bounded by the cap and not by the
  per-node degree cap, which is exactly what ADR-015 argued. The cap value 40
  itself remains an untuned ceiling justified by the downstream `max_evidence=12`
  budget, not by an optimisation on any split — that was true on test and is
  still true; it is not laundered into a "tuned on train" claim here.
- **ADR-016 (hit-rate is an arithmetic artifact; relevance-conditioned novelty)
  — UPHELD, and strengthened.** On train: `entity_level_mapping_rate = 0.1158`
  with 75.17 linked mentions per question gives
  `naive_expected_hit_rate_under_independence = 0.9999`, which again *exceeds*
  the observed `aggregate_hit_rate = 0.9540`. The permissive question-level hit
  definition is still measuring entity fan-out rather than coverage. This is the
  C5 contribution in `docs/00_RESEARCH_SPEC.md` §3, and it is now established on
  a split the project does not report on.
- **ADR-017 (MedQA category heuristic) — UPHELD.** The heuristic was originally
  tuned by inspecting a test question. Applied unchanged to 500 unseen train
  questions it does not degenerate: all six categories populate
  (pharmacology 127, pathology_diagnosis 116, treatment_management 101, other 92,
  physiology 41, anatomy 23), and the `other` catch-all takes **18.4%** of train
  questions vs 20.0% of test — no collapse into `other`, and no category
  starved. `anatomy` (n=23) now clears `MIN_CATEGORY_N_FOR_RATE = 15` and gets a
  reported rate on train, where at n=200 it was suppressed at n=8.
- **ADR-018 (`kg_rerank` adopt / keep-off) — VERDICT FLIPS. The primary metric
  is now a significant win, not a tie.** ADR-018 pre-registered the rule:
  primary metric `novel_and_relevant_rate_for_question`, paired bootstrap
  (10k resamples, seed 42), decide adopt/keep-off from the verdict and never
  from the point estimate. Re-run on train-500:

| metric | naive | kg_rerank | paired mean diff | 95% CI | verdict |
|---|---|---|---|---|---|
| `novel_and_relevant_rate_for_question` (**primary**) | 0.4237 | 0.4604 | **+0.0367** | **[+0.0044, +0.0681]** | **significant win** (was a *tie* at n=200 test) |
| `relevant_rate_for_question` (secondary) | 0.4744 | 0.5296 | +0.0551 | [+0.0203, +0.0889] | significant win (was a marginal win) |

  `n_paired = 477`, `n_excluded = 23` — the 23 exclusions are the zero-triple
  questions, identical in both variants, so this is not a variant-specific
  exclusion bias. ADR-018 predicted this outcome explicitly ("n=200 (192 paired)
  may simply be underpowered for an effect this size") and pre-committed to
  re-deciding on a larger n. That re-test has now run, on a clean split, and the
  tie resolved into a win.

**Multiplicity correction, run now rather than deferred (2026-09-07).** The
primary metric's CI lower bound is **+0.0044** — one-tenth of the point
estimate — and this is 2 comparisons read as one decision, so it is corrected
for multiplicity *before* that decision is recorded, not folded into the §6.7
grid-wide discipline still deferred to the pre-Phase-7 batch (that item covers
the ~9-metric × ~10-variant-pair Phase 7 grid; this is a single, narrow,
targeted correction for the one decision this ADR makes today).

`src.retrieval.metrics.paired_bootstrap_p` (new, AUDIT §6.8) computes a
two-sided percentile-bootstrap p-value from the *same* bootstrap distribution
`paired_bootstrap_ci` already draws (same `diffs`/`n_resamples`/`seed`), so
`p < α` is consistent with that call's own CI excluding zero by construction.
`src.retrieval.metrics.holm_bonferroni` applies the standard step-down
procedure over the two metrics together. Computed from the same train-500
`per_question` arrays (`scripts/kg_rerank_ablation.py`, now emitting `p`,
`holm_rank`, `holm_threshold`, `holm_survives` per metric):

| metric | p (two-sided) | Holm rank | Holm threshold (α=0.05, m=2) | survives |
|---|---|---|---|---|
| `relevant_rate_for_question` (secondary) | 0.0026 | 1 | 0.025 | **yes** |
| `novel_and_relevant_rate_for_question` (**primary**) | 0.0256 | 2 | 0.050 | **yes** |

**Both metrics survive Holm correction for this 2-comparison family.** This is
worth stating plainly because it was not the expected outcome going in — the
primary's CI lower bound sitting close to zero reads as fragile, and the
natural guess is that a stricter corrected threshold eats it. It does not,
for a specific, checkable reason: Holm's step-down procedure only tightens the
threshold for the *smaller* p-value in a family (here, the secondary, tested
against α/2 = 0.025, which its p=0.0026 clears easily); the *largest* p-value
in the family — here, the primary — is tested against the **same threshold as
the uncorrected case, α itself**, because it is the last hypothesis in the
step-down chain. With exactly two comparisons and the primary being the
weaker one, Holm correction changes nothing about the bar the primary has to
clear. It would have mattered with a third, even-weaker comparison in the
family, or if the primary had been the stronger of the two — neither is the
case here. So: **the primary result stands as a significant win, corrected for
this family, not merely uncorrected** — a materially different (better)
epistemic status than "recorded as exploratory" would suggest, and this ADR
is revised accordingly from an earlier draft that reasoned about the
correction rather than computing it.

**Decision on shipping: `kg_rerank` stays `false` by default and is promoted to
a first-class Phase 7 grid row.** The pre-registered rule speaks to the *KG
evidence-quality* verdict, and that verdict is now "win," Holm-corrected — so
the claim "`kg_rerank` improves KG evidence quality" is, from this ADR forward,
permitted and must cite this train-500 CI and its Holm-survival result. But
`novel_and_relevant_rate` is an intermediate proxy, not an outcome: no run yet
shows better triple selection moving any downstream metric, and flipping the
default now would silently change the evidence bundle underneath the committed
Phase 3/4 dev-20 results and make them non-comparable to everything after.
Adoption is therefore decided by the Phase 7 grid cell, on a downstream
metric — see the pre-registration deviation immediately below for why this is
a deviation from ADR-018's own rule, not a restatement of it.

**Pre-registration deviation, named explicitly.** ADR-018 pre-registered a
concrete rule: run the paired-bootstrap check on `novel_and_relevant_rate` at
a larger n, and "decide adopt/keep-off from that verdict, not the point
estimate." The verdict at n=500 — now Holm-corrected — is unambiguously
"win." **The rule said adopt. This ADR does not adopt; it keeps `kg_rerank`
off by default.** That is a deviation from the pre-registered decision
procedure, made *after* seeing the data the procedure was supposed to decide
on, and it is recorded as exactly that rather than presented as if ADR-018 had
always contemplated this outcome.

The reasons for the deviation are stated above (proxy-metric vs. downstream
outcome; comparability of the committed Phase 3/4 dev-20 results) and are
judged sound. But soundness of the reasons is not the point pre-registration
guards against — the point is that a rule abandoned after seeing favorable
data, for reasons that were not part of the rule, is exactly how motivated
post-hoc reasoning enters a project that otherwise holds itself to a strict
"decide before you look" discipline (CLAUDE.md §2, ADR-012's whole point).
Recording it as a plain default-vs-decision distinction, without naming the
deviation, would have let this ADR read as compliant with ADR-018 when it is
not. **The Phase 7 grid's `kg_rerank` row is therefore the thing that actually
settles this** — not as a formality, but because it is the only remaining
check on whether keeping the default off was the right call or a rationalised
one. If that grid cell also shows a downstream win, the deviation cost nothing
but a phase's delay; if it shows no downstream effect, the deviation will have
been the correct caution and ADR-018's original rule the wrong one to have
pre-registered in the first place (it targeted a proxy metric, not an
outcome). Either finding is informative and this ADR does not pre-judge it.

**Consequences.** ADR-015/016/017/018 keep their reasoning and are corrected of
record as to their evidential basis: each now cites the train-500 artifacts, and
their original test-split basis is superseded. ADR-020's "test-set hygiene held
since Phase 1" is false as written and is corrected by this ADR (and by a direct
correction landed in ADR-020 itself). `src.retrieval.metrics` gains
`paired_bootstrap_p` and `holm_bonferroni` (AUDIT §6.8), and
`scripts/kg_rerank_ablation.py` reports Holm-corrected significance by default
for its 2-metric family — any future re-run of this exact ablation inherits the
correction automatically. The Phase 7 grid gains a `kg_rerank` row, which is
now also the open item that resolves this ADR's pre-registration deviation, not
merely an ordinary ablation cell. The four test-split probe JSONs remain in
`results/phase2_kg_probe/` carrying `superseded: true` and must never be cited
as the basis of a decision.

## ADR-028 — Phase 5 evidential calibration: feature definitions, disjoint fit/held-out split, temperature scaling alone matches it, and gamma's KG-invisibility confirmed from a second direction
Date: 2026-09-07 · Phase: 5 · Status: accepted · amended same day after the
temperature-scaled-raw-logprob ablation (Decision 5) showed the original
calibrated-vs-raw-at-T=1 exit-test framing overstated what the result
established — corrected in place, not left standing

**Context.** `docs/01_ARCHITECTURE.md`'s calibration stage (`z_i = alpha*s_pos_i
- beta*s_neg_i + gamma*kg_i + delta*rerank_i`, `p = softmax(z/T)`) leaves the
`kg_i`/`rerank_i` feature definitions, the fit procedure, and the ECE
comparison protocol unspecified. The roadmap's exit test ("ECE on dev improves
vs. raw LLM option log-probs; ranked output includes per-option citations for
every question") also does not say how "raw LLM option log-probs" is obtained,
nor whether the fitting set and the reported-ECE set may be the same
questions. Each is decided here and recorded so a future reader does not
re-derive them differently.

**Decision 1 — feature definitions** (`src/calibration/scoring.py`):
`kg_support_count` = count of an option's SUPPORT cells whose cited evidence
is KG-modality (a signal independent of `s_pos`, which counts SUPPORT cells of
every modality — this is what lets gamma weight KG-backed support
differently from passage-backed support, per the architecture's separate
`kg_i` term). `mean_rerank_score` = mean MedCPT rerank score over an option's
non-neutral (SUPPORT **and** CONTRADICT) cells — both directions, because it
is meant as a retrieval-quality prior on the evidence actually consulted for
that option, independent of which way the audit read it.

**Decision 2 — raw LLM option log-probs is a real logprob readout, not
self-reported confidence.** Verified empirically before building on it:
LiteLLM's native `ollama/` provider path rejects `logprobs` outright
(`UnsupportedParamsError`); Ollama's OpenAI-compatible endpoint
(`/v1/chat/completions`) returns real per-token logprobs. `src.llm.client.chat`
now accepts `logprobs=True`, routing an Ollama call through LiteLLM's
`openai/` provider type at the same server's `/v1` path when logprobs are
requested — still through LiteLLM (CLAUDE.md: never call a provider SDK
directly), just a different LiteLLM provider type. `scripts/raw_logprob_baseline.py`
asks the model for a single-letter answer (so the option letter is the first
generated token) given the **same EvidenceBundle** the V4 agent pipeline saw
for that question — same information access, so the comparison isolates the
calibration mechanism's contribution rather than confounding it with an
evidence-access difference. Raw probability per option = softmax(1.0) over
just that option letter's logprob among the real `top_logprobs` alternatives;
a letter absent from `top_logprobs` gets a sentinel 5 nats below the worst
observed alternative (never dropped, never a crash).

**Decision 3 — disjoint fit/held-out split is mandatory, not deferred**
(overriding the originally-approved plan, per the project owner's correction
before implementation). Fitting a 5-parameter model and evaluating its ECE on
the *same* 20-60 questions is circular — C3 (evidential calibration) is the
contribution most likely to come out positive of the four in
`docs/00_RESEARCH_SPEC.md` §3, so a circular result here is worse than no
result. Ran V4 on a **120-question MedQA train slice** (seed 42), disjoint by
construction from Phase 3/4's dev-20
(`load_question_sample_excluding` — joins by resolved `question_id`, not index
arithmetic, since `random.sample` with a different `k` does not nest a smaller
sample inside a larger one at the same seed). Split by sorted `question_id`
into the first 60 (fit) and second 60 (held-out) — deterministic, no extra
randomness, and independently reproduced identically by
`scripts/raw_logprob_baseline.py` over the same question set so both scripts'
subsets align. `scripts/fit_calibration.py` asserts the two sets are disjoint
before fitting.

**Real numbers** (`results/phase5_calibration/medqa_dev_calibration.json`,
`results/medqa_V4_a714fe0-dirty_20260907T121703Z/`):

| | fit (n=60) | held-out (n=60) |
|---|---|---|
| ECE (5 equal-mass bins) | 0.0829 | **0.0724** |
| rank-1 accuracy (fitted weights, sanity only) | 0.267 | 0.267 |

Fitted weights: `alpha=-0.0139, beta=0.1098, gamma=0.2949, delta=0.0077,
temperature=0.2145`. The fit-vs-held-out ECE gap is **-0.0105** — held-out
ECE is *slightly better* than fit-set ECE, not worse. That is itself
informative: it is not the signature of overfitting (which would show
held-out ECE noticeably worse than fit-set ECE), and it is consistent with
the strong L2 regularization (lambda=0.01) and the bounded, low-dimensional
(5-parameter) fit keeping the model far from the fitting set's idiosyncrasies.
It should not be read as "the model generalizes well" beyond that — n=60 is
still small, and a gap this size is within plausible sampling noise either
way.

**Decision 4 — the ECE exit-test comparison gets a paired-bootstrap CI, not a
point estimate** (per the project owner's explicit correction: every other
A-vs-B claim in this project carries one, per ADR-012's precedent, and ECE at
n=60/5 bins is a single noisy number otherwise). ECE is not decomposable into
independent per-question differences the way `paired_bootstrap_ci` assumes (it
is a population-level, re-binned statistic), so a dedicated
`src.eval.metrics.paired_bootstrap_ece_diff` resamples question indices with
replacement (paired — the same resampled indices for both the calibrated and
raw-logprob sides on each draw) and recomputes the full `ece()` metric from
scratch on each of 10,000 resamples, exactly mirroring ADR-012's decision rule
(CI entirely below 0 → the calibrated side is significantly better; entirely
above 0 → raw is significantly better; otherwise → tie, reported as a tie).
The two sides are joined by `question_id` (`scripts/fit_calibration.py:
paired_calibrated_vs_raw`), never by list position, since the two scripts that
produce them run independently.

**Exit-test result (real, held-out, n=60 paired):**

| | value |
|---|---|
| Calibrated ECE (point) | 0.0724 |
| Raw-logprob ECE (point) | 0.5418 |
| Paired bootstrap mean diff (calibrated − raw) | -0.4204 |
| 95% CI | **[-0.5426, -0.2826]** |
| Verdict | **calibrated significantly better** (CI entirely below 0) |

The literal roadmap exit test passes, by a wide margin. **But this result on
its own is the wrong thing to read as evidence for C3, and an earlier draft of
this ADR read it that way before the ablation below was run — corrected here
rather than left standing.** A raw single-shot LLM confidence readout at
temperature 1 is a strawman-adjacent baseline for a very specific, boring
reason: temperature 1 is simply not the right temperature for this model's
confidence, and *any* miscalibrated-but-informative signal will look bad at
the wrong temperature and look better merely by rescaling it — with no
evidential feature (`s_pos`, `s_neg`, `kg_support_count`, `mean_rerank_score`)
involved at all. The 0.54-vs-0.07 gap above conflates two different possible
explanations — "the evidential features are doing real calibration work" and
"the raw signal just needed its temperature fixed" — and does not distinguish
between them. Decision 6 below runs the ablation that does.

**Decision 5 — a temperature-scaled-raw-logprob ablation is required to
isolate what C3 actually claims** (per the project owner, before Phase 6:
"the comparison right now is calibrated-evidential vs. uncalibrated logprobs,
and a reviewer will say the gain is temperature scaling, not evidential
calibration — because for the most part it might be"). Fit a single scalar
temperature T on the fit-half's *already-saved* logprobs (no new LLM calls):
`log(raw_probability)` per option is a valid stand-in for the true
(unobserved) logit for this purpose — softmax is shift-invariant, and
`log(p_i) = u_i - logsumexp(u)` is `u_i` shifted by a per-question constant,
so `softmax(log(p)/T) == softmax(u/T)` for any T (verified to floating-point
precision, `tests/test_calibration_softmax_rank.py`). `T` is fit by grid
search minimizing mean NLL of the gold option on the fit-60, then applied,
frozen, to the held-out-60.

**A real bug surfaced immediately and is recorded because it changes how this
result should be read by anyone reusing this method.** The grid search's
first run used `t_max=20` (borrowed from the evidential fit's own temperature
bound, which was appropriate for that model's much smaller logit scale but
not for raw log-probabilities). The fit saturated exactly at that boundary,
reporting `T=20` — and using it gave a *materially wrong* answer
(T-scaled-raw ECE 0.095, a real-looking but spurious gap vs. calibrated's
0.072). Widening the search and checking whether the NLL kept improving past
the old bound (it did, for a long way — the fit-set NLL surface flattens
toward `log(4)`, the uniform-prediction NLL, because the raw signal here
carries very little correlation with correctness at the fit-set's 60
questions) found the genuine fit-optimal **T≈92.03**, well inside a widened
range. `fit_temperature_only` now defaults to a much wider search
(`t_max=2000`) and `scripts/fit_calibration.py` detects boundary saturation
and auto-retries at 50× the range, warning loudly if it saturates again —
this exact failure mode (a plausible-looking, wrong optimum from an
insufficiently wide search) is now guarded against structurally, not just
noted here.

**Real ablation result (held-out, n=60, T=92.03 fit on the disjoint fit-60):**

| comparison | point diff | 95% CI | verdict |
|---|---|---|---|
| calibrated ECE (0.0724) vs. T-scaled-raw ECE (0.0725) | −0.0001 | [−0.1344, +0.0887] | **tie** |
| raw-at-T=1 ECE (0.5418) vs. T-scaled-raw ECE (0.0725) | +0.4694 | [+0.2168, +0.5563] | **T-scaling significantly better than raw** |

**This is the honest answer, and it is the outcome the project owner
predicted as the one requiring a restated claim.** Temperature-scaling the
raw logprobs *alone*, with zero evidential features, zero agent reasoning,
and zero new LLM calls, reaches ECE 0.0725 — statistically indistinguishable
from the full evidential pipeline's 0.0724. Essentially all of the 0.54→0.07
improvement documented above is a temperature-scaling effect, not an
evidential-calibration effect. **C3 ("evidential calibration for multi-choice")
is restated: at this sample size (n=60 fit / n=60 held-out), temperature
scaling of the raw signal suffices to reach the same ECE the evidential
features reach; the evidential features' additional contribution over
temperature scaling alone is not established.** This does not mean the
evidential features are worthless — `s_pos`/`s_neg` audit-derived counts are
a *different, and arguably more inspectable/groundable* signal than a raw
next-token confidence, and the features may earn their place at the larger
n the Phase 7 grid provides, or on a different axis (citations, abstention)
this ablation does not test — but the specific claim "evidential features
improve calibration beyond temperature scaling" is not what this run shows,
and must not be reported as if it were. The §4.3 budget asymmetry (12 items
for V4/V5 vs. up to ~52 for B2/B3) still separately forbids any comparison
to B2 (AUDIT 2026-09-06, deferred to the pre-Phase-7 batch) — a wholly
different limitation from this one.

**Decision 6 — report gamma's identifiability, not just its point estimate**
(per the project owner's explicit request, predicted from §5.5 Finding 2: only
1/108 audit cells on the dev-20 cite a KG triple). On this 120-question slice
the pattern reproduces almost exactly: **12 of 626 total cells (1.9%) are
KG-modality SUPPORT cells** — `kg_support_count` is 0 for the overwhelming
majority of (question, option) pairs. `bootstrap_weight_uncertainty` (200
resamples of the 60-question fit set) gives:

| param | mean | std | range |
|---|---|---|---|
| alpha | -0.0169 | 0.0718 | [-0.2777, +0.1934] |
| beta | +0.1139 | 0.0867 | [-0.2836, +0.3304] |
| **gamma** | **+0.2315** | **0.2340** | **[-0.4031, +0.6642]** |
| delta | +0.0071 | 0.0205 | [-0.0438, +0.1334] |

**Gamma's bootstrap std (0.234) is more than 3x alpha's (0.072) despite a
comparable point estimate, and its interval spans from negative to positive
— it is consistent with zero, unlike alpha and delta's tighter, mean-
dominated intervals.** This is exactly the signature of an unidentifiable
parameter: with almost no variance in `kg_support_count` across the fitting
set, the loss surface along the gamma axis is nearly flat, so different
resamples pull the point estimate around widely with no consistent direction.
**This is independent confirmation of the KG-invisibility finding from a
completely different direction than §5.5's raw count** — §5.5 observed that
the audit layer almost never cites KG evidence; this shows that even when a
downstream optimizer is handed the chance to weight that evidence, it cannot
find a stable weight to assign it, because the signal needed to constrain that
weight barely exists in the data. Both readings point at the same underlying
fact (the KG modality is nearly invisible to the agent pipeline as currently
wired), reached by unrelated measurements. Report gamma's bootstrap interval
alongside its point estimate wherever gamma is discussed; the point estimate
alone would misleadingly suggest a real, moderate positive effect.

**Calibration improved confidence quality while accuracy went down — stated
plainly, not as a footnote.** Phase 4's uncalibrated `rank-1-by-support` (raw
`s+ - s-` argmax, no temperature, no gamma/delta) was **0.35** on its own
dev-20 (`docs/04` §5.5 Finding 3). This phase's fitted, temperature-scaled
model gets **0.267** rank-1 accuracy on both the fit-60 and held-out-60 (same
number both places — see the fit-vs-held-out discussion above). These are
different question sets, not a controlled before/after on the same items, and
that caveat is real — but it is not a reason to soften the finding: a fitted
model that is *better calibrated* (ECE 0.07 vs. an implicit uncalibrated ECE
never measured for the 0.35 rule, since raw argmax carries no probability at
all to score) and *less accurate* than the crude uncalibrated rule it replaces
is a genuinely limited result, not a strict improvement, and the paper must
say so exactly that plainly. Reading the two results together: this phase
demonstrates a *well-calibrated* differential can be produced, and separately
that doing so did not preserve (on this evidence) the accuracy of the cruder
rule it's built on top of — two different claims, and only the first is
supported.

**Consequences.** `configs/v5.yaml` records the fitted weights (Phase 7's V5
variant: agents + calibration). The abstention threshold
(`tau=0.5458` at a 10% target risk on the held-out coverage-risk curve) and
the full per-question ranked output (all 120 questions, all options, with
citations — 63.1% of option-rows carry at least one real citation, the rest
legitimately ungrounded per CLAUDE.md's carve-out) are in
`results/phase5_calibration/medqa_dev_calibration.json`. `src.eval.metrics`
gains `paired_bootstrap_ece_diff`; `src.calibration.softmax_rank` gains
`fit_temperature_only` (with the boundary-saturation lesson from Decision 5
in its docstring); `src.llm.client.chat` gains `logprobs`
support; `src.retrieval` gains nothing (this ADR's bootstrap lives in
`src/eval/metrics.py` alongside `ece()`, not `src/retrieval/metrics.py`,
since it is ECE-specific, not a general per-item-diff bootstrap). Phase 7's
V5 grid cell is the natural next place gamma's identifiability gets tested
again, at n=500 — if it remains unidentifiable there too, gamma should be
dropped from the model rather than carried as a free parameter with no
signal to constrain it.

## ADR-029 — Phase 6 harness is minimal (run → score → write); the V-variant map, the uncalibrated-ranking rule, and the dual hallucination denominator
Date: 2026-09-07 · Phase: 6 · Status: accepted

**Context.** The roadmap's Phase 6 spec was "parallel, resumable, cached
harness + concurrency benchmark". `docs/SCHEDULE.md` (hard deadline
2026-09-12) cut that to "minimal: run, score, write" — concurrency
benchmarking is an engineering nicety, not a result. This ADR records what
that minimal harness *is*, plus three modelling choices the spec left open
that a reader would otherwise re-derive differently.

**Decision 1 — scope.** `src/eval/harness.py` is a sequential loop:
`run_eval(config, dataset, split, limit, variant, …)` → per-question predict →
`score_run` → `src.eval.results.write_run`. **Cut:** the `asyncio.Semaphore`
concurrency layer, resumability, and the 1/2/4 questions-per-hour benchmark.
`docs/04` §0.2's planning-estimate "3× concurrency speedup" is **not** replaced
by a measured number — Phase 7 runs sequentially at n=500 and that is
acceptable within the deadline. Resumability's absence means a crashed grid
job re-runs its variant from the top; the LLM cache (`.cache/llm/`) makes that
cheap for the agent calls, and calibration/NLI are deterministic.
`src/eval/ragas_eval.py` is not written — RAGAS is dropped (`docs/SCHEDULE.md`;
NLI entailment against cited evidence covers factual fidelity).

**Decision 2 — the V0–V5 → run mapping lives in one place** (`harness.VARIANTS`),
because the grid names V0–V5 while the code has baselines B0–B3 + V4/V5 and the
match is not 1:1:

| V-id | what it runs | modalities |
|---|---|---|
| V0 | `_run_b0` (zero-shot, no retrieval) | — |
| V1 | `_run_generation` | `{dense}` |
| V2 | `_run_generation` | `{dense, lexical}` — **new, never run before Phase 6** |
| V3 | `_run_generation` | `{dense, lexical, kg}` — **identical to baseline B2** |
| V4 | `run_consensus`, ranked by `(s+ - s-)` | (agent bundle) |
| V5 | `run_consensus` + `rank_options` with `configs/v5.yaml` weights | (agent bundle) |

The committed `results/medqa_B2_*` runs therefore map to **V3, not V2**;
`scripts/report.py` enforces this and V2's cell stays `TBD`. A `pubmedqa_B*`
run maps to no grid cell (PubMedQA smoke runs are not grid results).

**Decision 3 — ranking metrics (P@K / MRR / ECE) by variant.**
- V0–V3 emit one predicted option, not a ranking → P@K / MRR / ECE are
  reported `null` (the §5 grid shows `TBD`). P@1 for a ranked variant equals
  accuracy anyway.
- V4 has an agent SupportMatrix but no calibrated distribution → P@K / MRR are
  computed from the **uncalibrated `(s+ - s-)` support-margin ordering**, tagged
  `ranking_source = "uncalibrated_support_margin"` in `metrics.json` *and* in
  the table, so no reader takes V4's P@2 for a calibrated-model number. V4 ECE
  stays `null` (no probability vector to be calibrated).
- V5 gets all three from the calibrated `p_vector`.
This is why the pre-registration (`docs/04` §1.0) names **V4 vs V5 ranking** as
the comparison that isolates whether calibration improves *ordering* (P@K/MRR)
versus only *confidence* (ECE).

**Decision 4 — hallucination rate is reported under BOTH denominators**
(`src/eval/nli.py`), from one NLI pass:
- `hallucination_rate_predicted_option` — claims backing the answered option.
  **Primary.** It is what a reader taking the top-1 answer is exposed to, and
  it is comparable across variants that do not all produce a full differential.
- `hallucination_rate_all_support_cells` — every SUPPORT assertion across all
  options. The system's real output is a ranked differential with per-option
  citations, so every SUPPORT cell is an assertion on display; scoring only the
  winner undercounts what a reader sees.
A reviewer who prefers the other denominator reads it off the same
`metrics.json` — no re-run. For V0–V3 (one rationale, no per-option cells) the
two are equal by construction. Claim splitting is a small pure regex
(`split_claims`) that guards decimals and abbreviations — unit-tested on
adversarial strings, no LLM (an LLM splitter is another non-deterministic call
and makes the denominator unauditable).

**Decision 5 — `src/eval/results.py` git-diff decode.** `_git` used
`text=True`, which decodes subprocess output with the locale codec (cp1252 on
this Windows box) and raised `UnicodeDecodeError` in a reader thread the moment
a tracked file's diff carried a non-latin-1 byte — and `docs/*.md` are full of
non-ASCII (≈, minus signs, en-dashes). Every run made from a dirty tree with
non-ASCII changes got a corrupted `git_diff_sha256`, the exact provenance field
it records. Fixed to `encoding="utf-8", errors="replace"`. A bug fix, not a
design change.

**Consequences.** New: `src/eval/harness.py`, `src/eval/nli.py`,
`scripts/report.py`, `paper/tables/results_grid.{md,tex}`. Extended:
`src/eval/metrics.py` (`precision_at_k`, `reciprocal_rank_option`, `mean_*`),
`src/eval/baselines.py` (`Prediction.rationale`), `src/pipeline.py`
(`consensus_over_questions` — the per-question agent loop, now shared by
`_run_agents_eval` and the harness, not forked). §5.5's "H4 cannot be measured
— no p-vector is persisted" is resolved: V5 runs persist `p_vector` + `ranked`
per question. Exit test (dev-20 V5): acc 0.30, P@1 0.30, P@2 0.55, MRR 0.57,
ECE 0.080, halluc-rate 0.68 (predicted) / 0.72 (all cells), `abstention_rate`
1.00 (every calibrated p1 sits below `configs/v5.yaml: abstention_tau=0.5458`
— a real property of the near-degenerate Phase 5 fit, ADR-028, recorded not
hidden), `results/medqa_V5_aa73199-dirty_20260907T170604Z/`.

## ADR-030 — Phase 7 multiplicity pre-registered before any grid number; PubMedQA baseline is word-framed, not lettered
Date: 2026-09-07 · Phase: 6 (pre-Phase-7) · Status: accepted

**Context.** Two things had to be settled *before* the Wednesday grid launch,
because settling them afterwards is worthless: (1) AUDIT 2026-09-06 §6.7 — the
grid runs ~5 variants of paired-bootstrap comparisons across ~7 metrics and
nothing in `docs/04` §1 mentioned family-wise error, while ADR-012 already ran
3 uncorrected comparisons and ADR-018 ran 2; (2) the overnight PubMedQA smoke
run showed all four baselines at accuracy 0.50 with a one-class ("yes")
predictor — a degenerate result that would poison any PubMedQA grid cell.

**Decision 1 — pre-registered analysis plan, written into `docs/04` §1.0,
dated 2026-09-07.**
- **Confirmatory: exactly one test — V3 vs V5, accuracy (singular), MedQA
  n=500, alpha = 0.05 two-sided.** Not accuracy *and* Macro-F1 *and* ECE (that
  re-smuggles a family into the confirmatory slot). Two-sided because Phase 4
  §5.5 Finding 3 and Phase 5 ADR-028 both make "V5 does not beat V3" a live
  outcome — the direction may not be assumed.
- **A null result is the finding.** Written before the number is seen: if
  V3 = V5 on accuracy, the paper reports that the full stack does not beat
  tri-modal retrieval alone on MedQA at this scale, with the mechanism
  (fusion tie ADR-012, conflict never fires ADR-025, temperature scaling
  matches the features ADR-028). C1/C3 are already framed as diagnostic in
  `docs/00` §3, so this is consistent, not a salvage job.
- **Everything else is exploratory**, Holm–Bonferroni across a pre-registered
  family of 30 {variant-pair × metric} cells (enumerated in §1.0). Any test
  added after seeing grid numbers is reported as unplanned and excluded from
  the Holm family.

**Decision 2 — PubMedQA baselines use word-framed prompts** (`src/eval/
baselines.py: _PUBMEDQA_*_PROMPT`, `_PUBMEDQA_SYNONYMS`). Lettering yes/no/maybe
as A/B/C hid the task. Three prompt versions were run on the dev-20:

| Prompt | B0 pred dist | B0 acc | B3 `verifier_decision_rate` | B3 `n_supported` |
|---|---|---|---|---|
| v1: `A. yes / B. no / C. maybe` | `{A:14, B:5, C:1}` | 0.50 | 0.00 | `{0:12, 2:6, 3:2}` |
| v2: words + "use maybe when mixed/insufficient" | `{C:16, A:4}` | 0.25 | 0.15 | `{0:9, 1:3, 2:7, 3:1}` |
| v3 (kept): words, "give your best single answer" | `{C:14, A:6}` | 0.25 | 0.15 | `{0:9, 1:3, 2:7, 3:1}` |

**None of the three is non-degenerate** — the local 7B model never predicts
"no" under word framing and predicts one class 70–90% of the time under every
framing. This is a **model-capability ceiling, not a prompt bug**: it is
disclosed as a limitation, and PubMedQA stays out of the Phase 7 grid
(`docs/SCHEDULE.md` already cut it to "MedQA only"). v3 is kept because it is
the faithful statement of the task and it does not artificially cripple B3 —
under v3 the binary verifier still isolates a single option on only **3/20**
three-way questions (`verifier_decision_rate = 0.15`), which is the honest form
of the binary-bottleneck evidence for a 3-way task, reported in `docs/04` §5.4.
All v1/v2 PubMedQA runs are stamped `superseded` (by run id) pointing at the v3
run.

**Decision 3 — PubMedQA `contamination_assertion` runs for real**
(`src.eval.contamination.pubmedqa_contamination_report`, wired into both
`scripts/run_baselines.py` and `src/eval/harness.py`). It previously returned
`{"checked": False, "applies": True}` — knowing the check applied and skipping
it, the AUDIT §3.2 failure class. Now it calls `assert_no_contamination` with
`blocklisted_pmids("pubmedqa")` (the 500 `pqa_labeled` TEST PMIDs) against the
built corpus; `checked: True` is earned because the corpus was scanned. The
overnight and re-run PubMedQA runs record `passed: True`.

**Consequences.** `docs/04` §1.0 (pre-registration), §5.4 (word-framed B3
number + the lettered-prompt 0.00 as prompt-sensitivity), §5.5 Finding 3
(equal-budget B2 = 0.60 vs V4 = 0.35, now unconfounded). `src/eval/
contamination.py` gains `pubmedqa_contamination_report`. The multiplicity
family is recomputed and re-registered if the grid is expanded back toward
`docs/04` §5's full schema.

## ADR-031 — `hallucination_rate` at NLI threshold 0.5 is metric-strict, not an agent-layer signal; report it as a relative comparison only
Date: 2026-09-08 · Phase: 6 · Status: accepted

**Context.** The Phase 6 harness exit test (V5, dev-20) reported
`hallucination_rate_predicted_option = 0.682`
(`results/medqa_V5_aa73199-dirty_20260907T170604Z/`). Read naively that says
"the agent layer produces citations that support its claims only ~32% of the
time" — a strong agent-layer claim. But `docs/04` §3's criterion is DeBERTa-v3
NLI entailment **≥ 0.5** of a claim by its cited evidence, and that threshold
is strict: a clinical claim that is genuinely supported *after one inferential
step* scores below it. The number could be measuring the metric, not the
pipeline.

**The control** (`scripts/analysis/hallucination_control.py`,
`results/phase6_hallucination_control/control.json`): the **identical** scorer
(`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`), threshold (0.5)
and claim splitter (`src.eval.nli.split_claims`) run over the equal-budget B2
run's rationales + citations on the **same dev-20** (B2 re-run with
`Prediction.rationale` recorded:
`results/medqa_B2eq_79ee599-dirty_20260907T210514Z/`, acc/macro-F1 reproduced
exactly from the superseded `…aa73199…` run):

| pipeline (dev-20) | halluc-rate (predicted option) | n claims | uncited |
|---|---|---|---|
| B2 equal-budget (hybrid RAG, one LLM call) | **0.741** | 58 | 0 |
| V5 (agents + calibration) | **0.682** | 22 | 0 |

**Decision — the 0.682 is a metric-strictness artifact, reported as such.**
Both pipelines land near 0.7, **both with zero uncited claims**, so the number
is not about citation *presence* — it is DeBERTa NLI declining to score
cited-but-inferential clinical claims as entailed at ≥ 0.5. V5 is if anything
marginally *lower* than plain hybrid RAG (−0.060; n=20 single-seed, no
significance). Consequences:

1. `hallucination_rate` is used in Phase 7 as a **relative** comparison across
   variants at a fixed threshold (its role in the `§1.0` exploratory family),
   **never** as an absolute "X% of claims are hallucinated" statement in the
   paper — that framing measures the threshold.
2. The threshold (0.5) and the splitter (`split_claims`, regex, decimals /
   abbreviations guarded) are stated wherever the number appears (`docs/04`
   §5.7, §3) so it is auditable.
3. `invalid_citation_count` (a cited id absent from the bundle — a strictly
   worse, unambiguous failure) is reported alongside `hallucination_rate` for
   every variant; on the dev-20 both V5 and B2-equal-budget have
   `invalid_citation_count` ∈ {0, 1}.
4. A sensitivity sweep of the entailment threshold (0.3 / 0.5 / 0.7) on the
   Phase 7 V3/V5 runs is added to the exploratory reporting, so the paper can
   show the *ordering* between variants is threshold-stable even though the
   absolute rate is not. Not gating; recorded here so it is not dropped.

**What this does NOT license.** It is not evidence that V5's rationales *are*
well-grounded — only that they are no worse-grounded, under this metric, than a
one-call hybrid RAG baseline. The honest statement is "at this entailment
threshold the metric does not discriminate the two pipelines", not "V5 is
faithful".

**Consequences.** New: `scripts/analysis/hallucination_control.py`,
`results/phase6_hallucination_control/control.json`,
`results/medqa_B2eq_79ee599-dirty_20260907T210514Z/` (B2 re-run with rationale;
the earlier `…aa73199…` B2eq run is stamped `superseded`). `docs/04` §5.7
records the control; §3 and `§1.0` gain the "relative only" framing.

## ADR-032 — BM25 `k1`/`b` retuned on NFCorpus; defaults kept as a checked non-change
Date: 2026-09-08 · Phase: 6 (pre-Phase-7) · Status: accepted

**Context.** `configs/base.yaml` shipped the `rank_bm25` library defaults
(k1=1.5, b=0.75) with tuning parked in Phase 1. The roadmap's pre-Phase-7 list
requires: retune on a fixed dev slice, report Recall@10 / MRR before/after,
record the chosen values in an ADR and the config. NFCorpus (BEIR) is the
project's retrieval-quality dev slice (ADR-012) — 323 test queries with
human-judged graded qrels, no single-modality ceiling.

**Method** (`scripts/analysis/bm25_tune.py`, CPU-only, artifact
`results/phase1_bm25_tune/metrics.json`): rebuild `BM25Okapi` in memory for
each of an 8×6 grid — k1 ∈ {0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5}, b ∈
{0.3, 0.5, 0.65, 0.75, 0.9, 1.0} — over `data/processed/nfcorpus_corpus.jsonl`
(6,241 chunks), retrieve top-50 per query, dedupe to document rankings, score
mean nDCG@10 / Recall@10 / MRR against the qrels. Primary metric nDCG@10
(ADR-012). Best cell judged by a paired bootstrap of per-query nDCG@10 vs the
current default (10k resamples, seed 42), the project's standard significance
rule.

**Numbers.**

| config | nDCG@10 | Recall@10 | MRR |
|---|---|---|---|
| default k1=1.5, b=0.75 | 0.2929 | 0.6780 | 0.4923 |
| best cell k1=2.5, b=0.3 | 0.2993 | 0.6842 | 0.5063 |

Δ nDCG@10 (best − default) = **+0.0064**; paired-bootstrap mean +0.0064, 95%
CI **[+0.0008, +0.0126]**, p = 0.025 — the CI excludes zero, so the gain is
statistically detectable.

**Decision — keep the defaults (k1=1.5, b=0.75).** Reasons, all recorded:
1. **Below the practical floor.** The adoption bar was pre-set at Δ nDCG@10 ≥
   0.01; +0.0064 is under it.
2. **Selection-on-the-same-sample.** "Best" is the max over 48 cells chosen on
   the identical 323-query NFCorpus sample it is then tested on — the CI is
   optimistic and the true out-of-sample gain is smaller.
3. **BM25 is near-flat here.** Every one of the 48 cells lands at 0.29 ± 0.005
   nDCG@10. There is no meaningful ridge to move onto; low b (0.3) and high k1
   help by a hair, high b (0.9–1.0) hurts by a hair, and that is the whole
   dynamic range.
4. **NFCorpus is a domain proxy.** The real retrieval corpus is PubMed
   abstracts + 18 USMLE textbooks (113k chunks), not nutrition literature.
   Destabilising the retrieval config on a 0.006 proxy-metric gain immediately
   before the Phase 7 grid is not worth it.

This is not "the default was never checked" — it is "the default was checked
against a 48-cell grid and a significance test and is within noise of the grid
optimum". `configs/base.yaml` carries a comment to that effect so a future
reader does not re-open it.

**Consequences.** New: `scripts/analysis/bm25_tune.py`,
`results/phase1_bm25_tune/metrics.json`. `configs/base.yaml` unchanged in value,
annotated. If Phase 7 retrieval diagnostics on the real corpus show BM25
underperforming, this grid is re-run on the real corpus (slower — 113k-chunk
`BM25Okapi` rebuild per cell) rather than trusting the NFCorpus proxy; recorded
here so that path is known.

## ADR-033 — B3's `verifier_decision_rate` is prompt-sensitive; the "1/20 = 0.05" figure is retired, the claim is restated with its 0.05–0.35 range
Date: 2026-09-08 · Phase: 6 (pre-Phase-7) · Status: accepted

**Context.** `docs/04_EVALUATION.md` §5.4 reported the binary verifier B3's
`verifier_decision_rate = 0.05` on the MedQA dev-20 (isolates exactly one
SUPPORTED option on 1/20 questions), and framed it — after the context-window
check — as robust "direct empirical evidence for the binary bottleneck C2
targets". AUDIT 2026-09-06 §4.1 flagged that the number rests on a single
verifier phrasing while B2 gets a neutral instruction, leaving it open to a
"strawman by prompt construction" objection. The roadmap's *B3 verifier-collapse
robustness check* required running B3 under further phrasings.

**Method** (`scripts/analysis/b3_prompt_sensitivity.py`, artifact
`results/phase3_b3_prompt_sensitivity/metrics.json`). The B3 verify
loop (ask SUPPORTED / NOT_SUPPORTED per option, decide iff exactly one is
SUPPORTED) re-run on the same 20 MedQA dev questions, same evidence bundles,
temperature 0, seed 42, with three verify phrasings:

| phrasing | `verifier_decision_rate` | `n_supported` distribution |
|---|---|---|
| strict — "SUPPORTED only if the evidence positively supports it" (shipped) | **0.05** | `{0: 15, 1: 1, 2: 4}` |
| neutral — "is option X supported as the correct answer?" | **0.30** | `{0: 8, 1: 6, 2: 3, 3: 1, 4: 2}` |
| permissive — "could option X be correct? SUPPORTED if consistent with it" | **0.35** | `{0: 6, 1: 7, 2: 4, 3: 1, 4: 2}` |

Spread **0.30**. The strict instruction makes the verifier reject nearly
everything (15/20 questions have zero SUPPORTED); neutral and permissive
phrasings triple the decision rate.

**Decision.**
1. **The "1/20 = 0.05" headline figure is retired.** It is substantially an
   artifact of the strict verify instruction, not an intrinsic property of
   binary verification. §5.4 now reports all three rates and states the
   number as a **0.05–0.35 range**.
2. **The robustness claim is split by axis, honestly:** robust to the context
   window (0.05 at both `num_ctx` 4096 and 8192), **not** robust to verifier
   phrasing. §5.4 says both.
3. **What survives, and is what C2 may lean on:** under *every* phrasing tested
   the binary verifier isolates a single option on a minority of questions
   (best case 7/20 = 0.35) and returns multiple- or zero-SUPPORTED on the
   rest. A per-option yes/no head cannot rank mutually-exclusive options; no
   phrasing tested gets it above 0.35. That is a real limitation — but the
   defensible statement is "isolates a single option on ≤ 35% of questions",
   not "1/20".
4. **C2 (`docs/00` §3) is edited** to quote the range and to lean the
   audit-discrimination claim on the 108 non-neutral audit cells and the
   Phase 7 V4−audit leave-one-out cell, which do not depend on B3's prompt.
   The B3 comparison is demoted to a motivating illustration.
5. **The shipped `_VERIFY_PROMPT` is NOT changed.** Reverting to a more
   permissive phrasing to inflate the bottleneck contrast would be the same
   prompt-selection error in the other direction (cf. ADR-030's PubMedQA
   decision — keep the faithful prompt, report the number it gives). The
   strict phrasing is a defensible verifier instruction; it is simply not the
   only one, and the paper says so.
6. **Phase 7** re-measures `verifier_decision_rate` at n=500 for whichever
   phrasing the B3 row uses, and reports it with this dev-20 range as context.

**Consequences.** New: `scripts/analysis/b3_prompt_sensitivity.py`,
`results/phase3_b3_prompt_sensitivity/metrics.json`. Edited: `docs/04` §5.4
(MedQA B3 diagnostic rewritten; PubMedQA B3 para notes the strict-prompt floor),
`docs/00` §3 C2, `docs/02_ROADMAP.md` (the *B3 verifier-collapse robustness
check* item closed with this outcome). This weakens a talking point, not the
project: C2's real evidence is the audit layer's graded output and the Phase 7
leave-one-agent-out cells, and those are unaffected.

## ADR-034 — KG cache is bound to the question text, not just its positional id (AUDIT §6.5); the caches carry no current drift
Date: 2026-09-08 · Phase: 6 (pre-Phase-7) · Status: accepted

**Context.** `src/kg/dataset_questions.py` derives `question_id =
f"{split}_{idx:05d}"` — a positional index into the raw HF dataset file — and
the KG cache (`data/processed/kg_cache/<dataset>/<question_id>.json`) is keyed
on it. `data/MANIFEST.json` hashes the raw dataset file *contents*, not the
id↔question binding. So any re-fetch that reordered rows (a new HF revision, a
different download path) would silently re-associate every cached subgraph with
the wrong question, and nothing would catch it. AUDIT 2026-09-06 §6.5, MAJOR.

**Fix.**
1. `KGCacheEntry` gains `question_stem: str = ""` (`src/kg/traversal.py`).
   `scripts/precompute_kg_cache.py` writes `question_stem=question.question`
   into every new entry.
2. `get_subgraph_cached(dataset, question_id, cache_dir, *, expected_stem=None)`
   — when `expected_stem` is given and the file carries a non-empty
   `question_stem`, a whitespace/case-insensitive mismatch raises
   `KGCacheStemMismatch` naming the drift and the `--force` re-precompute
   command. A file with an empty stem (predates the field) is read without the
   check — graceful degradation, not a hard failure.
3. `TriModalPipeline.build_bundle` threads the query text through
   `_triples(query=...)` → `get_subgraph_cached(expected_stem=query)`, so every
   eval-time KG read (Phase 3+) is now checked.

**Backfill** (`scripts/kg_cache_backfill_stems.py`, CPU-only, no Neo4j).
Walks all four cache trees; for each file, looks up the question now at that
`question_id` in the current fetch and — **only if an integrity check passes** —
writes the stem in. Integrity check: ≥ 50% of the file's distinct linked-entity
surface strings (captured from the *original* question at build time) must
appear verbatim in the current question's link-text (stem + options, the exact
text `precompute_kg_cache` linked against). A file that fails is left untouched
and reported — that failure *is* the §6.5 condition.

**Result: 3633 / 3633 files backfilled, 0 integrity failures.** The current KG
caches carry **no drift** — every cached subgraph is bound to the question it
was built for. The fix is protective for Phase 7 (and for any future re-fetch),
not a repair of existing damage.

**Manifest.** `data/MANIFEST.json`'s `kg_cache` block was **already stale**
before this change — it recorded `medqa: 220 files` while disk held 1123 (the
Phase 3–5 dev-20 / calibration-120 / Phase 7 test-and-train-500 slices were
added without a manifest refresh), and `kg_cache_kg_rerank/medqa: 200` vs 1010
on disk (the ADR-027 train-500 kg_rerank re-test). `kg_cache_manifest --write`
now records the real counts and fresh checksums, and `--check` passes. The
`note`/`regen` fields are updated to acknowledge the growth, but the **exact
per-phase precompute provenance of the post-Phase-2 additions is not fully
reconstructed here** — that needs whoever ran those precomputes to confirm the
split/limit/seed, ideally with Neo4j up for a content re-verification. Flagged
as an open item; it does not block Phase 7 (the checksums are consistent and
the stem-binding check now guards correctness).

**Consequences.** New: `scripts/kg_cache_backfill_stems.py`,
`src.kg.traversal.KGCacheStemMismatch` / `_norm_stem` / `KGCacheEntry.
question_stem`. Edited: `src/pipeline.py` (`_triples` signature — `query`
kwarg), `scripts/precompute_kg_cache.py`, `scripts/kg_cache_manifest.py`
(notes + counts), `data/MANIFEST.json`, `tests/test_kg_traversal.py` (3 new
tests). The committed `medqa_V5_aa73199-dirty_20260907T170604Z` run is
unaffected — the KG subgraph content is byte-identical, only the `question_stem`
field was added to the cache files.
