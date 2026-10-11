# Agentic Multi-Evidence GraphRAG for Clinical Question Answering

Tri-modal retrieval, multi-agent evidence consensus, and evidential calibration — implemented in full, ablated component by component, and evaluated on MedQA-USMLE under a pre-registered protocol.

**This repository reports a negative result.** The assembled system does not outperform the generator it is built on. The study locates where the loss occurs and why.

---

## Result

MedQA-USMLE, 500 held-out questions, seed 42, single launch commit. Intervals are marginal bootstrap intervals (10,000 resamples, seed 42).

| Variant | Accuracy | P@2 | MRR | Grounded | 95% CI |
|---|---|---|---|---|---|
| V0 zero-shot | **0.582** | — | — | 0.000 | [0.538, 0.626] |
| V1 + dense | 0.566 | — | — | 1.000 | [0.522, 0.608] |
| V2 + lexical | 0.556 | — | — | 0.998 | [0.512, 0.598] |
| V3 + knowledge graph | 0.546 | — | — | 0.984 | [0.502, 0.590] |
| B3 binary verifier | 0.538 | — | — | 0.972 | [0.494, 0.582] |
| V4−conflict | 0.358 | 0.606 | 0.598 | 0.856 | [0.316, 0.400] |
| V5 full system | 0.316 | 0.582 | 0.571 | 0.756 | [0.276, 0.356] |
| V5-refit | 0.292 | 0.530 | 0.551 | 0.652 | [0.252, 0.332] |
| *chance* | *0.250* | *0.500* | *0.521* | — | — |

**The eight configurations fall into two tiers.** Within the retrieval tier no difference survives multiplicity correction — not zero-shot against dense retrieval (+0.016, CI [−0.008, +0.042]), and not the isolation of the knowledge-graph arm (+0.010, CI [−0.010, +0.030]). Every comparison that crosses the tier boundary is significant, at 0.19 to 0.27 accuracy. The pre-registered confirmatory test, V3 against V5, gives +0.230 in favour of V3 (CI [+0.172, +0.288], p < 0.0001) — the opposite direction to the registered prediction.

### Why it fails

The fitted calibration assigns a **negative weight to positive evidence** (α = −0.0139). Because contradiction already enters the logit through an explicit minus sign, supporting evidence is subtracted too, and the scoring rule becomes *choose the least contradicted option* rather than *choose the best supported one*.

- The top-ranked option lies in the least-contradicted set on **493 of 500** questions.
- **170 of 500** questions end with all four logits exactly equal; the ranking is decided by option order. The tie-break chose A on 120 of them, against gold answers distributed A 38 / B 47 / C 48 / D 37.
- On **68** questions the selected option had zero supporting evidence while another option had some.

Refitting with α constrained non-negative, a neutral default for the unscored case, and a seeded tie-break changes **207 of 500** predictions and does not improve accuracy. The failure is deeper than the sign: the evidential features are close to uninformative on this task.

---

## What is in the system

**Stage 1 — tri-modal retrieval.** MedCPT dense embeddings over a FAISS IVF-Flat index (768-d, inner product, nlist=100, nprobe=16); BM25 over the identical chunk set; PrimeKG entity linking and 1–2 hop traversal via Neo4j. Fused by reciprocal rank fusion (k=60), reranked by a cross-encoder, and cut to a 12-item evidence budget **partitioned into per-modality reserved slots** so that removing an arm actually removes its contribution.

**Stage 2 — multi-agent evidence consensus.** A bounded LangGraph state machine over a typed state object: a Hypothesis agent decomposing each option into atomic claims, an Audit agent emitting graded {SUPPORT, CONTRADICT, NEUTRAL} verdicts with a required citation identifier, and a Conflict agent adjudicating surviving contradictions. Citation validity is enforced at parse time — a verdict citing an identifier absent from that question's retrieved set is discarded. Budgets: 2 rounds, 12 generator calls, 12 evidence items per question.

**Stage 3 — evidential calibration.** A per-option logit `z = α·s⁺ − β·s⁻ + γ·kg + δ·rerank`, turned into a distribution by a temperature-scaled softmax, producing a ranked differential with an abstention gate.

**Generator.** Qwen2.5-7B-Instruct, served locally through Ollama, weights frozen, greedy decoding, 512-token cap. **No fine-tuning of any kind.** The only fitted parameters in the entire system are the five calibration scalars, fit on a disjoint train split.

**Corpus.** 113,465 chunks: 80,782 from all 18 USMLE textbooks distributed with MedQA, and 32,683 from a seed-42 20,000-document subsample of PubMedQA's unlabelled split. Chunks of 256 tokens with 32 overlap. Built once and never rebuilt, so every variant retrieved against a byte-identical index.

**Hardware.** One consumer laptop: 12 GB mobile GPU, 16 GB system RAM. No paid inference of any kind at any point (ADR-004).

---

## Evaluation protocol

The analysis plan was committed **before any variant had been run against the test split**. It fixed one confirmatory hypothesis and registered 35 further tests, of which 31 were executed as planned.

Tests fall into three families, each corrected separately by Holm–Bonferroni on its own size:

| Family | m | Specified |
|---|---|---|
| Planned | 31 | Before any test-set number existed |
| Unplanned | 18 | After the planned family was executed |
| Post-hoc | 7 | Motivated by an observed result |

They are corrected separately rather than pooled so that a test motivated by the data cannot draw on the multiplicity budget of a family it was never part of. All comparisons are paired — every variant answers the same 500 questions — and estimated by bootstrap resampling of questions, 10,000 resamples, seed 42, two-sided, α = 0.05.

Every test is reported, including the ones that found nothing.

---

## Reproducing the results

```bash
git clone https://github.com/VishwakSai1805/agentic-graphrag-clinical-qa.git
cd agentic-graphrag-clinical-qa
uv sync                       # or: pip install -e .

# Regenerate Table 1 from the committed per-question predictions
python -m scripts.analysis.marginal_cis
```

That script loads each variant's own `results/<run_id>/predictions.jsonl`, recomputes every point estimate, **asserts it against that run's published `metrics.json` before writing anything**, and emits `results/phase7_marginal_cis.json`. It has been verified to run end to end from a fresh clone.

Reproducing the pipeline itself additionally requires a local Ollama instance with `qwen2.5:7b-instruct`, a Neo4j instance loaded with PrimeKG, and the FAISS index built by `scripts/build_corpus.py`. Index and corpus artifacts are excluded from version control by size; the build script and its manifest are committed.

### Traceability

Every number in the paper is generated from a committed artifact by a script, never transcribed. Each run records the commit it was launched from, the resolved configuration after defaults and overrides, the model identifier actually served, per-question wall-clock, and a hash of the question-id list. Failed runs are retained rather than deleted.

---

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Retrieval, agent graph, calibration, metrics |
| `scripts/` | Corpus build, run launchers, analysis scripts |
| `scripts/analysis/` | One script per reported family of numbers |
| `results/` | Per-run `metrics.json`, configs, and the eight Phase 7 `predictions.jsonl` |
| `logs/` | The run logs cited by the ADRs |
| `docs/` | Specification, architecture, datasets, evaluation protocol, decision log |
| `paper/` | Manuscript sections, tables, figure |
| `tests/` | Unit tests (554 passing) |

### Documentation

| Document | Contents |
|---|---|
| [`docs/00_RESEARCH_SPEC.md`](docs/00_RESEARCH_SPEC.md) | Research question, claims C1–C4, and their status |
| [`docs/01_ARCHITECTURE.md`](docs/01_ARCHITECTURE.md) | Component contracts and agent interfaces |
| [`docs/02_ROADMAP.md`](docs/02_ROADMAP.md) | Phases 0–8 |
| [`docs/03_DATASETS.md`](docs/03_DATASETS.md) | Corpora and benchmarks, with exact counts |
| [`docs/04_EVALUATION.md`](docs/04_EVALUATION.md) | Pre-registration, the three test families, and all results |
| [`docs/05_DECISIONS.md`](docs/05_DECISIONS.md) | 51 architecture decision records |

The decision log is the most useful document in the repository for understanding why the project looks the way it does. It records the failures as well as the choices — including three reproducibility failure modes in which an evaluation returned confident, well-formed, entirely invalid results, and one planned robustness check that was blocked by a provider quota rather than abandoned.

---

## Reproducibility findings

Three failure modes encountered during this work are reported in the paper as contributions in their own right, because each produced output that passed every check in place at the time.

1. **Silent cache replay across generators.** Five configurations re-run against a different model exited successfully and wrote well-formed results naming that model, while every request was served from the original model's cache. Exposed only by wall-clock implausibility — one configuration "answered" 200 questions in six seconds.
2. **A liveness gate that passed from cache** while the endpoint it was probing returned HTTP 503. Caching belongs to the workload, never to the check that the workload can run.
3. **Truncation that does not surface as a parse failure.** With the answer letter preceding the citation list, a completion severed at the token cap parses cleanly with zero citations. Audited directly: 24 of 8,237 completions sat at the cap, with nothing between 490 and 510 tokens.

---

## Scope and limitations

Results are from MedQA-USMLE only; MedMCQA and PubMedQA are registered in the protocol and were not run. At n = 500 the paired intervals have a half-width of roughly 0.025, so differences below about 0.03 are not resolvable — the null results bound the size of any effect rather than establishing its absence. Every result uses the same 7B generator; the stronger-model sensitivity check was specified, implemented, and blocked by a free-tier quota of 20 requests per day against roughly 2,650 required (ADR-050). The launcher remains committed so that anyone with API access can run it.

This is a research artifact. It is **not** a clinical decision support system and must not be used for patient care. All data used is public and de-identified.

---

## Citation

```bibtex
@misc{graphrag_clinical_qa_2026,
  title  = {Agentic Multi-Evidence GraphRAG for Clinical Question Answering:
            Tri-Modal Retrieval with Multi-Agent Evidence Consensus and
            Evidential Calibration},
  author = {Gudivada, Vishwak Sai and Panda, Aalap Kishore and
            Kankanala, Havish Kumar},
  year   = {2026},
  note   = {Vellore Institute of Technology},
  url    = {https://github.com/VishwakSai1805/agentic-graphrag-clinical-qa}
}
```

## Development note

This project was developed with AI assistance (Claude, Anthropic) for experimental design discussion, code review and implementation, and manuscript drafting and editing. Development was AI-assisted; see ADR-052. All experiments were designed, executed and verified by the authors, who take full responsibility for the content of this repository and the accompanying paper.

## License

MIT. See [`LICENSE`](LICENSE).
