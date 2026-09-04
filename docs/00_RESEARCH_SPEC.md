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

**C1 — Tri-modal evidence.** Joint dense + lexical + knowledge-graph retrieval,
with KG triples treated as first-class evidence objects rather than context text.

**C2 — Multi-agent evidence consensus.** A bounded agent network that audits
retrieved triples against unstructured clinical text and explicitly resolves
contradictions, replacing the single binary verifier.

**C3 — Evidential calibration for multi-choice.** Per-option evidence scoring →
temperature-scaled softmax → ranked differential with per-option citation sets
and a calibration (ECE) guarantee.

**C4 — Empirical study.** Baseline parity on PubMedQA (binary), improvement on
MedQA-USMLE and MedMCQA (multi-choice), with an ablation grid isolating the
contribution of each modality and each agent.

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

- Retrieval corpus overlap with benchmark test items → `src/eval/contamination.py`.
- LLM version drift → pin model ids and record them in every `config.yaml`.
- PrimeKG coverage gaps for exam-style questions → report KG-hit rate per dataset.
- Agent cost/latency → report tokens and wall-clock per question alongside accuracy.
