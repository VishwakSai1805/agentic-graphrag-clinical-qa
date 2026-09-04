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
