---
description: Run a scored evaluation and write a reproducible result set
argument-hint: <dataset: pubmedqa|medqa|medmcqa> <variant: V0..V5> [limit]
allowed-tools: Read, Write, Edit, Bash, Glob, Grep, Task
---

Run evaluation: dataset **$1**, variant **$2**, limit **$3** (default: full split).

Protocol — follow `docs/04_EVALUATION.md` exactly:

1. Resolve the config for `$2` from `configs/`. If it does not exist, create it
   from `configs/base.yaml` and show me the diff before running.
2. Confirm the contamination assertion passes; abort the run if it does not.
3. Launch as a background script with logging, not inline:
   `nohup python -m src.eval.harness --config <cfg> --dataset $1 [--limit $3] > logs/<run_id>.log 2>&1 &`
   then poll the log rather than blocking.
4. On completion, verify `results/<run_id>/` contains `config.yaml`,
   `predictions.jsonl`, `metrics.json`.
5. Report: Accuracy, Macro-F1, RAGAS faithfulness, hallucination rate,
   invalid-citation count, P@1, P@2, MRR, ECE, plus cost/latency and the
   failure-mode counters.
6. Update the `docs/04_EVALUATION.md` §5 row for this cell — **copy numbers from
   `metrics.json`, never retype from memory**. Include the `run_id`.
7. Flag anything suspicious: accuracy above published SOTA, near-zero
   hallucination rate, ECE near zero, or a variant beating the oracle. Those mean
   a bug or leakage, and you should say so instead of celebrating.
