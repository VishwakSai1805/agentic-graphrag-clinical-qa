---
description: Run the full ablation grid from Phase 7 and assemble the comparison table
argument-hint: [dataset] [--dry-run]
allowed-tools: Read, Write, Edit, Bash, Glob, Grep, Task
---

Run the ablation grid in `docs/02_ROADMAP.md` Phase 7 for dataset **$1**
(default: medqa).

1. Enumerate every cell: V0–V5 plus leave-one-agent-out (V4−audit, V4−conflict),
   × 3 seeds wherever sampling is involved. Print the plan and the estimated
   total cost/time. If `--dry-run` was passed, stop here.
2. Check `results/` for cells already completed at the current git sha and skip
   them — never re-run a finished cell.
3. Execute sequentially as background scripts, one cell at a time, so a failure
   doesn't take down the grid. Log each `run_id`.
4. Aggregate: mean ± std per cell, paired bootstrap (10k) of V5 vs. V3 and
   V4 vs. V3, with p-values.
5. Write the table into `docs/04_EVALUATION.md` §5 and regenerate
   `paper/tables/ablation.tex` via `scripts/report.py`.
6. Write a short honest reading of the grid: which component actually earns its
   place, which does not, and what the leave-one-out results imply about the
   contribution claims in `docs/00_RESEARCH_SPEC.md` §3. If a component does not
   help, say so — that is a finding, not a failure.
