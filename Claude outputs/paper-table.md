---
description: Regenerate paper tables and figures from results/ — zero hand-typed numbers
argument-hint: [table-name]
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

Regenerate paper artifacts from `results/` only.

1. Run `python -m scripts.report --out paper/tables` (create/extend the script if
   the requested artifact isn't covered).
2. Artifacts: `main_results.tex`, `ablation.tex`, `calibration.tex`,
   `failure_modes.tex`, plus figures `reliability.pdf`, `coverage_risk.pdf`,
   `ablation_bars.pdf`.
3. Cross-check every generated cell against the source `metrics.json`. Report any
   number in `paper/` or `docs/` that is **not** traceable to a results file —
   those are contamination of the paper and must be removed.
4. If a cell is still `TBD`, list the exact `/run-eval` command that would fill it.
5. Do not adjust rounding, significance markers, or wording to make a result look
   stronger. Report what the files say.
