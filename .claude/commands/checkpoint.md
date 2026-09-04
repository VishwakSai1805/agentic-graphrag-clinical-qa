---
description: Close out a work session — sync docs, log decisions, commit, hand off
allowed-tools: Read, Edit, Write, Bash, Glob, Grep
---

Close out this session cleanly so the next one can start with `/clear`.

1. `git status` and `git diff --stat` — summarise what actually changed.
2. Update `docs/02_ROADMAP.md` phase statuses to match reality. Do not mark a
   phase `[x]` unless its exit test genuinely passed in this session.
3. Append ADRs to `docs/05_DECISIONS.md` for every non-obvious choice made
   today — including choices you made silently while implementing.
4. Update `docs/04_EVALUATION.md` §5 if any run produced numbers.
5. If a rule emerged that should govern all future sessions, add it to the right
   section of `CLAUDE.md` (keep it terse; that file is loaded every time).
6. Commit on a descriptive branch or main as appropriate:
   `git add -A && git commit -m "phase N: <what landed>"`.
7. Print a **handoff block**: current state in 5 lines, known-broken things,
   the exact first command the next session should run, and any question that
   needs my decision before work continues.
