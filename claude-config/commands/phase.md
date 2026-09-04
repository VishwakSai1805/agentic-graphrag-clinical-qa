---
description: Plan and execute one roadmap phase end-to-end, gated on the exit test
argument-hint: <phase-number>
allowed-tools: Read, Edit, Write, Glob, Grep, Bash, Task, TodoWrite
---

Execute **Phase $1** from `docs/02_ROADMAP.md`.

## Step 1 — Ground yourself
Read `CLAUDE.md`, `docs/01_ARCHITECTURE.md`, the Phase $1 block of
`docs/02_ROADMAP.md`, and any doc it references. Inspect what already exists in
the repo — do not assume the phase starts from nothing.

## Step 2 — Plan (no code yet)
Produce, in this order:
1. Deliverables — exact file paths.
2. Key interfaces — function/class signatures and the pydantic models only.
3. External dependencies to add, with versions and why.
4. The exit test verbatim, and how you will know it genuinely passed.
5. Risks: anything in the spec that is under-specified, contradictory, or that
   you believe will not work. Say so plainly rather than coding around it.
6. Estimated LLM/API cost of running the exit test.

**Stop and wait for my approval.**

## Step 3 — Implement
- Smallest vertical slice that satisfies the exit test first; polish after.
- Tests alongside logic, LLM mocked.
- Delegate bulk work to the specialist subagents in `.claude/agents/` so their
  output does not fill this context.
- Run `ruff check src tests` and `pytest -q` before declaring anything done.

## Step 4 — Prove it
Run the exit test. Paste the **real** output. If it fails, fix it — never
describe a passing run that did not happen, and never soften the exit test to
make it pass.

## Step 5 — Close out
- Flip Phase $1 to `[x]` in `docs/02_ROADMAP.md` with the commit sha.
- Append any ADRs to `docs/05_DECISIONS.md`.
- Summarise in ≤10 lines: what exists now, what is deliberately deferred, and the
  first thing the next session should do.
