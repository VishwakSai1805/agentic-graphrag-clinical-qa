# Next session — start here

**Done:** Phases 0–4, all committed to `master`. Phase 3 `b10c385`/`d445423`,
Phase 4 `e75de9c`/`6f9690e`/`1388904`, ADR-025 conflict fix `30700f9`/`ccba531`.
298 tests pass, ruff clean.

**2026-09-07 audit remediation session (`docs/AUDIT_2026-09-06.md`) is done and
committed.** Before this session, Phase 5 was unsound (calibration/corpus
overlap, §3.3) and the Phase 2 KG decisions rested on the wrong split (§3.1).
Both are fixed. Full detail lives in the ADRs below — this file is the entry
point, not the record.

---

## What changed today, in order

1. **Test-split leakage (§3.1) — fixed. ADR-027.** The Phase 2 KG probe
   originally ran on MedQA *test* (`medqa_n200.json`, `medqa_n500.json`,
   `medqa_n200_kg_rerank_*` — now stamped `superseded: true`, retained as
   provenance, never cite them). Re-ran on a 500-question MedQA **train**
   slice for both `kg_rerank` variants
   (`results/phase2_kg_probe/medqa_train_n500*.json` — now authoritative).
   ADR-015/016/017 upheld on the clean split. **ADR-018's verdict flipped**:
   `novel_and_relevant_rate` was a tie at n=200 test, is a **significant win**
   at n=500 train (+0.0367, 95% CI [+0.0044, +0.0681]) — and this was
   Holm-Bonferroni corrected for the 2-metric family the same day (**both
   metrics survive**, see `src.retrieval.metrics.paired_bootstrap_p` /
   `holm_bonferroni`, AUDIT §6.8). `kg_rerank` stays off by default anyway —
   **ADR-027 names this as a deviation from ADR-018's own pre-registered
   rule** (which said adopt on a cleared verdict), states the reasons, and
   points at the Phase 7 `kg_rerank` grid row as what actually settles it.
2. **Four integrity blockers — fixed.**
   - §2.1: `surviving_invalid_citation_count` is a real re-validation against
     `SupportMatrix.bundle_evidence_ids` (persisted per question), not a
     hardcoded `0`.
   - §3.2: `contamination_assertion` executes a real 8-gram stem-vs-retrieved-
     chunk overlap check (`src.eval.contamination.medqa_overlap_check`) — no
     more literal `{"checked": True}` dict.
   - §6.1: `git_sha7()` / `git_provenance()` append `-dirty` and record a
     `git_diff_sha256` + changed-file list when the worktree is dirty.
   - §6.2: the four canonical `medqa_B*` run ids are named explicitly in
     `docs/04_EVALUATION.md` §5.4; the four pre-`num_ctx` runs carry
     `superseded: true` (`scripts/mark_superseded.py`).
3. **Calibration/corpus overlap (§3.3) — fixed. ADR-026.** Calibration now
   fits on `pqa_labeled` **dev** (500 items), verified — not assumed — absent
   from the 20k-doc retrieval corpus
   (`src.eval.contamination.assert_calibration_fit_set_uncontaminated`,
   which **Phase 5 must call before fitting**). `pqa_artificial` is a
   corpus-source only now, never a fitting set.
4. **`docs/00_RESEARCH_SPEC.md` §3 rewritten.** C1 → diagnostic/negative
   pending the Phase 7 V2-vs-V3 cell; C2 → audit-discrimination, conflict
   resolution flagged unverified; "guarantee" dropped from C3; C4 marked
   pending; **C5 added** — the KG-hit-rate arithmetic artifact, the project's
   best-supported, fully-traceable finding, was previously unlisted.
5. **ADR-021 amended** with the no-op perturbation row: B0 (no retrieval,
   unaffected by the num_ctx fix) still flips 1/20 answers between the two
   batches, so the measured noise floor at n=20 is ±0.05 — the B1 accuracy
   delta is retracted as an effect; the decision itself still stands on the
   categorical B2/B3 evidence (parse_error 12/20→0/20).
6. **§5.4's B3 framing corrected** (the auditor's reading was wrong): both
   context conditions show `verifier_decision_rate = 0.05` with exactly one
   SUPPORTED option on 1/20 questions — only the failure *direction* flips
   (over-admit at 4096-ctx vs. over-reject at 8192-ctx). Reported as
   invariance under a perturbation that inverts the failure mode, which is
   stronger evidence for the binary-bottleneck claim than either run alone,
   not weaker.
7. **Five one-liner doc fixes** (§2.2–§2.6): ADR-024 status line now says
   "partly superseded by ADR-025"; CLAUDE.md's dead NLI model id corrected to
   the `-ling-wanli` variant; `configs/base.yaml`'s stale `kg_rerank` comment
   corrected; README's "Phase 0 — scaffold" corrected to "Phases 0–4 complete";
   ADR-020's "test-set hygiene held since Phase 1" corrected and scoped.

**Deferred, tracked in `docs/02_ROADMAP.md`'s "Pre-Phase-7 fixes" section:**
§1.1/§1.2 (real Phase 1 `metrics.json`), §4.1 (B3 prompt-sensitivity), §4.3
(equal-budget B2 comparator), §6.7 (grid-wide multiplicity discipline — the
kg_rerank decision above got its own targeted correction today, this is the
full ~9-metric × ~10-variant-pair version), all MINORs. Still open and not yet
triaged: §1.3–1.9, §2.3 *(fixed today, strike from this list)*, §3.4, §5.5,
§6.3, §6.4.

---

## Where the three contributions actually stand (`docs/00_RESEARCH_SPEC.md` §3
is the authoritative version — this is the short version)

- **C1** — diagnostic/negative, pending Phase 7 V2-vs-V3. No run yet shows the
  KG arm improving anything downstream; 1/108 audit cells cite KG even with
  the slot filled 17/20 times.
- **C2** — audit-discrimination is supported (108 cells, 61/80 options get
  signal, vs. B3's binary verifier isolating 1/20 under both context
  conditions tested). Conflict *resolution* is unverified: `conflict_fire_rate
  = 0.00` under the correct claim-level definition.
- **C3** — untested. Phase 5 decides it. No "guarantee" language; a temperature
  fit on 500 items is a real limitation to state alongside the ECE numbers.
- **C4** — pending Phase 7. PubMedQA has no baseline code path yet
  (`run_baselines.py --dataset` is `{medqa, medmcqa}` only).
- **C5 (new)** — SUPPORTED, fully traceable: the KG-hit-rate arithmetic
  artifact (~96% question-level "coverage" vs. 11% real per-entity mapping).
  Currently the strongest, most defensible result in the project.

---

## Delivery schedule

See `docs/SCHEDULE.md` — hard deadline Saturday 2026-09-12, day-by-day plan,
accepted scope cuts (MedQA-only grid, 5 cells not 8×3×3, RAGAS dropped,
minimal Phase 6 harness). Today (Mon 7) was the audit-remediation day on that
schedule. **Next: `/phase 5`** — calibration, ranked output, ECE, p-vectors
persisted, fit on `pqa_labeled` dev per ADR-026.

## Standing rules

- Per ADR-012/018/025/027: a result that undercuts a contribution is
  reported, not engineered around — including naming a deviation from a
  pre-registered decision rule when one happens (ADR-027).
- n=20 dev numbers are plumbing checks. No "A beats B" claims until Phase 7.
- Never select a hyper-parameter, threshold, or ablation decision by reading
  the split Phase 7 reports on (ADR-027 is what happens when this rule is
  violated and later caught).
