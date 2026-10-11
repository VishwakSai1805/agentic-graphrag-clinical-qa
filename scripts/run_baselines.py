"""Phase 3 exit test: score baselines B0-B3 on a MedQA slice and write
reproducible run dirs under `results/`. Also wired for medmcqa and pubmedqa
(AUDIT 2026-09-06 §5.4: pubmedqa was medqa/medmcqa-only, so C4's PubMedQA
baseline-parity claim had no code path). PubMedQA is 3-way yes/no/maybe, not
4-option MCQ -- `option_letters(dataset)` and `gold_option()` both branch on
that; `src/eval/baselines.py`'s prompt/parse/verify logic was already
dict-driven over whatever `options` it's given, so no change was needed there.

    uv run python -m scripts.run_baselines --dataset medqa --split train --limit 20
    uv run python -m scripts.run_baselines --dataset pubmedqa --split dev --limit 20

Roadmap Phase 3 exit test wording is "200-item MedQA dev slice"; the working
agreement (PROJECT_RULES.md §5) runs phase exit tests on a 20-question dev slice.
MedQA "dev" = a seeded sample of the train split (docs/03_DATASETS.md) -- the
official test split stays unseen until Phase 7. The KG cache for this slice is
precomputed by `scripts/precompute_kg_cache.py --dataset medqa --split train`.

One `results/<run_id>/` per baseline. `run_id = <dataset>_<variant>_<sha7>_<ts>`.
metrics.json carries: accuracy, macro_f1 (position-bias check, not a skill
metric for 4-option MCQ -- see src/eval/metrics.py), n_evaluated, grounded_rate,
invalid_citation_count, status counts, mean_llm_calls, contamination_assertion,
and (B3 only) verifier_decision_rate + n_supported distribution -- the last is
direct evidence for the binary-bottleneck the C2 contribution targets.

No cross-baseline comparison claim is made here: n=20 is far too small, and
PROJECT_RULES.md forbids "A beats B" from a raw mean. Significance is Phase 7's job.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.config import load_config  # noqa: E402
from src.eval.baselines import _PUBMEDQA_OPTIONS, BASELINES, Prediction, run_baseline  # noqa: E402
from src.eval.contamination import medqa_overlap_check  # noqa: E402
from src.eval.metrics import accuracy, macro_f1  # noqa: E402
from src.eval.results import make_run_id, write_run  # noqa: E402
from src.kg.dataset_questions import Question, load_question_sample  # noqa: E402
from src.llm.client import (  # noqa: E402
    StatsTracker,
    assert_generator_provenance,
    set_active_config,
)
from src.pipeline import TriModalPipeline  # noqa: E402
from src.retrieval.dense import DenseIndex  # noqa: E402
from src.retrieval.lexical import BM25Index  # noqa: E402
from src.retrieval.rerank import CrossEncoderReranker  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
CORPUS_PATH = REPO_ROOT / "data" / "processed" / "corpus.jsonl"
DENSE_DIR = REPO_ROOT / "indexes" / "dense"
BM25_DIR = REPO_ROOT / "indexes" / "bm25"

MEDQA_LETTERS = ("A", "B", "C", "D")
PUBMEDQA_LETTERS = ("A", "B", "C")  # yes/no/maybe -- see src.eval.baselines._PUBMEDQA_OPTIONS


def option_letters(dataset: str) -> tuple[str, ...]:
    """The fixed label set for macro_f1 per dataset. medqa/medmcqa are
    4-option MCQ; pubmedqa is 3-way yes/no/maybe, not 4-option -- using
    MEDQA_LETTERS for it would silently score a phantom "D" class."""
    return PUBMEDQA_LETTERS if dataset == "pubmedqa" else MEDQA_LETTERS


def print_gpu_status() -> None:
    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available() = {available}")
    if available:
        print(f"torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
        print(f"torch.cuda.get_device_capability(0) = {torch.cuda.get_device_capability(0)}")


def gold_option(q: Question) -> str:
    """MedQA `Question.answer` is the gold option *text*; medmcqa's is already
    a letter. Match text -> letter for medqa, asserting exactly one match.
    PubMedQA has no `options` on the Question itself (`_to_question` sets
    `options=None`) -- its 3-way yes/no/maybe label set is
    `src.eval.baselines._PUBMEDQA_OPTIONS`, the same fixed dict
    `_resolve_options` falls back to when building its prompt, so the gold
    letter and the prompt's option lettering never drift apart."""
    if q.dataset == "medmcqa":
        if q.answer not in (q.options or {}):
            raise ValueError(
                f"{q.question_id}: medmcqa answer {q.answer!r} not a valid option letter"
            )
        return q.answer
    if q.dataset == "medqa":
        if not q.options or q.answer is None:
            raise ValueError(f"{q.question_id}: missing options/answer")
        matches = [k for k, v in q.options.items() if v.strip() == q.answer.strip()]
        if len(matches) != 1:
            raise ValueError(
                f"{q.question_id}: gold text matched {len(matches)} options (expected 1) -- "
                "raw file / normalisation drift"
            )
        return matches[0]
    if q.dataset == "pubmedqa":
        if q.answer is None:
            raise ValueError(f"{q.question_id}: missing final_decision")
        matches = [
            k for k, v in _PUBMEDQA_OPTIONS.items() if v.strip() == q.answer.strip().lower()
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{q.question_id}: final_decision {q.answer!r} matched {len(matches)} of "
                f"{_PUBMEDQA_OPTIONS} (expected 1)"
            )
        return matches[0]
    raise NotImplementedError(f"gold_option not implemented for dataset {q.dataset!r}")


def contamination_assertion(dataset: str, questions: list[Question], pipeline) -> dict:
    """Run a REAL contamination check and return its executed outcome.

    AUDIT 2026-09-06 §3.2: this function previously returned a literal
    `{"checked": True, "applies": False, ...}` dict for MedQA without executing
    anything, and that unearned `checked: true` was written into every
    `results/medqa_B*/metrics.json` while `docs/04_EVALUATION.md` listed
    "contamination assertion passed" as a precondition for believing the
    results. PROJECT_RULES.md §2 requires the assertion to run in code.

    MedQA genuinely has no source-abstract blocklist (`blocklisted_pmids`
    raises for it by design), but the corpus does contain the 18 USMLE
    textbooks the items were written from (ADR-009). So the executed check is
    a stem-vs-retrieved-chunk verbatim n-gram overlap measurement
    (`src.eval.contamination.medqa_overlap_check`) over the real retrieval
    path, reported as a distribution. `checked`/`applies` come from that
    outcome. Without a retrieval pipeline (a B0-only run) nothing can be
    retrieved to compare against, and that is recorded as `checked: false`
    rather than being passed off as a clean check.
    """
    if dataset == "pubmedqa":
        # Real blocklist check: the pqa_labeled TEST 500 must not be indexed
        # (mid-Phase-6 fix -- previously this path returned an unearned
        # checked=False/applies=True, the AUDIT §3.2 failure class).
        from src.eval.contamination import pubmedqa_contamination_report

        return pubmedqa_contamination_report(CORPUS_PATH)
    if dataset != "medqa":
        return {
            "checked": False,
            "applies": True,
            "note": f"contamination check for {dataset} not wired in Phase 3",
        }
    if pipeline is None:
        return {
            "checked": False,
            "applies": True,
            "note": "no retrieval pipeline loaded (B0-only run): there are no retrieved "
            "chunks to measure question-stem overlap against, so no MedQA contamination "
            "check was executed for this run.",
        }
    top_k = pipeline.cfg.retrieval.top_k_rerank
    stems_and_chunks: list[tuple[str, list[str]]] = []
    for q in questions:
        bundle = pipeline.build_bundle(
            q.question, q.options or {}, dataset=dataset, question_id=q.question_id
        )
        stems_and_chunks.append((q.question, [e.text for e in bundle.passages]))
    result = medqa_overlap_check(stems_and_chunks)
    result["retrieved_passages_per_question"] = top_k
    return result


def summarise(preds: list[Prediction], baseline: str, dataset: str) -> dict:
    n = len(preds)
    status_counts = Counter(p.status for p in preds)
    grounded = sum(1 for p in preds if p.grounded)
    invalid_cites = sum(len(p.invalid_citations) for p in preds)
    calls = [p.n_llm_calls for p in preds]
    letters = option_letters(dataset)
    metrics: dict = {
        "n_evaluated": n,
        "accuracy": accuracy(preds),
        "macro_f1": macro_f1(preds, letters),
        "macro_f1_note": f"position-bias check, NOT a skill metric for {len(letters)}-option "
        "MCQ; Accuracy is the headline",
        "grounded_rate": (grounded / n) if n else 0.0,
        "invalid_citation_count": invalid_cites,
        "status_counts": dict(status_counts),
        "mean_llm_calls": (sum(calls) / n) if n else 0.0,
    }
    if baseline == "B3":
        decided = [p for p in preds if p.verifier_decided is not None]
        n_decided_true = sum(1 for p in decided if p.verifier_decided)
        metrics["verifier_decision_rate"] = (n_decided_true / len(decided)) if decided else 0.0
        metrics["n_supported_distribution"] = dict(
            Counter(p.n_supported for p in preds if p.n_supported is not None)
        )
        metrics["verifier_decision_rate_note"] = (
            "fraction of questions where exactly one option verified SUPPORTED (verifier decided) "
            "vs. fell back to B2-style generation. A high fallback rate is direct evidence for the "
            "binary bottleneck C2 targets, not a defect of this run."
        )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    """The CLI, as a value so it can be parsed in tests without running a job.
    See `tests/test_launcher_argv.py` and `src.eval.harness.build_parser`."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="medqa", choices=["medqa", "medmcqa", "pubmedqa"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--baselines", default=",".join(BASELINES))
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument(
        "--equal-budget",
        action="store_true",
        help="AUDIT 2026-09-06 SS4.3: trim B1/B2/B3's evidence bundle to "
        "agents.max_evidence via the same modality-reserved-slot split V4/V5 use, "
        "instead of up to 12 passages PLUS all cached KG triples (~52 total). "
        "Default off -- does not change any already-committed run's numbers.",
    )
    ap.add_argument(
        "--question-ids",
        default=None,
        help="JSON file with {'question_ids': [...]} pinning the exact question "
        "set and order (ADR-047). Overrides --limit. Required to reproduce a "
        "subset of an existing run: a seeded sample at a smaller --limit is a "
        "DIFFERENT draw, not a prefix.",
    )
    ap.add_argument("--llm-model", default=None, help="override fallback_chain[0].model")
    ap.add_argument("--llm-api-version", default=None, help="recorded in config.yaml")
    ap.add_argument(
        "--variant-suffix",
        default="",
        help="appended to the recorded variant/run_id (e.g. '-gemini'), so a run "
        "under a different generator can never be conflated with the grid cell "
        "of the same baseline name. Pass it as --variant-suffix=-gemini: the "
        "space form is unparseable because the value starts with a dash.",
    )
    ap.add_argument(
        "--allow-warm-cache",
        action="store_true",
        help="permit a hosted-provider run to be served mostly from cache "
        "(ADR-048). Use ONLY for a deliberate resume.",
    )
    return ap


def main() -> None:
    ap = build_parser()  # kept bound: ap.error(...) is used for validation below
    args = ap.parse_args()

    requested = [b.strip() for b in args.baselines.split(",") if b.strip()]
    bad = [b for b in requested if b not in BASELINES]
    if bad:
        ap.error(f"unknown baseline(s) {bad}; valid: {BASELINES}")

    print_gpu_status()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed

    # Stamp generator provenance into the config BEFORE write_run records it
    # (ADR-047): a hosted endpoint can change behind a fixed name, so the model
    # id, API version and resolution date all belong in the run's config.yaml.
    if args.llm_model is not None:
        if not cfg.llm.fallback_chain:
            ap.error("--llm-model given but the config has an empty fallback_chain")
        cfg.llm.fallback_chain[0].model = args.llm_model
    if args.llm_model is not None or args.llm_api_version is not None:
        import datetime as _dt

        cfg.llm.api_version = args.llm_api_version
        cfg.llm.model_resolved_utc = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ADR-048: src/eval/baselines.py calls chat() without `config=`, so without
    # this every call silently uses configs/base.yaml instead of --config.
    set_active_config(cfg)

    if args.question_ids:
        pinned = json.loads(Path(args.question_ids).read_text(encoding="utf-8"))["question_ids"]
        by_id = {q.question_id: q for q in load_question_sample(args.dataset, args.split, None)}
        missing = [qid for qid in pinned if qid not in by_id]
        if missing:
            ap.error(
                f"{len(missing)} pinned question_id(s) absent from "
                f"{args.dataset}/{args.split}: {missing[:5]}"
            )
        questions = [by_id[qid] for qid in pinned]
        print(f"[run_baselines] pinned question set: {len(questions)} ids from {args.question_ids}")
    else:
        questions = load_question_sample(args.dataset, args.split, args.limit, seed=seed)
    n_questions = len(questions)
    golds = {q.question_id: gold_option(q) for q in questions}
    print(
        f"[run_baselines] {n_questions} questions, dataset={args.dataset} "
        f"split={args.split} seed={seed}"
    )

    need_pipeline = any(b != "B0" for b in requested)
    pipeline = None
    if need_pipeline:
        if not CORPUS_PATH.exists():
            print(
                f"[run_baselines] ERROR: {CORPUS_PATH} missing -- run scripts.build_corpus",
                file=sys.stderr,
            )
            sys.exit(2)
        print("[run_baselines] loading dense / bm25 / reranker ...", flush=True)
        dense = DenseIndex.load(DENSE_DIR)
        bm25 = BM25Index.load(BM25_DIR)
        reranker = CrossEncoderReranker(max_length=cfg.retrieval.rerank.max_length)
        pipeline = TriModalPipeline(
            cfg, dense=dense, bm25=bm25, reranker=reranker, corpus_path=CORPUS_PATH
        )

    contam = contamination_assertion(args.dataset, questions, pipeline)
    if "max_overlap" in contam:  # MedQA n-gram overlap distribution
        detail = f" max_overlap={contam['max_overlap']:.3f} n_flagged={contam['n_flagged']}"
    elif "passed" in contam:  # PubMedQA blocklist check
        detail = f" passed={contam['passed']} blocklist_size={contam['blocklist_size']}"
    else:
        detail = ""
    print(
        f"[run_baselines] contamination check: checked={contam['checked']} "
        f"applies={contam['applies']}{detail}"
    )

    for baseline in requested:
        tracker = StatsTracker()
        preds: list[Prediction] = []
        for i, q in enumerate(questions):
            pred = run_baseline(
                baseline,
                q,
                golds[q.question_id],
                pipeline=pipeline,
                dataset=args.dataset,
                tracker=tracker,
                seed=seed,
                agents_cfg=cfg.agents if args.equal_budget else None,
            )
            preds.append(pred)
            if (i + 1) % 5 == 0 or (i + 1) == len(questions):
                print(f"[run_baselines] {baseline}: {i + 1}/{len(questions)}", flush=True)

        metrics = summarise(preds, baseline, args.dataset)
        metrics["contamination_assertion"] = contam
        metrics["llm_calls_total"] = tracker.summary().calls
        metrics["llm_cache_hits"] = tracker.summary().cache_hits
        # Raises before write_run, so a misattributed run never reaches results/.
        metrics["generator_provenance"] = assert_generator_provenance(
            cfg,
            tracker.summary(),
            run_label=f"{baseline}{args.variant_suffix}",
            allow_warm_cache=args.allow_warm_cache,
        )
        metrics["equal_budget"] = args.equal_budget
        if args.equal_budget:
            metrics["equal_budget_max_evidence"] = cfg.agents.max_evidence
            metrics["equal_budget_evidence_split"] = list(cfg.agents.evidence_split)

        # Equal-budget runs get a distinct variant suffix on disk (run_id +
        # write_run's recorded `variant`) so they can never be silently
        # conflated with the up-to-~52-item default runs when reading
        # results/ later -- `baseline` itself (B0/B1/B2/B3) stays unchanged
        # for dispatch and the B3-specific handling below.
        variant = f"{baseline}eq" if args.equal_budget else baseline
        variant = f"{variant}{args.variant_suffix}"
        run_id = make_run_id(args.dataset, variant)
        run_dir = write_run(
            run_id=run_id,
            cfg=cfg,
            dataset=args.dataset,
            variant=variant,
            split=args.split,
            limit=n_questions,
            seed=seed,
            predictions=[p.model_dump(mode="json") for p in preds],
            metrics=metrics,
        )
        print(
            f"[run_baselines] {baseline}: acc={metrics['accuracy']:.3f} "
            f"macroF1={metrics['macro_f1']:.3f} grounded={metrics['grounded_rate']:.2f} "
            f"invalid_cites={metrics['invalid_citation_count']} "
            f"calls={metrics['llm_calls_total']} -> {run_dir}"
        )
        if baseline == "B3":
            print(
                f"[run_baselines] B3 verifier_decision_rate="
                f"{metrics['verifier_decision_rate']:.2f} "
                f"n_supported_dist={metrics['n_supported_distribution']}"
            )


if __name__ == "__main__":
    main()
