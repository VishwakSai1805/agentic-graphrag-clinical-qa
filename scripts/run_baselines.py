"""Phase 3 exit test: score baselines B0-B3 on a MedQA slice and write
reproducible run dirs under `results/`.

    uv run python -m scripts.run_baselines --dataset medqa --split train --limit 20

Roadmap Phase 3 exit test wording is "200-item MedQA dev slice"; the working
agreement (CLAUDE.md §5) runs phase exit tests on a 20-question dev slice.
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
CLAUDE.md forbids "A beats B" from a raw mean. Significance is Phase 7's job.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.config import load_config  # noqa: E402
from src.eval.baselines import BASELINES, Prediction, run_baseline  # noqa: E402
from src.eval.contamination import medqa_overlap_check  # noqa: E402
from src.eval.metrics import accuracy, macro_f1  # noqa: E402
from src.eval.results import make_run_id, write_run  # noqa: E402
from src.kg.dataset_questions import Question, load_question_sample  # noqa: E402
from src.llm.client import StatsTracker  # noqa: E402
from src.pipeline import TriModalPipeline  # noqa: E402
from src.retrieval.dense import DenseIndex  # noqa: E402
from src.retrieval.lexical import BM25Index  # noqa: E402
from src.retrieval.rerank import CrossEncoderReranker  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
CORPUS_PATH = REPO_ROOT / "data" / "processed" / "corpus.jsonl"
DENSE_DIR = REPO_ROOT / "indexes" / "dense"
BM25_DIR = REPO_ROOT / "indexes" / "bm25"

MEDQA_LETTERS = ("A", "B", "C", "D")


def print_gpu_status() -> None:
    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available() = {available}")
    if available:
        print(f"torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
        print(f"torch.cuda.get_device_capability(0) = {torch.cuda.get_device_capability(0)}")


def gold_option(q: Question) -> str:
    """MedQA `Question.answer` is the gold option *text*; medmcqa's is already
    a letter. Match text -> letter for medqa, asserting exactly one match."""
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
    raise NotImplementedError(f"gold_option not implemented for dataset {q.dataset!r}")


def contamination_assertion(dataset: str, questions: list[Question], pipeline) -> dict:
    """Run a REAL contamination check and return its executed outcome.

    AUDIT 2026-09-06 §3.2: this function previously returned a literal
    `{"checked": True, "applies": False, ...}` dict for MedQA without executing
    anything, and that unearned `checked: true` was written into every
    `results/medqa_B*/metrics.json` while `docs/04_EVALUATION.md` listed
    "contamination assertion passed" as a precondition for believing the
    results. CLAUDE.md §2 requires the assertion to run in code.

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


def summarise(preds: list[Prediction], baseline: str) -> dict:
    n = len(preds)
    status_counts = Counter(p.status for p in preds)
    grounded = sum(1 for p in preds if p.grounded)
    invalid_cites = sum(len(p.invalid_citations) for p in preds)
    calls = [p.n_llm_calls for p in preds]
    metrics: dict = {
        "n_evaluated": n,
        "accuracy": accuracy(preds),
        "macro_f1": macro_f1(preds, MEDQA_LETTERS),
        "macro_f1_note": "position-bias check, NOT a skill metric for 4-option MCQ; "
        "Accuracy is the headline",
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="medqa", choices=["medqa", "medmcqa"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--baselines", default=",".join(BASELINES))
    ap.add_argument("--config", default=str(CONFIG_PATH))
    args = ap.parse_args()

    requested = [b.strip() for b in args.baselines.split(",") if b.strip()]
    bad = [b for b in requested if b not in BASELINES]
    if bad:
        ap.error(f"unknown baseline(s) {bad}; valid: {BASELINES}")

    print_gpu_status()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed

    questions = load_question_sample(args.dataset, args.split, args.limit, seed=seed)
    golds = {q.question_id: gold_option(q) for q in questions}
    print(
        f"[run_baselines] {len(questions)} questions, dataset={args.dataset} "
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
    print(
        f"[run_baselines] contamination check: checked={contam['checked']} "
        f"applies={contam['applies']}"
        + (
            f" max_overlap={contam['max_overlap']:.3f} n_flagged={contam['n_flagged']}"
            if contam.get("checked")
            else ""
        )
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
            )
            preds.append(pred)
            if (i + 1) % 5 == 0 or (i + 1) == len(questions):
                print(f"[run_baselines] {baseline}: {i + 1}/{len(questions)}", flush=True)

        metrics = summarise(preds, baseline)
        metrics["contamination_assertion"] = contam
        metrics["llm_calls_total"] = tracker.summary().calls
        metrics["llm_cache_hits"] = tracker.summary().cache_hits

        run_id = make_run_id(args.dataset, baseline)
        run_dir = write_run(
            run_id=run_id,
            cfg=cfg,
            dataset=args.dataset,
            variant=baseline,
            split=args.split,
            limit=args.limit,
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
