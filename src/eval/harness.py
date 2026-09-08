"""Phase 6 minimal evaluation harness: run a variant from a config, score it,
write one reproducible `results/<run_id>/`.

**Deliberately minimal** (`docs/SCHEDULE.md`, ADR-029): run -> score -> write.
No `asyncio.Semaphore`, no resumability, no concurrency benchmark -- those were
Phase 6's "nice to have" and are cut. Sequential loop, one process.

This is the single entrypoint Phase 7's ablation grid drives. It does not
re-implement retrieval / agents / calibration -- it orchestrates:

    build_bundle  ->  (agents) consensus_over_questions  ->  rank_options
                  ->  (baseline) _generate_answer
    then score with src.eval.{metrics,nli}
    then src.eval.results.write_run

Variant map (one source of truth -- `VARIANTS`):

    V0  zero-shot            = baseline B0, no retrieval
    V1  dense RAG            = {dense}
    V2  hybrid RAG           = {dense, lexical}          <- NEW, never run before Phase 6
    V3  +KG                  = {dense, lexical, kg}      <- == the committed medqa_B2_* runs
    V4  +agents              = consensus network, ranked by (s_pos - s_neg)
    V5  full                 = agents + Phase 5 calibrated softmax

Ranking metrics (P@K / MRR / ECE) per the approved Phase 6 plan:
  * V0-V3: reported as `null` -- a single predicted option is not a ranking.
  * V4:    P@K / MRR from the UNCALIBRATED `(s_pos - s_neg)` support-margin
           ordering (`ranking_source="uncalibrated_support_margin"`); ECE `null`.
  * V5:    P@K / MRR / ECE from the calibrated probability vector.

Hallucination rate (`src.eval.nli`) is reported under BOTH denominators
(predicted option / all SUPPORT cells) -- see ADR-029.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src.config import AppConfig, load_config
from src.eval.contamination import medqa_overlap_check, pubmedqa_contamination_report
from src.eval.metrics import accuracy, ece, macro_f1, mean_mrr, mean_precision_at_k
from src.eval.nli import ClaimRecord, score_hallucination, split_claims
from src.eval.results import RESULTS_DIR, make_run_id, patch_metrics, write_run

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "processed" / "corpus.jsonl"

MEDQA_LETTERS = ("A", "B", "C", "D")
PUBMEDQA_LETTERS = ("A", "B", "C")  # yes / no / maybe

_RANKED_SOURCES = ("calibrated_softmax", "uncalibrated_support_margin")


# --------------------------------------------------------------------------
# variant table
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantSpec:
    variant: str
    kind: Literal["baseline", "agents"]
    baseline: str | None = None
    modalities: frozenset[str] = field(default_factory=frozenset)
    calibration: bool = False


VARIANTS: dict[str, VariantSpec] = {
    "V0": VariantSpec("V0", "baseline", baseline="B0", modalities=frozenset()),
    "V1": VariantSpec("V1", "baseline", baseline="B1", modalities=frozenset({"dense"})),
    "V2": VariantSpec(
        "V2", "baseline", baseline="B2", modalities=frozenset({"dense", "lexical"})
    ),
    "V3": VariantSpec(
        "V3", "baseline", baseline="B2", modalities=frozenset({"dense", "lexical", "kg"})
    ),
    "V4": VariantSpec("V4", "agents", calibration=False),
    "V5": VariantSpec("V5", "agents", calibration=True),
    # Phase 7 leave-one-agent-out cell (ADR-035). Runs the agent path with
    # `--config configs/v4_no_conflict.yaml` (agents.run_conflict=false ->
    # graph routes audit->END). Ranked by the uncalibrated (s+ - s-) margin,
    # identical in shape to V4. V4-audit is deferred (needs the binary-audit
    # implementation, ADR-036).
    "V4-conflict": VariantSpec("V4-conflict", "agents", calibration=False),
}


def resolve_variant(cfg: AppConfig) -> str:
    """Derive the V-id from a resolved config, for the agent variants only.

    `configs/base.yaml` cannot distinguish V0-V3 (it has no variant field and
    the same retrieval block), so a baseline variant MUST be passed explicitly
    on the CLI -- this raises for that case rather than guessing.
    """
    if cfg.calibration.enabled:
        return "V5"
    if cfg.agents.enabled:
        return "V4"
    raise ValueError(
        "config has neither agents nor calibration enabled -- pass --variant "
        "explicitly (V0/V1/V2/V3) so the retrieval modality set is unambiguous"
    )


# --------------------------------------------------------------------------
# per-question record
# --------------------------------------------------------------------------


class QuestionRecord(BaseModel):
    question_id: str
    gold_option: str | None
    pred_option: str | None
    ranked: list[str]
    p_vector: list[float] | None
    ranking_source: Literal[
        "calibrated_softmax", "uncalibrated_support_margin", "single_prediction", "none"
    ]
    abstained: bool = False
    citations: dict[str, list[str]]  # option_id -> evidence ids cited FOR it
    invalid_citations: list[str]
    grounded: bool
    rationale_claims: list[str]
    claim_records: list[ClaimRecord] = []
    status: str
    n_llm_calls: int
    # Full agent SupportMatrix for agent variants (kept nested so Phase 5's
    # fit_calibration.py / raw_logprob_baseline.py can still read a V4 run's
    # predictions.jsonl -- ADR-029 schema-drift note).
    agent_matrix: dict | None = None


class _AccView(BaseModel):
    """Duck-typed view `src.eval.metrics.{accuracy,macro_f1}` accept."""

    pred_option: str | None
    gold_option: str | None


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------


def _evidence_text_map(evidence_objs) -> dict[str, str]:
    return {e.id: e.text for e in evidence_objs}


def _baseline_record(
    spec: VariantSpec, question, gold: str | None, pipeline, dataset: str, tracker, seed: int
) -> QuestionRecord:
    from src.eval.baselines import (
        _finalize,
        _generate_answer,
        _resolve_options,
        _run_b0,
    )

    options = _resolve_options(question)
    if spec.baseline == "B0":
        pred = _run_b0(question, gold or "", tracker, seed)
        id2text: dict[str, str] = {}
    else:
        bundle = pipeline.build_bundle(
            question.question,
            options,
            dataset=dataset,
            question_id=question.question_id,
            modalities=spec.modalities,
        )
        id2text = _evidence_text_map(bundle.evidence())
        letter, citations, status, rationale = _generate_answer(
            question, options, bundle, tracker, seed
        )
        pred = _finalize(
            question_id=question.question_id,
            letter=letter,
            citations=citations,
            status=status,
            gold_option=gold or "",
            valid_ids=bundle.valid_citation_ids(),
            n_llm_calls=1,
            rationale=rationale,
        )

    claims = split_claims(pred.rationale)
    cited_texts = [id2text[c] for c in pred.citations if c in id2text]
    claim_records = [
        ClaimRecord(
            question_id=pred.question_id,
            option_id=pred.pred_option,
            claim=claim,
            cited_evidence_texts=cited_texts,
            predicted_option=True,  # a baseline's single rationale IS the predicted option's
        )
        for claim in claims
    ]
    return QuestionRecord(
        question_id=pred.question_id,
        gold_option=gold,
        pred_option=pred.pred_option,
        ranked=[pred.pred_option] if pred.pred_option else [],
        p_vector=None,
        ranking_source="single_prediction",
        citations={pred.pred_option: list(pred.citations)} if pred.pred_option else {},
        invalid_citations=list(pred.invalid_citations),
        grounded=pred.grounded,
        rationale_claims=claims,
        claim_records=claim_records,
        status=pred.status,
        n_llm_calls=pred.n_llm_calls,
    )


def _agent_record(spec: VariantSpec, cfg: AppConfig, matrix, state) -> QuestionRecord:
    from src.calibration.scoring import CalibrationWeights, extract_option_features
    from src.calibration.softmax_rank import rank_options

    feats = extract_option_features(matrix)
    bundle_ids = set(matrix.bundle_evidence_ids)
    id2text = _evidence_text_map(state.evidence)

    if spec.calibration:
        weights = CalibrationWeights(
            alpha=cfg.calibration.alpha,
            beta=cfg.calibration.beta,
            gamma=cfg.calibration.gamma,
            delta=cfg.calibration.delta,
            temperature=cfg.calibration.temperature,
        )
        ranked_pairs = rank_options(feats, weights)  # [(oid, p), ...] desc
        ranked = [oid for oid, _ in ranked_pairs]
        pv = [p for _, p in ranked_pairs]
        p_vector: list[float] | None = pv
        ranking_source = "calibrated_softmax"
        abstained = len(pv) > 0 and pv[0] < cfg.calibration.abstention_tau
    else:
        margins = sorted(
            ((f.option_id, f.s_pos - f.s_neg) for f in feats),
            key=lambda kv: kv[1],
            reverse=True,
        )
        ranked = [oid for oid, _ in margins]
        p_vector = None
        ranking_source = "uncalibrated_support_margin"
        abstained = False

    pred_option = ranked[0] if ranked else None

    # citations & invalid citations from the matrix's SUPPORT cells
    from src.calibration.citations import cite_all

    citations = cite_all(matrix)
    invalid = sorted(
        {c.evidence_id for c in matrix.cells if c.evidence_id and c.evidence_id not in bundle_ids}
    )
    grounded = pred_option is not None and len(citations.get(pred_option, [])) > 0

    claim_records: list[ClaimRecord] = []
    predicted_claims: list[str] = []
    for cell in matrix.cells:
        if cell.label != "SUPPORT" or not (cell.rationale or "").strip():
            continue
        txt = id2text.get(cell.evidence_id)
        is_pred = cell.option_id == pred_option
        claim_records.append(
            ClaimRecord(
                question_id=matrix.question_id,
                option_id=cell.option_id,
                claim=cell.rationale.strip(),
                cited_evidence_texts=[txt] if txt else [],
                predicted_option=is_pred,
            )
        )
        if is_pred:
            predicted_claims.append(cell.rationale.strip())

    return QuestionRecord(
        question_id=matrix.question_id,
        gold_option=matrix.gold_option,
        pred_option=pred_option,
        ranked=ranked,
        p_vector=p_vector,
        ranking_source=ranking_source,  # type: ignore[arg-type]
        abstained=abstained,
        citations=citations,
        invalid_citations=invalid,
        grounded=grounded,
        rationale_claims=predicted_claims,
        claim_records=claim_records,
        status=matrix.status,
        n_llm_calls=matrix.llm_calls,
        agent_matrix=matrix.model_dump(mode="json"),
    )


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def score_run(
    records: list[QuestionRecord],
    *,
    dataset: str,
    nli_scorer=None,
    n_ece_bins: int = 15,
    ece_binning: str = "equal_width",
    compute_hallucination: bool = True,
) -> dict:
    """The `metrics.json` body for a completed run. `nli_scorer` is an
    injectable `(premise, hypothesis) -> float`; `None` loads the real NLI
    model (GPU) lazily.

    `compute_hallucination=False` (the grid path, ADR-037): skip the NLI pass
    entirely and set `hallucination: None`, `hallucination_pending: True`. The
    caller writes the run dir first, then scores hallucination separately so
    the ~1.5 GB NLI model never coexists with the retrieval models -- a real
    OOM on the 16 GB machine (Phase 7 grid launch). `score_hallucination_into`
    / `scripts/analysis/rescore_hallucination.py` fill it in afterwards.
    """
    n = len(records)
    letters = PUBMEDQA_LETTERS if dataset == "pubmedqa" else MEDQA_LETTERS
    acc_view = [_AccView(pred_option=r.pred_option, gold_option=r.gold_option) for r in records]

    metrics: dict = {
        "n_evaluated": n,
        "dataset": dataset,
        "accuracy": accuracy(acc_view),  # type: ignore[arg-type]  # duck-typed on .pred/.gold
        "macro_f1": macro_f1(acc_view, list(letters)),  # type: ignore[arg-type]
        "macro_f1_note": (
            f"position-bias check, not a skill metric for {len(letters)}-option MCQ"
        ),
    }

    ranked = [r for r in records if r.ranking_source in _RANKED_SOURCES]
    if ranked and len(ranked) == n:
        rankings = [r.ranked for r in records]
        golds = [r.gold_option or "" for r in records]
        metrics["p_at_1"] = mean_precision_at_k(rankings, golds, 1)
        metrics["p_at_2"] = mean_precision_at_k(rankings, golds, 2)
        metrics["mrr"] = mean_mrr(rankings, golds)
        metrics["ranking_source"] = ranked[0].ranking_source
    else:
        metrics["p_at_1"] = metrics["p_at_2"] = metrics["mrr"] = None
        metrics["ranking_source"] = "single_prediction"
        metrics["ranking_note"] = (
            "P@K / MRR are null: this variant emits a single predicted option, "
            "not a ranked differential (approved Phase 6 plan)."
        )

    ece_rows = [
        r for r in records if r.p_vector and r.gold_option and r.gold_option in r.ranked
    ]
    if ece_rows and len(ece_rows) == n:
        probs = [r.p_vector for r in records]
        gold_idx = [r.ranked.index(r.gold_option or "") for r in records]  # aligned to p_vector
        metrics["ece"] = ece(probs, gold_idx, n_bins=n_ece_bins, binning=ece_binning)  # type: ignore[arg-type]
        metrics["abstention_rate"] = sum(1 for r in records if r.abstained) / n if n else 0.0
    else:
        metrics["ece"] = None
        metrics["ece_note"] = (
            "null: no calibrated probability vector for this variant (only V5 has one)."
        )

    # hallucination -- one NLI pass, both denominators (ADR-029)
    claim_records = [cr for r in records for cr in r.claim_records]
    if not compute_hallucination:
        metrics["hallucination"] = None
        metrics["hallucination_rate"] = None
        metrics["hallucination_pending"] = bool(claim_records)
    elif claim_records:
        metrics["hallucination"] = score_hallucination(claim_records, scorer=nli_scorer)
        metrics["hallucination_rate"] = metrics["hallucination"][
            metrics["hallucination"]["primary"]
        ]
        metrics["hallucination_pending"] = False
    else:
        metrics["hallucination"] = None
        metrics["hallucination_rate"] = None
        metrics["hallucination_pending"] = False

    metrics["invalid_citation_count"] = sum(len(r.invalid_citations) for r in records)
    metrics["grounded_rate"] = (sum(1 for r in records if r.grounded) / n) if n else 0.0
    metrics["status_counts"] = dict(Counter(r.status for r in records))
    metrics["mean_llm_calls"] = (sum(r.n_llm_calls for r in records) / n) if n else 0.0

    # agent failure-mode counters, reusing the Phase 4 aggregator verbatim
    agent_matrices = [r.agent_matrix for r in records if r.agent_matrix is not None]
    if agent_matrices:
        from src.pipeline import _aggregate_agent_metrics

        agg = _aggregate_agent_metrics(agent_matrices)
        for key in (
            "kg_hit_rate",
            "budget_exceeded_rate",
            "conflict_fire_rate",
            "claim_level_conflicts_evaluated",
            "conflict_resolved",
            "conflict_unresolved",
            "conflict_spurious",
            "surviving_invalid_citation_count",
            "every_surviving_cell_has_valid_id",
            "raw_invalid_citation_count",
            "mean_rounds",
        ):
            metrics[f"agent_{key}"] = agg.get(key)

    return metrics


def score_hallucination_into(run_dir: Path, nli_scorer=None) -> dict | None:
    """Reconstruct `ClaimRecord`s from a written run's `predictions.jsonl`, run
    the NLI hallucination pass, and patch `metrics.json` in place (ADR-037).

    Called by `run_eval` *after* `write_run` and *after* the retrieval models
    are released, so the NLI model never coexists with them. Also the entry
    point for `scripts/analysis/rescore_hallucination.py` when a grid cell
    wrote its predictions but the NLI pass OOM'd or was skipped.
    """
    run_dir = Path(run_dir)
    rows = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    claim_records = [
        ClaimRecord(**cr) for row in rows for cr in row.get("claim_records", [])
    ]
    if not claim_records:
        patch_metrics(
            run_dir,
            {"hallucination": None, "hallucination_rate": None, "hallucination_pending": False},
        )
        return None
    hall = score_hallucination(claim_records, scorer=nli_scorer)
    patch_metrics(
        run_dir,
        {
            "hallucination": hall,
            "hallucination_rate": hall[hall["primary"]],
            "hallucination_pending": False,
        },
    )
    return hall


# --------------------------------------------------------------------------
# top-level run
# --------------------------------------------------------------------------


def _contamination(dataset: str, questions, pipeline) -> dict:
    """Executed contamination check (CLAUDE.md §2 -- run in code, not a
    comment). Mirrors `scripts.run_baselines.contamination_assertion`: a real
    MedQA stem-vs-retrieved-chunk n-gram overlap distribution; other datasets
    / a retrieval-free run record `checked: false` honestly."""
    if dataset == "pubmedqa":
        return pubmedqa_contamination_report(CORPUS_PATH)
    if dataset != "medqa" or pipeline is None:
        return {
            "checked": False,
            "applies": True,
            "note": f"no executed contamination check for dataset={dataset!r} "
            f"(pipeline_loaded={pipeline is not None})",
        }
    pairs = []
    for q in questions:
        bundle = pipeline.build_bundle(
            q.question, q.options or {}, dataset=dataset, question_id=q.question_id
        )
        pairs.append((q.question, [e.text for e in bundle.passages]))
    return medqa_overlap_check(pairs)


def run_eval(
    config_path: str,
    *,
    dataset: str,
    split: str,
    limit: int | None,
    variant: str | None = None,
    seed: int | None = None,
    exclude_limit: int | None = None,
    exclude_seed: int | None = None,
    results_dir: Path = RESULTS_DIR,
    nli_scorer=None,
    _pipeline=None,
) -> Path:
    """Run one variant end-to-end and write `results/<run_id>/`. Returns the
    run directory. `_pipeline` is a test seam (inject a fake TriModalPipeline).
    """
    from src.kg.dataset_questions import (
        gold_letter,
        load_question_sample,
        load_question_sample_excluding,
    )
    from src.llm.client import default_tracker

    cfg = load_config(config_path)
    seed = seed if seed is not None else cfg.seed
    variant = variant or resolve_variant(cfg)
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; valid: {sorted(VARIANTS)}")
    spec = VARIANTS[variant]

    if spec.kind == "agents" and not cfg.agents.enabled:
        raise ValueError(f"{variant} needs agents.enabled -- use configs/agents.yaml or v5.yaml")
    if variant == "V5" and not cfg.calibration.enabled:
        raise ValueError("V5 needs calibration.enabled -- use configs/v5.yaml")

    if exclude_limit is not None:
        excl_seed = exclude_seed if exclude_seed is not None else seed
        excluded = load_question_sample(dataset, split, exclude_limit, seed=excl_seed)
        questions = load_question_sample_excluding(
            dataset, split, limit or 0, seed, {q.question_id for q in excluded}
        )
    else:
        questions = load_question_sample(dataset, split, limit, seed=seed)
    print(f"[harness] {variant}: {len(questions)} questions, {dataset}/{split} seed={seed}")

    pipeline = _pipeline
    if pipeline is None and (spec.kind == "agents" or spec.baseline != "B0"):
        from src.pipeline import _load_pipeline

        if not CORPUS_PATH.exists():
            raise FileNotFoundError(f"{CORPUS_PATH} missing -- run scripts.build_corpus")
        print("[harness] loading dense / bm25 / reranker ...", flush=True)
        pipeline = _load_pipeline(cfg, REPO_ROOT)

    default_tracker.reset()
    records: list[QuestionRecord] = []

    if spec.kind == "agents":
        from src.pipeline import consensus_over_questions

        pairs = consensus_over_questions(pipeline, cfg, questions, dataset)
        for q, (matrix, state) in zip(questions, pairs, strict=True):
            if matrix.gold_option is None:
                matrix.gold_option = gold_letter(q)
            records.append(_agent_record(spec, cfg, matrix, state))
    else:
        for i, q in enumerate(questions):
            gold = gold_letter(q)
            records.append(
                _baseline_record(spec, q, gold, pipeline, dataset, default_tracker, seed)
            )
            if (i + 1) % 5 == 0 or (i + 1) == len(questions):
                print(f"[harness] {variant}: {i + 1}/{len(questions)}", flush=True)

    contam = _contamination(dataset, questions, pipeline)  # needs the pipeline

    # Free the retrieval models (dense + bm25 + cross-encoder, ~2 GB) BEFORE
    # the NLI hallucination pass loads DeBERTa (~1.5 GB). Both coexisting OOM'd
    # the 16 GB machine on the Phase 7 grid launch (ADR-037).
    if _pipeline is None:  # only free what we loaded, never a test-injected fake
        pipeline = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass

    metrics = score_run(
        records,
        dataset=dataset,
        nli_scorer=nli_scorer,
        n_ece_bins=cfg.calibration.n_ece_bins,
        ece_binning=cfg.calibration.ece_binning,
        compute_hallucination=False,  # deferred -> after write_run (ADR-037)
    )
    metrics["contamination_assertion"] = contam
    stats = default_tracker.summary()
    metrics["llm_calls_total"] = stats.calls
    metrics["llm_cache_hits"] = stats.cache_hits
    metrics["prompt_tokens_total"] = stats.prompt_tokens
    metrics["completion_tokens_total"] = stats.completion_tokens

    run_id = make_run_id(dataset, variant)
    run_dir = write_run(
        run_id=run_id,
        cfg=cfg,
        dataset=dataset,
        variant=variant,
        split=split,
        limit=limit,
        seed=seed,
        predictions=[r.model_dump(mode="json") for r in records],
        metrics=metrics,
        results_dir=results_dir,
    )
    # predictions.jsonl + non-NLI metrics are now durable. Score hallucination
    # separately; a failure here (OOM, model fetch) leaves the run intact.
    try:
        score_hallucination_into(run_dir, nli_scorer=nli_scorer)
    except Exception as e:  # pragma: no cover - the OOM path
        print(
            f"[harness] hallucination scoring DEFERRED ({type(e).__name__}: {e}).\n"
            f"[harness]   the run is complete and durable; fill it in with:\n"
            f"[harness]   uv run python -m scripts.analysis.rescore_hallucination {run_dir}"
        )

    final = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    _print_summary(variant, final, run_dir)
    return run_dir


def _print_summary(variant: str, metrics: dict, run_dir: Path) -> None:
    print("=" * 70)
    print(f"[harness] {variant} -> {run_dir}")
    print(f"  n={metrics['n_evaluated']}  acc={metrics['accuracy']:.3f}  "
          f"macroF1={metrics['macro_f1']:.3f}")
    print(f"  P@1={metrics['p_at_1']}  P@2={metrics['p_at_2']}  MRR={metrics['mrr']}")
    ece_val = metrics["ece"]["ece"] if isinstance(metrics["ece"], dict) else metrics["ece"]
    print(f"  ECE={ece_val}  ranking_source={metrics['ranking_source']}")
    if metrics.get("hallucination"):
        h = metrics["hallucination"]
        print(
            f"  hallucination_rate (predicted option, PRIMARY)="
            f"{h['hallucination_rate_predicted_option']:.3f} "
            f"(n={h['n_claims_predicted_option']})  |  "
            f"(all SUPPORT cells)={h['hallucination_rate_all_support_cells']:.3f} "
            f"(n={h['n_claims_all_support_cells']})"
        )
    print(f"  invalid_citation_count={metrics['invalid_citation_count']}  "
          f"grounded_rate={metrics['grounded_rate']:.2f}")
    print(f"  status={metrics['status_counts']}")


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default="medqa", choices=["medqa", "medmcqa", "pubmedqa"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--variant", default=None, choices=sorted(VARIANTS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--exclude-limit", type=int, default=None)
    ap.add_argument("--exclude-seed", type=int, default=None)
    args = ap.parse_args()

    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available() = {available}")
    if available:
        print(f"  {torch.cuda.get_device_name(0)} cap {torch.cuda.get_device_capability(0)}")

    run_dir = run_eval(
        args.config,
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        variant=args.variant,
        seed=args.seed,
        exclude_limit=args.exclude_limit,
        exclude_seed=args.exclude_seed,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
