"""Control for the Phase 6 hallucination-rate number (ADR-031, docs/04 §5.6).

The V5 dev-20 run scored `hallucination_rate_predicted_option = 0.682` — but
DeBERTa NLI at a 0.5 entailment threshold is strict, and a clinical claim that
is genuinely supported *after one step of inference* scores as not-entailed. So
0.682 may be measuring the metric's strictness rather than the agent layer's
behaviour.

This script runs the **identical** scorer / threshold / claim splitter
(`src.eval.nli.score_hallucination`, `split_claims`, DeBERTa NLI, threshold
0.5) over an equal-budget B2 run's rationales + citations on the SAME dev-20,
and prints both rates side by side.

  * B2 comparably high (~0.6+)  -> the number reflects the NLI criterion; the
    honest reading is "this threshold marks most clinical rationale claims as
    ungrounded" — a metric caveat, not an agent-layer finding.
  * B2 substantially lower      -> the agent layer specifically produces
    citations that do not support its claims — an agent-layer finding.

Usage:
    uv run python -m scripts.analysis.hallucination_control \\
        --b2-run results/medqa_B2eq_<sha>_<ts> \\
        --v5-run results/medqa_V5_<sha>_<ts>

The B2 run must have been produced by the current `scripts/run_baselines.py`
(which records `Prediction.rationale`, added in Phase 6 / ADR-029) — a
pre-Phase-6 B2 run has no rationale field and cannot be scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.config import load_config  # noqa: E402
from src.eval.baselines import _HYBRID_MODALITIES, _apply_equal_budget  # noqa: E402
from src.eval.nli import ClaimRecord, score_hallucination, split_claims  # noqa: E402
from src.kg.dataset_questions import load_question_sample  # noqa: E402
from src.kg.novelty import NLI_MODEL_NAME, get_default_scorer  # noqa: E402
from src.pipeline import _load_pipeline  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
OUT_DIR = REPO_ROOT / "results" / "phase6_hallucination_control"


def _load_predictions(run_dir: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if rows and "rationale" not in rows[0]:
        raise SystemExit(
            f"{run_dir}/predictions.jsonl has no 'rationale' field — it predates "
            "Phase 6. Re-run: scripts.run_baselines --baselines B2 --equal-budget"
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--b2-run", required=True, help="an equal-budget B2 run dir under results/")
    ap.add_argument("--v5-run", required=True, help="the V5 harness run dir to compare against")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    b2_dir = Path(args.b2_run)
    v5_dir = Path(args.v5_run)

    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  {torch.cuda.get_device_name(0)} cap {torch.cuda.get_device_capability(0)}")

    cfg = load_config(str(CONFIG_PATH))
    preds = _load_predictions(b2_dir)
    n = len(preds)
    questions = {
        q.question_id: q for q in load_question_sample("medqa", args.split, n, seed=args.seed)
    }

    print(f"[control] loading pipeline for {n} equal-budget bundles ...", flush=True)
    pipe = _load_pipeline(cfg, REPO_ROOT)
    kg_rank_fn = pipe.reranker.rerank if cfg.agents.kg_slot_rerank else None

    records: list[ClaimRecord] = []
    n_claims_by_q: dict[str, int] = {}
    for row in preds:
        qid = row["question_id"]
        q = questions[qid]
        bundle = pipe.build_bundle(
            q.question, q.options or {}, dataset="medqa", question_id=qid,
            modalities=_HYBRID_MODALITIES,
        )
        bundle = _apply_equal_budget(bundle, cfg.agents, kg_rank_fn)
        id2text = {e.id: e.text for e in bundle.evidence()}
        cited_texts = [id2text[c] for c in row.get("citations", []) if c in id2text]
        claims = split_claims(row.get("rationale", ""))
        n_claims_by_q[qid] = len(claims)
        for claim in claims:
            records.append(
                ClaimRecord(
                    question_id=qid,
                    option_id=row.get("pred_option"),
                    claim=claim,
                    cited_evidence_texts=cited_texts,
                    predicted_option=True,
                )
            )

    print(f"[control] scoring {len(records)} B2 claims with {NLI_MODEL_NAME} ...", flush=True)
    b2 = score_hallucination(records, scorer=get_default_scorer(), threshold=0.5)

    v5_metrics = json.loads((v5_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    v5 = v5_metrics["hallucination"]

    def _line(tag: str, h: dict) -> str:
        return (
            f"  {tag:<22} predicted-option {h['hallucination_rate_predicted_option']:.3f} "
            f"(n={h['n_claims_predicted_option']}, uncited={h['n_uncited_predicted_option']})"
        )

    print("\n" + "=" * 72)
    print("HALLUCINATION-RATE CONTROL — same scorer, threshold 0.5, same split_claims")
    print("=" * 72)
    print(_line("B2 equal-budget", b2))
    print(_line("V5 (agents+calib)", v5))
    delta = (
        v5["hallucination_rate_predicted_option"] - b2["hallucination_rate_predicted_option"]
    )
    print(f"\n  V5 − B2 = {delta:+.3f}")
    verdict = (
        "METRIC-STRICTNESS: B2 is comparably high — the 0.5 NLI threshold marks most "
        "clinical rationale claims ungrounded regardless of pipeline. State as a caveat."
        if b2["hallucination_rate_predicted_option"] >= 0.55
        else "AGENT-LAYER: B2 is substantially lower — the agent layer specifically "
        "produces citations that don't support its claims. Worth a section."
    )
    print(f"\n  Reading: {verdict}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "control.json"
    out_path.write_text(
        json.dumps(
            {
                "scorer": NLI_MODEL_NAME,
                "threshold": 0.5,
                "claim_splitter": "src.eval.nli.split_claims (regex, decimals+abbrev-guarded)",
                "split": args.split,
                "seed": args.seed,
                "n_questions": n,
                "b2_run": b2_dir.name,
                "v5_run": v5_dir.name,
                "b2_equal_budget": b2,
                "v5_from_metrics_json": v5,
                "v5_minus_b2_predicted_option": delta,
                "reading": verdict,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\n[control] wrote {out_path}")


if __name__ == "__main__":
    main()
