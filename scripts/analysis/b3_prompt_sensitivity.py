"""Pre-Phase-7 fix (AUDIT 2026-09-06 §4.1): B3 verifier prompt-sensitivity.

`docs/04_EVALUATION.md` §5.4 reports `verifier_decision_rate = 0.05` for the
binary verifier B3 on the MedQA dev-20 — the project's designated motivating
result for C2's "binary bottleneck". §5.4 already shows that rate is invariant
across a context-window change that inverts the failure direction, so the
remaining objection is **strawman-by-prompt-wording**: the number rests on one
verifier phrasing.

This runs the same B3 verify loop (build hybrid bundle → ask SUPPORTED /
NOT_SUPPORTED per option → decide iff exactly one is SUPPORTED) on the same
dev-20, with **three** verifier phrasings — the shipped strict one, a neutral
one, and a permissive one — and reports `verifier_decision_rate` and the
`n_supported` distribution for each. No production code changes: the verify
loop is re-implemented here over `src.eval.baselines` helpers.

    uv run python -m scripts.analysis.b3_prompt_sensitivity

Matches the reported §5.4 B3 run (`medqa_B3_d5e3af5_20260906T102134Z`):
default (non-equal-budget) bundle, `llm.num_ctx=8192`, temperature 0, seed 42.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.config import load_config  # noqa: E402
from src.eval.baselines import (  # noqa: E402
    _CITE_INSTRUCTION,
    _HYBRID_MODALITIES,
    _SYSTEM,
    _format_evidence,
    _parse_verdict,
    _resolve_options,
)
from src.kg.dataset_questions import load_question_sample  # noqa: E402
from src.llm.client import StatsTracker, chat  # noqa: E402
from src.pipeline import TriModalPipeline  # noqa: E402
from src.retrieval.dense import DenseIndex  # noqa: E402
from src.retrieval.lexical import BM25Index  # noqa: E402
from src.retrieval.rerank import CrossEncoderReranker  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
CORPUS_PATH = REPO_ROOT / "data" / "processed" / "corpus.jsonl"
OUT_DIR = REPO_ROOT / "results" / "phase3_b3_prompt_sensitivity"

_JSON_TAIL = (
    _CITE_INSTRUCTION
    + ' STRICT JSON: {{"verdict": "SUPPORTED" | "NOT_SUPPORTED", "evidence_id": "<id or null>"}}'
)

# key -> (label, template). {question} {letter} {text} {evidence} are filled.
PROMPTS: dict[str, tuple[str, str]] = {
    "strict_shipped": (
        "strict (the shipped _VERIFY_PROMPT)",
        "{question}\n\nGiven ONLY the evidence below, is option {letter} ({text}) the "
        "correct answer? Say SUPPORTED only if the evidence positively supports it, "
        "otherwise NOT_SUPPORTED.\n\nEvidence:\n{evidence}\n\n" + _JSON_TAIL,
    ),
    "neutral": (
        "neutral (no 'only if positively')",
        "{question}\n\nBased on the evidence below, is option {letter} ({text}) "
        "supported as the correct answer? Answer SUPPORTED or NOT_SUPPORTED.\n\n"
        "Evidence:\n{evidence}\n\n" + _JSON_TAIL,
    ),
    "permissive": (
        "permissive ('consistent with')",
        "{question}\n\nGiven the evidence below, could option {letter} ({text}) be the "
        "correct answer? Answer SUPPORTED if the evidence is consistent with it, "
        "otherwise NOT_SUPPORTED.\n\nEvidence:\n{evidence}\n\n" + _JSON_TAIL,
    ),
}


def main() -> None:
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  {torch.cuda.get_device_name(0)} cap {torch.cuda.get_device_capability(0)}")

    cfg = load_config(str(CONFIG_PATH))
    questions = load_question_sample("medqa", "train", 20, seed=42)
    print(f"[b3_sens] {len(questions)} MedQA dev-20 questions, seed 42")

    pipe = TriModalPipeline(
        cfg,
        dense=DenseIndex.load(REPO_ROOT / "indexes" / "dense"),
        bm25=BM25Index.load(REPO_ROOT / "indexes" / "bm25"),
        reranker=CrossEncoderReranker(max_length=cfg.retrieval.rerank.max_length),
        corpus_path=CORPUS_PATH,
    )

    # build every bundle once; all three prompts verify against the same evidence
    bundles = {}
    for q in questions:
        b = pipe.build_bundle(
            q.question, q.options or {}, dataset="medqa", question_id=q.question_id,
            modalities=_HYBRID_MODALITIES,
        )
        bundles[q.question_id] = (b, _format_evidence(b.evidence()))

    results: dict[str, dict] = {}
    for key, (label, template) in PROMPTS.items():
        tracker = StatsTracker()
        n_supported_per_q: list[int] = []
        for q in questions:
            options = _resolve_options(q)
            _bundle, evidence_block = bundles[q.question_id]
            n_supported = 0
            for letter in sorted(options):
                prompt = template.format(
                    question=q.question, letter=letter, text=options[letter],
                    evidence=evidence_block,
                )
                resp = chat(prompt, system=_SYSTEM, temperature=0, seed=42, tracker=tracker)
                verdict, _ev = _parse_verdict(resp.text)
                if verdict == "SUPPORTED":
                    n_supported += 1
            n_supported_per_q.append(n_supported)
        decided = sum(1 for n in n_supported_per_q if n == 1)
        dist = dict(sorted(Counter(n_supported_per_q).items()))
        results[key] = {
            "label": label,
            "verifier_decision_rate": decided / len(questions),
            "n_decided_exactly_one": decided,
            "n_supported_distribution": {str(k): v for k, v in dist.items()},
            "llm_calls": tracker.summary().calls,
        }
        print(
            f"  {label:<40} verifier_decision_rate={results[key]['verifier_decision_rate']:.2f} "
            f"n_supported={results[key]['n_supported_distribution']}"
        )

    rates = {k: v["verifier_decision_rate"] for k, v in results.items()}
    spread = max(rates.values()) - min(rates.values())
    print("\n" + "=" * 70)
    print(f"verifier_decision_rate across 3 phrasings: {rates}")
    print(f"spread = {spread:.2f}")
    print(
        "READING: "
        + (
            "invariant to wording — the §5.4 binary-bottleneck result is robust to "
            "prompt phrasing as well as to the context window."
            if spread <= 0.10
            else "sensitive to wording — §5.4's rate must be reported with this range, "
            "not as a single number."
        )
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "dataset": "medqa",
                "split": "train",
                "n_questions": len(questions),
                "seed": 42,
                "reported_comparator_run": "medqa_B3_d5e3af5_20260906T102134Z",
                "prompts": results,
                "verifier_decision_rate_by_prompt": rates,
                "spread": spread,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\n[b3_sens] wrote {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
