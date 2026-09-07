"""Phase 5 exit-test baseline: real per-option probabilities read directly off
the LLM's own next-token logprobs, for the ECE comparison
`docs/02_ROADMAP.md`'s Phase 5 exit test requires ("ECE on dev improves vs.
raw LLM option log-probs" -- also `docs/00_RESEARCH_SPEC.md` H3).

Same evidence access as the V4 agent pipeline (the full tri-modal
EvidenceBundle for the same question), so the comparison isolates the
calibration MECHANISM's contribution rather than confounding it with an
information-access difference. The model is asked for a single-letter answer
(so the option letter IS the first generated token) with `logprobs=True`; the
raw probability for each option is `softmax(1.0)` over just that option
letter's logprob among the returned `top_logprobs` alternatives -- no
weighting, no temperature scaling, exactly "read the model's own confidence
off its output distribution."

Ollama's native `ollama/` LiteLLM provider path rejects `logprobs` outright;
`src.llm.client.chat(logprobs=True)` routes this through the OpenAI-compatible
passthrough instead (verified empirically -- see that module).

Usage:
    uv run python -m scripts.raw_logprob_baseline --dataset medqa --split train \\
        --limit 120 --exclude-limit 20 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402
from src.eval.baselines import _format_evidence, _resolve_options  # noqa: E402
from src.eval.metrics import ece  # noqa: E402
from src.kg.dataset_questions import (  # noqa: E402
    gold_letter,
    load_question_sample,
    load_question_sample_excluding,
)
from src.llm.client import StatsTracker, chat  # noqa: E402
from src.pipeline import _load_pipeline  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "agents.yaml"
RESULTS_DIR = REPO_ROOT / "results" / "phase5_calibration"


def print_gpu_status() -> None:
    import torch

    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available() = {available}")
    if available:
        print(f"torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
        print(f"torch.cuda.get_device_capability(0) = {torch.cuda.get_device_capability(0)}")

_LETTER_PROMPT = (
    "{question}\n\nOptions:\n{options}\n\n"
    "Evidence:\n{evidence}\n\n"
    "Using the evidence above, choose the single best option. Respond with "
    "ONLY the letter of your choice (e.g. A). No punctuation, no other text."
)
_TOP_LOGPROBS = 12
_SENTINEL_MARGIN = 5.0  # logprob penalty below the worst OBSERVED alternative
_MIN_TOKENS_NEEDED = 2  # 1 for the letter, 1 headroom for a stray leading space


def _raw_probs_for_question(
    question, options: dict[str, str], evidence_text: str, seed: int, tracker: StatsTracker
) -> dict:
    """One LLM call with real logprobs; returns {letter: raw_probability} over
    exactly `options`'s letters plus the diagnostic fields recorded per
    question."""
    prompt = _LETTER_PROMPT.format(
        question=question.question,
        options=_format_options_sorted(options),
        evidence=evidence_text,
    )
    resp = chat(
        prompt,
        temperature=0,
        seed=seed,
        max_tokens=_MIN_TOKENS_NEEDED,
        logprobs=True,
        top_logprobs=_TOP_LOGPROBS,
        tracker=tracker,
    )
    letters = sorted(options)
    lp_by_token: dict[str, float] = {}
    if resp.logprobs:
        for alt in resp.logprobs[0]["top_logprobs"]:
            token = alt["token"].strip().upper()
            if token and (token not in lp_by_token or alt["logprob"] > lp_by_token[token]):
                lp_by_token[token] = alt["logprob"]

    observed = [lp_by_token[letter] for letter in letters if letter in lp_by_token]
    n_matched = len(observed)
    sentinel = (min(observed) - _SENTINEL_MARGIN) if observed else -50.0
    logprobs_by_letter = {letter: lp_by_token.get(letter, sentinel) for letter in letters}

    m = max(logprobs_by_letter.values())
    exps = {letter: math.exp(lp - m) for letter, lp in logprobs_by_letter.items()}
    total = sum(exps.values())
    raw_probs = {letter: v / total for letter, v in exps.items()}

    return {
        "raw_probs": raw_probs,
        "n_letters_matched_in_top_logprobs": n_matched,
        "n_letters_total": len(letters),
        "raw_text": resp.text,
    }


def _format_options_sorted(options: dict[str, str]) -> str:
    return "\n".join(f"{k}. {options[k]}" for k in sorted(options))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="medqa", choices=["medqa"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--exclude-limit", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print_gpu_status()
    cfg = load_config(args.config)

    excluded = load_question_sample(args.dataset, args.split, args.exclude_limit, seed=args.seed)
    exclude_ids = {q.question_id for q in excluded}
    questions = load_question_sample_excluding(
        args.dataset, args.split, args.limit, args.seed, exclude_ids
    )
    questions = sorted(questions, key=lambda q: q.question_id)
    print(
        f"[raw_logprob_baseline] {len(questions)} questions, {args.dataset}/{args.split} "
        f"(excluding {len(exclude_ids)} dev questions), seed={args.seed}"
    )

    pipe = _load_pipeline(cfg, REPO_ROOT)
    tracker = StatsTracker()

    per_question = []
    for i, q in enumerate(questions):
        options = _resolve_options(q)
        bundle = pipe.build_bundle(
            q.question, options, dataset=args.dataset, question_id=q.question_id
        )
        evidence_text = _format_evidence(bundle.evidence())
        result = _raw_probs_for_question(q, options, evidence_text, args.seed, tracker)
        gold = gold_letter(q)
        per_question.append(
            {
                "question_id": q.question_id,
                "gold_letter": gold,
                "letters": sorted(options),
                **result,
            }
        )
        if (i + 1) % 10 == 0 or (i + 1) == len(questions):
            print(f"[raw_logprob_baseline] {i + 1}/{len(questions)}", flush=True)

    n = len(per_question)
    # first half = fit set (matches scripts/fit_calibration.py's split), second = held-out
    mid = n // 2
    subsets = {"fit": per_question[:mid], "held_out": per_question[mid:]}
    ece_by_subset = {}
    for name, rows in subsets.items():
        rows_with_gold = [r for r in rows if r["gold_letter"] in r["letters"]]
        probs = [[r["raw_probs"][letter] for letter in r["letters"]] for r in rows_with_gold]
        gold_idx = [r["letters"].index(r["gold_letter"]) for r in rows_with_gold]
        ece_by_subset[name] = ece(probs, gold_idx, n_bins=5, binning="equal_mass")

    out = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "n_questions": n,
        "n_excluded_dev": len(exclude_ids),
        "llm_calls_total": tracker.summary().calls,
        "llm_cache_hits": tracker.summary().cache_hits,
        "per_question": per_question,
        "ece": ece_by_subset,
        "note": (
            "Raw per-option probability = softmax(1.0) over the first generated "
            "token's logprob for each option letter (from real top_logprobs, "
            "src.llm.client.chat(logprobs=True)), given the SAME EvidenceBundle "
            "the V4 agent pipeline saw for this question. No agent reasoning, no "
            "calibration weights, no temperature scaling -- this is the "
            "uncalibrated comparator docs/02_ROADMAP.md's Phase 5 exit test names."
        ),
    }
    default_out = RESULTS_DIR / f"{args.dataset}_raw_logprob_baseline.json"
    out_path = Path(args.out) if args.out else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 70)
    for name, e in ece_by_subset.items():
        print(f"[raw_logprob_baseline] {name} ECE (n={e['n']}, {e['binning']}): {e['ece']}")
    print(f"[raw_logprob_baseline] wrote {out_path}")


if __name__ == "__main__":
    main()
