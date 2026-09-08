"""Fill in a run's deferred NLI hallucination block (ADR-037).

The Phase 7 harness writes `predictions.jsonl` + all non-NLI metrics FIRST,
then scores hallucination separately so the DeBERTa NLI model (~1.5 GB) never
coexists in RAM with the retrieval models (a real OOM on the 16 GB machine).
If that second pass is skipped or OOMs, the run dir is still complete and
carries `metrics.hallucination_pending: true`.

This runs the pass standalone, from the already-written `predictions.jsonl`:

    uv run python -m scripts.analysis.rescore_hallucination RUN_DIR [RUN_DIR ...]

    # e.g. results/medqa_V5_<sha>_<ts>   results/medqa_V1_<sha>_<ts>

No LLM calls, no retrieval models -- only the NLI model. Idempotent: re-running
on a run that already has its hallucination block recomputes and rewrites it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.harness import score_hallucination_into  # noqa: E402


def main() -> None:
    run_dirs = [Path(a) for a in sys.argv[1:]]
    if not run_dirs:
        print(__doc__)
        raise SystemExit(2)

    for run_dir in run_dirs:
        mpath = run_dir / "metrics.json"
        if not mpath.exists():
            print(f"[rescore] SKIP {run_dir}: no metrics.json", file=sys.stderr)
            continue
        pending = json.loads(mpath.read_text(encoding="utf-8"))["metrics"].get(
            "hallucination_pending"
        )
        print(f"[rescore] {run_dir}  (hallucination_pending={pending})")
        hall = score_hallucination_into(run_dir)
        if hall is None:
            print("[rescore]   no citable claims -> hallucination stays null")
        else:
            print(
                f"[rescore]   predicted-option rate "
                f"{hall['hallucination_rate_predicted_option']:.3f} "
                f"(n={hall['n_claims_predicted_option']})  |  all-SUPPORT "
                f"{hall['hallucination_rate_all_support_cells']:.3f} "
                f"(n={hall['n_claims_all_support_cells']})"
            )
        print(f"[rescore]   patched {mpath}")


if __name__ == "__main__":
    main()
