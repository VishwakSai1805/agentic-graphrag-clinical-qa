"""Bound how much reported ungroundedness could be truncation rather than mechanism.

ADR-049. A completion cut at `llm.max_tokens` still parses: these prompts emit
the answer letter before the citation list, so `_parse_answer` returns `ok` and
`_citations_from_regex` recovers nothing from the severed array. The row is
scored `ok` with an EMPTY citation set -- ungrounded under CLAUDE.md section 2,
and invisible in `status_counts`.

So `ok` + zero citations is the observable footprint a truncated answer call
would leave. Counting it per cell puts a CEILING on the truncation exposure:
every such row is a candidate, not a confirmed victim (an empty citation set
also arises when the model simply cites nothing). If the ceiling is small
relative to a cell's ungrounded count, truncation cannot be the explanation for
that cell's grounded_rate and the docs/04 section 5.0a(c) zero-SUPPORT
mechanism stands unchallenged.

Tier matters and the two are NOT comparable:

  retrieval tier (V0-V3, B3) -- `grounded` means >=1 VALID citation, so an
    `ok` row with no citations is ungrounded BY THIS ROUTE. Ceiling is tight.

  agent tier (V4-conflict, V5, V5-refit) -- `grounded` means the predicted
    option has >=1 SUPPORT audit cell (`src/calibration/citations.py: cite()`).
    Groundedness there does not flow through the answer call's citation list at
    all, so an empty-citation `ok` row is NOT automatically an ungrounded row,
    and truncation of an ANSWER call cannot produce an ungrounded row. It could
    only do so by truncating an AUDIT call, which is a different footprint this
    script does not claim to measure.

    uv run python -m scripts.analysis.empty_citation_exposure

Writes results/phase7_grid/empty_citation_exposure.json. Read-only over
predictions; touches no run directory.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "phase7_grid" / "empty_citation_exposure.json"

# The eight cells of the docs/04 section 5.0a trade-off table, pinned by run dir.
# `expect` is that table's (accuracy, grounded_rate) -- asserted below so a
# wrong pin fails loudly instead of quietly measuring the wrong run.
CELLS = [
    ("V0", "medqa_V0_b24015a-dirty_20260911T080323Z", "no-retrieval", 0.582, 0.000),
    ("V1", "medqa_V1_41f7e14_20260908T212655Z", "retrieval", 0.566, 1.000),
    ("V2", "medqa_V2_b24015a-dirty_20260911T104513Z", "retrieval", 0.556, 0.998),
    ("V3", "medqa_V3_41f7e14-dirty_20260908T230633Z", "retrieval", 0.546, 0.984),
    ("B3", "medqa_B3_41f7e14-dirty_20260909T023658Z", "retrieval", 0.538, 0.972),
    ("V4-conflict", "medqa_V4-conflict_41f7e14-dirty_20260909T220306Z", "agent", 0.358, 0.856),
    ("V5", "medqa_V5_41f7e14-dirty_20260909T103452Z", "agent", 0.316, 0.756),
    ("V5-refit", "medqa_V5-refit_b24015a_20260911T074318Z", "agent", 0.292, 0.652),
]


def _rows(run_dir: Path) -> list[dict]:
    with (run_dir / "predictions.jsonl").open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _n_citations(row: dict) -> int:
    """Total citations on a row, across both schemas actually on disk.

    B3 writes `citations` as a flat list (one baseline answer); every other cell
    writes a per-option dict. Normalising here rather than special-casing B3 at
    the call site, so a future flat-list cell counts correctly by default.
    """
    cit = row.get("citations") or {}
    if isinstance(cit, list):
        return len(cit)
    return sum(len(v) for v in cit.values())


def main() -> None:
    cells = []
    for name, run_id, tier, exp_acc, exp_grounded in CELLS:
        run_dir = REPO / "results" / run_id
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]

        acc = round(float(metrics["accuracy"]), 3)
        grounded_rate = round(float(metrics["grounded_rate"]), 3)
        if (acc, grounded_rate) != (exp_acc, exp_grounded):
            raise SystemExit(
                f"{name}: run {run_id} reports acc={acc} grounded={grounded_rate}, "
                f"docs/04 section 5.0a expects {exp_acc}/{exp_grounded} -- wrong run pinned"
            )

        rows = _rows(run_dir)
        n_cit = [_n_citations(r) for r in rows]
        ok = [r["status"] == "ok" for r in rows]
        ungrounded = [not r.get("grounded") for r in rows]

        ok_empty = sum(1 for i in range(len(rows)) if ok[i] and n_cit[i] == 0)
        ok_empty_ungrounded = sum(
            1 for i in range(len(rows)) if ok[i] and n_cit[i] == 0 and ungrounded[i]
        )
        n_ungrounded = sum(ungrounded)

        # Where the ceiling is worth reading at all. V0 retrieves nothing, so all
        # 500 rows are `ok` with no citations BY CONSTRUCTION and the ratio is an
        # artefact, not an exposure. On the agent tier `grounded` is a SUPPORT
        # audit cell, so an answer call's citations cannot make a row ungrounded.
        meaningful = tier == "retrieval"
        why = {
            "no-retrieval": (
                "not meaningful: V0 has no retrieval, so every row is ok with zero "
                "citations by construction (docs/04 section 5.0a trade-off table)"
            ),
            "agent": (
                "not meaningful: grounded = predicted option has >=1 SUPPORT audit "
                "cell, so answer-call truncation cannot produce an ungrounded row; "
                "only audit-call truncation could, which this does not measure"
            ),
            "retrieval": (
                "meaningful: grounded = >=1 valid citation, so an ok row with zero "
                "citations is ungrounded by exactly the route truncation takes"
            ),
        }[tier]

        cells.append(
            {
                "variant": name,
                "run_id": run_id,
                "tier": tier,
                "n": len(rows),
                "accuracy": acc,
                "grounded_rate": grounded_rate,
                "n_ungrounded": n_ungrounded,
                "n_ok": sum(ok),
                "n_ok_empty_citations": ok_empty,
                "n_ok_empty_citations_and_ungrounded": ok_empty_ungrounded,
                # The ceiling only means anything on the retrieval tier, where
                # groundedness flows through the answer call's citation list.
                "truncation_ceiling_frac_of_ungrounded": (
                    round(ok_empty_ungrounded / n_ungrounded, 4) if n_ungrounded else None
                ),
                "ceiling_meaningful": meaningful,
                "ceiling_note": why,
            }
        )

    payload = {
        "adr": "ADR-049",
        "question": (
            "how much of each cell's reported ungroundedness could be answer-call "
            "truncation rather than the docs/04 section 5.0a(c) zero-SUPPORT mechanism?"
        ),
        "observable": "status == 'ok' AND zero citations across all options",
        "interpretation": (
            "a CEILING, not an estimate: every such row is a candidate, none is a "
            "confirmed truncation (a model may simply cite nothing). Meaningful on "
            "the retrieval tier only -- on the agent tier `grounded` is defined by "
            "SUPPORT audit cells, not by the answer call's citations, so answer-call "
            "truncation cannot produce an ungrounded row there."
        ),
        "cells": cells,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    hdr = (
        f"{'variant':<13}{'tier':<13}{'acc':>6}{'grnd':>7}"
        f"{'ungrnd':>8}{'ok+0cit':>9}{'both':>6}{'ceiling':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for c in cells:
        ceil = c["truncation_ceiling_frac_of_ungrounded"]
        if not c["ceiling_meaningful"]:
            shown = "--"
        elif ceil is None:
            shown = "n/a"
        else:
            shown = f"{ceil:.1%}"
        print(
            f"{c['variant']:<13}{c['tier']:<13}{c['accuracy']:>6.3f}{c['grounded_rate']:>7.3f}"
            f"{c['n_ungrounded']:>8}{c['n_ok_empty_citations']:>9}"
            f"{c['n_ok_empty_citations_and_ungrounded']:>6}{shown:>9}"
        )
    print(f"\nwrote {OUT.relative_to(REPO).as_posix()}")


if __name__ == "__main__":
    main()
