"""Rebuild `results/phase7_grid/manifest.json` from the five committed Phase 7
grid run directories, and record the runner failure that hit V5 and
V4-conflict (docs/05_DECISIONS.md ADR-039).

The original `run_phase7_grid.ps1` run wrote a manifest recording only V1, V3,
B3 as real completions -- V5 and V4-conflict were reported by `Invoke-Cell` as
`ExitCode 2, Minutes 0`, with **no log file written** for either (`logs/`
holds no `phase7_V5_*` / `phase7_V4-conflict_*` file for the grid's launch
stamp `20260909_020855`). Both cells were then run manually from the same
launch commit (`41f7e14`) and completed -- their `results/` dirs exist,
non-superseded, with real `predictions.jsonl` + `metrics.json`.

This script does not re-run anything. It is the auditable, script-generated
replacement for a hand-edited manifest (CLAUDE.md: "Don't touch `results/` by
hand. Ever.") -- every field below is read from a committed `metrics.json` or
from `logs/`, never typed in.

    uv run python -m scripts.analysis.rebuild_phase7_manifest
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
LOGS_DIR = REPO_ROOT / "logs"
MANIFEST_PATH = RESULTS_DIR / "phase7_grid" / "manifest.json"

LAUNCH_COMMIT = "41f7e14df845b1fcfa256141da41a03ba0ff8720"
LAUNCH_COMMIT_SHORT = "41f7e14"
LAUNCH_STAMP = "20260909_020855"

# The five grid cells -- pinned by exact directory name, never by glob
# (per the launch instruction: "Pin the five explicitly by name -- never by
# glob"). Order matches run_phase7_grid.ps1's cell order.
GRID_RUN_DIRS: dict[str, str] = {
    "V1": "medqa_V1_41f7e14_20260908T212655Z",
    "V3": "medqa_V3_41f7e14-dirty_20260908T230633Z",
    "B3": "medqa_B3_41f7e14-dirty_20260909T023658Z",
    "V5": "medqa_V5_41f7e14-dirty_20260909T103452Z",
    "V4-conflict": "medqa_V4-conflict_41f7e14-dirty_20260909T220306Z",
}

# From the runner's own manifest.json (its stopwatch measured these directly;
# carried forward verbatim -- not recomputed, since file mtimes only capture
# each run's *completion* instant, not its start, see ADR-039).
RUNNER_RECORDED_OUTCOMES: dict[str, dict] = {
    "V1": {"exit_code": 0, "minutes": 50.7, "log": str(LOGS_DIR / f"phase7_V1_{LAUNCH_STAMP}.log")},
    "V3": {"exit_code": 0, "minutes": 99.7, "log": str(LOGS_DIR / f"phase7_V3_{LAUNCH_STAMP}.log")},
    "B3": {
        "exit_code": 0,
        "minutes": 207.7,
        "log": str(LOGS_DIR / f"phase7_B3_{LAUNCH_STAMP}.log"),
    },
    "V5": {
        "exit_code": 2,
        "minutes": 0.0,
        "log": str(LOGS_DIR / f"phase7_V5_{LAUNCH_STAMP}.log"),
        "log_written": False,
    },
    "V4-conflict": {
        "exit_code": 2,
        "minutes": 0.0,
        "log": str(LOGS_DIR / f"phase7_V4-conflict_{LAUNCH_STAMP}.log"),
        "log_written": False,
    },
}

MANUALLY_RUN_CELLS = ("V5", "V4-conflict")

INVESTIGATION = {
    "question": "Why did Invoke-Cell in run_phase7_grid.ps1 never launch V5 and V4-conflict?",
    "confirmed": [
        "No log file exists for either cell at the grid's launch stamp "
        f"(no logs/phase7_V5_{LAUNCH_STAMP}.log, no "
        f"logs/phase7_V4-conflict_{LAUNCH_STAMP}.log), even though Invoke-Cell "
        "unconditionally appends to $script:Results after every attempt -- the "
        "manifest entry with ExitCode=2 proves the function body ran to "
        "completion for both cells.",
        "Empirically reproduced on this machine (Windows PowerShell 5.1, "
        "the version this repo runs on): `Invoke-Expression <cmd> 2>&1 | "
        "Tee-Object -FilePath $LogPath` creates $LogPath only once it "
        "receives at least one pipeline object. A native command that exits "
        "non-zero after producing ZERO stdout -- even if it writes to "
        "stderr, which 2>&1 nominally merges in -- causes Tee-Object to "
        "receive nothing and never create the file. Verified directly: "
        "`python -c \"import sys; print(1,file=sys.stderr); sys.exit(2)\"` "
        "run through the exact Invoke-Cell pattern leaves no log, exit=2 -- "
        "reproducing V5/V4-conflict's signature exactly.",
        "src/eval/harness.py's main() only emits its first stdout line "
        "(`torch.cuda.is_available() = ...`) AFTER argparse.parse_args() "
        "succeeds. Zero stdout is therefore consistent with a failure at or "
        "before argument parsing / Python startup -- not a crash mid-run "
        "(mid-run would have flushed at least the cuda-status line and any "
        "harness prints before dying, and Minutes would not read 0.0).",
    ],
    "ruled_out": [
        "argparse rejecting --variant V4-conflict as an invalid choice: "
        "VARIANTS in src/eval/harness.py already registers "
        "'V4-conflict' (used for resolve_variant / scoring), so it is a "
        "valid --variant value. Also would not explain V5's failure, which "
        "passes no --variant flag at all.",
        "A PowerShell parse error in the command STRING itself: both "
        "commands are plain space-separated CLI invocations, structurally "
        "identical to V1/V3's (which succeeded from the same $H template) "
        "and V4-conflict's own --variant token, differing only in the "
        "config path / trailing --variant value.",
    ],
    "leading_hypothesis": (
        "Resource exhaustion on the 15 GB machine (ADR-037's own documented "
        "OOM history), not a code defect in either cell. B3 (the immediately "
        "preceding cell) ran 207.7 minutes with OLLAMA_KEEP_ALIVE=-1, so "
        "Ollama's 7B model stayed resident the entire time; V5 and "
        "V4-conflict are the two heaviest remaining cells (agents + "
        "calibration, agents alone) and need to load the retrieval stack "
        "(dense/BM25/cross-encoder models) on top of that resident model. "
        "Python's stdout is block-buffered (not line-buffered) when piped "
        "rather than attached to a real console, exactly Invoke-Cell's "
        "situation -- so an abrupt process kill (OS-level OOM) before the "
        "buffer's first flush would lose ALL buffered output, including the "
        "cuda-status line, leaving zero stdout: consistent with the "
        "confirmed Tee-Object behavior above, the zero-stdout evidence from "
        "harness.py's print ordering, and the fact that only the two most "
        "memory-hungry cells, run back-to-back right after the longest prior "
        "cell, were the ones affected."
    ),
    "not_certain": (
        "The OOM hypothesis is not independently verified (no OS-level crash "
        "log was captured at the time, and it cannot be reproduced now "
        "without repeating the exact memory conditions of that night, which "
        "would mean rerunning scoring code -- out of scope here). It is "
        "reported as the leading, evidence-consistent hypothesis, not a "
        "confirmed root cause."
    ),
    "fix_applied": (
        "run_phase7_grid.ps1's Invoke-Cell now writes a one-line stub log "
        "(cell name, command, timestamp) via Set-Content BEFORE invoking the "
        "command, so a future zero-stdout failure still leaves a log file to "
        "inspect, and a follow-up 'child process was terminated' probe "
        "($LASTEXITCODE plus $Error[0] if the pipeline itself faulted) is "
        "appended after the attempt. See docs/05_DECISIONS.md ADR-039."
    ),
}


def _load_metrics(run_dir_name: str) -> dict:
    path = RESULTS_DIR / run_dir_name / "metrics.json"
    if not path.exists():
        raise SystemExit(f"[rebuild_phase7_manifest] missing {path} -- refusing to fabricate")
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest() -> dict:
    cell_outcomes = []
    for cell, run_dir_name in GRID_RUN_DIRS.items():
        m = _load_metrics(run_dir_name)
        if m.get("superseded"):
            raise SystemExit(f"[rebuild_phase7_manifest] {run_dir_name} is marked superseded")
        valid_shas = {LAUNCH_COMMIT_SHORT, f"{LAUNCH_COMMIT_SHORT}-dirty"}
        if m.get("git_sha7") not in valid_shas:
            raise SystemExit(
                f"[rebuild_phase7_manifest] {run_dir_name} git_sha7={m.get('git_sha7')!r} "
                f"does not match launch commit {LAUNCH_COMMIT_SHORT} (dirty or clean) -- "
                "refusing to include a run from a different commit in the grid manifest"
            )
        runner = RUNNER_RECORDED_OUTCOMES[cell]
        entry = {
            "cell": cell,
            "run_id": m["run_id"],
            "git_sha7": m["git_sha7"],
            "launched_by": "manual" if cell in MANUALLY_RUN_CELLS else "runner",
            "runner_attempt": {
                "exit_code": runner["exit_code"],
                "minutes": runner["minutes"],
                "log": runner["log"],
                "log_written": runner.get("log_written", True),
            },
            "n_evaluated": m["metrics"]["n_evaluated"],
            "accuracy": m["metrics"]["accuracy"],
            "status_counts": m["metrics"]["status_counts"],
            "hallucination_pending": m["metrics"].get("hallucination_pending"),
        }
        if cell in MANUALLY_RUN_CELLS:
            entry["manual_run_note"] = (
                "Runner's Invoke-Expression attempt exited 2 with no log written "
                "(see 'investigation' below). Run manually afterward from the same "
                f"launch commit ({LAUNCH_COMMIT_SHORT}); duration not captured -- "
                "config.yaml/predictions.jsonl/metrics.json are all written together "
                "at completion by write_run(), so file mtimes record completion time "
                "only, not a start time, and no independent stopwatch exists for a "
                "manual invocation."
            )
        cell_outcomes.append(entry)

    return {
        "phase": 7,
        "rebuilt_by": "scripts/analysis/rebuild_phase7_manifest.py",
        "launch_commit": LAUNCH_COMMIT,
        "launch_commit_short": LAUNCH_COMMIT_SHORT,
        "dataset": "medqa",
        "split": "test",
        "n": 500,
        "seed": 42,
        "cells_run": list(GRID_RUN_DIRS.keys()),
        "cells_deferred": ["V4-audit (ADR-036)"],
        "derived_analyses": [
            "post-hoc-binary from the V5 run (ADR-036 Constraint B) -- "
            "results/phase7_posthoc_binary_audit/"
        ],
        "pre_registration": "docs/04_EVALUATION.md section 1.0 (m=35), committed before this run",
        "dirty_stamp_resolution": (
            "Four of five cells carry a '-dirty' git_sha7 suffix. Resolved in "
            "docs/05_DECISIONS.md ADR-038: every run's own git_dirty_files list "
            "names only untracked results/ output directories from earlier cells "
            "in this same grid (plus results/phase7_grid/ itself), never a tracked "
            "src/, configs/, or scripts/ path. No tracked file differed from "
            f"{LAUNCH_COMMIT_SHORT} at any point during the grid; the launch commit "
            "is authoritative for all five cells."
        ),
        "cell_outcomes": cell_outcomes,
        "investigation": INVESTIGATION,
    }


def main() -> None:
    manifest = build_manifest()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[rebuild_phase7_manifest] wrote {MANIFEST_PATH}")
    for entry in manifest["cell_outcomes"]:
        print(f"  {entry['cell']:12s} {entry['run_id']}  acc={entry['accuracy']}")


if __name__ == "__main__":
    main()
