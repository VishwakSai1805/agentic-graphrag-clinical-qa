"""Reproducible run-output writer (CLAUDE.md "Every eval run is reproducible").

`run_id = <dataset>_<variant>_<git_sha7>_<utc_ts>`; each run writes
`results/<run_id>/{config.yaml,predictions.jsonl,metrics.json}`. Only
`config.yaml` and `metrics.json` are committed -- `predictions.jsonl` is
gitignored (`.gitignore`: `results/**/predictions.jsonl`).

This is the minimal Phase 3 writer. The Phase 6 harness
(`src/eval/harness.py`) adds resumability, caching and the parallel
semaphore on top of this same directory layout -- it should extend, not
replace, this module.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
from pathlib import Path

import yaml
from pydantic import BaseModel

from src.config import AppConfig, config_hash

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


def _git(*args: str) -> str | None:
    """Run a git command in REPO_ROOT, returning stdout or None if git is
    unavailable / the command fails. Never raises: a missing git must degrade
    to an honest "nogit" marker, not crash a finished eval before it writes."""
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            # Decode as UTF-8 explicitly: `text=True` uses the locale codec
            # (cp1252 on Windows), which raises UnicodeDecodeError in a reader
            # thread the moment a tracked file's diff contains a non-latin-1
            # byte (docs/*.md here are full of "≈", "−", en-dashes). That
            # corrupted `git_diff_sha256` for any run made from a dirty tree
            # with non-ASCII changes -- exactly the provenance this records.
            encoding="utf-8",
            errors="replace",
            cwd=REPO_ROOT,
            check=True,
        )
        return out.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_dirty_diff() -> str | None:
    """The full uncommitted diff (tracked changes + the list of untracked
    files), or None when git is unavailable. Empty string means a clean tree."""
    diff = _git("diff", "HEAD")
    if diff is None:
        return None
    untracked = _git("ls-files", "--others", "--exclude-standard") or ""
    return diff + untracked


def git_provenance() -> dict:
    """Full, honest provenance for a run.

    AUDIT 2026-09-06 §6.1: `git_sha7()` used to be a bare `git rev-parse
    --short=7 HEAD` with no dirty check. Every Phase 3/4 run was made from an
    uncommitted worktree, so the recorded SHA points at a tree that does NOT
    contain the code that produced the run -- e.g. `src/eval/baselines.py` does
    not exist at the `d5e3af5` stamped into the eight `results/medqa_B*` runs.
    Nobody can check out a recorded SHA and reproduce the numbers.

    From now on a run made from a dirty worktree is stamped `<sha7>-dirty` and
    carries `git_diff_sha256` (a hash of the exact uncommitted diff) plus the
    changed-file list, so the run is at minimum *identifiable* -- and re-running
    the same dirty tree reproduces the same diff hash.
    """
    sha = (_git("rev-parse", "--short=7", "HEAD") or "").strip()
    if not sha:
        return {
            "git_sha7": "nogit",
            "git_dirty": None,
            "git_diff_sha256": None,
            "git_dirty_files": [],
        }
    status = _git("status", "--porcelain")
    if status is None:
        return {
            "git_sha7": sha,
            "git_dirty": None,
            "git_diff_sha256": None,
            "git_dirty_files": [],
        }
    dirty_files = sorted(line[3:].strip() for line in status.splitlines() if line.strip())
    if not dirty_files:
        return {
            "git_sha7": sha,
            "git_dirty": False,
            "git_diff_sha256": None,
            "git_dirty_files": [],
        }
    diff = git_dirty_diff() or ""
    return {
        "git_sha7": f"{sha}-dirty",
        "git_dirty": True,
        "git_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "git_dirty_files": dirty_files,
    }


def git_sha7() -> str:
    """The run-id SHA component, suffixed `-dirty` when the worktree that
    produced the run has uncommitted changes (AUDIT 2026-09-06 §6.1)."""
    return git_provenance()["git_sha7"]


def utc_ts() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def make_run_id(
    dataset: str, variant: str, *, sha7: str | None = None, ts: str | None = None
) -> str:
    return f"{dataset}_{variant}_{sha7 or git_sha7()}_{ts or utc_ts()}"


class RunResult(BaseModel):
    run_id: str
    dataset: str
    variant: str
    split: str
    limit: int | None
    seed: int
    git_sha7: str
    # AUDIT 2026-09-06 §6.1: a run from a dirty worktree is not reproducible
    # from its SHA alone, so the SHA is never recorded on its own again.
    git_dirty: bool | None = None
    git_diff_sha256: str | None = None
    git_dirty_files: list[str] = []
    config_hash: str
    # AUDIT 2026-09-06 §6.2: a run that has been invalidated by a later fix must
    # say so IN ITS OWN metrics.json, so that a stranger -- or a report script --
    # reading `results/` cannot tabulate it by mistake. Stamped by
    # `scripts/mark_superseded.py`, never by hand.
    superseded: bool = False
    superseded_reason: str | None = None
    superseded_by: str | None = None
    metrics: dict


def write_run(
    *,
    run_id: str,
    cfg: AppConfig,
    dataset: str,
    variant: str,
    split: str,
    limit: int | None,
    seed: int,
    predictions: list[dict],
    metrics: dict,
    results_dir: Path = RESULTS_DIR,
) -> Path:
    """Writes the three run files and returns the run directory. `metrics` is
    written verbatim (the caller computes it); `n_evaluated` must be present
    in it (eval-scientist convention -- every metric reported with its n).
    """
    if "n_evaluated" not in metrics:
        raise ValueError("metrics must include 'n_evaluated' (report every metric with its n)")

    run_dir = Path(results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    result = RunResult(
        run_id=run_id,
        dataset=dataset,
        variant=variant,
        split=split,
        limit=limit,
        seed=seed,
        **git_provenance(),
        config_hash=config_hash(cfg),
        metrics=metrics,
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8"
    )
    return run_dir


def patch_metrics(run_dir: Path, updates: dict) -> None:
    """Merge `updates` into an already-written run's `metrics.json` -> `metrics`
    block, leaving everything else (provenance, config_hash, superseded flags)
    untouched. Used to fill in a metric that was deferred at write time -- e.g.
    the NLI hallucination block, computed after the retrieval models are freed
    so the two never coexist in RAM (ADR-037). Rewrites the file in place with
    the same formatting `write_run` uses."""
    path = Path(run_dir) / "metrics.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["metrics"] = {**doc.get("metrics", {}), **updates}
    path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
