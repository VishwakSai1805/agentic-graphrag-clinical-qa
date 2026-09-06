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
import json
import subprocess
from pathlib import Path

import yaml
from pydantic import BaseModel

from src.config import AppConfig, config_hash

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


def git_sha7() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=True,
        )
        return out.stdout.strip() or "nogit"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


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
    config_hash: str
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
        git_sha7=git_sha7(),
        config_hash=config_hash(cfg),
        metrics=metrics,
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8"
    )
    return run_dir
