"""Run-provenance writer tests (AUDIT 2026-09-06 §6.1, §6.2).

The audit found that `git_sha7()` was a bare `git rev-parse HEAD` with no
dirty check, so every committed run recorded a SHA whose tree does not contain
the code that produced it. These tests pin the corrected behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import load_config
from src.eval import results as R

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


@pytest.fixture
def cfg():
    return load_config(CONFIG_PATH)


def _fake_git(mapping):
    """Return a _git stand-in driven by a {args_tuple: stdout} mapping."""

    def _g(*args):
        return mapping.get(args)

    return _g


def test_clean_tree_records_bare_sha_and_no_diff_hash(monkeypatch):
    monkeypatch.setattr(
        R,
        "_git",
        _fake_git(
            {
                ("rev-parse", "--short=7", "HEAD"): "abc1234",
                ("status", "--porcelain"): "",
            }
        ),
    )
    prov = R.git_provenance()
    assert prov["git_sha7"] == "abc1234"
    assert prov["git_dirty"] is False
    assert prov["git_diff_sha256"] is None
    assert prov["git_dirty_files"] == []


def test_dirty_tree_suffixes_the_sha_and_hashes_the_diff(monkeypatch):
    monkeypatch.setattr(
        R,
        "_git",
        _fake_git(
            {
                ("rev-parse", "--short=7", "HEAD"): "abc1234\n",
                ("status", "--porcelain"): " M src/pipeline.py\n?? new.py\n",
                ("diff", "HEAD"): "diff --git a/src/pipeline.py\n+x\n",
                ("ls-files", "--others", "--exclude-standard"): "new.py\n",
            }
        ),
    )
    prov = R.git_provenance()
    assert prov["git_sha7"] == "abc1234-dirty"
    assert prov["git_dirty"] is True
    assert len(prov["git_diff_sha256"]) == 64
    assert prov["git_dirty_files"] == ["new.py", "src/pipeline.py"]


def test_same_dirty_tree_hashes_identically(monkeypatch):
    """The diff hash must be stable, so a re-run of the same dirty tree is
    at least identifiable as the same tree."""
    m = {
        ("rev-parse", "--short=7", "HEAD"): "abc1234\n",
        ("status", "--porcelain"): " M a.py\n",
        ("diff", "HEAD"): "some diff",
        ("ls-files", "--others", "--exclude-standard"): "",
    }
    monkeypatch.setattr(R, "_git", _fake_git(m))
    first = R.git_provenance()["git_diff_sha256"]
    monkeypatch.setattr(R, "_git", _fake_git(m))
    assert R.git_provenance()["git_diff_sha256"] == first


def test_different_dirty_content_hashes_differently(monkeypatch):
    base = {
        ("rev-parse", "--short=7", "HEAD"): "abc1234\n",
        ("status", "--porcelain"): " M a.py\n",
        ("ls-files", "--others", "--exclude-standard"): "",
    }
    monkeypatch.setattr(R, "_git", _fake_git({**base, ("diff", "HEAD"): "diff one"}))
    a = R.git_provenance()["git_diff_sha256"]
    monkeypatch.setattr(R, "_git", _fake_git({**base, ("diff", "HEAD"): "diff two"}))
    assert R.git_provenance()["git_diff_sha256"] != a


def test_no_git_available_degrades_to_nogit_without_raising(monkeypatch):
    monkeypatch.setattr(R, "_git", _fake_git({}))
    prov = R.git_provenance()
    assert prov["git_sha7"] == "nogit"
    assert prov["git_dirty"] is None


def test_git_sha7_is_the_provenance_sha(monkeypatch):
    monkeypatch.setattr(
        R,
        "_git",
        _fake_git(
            {
                ("rev-parse", "--short=7", "HEAD"): "abc1234\n",
                ("status", "--porcelain"): " M a.py\n",
                ("diff", "HEAD"): "d",
                ("ls-files", "--others", "--exclude-standard"): "",
            }
        ),
    )
    assert R.git_sha7() == "abc1234-dirty"


def test_write_run_persists_provenance_into_metrics_json(tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(
        R,
        "_git",
        _fake_git(
            {
                ("rev-parse", "--short=7", "HEAD"): "abc1234\n",
                ("status", "--porcelain"): " M src/pipeline.py\n",
                ("diff", "HEAD"): "a diff",
                ("ls-files", "--others", "--exclude-standard"): "",
            }
        ),
    )
    run_dir = R.write_run(
        run_id="medqa_TEST_abc1234-dirty_20260907T000000Z",
        cfg=cfg,
        dataset="medqa",
        variant="TEST",
        split="train",
        limit=1,
        seed=42,
        predictions=[{"question_id": "train_1"}],
        metrics={"n_evaluated": 1},
        results_dir=tmp_path,
    )
    doc = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert doc["git_sha7"] == "abc1234-dirty"
    assert doc["git_dirty"] is True
    assert doc["git_diff_sha256"]
    assert doc["git_dirty_files"] == ["src/pipeline.py"]


def test_new_runs_default_to_not_superseded(tmp_path, cfg):
    run_dir = R.write_run(
        run_id="medqa_TEST_x_20260907T000000Z",
        cfg=cfg,
        dataset="medqa",
        variant="TEST",
        split="train",
        limit=1,
        seed=42,
        predictions=[],
        metrics={"n_evaluated": 0},
        results_dir=tmp_path,
    )
    doc = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert doc["superseded"] is False
    assert doc["superseded_reason"] is None


def test_real_repo_provenance_is_self_consistent():
    """Against the actual repo: if git reports a dirty tree, the recorded SHA
    must say so."""
    prov = R.git_provenance()
    if prov["git_dirty"] is True:
        assert prov["git_sha7"].endswith("-dirty")
        assert prov["git_diff_sha256"]
    elif prov["git_dirty"] is False:
        assert not prov["git_sha7"].endswith("-dirty")
