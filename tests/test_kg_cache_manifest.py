"""Tests for scripts/kg_cache_manifest.py -- the aggregate checksum logic
that lets a gitignored KG cache tree be verified against data/MANIFEST.json.
"""

from __future__ import annotations

import json

from scripts.kg_cache_manifest import aggregate_sha256


def _write(root, rel, obj):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def test_empty_tree(tmp_path):
    digest, n = aggregate_sha256(tmp_path)
    assert n == 0
    assert len(digest) == 64


def test_stable_and_content_sensitive(tmp_path):
    _write(tmp_path, "medqa/train_00000.json", {"a": 1})
    _write(tmp_path, "medqa/train_00001.json", {"a": 2})
    d1, n1 = aggregate_sha256(tmp_path)
    assert n1 == 2
    # recompute -> identical
    d2, _ = aggregate_sha256(tmp_path)
    assert d1 == d2
    # change one file -> digest changes
    _write(tmp_path, "medqa/train_00001.json", {"a": 3})
    d3, _ = aggregate_sha256(tmp_path)
    assert d3 != d1


def test_rename_sensitive(tmp_path):
    _write(tmp_path, "medqa/a.json", {"x": 1})
    d1, _ = aggregate_sha256(tmp_path)
    (tmp_path / "medqa" / "a.json").rename(tmp_path / "medqa" / "b.json")
    d2, _ = aggregate_sha256(tmp_path)
    assert d1 != d2


def test_order_independent_of_creation(tmp_path, tmp_path_factory):
    _write(tmp_path, "d/z.json", {"v": 1})
    _write(tmp_path, "d/a.json", {"v": 2})
    other = tmp_path_factory.mktemp("other")
    _write(other, "d/a.json", {"v": 2})
    _write(other, "d/z.json", {"v": 1})
    assert aggregate_sha256(tmp_path)[0] == aggregate_sha256(other)[0]
