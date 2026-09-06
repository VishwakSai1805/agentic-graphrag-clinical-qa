"""Tests for src/llm/prompts.py + the versioned prompt files exist and format."""

from __future__ import annotations

import pytest

from src.llm.prompts import load_prompt, prompt_sha


@pytest.mark.parametrize("name", ["hypothesis", "audit", "conflict"])
def test_v1_prompt_loads(name):
    text = load_prompt(name, "v1")
    assert len(text) > 100
    assert "STRICT JSON" in text


def test_missing_prompt_raises_with_listing():
    with pytest.raises(FileNotFoundError):
        load_prompt("hypothesis", "v999")


def test_prompt_sha_stable_and_short():
    a = prompt_sha("audit", "v1")
    assert len(a) == 12
    assert a == prompt_sha("audit", "v1")


def test_hypothesis_prompt_formats():
    out = load_prompt("hypothesis", "v1").format(question="Q?", options="A. x\nB. y")
    assert "Q?" in out and "A. x" in out
    assert "{question}" not in out and "{options}" not in out


def test_audit_prompt_formats():
    out = load_prompt("audit", "v1").format(
        option_id="A", option_text="opt", claims="0: c", evidence="[id] text"
    )
    assert "opt" in out and "[id] text" in out


def test_conflict_prompt_formats():
    out = load_prompt("conflict", "v1").format(
        option_id="A", precedence="kg>textbook", support_block="s", contradict_block="c",
        precedence_winner="SUPPORT",
    )
    assert "kg>textbook" in out and "SUPPORT" in out
