"""The launcher's emitted commands must actually parse (ADR-047).

`scripts/run_gemini_sensitivity.ps1` emitted `--variant-suffix -gemini`. argparse
reads a token beginning with "-" as a new option, not as the preceding option's
value, so every cell died at launch with "argument --variant-suffix: expected
one argument" -- after the model probe, before any real work. Nothing local
caught it: the config resolved, the tests passed, the launcher parsed as
PowerShell, and the dry run printed the broken command without running it.

So this test reads the REAL command strings out of the real `.ps1`, resolves the
PowerShell interpolation, and feeds the resulting argv to the actual parsers.
It fails if anyone reverts to the space form, and it covers the whole bug class,
not just the one instance: no `--flag` may be followed by a value starting with
a dash.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PS1 = REPO_ROOT / "scripts" / "run_gemini_sensitivity.ps1"

EXPECTED_SUFFIX = {
    "V0": "-gemini",
    "V3": "-gemini",
    "B3": "-gemini",
    "V5": "-gemini",
    "V3rep": "-gemini-rep2",
}


def _scalar(text: str, name: str) -> str:
    m = re.search(rf'^\${name}\s*=\s*"([^"]*)"', text, re.M)
    assert m, f"${name} assignment not found in {PS1.name}"
    return m.group(1)


def _interp(s: str, env: dict[str, str]) -> str:
    for key in sorted(env, key=len, reverse=True):  # longest first: $H vs $Hrep
        s = s.replace(f"${key}", env[key])
    return s


def launcher_commands() -> dict[str, str]:
    """The five cell commands exactly as the launcher would run them."""
    text = PS1.read_text(encoding="utf-8")
    env = {
        "IdsPath": _scalar(text, "IdsPath"),
        "ApiVersion": _scalar(text, "ApiVersion"),
        # the probe picks the first candidate that the free tier serves
        "Model": re.search(
            r"\$ModelCandidates\s*=\s*@\(\s*\"([^\"]+)\"", text
        ).group(1),
    }
    for name in ("CommonArgs", "H", "B", "Hrep"):
        env[name] = _interp(_scalar(text, name), env)

    block = re.search(r"\$AllCells = \[ordered\]@\{(.*?)\n\}", text, re.S)
    assert block, "$AllCells block not found"
    cells = {
        m.group(1): _interp(m.group(2), env)
        for m in re.finditer(r'"(\w+)"\s*=\s*"([^"]*)"', block.group(1))
    }
    assert set(cells) == set(EXPECTED_SUFFIX), f"cells drifted: {sorted(cells)}"
    return cells


def _argv(command: str) -> tuple[str, list[str]]:
    """('src.eval.harness', [...flags...]) from 'uv run python -m MOD ...'."""
    tokens = shlex.split(command)
    assert tokens[:4] == ["uv", "run", "python", "-m"], tokens[:4]
    return tokens[4], tokens[5:]


def _parser_for(module: str):
    if module == "src.eval.harness":
        from src.eval.harness import build_parser
    elif module == "scripts.run_baselines":
        from scripts.run_baselines import build_parser
    else:  # pragma: no cover
        raise AssertionError(f"unexpected module {module!r}")
    return build_parser()


@pytest.mark.parametrize("cell", sorted(EXPECTED_SUFFIX))
def test_each_emitted_command_parses_with_the_right_suffix(cell):
    module, argv = _argv(launcher_commands()[cell])
    args = _parser_for(module).parse_args(argv)
    assert args.variant_suffix == EXPECTED_SUFFIX[cell]
    assert args.dataset == "medqa"
    assert args.split == "test"
    assert args.seed == 42
    assert args.llm_model.startswith("gemini/")
    assert args.question_ids.endswith("question_ids_200.json")


@pytest.mark.parametrize("cell", sorted(EXPECTED_SUFFIX))
def test_no_flag_is_followed_by_a_dash_leading_value(cell):
    """The bug class, not just the one instance: a value starting with '-' must
    be attached with '=', or argparse silently reads it as another option."""
    _, argv = _argv(launcher_commands()[cell])
    for flag, nxt in zip(argv, argv[1:], strict=False):  # deliberately offset by one
        if flag.startswith("--") and "=" not in flag:
            assert not nxt.startswith("-"), (
                f"{cell}: '{flag} {nxt}' is unparseable -- use {flag}={nxt}"
            )


def test_the_pinned_question_file_the_launcher_names_exists():
    _, argv = _argv(launcher_commands()["V3"])
    path = next(a.split("=", 1)[1] for a in argv if a.startswith("--question-ids="))
    assert (REPO_ROOT / path).exists(), f"launcher names a missing file: {path}"


def test_the_space_form_really_does_fail():
    """Proves this test is sensitive to the regression it guards against --
    otherwise it could pass for the wrong reason."""
    from src.eval.harness import build_parser

    ok = build_parser().parse_args(
        ["--config=configs/gemini_base.yaml", "--variant-suffix=-gemini"]
    )
    assert ok.variant_suffix == "-gemini"

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--config=configs/gemini_base.yaml", "--variant-suffix", "-gemini"]
        )


def test_run_baselines_accepts_the_same_suffix_form():
    from scripts.run_baselines import build_parser

    args = build_parser().parse_args(["--baselines=B3", "--variant-suffix=-gemini"])
    assert args.variant_suffix == "-gemini"
    assert args.baselines == "B3"
