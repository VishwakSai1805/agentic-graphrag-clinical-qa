"""Hypothesis agent: decompose each MCQ option into atomic verifiable claims.

Pure function over typed state (agents-engineer.md): one batched LLM call over
all options, strict-JSON parse with a single repair attempt (delegated to
`schemas.parse_or_repair`), then a deterministic post-validation into
`list[OptionClaims]`.

Determinism: `temperature=0` unless `cfg.hypothesis_sample` is set (an ablation
flag, ADR-001), and the sampling seed is always recorded by the caller.
"""

from __future__ import annotations

from src.agents.schemas import HypothesisOut, parse_or_repair
from src.agents.state import OptionClaims
from src.config import AgentsConfig
from src.llm.client import chat
from src.llm.prompts import load_prompt

# Imported by name so tests can monkeypatch `src.agents.hypothesis.chat`.
__all__ = ["run_hypothesis", "chat"]

_SYSTEM = (
    "You decompose clinical answer options into atomic verifiable claims. "
    "Respond with STRICT JSON only."
)

_MAX_CLAIMS_PER_OPTION = 4


def _format_options(options: dict[str, str]) -> str:
    """Render options as ``A. <text>`` lines, sorted by option key."""
    return "\n".join(f"{key}. {options[key]}" for key in sorted(options))


def run_hypothesis(
    query: str,
    options: dict[str, str],
    *,
    cfg: AgentsConfig,
    chat_fn=chat,
    seed: int = 42,
) -> tuple[list[OptionClaims], int]:
    """Decompose every option in `options` into atomic claims.

    Returns `(claims_per_option, n_llm_calls)`. `n_llm_calls` is 1 for the
    batched generation call plus 0 or 1 for a repair attempt. Raises
    `AgentParseError` (from `parse_or_repair`) if the model output cannot be
    validated even after the single repair -- the graph handles that.
    """
    prompt = load_prompt("hypothesis", cfg.prompt_versions["hypothesis"]).format(
        question=query,
        options=_format_options(options),
    )
    temperature = cfg.hypothesis_temperature if cfg.hypothesis_sample else 0.0

    resp = chat_fn(prompt, system=_SYSTEM, temperature=temperature, seed=seed)

    n_llm_calls = 1
    parsed, n_repair = parse_or_repair(
        resp.text,
        HypothesisOut,
        chat_fn=chat_fn,
        system=_SYSTEM,
        original_prompt=prompt,
        seed=seed,
    )
    n_llm_calls += n_repair

    return _to_option_claims(parsed, options), n_llm_calls


def _to_option_claims(
    parsed: HypothesisOut, options: dict[str, str]
) -> list[OptionClaims]:
    """Convert the raw schema output into one `OptionClaims` per known option.

    Post-validation:
    - Entries for an unknown `option_id` (not in `options`) are dropped.
    - Claims are stripped of empty strings and clamped to
      `_MAX_CLAIMS_PER_OPTION`.
    - Any option in `options` with no surviving entry gets a synthesized
      single fallback claim. Downstream the Audit agent iterates `options`;
      a missing claim set would silently skip auditing that option entirely.
    """
    by_option: dict[str, OptionClaims] = {}
    for entry in parsed.options:
        if entry.option_id not in options:
            continue
        if entry.option_id in by_option:
            continue  # first entry wins; ignore duplicates
        claims = [c for c in entry.claims if c and c.strip()][:_MAX_CLAIMS_PER_OPTION]
        by_option[entry.option_id] = OptionClaims(
            option_id=entry.option_id,
            claims=claims,
            required_evidence_types=list(entry.required_evidence_types),
        )

    result: list[OptionClaims] = []
    for key in sorted(options):
        entry = by_option.get(key)
        if entry is None or not entry.claims:
            result.append(
                OptionClaims(
                    option_id=key,
                    claims=[f"Option {key} ({options[key]}) is the correct answer."],
                    required_evidence_types=[],
                )
            )
        else:
            result.append(entry)
    return result
