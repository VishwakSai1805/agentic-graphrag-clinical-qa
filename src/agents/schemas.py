"""Strict LLM-output schemas for the three agents + `parse_or_repair`.

Contract (agents-engineer.md): every agent response is schema-validated; on a
validation failure there is exactly ONE repair attempt (the validation error
appended to the prompt), then a recorded hard failure -- never a silent coerce
or guess.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from src.agents.state import Label

T = TypeVar("T", bound=BaseModel)


class AgentParseError(RuntimeError):
    """An agent response failed schema validation twice (initial + one repair)."""


# --- agent output schemas -------------------------------------------------


class HypothesisClaimSet(BaseModel):
    option_id: str
    claims: list[str]
    required_evidence_types: list[str] = []


class HypothesisOut(BaseModel):
    options: list[HypothesisClaimSet]


class AuditCellOut(BaseModel):
    claim_idx: int
    evidence_id: str
    label: Label
    rationale: str = ""


class AuditOut(BaseModel):
    cells: list[AuditCellOut]


class ConflictOut(BaseModel):
    """v2 (ADR-025): the Conflict LLM only judges genuine-vs-spurious. Source
    precedence picks the winner mechanically, afterwards, in code."""

    genuine_contradiction: bool
    rationale: str = ""


# --- parsing ------------------------------------------------------------


def first_json_obj(text: str) -> dict | None:
    """First balanced ``{...}`` substring that `json.loads` parses to a dict.
    Tolerates prose or markdown fences around the JSON."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    return obj if isinstance(obj, dict) else None
        start = text.find("{", start + 1)
    return None


ChatFn = Callable[..., object]  # returns an object with a `.text` attribute


def parse_or_repair(
    raw_text: str,
    model: type[T],
    *,
    chat_fn: ChatFn,
    system: str | None,
    original_prompt: str,
    seed: int,
) -> tuple[T, int]:
    """Validate `raw_text` against `model`. On failure, make exactly one repair
    call (temperature=0) with the validation error appended, then validate
    again. Returns `(parsed, n_repair_calls)` (0 or 1). Raises `AgentParseError`
    if the repair also fails.
    """
    parsed, err = _try_parse(raw_text, model)
    if parsed is not None:
        return parsed, 0

    repair_prompt = (
        f"{original_prompt}\n\n"
        f"Your previous response could not be parsed:\n{raw_text}\n\n"
        f"The error was: {err}\n\n"
        f"Return ONLY a JSON object matching this schema, nothing else:\n"
        f"{json.dumps(model.model_json_schema())}"
    )
    resp = chat_fn(repair_prompt, system=system, temperature=0, seed=seed)
    parsed, err2 = _try_parse(resp.text, model)
    if parsed is not None:
        return parsed, 1
    raise AgentParseError(
        f"{model.__name__} failed validation after one repair attempt. "
        f"first error: {err}; repair error: {err2}"
    )


def _try_parse(raw_text: str, model: type[T]) -> tuple[T | None, str | None]:
    obj = first_json_obj(raw_text or "")
    if obj is None:
        return None, "no JSON object found in response"
    try:
        return model.model_validate(obj), None
    except ValidationError as exc:
        return None, str(exc)
