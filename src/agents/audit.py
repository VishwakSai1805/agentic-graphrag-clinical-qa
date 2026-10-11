"""Evidential Audit agent (Phase 4, docs/01_ARCHITECTURE.md "Agent contracts").

The graph calls `run_audit` ONCE per option. Given that option's atomic claims
(from the Hypothesis agent) and the question's evidence bundle, the agent emits
one `SupportCell` per (claim, evidence) pair that the evidence SUPPORTS or
CONTRADICTS -- judged against the retrieved evidence ONLY, never the model's own
knowledge.

Determinism (agents-engineer.md / PROJECT_RULES.md): Audit is `temperature=0`, always.
Only the Hypothesis agent may sample. `seed` is threaded through for provenance
and for `parse_or_repair`'s repair call.

Grounding hygiene:
- A cell citing an `evidence_id` not present in the provided bundle is discarded
  and counted in `n_invalid_dropped` (the honest signal -- a filtered matrix is
  clean by construction). The model's id is normalized first (Phase 3 / ADR-021
  lesson: models wrap the id in `[...]` / quotes because the prompt renders it
  bracketed).
- The `evidence` block is rendered WITHOUT leading item numbers -- a number is
  something the model cites instead of the id (ADR-021: Phase 3 B2 cited "12").
"""

from __future__ import annotations

from pydantic import BaseModel

from src.agents.schemas import AuditOut, parse_or_repair
from src.agents.state import SupportCell, source_rank
from src.config import AgentsConfig
from src.llm.client import chat
from src.llm.prompts import load_prompt
from src.schema import Evidence

_SYSTEM = "You judge clinical claims against retrieved evidence only. Respond with STRICT JSON only."  # noqa: E501

_MAX_RATIONALE_CHARS = 400

# The real gateway, captured at import. Used only as an identity sentinel: when
# a caller does not inject its own `chat_fn`, `run_audit` falls back to the
# current module-level `chat`, so tests that monkeypatch `src.agents.audit.chat`
# take effect even though the parameter default was bound at def time.
_DEFAULT_CHAT = chat


class AuditResult(BaseModel):
    """Return of one `run_audit` call (one option)."""

    cells: list[SupportCell]
    n_llm_calls: int
    n_invalid_dropped: int  # cells citing an evidence_id not in the provided evidence


def _format_claims(claims: list[str]) -> str:
    """Lines ``"{i}: {claim}"`` -- the index is the `claim_idx` the model must cite."""
    return "\n".join(f"{i}: {claim}" for i, claim in enumerate(claims))


def _format_evidence(evidence: list[Evidence]) -> str:
    """Lines ``"[{id}] {text}"`` -- NO leading item number (ADR-021: a numbered
    list makes the model cite the number instead of the evidence id)."""
    return "\n".join(f"[{e.id}] {e.text}" for e in evidence)


def _normalize_cite(token: str) -> str:
    """Strip the decoration the model wraps around an evidence id: outer
    whitespace, then ONE leading ``[`` / trailing ``]`` (the bracketed form
    `_format_evidence` renders), then surrounding quotes, then whitespace.
    Same normalization idea as Phase 3's `src/eval/baselines.py:_normalize_cite`.
    """
    token = token.strip()
    if token.startswith("["):
        token = token[1:]
    if token.endswith("]"):
        token = token[:-1]
    token = token.strip().strip("\"'").strip()
    return token


def run_audit(
    option_id: str,
    option_text: str,
    claims: list[str],
    evidence: list[Evidence],
    *,
    cfg: AgentsConfig,
    chat_fn=chat,
    seed: int = 42,
) -> AuditResult:
    """Audit one option's claims against the evidence bundle. Exactly one
    `chat_fn` call per invocation (plus at most one repair call inside
    `parse_or_repair`). Lets `AgentParseError` propagate."""
    if chat_fn is _DEFAULT_CHAT:
        chat_fn = chat  # honor a monkeypatched module-level `chat`

    prompt = load_prompt("audit", cfg.prompt_versions["audit"]).format(
        option_id=option_id,
        option_text=option_text,
        claims=_format_claims(claims),
        evidence=_format_evidence(evidence),
    )

    resp = chat_fn(prompt, system=_SYSTEM, temperature=0, seed=seed)
    parsed, n_repairs = parse_or_repair(
        resp.text,
        AuditOut,
        chat_fn=chat_fn,
        system=_SYSTEM,
        original_prompt=prompt,
        seed=seed,
    )
    n_llm_calls = 1 + n_repairs

    by_id = {e.id: e for e in evidence}
    n_claims = len(claims)

    cells: list[SupportCell] = []
    n_invalid_dropped = 0
    for cell in parsed.cells:
        cited = _normalize_cite(cell.evidence_id)
        ev = by_id.get(cited)
        if ev is None:
            # Hallucinated / mangled citation: discard and count it.
            n_invalid_dropped += 1
            continue
        # The schema permits NEUTRAL, but this agent only records non-neutral
        # cells (the prompt asks for SUPPORT/CONTRADICT only; NEUTRAL is omitted).
        if cell.label == "NEUTRAL":
            continue
        if cell.claim_idx not in range(n_claims):
            continue
        cells.append(
            SupportCell(
                option_id=option_id,
                claim_idx=cell.claim_idx,
                evidence_id=cited,
                label=cell.label,
                rationale=(cell.rationale or "")[:_MAX_RATIONALE_CHARS],
                source_rank=source_rank(ev, cfg.source_precedence),
                modality=ev.modality,
                rerank_score=ev.scores.get("rerank", 0.0),
            )
        )

    return AuditResult(
        cells=cells,
        n_llm_calls=n_llm_calls,
        n_invalid_dropped=n_invalid_dropped,
    )
