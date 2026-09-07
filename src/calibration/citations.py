"""Phase 5: per-option citation resolution.

`docs/00_RESEARCH_SPEC.md` §4 point 5: output is ranked `[(o_i, p_i,
cite(E_i))]` -- every option carries its own citation set, not just the
top-ranked one. This is `cite(E_i)`.
"""

from __future__ import annotations

from src.agents.state import SupportMatrix


def cite(option_id: str, matrix: SupportMatrix) -> list[str]:
    """Sorted, deduplicated evidence_ids from `option_id`'s SUPPORT cells in
    `matrix`. SUPPORT only (not CONTRADICT) -- a citation set justifies why an
    option is being presented as a candidate, and a CONTRADICT cell is
    evidence AGAINST the option, not a citation for it. Empty list (not an
    error) for an option with no SUPPORT cells -- an ungrounded option is a
    real, reportable outcome (CLAUDE.md's grounding rule), not a bug here.
    """
    ids = {
        cell.evidence_id
        for cell in matrix.cells
        if cell.option_id == option_id and cell.label == "SUPPORT"
    }
    return sorted(ids)


def cite_all(matrix: SupportMatrix) -> dict[str, list[str]]:
    """`{option_id: cite(option_id, matrix)}` for every option in the
    question -- the per-question citation map a ranked-output row needs."""
    return {option_id: cite(option_id, matrix) for option_id in matrix.options}
