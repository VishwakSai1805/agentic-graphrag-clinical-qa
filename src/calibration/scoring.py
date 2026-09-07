"""Phase 5 logit assembly: turns a `SupportMatrix` into one `OptionFeatures`
row per option, and a `CalibrationWeights` tuple into that option's raw logit.

`z_i = alpha*s_pos_i - beta*s_neg_i + gamma*kg_support_count_i +
delta*mean_rerank_score_i` (docs/01_ARCHITECTURE.md "3. MULTI-CHOICE
EVIDENTIAL CALIBRATION & RANKING"; docs/00_RESEARCH_SPEC.md §4 point 3).

Feature definitions (recorded here because neither doc pins them precisely --
see the ADR this phase adds to docs/05_DECISIONS.md):
  - `kg_support_count`: count of this option's SUPPORT cells whose cited
    evidence is KG-modality. A separate signal from `s_pos` (which counts
    SUPPORT cells of every modality) so gamma can weight KG-backed support
    differently from passage-backed support, per the architecture's
    "kg_i" term.
  - `mean_rerank_score`: mean MedCPT rerank score over this option's
    non-neutral (SUPPORT + CONTRADICT) cells -- both directions, not just
    SUPPORT, because it is meant as a retrieval-quality prior on the
    evidence actually consulted for this option, independent of which way
    the audit read it. 0.0 when the option has no non-neutral cells (no
    retrieval-quality signal to report, not "low quality").

No LLM calls, no I/O -- pure functions over an already-computed SupportMatrix,
directly unit-testable against hand-built matrices.
"""

from __future__ import annotations

from pydantic import BaseModel

from src.agents.state import SupportMatrix


class OptionFeatures(BaseModel):
    """One option's calibration inputs, extracted from a SupportMatrix."""

    option_id: str
    s_pos: int
    s_neg: int
    s_neu: int
    kg_support_count: int
    mean_rerank_score: float
    resolved: bool


class CalibrationWeights(BaseModel):
    """The fitted (or default/unfit) z_i = alpha*s+ - beta*s- + gamma*kg +
    delta*rerank parameters, plus the temperature applied to softmax(z/T)."""

    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.0
    delta: float = 0.0
    temperature: float = 1.0


def extract_option_features(matrix: SupportMatrix) -> list[OptionFeatures]:
    """One `OptionFeatures` per option in `matrix.options`, in that dict's
    iteration order. An option with zero cells (no signal at all) still gets
    a row -- all-zero features, not omitted -- so every option in the
    question is always scoreable."""
    by_option: dict[str, dict] = {
        oid: {"kg_support_count": 0, "rerank_scores": []} for oid in matrix.options
    }
    for cell in matrix.cells:
        acc = by_option.get(cell.option_id)
        if acc is None:  # a cell for an option not in matrix.options -- ignore defensively
            continue
        if cell.label in ("SUPPORT", "CONTRADICT"):
            acc["rerank_scores"].append(cell.rerank_score)
        if cell.label == "SUPPORT" and cell.modality == "kg":
            acc["kg_support_count"] += 1

    support_by_id = {os.option_id: os for os in matrix.option_support}
    out: list[OptionFeatures] = []
    for oid in matrix.options:
        os = support_by_id.get(oid)
        acc = by_option[oid]
        rerank_scores = acc["rerank_scores"]
        out.append(
            OptionFeatures(
                option_id=oid,
                s_pos=os.s_pos if os else 0,
                s_neg=os.s_neg if os else 0,
                s_neu=os.s_neu if os else 0,
                kg_support_count=acc["kg_support_count"],
                mean_rerank_score=(sum(rerank_scores) / len(rerank_scores))
                if rerank_scores
                else 0.0,
                resolved=os.resolved if os else True,
            )
        )
    return out


def compute_logit(features: OptionFeatures, weights: CalibrationWeights) -> float:
    """z_i for one option. `resolved` is NOT applied here (ADR-025:
    unresolved must not suppress a logit on a degenerate rule -- if a future
    ADR decides otherwise, that policy belongs in `softmax_rank.py`'s ranking
    step, as an explicit, separately-reviewable choice, not folded silently
    into this arithmetic)."""
    return (
        weights.alpha * features.s_pos
        - weights.beta * features.s_neg
        + weights.gamma * features.kg_support_count
        + weights.delta * features.mean_rerank_score
    )
