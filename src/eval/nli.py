"""Phase 6: hallucination-rate scoring by NLI entailment against *cited*
evidence (`docs/04_EVALUATION.md` §3).

A generated claim is *grounded* iff at least one piece of evidence the answer
cited for it entails it (NLI P(entailment) >= 0.5). A claim with no citation,
or whose cited ids do not resolve to any evidence text, is ungrounded by
construction (CLAUDE.md: "An answer with an empty citation set is a bug, and is
scored as ungrounded").

Two denominators are reported, from the **same** NLI pass (ADR-029):

  * ``hallucination_rate_predicted_option`` -- claims backing the option the
    system actually answered. **Primary.** It is what a reader who takes the
    top-1 answer is exposed to, and it is the number comparable across
    variants that do not all produce a full ranked differential.
  * ``hallucination_rate_all_support_cells`` -- every SUPPORT assertion the
    system makes across all options. The system's real output is a ranked
    differential with per-option citations, so every SUPPORT cell is an
    assertion on display; this denominator counts them all. Reported so a
    reviewer who prefers it does not need a re-run.

For the non-agent baselines (V0-V3) there is a single free-text rationale and
no per-option SUPPORT cells, so the two denominators are equal by
construction; both keys are still emitted.

The NLI model is `src.kg.novelty.NLI_MODEL_NAME` (the same DeBERTa-v3-large
MNLI/Fever/ANLI checkpoint Phase 2 pinned). This module never loads it
directly for tests -- pass a fake ``scorer``; the real one is
``src.kg.novelty.get_default_scorer()`` and is loaded lazily, once, only when
``scorer is None``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel

from src.kg.novelty import NLI_MODEL_NAME, ScoreFn

NLI_ENTAIL_THRESHOLD = 0.5

# Trailing tokens that end with "." but do NOT end a sentence. Compared
# lower-cased and stripped of a trailing period.
_ABBREV = frozenset(
    {
        "e.g",
        "i.e",
        "vs",
        "etc",
        "cf",
        "al",  # "et al."
        "approx",
        "no",
        "fig",
        "eq",
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "st",
    }
)


def split_claims(rationale: str) -> list[str]:
    """Split a free-text rationale into atomic claim strings (`docs/04` §3:
    "Split the answer rationale into atomic claims").

    Deterministic and dependency-free -- **no LLM** (an LLM splitter is another
    non-deterministic model call and would make the hallucination-rate
    denominator unauditable). Sentence boundary = ``.``/``!``/``?`` followed by
    whitespace, with two guards:

      * a decimal point inside a number ("5.5 mg") is not a boundary;
      * a fragment ending in a known abbreviation ("e.g.", "vs.", "et al.")
        is glued to the next fragment.

    Fragments shorter than 3 chars or containing no letter (stray list
    markers like "1.") are dropped. Returns ``[]`` for empty/blank input.
    """
    text = (rationale or "").strip()
    if not text:
        return []

    protected = re.sub(r"(?<=\d)\.(?=\d)", "\x00", text)  # mask decimal points
    pieces = re.split(r"(?<=[.!?])\s+", protected)

    claims: list[str] = []
    buffer = ""
    for piece in pieces:
        candidate = f"{buffer} {piece}".strip() if buffer else piece
        words = candidate.rstrip(".!?").split()
        last = words[-1].lower().rstrip(".") if words else ""
        if last in _ABBREV:
            buffer = candidate
            continue
        claims.append(candidate)
        buffer = ""
    if buffer:
        claims.append(buffer)

    out = []
    for claim in claims:
        cleaned = claim.replace("\x00", ".").strip()
        if len(cleaned) >= 3 and re.search(r"[A-Za-z]", cleaned):
            out.append(cleaned)
    return out


class ClaimRecord(BaseModel):
    """One claim to be checked, with the text of the evidence it cited.

    ``cited_evidence_texts`` is already resolved to text by the caller (the
    harness knows the bundle); an entry that could not be resolved is simply
    absent, so an empty list == "no usable citation" == ungrounded.
    """

    question_id: str
    option_id: str | None = None
    claim: str
    cited_evidence_texts: list[str] = []
    predicted_option: bool = False

    @property
    def has_citation(self) -> bool:
        return len(self.cited_evidence_texts) > 0


def _aggregate(scored: list[dict]) -> dict:
    n = len(scored)
    n_uncited = sum(1 for s in scored if not s["cited"])
    n_ungrounded = sum(1 for s in scored if not s["entailed"])
    return {
        "n_claims": n,
        "n_ungrounded": n_ungrounded,
        "n_uncited": n_uncited,
        # nothing asserted -> nothing hallucinated; n_claims is reported
        # alongside so a 0.0 from an empty set is never misread as "clean".
        "hallucination_rate": (n_ungrounded / n) if n else 0.0,
    }


def score_hallucination(
    records: Sequence[ClaimRecord],
    scorer: ScoreFn | None = None,
    threshold: float = NLI_ENTAIL_THRESHOLD,
) -> dict:
    """NLI-entail every claim in ``records`` against its cited evidence, once,
    and aggregate under both denominators (see module docstring / ADR-029).

    ``scorer(premise, hypothesis) -> P(entailment) in [0, 1]`` -- premise is a
    cited evidence text, hypothesis is the claim. Defaults to the real
    lazily-loaded model; pass a fake in tests.

    Returns a dict with, for each of the ``predicted_option`` and
    ``all_support_cells`` scopes: ``hallucination_rate_<scope>``,
    ``n_claims_<scope>``, ``n_ungrounded_<scope>``, ``n_uncited_<scope>``;
    plus ``threshold``, ``nli_model``, ``primary`` (the key name of the
    reported headline), and ``per_claim`` (one row per claim, with its max
    entailment score) for auditing.
    """
    if scorer is None:
        # get_default_scorer() -> the lazy singleton; src.kg.novelty._load_model
        # prints torch.cuda.is_available() unconditionally on first load
        # (CLAUDE.md §2), so no extra CUDA print is needed here.
        from src.kg.novelty import get_default_scorer

        scorer = get_default_scorer()

    per_claim: list[dict] = []
    for rec in records:
        if rec.has_citation:
            best = max(scorer(ev_text, rec.claim) for ev_text in rec.cited_evidence_texts)
        else:
            best = 0.0
        per_claim.append(
            {
                "question_id": rec.question_id,
                "option_id": rec.option_id,
                "claim": rec.claim,
                "predicted_option": rec.predicted_option,
                "cited": rec.has_citation,
                "max_entail_score": best,
                "entailed": rec.has_citation and best >= threshold,
            }
        )

    predicted = [s for s in per_claim if s["predicted_option"]]
    agg_pred = _aggregate(predicted)
    agg_all = _aggregate(per_claim)

    return {
        "primary": "hallucination_rate_predicted_option",
        "hallucination_rate_predicted_option": agg_pred["hallucination_rate"],
        "n_claims_predicted_option": agg_pred["n_claims"],
        "n_ungrounded_predicted_option": agg_pred["n_ungrounded"],
        "n_uncited_predicted_option": agg_pred["n_uncited"],
        "hallucination_rate_all_support_cells": agg_all["hallucination_rate"],
        "n_claims_all_support_cells": agg_all["n_claims"],
        "n_ungrounded_all_support_cells": agg_all["n_ungrounded"],
        "n_uncited_all_support_cells": agg_all["n_uncited"],
        "threshold": threshold,
        "nli_model": NLI_MODEL_NAME,
        "per_claim": per_claim,
    }
