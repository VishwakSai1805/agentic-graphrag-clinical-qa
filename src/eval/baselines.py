"""Phase 3 baselines B0-B3 for the multi-choice clinical QA eval.

These are the comparison points the agentic pipeline (Phases 4-5) has to beat.
Each `run_baseline` call returns exactly one `Prediction`; the eval harness
(Phase 6) aggregates them with `src.eval.metrics`.

  B0  zero-shot          no retrieval, no pipeline. Ungrounded BY DESIGN.
  B1  dense-only RAG      bundle from {"dense"}, one generation call.
  B2  hybrid RAG          bundle from {"dense","lexical","kg"}, one call.
  B3  binary verifier     MEGA-RAG-style: one SUPPORTED/NOT_SUPPORTED call per
                          option; decide iff exactly one option comes back
                          SUPPORTED, else fall back to a B2-style generation
                          call.

B3 deliberately gets the SAME model, temperature=0, retrieval budget and
prompt care as B2 -- it is the "binary bottleneck" this project claims to
remove (CLAUDE.md section 1), not a crippled strawman.

All model access goes through `src.llm.client.chat`, imported into this module
so tests can monkeypatch `src.eval.baselines.chat`. No provider SDK is touched.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from src.llm.client import LLMContextBudgetError, chat

if TYPE_CHECKING:
    from src.kg.dataset_questions import Question
    from src.llm.client import StatsTracker
    from src.pipeline import TriModalPipeline

BASELINES = ("B0", "B1", "B2", "B3")

_DENSE_MODALITIES = frozenset({"dense"})
_HYBRID_MODALITIES = frozenset({"dense", "lexical", "kg"})

# PubMedQA carries no MCQ option map on `Question` -- its gold answer is
# yes/no/maybe. Project that onto letters so the baselines stay dataset-generic
# (the caller still passes the gold letter in explicitly).
_PUBMEDQA_OPTIONS = {"A": "yes", "B": "no", "C": "maybe"}

_SYSTEM = (
    "You are a careful clinical question-answering assistant. "
    "Respond with STRICT JSON only -- no prose, no markdown fences."
)

_VERDICTS = ("SUPPORTED", "NOT_SUPPORTED")

_B0_PROMPT = (
    "{question}\n\nOptions:\n{options}\n\n"
    'Answer with STRICT JSON: {{"answer": "<letter>", "rationale": "<=3 sentences"}}'
)

_CITE_INSTRUCTION = (
    "Cite using the exact id shown in square brackets (for example "
    "chunk::textbook::First_Aid_Step1#42 or kg::Aspirin|indication|pain), "
    "not the position."
)

_GEN_PROMPT = (
    "{question}\n\nOptions:\n{options}\n\n"
    "Evidence:\n{evidence}\n\n"
    "Using ONLY the evidence above, pick the single best option and cite the "
    "evidence ids you relied on. " + _CITE_INSTRUCTION + " STRICT JSON: "
    '{{"answer": "<letter>", "citations": ["<id>", ...], "rationale": "..."}}'
)

_VERIFY_PROMPT = (
    "{question}\n\n"
    "Given ONLY the evidence below, is option {letter} ({text}) the correct "
    "answer? Say SUPPORTED only if the evidence positively supports it, "
    "otherwise NOT_SUPPORTED.\n\n"
    "Evidence:\n{evidence}\n\n"
    + _CITE_INSTRUCTION
    + ' STRICT JSON: {{"verdict": "SUPPORTED" | "NOT_SUPPORTED", "evidence_id": "<id or null>"}}'
)


class Prediction(BaseModel):
    question_id: str
    pred_option: str | None
    gold_option: str
    correct: bool
    citations: list[str]
    invalid_citations: list[str]  # cited ids NOT in bundle.valid_citation_ids()
    grounded: bool  # True iff >=1 valid citation survives; B0 always False
    status: Literal["ok", "parse_error", "no_answer", "context_exceeded"]
    n_llm_calls: int
    verifier_decided: bool | None = None  # B3 only
    n_supported: int | None = None  # B3 only
    per_option_verdict: dict[str, str] | None = None  # B3 only


# --------------------------------------------------------------------------
# parsing helpers
# --------------------------------------------------------------------------


def _first_json_obj(text: str) -> dict | None:
    """First balanced ``{...}`` substring that `json.loads` parses to a dict."""
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


def _normalize_cite(token: str) -> str:
    """Strip the decoration the model wraps around an evidence id: outer
    whitespace, then ONE leading ``[`` / trailing ``]`` (the bracketed form
    `_format_evidence` renders), then surrounding quotes, then whitespace.
    """
    token = token.strip()
    if token.startswith("["):
        token = token[1:]
    if token.endswith("]"):
        token = token[:-1]
    token = token.strip().strip("\"'").strip()
    return token


def _coerce_citations(value: object) -> list[str]:
    """Tolerate a list of strings OR a bare string OR None."""
    if isinstance(value, str):
        v = _normalize_cite(value)
        return [v] if v else []
    if isinstance(value, (list, tuple)):
        out = []
        for x in value:
            if isinstance(x, str):
                norm = _normalize_cite(x)
                if norm:
                    out.append(norm)
        return out
    return []


def _citations_from_regex(text: str) -> list[str]:
    m = re.search(r'citations"\s*:\s*\[([^\]]*)\]', text)
    if not m:
        return []
    out: list[str] = []
    for part in m.group(1).split(","):
        token = _normalize_cite(part)
        if token:
            out.append(token)
    return out


def _parse_answer(text: str, valid_letters: set[str]) -> tuple[str | None, list[str], str]:
    """Return ``(letter, citations, status)``.

    status is "no_answer" (blank output), "parse_error" (no letter, or a
    letter outside ``valid_letters``), or "ok".
    """
    if text is None or not text.strip():
        return None, [], "no_answer"

    letter: str | None = None
    citations: list[str] = []

    obj = _first_json_obj(text)
    if obj is not None:
        raw = obj.get("answer")
        if isinstance(raw, str) and raw.strip():
            letter = raw.strip().upper()
        citations = _coerce_citations(obj.get("citations"))

    if letter is None or letter not in valid_letters:
        m = re.search(r'["\']?answer["\']?\s*[:=]\s*["\']?([A-Za-z])', text)
        if m:
            letter = m.group(1).strip().upper()
    if not citations:
        citations = _citations_from_regex(text)

    if letter is None:
        return None, citations, "parse_error"
    if letter not in valid_letters:
        return None, citations, "parse_error"
    return letter, citations, "ok"


def _parse_verdict(text: str) -> tuple[str, str | None]:
    """Return ``(verdict, evidence_id)``.

    verdict is one of ``_VERDICTS`` or ``"UNPARSEABLE"`` (recorded honestly,
    counted as not-supported).
    """
    verdict: str | None = None
    evidence_id: str | None = None

    obj = _first_json_obj(text)
    if obj is not None:
        raw = obj.get("verdict")
        if isinstance(raw, str) and raw.strip().upper() in _VERDICTS:
            verdict = raw.strip().upper()
        ev = obj.get("evidence_id")
        if isinstance(ev, str) and ev.strip() and ev.strip().lower() != "null":
            evidence_id = ev.strip()

    if verdict is None:
        m = re.search(
            r'verdict["\']?\s*[:=]\s*["\']?(SUPPORTED|NOT[_ ]?SUPPORTED)', text, re.I
        )
        if m:
            token = m.group(1).upper().replace(" ", "_")
            verdict = "SUPPORTED" if token == "SUPPORTED" else "NOT_SUPPORTED"

    return (verdict or "UNPARSEABLE"), evidence_id


# --------------------------------------------------------------------------
# prompt formatting
# --------------------------------------------------------------------------


def _resolve_options(question: Question) -> dict[str, str]:
    return dict(question.options) if question.options else dict(_PUBMEDQA_OPTIONS)


def _format_options(options: dict[str, str]) -> str:
    return "\n".join(f"{k}. {options[k]}" for k in sorted(options))


def _format_evidence(items: list) -> str:
    # One item per line as `[<id>] <text>` with NO leading item number -- a
    # number is something the model will cite instead of the id (Phase 3 exit
    # run: B2 cited "12", "14"). The bracketed id is stripped back off on parse
    # by `_normalize_cite`.
    if not items:
        return "(no evidence retrieved)"
    return "\n".join(f"[{e.id}] {e.text}" for e in items)


# --------------------------------------------------------------------------
# prediction assembly
# --------------------------------------------------------------------------


def _finalize(
    *,
    question_id: str,
    letter: str | None,
    citations: list[str],
    status: str,
    gold_option: str,
    valid_ids: set[str],
    n_llm_calls: int,
    allow_grounded: bool = True,
) -> Prediction:
    invalid = [c for c in citations if c not in valid_ids]
    has_valid = any(c in valid_ids for c in citations)
    return Prediction(
        question_id=question_id,
        pred_option=letter,
        gold_option=gold_option,
        correct=(letter is not None and letter == gold_option),
        citations=list(citations),
        invalid_citations=invalid,
        grounded=(allow_grounded and has_valid),
        status=status,  # constrained to the Literal set by all call sites
        n_llm_calls=n_llm_calls,
    )


# --------------------------------------------------------------------------
# individual baselines
# --------------------------------------------------------------------------


def _run_b0(
    question: Question, gold_option: str, tracker: StatsTracker | None, seed: int
) -> Prediction:
    options = _resolve_options(question)
    prompt = _B0_PROMPT.format(question=question.question, options=_format_options(options))
    resp = chat(prompt, system=_SYSTEM, temperature=0, seed=seed, tracker=tracker)
    letter, _cites, status = _parse_answer(resp.text, set(options))
    # --- CLAUDE.md grounding carve-out ------------------------------------
    # CLAUDE.md: "An answer with an empty citation set is a bug, and is scored
    # as ungrounded." B0 is the documented exception: it runs zero retrieval,
    # so it has nothing to cite. citations=[] and grounded=False here are BY
    # CONSTRUCTION, not a pipeline failure -- B0 is our un-grounded floor.
    return _finalize(
        question_id=question.question_id,
        letter=letter,
        citations=[],
        status=status,
        gold_option=gold_option,
        valid_ids=set(),
        n_llm_calls=1,
        allow_grounded=False,
    )


def _generate_answer(
    question: Question,
    options: dict[str, str],
    bundle,
    tracker: StatsTracker | None,
    seed: int,
) -> tuple[str | None, list[str], str]:
    prompt = _GEN_PROMPT.format(
        question=question.question,
        options=_format_options(options),
        evidence=_format_evidence(bundle.evidence()),
    )
    resp = chat(prompt, system=_SYSTEM, temperature=0, seed=seed, tracker=tracker)
    return _parse_answer(resp.text, set(options))


def _run_generation(
    question: Question,
    gold_option: str,
    pipeline: TriModalPipeline,
    dataset: str,
    tracker: StatsTracker | None,
    seed: int,
    modalities: frozenset[str],
) -> Prediction:
    options = _resolve_options(question)
    bundle = pipeline.build_bundle(
        question.question,
        options,
        dataset=dataset,
        question_id=question.question_id,
        modalities=modalities,
    )
    letter, citations, status = _generate_answer(question, options, bundle, tracker, seed)
    return _finalize(
        question_id=question.question_id,
        letter=letter,
        citations=citations,
        status=status,
        gold_option=gold_option,
        valid_ids=bundle.valid_citation_ids(),
        n_llm_calls=1,
    )


def _run_b3(
    question: Question,
    gold_option: str,
    pipeline: TriModalPipeline,
    dataset: str,
    tracker: StatsTracker | None,
    seed: int,
) -> Prediction:
    options = _resolve_options(question)
    bundle = pipeline.build_bundle(
        question.question,
        options,
        dataset=dataset,
        question_id=question.question_id,
        modalities=_HYBRID_MODALITIES,
    )
    valid_ids = bundle.valid_citation_ids()
    evidence_block = _format_evidence(bundle.evidence())

    per_option_verdict: dict[str, str] = {}
    per_option_evidence: dict[str, str | None] = {}
    n_calls = 0
    for letter in sorted(options):
        prompt = _VERIFY_PROMPT.format(
            question=question.question,
            letter=letter,
            text=options[letter],
            evidence=evidence_block,
        )
        resp = chat(prompt, system=_SYSTEM, temperature=0, seed=seed, tracker=tracker)
        n_calls += 1
        verdict, evidence_id = _parse_verdict(resp.text)
        per_option_verdict[letter] = verdict
        per_option_evidence[letter] = evidence_id

    supported = [ltr for ltr, v in per_option_verdict.items() if v == "SUPPORTED"]
    n_supported = len(supported)

    if n_supported == 1:
        chosen = supported[0]
        ev = per_option_evidence[chosen]
        citations = [ev] if (ev is not None and ev in valid_ids) else []
        pred = _finalize(
            question_id=question.question_id,
            letter=chosen,
            citations=citations,
            status="ok",
            gold_option=gold_option,
            valid_ids=valid_ids,
            n_llm_calls=n_calls,
        )
        pred.verifier_decided = True
    else:
        # 0 or >=2 SUPPORTED: the binary verifier cannot decide. Fall back to
        # exactly the B2 generation call (same model, temp=0, same evidence).
        prompt = _GEN_PROMPT.format(
            question=question.question,
            options=_format_options(options),
            evidence=evidence_block,
        )
        resp = chat(prompt, system=_SYSTEM, temperature=0, seed=seed, tracker=tracker)
        n_calls += 1
        letter, citations, status = _parse_answer(resp.text, set(options))
        pred = _finalize(
            question_id=question.question_id,
            letter=letter,
            citations=citations,
            status=status,
            gold_option=gold_option,
            valid_ids=valid_ids,
            n_llm_calls=n_calls,
        )
        pred.verifier_decided = False

    pred.n_supported = n_supported
    pred.per_option_verdict = per_option_verdict
    return pred


# --------------------------------------------------------------------------
# public entrypoint
# --------------------------------------------------------------------------


def run_baseline(
    name: str,
    question: Question,
    gold_option: str,
    *,
    pipeline: TriModalPipeline | None,
    dataset: str,
    tracker: StatsTracker | None = None,
    seed: int = 42,
) -> Prediction:
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; expected one of {BASELINES}")
    if name != "B0" and pipeline is None:
        raise ValueError(f"baseline {name} requires a pipeline (got None)")

    try:
        if name == "B0":
            return _run_b0(question, gold_option, tracker, seed)
        if name == "B1":
            return _run_generation(
                question, gold_option, pipeline, dataset, tracker, seed, _DENSE_MODALITIES
            )
        if name == "B2":
            return _run_generation(
                question, gold_option, pipeline, dataset, tracker, seed, _HYBRID_MODALITIES
            )
        return _run_b3(question, gold_option, pipeline, dataset, tracker, seed)
    except LLMContextBudgetError:
        # Hard stop, not a silent truncation (CLAUDE.md budget-guard rule).
        # n_llm_calls=0: a multi-call B3 may have made several calls before the
        # oversized one, but we don't thread that count out -- 0 is the honest
        # floor. verifier_decided / n_supported / per_option_verdict stay None.
        return Prediction(
            question_id=question.question_id,
            pred_option=None,
            gold_option=gold_option,
            correct=False,
            citations=[],
            invalid_citations=[],
            grounded=False,
            status="context_exceeded",
            n_llm_calls=0,
        )
