"""Marginal-value diagnostic for KG evidence (Phase 2, substantiates
contribution C1): is a retrieved PrimeKG triple telling the LLM something the
dense-retrieval passages *don't already say*, or is it redundant with what
tri-modal retrieval would have surfaced anyway?

Reuses the exact threshold `docs/04_EVALUATION.md` §3 / CLAUDE.md §3 already
pin for Phase 6's hallucination-rate scoring (entailment >= 0.5) rather than
inventing a second convention.

**Model-id correction (flag for an ADR):** CLAUDE.md §3 pins
`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli`. Verified live against the
real Hugging Face Hub during this phase's implementation
(`HfApi.model_info`): that exact repo id does not exist (401/"Repository Not
Found"). The real, existing "large" DeBERTa-v3 model in this exact
MNLI+Fever+ANLI lineage from the same author is
`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (adds LingNLI +
WANLI training data on top of the same MNLI/Fever/ANLI base the pinned name
describes; same 3-way entailment/neutral/contradiction head, confirmed via
`AutoConfig.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}`).
This module uses the verified-real id, not the pinned-but-nonexistent one --
see the ADR this phase adds in docs/05_DECISIONS.md for the full record. Any
Phase 6 code pulling this model forward for hallucination-rate scoring
should use `NLI_MODEL_NAME` from this module rather than re-typing
CLAUDE.md's string, so the two phases can't silently diverge on which model
id is "the" real one.

**Relevance-conditioned addendum (this phase's review correction, see
docs/05_DECISIONS.md):** a high novel-relation-rate is a weak signal for C1
on its own -- an NLI model failing to entail an arbitrary/irrelevant graph
triple looks identical to it failing to entail a genuinely useful one.
`is_triple_relevant` is a second, independent, pure/ML-free check (string
substring match against the question stem + options) so `scripts/kg_probe.py`
can report `novel_and_relevant_rate` alongside the raw `novel_relation_rate`.

Three layers, deliberately separated so the decision logic is unit-testable
without downloading/loading the real model:
  - `decide_novel` (pure): given already-computed entailment scores, decide
    novel vs redundant. No torch, no model, fully deterministic.
  - `is_triple_relevant` (pure): given a triple and the question text/options,
    decide whether the triple is about something the question actually
    mentions. No torch, no model, fully deterministic.
  - `is_novel_relation`: does the real work -- verbalises `triple`, scores it
    against every one of `dense_passages` as the NLI hypothesis (each passage
    as the premise), and calls `decide_novel`. Loads the NLI model lazily,
    once, into a module-level singleton (`_get_default_scorer`) -- this is
    called many times per probe run (once per retrieved triple), so reloading
    per call would be a real cost, not just an inefficiency. `scorer` is
    injectable for tests (`tests/test_kg_novelty.py` mocks it and never
    touches torch or the network).

Per CLAUDE.md's GPU-script rule, `torch.cuda.is_available()` (+ device
name/capability if true) is printed unconditionally the first time the real
model is loaded -- not at import time (importing this module must not force
a multi-GB download/load; only actually using the default scorer does).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from src.kg.traversal import Triple
from src.kg.verbalize import verbalize_triple

NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

# (premise, hypothesis) -> P(entailment), in [0, 1].
ScoreFn = Callable[[str, str], float]

_model = None
_tokenizer = None
_device: str | None = None


def decide_novel(entailment_scores: Iterable[float], threshold: float = 0.5) -> bool:
    """Pure decision logic, no model involved. True (novel) iff none of
    `entailment_scores` reaches `threshold`; False (redundant) iff at least
    one does. An empty `entailment_scores` (no dense passages at all) is
    vacuously novel -- there is nothing entailing it."""
    return not any(score >= threshold for score in entailment_scores)


def is_triple_relevant(
    triple: Triple, question_text: str, options: dict[str, str] | None = None
) -> bool:
    """Cheap, inspectable relevance check for the C1 review's "novel != useful"
    point (docs/05_DECISIONS.md, this phase's ADR): a triple is "relevant" iff
    its head or tail entity string appears (case-insensitive substring match)
    in the question stem or in any answer-option text. No ML, no embedding --
    deliberately as simple/auditable as `categorize_medqa_question`, since this
    is a diagnostic label, not a modeling contribution. This will produce some
    false positives (a short/common entity string like "Pain" substring-matching
    unrelated text) and false negatives (paraphrases, e.g. "MI" vs "myocardial
    infarction") -- both are disclosed limitations of a substring heuristic, not
    hidden ones; see the Phase 2 report for how this affects the headline
    novel_and_relevant_rate."""
    haystacks = [question_text]
    if options:
        haystacks.extend(options.values())
    haystack = " ".join(h.lower() for h in haystacks)
    return triple.head.lower() in haystack or triple.tail.lower() in haystack


def _print_cuda_status() -> None:
    import torch

    available = torch.cuda.is_available()
    print(f"[kg.novelty] torch.cuda.is_available() = {available}")
    if available:
        print(f"[kg.novelty] torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
        print(
            "[kg.novelty] torch.cuda.get_device_capability(0) = "
            f"{torch.cuda.get_device_capability(0)}"
        )


def _load_model() -> None:
    global _model, _tokenizer, _device
    if _model is not None:
        return
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _print_cuda_status()
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[kg.novelty] loading NLI model {NLI_MODEL_NAME} on {_device} ...")
    _tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(_device).eval()


def _entailment_label_index(id2label: dict[int, str]) -> int:
    for idx, label in id2label.items():
        if "entail" in label.lower():
            return idx
    raise RuntimeError(
        f"no 'entailment' label found in {NLI_MODEL_NAME}'s id2label={id2label!r}"
    )


def _default_score(premise: str, hypothesis: str) -> float:
    """Real NLI entailment score, premise=dense passage, hypothesis=the
    verbalised KG triple text. Loads/caches the model on first call."""
    import torch

    _load_model()
    assert _model is not None and _tokenizer is not None
    with torch.no_grad():
        inputs = _tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(_device)
        logits = _model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
    id2label = {int(k): v for k, v in _model.config.id2label.items()}
    entail_idx = _entailment_label_index(id2label)
    return float(probs[entail_idx].item())


def get_default_scorer() -> ScoreFn:
    """Returns the lazy-singleton real-model scorer (module-level function,
    not a fresh object) -- callers wanting the real model just call this
    once and reuse the returned function across many `is_novel_relation`
    calls."""
    return _default_score


def is_novel_relation(
    triple: Triple,
    dense_passages: list[str],
    entailment_threshold: float = 0.5,
    scorer: ScoreFn | None = None,
) -> bool:
    """True (novel) iff the verbalised `triple` is NOT entailed (>= threshold)
    by any of `dense_passages`; False (redundant) iff at least one passage
    entails it. `scorer` defaults to the real NLI model (loaded once, lazily)
    -- pass a fake for tests."""
    score_fn = scorer if scorer is not None else get_default_scorer()
    hypothesis = verbalize_triple(triple).text
    scores = [score_fn(passage, hypothesis) for passage in dense_passages]
    return decide_novel(scores, entailment_threshold)
