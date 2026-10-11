"""Tokenizer-based chunking, shared by every corpus source (PubMed abstracts and
USMLE textbooks alike -- PROJECT_RULES.md/roadmap: identical chunking policy for both).

Chunk size and overlap are config-driven (configs/base.yaml:
retrieval.chunk_tokens / retrieval.chunk_overlap), not hardcoded here.

The tokenizer is an injectable dependency: anything exposing `encode(text) ->
list[int]` and `decode(ids) -> str]` works. `load_default_tokenizer()` lazily
wraps a real HF `AutoTokenizer` (only imported/downloaded on first use), so
`tests/test_chunking.py` can inject a lightweight in-memory stub and never
needs network access in CI -- the same "never hit a provider in CI" principle
PROJECT_RULES.md states for LLM calls applies to downloading a tokenizer from the
HF Hub in a unit test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

DEFAULT_TOKENIZER_MODEL = "ncbi/MedCPT-Article-Encoder"


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal tokenizer interface chunk_text() needs."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


@dataclass(frozen=True)
class TokenChunk:
    text: str
    index: int
    n_tokens: int
    start_token: int


class _HFTokenizerAdapter:
    """Adapts a HF `PreTrainedTokenizerBase` to the minimal Tokenizer protocol,
    stripping special tokens on both sides so chunk boundaries are pure content.
    """

    def __init__(self, hf_tokenizer: object) -> None:
        self._hf = hf_tokenizer

    def encode(self, text: str) -> list[int]:
        return self._hf.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]

    def decode(self, ids: list[int]) -> str:
        return self._hf.decode(ids, skip_special_tokens=True)  # type: ignore[attr-defined]


def load_default_tokenizer(model_name: str = DEFAULT_TOKENIZER_MODEL) -> Tokenizer:
    """Lazily loads a real HF tokenizer. Only call this from build scripts, never
    from unit tests (network/model download) -- inject a stub there instead.
    """
    from transformers import AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _HFTokenizerAdapter(hf_tokenizer)


def chunk_text(
    text: str,
    chunk_tokens: int = 256,
    chunk_overlap: int = 32,
    tokenizer: Tokenizer | None = None,
) -> list[TokenChunk]:
    """Splits `text` into overlapping token windows of `chunk_tokens` with
    `chunk_overlap` tokens of overlap between consecutive chunks. Pure function
    over an injected tokenizer -- no I/O, no model loading, unless `tokenizer`
    is left None (in which case a real HF tokenizer is lazily loaded).

    Returns an empty list for empty/whitespace-only text.
    """
    if chunk_tokens <= 0:
        raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_tokens:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than "
            f"chunk_tokens ({chunk_tokens})"
        )

    if not text or not text.strip():
        return []

    if tokenizer is None:
        tokenizer = load_default_tokenizer()

    ids = tokenizer.encode(text)
    if not ids:
        return []

    stride = chunk_tokens - chunk_overlap
    n = len(ids)
    chunks: list[TokenChunk] = []
    start = 0
    idx = 0
    while start < n:
        end = min(start + chunk_tokens, n)
        piece_ids = ids[start:end]
        piece_text = tokenizer.decode(piece_ids).strip()
        if piece_text:
            chunks.append(
                TokenChunk(text=piece_text, index=idx, n_tokens=len(piece_ids), start_token=start)
            )
            idx += 1
        if end == n:
            break
        start += stride

    return chunks
