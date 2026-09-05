import pytest

from src.retrieval.chunking import chunk_text


class StubTokenizer:
    """Deterministic whitespace tokenizer for tests -- no network, no model
    download, no GPU. Token id == position of the word in the *last* text
    passed to encode(); decode() looks words up from that same call, so a
    single chunk_text() call (one encode + many decode calls on slices of
    the same id sequence) round-trips correctly.
    """

    def __init__(self) -> None:
        self._words: list[str] = []

    def encode(self, text: str) -> list[int]:
        self._words = text.split()
        return list(range(len(self._words)))

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._words[i] for i in ids)


def test_chunk_text_basic_windowing():
    text = " ".join(f"w{i}" for i in range(10))  # w0 .. w9
    chunks = chunk_text(text, chunk_tokens=4, chunk_overlap=1, tokenizer=StubTokenizer())
    # stride = 3: windows start at 0, 3, 6; the window starting at 6 already
    # reaches token 10 (the end), so no tiny trailing remainder chunk is made.
    assert [c.start_token for c in chunks] == [0, 3, 6]
    assert chunks[0].text == "w0 w1 w2 w3"
    assert chunks[1].text == "w3 w4 w5 w6"
    assert chunks[-1].text == "w6 w7 w8 w9"


def test_chunk_text_consecutive_chunks_overlap_by_configured_amount():
    text = " ".join(f"w{i}" for i in range(12))
    overlap = 2
    chunks = chunk_text(text, chunk_tokens=5, chunk_overlap=overlap, tokenizer=StubTokenizer())
    assert len(chunks) >= 2
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        prev_words = prev.text.split()
        next_words = nxt.text.split()
        assert prev_words[-overlap:] == next_words[:overlap]


def test_chunk_text_short_text_yields_single_chunk():
    text = "just three words"
    chunks = chunk_text(text, chunk_tokens=256, chunk_overlap=32, tokenizer=StubTokenizer())
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].n_tokens == 3


def test_chunk_text_empty_text_returns_no_chunks():
    assert chunk_text("", tokenizer=StubTokenizer()) == []
    assert chunk_text("   ", tokenizer=StubTokenizer()) == []


def test_chunk_text_indices_are_sequential():
    text = " ".join(f"w{i}" for i in range(20))
    chunks = chunk_text(text, chunk_tokens=6, chunk_overlap=2, tokenizer=StubTokenizer())
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_overlap_must_be_smaller_than_chunk_tokens():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_tokens=10, chunk_overlap=10, tokenizer=StubTokenizer())
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_tokens=10, chunk_overlap=11, tokenizer=StubTokenizer())


def test_chunk_tokens_must_be_positive():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_tokens=0, chunk_overlap=0, tokenizer=StubTokenizer())
