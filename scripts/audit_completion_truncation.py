"""Audit every cached completion for truncation at the generator's token budget.

ADR-049. Answers one question with evidence instead of assumption: did the local
7B runs hit their own `llm.max_tokens` ceiling? The §0.6 stronger-model check
raises the Gemini budget to 1024 against base.yaml's 512 as a "parity
correction" (ADR-047), and that argument is only sound if the 7B never ran out
of room. It did, rarely -- this script is where that number comes from.

Truncation here is NOT visible in `status_counts`: these prompts put the answer
letter before the citation list, so a completion cut at the cap still parses as
`ok` and merely loses its citations (ADR-049 Decision 4). The only reliable
signal is the completion-length distribution itself, which is why this reads the
cache rather than the run records.

    uv run python -m scripts.audit_completion_truncation

Writes results/phase7_gemini/completion_truncation_audit.json. Reads only the
cache; touches no run directory.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / ".cache" / "llm"
OUT = REPO / "results" / "phase7_gemini" / "completion_truncation_audit.json"

# The local generator's own tokenizer -- a proxy tokenizer puts the spike in the
# wrong place and the whole point is whether lengths land exactly on the cap.
TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"
LOCAL_MAX_TOKENS = 512  # configs/base.yaml: llm.max_tokens
NEAR_CAP = LOCAL_MAX_TOKENS - 1  # 511: a stripped trailing token still counts


def _local_cache_entries() -> list[Path]:
    """Ollama entries only: `<cache>/<shard>/<hash>.json`, two levels deep.

    Namespaced providers (ADR-048, `cache_namespace_by_provider_model`) sit a
    level deeper under `<cache>/<provider>__<model>/<shard>/<hash>.json` and are
    deliberately excluded -- this audit is about the 7B arm.
    """
    return sorted(p for p in CACHE.glob("*/*.json") if len(p.parent.name) == 2)


def _shape(text: str) -> str:
    """Classify a completion by response shape.

    The cache stores only `{text, logprobs}` -- no prompt, no run id -- so a
    truncated entry CANNOT be attributed to the run that produced it. Content
    matching does not help either: the retrieved evidence a response quotes is
    shared across variants by construction, so a KG triple in a truncated
    completion appears in most runs' predictions. Shape is the only honest
    signal about which pipeline stage was truncated, and it says nothing about
    which cell.
    """
    head = text[:200]
    if '"answer"' in head:
        return "answer"  # a QA answer call (baseline tier or an agent's final answer)
    if '"cells"' in head or '"verdict"' in head:
        return "agent_json"  # audit/hypothesis structured output
    return "agent_prose"  # audit reasoning that never reached its JSON block


def main() -> None:
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    entries = _local_cache_entries()
    if not entries:
        raise SystemExit(f"no local cache entries under {CACHE}")

    lengths: list[int] = []
    at_cap: list[dict] = []
    for path in entries:
        text = (json.loads(path.read_text(encoding="utf-8")).get("text") or "")
        n = len(tok.encode(text))
        lengths.append(n)
        if n >= NEAR_CAP:
            at_cap.append(
                {
                    "cache_entry": path.relative_to(REPO).as_posix(),
                    "completion_tokens": n,
                    "shape": _shape(text),
                    "head": text[:80],
                    "tail": text[-80:],
                }
            )

    lengths.sort()

    def pct(q: float) -> int:
        return lengths[min(int(q * len(lengths)), len(lengths) - 1)]

    # The gap below the spike is the evidence that this is truncation rather
    # than a heavy tail: a natural distribution does not leave 490-510 empty.
    hist = Counter(lengths)
    gap_below = [n for n in range(460, LOCAL_MAX_TOKENS + 1) if hist[n]]

    payload = {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ"),
        "adr": "ADR-049",
        "question": (
            "did the local 7B runs truncate at configs/base.yaml llm.max_tokens=512?"
        ),
        "answer": "yes, rarely -- see truncated_fraction",
        "tokenizer": TOKENIZER,
        "cache_dir": CACHE.relative_to(REPO).as_posix(),
        "local_max_tokens": LOCAL_MAX_TOKENS,
        "n_completions": len(lengths),
        "distribution": {
            "mean": round(statistics.mean(lengths), 1),
            "median": pct(0.50),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": lengths[-1],
        },
        "n_at_exact_cap": hist[LOCAL_MAX_TOKENS],
        "n_at_or_above": len(at_cap),
        "truncated_fraction": round(len(at_cap) / len(lengths), 5),
        "occupied_lengths_460_to_cap": gap_below,
        "truncated_by_shape": dict(Counter(e["shape"] for e in at_cap)),
        "attribution_to_runs": (
            "IMPOSSIBLE with the current cache format, stated rather than "
            "guessed: entries are keyed by hash and carry no run id, and the "
            "evidence they quote is shared across variants, so content "
            "matching returns nearly every run. Shape says which STAGE "
            "truncated; nothing says which CELL."
        ),
        "truncated_entries": at_cap,
        "note": (
            "status_counts does not move for these: the answer letter precedes "
            "the citation list, so a truncated completion parses as ok with an "
            "empty citation set (ADR-049 Decision 4)."
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    d = payload["distribution"]
    print(f"completions: {payload['n_completions']}")
    print(f"mean {d['mean']} median {d['median']} p95 {d['p95']} p99 {d['p99']} max {d['max']}")
    print(f"at exact cap ({LOCAL_MAX_TOKENS}): {payload['n_at_exact_cap']}")
    print(f">= {NEAR_CAP}: {payload['n_at_or_above']} ({payload['truncated_fraction']:.2%})")
    print(f"occupied lengths in [460, {LOCAL_MAX_TOKENS}]: {gap_below}")
    print(f"by shape: {payload['truncated_by_shape']}")
    print(f"wrote {OUT.relative_to(REPO).as_posix()}")


if __name__ == "__main__":
    main()
