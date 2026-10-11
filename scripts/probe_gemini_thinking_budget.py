"""Can Gemini's reasoning budget be separated from the visible-output budget?

ADR-049. If it can, the section 0.6 token-budget asymmetry (Gemini 1024 vs the
7B's 512) can be REMOVED rather than disclosed: give reasoning its own
allowance and hold visible output at 512, exactly matching the 7B. That is
strictly better than documenting an asymmetry, so it is worth testing before
settling for the disclosure.

This probe calls litellm directly, which feature code must never do
(PROJECT_RULES.md section 2). It is a diagnostic, not a call site: `chat()`'s `Usage`
carries prompt/completion/total only, and the question here is specifically
whether `completion_tokens_details.reasoning_tokens` responds to a budget
parameter. The free-tier allowlist is asserted below before any call, so the
cost guard is not bypassed along with the gateway.

    uv run python -m scripts.probe_gemini_thinking_budget

Writes results/phase7_gemini/thinking_budget_probe.json. Makes ~4 real API
calls on the free tier.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import litellm
from dotenv import load_dotenv
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "phase7_gemini" / "thinking_budget_probe.json"

MODEL = "gemini/gemini-3.8-flash"
FREE_TIER_ALLOWED = {  # mirrors configs/gemini_base.yaml llm.free_tier_allowed
    "gemini:gemini/gemini-3.8-flash",
    "gemini:gemini/gemini-2.5-flash",
    "gemini:gemini/gemini-3.5-flash-lite",
}
VISIBLE_BUDGET = 512  # configs/base.yaml llm.max_tokens -- the 7B's budget

# A prompt that actually provokes reasoning; a trivial one under-reports.
PROMPT = (
    "A 63-year-old man presents with progressive dyspnoea, orthopnoea and "
    "bilateral pitting oedema. JVP is elevated. Echo shows an ejection "
    "fraction of 30%. Which single agent most improves survival?\n"
    "A. Furosemide\nB. Digoxin\nC. Carvedilol\nD. Amlodipine\n\n"
    'Reply as JSON: {"answer": "<letter>", "rationale": "<one sentence>"}'
)


def _details(resp) -> dict:
    usage = resp.get("usage", {}) or {}
    det = usage.get("completion_tokens_details") or {}
    if not isinstance(det, dict):
        det = det.model_dump() if hasattr(det, "model_dump") else dict(det)
    reasoning = int(det.get("reasoning_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    text = (resp["choices"][0]["message"].get("content") or "").strip()
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "visible_tokens_approx": max(completion - reasoning, 0),
        "finish_reason": resp["choices"][0].get("finish_reason"),
        "text_empty": not text,
        "text_head": text[:120],
    }


def _sent_thinking_config(**kwargs) -> dict | None:
    """What litellm would actually put in the request's thinkingConfig."""
    params: dict = {}
    VertexGeminiConfig().map_openai_params(
        non_default_params=dict(kwargs),
        optional_params=params,
        model=MODEL.split("/", 1)[1],
        drop_params=False,
    )
    return params.get("thinkingConfig")


def _attempt(name: str, hypothesis: str, **call_kwargs) -> dict:
    row: dict = {"attempt": name, "hypothesis": hypothesis, "call_kwargs": {}}
    for k, v in call_kwargs.items():
        row["call_kwargs"][k] = v
    try:
        row["thinking_config_litellm_would_send"] = _sent_thinking_config(
            **{k: v for k, v in call_kwargs.items() if k in ("thinking", "reasoning_effort")}
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic, record and continue
        row["thinking_config_litellm_would_send"] = f"<mapping raised: {exc}>"

    # Free-tier 503/429s are load, not evidence. Retry them, so a transient
    # outage cannot be mistaken for "the parameter was rejected" -- which is
    # exactly the conclusion this probe exists to reach or refuse.
    last: Exception | None = None
    for attempt_no in range(6):
        try:
            resp = litellm.completion(
                model=MODEL,
                messages=[{"role": "user", "content": PROMPT}],
                temperature=0.0,
                **call_kwargs,
            )
            row["ok"] = True
            row["api_attempts"] = attempt_no + 1
            row.update(_details(resp))
            return row
        except (litellm.ServiceUnavailableError, litellm.RateLimitError, litellm.Timeout) as exc:
            last = exc
            time.sleep(4.0 * (2**attempt_no))
        except Exception as exc:  # noqa: BLE001 - a rejection IS the finding here
            row["ok"] = False
            row["error_type"] = type(exc).__name__
            row["error"] = str(exc)[:400]
            return row
    row["ok"] = False
    row["error_type"] = type(last).__name__ if last else "Unknown"
    row["error"] = f"transient failure after 6 attempts (load, not a rejection): {str(last)[:300]}"
    row["transient"] = True
    return row


def main() -> None:
    # `--recompute-verdict` re-derives the verdict fields from the attempts
    # already stored in the output file, making NO API call. The section 0.6
    # canary needs the free-tier quota, so re-deriving a conclusion from calls
    # already made is preferable to spending four more to reach the same one.
    if "--recompute-verdict" in sys.argv:
        if not OUT.exists():
            raise SystemExit(f"{OUT} does not exist -- run the probe first")
        stored = json.loads(OUT.read_text(encoding="utf-8"))
        _finish(stored["attempts"], stored, recomputed=True)
        return

    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is not set (.env or environment)")
    ident = f"gemini:{MODEL}"
    if ident not in FREE_TIER_ALLOWED:
        raise SystemExit(f"{ident} is not on the free-tier allowlist -- refusing to call")

    attempts = [
        _attempt(
            "baseline",
            "no thinking parameter: reasoning and visible text share max_tokens",
            max_tokens=VISIBLE_BUDGET,
        ),
        _attempt(
            "thinking_budget_128",
            "if budget_tokens reaches the API, reasoning should be capped near 128",
            max_tokens=VISIBLE_BUDGET,
            thinking={"type": "enabled", "budget_tokens": 128},
        ),
        _attempt(
            "reasoning_effort_low",
            "Gemini 3 maps effort to thinkingLevel, not a token budget",
            max_tokens=VISIBLE_BUDGET,
            reasoning_effort="low",
        ),
        _attempt(
            "raw_thinking_config",
            "escape hatch: push thinkingBudget straight through as a raw kwarg",
            max_tokens=VISIBLE_BUDGET,
            thinkingConfig={"thinkingBudget": 128, "includeThoughts": False},
        ),
    ]

    _finish(attempts, None, recomputed=False)


def _finish(attempts: list[dict], prior: dict | None, *, recomputed: bool) -> None:
    """Derive the verdict from `attempts`, write the output file, print a summary."""
    # The verdict rests on the STATIC mapping, not on the token counts. Gemini
    # exposes no seed (ADR-047), so single-call reasoning_tokens differences sit
    # within run-to-run noise and cannot carry a conclusion on their own. If no
    # request carries a numeric `thinkingBudget`, no token count can rescue it.
    budget_reaches_api = any(
        isinstance(a.get("thinking_config_litellm_would_send"), dict)
        and "thinkingBudget" in a["thinking_config_litellm_would_send"]
        for a in attempts
    )
    base = next(a for a in attempts if a["attempt"] == "baseline")
    capped = next(a for a in attempts if a["attempt"] == "thinking_budget_128")
    token_evidence_consistent = bool(
        base.get("ok")
        and capped.get("ok")
        and capped.get("reasoning_tokens", 0) > 0
        and capped["reasoning_tokens"] < base.get("reasoning_tokens", 0) * 0.75
    )
    separable = bool(budget_reaches_api and token_evidence_consistent)
    reasoning_spread = sorted(a["reasoning_tokens"] for a in attempts if a.get("ok"))

    payload = {
        "generated_utc": (
            prior["generated_utc"]
            if prior and "generated_utc" in prior
            else datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
        ),
        "verdict_recomputed_utc": (
            datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ") if recomputed else None
        ),
        "verdict_recomputed_without_new_calls": recomputed,
        "adr": "ADR-049",
        "model": MODEL,
        "visible_budget_tested": VISIBLE_BUDGET,
        "question": (
            "can the reasoning budget be set separately from max_output_tokens, so "
            "visible output can be held at the 7B's 512 and the asymmetry removed?"
        ),
        "reasoning_budget_separable": separable,
        "numeric_thinking_budget_reaches_api": budget_reaches_api,
        "token_evidence_consistent_with_a_cap": token_evidence_consistent,
        "reasoning_tokens_observed": reasoning_spread,
        "verdict_basis": (
            "static mapping first: litellm routes Gemini 3+ to thinkingLevel and "
            "drops budget_tokens, so no request carries a numeric thinkingBudget. "
            "Reasoning-token counts are corroboration only -- Gemini has no seed, "
            "so single-call differences sit within noise (ADR-047)."
        ),
        "caveat_reasoning_effort": (
            "reasoning_effort=low DOES lower reasoning tokens, but it is a coarse "
            "behavioural setting, not a budget: it makes the generator think less, "
            "which changes the very thing section 0.6 measures. Not a legitimate fix."
        ),
        "attempts": attempts,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for a in attempts:
        print(f"--- {a['attempt']}")
        print(f"    thinkingConfig litellm sends: {a.get('thinking_config_litellm_would_send')}")
        if a["ok"]:
            print(
                f"    completion={a['completion_tokens']} "
                f"reasoning={a['reasoning_tokens']} "
                f"visible~={a['visible_tokens_approx']} "
                f"finish={a['finish_reason']} empty={a['text_empty']}"
            )
        else:
            print(f"    REJECTED {a['error_type']}: {a['error'][:160]}")
    print(f"\nnumeric thinkingBudget reaches the API: {budget_reaches_api}")
    print(f"reasoning_tokens observed across attempts: {reasoning_spread}")
    print(f"reasoning_budget_separable = {separable}")
    print(f"wrote {OUT.relative_to(REPO).as_posix()}")


if __name__ == "__main__":
    main()
