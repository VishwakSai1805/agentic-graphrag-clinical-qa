"""Provider-agnostic LLM gateway over LiteLLM.

Tries providers in `AppConfig.llm.fallback_chain` in order (default: local
Ollama first, free-tier Groq/Gemini as overflow). A provider whose
`api_key_env` is set but empty/missing is skipped without a network call.
Every call — cache hit or not — is cached on-disk and recorded in a
StatsTracker for token/latency accounting. Never call a provider SDK
directly from feature code; go through `chat()`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path
from threading import Lock

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel

from src.config import AppConfig, ProviderConfig, load_config

load_dotenv()
logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "base.yaml"
_config_cache: AppConfig | None = None

# Transient errors worth retrying / falling back on.
_RETRYABLE_EXC = (
    litellm.RateLimitError,
    litellm.APIConnectionError,
    litellm.Timeout,
    litellm.ServiceUnavailableError,
)


class LLMGatewayError(RuntimeError):
    """Raised when every provider in the fallback chain fails."""


class LLMCostGuardError(RuntimeError):
    """A provider/model outside the configured free-tier allowlist was about to
    be called while `llm.free_tier_only` is set (ADR-047, §0.6 Gemini check).

    This is a HARD STOP, deliberately not a "skip this provider and try the
    next one". The default `fallback_chain` ends at billable cloud providers,
    so silently falling through is precisely how a "free tier only" run would
    become a paid one -- and it would do so mid-run, after a rate limit, which
    is exactly when nobody is watching. Raised before any network call.
    """


class LLMContextBudgetError(RuntimeError):
    """The prompt plus the reserved completion space exceeds the configured
    context window (`llm.num_ctx - llm.context_completion_reserve`).

    This is a *recorded hard stop*, never a silent truncation (CLAUDE.md
    budget-guard rule). Silent truncation of an over-long prompt would drop
    whatever sits at the tail of the prompt -- for the tri-modal pipeline that
    is the KG-triple block, i.e. exactly the modality the hybrid baselines
    exist to measure -- and corrupt the numbers instead of failing loudly.
    Feature code that batches many calls (e.g. src/eval/baselines.py) catches
    this and records the item as `status="context_exceeded"`.
    """


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float


class ChatResponse(BaseModel):
    text: str
    model: str
    usage: Usage
    cached: bool
    prompt_hash: str
    latency_ms: float
    # Phase 5 (src/calibration): populated only when chat(logprobs=True) was
    # requested. One entry (the first generated token) with its logprob and
    # `top_logprobs` alternatives -- see `_extract_first_token_logprobs`.
    # None when logprobs weren't requested, or a provider returned none.
    logprobs: list[dict] | None = None


class CallStats(BaseModel):
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_ms: float = 0.0
    cost_usd: float = 0.0


class StatsTracker:
    """Accumulates token/latency/cost counters for one run. Thread-safe."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._stats = CallStats()

    def record(self, response: ChatResponse) -> None:
        with self._lock:
            self._stats.calls += 1
            if response.cached:
                self._stats.cache_hits += 1
            self._stats.prompt_tokens += response.usage.prompt_tokens
            self._stats.completion_tokens += response.usage.completion_tokens
            self._stats.total_latency_ms += response.latency_ms
            self._stats.cost_usd += response.usage.cost_usd

    def summary(self) -> CallStats:
        with self._lock:
            return self._stats.model_copy()

    def reset(self) -> None:
        with self._lock:
            self._stats = CallStats()


default_tracker = StatsTracker()


def _get_config() -> AppConfig:
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config(_DEFAULT_CONFIG_PATH)
    return _config_cache


def _count_prompt_tokens(model: str, prompt: str, system: str | None) -> int:
    """Best-effort token count of the system+user messages. `litellm.token_counter`
    uses the model's real tokenizer where it can and a tiktoken approximation
    otherwise -- close enough for a budget guard with a completion reserve."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        return int(litellm.token_counter(model=model, messages=messages))
    except Exception:  # noqa: BLE001 - never let the guard's own failure block a call
        return (len(system or "") + len(prompt)) // 4


def _cache_key(model: str, prompt: str, system: str | None, params: dict) -> str:
    payload = {"model": model, "system": system, "prompt": prompt, "params": params}
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _model_slug(model: str) -> str:
    """Filesystem-safe form of a model id (`gemini/gemini-3.8-flash` ->
    `gemini-gemini-3.8-flash`), for the cache namespace directory."""
    return "".join(ch if (ch.isalnum() or ch in "._-") else "-" for ch in model)


def _cache_dir_for(cfg: AppConfig, provider: ProviderConfig) -> Path:
    """The cache directory serving `provider` under `cfg`.

    With `llm.cache_namespace_by_provider_model` off (the default) this is the
    plain `cfg.cache_dir`, so every already-written `.cache/llm/` entry keeps
    being found. With it on, each (provider, model) gets its own subdirectory --
    the §0.6 Gemini run's replies live beside, never inside, the Ollama cache.
    """
    if not cfg.llm.cache_namespace_by_provider_model:
        return cfg.cache_dir
    return cfg.cache_dir / f"{provider.provider}__{_model_slug(provider.model)}"


def _assert_free_tier(provider: ProviderConfig, llm_cfg) -> None:
    """Cost guard (ADR-047). No-op unless `llm.free_tier_only` is set; then the
    provider/model must be on the allowlist or this raises before any call."""
    if not llm_cfg.free_tier_only:
        return
    ident = f"{provider.provider}:{provider.model}"
    if ident not in set(llm_cfg.free_tier_allowed):
        raise LLMCostGuardError(
            f"llm.free_tier_only is set and {ident!r} is not in "
            f"llm.free_tier_allowed ({sorted(llm_cfg.free_tier_allowed)}). "
            "Refusing to call it. This is a hard stop, not a fallback: add the "
            "provider:model to the allowlist only if you have confirmed it is "
            "served by a free tier for this API key."
        )


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / key[:2] / f"{key}.json"


def _read_cache(cache_dir: Path, key: str) -> dict | None:
    path = _cache_path(cache_dir, key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Corrupt cache entry %s, ignoring", path)
        return None


def _write_cache(cache_dir: Path, key: str, payload: dict) -> None:
    path = _cache_path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _provider_available(provider: ProviderConfig) -> bool:
    if provider.api_key_env is None:
        return True
    return bool(os.environ.get(provider.api_key_env))


def _logprobs_kwargs_and_routing(
    provider: ProviderConfig, top_logprobs: int
) -> tuple[dict, str, str | None]:
    """Returns (extra_kwargs, model, api_base_override) for a logprobs=True call.

    Phase 5 (src/calibration): LiteLLM's native `ollama/` provider path
    rejects `logprobs`/`top_logprobs` outright
    (`litellm.UnsupportedParamsError`) -- confirmed empirically against this
    project's own Ollama server, not assumed. Ollama's OpenAI-compatible
    endpoint (`/v1/chat/completions`) DOES return real per-token logprobs
    (also confirmed empirically), so a logprobs request against the ollama
    provider is routed through LiteLLM's `openai/` provider type pointed at
    that same server's `/v1` path instead of the native `ollama/` type. This
    still goes through LiteLLM (CLAUDE.md: "never call a provider SDK
    directly") -- it changes which LiteLLM provider *type* serves the call,
    not who serves it. Non-Ollama providers get `logprobs`/`top_logprobs`
    passed through as-is; if a fallback provider doesn't support them,
    LiteLLM raises and the existing per-provider fallback loop in `chat()`
    moves to the next provider, same as any other unsupported-param failure.
    """
    extra = {"logprobs": True, "top_logprobs": top_logprobs}
    if provider.provider == "ollama":
        bare_model = provider.model.removeprefix("ollama/").removeprefix("ollama_chat/")
        api_base = (provider.api_base or "http://localhost:11434").rstrip("/") + "/v1"
        return extra, f"openai/{bare_model}", api_base
    return extra, provider.model, provider.api_base


def _extract_first_token_logprobs(response) -> list[dict] | None:
    """Pulls the first generated token's `{token, logprob, top_logprobs}` out
    of a LiteLLM completion response, or None if the response carries no
    logprobs (a provider that silently ignored the request, or a
    zero-token completion). Only the first token matters here: every caller
    that requests logprobs uses a single-letter-answer prompt, so the option
    letter IS the first generated token."""
    lp = getattr(response.choices[0], "logprobs", None)
    if not lp or not getattr(lp, "content", None):
        return None
    first = lp.content[0]
    return [
        {
            "token": first.token,
            "logprob": first.logprob,
            "top_logprobs": [
                {"token": t.token, "logprob": t.logprob} for t in (first.top_logprobs or [])
            ],
        }
    ]


def _call_provider(
    provider: ProviderConfig,
    prompt: str,
    system: str | None,
    temperature: float,
    seed: int | None,
    max_tokens: int | None,
    max_retries: int,
    backoff_base_s: float,
    num_ctx: int | None = None,
    logprobs: bool = False,
    top_logprobs: int = 10,
) -> tuple[str, dict, list[dict] | None]:
    """Call one provider with retry/backoff. Raises on exhausted retries."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict = {
        "model": provider.model,
        "messages": messages,
        "temperature": temperature,
        "timeout": provider.timeout_s,
    }
    if provider.api_base:
        kwargs["api_base"] = provider.api_base
    # num_ctx is an Ollama-only option (llama.cpp context window). Groq/Gemini
    # reject unknown params, so only send it on the ollama path.
    if num_ctx is not None and provider.provider == "ollama":
        kwargs["num_ctx"] = num_ctx
    if seed is not None:
        kwargs["seed"] = seed
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if logprobs:
        extra, routed_model, routed_api_base = _logprobs_kwargs_and_routing(
            provider, top_logprobs
        )
        kwargs.update(extra)
        kwargs["model"] = routed_model
        if routed_api_base:
            kwargs["api_base"] = routed_api_base
        if provider.provider == "ollama":
            # LiteLLM's openai provider type requires a non-empty api_key even
            # against a local server that ignores it.
            kwargs["api_key"] = "ollama"

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = litellm.completion(**kwargs)
            text = response.choices[0].message.content or ""
            usage = response.get("usage", {}) or {}
            try:
                cost = litellm.completion_cost(completion_response=response)
            except Exception:
                cost = 0.0
            lp_out = _extract_first_token_logprobs(response) if logprobs else None
            return (
                text,
                {
                    "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
                    "completion_tokens": usage.get("completion_tokens", 0) or 0,
                    "total_tokens": usage.get("total_tokens", 0) or 0,
                    "cost_usd": cost or 0.0,
                },
                lp_out,
            )
        except _RETRYABLE_EXC as exc:
            last_exc = exc
            if attempt < max_retries:
                sleep_s = backoff_base_s * (2**attempt) + random.uniform(0, backoff_base_s)
                logger.info(
                    "Provider %s attempt %d/%d failed (%s), retrying in %.1fs",
                    provider.provider,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                    sleep_s,
                )
                time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def chat(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    max_tokens: int | None = None,
    use_cache: bool = True,
    tracker: StatsTracker | None = None,
    config: AppConfig | None = None,
    logprobs: bool = False,
    top_logprobs: int = 10,
) -> ChatResponse:
    """Chat through the configured provider fallback chain.

    Tries each provider in order; a provider is skipped if its required
    api key env var is unset/empty. Every attempted (provider, prompt,
    params) combination is cached independently, keyed on the model that
    would actually serve it.

    `logprobs=True` (Phase 5, src/calibration): requests real per-token
    logprobs on the completion (see `_logprobs_kwargs_and_routing` for how
    this is served against Ollama). Only affects the cache key when actually
    set -- a plain `chat(...)` call with `logprobs` left at its default
    computes the exact same cache key as before this parameter existed, so no
    existing `.cache/llm/` entry is invalidated by this feature.
    """
    cfg = config or _get_config()
    llm_cfg = cfg.llm
    resolved_temperature = temperature if temperature is not None else llm_cfg.temperature
    resolved_seed = seed if seed is not None else llm_cfg.seed
    resolved_max_tokens = max_tokens if max_tokens is not None else llm_cfg.max_tokens
    active_tracker = tracker or default_tracker

    # num_ctx is in `params` (and therefore the cache key) on purpose: changing
    # the context window changes model output, so a call under a bigger window
    # is a genuinely different call and must not replay a smaller-window cache
    # entry.
    params = {
        "temperature": resolved_temperature,
        "seed": resolved_seed,
        "max_tokens": resolved_max_tokens,
        "num_ctx": llm_cfg.num_ctx,
    }
    if logprobs:
        params["logprobs"] = True
        params["top_logprobs"] = top_logprobs

    # Hard context guard (CLAUDE.md budget-guard rule). Counted against the
    # first chain entry's tokenizer -- the local Ollama model is the binding
    # constraint; larger-window cloud fallbacks make this conservative, never
    # wrong. Raised before any cache read or network call.
    budget = llm_cfg.num_ctx - llm_cfg.context_completion_reserve
    ref_model = llm_cfg.fallback_chain[0].model if llm_cfg.fallback_chain else "gpt-3.5-turbo"
    n_prompt_tokens = _count_prompt_tokens(ref_model, prompt, system)
    if n_prompt_tokens > budget:
        raise LLMContextBudgetError(
            f"prompt is ~{n_prompt_tokens} tokens; context budget is {budget} "
            f"(num_ctx={llm_cfg.num_ctx} - context_completion_reserve="
            f"{llm_cfg.context_completion_reserve}). Trim the EvidenceBundle before calling."
        )

    errors: list[str] = []
    for provider in llm_cfg.fallback_chain:
        if not _provider_available(provider):
            logger.info("Skipping provider %s: %s not set", provider.provider, provider.api_key_env)
            continue

        # Cost guard BEFORE the cache read (ADR-047): a misconfigured chain must
        # surface immediately, not stay hidden behind cache hits until the first
        # miss -- which would be mid-run, on a billable provider.
        _assert_free_tier(provider, llm_cfg)

        provider_cache_dir = _cache_dir_for(cfg, provider)
        key = _cache_key(provider.model, prompt, system, params)

        if use_cache:
            cached = _read_cache(provider_cache_dir, key)
            if cached is not None:
                # A cache hit incurs no real tokens/cost/latency this run —
                # replaying the original call's usage would double-count
                # spend on every rerun of a cached question.
                response = ChatResponse(
                    text=cached["text"],
                    model=provider.model,
                    usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0, cost_usd=0.0),
                    cached=True,
                    prompt_hash=key,
                    latency_ms=0.0,
                    logprobs=cached.get("logprobs"),
                )
                active_tracker.record(response)
                return response

        start = time.monotonic()
        try:
            text, usage_dict, logprobs_out = _call_provider(
                provider,
                prompt,
                system,
                resolved_temperature,
                resolved_seed,
                resolved_max_tokens,
                llm_cfg.max_retries,
                llm_cfg.backoff_base_s,
                llm_cfg.num_ctx,
                logprobs,
                top_logprobs,
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad, we fall back
            errors.append(f"{provider.provider} ({provider.model}): {exc}")
            logger.warning("Provider %s exhausted, falling back: %s", provider.provider, exc)
            continue
        latency_ms = (time.monotonic() - start) * 1000

        if use_cache:
            _write_cache(provider_cache_dir, key, {"text": text, "logprobs": logprobs_out})

        response = ChatResponse(
            text=text,
            model=provider.model,
            usage=Usage(**usage_dict),
            cached=False,
            prompt_hash=key,
            latency_ms=latency_ms,
            logprobs=logprobs_out,
        )
        active_tracker.record(response)
        return response

    raise LLMGatewayError(
        "All providers in fallback_chain failed or were unavailable: " + "; ".join(errors)
    )
