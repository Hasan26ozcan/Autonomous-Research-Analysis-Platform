"""
app/services/llm_client.py
============================
Single source of truth for building ChatOpenAI clients across ARAP.

WHY THIS EXISTS
----------------
Every agent previously constructed its own ChatOpenAI() inline with
copy-pasted settings. That made two cross-cutting concerns impossible to
tune in one place:

  1. Rate-limit headroom — timeouts / retry counts had to be edited in 8 spots.
  2. 429 resilience — most clients used max_retries=0, so a transient Groq
     429 (HTTP 429 "Rate limit reached") killed the whole chunk/answer instead
     of being retried.

This factory centralizes client creation. It pulls api_key / base_url /
model / timeout / retries from settings so changing the provider or tuning
rate-limit behavior is a one-line change in app/core/config.py.

TOKEN TRACKING (Phase 9)
------------------------
Every client returned by make_llm() is a thin _LLMWrapper that records
prompt/completion token usage into a thread-local accumulator on each
invoke/ainvoke. The orchestrator resets the accumulator at the start of a
query and reads it back at the end (get_and_reset_token_usage()), so token
cost is captured per request without a global shared counter.

LLM CACHE (Phase 5)
-------------------
Pass use_cache=True to cache the model's response by (model, prompt) in
Redis (key llm:<hash>). Intended for deterministic, expensive, structured
calls (routing, HyDE, KG extraction) — NOT for open-ended generation. The
cache is best-effort: if Redis is down, calls just run normally.

429 RECOVERY
------------
The OpenAI SDK retries RateLimitError (429) and APITimeoutError with
exponential backoff that honors the server's Retry-After header. Setting
max_retries > 0 (via settings.llm_max_retries) therefore lets transient
Groq rate limits be absorbed automatically. This is the reactive safety net
that complements the PROACTIVE SlidingWindowRateLimiter in rate_limiter.py
(which paces requests before they're sent).
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.services.redis_cache import llm_cache_get, llm_cache_set

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Token usage accumulator (Phase 9)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A simple process-wide counter. The orchestrator resets it at the START of
# each query() and reads it back at the END (after the graph finishes running
# inside asyncio.to_thread). It is therefore accurate for the common case
# where queries run sequentially; under heavy concurrent load two overlapping
# queries could intermix their counts — acceptable for a best-effort metric.

_token_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}


def reset_token_usage() -> None:
    """Zero the token counter (call at the start of a query)."""
    _token_usage["prompt_tokens"] = 0
    _token_usage["completion_tokens"] = 0


def get_and_reset_token_usage() -> dict:
    """Return {prompt_tokens, completion_tokens, total_tokens} and reset."""
    snapshot = {
        "prompt_tokens": _token_usage["prompt_tokens"],
        "completion_tokens": _token_usage["completion_tokens"],
        "total_tokens": _token_usage["prompt_tokens"] + _token_usage["completion_tokens"],
    }
    reset_token_usage()
    return snapshot


def _record_usage(message: Any) -> None:
    """Add an AIMessage's token usage to the accumulator."""
    um = getattr(message, "usage_metadata", None)
    if not um:
        return
    if isinstance(um, dict):
        p = int(um.get("input_tokens", 0) or 0)
        c = int(um.get("output_tokens", 0) or 0)
    else:
        p = int(getattr(um, "input_tokens", 0) or 0)
        c = int(getattr(um, "output_tokens", 0) or 0)
    _token_usage["prompt_tokens"] += p
    _token_usage["completion_tokens"] += c


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Wrapper — adds token tracking + optional Redis LLM cache, stays transparent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _LLMWrapper:
    """
    Delegating wrapper around a ChatOpenAI client.

    Exposes `.invoke` / `.ainvoke` (with token tracking + optional caching)
    and forwards every other attribute/method to the underlying client via
    __getattr__, so callers that only do `client.invoke([...])` — every agent
    in ARAP — are unaffected, while tracing/callbacks still flow through.
    """

    def __init__(self, client: ChatOpenAI, use_cache: bool = False):
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_use_cache", use_cache)
        object.__setattr__(self, "model_name", getattr(client, "model_name", ""))

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails; delegate to the real client.
        return getattr(self._client, name)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        client = self._client
        if self._use_cache:
            cached = llm_cache_get(client.model_name, input)
            if cached is not None:
                return AIMessage(content=cached)
        result = client.invoke(input, config=config, **kwargs)
        _record_usage(result)
        if self._use_cache and getattr(result, "content", None):
            llm_cache_set(client.model_name, input, result.content)
        return result

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        client = self._client
        if self._use_cache:
            cached = llm_cache_get(client.model_name, input)
            if cached is not None:
                return AIMessage(content=cached)
        result = await client.ainvoke(input, config=config, **kwargs)
        _record_usage(result)
        if self._use_cache and getattr(result, "content", None):
            llm_cache_set(client.model_name, input, result.content)
        return result


def make_llm(
    model: str | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_retries: int | None = None,
    request_timeout: int | None = None,
    response_format: Any | None = None,
    use_cache: bool = False,
) -> _LLMWrapper:
    """
    Build a ChatOpenAI client with ARAP's shared, 429-resilient defaults.

    Per-agent overrides (model, temperature, max_tokens, response_format) are
    passed explicitly. Everything else falls back to settings so provider
    config lives in one place.

    Args:
        model:           Model id. Defaults to settings.llm_model.
        temperature:     Sampling temperature. Defaults to settings.temperature.
        max_tokens:      Output cap. If None, the SDK/model default is used.
        max_retries:     SDK retry count for 429/timeout. Defaults to
                         settings.llm_max_retries (reactive 429 recovery).
        request_timeout: Per-call timeout (s). Defaults to
                         settings.llm_request_timeout.
        response_format: Optional structured-output mode
                         (e.g. {"type": "json_object"}). If None, omitted.
        use_cache:       If True, cache responses in Redis by (model, prompt)
                         (Phase 5). For deterministic structured calls only.

    Returns:
        A _LLMWrapper around a configured ChatOpenAI instance. Behaves like
        a ChatOpenAI for `.invoke`/`.ainvoke` and records token usage.
    """
    kwargs: dict[str, Any] = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if response_format is not None:
        kwargs["response_format"] = response_format

    client = ChatOpenAI(  # type: ignore[call-arg]
        model=model or settings.llm_model,
        api_key=settings.openai_api_key,  # type: ignore[arg-type]
        base_url=settings.llm_base_url,
        temperature=temperature if temperature is not None else settings.temperature,
        max_retries=max_retries if max_retries is not None else settings.llm_max_retries,
        request_timeout=(
            request_timeout
            if request_timeout is not None
            else settings.llm_request_timeout
        ),
        **kwargs,
    )

    return _LLMWrapper(client, use_cache=use_cache)
