"""
app/services/redis_cache.py
=============================
Thin Redis cache/temp-state layer (Phase 5 of the ARAP architecture).

Redis already plays two roles in ARAP (LangGraph session checkpointer and the
Celery broker). This module adds the three remaining roles the spec lists:

  - Retrieval cache   retrieval:<hash>  → cached top-k chunks for a question
  - LLM cache         llm:<hash>        → cached model output for a prompt
  - Temp pipeline state  processing:<doc_id> → "ingest in flight" flag

Every function is BEST-EFFORT: if Redis is unreachable, the cache is silently
disabled and callers fall back to their normal (uncached) path. No call site
should ever fail because of this module — that is the whole point of keeping
caching out of the critical path.

Keys are SHA-256 hashes of their inputs (truncated to 32 hex chars) so they are
constant-length and never contain control characters.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

# Lazily-created module-level client + an "is Redis available?" latch.
# Once Redis fails to connect we stop retrying every call (avoids a thundering
# herd of connection attempts on every request).
_client = None
_enabled: bool | None = None


def _get_client():
    """Return a connected Redis client, or None if Redis is unavailable."""
    global _client, _enabled
    if _enabled is False:
        return None
    if _client is None:
        try:
            import redis as redis_lib
            _client = redis_lib.from_url(
                settings.redis_url, socket_connect_timeout=2, socket_timeout=2
            )
            _client.ping()
            _enabled = True
        except Exception as e:  # pragma: no cover - depends on Redis
            logger.warning("Redis cache unavailable (caching disabled): %s", e)
            _enabled = False
            return None
    return _client


def cache_enabled() -> bool:
    """Whether the cache is currently usable (for call-site logging only)."""
    return _get_client() is not None


def _hash(*parts: str) -> str:
    """Stable 32-char hex hash of the joined parts."""
    joined = "|".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Retrieval cache
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def retrieval_cache_get(question: str, doc_id: str | None) -> list[dict] | None:
    """
    Return cached top-k chunks for (question, doc_id), or None on miss/error.

    Cached chunks are plain JSON-serializable dicts (the retrieval agent's
    retrieved_chunks contract), so json round-trips losslessly.
    """
    c = _get_client()
    if c is None or not question:
        return None
    key = "retrieval:" + _hash(question, doc_id or "")
    try:
        raw = c.get(key)
        if raw:
            return json.loads(raw)
    except Exception as e:  # pragma: no cover - depends on Redis
        logger.debug("retrieval_cache_get failed (non-fatal): %s", e)
    return None


def retrieval_cache_set(
    question: str,
    doc_id: str | None,
    chunks: list[dict],
    ttl: int | None = None,
) -> None:
    """Cache the retrieved chunks for (question, doc_id). Best-effort."""
    c = _get_client()
    if c is None or not question or not chunks:
        return
    key = "retrieval:" + _hash(question, doc_id or "")
    try:
        c.set(
            key,
            json.dumps(chunks, ensure_ascii=False),
            ex=ttl if ttl is not None else settings.retrieval_cache_ttl_seconds,
        )
    except Exception as e:  # pragma: no cover - depends on Redis
        logger.debug("retrieval_cache_set failed (non-fatal): %s", e)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM response cache
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def llm_cache_get(model: str, messages: Any) -> str | None:
    """
    Return cached model output text for (model, messages), or None.

    `messages` may be a list of LangChain message objects or plain strings;
    we serialize by (type, content) so two semantically-identical prompts
    hit the same cache key.
    """
    c = _get_client()
    if c is None or not messages:
        return None
    key = "llm:" + _hash(model, _serialize_messages(messages))
    try:
        raw = c.get(key)
        if raw:
            return raw.decode("utf-8") if isinstance(raw, bytes) else raw
    except Exception as e:  # pragma: no cover - depends on Redis
        logger.debug("llm_cache_get failed (non-fatal): %s", e)
    return None


def llm_cache_set(
    model: str,
    messages: Any,
    text: str,
    ttl: int | None = None,
) -> None:
    """Cache model output `text` for (model, messages). Best-effort."""
    c = _get_client()
    if c is None or not messages or not text:
        return
    key = "llm:" + _hash(model, _serialize_messages(messages))
    try:
        c.set(
            key,
            text,
            ex=ttl if ttl is not None else settings.llm_cache_ttl_seconds,
        )
    except Exception as e:  # pragma: no cover - depends on Redis
        logger.debug("llm_cache_set failed (non-fatal): %s", e)


def _serialize_messages(messages: Any) -> str:
    """Render LangChain messages (or plain strings) into a stable string."""
    parts: list[str] = []
    for m in messages:
        mtype = getattr(m, "type", None) or type(m).__name__
        content = getattr(m, "content", None)
        if content is None:
            content = str(m)
        parts.append(f"{mtype}:{content}")
    return "\n".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Temporary pipeline state (ingest in-flight flag)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def pipeline_state_set(doc_id: str, ttl: int | None = None) -> None:
    """Mark `doc_id` as currently being ingested (processing:<doc_id>)."""
    c = _get_client()
    if c is None or not doc_id:
        return
    key = "processing:" + doc_id
    try:
        c.set(key, "1", ex=ttl or settings.pipeline_state_ttl_seconds)
    except Exception as e:  # pragma: no cover - depends on Redis
        logger.debug("pipeline_state_set failed (non-fatal): %s", e)


def pipeline_state_is_processing(doc_id: str) -> bool:
    """True if `doc_id` is currently being ingested."""
    c = _get_client()
    if c is None or not doc_id:
        return False
    key = "processing:" + doc_id
    try:
        return c.exists(key) > 0
    except Exception:  # pragma: no cover - depends on Redis
        return False


def pipeline_state_clear(doc_id: str) -> None:
    """Clear the processing:<doc_id> flag (call at end of ingest)."""
    c = _get_client()
    if c is None or not doc_id:
        return
    key = "processing:" + doc_id
    try:
        c.delete(key)
    except Exception as e:  # pragma: no cover - depends on Redis
        logger.debug("pipeline_state_clear failed (non-fatal): %s", e)
