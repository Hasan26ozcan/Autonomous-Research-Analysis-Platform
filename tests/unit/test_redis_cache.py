"""
tests/unit/test_redis_cache.py
================================
Unit tests for app/services/redis_cache.py (Phase 5 cache + temp state layer).

The module keeps a module-level ``_client`` / ``_enabled`` latch. Every test
resets that latch, then either:
  * patches ``redis.from_url`` with a fake client so the happy path runs, or
  * forces the disabled latch so the best-effort no-op paths run, or
  * makes the fake raise so the swallowed-exception paths run.

No real Redis is contacted.
"""

import json

from unittest.mock import MagicMock, patch

import pytest


class FakeRedis:
    """Minimal in-memory stand-in for a redis-py client."""

    def __init__(self):
        self.store: dict = {}
        self.published: list = []
        self.ping_calls = 0
        self.raise_on: Exception | None = None

    def ping(self) -> bool:
        self.ping_calls += 1
        return True

    def get(self, key):
        if self.raise_on is not None:
            raise self.raise_on
        return self.store.get(key)

    def set(self, key, value, ex=None):
        if self.raise_on is not None:
            raise self.raise_on
        self.store[key] = value

    def exists(self, key) -> int:
        if self.raise_on is not None:
            raise self.raise_on
        return 1 if key in self.store else 0

    def delete(self, key):
        if self.raise_on is not None:
            raise self.raise_on
        self.store.pop(key, None)

    def publish(self, channel, message):
        self.published.append((channel, message))


@pytest.fixture
def redis_cache():
    from app.services import redis_cache as module
    module._client = None
    module._enabled = None
    yield module
    module._client = None
    module._enabled = None


def _patch_client(module, fake: FakeRedis):
    """Point _get_client at a fake by patching redis.from_url."""
    return patch("redis.from_url", return_value=fake)


def _disable(module):
    module._client = None
    module._enabled = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# internals: _hash, _serialize_messages, _get_client / cache_enabled
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_hash_is_stable_and_32_chars(redis_cache):
    h1 = redis_cache._hash("a", "b", "c")
    h2 = redis_cache._hash("a", "b", "c")
    assert h1 == h2
    assert len(h1) == 32
    assert redis_cache._hash("a", "b", "x") != h1


def test_serialize_messages_handles_objects_and_strings(redis_cache):
    class Msg:
        type = "human"
        content = "hello"

    class NoContent:
        type = "ai"

    out = redis_cache._serialize_messages([Msg(), "plain", NoContent()])
    assert "human:hello" in out
    assert "plain" in out
    # No content → falls back to str(m).
    assert "NoContent" in out


def test_get_client_connects_and_caches(redis_cache):
    fake = FakeRedis()
    with _patch_client(redis_cache, fake):
        first = redis_cache._get_client()
        second = redis_cache._get_client()
    assert first is fake is second
    assert fake.ping_calls == 1  # only the first call pings


def test_get_client_returns_none_when_disabled_latch_set(redis_cache):
    _disable(redis_cache)
    assert redis_cache._get_client() is None


def test_cache_enabled_true_when_connected(redis_cache):
    fake = FakeRedis()
    with _patch_client(redis_cache, fake):
        assert redis_cache.cache_enabled() is True


def test_cache_enabled_false_when_connection_fails(redis_cache):
    with patch("redis.from_url", side_effect=ConnectionError("refused")):
        assert redis_cache.cache_enabled() is False
    # Latch prevents retry.
    assert redis_cache._enabled is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# retrieval cache
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_retrieval_cache_set_and_get(redis_cache):
    fake = FakeRedis()
    with _patch_client(redis_cache, fake):
        chunks = [{"text": "x", "score": 0.9}]
        redis_cache.retrieval_cache_set("what is X?", "doc1", chunks)
        got = redis_cache.retrieval_cache_get("what is X?", "doc1")
    assert got == chunks
    # Key is namespaced + hashed.
    assert any(k.startswith("retrieval:") for k in fake.store)


def test_retrieval_cache_get_miss_returns_none(redis_cache):
    fake = FakeRedis()
    with _patch_client(redis_cache, fake):
        assert redis_cache.retrieval_cache_get("nope", "doc1") is None


def test_retrieval_cache_set_noop_for_empty_question(redis_cache):
    fake = FakeRedis()
    with _patch_client(redis_cache, fake):
        redis_cache.retrieval_cache_set("", "doc1", [{"text": "x"}])
    assert fake.store == {}


def test_retrieval_cache_noop_when_disabled(redis_cache):
    _disable(redis_cache)
    assert redis_cache.retrieval_cache_get("q", "doc1") is None
    # set must not raise either.
    redis_cache.retrieval_cache_set("q", "doc1", [{"text": "x"}])


def test_retrieval_cache_swallows_get_error(redis_cache):
    fake = FakeRedis()
    fake.raise_on = RuntimeError("down")
    with _patch_client(redis_cache, fake):
        assert redis_cache.retrieval_cache_get("q", "doc1") is None


def test_retrieval_cache_swallows_set_error(redis_cache):
    fake = FakeRedis()
    fake.raise_on = RuntimeError("down")
    with _patch_client(redis_cache, fake):
        redis_cache.retrieval_cache_set("q", "doc1", [{"text": "x"}])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# llm cache
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_llm_cache_set_and_get(redis_cache):
    fake = FakeRedis()
    with _patch_client(redis_cache, fake):
        redis_cache.llm_cache_set("gpt", [("human", "hi")], "response text")
        got = redis_cache.llm_cache_get("gpt", [("human", "hi")])
    assert got == "response text"
    assert any(k.startswith("llm:") for k in fake.store)


def test_llm_cache_get_decodes_bytes(redis_cache):
    fake = FakeRedis()
    # Store bytes at the exact key the module will compute, so the get hits.
    key = "llm:" + redis_cache._hash("gpt", redis_cache._serialize_messages("anything"))
    fake.store[key] = b"bytes-response"
    with _patch_client(redis_cache, fake):
        assert redis_cache.llm_cache_get("gpt", "anything") == "bytes-response"


def test_llm_cache_noop_for_empty_messages(redis_cache):
    fake = FakeRedis()
    with _patch_client(redis_cache, fake):
        redis_cache.llm_cache_set("gpt", [], "text")
    assert fake.store == {}


def test_llm_cache_noop_when_disabled(redis_cache):
    _disable(redis_cache)
    assert redis_cache.llm_cache_get("gpt", "m") is None
    redis_cache.llm_cache_set("gpt", "m", "text")


def test_llm_cache_swallows_error(redis_cache):
    fake = FakeRedis()
    fake.raise_on = RuntimeError("down")
    with _patch_client(redis_cache, fake):
        assert redis_cache.llm_cache_get("gpt", "m") is None
        redis_cache.llm_cache_set("gpt", "m", "text")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# pipeline temp state
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_pipeline_state_set_is_processing_clear(redis_cache):
    fake = FakeRedis()
    with _patch_client(redis_cache, fake):
        redis_cache.pipeline_state_set("doc1")
        # Key is namespaced while the flag is set.
        assert any(k.startswith("processing:") for k in fake.store)
        assert redis_cache.pipeline_state_is_processing("doc1") is True
        redis_cache.pipeline_state_clear("doc1")
        assert redis_cache.pipeline_state_is_processing("doc1") is False
    # After clear the flag is gone.
    assert not any(k.startswith("processing:") for k in fake.store)


def test_pipeline_state_noop_when_disabled(redis_cache):
    _disable(redis_cache)
    assert redis_cache.pipeline_state_is_processing("doc1") is False
    redis_cache.pipeline_state_set("doc1")
    redis_cache.pipeline_state_clear("doc1")


def test_pipeline_state_swallows_error(redis_cache):
    fake = FakeRedis()
    fake.raise_on = RuntimeError("down")
    with _patch_client(redis_cache, fake):
        redis_cache.pipeline_state_set("doc1")
        assert redis_cache.pipeline_state_is_processing("doc1") is False
        redis_cache.pipeline_state_clear("doc1")
