"""
tests/unit/test_llm_client.py
=============================
Unit tests for app/services/llm_client.py.

Covers:
  * the process-wide token accumulator (reset / get_and_reset)
  * _record_usage for dict- and object-style usage_metadata (and None)
  * _LLMWrapper.invoke / ainvoke with and without the Redis LLM cache
  * attribute delegation via __getattr__
  * make_llm() factory returns a wrapper exposing model_name
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage


@pytest.fixture
def llm_client():
    from app.services import llm_client as module
    module.reset_token_usage()
    yield module
    module.reset_token_usage()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# token accumulator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_reset_and_get_token_usage(llm_client):
    llm_client._token_usage["prompt_tokens"] = 12
    llm_client._token_usage["completion_tokens"] = 3
    snapshot = llm_client.get_and_reset_token_usage()
    assert snapshot == {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    # Must reset to zero after reading.
    assert llm_client.get_and_reset_token_usage() == {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _record_usage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_record_usage_from_dict_metadata(llm_client):
    msg = AIMessage(content="x", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    llm_client._record_usage(msg)
    snap = llm_client.get_and_reset_token_usage()
    assert snap["prompt_tokens"] == 10
    assert snap["completion_tokens"] == 5


def test_record_usage_from_object_metadata(llm_client):
    # AIMessage itself validates usage_metadata as a dict, so to exercise the
    # object-style branch of _record_usage we wrap a plain object that exposes
    # input_tokens / output_tokens attributes (as a real UsageMetadata would).
    class Usage:
        input_tokens = 7
        output_tokens = 2

    class FakeMessage:
        usage_metadata = Usage()

    llm_client._record_usage(FakeMessage())
    snap = llm_client.get_and_reset_token_usage()
    assert snap["prompt_tokens"] == 7
    assert snap["completion_tokens"] == 2


def test_record_usage_noop_without_metadata(llm_client):
    msg = AIMessage(content="x")  # no usage_metadata
    llm_client._record_usage(msg)
    snap = llm_client.get_and_reset_token_usage()
    assert snap["prompt_tokens"] == 0
    assert snap["completion_tokens"] == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _LLMWrapper.invoke / ainvoke
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_wrapper_invoke_without_cache(llm_client):
    client = MagicMock()
    client.model_name = "gpt-4o"
    client.invoke.return_value = AIMessage(
        content="answer", usage_metadata={"input_tokens": 4, "output_tokens": 1, "total_tokens": 5}
    )
    wrapper = llm_client._LLMWrapper(client)
    result = wrapper.invoke([("human", "hi")])
    assert result.content == "answer"
    client.invoke.assert_called_once()
    # Token usage must be recorded through the wrapper.
    assert llm_client.get_and_reset_token_usage()["prompt_tokens"] == 4


def test_wrapper_invoke_cache_hit_skips_client(llm_client):
    client = MagicMock()
    client.model_name = "gpt-4o"
    with patch.object(llm_client, "llm_cache_get", return_value="cached answer"):
        wrapper = llm_client._LLMWrapper(client, use_cache=True)
        result = wrapper.invoke([("human", "hi")])
    # Cache hit → wrapped returns an AIMessage without calling the client.
    assert isinstance(result, AIMessage)
    assert result.content == "cached answer"
    client.invoke.assert_not_called()


def test_wrapper_invoke_cache_miss_writes_cache(llm_client):
    client = MagicMock()
    client.model_name = "gpt-4o"
    client.invoke.return_value = AIMessage(
        content="fresh", usage_metadata={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
    )
    with patch.object(llm_client, "llm_cache_get", return_value=None) as mock_get, \
         patch.object(llm_client, "llm_cache_set") as mock_set:
        wrapper = llm_client._LLMWrapper(client, use_cache=True)
        result = wrapper.invoke([("human", "q")])
    assert result.content == "fresh"
    mock_get.assert_called_once()
    # On a miss with a non-empty result, the response is written to the cache.
    mock_set.assert_called_once()
    assert mock_set.call_args.args[2] == "fresh"


async def test_wrapper_ainvoke_without_cache(llm_client):
    client = MagicMock()
    client.model_name = "gpt-4o"
    client.ainvoke = AsyncMock(return_value=AIMessage(
        content="async answer", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    ))
    wrapper = llm_client._LLMWrapper(client)
    result = await wrapper.ainvoke([("human", "hi")])
    assert result.content == "async answer"
    client.ainvoke.assert_called_once()
    assert llm_client.get_and_reset_token_usage()["completion_tokens"] == 1


async def test_wrapper_ainvoke_cache_hit(llm_client):
    client = MagicMock()
    client.model_name = "gpt-4o"
    with patch.object(llm_client, "llm_cache_get", return_value="cached"):
        wrapper = llm_client._LLMWrapper(client, use_cache=True)
        result = await wrapper.ainvoke([("human", "hi")])
    assert result.content == "cached"
    client.ainvoke.assert_not_called()


async def test_wrapper_ainvoke_cache_miss_writes_cache(llm_client):
    client = MagicMock()
    client.model_name = "gpt-4o"
    client.ainvoke = AsyncMock(return_value=AIMessage(
        content="fresh async",
        usage_metadata={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    ))
    # Cache miss → the real client runs, usage is recorded, and the response
    # is written back to the cache. Mirrors the sync invoke path (line 145).
    with patch.object(llm_client, "llm_cache_get", return_value=None) as mock_get, \
         patch.object(llm_client, "llm_cache_set") as mock_set:
        wrapper = llm_client._LLMWrapper(client, use_cache=True)
        result = await wrapper.ainvoke([("human", "q")])
    assert result.content == "fresh async"
    mock_get.assert_called_once()
    mock_set.assert_called_once()
    assert mock_set.call_args.args[2] == "fresh async"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# attribute delegation + factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_wrapper_delegates_unknown_attrs(llm_client):
    client = MagicMock()
    client.model_name = "gpt-4o"
    client.some_attribute = "delegated"
    wrapper = llm_client._LLMWrapper(client)
    # Not set on the wrapper → must fall through to the underlying client.
    assert wrapper.some_attribute == "delegated"


def test_wrapper_exposes_model_name(llm_client):
    client = MagicMock()
    client.model_name = "gpt-4o-mini"
    wrapper = llm_client._LLMWrapper(client)
    assert wrapper.model_name == "gpt-4o-mini"


def test_make_llm_returns_wrapper(llm_client):
    fake_client = MagicMock()
    fake_client.model_name = "gpt-4o"
    with patch.object(llm_client, "ChatOpenAI", return_value=fake_client) as mock_chat:
        wrapper = llm_client.make_llm(model="gpt-4o", temperature=0.0, use_cache=True)
    assert isinstance(wrapper, llm_client._LLMWrapper)
    assert wrapper.model_name == "gpt-4o"
    # make_llm passed through the model + temperature to ChatOpenAI.
    _, kwargs = mock_chat.call_args
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["temperature"] == 0.0
