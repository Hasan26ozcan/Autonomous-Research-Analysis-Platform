"""
tests/unit/test_analytics.py
============================
Unit tests for app/services/analytics.py (Phase 11 read-only dashboard queries).

All queries are best-effort and return empty/zeroed structures when Postgres
is unreachable or errors. We patch ``get_pool()`` with a fake pool that yields
scripted ``RealDictCursor``-style rows. ``FakeRow`` supports both
``row["col"]`` and ``row[0]`` access so ``_fetchval`` (which indexes by 0)
and the result readers (which index by name) are both satisfied.
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from tests.pg_helpers import FakeRow, make_pool


@pytest.fixture
def analytics():
    from app.services import analytics as module
    return module


def _patch_pool(module, pool):
    return patch.object(module, "get_pool", return_value=pool)



def test_summary_returns_aggregates(analytics):
    # Scripted values consumed in order by the 7 _fetchval calls inside summary().
    # Each scalar is wrapped in a FakeRow so ``row[0]`` (used by _fetchval)
    # returns the value, mirroring a single-column query result.
    pool = make_pool(script=[
        FakeRow({"v": 5}), FakeRow({"v": 3}), FakeRow({"v": 2}),
        FakeRow({"v": 0.91}), FakeRow({"v": 123.45}),
        FakeRow({"v": 0.8}), FakeRow({"v": 0.75}),
    ])
    with _patch_pool(analytics, pool):
        result = analytics.summary()
    assert result == {
        "total_queries": 5,
        "total_documents": 3,
        "total_evals": 2,
        "avg_latency_ms": 123.45,
        "avg_faithfulness": 0.91,
        "avg_precision": 0.8,
        "avg_recall": 0.75,
    }


def test_summary_zeroes_none_when_no_rows(analytics):
    # Empty script → every _fetchval returns None → int(None or 0) == 0.
    pool = make_pool(script=[])
    with _patch_pool(analytics, pool):
        result = analytics.summary()
    assert result["total_queries"] == 0
    assert result["avg_faithfulness"] is None
    assert result["avg_precision"] is None
    assert result["avg_recall"] is None


def test_summary_returns_empty_when_pool_unavailable(analytics):
    with _patch_pool(analytics, None):
        result = analytics.summary()
    assert result["total_queries"] == 0
    assert result["avg_latency_ms"] is None


def test_summary_returns_empty_on_db_error(analytics):
    pool = make_pool(raise_on=RuntimeError("down"))
    with _patch_pool(analytics, pool):
        result = analytics.summary()
    assert result["total_queries"] == 0



def test_top_documents_returns_rows(analytics):
    rows = [
        FakeRow({"doc_id": "d1", "filename": "a.pdf", "query_count": 9, "avg_faithfulness": 0.88}),
        FakeRow({"doc_id": "d2", "filename": "b.pdf", "query_count": 4, "avg_faithfulness": None}),
    ]
    pool = make_pool(script=rows)
    with _patch_pool(analytics, pool):
        result = analytics.top_documents(limit=10)
    assert len(result) == 2
    assert result[0]["doc_id"] == "d1"
    assert result[0]["query_count"] == 9
    assert result[0]["avg_faithfulness"] == pytest.approx(0.88)
    # None faithfulness must survive through, not become 0.
    assert result[1]["avg_faithfulness"] is None


def test_top_documents_respects_limit_param(analytics):
    pool = make_pool(script=[])
    with _patch_pool(analytics, pool):
        analytics.top_documents(limit=7)
    # The LIMIT placeholder must carry the requested value.
    assert any(
        "LIMIT" in sql and params == (7,)
        for sql, params in pool.all_executed
    )


def test_top_documents_empty_when_pool_unavailable(analytics):
    with _patch_pool(analytics, None):
        assert analytics.top_documents() == []


def test_top_documents_empty_on_db_error(analytics):
    pool = make_pool(raise_on=RuntimeError("down"))
    with _patch_pool(analytics, pool):
        assert analytics.top_documents() == []



def test_eval_trend_returns_runs(analytics):
    rows = [
        FakeRow({
            "id": 1,
            "created_at": datetime(2026, 1, 2, 3, 4, 5),
            "num_questions": 20,
            "status": "completed",
            "average_faithfulness": 0.92,
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })
    ]
    pool = make_pool(script=rows)
    with _patch_pool(analytics, pool):
        result = analytics.eval_trend(limit=20)
    assert len(result) == 1
    r = result[0]
    assert r["run_id"] == 1
    assert r["num_questions"] == 20
    assert r["status"] == "completed"
    assert r["average_faithfulness"] == pytest.approx(0.92)
    assert r["created_at"] == "2026-01-02T03:04:05"
    assert r["token_usage"]["prompt_tokens"] == 10


def test_eval_trend_handles_null_created_at(analytics):
    rows = [FakeRow({
        "id": 2, "created_at": None, "num_questions": 5,
        "status": "running", "average_faithfulness": None, "token_usage": None,
    })]
    pool = make_pool(script=rows)
    with _patch_pool(analytics, pool):
        result = analytics.eval_trend()
    assert result[0]["created_at"] is None


def test_eval_trend_empty_when_pool_unavailable(analytics):
    with _patch_pool(analytics, None):
        assert analytics.eval_trend() == []


def test_eval_trend_empty_on_db_error(analytics):
    pool = make_pool(raise_on=RuntimeError("down"))
    with _patch_pool(analytics, pool):
        assert analytics.eval_trend() == []



def test_round_helper(analytics):
    assert analytics._round(None) is None
    assert analytics._round(0.123456789) == pytest.approx(0.1235)
    assert analytics._round("not a number") is None
    assert analytics._round(3) == pytest.approx(3.0)
