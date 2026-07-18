"""
tests/unit/test_log_store.py
=============================
Unit tests for app/services/log_store.py (Phase 10 structured logging).

All three writers (api / worker / pipeline) are best-effort: they no-op when
the pool is unavailable, skip empty input, and swallow DB errors. We patch
``get_pool()`` with a fake pool and assert on the SQL that gets executed (or
the absence of it on the defensive branches).
"""

from unittest.mock import MagicMock, patch

import pytest

from tests.pg_helpers import make_pool


@pytest.fixture
def log_store():
    from app.services import log_store as module
    return module


def _patch_pool(module, pool):
    return patch.object(module, "get_pool", return_value=pool)



def test_log_api_writes_row(log_store):
    pool = make_pool()
    with _patch_pool(log_store, pool):
        log_store.log_api("GET", "/query", 200, 12.5, user_id="u1", session_id="s1")
    assert any("INSERT INTO api_log" in sql for sql, _ in pool.all_executed)
    assert pool.committed is True


def test_log_api_handles_db_error(log_store):
    pool = make_pool(raise_on=RuntimeError("down"))
    with _patch_pool(log_store, pool):
        # Must not raise.
        log_store.log_api("GET", "/query", 200, 1.0)


def test_log_api_noop_when_pool_unavailable(log_store):
    with _patch_pool(log_store, None):
        log_store.log_api("GET", "/query", 200, 1.0)



def test_log_worker_writes_row(log_store):
    pool = make_pool()
    with _patch_pool(log_store, pool):
        log_store.log_worker("ingest", "doc1", "info", "started")
    assert any("INSERT INTO worker_log" in sql for sql, _ in pool.all_executed)
    assert pool.committed is True


def test_log_worker_handles_db_error(log_store):
    pool = make_pool(raise_on=RuntimeError("down"))
    with _patch_pool(log_store, pool):
        log_store.log_worker("ingest", "doc1", "error", "boom")


def test_log_worker_noop_when_pool_unavailable(log_store):
    with _patch_pool(log_store, None):
        log_store.log_worker("ingest", "doc1", "info", "x")



def test_log_pipeline_batch_writes_one_row_per_node(log_store):
    pool = make_pool()
    with _patch_pool(log_store, pool):
        log_store.log_pipeline_batch(
            "sess1", "single", {"retrieval": 120.0, "generation": 340.5}
        )
    assert any("INSERT INTO pipeline_log" in sql for sql, _ in pool.all_executed)
    assert pool.committed is True


def test_log_pipeline_batch_skips_non_numeric_latency(log_store):
    pool = make_pool()
    with _patch_pool(log_store, pool):
        log_store.log_pipeline_batch(
            "sess1", "single", {"retrieval": 120.0, "broken": "n/a"}
        )
    # Only the numeric node produces a row; 'broken' is filtered out.
    rows = [params for sql, params in pool.all_executed if "INSERT INTO pipeline_log" in sql]
    assert rows, "expected at least the numeric node to be written"
    # The executemany call packs the rows as its params (already in `rows`).
    written = rows[0]
    assert all(isinstance(ms, float) for *_, ms, _qt in written)


def test_log_pipeline_batch_noop_for_empty_latency(log_store):
    pool = MagicMock()
    with _patch_pool(log_store, pool):
        log_store.log_pipeline_batch("sess1", "single", {})
    pool.getconn.assert_not_called()


def test_log_pipeline_batch_noop_when_pool_unavailable(log_store):
    with _patch_pool(log_store, None):
        log_store.log_pipeline_batch("sess1", "single", {"retrieval": 1.0})


def test_log_pipeline_batch_handles_db_error(log_store):
    pool = make_pool(raise_on=RuntimeError("down"))
    with _patch_pool(log_store, pool):
        log_store.log_pipeline_batch("sess1", "single", {"retrieval": 1.0})


def test_log_pipeline_batch_noop_when_all_latency_non_numeric(log_store):
    """A latency dict whose values are all non-numeric yields no rows → early
    return (line 110) before any SQL is executed."""
    pool = make_pool()
    with _patch_pool(log_store, pool):
        log_store.log_pipeline_batch("sess1", "single", {"a": "n/a", "b": "skip"})
    rows = [p for sql, p in pool.all_executed if "INSERT INTO pipeline_log" in sql]
    assert rows == []
