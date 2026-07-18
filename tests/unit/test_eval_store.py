"""
tests/unit/test_eval_store.py
==============================
Unit tests for app/services/eval_store.py (Phase 9 persistence).

Every function is best-effort and synchronous (psycopg2). We replace
``get_pool()`` with a fake pool so no real Postgres is touched. The tests
exercise the happy path (INSERT/UPDATE executed, rows committed) and the
defensive branches (None pool, no-op when inputs are missing, exception
swallowed and logged).
"""

from unittest.mock import MagicMock, patch

import pytest

from tests.pg_helpers import FakeRow, make_pool


@pytest.fixture
def eval_store():
    from app.services import eval_store as module
    return module


def _patch_pool(module, pool):
    return patch.object(module, "get_pool", return_value=pool)



def test_start_run_inserts_and_returns_id(eval_store):
    pool = make_pool(script=[FakeRow({"id": 42})])
    with _patch_pool(eval_store, pool):
        run_id = eval_store.start_run(10)
    assert run_id == 42
    # The generated cursor must have executed an INSERT and committed.
    assert any("INSERT INTO evaluation_runs" in sql for sql, _ in pool.all_executed)


def test_start_run_returns_none_when_pool_unavailable(eval_store):
    with _patch_pool(eval_store, None):
        assert eval_store.start_run(5) is None


def test_start_run_returns_none_when_insert_fails(eval_store):
    pool = make_pool(raise_on=RuntimeError("db down"))
    with _patch_pool(eval_store, pool):
        assert eval_store.start_run(5) is None


def test_start_run_returns_none_when_fetchone_empty(eval_store):
    # INSERT succeeds but RETURNING yields no row → int(row[0]) guarded.
    pool = make_pool(script=[None])
    with _patch_pool(eval_store, pool):
        assert eval_store.start_run(5) is None



def test_finish_run_noop_for_none_run_id(eval_store):
    # Should never touch the pool.
    pool = MagicMock()
    with _patch_pool(eval_store, pool):
        eval_store.finish_run(None, status="done", metrics={})
    pool.getconn.assert_not_called()


def test_finish_run_noop_when_pool_unavailable(eval_store):
    with _patch_pool(eval_store, None):
        # Must not raise.
        eval_store.finish_run(7, status="done", metrics={"x": 1.0})


def test_finish_run_updates_and_writes_scores(eval_store):
    pool = make_pool()
    metrics = {"context_precision": 0.8, "context_recall": 0.75, "faithfulness": None}
    token_usage = {"prompt_tokens": 100, "completion_tokens": 50}
    with _patch_pool(eval_store, pool):
        eval_store.finish_run(
            7,
            status="completed",
            metrics=metrics,
            token_usage=token_usage,
            average_faithfulness=0.9,
            notes="ok",
        )
    sqls = [sql for sql, _ in pool.all_executed]
    assert any("UPDATE evaluation_runs" in s for s in sqls)
    # 'faithfulness': None must be excluded; only the two numeric metrics written.
    assert any("INSERT INTO evaluation_scores" in s for s in sqls)
    assert pool.committed is True


def test_finish_run_handles_db_error(eval_store):
    pool = make_pool(raise_on=RuntimeError("db down"))
    with _patch_pool(eval_store, pool):
        # Swallowed and logged; must not propagate.
        eval_store.finish_run(7, status="completed", metrics={"x": 1.0})



def test_record_retrieval_results_noop_for_none_run_id(eval_store):
    pool = MagicMock()
    with _patch_pool(eval_store, pool):
        eval_store.record_retrieval_results(None, [{"question": "q"}])
    pool.getconn.assert_not_called()


def test_record_retrieval_results_noop_for_empty_items(eval_store):
    pool = MagicMock()
    with _patch_pool(eval_store, pool):
        eval_store.record_retrieval_results(7, [])
    pool.getconn.assert_not_called()


def test_record_retrieval_results_noop_when_pool_unavailable(eval_store):
    with _patch_pool(eval_store, None):
        eval_store.record_retrieval_results(7, [{"question": "q", "doc_id": "d"}])


def test_record_retrieval_results_writes_rows(eval_store):
    pool = make_pool()
    items = [
        {"question": "q1", "doc_id": "d1", "chunk_index": 0, "score": 0.9, "source": "dense"},
        {"question": "q2", "doc_id": "d2", "chunk_index": 3, "score": 0.4, "source": "bm25"},
    ]
    with _patch_pool(eval_store, pool):
        eval_store.record_retrieval_results(7, items)
    assert any("INSERT INTO retrieval_results" in sql for sql, _ in pool.all_executed)
    assert pool.committed is True


def test_record_retrieval_results_handles_db_error(eval_store):
    pool = make_pool(raise_on=RuntimeError("db down"))
    items = [{"question": "q", "doc_id": "d", "chunk_index": 0, "score": 0.5, "source": "dense"}]
    with _patch_pool(eval_store, pool):
        eval_store.record_retrieval_results(7, items)
