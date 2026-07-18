"""
tests/unit/test_postgres_store.py
=================================
Unit tests for app/services/postgres_store.py (the shared write path).

This complements tests/unit/test_postgres_persistence.py, which is the
integration suite that hits a *real* Postgres for re-ingest behavior. Here we
exercise the full surface of every public function against a fake pool so the
test runs offline: success paths (SQL executed + committed), defensive no-ops
(pool None, empty/guarded inputs), and the best-effort exception branches.

get_pool() itself is also tested: it lazily builds a real pool when psycopg2
can connect and returns None (caching the failure) when it cannot.
"""

from unittest.mock import patch

import pytest

from tests.pg_helpers import FakeRow, make_pool


@pytest.fixture
def postgres_store():
    from app.services import postgres_store as module
    # Reset the lazily-cached pool so each test starts clean.
    module._pool = None
    yield module
    module._pool = None


def _patch_pool(module, pool):
    # Context manager that replaces get_pool with a stub returning `pool`,
    # and restores the real function on exit (no global state leak).
    return patch.object(module, "get_pool", return_value=pool)



def test_get_pool_returns_none_when_init_fails(postgres_store, monkeypatch):
    # Force the connection pool constructor to raise → get_pool returns None.
    def _boom(*a, **k):
        raise RuntimeError("no postgres")

    monkeypatch.setattr(postgres_store.pg_pool, "SimpleConnectionPool", _boom)
    postgres_store._pool = None
    assert postgres_store.get_pool() is None
    # The failure is cached: a second call must NOT retry the constructor.
    assert postgres_store.get_pool() is None


def test_get_pool_builds_real_pool_when_available(postgres_store, monkeypatch):
    fake_pool = object()
    monkeypatch.setattr(
        postgres_store.pg_pool, "SimpleConnectionPool", lambda **k: fake_pool
    )
    postgres_store._pool = None
    assert postgres_store.get_pool() is fake_pool



def test_record_document_upserts_and_commits(postgres_store, monkeypatch):
    pool = make_pool()
    monkeypatch.setattr(postgres_store, "get_pool", lambda: pool)
    postgres_store.record_document("doc1", "a.pdf", 10, 3, status="ready", total_pages=5)
    assert any("INSERT INTO documents" in sql for sql, _ in pool.all_executed)
    assert pool.committed is True


def test_record_document_defaults_status_to_ready(postgres_store, monkeypatch):
    pool = make_pool()
    monkeypatch.setattr(postgres_store, "get_pool", lambda: pool)
    postgres_store.record_document("doc1", "a.pdf", 10, 3)
    # The default status "ready" must be sent as a parameter.
    inserts = [params for sql, params in pool.all_executed if "INSERT INTO documents" in sql]
    assert inserts, "expected an INSERT"
    # params order: doc_id, filename, chunk_count, kg_triples, status, total_pages
    assert inserts[0][4] == "ready"


def test_record_document_noop_when_pool_unavailable(postgres_store, monkeypatch):
    monkeypatch.setattr(postgres_store, "get_pool", lambda: None)
    # Must not raise.
    postgres_store.record_document("doc1", "a.pdf", 10, 3)


def test_record_document_swallows_db_error(postgres_store, monkeypatch):
    pool = make_pool(raise_on=RuntimeError("down"))
    monkeypatch.setattr(postgres_store, "get_pool", lambda: pool)
    postgres_store.record_document("doc1", "a.pdf", 10, 3)



def test_update_document_status_updates(postgres_store, monkeypatch):
    pool = make_pool()
    monkeypatch.setattr(postgres_store, "get_pool", lambda: pool)
    postgres_store.update_document_status("doc1", "processing")
    updates = [params for sql, params in pool.all_executed if "UPDATE documents" in sql]
    assert updates and updates[0] == ("processing", "doc1")
    assert pool.committed is True


def test_update_document_status_noop_when_pool_unavailable(postgres_store, monkeypatch):
    monkeypatch.setattr(postgres_store, "get_pool", lambda: None)
    postgres_store.update_document_status("doc1", "processing")


def test_update_document_status_swallows_db_error(postgres_store, monkeypatch):
    pool = make_pool(raise_on=RuntimeError("down"))
    monkeypatch.setattr(postgres_store, "get_pool", lambda: pool)
    postgres_store.update_document_status("doc1", "processing")



def test_record_chunk_metadata_executemany(postgres_store, monkeypatch):
    pool = make_pool()
    monkeypatch.setattr(postgres_store, "get_pool", lambda: pool)
    chunks = [
        {"chunk_index": 0, "text": "a", "word_count": 5, "page": 1},
        {"chunk_index": 1, "text": "b", "word_count": 6, "page": 2},
    ]
    postgres_store.record_chunk_metadata("doc1", chunks)
    assert any("INSERT INTO document_chunks" in sql for sql, _ in pool.all_executed)
    assert pool.committed is True


def test_record_chunk_metadata_noop_for_empty(postgres_store, monkeypatch):
    pool = make_pool()
    monkeypatch.setattr(postgres_store, "get_pool", lambda: pool)
    postgres_store.record_chunk_metadata("doc1", [])
    assert not any("document_chunks" in sql for sql, _ in pool.all_executed)


def test_record_chunk_metadata_noop_when_pool_unavailable(postgres_store, monkeypatch):
    monkeypatch.setattr(postgres_store, "get_pool", lambda: None)
    postgres_store.record_chunk_metadata("doc1", [{"chunk_index": 0}])


def test_record_chunk_metadata_swallows_db_error(postgres_store, monkeypatch):
    pool = make_pool(raise_on=RuntimeError("down"))
    monkeypatch.setattr(postgres_store, "get_pool", lambda: pool)
    postgres_store.record_chunk_metadata("doc1", [{"chunk_index": 0}])



def test_record_user_returns_id(postgres_store, monkeypatch):
    pool = make_pool(script=[FakeRow({"id": 99})])
    monkeypatch.setattr(postgres_store, "get_pool", lambda: pool)
    assert postgres_store.record_user("alice") == 99
    assert any("INSERT INTO users" in sql for sql, _ in pool.all_executed)


def test_record_user_noop_when_pool_unavailable(postgres_store, monkeypatch):
    monkeypatch.setattr(postgres_store, "get_pool", lambda: None)
    assert postgres_store.record_user("alice") is None


def test_record_user_swallows_db_error(postgres_store, monkeypatch):
    pool = make_pool(raise_on=RuntimeError("down"))
    monkeypatch.setattr(postgres_store, "get_pool", lambda: pool)
    assert postgres_store.record_user("alice") is None



def test_record_memory_writes(postgres_store, monkeypatch):
    pool = make_pool()
    monkeypatch.setattr(postgres_store, "get_pool", lambda: pool)
    postgres_store.record_memory("u1", "user prefers concise answers")
    assert any("INSERT INTO memories" in sql for sql, _ in pool.all_executed)


@pytest.mark.parametrize("user_id,text", [
    ("", "something"),          # empty user_id
    ("anonymous", "something"), # anonymous user
    ("u1", ""),                 # empty memory text
])
def test_record_memory_noop_for_invalid_inputs(postgres_store, monkeypatch, user_id, text):
    pool = make_pool()
    monkeypatch.setattr(postgres_store, "get_pool", lambda: pool)
    postgres_store.record_memory(user_id, text)
    assert not any("INSERT INTO memories" in sql for sql, _ in pool.all_executed)


def test_record_memory_noop_when_pool_unavailable(postgres_store, monkeypatch):
    monkeypatch.setattr(postgres_store, "get_pool", lambda: None)
    postgres_store.record_memory("u1", "remember this")


def test_record_memory_swallows_db_error(postgres_store, monkeypatch):
    pool = make_pool(raise_on=RuntimeError("down"))
    monkeypatch.setattr(postgres_store, "get_pool", lambda: pool)
    postgres_store.record_memory("u1", "remember this")



def test_record_conversation_writes(postgres_store, monkeypatch):
    pool = make_pool()
    monkeypatch.setattr(postgres_store, "get_pool", lambda: pool)
    postgres_store.record_conversation("s1", "u1", "question?", "answer!")
    assert any("INSERT INTO conversations" in sql for sql, _ in pool.all_executed)
    assert pool.committed is True


def test_record_conversation_noop_for_empty_question(postgres_store, monkeypatch):
    pool = make_pool()
    monkeypatch.setattr(postgres_store, "get_pool", lambda: pool)
    postgres_store.record_conversation("s1", "u1", "", "answer!")
    assert not any("INSERT INTO conversations" in sql for sql, _ in pool.all_executed)


def test_record_conversation_noop_when_pool_unavailable(postgres_store, monkeypatch):
    monkeypatch.setattr(postgres_store, "get_pool", lambda: None)
    postgres_store.record_conversation("s1", "u1", "q?", "a!")


def test_record_conversation_swallows_db_error(postgres_store, monkeypatch):
    pool = make_pool(raise_on=RuntimeError("down"))
    monkeypatch.setattr(postgres_store, "get_pool", lambda: pool)
    postgres_store.record_conversation("s1", "u1", "q?", "a!")



def test_record_query_writes(postgres_store, monkeypatch):
    pool = make_pool()
    monkeypatch.setattr(postgres_store, "get_pool", lambda: pool)
    postgres_store.record_query(
        "s1", "u1", "doc1", "question?", "answer!",
        "single", 0.9, 0.8, 1, {"retrieval": 12.0},
    )
    assert any("INSERT INTO query_history" in sql for sql, _ in pool.all_executed)
    assert pool.committed is True


def test_record_query_noop_when_pool_unavailable(postgres_store, monkeypatch):
    monkeypatch.setattr(postgres_store, "get_pool", lambda: None)
    postgres_store.record_query("s1", "u1", "doc1", "q?", "a!", "single", None, None, 0, {})


def test_record_query_swallows_db_error(postgres_store, monkeypatch):
    pool = make_pool(raise_on=RuntimeError("down"))
    monkeypatch.setattr(postgres_store, "get_pool", lambda: pool)
    postgres_store.record_query("s1", "u1", "doc1", "q?", "a!", "single", None, None, 0, {})
