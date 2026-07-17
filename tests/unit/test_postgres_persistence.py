"""
tests/unit/test_postgres_persistence.py
=========================================
Integration tests for the PostgreSQL write path (psycopg2).

These exercise the real re-ingest behavior that unit tests can't:
  - record_document flips a document back to "ready" on re-ingest (BUG 3)
  - record_chunk_metadata upserts (does not fail/duplicate) on re-ingest (BUG 4)

They require a reachable PostgreSQL whose schema was created by
scripts/init_db.sql (this happens automatically on `docker compose up`).
When no Postgres is reachable (e.g. a local checkout without the stack
running), the whole module skips — so it never fails a unit run offline.

The tests use freshly generated doc_ids and delete only their own rows in
teardown, so they're safe to run against a live dev database.
"""

import uuid

import psycopg2
import pytest

from app.core.config import settings
from app.services.postgres_store import (
    record_chunk_metadata,
    record_document,
    update_document_status,
)


def _connect():
    try:
        return psycopg2.connect(settings.postgres_url, connect_timeout=3)
    except Exception:
        return None


@pytest.fixture(scope="module")
def conn():
    c = _connect()
    if c is None:
        pytest.skip("PostgreSQL not reachable — skipping persistence integration tests")
    # The schema must already exist (init_db.sql runs at container start).
    try:
        with c.cursor() as cur:
            cur.execute("SELECT 1 FROM documents LIMIT 1")
    except Exception:
        pytest.skip("documents table absent (init_db.sql not applied)")
    yield c
    c.close()


def _status(c, doc_id):
    with c.cursor() as cur:
        cur.execute("SELECT status FROM documents WHERE doc_id=%s", (doc_id,))
        row = cur.fetchone()
    return row[0] if row else None


def _chunk_rows(c, doc_id):
    with c.cursor() as cur:
        cur.execute(
            "SELECT text, token_count FROM document_chunks "
            "WHERE document_id=%s AND chunk_index=0",
            (doc_id,),
        )
        return cur.fetchone()


def _delete(conn, doc_id):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM document_chunks WHERE document_id=%s", (doc_id,))
        cur.execute("DELETE FROM documents WHERE doc_id=%s", (doc_id,))
    conn.commit()


def test_record_document_status_flips_on_reingest(conn):
    """Re-ingesting the same doc_id must return it to 'ready', not strand it."""
    doc_id = f"test_{uuid.uuid4().hex}"
    try:
        record_document(doc_id, "a.pdf", 1, 0, status="ready")
        assert _status(conn, doc_id) == "ready"

        update_document_status(doc_id, "processing")
        assert _status(conn, doc_id) == "processing"

        # Re-ingest: the ON CONFLICT DO UPDATE must refresh status -> ready (BUG 3).
        record_document(doc_id, "a.pdf", 2, 0, status="ready")
        assert _status(conn, doc_id) == "ready"
    finally:
        _delete(conn, doc_id)


def test_record_chunk_metadata_upserts_on_reingest(conn):
    """Re-ingesting the same chunk_index must UPDATE, not raise/duplicate (BUG 4)."""
    doc_id = f"test_{uuid.uuid4().hex}"
    try:
        record_chunk_metadata(doc_id, [
            {"chunk_index": 0, "text": "original", "word_count": 10, "page": 1},
        ])
        # Second insert with the same (document_id, chunk_index): must upsert.
        record_chunk_metadata(doc_id, [
            {"chunk_index": 0, "text": "updated", "word_count": 12, "page": 1},
        ])

        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM document_chunks WHERE document_id=%s",
                (doc_id,),
            )
            count = cur.fetchone()[0]
        assert count == 1, "re-ingest should upsert, not create a duplicate row"

        text, token_count = _chunk_rows(conn, doc_id)
        assert text == "updated"
        assert token_count == 12
    finally:
        _delete(conn, doc_id)
