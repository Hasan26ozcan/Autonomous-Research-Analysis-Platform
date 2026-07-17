"""
PostgreSQL persistence for ARAP.

Writes to the tables defined in scripts/init_db.sql:
  - documents:     one row per ingested PDF (doc_id, filename, counts)
  - document_chunks: chunk metadata mirror (text + token_count + page)
  - users:         one row per ingested-by user
  - memories:      long-term memory facts mirrored from Mem0 (per user)
  - conversations: one row per Q&A turn (Phase 8)
  - query_history: one row per /query call (audit log + RAGAS test set)

These tables existed in the schema from day one (scripts/init_db.sql runs
automatically on Postgres' first boot) but nothing in the application ever
wrote to them - the migration ran, the tables were just never fed. This
module is that missing write path.

Design notes:
  - A single lazily-created psycopg2 connection pool is shared process-wide.
  - psycopg2 is *synchronous* and loop-agnostic. This matters: the same
    pool is used from three very different execution contexts - the FastAPI
    event loop, the worker-thread loop spun up by `asyncio.to_thread` for the
    synchronous LangGraph nodes, and the per-task loops inside the Celery
    ingest worker. An async driver (asyncpg) binds its connections to the
    event loop that created the pool, so sharing one pool across those
    contexts silently drops writes on every loop except the first one. A
    synchronous driver has no event loop, so it just works everywhere.
  - Every public function is wrapped in try/except and only *logs* on
    failure - the same graceful-degradation pattern used for the Redis
    checkpointer and Mem0 client elsewhere in this codebase. A Postgres
    outage should never fail an /ingest or /query request; this table is an
    audit log / evaluation dataset, not on the critical path.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import psycopg2
from psycopg2 import pool as pg_pool

from app.core.config import settings

logger = logging.getLogger(__name__)

_pool: pg_pool.SimpleConnectionPool | None = None


def get_pool() -> pg_pool.SimpleConnectionPool | None:
    """
    Lazily create (once) and return the shared connection pool, or None on failure.

    Used by sibling persistence modules (eval_store, log_store, analytics) so the
    whole app shares one connection pool instead of opening several.

    psycopg2 connections are NOT safe to share across threads, but the pool hands
    out a dedicated connection per `getconn()` call and the caller returns it with
    `putconn()`, so a single shared pool is safe to use from every thread/loop.
    """
    global _pool
    if _pool is None:
        try:
            _pool = pg_pool.SimpleConnectionPool(
                minconn=1,
                maxconn=5,
                dsn=settings.postgres_url,
                connect_timeout=5,
            )
            logger.info("PostgreSQL connection pool initialized.")
        except Exception as e:
            logger.warning("PostgreSQL pool init failed (persistence disabled): %s", e)
            _pool = None
    return _pool


def record_document(
    doc_id: str,
    filename: str,
    chunk_count: int,
    kg_triples: int,
    *,
    status: str | None = None,
    total_pages: int = 0,
) -> None:
    """
    Upsert one row into `documents` after a successful /ingest.

    ON CONFLICT (doc_id) DO UPDATE handles re-ingesting the same file
    (doc_id is a content hash - same bytes always produce the same id).
    The conflict branch also refreshes `status`, so a re-ingest reliably
    flips the document back to "ready" instead of stranding it in
    "processing".

    Args:
        status:      Lifecycle state ('processing' | 'ready' | 'error').
                    If None, defaults to "ready" (the normal post-ingest state).
        total_pages: Total pages parsed from the PDF.
    """
    pool = get_pool()
    if pool is None:
        return
    try:
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents
                        (doc_id, filename, chunk_count, kg_triples, status, total_pages)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id) DO UPDATE
                    SET filename    = EXCLUDED.filename,
                        chunk_count = EXCLUDED.chunk_count,
                        kg_triples  = EXCLUDED.kg_triples,
                        total_pages = EXCLUDED.total_pages,
                        status      = EXCLUDED.status
                    """,
                    (
                        doc_id,
                        filename,
                        chunk_count,
                        kg_triples,
                        status if status is not None else "ready",
                        total_pages,
                    ),
                )
            conn.commit()
        finally:
            pool.putconn(conn)
    except Exception as e:
        logger.warning("record_document() failed for doc_id=%s: %s", doc_id, e)


def update_document_status(doc_id: str, status: str) -> None:
    """Move a document through its ingest lifecycle."""
    pool = get_pool()
    if pool is None:
        return
    try:
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET status = %s WHERE doc_id = %s",
                    (status, doc_id),
                )
            conn.commit()
        finally:
            pool.putconn(conn)
    except Exception as e:
        logger.warning(
            "update_document_status() failed for doc_id=%s: %s", doc_id, e
        )


def record_chunk_metadata(doc_id: str, chunks: list[dict]) -> None:
    """
    Bulk upsert chunk metadata into `document_chunks`.

    Postgres holds the chunk text + token_count (metadata); Qdrant holds the
    dense vector. This mirrors the spec's `document_chunks` table and lets us
    list/inspect a document's chunks without a vector round-trip.

    ON CONFLICT (document_id, chunk_index) DO UPDATE makes re-ingesting the
    same file replace the chunk rows instead of failing on a duplicate key.
    The backing UNIQUE constraint (uq_document_chunk) is created in
    scripts/init_db.sql.
    """
    pool = get_pool()
    if pool is None or not chunks:
        return
    try:
        rows = [
            (
                doc_id,
                int(c.get("chunk_index", 0)),
                c.get("text", ""),
                int(c.get("word_count", 0)),
                int(c.get("page", 0)),
            )
            for c in chunks
        ]
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO document_chunks
                        (document_id, chunk_index, text, token_count, page)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (document_id, chunk_index) DO UPDATE
                    SET text        = EXCLUDED.text,
                        token_count = EXCLUDED.token_count,
                        page        = EXCLUDED.page
                    """,
                    rows,
                )
            conn.commit()
        finally:
            pool.putconn(conn)
    except Exception as e:
        logger.warning(
            "record_chunk_metadata() failed for doc_id=%s: %s", doc_id, e
        )


def record_user(username: str) -> int | None:
    """Upsert a user by username, returning the user id (best-effort)."""
    pool = get_pool()
    if pool is None:
        return None
    try:
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (username)
                    VALUES (%s)
                    ON CONFLICT (username) DO UPDATE
                    SET username = EXCLUDED.username
                    RETURNING id
                    """,
                    (username,),
                )
                row = cur.fetchone()
            conn.commit()
            return int(row[0]) if row else None
        finally:
            pool.putconn(conn)
    except Exception as e:
        logger.warning("record_user() failed for '%s': %s", username, e)
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 8 — Memory / Conversation metadata
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def record_memory(user_id: str, memory_text: str) -> None:
    """
    Persist a long-term memory fact to Postgres.

    UNIQUE (user_id, memory_text) makes this idempotent: re-recording an
    already-known fact is a no-op rather than a duplicate row.
    """
    if not user_id or user_id == "anonymous" or not memory_text:
        return
    pool = get_pool()
    if pool is None:
        return
    try:
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memories (user_id, memory_text)
                    VALUES (%s, %s)
                    ON CONFLICT (user_id, memory_text) DO NOTHING
                    """,
                    (user_id, memory_text),
                )
            conn.commit()
        finally:
            pool.putconn(conn)
    except Exception as e:
        logger.warning("record_memory() failed for user_id=%s: %s", user_id, e)


def record_conversation(
    session_id: str,
    user_id: str,
    question: str,
    answer: str,
) -> None:
    """Persist one Q&A turn as conversation metadata in Postgres."""
    if not question:
        return
    pool = get_pool()
    if pool is None:
        return
    try:
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversations (session_id, user_id, question, answer)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (session_id, user_id, question, answer),
                )
            conn.commit()
        finally:
            pool.putconn(conn)
    except Exception as e:
        logger.warning(
            "record_conversation() failed for session=%s: %s", session_id, e
        )


def record_query(
    session_id: str,
    user_id: str,
    doc_id: str | None,
    question: str,
    answer: str,
    query_type: str | None,
    faithfulness_score: float | None,
    retrieval_score: float | None,
    retry_count: int,
    latency_ms: dict[str, Any] | None,
) -> None:
    """Insert one row into `query_history` after a completed /query call."""
    pool = get_pool()
    if pool is None:
        return
    try:
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO query_history (
                        session_id, user_id, doc_id, question, answer,
                        query_type, faithfulness_score, retrieval_score,
                        retry_count, latency_ms
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        session_id,
                        user_id,
                        doc_id,
                        question,
                        answer,
                        query_type,
                        faithfulness_score,
                        retrieval_score,
                        retry_count,
                        json.dumps(latency_ms or {}),
                    ),
                )
            conn.commit()
        finally:
            pool.putconn(conn)
    except Exception as e:
        logger.warning("record_query() failed for session_id=%s: %s", session_id, e)
