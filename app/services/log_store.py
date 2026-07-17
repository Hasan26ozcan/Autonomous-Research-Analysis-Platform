"""
app/services/log_store.py
==========================

Phase 10 — structured logging to PostgreSQL.

Three log streams the spec calls for, each its own table (scripts/init_db.sql):
  - api_log       : every HTTP request (method, path, status, latency)
  - worker_log    : Celery ingest worker events (start / done / error)
  - pipeline_log  : per-node latency across the query graph

All functions are synchronous (psycopg2) and BEST-EFFORT: if Postgres is
unreachable the log line is dropped (logged at warning level) and the caller
proceeds. Logging must never fail a request, an ingest, or a query.

They share the one connection pool exposed by postgres_store.get_pool().

psycopg2 is loop-agnostic, so these are safe to call from the FastAPI event
loop (e.g. the api_log middleware), the Celery worker, or any thread.
"""

from __future__ import annotations

import logging
from typing import Any

from app.services.postgres_store import get_pool

logger = logging.getLogger(__name__)


def log_api(
    method: str,
    path: str,
    status_code: int,
    latency_ms: float,
    user_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """Append one row to api_log."""
    pool = get_pool()
    if pool is None:
        return
    try:
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO api_log (method, path, status_code, latency_ms, user_id, session_id)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (method, path, status_code, latency_ms, user_id, session_id),
                )
            conn.commit()
        finally:
            pool.putconn(conn)
    except Exception as e:
        logger.warning("log_api() failed (non-fatal): %s", e)


def log_worker(
    task: str,
    doc_id: str | None,
    level: str,
    message: str,
) -> None:
    """Append one row to worker_log."""
    pool = get_pool()
    if pool is None:
        return
    try:
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO worker_log (task, doc_id, level, message)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (task, doc_id, level, message),
                )
            conn.commit()
        finally:
            pool.putconn(conn)
    except Exception as e:
        logger.warning("log_worker() failed (non-fatal): %s", e)


def log_pipeline_batch(
    session_id: str | None,
    query_type: str | None,
    latency: dict[str, Any],
) -> None:
    """
    Append one pipeline_log row per node in `latency` (the query graph's
    latency_ms dict: {node_name: ms}). Single transaction for efficiency.
    """
    if not latency:
        return
    pool = get_pool()
    if pool is None:
        return
    rows = [
        (session_id, str(node), float(ms), query_type)
        for node, ms in latency.items()
        if isinstance(ms, (int, float))
    ]
    if not rows:
        return
    try:
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO pipeline_log (session_id, node, latency_ms, query_type)
                    VALUES (%s, %s, %s, %s)
                    """,
                    rows,
                )
            conn.commit()
        finally:
            pool.putconn(conn)
    except Exception as e:
        logger.warning("log_pipeline_batch() failed (non-fatal): %s", e)
