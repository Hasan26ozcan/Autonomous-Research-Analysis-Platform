"""
app/services/analytics.py
=========================
Phase 11 — analytics queries over the PostgreSQL tables populated by the rest
of the system (query_history, documents, evaluation_runs, evaluation_scores).

Everything is read-only and BEST-EFFORT: if Postgres is unreachable each
function returns an empty/zeroed structure so the /analytics dashboard still
renders (just with no data) instead of erroring.

Synchronous (psycopg2). Uses RealDictCursor so rows are addressable by column
name. Shares the connection pool from postgres_store.get_pool().
"""

from __future__ import annotations

import logging
from typing import Any

from psycopg2.extras import RealDictCursor

from app.services.postgres_store import get_pool

logger = logging.getLogger(__name__)


def _fetchval(conn, sql: str, params: tuple = ()) -> Any:
    """psycopg2 equivalent of asyncpg's conn.fetchval — first column of row 0."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    return row[0] if row else None


def summary() -> dict[str, Any]:
    """
    Headline metrics for the dashboard:
      - total_queries, total_documents, total_evals
      - avg_latency_ms   (mean total latency across query_history)
      - avg_faithfulness (mean NLI faithfulness across query_history)
      - avg_precision / avg_recall (mean of latest evaluation_scores)
    """
    empty = {
        "total_queries": 0,
        "total_documents": 0,
        "total_evals": 0,
        "avg_latency_ms": None,
        "avg_faithfulness": None,
        "avg_precision": None,
        "avg_recall": None,
    }
    pool = get_pool()
    if pool is None:
        return empty
    try:
        conn = pool.getconn()
        try:
            total_queries = _fetchval(conn, "SELECT COUNT(*) FROM query_history") or 0
            total_documents = _fetchval(conn, "SELECT COUNT(*) FROM documents") or 0
            total_evals = _fetchval(conn, "SELECT COUNT(*) FROM evaluation_runs") or 0
            avg_faithfulness = _fetchval(
                conn,
                "SELECT AVG(faithfulness_score) FROM query_history "
                "WHERE faithfulness_score IS NOT NULL",
            )
            # Average total latency: sum every node's ms per row, then average.
            avg_latency = _fetchval(
                conn,
                """
                SELECT AVG(total_ms) FROM (
                    SELECT (
                        SELECT COALESCE(SUM(value::float), 0)
                        FROM jsonb_each_text(latency_ms)
                    ) AS total_ms
                    FROM query_history
                    WHERE latency_ms IS NOT NULL
                ) sub
                """,
            )
            avg_precision = _fetchval(
                conn,
                "SELECT AVG(metric_value) FROM evaluation_scores "
                "WHERE metric_name = 'context_precision'",
            )
            avg_recall = _fetchval(
                conn,
                "SELECT AVG(metric_value) FROM evaluation_scores "
                "WHERE metric_name = 'context_recall'",
            )
        finally:
            pool.putconn(conn)
        return {
            "total_queries": int(total_queries),
            "total_documents": int(total_documents),
            "total_evals": int(total_evals),
            "avg_latency_ms": _round(avg_latency),
            "avg_faithfulness": _round(avg_faithfulness),
            "avg_precision": _round(avg_precision),
            "avg_recall": _round(avg_recall),
        }
    except Exception as e:
        logger.warning("analytics.summary() failed: %s", e)
        return empty


def top_documents(limit: int = 10) -> list[dict[str, Any]]:
    """
    Most-queried documents with their average faithfulness.

    Joins query_history (by doc_id) onto documents for the filename.
    """
    pool = get_pool()
    if pool is None:
        return []
    try:
        conn = pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        d.doc_id      AS doc_id,
                        d.filename    AS filename,
                        COUNT(q.id)   AS query_count,
                        AVG(q.faithfulness_score) AS avg_faithfulness
                    FROM documents d
                    LEFT JOIN query_history q ON q.doc_id = d.doc_id
                    GROUP BY d.doc_id, d.filename
                    ORDER BY query_count DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        finally:
            pool.putconn(conn)
        return [
            {
                "doc_id": r["doc_id"],
                "filename": r["filename"],
                "query_count": int(r["query_count"] or 0),
                "avg_faithfulness": _round(r["avg_faithfulness"]),
            }
            for r in rows
        ]
    except Exception as e:
        logger.warning("analytics.top_documents() failed: %s", e)
        return []


def eval_trend(limit: int = 20) -> list[dict[str, Any]]:
    """Recent evaluation runs (newest first) with headline metrics."""
    pool = get_pool()
    if pool is None:
        return []
    try:
        conn = pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, created_at, num_questions, status,
                           average_faithfulness, token_usage
                    FROM evaluation_runs
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        finally:
            pool.putconn(conn)
        return [
            {
                "run_id": int(r["id"]),
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "num_questions": int(r["num_questions"] or 0),
                "status": r["status"],
                "average_faithfulness": _round(r["average_faithfulness"]),
                "token_usage": r["token_usage"],
            }
            for r in rows
        ]
    except Exception as e:
        logger.warning("analytics.eval_trend() failed: %s", e)
        return []


def _round(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None
