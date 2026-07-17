"""
app/services/eval_store.py
===========================
Phase 9 — persistence for evaluation runs (faithfulness, precision, recall,
latency, token usage) into PostgreSQL.

Tables (created in scripts/init_db.sql):
  - evaluation_runs     : one row per evaluation run
  - evaluation_scores   : one row per metric per run
  - retrieval_results   : one row per retrieved chunk observed during a run

Synchronous (psycopg2) and BEST-EFFORT: if Postgres is unreachable, evaluation
still runs and reports to JSON — it just isn't persisted. This matches the
pattern used by postgres_store.py.

psycopg2 is loop-agnostic, so these functions are safe to call from the
FastAPI event loop, the Celery worker, or any thread.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.services.postgres_store import get_pool

logger = logging.getLogger(__name__)


def start_run(num_questions: int) -> int | None:
    """Insert a new evaluation_runs row; return its run_id (or None on failure)."""
    pool = get_pool()
    if pool is None:
        return None
    try:
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO evaluation_runs (num_questions, status)
                    VALUES (%s, 'running')
                    RETURNING id
                    """,
                    (num_questions,),
                )
                row = cur.fetchone()
            conn.commit()
            return int(row[0]) if row else None
        finally:
            pool.putconn(conn)
    except Exception as e:
        logger.warning("eval_store.start_run() failed: %s", e)
        return None


def finish_run(
    run_id: int | None,
    *,
    status: str,
    metrics: dict[str, float | None],
    token_usage: dict[str, int] | None = None,
    average_faithfulness: float | None = None,
    notes: str | None = None,
) -> None:
    """
    Finalize an evaluation run: write its status, aggregate faithfulness,
    token usage, and one evaluation_scores row per metric.

    No-op if run_id is None (Postgres was unavailable at start_run).
    """
    if run_id is None:
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
                    UPDATE evaluation_runs
                    SET status = %s,
                        average_faithfulness = %s,
                        token_usage = %s,
                        notes = %s
                    WHERE id = %s
                    """,
                    (
                        status,
                        average_faithfulness,
                        json.dumps(token_usage or {}, ensure_ascii=False),
                        notes,
                        run_id,
                    ),
                )
                score_rows = [
                    (run_id, name, float(value))
                    for name, value in (metrics or {}).items()
                    if value is not None
                ]
                if score_rows:
                    cur.executemany(
                        """
                        INSERT INTO evaluation_scores (run_id, metric_name, metric_value)
                        VALUES (%s, %s, %s)
                        """,
                        score_rows,
                    )
            conn.commit()
        finally:
            pool.putconn(conn)
    except Exception as e:
        logger.warning("eval_store.finish_run() failed for run_id=%s: %s", run_id, e)


def record_retrieval_results(
    run_id: int | None,
    items: list[dict[str, Any]],
) -> None:
    """
    Persist per-question retrieval results observed during a run.

    items: list of {
        "question": str, "doc_id": str, "chunk_index": int,
        "score": float, "source": str
    }
    """
    if run_id is None or not items:
        return
    pool = get_pool()
    if pool is None:
        return
    try:
        rows = [
            (
                run_id,
                it.get("question", ""),
                it.get("doc_id", ""),
                int(it.get("chunk_index", 0)),
                float(it.get("score", 0.0) or 0.0),
                it.get("source", ""),
            )
            for it in items
        ]
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO retrieval_results
                        (run_id, question, doc_id, chunk_index, score, source)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    rows,
                )
            conn.commit()
        finally:
            pool.putconn(conn)
    except Exception as e:
        logger.warning(
            "eval_store.record_retrieval_results() failed for run_id=%s: %s",
            run_id, e,
        )
