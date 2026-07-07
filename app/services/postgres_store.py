"""
PostgreSQL persistence for ARAP.

Writes to the two tables defined in scripts/init_db.sql:
  - documents:     one row per ingested PDF (doc_id, filename, counts)
  - query_history: one row per /query call (audit log + RAGAS test set)

These tables existed in the schema from day one (scripts/init_db.sql runs
automatically on Postgres' first boot) but nothing in the application ever
wrote to them - the migration ran, the tables were just never fed. This
module is that missing write path.

Design notes:
  - A single lazily-created asyncpg connection pool is shared process-wide.
  - Every public function is wrapped in try/except and only *logs* on
    failure - the same graceful-degradation pattern already used for the
    Redis checkpointer and Mem0 client elsewhere in this codebase. A
    Postgres outage should never fail an /ingest or /query request; this
    table is an audit log / evaluation dataset, not on the critical path.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import asyncpg

from app.core.config import settings

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def _get_pool() -> asyncpg.Pool | None:
    """Lazily create (once) and return the shared connection pool, or None on failure."""
    global _pool
    if _pool is None:
        try:
            _pool = await asyncpg.create_pool(
                settings.postgres_url,
                min_size=1,
                max_size=5,
                command_timeout=5,
            )
            logger.info("PostgreSQL connection pool initialized.")
        except Exception as e:
            logger.warning("PostgreSQL pool init failed (persistence disabled): %s", e)
            _pool = None
    return _pool


async def record_document(
    doc_id: str,
    filename: str,
    chunk_count: int,
    kg_triples: int,
) -> None:
    """
    Upsert one row into `documents` after a successful /ingest.

    ON CONFLICT (doc_id) DO UPDATE handles re-ingesting the same file
    (doc_id is a content hash - same bytes always produce the same id).
    """
    pool = await _get_pool()
    if pool is None:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO documents (doc_id, filename, chunk_count, kg_triples)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (doc_id) DO UPDATE
                SET filename    = EXCLUDED.filename,
                    chunk_count = EXCLUDED.chunk_count,
                    kg_triples  = EXCLUDED.kg_triples
                """,
                doc_id, filename, chunk_count, kg_triples,
            )
    except Exception as e:
        logger.warning("record_document() failed for doc_id=%s: %s", doc_id, e)


async def record_query(
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
    pool = await _get_pool()
    if pool is None:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO query_history (
                    session_id, user_id, doc_id, question, answer,
                    query_type, faithfulness_score, retrieval_score,
                    retry_count, latency_ms
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                session_id, user_id, doc_id, question, answer,
                query_type, faithfulness_score, retrieval_score,
                retry_count, json.dumps(latency_ms or {}),
            )
    except Exception as e:
        logger.warning("record_query() failed for session_id=%s: %s", session_id, e)
