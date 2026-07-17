-- =============================================================================
-- ARAP — PostgreSQL Schema
-- =============================================================================
-- Executed automatically by PostgreSQL on first docker compose up.
-- Tables use IF NOT EXISTS / conditional ALTERs so re-runs are safe (idempotent).

-- Document registry
-- doc_id = SHA-256 of PDF bytes (first 16 hex chars) — same file = same doc_id
CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    doc_id      VARCHAR(64) UNIQUE NOT NULL,
    filename    VARCHAR(512) NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    kg_triples  INTEGER DEFAULT 0,
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Add Phase 1 columns (status / total_pages) without dropping existing data.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'documents' AND column_name = 'status'
    ) THEN
        ALTER TABLE documents ADD COLUMN status VARCHAR(32) DEFAULT 'processing';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'documents' AND column_name = 'total_pages'
    ) THEN
        ALTER TABLE documents ADD COLUMN total_pages INTEGER DEFAULT 0;
    END IF;
END $$;

-- Phase 1: users table
CREATE TABLE IF NOT EXISTS users (
    id          SERIAL PRIMARY KEY,
    username    VARCHAR(256) UNIQUE NOT NULL,
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Phase 1: document_chunks — chunk metadata mirror (Postgres holds metadata,
-- Qdrant holds the vectors). Lets us answer "what is in this document?" without
-- a vector round-trip.
CREATE TABLE IF NOT EXISTS document_chunks (
    id           SERIAL PRIMARY KEY,
    document_id  VARCHAR(64) NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    text         TEXT,
    token_count  INTEGER DEFAULT 0,
    page         INTEGER DEFAULT 0,
    created_at   TIMESTAMP DEFAULT NOW()
);

-- Query history
-- Dual purpose: audit log + RAGAS evaluation test set
CREATE TABLE IF NOT EXISTS query_history (
    id                 SERIAL PRIMARY KEY,
    session_id         VARCHAR(64),
    user_id            VARCHAR(256),
    doc_id             VARCHAR(64),
    question           TEXT NOT NULL,
    answer             TEXT,
    query_type         VARCHAR(32),       -- direct/single/multi_hop/graph
    faithfulness_score FLOAT,             -- NLI judge score (0.0-1.0)
    retrieval_score    FLOAT,             -- avg rerank score
    retry_count        INTEGER DEFAULT 0,
    latency_ms         JSONB,             -- per-node breakdown
    created_at         TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_qh_session    ON query_history(session_id);
CREATE INDEX IF NOT EXISTS idx_qh_user       ON query_history(user_id);
CREATE INDEX IF NOT EXISTS idx_qh_created    ON query_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_qh_query_type ON query_history(query_type);
CREATE INDEX IF NOT EXISTS idx_docs_doc_id   ON documents(doc_id);
CREATE INDEX IF NOT EXISTS idx_dc_doc        ON document_chunks(document_id, chunk_index);

-- Unique (document_id, chunk_index) so record_chunk_metadata can ON CONFLICT
-- DO UPDATE on re-ingest (replace chunk rows instead of failing on a dup key).
-- Idempotent: guarded so re-running init_db.sql is safe.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'uq_document_chunk'
          AND table_name = 'document_chunks'
    ) THEN
        ALTER TABLE document_chunks
            ADD CONSTRAINT uq_document_chunk
            UNIQUE (document_id, chunk_index);
    END IF;
END $$;

-- Phase 8: long-term memory metadata (what topics a user has engaged with).
CREATE TABLE IF NOT EXISTS memories (
    id          SERIAL PRIMARY KEY,
    user_id     VARCHAR(256) NOT NULL,
    memory_text TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT NOW(),
    UNIQUE (user_id, memory_text)
);

-- Phase 8: conversation metadata — one row per persisted Q&A turn.
CREATE TABLE IF NOT EXISTS conversations (
    id          SERIAL PRIMARY KEY,
    session_id  VARCHAR(64),
    user_id     VARCHAR(256),
    question    TEXT,
    answer      TEXT,
    created_at  TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_conv_user      ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conv_session   ON conversations(session_id);

-- Phase 9: evaluation run registry.
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id                  SERIAL PRIMARY KEY,
    created_at          TIMESTAMP DEFAULT NOW(),
    num_questions       INTEGER DEFAULT 0,
    status              VARCHAR(32),
    average_faithfulness FLOAT,
    token_usage         JSONB,
    notes               TEXT
);

-- Phase 9: per-metric scores for a run (faithfulness, answer_relevancy, ...).
CREATE TABLE IF NOT EXISTS evaluation_scores (
    id          SERIAL PRIMARY KEY,
    run_id      INTEGER NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
    metric_name VARCHAR(64) NOT NULL,
    metric_value FLOAT
);
CREATE INDEX IF NOT EXISTS idx_es_run ON evaluation_scores(run_id);

-- Phase 9: per-question retrieval results observed during a run.
CREATE TABLE IF NOT EXISTS retrieval_results (
    id           SERIAL PRIMARY KEY,
    run_id       INTEGER NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
    question     TEXT,
    doc_id       VARCHAR(64),
    chunk_index  INTEGER,
    score        FLOAT,
    source       VARCHAR(32)
);
CREATE INDEX IF NOT EXISTS idx_rr_run ON retrieval_results(run_id);

-- Phase 10: API access log.
CREATE TABLE IF NOT EXISTS api_log (
    id           SERIAL PRIMARY KEY,
    ts           TIMESTAMP DEFAULT NOW(),
    method       VARCHAR(16),
    path         TEXT,
    status_code  INTEGER,
    latency_ms   FLOAT,
    user_id      VARCHAR(256),
    session_id   VARCHAR(64)
);

-- Phase 10: worker (Celery ingest) log.
CREATE TABLE IF NOT EXISTS worker_log (
    id          SERIAL PRIMARY KEY,
    ts          TIMESTAMP DEFAULT NOW(),
    task        VARCHAR(128),
    doc_id      VARCHAR(64),
    level       VARCHAR(16),
    message     TEXT
);

-- Phase 10: pipeline node execution log (per-node latency across the query graph).
CREATE TABLE IF NOT EXISTS pipeline_log (
    id           SERIAL PRIMARY KEY,
    ts           TIMESTAMP DEFAULT NOW(),
    session_id   VARCHAR(64),
    node         VARCHAR(64),
    latency_ms   FLOAT,
    query_type   VARCHAR(32)
);

-- Evaluation view: high-quality Q&A pairs for RAGAS
CREATE OR REPLACE VIEW evaluation_test_set AS
SELECT question, answer AS ground_truth, query_type, faithfulness_score, created_at
FROM query_history
WHERE answer IS NOT NULL AND LENGTH(answer) > 50 AND faithfulness_score IS NOT NULL
ORDER BY created_at DESC
LIMIT 100;
