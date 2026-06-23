-- =============================================================================
-- ARAP — PostgreSQL Schema
-- =============================================================================
-- Executed automatically by PostgreSQL on first docker compose up.
-- Tables use IF NOT EXISTS so re-runs are safe (idempotent).

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

-- Evaluation view: high-quality Q&A pairs for RAGAS
CREATE OR REPLACE VIEW evaluation_test_set AS
SELECT question, answer AS ground_truth, query_type, faithfulness_score, created_at
FROM query_history
WHERE answer IS NOT NULL AND LENGTH(answer) > 50 AND faithfulness_score IS NOT NULL
ORDER BY created_at DESC
LIMIT 100;
