"""
app/core/config.py
==================
Centralized configuration for the entire ARAP system.

Design principles:
  - Every secret and tunable parameter lives here — no magic strings elsewhere.
  - All values loaded from env vars via pydantic-settings (.env auto-read).
  - Sensible defaults so the system boots with minimal setup.
  - Settings grouped by subsystem for easy navigation.

Usage anywhere in the codebase:
    from app.core.config import settings
    print(settings.llm_model)     # "gpt-4o"
    print(settings.qdrant_host)   # "localhost"
"""

from __future__ import annotations
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Single source of truth for all ARAP configuration.

    pydantic-settings automatically:
      1. Reads values from environment variables (case-insensitive)
      2. Falls back to a .env file in the working directory
      3. Falls back to the default values defined below
      4. Validates types — e.g. qdrant_port must be int, not string
    """

    # ── LLM Models ────────────────────────────────────────────────────────────
    # Three models with different cost/quality tradeoffs:
    #   llm_model    — most capable, only for final answer generation
    #   router_model — fast and cheap, for routing/rewriting/extraction
    #   judge_model  — cheap, for LLM-based fallback judging
    # Primary faithfulness judging uses LOCAL NLI (DeBERTa-v3) — no API cost.
    openai_api_key: str = Field(default="", description="OpenAI API key")
    llm_model: str = Field(default="gpt-4o", description="Primary generation model")
    router_model: str = Field(default="gpt-4o-mini", description="Fast model for routing/rewriting")
    judge_model: str = Field(default="gpt-4o-mini", description="Fallback LLM judge model")
    temperature: float = Field(default=0.1, description="Low temperature for factual RAG")
    max_tokens: int = Field(default=2048)
    max_retries: int = Field(default=2, description="Max faithfulness retry attempts")

    # ── Embedding ─────────────────────────────────────────────────────────────
    # Default: MiniLM-L6-v2 (384 dims, CPU-friendly, fast)
    # SOTA 2026: Qwen/Qwen3-Embedding-4B (2560 dims, GPU required, top MTEB)
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim: int = Field(default=384, description="MUST match embedding_model output dim")

    # ── Chunking ──────────────────────────────────────────────────────────────
    # 512 words ≈ 700 tokens — fits in cross-encoder max_length.
    # 64-word overlap prevents losing information at chunk boundaries.
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)

    # ── Qdrant — Vector Database ──────────────────────────────────────────────
    # Chosen over FAISS: built-in metadata filtering, REST+gRPC, production-ready.
    # Chosen over Pinecone: self-hostable, no per-query cost.
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection: str = Field(default="arap_docs")

    # ── Neo4j — Knowledge Graph ───────────────────────────────────────────────
    # Graph traversal for entity/relationship queries that vector search cannot
    # answer: "Which papers does author X cite?" needs Cypher, not cosine.
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")

    # ── Redis ─────────────────────────────────────────────────────────────────
    # Role 1: LangGraph checkpointer — persists conversation state across requests
    # Role 2: Celery broker — async ingestion task queue
    redis_url: str = Field(default="redis://localhost:6379/0")
    session_ttl_seconds: int = Field(default=3600)

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    # Document metadata + query history (doubles as RAGAS evaluation test set)
    postgres_url: str = Field(default="postgresql://user:password@localhost:5432/arap")

    # ── Mem0 — Long-Term Memory ───────────────────────────────────────────────
    # Chhikara et al. (2025), arXiv:2504.19413:
    # 91% lower p95 latency, 90%+ token cost reduction vs full-context.
    # SOTA on multi-hop, temporal, and open-domain memory queries.
    mem0_api_key: str = Field(default="", description="Leave empty for self-hosted")
    mem0_base_url: str = Field(default="http://localhost:3000")

    # ── LangSmith — Observability ─────────────────────────────────────────────
    # Traces every LangGraph node: input/output, token counts, latency.
    # Enable by setting LANGCHAIN_TRACING_V2=true — zero code changes needed.
    langchain_api_key: str = Field(default="")
    langchain_tracing_v2: bool = Field(default=False)
    langchain_project: str = Field(default="arap")

    # ── Retrieval Parameters ──────────────────────────────────────────────────
    # Pipeline: top_k_retrieval candidates → cross-encoder rerank → top_k_final
    # RRF formula: score(d) = Σ [ weight × 1 / (k + rank(d)) ]
    # rrf_k=60 from Cormack et al. (SIGIR 2009) — original RRF paper.
    top_k_retrieval: int = Field(default=10, description="Candidates before reranking")
    top_k_final: int = Field(default=5, description="Chunks sent to LLM after reranking")
    rrf_k: int = Field(default=60, description="RRF constant, standard value from original paper")
    dense_weight: float = Field(default=0.7, description="Dense search weight in RRF merge")
    bm25_weight: float = Field(default=0.3, description="BM25 search weight in RRF merge")

    # ── Faithfulness Judge ────────────────────────────────────────────────────
    # Model: cross-encoder/nli-deberta-v3-small
    #   - Best NLI model at this size (2024-2026 NLI benchmarks)
    #   - Runs on CPU in ~50ms per sentence pair, no API cost
    #   - 3 labels: contradiction=0, neutral=0.5, entailment=1.0
    # Below faithfulness_threshold → retry generation with stricter prompt
    faithfulness_threshold: float = Field(default=0.75)
    nli_model: str = Field(default="cross-encoder/nli-deberta-v3-small")

    # ── Celery — Async Task Queue ─────────────────────────────────────────────
    # PDF ingestion takes 30-90s. Celery lets /ingest return immediately.
    celery_broker_url: str = Field(default="redis://localhost:6379/1")
    celery_result_backend: str = Field(default="redis://localhost:6379/2")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Module-level singleton — import this everywhere.
# Never instantiate Settings() directly; always use this singleton.
settings = Settings()
