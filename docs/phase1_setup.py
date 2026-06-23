# =============================================================================
# ARAP — Adaptive Research & Analysis Platform
# Phase 1: Project Setup
#
# This single file contains everything needed to understand and reproduce
# the project's foundation:
#
#   1. Directory structure
#   2. app/core/config.py       — centralized configuration
#   3. app/core/state.py        — LangGraph shared state schema
#   4. requirements.txt         — all dependencies with explanations
#   5. .env.example             — environment variable template
#   6. .gitignore               — what to never commit
#   7. scripts/init_db.sql      — PostgreSQL schema
#
# After reading this file you will understand:
#   - Every configuration knob in the system and why it exists
#   - The exact shape of data flowing through LangGraph
#   - Every external dependency and the research reason for choosing it
# =============================================================================


# =============================================================================
# SECTION 1 — DIRECTORY STRUCTURE
# =============================================================================
#
# arap/
# ├── app/
# │   ├── core/
# │   │   ├── __init__.py
# │   │   ├── config.py          ← Section 2 of this file
# │   │   └── state.py           ← Section 3 of this file
# │   ├── agents/
# │   │   ├── __init__.py
# │   │   ├── router.py          ← Phase 4
# │   │   ├── retrieval_agent.py ← Phase 5
# │   │   ├── graph_agent.py     ← Phase 6
# │   │   └── generator.py       ← Phase 7
# │   ├── api/
# │   │   ├── __init__.py
# │   │   └── main.py            ← Phase 8
# │   └── services/
# │       ├── __init__.py
# │       ├── chunker.py         ← Phase 2
# │       ├── embedder.py        ← Phase 2
# │       ├── vector_store.py    ← Phase 2
# │       ├── bm25_index.py      ← Phase 2
# │       ├── contextual_enricher.py ← Phase 3
# │       └── tasks.py           ← Phase 8 (Celery async workers)
# ├── evaluation/
# │   ├── __init__.py
# │   └── ragas_eval.py          ← Phase 9
# ├── tests/
# │   ├── __init__.py
# │   ├── unit/                  ← Phase 9
# │   └── integration/           ← Phase 9
# ├── scripts/
# │   └── init_db.sql            ← Section 7 of this file
# ├── docs/
# │   └── architecture.md        ← Phase 9
# ├── Dockerfile                 ← Phase 9
# ├── docker-compose.yml         ← Phase 9
# ├── requirements.txt           ← Section 4 of this file
# ├── .env.example               ← Section 5 of this file
# ├── .gitignore                 ← Section 6 of this file
# └── README.md                  ← Phase 9


# =============================================================================
# SECTION 2 — app/core/config.py
# =============================================================================
#
# Save this block as: app/core/config.py
# ─────────────────────────────────────

CONFIG_PY = '''
"""
app/core/config.py
==================
Centralized configuration for the entire ARAP system.

Design principles:
  - Every secret and tunable parameter lives here — no magic strings elsewhere.
  - All values are loaded from environment variables via pydantic-settings,
    which reads a .env file automatically in development.
  - Sensible defaults are provided so the system boots with minimal setup.
  - Settings are grouped by their subsystem for easy navigation.

Usage:
  from app.core.config import settings
  print(settings.llm_model)          # "gpt-4o"
  print(settings.qdrant_host)        # "localhost"
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
      4. Validates types — e.g. redis_port must be an int, not a string
    """

    # ──────────────────────────────────────────────────────────────────────────
    # LLM MODELS
    # We use three different models with different cost/quality tradeoffs:
    #   llm_model     — most capable, used only for final answer generation
    #   router_model  — fast and cheap, used for routing, rewriting, extraction
    #   judge_model   — cheap NLI-based judge (or gpt-4o-mini for LLM judging)
    # ──────────────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key")

    llm_model: str = Field(
        default="gpt-4o",
        description=(
            "Primary generation model. GPT-4o gives the best answer quality. "
            "Switch to gpt-4o-mini to cut costs by ~10x at some quality loss."
        ),
    )
    router_model: str = Field(
        default="gpt-4o-mini",
        description=(
            "Fast model for routing, query rewriting, entity extraction. "
            "gpt-4o-mini is 10x cheaper and fast enough for these structured tasks."
        ),
    )
    judge_model: str = Field(
        default="gpt-4o-mini",
        description=(
            "Model used for LLM-based faithfulness judging as fallback. "
            "Primary judging uses local NLI (DeBERTa-v3) — no API cost."
        ),
    )
    temperature: float = Field(
        default=0.1,
        description=(
            "Generation temperature. Low (0.0-0.2) for factual RAG answers. "
            "Never use high temperature for document Q&A — increases hallucination."
        ),
    )
    max_tokens: int = Field(
        default=2048,
        description="Max tokens for generated answers. 2048 covers most documents.",
    )
    max_retries: int = Field(
        default=2,
        description=(
            "Max faithfulness retry attempts. If the judge fails twice, "
            "we return the best available answer rather than looping forever."
        ),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # EMBEDDING MODEL
    # sentence-transformers/all-MiniLM-L6-v2:
    #   - 384 dimensions, runs on CPU, ~80ms per batch of 32 chunks
    #   - Good baseline for English text
    # Qwen/Qwen3-Embedding-4B (SOTA 2026):
    #   - 2560 dimensions, needs GPU, top MTEB leaderboard score
    #   - Switch by changing embedding_model + embedding_dim
    # ──────────────────────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description=(
            "Sentence embedding model. Default is fast CPU-friendly MiniLM. "
            "For best accuracy: 'Qwen/Qwen3-Embedding-4B' (needs GPU, dim=2560)."
        ),
    )
    embedding_dim: int = Field(
        default=384,
        description=(
            "Vector dimension — MUST match embedding_model output. "
            "MiniLM-L6-v2 → 384. Qwen3-Embedding-4B → 2560."
        ),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # CHUNKING
    # Chunking strategy significantly impacts retrieval quality.
    # chunk_size = 512 words ≈ 700 tokens — fits in cross-encoder max length.
    # chunk_overlap = 64 words ensures no information is lost at boundaries.
    # contextual_window = surrounding text given to LLM for context generation.
    # ──────────────────────────────────────────────────────────────────────────
    chunk_size: int = Field(
        default=512,
        description=(
            "Words per chunk. 512 words ≈ 700 tokens. "
            "Larger chunks → more context per retrieval hit but less precision. "
            "Smaller chunks → higher precision but may miss multi-sentence reasoning."
        ),
    )
    chunk_overlap: int = Field(
        default=64,
        description=(
            "Word overlap between consecutive chunks. "
            "Prevents losing information at chunk boundaries. "
            "64 words ≈ 2-3 sentences of overlap."
        ),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # QDRANT — Vector Database
    # HNSW (Hierarchical Navigable Small World) index for approximate nearest
    # neighbor search. Qdrant is chosen over FAISS because:
    #   - Built-in metadata filtering (filter by doc_id without post-processing)
    #   - Native hybrid search support (sparse + dense)
    #   - Production-ready: REST + gRPC, Docker, cloud option
    # ──────────────────────────────────────────────────────────────────────────
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection: str = Field(default="arap_docs")

    # ──────────────────────────────────────────────────────────────────────────
    # NEO4J — Knowledge Graph
    # Used by the Graph Agent for entity/relationship queries that vector
    # search cannot answer: "Which papers does author X cite?" requires
    # graph traversal, not cosine similarity.
    # ──────────────────────────────────────────────────────────────────────────
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")

    # ──────────────────────────────────────────────────────────────────────────
    # REDIS
    # Two roles:
    #   1. LangGraph checkpointer — persists conversation state across requests
    #      so multi-turn conversations work without re-sending history
    #   2. Celery broker — queues async ingestion tasks so /ingest returns
    #      immediately instead of blocking for 30+ seconds
    # ──────────────────────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0")
    session_ttl_seconds: int = Field(
        default=3600,
        description="How long to keep conversation state in Redis (1 hour default).",
    )

    # ──────────────────────────────────────────────────────────────────────────
    # POSTGRESQL
    # Stores document metadata and query history.
    # Query history is used as the test set for RAGAS evaluation —
    # real user questions produce more meaningful eval scores than synthetic ones.
    # ──────────────────────────────────────────────────────────────────────────
    postgres_url: str = Field(
        default="postgresql://user:password@localhost:5432/arap"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # MEM0 — Long-Term Memory
    # Mem0 gives agents persistent memory across sessions.
    # It extracts semantic facts from conversation turns and deduplicates them,
    # so "user mentioned working at Omdena" is stored once, not per message.
    #
    # Reference:
    #   Chhikara et al. (2025). "Mem0: Building Production-Ready AI Agents
    #   with Scalable Long-Term Memory." arXiv:2504.19413
    #   Key findings: 91% lower p95 latency, 90%+ token cost reduction vs
    #   full-context approaches, best performance on multi-hop & temporal queries.
    # ──────────────────────────────────────────────────────────────────────────
    mem0_api_key: str = Field(
        default="",
        description="Leave empty to use self-hosted Mem0 (via mem0_base_url).",
    )
    mem0_base_url: str = Field(
        default="http://localhost:3000",
        description="Self-hosted Mem0 server URL.",
    )

    # ──────────────────────────────────────────────────────────────────────────
    # LANGSMITH — Observability
    # LangSmith traces every LangGraph node execution with:
    #   - Input/output at each node
    #   - Token counts and latency per step
    #   - Full conversation thread view
    # Essential for debugging multi-hop failures and faithfulness regressions.
    # ──────────────────────────────────────────────────────────────────────────
    langchain_api_key: str = Field(default="")
    langchain_tracing_v2: bool = Field(default=False)
    langchain_project: str = Field(default="arap")

    # ──────────────────────────────────────────────────────────────────────────
    # RETRIEVAL PARAMETERS
    # top_k_retrieval: how many candidates to fetch before re-ranking
    # top_k_final: how many chunks to send to the LLM after re-ranking
    # The gap (10 → 5) is the re-ranking stage removing noisy results.
    # rrf_k: constant in the RRF formula. 60 is from the original RRF paper.
    #   score(d) = Σ [ weight × 1/(k + rank(d)) ]
    # ──────────────────────────────────────────────────────────────────────────
    top_k_retrieval: int = Field(
        default=10,
        description="Candidate chunks fetched before re-ranking. Intentionally wide.",
    )
    top_k_final: int = Field(
        default=5,
        description="Chunks sent to LLM after cross-encoder re-ranking.",
    )
    rrf_k: int = Field(
        default=60,
        description=(
            "RRF constant. Higher k = less reward for being rank 1. "
            "60 is the standard from Cormack et al. (SIGIR 2009)."
        ),
    )
    dense_weight: float = Field(
        default=0.7,
        description=(
            "Weight for dense vector search in RRF merge. "
            "0.7 because semantic similarity outperforms keyword matching "
            "for most analytical document questions."
        ),
    )
    bm25_weight: float = Field(
        default=0.3,
        description=(
            "Weight for BM25 keyword search in RRF merge. "
            "0.3 — still valuable for exact term matching (names, codes, IDs)."
        ),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # FAITHFULNESS JUDGE
    # NLI (Natural Language Inference) scores whether each sentence in the
    # generated answer is ENTAILED by the retrieved context.
    #
    # Model: cross-encoder/nli-deberta-v3-small
    #   - DeBERTa-v3 is the best NLI model at this size (2024-2026)
    #   - "small" runs on CPU in ~50ms per sentence pair
    #   - 3 labels: contradiction (0), neutral (0.5), entailment (1)
    #
    # faithfulness_threshold:
    #   - Below 0.75 → retry generation with a stricter prompt
    #   - After max_retries → return best available answer with a warning
    # ──────────────────────────────────────────────────────────────────────────
    faithfulness_threshold: float = Field(
        default=0.75,
        description=(
            "Minimum average NLI entailment score across answer sentences. "
            "Below this, the judge triggers a generation retry."
        ),
    )
    nli_model: str = Field(
        default="cross-encoder/nli-deberta-v3-small",
        description=(
            "Local NLI model for faithfulness scoring. No API cost. "
            "DeBERTa-v3 outperforms BERT-based NLI models on NLI benchmarks."
        ),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # CELERY — Async Task Queue
    # PDF ingestion is slow (chunking + embedding + KG extraction can take
    # 30-90 seconds for large documents). Celery lets /ingest return a job_id
    # immediately while the work happens in the background.
    # ──────────────────────────────────────────────────────────────────────────
    celery_broker_url: str = Field(default="redis://localhost:6379/1")
    celery_result_backend: str = Field(default="redis://localhost:6379/2")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        # pydantic-settings reads environment variables case-insensitively,
        # so OPENAI_API_KEY and openai_api_key both work.


# Module-level singleton — import this everywhere
settings = Settings()
'''


# =============================================================================
# SECTION 3 — app/core/state.py
# =============================================================================
#
# Save this block as: app/core/state.py
# ──────────────────────────────────────

STATE_PY = '''
"""
app/core/state.py
==================
Shared state schema for all LangGraph agents in ARAP.

LangGraph fundamentals:
  - A "graph" in LangGraph is a directed acyclic (or cyclic) computation graph
    where each node is a Python function that reads and updates a shared state.
  - The state is a TypedDict — a typed dictionary that every node can read from
    and write to. Nodes receive the full state and return a PARTIAL update.
    LangGraph merges the update automatically (no need to copy the whole state).
  - Conditional edges are functions that READ the state and return the NAME
    of the next node to execute. This is how adaptive routing works.

State design principles:
  - Every field is Optional (total=False) because no single node sets all fields.
  - Fields are grouped by which phase of the pipeline populates them.
  - Non-serializable fields (raw_bytes, embeddings) are cleaned before streaming.

Example flow through state:
  router sets:        query_type="multi_hop", long_term_memories=[...]
  retrieval sets:     sub_questions=[...], retrieved_chunks=[...], retrieval_score=0.87
  generator sets:     draft_answer="...", sources=[...]
  judge sets:         faithfulness_score=0.91, judge_passed=True, answer="..."
  memory_store sets:  (nothing — side effect only)
"""

from __future__ import annotations
from typing import TypedDict, Literal


# ──────────────────────────────────────────────────────────────────────────────
# Query type taxonomy (Adaptive RAG pattern)
#
# The router classifies every incoming question into one of these four types.
# Each type triggers a different retrieval strategy:
#
#   direct    → Skip retrieval. LLM answers from parametric knowledge.
#               Used for: definitions, general knowledge, simple math.
#               Example: "What is cosine similarity?"
#
#   single    → One round of hybrid retrieval (BM25 + dense + rerank).
#               Used for: specific facts that exist in the document.
#               Example: "What revenue did the company report in Q3?"
#
#   multi_hop → Query decomposition + iterative retrieval.
#               Used for: multi-step reasoning, comparisons, synthesis.
#               Example: "How does the method in section 3 address the
#                         limitations described in section 6?"
#
#   graph     → Knowledge graph traversal via Neo4j Cypher queries.
#               Used for: entity relationships, co-occurrence, citations.
#               Example: "Which authors appear in both cited papers?"
# ──────────────────────────────────────────────────────────────────────────────
QueryType = Literal["direct", "single", "multi_hop", "graph"]


class AgentState(TypedDict, total=False):
    """
    The single shared state object that flows through the entire ARAP pipeline.

    total=False means ALL fields are optional — TypedDict normally requires
    all fields to be present. With total=False, nodes only need to return
    the fields they actually modified.

    Grouped by pipeline phase for readability.
    """

    # ── Identity ───────────────────────────────────────────────────────────────
    # These are set at the entry point and remain constant throughout the run.
    session_id: str
    # Unique per browser tab / API call. Used as LangGraph thread_id for
    # Redis-backed checkpointing — enables multi-turn conversation memory.

    user_id: str
    # Identifies the user for Mem0 long-term memory lookup and storage.
    # Can be an email, UUID, or any stable identifier.

    # ── Input ──────────────────────────────────────────────────────────────────
    question: str
    # The user's raw question, exactly as typed. Never modified after entry.

    doc_id: str | None
    # Optional scope filter. If set, retrieval is restricted to chunks from
    # this document. If None, retrieval searches across all documents.

    top_k: int
    # Number of chunks to send to the generator after re-ranking.
    # Default from settings.top_k_final (5). Can be overridden per request.

    # ── Ingestion path ─────────────────────────────────────────────────────────
    # These fields are only used during document ingestion, not during queries.
    raw_bytes: bytes | None
    # Raw PDF binary. Stored in state so the chunk node can process it.
    # Cleaned out before any streaming or logging (not JSON-serializable).

    filename: str | None
    # Original filename, stored in chunk metadata for source attribution.

    chunks: list[dict]
    # List of chunk dicts after PDFChunker runs. Each chunk has:
    #   {"text": str, "page": int, "filename": str, "doc_id": str, "chunk_index": int}

    embeddings: list[list[float]]
    # Parallel list to chunks. embeddings[i] is the vector for chunks[i].
    # Cleaned before streaming — float lists are large and not useful to clients.

    chunk_count: int
    # Total chunks stored. Returned to the client in the /ingest response.

    kg_entities: list[dict]
    # Entity/relation triples extracted from chunks and written to Neo4j.
    # Each triple: {"head": str, "relation": str, "tail": str, "confidence": float}

    # ── Adaptive routing ───────────────────────────────────────────────────────
    query_type: QueryType
    # Set by the router node. Determines which retrieval branch executes.

    routing_confidence: float
    # Router's confidence in its classification (0.0–1.0).
    # Logged to LangSmith for monitoring routing accuracy over time.

    # ── Query transformation ───────────────────────────────────────────────────
    sub_questions: list[str]
    # For multi_hop queries: the decomposed atomic sub-questions.
    # Example: "How does X compare to Y?" →
    #   ["What is X?", "What is Y?", "How do X and Y differ?"]

    rewritten_query: str
    # HyDE (Hypothetical Document Embeddings) rewrite of the question.
    # Instead of embedding the question, we embed a hypothetical answer —
    # this reduces the vocabulary mismatch between questions and documents.
    # Reference: Gao et al. (2022), arXiv:2212.10496

    # ── Retrieval results ──────────────────────────────────────────────────────
    retrieved_chunks: list[dict]
    # Top-k chunks after hybrid search + cross-encoder reranking.
    # Each chunk: {text, page, filename, doc_id, rrf_score, rerank_score}

    kg_paths: list[dict]
    # Knowledge graph traversal results for graph-type queries.
    # Each path is a dict of Cypher result fields (node names, edge types).

    retrieval_score: float
    # Average rerank score of retrieved chunks. Used for quality monitoring.

    # ── Long-term memory ───────────────────────────────────────────────────────
    long_term_memories: list[dict]
    # Fetched from Mem0 at routing time. Contains past facts about this user.
    # Each memory: {"memory": str, "score": float}
    # Injected into the generator prompt to personalize responses.

    session_context: str
    # Redis-backed summary of the current session conversation.
    # Enables multi-turn Q&A without re-sending full history each time.

    # ── Generation ─────────────────────────────────────────────────────────────
    draft_answer: str
    # The raw LLM output before faithfulness judging.
    # Kept separate from `answer` so retries can compare drafts.

    answer: str
    # The FINAL answer after passing the faithfulness judge.
    # This is what gets returned to the client and stored in Mem0.

    sources: list[dict]
    # Formatted source list for client display. Each source:
    #   {"index": int, "text": str (truncated), "page": int,
    #    "filename": str, "doc_id": str, "rerank_score": float}

    # ── Faithfulness judging ───────────────────────────────────────────────────
    faithfulness_score: float
    # Average NLI entailment score across all answer sentences (0.0–1.0).
    # 1.0 = every sentence is fully entailed by retrieved context.
    # 0.0 = no sentences are supported (pure hallucination).

    judge_passed: bool
    # True if faithfulness_score >= settings.faithfulness_threshold.
    # When True (or retry_count >= max_retries), the answer is finalized.

    retry_count: int
    # How many generation retries have occurred. Initialized to 0.
    # Incremented by the judge node on failure. Caps at settings.max_retries.

    # ── Evaluation ────────────────────────────────────────────────────────────
    ragas_scores: dict[str, float]
    # Set during evaluation runs, not during normal query execution.
    # Keys: faithfulness, answer_relevancy, context_precision, context_recall

    # ── Observability ─────────────────────────────────────────────────────────
    error: str | None
    # Set if any node raises an exception. Allows graceful error responses.

    latency_ms: dict[str, float]
    # Per-node execution time in milliseconds. Accumulated across nodes.
    # Example: {"router": 312, "retrieval": 847, "generation": 1240, "judge": 180}
    # Logged to LangSmith and returned in API responses for performance analysis.
'''


# =============================================================================
# SECTION 4 — requirements.txt
# =============================================================================
#
# Save this block as: requirements.txt
# ─────────────────────────────────────

REQUIREMENTS_TXT = '''
# =============================================================================
# ARAP — Python Dependencies
# =============================================================================
# Every package is pinned to a specific version for reproducibility.
# Comments explain WHY each package is here — not just what it does.

# ── Web framework ──────────────────────────────────────────────────────────────
fastapi==0.115.0
# Async-first REST framework. Chosen over Flask/Django because:
#   - Native async/await support for WebSocket streaming
#   - Automatic OpenAPI documentation at /docs
#   - Pydantic integration for request/response validation

uvicorn[standard]==0.30.6
# ASGI server for FastAPI. [standard] includes websockets and httptools.

pydantic==2.8.2
# Data validation. Used for request/response schemas and settings.

pydantic-settings==2.4.0
# Loads config from environment variables and .env files.
# Separate package from pydantic since v2 — required for Settings class.

python-multipart==0.0.12
# Enables file upload parsing in FastAPI (needed for /ingest endpoint).

websockets==13.1
# WebSocket protocol support for real-time streaming responses.

# ── LangGraph / LangChain ecosystem ───────────────────────────────────────────
langgraph==0.2.50
# The orchestration framework. Manages the stateful agent graph,
# conditional edges, retry loops, and Redis-backed checkpointing.
# Chosen over plain LangChain because cyclic graphs (retry loops)
# are first-class citizens.

langgraph-checkpoint-redis==0.0.6
# Redis backend for LangGraph's checkpointer.
# Enables conversation state persistence across HTTP requests.

langchain==0.3.7
# Core LangChain primitives: messages, runnables, output parsers.

langchain-openai==0.2.10
# OpenAI-specific LangChain integration (ChatOpenAI class).

langchain-core==0.3.15
# LangChain core interfaces. Installed explicitly to pin the version.

langsmith==0.1.141
# Observability SDK. Traces every node in LangGraph automatically
# when LANGCHAIN_TRACING_V2=true. No code changes needed.

# ── LLM provider ──────────────────────────────────────────────────────────────
openai==1.51.0
# OpenAI Python SDK. Used by langchain-openai under the hood.

# ── PDF parsing ────────────────────────────────────────────────────────────────
pymupdf==1.24.10
# Layout-aware PDF text extraction. Chosen over PyPDF2/pdfplumber because:
#   - Preserves paragraph structure better than alternatives
#   - Provides page numbers per text block (essential for source citation)
#   - Fast C-based implementation

# ── Embeddings ─────────────────────────────────────────────────────────────────
sentence-transformers==3.2.1
# Local embedding inference. Downloads models from HuggingFace Hub.
# Used for:
#   - Document chunk embedding (MiniLM-L6-v2, default)
#   - Query embedding at search time
#   - Cross-encoder reranking (ms-marco-MiniLM)
#   - NLI faithfulness scoring (nli-deberta-v3-small)

# ── Vector database ────────────────────────────────────────────────────────────
qdrant-client==1.12.0
# Qdrant Python client. Supports both REST and gRPC.
# Qdrant chosen over alternatives:
#   vs FAISS: Qdrant has built-in metadata filtering and HTTP API
#   vs Pinecone: self-hostable, no per-query cost
#   vs Chroma: better production scalability and filtering

# ── BM25 keyword search ────────────────────────────────────────────────────────
rank-bm25==0.2.2
# BM25Okapi implementation. In-memory, no extra infrastructure.
# Provides keyword matching as the sparse component of hybrid search.
# BM25 catches exact term matches that dense search misses (names, IDs, codes).

# ── Knowledge graph ────────────────────────────────────────────────────────────
neo4j==5.25.0
# Official Neo4j Python driver. Bolt protocol.
# Neo4j chosen for:
#   - Cypher query language (readable, expressive)
#   - APOC plugin ecosystem for graph algorithms
#   - Free community edition sufficient for this project

# ── Long-term memory ───────────────────────────────────────────────────────────
mem0ai==0.1.29
# Mem0 Python client for long-term agent memory.
# Can connect to Mem0 cloud (api_key) or self-hosted (base_url).
# Automatically extracts, deduplicates, and retrieves semantic memories.

# ── Databases ──────────────────────────────────────────────────────────────────
asyncpg==0.30.0
# Async PostgreSQL driver. Used for query history logging.
# asyncpg is 3-5x faster than psycopg2 for async workloads.

sqlalchemy[asyncio]==2.0.36
# ORM layer over asyncpg. Used for structured DB access in services.

psycopg2-binary==2.9.10
# Sync PostgreSQL driver. Used by Alembic migrations and Celery.

redis==5.2.0
# Redis Python client. Used for session TTL management and Celery.

# ── Async task queue ───────────────────────────────────────────────────────────
celery[redis]==5.4.0
# Distributed task queue. Handles background PDF ingestion.
# /ingest endpoint enqueues a task and returns immediately.
# Worker process consumes the queue and does the heavy lifting.

flower==2.0.1
# Celery monitoring web UI. Access at http://localhost:5555.
# Shows task queue depth, worker status, and task history.

# ── Evaluation ─────────────────────────────────────────────────────────────────
ragas==0.1.21
# RAGAS: Retrieval Augmented Generation Assessment.
# Computes: faithfulness, answer_relevancy, context_precision, context_recall
# Reference: Es et al. (2023), arXiv:2309.15217

datasets==3.1.0
# HuggingFace datasets library. Required by RAGAS for test set formatting.

# ── Utilities ──────────────────────────────────────────────────────────────────
python-dotenv==1.0.1
# Loads .env file. Required by pydantic-settings.

httpx==0.27.2
# Async HTTP client. Used by FastAPI TestClient in tests.

numpy==1.26.4
# Numerical computing. Required by sentence-transformers and RAGAS.

tenacity==9.0.0
# Retry decorator with exponential backoff.
# Used to wrap external API calls (OpenAI, Neo4j) for resilience.

# ── Testing ────────────────────────────────────────────────────────────────────
pytest==8.3.3
# Test runner. Run with: pytest tests/ -v --cov=app --cov-report=html

pytest-asyncio==0.24.0
# Async test support. Required for testing async FastAPI endpoints.

pytest-cov==5.0.0
# Coverage reporting. Shows which lines are untested.
'''


# =============================================================================
# SECTION 5 — .env.example
# =============================================================================
#
# Save this block as: .env.example
# Copy to .env and fill in real values before running.
# ────────────────────────────────────────────────────

ENV_EXAMPLE = '''
# =============================================================================
# ARAP — Environment Variables
# =============================================================================
# 1. Copy this file:  cp .env.example .env
# 2. Fill in OPENAI_API_KEY (required) and any optional keys
# 3. Never commit .env to git — it contains secrets

# ── REQUIRED ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY=sk-...

# ── LangSmith (strongly recommended for debugging) ────────────────────────────
# Get your key at: https://smith.langchain.com
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=arap

# ── Mem0 long-term memory ──────────────────────────────────────────────────────
# Option A — Mem0 cloud:  set MEM0_API_KEY, leave MEM0_BASE_URL empty
# Option B — Self-hosted: leave MEM0_API_KEY empty, docker compose handles it
MEM0_API_KEY=
MEM0_BASE_URL=http://localhost:3000

# ── Knowledge graph ───────────────────────────────────────────────────────────
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password_here

# ── Vector database ───────────────────────────────────────────────────────────
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=arap_docs

# ── Relational database ───────────────────────────────────────────────────────
POSTGRES_URL=postgresql://user:password@localhost:5432/arap

# ── Cache / state ─────────────────────────────────────────────────────────────
REDIS_URL=redis://localhost:6379/0

# ── LLM model selection ───────────────────────────────────────────────────────
# Change to gpt-4o-mini for everything to reduce costs during development
LLM_MODEL=gpt-4o
ROUTER_MODEL=gpt-4o-mini
JUDGE_MODEL=gpt-4o-mini

# ── Embedding model ───────────────────────────────────────────────────────────
# Default (fast, CPU):  sentence-transformers/all-MiniLM-L6-v2  dim=384
# SOTA 2026 (GPU req):  Qwen/Qwen3-Embedding-4B                 dim=2560
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384

# ── Faithfulness judge thresholds ────────────────────────────────────────────
FAITHFULNESS_THRESHOLD=0.75
NLI_MODEL=cross-encoder/nli-deberta-v3-small

# ── Retrieval parameters ──────────────────────────────────────────────────────
TOP_K_RETRIEVAL=10
TOP_K_FINAL=5
DENSE_WEIGHT=0.7
BM25_WEIGHT=0.3
'''


# =============================================================================
# SECTION 6 — .gitignore
# =============================================================================

GITIGNORE = '''
# Environment secrets — never commit
.env
*.key
*.pem

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
.venv/
venv/
env/
*.egg-info/
dist/
build/
.eggs/

# Testing & coverage
.pytest_cache/
htmlcov/
.coverage
coverage.xml

# IDE
.vscode/
.idea/
*.swp
*.swo

# Docker volumes (if mounted locally)
qdrant_storage/
neo4j_data/
postgres_data/
redis_data/

# OS
.DS_Store
Thumbs.db

# Model cache (large files)
.cache/
models/

# LangSmith local traces
.langsmith/
'''


# =============================================================================
# SECTION 7 — scripts/init_db.sql
# =============================================================================
#
# Save this block as: scripts/init_db.sql
# Run automatically by PostgreSQL on first docker compose up.
# ──────────────────────────────────────────

INIT_DB_SQL = '''
-- =============================================================================
-- ARAP — PostgreSQL Schema
-- =============================================================================
-- This file is mounted into the postgres container and executed on first boot.
-- Tables are created with IF NOT EXISTS so re-runs are safe.

-- Documents table
-- Tracks every PDF that has been ingested into the system.
-- doc_id is a SHA-256 hash of the PDF bytes (first 16 hex chars) —
-- this means the same file uploaded twice gets the same doc_id (idempotent).
CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    doc_id      VARCHAR(64) UNIQUE NOT NULL,
    filename    VARCHAR(512) NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    kg_triples  INTEGER DEFAULT 0,     -- triples stored in Neo4j for this doc
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Query history table
-- Every user query and its result is stored here.
-- Two purposes:
--   1. RAGAS evaluation — real queries make better test sets than synthetic ones
--   2. Analytics — query_type distribution shows how the router performs
CREATE TABLE IF NOT EXISTS query_history (
    id                 SERIAL PRIMARY KEY,
    session_id         VARCHAR(64),
    user_id            VARCHAR(256),
    doc_id             VARCHAR(64),
    question           TEXT NOT NULL,
    answer             TEXT,
    query_type         VARCHAR(32),       -- direct/single/multi_hop/graph
    faithfulness_score FLOAT,             -- NLI judge score (0.0-1.0)
    retrieval_score    FLOAT,             -- avg rerank score of retrieved chunks
    retry_count        INTEGER DEFAULT 0, -- how many judge retries occurred
    latency_ms         JSONB,             -- per-node latency breakdown
    created_at         TIMESTAMP DEFAULT NOW()
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_qh_session    ON query_history(session_id);
CREATE INDEX IF NOT EXISTS idx_qh_user       ON query_history(user_id);
CREATE INDEX IF NOT EXISTS idx_qh_created    ON query_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_qh_query_type ON query_history(query_type);
CREATE INDEX IF NOT EXISTS idx_docs_doc_id   ON documents(doc_id);

-- View: evaluation test set
-- Selects recent high-quality Q&A pairs for RAGAS evaluation.
-- "High quality" means faithfulness score was measured (judge ran)
-- and the answer is non-trivial (> 50 characters).
CREATE OR REPLACE VIEW evaluation_test_set AS
SELECT
    question,
    answer AS ground_truth,
    query_type,
    faithfulness_score,
    created_at
FROM query_history
WHERE
    answer IS NOT NULL
    AND LENGTH(answer) > 50
    AND faithfulness_score IS NOT NULL
ORDER BY created_at DESC
LIMIT 100;
'''


# =============================================================================
# HOW TO USE THIS FILE
# =============================================================================
#
# This is a documentation + template file. To set up Phase 1:
#
# 1. Create the directory structure:
#    mkdir -p arap/{app/{core,agents,api,services},evaluation,tests/{unit,integration},scripts,docs}
#    touch arap/app/__init__.py arap/app/core/__init__.py arap/app/agents/__init__.py
#    touch arap/app/api/__init__.py arap/app/services/__init__.py
#    touch arap/evaluation/__init__.py arap/tests/__init__.py
#
# 2. Write each section to its file:
#    The CONFIG_PY string → app/core/config.py
#    The STATE_PY string  → app/core/state.py
#    The REQUIREMENTS_TXT string → requirements.txt
#    The ENV_EXAMPLE string → .env.example
#    The GITIGNORE string → .gitignore
#    The INIT_DB_SQL string → scripts/init_db.sql
#
# 3. Create your .env:
#    cp .env.example .env
#    # Then add your OPENAI_API_KEY
#
# 4. Install dependencies:
#    python -m venv .venv
#    source .venv/bin/activate        # Windows: .venv\Scripts\activate
#    pip install -r requirements.txt
#
# 5. Verify config loads:
#    python -c "from app.core.config import settings; print(settings.llm_model)"
#    # Should print: gpt-4o
#
# 6. Verify state imports:
#    python -c "from app.core.state import AgentState, QueryType; print('OK')"
#    # Should print: OK
#
# Next: Phase 2 — Core Services (chunker, embedder, vector_store, bm25_index)
# =============================================================================
