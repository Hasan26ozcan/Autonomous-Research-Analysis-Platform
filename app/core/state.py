"""
app/core/state.py
==================
Shared state schema for all LangGraph agents in ARAP.

LangGraph fundamentals:
  - Each node is a Python function: (AgentState) -> dict (partial update)
  - Nodes receive the FULL state, return ONLY the fields they changed
  - LangGraph merges partial updates automatically
  - Conditional edges are functions: (AgentState) -> str (next node name)

Why TypedDict with total=False?
  All fields are optional because no single node sets all of them.
  Nodes only return what they changed — no need to copy the whole state.

State flow by node:
  Entry point   → session_id, user_id, question, doc_id, top_k
  router        → query_type, routing_confidence, long_term_memories
  retrieval     → rewritten_query, retrieved_chunks, retrieval_score
  retrieval_multi → sub_questions + all retrieval fields
  graph_retrieve  → kg_paths + retrieval fields
  generator     → draft_answer, sources
  judge         → faithfulness_score, judge_passed, answer (if passed)
  memory_store  → (side effect only — no state changes)
"""

from __future__ import annotations
from typing import TypedDict, Literal


# Adaptive RAG query taxonomy — Jeong et al. (2024), arXiv:2403.14403
QueryType = Literal["direct", "single", "multi_hop", "graph"]


class AgentState(TypedDict, total=False):
    """
    Single shared state object flowing through the entire ARAP pipeline.
    Every field is optional (total=False) — nodes only set what they touch.
    """

    # ── Identity (set at entry, constant throughout) ───────────────────────────
    session_id: str
    # Unique per conversation. Used as LangGraph thread_id for Redis checkpointing.

    user_id: str
    # Stable user identifier. Used as Mem0 user_id for long-term memory.

    # ── Input ──────────────────────────────────────────────────────────────────
    question: str
    # Raw user question — never modified. All rewrites produce new fields.

    doc_id: str | None
    # Optional: restrict retrieval to a single document. None = search all docs.

    top_k: int
    # Chunks to send to LLM after reranking. Default: settings.top_k_final (5).

    # ── Ingestion path (only used during /ingest, not during /query) ───────────
    raw_bytes: bytes | None
    # Raw PDF binary. Cleaned before streaming — not JSON-serializable.

    filename: str | None
    # Original PDF filename. Stored in chunk metadata for source attribution.

    chunks: list[dict]
    # Output of PDFChunker. Each chunk:
    # {"text": str, "page": int, "filename": str, "doc_id": str,
    #  "chunk_index": int, "context_prepended": bool}

    embeddings: list[list[float]]
    # embeddings[i] is the dense vector for chunks[i].
    # Cleaned before streaming — large and not useful to clients.

    chunk_count: int
    # Total chunks stored. Returned in /ingest response.

    kg_entities: list[dict]
    # Entity/relation triples extracted and written to Neo4j during ingestion.
    # Each triple: {"head": str, "relation": str, "tail": str, "confidence": float}

    # ── Adaptive routing ───────────────────────────────────────────────────────
    query_type: QueryType
    # Set by router. LangGraph reads via router.get_route() conditional edge.

    routing_confidence: float
    # Router's confidence (0.0–1.0). Logged for monitoring accuracy drift.

    # ── Query transformation ───────────────────────────────────────────────────
    sub_questions: list[str]
    # multi_hop only: atomic sub-questions from query decomposition.
    # Each retrieved independently; results merged before generation.

    rewritten_query: str
    # HyDE rewrite: hypothetical answer embedded instead of raw question.
    # Reduces vocabulary mismatch between queries and documents.
    # Reference: Gao et al. (2022), arXiv:2212.10496

    # ── Retrieval results ──────────────────────────────────────────────────────
    retrieved_chunks: list[dict]
    # Final top-k chunks after hybrid search + cross-encoder reranking.
    # Each chunk: {text, page, filename, doc_id, rrf_score, rerank_score}

    kg_paths: list[dict]
    # Neo4j Cypher traversal results for graph-type queries.

    retrieval_score: float
    # Average rerank score of retrieved chunks. Logged for quality monitoring.

    # ── Long-term memory ───────────────────────────────────────────────────────
    long_term_memories: list[dict]
    # Fetched from Mem0 at routing time. Past facts about this user.
    # Each memory: {"memory": str, "score": float}
    # Injected into generator prompt to personalize responses.

    session_context: str
    # Redis-backed conversation summary for multi-turn Q&A.

    # ── Generation ─────────────────────────────────────────────────────────────
    draft_answer: str
    # Raw LLM output BEFORE faithfulness judging.
    # Kept separate from `answer` so retries can compare drafts.

    answer: str
    # FINAL answer after the faithfulness judge approves it.
    # What gets returned to client and stored in Mem0.

    sources: list[dict]
    # Formatted source list for client. Each source:
    # {"index": int, "text": str (truncated 300 chars),
    #  "page": int, "filename": str, "doc_id": str, "rerank_score": float}

    # ── Faithfulness judging ───────────────────────────────────────────────────
    faithfulness_score: float
    # Avg NLI entailment score across all answer sentences (0.0–1.0).
    # 1.0 = every sentence entailed by context (no hallucination)
    # 0.0 = no sentences supported (pure confabulation)

    judge_passed: bool
    # True when faithfulness_score >= threshold OR retry_count >= max_retries.
    # The conditional edge generator.should_retry() reads this field.

    retry_count: int
    # Initialized to 0. Incremented by judge on failure. Caps at max_retries.

    # ── Evaluation ────────────────────────────────────────────────────────────
    ragas_scores: dict[str, float]
    # Populated only during /eval runs. Keys: faithfulness, answer_relevancy,
    # context_precision, context_recall

    # ── Observability ─────────────────────────────────────────────────────────
    error: str | None
    # Set if a node raises an unrecoverable exception. Enables structured errors.

    latency_ms: dict[str, float]
    # Per-node execution time (ms), accumulated across the pipeline.
    # {"router": 312, "retrieval": 847, "generation": 1240, "judge": 180}
    # Returned in API responses + sent to LangSmith.
