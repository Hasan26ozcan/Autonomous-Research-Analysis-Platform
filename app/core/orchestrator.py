"""
app/core/orchestrator.py
=========================
Assembles every phase into two compiled LangGraph graphs and exposes
a clean async API that the FastAPI layer (main.py) calls.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE TWO GRAPHS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INGEST GRAPH — one-time document processing per PDF upload
─────────────────────────────────────────────────────────
  chunk_document        (Phase 2) PDF → overlapping chunks with metadata
      ↓
  enrich_chunks         (Phase 3) prepend LLM context to each chunk
      ↓
  embed_chunks          (Phase 2) sentence-transformers → dense vectors
      ↓
  store_chunks          (Phase 2) upsert vectors + payload to Qdrant
      ↓
  index_chunks          (Phase 2) add chunk texts to BM25 in-memory corpus
      ↓
  extract_and_store_node (Phase 6) LLM triple extraction → Neo4j batch write
      ↓
  END

QUERY GRAPH — per-request adaptive retrieval + generation
──────────────────────────────────────────────────────────
                     ┌─────────────────────────────────┐
  router             │  Phase 4: classify + Mem0 fetch  │
      ↓              └─────────────────────────────────┘
  get_route() conditional edge:
    "direct"    ──→ direct_answer ──────────────────────────────┐
    "single"    ──→ retrieve       (Phase 5: HyDE+hybrid+rerank)│
    "multi_hop" ──→ retrieve_multi (Phase 5: decompose+multi)   │
    "graph"     ──→ graph_retrieve (Phase 6: Neo4j Cypher)      │
                         ↓ (all three retrieval routes converge) │
                    merge_results   (no-op, convergence point)   │
                         ↓                                       │
                    generate       (Phase 7: GPT-4o + context)   │
                         ↓                                       │
                    judge          (Phase 7: NLI faithfulness)   │
                         ↓                                       │
             should_retry() conditional edge:                    │
               "generate"     ──→ generate  (retry loop)        │
               "memory_store" ──→ memory_store ←────────────────┘
                                       ↓
                                      END

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORT CONVENTION — singletons vs classes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every agent module exposes a module-level singleton:
  app.agents.router          → router_agent    (RouterAgent)
  app.agents.retrieval_agent → retrieval_agent (RetrievalAgent)
  app.agents.graph_agent     → kg_agent        (KnowledgeGraphAgent)
  app.agents.generator       → generator       (AnswerGenerator)

Service modules expose LangGraph-compatible node FUNCTIONS:
  app.services.chunker             → chunk_document
  app.services.contextual_enricher → enrich_chunks
  app.services.embedder            → embed_chunks
  app.services.vector_store        → store_chunks
  app.services.bm25_index          → index_chunks

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANGSMITH TRACING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Enabled automatically when LANGCHAIN_TRACING_V2=true in .env.
Every LangGraph node appears as a named span in LangSmith:
  - input/output state at each node
  - per-node token counts and latency
  - full conversation thread view via thread_id (= session_id)
Zero code changes needed — LangSmith SDK intercepts at the LangChain layer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REDIS CHECKPOINTER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The query graph uses a Redis-backed checkpointer keyed by session_id.
This enables multi-turn conversations: every invocation restores the
prior conversation state automatically via LangGraph's thread_id mechanism.
The ingest graph does NOT use a checkpointer — ingestion is stateless
(each PDF is processed independently; there is no conversation to persist).
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

# ── Phase 7: Generator + Judge + Memory Store ─────────────────────────────────
from app.agents.generator import generator

# ── Phase 6: Knowledge Graph Agent ────────────────────────────────────────────
from app.agents.graph_agent import kg_agent

# ── Phase 5: Retrieval Agent ───────────────────────────────────────────────────
from app.agents.retrieval_agent import retrieval_agent

# ── Phase 4: Adaptive Router ───────────────────────────────────────────────────
from app.agents.router import router_agent
from app.core.config import settings
from app.core.state import AgentState
from app.services.bm25_index import index_chunks

# ── Phase 2 + Phase 3: Service node functions ─────────────────────────────────
from app.services.chunker import chunk_document
from app.services.contextual_enricher import enrich_chunks
from app.services.embedder import embed_chunks
from app.services.llm_client import make_llm
from app.services.vector_store import store_chunks

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LANGSMITH — enable at module load time if configured
# Must happen before any LangChain/LangGraph objects are created.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if settings.langchain_tracing_v2 and settings.langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    logger.info("LangSmith tracing enabled for project '%s'", settings.langchain_project)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INGEST GRAPH — helper nodes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# All ingest pipeline node functions are imported directly from service modules
# (chunk_document, enrich_chunks, embed_chunks, store_chunks, index_chunks).
# No additional wrapper needed — they already match the LangGraph node signature:
#   (state: AgentState) -> dict   (partial state update)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QUERY GRAPH — helper nodes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def direct_answer(state: AgentState) -> dict:
    """
    LangGraph node: answer questions that need NO document retrieval.

    Called when router_agent classifies query_type="direct" — meaning the
    LLM already knows the answer from parametric knowledge (e.g. "What is
    cosine similarity?"). Running retrieval for these questions wastes ~800ms
    and adds context noise.

    Uses router_model (gpt-4o-mini) because this is a lightweight generation
    task — the question is conceptual, not document-specific. Saves llm_model
    (gpt-4o) for the heavy generation in Phase 7 generator.

    Injects long_term_memories from Mem0 (fetched by router in Phase 4)
    so the answer can be personalized ("Given that you work with flood
    prediction models...") even without document retrieval.

    Reads from AgentState:
        question            (str)
        long_term_memories  (list[dict])  from Phase 4

    Writes to AgentState:
        answer              (str)   final answer (no judging needed for direct)
        sources             (list)  empty (no documents cited)
        faithfulness_score  (float) 1.0 (parametric knowledge, no hallucination risk)
        judge_passed        (bool)  True (skip judge — nothing to ground-check)
        draft_answer        (str)   same as answer (no judging step)
    """
    llm = make_llm(
        model=settings.router_model,
        temperature=0.1,
    )

    question: str = state.get("question", "")
    memories: list[dict] = state.get("long_term_memories") or []

    # Build personalization prefix from Mem0 memories
    memory_lines = "\n".join(
        f"- {m['memory']}"
        for m in memories[:5]
        if m.get("memory")
    )
    if memory_lines:
        prompt = (
            f"[User context from memory]\n{memory_lines}\n\n"
            f"Question: {question}"
        )
    else:
        prompt = question

    response = llm.invoke([
        SystemMessage(content=(
            "You are a helpful AI assistant. "
            "Answer the user's question concisely and accurately. "
            "If user context is provided, personalize your answer accordingly."
        )),
        HumanMessage(content=prompt),
    ])
    answer = response.content.strip() if isinstance(response.content, str) else ""

    return {
        "answer":            answer,
        "draft_answer":      answer,
        "sources":           [],
        "faithfulness_score": 1.0,
        "judge_passed":      True,
    }


def merge_results(state: AgentState) -> dict | None:
    """
    LangGraph node: convergence point for all retrieval branches.

    All three retrieval routes (retrieve, retrieve_multi, graph_retrieve)
    connect to this node before passing to generate(). This node is
    currently a no-op pass-through (no state changes), but it serves as a
    clean architectural convergence point.

    Future enhancement: deduplicate retrieved_chunks if the graph and
    retrieval agents returned overlapping chunks.

    IMPORTANT — do NOT return {} here:
      In LangGraph 0.2.x, returning an empty dict {} from a node raises
      InvalidUpdateError ("Expected node session_id to update at least one
      of [...], got {}"). This is because the empty dict vacuously satisfies
      "none of the returned keys are valid state channels". The correct
      no-op for a node that intentionally changes nothing is to return None,
      which LangGraph translates to SKIP_WRITE for every channel (state left
      unchanged). Return {} will crash the whole query graph.
    """
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ORCHESTRATOR CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ARAPOrchestrator:
    """
    Assembles all agent and service nodes into compiled LangGraph graphs
    and provides the async API that FastAPI (main.py) calls.

    Caching:
        Both graphs are compiled once and cached in instance variables.
        LangGraph compilation is expensive (validates graph structure,
        resolves all edges, wires the checkpointer). Recompiling per
        request would add 50-200ms overhead. The cached graphs are
        thread-safe for concurrent invocation.

    Checkpointer:
        The query graph uses a Redis-backed LangGraph checkpointer.
        Redis key = thread_id = session_id, so every session's state
        is isolated and persisted across HTTP requests automatically.
        The ingest graph has no checkpointer — it's stateless per run.
    """

    def __init__(self):
        self._ingest_graph = None
        self._query_graph = None
        self._checkpointer = None

    # ── Checkpointer ──────────────────────────────────────────────────────────

    @property
    def checkpointer(self):
        """
        Lazy Redis checkpointer — created on first query graph access.

        RedisSaver stores LangGraph conversation state in Redis with
        TTL = settings.session_ttl_seconds (default 1 hour).
        Each session_id gets its own Redis key so sessions are isolated.

        Falls back to None if Redis is unreachable — the graph still
        works without checkpointing (no multi-turn memory, but no crash).
        """
        if self._checkpointer is None:
            try:
                from langgraph.checkpoint.redis import RedisSaver
                # NOTE: RedisSaver.from_conn_string() is a @contextmanager - calling
                # it directly (without `with ... as saver:`) returns a
                # _GeneratorContextManager object, not an actual RedisSaver. That
                # object doesn't implement the checkpointer interface (e.g. it has
                # no get_next_version), which is why compiling the graph with it
                # raised: "'_GeneratorContextManager' object has no attribute
                # 'get_next_version'". Since this checkpointer needs to live for
                # the whole lifetime of the app (not just a `with` block), we
                # construct RedisSaver directly instead and call .setup() once to
                # create its Redis search indices.
                self._checkpointer = RedisSaver(redis_url=settings.redis_url)
                self._checkpointer.setup()
                logger.info("LangGraph Redis checkpointer initialized.")
            except Exception as e:
                logger.warning(
                    "Redis checkpointer init failed (multi-turn memory disabled): %s", e
                )
                self._checkpointer = None
        return self._checkpointer

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Graph builders
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def build_ingest_graph(self):
        """
        Build and compile the document ingestion LangGraph.

        Node execution order (strictly linear, no branching):
          chunk_document        → PDF bytes → overlapping chunks (Phase 2)
          enrich_chunks         → LLM context prepended to each chunk (Phase 3)
          embed_chunks          → sentence-transformers dense vectors (Phase 2)
          store_chunks          → Qdrant HNSW upsert (Phase 2)
          index_chunks          → BM25 in-memory corpus update (Phase 2)
          extract_and_store_node → Neo4j triple extraction + batch write (Phase 6)

        Why enrich BEFORE embed?
          The contextual enricher (Phase 3) prepends a 2-3 sentence context
          description to each chunk's text. We embed the ENRICHED text so
          the stored vectors encode both content AND document context.
          If we embedded before enriching, we'd have to re-embed after,
          doubling the embedding cost with no benefit.

        Why KG extraction LAST?
          KG extraction is the slowest step (~1 LLM call per chunk).
          Running it last means the document is already fully searchable
          via Qdrant + BM25 before KG extraction completes. If KG
          extraction fails or times out, the document is still queryable
          via vector and keyword search — graceful degradation.
        """
        g = StateGraph(AgentState)

        # Register all nodes with their LangGraph-compatible node functions
        g.add_node("chunk",      chunk_document)                    # Phase 2
        g.add_node("enrich",     enrich_chunks)                     # Phase 3
        g.add_node("embed",      embed_chunks)                      # Phase 2
        g.add_node("store",      store_chunks)                      # Phase 2
        g.add_node("index_bm25", index_chunks)                      # Phase 2
        g.add_node("extract_kg", kg_agent.extract_and_store_node)   # Phase 6

        # Linear pipeline — each step feeds directly into the next
        g.set_entry_point("chunk")
        g.add_edge("chunk",      "enrich")
        g.add_edge("enrich",     "embed")
        g.add_edge("embed",      "store")
        g.add_edge("store",      "index_bm25")
        g.add_edge("index_bm25", "extract_kg")
        g.add_edge("extract_kg", END)

        # No checkpointer for ingest — stateless per document
        return g.compile()

    def build_query_graph(self):
        """
        Build and compile the adaptive query LangGraph.

        Graph topology:
          Entry: router
          Branches: direct / retrieve / retrieve_multi / graph_retrieve
          Convergence: merge → generate → judge
          Retry loop: judge → (rejected) → generate → judge
          Exit: memory_store → END

        Conditional edges:
          1. router → get_route() → {direct, single, multi_hop, graph}
             Routes to the appropriate retrieval strategy per query.
          2. judge → should_retry() → {generate, memory_store}
             Loops back to generate if faithfulness score is too low,
             proceeds to memory_store when judge approves or retries exhausted.

        The retry loop is the graph's only cycle. LangGraph supports cycles
        explicitly (unlike DAG-only frameworks) — this is one of the reasons
        we chose LangGraph over plain LangChain for ARAP.
        """
        g = StateGraph(AgentState)

        # ── Register all nodes ─────────────────────────────────────────────────
        g.add_node("router",         router_agent.route)           # Phase 4
        g.add_node("direct",         direct_answer)                # Phase 8 (this file)
        g.add_node("retrieve",       retrieval_agent.retrieve)     # Phase 5
        g.add_node("retrieve_multi", retrieval_agent.retrieve_multi) # Phase 5
        g.add_node("graph_retrieve", kg_agent.graph_retrieve)      # Phase 6
        g.add_node("merge",          merge_results)                # Phase 8 (this file)
        g.add_node("generate",       generator.generate)           # Phase 7
        g.add_node("judge",          generator.judge)              # Phase 7
        g.add_node("memory_store",   generator.store_memory)       # Phase 7

        # ── Entry point ────────────────────────────────────────────────────────
        g.set_entry_point("router")

        # ── Conditional edge 1: router → retrieval branch ─────────────────────
        # router_agent.get_route() reads query_type from state and returns
        # one of "direct", "single", "multi_hop", "graph".
        # The map below translates each value to the corresponding node name.
        g.add_conditional_edges(
            "router",
            router_agent.get_route,
            {
                "direct":    "direct",
                "single":    "retrieve",
                "multi_hop": "retrieve_multi",
                "graph":     "graph_retrieve",
            },
        )

        # ── All retrieval branches converge at merge ───────────────────────────
        g.add_edge("retrieve",       "merge")
        g.add_edge("retrieve_multi", "merge")
        g.add_edge("graph_retrieve", "merge")

        # ── Linear: merge → generate → judge ──────────────────────────────────
        g.add_edge("merge",    "generate")
        g.add_edge("generate", "judge")

        # ── Conditional edge 2: judge → retry loop or proceed ─────────────────
        # generator.should_retry() reads judge_passed from state.
        # "generate" → retry (loop back with stricter prompt)
        # "memory_store" → proceed to persist and finish
        g.add_conditional_edges(
            "judge",
            generator.should_retry,
            {
                "generate":     "generate",
                "memory_store": "memory_store",
            },
        )

        # ── Direct path also flows through memory_store ────────────────────────
        # Even direct answers are stored in Mem0 for long-term memory.
        g.add_edge("direct",       "memory_store")

        # ── Terminal edge ──────────────────────────────────────────────────────
        g.add_edge("memory_store", END)

        # Compile WITH Redis checkpointer for multi-turn conversation support
        return g.compile(checkpointer=self.checkpointer)

    # ── Cached graph properties ───────────────────────────────────────────────

    @property
    def ingest_graph(self):
        """Lazy compile and cache the ingest graph."""
        if self._ingest_graph is None:
            logger.info("Compiling ingest graph...")
            self._ingest_graph = self.build_ingest_graph()
            logger.info("Ingest graph compiled.")
        return self._ingest_graph

    @property
    def query_graph(self):
        """Lazy compile and cache the query graph."""
        if self._query_graph is None:
            logger.info("Compiling query graph...")
            self._query_graph = self.build_query_graph()
            logger.info("Query graph compiled.")
        return self._query_graph

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Public Async API — called by FastAPI endpoints
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def ingest(self, pdf_bytes: bytes, filename: str) -> dict:
        """
        Run the ingest pipeline for one PDF.

        Delegates to ingest_graph.invoke() via asyncio.to_thread() because
        LangGraph's synchronous invoke() would block the FastAPI event loop.
        All LLM calls inside the graph (enricher, KG extractor) are
        synchronous ChatOpenAI calls — running them in a thread pool keeps
        the async event loop responsive to other requests during ingestion.

        Args:
            pdf_bytes: Raw PDF binary from the uploaded file.
            filename:  Original filename for metadata and source attribution.

        Returns:
            {
                "doc_id":      str,  SHA-256 content hash (first 16 hex chars)
                "chunk_count": int,  total chunks stored in Qdrant + BM25
                "kg_triples":  int,  total entity-relation triples in Neo4j
            }
        """
        init_state: AgentState = {
            "raw_bytes":   pdf_bytes,
            "filename":    filename,
            "chunks":      [],
            "embeddings":  [],
            "chunk_count": 0,
            "kg_entities": [],
        }

        # Run the blocking graph in a thread pool to avoid blocking event loop
        final_state = await asyncio.to_thread(
            self.ingest_graph.invoke,
            init_state,
        )

        result = {
            "doc_id":      final_state.get("doc_id"),
            "chunk_count": final_state.get("chunk_count", 0),
            "kg_triples":  len(final_state.get("kg_entities") or []),
        }

        # Best-effort audit log - never blocks or fails the response.
        # These are synchronous (psycopg2) and best-effort.
        from app.services.postgres_store import (
            record_chunk_metadata,
            record_document,
            update_document_status,
        )
        update_document_status(result["doc_id"], "processing")
        record_chunk_metadata(result["doc_id"], final_state.get("chunks") or [])
        record_document(
            doc_id=result["doc_id"],
            filename=filename,
            chunk_count=result["chunk_count"],
            kg_triples=result["kg_triples"],
            status="ready",
        )

        return result

    async def query(
        self,
        question: str,
        session_id: str,
        user_id: str,
        doc_id: str | None = None,
        top_k: int = settings.top_k_final,
    ) -> dict:
        """
        Run the full adaptive query pipeline for one question.

        Returns the final answer, sources, and observability metadata.
        This is a synchronous-result endpoint — the caller waits for
        the complete pipeline to finish before receiving a response.

        For real-time streaming, use stream_query() instead.

        Args:
            question:   Raw user question string.
            session_id: Unique conversation identifier (e.g. browser tab UUID).
                        Used as LangGraph thread_id for Redis checkpointing.
            user_id:    Stable user identifier for Mem0 memory personalization.
            doc_id:     Optional — scope retrieval to a single document.
                        None = search across all ingested documents.
            top_k:      Number of chunks to send to the generator.

        Returns:
            {
                "answer":            str,         final approved answer
                "sources":           list[dict],  formatted source citations
                "query_type":        str,          "direct"|"single"|"multi_hop"|"graph"
                "faithfulness_score": float,       NLI entailment score (0.0-1.0)
                "latency_ms":        dict,         per-node timing breakdown
            }
        """
        init_state: AgentState = {
            "question":    question,
            "session_id":  session_id,
            "user_id":     user_id,
            "doc_id":      doc_id,
            "top_k":       top_k,
            "retry_count": 0,
            "latency_ms":  {},
        }

        # Phase 9: reset the process-wide token counter so the count returned
        # at the end reflects only THIS query's LLM calls.
        from app.services.llm_client import (
            get_and_reset_token_usage,
            reset_token_usage,
        )
        reset_token_usage()

        # LangGraph thread_id = session_id → Redis checkpoint key
        config = {"configurable": {"thread_id": session_id}}

        final_state = await asyncio.to_thread(
            self.query_graph.invoke,
            init_state,
            config,
        )

        token_usage = get_and_reset_token_usage()

        # Phase 10: log per-node latency to pipeline_log (best-effort).
        try:
            from app.services.log_store import log_pipeline_batch
            log_pipeline_batch(
                session_id,
                final_state.get("query_type"),
                final_state.get("latency_ms") or {},
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("pipeline log failed (non-fatal): %s", e)

        result = {
            "answer":             final_state.get("answer", ""),
            "sources":            final_state.get("sources") or [],
            "query_type":         final_state.get("query_type"),
            "faithfulness_score": final_state.get("faithfulness_score"),
            "latency_ms":         final_state.get("latency_ms") or {},
            "token_usage":        token_usage,
        }

        # Best-effort audit log - never blocks or fails the response.
        # Synchronous (psycopg2) and best-effort.
        from app.services.postgres_store import record_query
        record_query(
            session_id=session_id,
            user_id=user_id,
            doc_id=doc_id,
            question=question,
            answer=result["answer"],
            query_type=result["query_type"],
            faithfulness_score=result["faithfulness_score"],
            retrieval_score=final_state.get("retrieval_score"),
            retry_count=final_state.get("retry_count") or 0,
            latency_ms=result["latency_ms"],
        )

        return result

    async def stream_query(
        self,
        question: str,
        session_id: str,
        user_id: str,
        doc_id: str | None = None,
    ) -> AsyncIterator[dict]:
        """
        Stream intermediate state updates as the query graph executes.

        Used by the WebSocket endpoint in main.py. The client receives
        one event per LangGraph node execution, enabling real-time UI
        updates:
          {"node": "router",       "data": {"query_type": "single", ...}}
          {"node": "retrieve",     "data": {"retrieved_chunks": [...], ...}}
          {"node": "generate",     "data": {"draft_answer": "...", ...}}
          {"node": "judge",        "data": {"faithfulness_score": 0.91, ...}}
          {"node": "memory_store", "data": {}}

        Non-serializable fields (raw_bytes, embeddings) are stripped from
        every update before yielding — they are large, not useful to clients,
        and cannot be JSON-serialized.

        Args:
            question:   Raw user question.
            session_id: Conversation thread identifier.
            user_id:    Mem0 user identifier.
            doc_id:     Optional document scope filter.

        Yields:
            {"node": str, "data": dict}  one dict per node execution
        """
        init_state: AgentState = {
            "question":    question,
            "session_id":  session_id,
            "user_id":     user_id,
            "doc_id":      doc_id,
            "top_k":       settings.top_k_final,
            "retry_count": 0,
            "latency_ms":  {},
        }
        config = {"configurable": {"thread_id": session_id}}

        # query_graph.stream() yields one event per node, mode="updates"
        # means each event is {node_name: partial_state_update}
        for event in self.query_graph.stream(init_state, config, stream_mode="updates"):
            for node_name, node_output in event.items():
                yield {
                    "node": node_name,
                    "data": _safe_serialize(node_output),
                }

    async def health(self) -> dict:
        """
        Check reachability of all infrastructure components.

        Called by the /health FastAPI endpoint. Each check is
        independent — one failure does not block the others.

        Returns:
            {
                "qdrant": "ok" | "unreachable",
                "neo4j":  "ok" | "unreachable",
                "redis":  "ok" | "unreachable",
            }
        """
        status: dict[str, str] = {}

        # Each check is a blocking I/O call; run them off the event loop in
        # worker threads so this coroutine awaits real work instead of blocking
        # the loop. The checks remain independent — one failure does not affect
        # the others (gather with return_exceptions=True).
        def _check_qdrant() -> bool:
            try:
                from qdrant_client import QdrantClient
                client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                    timeout=2,
                )
                client.get_collections()
                return True
            except Exception:
                return False

        def _check_neo4j() -> bool:
            try:
                from neo4j import GraphDatabase
                driver = GraphDatabase.driver(
                    settings.neo4j_uri,
                    auth=(settings.neo4j_user, settings.neo4j_password),
                )
                driver.verify_connectivity()
                driver.close()
                return True
            except Exception:
                return False

        def _check_redis() -> bool:
            try:
                import redis as redis_lib
                r = redis_lib.from_url(settings.redis_url, socket_connect_timeout=2)
                r.ping()
                return True
            except Exception:
                return False

        ok_qdrant, ok_neo4j, ok_redis = await asyncio.gather(
            asyncio.to_thread(_check_qdrant),
            asyncio.to_thread(_check_neo4j),
            asyncio.to_thread(_check_redis),
            return_exceptions=True,
        )

        status["qdrant"] = "ok" if ok_qdrant is True else "unreachable"
        status["neo4j"] = "ok" if ok_neo4j is True else "unreachable"
        status["redis"] = "ok" if ok_redis is True else "unreachable"

        return status


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _safe_serialize(state: dict) -> dict:
    """
    Strip non-JSON-serializable fields from a state update before streaming.

    Fields stripped:
      raw_bytes   — PDF binary data (bytes, not JSON-serializable, large)
      embeddings  — float vectors (list[list[float]], very large, not useful to client)

    Also strips None values — clients should treat missing keys as None.

    Args:
        state: A partial state update dict from a LangGraph node.

    Returns:
        Cleaned dict safe for json.dumps() and WebSocket delivery.
    """
    _STRIP = {"raw_bytes", "embeddings"}
    return {
        k: v
        for k, v in state.items()
        if k not in _STRIP and v is not None
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Module-Level Singleton
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

orchestrator = ARAPOrchestrator()
