"""
ARAP — Main LangGraph Orchestration
=====================================
Assembles all specialist agents into two compiled graphs:

  1. ingest_graph  — document processing pipeline
     chunk → embed → contextual_enrich → store → kg_extract

  2. query_graph   — adaptive retrieval + generation pipeline
     router → [direct | single | multi_hop | graph] → merge → generate
     → judge → [retry? | memory_store] → respond

The conditional edges implement the Adaptive RAG routing pattern:
each query type follows a different path through the graph,
avoiding unnecessary computation for simple queries
and enabling rich multi-step reasoning for complex ones.

LangSmith tracing is enabled automatically when LANGCHAIN_TRACING_V2=true.
Every node execution appears as a named span in the LangSmith dashboard.
"""
from __future__ import annotations
import os
import asyncio
import logging
from typing import AsyncIterator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import RedisSaver

from app.core.state import AgentState
from app.core.config import settings
from app.agents.router import router
from app.agents.retrieval_agent import retrieval_agent
from app.agents.graph_agent import kg_agent
from app.agents.generator import generator

logger = logging.getLogger(__name__)

# ── LangSmith tracing ──────────────────────────────────────────────────────────
if settings.langchain_tracing_v2 and settings.langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project


# ── Direct answer node (no retrieval) ─────────────────────────────────────────

def direct_answer(state: AgentState) -> AgentState:
    """
    For 'direct' queries: LLM answers from parametric knowledge.
    No retrieval. Skips the full pipeline for simple factual questions.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatOpenAI(
        model=settings.router_model,
        api_key=settings.openai_api_key,
        temperature=0.1,
    )
    memories = state.get("long_term_memories", [])
    mem_text = "\n".join(f"- {m['memory']}" for m in memories) if memories else ""
    prompt = state["question"]
    if mem_text:
        prompt = f"[User context]\n{mem_text}\n\nQuestion: {prompt}"

    resp = llm.invoke([
        SystemMessage(content="You are a helpful AI assistant. Answer the user's question concisely."),
        HumanMessage(content=prompt),
    ])
    return {
        "answer": resp.content.strip(),
        "sources": [],
        "faithfulness_score": 1.0,
        "judge_passed": True,
    }


# ── No-op merge node ───────────────────────────────────────────────────────────

def merge_results(state: AgentState) -> AgentState:
    """
    Collects results from parallel retrieval branches.
    Currently a pass-through; could deduplicate chunks across branches.
    """
    return {}


# ── Graph builders ─────────────────────────────────────────────────────────────

class ARAPOrchestrator:
    """
    Builds and caches compiled LangGraph graphs.
    Provides async streaming for WebSocket delivery.
    """

    def __init__(self):
        self._checkpointer = None
        self._ingest_graph = None
        self._query_graph = None

    @property
    def checkpointer(self):
        """Redis-backed checkpointer for conversation persistence."""
        if self._checkpointer is None:
            self._checkpointer = RedisSaver.from_conn_string(settings.redis_url)
        return self._checkpointer

    def build_ingest_graph(self):
        """Document ingestion pipeline."""
        from app.services.chunker import chunk_document
        from app.services.embedder import embed_chunks
        from app.services.contextual_enricher import enrich_chunks
        from app.services.vector_store import store_chunks
        from app.services.bm25_index import index_chunks

        g = StateGraph(AgentState)
        g.add_node("chunk", chunk_document)
        g.add_node("embed", embed_chunks)
        g.add_node("enrich", enrich_chunks)       # contextual retrieval
        g.add_node("store_vectors", store_chunks)
        g.add_node("index_bm25", index_chunks)
        g.add_node("extract_kg", kg_agent.extract_and_store_node)

        g.set_entry_point("chunk")
        g.add_edge("chunk", "embed")
        g.add_edge("embed", "enrich")
        g.add_edge("enrich", "store_vectors")
        g.add_edge("store_vectors", "index_bm25")
        g.add_edge("index_bm25", "extract_kg")
        g.add_edge("extract_kg", END)

        return g.compile()

    def build_query_graph(self):
        """
        Adaptive query pipeline with conditional routing.

        Routing logic:
          router → get_route() → {
            "direct"    → direct_answer → memory_store → END
            "single"    → retrieve → merge → generate → judge → ...
            "multi_hop" → retrieve_multi_hop → merge → generate → judge → ...
            "graph"     → graph_retrieve → merge → generate → judge → ...
          }

          judge → should_retry() → {
            judge_passed or max_retries → memory_store → END
            else                        → generate (retry with stricter prompt)
          }
        """
        g = StateGraph(AgentState)

        # Nodes
        g.add_node("router", router.route)
        g.add_node("direct", direct_answer)
        g.add_node("retrieve", retrieval_agent.retrieve)
        g.add_node("retrieve_multi", retrieval_agent.retrieve_multi_hop)
        g.add_node("graph_retrieve", kg_agent.graph_retrieve)
        g.add_node("merge", merge_results)
        g.add_node("generate", generator.generate)
        g.add_node("judge", generator.judge)
        g.add_node("memory_store", generator.store_memory)

        # Entry
        g.set_entry_point("router")

        # Conditional routing after router
        g.add_conditional_edges(
            "router",
            router.get_route,
            {
                "direct":    "direct",
                "single":    "retrieve",
                "multi_hop": "retrieve_multi",
                "graph":     "graph_retrieve",
            },
        )

        # All retrieval paths converge at merge
        g.add_edge("retrieve",       "merge")
        g.add_edge("retrieve_multi", "merge")
        g.add_edge("graph_retrieve", "merge")
        g.add_edge("merge",          "generate")
        g.add_edge("generate",       "judge")

        # Judge conditional: retry or finish
        g.add_conditional_edges(
            "judge",
            generator.should_retry,
            {
                "generate":     "generate",
                "memory_store": "memory_store",
            },
        )

        g.add_edge("direct",       "memory_store")
        g.add_edge("memory_store", END)

        return g.compile(checkpointer=self.checkpointer)

    @property
    def ingest_graph(self):
        if self._ingest_graph is None:
            self._ingest_graph = self.build_ingest_graph()
        return self._ingest_graph

    @property
    def query_graph(self):
        if self._query_graph is None:
            self._query_graph = self.build_query_graph()
        return self._query_graph

    # ── Public API ─────────────────────────────────────────────────────────────

    async def ingest(self, pdf_bytes: bytes, filename: str) -> dict:
        init: AgentState = {
            "raw_bytes": pdf_bytes,
            "filename": filename,
            "chunks": [],
            "embeddings": [],
            "chunk_count": 0,
            "kg_entities": [],
        }
        final = await asyncio.to_thread(self.ingest_graph.invoke, init)
        return {
            "doc_id": final.get("doc_id"),
            "chunk_count": final.get("chunk_count", 0),
            "kg_triples": len(final.get("kg_entities", [])),
        }

    async def query(
        self,
        question: str,
        session_id: str,
        user_id: str,
        doc_id: str | None = None,
        top_k: int = 5,
    ) -> dict:
        init: AgentState = {
            "question": question,
            "session_id": session_id,
            "user_id": user_id,
            "doc_id": doc_id,
            "top_k": top_k,
            "retry_count": 0,
            "latency_ms": {},
        }
        config = {"configurable": {"thread_id": session_id}}
        final = await asyncio.to_thread(self.query_graph.invoke, init, config)
        return {
            "answer": final.get("answer", ""),
            "sources": final.get("sources", []),
            "query_type": final.get("query_type"),
            "faithfulness_score": final.get("faithfulness_score"),
            "latency_ms": final.get("latency_ms", {}),
        }

    async def stream_query(
        self,
        question: str,
        session_id: str,
        user_id: str,
        doc_id: str | None = None,
    ) -> AsyncIterator[dict]:
        """
        Yield intermediate state updates as the graph executes.
        Used for WebSocket streaming — client sees retrieval progress,
        then judge result, then final answer.
        """
        init: AgentState = {
            "question": question,
            "session_id": session_id,
            "user_id": user_id,
            "doc_id": doc_id,
            "top_k": settings.top_k_final,
            "retry_count": 0,
            "latency_ms": {},
        }
        config = {"configurable": {"thread_id": session_id}}

        for event in self.query_graph.stream(init, config, stream_mode="updates"):
            for node_name, node_output in event.items():
                yield {"node": node_name, "data": _safe_serialize(node_output)}


def _safe_serialize(state: dict) -> dict:
    """Strip non-serializable fields (bytes, embeddings) before JSON streaming."""
    skip = {"raw_bytes", "embeddings"}
    return {k: v for k, v in state.items() if k not in skip and v is not None}


orchestrator = ARAPOrchestrator()
