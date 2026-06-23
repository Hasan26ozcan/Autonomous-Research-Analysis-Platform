"""
Retrieval Agent
===============
Implements 2026 production retrieval stack:

  1. Query rewriting — HyDE (Hypothetical Document Embeddings) and step-back
     prompting improve recall for ambiguous queries.

  2. Hybrid search — BM25 (keyword) + Qdrant dense vector search,
     merged with Reciprocal Rank Fusion (RRF). Dense weight 0.7, BM25 0.3.

  3. Contextual retrieval — each chunk is stored with injected surrounding
     context (Anthropic 2024 technique: 15-25% recall improvement).

  4. Cross-encoder re-ranking — a local cross-encoder scores each chunk
     against the query for precise semantic relevance. Removes BM25 noise.

References:
  - HyDE: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
    (Gao et al., 2022) — arXiv:2212.10496
  - Contextual Retrieval: Anthropic blog, October 2024
  - Cross-encoder re-ranking: "Passage Re-ranking with BERT" (Nogueira & Cho, 2019)
  - RRF: "Reciprocal Rank Fusion outperforms Condorcet" (Cormack et al., 2009)
"""
from __future__ import annotations
import time
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder

from app.core.state import AgentState
from app.core.config import settings
from app.services.vector_store import VectorStore
from app.services.bm25_index import BM25Index

logger = logging.getLogger(__name__)


HYDE_SYSTEM = """You are a document retrieval assistant.
Given a question, write a short hypothetical passage (3-5 sentences) that would
PERFECTLY answer this question. Do not hedge — write as if you found the answer.
This passage will be used as a dense retrieval query (HyDE technique)."""

STEPBACK_SYSTEM = """Rewrite the user's specific question as a broader, more general question
that would surface the background knowledge needed to answer it.
Return ONLY the rewritten question, nothing else."""


class RetrievalAgent:
    """
    Handles single-hop and multi-hop retrieval for the query pipeline.
    Multi-hop: iteratively decomposes sub-questions and retrieves for each.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.router_model,
            api_key=settings.openai_api_key,
            temperature=0.0,
        )
        self.vector_store = VectorStore()
        self.bm25 = BM25Index()
        self._reranker: Optional[CrossEncoder] = None

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
            )
        return self._reranker

    # ── Main node ──────────────────────────────────────────────────────────────

    def retrieve(self, state: AgentState) -> AgentState:
        """LangGraph node: single-hop retrieval with HyDE + hybrid + rerank."""
        t0 = time.perf_counter()
        question = state["question"]
        top_k = state.get("top_k", settings.top_k_final)

        rewritten = self._hyde_rewrite(question)
        chunks = self._hybrid_search(rewritten, state.get("doc_id"), top_k * 2)
        ranked = self._rerank(question, chunks, top_k)

        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "rewritten_query": rewritten,
            "retrieved_chunks": ranked,
            "retrieval_score": self._avg_score(ranked),
            "latency_ms": {**(state.get("latency_ms") or {}), "retrieval": elapsed},
        }

    def retrieve_multi_hop(self, state: AgentState) -> AgentState:
        """
        LangGraph node: multi-hop retrieval.
        Decomposes the question, retrieves for each sub-question, deduplicates.
        """
        t0 = time.perf_counter()
        question = state["question"]
        top_k = state.get("top_k", settings.top_k_final)

        sub_questions = self._decompose(question)
        logger.info("Multi-hop: %d sub-questions from '%s'", len(sub_questions), question)

        all_chunks: dict[str, dict] = {}
        for sq in sub_questions:
            rewritten = self._hyde_rewrite(sq)
            chunks = self._hybrid_search(rewritten, state.get("doc_id"), top_k)
            for c in chunks:
                key = c["text"][:80]
                if key not in all_chunks or c["score"] > all_chunks[key]["score"]:
                    all_chunks[key] = c

        deduped = sorted(all_chunks.values(), key=lambda x: x["score"], reverse=True)
        ranked = self._rerank(question, deduped, top_k)

        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "sub_questions": sub_questions,
            "retrieved_chunks": ranked,
            "retrieval_score": self._avg_score(ranked),
            "latency_ms": {**(state.get("latency_ms") or {}), "retrieval_multi": elapsed},
        }

    # ── Query rewriting ────────────────────────────────────────────────────────

    def _hyde_rewrite(self, question: str) -> str:
        """
        HyDE: generate a hypothetical answer, embed it instead of the raw question.
        Reduces vocabulary mismatch between query and document space.
        """
        try:
            resp = self.llm.invoke([
                SystemMessage(content=HYDE_SYSTEM),
                HumanMessage(content=question),
            ])
            return resp.content.strip()
        except Exception as e:
            logger.warning("HyDE rewrite failed, using raw question: %s", e)
            return question

    def _decompose(self, question: str) -> list[str]:
        """Break a complex question into atomic sub-questions."""
        system = (
            "Decompose the following complex question into 2-4 simpler, "
            "atomic sub-questions that together answer the original. "
            "Return one sub-question per line. No numbering or bullets."
        )
        resp = self.llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=question),
        ])
        lines = [l.strip() for l in resp.content.strip().split("\n") if l.strip()]
        return lines if lines else [question]

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def _hybrid_search(
        self,
        query: str,
        doc_id: str | None,
        top_k: int,
    ) -> list[dict]:
        """BM25 + dense cosine, merged with RRF."""
        from app.services.embedder import embedder
        query_vec = embedder.embed_single(query)

        dense = self.vector_store.search(query_vec, doc_id=doc_id, top_k=top_k * 2)
        bm25 = self.bm25.search(query, doc_id=doc_id, top_k=top_k * 2)
        return self._rrf_merge(dense, bm25, top_k)

    def _rrf_merge(
        self,
        dense: list[dict],
        bm25: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion.
        score(d) = Σ_list [ weight × 1 / (k + rank(d)) ]
        k=60 is the standard constant from the original RRF paper.
        """
        k = settings.rrf_k
        scores: dict[str, float] = {}
        meta: dict[str, dict] = {}

        def _add(results: list[dict], weight: float):
            for rank, item in enumerate(results, start=1):
                key = item["text"][:100]
                scores[key] = scores.get(key, 0.0) + weight / (k + rank)
                meta.setdefault(key, item)

        _add(dense, settings.dense_weight)
        _add(bm25, settings.bm25_weight)

        ranked = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
        return [{**meta[k], "rrf_score": round(scores[k], 6)} for k in ranked]

    # ── Re-ranking ─────────────────────────────────────────────────────────────

    def _rerank(self, question: str, chunks: list[dict], top_k: int) -> list[dict]:
        """
        Cross-encoder re-ranking: scores each (question, chunk) pair for relevance.
        More accurate than vector similarity alone — eliminates BM25 false positives.
        """
        if not chunks:
            return []
        pairs = [(question, c["text"]) for c in chunks]
        scores = self.reranker.predict(pairs)
        scored = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )
        result = []
        for score, chunk in scored[:top_k]:
            result.append({**chunk, "rerank_score": float(score)})
        return result

    @staticmethod
    def _avg_score(chunks: list[dict]) -> float:
        if not chunks:
            return 0.0
        return sum(c.get("rerank_score", c.get("rrf_score", 0)) for c in chunks) / len(chunks)


retrieval_agent = RetrievalAgent()
