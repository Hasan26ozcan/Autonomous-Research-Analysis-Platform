"""
app/agents/retrieval_agent.py
==============================
Retrieval Agent — finds the most relevant document chunks for a given question.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE RETRIEVAL STACK (4 layers, each building on the previous)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Layer 1 — HyDE Query Rewriting
  Problem: raw questions and stored chunk texts live in different
  vocabulary spaces. "What caused the floods?" and "Heavy rainfall
  triggered widespread inundation" both mean the same thing, but their
  dense vectors are far apart because the surface form is so different.

  Solution (HyDE — Hypothetical Document Embeddings):
    Instead of embedding the question, generate a hypothetical answer
    that WOULD appear in the document, then embed that.
    "Heavy precipitation events exceeded river capacity, causing downstream
     flooding across three provinces" → this vector is much closer to the
     actual chunks that describe flood causes.

  Reference: Gao et al. (2022), "Precise Zero-Shot Dense Retrieval
             without Relevance Labels." arXiv:2212.10496

Layer 2 — Hybrid Search (BM25 + dense)
  Neither BM25 nor dense alone is sufficient:
    - BM25 is great for: exact terms, names, codes ("Table 4", "BERT", "2024")
    - Dense is great for: semantic paraphrases, cross-lingual, intent matching
  Running both and merging the results always outperforms either alone.

  Services used (from Phase 2):
    - app.services.vector_store.VectorStore.search() → dense results
    - app.services.bm25_index.BM25Index.search()     → keyword results

Layer 3 — RRF Fusion
  Problem: how to combine two ranked lists with incompatible scores?
    BM25 scores are unbounded floats. Dense scores are cosine similarities.
    You cannot simply average them — they're on different scales.

  Solution (RRF — Reciprocal Rank Fusion):
    Ignore the raw scores entirely. Use only the RANK of each document
    in each list. Apply the formula:

      rrf_score(doc) = Σ_list [ weight × 1 / (k + rank(doc)) ]

    where k=60 prevents top-ranked documents from dominating, and
    weights (0.7 dense, 0.3 BM25) reflect empirical performance.

    Key property: a document appearing in BOTH lists gets scores from
    both — naturally rewarding evidence from multiple retrieval methods.

  Reference: Cormack et al. (2009), "Reciprocal Rank Fusion outperforms
             Condorcet and individual rank learning methods." SIGIR 2009.

Layer 4 — Cross-Encoder Re-ranking
  Problem: RRF picks the best of what we retrieved, but retrieval
  itself is approximate. Some top-10 BM25 hits are keyword matches
  without semantic relevance. We need to filter these out.

  Solution: score each (question, chunk) pair with a cross-encoder.
    Unlike bi-encoders (which embed query and document separately),
    cross-encoders see BOTH texts simultaneously — enabling full attention
    between every token in the question and every token in the chunk.
    This produces much more accurate relevance scores.

  Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Trained on MS MARCO passage ranking (Microsoft, 2016)
    - Fast enough for CPU inference at query time (~80ms for 10 pairs)
    - Standard choice for production reranking in 2024-2026

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TWO NODE FUNCTIONS (for two LangGraph routes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

retrieve()        — single route: one HyDE rewrite → one hybrid search
retrieve_multi()  — multi_hop route: decompose → retrieve per sub-question
                    → deduplicate → cross-encoder re-rank over merged set

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STATE INTEGRATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Both node functions read from AgentState:
    question          (str):        raw user question (never modified)
    doc_id            (str | None): optional per-document scope filter
    top_k             (int):        final chunk count after re-ranking
    latency_ms        (dict):       accumulated per-node timing

Both node functions write to AgentState (partial update):
    retrieved_chunks  (list[dict]): top-k chunks with rrf_score + rerank_score
    retrieval_score   (float):      avg rerank_score across retrieved chunks
    rewritten_query   (str):        HyDE-rewritten query (single) or original
    sub_questions     (list[str]):  decomposed sub-questions (multi_hop only)
    latency_ms        (dict):       previous dict + {"retrieval": <ms>}

retrieved_chunks field contract (matches AgentState comment + generator expectation):
    {
        "text":         str,    # chunk text (may include [Context: ...] prefix)
        "page":         int,    # page number in original PDF
        "filename":     str,    # original PDF filename
        "doc_id":       str,    # SHA-256 based document identifier
        "chunk_index":  int,    # position of chunk in document
        "rrf_score":    float,  # Reciprocal Rank Fusion score (pre-reranking)
        "rerank_score": float,  # cross-encoder score (final relevance measure)
        "source":       str,    # "dense", "bm25", or "both" (for observability)
    }
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.core.config import settings
from app.services.embedder import embedder
from app.services.vector_store import vector_store
from app.services.bm25_index import bm25_index

if TYPE_CHECKING:
    from app.core.state import AgentState

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROMPTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HYDE_SYSTEM_PROMPT = """\
You are a document retrieval assistant implementing the HyDE technique
(Hypothetical Document Embeddings, Gao et al. 2022).

Your task: given a question, write a SHORT passage (3-5 sentences) that
WOULD BE the ideal answer to this question as it might appear in a
technical document, research paper, or report.

Rules:
  - Write as if you FOUND the answer in a document (do not hedge).
  - Use technical, document-like language — not conversational.
  - Do not mention that this is hypothetical.
  - Do not say "According to the document" or similar.
  - Write exactly 3-5 sentences. No more, no less.

This passage will be embedded and used for similarity search.
Return ONLY the passage text, no preamble.\
"""

DECOMPOSE_SYSTEM_PROMPT = """\
You are a question decomposition assistant for a multi-hop RAG system.

Your task: break the user's complex question into 2-4 ATOMIC sub-questions.
Each sub-question must be:
  - Self-contained (answerable without the others)
  - Concrete (asks for one specific fact or comparison)
  - Necessary (each one contributes to answering the original question)

Rules:
  - Return each sub-question on its own line.
  - No numbering, no bullet points, no explanations.
  - If the question is simple (only needs one retrieval), return it unchanged.
  - Maximum 4 sub-questions.

Example:
  Original: "How does the preprocessing in section 3 address the limitations in section 6?"
  Output:
  What preprocessing steps are described in section 3?
  What limitations are identified in section 6?
  How do the preprocessing steps in section 3 relate to those limitations?\
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RETRIEVAL AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RetrievalAgent:
    """
    Implements the full 4-layer retrieval pipeline for ARAP.

    Singleton pattern — shared across all requests in one process.
    Two separate LangGraph node methods:
      retrieve()       → registered as "retrieve" node (single route)
      retrieve_multi() → registered as "retrieve_multi" node (multi_hop route)

    Both methods call the same internal hybrid search + reranking pipeline.
    The difference is in how the query is prepared before hitting that pipeline.

    Service dependencies (imported at module level, lazy-initialized):
      embedder      — app.services.embedder.Embedder
      vector_store  — app.services.vector_store.VectorStore
      bm25_index    — app.services.bm25_index.BM25Index
      cross_encoder — sentence_transformers.CrossEncoder (lazy property)
      llm           — ChatOpenAI (router_model, for HyDE + decomposition)
    """

    def __init__(self):
        # Use router_model (gpt-4o-mini) for HyDE + decomposition.
        # These are structured generation tasks — cheaper model is sufficient.
        # We save llm_model (gpt-4o) for final answer generation in Phase 7.
        self.llm = ChatOpenAI(
            model=settings.router_model,
            api_key=settings.openai_api_key,
            temperature=0.0,    # deterministic HyDE and decomposition
            max_tokens=300,     # 3-5 sentence HyDE passage fits in 300 tokens
        )
        self._cross_encoder: CrossEncoder | None = None  # lazy-loaded

    # ── Lazy cross-encoder ────────────────────────────────────────────────────

    @property
    def cross_encoder(self) -> CrossEncoder:
        """
        Lazy-load the cross-encoder model.

        Model: cross-encoder/ms-marco-MiniLM-L-6-v2
          Trained on MS MARCO (~8.8M passage-query pairs).
          Produces a relevance score for each (query, passage) pair.
          Runs on CPU: ~8ms per pair, ~80ms for 10 pairs.
          Much more accurate than bi-encoder cosine similarity alone.

        Why lazy?
          CrossEncoder downloads ~90MB on first use. Lazy init prevents
          download at import time, keeping startup fast.
        """
        if self._cross_encoder is None:
            logger.info("Loading cross-encoder model (first call, may download)...")
            t0 = time.perf_counter()
            self._cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,   # truncate at 512 tokens (model limit)
            )
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("Cross-encoder loaded in %.0fms", elapsed)
        return self._cross_encoder

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LangGraph Node: retrieve() — single route
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def retrieve(self, state: "AgentState") -> dict:
        """
        LangGraph node: single-hop retrieval.

        Called when router classified query_type = "single".
        One HyDE rewrite → one hybrid search → cross-encoder rerank.

        Pipeline:
          1. HyDE: rewrite question → hypothetical answer passage
          2. Embed: vectorize the hypothetical passage
          3. Hybrid: BM25 search + Qdrant dense search
          4. RRF: merge both result lists by rank
          5. Rerank: cross-encoder scores all RRF results
          6. Return top_k after reranking

        Reads from AgentState:
            question   (str):       raw user question
            doc_id     (str|None):  optional document scope filter
            top_k      (int):       final number of chunks to return
            latency_ms (dict):      accumulated timing

        Writes to AgentState:
            retrieved_chunks (list[dict]): top-k chunks (see field contract above)
            retrieval_score  (float):      avg rerank_score
            rewritten_query  (str):        HyDE passage used for embedding
            latency_ms       (dict):       + {"retrieval": <ms>}
        """
        t0 = time.perf_counter()

        question: str = state.get("question", "")
        doc_id: str | None = state.get("doc_id")
        top_k: int = state.get("top_k") or settings.top_k_final

        if not question:
            logger.error("retrieve() called with empty question")
            return self._empty_update(state, t0, node_key="retrieval")

        # Layer 1: HyDE rewrite
        hyde_passage = self._hyde_rewrite(question)

        # Layers 2-4: embed → hybrid search → RRF → rerank
        chunks = self._full_retrieval_pipeline(
            query_text=hyde_passage,
            original_question=question,
            doc_id=doc_id,
            top_k=top_k,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "retrieve(): %d chunks in %.0fms | question='%s...'",
            len(chunks), elapsed_ms, question[:50],
        )

        return {
            "retrieved_chunks": chunks,
            "retrieval_score":  _avg_rerank_score(chunks),
            "rewritten_query":  hyde_passage,
            "latency_ms": {
                **(state.get("latency_ms") or {}),
                "retrieval": round(elapsed_ms, 2),
            },
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LangGraph Node: retrieve_multi() — multi_hop route
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def retrieve_multi(self, state: "AgentState") -> dict:
        """
        LangGraph node: multi-hop retrieval with query decomposition.

        Called when router classified query_type = "multi_hop".
        Decomposes the complex question into atomic sub-questions,
        retrieves for each sub-question independently, deduplicates,
        then applies cross-encoder reranking over the merged set.

        Why retrieve per sub-question?
          A question like "How does method A in section 3 compare with
          method B in section 6?" requires chunks from BOTH sections.
          A single retrieval call biased toward one section's vocabulary
          would miss the other. Decomposing into:
            - "What is method A in section 3?"
            - "What is method B in section 6?"
            - "How do methods A and B differ?"
          and retrieving for each guarantees coverage of both sections.

        Deduplication strategy:
          Two chunks with the same (doc_id, chunk_index) are identical.
          We keep the one with the higher rrf_score when deduplicating.
          This prevents the generator from seeing the same text twice,
          which would waste context window tokens.

        Reads from AgentState:
            question   (str):       complex multi-hop question
            doc_id     (str|None):  optional document scope filter
            top_k      (int):       final chunk count after reranking
            latency_ms (dict):      accumulated timing

        Writes to AgentState:
            retrieved_chunks (list[dict]): deduplicated top-k chunks
            retrieval_score  (float):      avg rerank_score
            sub_questions    (list[str]):  the decomposed sub-questions
            rewritten_query  (str):        original question (not HyDE here)
            latency_ms       (dict):       + {"retrieval_multi": <ms>}
        """
        t0 = time.perf_counter()

        question: str = state.get("question", "")
        doc_id: str | None = state.get("doc_id")
        top_k: int = state.get("top_k") or settings.top_k_final

        if not question:
            logger.error("retrieve_multi() called with empty question")
            return self._empty_update(state, t0, node_key="retrieval_multi")

        # Step 1: Decompose into sub-questions
        sub_questions = self._decompose_question(question)
        logger.info(
            "retrieve_multi(): %d sub-questions from '%s...'",
            len(sub_questions), question[:50],
        )

        # Step 2: Retrieve for each sub-question, accumulate all candidates
        # key = (doc_id, chunk_index) → deduplication key
        # value = best chunk dict seen so far for this key
        candidate_pool: dict[tuple, dict] = {}

        for sub_q in sub_questions:
            hyde_passage = self._hyde_rewrite(sub_q)
            sub_chunks = self._full_retrieval_pipeline(
                query_text=hyde_passage,
                original_question=sub_q,
                doc_id=doc_id,
                # Retrieve more per sub-question: after dedup we need top_k total
                top_k=max(top_k, settings.top_k_retrieval),
                skip_final_rerank=True,   # rerank once at the end, not per sub-q
            )
            for chunk in sub_chunks:
                dedup_key = (chunk.get("doc_id", ""), chunk.get("chunk_index", 0))
                existing = candidate_pool.get(dedup_key)
                if existing is None or chunk.get("rrf_score", 0) > existing.get("rrf_score", 0):
                    candidate_pool[dedup_key] = chunk

        # Step 3: Cross-encoder rerank over the deduplicated merged pool
        all_candidates = list(candidate_pool.values())

        if not all_candidates:
            logger.warning("retrieve_multi(): no candidates after deduplication")
            return self._empty_update(state, t0, node_key="retrieval_multi")

        # Rerank merged pool against the ORIGINAL question (not sub-questions)
        # This ensures the final chunks are relevant to what the user actually asked
        final_chunks = self._rerank(
            question=question,
            chunks=all_candidates,
            top_k=top_k,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "retrieve_multi(): %d candidates → %d final chunks in %.0fms",
            len(all_candidates), len(final_chunks), elapsed_ms,
        )

        return {
            "retrieved_chunks": final_chunks,
            "retrieval_score":  _avg_rerank_score(final_chunks),
            "sub_questions":    sub_questions,
            "rewritten_query":  question,   # original question, not HyDE passage
            "latency_ms": {
                **(state.get("latency_ms") or {}),
                "retrieval_multi": round(elapsed_ms, 2),
            },
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Private: Layer 1 — HyDE Query Rewriting
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(Exception),
        reraise=False,   # on failure, return None → caller falls back to original question
    )
    def _hyde_rewrite(self, question: str) -> str:
        """
        Generate a hypothetical answer passage for dense embedding.

        The key insight from HyDE:
          Questions and answers have different statistical distributions
          in the embedding space. "What caused the flood?" lives in
          question-space. "Heavy rainfall overwhelmed drainage systems..."
          lives in answer-space — much closer to where document chunks are.

        By generating a hypothetical answer and embedding THAT, we close
        the vocabulary gap between the question and the document.

        On any LLM failure (timeout, rate limit, API error):
          Returns the original question unchanged. The pipeline continues
          with a less optimal embedding — retrieval quality degrades
          slightly but does not fail.

        Args:
            question: Raw user question string.

        Returns:
            3-5 sentence hypothetical answer passage, or original question on error.
        """
        try:
            response = self.llm.invoke([
                SystemMessage(content=HYDE_SYSTEM_PROMPT),
                HumanMessage(content=f"Question: {question}"),
            ])
            passage = response.content.strip()
            logger.debug("HyDE passage (first 80 chars): '%s...'", passage[:80])
            return passage
        except Exception as e:
            logger.warning(
                "HyDE rewrite failed (using original question): %s", str(e)[:80]
            )
            return question   # graceful fallback — retrieval still works

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Private: Multi-hop Question Decomposition
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _decompose_question(self, question: str) -> list[str]:
        """
        Break a complex question into atomic, self-contained sub-questions.

        Each sub-question will be retrieved for independently. Together
        they must cover all information needed to answer the original question.

        Parsing strategy:
          LLM returns one sub-question per line (instructed by prompt).
          We split on newlines and strip each line.
          Lines with fewer than 5 words are likely artifacts — filtered out.
          If parsing yields nothing, return [question] as a safe fallback.

        Args:
            question: The complex multi-hop question to decompose.

        Returns:
            List of 1-4 sub-question strings.
            Always returns at least [question] (never empty list).
        """
        try:
            response = self.llm.invoke([
                SystemMessage(content=DECOMPOSE_SYSTEM_PROMPT),
                HumanMessage(content=f"Question to decompose: {question}"),
            ])
            lines = response.content.strip().split("\n")
            sub_questions = [
                line.strip()
                for line in lines
                if len(line.strip().split()) >= 5   # filter noise lines
            ]
            if sub_questions:
                logger.info(
                    "Decomposed '%s...' → %d sub-questions",
                    question[:50], len(sub_questions),
                )
                return sub_questions[:4]    # cap at 4 to control latency
        except Exception as e:
            logger.warning("Decomposition failed (using original): %s", str(e)[:80])

        return [question]   # safe fallback: treat as single-hop

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Private: Layers 2-3 — Hybrid Search + RRF Fusion
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _hybrid_search(
        self,
        query_text: str,
        doc_id: str | None,
        top_k: int,
    ) -> list[dict]:
        """
        Run BM25 + dense vector search and merge results with RRF.

        Args:
            query_text: The text to search for (HyDE passage or sub-question).
            doc_id:     Optional document scope filter (None = all documents).
            top_k:      Number of results to return after RRF.

        Returns:
            List of chunk dicts with rrf_score field added.
            Sorted by rrf_score descending.

        The two searches use different services from Phase 2:
          - vector_store.search(): Qdrant HNSW dense search
          - bm25_index.search():   in-memory BM25Okapi keyword search

        Both return the same field names:
          {text, page, filename, doc_id, chunk_index, score, source}
        This common schema is what makes RRF merger clean — no field mapping needed.
        """
        # Candidates = top_k * 2 before RRF so we have enough to merge from
        candidate_count = top_k * 2

        # Dense search: embed the query text, search Qdrant HNSW index
        query_vector = embedder.embed_single(query_text)
        dense_results = vector_store.search(
            query_vector=query_vector,
            doc_id=doc_id,
            top_k=candidate_count,
        )

        # BM25 search: tokenize the query text, score all corpus documents
        bm25_results = bm25_index.search(
            query=query_text,
            doc_id=doc_id,
            top_k=candidate_count,
        )

        logger.debug(
            "Hybrid search: %d dense + %d BM25 candidates",
            len(dense_results), len(bm25_results),
        )

        # RRF merge: combine by rank, not by raw score
        return self._rrf_merge(dense_results, bm25_results, top_k=top_k)

    def _rrf_merge(
        self,
        dense: list[dict],
        bm25: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion: merge two ranked lists into one.

        Formula:
            rrf_score(doc) = Σ_list [ weight_list × 1 / (k + rank(doc)) ]

        Where:
            k = settings.rrf_k (60, from original RRF paper)
            weight for dense results = settings.dense_weight (0.7)
            weight for BM25 results  = settings.bm25_weight  (0.3)
            rank = 1-based position in each result list

        Deduplication:
            The same chunk may appear in both dense and BM25 results.
            We use the first 100 characters of chunk text as the dedup key.
            When a chunk appears in both lists, its rrf_score is the SUM
            of its scores from both lists — naturally rewarding chunks
            found by multiple retrieval methods.

        Args:
            dense:  Dense search results (sorted by cosine similarity).
            bm25:   BM25 results (sorted by BM25 score).
            top_k:  Number of results to return after merging.

        Returns:
            List of chunk dicts with rrf_score field, sorted descending.
        """
        k = settings.rrf_k
        accumulated_scores: dict[str, float] = {}
        chunk_registry: dict[str, dict] = {}

        def _accumulate(results: list[dict], weight: float) -> None:
            for rank, chunk in enumerate(results, start=1):
                # Dedup key: first 100 chars of text.
                # Avoids full text comparison (slow) while being collision-resistant
                # for typical chunk lengths (512 words ≈ 3000 chars).
                dedup_key = chunk["text"][:100]
                accumulated_scores[dedup_key] = (
                    accumulated_scores.get(dedup_key, 0.0)
                    + weight * (1.0 / (k + rank))
                )
                # Register chunk metadata (first occurrence wins)
                chunk_registry.setdefault(dedup_key, chunk)

        _accumulate(dense, settings.dense_weight)
        _accumulate(bm25,  settings.bm25_weight)

        # Sort by accumulated RRF score descending, take top_k
        sorted_keys = sorted(
            accumulated_scores,
            key=lambda key: accumulated_scores[key],
            reverse=True,
        )[:top_k]

        return [
            {
                **chunk_registry[key],
                "rrf_score": round(accumulated_scores[key], 6),
                # Normalize source label: if chunk appeared in both lists,
                # it was registered from whichever list processed it first.
                # We don't change "source" here — reranker adds "rerank_score".
            }
            for key in sorted_keys
        ]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Private: Layer 4 — Cross-Encoder Re-ranking
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _rerank(
        self,
        question: str,
        chunks: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        Cross-encoder re-ranking: score each (question, chunk) pair.

        The cross-encoder sees both texts simultaneously with full attention
        between every token pair. This is fundamentally more accurate than
        the bi-encoder approach used in dense retrieval, but too slow to
        run over the entire corpus (hence it runs only on the top-k RRF results).

        Scoring:
          The ms-marco cross-encoder outputs an unbounded float (logit).
          Higher values = more relevant. We do NOT normalize these —
          the relative ranking is what matters, not the absolute value.
          The generator receives rerank_score as a quality signal but
          doesn't use it mathematically.

        Performance:
          10 pairs × ~8ms per pair ≈ 80ms on CPU.
          Acceptable for query-time latency.

        Args:
            question: The ORIGINAL user question (not HyDE passage).
                      We rerank against what the user actually asked, not
                      what we searched for — this is the key distinction.
            chunks:   List of chunks from RRF (already has rrf_score).
            top_k:    Number of chunks to return after reranking.

        Returns:
            List of top_k chunks sorted by rerank_score descending.
            Each chunk has rerank_score field added.
        """
        if not chunks:
            return []

        # Build (question, chunk_text) pairs for the cross-encoder.
        # We use original_text if available (strips the [Context: ...] prefix)
        # so the cross-encoder sees only the actual content, not our metadata.
        pairs = [
            (question, chunk.get("original_text") or chunk["text"])
            for chunk in chunks
        ]

        t0 = time.perf_counter()
        scores: list[float] = self.cross_encoder.predict(pairs).tolist()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Cross-encoder scored %d pairs in %.0fms", len(pairs), elapsed_ms
        )

        # Attach rerank_score to each chunk and sort descending
        scored_chunks = [
            {**chunk, "rerank_score": float(score)}
            for chunk, score in zip(chunks, scores)
        ]
        scored_chunks.sort(key=lambda c: c["rerank_score"], reverse=True)

        return scored_chunks[:top_k]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Private: Full Pipeline Combinator
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _full_retrieval_pipeline(
        self,
        query_text: str,
        original_question: str,
        doc_id: str | None,
        top_k: int,
        skip_final_rerank: bool = False,
    ) -> list[dict]:
        """
        Layers 2-4 combined: hybrid search → RRF → (optional) rerank.

        Used by both retrieve() and retrieve_multi(). The skip_final_rerank
        flag is used by retrieve_multi() to skip per-sub-question reranking —
        it reranks once at the end over the merged pool instead.

        Args:
            query_text:         Text to embed and search (HyDE passage or sub-question).
            original_question:  Original user question (used for cross-encoder scoring).
            doc_id:             Optional document scope filter.
            top_k:              Final chunk count.
            skip_final_rerank:  If True, return RRF results without cross-encoder.

        Returns:
            List of chunk dicts with rrf_score (always) and rerank_score (if reranked).
        """
        # Hybrid: BM25 + dense → RRF
        rrf_results = self._hybrid_search(
            query_text=query_text,
            doc_id=doc_id,
            top_k=settings.top_k_retrieval,   # get top_k_retrieval before reranking
        )

        if skip_final_rerank or not rrf_results:
            return rrf_results

        # Cross-encoder rerank
        return self._rerank(
            question=original_question,
            chunks=rrf_results,
            top_k=top_k,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Private: Empty State Update Helper
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _empty_update(
        self, state: "AgentState", t0: float, node_key: str
    ) -> dict:
        """
        Return a safe empty state update when retrieval has nothing to return.

        The generator will receive an empty retrieved_chunks list and must
        respond with "I could not find relevant information in the documents."
        This is preferable to an error — the user gets a response, not a crash.
        """
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "retrieved_chunks": [],
            "retrieval_score":  0.0,
            "rewritten_query":  state.get("question", ""),
            "latency_ms": {
                **(state.get("latency_ms") or {}),
                node_key: round(elapsed_ms, 2),
            },
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _avg_rerank_score(chunks: list[dict]) -> float:
    """
    Average rerank_score across all retrieved chunks.

    Falls back to avg rrf_score if rerank_score is absent
    (e.g. when skip_final_rerank=True was used).

    Used as retrieval_score in AgentState — a single float that
    summarizes retrieval quality for monitoring and logging.
    """
    if not chunks:
        return 0.0
    scores = [
        c.get("rerank_score") if c.get("rerank_score") is not None
        else c.get("rrf_score", 0.0)
        for c in chunks
    ]
    return round(sum(scores) / len(scores), 4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Module-Level Singleton
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# One agent per process — LLM client and cross-encoder are shared.
# Registered in orchestrator.py as:
#   g.add_node("retrieve",       retrieval_agent.retrieve)
#   g.add_node("retrieve_multi", retrieval_agent.retrieve_multi)
retrieval_agent = RetrievalAgent()
