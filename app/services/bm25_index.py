"""
app/services/bm25_index.py
===========================
BM25 keyword search index — the sparse component of hybrid retrieval.

Why do we need BM25 if we have dense vector search?
  Dense vector search finds semantically similar content but can miss:
    - Exact term matches (product codes, paper IDs, person names)
    - Rare or technical vocabulary not well-represented in the embedding space
    - Acronyms and abbreviations ("RAG", "HNSW", "NLI")
    - Numbers and dates ("Q3 2024", "97%", "Section 4.2")

  BM25 catches these exact matches that dense embeddings can miss.
  Combining both (hybrid search with RRF fusion) consistently outperforms
  either method alone by 15-25% on document Q&A benchmarks.

What is BM25?
  BM25 (Best Match 25) is a probabilistic ranking function.
  It scores a document d for query q as:

    score(d, q) = Σ_term [IDF(term) × TF(term, d) × (k1 + 1)]
                          ─────────────────────────────────────
                          [TF(term, d) + k1 × (1 - b + b × |d|/avgdl)]

  Where:
    TF(term, d)  = term frequency in document d
    IDF(term)    = log((N - df + 0.5) / (df + 0.5)) — penalizes common terms
    |d|          = length of document d in words
    avgdl        = average document length in the corpus
    k1 = 1.5     = term frequency saturation parameter
    b  = 0.75    = length normalization parameter

  Compared to TF-IDF, BM25:
    - Saturates TF (adding more occurrences of a term gives diminishing returns)
    - Normalizes by document length (short documents are not penalized)
    - Has been the standard baseline in information retrieval since 2009

  Reference: Robertson & Zaragoza (2009), "The Probabilistic Relevance Framework"

rank_bm25 (BM25Okapi):
  We use the rank_bm25 Python library with the BM25Okapi variant.
  It runs entirely in-memory — no extra infrastructure needed.
  Trade-off: all chunks must fit in RAM. For millions of chunks, use
  Elasticsearch or OpenSearch which implement BM25 on disk.

Persistence strategy:
  The BM25 corpus (list of tokenized texts) is rebuilt from Qdrant on
  startup using load_from_qdrant(). This means:
    - No separate persistence needed (Qdrant is the source of truth)
    - Restart-safe: BM25 is rebuilt from whatever is in Qdrant
    - One-time cost at startup: typically <5 seconds for 10,000 chunks
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from app.core.config import settings

if TYPE_CHECKING:
    from app.core.state import AgentState

logger = logging.getLogger(__name__)


class BM25Index:
    """
    In-memory BM25 index for keyword retrieval.

    The index consists of:
      _corpus_tokens: list of tokenized documents (list of word lists)
      _corpus_meta:   parallel list of metadata dicts (text, page, doc_id, etc.)
      _bm25:          BM25Okapi instance built from _corpus_tokens

    Thread safety:
      BM25Okapi.get_scores() is read-only and thread-safe.
      add_chunks() modifies _corpus and rebuilds the BM25 object — use a lock
      if multiple workers share one BM25Index instance.
      In ARAP, ingestion runs in Celery workers (separate processes), so
      thread safety is not a concern for the default deployment.
    """

    def __init__(self):
        self._corpus_tokens: list[list[str]] = []  # tokenized texts
        self._corpus_meta: list[dict] = []         # chunk metadata (parallel)
        self._bm25 = None                          # BM25Okapi instance (lazy)
        self._dirty = False                        # True when corpus changed but BM25 not rebuilt

    # ── Index management ───────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[dict]) -> None:
        """
        Add chunks to the BM25 corpus.

        Called during document ingestion, after chunking.
        Tokenizes each chunk's text and appends to the corpus.
        Marks the index as dirty (BM25 will be rebuilt on next search).

        Args:
            chunks: List of chunk dicts. Must have "text" key.
                    Other keys (page, doc_id, filename) stored as metadata.
        """
        for chunk in chunks:
            tokens = self._tokenize(chunk["text"])
            self._corpus_tokens.append(tokens)
            self._corpus_meta.append({
                "text":        chunk["text"],
                "page":        chunk.get("page", 0),
                "filename":    chunk.get("filename", ""),
                "doc_id":      chunk.get("doc_id", ""),
                "chunk_index": chunk.get("chunk_index", 0),
            })
        self._dirty = True
        logger.debug("BM25: added %d chunks. Total corpus: %d", len(chunks), len(self._corpus_tokens))

    def remove_by_doc(self, doc_id: str) -> int:
        """
        Remove all chunks belonging to a specific document from the index.

        Used when re-ingesting a document to prevent duplicate entries.
        After removal, the BM25 index is rebuilt from the remaining corpus.

        Args:
            doc_id: Document identifier to remove.

        Returns:
            Number of chunks removed.
        """
        original_len = len(self._corpus_tokens)

        # Keep only chunks NOT belonging to this doc_id
        paired = [
            (tokens, meta)
            for tokens, meta in zip(self._corpus_tokens, self._corpus_meta)
            if meta.get("doc_id") != doc_id
        ]

        if paired:
            self._corpus_tokens, self._corpus_meta = zip(*paired)
            self._corpus_tokens = list(self._corpus_tokens)
            self._corpus_meta = list(self._corpus_meta)
        else:
            self._corpus_tokens = []
            self._corpus_meta = []

        removed = original_len - len(self._corpus_tokens)
        self._dirty = True
        logger.info("BM25: removed %d chunks for doc_id='%s'", removed, doc_id)
        return removed

    def load_from_qdrant(self, qdrant_client, collection: str) -> None:
        """
        Rebuild the BM25 corpus from all vectors stored in Qdrant.

        Called at startup to synchronize BM25 with the persisted Qdrant data.
        Qdrant is the source of truth — BM25 is an in-memory cache.

        Strategy:
          - Scroll through ALL points in Qdrant (no limit on total)
          - Extract text and metadata from each point's payload
          - Build BM25 corpus from extracted texts

        Performance:
          ~1-3 seconds per 10,000 chunks (network + tokenization).
          For very large collections (>100K chunks), consider:
            - Storing BM25 corpus as a pickle file and loading from disk
            - Using Elasticsearch/OpenSearch for disk-backed BM25

        Args:
            qdrant_client: QdrantClient instance (from VectorStore)
            collection:    Qdrant collection name
        """
        logger.info("Building BM25 index from Qdrant collection '%s'...", collection)
        t0 = time.perf_counter()

        self._corpus_tokens = []
        self._corpus_meta = []

        # Scroll through all points in batches of 1000
        # Qdrant scroll returns (records, next_offset) — loop until next_offset is None
        offset = None
        total = 0

        while True:
            records, next_offset = qdrant_client.scroll(
                collection_name=collection,
                offset=offset,
                limit=1000,
                with_payload=True,
                with_vectors=False,  # don't need vectors for BM25
            )

            for record in records:
                payload = record.payload or {}
                text = payload.get("text", "")
                if text:
                    self._corpus_tokens.append(self._tokenize(text))
                    self._corpus_meta.append({
                        "text":        text,
                        "page":        payload.get("page", 0),
                        "filename":    payload.get("filename", ""),
                        "doc_id":      payload.get("doc_id", ""),
                        "chunk_index": payload.get("chunk_index", 0),
                    })
                    total += 1

            if next_offset is None:
                break
            offset = next_offset

        self._dirty = True
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "BM25 index built: %d chunks in %.0fms",
            total, elapsed,
        )

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        doc_id: str | None = None,
        top_k: int = settings.top_k_retrieval,
    ) -> list[dict]:
        """
        BM25 keyword search over the in-memory corpus.

        Steps:
          1. Tokenize the query the same way as the corpus
          2. Build BM25Okapi if dirty (corpus changed since last build)
          3. Compute BM25 scores for all documents
          4. Filter by doc_id if provided
          5. Rank by score, return top_k results

        Args:
            query:  Raw user question (or HyDE rewrite)
            doc_id: Optional document scope filter
            top_k:  Maximum results to return

        Returns:
            List of result dicts sorted by BM25 score (highest first):
              {"text": str, "page": int, "filename": str, "doc_id": str,
               "score": float (0.0-1.0, normalized), "source": "bm25"}
            Returns [] if corpus is empty or no results have score > 0.

        Note on score normalization:
          BM25 raw scores are unbounded positive floats.
          We normalize to [0, 1] by dividing by the max score in the result set.
          This makes BM25 scores comparable to dense cosine scores in RRF.
        """
        if not self._corpus_tokens:
            logger.debug("BM25 search called on empty corpus — returning []")
            return []

        # Rebuild BM25Okapi if corpus was modified since last build
        if self._dirty or self._bm25 is None:
            self._rebuild()

        query_tokens = self._tokenize(query)
        if not query_tokens:
            logger.debug("BM25: empty query tokens — returning []")
            return []

        # Compute scores for all documents in the corpus
        t0 = time.perf_counter()
        all_scores = self._bm25.get_scores(query_tokens)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("BM25 scored %d docs in %.1fms", len(all_scores), elapsed)

        # Pair scores with metadata, apply doc_id filter
        scored: list[tuple[float, dict]] = []
        for score, meta in zip(all_scores, self._corpus_meta):
            if score <= 0.0:
                continue  # skip documents with zero relevance
            if doc_id and meta.get("doc_id") != doc_id:
                continue  # skip documents from other docs
            scored.append((float(score), meta))

        if not scored:
            return []

        # Sort by score descending, take top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        # Normalize scores to [0, 1] for RRF compatibility
        max_score = top[0][0]
        results = [
            {
                **meta,
                "score":  score / (max_score + 1e-9),  # avoid division by zero
                "source": "bm25",
            }
            for score, meta in top
        ]

        return results

    # ── Private helpers ────────────────────────────────────────────────────────

    def _rebuild(self) -> None:
        """
        Rebuild the BM25Okapi model from the current corpus.

        BM25Okapi precomputes:
          - Inverse document frequency (IDF) for every unique term
          - Document length normalization factors
        This makes get_scores() fast (no precomputation at query time).

        Called automatically when _dirty=True.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise RuntimeError("rank_bm25 not installed. Run: pip install rank-bm25")

        t0 = time.perf_counter()
        self._bm25 = BM25Okapi(self._corpus_tokens)
        self._dirty = False
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(
            "BM25Okapi rebuilt: %d documents in %.0fms",
            len(self._corpus_tokens), elapsed,
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Tokenize text for BM25 indexing and querying.

        Strategy: lowercase + split on whitespace.
        Simple tokenization is intentional:
          - Consistent with how BM25 is typically deployed
          - Avoids stemming artifacts (e.g. "running" → "run" is lossy)
          - Multilingual-safe (no English-specific stemmer)

        For better recall, consider adding:
          - Stop word removal (may hurt precision for specific queries)
          - Stemming with nltk.PorterStemmer (English only)
          - Character n-grams for typo tolerance

        Args:
            text: Raw text string

        Returns:
            List of lowercase word tokens
        """
        return text.lower().split()

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of documents in the corpus."""
        return len(self._corpus_tokens)

    @property
    def is_empty(self) -> bool:
        """True if no documents have been indexed."""
        return len(self._corpus_tokens) == 0


# ── LangGraph node function ────────────────────────────────────────────────────

# Module-level singleton — shared across requests in the same process.
bm25_index = BM25Index()


def index_chunks(state: "AgentState") -> dict:
    """
    LangGraph node function for the ingest pipeline.

    Adds the document's chunks to the in-memory BM25 corpus.
    Runs AFTER store_chunks (chunks are already in Qdrant at this point).

    Reads from state:
        chunks (list[dict]): enriched chunks from the ingest pipeline

    Writes to state (partial update):
        (none — BM25 index update is a side effect)

    Side effects:
        Updates bm25_index.corpus and marks it dirty.
        The BM25 model will be rebuilt on the next search() call.
    """
    chunks: list[dict] = state.get("chunks", [])

    if not chunks:
        logger.warning("index_chunks: no chunks in state")
        return {}

    bm25_index.add_chunks(chunks)
    logger.info("BM25: indexed %d chunks. Total corpus size: %d", len(chunks), bm25_index.size)

    return {}
