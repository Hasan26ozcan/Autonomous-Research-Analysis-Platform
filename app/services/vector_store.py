"""
app/services/vector_store.py
=============================
Qdrant HNSW vector database operations: upsert, search, filter, delete.

What is Qdrant?
  Qdrant is a purpose-built vector database. Unlike storing vectors in
  PostgreSQL (slow, no ANN index) or using FAISS (in-memory, no filtering),
  Qdrant provides:
    - HNSW index: approximate nearest neighbor search in O(log n)
    - Metadata filtering: "find the 5 nearest vectors WHERE doc_id='abc'"
    - REST + gRPC API: queryable from any language
    - Persistent storage: data survives restarts
    - Production features: collection aliases, snapshots, cluster mode

HNSW — Hierarchical Navigable Small World:
  A graph-based ANN algorithm that builds a multi-layer proximity graph
  during indexing. Search traverses from coarse top layers to fine bottom
  layers, finding approximate nearest neighbors in O(log n) time.
  Trade-off: slightly approximate (misses ~1-5% of true nearest neighbors)
  but orders of magnitude faster than exact search for large collections.
  Reference: Malkov & Yashunin (2018), arXiv:1603.09320

Collection design:
  - One collection for all documents (arap_docs)
  - Vectors: 384-dim float32 (MiniLM) or 2560-dim (Qwen3-Embedding)
  - Distance: Cosine (equivalent to dot product for unit-norm vectors)
  - Payload (metadata): text, page, filename, doc_id, chunk_index
  - Payload indexes on doc_id enable fast per-document filtering

Point ID strategy:
  Qdrant requires each point to have a unique UUID or unsigned integer ID.
  We generate UUID4 for each chunk because:
    - UUIDs are collision-free across distributed workers
    - They don't reveal information about insertion order
    - Re-uploading the same PDF can use upsert safely (same doc_id filter + delete)
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING

from app.core.config import settings

if TYPE_CHECKING:
    from app.core.state import AgentState

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages all Qdrant operations for ARAP.

    Lifecycle:
      1. At startup: ensure_collection() creates the collection if it doesn't exist
      2. During /ingest: upsert() stores chunk embeddings with metadata
      3. During /query: search() finds semantically similar chunks
      4. During re-ingestion: delete_by_doc() removes old chunks before re-upsert

    Connection is lazy — Qdrant client is created on first use, not at import time.
    This prevents startup failures if Qdrant isn't running yet.
    """

    def __init__(
        self,
        host: str = settings.qdrant_host,
        port: int = settings.qdrant_port,
        collection: str = settings.qdrant_collection,
        embedding_dim: int = settings.embedding_dim,
    ):
        self.host = host
        self.port = port
        self.collection = collection
        self.embedding_dim = embedding_dim
        self._client = None  # lazy-initialized

    @property
    def client(self):
        """
        Lazy Qdrant client initialization.

        On first access:
          1. Creates the QdrantClient (connects via HTTP to localhost:6333)
          2. Calls ensure_collection() to create the collection if needed
          3. Caches the client in self._client for subsequent calls

        Why lazy?
          - Fast imports at startup (Qdrant may not be running yet)
          - docker compose starts Qdrant and the API simultaneously;
            lazy init gives Qdrant time to become ready
        """
        if self._client is None:
            from qdrant_client import QdrantClient

            logger.info(
                "Connecting to Qdrant at %s:%d", self.host, self.port
            )
            self._client = QdrantClient(
                host=self.host,
                port=self.port,
                timeout=10,         # seconds before connection attempt fails
                prefer_grpc=False,  # use REST (simpler, slightly slower than gRPC)
            )
            self._ensure_collection()

        return self._client

    # ── Collection management ──────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """
        Create the Qdrant collection if it doesn't already exist.

        Collection settings:
          - vectors size: must match embedding model output dimension
          - distance: Cosine (preferred for normalized embeddings)
          - on_disk_payload: True — stores metadata on disk, not RAM
            (important for large deployments with many chunks)

        Payload indexes:
          - doc_id indexed for fast per-document filtering
            ("find chunks WHERE doc_id='abc'" becomes O(log n), not O(n))

        This method is idempotent — safe to call multiple times.
        """
        from qdrant_client.models import (
            Distance,
            VectorParams,
            PayloadSchemaType,
        )

        existing = [c.name for c in self._client.get_collections().collections]

        if self.collection not in existing:
            logger.info("Creating Qdrant collection '%s'...", self.collection)
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                    on_disk=False,  # keep vectors in RAM for speed
                ),
                on_disk_payload=True,  # store metadata on disk (saves RAM)
            )
            # Create index on doc_id payload field for fast filtering
            self._client.create_payload_index(
                collection_name=self.collection,
                field_name="doc_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info(
                "Created collection '%s' (dim=%d, distance=Cosine)",
                self.collection, self.embedding_dim,
            )
        else:
            logger.debug("Collection '%s' already exists.", self.collection)

    # ── Write operations ───────────────────────────────────────────────────────

    def upsert(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        """
        Store chunk embeddings and metadata in Qdrant.

        Uses Qdrant's upsert operation (insert or update):
          - If a point with the same ID already exists, it's overwritten
          - If it's new, it's inserted
          - Points are processed in batches of 100 to avoid large payloads

        Args:
            chunks:     List of chunk dicts (from chunker or contextual enricher)
            embeddings: Parallel list of float vectors (from embedder)
                        Must be same length as chunks.

        Returns:
            Number of points upserted.

        Raises:
            ValueError: If chunks and embeddings have different lengths.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length"
            )

        if not chunks:
            return 0

        from qdrant_client.models import PointStruct

        # Build PointStruct list — each point has an ID, vector, and payload
        points = [
            PointStruct(
                id=str(uuid.uuid4()),   # unique UUID per chunk
                vector=embedding,
                payload={
                    "text":        chunk["text"],
                    "page":        chunk.get("page", 0),
                    "filename":    chunk.get("filename", ""),
                    "doc_id":      chunk.get("doc_id", ""),
                    "chunk_index": chunk.get("chunk_index", i),
                    "word_count":  chunk.get("word_count", 0),
                },
            )
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        # Batch upsert in groups of 100 to avoid HTTP payload size limits
        batch_size = 100
        total_upserted = 0

        t0 = time.perf_counter()
        for batch_start in range(0, len(points), batch_size):
            batch = points[batch_start : batch_start + batch_size]
            self.client.upsert(
                collection_name=self.collection,
                points=batch,
                wait=True,  # wait for indexing before returning (consistency)
            )
            total_upserted += len(batch)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Upserted %d points to Qdrant in %.0fms",
            total_upserted, elapsed,
        )
        return total_upserted

    def delete_by_doc(self, doc_id: str) -> int:
        """
        Delete all chunks belonging to a specific document.

        Used when re-ingesting a document: delete old chunks first,
        then upsert new ones. Without this, you'd accumulate duplicate chunks.

        Args:
            doc_id: The document identifier (SHA-256 hash of PDF bytes).

        Returns:
            Number of points deleted.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        logger.info("Deleting all chunks for doc_id='%s'...", doc_id)
        result = self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id),
                    )
                ]
            ),
            wait=True,
        )
        logger.info("Deleted chunks for doc_id='%s'", doc_id)
        return 0  # Qdrant doesn't return count from delete; log is sufficient

    # ── Read operations ────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        doc_id: str | None = None,
        top_k: int = settings.top_k_retrieval,
    ) -> list[dict]:
        """
        Dense vector similarity search using HNSW index.

        Finds the top_k chunks most semantically similar to the query vector.
        If doc_id is provided, restricts results to that document.

        Args:
            query_vector: Embedded query text (float list, same dim as chunks)
            doc_id:       Optional document scope filter
            top_k:        Number of results to return

        Returns:
            List of result dicts sorted by cosine similarity (highest first):
              {"text": str, "page": int, "filename": str, "doc_id": str,
               "chunk_index": int, "score": float, "source": "dense"}

        Performance:
            HNSW search is O(log n) — typically < 10ms for collections
            up to 1 million vectors on commodity hardware.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Build payload filter if doc_id is provided
        query_filter = None
        if doc_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id),
                    )
                ]
            )

        t0 = time.perf_counter()
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,     # include metadata (text, page, etc.)
            with_vectors=False,    # don't return the raw vectors (large, not needed)
            score_threshold=0.0,   # no minimum score filter — reranker handles quality
        )
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("Qdrant search: %d results in %.1fms", len(results), elapsed)

        return [
            {
                "text":        r.payload["text"],
                "page":        r.payload.get("page", 0),
                "filename":    r.payload.get("filename", ""),
                "doc_id":      r.payload.get("doc_id", ""),
                "chunk_index": r.payload.get("chunk_index", 0),
                "score":       float(r.score),
                "source":      "dense",
            }
            for r in results
        ]

    def get_collection_info(self) -> dict:
        """
        Return collection statistics for the /health endpoint.

        Returns:
            {"points_count": int, "indexed_vectors": int, "status": str}
        """
        info = self.client.get_collection(self.collection)
        return {
            "points_count":    info.points_count,
            "vectors_count":   info.vectors_count,
            "status":          str(info.status),
        }


# ── LangGraph node function ────────────────────────────────────────────────────

# Module-level singleton
vector_store = VectorStore()


def store_chunks(state: "AgentState") -> dict:
    """
    LangGraph node function for the ingest pipeline.

    Reads chunks and embeddings from state, upserts them into Qdrant.
    This node runs AFTER contextual enrichment (the chunks already have
    their context-prepended text) and AFTER embedding.

    Reads from state:
        chunks     (list[dict]):        enriched chunks with metadata
        embeddings (list[list[float]]): vectors corresponding to each chunk

    Writes to state (partial update):
        (none — upsert is a side effect, state unchanged)

    Side effects:
        Writes to Qdrant. The chunk_count in state was set by chunk_document;
        we don't change it here.
    """
    chunks: list[dict] = state.get("chunks", [])
    embeddings: list[list[float]] = state.get("embeddings", [])

    if not chunks or not embeddings:
        logger.warning("store_chunks: no chunks or embeddings in state")
        return {}

    vector_store.upsert(chunks=chunks, embeddings=embeddings)

    # Return empty dict — no state fields to update.
    # The side effect (Qdrant write) has occurred.
    return {}
