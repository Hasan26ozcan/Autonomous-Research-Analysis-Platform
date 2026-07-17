"""Services package exports for the autonomous research analysis platform.

This package exposes the retrieval, chunking, enrichment, embedding,
and vector storage services that power the ingestion and query pipelines.
"""

from __future__ import annotations

from app.services.bm25_index import BM25Index
from app.services.chunker import PDFChunker, chunk_document
from app.services.contextual_enricher import ContextualEnricher, enrich_chunks
from app.services.embedder import Embedder, embed_chunks, embedder
from app.services.vector_store import VectorStore, store_chunks

__all__ = [
    "BM25Index",
    "ContextualEnricher",
    "Embedder",
    "PDFChunker",
    "VectorStore",
    "chunk_document",
    "embed_chunks",
    "embedder",
    "enrich_chunks",
    "store_chunks",
]
