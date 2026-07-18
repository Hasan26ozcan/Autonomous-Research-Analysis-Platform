import os
import tempfile
from typing import Any

from app.agents.graph_agent import kg_agent
from app.core.config import settings
from app.core.logging import logger
from app.services.bm25_index import bm25_index
from app.services.chunker import chunk_pdf, pdf_page_count
from app.services.contextual_enricher import enrich_chunk
from app.services.embedder import embedder
from app.services.log_store import log_worker
from app.services.postgres_store import (
    record_chunk_metadata,
    record_document,
    update_document_status,
)
from app.services.redis_cache import pipeline_state_clear, pipeline_state_set
from app.services.vector_store import vector_store


def run_ingest_pipeline(
    file_content: bytes, filename: str
) -> dict[str, Any]:
    """
    Full synchronous ingest pipeline – runs inside Celery worker.

    Steps:
      1. Write bytes to temporary PDF file
      2. Chunk the PDF (overlapping sliding window)
      3. Contextual enrichment (LLM adds 2‑3 sentence context to each chunk)
      4. Embed all chunks (batch)
      5. Upsert vectors to Qdrant
      6. Index chunks in BM25 (keyword search)
      7. Parallel KG extraction (5 workers) → write triples to Neo4j
      8. Save document metadata to PostgreSQL (Phase 1: status + chunks + pages)

    Returns:
        {
            "status": "success" or "error",
            "doc_id": str,
            "chunks": int,
            "kg_triples": int
        }
    """
    temp_path = None
    doc_id = "unknown"
    chunks: list = []
    try:
        # 1. Write to temporary file
        temp_path = _write_temp_pdf(file_content)

        # 2. Chunking
        chunks = chunk_pdf(temp_path)
        if not chunks:
            logger.warning(
                f"No chunks extracted from {filename} – document may be empty or unreadable."
            )
            return {
                "status": "error",
                "doc_id": "unknown",
                "chunks": 0,
                "kg_triples": 0,
                "error": "No text extracted from PDF"
            }

        doc_id = chunks[0].get("doc_id", "unknown")
        logger.info(f"Chunked {filename}: {len(chunks)} chunks (doc_id={doc_id})")

        _mark_ingest_start(doc_id, filename)
        total_pages = _prepare_document(temp_path, doc_id)
        enriched_chunks = _enrich_chunks(chunks, doc_id)
        embeddings = embedder.embed([c["text"] for c in enriched_chunks])
        _store_chunks(enriched_chunks, embeddings, doc_id)
        all_triples = _extract_kg(enriched_chunks, doc_id)
        _save_metadata(doc_id, filename, chunks, all_triples, total_pages)
        _finalize_ingest(doc_id, chunks, all_triples)

        return {
            "status": "success",
            "doc_id": doc_id,
            "chunks": len(chunks),
            "kg_triples": len(all_triples)
        }

    except Exception as e:
        return _handle_ingest_failure(e, doc_id, chunks)

    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _write_temp_pdf(file_content: bytes) -> str:
    """Write uploaded bytes to a temporary PDF and return its path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_content)
        return tmp.name


def _mark_ingest_start(doc_id: str, filename: str) -> None:
    """Mark the document as being ingested and log the start (best-effort)."""
    try:
        pipeline_state_set(doc_id)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("pipeline_state_set failed (non-fatal): %s", e)
    try:
        log_worker("ingest", doc_id, "info", f"started ingest of {filename}")
    except Exception:  # pragma: no cover - defensive
        pass


def _prepare_document(temp_path: str, doc_id: str) -> int:
    """Record page count and set document status to 'processing' (best-effort)."""
    try:
        total_pages = pdf_page_count(temp_path)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not read PDF page count (non-fatal): %s", e)
        total_pages = 0
    try:
        update_document_status(doc_id, "processing")
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("update_document_status(processing) failed: %s", e)
    return total_pages


def _enrich_chunks(chunks: list, doc_id: str) -> list:
    """Contextually enrich each chunk (LLM adds context). Returns enriched list."""
    doc_anchor = " ".join(chunks[0]["text"].split()[:400]) if chunks else ""
    enriched_chunks = []
    for idx, chunk in enumerate(chunks):
        enriched_text = enrich_chunk(
            chunk["text"],
            chunk.get("page", 1),
            doc_id,
            idx,
            doc_anchor=doc_anchor,
        )
        enriched_chunks.append({
            **chunk,
            "text": enriched_text,          # enriched text
            "original_text": chunk["text"]  # keep original for cross‑encoder
        })

    success_count = sum(1 for c in enriched_chunks if c["text"] != c["original_text"])
    logger.info(f"Contextual enrichment: {success_count}/{len(chunks)} chunks enriched.")
    return enriched_chunks


def _store_chunks(enriched_chunks: list, embeddings: list, doc_id: str) -> None:
    """Upsert vectors to Qdrant, index in BM25, and signal API reload."""
    vector_store.upsert(enriched_chunks, embeddings)
    logger.info(f"Upserted {len(enriched_chunks)} points to Qdrant.")

    bm25_index.remove_by_doc(doc_id)
    bm25_index.add_chunks(enriched_chunks)
    logger.info(f"BM25 indexed {len(enriched_chunks)} chunks.")

    _notify_bm25_reload()


def _extract_kg(enriched_chunks: list, doc_id: str) -> list:
    """Run parallel KG extraction and write triples to Neo4j."""
    kg_result = kg_agent.extract_and_store_node(
        {"chunks": enriched_chunks, "doc_id": doc_id}
    )
    all_triples = kg_result.get("kg_entities", [])
    logger.info(
        f"KG extraction complete: {len(all_triples)} triples from "
        f"{len(enriched_chunks)} chunks, written to Neo4j."
    )
    return all_triples


def _save_metadata(
    doc_id: str, filename: str, chunks: list, all_triples: list, total_pages: int
) -> None:
    """Persist chunk + document metadata to PostgreSQL (best-effort)."""
    try:
        record_chunk_metadata(doc_id, chunks)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("PostgreSQL chunk metadata save failed (non-fatal): %s", e)

    try:
        record_document(
            doc_id, filename, len(chunks), len(all_triples),
            status="ready", total_pages=total_pages,
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("PostgreSQL metadata save failed (non-fatal): %s", e)


def _finalize_ingest(doc_id: str, chunks: list, all_triples: list) -> None:
    """Clear the in-flight flag and log completion (best-effort)."""
    try:
        pipeline_state_clear(doc_id)
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        log_worker(
            "ingest", doc_id, "info",
            f"completed: {len(chunks)} chunks, {len(all_triples)} triples",
        )
    except Exception:  # pragma: no cover - defensive
        pass


def _handle_ingest_failure(e: Exception, doc_id: str, chunks: list) -> dict:
    """Mark failure, clear in-flight flag, log, and return a structured error."""
    logger.error(f"Ingest pipeline failed: {str(e)}", exc_info=True)
    try:
        update_document_status(doc_id, "error")
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        pipeline_state_clear(doc_id)
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        log_worker("ingest", doc_id, "error", str(e)[:500])
    except Exception:  # pragma: no cover - defensive
        pass
    return {
        "status": "error",
        "doc_id": doc_id,
        "chunks": len(chunks),
        "kg_triples": 0,
        "error": str(e),
    }


def _notify_bm25_reload() -> None:
    """
    Signal running API processes to rebuild their in-memory BM25 index.

    Ingestion runs in Celery worker processes, but queries are served from the
    API process's own BM25 singleton. Publishing a reload signal on the Redis
    channel lets the API rebuild BM25 from Qdrant (source of truth) so newly
    ingested documents are keyword-searchable without restarting the API.
    Best-effort and non-fatal.
    """
    try:
        import redis as redis_lib
        r = redis_lib.from_url(settings.redis_url)
        r.publish("arap:bm25:reload", "1")
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("BM25 reload notify failed (non-fatal): %s", e)
