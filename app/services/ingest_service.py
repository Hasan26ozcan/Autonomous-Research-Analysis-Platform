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
    file_content: bytes, filename: str, user_id: str = "default"
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_content)
            temp_path = tmp.name

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

        # Phase 5: mark this document as being ingested (processing:<doc_id>).
        # Lets a second concurrent upload of the same file be detected, and
        # the flag auto-expires via settings.pipeline_state_ttl_seconds.
        try:
            pipeline_state_set(doc_id)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("pipeline_state_set failed (non-fatal): %s", e)

        # Phase 10: worker log — ingest started.
        try:
            log_worker("ingest", doc_id, "info", f"started ingest of {filename}")
        except Exception:  # pragma: no cover - defensive
            pass

        # Phase 1: mark the document as being processed (metadata lifecycle).
        try:
            total_pages = pdf_page_count(temp_path)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Could not read PDF page count (non-fatal): %s", e)
            total_pages = 0
        try:
            update_document_status(doc_id, "processing")
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("update_document_status(processing) failed: %s", e)

        # 3. Contextual Enrichment (per‑chunk LLM call to add context)
        # Build ONE document anchor (first 400 words of the document beginning)
        # and pass it to every chunk so the enricher situates each chunk in the
        # full document — instead of each chunk using its own text as the anchor
        # (which degraded enrichment quality in the Celery ingest path).
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

        # 4. Embedding (batch)
        texts = [c["text"] for c in enriched_chunks]
        embeddings = embedder.embed(texts)   # returns list of lists

        # 5. Upsert to Qdrant
        # vector_store.upsert builds the PointStructs (with deterministic,
        # dedupe-friendly IDs) from (chunks, embeddings) and batches the write,
        # mirroring the LangGraph store_chunks node.
        vector_store.upsert(enriched_chunks, embeddings)
        logger.info(f"Upserted {len(enriched_chunks)} points to Qdrant.")

        # 6. BM25 Index (keyword search)
        # bm25_index exposes add_chunks(chunks) (chunks = list of dicts with
        # text/page/doc_id/...). Remove any pre-existing entries for this doc_id
        # first so re-ingesting the same PDF replaces, never duplicates.
        bm25_index.remove_by_doc(doc_id)
        bm25_index.add_chunks(enriched_chunks)
        logger.info(f"BM25 indexed {len(enriched_chunks)} chunks.")

        # Notify API processes to rebuild their in-memory BM25 index from
        # Qdrant (the source of truth) so this doc becomes keyword-searchable
        # without an API restart. Best-effort, non-fatal.
        _notify_bm25_reload()

        # 7. Parallel KG Extraction (5 workers) → write triples to Neo4j
        # Delegates to the same KnowledgeGraphAgent node the LangGraph ingest
        # graph uses: parallel LLM extraction across chunks + batched Neo4j upsert.
        kg_result = kg_agent.extract_and_store_node(
            {"chunks": enriched_chunks, "doc_id": doc_id}
        )
        all_triples = kg_result.get("kg_entities", [])
        logger.info(
            f"KG extraction complete: {len(all_triples)} triples from "
            f"{len(chunks)} chunks, written to Neo4j."
        )

        # 8. Save metadata to PostgreSQL (Phase 1)
        # record_document / record_chunk_metadata are now synchronous (psycopg2)
        # and best-effort: they catch their own errors and log them. A Postgres
        # outage should never fail an ingest - these audit tables are non-critical.
        try:
            record_chunk_metadata(doc_id, enriched_chunks)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("PostgreSQL chunk metadata save failed (non-fatal): %s", e)

        try:
            record_document(
                doc_id, filename, len(chunks), len(all_triples),
                status="ready", total_pages=total_pages,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("PostgreSQL metadata save failed (non-fatal): %s", e)

        # Phase 5: clear the in-flight ingest flag (success path).
        try:
            pipeline_state_clear(doc_id)
        except Exception:  # pragma: no cover - defensive
            pass

        # Phase 10: worker log — ingest completed.
        try:
            log_worker(
                "ingest", doc_id, "info",
                f"completed: {len(chunks)} chunks, {len(all_triples)} triples",
            )
        except Exception:  # pragma: no cover - defensive
            pass

        return {
            "status": "success",
            "doc_id": doc_id,
            "chunks": len(chunks),
            "kg_triples": len(all_triples)
        }

    except Exception as e:
        logger.error(f"Ingest pipeline failed: {str(e)}", exc_info=True)
        # Phase 1: mark the document as failed so its lifecycle is observable.
        try:
            update_document_status(doc_id, "error")
        except Exception:  # pragma: no cover - defensive
            pass
        # Phase 5: clear the in-flight ingest flag.
        try:
            pipeline_state_clear(doc_id)
        except Exception:  # pragma: no cover - defensive
            pass
        # Phase 10: worker log — ingest failed.
        try:
            log_worker("ingest", doc_id, "error", str(e)[:500])
        except Exception:  # pragma: no cover - defensive
            pass
        # Return a structured error so the caller knows what happened
        return {
            "status": "error",
            "doc_id": doc_id if 'doc_id' in locals() else "unknown",
            "chunks": len(chunks) if 'chunks' in locals() else 0,
            "kg_triples": 0,
            "error": str(e)
        }

    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


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
