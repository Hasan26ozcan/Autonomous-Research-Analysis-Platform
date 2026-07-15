import os
import tempfile
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from app.services.chunker import chunk_pdf
from app.services.contextual_enricher import enrich_chunk
from app.services.embedder import embedder
from app.services.vector_store import upsert_vectors
from app.services.bm25_index import bm25_index
from app.agents.graph_agent import extract_triples_from_chunks, write_triples_to_neo4j
from app.services.postgres_store import postgres_store
from app.core.logging import logger


def run_ingest_pipeline(file_content: bytes, filename: str, user_id: str = "default") -> Dict[str, Any]:
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
      8. Save document metadata to PostgreSQL

    Returns:
        {
            "status": "success" or "error",
            "doc_id": str,
            "chunks": int,
            "kg_triples": int
        }
    """
    temp_path = None
    try:
        # 1. Write to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_content)
            temp_path = tmp.name

        # 2. Chunking
        chunks = chunk_pdf(temp_path)
        if not chunks:
            logger.warning(f"No chunks extracted from {filename} – document may be empty or unreadable.")
            return {
                "status": "error",
                "doc_id": "unknown",
                "chunks": 0,
                "kg_triples": 0,
                "error": "No text extracted from PDF"
            }

        doc_id = chunks[0].get("doc_id", "unknown")
        logger.info(f"Chunked {filename}: {len(chunks)} chunks (doc_id={doc_id})")

        # 3. Contextual Enrichment (per‑chunk LLM call to add context)
        enriched_chunks = []
        for idx, chunk in enumerate(chunks):
            enriched_text = enrich_chunk(
                chunk["text"],
                chunk.get("page", 1),
                doc_id,
                idx
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
        points = [
            {
                "id": f"{doc_id}_{i}",
                "vector": embeddings[i],
                "payload": {
                    "doc_id": doc_id,
                    "text": texts[i],
                    "page": enriched_chunks[i].get("page", 0),
                    "user_id": user_id
                }
            }
            for i in range(len(enriched_chunks))
        ]
        upsert_vectors(points)
        logger.info(f"Upserted {len(points)} points to Qdrant.")

        # 6. BM25 Index (keyword search)
        bm25_index.index_chunks(doc_id, texts)
        logger.info(f"BM25 indexed {len(texts)} chunks.")

        # 7. Parallel KG Extraction (5 workers)
        def extract_single(chunk_text: str) -> List[Dict[str, Any]]:
            # Returns list of triples for this chunk
            return extract_triples_from_chunks([chunk_text], doc_id)

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(extract_single, texts))

        all_triples = [triple for sublist in results for triple in sublist]
        write_triples_to_neo4j(all_triples, doc_id)
        logger.info(
            f"KG extraction complete: {len(all_triples)} triples from "
            f"{len(chunks)} chunks, written to Neo4j."
        )

        # 8. Save metadata to PostgreSQL
        postgres_store.save_document_metadata(doc_id, filename, user_id, len(chunks))

        return {
            "status": "success",
            "doc_id": doc_id,
            "chunks": len(chunks),
            "kg_triples": len(all_triples)
        }

    except Exception as e:
        logger.error(f"Ingest pipeline failed: {str(e)}", exc_info=True)
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