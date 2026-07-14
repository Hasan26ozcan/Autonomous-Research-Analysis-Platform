import os
import json
import tempfile
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.services.chunker import chunk_pdf
from app.services.contextual_enricher import enrich_chunk
from app.services.embedder import embedder
from app.services.vector_store import qdrant_client, upsert_vectors
from app.services.bm25_index import bm25_index
from app.agents.graph_agent import extract_triples_from_chunks, write_triples_to_neo4j
from app.services.postgres_store import postgres_store
from app.core.logging import logger

def run_ingest_pipeline(file_content: bytes, filename: str, user_id: str = "default") -> Dict[str, Any]:
    """
    Tüm ingest sürecini senkron olarak çalıştırır (Celery worker içinde çalışır).
    """
    temp_path = None
    try:
        # 1. Geçici dosyaya yaz
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_content)
            temp_path = tmp.name

        # 2. Chunking
        chunks = chunk_pdf(temp_path)
        doc_id = chunks[0].get("doc_id") if chunks else "unknown"
        logger.info(f"Chunked {filename}: {len(chunks)} chunks (doc_id={doc_id})")

        # 3. Contextual Enrichment (Artık hızlı ve boş dönmüyor)
        enriched_chunks = []
        for idx, chunk in enumerate(chunks):
            # Model llama-3.1 olarak enricher içinde ayarlandı, max_tokens=512
            enriched_text = enrich_chunk(chunk["text"], chunk.get("page", 1), doc_id, idx)
            enriched_chunks.append({
                **chunk,
                "text": enriched_text,  # Zenginleştirilmiş metin
                "original_text": chunk["text"]
            })
        
        success_count = sum(1 for c in enriched_chunks if c["text"] != c["original_text"])
        logger.info(f"Contextual enrichment: {success_count}/{len(chunks)} chunks enriched.")

        # 4. Embedding (Batch)
        texts = [c["text"] for c in enriched_chunks]
        embeddings = embedder.embed(texts)  # Batch embed

        # 5. Vektör DB'ye yaz (Qdrant)
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

        # 6. BM25 Index (Anahtar kelime araması için)
        bm25_index.index_chunks(doc_id, texts)
        logger.info(f"BM25 indexed {len(texts)} chunks.")

        # 7. KG Extraction (PARALEL)
        # ThreadPool ile 5'erli paralel çalıştır
        def extract_single(chunk_text):
            # Graph agent içindeki LLM çağrısı artık llama-3.1 ve max_tokens=4096
            return extract_triples_from_chunks([chunk_text], doc_id)
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(extract_single, texts))
        
        all_triples = [triple for sublist in results for triple in sublist]
        write_triples_to_neo4j(all_triples, doc_id)
        logger.info(f"KG extraction complete: {len(all_triples)} triples from {len(chunks)} chunks, written to Neo4j.")

        # 8. PostgreSQL'e kaydet (metadata)
        postgres_store.save_document_metadata(doc_id, filename, user_id, len(chunks))

        return {
            "status": "success",
            "doc_id": doc_id,
            "chunks": len(chunks),
            "kg_triples": len(all_triples)
        }

    except Exception as e:
        logger.error(f"Ingest pipeline failed: {str(e)}", exc_info=True)
        raise e
    finally:
        # Geçici dosyayı temizle
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)