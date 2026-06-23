"""
ARAP — FastAPI Application
===========================
Endpoints:

  POST /ingest            Upload a PDF; triggers async ingestion pipeline
  POST /query             Synchronous query (JSON response)
  WS   /ws/{session_id}  WebSocket streaming — real-time node-by-node updates
  GET  /eval              Run RAGAS evaluation suite
  GET  /health            Health check + component status
  GET  /docs              Auto-generated OpenAPI docs (FastAPI built-in)
"""
from __future__ import annotations
import uuid
import logging
import json
import asyncio

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.core.orchestrator import orchestrator
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Adaptive Research & Analysis Platform (ARAP)",
    description=(
        "Production-grade multi-agent RAG system with adaptive routing, "
        "knowledge graph, Mem0 long-term memory, and faithfulness judging. "
        "Built with LangGraph · Qdrant · Neo4j · Redis · RAGAS · LangSmith."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(default="anonymous")
    doc_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    query_type: str | None
    faithfulness_score: float | None
    session_id: str
    latency_ms: dict


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    kg_triples: int
    message: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check with component reachability."""
    status = {"api": "ok"}
    # Qdrant
    try:
        from qdrant_client import QdrantClient
        c = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=2)
        c.get_collections()
        status["qdrant"] = "ok"
    except Exception:
        status["qdrant"] = "unreachable"
    # Neo4j
    try:
        from neo4j import GraphDatabase
        d = GraphDatabase.driver(settings.neo4j_uri,
                                 auth=(settings.neo4j_user, settings.neo4j_password))
        d.verify_connectivity()
        status["neo4j"] = "ok"
    except Exception:
        status["neo4j"] = "unreachable"
    # Redis
    try:
        import redis
        r = redis.from_url(settings.redis_url, socket_connect_timeout=2)
        r.ping()
        status["redis"] = "ok"
    except Exception:
        status["redis"] = "unreachable"

    ok = all(v == "ok" for v in status.values())
    return {"status": "ok" if ok else "degraded", "components": status}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload a PDF document. Triggers the full ingestion pipeline:
      chunk → embed → contextual enrichment → Qdrant store → BM25 index → Neo4j KG
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:  # 50 MB cap
        raise HTTPException(status_code=413, detail="File too large (max 50 MB).")

    try:
        result = await orchestrator.ingest(contents, filename=file.filename)
        return IngestResponse(
            doc_id=result["doc_id"],
            filename=file.filename,
            chunk_count=result["chunk_count"],
            kg_triples=result.get("kg_triples", 0),
            message="Document ingested successfully.",
        )
    except Exception as e:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Synchronous query endpoint.
    The adaptive router selects direct / single / multi_hop / graph retrieval.
    Returns answer with source citations and faithfulness score.
    """
    try:
        result = await orchestrator.query(
            question=req.question,
            session_id=req.session_id,
            user_id=req.user_id,
            doc_id=req.doc_id,
            top_k=req.top_k,
        )
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            query_type=result.get("query_type"),
            faithfulness_score=result.get("faithfulness_score"),
            session_id=req.session_id,
            latency_ms=result.get("latency_ms", {}),
        )
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{session_id}")
async def websocket_query(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time streaming.

    Client sends: {"question": "...", "user_id": "...", "doc_id": "..."}
    Server streams: {"node": "<node_name>", "data": {...}} per LangGraph step
    Final message:  {"type": "done", "answer": "...", "sources": [...]}
    """
    await websocket.accept()
    logger.info("WS connection: session_id=%s", session_id)

    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)
            question = payload.get("question", "").strip()
            if not question:
                await websocket.send_json({"type": "error", "message": "Empty question."})
                continue

            final_answer = ""
            final_sources: list = []

            async for event in orchestrator.stream_query(
                question=question,
                session_id=session_id,
                user_id=payload.get("user_id", "anonymous"),
                doc_id=payload.get("doc_id"),
            ):
                await websocket.send_json({"type": "update", **event})
                if "answer" in event.get("data", {}):
                    final_answer = event["data"]["answer"]
                if "sources" in event.get("data", {}):
                    final_sources = event["data"]["sources"]

            await websocket.send_json({
                "type": "done",
                "answer": final_answer,
                "sources": final_sources,
            })

    except WebSocketDisconnect:
        logger.info("WS disconnected: session_id=%s", session_id)
    except Exception as e:
        logger.exception("WS error: session_id=%s", session_id)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


@app.get("/eval")
async def run_evaluation():
    """
    Run the full RAGAS evaluation suite.
    Returns faithfulness, answer_relevancy, context_precision, context_recall.
    Evaluation uses the query_history table in PostgreSQL as the test set.
    """
    try:
        from evaluation.ragas_eval import run_ragas
        scores = await run_ragas(orchestrator)
        return {"status": "ok", "scores": scores}
    except Exception as e:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
