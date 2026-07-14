"""
app/api/main.py
================
FastAPI application — all HTTP endpoints and WebSocket streaming.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENDPOINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  GET  /health
    Returns reachability status for Qdrant, Neo4j, Redis.
    Used by Docker health checks and monitoring dashboards.

  POST /ingest
    Upload a PDF → triggers full ingest pipeline (Phase 2-3-6) **asynchronously**.
    Returns task_id immediately. Use /ingest/status/{task_id} to track progress.
    Max file size: 50 MB. Only .pdf extension accepted.

  GET  /ingest/status/{task_id}
    Query the status of an ongoing ingest task.
    Returns PENDING, STARTED, SUCCESS, or FAILURE with details.

  POST /query
    Synchronous question answering via adaptive RAG pipeline (Phase 4-7).
    Returns answer, sources, query_type, faithfulness_score, latency_ms.
    Caller blocks until the full pipeline completes.

  WS   /ws/{session_id}
    WebSocket streaming endpoint. Client sends question JSON, server
    streams one update per LangGraph node, then sends a final "done" message.
    session_id becomes the LangGraph thread_id for conversation memory.

  GET  /docs
    OpenAPI documentation (FastAPI built-in, auto-generated from schemas).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUEST / RESPONSE SCHEMAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  IngestResponse      — task_id, status, message
  IngestStatusResponse — task_id, status, result (if done), error (if failed)
  QueryRequest        — question, session_id, user_id, doc_id, top_k
  QueryResponse       — answer, sources, query_type, faithfulness_score,
                        session_id, latency_ms
  SourceItem          — index, text, page, filename, doc_id, chunk_index,
                        rerank_score
  HealthResponse      — api, qdrant, neo4j, redis

All schemas use Pydantic v2 (FastAPI 0.115 default).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WEBSOCKET PROTOCOL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Client → Server:
    {"question": "...", "user_id": "...", "doc_id": "..."}

  Server → Client (one per LangGraph node):
    {"type": "update", "node": "router",   "data": {"query_type": "single", ...}}
    {"type": "update", "node": "retrieve", "data": {"retrieved_chunks": [...], ...}}
    {"type": "update", "node": "generate", "data": {"draft_answer": "...", ...}}
    {"type": "update", "node": "judge",    "data": {"faithfulness_score": 0.91, ...}}

  Server → Client (final message):
    {"type": "done", "answer": "...", "sources": [...], "faithfulness_score": 0.91}

  Server → Client (on error):
    {"type": "error", "message": "..."}

  The session stays open for multiple questions — client can send
  another question after receiving "done" on the same WebSocket connection.
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from celery.result import AsyncResult

from app.core.config import settings
from app.core.orchestrator import orchestrator
from app.services.tasks import ingest_document_task
from app.core.celery_app import celery_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LIFESPAN — startup / shutdown hooks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager — replaces deprecated @app.on_event.

    Startup:
      - Warm up the embedding model (downloads ~90MB on first run).
        Without warmup, the FIRST /ingest request would wait 3-5 seconds
        for the model to load. Warmup moves this cost to API startup.
      - Trigger lazy compilation of both LangGraph graphs.
        Compilation validates the graph structure and wires all edges.
        Catching errors here (misconfigured edges, missing nodes) is
        better than discovering them on the first real request.

    Shutdown:
      Currently a no-op — connections clean up themselves.
      Future: graceful Celery worker drain, Neo4j driver close.
    """
    logger.info("ARAP starting up...")

    # Warm up the embedding model
    try:
        from app.services.embedder import embedder
        embedder.warmup()
    except Exception as e:
        logger.warning("Embedding model warmup failed (non-fatal): %s", e)

    # Trigger LangGraph compilation (catches wiring errors early)
    try:
        _ = orchestrator.ingest_graph
        _ = orchestrator.query_graph
        logger.info("Both LangGraph graphs compiled successfully.")
    except Exception as e:
        logger.error("LangGraph compilation failed — API may not work: %s", e)

    logger.info("ARAP ready.")
    yield
    logger.info("ARAP shutting down.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FASTAPI APP INSTANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app = FastAPI(
    title="Adaptive Research & Analysis Platform (ARAP)",
    description=(
        "Production-grade multi-agent RAG system.\n\n"
        "**Stack:** LangGraph · GPT-4o · Qdrant · Neo4j · Mem0 · "
        "RAGAS · LangSmith · Redis · PostgreSQL\n\n"
        "**Features:** Adaptive routing (4 strategies) · HyDE query rewriting · "
        "Hybrid BM25+dense search · Cross-encoder reranking · "
        "Contextual retrieval · NLI faithfulness judging · "
        "Long-term memory · WebSocket streaming"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins in development. Restrict in production via env config.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PYDANTIC SCHEMAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class IngestResponse(BaseModel):
    task_id: str
    status: str = "queued"
    message: str = "Document ingestion started. Use /ingest/status/{task_id} to track progress."


class IngestStatusResponse(BaseModel):
    task_id: str
    status: str  # PENDING, STARTED, SUCCESS, FAILURE
    result: dict | None = None
    error: str | None = None


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The user's question in any language.",
    )
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description=(
            "Unique conversation identifier. Reuse across requests for "
            "multi-turn conversation memory. Auto-generated if omitted."
        ),
    )
    user_id: str = Field(
        default="anonymous",
        description=(
            "Stable user identifier for long-term Mem0 memory. "
            "Use 'anonymous' for unauthenticated users."
        ),
    )
    doc_id: str | None = Field(
        default=None,
        description=(
            "Scope retrieval to a specific document (doc_id from /ingest). "
            "Omit to search across all ingested documents."
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve and send to the generator.",
    )


class SourceItem(BaseModel):
    """
    One source citation returned with the answer.
    index matches the [Source N] reference in the answer text.
    """
    index: int
    text: str          # truncated chunk text for display (≤ 300 chars)
    page: int | None
    filename: str
    doc_id: str
    chunk_index: int
    rerank_score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    query_type: str | None = Field(
        description="Routing decision: direct | single | multi_hop | graph"
    )
    faithfulness_score: float | None = Field(
        description="NLI entailment score (0.0–1.0). None for direct answers."
    )
    session_id: str
    latency_ms: dict[str, float]


class HealthResponse(BaseModel):
    api: str = "ok"
    qdrant: str
    neo4j: str
    redis: str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """
    Health check — returns reachability status for all infrastructure.

    Used by:
      - Docker Compose health checks (determines when API is "healthy")
      - Kubernetes liveness / readiness probes
      - Monitoring dashboards (Grafana, Datadog)

    All checks are non-blocking and have a 2-second timeout each.
    Returns 200 even if components are unreachable (caller reads the fields).
    """
    component_status = await orchestrator.health()
    return HealthResponse(api="ok", **component_status)


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_document(file: UploadFile = File(...), user_id: str = "default"):
    """
    Upload a PDF and process it through the full ingestion pipeline **asynchronously**.

    Pipeline runs in a Celery worker (non-blocking):
      1. chunk_document            — PDF → overlapping text chunks
      2. enrich_chunks             — prepend LLM context to each chunk
      3. embed_chunks              — dense vectors via sentence-transformers
      4. store_chunks              — upsert to Qdrant HNSW index
      5. index_chunks              — add to BM25 in-memory corpus
      6. extract_and_store_node    — entity/relation triples → Neo4j

    Returns immediately with a task_id. Use /ingest/status/{task_id} to check progress.

    Validation:
      - Only .pdf files accepted (extension check)
      - Maximum 50 MB file size
    """
    # Extension validation
    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Received: " + filename,
        )

    # Read the file bytes
    contents = await file.read()

    # Size validation (50 MB cap)
    max_bytes = 50 * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is 50 MB. "
                   f"Received: {len(contents) / 1024 / 1024:.1f} MB.",
        )

    # Send task to Celery worker
    task = ingest_document_task.delay(
        file_content=contents,
        filename=filename,
        user_id=user_id
    )

    logger.info("Ingest task queued: task_id=%s, filename=%s", task.id, filename)

    return IngestResponse(
        task_id=task.id,
        status="queued",
        message="Document ingestion started. Use /ingest/status/{task_id} to track progress."
    )


@app.get("/ingest/status/{task_id}", response_model=IngestStatusResponse, tags=["Documents"])
async def get_ingest_status(task_id: str):
    """
    Get the status of an ingest task.

    Possible statuses:
      - PENDING   : waiting for a worker
      - STARTED   : worker started processing
      - SUCCESS   : completed successfully (result contains doc_id, chunk_count, kg_triples)
      - FAILURE   : failed (error field contains exception message)
    """
    task_result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "status": task_result.status,
    }

    if task_result.status == "SUCCESS":
        response["result"] = task_result.result
    elif task_result.status == "FAILURE":
        response["error"] = str(task_result.info)
    elif task_result.status in ("PENDING", "STARTED"):
        # No additional info yet
        pass

    return IngestStatusResponse(**response)


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(req: QueryRequest):
    """
    Answer a question using the adaptive RAG pipeline (synchronous).

    The pipeline runs fully before returning a response:
      Phase 4 — router classifies question → fetches Mem0 memories
      Phase 5 — hybrid retrieval (HyDE + BM25 + dense + RRF + rerank)
              OR Phase 6 — Neo4j graph traversal (for graph queries)
      Phase 7 — GPT-4o generation → NLI faithfulness judge (→ retry?)
      Phase 7 — Mem0 memory store

    For real-time streaming of each step, use the WebSocket endpoint /ws/{session_id}.

    The session_id enables multi-turn conversations — reuse the same
    session_id across multiple /query calls to build on prior context
    (LangGraph Redis checkpointer persists state per session_id).
    """
    try:
        result = await orchestrator.query(
            question=req.question,
            session_id=req.session_id,
            user_id=req.user_id,
            doc_id=req.doc_id,
            top_k=req.top_k,
        )

        logger.info(
            "Query completed: type=%s faithfulness=%.2f session=%s",
            result.get("query_type"), result.get("faithfulness_score") or 0.0, req.session_id,
        )

        return QueryResponse(
            answer=result.get("answer", ""),
            sources=[SourceItem(**s) for s in (result.get("sources") or [])],
            query_type=result.get("query_type"),
            faithfulness_score=result.get("faithfulness_score"),
            session_id=req.session_id,
            latency_ms=result.get("latency_ms") or {},
        )

    except Exception as e:
        logger.exception("Query failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)[:200]}",
        )


@app.websocket("/ws/{session_id}")
async def websocket_query(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time streaming.

    Accepts multiple questions per connection (the loop continues until
    the client disconnects). Each question triggers the full query pipeline
    and streams one event per LangGraph node.

    Protocol:
      Client sends:  {"question": "...", "user_id": "...", "doc_id": "..."}
      Server yields: {"type": "update", "node": "...", "data": {...}}  × N nodes
      Server sends:  {"type": "done", "answer": "...", "sources": [...], ...}

    Error handling:
      Any exception inside the pipeline sends {"type": "error", "message": "..."}
      and the loop continues — the connection stays open for the next question.

    Session continuity:
      session_id in the URL path maps to the LangGraph thread_id, ensuring
      conversation history is maintained across multiple questions in the
      same WebSocket connection AND across reconnections with the same session_id.
    """
    await websocket.accept()
    logger.info("WebSocket connected: session_id=%s", session_id)

    try:
        while True:
            # Wait for the next question from the client
            raw_message = await websocket.receive_text()

            try:
                payload = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type":    "error",
                    "message": "Invalid JSON. Expected: {\"question\": \"...\", ...}",
                })
                continue

            question = payload.get("question", "").strip()
            if not question:
                await websocket.send_json({
                    "type":    "error",
                    "message": "Field 'question' is required and must be non-empty.",
                })
                continue

            user_id = payload.get("user_id", "anonymous")
            doc_id  = payload.get("doc_id")

            # Accumulate final answer/sources/score across all streamed events
            final_answer = ""
            final_sources: list = []
            final_score: float | None = None
            final_type: str | None = None

            try:
                async for event in orchestrator.stream_query(
                    question=question,
                    session_id=session_id,
                    user_id=user_id,
                    doc_id=doc_id,
                ):
                    # Stream each node update to the client
                    await websocket.send_json({"type": "update", **event})

                    # Extract final answer and sources from the last updates
                    data = event.get("data", {})
                    if data.get("answer"):
                        final_answer = data["answer"]
                    if data.get("sources"):
                        final_sources = data["sources"]
                    if data.get("faithfulness_score") is not None:
                        final_score = data["faithfulness_score"]
                    if data.get("query_type"):
                        final_type = data["query_type"]

                # Send the final consolidated response
                await websocket.send_json({
                    "type":              "done",
                    "answer":            final_answer,
                    "sources":           final_sources,
                    "faithfulness_score": final_score,
                    "query_type":        final_type,
                })
                logger.info(
                    "WebSocket query done: session=%s type=%s faithfulness=%.2f",
                    session_id, final_type, final_score or 0.0,
                )

            except Exception as e:
                logger.exception("WebSocket pipeline error: session=%s %s", session_id, e)
                await websocket.send_json({
                    "type":    "error",
                    "message": f"Pipeline error: {str(e)[:200]}",
                })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session_id=%s", session_id)
    except Exception as e:
        logger.exception("WebSocket fatal error: session=%s %s", session_id, e)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UVICORN ENTRY POINT (for local dev without docker)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,    # auto-reload on file changes (dev only)
        log_level="info",
    )