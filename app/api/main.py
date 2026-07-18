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

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from celery.result import AsyncResult
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.orchestrator import orchestrator
from app.services import analytics as analytics_service
from app.services.log_store import log_api
from app.services.tasks import ingest_document_task
from evaluation.ragas_eval import run_ragas_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LIFESPAN — startup / shutdown hooks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _bm25_reload_listener() -> None:
    """
    Background daemon thread: rebuild the API's in-memory BM25 keyword index
    from Qdrant whenever a Celery ingest worker publishes a reload signal.

    Ingestion runs in a separate Celery worker process, so the API's own BM25
    singleton would otherwise never see newly ingested documents. This listener
    keeps it in sync. Runs in a daemon thread (started in lifespan); any failure
    is caught and logged so the API itself is never affected.
    """
    try:
        import redis as redis_lib
        r = redis_lib.from_url(settings.redis_url)
        pubsub = r.pubsub()
        pubsub.subscribe("arap:bm25:reload")
        for message in pubsub.listen():
            if message.get("type") == "message":
                try:
                    logger.info("BM25 reload signal received — rebuilding from Qdrant")
                    from app.services.bm25_index import bm25_index
                    from app.services.vector_store import vector_store
                    bm25_index.load_from_qdrant(vector_store.client, settings.qdrant_collection)
                    logger.info(
                        "BM25 index reloaded from Qdrant (%d chunks).", bm25_index.size
                    )
                except Exception as e:  # pragma: no cover - depends on Qdrant
                    logger.warning("BM25 reload failed (non-fatal): %s", e)
    except Exception as e:  # pragma: no cover - depends on Redis availability
        logger.warning("BM25 reload listener stopped (non-fatal): %s", e)


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

    # Rebuild the API process's in-memory BM25 keyword index from Qdrant
    # (the source of truth). Ingestion runs in a separate Celery worker
    # process, so the API's own BM25 would otherwise stay empty and hybrid
    # retrieval would silently degrade to dense-only. Best-effort, non-fatal.
    try:
        from app.services.bm25_index import bm25_index
        from app.services.vector_store import vector_store
        bm25_index.load_from_qdrant(vector_store.client, settings.qdrant_collection)
        logger.info("BM25 index loaded from Qdrant (%d chunks).", bm25_index.size)
    except Exception as e:  # pragma: no cover - depends on Qdrant availability
        logger.warning("BM25 initial load from Qdrant failed (non-fatal): %s", e)

    # Start a background listener so the API rebuilds its BM25 index whenever a
    # Celery ingest worker publishes a reload signal after finishing a document.
    # Daemon thread: won't block shutdown; any error is logged, not fatal.
    try:
        import threading
        threading.Thread(target=_bm25_reload_listener, daemon=True).start()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("BM25 reload listener failed to start (non-fatal): %s", e)

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 10 — API access logging middleware
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Every request is recorded to api_log (Postgres) with method, path, status
# and latency. Health/docs/openapi are skipped to avoid log noise. The write
# is best-effort and never blocks or fails the response.

_SKIP_LOG_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


@app.middleware("http")
async def api_log_middleware(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if request.url.path not in _SKIP_LOG_PATHS:
        try:
            # log_api is synchronous (psycopg2); run it off the event loop so a
            # slow/blocked Postgres write never delays the response.
            await asyncio.to_thread(
                log_api,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                latency_ms=round(elapsed_ms, 2),
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("api_log_middleware write failed (non-fatal): %s", e)

    return response


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


class EvalRequest(BaseModel):
    """Phase 9 — run the RAGAS evaluation suite and persist results to Postgres."""
    limit: int = Field(default=20, ge=1, le=200, description="Number of questions to evaluate")
    save: str | None = Field(default=None, description="Optional JSON path to also save the report")
    no_seed: bool = Field(default=False, description="Do not fall back to seeded questions")


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

    # Send task to Celery worker.
    # Fail gracefully if the broker (Redis) is unreachable, so the caller
    # gets a clear 500 instead of an unhandled exception / stack trace.
    try:
        task = ingest_document_task.delay(
            file_content=contents,
            filename=filename,
            user_id=user_id,
        )
    except Exception as exc:
        logger.error("Failed to enqueue ingest task (broker unreachable?): %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Ingestion queue is temporarily unavailable. Please try again later.",
        ) from exc

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
        ) from e


@app.post("/eval", tags=["Evaluation"])
async def eval_endpoint(req: EvalRequest):
    """
    Phase 9 — run the RAGAS evaluation suite and persist results to Postgres.

    Runs each question through the live query pipeline, computes RAGAS metrics
    (faithfulness / answer_relevancy / context_precision / context_recall),
    captures token usage, and writes everything to the evaluation_runs /
    evaluation_scores / retrieval_results tables (scripts/init_db.sql).

    The suite can take minutes (it issues many LLM calls), so it runs in a
    worker thread with its own event loop to keep the API responsive.
    """
    try:
        report = await asyncio.to_thread(
            lambda: asyncio.run(
                run_ragas_evaluation(
                    orchestrator,
                    limit=req.limit,
                    include_seed=not req.no_seed,
                    save_path=req.save,
                )
            )
        )
        return report
    except Exception as e:
        logger.exception("Evaluation failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)[:200]}",
        ) from e


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 11 — Analytics (JSON endpoints + HTML dashboard)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/analytics/summary", tags=["Analytics"])
async def analytics_summary():
    """Headline metrics: totals, avg latency / faithfulness / precision / recall."""
    return await asyncio.to_thread(analytics_service.summary)


@app.get("/analytics/documents", tags=["Analytics"])
async def analytics_documents(limit: int = 10):
    """Top documents by query volume, with average faithfulness."""
    return await asyncio.to_thread(analytics_service.top_documents, limit=limit)


@app.get("/analytics/eval-trend", tags=["Analytics"])
async def analytics_eval_trend(limit: int = 20):
    """Recent evaluation runs with headline metrics."""
    return await asyncio.to_thread(analytics_service.eval_trend, limit=limit)


@app.get("/analytics", response_class=HTMLResponse, tags=["Analytics"])
async def analytics_dashboard():
    """
    Minimal self-contained HTML dashboard (no external libraries).

    Fetches the three JSON analytics endpoints on load and renders them as
    stat cards + tables. Meant as an at-a-glance operational view, not a BI tool.
    """
    return HTMLResponse(content=_ANALYTICS_HTML)


_ANALYTICS_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ARAP Analytics</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 0; padding: 24px; background: #0f1420; color: #e7ecf3; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: #8b97a8; font-size: 13px; margin-bottom: 20px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr));
           gap: 14px; margin-bottom: 28px; }
  .card { background: #171e2e; border: 1px solid #26304a; border-radius: 12px;
          padding: 16px; }
  .card .label { color: #8b97a8; font-size: 12px; text-transform: uppercase;
                 letter-spacing: .04em; }
  .card .value { font-size: 26px; font-weight: 650; margin-top: 6px; }
  h2 { font-size: 15px; margin: 22px 0 10px; }
  table { width: 100%; border-collapse: collapse; background: #171e2e;
          border: 1px solid #26304a; border-radius: 12px; overflow: hidden; }
  th, td { text-align: left; padding: 10px 12px; font-size: 13px;
           border-bottom: 1px solid #26304a; }
  th { color: #8b97a8; font-weight: 600; background: #131a29; }
  tr:last-child td { border-bottom: none; }
  .muted { color: #8b97a8; }
  button { background: #2c6cf0; color: #fff; border: 0; border-radius: 8px;
           padding: 8px 14px; font-size: 13px; cursor: pointer; }
</style>
</head>
<body>
  <h1>ARAP Analytics</h1>
  <div class="sub">Phase 11 dashboard &middot; <button onclick="loadAll()">Refresh</button></div>

  <div class="cards" id="cards"></div>

  <h2>Top documents</h2>
  <table id="docs"><thead><tr>
    <th>Filename</th><th>doc_id</th><th>Queries</th><th>Avg faithfulness</th>
  </tr></thead><tbody></tbody></table>

  <h2>Evaluation runs</h2>
  <table id="evals"><thead><tr>
    <th>Run</th><th>When</th><th>Questions</th><th>Status</th><th>Avg faithfulness</th>
  </tr></thead><tbody></tbody></table>

<script>
function fmt(v) { return (v === null || v === undefined) ? '&mdash;' : v; }

async function loadAll() {
  try {
    const s = await (await fetch('/analytics/summary')).json();
    const cards = [
      ['Total queries', fmt(s.total_queries)],
      ['Documents', fmt(s.total_documents)],
      ['Eval runs', fmt(s.total_evals)],
      ['Avg latency (ms)', fmt(s.avg_latency_ms)],
      ['Avg faithfulness', fmt(s.avg_faithfulness)],
      ['Avg precision', fmt(s.avg_precision)],
      ['Avg recall', fmt(s.avg_recall)],
    ];
    document.getElementById('cards').innerHTML = cards.map(
      c => `<div class="card"><div class="label">${c[0]}</div>`
         + `<div class="value">${c[1]}</div></div>`
    ).join('');

    const docs = await (await fetch('/analytics/documents')).json();
    document.querySelector('#docs tbody').innerHTML = docs.length ? docs.map(d =>
      `<tr><td>${fmt(d.filename)}</td><td class="muted">${fmt(d.doc_id)}</td>
       <td>${fmt(d.query_count)}</td><td>${fmt(d.avg_faithfulness)}</td></tr>`
    ).join('') : '<tr><td colspan="4" class="muted">No documents yet.</td></tr>';

    const evals = await (await fetch('/analytics/eval-trend')).json();
    document.querySelector('#evals tbody').innerHTML = evals.length ? evals.map(e =>
      `<tr><td>#${fmt(e.run_id)}</td><td class="muted">${fmt(e.created_at)}</td>
       <td>${fmt(e.num_questions)}</td><td>${fmt(e.status)}</td>
       <td>${fmt(e.average_faithfulness)}</td></tr>`
    ).join('') : '<tr><td colspan="5" class="muted">No evaluation runs yet.</td></tr>';
  } catch (err) {
    document.getElementById('cards').innerHTML =
      '<div class="card"><div class="label">Error</div><div class="value">&mdash;</div></div>';
  }
}
loadAll();
</script>
</body>
</html>"""


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

def main() -> None:
    """Entry point for running the API server directly.

    Invoked via ``python -m app.api.main`` (see the ``if __name__ == "__main__"``
    guard below). Extracted into a function so the uvicorn launch is unit-testable
    without actually binding a socket.
    """
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,    # auto-reload on file changes (dev only)
        log_level="info",
    )


if __name__ == "__main__":  # pragma: no cover - only executed when run as __main__
    main()
