# Adaptive Research & Analysis Platform (ARAP)

> Production-style, **multi-agent Retrieval-Augmented Generation (RAG)** platform for answering
> questions over PDF documents and research material. ARAP combines **adaptive query routing**,
> **hybrid retrieval**, **contextual chunk enrichment**, **knowledge-graph traversal**, a
> **local NLI faithfulness judge** with a retry loop, and **long-term memory** — all orchestrated
> by two compiled **LangGraph** graphs and served over a **FastAPI** HTTP + WebSocket API.

---

## Table of Contents

- [What ARAP does](#what-arap-does)
- [Key features](#key-features)
- [Architecture](#architecture)
  - [System overview](#system-overview)
  - [The two LangGraph graphs](#the-two-langgraph-graphs)
  - [Adaptive routing (4 strategies)](#adaptive-routing-4-strategies)
  - [The retrieval stack (4 layers)](#the-retrieval-stack-4-layers)
  - [Contextual enrichment](#contextual-enrichment)
  - [Knowledge graph (Neo4j)](#knowledge-graph-neo4j)
  - [Faithfulness judging & retry loop](#faithfulness-judging--retry-loop)
  - [Long-term memory (Mem0)](#long-term-memory-mem0)
  - [Caching, rate limiting & resilience](#caching-rate-limiting--resilience)
  - [Observability & evaluation](#observability--evaluation)
  - [Infrastructure services](#infrastructure-services)
- [Project structure](#project-structure)
- [Quickstart](#quickstart)
  - [Option A — Docker Compose (recommended)](#option-a--docker-compose-recommended)
  - [Option B — Local (virtualenv)](#option-b--local-virtualenv)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [WebSocket protocol](#websocket-protocol)
- [Evaluation (RAGAS)](#evaluation-ragas)
- [Testing](#testing)
- [Design principles](#design-principles)
- [Tech stack](#tech-stack)
- [License](#license)

---

## What ARAP does

ARAP is built for document-heavy workflows where a single-vector-search RAG is not enough. For
each question it **routes** to the right strategy, **retrieves** evidence with a hybrid
BM25 + dense pipeline, optionally walks a **knowledge graph**, **generates** a grounded answer
with source citations, and **self-checks** the answer for faithfulness — retrying with a stricter
prompt when it detects hallucination.

The pipeline can:

- ingest PDF documents and prepare them for retrieval (chunk → enrich → embed → store);
- extract an entity/relationship knowledge graph into Neo4j during ingestion;
- route each question to the optimal retrieval strategy (`direct` / `single` / `multi_hop` / `graph`);
- retrieve relevant evidence with HyDE rewriting + BM25 + dense vector search + RRF fusion + cross-encoder reranking;
- optionally use knowledge-graph traversal for relationship-aware questions;
- generate grounded answers with inline `[Source N]` citations;
- judge answer faithfulness with a local NLI model and retry when it falls short;
- personalize answers with Mem0 long-term memory;
- stream progress and final results over WebSocket;
- log every request, pipeline node, and query to PostgreSQL for analytics and offline evaluation.

---

## Key features

| Area | Capability |
| --- | --- |
| **Adaptive routing** | Four routing strategies (`direct`, `single`, `multi_hop`, `graph`) classified by an LLM before any retrieval. |
| **Hybrid retrieval** | BM25 (lexical) + dense vectors, merged with Reciprocal Rank Fusion, then re-ranked with a cross-encoder. |
| **HyDE rewriting** | Embeds a hypothetical answer instead of the raw question to close the vocabulary gap. |
| **Contextual retrieval** | Anthropic-style context prepended to each chunk before embedding for better recall. |
| **Knowledge graph** | Entity/relation triples extracted into Neo4j; read-only Cypher generation for graph questions. |
| **Faithfulness judge** | Local NLI (DeBERTa-v3-small) scores every answer sentence; retry loop on low scores. |
| **Long-term memory** | Mem0 personalizes answers from past conversations (self-hosted or hosted). |
| **Async ingestion** | Heavy PDF ingestion runs in a Celery worker; the API returns a task id immediately. |
| **Streaming** | WebSocket endpoint streams one event per LangGraph node in real time. |
| **Observability** | LangSmith tracing, PostgreSQL access/pipeline/conversation logs, and a built-in HTML analytics dashboard. |
| **Evaluation** | RAGAS metrics (faithfulness, answer relevancy, context precision/recall) persisted to PostgreSQL. |
| **Provider-agnostic LLM** | Point `LLM_BASE_URL` at OpenAI, Groq, or Ollama — no code changes. |

---

## Architecture

### System overview

```
                         ┌─────────────────────────────────────────────────────────┐
        PDF upload ─────▶│  FastAPI (app/api/main.py)                              │
                         │   • /ingest  → Celery task (async)                       │
        question  ─────▶│   • /query   → ARAPOrchestrator.query()                  │
                         │   • /ws/{sid} → ARAPOrchestrator.stream_query()         │
                         └───────────────┬───────────────────────┬────────────────┘
                                         │                        │
                            ┌────────────▼─────────┐   ┌──────────▼───────────┐
                            │  Celery worker        │   │  LangGraph QUERY graph │
                            │  (ingest pipeline)    │   │  (per-request)          │
                            └────────────┬─────────┘   └──────────┬───────────┘
                                         │                        │
        ┌────────────────────────────────┼────────────────────────┼───────────────────┐
        │  Qdrant (vectors)   Neo4j (KG)   Redis (cache+checkpointer+broker)   PostgreSQL  │
        └────────────────────────────────────────────────────────────────────────────────┘
```

### The two LangGraph graphs

ARAP compiles **two** LangGraph state graphs (`app/core/orchestrator.py`). Both are built once
and cached; the query graph is compiled with a **Redis checkpointer** keyed by `session_id` so
multi-turn conversations persist automatically.

**INGEST graph** — runs once per PDF (in the Celery worker):

```
chunk_document ─▶ enrich_chunks ─▶ embed_chunks ─▶ store_chunks ─▶ index_chunks ─▶ extract_and_store_node ─▶ END
   (Phase 2)        (Phase 3)       (Phase 2)       (Phase 2)      (Phase 2)         (Phase 6: Neo4j triples)
```

**QUERY graph** — runs per request, with an adaptive branch and a retry cycle:

```
                 ┌──────────────────────────────────────────────────┐
   router ───────▶│ Phase 4: classify + fetch Mem0 memories          │
       │          └──────────────────────────────────────────────────┘
       │ get_route() conditional edge:
       ├─ "direct"    ─▶ direct_answer ─────────────────┐
       ├─ "single"    ─▶ retrieve (HyDE+hybrid+rerank)  │
       ├─ "multi_hop" ─▶ retrieve_multi (decompose+multi)│
       └─ "graph"     ─▶ graph_retrieve (Neo4j Cypher)   │
       │                 └─────────▶ merge_results (convergence) ─▶ generate ─▶ judge ─┐
       │                                                                                │
       │                                          should_retry() conditional edge:     │
       │                                            "generate"     ──▶ generate (retry) │
       │                                            "memory_store" ──▶ memory_store ──▶ END
```

Every node is a plain function `(AgentState) -> dict` (or `None` for side-effect-only nodes) that
returns only the state fields it changed. The shared `AgentState` schema is defined in
`app/core/state.py`.

### Adaptive routing (4 strategies)

`app/agents/router.py` classifies each question *before* any retrieval (Jeong et al., 2024,
*Adaptive RAG*, arXiv:2403.14403) and fetches the user's long-term Mem0 memories in the same node:

| Strategy | When | Next step |
| --- | --- | --- |
| `direct` | General knowledge the LLM already knows (e.g. "What is cosine similarity?") | `direct_answer` — no retrieval, saves ~800 ms. |
| `single` | One chunk answers it (e.g. "What revenue was reported in Q3 2024?") | `retrieve` — one hybrid round. |
| `multi_hop` | Needs reasoning across sections (e.g. "How does section 3 address section 7's limitations?") | `retrieve_multi` — decompose → retrieve per sub-question → merge → rerank. |
| `graph` | Structural relationships between named entities (e.g. "Which authors co-appear in papers cited by both chapter 2 and 5?") | `graph_retrieve` — Neo4j Cypher. |

The classifier is cheap (`router_model`, ~$0.0001) and falls back to `single` on any error.

### The retrieval stack (4 layers)

`app/agents/retrieval_agent.py` implements a four-layer pipeline (used by both `retrieve` and
`retrieve_multi`):

1. **HyDE rewriting** — generate a hypothetical answer passage and embed *that* instead of the
   question, closing the question/doc vocabulary gap (Gao et al., 2022, arXiv:2212.10496).
2. **Hybrid search** — BM25 (`rank_bm25`) catches exact terms/codes/names; dense vectors
   (`sentence-transformers` → Qdrant HNSW) catch paraphrase/semantics.
3. **RRF fusion** — Reciprocal Rank Fusion (`score = Σ weight · 1/(k + rank)`, `k=60`,
   dense `0.7` / BM25 `0.3`) merges the two ranked lists by *rank*, ignoring incompatible raw
   scores (Cormack et al., SIGIR 2009).
4. **Cross-encoder reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each
   (question, chunk) pair with full attention and returns the top-`k` (CPU, ~80 ms / 10 pairs).

Retrieval results are cached in Redis keyed by `(question, doc_id)`.

### Contextual enrichment

`app/services/contextual_enricher.py` implements Anthropic's *Contextual Retrieval*: before
embedding, an LLM prepends a 2–3 sentence context description to each chunk
(`[Context: ...]\n\n<original text>`). The enriched text is what gets **embedded** and **stored**,
while the original text is preserved (`original_text`) for display and NLI scoring. Enrichment
degrades gracefully — a failed chunk keeps its original text.

### Knowledge graph (Neo4j)

`app/agents/graph_agent.py` has two responsibilities:

- **Ingestion (`extract_and_store_node`)** — extracts `(head, relation, tail, confidence)` triples
  from every chunk in parallel (`ThreadPoolExecutor`, 5 workers) and writes them to Neo4j in a
  single idempotent `UNWIND … MERGE` batch. Runs *last* in the ingest graph so the document is
  already searchable via Qdrant/BM25 even if KG extraction is slow or partially fails.
- **Query (`graph_retrieve`)** — extracts entities from the question, generates a **read-only**
  Cypher query, validates it, and executes it in a Neo4j read transaction.

**Cypher safety (3 layers):** (1) prompt forbids write keywords; (2) `_validate_cypher()` rejects
any query containing `CREATE/MERGE/DELETE/SET/…`, missing `LIMIT`, or missing `MATCH`;
(3) `session.execute_read()` enforces read-only at the driver level.

### Faithfulness judging & retry loop

`app/agents/generator.py` owns generation, judging, and memory storage:

- **generate()** — assembles a context window from retrieved chunks + KG paths + Mem0 memories,
  calls `llm_model` (GPT-4o by default) with a strict grounding prompt that requires `[Source N]`
  citations. Uses a stricter prompt on retry attempts.
- **judge()** — splits the draft into sentences and scores each against the concatenated context
  with a local NLI model (`cross-encoder/nli-deberta-v3-small`). `faithfulness_score` = mean
  entailment probability. If below `faithfulness_threshold` (0.75) and retries remain, the graph
  loops back to `generate`; otherwise the answer is finalized (best available, never an infinite loop).
- **store_memory()** — persists the approved Q&A turn to Mem0 for future personalization.

Using a local NLI model instead of LLM-as-judge avoids API cost and ~2–3 s latency per answer.

### Long-term memory (Mem0)

The router and generator use Mem0 to personalize answers. By default it runs in **embedded
( self-hosted)** mode: it points at the *same* Qdrant instance (separate `arap_memories` collection)
and the *same* local `sentence-transformers` model, and calls whatever OpenAI-compatible LLM
endpoint the app already uses — so **no separate Mem0 server or account is required**. Set
`MEM0_API_KEY` to use the hosted Mem0 Platform instead. Memory failures are non-fatal.

### Caching, rate limiting & resilience

- **Redis** plays five roles: LangGraph session checkpointer, Celery broker, retrieval-result
  cache, LLM-response cache, and a transient "ingest in flight" flag (`app/services/redis_cache.py`).
  All cache calls are best-effort — if Redis is down, the app falls back to the uncached path.
- **Proactive rate limiter** (`app/services/rate_limiter.py`) — a thread-safe sliding-window
  limiter paces LLM calls to stay under the provider's RPM/TPM budget (tuned for Groq's free tier)
  *before* sending, avoiding most 429s. The OpenAI SDK adds reactive retries on the rest.
- **Graceful degradation everywhere** — Mem0, Neo4j KG, NLI, BM25, and PostgreSQL are all
  optional; a failure in any one degrades the answer rather than crashing the request.
- **Idempotent ingest** — `doc_id` is the SHA-256 of the PDF bytes (first 16 hex chars), so
  re-uploading the same file upserts rather than duplicating.

### Observability & evaluation

- **LangSmith** tracing is enabled automatically when `LANGCHAIN_TRACING_V2=true` — every node
  appears as a named span with input/output, token counts, and latency.
- **PostgreSQL logging** — `api_log`, `worker_log`, `pipeline_log` (per-node latency),
  `query_history` (also an auto RAGAS test set), `conversations`, and `memories`.
- **Analytics dashboard** — `GET /analytics` renders a self-contained HTML page with summary
  stats, top documents, and evaluation trends (`app/services/analytics.py`).
- **RAGAS evaluation** — `POST /eval` (or `python -m evaluation.ragas_eval`) runs the live
  pipeline over real or seed questions and persists faithfulness / answer-relevancy /
  context-precision / context-recall to PostgreSQL.

### Infrastructure services

| Service | Image | Role |
| --- | --- | --- |
| **Qdrant** | `qdrant/qdrant:v1.12.0` | Dense vector store (`arap_docs` collection, cosine, HNSW). |
| **Neo4j** | `neo4j:5.25-community` | Knowledge graph (triples, Cypher traversal). |
| **Redis Stack** | `redis/redis-stack:latest` | Checkpointer + Celery broker + caches (RediSearch module). |
| **PostgreSQL** | `postgres:16-alpine` | Metadata, query/conversation history, eval results, logs. |
| **API** | built from `Dockerfile` | FastAPI + Uvicorn. |
| **Worker** | built from `Dockerfile` | Celery worker for async ingestion. |

---

## Project structure

```text
ARAP/
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                      # FastAPI app: /health, /ingest, /query, /ws, /eval, /analytics
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── router.py                    # Adaptive router + Mem0 fetch (Phase 4)
│   │   ├── retrieval_agent.py          # HyDE + hybrid + RRF + cross-encoder rerank (Phase 5)
│   │   ├── graph_agent.py               # Neo4j triple extraction + read-only Cypher (Phase 6)
│   │   └── generator.py                 # Generate + NLI faithfulness judge + Mem0 store (Phase 7)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                    # Single source of truth for all settings (pydantic-settings)
│   │   ├── orchestrator.py              # Builds & runs the two LangGraph graphs
│   │   ├── state.py                     # Shared AgentState TypedDict schema
│   │   ├── celery_app.py                # Celery application instance
│   │   └── logging.py                   # Shared logging configuration
│   └── services/
│       ├── __init__.py
│       ├── chunker.py                   # PDF → overlapping chunks (PyMuPDF) (Phase 2)
│       ├── contextual_enricher.py       # Prepend LLM context to chunks (Phase 3)
│       ├── embedder.py                  # sentence-transformers embeddings (Phase 2)
│       ├── vector_store.py              # Qdrant upsert/search (Phase 2)
│       ├── bm25_index.py                # In-memory BM25 keyword index (Phase 2)
│       ├── llm_client.py                # make_llm() factory: 429-resilient, token tracking, LLM cache
│       ├── rate_limiter.py              # Proactive sliding-window RPM/TPM limiter
│       ├── redis_cache.py               # Retrieval/LLM caches + ingest state flag
│       ├── ingest_service.py            # Synchronous ingest pipeline run inside Celery
│       ├── tasks.py                     # Celery task wrapping the ingest pipeline
│       ├── postgres_store.py            # psycopg2 persistence (documents, queries, conversations…)
│       ├── eval_store.py                # Persist RAGAS evaluation runs/scores
│       ├── log_store.py                 # API / worker / pipeline log writes
│       └── analytics.py                 # Aggregations for the /analytics dashboard
├── evaluation/
│   ├── __init__.py
│   └── ragas_eval.py                    # RAGAS evaluation suite + seed questions
├── scripts/
│   └── init_db.sql                      # Idempotent PostgreSQL schema (auto-run on first boot)
├── docs/
│   └── phase1_setup.py                  # Reference setup notes / dependency manifest
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── integration/
│   │   └── __init__.py
│   └── unit/
│       └── test_phase*.py              # Per-phase unit tests (services → orchestrator → API)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
├── README.md
└── .env.example
```

---

## Quickstart

### Option A — Docker Compose (recommended)

1. **Clone & configure**

   ```bash
   git clone <your-repo-url>
   cd Autonomous-Research-Analysis-Platform
   cp .env.example .env
   ```

   Edit `.env` and set at minimum `OPENAI_API_KEY`. To use a free OpenAI-compatible provider
   (Groq or Ollama) instead, set `LLM_BASE_URL` and the three `*_MODEL` variables (see
   [Configuration](#configuration)).

2. **Start everything**

   ```bash
   docker compose up -d --build
   ```

   This starts `qdrant`, `neo4j`, `redis`, `postgres`, the `api`, and the `worker`. The API is
   healthy once `http://localhost:8000/health` returns `ok` for all components.

3. **Ingest a document**

   ```bash
   curl -X POST http://localhost:8000/ingest \
     -F "file=@research_paper.pdf"
   # → { "task_id": "...", "status": "queued", ... }
   ```

   Track progress with `GET /ingest/status/{task_id}`.

4. **Ask a question**

   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What is the main contribution of this document?",
       "user_id": "demo-user"
     }'
   ```

### Option B — Local (virtualenv)

Requires Python 3.10+ and running Qdrant / Neo4j / Redis / PostgreSQL instances (e.g. via the
compose file's infra services only: `docker compose up -d qdrant neo4j redis postgres`).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

cp .env.example .env             # fill in OPENAI_API_KEY / LLM_BASE_URL

# Terminal 1 — API
uvicorn app.api.main:app --reload

# Terminal 2 — Celery worker (for /ingest)
celery -A app.core.celery_app worker --loglevel=info --concurrency=2
```

Then open [http://localhost:8000/docs](http://localhost:8000/docs) or
[http://localhost:8000/analytics](http://localhost:8000/analytics).

---

## Configuration

All configuration lives in `app/core/config.py` and is loaded from environment variables (or
`.env`) via `pydantic-settings`. Every variable has a safe default, so you only override what
differs. The most important ones:

| Variable | Default | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | API key for the LLM (or your OpenAI-compatible provider). |
| `LLM_BASE_URL` | _(empty)_ | Point at OpenAI, Groq (`https://api.groq.com/openai/v1`), or Ollama (`http://localhost:11434/v1`). Empty = OpenAI. |
| `LLM_MODEL` | `gpt-4o` | Generation model (used only for the final answer). |
| `ROUTER_MODEL` | `gpt-4o-mini` | Fast/cheap model for routing, HyDE, KG extraction. |
| `JUDGE_MODEL` | `gpt-4o-mini` | Fallback LLM judge. |
| `LLM_RPM_LIMIT` / `LLM_TPM_LIMIT` | `30` / `6000` | Proactive rate-limiter budgets. |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model (384-d). Swap for Qwen3-Embedding-4B on GPU. |
| `EMBEDDING_DIM` | `384` | Must match the embedding model output dimension. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `64` | Words per chunk / overlap. |
| `QDRANT_HOST` / `QDRANT_PORT` / `QDRANT_COLLECTION` | `localhost` / `6333` / `arap_docs` | Vector DB. |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | `bolt://localhost:7687` / `neo4j` / `password` | Knowledge graph. |
| `REDIS_URL` | `redis://localhost:6379/0` | Checkpointer + cache. |
| `POSTGRES_URL` | `postgresql://user:password@localhost:5432/arap` | Metadata / history / eval. |
| `MEM0_API_KEY` | _(empty)_ | Leave empty for self-hosted Mem0; set for hosted platform. |
| `MEM0_BASE_URL` | `http://localhost:3000` | Mem0 endpoint. |
| `LANGCHAIN_TRACING_V2` | `false` | Set `true` to enable LangSmith tracing. |
| `LANGCHAIN_API_KEY` | — | LangSmith project key. |
| `TOP_K_RETRIEVAL` / `TOP_K_FINAL` | `10` / `5` | Candidates before rerank / chunks sent to LLM. |
| `RRF_K` / `DENSE_WEIGHT` / `BM25_WEIGHT` | `60` / `0.7` / `0.3` | RRF fusion parameters. |
| `FAITHFULNESS_THRESHOLD` | `0.75` | NLI score below this triggers a retry. |
| `NLI_MODEL` | `cross-encoder/nli-deberta-v3-small` | Local faithfulness model. |
| `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` | `redis://localhost:6379/1` / `/2` | Celery broker/result. |

---

## API reference

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Reachability of Qdrant, Neo4j, Redis. |
| `POST` | `/ingest` | Upload a PDF (`.pdf`, ≤50 MB). Returns `task_id` immediately; ingestion runs in Celery. |
| `GET` | `/ingest/status/{task_id}` | Ingest status: `PENDING` / `STARTED` / `SUCCESS` / `FAILURE`. |
| `POST` | `/query` | Synchronous adaptive Q&A. Returns `answer`, `sources`, `query_type`, `faithfulness_score`, `latency_ms`. |
| `WS` | `/ws/{session_id}` | Streaming Q&A — one event per LangGraph node, then a final `done` message. |
| `POST` | `/eval` | Run the RAGAS evaluation suite and persist results to PostgreSQL. |
| `GET` | `/analytics` | Built-in HTML analytics dashboard. |
| `GET` | `/docs` | Auto-generated OpenAPI/Swagger docs. |

**Example `/query` response**

```json
{
  "answer": "The proposed method achieves state-of-the-art results [Source 1]…",
  "sources": [
    { "index": 1, "text": "…", "page": 7, "filename": "paper.pdf",
      "doc_id": "a3f8b12c", "chunk_index": 23, "rerank_score": 0.91 }
  ],
  "query_type": "single",
  "faithfulness_score": 0.92,
  "session_id": "bb1c…",
  "latency_ms": { "router": 312, "retrieval": 847, "generation": 1240, "judge": 180 }
}
```

---

## WebSocket protocol

Connect to `ws://localhost:8000/ws/{session_id}` (the `session_id` becomes the LangGraph
`thread_id`, enabling multi-turn memory). The connection stays open for multiple questions.

**Client → Server**

```json
{ "question": "How does section 3 address the limitations in section 7?", "user_id": "demo", "doc_id": null }
```

**Server → Client** (one `update` per node, then a `done`)

```json
{ "type": "update", "node": "router",   "data": { "query_type": "multi_hop", ... } }
{ "type": "update", "node": "retrieve_multi", "data": { "sub_questions": [...], ... } }
{ "type": "update", "node": "generate", "data": { "draft_answer": "..." } }
{ "type": "update", "node": "judge",    "data": { "faithfulness_score": 0.91 } }
{ "type": "done",   "answer": "...", "sources": [...], "faithfulness_score": 0.91, "query_type": "multi_hop" }
```

Errors arrive as `{ "type": "error", "message": "..." }`; the connection remains open for the next question.

---

## Evaluation (RAGAS)

```bash
# From the CLI (against the live pipeline)
python -m evaluation.ragas_eval --limit 20 --save reports/ragas_report.json

# Or via the API
curl -X POST http://localhost:8000/eval -H "Content-Type: application/json" \
  -d '{ "limit": 20 }'
```

The suite loads real questions from `query_history` (falling back to seed questions), runs them
through the orchestrator, computes RAGAS metrics, and writes the run + scores to PostgreSQL
(`evaluation_runs`, `evaluation_scores`, `retrieval_results`). If the `ragas`/`datasets`
packages are unavailable, it returns a partial report using the pipeline's own faithfulness score.

---

## Testing

```bash
# Unit tests (per-phase: services → orchestrator → API/WebSocket)
pytest tests/unit -q

# Integration tests
pytest tests/integration -q

# With coverage
pytest --cov=app --cov=evaluation -q
```

Tests use `pytest-asyncio` (auto mode), `httpx`/`httpx-ws` for the ASGI/WebSocket client, and
extensive mocking of the LLM/orchestrator so they run without live infrastructure.

---

## Design principles

- **Every pipeline stage is an explicit, swappable node.** The two LangGraph graphs make routing,
  retrieval, generation, and judging visible and easy to extend.
- **Cheap model for cheap work.** Routing, HyDE, decomposition, and KG extraction use
  `router_model`; only final generation uses `llm_model`.
- **Local inference where possible.** Embeddings, the cross-encoder reranker, and the NLI judge
  all run on CPU with no API cost.
- **Graceful degradation.** No single optional subsystem (Mem0, Neo4j, NLI, BM25, PostgreSQL,
  Redis) can crash a request.
- **Observability by default.** Latency, faithfulness, token usage, and retrieval quality are
  logged at every node.
- **Provider-agnostic.** One `LLM_BASE_URL` switch moves the whole app between OpenAI, Groq, and
  Ollama.

---

## Tech stack

| Layer | Choice |
| --- | --- |
| API | FastAPI, Uvicorn, WebSockets, Pydantic v2 |
| Orchestration | LangGraph (StateGraph), LangChain, Redis checkpointer |
| Retrieval | Qdrant, `rank-bm25`, `sentence-transformers`, cross-encoder reranking |
| Knowledge graph | Neo4j (Cypher) |
| Long-term memory | Mem0 (self-hosted embedded or hosted) |
| Async jobs | Celery + Redis broker |
| Persistence | PostgreSQL (psycopg2) |
| Evaluation | RAGAS, `datasets` |
| Observability | LangSmith, PostgreSQL logs, HTML analytics dashboard |
| Infra | Docker Compose (Qdrant, Neo4j, Redis Stack, Postgres) |

---

## License

MIT
