# Adaptive Research & Analysis Platform (ARAP)

**ARAP turns your PDFs and research material into a question-answering system that
*cites its sources and checks its own work*.**

> Production-style, **multi-agent Retrieval-Augmented Generation (RAG)** platform for answering
> questions over PDF documents and research material. ARAP combines **adaptive query routing**,
> **hybrid retrieval**, **contextual chunk enrichment**, **knowledge-graph traversal**, a
> **local NLI faithfulness judge** with a retry loop, and **long-term memory** — all orchestrated
> by two compiled **LangGraph** graphs and served over a **FastAPI** HTTP + WebSocket API.

<p align="center">
  <img src="https://github.com/Hasan26ozcan/Autonomous-Research-Analysis-Platform/actions/workflows/ci.yml/badge.svg" alt="CI">
  <img src="https://codecov.io/gh/Hasan26ozcan/Autonomous-Research-Analysis-Platform/branch/main/graph/badge.svg" alt="Coverage">
</p>

---

## Table of Contents

- [What ARAP does](#what-arap-does)
- [Example: a real end-to-end query](#example-a-real-end-to-end-query)
- [Key features](#key-features)
- [Architecture](#architecture)
  - [The two LangGraph graphs](#the-two-langgraph-graphs)
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
- [Code quality & CI](#code-quality--ci)
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

## Example: a real end-to-end query

To show what ARAP actually produces (not a hand-written sample), here is a **real**
response from the pipeline. A research PDF — *"The LLM Fallacy: Misattribution in
AI-Assisted Cognitive Workflows"* by Hyunwoo Kim — was ingested, then asked a single
question. Everything below is the platform's own output, lightly formatted.

> **Question:** "What is the topic of this article?"

**Answer**

> The topic of this article is the **LLM Fallacy**, which refers to a cognitive
> attribution error where individuals misinterpret outputs generated with the assistance
> of Large Language Models (LLMs) as evidence of their own independent competence,
> leading to a systematic divergence between perceived and actual capability.

**Sources** — retrieved, reranked, and cited by the generator:

| # | Page | Chunk | Rerank score | Snippet |
| --- | --- | --- | --- | --- |
| 1 | 5 | 5 | −7.0052 | "…cognitive attribution errors in AI-assisted workflows…" |
| 2 | 6 | 6 | −7.4964 | "…manifestations across cognitive tasks…" |
| 3 | 10 | 12 | −7.6813 | "…misattribution of LLM-assisted outputs as evidence of human competence…" |
| 4 | 11 | 13 | −7.9684 | "…Guidelines…" |
| 5 | 1 | 0 | −8.0122 | "The LLM Fallacy:…" |

**Pipeline metrics** (per stage, in milliseconds):

| Stage | Latency (ms) |
| --- | --- |
| Router | 3,777 |
| Retrieval | 3,934 |
| Generation | 703 |
| Faithfulness judge | 2,534 |

- **Query type:** `single` — the router decided one hybrid retrieval round was enough.
- **Faithfulness score:** `0.994` — the answer is almost entirely grounded in the
  cited chunks; the local NLI judge found no meaningful contradiction.
- **Session id:** `test1`.

This single exchange exercises the whole stack described below: adaptive routing,
hybrid retrieval + cross-encoder reranking, grounded generation with `[Source N]`
citations, the faithfulness judge, and structured latency logging.

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

ARAP is a **multi-agent, retrieval-augmented generation (RAG)** platform built from two compiled
**LangGraph** graphs and served over **FastAPI**. At a glance, every request flows through the
same shape:

```
   PDF upload / question
            │
            ▼
   FastAPI  (app/api/main.py)
     ├─ /ingest  ──▶  Celery worker  ──▶  INGEST graph
     │                 (async, once per PDF)    chunk → enrich → embed
     │                                              → store → index → KG
     └─ /query, /ws  ──▶  QUERY graph  (per request)
                          router → retrieve → generate → judge
            │
            ▼
   Infrastructure:  Qdrant · Neo4j · Redis · PostgreSQL · (Mem0)
```

### The two LangGraph graphs

`app/core/orchestrator.py` compiles and caches two state graphs — this is the spine of the whole
system:

- **INGEST graph** — runs **once per PDF**, off the request path, inside a Celery worker. A
  linear, branchless pipeline (`chunk → enrich → embed → store → index → extract KG`) so heavy
  documents never block the API. KG extraction runs last, so a document stays searchable even if
  graph extraction is slow.
- **QUERY graph** — runs **per request**, adapting to the question. The `router` node classifies
  the question into one of four strategies (`direct` / `single` / `multi_hop` / `graph`), retrieval
  runs a 4-layer hybrid stack, then `generate` writes a cited answer that a local NLI judge scores —
  retrying once if it detects hallucination. A Redis checkpointer keyed by `session_id` keeps
  multi-turn context.

Every node is a plain `(AgentState) → dict` function; the shared schema lives in
`app/core/state.py`.

<details>
<summary>How the pieces fit — routing, retrieval, knowledge graph, memory & resilience</summary>

**Adaptive routing (4 strategies).** `app/agents/router.py` classifies each question *before*
retrieval and fetches the user's Mem0 memories in the same step:

| Strategy | Used when | Next step |
| --- | --- | --- |
| `direct` | General knowledge the LLM already knows | answer directly, no retrieval (~800 ms saved) |
| `single` | One chunk answers it | one hybrid retrieval round |
| `multi_hop` | Reasoning across sections | decompose → retrieve per sub-question → merge |
| `graph` | Relationships between named entities | read-only Neo4j Cypher |

It falls back to `single` on any error.

**The retrieval stack (4 layers).** `app/agents/retrieval_agent.py`:
(1) **HyDE** rewrites the question as a hypothetical answer and embeds that;
(2) **hybrid search** — BM25 (lexical) + dense vectors (Qdrant);
(3) **RRF fusion** (`k=60`, dense `0.7` / BM25 `0.3`);
(4) **cross-encoder rerank** (`nli-deberta-v3-small`) returns the top-`k`.
Results are cached in Redis by `(question, doc_id)`.

**Contextual enrichment.** Before embedding, an LLM prepends a 2–3 sentence context to each chunk
(Anthropic's *Contextual Retrieval*); the original text is kept for display and scoring.

**Knowledge graph (Neo4j).** `app/agents/graph_agent.py` extracts `(head, relation, tail)` triples
during ingest and generates **read-only** Cypher at query time, guarded by a 3-layer safety check
(prompt + `_validate_cypher()` + `execute_read()`).

**Faithfulness judge & retry.** A local NLI model scores each answer sentence
(`1 − P(contradiction)`); below `FAITHFULNESS_THRESHOLD` (0.90) the graph loops back to `generate`
once, then finalizes the best answer. No API cost, no infinite loop.

**Long-term memory (Mem0).** Personalizes answers from the same Qdrant instance and LLM endpoint —
self-hosted by default, no extra server needed.

**Resilience.** Redis serves as checkpointer + broker + cache (all best-effort). A proactive
sliding-window rate limiter paces LLM calls. Mem0, Neo4j, NLI, BM25, and PostgreSQL are all
optional — a failure degrades the answer, never crashes the request. Ingest is idempotent
(`doc_id` = SHA-256 of the PDF bytes).

**Observability.** LangSmith tracing (opt-in), PostgreSQL per-node/pipeline logging, a built-in
HTML `/analytics` dashboard, and RAGAS evaluation persisted to PostgreSQL.

</details>

> 📊 For a fully diagram-driven walkthrough — system overview, both graphs, the retrieval stack,
> service dependencies, the Docker layout, and an end-to-end sequence — open
> **[`architecture.md`](architecture.md)**.

### Infrastructure services

| Service | Image | Role |
| --- | --- | --- |
| **Qdrant** | `qdrant/qdrant:v1.12.0` | Dense vector store (`arap_docs` collection). |
| **Neo4j** | `neo4j:5.25-community` | Knowledge graph (triples, Cypher traversal). |
| **Redis Stack** | `redis/redis-stack:latest` | Checkpointer + Celery broker + caches. |
| **PostgreSQL** | `postgres:16-alpine` | Metadata, history, eval results, logs. |
| **API / Worker** | built from `Dockerfile` | FastAPI+Uvicorn, and the Celery ingest worker. |

---

## Project structure

```text
ARAP/
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                      # FastAPI app: /health, /ingest, /query, /ws, /eval, /analytics (+ /analytics/* JSON)
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
| `FAITHFULNESS_THRESHOLD` | `0.90` | Mean faithfulness (1 - P(contradiction)) below this triggers a retry. |
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
| `GET` | `/analytics` | Built-in HTML analytics dashboard (fetches the three JSON endpoints below). |
| `GET` | `/analytics/summary` | Headline metrics: total queries, documents, eval runs, avg latency / faithfulness / precision / recall. |
| `GET` | `/analytics/documents` | Top documents by query volume with average faithfulness. |
| `GET` | `/analytics/eval-trend` | Recent evaluation runs with headline metrics. |
| `GET` | `/docs` | Auto-generated OpenAPI/Swagger docs. |

**Example `/query` response**

```json
{
  "answer": "The proposed method achieves state-of-the-art results [Source 1]…",
  "sources": [
    { "index": 1, "text": "…", "page": 7, "filename": "paper.pdf",
      "doc_id": "a3f8b12c", "chunk_index": 23, "rerank_score": -7.01 }
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

## Code quality & CI

This repo runs an automated quality gate on every push to `main` and on every PR that targets
`main` (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

| Job | Tool | What it checks | Merge status |
| --- | --- | --- | --- |
| `unit-tests` | pytest + pytest-cov | Unit suite + coverage | **Required** (blocks merge) |
| `lint` | [Ruff](https://docs.astral.sh/ruff/) | Style, unused imports, undefined names, pyupgrade | Informational |
| `type-check` | [mypy](https://mypy-lang.org/) | Static type hints | Informational |
| `security` | [pip-audit](https://pypi.org/project/pip-audit/) | Known vulnerabilities in `requirements.txt` | Informational |
| `pre-commit` | [pre-commit](https://pre-commit.com/) | Same hooks as the local git hook | Informational |

Coverage is uploaded to [Codecov](https://codecov.io/) and a per-PR comment reports the change
("this PR increases/decreases coverage by X%"). The CI and coverage badges at the top of this
README reflect the latest `main` run.

**Local setup (recommended):** install the pre-commit hook so the same checks run *before every
commit*, catching issues on your machine instead of only in CI:

```bash
pip install pre-commit
pre-commit install
```

[Dependabot](https://docs.github.com/en/code-security/dependabot) keeps `requirements.txt` and the
GitHub Actions used by the workflow up to date (`.github/dependabot.yml`). The full walkthrough —
including how to enable Dependabot security alerts, configure branch protection so `main` can't be
merged with a red test suite, and promote the informational jobs to required — lives in
[`docs/CI_QUALITY.md`](docs/CI_QUALITY.md).

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
