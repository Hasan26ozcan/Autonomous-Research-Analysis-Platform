# ARAP — Architecture Diagrams

This file describes the architecture of the **Adaptive Research & Analysis
Platform (ARAP)** using Mermaid diagrams. The diagrams are based on the actual
code topology in `app/core/orchestrator.py`, `app/api/main.py`, and
`app/agents/*`. GitHub and VS Code (Mermaid extension) render this file directly.

---

## 1. System Overview

Client requests arrive at FastAPI. `/ingest` hands the heavy PDF processing off
to a **Celery worker** (Redis broker); `/query` and `/ws` run the per-request
**QUERY LangGraph**. All persistent and transient state lives on five
infrastructure services.

```mermaid
flowchart TB
    Client["Client<br/>(Browser / API client)"]

    subgraph API["FastAPI (app/api/main.py)"]
        EP_INGEST["POST /ingest<br/>(async → task_id)"]
        EP_QUERY["POST /query<br/>(sync)"]
        EP_WS["WS /ws/{session_id}<br/>(stream)"]
        EP_EVAL["POST /eval"]
        EP_ANALYTICS["GET /analytics/*"]
        Orch["ARAPOrchestrator<br/>(app/core/orchestrator.py)"]
        EP_INGEST --> Orch
        EP_QUERY --> Orch
        EP_WS --> Orch
    end

    subgraph WORKER["Celery Worker (Docker)"]
        TASK["ingest_document_task<br/>(app/services/tasks.py)"]
        INGEST_G["INGEST LangGraph<br/>(chunk→enrich→embed→store→index→KG)"]
        TASK --> INGEST_G
    end

    Client -->|PDF / question| Orch
    EP_INGEST -.->|enqueue| BROKER[(Redis broker)]
    BROKER -.->|consume| TASK

    Orch -->|compile & run| QUERY_G["QUERY LangGraph<br/>(router→retrieve→generate→judge)"]
    Orch -.->|Redis checkpointer| REDIS
    TASK -.->|BM25 reload signal| REDIS

    subgraph INFRA["Infrastructure Services"]
        QDRANT[(Qdrant<br/>vectors)]
        NEO4J[(Neo4j<br/>knowledge graph)]
        REDIS[(Redis Stack<br/>cache/checkpointer/broker)]
        PG[(PostgreSQL<br/>metadata/logs/eval)]
        MEM0[(Mem0<br/>long-term memory)]
    end

    INGEST_G --> QDRANT
    INGEST_G --> NEO4J
    QUERY_G --> QDRANT
    QUERY_G --> NEO4J
    QUERY_G --> MEM0
    EP_EVAL --> PG
    EP_ANALYTICS --> PG
    TASK --> PG
    Orch -.->|health check| QDRANT
    Orch -.->|health check| NEO4J
    Orch -.->|health check| REDIS
```

---

## 2. INGEST Graph (Document Processing — once per PDF)

`app/core/orchestrator.py → build_ingest_graph()`. A linear, branchless
pipeline. **KG extraction runs last** so the document stays searchable via
Qdrant/BM25 even if extraction is slow (graceful degradation). No checkpointer
(stateless).

```mermaid
flowchart LR
    START(["START"]) --> CHUNK["chunk_document<br/>(Phase 2 · PyMuPDF)"]
    CHUNK --> ENRICH["enrich_chunks<br/>(Phase 3 · LLM context)"]
    ENRICH --> EMBED["embed_chunks<br/>(Phase 2 · sentence-transformers)"]
    EMBED --> STORE["store_chunks<br/>(Qdrant HNSW upsert)"]
    STORE --> INDEX["index_chunks<br/>(BM25 corpus)"]
    INDEX --> KG["extract_and_store_node<br/>(Phase 6 · Neo4j triples)"]
    KG --> END(["END"])

    STORE --> QDRANT[(Qdrant)]
    INDEX --> BM25[(BM25 in-memory)]
    KG --> NEO4J[(Neo4j)]
```

---

## 3. QUERY Graph (Query — per request, adaptive)

`app/core/orchestrator.py → build_query_graph()`. Two conditional edges:
(1) `router → get_route()` branches to 4 strategies; (2) `judge → should_retry()`
drives the retrieve/generate retry loop. The single cycle (retry loop) relies on
LangGraph's native cycle support. The Redis checkpointer preserves multi-turn
memory keyed by `session_id`.

```mermaid
flowchart TD
    START(["START"]) --> ROUTER["router<br/>(Phase 4 · classify + Mem0 fetch)"]

    ROUTER -->|get_route: direct| DIRECT["direct_answer<br/>(no retrieval)"]
    ROUTER -->|get_route: single| RETRIEVE["retrieve<br/>(HyDE+hybrid+rerank)"]
    ROUTER -->|get_route: multi_hop| RMULTI["retrieve_multi<br/>(decompose+multi)"]
    ROUTER -->|get_route: graph| GRAPH["graph_retrieve<br/>(Neo4j Cypher)"]

    RETRIEVE --> MERGE["merge_results<br/>(convergence)"]
    RMULTI --> MERGE
    GRAPH --> MERGE
    DIRECT --> MEMSTORE

    MERGE --> GEN["generate<br/>(Phase 7 · GPT-4o + sources)"]
    GEN --> JUDGE["judge<br/>(Phase 7 · NLI faithfulness)"]

    JUDGE -->|should_retry: generate| GEN
    JUDGE -->|should_retry: memory_store| MEMSTORE["memory_store<br/>(Mem0 persist)"]

    MEMSTORE --> END(["END"])

    DIRECT -.->|faithfulness_score=1.0| MEMSTORE
```

> **Adaptive routing strategies** (`app/agents/router.py`):
> `direct` (parametric knowledge, no retrieval) · `single` (one hybrid round) ·
> `multi_hop` (decompose into sub-questions + merge) · `graph` (Neo4j Cypher).
> Falls back to `single` on any error.

---

## 4. Retrieval Stack (4 Layers)

Implemented in `app/agents/retrieval_agent.py`, used by both `retrieve` and
`retrieve_multi`.

```mermaid
flowchart LR
    Q["Question"] --> HYDE["1 · HyDE rewriting<br/>(hypothetical answer embed)"]
    HYDE --> HYBRID["2 · Hybrid search<br/>BM25 (lexical) + dense (Qdrant)"]
    BM25[(BM25)] --> HYBRID
    QDRANT[(Qdrant)] --> HYBRID
    HYBRID --> RRF["3 · RRF fusion<br/>(k=60, dense 0.7 / BM25 0.3)"]
    RRF --> RERANK["4 · Cross-encoder rerank<br/>(top-k, NLI model)"]
    RERANK --> OUT["Retrieved chunks<br/>+ rerank_score"]
```

> Results are cached in Redis keyed by `(question, doc_id)`.

---

## 5. Service Layer and Dependencies

```mermaid
flowchart TB
    subgraph AGENTS["Agents (app/agents)"]
        ROUTER_A["router.py"]
        RET_A["retrieval_agent.py"]
        GRAPH_A["graph_agent.py"]
        GEN_A["generator.py"]
    end

    subgraph CORE["Core (app/core)"]
        ORCH["orchestrator.py"]
        STATE["state.py (AgentState)"]
        CFG["config.py (pydantic-settings)"]
        CELERY["celery_app.py"]
    end

    subgraph SERVICES["Services (app/services)"]
        CHUNK["chunker.py"]
        ENRICH["contextual_enricher.py"]
        EMBED["embedder.py"]
        VEC["vector_store.py"]
        BM25S["bm25_index.py"]
        LLM["llm_client.py"]
        RATE["rate_limiter.py"]
        CACHE["redis_cache.py"]
        INGEST["ingest_service.py"]
        PGSTORE["postgres_store.py"]
        EVALS["eval_store.py"]
        LOGS["log_store.py"]
        ANALYTICS["analytics.py"]
    end

    ORCH --> AGENTS
    ORCH --> STATE
    ORCH --> CFG
    CELERY --> INGEST
    AGENTS --> SERVICES
    SERVICES --> LLM
    SERVICES --> EMBED
    SERVICES --> VEC
    SERVICES --> BM25S
    SERVICES --> CACHE
    SERVICES --> RATE
    SERVICES --> PGSTORE
```

---

## 6. Infrastructure Services (Docker Compose)

| Service | Image | Role |
| --- | --- | --- |
| **API** | `Dockerfile` | FastAPI + Uvicorn (HTTP + WebSocket) |
| **Worker** | `Dockerfile` | Celery worker (async ingest) |
| **Qdrant** | `qdrant/qdrant:v1.12.0` | Dense vector store (`arap_docs`) |
| **Neo4j** | `neo4j:5.25-community` | Knowledge graph (Cypher) |
| **Redis Stack** | `redis/redis-stack:latest` | Checkpointer + broker + cache |
| **PostgreSQL** | `postgres:16-alpine` | Metadata, logs, eval, history |

```mermaid
flowchart LR
    subgraph DOCKER["docker-compose"]
        API_SVC["api"] -->|broker| REDIS_SVC[(redis-stack)]
        WORKER_SVC["worker"] -->|broker| REDIS_SVC
        API_SVC --> QDRANT_SVC[(qdrant)]
        WORKER_SVC --> QDRANT_SVC
        WORKER_SVC --> NEO4J_SVC[(neo4j)]
        API_SVC --> PG_SVC[(postgresql)]
        WORKER_SVC --> PG_SVC
    end
```

---

## 7. End-to-End Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant API as FastAPI
    participant O as Orchestrator
    participant G as QUERY Graph
    participant R as Redis
    participant DB as PostgreSQL

    C->>API: POST /query or WS /ws
    API->>O: query() / stream_query()
    O->>G: invoke(state, thread_id=session_id)
    G->>G: router → retrieve → generate → judge
    G-->>R: checkpoint (session state)
    G-->>O: answer + sources + faithfulness_score
    O->>DB: record_query() (best-effort)
    O-->>API: result
    API-->>C: JSON / stream events

    Note over C,DB: /ingest uses a separate path: API → Redis broker → Celery worker → INGEST Graph → Qdrant/Neo4j/PG
```
