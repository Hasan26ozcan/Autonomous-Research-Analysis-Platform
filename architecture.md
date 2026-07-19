# ARAP — Mimari Diyagramlar (Architecture)

Bu dosya **Adaptive Research & Analysis Platform (ARAP)**'in mimarisini
Mermaid diyagramları ile açıklar. Diyagramlar `app/core/orchestrator.py`,
`app/api/main.py` ve `app/agents/*` modüllerindeki gerçek kod topolojisine
dayanmaktadır. GitHub ve VS Code (Mermaid uzantısı) bu dosyayı doğrudan render eder.

---

## 1. Sistem Genel Bakışı (System Overview)

Kullanıcı/istemci istekleri FastAPI'ye gelir. `/ingest` zaman alan PDF
işlemeyi **Celery worker**'a (Redis broker) devreder; `/query` ve `/ws` ise
her istekte derlenen **QUERY LangGraph**'ını çalıştırır. Tüm kalıcı ve
geçici durum beş altyapı servisi üzerinde tutulur.

```mermaid
flowchart TB
    Client["İstemci<br/>(Tarayıcı / API client)"]

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

    subgraph INFRA["Altyapı Servisleri"]
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

## 2. INGEST Grafiği (Belge İşleme — bir kez PDF başına)

`app/core/orchestrator.py → build_ingest_graph()`. Doğrusal, dalsız boru
hattı. **KG çıkarımı sona konur** ki belge Qdrant/BM25 üzerinden hâlâ
aranabilir olsun (graceful degradation). Checkpointer yoktur (stateless).

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

## 3. QUERY Grafiği (Sorgu — istek başına, adaptif)

`app/core/orchestrator.py → build_query_graph()`. İki koşullu kenar vardır:
(1) `router → get_route()` 4 stratejiye dallanır; (2) `judge → should_retry()`
retrieve/üretim döngüsü. Tek döngü (retry loop) LangGraph'ın döngü desteğiyle
sağlanır. Redis checkpointer `session_id` ile çoklu-tur hafızayı korur.

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

> **Adaptif yönlendirme stratejileri** (`app/agents/router.py`):
> `direct` (parametrik bilgi, retrieve yok) · `single` (tek hibrit tur) ·
> `multi_hop` (alt sorulara böl + birleştir) · `graph` (Neo4j Cypher).
> Hata durumunda `single`'a düşer.

---

## 4. Retrieval Stack (4 Katman)

`app/agents/retrieval_agent.py` içinde hem `retrieve` hem `retrieve_multi`
tarafından kullanılır.

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

> Sonuçlar Redis'te `(question, doc_id)` anahtarıyla cache'lenir.

---

## 5. Servis Katmanı ve Bağımlılıklar

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

## 6. Altyapı Servisleri (Docker Compose)

| Servis | Görüntü | Rol |
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

## 7. Akış Özeti (End-to-End)

```mermaid
sequenceDiagram
    participant C as İstemci
    participant API as FastAPI
    participant O as Orchestrator
    participant G as QUERY Graph
    participant R as Redis
    participant DB as PostgreSQL

    C->>API: POST /query veya WS /ws
    API->>O: query() / stream_query()
    O->>G: invoke(state, thread_id=session_id)
    G->>G: router → retrieve → generate → judge
    G-->>R: checkpoint (session state)
    G-->>O: answer + sources + faithfulness_score
    O->>DB: record_query() (best-effort)
    O-->>API: result
    API-->>C: JSON / stream events

    Note over C,DB: /ingest ayrı yolda: API → Redis broker → Celery worker → INGEST Graph → Qdrant/Neo4j/PG
```
