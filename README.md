# Adaptive Research & Analysis Platform (ARAP)

> A production-grade, multi-agent RAG system implementing the 2026 state-of-the-art:
> Adaptive routing · Knowledge graph · Contextual retrieval · Faithfulness judging · Long-term memory

---

## What makes this different from a basic RAG pipeline

| Feature | Basic RAG (2023) | ARAP (2026) |
|---|---|---|
| Retrieval strategy | Fixed vector search | Adaptive routing (4 strategies) |
| Query handling | Raw question → embed | HyDE rewrite + query decomposition |
| Search | Dense-only | BM25 + dense + RRF fusion |
| Re-ranking | None | Cross-encoder (ms-marco-MiniLM) |
| Chunk context | Isolated chunks | Contextual retrieval (Anthropic 2024) |
| Hallucination control | None | NLI faithfulness judge + retry loop |
| Relationship queries | Impossible | Knowledge graph (Neo4j) traversal |
| Memory | Single session | Mem0 long-term memory (episodic + semantic) |
| Delivery | Polling | WebSocket streaming |
| Evaluation | Manual | RAGAS (4 metrics, automated) |
| Observability | Logs | LangSmith traces + spans |

---

## Architecture

```
Client (PDF / question)
        │
        ▼
FastAPI Gateway  ──────────────────────────────────── LangSmith tracing
  /ingest · /query · /ws · /eval
        │
        ▼
Adaptive Query Router  (gpt-4o-mini classifier)
  ┌─────┬───────────┬────────────┬──────────┐
  │     │           │            │          │
direct  single   multi_hop     graph        │
  │     │           │            │          │
  │  Retrieval   Retrieval    Graph         │
  │   Agent      Agent        Agent         │
  │  (hybrid)   (decompose)  (Neo4j)        │
  │     │           │            │          │
  └─────┴───────────┴────────────┘          │
        │  HyDE + BM25 + Qdrant dense       │
        │  RRF fusion + cross-encoder rerank │
        ▼                                    │
   Generator  (gpt-4o, source citations)     │
        │                                    │
        ▼                                    │
Faithfulness Judge  (NLI deberta-v3)         │
   passed? ──no──► retry (max 2x)           │
        │                                    │
        ▼                                    │
  Mem0 Memory Store ◄──────────────────────┘
  (episodic + semantic per user)
        │
        ▼
Streamed response  (WebSocket + JSON)

Storage:  Qdrant (HNSW vectors) · Neo4j (knowledge graph) · Redis (sessions) · PostgreSQL (metadata)
Eval:     RAGAS — faithfulness · answer_relevancy · context_precision · context_recall
```

---

## Key design decisions and their research basis

### Adaptive routing
The 2026 state of the art is Adaptive RAG — a query classifier that routes each question to the appropriate retrieval strategy based on its complexity, making the system economically viable in production. Direct questions skip retrieval entirely. Simple factual questions use single-hop. Complex analytical questions trigger multi-hop decomposition. Entity/relationship questions go to the knowledge graph.

### Contextual retrieval
Each chunk is prepended with a 2-3 sentence LLM-generated context description before embedding, anchoring it in the full document. Published by Anthropic in late 2024 and widely adopted through 2025, this technique delivers 15–25% retrieval recall improvement with minimal latency overhead.

### Faithfulness judging
Production RAG systems require a faithfulness judge gating the final response — without it, the agent can retrieve 8 chunks, use 6 of them, and invent the 7th fact entirely with no span scoring faithfulness. ARAP uses NLI (DeBERTa-v3) to score entailment between each answer sentence and the retrieved context, retrying generation up to 2× if the score falls below threshold.

### Mem0 long-term memory
Mem0 achieves 91% lower p95 latency and saves more than 90% token cost compared to full-context approaches, while outperforming all existing memory systems across single-hop, temporal, multi-hop, and open-domain question categories.

### Knowledge graph
Graph RAG utilizes a knowledge graph to understand how information is structurally connected — enabling relationship-aware knowledge retrieval that pure vector search cannot provide. During ingestion, entity-relation triples are extracted and stored in Neo4j. Graph queries use LLM-generated Cypher.

---

## Quickstart

### 1. Clone and configure

```bash
git clone https://github.com/Hasan26ozcan/arap.git
cd arap
cp .env.example .env
# Fill in: OPENAI_API_KEY, and optionally LANGCHAIN_API_KEY for LangSmith
```

### 2. Start all services (one command)

```bash
docker compose up --build
```

Services started:
- **API** → `http://localhost:8000`
- **Neo4j browser** → `http://localhost:7474`
- **Qdrant dashboard** → `http://localhost:6333/dashboard`
- **API docs** → `http://localhost:8000/docs`

### 3. Upload a document

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@research_paper.pdf"
```

```json
{
  "doc_id": "a3f8b12c9d4e",
  "filename": "research_paper.pdf",
  "chunk_count": 84,
  "kg_triples": 312,
  "message": "Document ingested successfully."
}
```

### 4. Ask a question (REST)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does the proposed method compare to baseline approaches?",
    "user_id": "hasan",
    "doc_id": "a3f8b12c9d4e"
  }'
```

```json
{
  "answer": "The proposed method outperforms all baselines [Source 1]. Specifically, it achieves 23% higher F1 on the benchmark dataset [Source 2]...",
  "sources": [
    {"index": 1, "text": "...chunk text...", "page": 7, "rerank_score": 0.9241},
    {"index": 2, "text": "...chunk text...", "page": 12, "rerank_score": 0.8876}
  ],
  "query_type": "multi_hop",
  "faithfulness_score": 0.91,
  "latency_ms": {"router": 312, "retrieval_multi": 847, "generation": 1240, "judge": 180}
}
```

### 5. Stream via WebSocket

```python
import asyncio, websockets, json

async def stream():
    async with websockets.connect("ws://localhost:8000/ws/my-session") as ws:
        await ws.send(json.dumps({
            "question": "What are the key limitations?",
            "user_id": "hasan",
        }))
        async for msg in ws:
            event = json.loads(msg)
            if event["type"] == "update":
                print(f"[{event['node']}] {list(event['data'].keys())}")
            elif event["type"] == "done":
                print("\nFinal answer:", event["answer"][:200])
                break

asyncio.run(stream())
```

### 6. Run RAGAS evaluation

```bash
curl http://localhost:8000/eval
```

```json
{
  "scores": {
    "faithfulness": 0.912,
    "answer_relevancy": 0.887,
    "context_precision": 0.843,
    "context_recall": 0.791,
    "num_questions": 20
  }
}
```

---

## Tech stack

| Layer | Technology | Version | Why |
|---|---|---|---|
| Orchestration | LangGraph | 0.2.50 | Stateful cyclic agent graphs with conditional edges |
| API | FastAPI | 0.115 | Async, WebSocket support, auto OpenAPI docs |
| LLM | GPT-4o / GPT-4o-mini | latest | Generator + router + judge |
| Embedding | sentence-transformers MiniLM-L6-v2 | 3.2.1 | Fast, normalized 384-dim vectors |
| Vector DB | Qdrant HNSW | 1.12 | HNSW index, cosine similarity, metadata filtering |
| KG | Neo4j | 5.25 | Cypher queries, entity/relation traversal |
| Memory | Mem0 | 0.1.29 | 3-tier memory (episodic, semantic, graph) |
| BM25 | rank-bm25 | 0.2.2 | BM25Okapi keyword search (no extra infra) |
| Re-ranking | CrossEncoder ms-marco-MiniLM-L-6 | 3.2.1 | Local cross-encoder, no API cost |
| Faithfulness | DeBERTa-v3-small NLI | 3.2.1 | NLI entailment scoring, local inference |
| Cache / state | Redis 7.4 | - | Session persistence, LangGraph checkpointer |
| Metadata | PostgreSQL 16 | - | Document registry, query history |
| Task queue | Celery + Redis | 5.4.0 | Async ingestion workers |
| Evaluation | RAGAS | 0.1.21 | 4-metric automated RAG evaluation |
| Observability | LangSmith | 0.1.141 | Trace every node, latency, token cost |

---

## Project structure

```
arap/
├── app/
│   ├── api/
│   │   └── main.py              FastAPI app, all endpoints + WebSocket
│   ├── core/
│   │   ├── config.py            Centralized settings (pydantic-settings)
│   │   ├── state.py             LangGraph AgentState TypedDict
│   │   └── orchestrator.py      Graph assembly + async API
│   ├── agents/
│   │   ├── router.py            Adaptive query router + Mem0 fetch
│   │   ├── retrieval_agent.py   HyDE + hybrid search + rerank
│   │   ├── graph_agent.py       Neo4j entity extraction + Cypher
│   │   └── generator.py        Generator + NLI faithfulness judge + Mem0 store
│   └── services/
│       ├── chunker.py           PyMuPDF chunker with overlap
│       ├── embedder.py          sentence-transformers wrapper
│       ├── contextual_enricher.py  Anthropic contextual retrieval
│       ├── vector_store.py      Qdrant HNSW operations
│       ├── bm25_index.py        BM25 corpus management
│       └── tasks.py             Celery async tasks
├── evaluation/
│   └── ragas_eval.py            RAGAS 4-metric suite
├── tests/
│   ├── unit/                    Component-level pytest tests
│   └── integration/             End-to-end pipeline tests
├── scripts/
│   └── init_db.sql              PostgreSQL schema
├── docker-compose.yml           6-service production setup
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Research references

1. Jeong et al. (2024). *Adaptive RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity*. arXiv:2403.14403
2. Gao et al. (2022). *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE). arXiv:2212.10496
3. Anthropic (2024). *Introducing Contextual Retrieval*. anthropic.com/news/contextual-retrieval
4. Cormack et al. (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*. SIGIR 2009
5. Tang et al. (2024). *MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents*. arXiv:2404.10774
6. Edge et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*. arXiv:2404.16130
7. Chhikara et al. (2025). *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory*. arXiv:2504.19413
8. Es et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. arXiv:2309.15217

---

## License

MIT
