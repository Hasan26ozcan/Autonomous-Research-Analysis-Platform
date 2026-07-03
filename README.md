# Adaptive Research & Analysis Platform (ARAP)

ARAP is a production-style, multi-agent retrieval augmented generation system for answering questions over PDF documents and research material. It combines adaptive routing, hybrid retrieval, contextual enrichment, graph-based retrieval, faithfulness judging, and memory-aware generation in a single workflow.

## What ARAP does

ARAP is designed for document-heavy workflows where a simple vector search is not enough. The pipeline can:

- ingest PDF documents and prepare them for retrieval,
- route each question to the right retrieval strategy,
- retrieve relevant evidence with BM25 + dense vector search,
- rerank the most useful chunks,
- optionally use knowledge graph traversal for entity-relationship questions,
- generate grounded answers with source attribution,
- judge answer faithfulness and retry when needed,
- stream progress and final results over WebSocket.

## Core architecture

The project is organized around a FastAPI entrypoint and a LangGraph-based orchestration layer.

- FastAPI exposes the HTTP and WebSocket interface.
- LangGraph composes the document ingestion and query execution pipelines.
- Agents handle routing, retrieval, graph querying, and answer generation.
- Services manage chunking, embedding, vector storage, BM25 indexing, and enrichment.

## Main features

- Adaptive query routing with four strategies: direct, single, multi-hop, and graph.
- Hybrid retrieval using BM25, dense embeddings, Reciprocal Rank Fusion, and reranking.
- Contextual chunk enrichment for better retrieval recall.
- Knowledge graph retrieval with Neo4j for relationship-aware questions.
- Faithfulness judging with an NLI-based scorer and retry loop.
- Long-term memory support via Mem0.
- WebSocket streaming for interactive responses.
- Evaluation support through RAGAS-based metrics.

## Quickstart

### 1. Prerequisites

- Python 3.10+
- Docker Desktop / Docker Compose
- An OpenAI API key

### 2. Clone and configure

```bash
git clone <your-repo-url>
cd Autonomous-Research-Analysis-Platform
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local environment file:

```bash
copy .env.example .env
```

Then fill in the required values, especially `OPENAI_API_KEY`.

### 3. Start the supporting services

```bash
docker compose up -d qdrant neo4j redis postgres
```

### 4. Run the API

```bash
uvicorn app.api.main:app --reload
```

You can then open:

- http://localhost:8000/health
- http://localhost:8000/docs

### 5. Ingest a document

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@research_paper.pdf"
```

### 6. Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main contribution of this document?",
    "user_id": "demo-user"
  }'
```

## Project structure

```text
ARAP/
├── app/
│   ├── agents/
│   │   ├── generator.py
│   │   ├── graph_agent.py
│   │   ├── retrieval_agent.py
│   │   └── router.py
│   ├── api/
│   │   └── main.py
│   ├── core/
│   │   ├── config.py
│   │   ├── orchestrator.py
│   │   └── state.py
│   └── services/
│       ├── bm25_index.py
│       ├── chunker.py
│       ├── contextual_enricher.py
│       ├── embedder.py
│       └── vector_store.py
├── docs/
│   └── phase1_setup.py
├── evaluation/
│   └── ragas_eval.py
├── scripts/
│   └── init_db.sql
├── tests/
│   ├── integration/
│   └── unit/
├── Dockerfile
├── docker-compose.yml
├── README.md
├── requirements.txt
└── .env.example
```

## Development and testing

Run unit tests:

```bash
pytest tests/unit -q
```

Run integration tests:

```bash
pytest tests/integration -q
```

Run the evaluation suite:

```bash
python -m evaluation.ragas_eval --limit 20
```

## Environment variables

The application uses `pydantic-settings` and reads values from `.env` or environment variables. The most important ones are:

- `OPENAI_API_KEY`
- `QDRANT_HOST` / `QDRANT_PORT`
- `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`
- `REDIS_URL`
- `POSTGRES_URL`
- `LANGCHAIN_API_KEY` (optional)
- `MEM0_API_KEY` (optional)

## Notes

This repository is structured as a modular RAG platform rather than a single-file demo. The intent is to make each stage of the pipeline explicit and easy to extend.

## License

MIT
