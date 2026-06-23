"""
Knowledge Graph Agent
=====================
Handles "graph" query type: questions requiring entity/relationship traversal.

Pipeline:
  1. Entity & relation extraction from the user query (LLM + NER)
  2. Cypher query generation for Neo4j
  3. Graph traversal — returns entity paths, not just text chunks
  4. Path serialization → context for the generator

During ingestion, this agent also extracts entities from document chunks
and stores them as (head, relation, tail) triples in Neo4j.

Why a knowledge graph?
  Vector search finds semantically similar chunks but misses structural
  relationships: "Who co-authored papers with X?" or "Which companies are
  subsidiaries of Y?" require graph traversal, not cosine similarity.

Reference:
  Microsoft GraphRAG (Edge et al., 2024): graph-based community summarization
  HippoRAG2 (Gutiérrez et al., 2025): phrase-level KG + passage retrieval
  Document GraphRAG (MDPI Electronics, May 2025): doi:10.3390/electronics14112102
"""
from __future__ import annotations
import time
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.state import AgentState
from app.core.config import settings

logger = logging.getLogger(__name__)


EXTRACTION_SYSTEM = """Extract all named entities and their relationships from the text.
Return a JSON array of triples:
[{"head": "<entity>", "relation": "<verb/relation>", "tail": "<entity>", "confidence": <0-1>}]
Include only concrete, factual relationships. Confidence below 0.5 → omit.
Return ONLY the JSON array, no other text."""

CYPHER_SYSTEM = """You are a Neo4j Cypher expert.
Given a natural language question and a list of entities found in the query,
generate a Cypher query to answer it.

Schema: Nodes have label :Entity with property {name, type, doc_id}.
        Edges have type :RELATES_TO with property {relation, confidence, doc_id}.

Rules:
- Use MATCH and OPTIONAL MATCH only (no DELETE/CREATE/MERGE)
- Always LIMIT results to 20
- Return node names and relationship types

Return ONLY the Cypher query."""


class KnowledgeGraphAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.router_model,
            api_key=settings.openai_api_key,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        self.cypher_llm = ChatOpenAI(
            model=settings.router_model,
            api_key=settings.openai_api_key,
            temperature=0.0,
        )
        self._driver = None

    @property
    def driver(self):
        if self._driver is None:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
        return self._driver

    # ── Query node ─────────────────────────────────────────────────────────────

    def graph_retrieve(self, state: AgentState) -> AgentState:
        """LangGraph node: extract entities → generate Cypher → traverse → return paths."""
        t0 = time.perf_counter()
        question = state["question"]

        # Entity extraction from question
        entities = self._extract_query_entities(question)
        logger.info("KG agent: extracted %d entities from query", len(entities))

        # Generate Cypher
        cypher = self._generate_cypher(question, entities)
        logger.info("KG agent: cypher=%s", cypher)

        # Execute traversal
        paths = self._run_cypher(cypher, doc_id=state.get("doc_id"))

        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "kg_paths": paths,
            "latency_ms": {**(state.get("latency_ms") or {}), "graph": elapsed},
        }

    # ── Ingestion: entity extraction from document chunks ──────────────────────

    def extract_and_store(self, chunks: list[dict], doc_id: str) -> int:
        """
        Called during document ingestion.
        Extracts triples from each chunk and writes them to Neo4j.
        Returns count of stored triples.
        """
        stored = 0
        for chunk in chunks:
            triples = self._extract_triples(chunk["text"])
            for triple in triples:
                if triple.get("confidence", 0) >= 0.5:
                    self._upsert_triple(triple, doc_id=doc_id, source_page=chunk.get("page"))
                    stored += 1
        logger.info("KG: stored %d triples for doc_id=%s", stored, doc_id)
        return stored

    # ── Private helpers ────────────────────────────────────────────────────────

    def _extract_triples(self, text: str) -> list[dict]:
        import json
        try:
            resp = self.llm.invoke([
                SystemMessage(content=EXTRACTION_SYSTEM),
                HumanMessage(content=text[:2000]),
            ])
            raw = resp.content.strip()
            data = json.loads(raw)
            if isinstance(data, list):
                return data
            return data.get("triples", [])
        except Exception as e:
            logger.warning("Triple extraction failed: %s", e)
            return []

    def _extract_query_entities(self, question: str) -> list[str]:
        """Simple NER: extract entity names from the user query."""
        system = (
            "Extract all named entities (people, organizations, products, concepts) "
            "from the question. Return a JSON array of strings. "
            'Example: {"entities": ["BERT", "Google", "2024"]}.'
        )
        import json
        try:
            resp = self.llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=question),
            ])
            data = json.loads(resp.content)
            return data.get("entities", [])
        except Exception:
            return []

    def _generate_cypher(self, question: str, entities: list[str]) -> str:
        entity_list = ", ".join(f'"{e}"' for e in entities[:8])
        prompt = (
            f"Question: {question}\n"
            f"Entities found in question: [{entity_list}]\n"
            "Generate a Cypher query to answer this question."
        )
        resp = self.cypher_llm.invoke([
            SystemMessage(content=CYPHER_SYSTEM),
            HumanMessage(content=prompt),
        ])
        return resp.content.strip().strip("```cypher").strip("```").strip()

    def _run_cypher(self, cypher: str, doc_id: str | None) -> list[dict]:
        """Execute Cypher and return serialized paths."""
        try:
            params = {"doc_id": doc_id} if doc_id else {}
            with self.driver.session() as session:
                result = session.run(cypher, **params)
                paths = []
                for record in result:
                    paths.append({k: str(v) for k, v in record.items()})
                return paths[:20]
        except Exception as e:
            logger.error("Neo4j query failed: %s | cypher=%s", e, cypher)
            return []

    def _upsert_triple(self, triple: dict, doc_id: str, source_page: int | None):
        """Write a single (head, relation, tail) triple to Neo4j."""
        cypher = """
        MERGE (h:Entity {name: $head, doc_id: $doc_id})
        MERGE (t:Entity {name: $tail, doc_id: $doc_id})
        MERGE (h)-[r:RELATES_TO {relation: $relation, doc_id: $doc_id}]->(t)
        ON CREATE SET r.confidence = $confidence, r.page = $page
        ON MATCH  SET r.confidence = CASE WHEN r.confidence < $confidence
                                          THEN $confidence ELSE r.confidence END
        """
        try:
            with self.driver.session() as session:
                session.run(cypher, {
                    "head": triple["head"],
                    "tail": triple["tail"],
                    "relation": triple.get("relation", "related_to"),
                    "confidence": triple.get("confidence", 0.5),
                    "doc_id": doc_id,
                    "page": source_page,
                })
        except Exception as e:
            logger.warning("Neo4j upsert failed: %s", e)


kg_agent = KnowledgeGraphAgent()
