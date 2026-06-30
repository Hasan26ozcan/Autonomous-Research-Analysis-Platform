"""
app/agents/graph_agent.py
===========================
Knowledge Graph Agent — entity/relationship extraction and Neo4j Cypher traversal.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY VECTOR SEARCH CANNOT ANSWER EVERYTHING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dense + BM25 hybrid search (Phase 5) finds chunks that are TEXTUALLY similar
to a question. This works for the vast majority of questions. But consider:

  "Which authors appear in papers cited by both chapter 2 and chapter 5?"

No single chunk contains the answer. The information is scattered:
  - Chapter 2's references list mentions authors A, B, C
  - Chapter 5's references list mentions authors B, D, E
  - The answer (author B) requires INTERSECTING two separate entity sets

This is a structural relationship query, not a semantic similarity query.
Vector search has no notion of "intersection" or "co-occurrence across
distant parts of a document." A knowledge graph does — that's its entire
purpose: a graph datastore where relationships ARE the primary data
structure, not a side effect of text proximity.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TWO RESPONSIBILITIES OF THIS AGENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. INGESTION TIME — extract_and_store_node()
   For every chunk produced during /ingest, extract (head, relation, tail)
   triples using an LLM, then write them to Neo4j as a property graph:
     (Entity {name: head})-[:RELATES_TO {relation, confidence}]->(Entity {name: tail})
   This runs ONCE per document, after chunking/enrichment/embedding/storage.

2. QUERY TIME — graph_retrieve()
   When the router classifies a question as query_type="graph":
     a. Extract entity names mentioned in the QUESTION (lightweight NER)
     b. Generate a Cypher query using those entities as anchors
     c. Execute the Cypher query against Neo4j
     d. Serialize results into kg_paths for the generator

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PIPELINE POSITION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ingest graph (see orchestrator.build_ingest_graph):
    chunk_document → enrich_chunks → embed_chunks → store_chunks
        → index_chunks → extract_and_store_node (THIS FILE) → END

Query graph (see orchestrator.build_query_graph), graph branch only:
    router → [query_type == "graph"] → graph_retrieve (THIS FILE)
        → merge → generate → judge → memory_store → END

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SAFETY: LLM-GENERATED CYPHER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Letting an LLM write Cypher that executes directly against the database is
a real injection risk if not constrained. We mitigate this with THREE layers:

  Layer 1 — Prompt constraints: the system prompt explicitly forbids
            CREATE, DELETE, MERGE, SET, REMOVE, DETACH — read-only only.
  Layer 2 — Static validation: _validate_cypher() rejects any query
            containing write keywords BEFORE execution, regardless of
            what the prompt said (defense in depth — prompts can be bypassed).
  Layer 3 — Read-only Neo4j session: we explicitly request a READ
            transaction via session.execute_read(), which Neo4j enforces
            at the driver level — write operations raise an exception.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STATE INTEGRATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

graph_retrieve() reads from AgentState:
    question   (str):       raw user question
    doc_id     (str|None):  optional document scope filter
    latency_ms (dict):      accumulated per-node timing

graph_retrieve() writes to AgentState (partial update):
    kg_paths   (list[dict]): traversal results, each a flat dict of
                              Cypher RETURN column → string value
    latency_ms (dict):       previous dict + {"graph": <ms>}

extract_and_store_node() reads from AgentState:
    chunks     (list[dict]): enriched chunks from the ingest pipeline
    doc_id     (str):        document identifier (set by chunk_document)

extract_and_store_node() writes to AgentState (partial update):
    kg_entities (list[dict]): all triples extracted across all chunks
                               [{"head", "relation", "tail", "confidence"}, ...]
                               Used by orchestrator.ingest() to report
                               kg_triples count in the /ingest API response.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.core.config import settings

if TYPE_CHECKING:
    from app.core.state import AgentState

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROMPTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TRIPLE_EXTRACTION_SYSTEM = """\
You are an information extraction assistant building a knowledge graph.

Your task: read the given text and extract factual relationships as triples.
A triple is (head_entity, relation, tail_entity) — a directed, labeled edge
between two named entities.

RULES:
  1. Only extract CONCRETE, FACTUAL relationships explicitly stated in the text.
  2. Entities must be proper nouns: people, organizations, products, datasets,
     locations, named methods/models, or specific dates/versions.
     Do NOT extract generic nouns like "the model" or "the result" as entities.
  3. Relations should be short verb phrases: "developed_by", "cites",
     "uses_dataset", "achieved", "located_in", "collaborated_with", etc.
  4. Assign a confidence score 0.0-1.0: how certain are you this relationship
     is explicitly stated (not inferred)? Only include triples with confidence >= 0.5.
  5. Extract AT MOST 8 triples per text — prioritize the most important ones.
  6. If no clear entity relationships exist, return an empty array.

OUTPUT FORMAT — return ONLY a JSON object with this exact shape:
{
  "triples": [
    {"head": "<entity>", "relation": "<relation>", "tail": "<entity>", "confidence": <float>}
  ]
}

Example:
Text: "FloodNet was developed by researchers at MIT and uses the ERA5 dataset
       for training. It outperforms the baseline CNN model by 23%."
Output:
{
  "triples": [
    {"head": "FloodNet", "relation": "developed_by", "tail": "MIT", "confidence": 0.95},
    {"head": "FloodNet", "relation": "uses_dataset", "tail": "ERA5", "confidence": 0.9},
    {"head": "FloodNet", "relation": "outperforms", "tail": "baseline CNN model", "confidence": 0.85}
  ]
}\
"""

QUERY_ENTITY_EXTRACTION_SYSTEM = """\
You are a named entity recognizer for a knowledge graph query system.

Extract all named entities (people, organizations, products, datasets,
locations, methods, models) mentioned in the user's question.

Return ONLY a JSON object: {"entities": ["entity1", "entity2", ...]}
If no named entities are present, return {"entities": []}.
Do not include generic terms like "the paper" or "the method" unless
they are part of a proper name.\
"""

CYPHER_GENERATION_SYSTEM = """\
You are a Neo4j Cypher query generator for a READ-ONLY knowledge graph API.

Graph schema:
  Nodes:  (:Entity {name: string, doc_id: string})
  Edges:  (:Entity)-[:RELATES_TO {relation: string, confidence: float,
                                  doc_id: string, page: integer}]->(:Entity)

Your task: given a natural language question and the named entities found
in it, write a Cypher query that retrieves the relevant graph data to
answer the question.

ABSOLUTE RULES (violating these will cause the query to be rejected):
  1. READ-ONLY. Never use CREATE, MERGE, SET, DELETE, REMOVE, DETACH, DROP.
  2. Only use MATCH and OPTIONAL MATCH for graph traversal.
  3. Always include a LIMIT clause, maximum 20.
  4. If a doc_id parameter is provided, filter all nodes/edges by it using
     $doc_id (already bound — just reference it in your WHERE clause).
  5. Return entity names and relationship types as columns, not full nodes.
  6. Use case-insensitive matching: WHERE toLower(e.name) CONTAINS toLower($entity_name)
     when matching against the entities found in the question (use multiple
     WHERE clauses with OR if multiple entities were extracted).

Return ONLY the raw Cypher query text. No markdown, no backticks, no explanation.

Example:
Question: "Which organizations is FloodNet associated with?"
Entities: ["FloodNet"]
Output:
MATCH (h:Entity)-[r:RELATES_TO]->(t:Entity)
WHERE toLower(h.name) CONTAINS toLower('FloodNet')
RETURN h.name AS head, r.relation AS relation, t.name AS tail, r.confidence AS confidence
LIMIT 20\
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRUCTURED OUTPUT MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Triple(BaseModel):
    """
    A single (head, relation, tail) fact extracted from document text.

    Pydantic validation ensures:
      - confidence is always in [0.0, 1.0]
      - head/tail/relation are non-empty strings (entities can't be blank)
    """
    head: str = Field(min_length=1, description="Subject entity name")
    relation: str = Field(min_length=1, description="Relationship label")
    tail: str = Field(min_length=1, description="Object entity name")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")

    @field_validator("head", "relation", "tail")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class TripleExtractionResult(BaseModel):
    """Validated LLM output for triple extraction from a chunk."""
    triples: list[Triple] = Field(default_factory=list)


class EntityExtractionResult(BaseModel):
    """Validated LLM output for entity extraction from a question."""
    entities: list[str] = Field(default_factory=list)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CYPHER SAFETY VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Keywords that indicate a WRITE operation. Case-insensitive match.
# This is layer 2 of our 3-layer defense (see file header).
_FORBIDDEN_CYPHER_KEYWORDS = (
    "CREATE", "MERGE", "DELETE", "SET", "REMOVE",
    "DETACH", "DROP", "CALL DB.", "LOAD CSV",
)


def _validate_cypher(cypher: str) -> tuple[bool, str]:
    """
    Static safety check on LLM-generated Cypher BEFORE execution.

    This is independent of what the system prompt instructed — we never
    trust the LLM's compliance alone. Defense in depth: even if a prompt
    injection or model error produces a write query, this check blocks it.

    Args:
        cypher: The raw Cypher query string to validate.

    Returns:
        (is_safe, reason) tuple.
        is_safe=True  → query may proceed to execution
        is_safe=False → reason explains why it was rejected
    """
    if not cypher or not cypher.strip():
        return False, "Empty Cypher query"

    upper_cypher = cypher.upper()

    for keyword in _FORBIDDEN_CYPHER_KEYWORDS:
        # Word boundary check to avoid false positives like "Setting" containing "SET"
        if re.search(rf"\b{re.escape(keyword)}\b", upper_cypher):
            return False, f"Forbidden write keyword detected: {keyword}"

    if "LIMIT" not in upper_cypher:
        return False, "Query missing required LIMIT clause"

    if not ("MATCH" in upper_cypher):
        return False, "Query must contain at least one MATCH clause"

    return True, "OK"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KNOWLEDGE GRAPH AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class KnowledgeGraphAgent:
    """
    Manages both ingestion-time triple extraction and query-time graph traversal.

    LLM choice: router_model (gpt-4o-mini) for ALL three LLM calls
    (triple extraction, query entity extraction, Cypher generation).
    Consistent with Phase 4 (router) and Phase 5 (retrieval) — these are
    structured extraction/generation tasks, not open-ended reasoning,
    so the cheaper model is sufficient and keeps ingestion cost low.

    Neo4j connection: lazy-initialized, same pattern as VectorStore (Phase 2)
    and BM25Index — prevents startup crashes if Neo4j isn't ready yet
    (relevant during docker-compose cold starts where services race to boot).
    """

    def __init__(self):
        # response_format json_object forces valid JSON, same pattern as router.py
        self.extraction_llm = ChatOpenAI(
            model=settings.router_model,
            api_key=settings.openai_api_key,
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=600,    # up to 8 triples in JSON needs more room than routing
        )
        self.cypher_llm = ChatOpenAI(
            model=settings.router_model,
            api_key=settings.openai_api_key,
            temperature=0.0,
            max_tokens=250,    # Cypher queries are short
            # NOTE: no response_format here — Cypher is not JSON, it's a query string
        )
        self._driver = None   # lazy Neo4j driver

    # ── Lazy Neo4j Driver ─────────────────────────────────────────────────────

    @property
    def driver(self):
        """
        Lazy Neo4j driver initialization.

        Why lazy?
          Same rationale as VectorStore.client (Phase 2): importing this
          module must not crash if Neo4j isn't running yet. The driver
          connects on first actual graph operation, not at import time.

        verify_connectivity() is called once on first access to fail fast
        with a clear error message rather than a cryptic timeout deep inside
        a Cypher execution call.
        """
        if self._driver is None:
            from neo4j import GraphDatabase
            logger.info("Connecting to Neo4j at %s", settings.neo4j_uri)
            self._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            try:
                self._driver.verify_connectivity()
                logger.info("Neo4j connection verified.")
            except Exception as e:
                logger.error("Neo4j connectivity check failed: %s", e)
                raise
        return self._driver

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LangGraph Node (INGEST GRAPH): extract_and_store_node()
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def extract_and_store_node(self, state: "AgentState") -> dict:
        """
        LangGraph node for the INGEST pipeline (runs once per document).

        Registered in orchestrator.build_ingest_graph() as the final step:
            chunk_document → enrich_chunks → embed_chunks → store_chunks
                → index_chunks → extract_and_store_node (THIS) → END

        For every chunk in state["chunks"], extract entity/relation triples
        via LLM and write them to Neo4j as a property graph. Triples are
        deduplicated and accumulated across all chunks before being written
        in a single batch transaction for efficiency.

        Why run AFTER store_chunks/index_chunks (not in parallel)?
          KG extraction is the slowest ingestion step (one LLM call per chunk).
          Running it last means a document is fully searchable (Qdrant + BM25)
          even if KG extraction is still in progress or partially fails —
          graceful degradation: the document remains queryable via vector/BM25
          search even without complete KG coverage.

        Reads from AgentState:
            chunks (list[dict]): enriched chunks, each with "text" and "doc_id"
            doc_id (str):        document identifier (set by chunk_document)

        Writes to AgentState (partial update):
            kg_entities (list[dict]): all triples extracted, as plain dicts
                                       [{"head", "relation", "tail", "confidence"}, ...]
                                       orchestrator.ingest() reports len() of this
                                       as "kg_triples" in the API response.

        Failure handling:
          Per-chunk extraction failures are logged and skipped — one bad
          chunk does not abort extraction for the rest of the document.
          A total Neo4j connectivity failure degrades gracefully: kg_entities
          returns [] and the document is still usable for vector/BM25 retrieval.
        """
        t0 = time.perf_counter()

        chunks: list[dict] = state.get("chunks", [])
        doc_id: str = state.get("doc_id", "")

        if not chunks:
            logger.warning("extract_and_store_node: no chunks in state — skipping")
            return {"kg_entities": []}

        # Step 1: Extract triples from every chunk (LLM calls)
        all_triples: list[Triple] = []
        chunks_with_triples = 0

        for chunk in chunks:
            # Use original_text if available (post Phase-3 enrichment) to avoid
            # extracting entities from our own "[Context: ...]" wrapper text.
            text_for_extraction = chunk.get("original_text") or chunk.get("text", "")
            page = chunk.get("page")

            triples = self._extract_triples_from_text(text_for_extraction)
            if triples:
                chunks_with_triples += 1
                all_triples.extend(triples)

        if not all_triples:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "KG extraction: 0 triples found across %d chunks (%.0fms)",
                len(chunks), elapsed_ms,
            )
            return {"kg_entities": []}

        # Step 2: Write all triples to Neo4j in a single batch (efficient,
        # one transaction instead of N round-trips)
        written_count = self._batch_upsert_triples(all_triples, doc_id=doc_id, chunks=chunks)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "KG extraction complete: %d triples from %d/%d chunks, "
            "%d written to Neo4j, doc_id=%s (%.0fms)",
            len(all_triples), chunks_with_triples, len(chunks),
            written_count, doc_id, elapsed_ms,
        )

        return {
            "kg_entities": [t.model_dump() for t in all_triples],
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LangGraph Node (QUERY GRAPH): graph_retrieve()
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def graph_retrieve(self, state: "AgentState") -> dict:
        """
        LangGraph node for the QUERY pipeline, "graph" route only.

        Called when router_agent.route() classified query_type="graph"
        (see Phase 4: app/agents/router.py).

        Three-step pipeline:
          1. Extract named entities mentioned in the question (lightweight NER)
          2. Generate a read-only Cypher query anchored on those entities
          3. Execute the query (with safety validation) and serialize results

        Reads from AgentState:
            question   (str):       the user's relationship/entity question
            doc_id     (str|None):  optional document scope filter
            latency_ms (dict):      accumulated per-node timing

        Writes to AgentState (partial update):
            kg_paths   (list[dict]): traversal results, each row is a flat
                                      dict of Cypher RETURN columns → values,
                                      all stringified for JSON/state safety
            latency_ms (dict):       previous dict + {"graph": <ms>}

        Failure handling:
          If entity extraction fails → empty entity list → Cypher generation
          still attempted (whole-graph query with just the question text as
          context). If Cypher validation fails → kg_paths=[] returned, the
          generator will see no graph evidence and can fall back to saying
          "no relationship information found" rather than crashing.
        """
        t0 = time.perf_counter()

        question: str = state.get("question", "")
        doc_id: str | None = state.get("doc_id")

        if not question:
            logger.error("graph_retrieve() called with empty question")
            return self._empty_update(state, t0)

        # Step 1: Extract entities from the question
        entities = self._extract_query_entities(question)
        logger.info(
            "Graph query: extracted %d entities %s from '%s...'",
            len(entities), entities, question[:50],
        )

        # Step 2: Generate Cypher
        cypher = self._generate_cypher(question, entities)

        # Step 3: Validate + execute
        is_safe, reason = _validate_cypher(cypher)
        if not is_safe:
            logger.error(
                "Graph query: generated Cypher REJECTED (%s). Query was: %s",
                reason, cypher[:200],
            )
            return self._empty_update(state, t0)

        paths = self._execute_cypher(cypher, doc_id=doc_id)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "graph_retrieve(): %d paths found in %.0fms",
            len(paths), elapsed_ms,
        )

        return {
            "kg_paths": paths,
            "latency_ms": {
                **(state.get("latency_ms") or {}),
                "graph": round(elapsed_ms, 2),
            },
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Private: Triple Extraction (Ingestion)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(Exception),
        reraise=False,   # caller handles None/failure gracefully
    )
    def _call_extraction_llm(self, text: str) -> str:
        """
        Isolated LLM call for triple extraction. Retries on transient errors.
        Truncates input to 1500 words to control token cost and stay within
        the model's effective attention range for accurate extraction.
        """
        truncated = " ".join(text.split()[:1500])
        response = self.extraction_llm.invoke([
            SystemMessage(content=TRIPLE_EXTRACTION_SYSTEM),
            HumanMessage(content=f"Text:\n{truncated}"),
        ])
        return response.content

    def _extract_triples_from_text(self, text: str) -> list[Triple]:
        """
        Extract validated Triple objects from one chunk of text.

        On any failure (LLM error, JSON parse error, validation error),
        logs a warning and returns an empty list — extraction for this
        chunk is skipped, but the rest of the document continues processing.
        This graceful degradation is essential: KG extraction must never
        block document ingestion (vector + BM25 indexing already succeeded
        by the time this node runs).

        Args:
            text: Chunk text to extract relationships from.

        Returns:
            List of validated Triple objects (confidence >= 0.5, enforced
            both by the prompt instruction and downstream by Pydantic's
            implicit acceptance of whatever the LLM returns — we do NOT
            re-filter by confidence here since the prompt already filters;
            see TRIPLE_EXTRACTION_SYSTEM rule 4).
        """
        if not text or len(text.split()) < 10:
            return []   # too short to contain meaningful relationships

        try:
            raw_json = self._call_extraction_llm(text)
            parsed = json.loads(raw_json)
            result = TripleExtractionResult(**parsed)
            return result.triples
        except json.JSONDecodeError as e:
            logger.warning("KG extraction: JSON parse error (skipping chunk): %s", e)
            return []
        except Exception as e:
            logger.warning("KG extraction: failed for chunk (skipping): %s", str(e)[:120])
            return []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Private: Neo4j Batch Write
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _batch_upsert_triples(
        self,
        triples: list[Triple],
        doc_id: str,
        chunks: list[dict],
    ) -> int:
        """
        Write all triples to Neo4j in a single batch transaction using UNWIND.

        Cypher pattern (idempotent — safe to re-run on re-ingestion):
            UNWIND $triples AS triple
            MERGE (h:Entity {name: triple.head, doc_id: $doc_id})
            MERGE (t:Entity {name: triple.tail, doc_id: $doc_id})
            MERGE (h)-[r:RELATES_TO {relation: triple.relation, doc_id: $doc_id}]->(t)
            ON CREATE SET r.confidence = triple.confidence
            ON MATCH  SET r.confidence = CASE WHEN r.confidence < triple.confidence
                                              THEN triple.confidence ELSE r.confidence END

        Why MERGE instead of CREATE?
          MERGE is idempotent — if the same (head, relation, tail) triple is
          extracted again (e.g. document re-ingested, or the same fact stated
          in two chunks), MERGE updates the existing edge instead of creating
          a duplicate. The ON MATCH clause keeps the HIGHER confidence score
          if the same relationship is found multiple times — later extractions
          don't accidentally downgrade a high-confidence existing edge.

        Why UNWIND for batching?
          A single UNWIND query with all triples is dramatically faster than
          N individual MERGE queries (one Neo4j round-trip instead of N).
          For a 200-chunk document with ~3 triples/chunk, this is 600
          potential round-trips collapsed into 1.

        Page attribution:
          We don't have an easy way to know exactly which chunk produced
          which triple after the fact (triples are accumulated across all
          chunks before this method runs). For simplicity and correctness,
          we omit per-triple page numbers here. If precise page attribution
          is needed, _extract_triples_from_text could be modified to tag
          each Triple with its source page at extraction time.

        Args:
            triples: All validated Triple objects to write.
            doc_id:  Document identifier — every node/edge is scoped to this
                     doc_id, enabling per-document filtering at query time.
            chunks:  Unused in this implementation (reserved for future page
                     attribution enhancement) — kept in signature for API
                     stability and to document the design intent above.

        Returns:
            Number of triples submitted for writing. Note: this is the
            INPUT count, not necessarily the count of NEW edges created
            (MERGE may update existing edges rather than create new ones).
        """
        if not triples:
            return 0

        cypher = """
        UNWIND $triples AS triple
        MERGE (h:Entity {name: triple.head, doc_id: $doc_id})
        MERGE (t:Entity {name: triple.tail, doc_id: $doc_id})
        MERGE (h)-[r:RELATES_TO {relation: triple.relation, doc_id: $doc_id}]->(t)
        ON CREATE SET r.confidence = triple.confidence
        ON MATCH SET r.confidence = CASE
            WHEN r.confidence < triple.confidence THEN triple.confidence
            ELSE r.confidence
        END
        """

        triples_payload = [t.model_dump() for t in triples]

        try:
            with self.driver.session() as session:
                session.execute_write(
                    lambda tx: tx.run(cypher, triples=triples_payload, doc_id=doc_id)
                )
            logger.info(
                "Neo4j: batch-wrote %d triples for doc_id=%s", len(triples), doc_id
            )
            return len(triples)
        except Exception as e:
            logger.error(
                "Neo4j: batch write failed for doc_id=%s (KG storage skipped, "
                "document remains searchable via vector/BM25): %s",
                doc_id, str(e)[:150],
            )
            return 0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Private: Query-Time Entity Extraction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _extract_query_entities(self, question: str) -> list[str]:
        """
        Lightweight NER over the user's question — extract entity names
        to anchor the Cypher query (e.g. "Which papers does X cite?" → ["X"]).

        Separate, simpler prompt from triple extraction (QUERY_ENTITY_EXTRACTION_SYSTEM)
        because this task is single-purpose: list entities, not extract relationships.

        On failure, returns [] — Cypher generation can still proceed with
        a question-only Cypher query (less precise, but not blocked).

        Args:
            question: The raw user question.

        Returns:
            List of entity name strings, e.g. ["FloodNet", "MIT", "ERA5"].
        """
        try:
            response = self.extraction_llm.invoke([
                SystemMessage(content=QUERY_ENTITY_EXTRACTION_SYSTEM),
                HumanMessage(content=f"Question: {question}"),
            ])
            parsed = json.loads(response.content)
            result = EntityExtractionResult(**parsed)
            return result.entities
        except Exception as e:
            logger.warning(
                "Query entity extraction failed (proceeding with empty list): %s",
                str(e)[:100],
            )
            return []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Private: Cypher Generation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _generate_cypher(self, question: str, entities: list[str]) -> str:
        """
        Generate a read-only Cypher query using the LLM, anchored on
        entities extracted from the question.

        The generated query is NOT trusted blindly — _validate_cypher()
        runs on the output before execution (see graph_retrieve()).

        Args:
            question: The user's original question.
            entities: Entity names extracted via _extract_query_entities().

        Returns:
            Raw Cypher query string. May be empty or invalid — caller
            MUST validate before executing.
        """
        entity_list_str = ", ".join(f'"{e}"' for e in entities[:8]) if entities else "(none found)"
        user_prompt = (
            f"Question: {question}\n"
            f"Entities found in question: [{entity_list_str}]\n\n"
            "Generate the Cypher query."
        )

        try:
            response = self.cypher_llm.invoke([
                SystemMessage(content=CYPHER_GENERATION_SYSTEM),
                HumanMessage(content=user_prompt),
            ])
            # Strip potential markdown fences the model might add despite instructions
            cypher = response.content.strip()
            cypher = re.sub(r"^```(?:cypher)?\s*", "", cypher)
            cypher = re.sub(r"\s*```$", "", cypher)
            return cypher.strip()
        except Exception as e:
            logger.error("Cypher generation failed: %s", str(e)[:120])
            return ""   # _validate_cypher() will reject empty string

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Private: Cypher Execution
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _execute_cypher(self, cypher: str, doc_id: str | None) -> list[dict]:
        """
        Execute a validated Cypher query in a READ transaction.

        Uses session.execute_read() — Neo4j's driver-level guarantee that
        the transaction function cannot perform write operations, even if
        a write keyword somehow slipped past _validate_cypher() (the third
        layer of our defense-in-depth strategy, see file header).

        Result serialization:
          Neo4j Record objects are NOT JSON-serializable and cannot live
          in AgentState (a TypedDict that must support LangGraph's internal
          serialization for checkpointing). We convert every value to a
          plain string via str(), producing flat dicts safe for state.

        Args:
            cypher: Pre-validated Cypher query string.
            doc_id: Optional document scope — passed as the $doc_id Cypher
                    parameter (the generated query references it, but if it
                    doesn't use $doc_id, Neo4j simply ignores the unused param).

        Returns:
            List of result row dicts (max 20, enforced by the query's own
            LIMIT clause, which _validate_cypher() requires to be present).
            Returns [] on any execution error (e.g. Neo4j unreachable,
            malformed query that passed static validation but fails at
            runtime due to schema mismatch).
        """
        try:
            params = {"doc_id": doc_id} if doc_id else {}

            with self.driver.session() as session:
                records = session.execute_read(
                    lambda tx: list(tx.run(cypher, **params))
                )

            paths = [
                {key: str(value) for key, value in record.items()}
                for record in records
            ]
            return paths[:20]   # defense-in-depth cap, even though query has its own LIMIT

        except Exception as e:
            logger.error(
                "Cypher execution failed (returning empty kg_paths): %s | query: %s",
                str(e)[:150], cypher[:200],
            )
            return []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Private: Empty State Update Helper
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _empty_update(self, state: "AgentState", t0: float) -> dict:
        """
        Safe fallback state update when graph retrieval cannot proceed.

        The generator (Phase 7) handles an empty kg_paths list gracefully —
        it simply omits the "[Knowledge Graph Paths]" section from the
        context window, and the LLM responds based on whatever other
        context is available (or states it cannot find relationship info).
        """
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "kg_paths": [],
            "latency_ms": {
                **(state.get("latency_ms") or {}),
                "graph": round(elapsed_ms, 2),
            },
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Module-Level Singleton
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# One agent per process — LLM clients and Neo4j driver are shared.
# Registered in orchestrator.py as:
#
#   Ingest graph:
#     g.add_node("extract_kg", kg_agent.extract_and_store_node)
#
#   Query graph (graph route):
#     g.add_node("graph_retrieve", kg_agent.graph_retrieve)
#     g.add_conditional_edges("router", router_agent.get_route, {
#         ..., "graph": "graph_retrieve",
#     })
kg_agent = KnowledgeGraphAgent()
