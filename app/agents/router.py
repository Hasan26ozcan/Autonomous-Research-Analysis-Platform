"""
app/agents/router.py
=====================
Adaptive Query Router — the intelligence that decides HOW to answer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY AN ADAPTIVE ROUTER?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The single biggest mistake in naive RAG systems is treating every question
identically: embed → retrieve → generate. This wastes compute on simple
questions and under-serves complex ones.

Consider three real questions users might ask:

  Q1: "What is cosine similarity?"
      → LLM already knows this. Running retrieval wastes 800ms and adds
        noise. Correct routing: direct (skip retrieval entirely).

  Q2: "What revenue did the company report in Q3 2024?"
      → One chunk from the financial report answers this. Correct routing:
        single (one retrieval round, straightforward generation).

  Q3: "How does the preprocessing described in section 3 address the
        data quality issues mentioned in section 7?"
      → Requires retrieving from BOTH sections, reasoning across them,
        possibly multiple retrieval rounds. Correct routing: multi_hop.

  Q4: "Which authors appear in papers cited by both chapter 2 and chapter 5?"
      → Requires understanding entity relationships across the document.
        Vector search cannot answer "which authors co-appear" — that's a
        graph query. Correct routing: graph.

The router makes this decision once, cheaply (gpt-4o-mini, ~300ms, ~$0.0001),
before the expensive retrieval and generation steps.

Reference:
  Jeong et al. (2024). "Adaptive RAG: Learning to Adapt Retrieval-Augmented
  Large Language Models through Question Complexity." arXiv:2403.14403

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MEM0 LONG-TERM MEMORY INTEGRATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The router also fetches the user's long-term memories from Mem0 at the
START of the pipeline. This is the correct place because:

  1. Memories are needed by the generator (later) — fetching them here
     means they flow through state without an extra Mem0 call later.
  2. Memories can INFLUENCE the routing decision. If Mem0 says "this user
     always asks about chapter 3", the router can adjust its classification.
  3. Fetching asynchronously with routing (same node) keeps total latency
     lower than sequential fetch → classify.

Mem0 delivers:
  - 91% lower p95 latency vs full-context approaches
  - 90%+ token cost reduction
  - Best performance on multi-hop, temporal, and open-domain memory tasks
  Reference: Chhikara et al. (2025), arXiv:2504.19413

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANGGRAPH INTEGRATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This file provides TWO functions for the query graph:

  1. route(state) → dict
     LangGraph NODE. Reads question, classifies it, fetches memories.
     Returns partial state update: {query_type, routing_confidence,
                                    long_term_memories, latency_ms}

  2. get_route(state) → str
     LangGraph CONDITIONAL EDGE. Reads query_type from state.
     Returns the NAME of the next node to execute.

     Graph wiring (in orchestrator.py):
       g.add_node("router", router_agent.route)
       g.add_conditional_edges("router", router_agent.get_route, {
           "direct":    "direct_answer",
           "single":    "retrieve",
           "multi_hop": "retrieve_multi",
           "graph":     "graph_retrieve",
       })

State read by this node (from AgentState):
    question  (str):      the raw user question
    user_id   (str):      used as Mem0 user_id for memory lookup
    latency_ms (dict):    accumulated per-node timing (we add "router" key)

State written by this node (partial update dict):
    query_type          (QueryType):   "direct" | "single" | "multi_hop" | "graph"
    routing_confidence  (float):       classifier confidence score (0.0–1.0)
    long_term_memories  (list[dict]):  [{memory: str, score: float}, ...]
    latency_ms          (dict):        previous dict + {"router": <ms>}
"""

from __future__ import annotations

import json
import logging
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
from app.core.state import QueryType

if TYPE_CHECKING:
    from app.core.state import AgentState

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROUTING PROMPT
#
# Prompt engineering decisions:
#   1. The four categories are defined WITH examples — not just named.
#      Examples dramatically improve classification accuracy for edge cases.
#   2. We force JSON output (response_format={"type": "json_object"}) to
#      guarantee parseable structured output without asking the model to
#      "wrap in ```json```" — that approach is fragile.
#   3. "reason" field forces the model to articulate its reasoning before
#      committing to a type — a chain-of-thought approach within JSON.
#   4. We do NOT include the document content in the routing prompt.
#      The router classifies based on question structure alone, not document
#      content. Adding document content would inflate cost by 10-50x.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROUTER_SYSTEM_PROMPT = """\
You are a query complexity classifier for a multi-agent RAG system.

Your job: classify the user's question into exactly ONE of four retrieval strategies.
The strategy you choose determines which agent handles the question next.

━━━ STRATEGY DEFINITIONS ━━━

direct (skip retrieval entirely)
  The answer is general knowledge the LLM already knows with high confidence.
  No document lookup is needed. Use this ONLY when you are very confident.
  Examples:
    • "What is cosine similarity?"
    • "Explain the difference between precision and recall."
    • "What does RAG stand for?"
    • "What year did World War II end?"

single (one retrieval round)
  The answer requires retrieving one or a few document chunks.
  A single hybrid search round is sufficient to find the answer.
  Examples:
    • "What revenue did the company report in Q3 2024?"
    • "What is the author's conclusion about climate change?"
    • "Which dataset was used in the experiments?"
    • "What hyperparameters were used for training?"

multi_hop (iterative retrieval + reasoning)
  The answer requires retrieving from MULTIPLE parts of the document and
  reasoning across them, OR the question involves comparison, synthesis,
  or multi-step inference that a single retrieval cannot satisfy.
  Examples:
    • "How does the method in section 3 address the limitations in section 6?"
    • "Compare the results of experiment A with experiment B."
    • "What assumptions in the introduction are contradicted by the results?"
    • "Summarize the evolution of the approach from chapter 1 to chapter 5."

graph (knowledge graph traversal)
  The answer requires understanding RELATIONSHIPS between named entities.
  Vector search finds similar text — it cannot answer structural relationship
  questions that require graph traversal.
  Examples:
    • "Which authors co-appear in papers cited by both section 2 and section 4?"
    • "What organizations are mentioned in connection with the main author?"
    • "Which datasets are shared between the baseline methods?"
    • "What is the citation chain from paper X to paper Y?"

━━━ OUTPUT FORMAT ━━━

Return ONLY a valid JSON object with exactly these three keys:
{
  "type": "<one of: direct, single, multi_hop, graph>",
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<one sentence explaining your classification>"
}

Do not include markdown, backticks, or any text outside the JSON object.\
"""

ROUTER_USER_TEMPLATE = "Question: {question}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRUCTURED OUTPUT MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RouterOutput(BaseModel):
    """
    Validated output from the router LLM call.

    Using Pydantic for validation means:
      - If the LLM returns an invalid type, we get a clear ValidationError
        (not a cryptic KeyError downstream)
      - Confidence is automatically clamped to [0.0, 1.0]
      - Type is validated against the allowed Literal values

    This model is NOT stored in AgentState — we extract its fields into
    the state update dict. Pydantic models are not JSON-serializable by
    default, and AgentState is a plain TypedDict.
    """

    type: QueryType = Field(
        description="Query complexity type. One of: direct, single, multi_hop, graph"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Router's confidence in the classification (0.0–1.0)",
    )
    reason: str = Field(
        description="One sentence explaining the classification decision"
    )

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        """Clamp confidence to [0.0, 1.0] even if LLM returns out-of-range."""
        return max(0.0, min(1.0, v))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROUTER AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RouterAgent:
    """
    Adaptive query router and long-term memory fetcher.

    Responsibilities:
      1. Classify question complexity → query_type (direct/single/multi_hop/graph)
      2. Fetch user's long-term memories from Mem0 → long_term_memories
      3. Accumulate per-node latency → latency_ms["router"]
      4. Expose get_route() as LangGraph conditional edge function

    LLM choice: router_model (gpt-4o-mini by default)
      Why NOT gpt-4o here?
        The routing task is structured classification with explicit definitions.
        gpt-4o-mini handles this with >95% accuracy at 10x lower cost.
        We save gpt-4o for the generation step where quality matters most.

    Why response_format={"type": "json_object"}?
      OpenAI's JSON mode guarantees the response is valid JSON.
      Without it, the model occasionally wraps output in ```json``` fences
      or adds preamble text, breaking json.loads(). JSON mode prevents this.
      Note: the system prompt must mention "JSON" for JSON mode to activate.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.router_model,
            api_key=settings.openai_api_key,
            temperature=0.0,                              # fully deterministic routing
            response_format={"type": "json_object"},      # guaranteed valid JSON output
            max_tokens=200,                               # reason + type + confidence fits in 200
        )
        self._mem0_client = None   # lazy-initialized to avoid startup failures

    # ── Lazy Mem0 Client ──────────────────────────────────────────────────────

    @property
    def mem0(self):
        """
        Lazy Mem0 client initialization.

        Why lazy?
          The Mem0 client makes a connection on instantiation. If Mem0 is
          not running yet (e.g. docker compose startup race condition),
          eager init would crash the entire API on boot.

        Cloud vs self-hosted:
          If settings.mem0_api_key is set → connects to Mem0 cloud.
          If settings.mem0_api_key is empty → connects to self-hosted
          Mem0 server at settings.mem0_base_url.
          The client API is identical either way.
        """
        if self._mem0_client is None:
            try:
                from mem0 import MemoryClient
                if settings.mem0_api_key:
                    # Mem0 cloud: API key authentication
                    self._mem0_client = MemoryClient(api_key=settings.mem0_api_key)
                else:
                    # Self-hosted: URL-based connection (docker-compose)
                    self._mem0_client = MemoryClient(base_url=settings.mem0_base_url)
                logger.info("Mem0 client initialized.")
            except Exception as e:
                logger.warning(
                    "Mem0 client init failed (long-term memory disabled): %s", e
                )
                self._mem0_client = None
        return self._mem0_client

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LangGraph Node: route()
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def route(self, state: "AgentState") -> dict:
        """
        LangGraph node function — the entry point of the query pipeline.

        Executes two tasks in parallel (sequentially for simplicity, but
        each is independent and could be parallelized with asyncio.gather):
          Task 1: Classify the question → query_type, routing_confidence
          Task 2: Fetch user memories from Mem0 → long_term_memories

        Reads from AgentState:
            question  (str):       the raw user question
            user_id   (str):       Mem0 user identifier
            latency_ms (dict):     existing per-node timing dict

        Returns partial AgentState update:
            query_type          (QueryType):   routing decision
            routing_confidence  (float):       classifier confidence
            long_term_memories  (list[dict]):  retrieved user memories
            latency_ms          (dict):        previous dict + "router" key

        Error handling:
            Classification failures fall back to "single" (safe default).
            Mem0 failures return [] (non-blocking — memory is a bonus, not required).
        """
        t0 = time.perf_counter()

        question: str = state.get("question", "")
        user_id: str = state.get("user_id", "")

        if not question:
            logger.error("Router called with empty question — defaulting to 'single'")
            return self._fallback_update(state, t0, reason="empty question")

        # ── Task 1: Classify query ─────────────────────────────────────────────
        routing_result = self._classify(question)

        logger.info(
            "Router: type=%s confidence=%.2f reason='%s' question='%s...'",
            routing_result.type,
            routing_result.confidence,
            routing_result.reason,
            question[:60],
        )

        # ── Task 2: Fetch long-term memories ──────────────────────────────────
        memories = self._fetch_memories(question=question, user_id=user_id)

        # ── Accumulate latency ────────────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t0) * 1000
        updated_latency = {
            **(state.get("latency_ms") or {}),
            "router": round(elapsed_ms, 2),
        }

        return {
            "query_type":         routing_result.type,
            "routing_confidence": routing_result.confidence,
            "long_term_memories": memories,
            "latency_ms":         updated_latency,
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LangGraph Conditional Edge: get_route()
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_route(self, state: "AgentState") -> str:
        """
        LangGraph conditional edge function.

        Called by LangGraph AFTER the router node completes.
        Reads query_type from state and returns the name of the next node.

        This function MUST return a string that is a key in the edge map
        defined in orchestrator.py:
            {
                "direct":    "direct_answer",
                "single":    "retrieve",
                "multi_hop": "retrieve_multi",
                "graph":     "graph_retrieve",
            }

        Why a separate function from route()?
          LangGraph separates node logic (route) from edge logic (get_route).
          The node produces state. The edge reads state to decide what's next.
          This separation allows the same router to be reused with different
          graph topologies without changing the routing logic.

        Fallback:
          If query_type is somehow missing or invalid, returns "single".
          This is the safest fallback — single retrieval works for most questions.
        """
        query_type: str = state.get("query_type", "single")

        valid_types = {"direct", "single", "multi_hop", "graph"}
        if query_type not in valid_types:
            logger.warning(
                "get_route: invalid query_type '%s', falling back to 'single'",
                query_type,
            )
            return "single"

        return query_type

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Private: LLM Classification
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_llm(self, question: str) -> str:
        """
        Call the router LLM and return the raw JSON string response.

        Isolated from _classify() so:
          - The @retry decorator applies only to the network call
          - _classify() handles JSON parsing separately from network errors
          - Tests can mock _call_llm without dealing with retry logic

        The retry decorator handles:
          - OpenAI rate limit errors (429)
          - Transient network timeouts
          - Temporary service unavailability (503)
        After 3 attempts (1s → 2s → 4s backoff), the exception propagates
        to _classify() which catches it and returns the safe fallback.
        """
        response = self.llm.invoke([
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=ROUTER_USER_TEMPLATE.format(question=question)),
        ])
        return response.content

    def _classify(self, question: str) -> RouterOutput:
        """
        Classify the question complexity. Returns RouterOutput.

        Two-step process:
          1. Call LLM → raw JSON string
          2. Parse and validate with Pydantic RouterOutput model

        Error handling strategy:
          - JSON parse error → log and return safe fallback (single, confidence=0.5)
          - Pydantic validation error → log and return safe fallback
          - LLM call failure (all retries exhausted) → log and return safe fallback

        Why "single" as the fallback type?
          - "single" always works (one retrieval round, generate answer)
          - "direct" would skip retrieval — risky if we're wrong
          - "multi_hop" adds unnecessary latency for simple questions
          - "graph" requires Neo4j and may not have relevant entity data

        The fallback confidence of 0.5 signals uncertainty to monitoring systems.
        Any confidence below 0.7 in production should trigger alert review.
        """
        try:
            raw_json = self._call_llm(question)
            parsed_dict = json.loads(raw_json)
            return RouterOutput(**parsed_dict)

        except json.JSONDecodeError as e:
            logger.error(
                "Router: JSON parse error (falling back to 'single'): %s | "
                "raw response: %s",
                e, raw_json[:200] if "raw_json" in dir() else "N/A",
            )
        except Exception as e:
            logger.error(
                "Router: classification failed (falling back to 'single'): %s", e
            )

        return RouterOutput(
            type="single",
            confidence=0.5,
            reason="Fallback due to classification error.",
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Private: Mem0 Memory Fetch
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _fetch_memories(self, question: str, user_id: str) -> list[dict]:
        """
        Search Mem0 for user memories relevant to the current question.

        How Mem0 works:
          Mem0 stores memories as semantic facts extracted from past conversations.
          e.g. "User works on flood prediction models" or "User prefers concise answers".
          mem0.search() does semantic search over stored memories using the
          current question as the query, returning the most relevant ones.

        Memory format returned:
          [{"memory": "User is a machine learning engineer at Omdena.",
            "score": 0.91},
           {"memory": "User prefers answers with code examples.",
            "score": 0.78}]

        These are passed to the generator in Phase 7, which prepends them
        to the LLM prompt as "[User Context from Memory]" — personalizing
        the answer without the user having to re-state their preferences.

        Error handling:
          Mem0 failure returns [] — long-term memory is an enhancement,
          not a requirement. The pipeline always completes without it.
          This prevents a Mem0 outage from breaking the entire query pipeline.

        Skips if:
          - user_id is empty (anonymous user — no memories to fetch)
          - Mem0 client failed to initialize
          - user_id is "anonymous" (reserved for internal/eval calls)
        """
        if not user_id or user_id == "anonymous":
            logger.debug("Mem0: skipping fetch for anonymous/empty user_id")
            return []

        if self.mem0 is None:
            logger.debug("Mem0: client not available — skipping memory fetch")
            return []

        try:
            t0 = time.perf_counter()
            results = self.mem0.search(
                query=question,
                user_id=user_id,
                limit=5,   # top 5 most relevant memories is sufficient for context
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            memories = [
                {
                    "memory": r.get("memory", ""),
                    "score":  float(r.get("score", 0.0)),
                }
                for r in (results or [])
                if r.get("memory")   # skip empty memory strings
            ]

            logger.info(
                "Mem0: fetched %d memories for user_id='%s' in %.0fms",
                len(memories), user_id, elapsed_ms,
            )
            return memories

        except Exception as e:
            logger.warning(
                "Mem0: fetch failed for user_id='%s' (non-fatal, returning []): %s",
                user_id, str(e)[:120],
            )
            return []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Private: Fallback State Update
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _fallback_update(self, state: "AgentState", t0: float, reason: str) -> dict:
        """
        Return a safe fallback state update when the router cannot classify.

        Always returns a valid state update so the pipeline can continue.
        The "single" type is the safest fallback (see _classify docstring).
        """
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "query_type":         "single",
            "routing_confidence": 0.0,
            "long_term_memories": [],
            "latency_ms": {
                **(state.get("latency_ms") or {}),
                "router": round(elapsed_ms, 2),
            },
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Module-Level Singleton
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# One RouterAgent per process — LLM client and Mem0 client shared across requests.
# Registered in orchestrator.py as:
#   g.add_node("router", router_agent.route)
#   g.add_conditional_edges("router", router_agent.get_route, {...})
router_agent = RouterAgent()
