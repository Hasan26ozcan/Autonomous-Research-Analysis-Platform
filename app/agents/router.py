"""
Adaptive Query Router
=====================
Implements the 2026 Adaptive RAG pattern:
  a small, fast LLM call classifies each query into one of four types,
  then LangGraph conditional edges route to the appropriate specialist agent.

Classification types:
  direct    — LLM can answer from parametric knowledge, no retrieval needed
  single    — one retrieval round against Qdrant (simple factual question)
  multi_hop — iterative retrieval with query decomposition (complex reasoning)
  graph     — requires Knowledge Graph traversal (entity/relationship questions)

Reference:
  Adaptive RAG: "Learning to Adapt Retrieval-Augmented Large Language Models
  through Question Complexity" (Jeong et al., 2024) — arXiv:2403.14403
"""
from __future__ import annotations
import time
import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.core.state import AgentState, QueryType
from app.core.config import settings

logger = logging.getLogger(__name__)


ROUTER_SYSTEM = """You are a query complexity classifier for a RAG system.
Classify the user's question into exactly one of these types:

  direct    — The answer is general knowledge the LLM already knows with high confidence.
              No document lookup is needed.
              Example: "What is attention in transformers?"

  single    — The answer requires retrieving one or a few document chunks.
              One retrieval round is sufficient.
              Example: "What did the Q4 2025 earnings report say about revenue?"

  multi_hop — The answer requires retrieving from multiple sources or reasoning
              across several steps. Decomposition is needed.
              Example: "Compare the methodology in section 3 with the limitations in section 6."

  graph     — The answer requires understanding relationships between named entities,
              organizations, concepts, or any multi-hop entity traversal.
              Example: "Which authors collaborated on both papers cited in chapter 2?"

Respond with a JSON object: {"type": "<type>", "confidence": <0.0-1.0>, "reason": "<brief>"}
Return ONLY the JSON. No other text."""


class RouterOutput(BaseModel):
    type: QueryType
    confidence: float
    reason: str


class AdaptiveRouter:
    """
    Wraps the LLM-based classifier and also pulls Mem0 long-term memories
    to inject user context before routing decisions.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.router_model,
            api_key=settings.openai_api_key,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        self._mem0 = None

    @property
    def mem0(self):
        if self._mem0 is None:
            from mem0 import MemoryClient
            self._mem0 = MemoryClient(
                api_key=settings.mem0_api_key or None,
                base_url=settings.mem0_base_url,
            )
        return self._mem0

    def route(self, state: AgentState) -> AgentState:
        """
        LangGraph node: classify query + fetch long-term memories.
        Returns partial state update.
        """
        t0 = time.perf_counter()
        question = state["question"]

        # ── 1. Classify query complexity ───────────────────────────────────────
        messages = [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Question: {question}"),
        ]
        raw = self.llm.invoke(messages)
        parsed = RouterOutput.model_validate_json(raw.content)

        logger.info(
            "Router: type=%s confidence=%.2f reason=%s",
            parsed.type, parsed.confidence, parsed.reason,
        )

        # ── 2. Fetch long-term memories from Mem0 ─────────────────────────────
        memories: list[dict] = []
        if state.get("user_id"):
            try:
                results = self.mem0.search(
                    query=question,
                    user_id=state["user_id"],
                    limit=5,
                )
                memories = [{"memory": r["memory"], "score": r.get("score", 0)} for r in results]
                logger.info("Mem0: fetched %d memories for user=%s", len(memories), state["user_id"])
            except Exception as e:
                logger.warning("Mem0 fetch failed (non-fatal): %s", e)

        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "query_type": parsed.type,
            "routing_confidence": parsed.confidence,
            "long_term_memories": memories,
            "latency_ms": {**(state.get("latency_ms") or {}), "router": elapsed},
        }

    def get_route(self, state: AgentState) -> str:
        """
        LangGraph conditional edge function.
        Returns the name of the next node to execute.
        """
        return state.get("query_type", "single")


# Singleton for use as LangGraph node
router = AdaptiveRouter()
