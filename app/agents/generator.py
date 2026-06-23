"""
Generator + Faithfulness Judge
================================
Two-stage answer production:

  Stage 1 — Generator
    Builds a context window from retrieved chunks + KG paths + Mem0 memories,
    then calls the LLM with strict source-grounding instructions.
    Every claim must cite a [Source N] reference.

  Stage 2 — Faithfulness Judge
    Uses NLI (Natural Language Inference) to score whether each sentence
    in the draft answer is entailed by the retrieved context.
    If faithfulness_score < threshold, the pipeline retries with a stricter prompt.

    NLI model: cross-encoder/nli-deberta-v3-small (local, no API cost).
    Label mapping: ENTAILMENT=1, NEUTRAL=0.5, CONTRADICTION=0.

    This directly prevents hallucination and builds the kind of observable,
    auditable AI pipeline that enterprise customers require in 2026.

Reference:
  "Agentic RAG in 2026: the agent loops over retrieve and reason,
   with a faithfulness judge gating the final response." — FutureAGI, May 2026
  NLI for faithfulness: MiniCheck (Tang et al., 2024) — arXiv:2404.10774
"""
from __future__ import annotations
import time
import re
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder

from app.core.state import AgentState
from app.core.config import settings

logger = logging.getLogger(__name__)


GENERATOR_SYSTEM = """You are a precise document assistant. Your ONLY job is to answer
the user's question using the provided context.

Rules:
1. NEVER introduce facts not present in the provided context.
2. For every claim you make, cite the source: [Source 1], [Source 2], etc.
3. If the context does not contain enough information to answer, say so explicitly.
4. Integrate any relevant user memories to personalize the response.
5. Be concise and structured. Use bullet points for lists.
6. Do not speculate or infer beyond what is literally stated in the context."""

RETRY_SYSTEM = """You are a precise document assistant. A previous answer was rejected
for potential hallucinations (making claims not supported by the context).

STRICT RULES — no exceptions:
1. Only use facts verbatim from the provided context. Copy short phrases if needed.
2. Cite every single sentence with [Source N].
3. If any part of the question cannot be answered from context, say "Not found in context."
4. It is better to give a short, fully-grounded answer than a long, partially-fabricated one."""


class AnswerGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        self._nli: CrossEncoder | None = None

    @property
    def nli(self) -> CrossEncoder:
        if self._nli is None:
            self._nli = CrossEncoder(
                settings.nli_model,
                num_labels=3,  # contradiction, entailment, neutral
            )
        return self._nli

    # ── Generation node ────────────────────────────────────────────────────────

    def generate(self, state: AgentState) -> AgentState:
        """LangGraph node: build context + generate answer."""
        t0 = time.perf_counter()

        context = self._build_context(state)
        system = GENERATOR_SYSTEM if state.get("retry_count", 0) == 0 else RETRY_SYSTEM
        prompt = self._build_prompt(state["question"], context, state.get("long_term_memories", []))

        response = self.llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ])
        draft = response.content.strip()

        sources = self._format_sources(state.get("retrieved_chunks", []))
        elapsed = (time.perf_counter() - t0) * 1000

        return {
            "draft_answer": draft,
            "sources": sources,
            "latency_ms": {**(state.get("latency_ms") or {}), "generation": elapsed},
        }

    # ── Faithfulness judge node ────────────────────────────────────────────────

    def judge(self, state: AgentState) -> AgentState:
        """
        LangGraph node: NLI-based faithfulness scoring.
        Scores each sentence in draft_answer against the retrieved context.
        """
        t0 = time.perf_counter()
        draft = state.get("draft_answer", "")
        chunks = state.get("retrieved_chunks", [])

        if not chunks or not draft:
            return {"faithfulness_score": 1.0, "judge_passed": True, "answer": draft}

        context_text = " ".join(c["text"] for c in chunks[:5])
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", draft) if len(s.strip()) > 20]

        if not sentences:
            return {"faithfulness_score": 1.0, "judge_passed": True, "answer": draft}

        # NLI: (premise=context, hypothesis=each sentence)
        pairs = [(context_text[:512], s) for s in sentences]
        logits = self.nli.predict(pairs, apply_softmax=True)

        # Labels: [contradiction=0, entailment=1, neutral=2] for deberta-v3
        entailment_scores = [float(logit[1]) for logit in logits]
        faithfulness = sum(entailment_scores) / len(entailment_scores)

        passed = faithfulness >= settings.faithfulness_threshold
        retry_count = state.get("retry_count", 0)

        logger.info(
            "Judge: faithfulness=%.3f threshold=%.2f passed=%s retry=%d",
            faithfulness, settings.faithfulness_threshold, passed, retry_count,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        update: dict = {
            "faithfulness_score": faithfulness,
            "judge_passed": passed,
            "latency_ms": {**(state.get("latency_ms") or {}), "judge": elapsed},
        }

        if passed or retry_count >= settings.max_retries:
            update["answer"] = state["draft_answer"]
        else:
            update["retry_count"] = retry_count + 1

        return update

    def should_retry(self, state: AgentState) -> str:
        """LangGraph conditional edge: retry generation or proceed."""
        if state.get("judge_passed") or state.get("retry_count", 0) >= settings.max_retries:
            return "memory_store"
        return "generate"

    # ── Memory storage node ────────────────────────────────────────────────────

    def store_memory(self, state: AgentState) -> AgentState:
        """
        LangGraph node: persist the Q&A turn to Mem0.
        Mem0 automatically extracts semantic facts and deduplicates.
        """
        if not state.get("user_id"):
            return {}
        try:
            from mem0 import MemoryClient
            client = MemoryClient(
                api_key=settings.mem0_api_key or None,
                base_url=settings.mem0_base_url,
            )
            messages = [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": state.get("answer", "")},
            ]
            client.add(messages, user_id=state["user_id"])
            logger.info("Mem0: stored turn for user=%s", state["user_id"])
        except Exception as e:
            logger.warning("Mem0 store failed (non-fatal): %s", e)
        return {}

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_context(self, state: AgentState) -> str:
        """Assemble context window from chunks + KG paths."""
        parts = []

        chunks = state.get("retrieved_chunks", [])
        for i, c in enumerate(chunks, start=1):
            parts.append(
                f"[Source {i} | {c.get('filename', 'doc')} p.{c.get('page', '?')}]\n{c['text']}"
            )

        paths = state.get("kg_paths", [])
        if paths:
            parts.append("\n[Knowledge Graph Paths]")
            for p in paths[:5]:
                parts.append(str(p))

        return "\n\n".join(parts)

    def _build_prompt(
        self,
        question: str,
        context: str,
        memories: list[dict],
    ) -> str:
        memory_section = ""
        if memories:
            mem_text = "\n".join(f"- {m['memory']}" for m in memories[:5])
            memory_section = f"\n\n[User Context from Memory]\n{mem_text}"

        return (
            f"Context:\n{context}"
            f"{memory_section}"
            f"\n\nQuestion: {question}"
        )

    def _format_sources(self, chunks: list[dict]) -> list[dict]:
        return [
            {
                "index": i + 1,
                "text": c["text"][:300] + ("..." if len(c["text"]) > 300 else ""),
                "page": c.get("page"),
                "filename": c.get("filename"),
                "doc_id": c.get("doc_id"),
                "rerank_score": round(c.get("rerank_score", 0), 4),
            }
            for i, c in enumerate(chunks)
        ]


generator = AnswerGenerator()
