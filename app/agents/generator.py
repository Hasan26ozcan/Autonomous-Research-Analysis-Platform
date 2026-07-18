"""
app/agents/generator.py
========================
Generator + Faithfulness Judge + Mem0 Memory Store.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THREE RESPONSIBILITIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. GENERATOR — generate()
   Assembles a context window from three sources:
     a. retrieved_chunks     → from Phase 5 (RetrievalAgent) or Phase 6 (GraphAgent)
     b. kg_paths             → from Phase 6 (KnowledgeGraphAgent), if any
     c. long_term_memories   → from Phase 4 (RouterAgent via Mem0)

   Calls GPT-4o (llm_model) with a strict grounding prompt.
   Every claim in the answer must cite a source: [Source 1], [Source 2], etc.
   Produces draft_answer — NOT the final answer yet.

2. FAITHFULNESS JUDGE — judge()
   Scores whether the draft_answer is grounded in the retrieved context.

   Method: NLI (Natural Language Inference) using DeBERTa-v3-small.
     - Splits draft_answer into sentences
     - Scores each sentence as: entailment / neutral / contradiction
       against the concatenated retrieved context
     - faithfulness_score = mean entailment probability across sentences
     - If score < settings.faithfulness_threshold (0.75): triggers retry
     - After settings.max_retries retries: finalizes with best available answer

   Why NLI instead of LLM-as-judge?
     NLI is local inference (~50ms per sentence pair, no API cost).
     LLM-as-judge costs ~$0.002 per evaluation and adds 2-3 seconds latency.
     For production at scale, NLI is the correct trade-off.
     DeBERTa-v3-small achieves SOTA NLI accuracy at this model size class.

   Reference:
     Tang et al. (2024). "MiniCheck: Efficient Fact-Checking of LLMs on
     Grounding Documents." arXiv:2404.10774

3. MEMORY STORE — store_memory()
   After judge approves the answer, persists the (question, answer) turn
   to Mem0 for the user. Next time this user asks a related question,
   the router fetches these memories and the generator has personalization
   context without re-reading the full conversation history.

   Uses the same Mem0 connection pattern as Phase 4 (RouterAgent) —
   cloud with api_key, or self-hosted with base_url.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANGGRAPH WIRING (defined in orchestrator.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  merge → generate → judge → should_retry() ──┐
               ↑                               │  "generate" (retry path)
               └───────────────────────────────┘
                             │
                             └── "memory_store" (pass path) → END

  should_retry() returns:
    "generate"     → if not passed AND retry_count < max_retries
    "memory_store" → if passed OR retry_count >= max_retries

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STATE CONTRACT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

generate() reads:
    question            (str):        raw user question
    retrieved_chunks    (list[dict]): from Phase 5 — text, page, filename,
                                       doc_id, chunk_index, rerank_score
    kg_paths            (list[dict]): from Phase 6 — head, relation, tail (optional)
    long_term_memories  (list[dict]): from Phase 4 — memory, score (optional)
    retry_count         (int):        how many times judge has already rejected
    latency_ms          (dict):       accumulated per-node timing

generate() writes:
    draft_answer  (str):        raw LLM output before judging
    sources       (list[dict]): formatted source list for client display
    latency_ms    (dict):       + {"generation": <ms>}

judge() reads:
    draft_answer        (str):        from generate()
    retrieved_chunks    (list[dict]): used as NLI premise text
    retry_count         (int):        to decide if we've hit max_retries
    latency_ms          (dict):       accumulated timing

judge() writes:
    faithfulness_score  (float):  NLI entailment mean (0.0–1.0)
    judge_passed        (bool):   True if score >= threshold or retries exhausted
    answer              (str):    set only when judge_passed=True
    retry_count         (int):    incremented on failure
    latency_ms          (dict):   + {"judge": <ms>}

should_retry() reads:
    judge_passed  (bool)
    retry_count   (int)

store_memory() reads:
    question  (str)
    answer    (str)
    user_id   (str)
    (writes nothing to state — pure side effect)
"""

from __future__ import annotations

import logging
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder

from app.core.config import settings

# NOTE: must be a real (non-TYPE_CHECKING) import - see graph_agent.py note.
# generate()/judge()/should_retry() use `state: "AgentState"` as a
# runtime-resolved string annotation (LangGraph calls typing.get_type_hints()
# on node functions), so AgentState must actually be bound in this module's
# namespace at runtime.
from app.core.state import AgentState
from app.services.llm_client import make_llm
from app.services.postgres_store import record_conversation

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GENERATION PROMPTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Standard prompt: used on first generation attempt (retry_count == 0)
GENERATOR_SYSTEM_PROMPT = """\
You are a precise document assistant. Your task is to answer the user's
question using ONLY the information provided in the context below.

RULES — follow every one, no exceptions:
  1. Ground every claim in the provided context. Never introduce facts
     you know from pre-training if they are not confirmed by the context.
  2. Cite the source of every claim using [Source N] inline, where N is
     the source index number from the context.
  3. If a piece of information does not appear in the context, say so
     explicitly: "This information was not found in the provided document."
  4. Use [User Context] to personalize your answer when relevant, but
     never cite user context as a document source.
  5. Be concise and well-structured. Use bullet points for lists, bold
     for key terms, and short paragraphs for explanations.
  6. Do not speculate, infer beyond what is stated, or pad the answer.\
"""

# Stricter prompt: used on retry attempts after judge rejection.
# More directive language and explicit grounding instruction.
GENERATOR_RETRY_PROMPT = """\
You are a strict document assistant. A previous answer you generated was
REJECTED for including claims not fully supported by the retrieved context.

STRICT RULES for this retry:
  1. Only state facts that are LITERALLY present in the context.
     If you need to be certain, quote a short phrase from the source directly.
  2. Every single sentence must end with a [Source N] citation.
  3. If the question cannot be fully answered from the context, answer
     only the parts that CAN be answered, and state what is missing.
  4. Do NOT fill gaps with general knowledge. A shorter, fully-grounded
     answer is far better than a complete but partially fabricated one.\
"""

# Context assembly templates
CONTEXT_SOURCE_TEMPLATE = (
    "[Source {index} | {filename} — Page {page}]\n{text}"
)
KG_PATHS_HEADER = "\n[Knowledge Graph — Relationship Paths]\n"
USER_MEMORY_HEADER = "\n[User Context from Memory]\n"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NLI LABEL MAPPING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# cross-encoder/nli-deberta-v3-small output label order:
# Position 0 = contradiction, Position 1 = entailment, Position 2 = neutral
# We want the ENTAILMENT probability for faithfulness scoring.
_NLI_ENTAILMENT_INDEX = 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANSWER GENERATOR CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AnswerGenerator:
    """
    Implements the generate → judge → (retry?) → memory_store pipeline.

    LLM choice: llm_model (gpt-4o by default).
      Unlike the router and retrieval agents (which use gpt-4o-mini for
      classification/structured tasks), generation requires deep reasoning
      and high-quality language production — gpt-4o is the right choice here.

    NLI model: cross-encoder/nli-deberta-v3-small (local, no API cost).
      Loaded lazily on first judge() call. ~150MB download, CPU inference.
      Shared across all requests in the same process — expensive to load,
      cheap to use once loaded.

    Mem0 client: lazy-initialized, same pattern as Phase 4 (RouterAgent).
    """

    def __init__(self):
        self.llm = make_llm(
            model=settings.llm_model,
            max_tokens=settings.max_tokens,
        )
        self._nli_model: CrossEncoder | None = None
        self._mem0_client = None

    # ── Lazy NLI model ────────────────────────────────────────────────────────

    @property
    def nli(self) -> CrossEncoder:
        """
        Lazy-load the NLI cross-encoder for faithfulness scoring.

        Model: cross-encoder/nli-deberta-v3-small
          - 3 output labels: [contradiction, entailment, neutral]
          - apply_softmax=True in predict() converts logits to probabilities
          - We extract index 1 (entailment probability) per sentence pair

        Why DeBERTa over BERT for NLI?
          DeBERTa uses disentangled attention — content and position are
          attended separately. This consistently outperforms BERT-based
          models on NLI benchmarks (MNLI, SNLI, NLI-hard) since 2021.
        """
        if self._nli_model is None:
            logger.info(
                "Loading NLI model '%s' (first judge() call, may download)...",
                settings.nli_model,
            )
            t0 = time.perf_counter()
            self._nli_model = CrossEncoder(
                settings.nli_model,
                num_labels=3,         # contradiction / entailment / neutral
            )
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("NLI model loaded in %.0fms", elapsed)
        return self._nli_model

    # ── Lazy Mem0 client ──────────────────────────────────────────────────────

    @property
    def mem0(self):
        """
        Lazy Mem0 client, consistent with Phase 4 (RouterAgent.mem0).

        Embedded mode (default, no Mem0 account needed): points at the
        already-running Qdrant (separate collection) + local
        sentence-transformers embedder + whatever OpenAI-compatible LLM
        endpoint the app is already configured with (e.g. Groq). See
        RouterAgent.mem0 for the full rationale - kept identical here so
        both agents share one consistent memory store.

        store_memory() is a best-effort operation - if Mem0 is unavailable,
        the query still completes and the answer is still returned.
        """
        if self._mem0_client is None:
            try:
                if settings.mem0_api_key:
                    from mem0 import MemoryClient
                    self._mem0_client = MemoryClient(api_key=settings.mem0_api_key)
                    logger.info("Generator: Mem0 client initialized (hosted Mem0 Platform).")
                else:
                    from mem0 import Memory
                    config = {
                        "vector_store": {
                            "provider": "qdrant",
                            "config": {
                                "collection_name": settings.mem0_collection_name,
                                "host": settings.qdrant_host,
                                "port": settings.qdrant_port,
                                "embedding_model_dims": settings.embedding_dim,
                            },
                        },
                        "embedder": {
                            "provider": "huggingface",
                            "config": {
                                "model": settings.embedding_model,
                                "embedding_dims": settings.embedding_dim,
                            },
                        },
                        "llm": {
                            "provider": "openai",
                            "config": {
                                "model": settings.router_model,
                                "api_key": settings.openai_api_key,
                                "openai_base_url": settings.llm_base_url,
                            },
                        },
                    }
                    self._mem0_client = Memory.from_config(config)
                    logger.info("Generator: Mem0 initialized (embedded) for store_memory.")
            except Exception as e:
                logger.warning(
                    "Generator: Mem0 client init failed (memory storage disabled): %s", e
                )
                self._mem0_client = None
        return self._mem0_client

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LangGraph Node 1: generate()
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def generate(self, state: AgentState) -> dict:
        """
        LangGraph node: build context window and call LLM for answer generation.

        Runs AFTER all retrieval branches (single, multi_hop, graph) have
        merged at the "merge" node in the query graph. At this point, state
        contains retrieved_chunks (Phase 5) and possibly kg_paths (Phase 6)
        and long_term_memories (Phase 4).

        On first call (retry_count == 0): uses GENERATOR_SYSTEM_PROMPT.
        On retry (retry_count >= 1): uses GENERATOR_RETRY_PROMPT, which is
        stricter and explicitly instructs the model to avoid the hallucination
        patterns that caused the judge to reject the previous draft.

        Reads from AgentState:
            question           (str):        raw user question
            retrieved_chunks   (list[dict]): Phase 5 output
            kg_paths           (list[dict]): Phase 6 output (may be empty)
            long_term_memories (list[dict]): Phase 4 output (may be empty)
            retry_count        (int):        0 on first call, incremented by judge

        Writes to AgentState (partial update):
            draft_answer (str):        raw LLM response (before judging)
            sources      (list[dict]): formatted for client display
            latency_ms   (dict):       + {"generation": <ms>}
        """
        t0 = time.perf_counter()

        question: str = state.get("question", "")
        retrieved_chunks: list[dict] = state.get("retrieved_chunks") or []
        kg_paths: list[dict] = state.get("kg_paths") or []
        memories: list[dict] = state.get("long_term_memories") or []
        retry_count: int = state.get("retry_count") or 0

        # Build the full context window
        context = self._build_context(retrieved_chunks, kg_paths)
        user_prompt = self._build_user_prompt(question, context, memories)

        # First attempt uses standard prompt; retries use the stricter one
        system_prompt = (
            GENERATOR_SYSTEM_PROMPT if retry_count == 0
            else GENERATOR_RETRY_PROMPT
        )

        if retry_count > 0:
            logger.info(
                "generate(): retry attempt %d/%d for question='%s...'",
                retry_count, settings.max_retries, question[:50],
            )

        # LLM call — this is the primary cost center in the pipeline
        from app.services.rate_limiter import estimate_tokens, groq_rate_limiter
        groq_rate_limiter.acquire(estimate_tokens(
            system_prompt, user_prompt, max_output_tokens=settings.max_tokens,
        ))
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        draft_answer = response.content.strip()

        # Format sources for client display
        sources = self._format_sources(retrieved_chunks)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "generate(): %d chars draft, %d sources in %.0fms (retry=%d)",
            len(draft_answer), len(sources), elapsed_ms, retry_count,
        )

        return {
            "draft_answer": draft_answer,
            "sources":      sources,
            "latency_ms": {
                **(state.get("latency_ms") or {}),
                "generation": round(elapsed_ms, 2),
            },
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LangGraph Node 2: judge()
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def judge(self, state: AgentState) -> dict:
        """
        LangGraph node: NLI-based faithfulness scoring of draft_answer.

        Runs AFTER generate(). Reads draft_answer from state, scores each
        sentence for entailment against the retrieved context, and either:
          a. Approves the answer (faithfulness_score >= threshold):
             Sets judge_passed=True and answer=draft_answer.
          b. Rejects and triggers retry (score < threshold, retries available):
             Sets judge_passed=False and increments retry_count.
             The conditional edge should_retry() reads judge_passed and routes
             back to generate() for another attempt.
          c. Forces finalization (retries exhausted):
             Sets judge_passed=True and answer=draft_answer regardless of score.
             This prevents infinite loops — the user gets the best available
             answer rather than an error.

        NLI Scoring methodology:
          1. Split draft_answer into sentences using regex.
             Sentences shorter than 20 characters are skipped
             (citations like "[Source 1]" alone are not factual claims).
          2. Build (premise, hypothesis) pairs:
               premise    = concatenated retrieved_chunks text (first 5 chunks,
                             ~2000 tokens — balances coverage vs model limits)
               hypothesis = each answer sentence
          3. Run cross-encoder forward pass with apply_softmax=True.
          4. Extract entailment probability (index 1 of softmax output).
          5. faithfulness_score = mean across all sentence scores.

        Reads from AgentState:
            draft_answer     (str):        from generate()
            retrieved_chunks (list[dict]): used as NLI premise
            retry_count      (int):        checked against max_retries
            latency_ms       (dict):       accumulated timing

        Writes to AgentState:
            faithfulness_score (float):  0.0–1.0, NLI entailment mean
            judge_passed       (bool):   True if approved or retries exhausted
            answer             (str):    set when judge_passed=True
            retry_count        (int):    incremented on rejection
            latency_ms         (dict):   + {"judge": <ms>}
        """
        t0 = time.perf_counter()

        draft: str = state.get("draft_answer") or ""
        retrieved_chunks: list[dict] = state.get("retrieved_chunks") or []
        retry_count: int = state.get("retry_count") or 0

        # ── Edge case: nothing to judge ────────────────────────────────────────
        if not draft:
            logger.warning("judge(): empty draft_answer — passing immediately")
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "faithfulness_score": 0.0,
                "judge_passed":       True,    # pass to avoid infinite retry on empty
                "answer":             draft,
                "latency_ms": {
                    **(state.get("latency_ms") or {}),
                    "judge": round(elapsed_ms, 2),
                },
            }

        # ── Edge case: no retrieved context to judge against ───────────────────
        if not retrieved_chunks:
            logger.info(
                "judge(): no retrieved chunks — cannot assess faithfulness. "
                "Passing with score=1.0 (direct answer path or empty retrieval)."
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "faithfulness_score": 1.0,
                "judge_passed":       True,
                "answer":             draft,
                "latency_ms": {
                    **(state.get("latency_ms") or {}),
                    "judge": round(elapsed_ms, 2),
                },
            }

        # ── Build NLI premise from retrieved chunks ────────────────────────────
        # Use original_text when available (strips the [Context: ...] prefix
        # added by Phase 3's contextual enricher). The NLI model should judge
        # whether answer claims are entailed by DOCUMENT content, not by
        # our own metadata wrappers.
        #
        # Take first 5 chunks — beyond that, the concatenated text exceeds
        # the cross-encoder's 512-token limit and gets silently truncated,
        # producing unreliable entailment scores for later sentences.
        premise = self._judge_build_premise(retrieved_chunks)

        # ── Split draft into sentences ─────────────────────────────────────────
        # Split on sentence-ending punctuation followed by whitespace.
        # Minimum 20 chars filters out inline citations like "[Source 1]"
        # and headers like "**Key Findings:**" that are not factual claims.
        sentences = self._judge_split_sentences(draft)

        if not sentences:
            # Draft has no sentence-length claims (e.g. one-liner answer)
            # Pass without NLI — a one-sentence answer is likely fine
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "faithfulness_score": 1.0,
                "judge_passed":       True,
                "answer":             draft,
                "latency_ms": {
                    **(state.get("latency_ms") or {}),
                    "judge": round(elapsed_ms, 2),
                },
            }

        # ── NLI scoring ────────────────────────────────────────────────────────
        pairs = [(premise, sentence) for sentence in sentences]

        t_nli = time.perf_counter()
        raw_scores = self.nli.predict(pairs, apply_softmax=True)  # type: ignore[arg-type]
        nli_elapsed_ms = (time.perf_counter() - t_nli) * 1000

        # raw_scores shape: (n_sentences, 3) — [contradiction, entailment, neutral]
        entailment_scores = [
            float(row[_NLI_ENTAILMENT_INDEX])
            for row in raw_scores
        ]
        faithfulness_score = sum(entailment_scores) / len(entailment_scores)

        logger.info(
            "judge(): faithfulness=%.3f (threshold=%.2f) | "
            "%d sentences, NLI in %.0fms | retry=%d/%d",
            faithfulness_score, settings.faithfulness_threshold,
            len(sentences), nli_elapsed_ms,
            retry_count, settings.max_retries,
        )

        # ── Decision: pass, reject, or force-finalize ─────────────────────────
        retries_exhausted = retry_count >= settings.max_retries
        passed = faithfulness_score >= settings.faithfulness_threshold

        return self._judge_build_update(
            state, t0, faithfulness_score, draft, retry_count,
            retries_exhausted, passed,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LangGraph Conditional Edge: should_retry()
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Private: judge() helpers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _judge_build_premise(self, retrieved_chunks: list[dict]) -> str:
        """
        Build the NLI premise text from the first 5 retrieved chunks.

        Uses original_text (pre-enrichment) when available so the NLI model
        judges entailment against document content, not our metadata wrappers.
        A character cap guards against exceeding the cross-encoder's limit.
        """
        premise_parts = []
        for chunk in retrieved_chunks[:5]:
            text = chunk.get("original_text") or chunk.get("text", "")
            if text:
                premise_parts.append(text)
        return " ".join(premise_parts)[:4000]   # extra safety cap in characters

    def _judge_split_sentences(self, draft: str) -> list[str]:
        """
        Split the draft answer into sentence-length claims for NLI scoring.

        Splits on sentence-ending punctuation + whitespace; the 20-char minimum
        filters inline citations and headers that are not factual claims.
        """
        return [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", draft)
            if len(s.strip()) >= 20
        ]

    def _judge_build_update(
        self,
        state: AgentState,
        t0: float,
        faithfulness_score: float,
        draft: str,
        retry_count: int,
        retries_exhausted: bool,
        passed: bool,
    ) -> dict:
        """
        Build the judge() state update: pass, reject, or force-finalize.

        Mirrors the inlined decision logic exactly: on pass (or exhausted
        retries) approves with answer=draft; otherwise rejects and bumps
        retry_count to route back to generate() with the stricter prompt.
        """
        elapsed_ms = (time.perf_counter() - t0) * 1000
        update: dict = {
            "faithfulness_score": round(faithfulness_score, 4),
            "latency_ms": {
                **(state.get("latency_ms") or {}),
                "judge": round(elapsed_ms, 2),
            },
        }

        if passed or retries_exhausted:
            if retries_exhausted and not passed:
                logger.warning(
                    "judge(): max_retries (%d) reached with faithfulness=%.3f < %.2f. "
                    "Finalizing best-available answer.",
                    settings.max_retries, faithfulness_score,
                    settings.faithfulness_threshold,
                )
            update["judge_passed"] = True
            update["answer"] = draft
        else:
            # Reject: route back to generate() with stricter prompt
            logger.info(
                "judge(): REJECTED (faithfulness=%.3f < %.2f). "
                "Routing to generate() retry.",
                faithfulness_score, settings.faithfulness_threshold,
            )
            update["judge_passed"] = False
            update["retry_count"] = retry_count + 1

        return update

    def should_retry(self, state: AgentState) -> str:
        """
        LangGraph conditional edge function called after judge().

        Reads judge_passed from state and returns the name of the next node.

        Return values (must match orchestrator.py edge map):
            "generate"     → re-run generate() with stricter prompt
            "memory_store" → proceed to persist the approved answer

        Wiring in orchestrator.build_query_graph():
            g.add_conditional_edges(
                "judge",
                generator.should_retry,
                {
                    "generate":     "generate",
                    "memory_store": "memory_store",
                },
            )
        """
        judge_passed: bool = state.get("judge_passed", True)

        if judge_passed:
            logger.debug("should_retry(): judge passed → memory_store")
            return "memory_store"
        else:
            logger.info("should_retry(): judge rejected → generate (retry)")
            return "generate"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LangGraph Node 3: store_memory()
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def store_memory(self, state: AgentState) -> dict | None:
        """
        LangGraph node: persist the approved Q&A turn to Mem0.

        Runs AFTER judge() approves the answer (or exhausts retries).
        This is the last node before END in the query graph.

        What Mem0 does with the turn:
          Mem0's add() method sends the conversation turn to Mem0's memory
          extraction pipeline. Mem0 automatically:
            - Extracts semantic facts: "User asked about flood prediction models"
            - Deduplicates: if "User works at Omdena" already exists, it's updated
            - Creates embeddings for future semantic search
          Next time this user queries the system, RouterAgent._fetch_memories()
          retrieves these stored facts and injects them into the generator prompt.

        Skips if:
          - user_id is empty or "anonymous" (no persistent identity to store against)
          - Mem0 client is unavailable (Mem0 outage should not block response)
          - answer is empty (nothing meaningful to store)

        Why this is a separate node (not part of generate/judge)?
          Separation of concerns: the LangGraph graph can be modified to
          run store_memory() in parallel with the response delivery in future
          versions. Keeping it as a separate node makes that refactor trivial.

        Reads from AgentState:
            question  (str):  original user question
            answer    (str):  approved final answer
            user_id   (str):  Mem0 user identifier

        Writes to AgentState:
            (nothing — pure side effect)
        """
        question: str = state.get("question", "")
        answer: str = state.get("answer", "")
        user_id: str = state.get("user_id", "")

        # Skip conditions
        if not user_id or user_id == "anonymous":
            logger.debug("store_memory(): skipping for anonymous/empty user_id")
            return None

        if not answer:
            logger.debug("store_memory(): skipping — empty answer")
            return None

        if self.mem0 is None:
            logger.debug("store_memory(): Mem0 unavailable — skipping")
            return None

        try:
            # Mem0 expects a list of message dicts in conversation format
            messages = [
                {"role": "user",      "content": question},
                {"role": "assistant", "content": answer},
            ]
            self.mem0.add(messages, user_id=user_id)
            logger.info("store_memory(): stored turn for user_id='%s'", user_id)
        except Exception as e:
            # Never let Mem0 failure surface as an API error
            logger.warning(
                "store_memory(): failed for user_id='%s' (non-fatal): %s",
                user_id, str(e)[:120],
            )

        # Phase 8: persist the Q&A turn as conversation metadata in Postgres
        # (Conversation metadata → PostgreSQL). Best-effort; never blocks.
        # record_conversation is synchronous (psycopg2) and safe to call from
        # this worker-thread context.
        try:
            record_conversation(
                state.get("session_id", ""), user_id, question, answer,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("store_memory(): record_conversation failed (non-fatal): %s", e)

        # Pure side-effect node: no state fields are modified, so return None
        # (the LangGraph no-op). Returning {} here raises InvalidUpdateError
        # in LangGraph 0.2.x — see merge_results() for the rationale.
        return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Private: Context Window Assembly
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _build_context(
        self,
        retrieved_chunks: list[dict],
        kg_paths: list[dict],
    ) -> str:
        """
        Assemble the context window from retrieved chunks and KG paths.

        Context structure:
            [Source 1 | filename.pdf — Page 7]
            <chunk text>

            [Source 2 | filename.pdf — Page 12]
            <chunk text>

            [Knowledge Graph — Relationship Paths]
            head → relation → tail (confidence: 0.91)
            ...

        Design decisions:
          1. We use original_text (Phase 3 field) when available, not the
             enriched text. The [Context: ...] prefix added by the contextual
             enricher is useful for embedding/retrieval but would confuse the
             LLM if included verbatim in the generation prompt — it would
             see our own metadata as document content.

          2. Source indices are 1-based (matches [Source N] citation format
             instructed in the system prompt).

          3. KG paths are rendered as human-readable relationship strings
             rather than raw JSON dicts, making them easier for the LLM to
             cite and reason about.

          4. If kg_paths is empty (non-graph query route), the KG section
             is omitted entirely — no empty section headers.

        Args:
            retrieved_chunks: Phase 5 retrieval output.
            kg_paths:         Phase 6 graph traversal output (may be empty).

        Returns:
            Formatted context string ready to be inserted into the user prompt.
        """
        parts: list[str] = []

        # ── Document chunks ────────────────────────────────────────────────────
        for i, chunk in enumerate(retrieved_chunks, start=1):
            # Prefer original_text (pre-enrichment) for clean LLM context
            text = chunk.get("original_text") or chunk.get("text", "")
            filename = chunk.get("filename", "document")
            page = chunk.get("page", "?")

            source_block = CONTEXT_SOURCE_TEMPLATE.format(
                index=i,
                filename=filename,
                page=page,
                text=text,
            )
            parts.append(source_block)

        # ── Knowledge graph paths ──────────────────────────────────────────────
        if kg_paths:
            parts.append(KG_PATHS_HEADER)
            for path in kg_paths[:10]:   # cap at 10 paths to control context length
                # Render each path as a readable relationship string.
                # KG paths are plain dicts of Cypher RETURN columns (Phase 6
                # stringifies all values, so everything here is already a str).
                head     = path.get("head", "?")
                relation = path.get("relation", "?")
                tail     = path.get("tail", "?")
                conf     = path.get("confidence", "")
                conf_str = f" (confidence: {conf})" if conf and conf != "None" else ""
                parts.append(f"  {head} → {relation} → {tail}{conf_str}")

        return "\n\n".join(parts)

    def _build_user_prompt(
        self,
        question: str,
        context: str,
        memories: list[dict],
    ) -> str:
        """
        Build the full user prompt combining context, memories, and question.

        Structure:
            Context:
            <source blocks and KG paths>

            [User Context from Memory]    ← only if memories exist
            - User works as an ML engineer at Omdena.
            - User prefers concise answers with code examples.

            Question: <question>

        Memory injection:
          Memories are placed between the context and the question.
          This ordering follows the "lost in the middle" research finding
          (Liu et al., 2023): LLMs best attend to information at the start
          and end of the context window. Placing memories near the question
          (end of context) maximizes their influence on the answer.

        Args:
            question:  Raw user question.
            context:   Output of _build_context().
            memories:  List of {memory: str, score: float} dicts from Mem0.

        Returns:
            Complete user message string for the LLM.
        """
        memory_section = ""
        if memories:
            memory_lines = "\n".join(
                f"  - {m['memory']}"
                for m in memories[:5]   # top 5 most relevant memories
                if m.get("memory")
            )
            if memory_lines:
                memory_section = f"{USER_MEMORY_HEADER}{memory_lines}\n"

        return (
            f"Context:\n{context}"
            f"\n\n{memory_section}"
            f"Question: {question}"
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Private: Source Formatting
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _format_sources(self, retrieved_chunks: list[dict]) -> list[dict]:
        """
        Format retrieved chunks into the sources list for the API response.

        Sources are shown to end users in the frontend — they need to be
        clean, truncated, and structured for display.

        Source dict contract (matches AgentState.sources comment in state.py):
            {
                "index":        int,    # 1-based, matches [Source N] citations
                "text":         str,    # truncated to 300 chars for display
                "page":         int,
                "filename":     str,
                "doc_id":       str,
                "chunk_index":  int,
                "rerank_score": float,  # cross-encoder score from Phase 5
            }

        Text truncation:
          300 characters is enough for users to understand which part of
          the document is being cited without overwhelming the UI.
          Full text is available in retrieved_chunks in state for any
          internal use that needs it.

        Args:
            retrieved_chunks: Phase 5 output chunks.

        Returns:
            List of source dicts for client display.
        """
        sources = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            # Show original_text (without [Context: ...] prefix) to users
            display_text = chunk.get("original_text") or chunk.get("text", "")
            truncated = (
                display_text[:300] + "..."
                if len(display_text) > 300
                else display_text
            )
            sources.append({
                "index":        i,
                "text":         truncated,
                "page":         chunk.get("page"),
                "filename":     chunk.get("filename", ""),
                "doc_id":       chunk.get("doc_id", ""),
                "chunk_index":  chunk.get("chunk_index", 0),
                "rerank_score": round(chunk.get("rerank_score", 0.0), 4),
            })
        return sources


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Module-Level Singleton
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# One AnswerGenerator per process — LLM client, NLI model, and Mem0 client shared.
#
# Registered in orchestrator.py as:
#   g.add_node("generate",      generator.generate)
#   g.add_node("judge",         generator.judge)
#   g.add_node("memory_store",  generator.store_memory)
#   g.add_conditional_edges(
#       "judge",
#       generator.should_retry,
#       {"generate": "generate", "memory_store": "memory_store"},
#   )
generator = AnswerGenerator()
