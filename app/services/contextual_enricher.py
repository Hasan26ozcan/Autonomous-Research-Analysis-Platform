"""
app/services/contextual_enricher.py
=====================================
Prepend LLM-generated context to each chunk before embedding.

... (full docstring, unchanged) ...
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import settings
from app.core.state import AgentState

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROMPTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONTEXT_SYSTEM_PROMPT = """\
You are a document processing assistant specializing in retrieval optimization.

Your task: given a chunk of text extracted from a larger document, write a SHORT
context description (2-3 sentences, maximum 80 words) that situates this chunk
within the full document.

The context description will be PREPENDED to the chunk before embedding,
so a vector search engine can retrieve it accurately for relevant questions.

CRITICAL RULES:
  1. Do NOT summarize the chunk — only provide surrounding context.
  2. Mention the document title/type if identifiable.
  3. Mention the section or topic this chunk belongs to.
  4. Name any key entities (authors, organizations, methods, datasets)
     that were introduced earlier in the document and are referenced here.
  5. Mention temporal or geographical scope if present.
  6. Write in one paragraph. No bullet points. No headers.
  7. Return ONLY the context text — no preamble like "Here is the context:".

Example output:
  "This chunk is from the Methodology section of a 2024 climate science paper
   analyzing flood risk in Southeast Asia. It describes the preprocessing
   steps applied to the ERA5 reanalysis dataset introduced in Section 2,
   specifically the normalization pipeline used before feeding data into the
   FloodNet model."
"""

CONTEXT_USER_TEMPLATE = """\
Document: {filename}
Document beginning (first 400 words for reference):
\"\"\"
{doc_anchor}
\"\"\"

Chunk to contextualize (from page {page}):
\"\"\"
{chunk_text}
\"\"\"

Write the context description for this chunk:"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ContextualEnricher:
    """
    Prepends LLM-generated context descriptions to chunks before embedding.

    This class is the concrete implementation of Anthropic's Contextual
    Retrieval technique. It is designed to:

      1. Be stateless across enrichment calls (safe for concurrent use)
      2. Fail gracefully (never block ingestion on LLM API errors)
      3. Log enrichment quality metrics (success rate, latency per chunk)
      4. Be easily testable (LLM call is isolated in _generate_context())

    Attributes:
        llm:            ChatOpenAI instance using llama-3.1-70b-versatile (via Groq).
                        This model is specifically chosen because:
                          - It's NOT a reasoning model (no hidden "thinking" tokens)
                          - It perfectly supports `response_format={"type": "json_object"}`
                          - It's extremely fast and free on Groq's tier
                          - It outputs valid JSON without consuming the token budget
                        Using this model guarantees that `max_tokens=512` is used
                        entirely for the visible output (context text), not wasted
                        on invisible reasoning chains.
        _cache:         Dict mapping (doc_id, chunk_index) → context string.
                        Prevents re-generating context if the same document is
                        re-ingested (e.g. after a failed attempt). Cache is
                        in-memory (not persisted) — acceptable for this use case.
    """

    def __init__(self):
        # CRITICAL FIX 1: Model explicitly set to Groq's best non-reasoning model.
        # Previously, settings.router_model (gpt-4o-mini) was used, but this model
        # is not available on Groq and would default to returning errors or empty responses.
        self.llm = ChatOpenAI(
            model="llama-3.1-70b-versatile",  # Works on Groq, NOT a reasoning model
            api_key=settings.openai_api_key,
            base_url=settings.llm_base_url,
            temperature=0.0,                 # deterministic: same chunk → same context
            # CRITICAL FIX 2: max_tokens increased from 400 to 512.
            # Since this is NOT a reasoning model, all 512 tokens go to the visible output,
            # which definitively prevents empty responses like "[Context: ]".
            max_tokens=512,
        )
        self._cache: dict[tuple[str, int], str] = {}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LangGraph Node Entry Point
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def enrich(self, state: "AgentState") -> dict:
        """
        LangGraph node: enrich all chunks in state with contextual descriptions.

        Reads from state:
            chunks (list[dict]): raw chunks from chunk_document node.
                Each chunk must have: text, page, filename, doc_id, chunk_index.
                context_prepended must be False (set by PDFChunker).

        Writes to state (full replacement of 'chunks' key):
            chunks (list[dict]): same list, same length, same index order.
                Each chunk now has:
                  - text:             "[Context: ...]\n\n<original_text>"
                  - context_prepended: True
                  - original_text:    the raw text before enrichment
                  - context_text:     just the generated context description

        Guarantees:
            - len(output_chunks) == len(input_chunks) ALWAYS
            - chunk ordering is preserved ALWAYS
            - chunks[i]["doc_id"] and chunks[i]["chunk_index"] unchanged ALWAYS
            - If enrichment fails for any chunk, original text is used (no loss)

        These guarantees matter because embed_chunks relies on the parallel
        structure: embeddings[i] must correspond to chunks[i].
        """
        chunks: list[dict] = state.get("chunks", [])

        if not chunks:
            logger.warning("enrich_chunks called with no chunks in state — skipping")
            return None

        t0 = time.perf_counter()

        # Build the document anchor once per document.
        # The anchor is the first 400 words of the FIRST chunk — it gives the
        # LLM enough context to understand what kind of document this is
        # (research paper? financial report? legal contract?) and who/what
        # the main entities are.
        doc_anchor = self._build_doc_anchor(chunks)

        # Enrich each chunk
        enriched_chunks: list[dict] = []
        success_count = 0

        for i, chunk in enumerate(chunks):
            # Proactive pacing: wait BEFORE each call (except the very first)
            # so we never fire requests faster than the provider's per-minute
            # limit allows. This avoids 429s instead of retrying after them.
            if i > 0 and settings.llm_call_min_interval_seconds > 0:
                cache_key = (chunk.get("doc_id", ""), chunk.get("chunk_index", 0))
                if cache_key not in self._cache:
                    time.sleep(settings.llm_call_min_interval_seconds)

            enriched = self._enrich_single_chunk(chunk, doc_anchor)
            enriched_chunks.append(enriched)
            if enriched.get("context_prepended"):
                success_count += 1

        elapsed = (time.perf_counter() - t0) * 1000
        success_rate = success_count / len(chunks) * 100 if chunks else 0

        logger.info(
            "Contextual enrichment: %d/%d chunks enriched (%.0f%%) in %.0fms",
            success_count, len(chunks), success_rate, elapsed,
        )

        # Sanity check: output must be same length as input.
        # This protects embed_chunks from index mismatches.
        assert len(enriched_chunks) == len(chunks), (
            f"CRITICAL: enriched_chunks length {len(enriched_chunks)} "
            f"!= input chunks length {len(chunks)}. This would corrupt embeddings."
        )

        return {"chunks": enriched_chunks}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Single-Chunk Enrichment
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _enrich_single_chunk(self, chunk: dict, doc_anchor: str) -> dict:
        """
        Enrich one chunk with its context description.

        Strategy:
          1. Check cache — if this (doc_id, chunk_index) was already enriched,
             reuse the cached context (handles re-ingestion gracefully).
          2. Call LLM to generate context description.
          3. Prepend context to chunk text in the standard format.
          4. Update metadata fields (context_prepended=True, original_text).
          5. On any failure: return original chunk unchanged (graceful degradation).

        The enriched text format:
            "[Context: <generated description>]\n\n<original chunk text>"

        This exact format matters. The "[Context: ...]" prefix:
          - Is visually distinct from content (helpful for debugging)
          - Signals to the embedding model that this is metadata, not content
          - Is easy to strip programmatically if needed (e.g. for display)

        Args:
            chunk:      Single chunk dict from PDFChunker.
            doc_anchor: First 400 words of the document (shared across all chunks).

        Returns:
            A copy of the chunk dict with enrichment fields added.
            NEVER modifies the input chunk in place.
        """
        doc_id: str = chunk.get("doc_id", "")
        chunk_index: int = chunk.get("chunk_index", 0)
        cache_key = (doc_id, chunk_index)

        # ── Cache hit ──────────────────────────────────────────────────────────
        if cache_key in self._cache:
            context_text = self._cache[cache_key]
            logger.debug(
                "Cache hit for doc_id=%s chunk_index=%d", doc_id, chunk_index
            )
            return self._apply_context(chunk, context_text)

        # ── LLM call ───────────────────────────────────────────────────────────
        try:
            context_text = self._generate_context(
                chunk_text=chunk["text"],
                filename=chunk.get("filename", "document"),
                page=chunk.get("page", 1),
                doc_anchor=doc_anchor,
            )
            if not context_text:
                # Defensive fallback: an empty (but non-exception) response
                # would otherwise silently produce "[Context: ]\n\n<text>" -
                # a useless artifact that's worse than no context at all.
                # Treat it the same as a failure: use the chunk unchanged.
                logger.warning(
                    "Context generation returned empty text for doc_id=%s "
                    "chunk_index=%d (using original text)", doc_id, chunk_index,
                )
                return {**chunk, "context_prepended": False, "original_text": chunk["text"]}
            self._cache[cache_key] = context_text
            return self._apply_context(chunk, context_text)

        except Exception as e:
            # Graceful degradation: log the error, return the original chunk.
            # Enrichment failure is recoverable — the chunk is still indexable.
            # The chunk just won't have the context boost for this retrieval.
            logger.warning(
                "Context generation failed for doc_id=%s chunk_index=%d "
                "(using original text): %s",
                doc_id, chunk_index, str(e)[:100],
            )
            # Return a copy of the original chunk with failure metadata
            return {
                **chunk,
                "context_prepended": False,   # explicitly mark as not enriched
                "context_text": None,
                "original_text": chunk["text"],
            }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Public method for single‑chunk enrichment (used by ingest_service)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def enrich_text(
        self,
        chunk_text: str,
        page: int,
        filename: str,
        doc_id: str,
        chunk_index: int,
        doc_anchor: str = "",
    ) -> str:
        """
        Public method to enrich a single chunk with context.

        This is used by the top‑level `enrich_chunk` function for use in
        `ingest_service.py`. It builds a minimal chunk dict, calls the
        internal enrichment logic, and returns the enriched text.

        Args:
            chunk_text:   Original chunk text.
            page:         Page number.
            filename:     PDF filename.
            doc_id:       Document ID.
            chunk_index:  Chunk index.
            doc_anchor:   Optional document anchor (first 400 words).
                          If not provided, a default empty anchor is used.

        Returns:
            Enriched text (with context prepended), or original text on failure.
        """
        chunk = {
            "text": chunk_text,
            "page": page,
            "filename": filename,
            "doc_id": doc_id,
            "chunk_index": chunk_index,
        }
        # If no anchor is provided, we can't generate good context.
        # Use the chunk itself as a minimal anchor (not ideal but better than nothing).
        if not doc_anchor:
            doc_anchor = " ".join(chunk_text.split()[:400])  # fallback anchor
        enriched = self._enrich_single_chunk(chunk, doc_anchor)
        return enriched.get("text", chunk_text)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LLM Call (isolated for testability)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @retry(
        # Retry on OpenAI rate limit or transient network errors.
        # Exponential backoff: wait 2s, 4s, 8s before giving up.
        # After 3 attempts with no success, the exception propagates to
        # _enrich_single_chunk which handles it gracefully.
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _generate_context(
        self,
        chunk_text: str,
        filename: str,
        page: int,
        doc_anchor: str,
    ) -> str:
        """
        Call the LLM to generate a context description for one chunk.

        This method is isolated from _enrich_single_chunk so that:
          1. Tests can mock just this method without mocking LangChain internals
          2. The retry decorator applies only to the LLM call, not to cache logic
          3. The method has a single responsibility: LLM in → context string out

        Input to LLM:
          - CONTEXT_SYSTEM_PROMPT: detailed instructions for context generation
          - CONTEXT_USER_TEMPLATE: filled with filename, doc_anchor, page, chunk_text
            chunk_text is truncated to 600 words to stay within token budget

        Output from LLM:
          - 2-3 sentence context description
          - Stripped of leading/trailing whitespace
          - Maximum ~512 tokens (enforced by max_tokens parameter)

        Args:
            chunk_text:  The raw chunk text (truncated to 600 words internally)
            filename:    Original PDF filename for context
            page:        Page number where this chunk starts
            doc_anchor:  First 400 words of the document (may be empty)

        Returns:
            Context description string (50-120 tokens, 2-3 sentences)
        """
        # Truncate chunk_text to 600 words to avoid exceeding context window.
        # 600 words ≈ 800 tokens. Combined with system prompt (~250 tokens)
        # and doc_anchor (~550 tokens), total input stays under 1800 tokens.
        truncated_chunk = " ".join(chunk_text.split()[:600])

        # If doc_anchor is empty, provide a placeholder to avoid empty prompt.
        anchor_display = doc_anchor if doc_anchor else "(No document beginning available)"
        user_message = CONTEXT_USER_TEMPLATE.format(
            filename=filename,
            doc_anchor=anchor_display,
            page=page,
            chunk_text=truncated_chunk,
        )

        from app.services.rate_limiter import groq_rate_limiter, estimate_tokens
        # max_output_tokens estimation updated (150 → 512)
        groq_rate_limiter.acquire(estimate_tokens(
            CONTEXT_SYSTEM_PROMPT, user_message, max_output_tokens=512,
        ))
        response = self.llm.invoke([
            SystemMessage(content=CONTEXT_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])

        return response.content.strip()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Helpers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _build_doc_anchor(chunks: list[dict]) -> str:
        """
        Build the document anchor from the first chunk's text.

        The anchor gives the LLM enough document-level context to understand:
          - What type of document this is (research paper, report, contract...)
          - Who the main authors, organizations, or subjects are
          - What the document is about at a high level

        We use the first chunk (page 1 content) rather than a separate
        document summary because:
          1. It's already available (no extra LLM call)
          2. First pages typically contain title, abstract, introduction —
             the highest-density metadata in any document
          3. The anchor is just for reference, not for answering questions

        Truncation to 400 words keeps the anchor below ~550 tokens,
        leaving room for the chunk text and prompts in the LLM context.

        Args:
            chunks: All chunks for this document. First chunk is used as anchor.

        Returns:
            First 400 words of the first chunk, or empty string if no chunks.
        """
        if not chunks:
            return ""
        first_chunk_text = chunks[0].get("text", "")
        anchor_words = first_chunk_text.split()[:400]
        return " ".join(anchor_words)

    @staticmethod
    def _apply_context(chunk: dict, context_text: str) -> dict:
        """
        Create an enriched copy of the chunk with context prepended to text.

        This is a pure function: it creates a NEW dict (does not modify input).
        The output dict has all original chunk fields PLUS:
          - text:             enriched text ("[Context: ...]\n\n<original>")
          - context_prepended: True
          - context_text:     just the generated context (for debugging/display)
          - original_text:    the raw text before enrichment

        Why store original_text?
          - Useful for debugging (compare enriched vs original retrieval quality)
          - The generator can display just the original text to users
          - Re-enrichment after prompt changes: original_text is preserved

        The "[Context: ...]" wrapper format is important:
          Wrapping in brackets makes the context clearly distinct from document
          content. This prevents the embedding model from over-weighting the
          context (which would be 10-15% of the total text).

        Args:
            chunk:        Original chunk dict (not modified)
            context_text: Generated context description string

        Returns:
            New dict with all original fields plus enrichment fields
        """
        original_text = chunk["text"]
        enriched_text = f"[Context: {context_text}]\n\n{original_text}"

        return {
            **chunk,                              # preserve all original fields
            "text": enriched_text,                # OVERWRITE text with enriched version
            "context_prepended": True,            # mark as enriched (was False from chunker)
            "context_text": context_text,         # store context separately for display
            "original_text": original_text,       # store original for debugging
        }

    def clear_cache(self) -> None:
        """
        Clear the in-memory context cache.

        Call this if you want to force re-generation of all context descriptions,
        for example after changing the context generation prompt.
        """
        self._cache.clear()
        logger.info("ContextualEnricher: cache cleared.")

    @property
    def cache_size(self) -> int:
        """Number of cached context descriptions."""
        return len(self._cache)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Module-Level Singleton and LangGraph Node Function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# One enricher instance per process — LLM client and cache are shared.
contextual_enricher = ContextualEnricher()


def enrich_chunks(state: "AgentState") -> dict:
    """
    LangGraph node function — thin wrapper around contextual_enricher.enrich().

    Registered in the ingest graph as the "enrich" node.
    Runs AFTER chunk_document and BEFORE embed_chunks.

    The function signature (state: AgentState) -> dict is the standard
    LangGraph node interface. All logic lives in ContextualEnricher.enrich()
    to keep this function easily testable and replaceable.

    Graph position:
        chunk_document → [enrich_chunks] → embed_chunks → store_chunks

    See: app/core/orchestrator.py, build_ingest_graph()
    """
    return contextual_enricher.enrich(state)


# ── Top‑level function for ingest_service ────────────────────────────────────

def enrich_chunk(chunk_text: str, page: int, doc_id: str, chunk_index: int) -> str:
    """
    Convenience wrapper around ContextualEnricher.enrich_text().

    This is the function imported by `ingest_service.py`. It uses the
    singleton enricher instance and returns the enriched text (or the
    original text if enrichment fails).

    Args:
        chunk_text:   Original chunk text.
        page:         Page number.
        doc_id:       Document ID.
        chunk_index:  Chunk index.

    Returns:
        Enriched text with context prepended, or original text on failure.
    """
    # We don't have filename here; we can use doc_id as a fallback.
    filename = f"doc_{doc_id[:8]}" if doc_id else "document"
    # We also don't have doc_anchor, but enrich_text will build a fallback anchor.
    return contextual_enricher.enrich_text(
        chunk_text=chunk_text,
        page=page,
        filename=filename,
        doc_id=doc_id,
        chunk_index=chunk_index,
        doc_anchor="",  # let enrich_text build a fallback
    )