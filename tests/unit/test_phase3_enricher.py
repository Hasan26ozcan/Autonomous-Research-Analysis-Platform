"""
tests/unit/test_phase3_enricher.py
====================================
Unit tests for the ContextualEnricher (Phase 3).

Test strategy:
  - No real LLM calls: _generate_context() is mocked throughout.
  - Tests verify the BEHAVIOR of the enricher, not LLM output quality.
  - Specifically tested:
      1. Output structure integrity (same length, same order, same doc_id)
      2. Enriched text format ("[Context: ...]")
      3. Graceful degradation when LLM fails
      4. Cache behavior (reuse without re-calling LLM)
      5. State integration (AgentState in → partial update out)
      6. Edge cases (empty input, single chunk, partial failures)

Run:
    pytest tests/unit/test_phase3_enricher.py -v
"""

import pytest
from unittest.mock import MagicMock, patch


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_chunk(
    text: str,
    chunk_index: int = 0,
    page: int = 1,
    doc_id: str = "doc_abc",
    filename: str = "test.pdf",
) -> dict:
    """
    Build a chunk dict exactly as PDFChunker produces it.
    Matches the output contract from app/services/chunker.py.
    """
    return {
        "text": text,
        "page": page,
        "filename": filename,
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "word_count": len(text.split()),
        "context_prepended": False,   # PDFChunker always sets this to False
    }


def make_chunks(n: int, doc_id: str = "doc_abc") -> list[dict]:
    """Build n chunks with predictable content."""
    return [
        make_chunk(
            text=" ".join([f"word{i}_{j}" for j in range(50)]),
            chunk_index=i,
            page=i + 1,
            doc_id=doc_id,
        )
        for i in range(n)
    ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Output Structure Tests
# These tests verify that the enricher never corrupts the chunk list.
# Structural integrity is critical because embed_chunks depends on the
# parallel correspondence between chunks[i] and embeddings[i].
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestOutputStructureIntegrity:

    def setup_method(self):
        from app.services.contextual_enricher import ContextualEnricher
        self.enricher = ContextualEnricher()
        # Mock LLM to avoid real API calls
        self.enricher.llm = MagicMock()
        self.enricher.llm.invoke.return_value = MagicMock(
            content="This chunk is from the introduction of a research paper."
        )

    def test_output_length_equals_input_length(self):
        """
        CRITICAL: output chunk count must ALWAYS equal input chunk count.
        embed_chunks relies on chunks[i] ↔ embeddings[i] correspondence.
        A length mismatch would corrupt ALL embeddings for this document.
        """
        chunks = make_chunks(5)
        result = self.enricher.enrich({"chunks": chunks})
        assert len(result["chunks"]) == 5

    def test_chunk_ordering_is_preserved(self):
        """
        Chunks must come out in the same order they went in.
        chunk_index must match position in the output list.
        """
        chunks = make_chunks(4)
        result = self.enricher.enrich({"chunks": chunks})
        for i, chunk in enumerate(result["chunks"]):
            assert chunk["chunk_index"] == i, (
                f"Chunk at position {i} has chunk_index={chunk['chunk_index']}. "
                "Order was corrupted."
            )

    def test_doc_id_preserved_on_all_chunks(self):
        """doc_id must not be modified — it's the document's identity key."""
        chunks = make_chunks(3, doc_id="my_doc_123")
        result = self.enricher.enrich({"chunks": chunks})
        for chunk in result["chunks"]:
            assert chunk["doc_id"] == "my_doc_123"

    def test_page_numbers_preserved(self):
        """Page numbers must survive enrichment — used in source citations."""
        chunks = make_chunks(3)
        result = self.enricher.enrich({"chunks": chunks})
        original_pages = [c["page"] for c in chunks]
        enriched_pages = [c["page"] for c in result["chunks"]]
        assert original_pages == enriched_pages

    def test_word_count_preserved(self):
        """word_count reflects the ORIGINAL chunk, not the enriched version."""
        chunks = make_chunks(2)
        result = self.enricher.enrich({"chunks": chunks})
        for orig, enriched in zip(chunks, result["chunks"]):
            assert enriched["word_count"] == orig["word_count"]

    def test_returns_dict_with_chunks_key(self):
        """
        LangGraph node must return a dict with key 'chunks'.
        LangGraph merges this into the running AgentState.
        """
        result = self.enricher.enrich({"chunks": make_chunks(1)})
        assert isinstance(result, dict)
        assert "chunks" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enriched Text Format Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEnrichedTextFormat:

    def setup_method(self):
        from app.services.contextual_enricher import ContextualEnricher
        self.enricher = ContextualEnricher()
        self.fake_context = "This chunk comes from the Results section of a flood prediction paper."
        self.enricher.llm = MagicMock()
        self.enricher.llm.invoke.return_value = MagicMock(content=self.fake_context)

    def test_enriched_text_starts_with_context_bracket(self):
        """
        Enriched text must start with '[Context: ' prefix.
        This format is expected by the generator for display stripping.
        """
        chunks = [make_chunk("The results show 23% improvement.")]
        result = self.enricher.enrich({"chunks": chunks})
        assert result["chunks"][0]["text"].startswith("[Context: ")

    def test_enriched_text_contains_original_text(self):
        """
        The original chunk text must appear in the enriched text.
        Embedding the enriched text without the original would defeat the purpose.
        """
        original = "The results show 23% improvement over baseline."
        chunks = [make_chunk(original)]
        result = self.enricher.enrich({"chunks": chunks})
        assert original in result["chunks"][0]["text"]

    def test_enriched_text_contains_llm_context(self):
        """LLM-generated context must appear in the enriched text."""
        chunks = [make_chunk("Some chunk text here.")]
        result = self.enricher.enrich({"chunks": chunks})
        assert self.fake_context in result["chunks"][0]["text"]

    def test_context_and_original_separated_by_blank_line(self):
        """
        Context and original text must be separated by double newline.
        Format: "[Context: ...]\n\n<original text>"
        """
        chunks = [make_chunk("Original text goes here.")]
        result = self.enricher.enrich({"chunks": chunks})
        enriched_text = result["chunks"][0]["text"]
        assert "\n\n" in enriched_text
        parts = enriched_text.split("\n\n", 1)
        assert len(parts) == 2
        assert parts[0].startswith("[Context:")
        assert "Original text goes here." in parts[1]

    def test_context_prepended_flag_is_true(self):
        """
        context_prepended must be True after enrichment.
        PDFChunker sets it False; enricher must set it True.
        This flag is read by downstream components to know enrichment status.
        """
        chunks = [make_chunk("Some text.")]
        result = self.enricher.enrich({"chunks": chunks})
        assert result["chunks"][0]["context_prepended"] is True

    def test_original_text_stored_separately(self):
        """
        original_text field must preserve the pre-enrichment text exactly.
        Used by generator to display clean text to users (not the context wrapper).
        """
        original = "The model achieves 97% accuracy on the test set."
        chunks = [make_chunk(original)]
        result = self.enricher.enrich({"chunks": chunks})
        assert result["chunks"][0]["original_text"] == original

    def test_context_text_stored_separately(self):
        """
        context_text field must store just the generated context (no chunk text).
        Useful for debugging enrichment quality without parsing the combined text.
        """
        chunks = [make_chunk("Some chunk.")]
        result = self.enricher.enrich({"chunks": chunks})
        assert result["chunks"][0]["context_text"] == self.fake_context


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Graceful Degradation Tests
# These tests verify that LLM failures never block document ingestion.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGracefulDegradation:

    def setup_method(self):
        from app.services.contextual_enricher import ContextualEnricher
        self.enricher = ContextualEnricher()

    def test_llm_exception_does_not_raise(self):
        """
        When the LLM raises any exception, enrichment must NOT propagate it.
        The ingestion pipeline must always complete.
        """
        self.enricher.llm = MagicMock()
        self.enricher.llm.invoke.side_effect = RuntimeError("OpenAI API timeout")

        chunks = make_chunks(2)
        # This must NOT raise
        result = self.enricher.enrich({"chunks": chunks})
        assert "chunks" in result

    def test_failed_chunk_uses_original_text(self):
        """
        A chunk whose LLM call fails must still appear in output with its
        original text unchanged. The chunk is not dropped.
        """
        original_text = "The loss function converges after 100 epochs."
        self.enricher.llm = MagicMock()
        self.enricher.llm.invoke.side_effect = ConnectionError("Network unreachable")

        chunks = [make_chunk(original_text)]
        result = self.enricher.enrich({"chunks": chunks})

        assert len(result["chunks"]) == 1
        assert result["chunks"][0]["text"] == original_text
        assert result["chunks"][0]["context_prepended"] is False

    def test_partial_failure_enriches_successful_chunks(self):
        """
        If LLM succeeds for chunks 0 and 2 but fails for chunk 1,
        output must have 3 chunks: enriched, original, enriched.
        No chunk is dropped.
        """
        fake_context = "Context from successful LLM call."
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise TimeoutError("LLM timeout on chunk 1")
            return MagicMock(content=fake_context)

        self.enricher.llm = MagicMock()
        self.enricher.llm.invoke.side_effect = side_effect

        chunks = make_chunks(3)
        result = self.enricher.enrich({"chunks": chunks})

        assert len(result["chunks"]) == 3
        # Chunk 0: enriched
        assert result["chunks"][0]["context_prepended"] is True
        # Chunk 1: failed — original text preserved
        assert result["chunks"][1]["context_prepended"] is False
        # Chunk 2: enriched
        assert result["chunks"][2]["context_prepended"] is True

    def test_empty_chunks_returns_empty_update(self):
        """
        When state has no chunks, return empty dict (no-op LangGraph update).
        This can happen if chunk_document produced no output (empty PDF).
        """
        result = self.enricher.enrich({"chunks": []})
        assert result == {}

    def test_missing_chunks_key_returns_empty_update(self):
        """
        When 'chunks' key is absent from state entirely, return empty dict.
        This protects against malformed state from upstream nodes.
        """
        result = self.enricher.enrich({})
        assert result == {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cache Behavior Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCacheBehavior:

    def setup_method(self):
        from app.services.contextual_enricher import ContextualEnricher
        self.enricher = ContextualEnricher()
        self.enricher.llm = MagicMock()
        self.enricher.llm.invoke.return_value = MagicMock(
            content="Cached context description."
        )

    def test_same_chunk_uses_cache_on_second_call(self):
        """
        Enriching the same (doc_id, chunk_index) twice must call LLM only once.
        Cache hit prevents redundant API calls on re-ingestion.
        """
        chunk = make_chunk("Identical content.", chunk_index=0, doc_id="doc1")

        self.enricher.enrich({"chunks": [chunk]})  # first: LLM called
        self.enricher.enrich({"chunks": [chunk]})  # second: cache hit

        assert self.enricher.llm.invoke.call_count == 1

    def test_different_chunks_each_call_llm(self):
        """
        Different (doc_id, chunk_index) combinations must each call the LLM.
        Cache key is (doc_id, chunk_index) — different means different entry.
        """
        chunk_a = make_chunk("Content A.", chunk_index=0, doc_id="doc1")
        chunk_b = make_chunk("Content B.", chunk_index=1, doc_id="doc1")
        chunk_c = make_chunk("Content C.", chunk_index=0, doc_id="doc2")

        self.enricher.enrich({"chunks": [chunk_a, chunk_b, chunk_c]})
        assert self.enricher.llm.invoke.call_count == 3

    def test_cache_size_grows_with_unique_chunks(self):
        """cache_size property must reflect number of unique (doc_id, chunk_index) pairs."""
        chunks = make_chunks(4, doc_id="doc1")
        self.enricher.enrich({"chunks": chunks})
        assert self.enricher.cache_size == 4

    def test_clear_cache_resets_to_zero(self):
        """After clear_cache(), the next enrich call must re-call the LLM."""
        chunks = make_chunks(2)
        self.enricher.enrich({"chunks": chunks})
        assert self.enricher.cache_size == 2

        self.enricher.clear_cache()
        assert self.enricher.cache_size == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Doc Anchor Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDocAnchor:

    def test_anchor_uses_first_chunk_text(self):
        """
        Doc anchor must come from the first chunk (page 1 = title/abstract).
        The first chunk provides the best document-level context.
        """
        from app.services.contextual_enricher import ContextualEnricher
        chunks = [
            make_chunk("First chunk text. " * 20, chunk_index=0),
            make_chunk("Second chunk text.", chunk_index=1),
        ]
        anchor = ContextualEnricher._build_doc_anchor(chunks)
        assert "First chunk text." in anchor
        assert "Second chunk text." not in anchor

    def test_anchor_truncates_to_400_words(self):
        """
        Anchor must not exceed 400 words to stay within LLM token budget.
        Combined with prompt and chunk text, total input must stay < 2000 tokens.
        """
        from app.services.contextual_enricher import ContextualEnricher
        long_text = " ".join([f"word{i}" for i in range(600)])
        chunks = [make_chunk(long_text)]
        anchor = ContextualEnricher._build_doc_anchor(chunks)
        assert len(anchor.split()) <= 400

    def test_anchor_returns_empty_string_for_no_chunks(self):
        """_build_doc_anchor with empty list must return empty string, not raise."""
        from app.services.contextual_enricher import ContextualEnricher
        anchor = ContextualEnricher._build_doc_anchor([])
        assert anchor == ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LangGraph Node Integration Test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLangGraphNodeIntegration:

    def test_enrich_chunks_node_function_delegates_correctly(self):
        """
        The module-level enrich_chunks() function must call
        contextual_enricher.enrich() and return its result.
        Tests that the node wrapper correctly delegates to the singleton.
        """
        from app.services import contextual_enricher as ce_module
        from app.services.contextual_enricher import enrich_chunks

        fake_output = {"chunks": [make_chunk("enriched text")]}
        mock_enricher = MagicMock()
        mock_enricher.enrich.return_value = fake_output

        original = ce_module.contextual_enricher
        ce_module.contextual_enricher = mock_enricher

        try:
            state = {"chunks": [make_chunk("original text")]}
            result = enrich_chunks(state)
            mock_enricher.enrich.assert_called_once_with(state)
            assert result == fake_output
        finally:
            ce_module.contextual_enricher = original

    def test_node_output_compatible_with_embed_chunks_input(self):
        """
        Integration test: enrich_chunks output must be valid input for embed_chunks.
        Specifically: result["chunks"] must be a list of dicts with "text" key.
        This is the contract between Phase 3 and Phase 2 (embed_chunks).
        """
        from app.services.contextual_enricher import ContextualEnricher
        from app.services.embedder import embed_chunks

        enricher = ContextualEnricher()
        enricher.llm = MagicMock()
        enricher.llm.invoke.return_value = MagicMock(content="Generated context.")

        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]] * 3)

        chunks = make_chunks(3)
        enrichment_result = enricher.enrich({"chunks": chunks})

        # The enriched chunks must work as input to embed_chunks
        with patch("app.services.embedder.embedder._model", mock_model):
            embedding_result = embed_chunks(enrichment_result)

        assert "embeddings" in embedding_result
        assert len(embedding_result["embeddings"]) == 3
