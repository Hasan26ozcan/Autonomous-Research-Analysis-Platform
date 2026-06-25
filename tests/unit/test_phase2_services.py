"""
tests/unit/test_phase2_services.py
====================================
Unit tests for Phase 2 core services:
  - PDFChunker
  - Embedder
  - VectorStore
  - BM25Index

Design philosophy:
  - No external dependencies: Qdrant, OpenAI, Neo4j are all mocked
  - Tests verify behavior, not implementation details
  - Each test has a single clear assertion about one thing
  - Tests are fast (< 1 second each) and can run offline

Run with:
    pytest tests/unit/test_phase2_services.py -v
    pytest tests/unit/test_phase2_services.py -v --tb=short  # shorter tracebacks
"""

import pytest
from unittest.mock import MagicMock, patch


# =============================================================================
# PDFChunker tests
# =============================================================================

class TestPDFChunker:
    """Tests for app/services/chunker.py"""

    def setup_method(self):
        from app.services.chunker import PDFChunker
        self.chunker = PDFChunker(chunk_size=10, chunk_overlap=3)

    def test_raises_on_invalid_params(self):
        """chunk_overlap must be less than chunk_size."""
        from app.services.chunker import PDFChunker
        with pytest.raises(ValueError, match="chunk_overlap"):
            PDFChunker(chunk_size=10, chunk_overlap=10)

    def test_clean_text_collapses_whitespace(self):
        """Multiple spaces/newlines become a single space."""
        raw = "Hello   World\n\nThis  is\ta  test."
        result = self.chunker._clean_text(raw)
        assert "  " not in result
        assert "\n" not in result
        assert result == "Hello World This is a test."

    def test_build_word_stream_pairs_words_with_pages(self):
        """Every word in the stream must carry its page number."""
        pages = [
            {"page": 1, "text": "hello world"},
            {"page": 2, "text": "foo bar"},
        ]
        stream = self.chunker._build_word_stream(pages)
        assert ("hello", 1) in stream
        assert ("world", 1) in stream
        assert ("foo", 2) in stream
        assert ("bar", 2) in stream

    def test_slide_window_produces_overlapping_chunks(self):
        """Consecutive chunks must share chunk_overlap words."""
        # 30 words, chunk_size=10, overlap=3 → step=7
        words = [f"word{i}" for i in range(30)]
        pages = [{"page": 1, "text": " ".join(words)}]
        stream = self.chunker._build_word_stream(pages)
        chunks = self.chunker._slide_window(stream, doc_id="test", filename="test.pdf")

        assert len(chunks) > 1

        # The last 3 words of chunk[0] should appear at start of chunk[1]
        chunk0_words = chunks[0]["text"].split()
        chunk1_words = chunks[1]["text"].split()
        overlap = set(chunk0_words[-3:]) & set(chunk1_words[:3])
        assert len(overlap) >= 1, "Expected overlapping words between consecutive chunks"

    def test_chunks_have_required_metadata_keys(self):
        """Every chunk must carry text, page, filename, doc_id, chunk_index."""
        words = [f"word{i}" for i in range(50)]
        pages = [{"page": 3, "text": " ".join(words)}]
        stream = self.chunker._build_word_stream(pages)
        chunks = self.chunker._slide_window(stream, doc_id="abc123", filename="test.pdf")

        required_keys = {"text", "page", "filename", "doc_id", "chunk_index", "word_count"}
        for chunk in chunks:
            assert required_keys.issubset(chunk.keys()), (
                f"Chunk missing keys: {required_keys - chunk.keys()}"
            )

    def test_doc_id_is_deterministic(self):
        """Same PDF bytes must always produce the same doc_id."""
        pdf_bytes = b"fake pdf content for testing"
        id1 = self.chunker.get_doc_id(pdf_bytes)
        id2 = self.chunker.get_doc_id(pdf_bytes)
        assert id1 == id2

    def test_different_pdfs_produce_different_doc_ids(self):
        """Different content must produce different doc_ids."""
        id1 = self.chunker.get_doc_id(b"content one")
        id2 = self.chunker.get_doc_id(b"content two")
        assert id1 != id2

    def test_chunk_document_node_returns_correct_keys(self):
        """LangGraph node must return chunks, doc_id, chunk_count."""
        from app.services.chunker import chunk_document

        fake_pages = [{"page": 1, "text": " ".join([f"word{i}" for i in range(100)])}]

        with patch("app.services.chunker.PDFChunker._extract_pages", return_value=fake_pages):
            result = chunk_document({
                "raw_bytes": b"fake pdf",
                "filename": "test.pdf",
            })

        assert "chunks" in result
        assert "doc_id" in result
        assert "chunk_count" in result
        assert result["chunk_count"] == len(result["chunks"])

    def test_chunk_document_handles_missing_bytes(self):
        """Node must return error gracefully if raw_bytes is missing."""
        from app.services.chunker import chunk_document
        result = chunk_document({"filename": "test.pdf"})
        assert "error" in result
        assert result["chunk_count"] == 0


# =============================================================================
# Embedder tests
# =============================================================================

class TestEmbedder:
    """Tests for app/services/embedder.py"""

    def setup_method(self):
        from app.services.embedder import Embedder
        # Use a tiny model for testing — don't download full MiniLM in CI
        self.embedder = Embedder()

    def test_embed_returns_correct_count(self):
        """embed() must return one vector per input text."""
        from app.services.embedder import Embedder
        embedder = Embedder()

        # Mock the internal model to avoid downloading anything
        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.get_sentence_embedding_dimension.return_value = 3
        embedder._model = mock_model

        texts = ["hello world", "foo bar"]
        result = embedder.embed(texts)

        assert len(result) == 2

    def test_embed_empty_list_returns_empty(self):
        """embed([]) must return [] without calling the model."""
        from app.services.embedder import Embedder
        embedder = Embedder()
        embedder._model = MagicMock()  # if model is called, test fails

        result = embedder.embed([])
        assert result == []
        embedder._model.encode.assert_not_called()

    def test_embed_single_returns_one_vector(self):
        """embed_single() must return a flat list, not a list of lists."""
        from app.services.embedder import Embedder
        embedder = Embedder()

        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        embedder._model = mock_model

        result = embedder.embed_single("test sentence")
        assert isinstance(result, list)
        assert not isinstance(result[0], list), "embed_single should return flat list"

    def test_embed_chunks_node_returns_embeddings_key(self):
        """LangGraph node must return {'embeddings': [...]}."""
        from app.services.embedder import embed_chunks
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        with patch("app.services.embedder.embedder._model", mock_model):
            result = embed_chunks({
                "chunks": [
                    {"text": "hello world"},
                    {"text": "foo bar"},
                ]
            })

        assert "embeddings" in result
        assert len(result["embeddings"]) == 2

    def test_embed_chunks_node_handles_empty_chunks(self):
        """Node must return empty embeddings list gracefully."""
        from app.services.embedder import embed_chunks
        result = embed_chunks({"chunks": []})
        assert result == {"embeddings": []}


# =============================================================================
# VectorStore tests
# =============================================================================

class TestVectorStore:
    """Tests for app/services/vector_store.py"""

    def _make_store(self):
        """Create a VectorStore with a mocked Qdrant client."""
        from app.services.vector_store import VectorStore
        store = VectorStore()

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        store._client = mock_client
        store._ensure_collection = MagicMock()
        return store, mock_client

    def test_upsert_raises_on_length_mismatch(self):
        """upsert() must raise ValueError if chunks and embeddings differ in length."""
        store, _ = self._make_store()
        with pytest.raises(ValueError, match="same length"):
            store.upsert(
                chunks=[{"text": "one"}, {"text": "two"}],
                embeddings=[[0.1, 0.2]],  # only 1 embedding for 2 chunks
            )

    def test_upsert_returns_zero_for_empty_input(self):
        """upsert() on empty lists must return 0 without calling Qdrant."""
        store, mock_client = self._make_store()
        result = store.upsert(chunks=[], embeddings=[])
        assert result == 0
        mock_client.upsert.assert_not_called()

    def test_upsert_calls_qdrant_with_correct_collection(self):
        """upsert() must call client.upsert with the configured collection name."""
        store, mock_client = self._make_store()
        store.upsert(
            chunks=[{"text": "hello", "page": 1, "filename": "f.pdf",
                     "doc_id": "abc", "chunk_index": 0, "word_count": 5}],
            embeddings=[[0.1, 0.2, 0.3]],
        )
        mock_client.upsert.assert_called_once()
        call_kwargs = mock_client.upsert.call_args.kwargs
        assert call_kwargs["collection_name"] == store.collection

    def test_search_applies_doc_id_filter(self):
        """search() with doc_id must pass a filter to Qdrant."""
        from qdrant_client.models import Filter
        store, mock_client = self._make_store()
        mock_client.search.return_value = []

        store.search(query_vector=[0.1, 0.2], doc_id="abc123", top_k=5)

        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["query_filter"] is not None

    def test_search_no_filter_when_doc_id_is_none(self):
        """search() without doc_id must pass query_filter=None."""
        store, mock_client = self._make_store()
        mock_client.search.return_value = []

        store.search(query_vector=[0.1, 0.2], doc_id=None, top_k=5)

        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["query_filter"] is None

    def test_store_chunks_node_calls_upsert(self):
        """store_chunks LangGraph node must trigger upsert side effect."""
        from app.services import vector_store as vs_module

        mock_store = MagicMock()
        original = vs_module.vector_store
        vs_module.vector_store = mock_store

        try:
            from app.services.vector_store import store_chunks
            store_chunks({
                "chunks": [{"text": "hello"}],
                "embeddings": [[0.1, 0.2]],
            })
            mock_store.upsert.assert_called_once()
        finally:
            vs_module.vector_store = original


# =============================================================================
# BM25Index tests
# =============================================================================

class TestBM25Index:
    """Tests for app/services/bm25_index.py"""

    def setup_method(self):
        from app.services.bm25_index import BM25Index
        self.index = BM25Index()

    def _make_chunks(self, texts: list[str], doc_id: str = "doc1") -> list[dict]:
        return [
            {"text": t, "page": i + 1, "filename": "test.pdf",
             "doc_id": doc_id, "chunk_index": i}
            for i, t in enumerate(texts)
        ]

    def test_empty_index_returns_no_results(self):
        """Searching an empty index must return []."""
        result = self.index.search("anything")
        assert result == []

    def test_exact_term_match_returns_results(self):
        """Exact keyword match must appear in results."""
        chunks = self._make_chunks([
            "The quick brown fox jumps over the lazy dog",
            "Python is a programming language for machine learning",
            "Qdrant is a vector database for similarity search",
        ])
        self.index.add_chunks(chunks)
        results = self.index.search("vector database similarity")
        assert len(results) > 0
        assert any("Qdrant" in r["text"] for r in results)

    def test_rare_term_ranks_higher_than_common_term(self):
        """BM25 IDF means rare terms score higher than common 'the', 'is', etc."""
        chunks = self._make_chunks([
            "HNSW is an efficient algorithm for approximate nearest neighbor search",
            "The cat sat on the mat and the dog sat on the log",
        ])
        self.index.add_chunks(chunks)
        results = self.index.search("HNSW algorithm")
        # The HNSW chunk should rank first
        assert "HNSW" in results[0]["text"]

    def test_doc_id_filter_restricts_results(self):
        """search() with doc_id must only return chunks from that document."""
        chunks_a = self._make_chunks(["machine learning neural networks"], doc_id="doc_a")
        chunks_b = self._make_chunks(["machine learning deep learning"], doc_id="doc_b")
        self.index.add_chunks(chunks_a)
        self.index.add_chunks(chunks_b)

        results = self.index.search("machine learning", doc_id="doc_a")
        assert all(r["doc_id"] == "doc_a" for r in results)

    def test_scores_are_normalized_to_unit_range(self):
        """All returned scores must be in [0, 1] range."""
        chunks = self._make_chunks([
            "neural network transformer attention",
            "convolutional neural network image classification",
            "recurrent neural network sequence modeling",
        ])
        self.index.add_chunks(chunks)
        results = self.index.search("neural network")
        for r in results:
            assert 0.0 <= r["score"] <= 1.0, (
                f"Score {r['score']} out of [0, 1] range"
            )

    def test_remove_by_doc_removes_correct_chunks(self):
        """After remove_by_doc('doc_a'), searching must not return doc_a chunks."""
        chunks_a = self._make_chunks(["unique term xylophone doc_a only"], doc_id="doc_a")
        chunks_b = self._make_chunks(["another document with different content"], doc_id="doc_b")
        self.index.add_chunks(chunks_a)
        self.index.add_chunks(chunks_b)

        removed = self.index.remove_by_doc("doc_a")
        assert removed == 1

        results = self.index.search("xylophone")
        assert all(r["doc_id"] != "doc_a" for r in results)

    def test_index_size_updates_after_adding(self):
        """size property must reflect current corpus length."""
        assert self.index.size == 0
        self.index.add_chunks(self._make_chunks(["text one", "text two", "text three"]))
        assert self.index.size == 3

    def test_results_have_required_keys(self):
        """Every search result must have text, page, doc_id, score, source."""
        self.index.add_chunks(self._make_chunks(["hello world foo bar baz"]))
        results = self.index.search("hello")
        required = {"text", "page", "doc_id", "score", "source"}
        for r in results:
            assert required.issubset(r.keys())

    def test_bm25_source_field_is_correct(self):
        """Results from BM25 search must have source='bm25'."""
        self.index.add_chunks(self._make_chunks(["test content for bm25"]))
        results = self.index.search("test content")
        assert all(r["source"] == "bm25" for r in results)

    def test_index_chunks_node_adds_to_corpus(self):
        """LangGraph node must increase corpus size."""
        from app.services import bm25_index as bm25_module
        from app.services.bm25_index import BM25Index

        fresh_index = BM25Index()
        original = bm25_module.bm25_index
        bm25_module.bm25_index = fresh_index

        try:
            from app.services.bm25_index import index_chunks
            index_chunks({
                "chunks": [
                    {"text": "hello world", "page": 1, "filename": "f.pdf",
                     "doc_id": "abc", "chunk_index": 0},
                    {"text": "foo bar baz", "page": 2, "filename": "f.pdf",
                     "doc_id": "abc", "chunk_index": 1},
                ]
            })
            assert fresh_index.size == 2
        finally:
            bm25_module.bm25_index = original
