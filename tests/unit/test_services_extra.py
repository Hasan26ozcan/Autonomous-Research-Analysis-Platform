"""
tests/unit/test_services_extra.py
=================================
Extra coverage for app/services modules that already have Phase tests but
still leave specific branches unexercised:

  * bm25_index  — load_from_qdrant, empty-query short circuit, rank_bm25
                  ImportError, index_chunks node branches, size/is_empty
  * chunker     — empty-bytes guard, no-text / too-short pages, page_count,
                  real fitz extraction path, chunk_pdf / pdf_page_count wrappers
  * vector_store— delete_by_doc, get_collection_info, _ensure_collection create
                  branch, store_chunks empty-input node
  * embedder    — get_dimension

All external systems (Qdrant, fitz/PyMuPDF) are mocked; nothing touches disk
or the network beyond a tiny temp file for the *pdf path wrappers.
"""

import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BM25Index — remaining branches
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBM25Extra:
    def setup_method(self):
        from app.services.bm25_index import BM25Index
        self.index = BM25Index()

    def _chunks(self, texts, doc_id="d1"):
        return [{"text": t, "page": i + 1, "filename": "f.pdf",
                 "doc_id": doc_id, "chunk_index": i} for i, t in enumerate(texts)]

    def test_empty_query_tokens_short_circuits(self):
        self.index.add_chunks(self._chunks(["hello world", "foo bar"]))
        # Whitespace-only query tokenizes to [] → returns [] immediately.
        assert self.index.search("    ") == []

    def test_size_and_is_empty_properties(self):
        assert self.index.is_empty is True
        assert self.index.size == 0
        self.index.add_chunks(self._chunks(["one two three"]))
        assert self.index.is_empty is False
        assert self.index.size == 1

    def test_load_from_qdrant_builds_corpus(self):
        class _Rec:
            def __init__(self, payload): self.payload = payload
        calls = {"n": 0}

        def _scroll(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return [_Rec({"text": "alpha beta", "doc_id": "d1",
                              "page": 1, "chunk_index": 0, "filename": "f.pdf"})], "next"
            return [], None  # second page exhausts the scroll

        fake_client = MagicMock()
        fake_client.scroll.side_effect = _scroll
        self.index.load_from_qdrant(fake_client, "arap_docs")
        assert self.index.size == 1

    def test_rebuild_raises_when_rank_bm25_unavailable(self):
        self.index.add_chunks(self._chunks(["need real tokenization here"]))

        class _FakeRankBm25:
            def __getattr__(self, name):
                raise ImportError("rank_bm25 not installed")

        # Force `from rank_bm25 import BM25Okapi` to fail inside _rebuild().
        with patch.dict(sys.modules, {"rank_bm25": _FakeRankBm25()}):
            with pytest.raises(RuntimeError, match="rank_bm25"):
                self.index.search("tokenization")

    def test_index_chunks_node_noop_for_empty(self):
        from app.services.bm25_index import index_chunks
        # Empty chunks → warning + None, no corpus mutation.
        assert index_chunks({"chunks": []}) is None

    def test_index_chunks_node_removes_stale_entries(self):
        from app.services import bm25_index as mod
        from app.services.bm25_index import BM25Index, index_chunks

        original = mod.bm25_index
        fresh = BM25Index()
        fresh.add_chunks(self._chunks(["stale entry only here"], doc_id="docX"))
        mod.bm25_index = fresh
        try:
            # Re-indexing the same doc_id must remove the stale entry first.
            index_chunks({"chunks": self._chunks(["fresh content"], doc_id="docX")})
            # One stale removed, one fresh added → size back to 1.
            assert fresh.size == 1
        finally:
            mod.bm25_index = original


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# chunker — remaining branches (fitz / file wrappers / guards)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _FakePage:
    def __init__(self, text):
        self._text = text

    def get_text(self, mode):
        return self._text


class _FakeDoc:
    def __init__(self, pages):
        self._pages = pages
        self.page_count = len(pages)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self._pages)


class TestChunkerExtra:
    def setup_method(self):
        from app.services.chunker import PDFChunker
        self.chunker = PDFChunker(chunk_size=512, chunk_overlap=64)

    def test_chunk_raises_on_empty_bytes(self):
        import pytest as _pytest
        with _pytest.raises(ValueError, match="cannot be empty"):
            self.chunker.chunk(b"", "x.pdf")

    def test_chunk_returns_empty_when_no_text_extracted(self):
        with patch.object(self.chunker, "_extract_pages", return_value=[]):
            assert self.chunker.chunk(b"pdf", "x.pdf") == []

    def test_chunk_returns_empty_when_pages_too_short(self):
        # Pages with < 10 words are skipped by _extract_pages → no word stream.
        with patch.object(
            self.chunker, "_extract_pages",
            return_value=[{"page": 1, "text": "too short"}],
        ):
            assert self.chunker.chunk(b"pdf", "x.pdf") == []

    def test_page_count_via_fitz(self):
        import types
        fake_fitz = types.ModuleType("fitz")
        fake_fitz.open = lambda **kw: _FakeDoc([_FakePage("x")] * 3)
        with patch.dict(sys.modules, {"fitz": fake_fitz}):
            assert self.chunker.page_count(b"pdf") == 3

    def test_extract_pages_real_fitz_path(self):
        import types
        fake_fitz = types.ModuleType("fitz")
        fake_fitz.open = lambda **kw: _FakeDoc([_FakePage("word " * 30 + "more words here")])
        with patch.dict(sys.modules, {"fitz": fake_fitz}):
            pages = self.chunker._extract_pages(b"pdf")
        assert len(pages) == 1
        assert len(pages[0]["text"].split()) >= 10

    def test_chunk_pdf_reads_from_filesystem(self):
        from app.services.chunker import PDFChunker, chunk_pdf
        pages = [{"page": 1, "text": " ".join(f"word{i}" for i in range(60))}]
        with patch.object(PDFChunker, "_extract_pages", lambda self, b: pages):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(b"%PDF-1.4 fake")
                path = tmp.name
            try:
                chunks = chunk_pdf(path)
            finally:
                import os
                os.unlink(path)
        assert len(chunks) >= 1
        assert chunks[0]["doc_id"] == self.chunker.get_doc_id(b"%PDF-1.4 fake")

    def test_pdf_page_count_reads_from_filesystem(self):
        from app.services.chunker import PDFChunker, pdf_page_count
        with patch.object(PDFChunker, "page_count", lambda self, b: 9):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(b"%PDF-1.4 fake")
                path = tmp.name
            try:
                assert pdf_page_count(path) == 9
            finally:
                import os
                os.unlink(path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VectorStore — remaining branches (delete / info / ensure / empty node)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestVectorStoreExtra:
    def _make(self):
        from app.services.vector_store import VectorStore
        store = VectorStore()
        # Set _client directly so the lazy `client` property returns it without
        # triggering _ensure_collection (which would hit a real Qdrant).
        store._client = MagicMock()
        return store

    def test_delete_by_doc_calls_client(self):
        from qdrant_client.models import Filter
        store = self._make()
        assert store.delete_by_doc("docX") == 0
        store._client.delete.assert_called_once()
        # A filter scoped to the doc_id must have been passed.
        selector = store._client.delete.call_args.kwargs["points_selector"]
        assert isinstance(selector, Filter)

    def test_get_collection_info(self):
        store = self._make()
        info = MagicMock(points_count=10, vectors_count=10, status="green")
        store._client.get_collection.return_value = info
        res = store.get_collection_info()
        assert res == {"points_count": 10, "vectors_count": 10, "status": "green"}

    def test_ensure_collection_creates_when_absent(self):
        store = self._make()
        # Collection list does NOT contain the configured collection yet.
        store._client.get_collections.return_value = MagicMock(
            collections=[MagicMock(name="some_other")]
        )
        store._ensure_collection()
        store._client.create_collection.assert_called_once()
        store._client.create_payload_index.assert_called_once()

    def test_store_chunks_node_returns_none_when_empty(self):
        from app.services.vector_store import store_chunks
        # No chunks/embeddings → node short-circuits without touching Qdrant.
        assert store_chunks({"chunks": [], "embeddings": []}) is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Embedder — get_dimension
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_embedder_get_dimension():
    from app.services.embedder import Embedder
    embedder = Embedder()
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 384
    embedder._model = model  # bypass lazy load
    assert embedder.get_dimension() == 384
