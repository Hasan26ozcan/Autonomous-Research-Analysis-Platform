"""
app/services/chunker.py
========================
PDF → overlapping text chunks with rich metadata.

Why chunking matters:
  LLMs have a context window limit — you cannot feed an entire document
  at once. Chunking splits the document into smaller pieces that fit in
  the context window while preserving enough local context for the LLM
  to understand each piece in isolation.

Chunking strategy used here: sliding window over words
  - chunk_size=512 words ≈ 700 tokens (fits cross-encoder max_length=512)
  - chunk_overlap=64 words ≈ 2-3 sentences
  - Overlap prevents losing information at chunk boundaries.
    Without overlap, a sentence split across two chunks would be partially
    invisible to each chunk's embedding — breaking retrieval for that fact.

Why PyMuPDF (fitz)?
  - Layout-aware: respects paragraph and column structure
  - Provides per-block page numbers (essential for source citation)
  - Fast C-based implementation: 10x faster than PyPDF2 on large PDFs
  - Handles scanned PDFs via OCR plugin (future extension point)

LangGraph integration:
  chunk_document() is registered as a LangGraph node in the ingest graph.
  It receives the full AgentState and returns a partial update dict:
    {"chunks": [...], "doc_id": "abc123", "chunk_count": 84}

Example output chunk:
    {
        "text": "The proposed method achieves state-of-the-art results...",
        "page": 7,
        "filename": "research_paper.pdf",
        "doc_id": "a3f8b12c",
        "chunk_index": 23,
        "word_count": 512,
        "context_prepended": False   # set to True after contextual enrichment
    }
"""

from __future__ import annotations

import hashlib
import io
import re
import logging
from typing import TYPE_CHECKING

from app.core.config import settings

if TYPE_CHECKING:
    from app.core.state import AgentState

logger = logging.getLogger(__name__)


class PDFChunker:
    """
    Converts raw PDF bytes into a list of overlapping text chunks.

    The chunking pipeline:
      1. Extract text from each page using PyMuPDF
      2. Clean extracted text (normalize whitespace, remove control chars)
      3. Concatenate all pages into a stream of (word, page_number) pairs
      4. Slide a window of chunk_size words with chunk_overlap step
      5. Attach metadata (page, filename, doc_id, chunk_index) to each chunk

    Keeping (word, page_number) pairs through the pipeline means each
    chunk inherits the page number of its MAJORITY page — this is the
    page number shown in source citations to users.
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ):
        """
        Args:
            chunk_size:    Words per chunk. 512 words ≈ 700 tokens.
            chunk_overlap: Words shared between consecutive chunks.
                           Must be less than chunk_size.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Step size = how many words we advance between chunks.
        # With chunk_size=512 and overlap=64, step=448.
        # Each new chunk shares 64 words with the previous one.
        self.step = chunk_size - chunk_overlap

    # ── Public API ─────────────────────────────────────────────────────────────

    def chunk(self, pdf_bytes: bytes, filename: str = "document.pdf") -> list[dict]:
        """
        Main entry point. Convert raw PDF bytes into overlapping text chunks.

        Args:
            pdf_bytes: Raw bytes of the PDF file (from file.read())
            filename:  Original filename for metadata and source attribution

        Returns:
            List of chunk dicts, each containing:
              text, page, filename, doc_id, chunk_index, word_count,
              context_prepended

        Raises:
            RuntimeError: If PyMuPDF is not installed
            ValueError:   If pdf_bytes is empty or not a valid PDF
        """
        if not pdf_bytes:
            raise ValueError("pdf_bytes cannot be empty")

        # Generate a stable doc_id from the PDF content.
        # SHA-256 hash means the same file always gets the same doc_id —
        # uploading the same PDF twice is idempotent (no duplicate chunks).
        doc_id = hashlib.sha256(pdf_bytes).hexdigest()[:16]

        # Extract text page by page
        pages = self._extract_pages(pdf_bytes)
        if not pages:
            logger.warning("No text extracted from PDF: %s", filename)
            return []

        # Build a flat stream of (word, page_number) tuples
        word_stream = self._build_word_stream(pages)
        if not word_stream:
            logger.warning("No words found after extraction: %s", filename)
            return []

        # Slide window over the word stream to produce chunks
        chunks = self._slide_window(
            word_stream=word_stream,
            doc_id=doc_id,
            filename=filename,
        )

        logger.info(
            "Chunked '%s': %d pages → %d chunks (doc_id=%s)",
            filename, len(pages), len(chunks), doc_id,
        )
        return chunks

    def get_doc_id(self, pdf_bytes: bytes) -> str:
        """Return the stable doc_id for a PDF without full chunking."""
        return hashlib.sha256(pdf_bytes).hexdigest()[:16]

    # ── Private: PDF text extraction ───────────────────────────────────────────

    def _extract_pages(self, pdf_bytes: bytes) -> list[dict]:
        """
        Extract text from each page of the PDF using PyMuPDF.

        PyMuPDF's get_text("text") mode returns plain text with newlines
        between blocks. This preserves paragraph structure better than
        "html" or "dict" modes for our purposes.

        Returns:
            List of {"page": int, "text": str} dicts, one per page.
            Pages with no extractable text are skipped.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise RuntimeError(
                "PyMuPDF not installed. Run: pip install pymupdf"
            )

        pages = []
        with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
            for page_num, page in enumerate(doc, start=1):
                raw_text = page.get_text("text")
                cleaned = self._clean_text(raw_text)
                if len(cleaned.split()) >= 10:  # skip near-empty pages
                    pages.append({"page": page_num, "text": cleaned})

        return pages

    # ── Private: text cleaning ─────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Normalize extracted PDF text for embedding quality.

        Operations:
          1. Collapse multiple whitespace/newlines into single spaces
             (PDFs often have excessive line breaks between words)
          2. Remove non-printable control characters (common in scanned PDFs)
          3. Strip leading/trailing whitespace

        Note: We keep standard Latin characters and extended Unicode
        (accented chars, Turkish chars like ş ğ ı ö ü ç) for multilingual support.
        """
        # Step 1: Replace all whitespace sequences (spaces, tabs, newlines) with single space
        text = re.sub(r"\s+", " ", text)

        # Step 2: Remove ASCII control characters (0x00-0x1F, 0x7F) except space
        # Keep extended Latin (0xC0-0x024F) for multilingual support
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

        # Step 3: Remove sequences of 3+ repeated special chars (OCR artifacts)
        # e.g. "------", "......" from table borders in PDFs
        text = re.sub(r"([^\w\s])\1{2,}", "", text)

        return text.strip()

    # ── Private: word stream ───────────────────────────────────────────────────

    @staticmethod
    def _build_word_stream(pages: list[dict]) -> list[tuple[str, int]]:
        """
        Convert pages into a flat list of (word, page_number) tuples.

        Keeping page numbers paired with individual words allows each chunk
        to inherit the correct page number later, even when a chunk spans
        a page boundary.

        Example output (truncated):
            [("The", 1), ("proposed", 1), ("method", 1), ..., ("results", 2), ...]
        """
        stream: list[tuple[str, int]] = []
        for page in pages:
            for word in page["text"].split():
                if word:  # skip empty strings from double spaces
                    stream.append((word, page["page"]))
        return stream

    # ── Private: sliding window ────────────────────────────────────────────────

    def _slide_window(
        self,
        word_stream: list[tuple[str, int]],
        doc_id: str,
        filename: str,
    ) -> list[dict]:
        """
        Produce overlapping chunks by sliding a window over the word stream.

        Window mechanics:
          - Window size: self.chunk_size words
          - Advance step: self.step words (= chunk_size - chunk_overlap)
          - Each chunk shares self.chunk_overlap words with the next chunk

        Page assignment:
          Each chunk is assigned the page number of its MIDDLE word.
          This is more accurate than using the first or last word's page
          when chunks span page boundaries.

        Minimum chunk size:
          Chunks with fewer than 30 words are discarded. These are usually
          isolated headers, footers, or page numbers that add noise.

        Returns:
            List of chunk dicts with all metadata fields populated.
        """
        chunks: list[dict] = []
        total_words = len(word_stream)
        chunk_index = 0
        start = 0

        while start < total_words:
            end = min(start + self.chunk_size, total_words)
            window = word_stream[start:end]

            # Skip windows that are too short to be meaningful
            if len(window) < 30:
                break

            # Build the chunk text by joining words
            chunk_text = " ".join(word for word, _ in window)

            # Assign the page number from the middle word of the window.
            # Using the middle (not first) gives the most representative page
            # when a chunk spans a page boundary.
            middle_idx = len(window) // 2
            page_number = window[middle_idx][1]

            chunks.append({
                "text": chunk_text,
                "page": page_number,
                "filename": filename,
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "word_count": len(window),
                "context_prepended": False,  # set True by contextual_enricher
            })

            chunk_index += 1

            # If we've reached the end, stop
            if end == total_words:
                break

            start += self.step

        return chunks


# ── LangGraph node function ────────────────────────────────────────────────────

# Module-level chunker singleton
_chunker = PDFChunker()


def chunk_document(state: "AgentState") -> dict:
    """
    LangGraph node function for the ingest pipeline.

    Reads from state:
        raw_bytes (bytes): raw PDF content
        filename  (str):   original filename

    Writes to state (partial update):
        chunks      (list[dict]): all text chunks with metadata
        doc_id      (str):        stable content-based document identifier
        chunk_count (int):        total number of chunks produced

    This function is pure in the LangGraph sense — it reads from state,
    computes a result, and returns ONLY the fields it changed.
    It does NOT modify any other state fields.
    """
    raw_bytes: bytes | None = state.get("raw_bytes")
    filename: str = state.get("filename") or "document.pdf"

    if not raw_bytes:
        logger.error("chunk_document called with no raw_bytes in state")
        return {"error": "No PDF bytes in state", "chunks": [], "chunk_count": 0}

    chunks = _chunker.chunk(raw_bytes, filename=filename)
    doc_id = _chunker.get_doc_id(raw_bytes)

    return {
        "chunks": chunks,
        "doc_id": doc_id,
        "chunk_count": len(chunks),
    }
