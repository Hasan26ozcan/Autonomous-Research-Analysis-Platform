"""
app/services/embedder.py
=========================
Text → dense float vectors for semantic similarity search.

What is embedding?
  Embedding maps text into a high-dimensional vector space where semantically
  similar texts are close to each other (measured by cosine similarity).
  "The patient has diabetes" and "The individual suffers from high blood sugar"
  would be near each other, while "The stock market crashed" would be far away.
  This is the foundation that makes semantic RAG retrieval work.

Model hierarchy (by quality, 2026):
  Tier 1 — Qwen/Qwen3-Embedding-4B (2560 dims)
    SOTA on MTEB leaderboard as of mid-2026. Needs GPU (4-8GB VRAM).
    Best choice for production if GPU is available.

  Tier 2 — sentence-transformers/all-MiniLM-L6-v2 (384 dims)
    Default in ARAP. Excellent for English text on CPU.
    ~80ms per batch of 32 chunks on a modern CPU.
    Well-tested across millions of production deployments.

  Tier 3 — text-embedding-3-small (1536 dims, OpenAI API)
    Hosted option. No local model download, but costs money per token
    and adds network latency. Not used by default.

Why normalized embeddings?
  normalize_embeddings=True divides every vector by its L2 norm,
  making all vectors unit length. This means:
    cosine_similarity(a, b) = dot_product(a, b)
  Qdrant's cosine distance becomes equivalent to negative dot product,
  which is faster to compute. It also makes scores consistently in [-1, 1].

Lazy loading:
  The model is downloaded from HuggingFace Hub on first use (~90MB for MiniLM).
  We use lazy initialization (@property) so importing this module does NOT
  trigger a download — only the first actual embed() call does.

LangGraph integration:
  embed_chunks() is registered as a node in the ingest graph.
  It receives state with chunks and returns embeddings.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from app.core.config import settings

if TYPE_CHECKING:
    from app.core.state import AgentState

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wraps sentence-transformers for local embedding inference.

    Supports any model on the sentence-transformers hub.
    Switch model by changing EMBEDDING_MODEL in .env — no code changes needed.

    Thread safety:
        SentenceTransformer models are NOT thread-safe for concurrent encode()
        calls on CPU. In production, use one Embedder per Celery worker process,
        or add a threading.Lock() around encode() if needed.
    """

    def __init__(
        self,
        model_name: str = settings.embedding_model,
        device: str | None = None,
    ):
        """
        Args:
            model_name: HuggingFace model ID or local path.
                        Default: settings.embedding_model
            device:     "cpu", "cuda", "mps", or None (auto-detect).
                        None lets sentence-transformers pick the best available.
        """
        self.model_name = model_name
        self.device = device
        self._model = None  # lazy-loaded on first embed() call

    @property
    def model(self):
        """
        Lazy-load the sentence-transformers model.

        Why lazy?
          - Importing this module (at startup) should be instant
          - Model loading takes 1-5 seconds and downloads ~90MB on first run
          - Only load when actually needed (first embed() call)

        The model is cached in self._model after first load,
        so subsequent calls are instant.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Loading embedding model '%s' (first call, may download)...",
                self.model_name,
            )
            t0 = time.perf_counter()
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                "Embedding model loaded in %.0fms. "
                "Dimension: %d. Device: %s",
                elapsed,
                self._model.get_sentence_embedding_dimension(),
                self._model.device,
            )
        return self._model

    # ── Core embedding methods ─────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts into dense float vectors.

        Processing:
          1. Batch texts into groups of batch_size (default 32)
          2. Tokenize each batch (truncate to model max_length if needed)
          3. Run forward pass through the transformer
          4. Mean-pool the token embeddings into one vector per text
          5. L2-normalize each vector (for cosine similarity = dot product)

        Args:
            texts: List of strings to embed. Can be empty (returns []).
                   Very long texts are automatically truncated to model max_length
                   (usually 512 tokens for MiniLM, 8192 for larger models).

        Returns:
            List of float vectors, one per input text.
            All vectors have length settings.embedding_dim (e.g. 384 for MiniLM).
            All vectors are L2-normalized (unit length).

        Performance (MiniLM-L6-v2 on CPU):
            ~80ms for 32 texts, ~2.5ms per text.
            Batching amortizes tokenization overhead significantly.
        """
        if not texts:
            return []

        t0 = time.perf_counter()

        vectors = self.model.encode(
            texts,
            batch_size=32,              # process 32 texts at once (memory vs speed tradeoff)
            normalize_embeddings=True,  # L2 normalize → cosine sim = dot product
            show_progress_bar=False,    # suppress tqdm output in production
            convert_to_numpy=True,      # return numpy array (faster than torch tensor)
        )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Embedded %d texts in %.0fms (%.1fms/text)",
            len(texts), elapsed, elapsed / len(texts),
        )

        # Convert numpy array to Python list of lists for JSON serialization
        # and LangGraph state compatibility (TypedDict doesn't support numpy)
        return vectors.tolist()

    def embed_single(self, text: str) -> list[float]:
        """
        Embed a single text string. Convenience wrapper around embed().

        Used at query time to embed the user's question (or its HyDE rewrite)
        before searching Qdrant.

        Returns:
            Single float vector of length settings.embedding_dim.
        """
        results = self.embed([text])
        return results[0] if results else []

    def get_dimension(self) -> int:
        """
        Return the vector dimension for the loaded model.

        Used when creating the Qdrant collection — the collection's vector
        size must match the embedding model's output dimension exactly.
        Calling this at startup validates that settings.embedding_dim is correct.
        """
        return self.model.get_sentence_embedding_dimension()

    def warmup(self) -> None:
        """
        Pre-load the model by embedding a dummy sentence.

        Call this at API startup so the first real user request doesn't
        experience the 1-5 second model loading delay.

        Usage in FastAPI lifespan:
            @app.on_event("startup")
            async def startup():
                embedder.warmup()
        """
        logger.info("Warming up embedding model...")
        self.embed_single("warmup")
        logger.info("Embedding model warmed up.")


# ── LangGraph node function ────────────────────────────────────────────────────

# Module-level singleton — shared across all requests in a process.
# One model loaded once, used many times.
embedder = Embedder()


def embed_chunks(state: "AgentState") -> dict:
    """
    LangGraph node function for the ingest pipeline.

    Takes the chunks produced by chunk_document and generates a dense
    vector embedding for each chunk's text. The embeddings are stored
    in state and later written to Qdrant by store_chunks.

    Reads from state:
        chunks (list[dict]): chunk dicts from chunk_document node

    Writes to state (partial update):
        embeddings (list[list[float]]): one vector per chunk, in same order

    Design note:
        embeddings[i] corresponds to chunks[i] by index.
        This parallel-list pattern is maintained throughout the ingest pipeline.
        When storing to Qdrant, we zip(chunks, embeddings) to keep them paired.
    """
    chunks: list[dict] = state.get("chunks", [])

    if not chunks:
        logger.warning("embed_chunks called with no chunks in state")
        return {"embeddings": []}

    # Extract just the text from each chunk for embedding.
    # After contextual enrichment (Phase 3), each chunk's text will have
    # a 2-3 sentence context prepended — we embed the enriched text,
    # not the raw text, which is why embed_chunks runs AFTER enrich_chunks.
    texts = [chunk["text"] for chunk in chunks]

    t0 = time.perf_counter()
    embeddings = embedder.embed(texts)
    elapsed = (time.perf_counter() - t0) * 1000

    logger.info(
        "Embedded %d chunks in %.0fms (dim=%d)",
        len(embeddings), elapsed, len(embeddings[0]) if embeddings else 0,
    )

    return {"embeddings": embeddings}
