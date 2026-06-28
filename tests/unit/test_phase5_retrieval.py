"""
tests/unit/test_phase5_retrieval.py
=====================================
Unit tests for the Retrieval Agent (Phase 5).

Coverage:
  1.  HyDE rewriting — graceful fallback on failure
  2.  Question decomposition — output format, fallback, cap at 4
  3.  RRF fusion — formula correctness, deduplication, weight application
  4.  Cross-encoder reranking — score attachment, sort order, top_k
  5.  retrieve() node — state read/write contract
  6.  retrieve_multi() node — deduplication, sub_questions in state
  7.  _avg_rerank_score() helper — correctness, empty input
  8.  Downstream compatibility — output feeds correctly into generator (Phase 7)

All external calls (LLM, Qdrant, BM25, CrossEncoder) are mocked.
Tests run offline in < 2 seconds.

Run:
    pytest tests/unit/test_phase5_retrieval.py -v
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures & Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_chunk(
    text: str = "Default chunk text " * 5,
    doc_id: str = "doc_abc",
    chunk_index: int = 0,
    page: int = 1,
    score: float = 0.8,
    source: str = "dense",
) -> dict:
    """
    Build a chunk dict matching the exact schema returned by VectorStore.search()
    and BM25Index.search() (Phase 2 services).
    """
    return {
        "text":        text,
        "page":        page,
        "filename":    "test.pdf",
        "doc_id":      doc_id,
        "chunk_index": chunk_index,
        "score":       score,
        "source":      source,
    }


def make_chunks(n: int, source: str = "dense") -> list[dict]:
    return [
        make_chunk(
            text=f"Chunk text number {i} with enough words to be meaningful content here",
            chunk_index=i,
            score=0.9 - i * 0.05,
            source=source,
        )
        for i in range(n)
    ]


def make_agent_with_mocks(
    hyde_response: str = "Hypothetical answer passage for dense search.",
    dense_results: list | None = None,
    bm25_results: list | None = None,
    reranker_scores: list[float] | None = None,
):
    """
    Build a RetrievalAgent with all external dependencies mocked.
    Returns (agent, mocks_dict) for detailed inspection.
    """
    from app.agents.retrieval_agent import RetrievalAgent

    agent = RetrievalAgent()

    # Mock LLM (HyDE + decomposition)
    agent.llm = MagicMock()
    agent.llm.invoke.return_value = MagicMock(content=hyde_response)

    # Mock cross-encoder
    mock_ce = MagicMock()
    if reranker_scores is not None:
        mock_ce.predict.return_value = np.array(reranker_scores)
    else:
        # Default: assign descending scores to preserve input order
        def _dynamic_scores(pairs):
            n = len(pairs)
            return np.array([1.0 - i * 0.1 for i in range(n)])
        mock_ce.predict.side_effect = _dynamic_scores
    agent._cross_encoder = mock_ce

    # Mock Phase 2 services at module level
    mock_vs = MagicMock()
    mock_vs.search.return_value = dense_results if dense_results is not None else make_chunks(5)

    mock_bm25 = MagicMock()
    mock_bm25.search.return_value = bm25_results if bm25_results is not None else make_chunks(3, "bm25")

    mock_embedder = MagicMock()
    mock_embedder.embed_single.return_value = [0.1] * 384

    return agent, {
        "vs": mock_vs,
        "bm25": mock_bm25,
        "embedder": mock_embedder,
        "ce": mock_ce,
    }


def run_retrieve(agent, mocks, state: dict) -> dict:
    """Run agent.retrieve() with mocked Phase 2 services."""
    import app.agents.retrieval_agent as ra_module
    orig_vs = ra_module.vector_store
    orig_bm25 = ra_module.bm25_index
    orig_emb = ra_module.embedder
    ra_module.vector_store = mocks["vs"]
    ra_module.bm25_index   = mocks["bm25"]
    ra_module.embedder     = mocks["embedder"]
    try:
        return agent.retrieve(state)
    finally:
        ra_module.vector_store = orig_vs
        ra_module.bm25_index   = orig_bm25
        ra_module.embedder     = orig_emb


def run_retrieve_multi(agent, mocks, state: dict) -> dict:
    """Run agent.retrieve_multi() with mocked Phase 2 services."""
    import app.agents.retrieval_agent as ra_module
    orig_vs = ra_module.vector_store
    orig_bm25 = ra_module.bm25_index
    orig_emb = ra_module.embedder
    ra_module.vector_store = mocks["vs"]
    ra_module.bm25_index   = mocks["bm25"]
    ra_module.embedder     = mocks["embedder"]
    try:
        return agent.retrieve_multi(state)
    finally:
        ra_module.vector_store = orig_vs
        ra_module.bm25_index   = orig_bm25
        ra_module.embedder     = orig_emb


def make_state(
    question: str = "What is the main finding?",
    doc_id: str | None = "doc_xyz",
    top_k: int = 5,
    latency_ms: dict | None = None,
) -> dict:
    return {
        "question":    question,
        "doc_id":      doc_id,
        "top_k":       top_k,
        "user_id":     "user_123",
        "session_id":  "session_abc",
        "retry_count": 0,
        "latency_ms":  latency_ms or {},
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HyDE Rewriting Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestHydeRewrite:

    def test_hyde_returns_llm_content(self):
        """_hyde_rewrite() must return the LLM's content string."""
        from app.agents.retrieval_agent import RetrievalAgent
        agent = RetrievalAgent()
        expected = "Heavy rainfall exceeded river capacity causing downstream flooding."
        agent.llm = MagicMock()
        agent.llm.invoke.return_value = MagicMock(content=expected)
        result = agent._hyde_rewrite("What caused the flood?")
        assert result == expected

    def test_hyde_falls_back_to_question_on_llm_error(self):
        """On LLM failure, _hyde_rewrite() must return original question (not raise)."""
        from app.agents.retrieval_agent import RetrievalAgent
        agent = RetrievalAgent()
        agent.llm = MagicMock()
        agent.llm.invoke.side_effect = ConnectionError("API timeout")
        question = "What is the impact of climate change?"
        result = agent._hyde_rewrite(question)
        assert result == question

    def test_hyde_strips_whitespace_from_response(self):
        """LLM response may have leading/trailing whitespace — strip it."""
        from app.agents.retrieval_agent import RetrievalAgent
        agent = RetrievalAgent()
        agent.llm = MagicMock()
        agent.llm.invoke.return_value = MagicMock(content="  Passage with spaces.  ")
        result = agent._hyde_rewrite("What?")
        assert result == "Passage with spaces."


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Question Decomposition Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDecomposeQuestion:

    def setup_method(self):
        from app.agents.retrieval_agent import RetrievalAgent
        self.agent = RetrievalAgent()

    def test_returns_list_of_strings(self):
        self.agent.llm = MagicMock()
        self.agent.llm.invoke.return_value = MagicMock(
            content="What is method A?\nWhat is method B?\nHow do they differ?"
        )
        result = self.agent._decompose_question("Compare method A with method B.")
        assert isinstance(result, list)
        assert all(isinstance(q, str) for q in result)

    def test_parses_newline_separated_output(self):
        self.agent.llm = MagicMock()
        self.agent.llm.invoke.return_value = MagicMock(
            content="What preprocessing steps are used?\nWhat limitations exist?\nHow do they relate?"
        )
        result = self.agent._decompose_question("How does preprocessing address limitations?")
        assert len(result) == 3

    def test_caps_at_four_sub_questions(self):
        """Even if LLM returns 6 lines, we must cap at 4."""
        self.agent.llm = MagicMock()
        self.agent.llm.invoke.return_value = MagicMock(
            content="\n".join([
                "First sub-question here please?",
                "Second sub-question here indeed?",
                "Third sub-question right now?",
                "Fourth sub-question is good?",
                "Fifth sub-question too many?",
                "Sixth sub-question way too many?",
            ])
        )
        result = self.agent._decompose_question("Complex question")
        assert len(result) <= 4

    def test_filters_short_noise_lines(self):
        """Lines with fewer than 5 words are artifacts — must be filtered."""
        self.agent.llm = MagicMock()
        self.agent.llm.invoke.return_value = MagicMock(
            content="What are the main findings of this research paper?\nOK\n\nWhat methodology was used?"
        )
        result = self.agent._decompose_question("Complex question here")
        # "OK" and empty line must be filtered
        assert all(len(q.split()) >= 5 for q in result)

    def test_falls_back_to_original_on_llm_error(self):
        """On LLM failure, return [original_question] — never empty list."""
        self.agent.llm = MagicMock()
        self.agent.llm.invoke.side_effect = RuntimeError("Network error")
        original = "What is the relationship between X and Y?"
        result = self.agent._decompose_question(original)
        assert result == [original]

    def test_falls_back_when_all_lines_too_short(self):
        """If LLM returns only noise, fall back to [original_question]."""
        self.agent.llm = MagicMock()
        self.agent.llm.invoke.return_value = MagicMock(content="OK\nYes\nNo")
        original = "What is the main contribution?"
        result = self.agent._decompose_question(original)
        assert result == [original]

    def test_always_returns_non_empty_list(self):
        """_decompose_question must NEVER return an empty list."""
        self.agent.llm = MagicMock()
        self.agent.llm.invoke.side_effect = Exception("Any error")
        result = self.agent._decompose_question("Some question")
        assert len(result) >= 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RRF Fusion Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRRFMerge:

    def setup_method(self):
        from app.agents.retrieval_agent import RetrievalAgent
        self.agent = RetrievalAgent()

    def test_chunk_in_both_lists_scores_higher_than_exclusive(self):
        """
        A chunk appearing in BOTH dense and BM25 results gets scores from
        both lists added together. It must rank above chunks in only one list.
        This is the core property of RRF — multi-list evidence is rewarded.
        """
        shared_text = "This chunk appears in both dense and BM25 results for sure."
        shared = make_chunk(text=shared_text, chunk_index=0, score=0.9, source="dense")
        dense_only = make_chunk(text="Dense only chunk different content here.", chunk_index=1)
        bm25_copy = {**shared, "source": "bm25", "score": 0.8}

        result = self.agent._rrf_merge(
            dense=[shared, dense_only],
            bm25=[bm25_copy],
            top_k=3,
        )
        top_text = result[0]["text"][:40]
        assert "This chunk appears in both" in top_text, (
            "Chunk appearing in both lists must rank first"
        )

    def test_respects_top_k_limit(self):
        """_rrf_merge must return exactly top_k results."""
        dense = make_chunks(8, "dense")
        bm25  = make_chunks(6, "bm25")
        result = self.agent._rrf_merge(dense, bm25, top_k=4)
        assert len(result) == 4

    def test_all_results_have_rrf_score(self):
        """Every chunk in the output must have rrf_score field."""
        dense = make_chunks(4)
        bm25  = make_chunks(3, "bm25")
        result = self.agent._rrf_merge(dense, bm25, top_k=5)
        for chunk in result:
            assert "rrf_score" in chunk, f"Missing rrf_score in chunk: {chunk.keys()}"
            assert isinstance(chunk["rrf_score"], float)

    def test_rrf_scores_are_positive(self):
        """RRF scores must always be positive (they are sums of 1/(k+rank))."""
        dense = make_chunks(3)
        result = self.agent._rrf_merge(dense, [], top_k=3)
        for chunk in result:
            assert chunk["rrf_score"] > 0

    def test_empty_dense_with_only_bm25_works(self):
        """If dense returns nothing, BM25 results must still be returned."""
        bm25 = make_chunks(3, "bm25")
        result = self.agent._rrf_merge([], bm25, top_k=3)
        assert len(result) == 3

    def test_both_empty_returns_empty(self):
        """If both search results are empty, output must be empty."""
        result = self.agent._rrf_merge([], [], top_k=5)
        assert result == []

    def test_dense_weight_higher_than_bm25(self):
        """
        Dense-only chunk at rank 1 vs BM25-only chunk at rank 1.
        Dense must score higher because dense_weight > bm25_weight.

        dense_score  = 0.7 × 1/(60+1) = 0.01148
        bm25_score   = 0.3 × 1/(60+1) = 0.00492
        """
        dense_chunk = make_chunk(text="Dense exclusive content A B C D E F G H I J", chunk_index=10, source="dense")
        bm25_chunk  = make_chunk(text="BM25 exclusive content K L M N O P Q R S T", chunk_index=11, source="bm25")

        result = self.agent._rrf_merge([dense_chunk], [bm25_chunk], top_k=2)
        # Dense chunk must rank first (higher weight)
        assert result[0]["text"].startswith("Dense exclusive")

    def test_rrf_preserves_original_chunk_fields(self):
        """_rrf_merge must NOT strip original chunk fields (doc_id, page, etc.)."""
        chunk = make_chunk(text="Complete chunk with all metadata fields", chunk_index=5, page=7)
        result = self.agent._rrf_merge([chunk], [], top_k=1)
        assert result[0]["page"] == 7
        assert result[0]["doc_id"] == "doc_abc"
        assert result[0]["chunk_index"] == 5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cross-Encoder Re-ranking Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRerank:

    def setup_method(self):
        from app.agents.retrieval_agent import RetrievalAgent
        self.agent = RetrievalAgent()
        self.mock_ce = MagicMock()
        self.agent._cross_encoder = self.mock_ce

    def test_rerank_adds_rerank_score_to_each_chunk(self):
        """Every chunk in output must have rerank_score field."""
        chunks = make_chunks(3)
        self.mock_ce.predict.return_value = np.array([0.9, 0.5, 0.7])
        result = self.agent._rerank("What is X?", chunks, top_k=3)
        for c in result:
            assert "rerank_score" in c

    def test_rerank_sorts_by_score_descending(self):
        """Output must be sorted by rerank_score highest first."""
        chunks = make_chunks(3)
        # Assign scores out of order
        self.mock_ce.predict.return_value = np.array([0.3, 0.9, 0.6])
        result = self.agent._rerank("Some question?", chunks, top_k=3)
        scores = [c["rerank_score"] for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_respects_top_k(self):
        """Output must contain exactly top_k chunks."""
        chunks = make_chunks(8)
        self.mock_ce.predict.return_value = np.array([0.9 - i*0.1 for i in range(8)])
        result = self.agent._rerank("Question?", chunks, top_k=3)
        assert len(result) == 3

    def test_rerank_uses_original_text_when_available(self):
        """
        Cross-encoder must receive original_text (without [Context: ...] prefix)
        when it's available. This ensures the model scores actual content,
        not our metadata wrapper from contextual enrichment (Phase 3).
        """
        chunks = [
            {
                **make_chunk("text"),
                "original_text": "The actual clean content without context prefix.",
                "text": "[Context: Generated context.]\n\nThe actual clean content without context prefix.",
            }
        ]
        self.mock_ce.predict.return_value = np.array([0.85])
        self.agent._rerank("Question about content?", chunks, top_k=1)

        call_pairs = self.mock_ce.predict.call_args[0][0]
        assert "actual clean content" in call_pairs[0][1]
        assert "[Context:" not in call_pairs[0][1]

    def test_rerank_falls_back_to_text_when_no_original_text(self):
        """When original_text is absent (not enriched), use chunk text directly."""
        chunks = [make_chunk("Plain chunk text without context wrapper.")]
        self.mock_ce.predict.return_value = np.array([0.7])
        self.agent._rerank("Question?", chunks, top_k=1)

        call_pairs = self.mock_ce.predict.call_args[0][0]
        assert "Plain chunk text" in call_pairs[0][1]

    def test_rerank_empty_input_returns_empty(self):
        """Empty input must return [] without calling cross-encoder."""
        result = self.agent._rerank("Question?", [], top_k=5)
        assert result == []
        self.mock_ce.predict.assert_not_called()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# retrieve() Node — State Contract Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRetrieveNode:

    def test_returns_retrieved_chunks_key(self):
        agent, mocks = make_agent_with_mocks()
        result = run_retrieve(agent, mocks, make_state())
        assert "retrieved_chunks" in result
        assert isinstance(result["retrieved_chunks"], list)

    def test_returns_retrieval_score_key(self):
        agent, mocks = make_agent_with_mocks()
        result = run_retrieve(agent, mocks, make_state())
        assert "retrieval_score" in result
        assert isinstance(result["retrieval_score"], float)

    def test_returns_rewritten_query_key(self):
        agent, mocks = make_agent_with_mocks(hyde_response="Hypothetical passage here.")
        result = run_retrieve(agent, mocks, make_state())
        assert "rewritten_query" in result
        assert result["rewritten_query"] == "Hypothetical passage here."

    def test_returns_latency_ms_with_retrieval_key(self):
        agent, mocks = make_agent_with_mocks()
        result = run_retrieve(agent, mocks, make_state())
        assert "latency_ms" in result
        assert "retrieval" in result["latency_ms"]

    def test_preserves_existing_latency_keys(self):
        """
        retrieve() must add "retrieval" to latency_ms, not replace it.
        Router already wrote "router" key — that must survive.
        """
        state = make_state(latency_ms={"router": 312.5})
        agent, mocks = make_agent_with_mocks()
        result = run_retrieve(agent, mocks, state)
        assert result["latency_ms"]["router"] == 312.5
        assert "retrieval" in result["latency_ms"]

    def test_output_keys_valid_in_agent_state(self):
        """All keys returned by retrieve() must be valid AgentState fields."""
        from app.core.state import AgentState
        valid_keys = set(AgentState.__annotations__.keys())
        agent, mocks = make_agent_with_mocks()
        result = run_retrieve(agent, mocks, make_state())
        for key in result:
            assert key in valid_keys, (
                f"retrieve() returned '{key}' not in AgentState. "
                "Add it to state.py or remove from return dict."
            )

    def test_each_chunk_has_rrf_score(self):
        """
        Every chunk in retrieved_chunks must have rrf_score.
        This is set by _rrf_merge and required by the generator for logging.
        """
        agent, mocks = make_agent_with_mocks()
        result = run_retrieve(agent, mocks, make_state())
        for chunk in result["retrieved_chunks"]:
            assert "rrf_score" in chunk, (
                f"Chunk missing rrf_score: {list(chunk.keys())}"
            )

    def test_each_chunk_has_rerank_score(self):
        """
        Every chunk in retrieved_chunks must have rerank_score.
        This is set by _rerank and is what the generator shows users.
        """
        agent, mocks = make_agent_with_mocks()
        result = run_retrieve(agent, mocks, make_state())
        for chunk in result["retrieved_chunks"]:
            assert "rerank_score" in chunk, (
                f"Chunk missing rerank_score: {list(chunk.keys())}"
            )

    def test_empty_question_returns_empty_chunks(self):
        """retrieve() with no question must return empty retrieved_chunks."""
        agent, mocks = make_agent_with_mocks()
        result = run_retrieve(agent, mocks, make_state(question=""))
        assert result["retrieved_chunks"] == []
        assert result["retrieval_score"] == 0.0

    def test_top_k_is_respected(self):
        """retrieve() must return at most top_k chunks."""
        agent, mocks = make_agent_with_mocks(
            dense_results=make_chunks(10),
            bm25_results=make_chunks(8, "bm25"),
            reranker_scores=[1.0 - i*0.05 for i in range(10)],
        )
        result = run_retrieve(agent, mocks, make_state(top_k=3))
        assert len(result["retrieved_chunks"]) <= 3

    def test_hyde_llm_called_once(self):
        """HyDE must call LLM exactly once per retrieve() invocation."""
        agent, mocks = make_agent_with_mocks()
        run_retrieve(agent, mocks, make_state())
        assert agent.llm.invoke.call_count == 1

    def test_embedder_called_with_hyde_passage(self):
        """
        embed_single() must be called with the HyDE passage, not the raw question.
        This is the fundamental correctness invariant of HyDE.
        """
        hyde_passage = "Unique hypothetical passage XYZ for testing."
        agent, mocks = make_agent_with_mocks(hyde_response=hyde_passage)
        run_retrieve(agent, mocks, make_state("What is the answer?"))
        call_args = mocks["embedder"].embed_single.call_args[0][0]
        assert call_args == hyde_passage


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# retrieve_multi() Node Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRetrieveMultiNode:

    def _make_agent_with_decompose(self, sub_questions: list[str]):
        """Build agent where LLM returns sub_questions on first call, HyDE on subsequent."""
        from app.agents.retrieval_agent import RetrievalAgent
        agent = RetrievalAgent()
        call_count = [0]

        def llm_side_effect(messages):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: decomposition
                return MagicMock(content="\n".join(sub_questions))
            else:
                # Subsequent calls: HyDE
                return MagicMock(content="Hypothetical answer passage.")

        agent.llm = MagicMock()
        agent.llm.invoke.side_effect = llm_side_effect

        mock_ce = MagicMock()
        mock_ce.predict.side_effect = lambda pairs: np.array(
            [1.0 - i * 0.1 for i in range(len(pairs))]
        )
        agent._cross_encoder = mock_ce
        return agent

    def test_sub_questions_in_state_output(self):
        """sub_questions must be written to state by retrieve_multi()."""
        agent = self._make_agent_with_decompose([
            "What preprocessing steps are used in section three?",
            "What limitations are identified in section six?",
        ])
        mocks = {
            "vs":      MagicMock(search=MagicMock(return_value=make_chunks(3))),
            "bm25":    MagicMock(search=MagicMock(return_value=make_chunks(2, "bm25"))),
            "embedder": MagicMock(embed_single=MagicMock(return_value=[0.1]*384)),
        }
        result = run_retrieve_multi(agent, mocks, make_state(
            "How does preprocessing in section 3 address limitations in section 6?"
        ))
        assert "sub_questions" in result
        assert len(result["sub_questions"]) == 2

    def test_deduplication_prevents_same_chunk_twice(self):
        """
        If the same chunk (same doc_id + chunk_index) appears in multiple
        sub-question retrievals, it must appear only ONCE in the final output.
        """
        agent = self._make_agent_with_decompose([
            "First sub-question about topic A?",
            "Second sub-question about topic B?",
        ])
        # Both sub-questions retrieve the SAME chunk (chunk_index=0)
        same_chunk = make_chunk(text="Same content " * 5, chunk_index=0, doc_id="doc_x")

        mocks = {
            "vs":      MagicMock(search=MagicMock(return_value=[same_chunk])),
            "bm25":    MagicMock(search=MagicMock(return_value=[])),
            "embedder": MagicMock(embed_single=MagicMock(return_value=[0.1]*384)),
        }
        result = run_retrieve_multi(agent, mocks, make_state("Complex multi-hop question here?"))

        # Count how many times chunk_index=0 / doc_id=doc_x appears
        dedup_keys = [
            (c.get("doc_id"), c.get("chunk_index"))
            for c in result["retrieved_chunks"]
        ]
        assert len(dedup_keys) == len(set(dedup_keys)), (
            "Duplicate (doc_id, chunk_index) found in retrieved_chunks"
        )

    def test_output_has_retrieval_multi_latency_key(self):
        """retrieve_multi() must write 'retrieval_multi' (not 'retrieval') to latency_ms."""
        agent = self._make_agent_with_decompose([
            "First valid sub-question that is long enough?",
        ])
        mocks = {
            "vs":      MagicMock(search=MagicMock(return_value=make_chunks(3))),
            "bm25":    MagicMock(search=MagicMock(return_value=[])),
            "embedder": MagicMock(embed_single=MagicMock(return_value=[0.1]*384)),
        }
        result = run_retrieve_multi(agent, mocks, make_state("Multi-hop question here?"))
        assert "retrieval_multi" in result["latency_ms"]

    def test_rerank_uses_original_question_not_sub_question(self):
        """
        The cross-encoder must score chunks against the ORIGINAL question,
        not the sub-questions. This ensures final chunks are relevant to
        what the user actually asked — not just to a decomposed subset.
        """
        original_q = "How does A compare to B in terms of C?"
        agent = self._make_agent_with_decompose([
            "What is A in terms of C from document?",
            "What is B in terms of C from document?",
        ])

        captured_questions = []
        def capture_pairs(pairs):
            captured_questions.extend([p[0] for p in pairs])
            return np.array([0.8] * len(pairs))

        agent._cross_encoder = MagicMock()
        agent._cross_encoder.predict.side_effect = capture_pairs

        mocks = {
            "vs":      MagicMock(search=MagicMock(return_value=make_chunks(3))),
            "bm25":    MagicMock(search=MagicMock(return_value=[])),
            "embedder": MagicMock(embed_single=MagicMock(return_value=[0.1]*384)),
        }
        run_retrieve_multi(agent, mocks, make_state(question=original_q))
        # All reranking queries must be the ORIGINAL question
        assert all(q == original_q for q in captured_questions), (
            f"Cross-encoder received sub-questions instead of original: {set(captured_questions)}"
        )

    def test_output_keys_valid_in_agent_state(self):
        """All keys returned by retrieve_multi() must be valid AgentState fields."""
        from app.core.state import AgentState
        valid_keys = set(AgentState.__annotations__.keys())
        agent = self._make_agent_with_decompose(["Valid sub-question here?"])
        mocks = {
            "vs":      MagicMock(search=MagicMock(return_value=make_chunks(2))),
            "bm25":    MagicMock(search=MagicMock(return_value=[])),
            "embedder": MagicMock(embed_single=MagicMock(return_value=[0.1]*384)),
        }
        result = run_retrieve_multi(agent, mocks, make_state())
        for key in result:
            assert key in valid_keys, f"retrieve_multi() returned invalid key '{key}'"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _avg_rerank_score Helper Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestAvgRerankScore:

    def test_correct_average_with_rerank_scores(self):
        from app.agents.retrieval_agent import _avg_rerank_score
        chunks = [
            {**make_chunk(), "rerank_score": 0.9},
            {**make_chunk(), "rerank_score": 0.7},
            {**make_chunk(), "rerank_score": 0.8},
        ]
        result = _avg_rerank_score(chunks)
        assert abs(result - 0.8) < 0.001

    def test_falls_back_to_rrf_score_when_no_rerank_score(self):
        """If rerank_score absent, use rrf_score as fallback."""
        from app.agents.retrieval_agent import _avg_rerank_score
        chunks = [
            {**make_chunk(), "rrf_score": 0.6},
            {**make_chunk(), "rrf_score": 0.4},
        ]
        result = _avg_rerank_score(chunks)
        assert abs(result - 0.5) < 0.001

    def test_returns_zero_for_empty_list(self):
        from app.agents.retrieval_agent import _avg_rerank_score
        assert _avg_rerank_score([]) == 0.0

    def test_result_is_rounded_to_4_decimals(self):
        from app.agents.retrieval_agent import _avg_rerank_score
        chunks = [{**make_chunk(), "rerank_score": 1/3}]
        result = _avg_rerank_score(chunks)
        assert result == round(1/3, 4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Downstream Compatibility — Phase 7 Generator Expectations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDownstreamCompatibility:
    """
    Verify that retrieve() output matches exactly what Phase 7 (generator)
    expects to read from AgentState.retrieved_chunks.

    Generator reads:
        chunk["text"]          — for building context window
        chunk.get("original_text") — for display to users (optional)
        chunk.get("page")      — for source citation
        chunk.get("filename")  — for source citation
        chunk.get("doc_id")    — for source citation
        chunk.get("rerank_score") — for source quality display
        chunk.get("chunk_index")  — for deduplication
    """

    def test_chunks_have_text_field(self):
        agent, mocks = make_agent_with_mocks()
        result = run_retrieve(agent, mocks, make_state())
        for chunk in result["retrieved_chunks"]:
            assert "text" in chunk and chunk["text"]

    def test_chunks_have_page_field(self):
        agent, mocks = make_agent_with_mocks()
        result = run_retrieve(agent, mocks, make_state())
        for chunk in result["retrieved_chunks"]:
            assert "page" in chunk

    def test_chunks_have_filename_field(self):
        agent, mocks = make_agent_with_mocks()
        result = run_retrieve(agent, mocks, make_state())
        for chunk in result["retrieved_chunks"]:
            assert "filename" in chunk

    def test_chunks_have_doc_id_field(self):
        agent, mocks = make_agent_with_mocks()
        result = run_retrieve(agent, mocks, make_state())
        for chunk in result["retrieved_chunks"]:
            assert "doc_id" in chunk

    def test_singleton_is_retrieval_agent_instance(self):
        from app.agents.retrieval_agent import retrieval_agent, RetrievalAgent
        assert isinstance(retrieval_agent, RetrievalAgent)
