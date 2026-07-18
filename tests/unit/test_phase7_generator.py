"""
tests/unit/test_phase7_generator.py
=====================================
Unit tests for Phase 7: Generator, Faithfulness Judge, Memory Store.

Coverage:
  1.  _build_context()     — source blocks, KG paths, original_text preference
  2.  _build_user_prompt() — memory injection, structure, capping
  3.  _format_sources()    — field contract, truncation, 1-based index
  4.  generate()           — LLM call, prompt switching on retry, state contract
  5.  judge()              — NLI scoring, pass/reject/force-finalize, edge cases
  6.  should_retry()       — conditional edge correctness for all states
  7.  store_memory()       — Mem0 add call, skip conditions, failure handling
  8.  Retry loop           — generate → judge → generate (retry) → judge
  9.  Downstream API       — sources and answer format for Phase 8

All LLM, NLI, and Mem0 calls are mocked. Tests run offline in < 2 seconds.

Run:
    pytest tests/unit/test_phase7_generator.py -v
"""
from unittest.mock import MagicMock

import numpy as np
import pytest


def make_chunk(
    text: str = "The model achieved 97% accuracy on the test dataset.",
    original_text: str | None = None,
    page: int = 3,
    filename: str = "paper.pdf",
    doc_id: str = "doc_abc",
    chunk_index: int = 0,
    rerank_score: float = 0.92,
    rrf_score: float = 0.014,
    source: str = "dense",
) -> dict:
    """
    Build a retrieved_chunk dict matching Phase 5 (RetrievalAgent) output.
    original_text mimics Phase 3 (contextual enricher) enrichment output.
    """
    chunk = {
        "text":         text,
        "page":         page,
        "filename":     filename,
        "doc_id":       doc_id,
        "chunk_index":  chunk_index,
        "rerank_score": rerank_score,
        "rrf_score":    rrf_score,
        "source":       source,
    }
    if original_text is not None:
        chunk["original_text"] = original_text
    return chunk


def make_chunk_no_original(
    text: str = "Plain chunk text without enrichment.",
    **kwargs,
) -> dict:
    """Build a chunk WITHOUT original_text key (pre-Phase 3 or skipped enrichment)."""
    chunk = make_chunk(text=text, **kwargs)
    chunk.pop("original_text", None)
    return chunk


def make_kg_path(
    head: str = "FloodNet",
    relation: str = "developed_by",
    tail: str = "MIT",
    confidence: str = "0.91",
) -> dict:
    """Phase 6 (GraphAgent) stringifies all Neo4j record values."""
    return {"head": head, "relation": relation, "tail": tail, "confidence": confidence}


def make_memory(memory: str, score: float = 0.85) -> dict:
    """Phase 4 (RouterAgent via Mem0) memory dict format."""
    return {"memory": memory, "score": score}


def make_state(
    question: str = "What accuracy did the model achieve?",
    retrieved_chunks: list | None = None,
    kg_paths: list | None = None,
    long_term_memories: list | None = None,
    draft_answer: str = "",
    answer: str = "",
    user_id: str = "user_123",
    session_id: str = "sess_abc",
    retry_count: int = 0,
    judge_passed: bool | None = None,
    faithfulness_score: float | None = None,
    latency_ms: dict | None = None,
) -> dict:
    state: dict = {
        "question":           question,
        "retrieved_chunks":   retrieved_chunks if retrieved_chunks is not None else [make_chunk()],
        "kg_paths":           kg_paths or [],
        "long_term_memories": long_term_memories or [],
        "draft_answer":       draft_answer,
        "answer":             answer,
        "user_id":            user_id,
        "session_id":         session_id,
        "retry_count":        retry_count,
        "latency_ms":         latency_ms or {},
    }
    if judge_passed is not None:
        state["judge_passed"] = judge_passed
    if faithfulness_score is not None:
        state["faithfulness_score"] = faithfulness_score
    return state


def make_generator(
    llm_response: str = "The model achieved 97% accuracy [Source 1].",
    nli_scores: list[list[float]] | None = None,
    mem0_available: bool = True,
):
    """
    Build an AnswerGenerator with all external dependencies mocked.

    nli_scores: list of [contradiction, entailment, neutral] per sentence.
                Default = high entailment (0.90) → judge always passes.
    """
    from app.agents.generator import AnswerGenerator

    gen = AnswerGenerator()

    # Mock LLM (gpt-4o)
    gen.llm = MagicMock()
    gen.llm.invoke.return_value = MagicMock(content=llm_response)

    # Mock NLI cross-encoder
    mock_nli = MagicMock()
    if nli_scores is not None:
        mock_nli.predict.return_value = np.array(nli_scores)
    else:
        def _default_scores(pairs, **kwargs):
            return np.array([[0.05, 0.90, 0.05]] * len(pairs))
        mock_nli.predict.side_effect = _default_scores
    gen._nli_model = mock_nli

    # Mock Mem0
    gen._mem0_client = MagicMock() if mem0_available else None

    return gen



class TestBuildContext:

    def setup_method(self):
        self.gen = make_generator()

    def test_context_contains_source_index(self):
        context = self.gen._build_context([make_chunk()], kg_paths=[])
        assert "[Source 1 |" in context

    def test_context_contains_filename_and_page(self):
        context = self.gen._build_context(
            [make_chunk(filename="research.pdf", page=5)], kg_paths=[]
        )
        assert "research.pdf" in context
        assert "Page 5" in context

    def test_source_indices_are_sequential(self):
        chunks = [make_chunk(chunk_index=i) for i in range(3)]
        context = self.gen._build_context(chunks, kg_paths=[])
        assert "[Source 1 |" in context
        assert "[Source 2 |" in context
        assert "[Source 3 |" in context

    def test_uses_original_text_over_enriched_text(self):
        """
        original_text (without [Context: ...] prefix) must be used, not the
        enriched text. The LLM must never see our own metadata wrappers.
        """
        original = "The clean original chunk content from the document."
        enriched = "[Context: This is from section 3.]\n\n" + original
        chunk = make_chunk(text=enriched, original_text=original)
        context = self.gen._build_context([chunk], kg_paths=[])

        assert original in context
        assert "[Context:" not in context

    def test_falls_back_to_text_when_no_original_text(self):
        chunk = make_chunk_no_original(text="Plain chunk text here.")
        context = self.gen._build_context([chunk], kg_paths=[])
        assert "Plain chunk text here." in context

    def test_kg_paths_section_included_when_present(self):
        paths = [make_kg_path()]
        context = self.gen._build_context([], kg_paths=paths)
        assert "Knowledge Graph" in context
        assert "FloodNet" in context
        assert "MIT" in context

    def test_kg_paths_section_omitted_when_empty(self):
        context = self.gen._build_context([make_chunk()], kg_paths=[])
        assert "Knowledge Graph" not in context

    def test_kg_paths_capped_at_ten(self):
        paths = [make_kg_path(head=f"Entity{i}") for i in range(15)]
        context = self.gen._build_context([], kg_paths=paths)
        count = sum(1 for i in range(15) if f"Entity{i}" in context)
        assert count <= 10

    def test_kg_path_rendered_as_readable_arrow_string(self):
        paths = [make_kg_path("FloodNet", "uses_dataset", "ERA5", "0.85")]
        context = self.gen._build_context([], kg_paths=paths)
        assert "→" in context
        assert "uses_dataset" in context

    def test_empty_input_returns_empty_string(self):
        context = self.gen._build_context([], kg_paths=[])
        assert context.strip() == ""



class TestBuildUserPrompt:

    def setup_method(self):
        self.gen = make_generator()

    def test_prompt_contains_question(self):
        prompt = self.gen._build_user_prompt(
            "What accuracy was achieved?", "Context here.", []
        )
        assert "What accuracy was achieved?" in prompt

    def test_prompt_contains_context(self):
        prompt = self.gen._build_user_prompt("Question?", "Unique context XYZ.", [])
        assert "Unique context XYZ." in prompt

    def test_memories_injected_when_present(self):
        memories = [make_memory("User is an ML engineer specializing in RAG.")]
        prompt = self.gen._build_user_prompt("Q?", "Context.", memories)
        assert "User is an ML engineer" in prompt
        assert "User Context" in prompt

    def test_memories_omitted_when_empty(self):
        prompt = self.gen._build_user_prompt("Q?", "Context.", [])
        assert "User Context" not in prompt

    def test_memories_capped_at_five(self):
        memories = [make_memory(f"Memory number {i}.") for i in range(10)]
        prompt = self.gen._build_user_prompt("Q?", "Context.", memories)
        count = sum(1 for i in range(10) if f"Memory number {i}" in prompt)
        assert count <= 5

    def test_question_appears_after_context(self):
        prompt = self.gen._build_user_prompt("My question here?", "Context text here.", [])
        context_pos = prompt.find("Context text here.")
        question_pos = prompt.find("My question here?")
        assert context_pos < question_pos, "Context must precede question in prompt"



class TestFormatSources:

    def setup_method(self):
        self.gen = make_generator()

    def test_sources_are_one_based_indexed(self):
        chunks = [make_chunk(chunk_index=i) for i in range(2)]
        sources = self.gen._format_sources(chunks)
        assert sources[0]["index"] == 1
        assert sources[1]["index"] == 2

    def test_text_truncated_at_300_chars(self):
        chunk = make_chunk(text="A" * 500)
        sources = self.gen._format_sources([chunk])
        # 300 chars + "..." = 303 at most
        assert len(sources[0]["text"]) <= 303

    def test_short_text_not_truncated(self):
        chunk = make_chunk(text="Short text.")
        sources = self.gen._format_sources([chunk])
        assert sources[0]["text"] == "Short text."

    def test_uses_original_text_for_display(self):
        """original_text (not enriched) is what users see in the frontend."""
        original = "Clean text for user display."
        enriched = "[Context: Generated context.]\n\n" + original
        chunk = make_chunk(text=enriched, original_text=original)
        sources = self.gen._format_sources([chunk])
        assert sources[0]["text"] == original
        assert "[Context:" not in sources[0]["text"]

    def test_all_required_fields_present(self):
        """
        Required fields for QueryResponse.sources (Phase 8 API model):
        index, text, page, filename, doc_id, chunk_index, rerank_score
        """
        sources = self.gen._format_sources([make_chunk()])
        required = {"index", "text", "page", "filename", "doc_id", "chunk_index", "rerank_score"}
        assert required.issubset(sources[0].keys())

    def test_rerank_score_rounded_to_4_decimals(self):
        chunk = make_chunk(rerank_score=0.923456789)
        sources = self.gen._format_sources([chunk])
        assert sources[0]["rerank_score"] == round(0.923456789, 4)

    def test_empty_chunks_returns_empty_list(self):
        assert self.gen._format_sources([]) == []



class TestGenerateNode:

    def test_returns_draft_answer(self):
        gen = make_generator(llm_response="The accuracy is 97% [Source 1].")
        result = gen.generate(make_state())
        assert result["draft_answer"] == "The accuracy is 97% [Source 1]."

    def test_returns_sources_list(self):
        gen = make_generator()
        result = gen.generate(make_state())
        assert isinstance(result["sources"], list)

    def test_returns_latency_ms_with_generation_key(self):
        gen = make_generator()
        result = gen.generate(make_state())
        assert "generation" in result["latency_ms"]

    def test_preserves_existing_latency_keys(self):
        """generate() must ADD 'generation' to latency_ms, not replace it."""
        gen = make_generator()
        state = make_state(latency_ms={"router": 312.0, "retrieval": 847.0})
        result = gen.generate(state)
        assert result["latency_ms"]["router"] == pytest.approx(312.0)
        assert result["latency_ms"]["retrieval"] == pytest.approx(847.0)
        assert "generation" in result["latency_ms"]

    def test_uses_standard_prompt_on_first_attempt(self):
        """retry_count=0 → GENERATOR_SYSTEM_PROMPT (mentions 'precise document assistant')."""
        from langchain_core.messages import SystemMessage
        gen = make_generator()
        gen.generate(make_state(retry_count=0))
        system_msg = gen.llm.invoke.call_args[0][0][0]
        assert isinstance(system_msg, SystemMessage)
        assert "precise document assistant" in system_msg.content.lower()

    def test_uses_stricter_prompt_on_retry(self):
        """retry_count >= 1 → GENERATOR_RETRY_PROMPT (mentions 'REJECTED')."""
        from langchain_core.messages import SystemMessage
        gen = make_generator()
        gen.generate(make_state(retry_count=1))
        system_msg = gen.llm.invoke.call_args[0][0][0]
        assert isinstance(system_msg, SystemMessage)
        assert "REJECTED" in system_msg.content

    def test_llm_called_exactly_once(self):
        gen = make_generator()
        gen.generate(make_state())
        assert gen.llm.invoke.call_count == 1

    def test_output_keys_valid_in_agent_state(self):
        from app.core.state import AgentState
        valid_keys = set(AgentState.__annotations__.keys())
        gen = make_generator()
        result = gen.generate(make_state())
        for key in result:
            assert key in valid_keys, (
                f"generate() returned '{key}' which is not an AgentState field"
            )

    def test_returns_exactly_three_keys(self):
        gen = make_generator()
        result = gen.generate(make_state())
        assert set(result.keys()) == {"draft_answer", "sources", "latency_ms"}

    def test_strips_whitespace_from_llm_response(self):
        gen = make_generator(llm_response="  Padded answer.  ")
        result = gen.generate(make_state())
        assert result["draft_answer"] == "Padded answer."

    def test_sources_count_matches_chunk_count(self):
        chunks = [make_chunk(chunk_index=i) for i in range(4)]
        gen = make_generator()
        result = gen.generate(make_state(retrieved_chunks=chunks))
        assert len(result["sources"]) == 4



class TestJudgeNode:

    def test_passes_when_entailment_above_threshold(self):
        """High entailment → judge_passed=True and answer set."""
        gen = make_generator(nli_scores=[[0.02, 0.95, 0.03]])
        draft = "The model achieved 97% accuracy on the test dataset."
        result = gen.judge(make_state(draft_answer=draft))
        assert result["judge_passed"] is True
        assert result["answer"] == draft

    def test_rejects_when_entailment_below_threshold(self):
        """Low entailment → judge_passed=False, retry_count incremented."""
        gen = make_generator(nli_scores=[[0.50, 0.40, 0.10]])
        draft = "The model achieved 97% accuracy on the test dataset."
        result = gen.judge(make_state(draft_answer=draft, retry_count=0))
        assert result["judge_passed"] is False
        assert result["retry_count"] == 1

    def test_faithfulness_score_is_mean_of_sentence_scores(self):
        """
        Two sentences: entailment [0.80, 0.60] → faithfulness = 0.70.
        Mean is computed across ALL scoreable sentences (>= 20 chars).
        """
        gen = make_generator(nli_scores=[
            [0.10, 0.80, 0.10],
            [0.30, 0.60, 0.10],
        ])
        draft = "First complete sentence right here. Second complete sentence right here."
        result = gen.judge(make_state(draft_answer=draft))
        expected = round((0.80 + 0.60) / 2, 4)
        assert abs(result["faithfulness_score"] - expected) < 0.001

    def test_force_finalizes_when_retries_exhausted(self):
        """
        retry_count >= max_retries → finalize regardless of faithfulness score.
        Prevents infinite retry loops even when LLM keeps hallucinating.
        """
        from app.core.config import settings
        gen = make_generator(nli_scores=[[0.60, 0.30, 0.10]])
        draft = "Potentially hallucinated answer right here."
        state = make_state(draft_answer=draft, retry_count=settings.max_retries)
        result = gen.judge(state)
        assert result["judge_passed"] is True
        assert result["answer"] == draft

    def test_passes_immediately_when_no_retrieved_chunks(self):
        """No context = nothing to judge against → pass with score=1.0."""
        gen = make_generator()
        result = gen.judge(make_state(
            draft_answer="General knowledge answer here.",
            retrieved_chunks=[],
        ))
        assert result["judge_passed"] is True
        assert result["faithfulness_score"] == pytest.approx(1.0)
        gen._nli_model.predict.assert_not_called()

    def test_passes_immediately_for_empty_draft(self):
        gen = make_generator()
        result = gen.judge(make_state(draft_answer=""))
        assert result["judge_passed"] is True

    def test_passes_immediately_for_very_short_draft(self):
        """Drafts with no sentences >= 20 chars pass automatically."""
        gen = make_generator()
        result = gen.judge(make_state(draft_answer="Yes."))
        assert result["judge_passed"] is True
        gen._nli_model.predict.assert_not_called()

    def test_nli_uses_original_text_as_premise(self):
        """
        NLI premise must come from original_text, not the enriched text.
        Judging against [Context: ...] wrappers would distort the score.
        """
        original = "The actual document content used as NLI premise."
        enriched = "[Context: Generated header.]\n\n" + original
        chunk = make_chunk(text=enriched, original_text=original)

        gen = make_generator()
        gen.judge(make_state(
            draft_answer="Sentence about document content goes right here.",
            retrieved_chunks=[chunk],
        ))

        call_pairs = gen._nli_model.predict.call_args[0][0]
        premise_used = call_pairs[0][0]
        assert original in premise_used
        assert "[Context:" not in premise_used

    def test_judge_uses_at_most_five_chunks_for_premise(self):
        """
        NLI premise is built from first 5 chunks only. Beyond 5, concatenated
        text exceeds cross-encoder's 512-token limit → unreliable scores.
        """
        chunks = [
            make_chunk(text=f"Content of chunk number {i}." * 5, chunk_index=i)
            for i in range(8)
        ]
        gen = make_generator()
        gen.judge(make_state(
            draft_answer="Answer sentence about content here.",
            retrieved_chunks=chunks,
        ))
        call_pairs = gen._nli_model.predict.call_args[0][0]
        premise = call_pairs[0][0]
        assert "Content of chunk number 5." not in premise
        assert "Content of chunk number 6." not in premise

    def test_output_keys_valid_in_agent_state(self):
        from app.core.state import AgentState
        valid_keys = set(AgentState.__annotations__.keys())
        gen = make_generator()
        result = gen.judge(make_state(
            draft_answer="The model achieved 97% accuracy here."
        ))
        for key in result:
            assert key in valid_keys, (
                f"judge() returned '{key}' which is not an AgentState field"
            )

    def test_returns_latency_ms_with_judge_key(self):
        gen = make_generator()
        result = gen.judge(make_state(
            draft_answer="The model achieved 97% accuracy here."
        ))
        assert "judge" in result["latency_ms"]

    def test_preserves_prior_latency_keys(self):
        gen = make_generator()
        prior = {"router": 312.0, "retrieval": 847.0, "generation": 1240.0}
        result = gen.judge(make_state(
            draft_answer="The model achieved 97% accuracy here.",
            latency_ms=prior,
        ))
        assert result["latency_ms"]["router"] == pytest.approx(312.0)
        assert result["latency_ms"]["generation"] == pytest.approx(1240.0)
        assert "judge" in result["latency_ms"]

    def test_answer_not_set_when_judge_rejects(self):
        """On rejection, 'answer' key must NOT appear in result (only draft_answer exists)."""
        gen = make_generator(nli_scores=[[0.60, 0.30, 0.10]])
        result = gen.judge(make_state(
            draft_answer="Hallucinated answer sentence here.",
            retry_count=0,
        ))
        assert result["judge_passed"] is False
        assert "answer" not in result



class TestShouldRetry:

    def setup_method(self):
        self.gen = make_generator()

    def test_returns_memory_store_when_judge_passed_true(self):
        assert self.gen.should_retry({"judge_passed": True}) == "memory_store"

    def test_returns_generate_when_judge_passed_false(self):
        assert self.gen.should_retry({"judge_passed": False}) == "generate"

    def test_returns_memory_store_when_judge_passed_missing(self):
        """Missing judge_passed defaults to True → proceed."""
        assert self.gen.should_retry({}) == "memory_store"

    def test_always_returns_string(self):
        for passed in [True, False]:
            result = self.gen.should_retry({"judge_passed": passed})
            assert isinstance(result, str)

    def test_return_values_match_orchestrator_edge_map(self):
        """
        Return values must exactly match keys in orchestrator.py's edge map:
            {"generate": "generate", "memory_store": "memory_store"}
        A mismatch would silently route to wrong node or crash LangGraph.
        """
        valid_returns = {"generate", "memory_store"}
        for passed in [True, False]:
            result = self.gen.should_retry({"judge_passed": passed})
            assert result in valid_returns



class TestStoreMemory:

    def test_calls_mem0_add_with_correct_messages(self):
        gen = make_generator()
        gen.store_memory(make_state(
            question="What accuracy was achieved?",
            answer="The model achieved 97% accuracy [Source 1].",
        ))
        gen._mem0_client.add.assert_called_once()
        messages = gen._mem0_client.add.call_args[0][0]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What accuracy was achieved?"
        assert messages[1]["role"] == "assistant"
        assert "97%" in messages[1]["content"]

    def test_calls_mem0_add_with_user_id(self):
        gen = make_generator()
        gen.store_memory(make_state(user_id="user_xyz", answer="An answer."))
        call_kwargs = gen._mem0_client.add.call_args[1]
        assert call_kwargs.get("user_id") == "user_xyz"

    def test_skips_when_user_id_is_anonymous(self):
        gen = make_generator()
        gen.store_memory(make_state(user_id="anonymous", answer="An answer."))
        gen._mem0_client.add.assert_not_called()

    def test_skips_when_user_id_is_empty(self):
        gen = make_generator()
        gen.store_memory(make_state(user_id="", answer="An answer."))
        gen._mem0_client.add.assert_not_called()

    def test_skips_when_answer_is_empty(self):
        gen = make_generator()
        gen.store_memory(make_state(answer=""))
        gen._mem0_client.add.assert_not_called()

    def test_skips_when_mem0_unavailable(self):
        gen = make_generator(mem0_available=False)
        result = gen.store_memory(make_state(answer="An answer."))
        assert result is None

    def test_mem0_exception_does_not_propagate(self):
        gen = make_generator()
        gen._mem0_client.add.side_effect = ConnectionError("Mem0 unreachable")
        result = gen.store_memory(make_state(answer="An answer."))
        assert result is None

    def test_returns_none(self):
        """store_memory is a pure side-effect node — must not modify state."""
        gen = make_generator()
        result = gen.store_memory(make_state(answer="An answer."))
        assert result is None



class TestRetryLoop:
    """
    Simulates the full generate → judge → retry loop WITHOUT LangGraph.
    Manually calls node functions in sequence to verify the retry contract.
    """

    def test_single_attempt_passes_and_sets_answer(self):
        gen = make_generator(
            llm_response="The model achieved 97% accuracy [Source 1].",
            nli_scores=[[0.02, 0.92, 0.06]],
        )
        state = make_state(retry_count=0)

        state.update(gen.generate(state))
        judge_result = gen.judge(state)

        assert judge_result["judge_passed"] is True
        assert judge_result["answer"] == "The model achieved 97% accuracy [Source 1]."

    def test_retry_uses_stricter_prompt_on_second_call(self):
        """
        When first attempt fails, second generate() must use GENERATOR_RETRY_PROMPT.
        Verified by checking SystemMessage content of second LLM invocation.
        """
        from langchain_core.messages import SystemMessage

        gen = make_generator(
            llm_response="This proposed method improves the baseline accuracy substantially.",
            nli_scores=[[0.50, 0.30, 0.20]],
        )
        state = make_state(retry_count=0)

        # First generate + judge (fails)
        state.update(gen.generate(state))
        judge1 = gen.judge(state)
        assert judge1["judge_passed"] is False
        state.update(judge1)  # retry_count is now 1

        # Second generate (retry)
        gen.generate(state)

        second_call_args = gen.llm.invoke.call_args_list[1]
        system_msg = second_call_args[0][0][0]
        assert isinstance(system_msg, SystemMessage)
        assert "REJECTED" in system_msg.content

    def test_max_retries_forces_finalization_with_low_score(self):
        from app.core.config import settings
        gen = make_generator(
            llm_response="Potentially hallucinated answer here.",
            nli_scores=[[0.60, 0.20, 0.20]],
        )
        state = make_state(
            draft_answer="Potentially hallucinated answer here.",
            retry_count=settings.max_retries,
        )
        result = gen.judge(state)
        assert result["judge_passed"] is True
        assert result["answer"] == "Potentially hallucinated answer here."

    def test_route_follows_correct_path_after_rejection(self):
        """should_retry() must return 'generate' after judge rejects."""
        gen = make_generator(nli_scores=[[0.60, 0.30, 0.10]])
        state = make_state(
            draft_answer="Rejected answer sentence here.",
            retry_count=0,
        )
        judge_result = gen.judge(state)
        state.update(judge_result)

        route = gen.should_retry(state)
        assert route == "generate"

    def test_route_follows_correct_path_after_approval(self):
        """should_retry() must return 'memory_store' after judge approves."""
        gen = make_generator(nli_scores=[[0.02, 0.92, 0.06]])
        state = make_state(
            draft_answer="Approved answer sentence here.",
            retry_count=0,
        )
        judge_result = gen.judge(state)
        state.update(judge_result)

        route = gen.should_retry(state)
        assert route == "memory_store"



class TestDownstreamAPICompatibility:
    """
    orchestrator.query() (Phase 8) reads:
        final.get("answer", "")
        final.get("sources", [])
        final.get("faithfulness_score")
        final.get("latency_ms", {})
    All must be present and correctly typed.
    """

    def test_answer_is_string(self):
        gen = make_generator(nli_scores=[[0.02, 0.92, 0.06]])
        result = gen.judge(make_state(
            draft_answer="The accuracy is 97%."
        ))
        if "answer" in result:
            assert isinstance(result["answer"], str)

    def test_sources_is_list_of_dicts(self):
        gen = make_generator()
        result = gen.generate(make_state())
        assert isinstance(result["sources"], list)
        for s in result["sources"]:
            assert isinstance(s, dict)

    def test_faithfulness_score_is_float_in_unit_range(self):
        gen = make_generator(nli_scores=[[0.02, 0.92, 0.06]])
        result = gen.judge(make_state(
            draft_answer="The model achieved high accuracy here."
        ))
        score = result["faithfulness_score"]
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_latency_ms_is_dict_of_floats(self):
        gen = make_generator()
        result = gen.generate(make_state())
        for val in result["latency_ms"].values():
            assert isinstance(val, float)

    def test_singleton_is_answer_generator_instance(self):
        from app.agents.generator import AnswerGenerator, generator
        assert isinstance(generator, AnswerGenerator)



class TestGeneratorExtra:

    def test_nli_model_lazy_load(self, monkeypatch):
        """First access of the nli_model property loads the NLI
        cross-encoder (generator.py lines 233-243)."""
        from unittest.mock import patch

        from app.agents.generator import AnswerGenerator

        gen = AnswerGenerator()
        gen._nli_model = None  # force the load branch
        fake = MagicMock()
        with patch("app.agents.generator.CrossEncoder", return_value=fake) as mock_ce:
            model = gen.nli
        mock_ce.assert_called_once()
        assert model is fake

    def test_mem0_hosted_init(self, monkeypatch):
        """settings.mem0_api_key set → hosted MemoryClient
        (generator.py lines 266-268)."""
        from unittest.mock import patch

        from app.agents.generator import AnswerGenerator
        from app.core.config import settings

        monkeypatch.setattr(settings, "mem0_api_key", "fake-key")
        gen = AnswerGenerator()
        gen._mem0_client = None
        with patch("mem0.MemoryClient", return_value=MagicMock()) as mock_client:
            client = gen.mem0
        mock_client.assert_called_once()
        assert client is not None

    def test_mem0_embedded_init_logs(self, monkeypatch):
        """No api key → embedded Memory.from_config; success log
        (line 298) when from_config succeeds."""
        from unittest.mock import patch

        from app.agents.generator import AnswerGenerator
        from app.core.config import settings

        monkeypatch.setattr(settings, "mem0_api_key", None)
        gen = AnswerGenerator()
        gen._mem0_client = None
        with patch("mem0.Memory.from_config", return_value=MagicMock()) as mock_from:
            client = gen.mem0
        mock_from.assert_called_once()
        assert client is not None
