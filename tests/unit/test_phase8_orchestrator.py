"""
tests/unit/test_phase8_orchestrator.py
========================================
Unit tests for Phase 8: Orchestrator + FastAPI API + WebSocket.

Coverage:
  1.  _safe_serialize()        — strips raw_bytes/embeddings, removes None values
  2.  direct_answer()          — LLM call, memory injection, state write contract
  3.  merge_results()          — no-op pass-through
  4.  ARAPOrchestrator.ingest()— delegates to ingest_graph, returns correct keys
  5.  ARAPOrchestrator.query() — delegates to query_graph, returns correct keys
  6.  ARAPOrchestrator.health()— independent component checks, no exception
  7.  FastAPI /health          — 200 + correct schema
  8.  FastAPI /ingest          — PDF validation, size cap, 200 + IngestResponse
  9.  FastAPI /query           — request validation, 200 + QueryResponse schema
  10. FastAPI /ws/{session_id} — WebSocket protocol: update events, done, error

All external calls (LangGraph graphs, LLM, Qdrant, Neo4j, Redis) are mocked.
FastAPI tests use httpx.AsyncClient (AsyncClient for ASGI) + anyio backend.

Run:
    pytest tests/unit/test_phase8_orchestrator.py -v
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_final_state(
    answer: str = "The model achieved 97% accuracy [Source 1].",
    query_type: str = "single",
    faithfulness_score: float = 0.91,
    sources: list | None = None,
    latency_ms: dict | None = None,
    doc_id: str = "doc_abc",
    chunk_count: int = 42,
    kg_entities: list | None = None,
) -> dict:
    """Simulate the final AgentState dict returned by LangGraph graph.invoke()."""
    return {
        "answer":             answer,
        "query_type":         query_type,
        "faithfulness_score": faithfulness_score,
        "sources":            sources or [
            {
                "index": 1, "text": "Chunk text here.", "page": 3,
                "filename": "paper.pdf", "doc_id": doc_id,
                "chunk_index": 0, "rerank_score": 0.92,
            }
        ],
        "latency_ms":         latency_ms or {"router": 312.0, "retrieval": 847.0},
        "doc_id":             doc_id,
        "chunk_count":        chunk_count,
        "kg_entities":        kg_entities or [],
        "judge_passed":       True,
        "retry_count":        0,
    }


def make_async_client():
    """
    Build an httpx.AsyncClient targeting the FastAPI app.
    Patches orchestrator so no real infrastructure is needed.
    """
    from httpx import ASGITransport, AsyncClient

    from app.api.main import app
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _safe_serialize() Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSafeSerialize:

    def test_strips_raw_bytes(self):
        """raw_bytes (PDF binary) must never reach the WebSocket client."""
        from app.core.orchestrator import _safe_serialize
        state = {"raw_bytes": b"%PDF-1.4 binary data", "answer": "OK"}
        result = _safe_serialize(state)
        assert "raw_bytes" not in result
        assert result["answer"] == "OK"

    def test_strips_embeddings(self):
        """embeddings (large float lists) must never reach the WebSocket client."""
        from app.core.orchestrator import _safe_serialize
        state = {"embeddings": [[0.1] * 384] * 50, "query_type": "single"}
        result = _safe_serialize(state)
        assert "embeddings" not in result
        assert result["query_type"] == "single"

    def test_strips_none_values(self):
        """None values are stripped — clients treat missing keys as None."""
        from app.core.orchestrator import _safe_serialize
        state = {"answer": "Yes", "faithfulness_score": None, "kg_paths": None}
        result = _safe_serialize(state)
        assert "faithfulness_score" not in result
        assert "kg_paths" not in result
        assert result["answer"] == "Yes"

    def test_preserves_non_stripped_fields(self):
        from app.core.orchestrator import _safe_serialize
        state = {
            "answer":            "The accuracy is 97%.",
            "query_type":        "single",
            "faithfulness_score": 0.91,
            "latency_ms":        {"router": 312.0},
        }
        result = _safe_serialize(state)
        assert result["answer"] == "The accuracy is 97%."
        assert result["query_type"] == "single"
        assert result["faithfulness_score"] == 0.91

    def test_empty_state_returns_empty_dict(self):
        from app.core.orchestrator import _safe_serialize
        assert _safe_serialize({}) == {}

    def test_result_is_json_serializable(self):
        """Output must be serializable — WebSocket sends it as JSON."""
        from app.core.orchestrator import _safe_serialize
        state = {
            "answer":      "Answer here.",
            "sources":     [{"index": 1, "text": "Chunk.", "page": 1}],
            "latency_ms":  {"router": 100.5},
            "raw_bytes":   b"binary",
            "embeddings":  [[0.1, 0.2]],
        }
        result = _safe_serialize(state)
        # Must not raise
        json.dumps(result)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# direct_answer() Node Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDirectAnswer:

    def _run(self, state: dict, llm_response: str = "Cosine similarity measures angle."):
        """Run direct_answer() with a mocked LLM.

        direct_answer() builds its client via ``make_llm()`` (not a module-level
        ``ChatOpenAI`` symbol), so we patch ``make_llm`` and return a canned
        invoke() result. Returns (result, mock_llm_instance) so callers can
        inspect the prompt that was sent.
        """
        from app.core.orchestrator import direct_answer
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = MagicMock(content=llm_response)
        with patch("app.core.orchestrator.make_llm", return_value=mock_instance):
            result = direct_answer(state)
        return result, mock_instance

    def test_returns_answer_string(self):
        result, _ = self._run({"question": "What is cosine similarity?"})
        assert result["answer"] == "Cosine similarity measures angle."

    def test_sets_judge_passed_true(self):
        """Direct answers skip judging — judge_passed must be True."""
        result, _ = self._run({"question": "What is RAG?"})
        assert result["judge_passed"] is True

    def test_sets_faithfulness_score_to_one(self):
        """Parametric knowledge answers have no hallucination risk — score=1.0."""
        result, _ = self._run({"question": "What is RAG?"})
        assert result["faithfulness_score"] == 1.0

    def test_returns_empty_sources(self):
        """Direct answers cite no documents — sources must be []."""
        result, _ = self._run({"question": "What is RAG?"})
        assert result["sources"] == []

    def test_injects_memories_into_prompt(self):
        """Mem0 memories must appear in the LLM prompt for personalization."""
        from langchain_core.messages import HumanMessage

        state = {
            "question": "What is RAG?",
            "long_term_memories": [
                {"memory": "User works on flood prediction.", "score": 0.9},
            ],
        }
        _, mock_instance = self._run(state)

        call_messages = mock_instance.invoke.call_args[0][0]
        human_msg = call_messages[1]
        assert isinstance(human_msg, HumanMessage)
        assert "flood prediction" in human_msg.content

    def test_skips_memory_section_when_no_memories(self):

        _, mock_instance = self._run(
            {"question": "What is RAG?", "long_term_memories": []}
        )

        call_messages = mock_instance.invoke.call_args[0][0]
        human_msg = call_messages[1]
        assert "User context" not in human_msg.content

    def test_strips_whitespace_from_llm_response(self):
        result, _ = self._run({"question": "What is RAG?"}, llm_response="  Padded answer.  ")
        assert result["answer"] == "Padded answer."

    def test_output_keys_valid_in_agent_state(self):
        from app.core.state import AgentState
        valid_keys = set(AgentState.__annotations__.keys())
        result, _ = self._run({"question": "What is X?"})
        for key in result:
            assert key in valid_keys, f"direct_answer() returned invalid key '{key}'"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# merge_results() Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestMergeResults:

    def test_returns_none(self):
        """merge_results is a no-op convergence point — must return None."""
        from app.core.orchestrator import merge_results
        result = merge_results({
            "question": "Test?",
            "retrieved_chunks": [{"text": "chunk"}],
        })
        assert result is None

    def test_does_not_modify_state(self):
        from app.core.orchestrator import merge_results
        state = {"question": "Test?", "retrieved_chunks": [{"text": "chunk"}]}
        original_state = dict(state)
        merge_results(state)
        assert state == original_state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARAPOrchestrator.ingest() Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestOrchestratorIngest:

    def _make_orchestrator(self, final_state: dict):
        from app.core.orchestrator import ARAPOrchestrator
        orc = ARAPOrchestrator()
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = final_state
        orc._ingest_graph = mock_graph
        return orc

    @pytest.mark.asyncio
    async def test_returns_doc_id(self):
        final = make_final_state(doc_id="abc123", chunk_count=42)
        orc = self._make_orchestrator(final)
        result = await orc.ingest(b"%PDF-1.4", "test.pdf")
        assert result["doc_id"] == "abc123"

    @pytest.mark.asyncio
    async def test_returns_chunk_count(self):
        final = make_final_state(chunk_count=84)
        orc = self._make_orchestrator(final)
        result = await orc.ingest(b"%PDF-1.4", "test.pdf")
        assert result["chunk_count"] == 84

    @pytest.mark.asyncio
    async def test_returns_kg_triples_count(self):
        final = make_final_state(kg_entities=[
            {"head": "A", "relation": "r", "tail": "B", "confidence": 0.9},
            {"head": "C", "relation": "r", "tail": "D", "confidence": 0.8},
        ])
        orc = self._make_orchestrator(final)
        result = await orc.ingest(b"%PDF-1.4", "test.pdf")
        assert result["kg_triples"] == 2

    @pytest.mark.asyncio
    async def test_passes_pdf_bytes_and_filename_to_graph(self):
        final = make_final_state()
        orc = self._make_orchestrator(final)
        await orc.ingest(b"%PDF-1.4 binary", "report.pdf")

        init_state = orc._ingest_graph.invoke.call_args[0][0]
        assert init_state["raw_bytes"] == b"%PDF-1.4 binary"
        assert init_state["filename"] == "report.pdf"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARAPOrchestrator.query() Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestOrchestratorQuery:

    def _make_orchestrator(self, final_state: dict):
        from app.core.orchestrator import ARAPOrchestrator
        orc = ARAPOrchestrator()
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = final_state
        orc._query_graph = mock_graph
        orc._checkpointer = MagicMock()
        return orc

    @pytest.mark.asyncio
    async def test_returns_answer(self):
        final = make_final_state(answer="The accuracy is 97%.")
        orc = self._make_orchestrator(final)
        result = await orc.query("What accuracy?", "sess_1", "user_1")
        assert result["answer"] == "The accuracy is 97%."

    @pytest.mark.asyncio
    async def test_returns_sources(self):
        final = make_final_state()
        orc = self._make_orchestrator(final)
        result = await orc.query("What accuracy?", "sess_1", "user_1")
        assert isinstance(result["sources"], list)

    @pytest.mark.asyncio
    async def test_returns_query_type(self):
        final = make_final_state(query_type="multi_hop")
        orc = self._make_orchestrator(final)
        result = await orc.query("Complex question?", "sess_1", "user_1")
        assert result["query_type"] == "multi_hop"

    @pytest.mark.asyncio
    async def test_returns_faithfulness_score(self):
        final = make_final_state(faithfulness_score=0.88)
        orc = self._make_orchestrator(final)
        result = await orc.query("Question?", "sess_1", "user_1")
        assert abs(result["faithfulness_score"] - 0.88) < 0.001

    @pytest.mark.asyncio
    async def test_returns_latency_ms(self):
        final = make_final_state(latency_ms={"router": 312.0, "retrieval": 847.0})
        orc = self._make_orchestrator(final)
        result = await orc.query("Question?", "sess_1", "user_1")
        assert result["latency_ms"]["router"] == 312.0

    @pytest.mark.asyncio
    async def test_passes_session_id_as_thread_id(self):
        """
        LangGraph checkpointer uses thread_id = session_id.
        The config dict must carry session_id as thread_id.
        """
        final = make_final_state()
        orc = self._make_orchestrator(final)
        await orc.query("Question?", "my-session-abc", "user_1")

        call_args = orc._query_graph.invoke.call_args
        config = call_args[0][1]
        assert config["configurable"]["thread_id"] == "my-session-abc"

    @pytest.mark.asyncio
    async def test_initializes_retry_count_to_zero(self):
        """retry_count must start at 0 so the first generate() uses standard prompt."""
        final = make_final_state()
        orc = self._make_orchestrator(final)
        await orc.query("Question?", "sess_1", "user_1")
        init_state = orc._query_graph.invoke.call_args[0][0]
        assert init_state["retry_count"] == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARAPOrchestrator.health() Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestOrchestratorHealth:

    @pytest.mark.asyncio
    async def test_all_ok_when_services_reachable(self):
        from app.core.orchestrator import ARAPOrchestrator
        orc = ARAPOrchestrator()

        with patch("qdrant_client.QdrantClient") as MockQdrant, \
             patch("neo4j.GraphDatabase") as MockNeo4j, \
             patch("redis.from_url") as MockRedis:

            mock_qdrant = MagicMock()
            MockQdrant.return_value = mock_qdrant

            mock_driver = MagicMock()
            MockNeo4j.driver.return_value = mock_driver

            mock_redis = MagicMock()
            MockRedis.return_value = mock_redis

            status = await orc.health()

        assert status["qdrant"] == "ok"
        assert status["neo4j"] == "ok"
        assert status["redis"] == "ok"

    @pytest.mark.asyncio
    async def test_unreachable_when_services_fail(self):
        from app.core.orchestrator import ARAPOrchestrator
        orc = ARAPOrchestrator()

        with patch("qdrant_client.QdrantClient", side_effect=ConnectionError), \
             patch("neo4j.GraphDatabase") as MockNeo4j, \
             patch("redis.from_url") as MockRedis:

            MockNeo4j.driver.side_effect = ConnectionError
            MockRedis.side_effect = ConnectionError

            status = await orc.health()

        assert status["qdrant"] == "unreachable"
        assert status["neo4j"] == "unreachable"
        assert status["redis"] == "unreachable"

    @pytest.mark.asyncio
    async def test_never_raises_even_when_all_fail(self):
        """health() must always return a dict, never raise."""
        from app.core.orchestrator import ARAPOrchestrator
        orc = ARAPOrchestrator()

        with patch("qdrant_client.QdrantClient", side_effect=RuntimeError), \
             patch("neo4j.GraphDatabase", side_effect=RuntimeError), \
             patch("redis.from_url") as MockRedis:
            MockRedis.side_effect = RuntimeError
            result = await orc.health()

        assert isinstance(result, dict)
        assert "qdrant" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI Endpoint Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFastAPIHealth:

    @pytest.mark.asyncio
    async def test_health_returns_200(self):
        with patch("app.api.main.orchestrator") as mock_orc:
            mock_orc.health = AsyncMock(return_value={
                "qdrant": "ok", "neo4j": "ok", "redis": "ok"
            })
            async with make_async_client() as client:
                resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_returns_all_component_fields(self):
        with patch("app.api.main.orchestrator") as mock_orc:
            mock_orc.health = AsyncMock(return_value={
                "qdrant": "ok", "neo4j": "unreachable", "redis": "ok"
            })
            async with make_async_client() as client:
                resp = await client.get("/health")
        body = resp.json()
        assert "api" in body
        assert "qdrant" in body
        assert "neo4j" in body
        assert "redis" in body

    @pytest.mark.asyncio
    async def test_health_api_field_is_always_ok(self):
        with patch("app.api.main.orchestrator") as mock_orc:
            mock_orc.health = AsyncMock(return_value={
                "qdrant": "unreachable", "neo4j": "unreachable", "redis": "unreachable"
            })
            async with make_async_client() as client:
                resp = await client.get("/health")
        assert resp.json()["api"] == "ok"


class TestFastAPIIngest:
    """
    /ingest is asynchronous: it validates the upload, hands the bytes to a
    Celery task via ingest_document_task.delay(), and returns a task_id
    immediately (IngestResponse: task_id, status, message). The Celery task
    is mocked so nothing touches Redis or a real worker.
    """

    @pytest.mark.asyncio
    async def test_rejects_non_pdf_extension(self):
        async with make_async_client() as client:
            resp = await client.post(
                "/ingest",
                files={"file": ("document.txt", b"text content", "text/plain")},
            )
        assert resp.status_code == 400
        assert "PDF" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_rejects_oversized_file(self):
        """Files over 50MB must return 413."""
        large_content = b"A" * (51 * 1024 * 1024)
        async with make_async_client() as client:
            resp = await client.post(
                "/ingest",
                files={"file": ("big.pdf", large_content, "application/pdf")},
            )
        assert resp.status_code == 413

    @pytest.mark.asyncio
    async def test_valid_pdf_returns_200_with_correct_schema(self):
        mock_task = MagicMock()
        mock_task.id = "task-abc-123"
        with patch("app.api.main.ingest_document_task.delay",
                   return_value=mock_task) as mock_delay:
            async with make_async_client() as client:
                resp = await client.post(
                    "/ingest",
                    files={"file": ("test.pdf", b"%PDF-1.4 content here", "application/pdf")},
                )
        assert resp.status_code == 200
        body = resp.json()
        # Async ingest returns a queued-task envelope, not pipeline results.
        assert body["task_id"] == "task-abc-123"
        assert body["status"] == "queued"
        assert "message" in body
        mock_delay.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_returns_500_on_pipeline_error(self):
        # If enqueueing the Celery task fails (e.g. broker unreachable),
        # the endpoint surfaces a 500 rather than silently dropping the upload.
        with patch("app.api.main.ingest_document_task.delay",
                   side_effect=RuntimeError("broker down")):
            async with make_async_client() as client:
                resp = await client.post(
                    "/ingest",
                    files={"file": ("test.pdf", b"%PDF-1.4 content", "application/pdf")},
                )
        assert resp.status_code == 500


class TestFastAPIQuery:

    def _mock_query(self, answer: str = "The accuracy is 97%.", query_type: str = "single"):
        return patch(
            "app.api.main.orchestrator.query",
            new=AsyncMock(return_value={
                "answer":             answer,
                "sources":            [{
                    "index": 1, "text": "Chunk.", "page": 1,
                    "filename": "paper.pdf", "doc_id": "doc_abc",
                    "chunk_index": 0, "rerank_score": 0.92,
                }],
                "query_type":         query_type,
                "faithfulness_score": 0.91,
                "latency_ms":         {"router": 312.0},
            }),
        )

    @pytest.mark.asyncio
    async def test_valid_query_returns_200(self):
        with self._mock_query():
            async with make_async_client() as client:
                resp = await client.post(
                    "/query",
                    json={"question": "What accuracy did the model achieve?"},
                )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_response_has_all_required_fields(self):
        with self._mock_query():
            async with make_async_client() as client:
                resp = await client.post(
                    "/query",
                    json={"question": "What accuracy did the model achieve?"},
                )
        body = resp.json()
        required = {"answer", "sources", "query_type", "faithfulness_score",
                    "session_id", "latency_ms"}
        assert required.issubset(body.keys())

    @pytest.mark.asyncio
    async def test_rejects_question_shorter_than_3_chars(self):
        with patch("app.api.main.orchestrator"):
            async with make_async_client() as client:
                resp = await client.post("/query", json={"question": "AB"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_rejects_question_longer_than_2000_chars(self):
        with patch("app.api.main.orchestrator"):
            async with make_async_client() as client:
                resp = await client.post(
                    "/query", json={"question": "A" * 2001}
                )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_top_k_max_is_20(self):
        with patch("app.api.main.orchestrator"):
            async with make_async_client() as client:
                resp = await client.post(
                    "/query",
                    json={"question": "Valid question here?", "top_k": 25},
                )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_auto_generates_session_id_if_omitted(self):
        with self._mock_query():
            async with make_async_client() as client:
                resp = await client.post(
                    "/query",
                    json={"question": "What accuracy did the model achieve?"},
                )
        body = resp.json()
        assert "session_id" in body
        assert len(body["session_id"]) > 0

    @pytest.mark.asyncio
    async def test_returns_500_on_pipeline_error(self):
        with patch("app.api.main.orchestrator.query",
                   new=AsyncMock(side_effect=RuntimeError("Graph error"))):
            async with make_async_client() as client:
                resp = await client.post(
                    "/query",
                    json={"question": "What accuracy did the model achieve?"},
                )
        assert resp.status_code == 500


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WebSocket Protocol Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestWebSocket:

    async def _make_ws_client(self):
        """httpx WebSocket client via ASGI transport."""
        from httpx import ASGITransport, AsyncClient

        from app.api.main import app
        transport = ASGITransport(app=app)
        client = AsyncClient(transport=transport, base_url="http://test")
        return client

    def _mock_stream(self, events: list[dict]):
        """
        Mock orchestrator.stream_query() to yield the given events as async generator.
        """
        async def _gen(*args, **kwargs):
            for event in events:
                yield event

        return patch("app.api.main.orchestrator.stream_query", side_effect=_gen)

    @pytest.mark.asyncio
    async def test_websocket_sends_update_events(self):
        """Each LangGraph node must produce a 'type: update' event to the client."""
        from fastapi.testclient import TestClient

        from app.api.main import app

        stream_events = [
            {"node": "router",   "data": {"query_type": "single"}},
            {"node": "retrieve", "data": {"retrieval_score": 0.87}},
            {"node": "generate", "data": {"draft_answer": "Answer here."}},
            {"node": "judge",    "data": {
                "faithfulness_score": 0.91,
                "judge_passed": True,
                "answer": "Answer here.",
            }},
        ]

        with self._mock_stream(stream_events):
            with TestClient(app) as client:
                with client.websocket_connect("/ws/test-session") as ws:
                    ws.send_text(json.dumps({
                        "question": "What is the accuracy?",
                        "user_id":  "user_1",
                    }))
                    received = []
                    for _ in range(len(stream_events) + 1):  # +1 for "done"
                        msg = json.loads(ws.receive_text())
                        received.append(msg)

        types = [m["type"] for m in received]
        assert "update" in types
        assert "done" in types

    @pytest.mark.asyncio
    async def test_websocket_done_contains_answer(self):
        """The final 'done' message must contain the answer field."""
        from fastapi.testclient import TestClient

        from app.api.main import app

        stream_events = [
            {"node": "judge", "data": {
                "faithfulness_score": 0.91,
                "judge_passed": True,
                "answer": "The model achieved 97% accuracy.",
            }},
        ]

        with self._mock_stream(stream_events):
            with TestClient(app) as client:
                with client.websocket_connect("/ws/test-session") as ws:
                    ws.send_text(json.dumps({
                        "question": "What accuracy?",
                        "user_id":  "user_1",
                    }))
                    messages = []
                    for _ in range(2):
                        messages.append(json.loads(ws.receive_text()))

        done_msg = next(m for m in messages if m.get("type") == "done")
        assert done_msg["answer"] == "The model achieved 97% accuracy."

    @pytest.mark.asyncio
    async def test_websocket_sends_error_on_invalid_json(self):
        """Malformed JSON from client must produce error message, not disconnect."""
        from fastapi.testclient import TestClient

        from app.api.main import app

        with patch("app.api.main.orchestrator"):
            with TestClient(app) as client:
                with client.websocket_connect("/ws/test-session") as ws:
                    ws.send_text("this is not json {{{")
                    msg = json.loads(ws.receive_text())

        assert msg["type"] == "error"
        assert "JSON" in msg["message"]

    @pytest.mark.asyncio
    async def test_websocket_sends_error_on_empty_question(self):
        """Empty question must return error without calling pipeline."""
        from fastapi.testclient import TestClient

        from app.api.main import app

        with patch("app.api.main.orchestrator.stream_query") as mock_stream:
            with TestClient(app) as client:
                with client.websocket_connect("/ws/test-session") as ws:
                    ws.send_text(json.dumps({"question": ""}))
                    msg = json.loads(ws.receive_text())

        assert msg["type"] == "error"
        mock_stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_continues_after_pipeline_error(self):
        """
        If the pipeline raises during streaming, the connection must stay open.
        The client receives an error message and can send another question.
        """
        from fastapi.testclient import TestClient

        from app.api.main import app

        async def _failing_stream(*args, **kwargs):
            raise RuntimeError("Pipeline exploded")
            yield  # make it an async generator

        with patch("app.api.main.orchestrator.stream_query", side_effect=_failing_stream):
            with TestClient(app) as client:
                with client.websocket_connect("/ws/test-session") as ws:
                    ws.send_text(json.dumps({"question": "What is the model?"}))
                    msg = json.loads(ws.receive_text())

        assert msg["type"] == "error"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Graph Wiring Smoke Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGraphWiring:
    """
    Verify that ARAPOrchestrator can compile both graphs without errors.
    These tests catch misconfigured edges, missing node functions, or
    wrong conditional edge return values — all wiring bugs that would
    only surface at runtime otherwise.
    """

    def test_ingest_graph_compiles_without_error(self):
        """build_ingest_graph() must not raise."""
        from app.core.orchestrator import ARAPOrchestrator
        orc = ARAPOrchestrator()
        graph = orc.build_ingest_graph()
        assert graph is not None

    def test_query_graph_compiles_without_error(self):
        """build_query_graph() must not raise (even without Redis checkpointer)."""
        from app.core.orchestrator import ARAPOrchestrator
        orc = ARAPOrchestrator()
        orc._checkpointer = None   # skip Redis for this test
        graph = orc.build_query_graph()
        assert graph is not None

    def test_ingest_graph_is_cached_after_first_access(self):
        """Second access must return the SAME compiled graph object."""
        from app.core.orchestrator import ARAPOrchestrator
        orc = ARAPOrchestrator()
        g1 = orc.ingest_graph
        g2 = orc.ingest_graph
        assert g1 is g2

    def test_query_graph_is_cached_after_first_access(self):
        from app.core.orchestrator import ARAPOrchestrator
        orc = ARAPOrchestrator()
        orc._checkpointer = None
        g1 = orc.query_graph
        g2 = orc.query_graph
        assert g1 is g2

    def test_singleton_is_arap_orchestrator_instance(self):
        from app.core.orchestrator import ARAPOrchestrator, orchestrator
        assert isinstance(orchestrator, ARAPOrchestrator)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Additional coverage: branches not exercised by the happy-path tests above
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Targets (per coverage report):
#   orchestrator.py: 128-131 (LangSmith tracing env setup),
#                     296-298 (Redis checkpointer init success path),
#                     662-677 (stream_query() real body)
#   main.py:          123-132 (_bm25_reload_listener body),
#                     163-164 / 171-172 (lifespan except branches),
#                     178-182 (lifespan BM25 load + success log),
#                     451-466 (get_ingest_status SUCCESS/FAILURE/PENDING),
#                     531-545 (eval_endpoint try/except),
#                     558 / 564 / 570 / 581 (analytics endpoints),
#                     745 (WebSocket final_sources accumulation),
#                     773-774 (WebSocket top-level fatal except),
#                     781-789 (main() entry point — guard is pragmatized)


class TestOrchestratorCheckpointer:

    def test_checkpointer_initializes_when_redis_reachable(self):
        """RedisSaver(...) + .setup() + info log (orchestrator.py 296-298)."""
        from app.core.orchestrator import ARAPOrchestrator
        orc = ARAPOrchestrator()
        orc._checkpointer = None  # force the lazy-init branch
        fake_saver = MagicMock()
        with patch("langgraph.checkpoint.redis.RedisSaver", return_value=fake_saver):
            cp = orc.checkpointer
        assert cp is fake_saver
        fake_saver.setup.assert_called_once()

    def test_checkpointer_returns_none_when_redis_unreachable(self):
        """On any Redis error the checkpointer degrades to None (multi-turn
        memory off, but the app keeps working)."""
        from app.core.orchestrator import ARAPOrchestrator
        orc = ARAPOrchestrator()
        orc._checkpointer = None
        with patch("langgraph.checkpoint.redis.RedisSaver",
                   side_effect=ConnectionError("no redis")):
            cp = orc.checkpointer
        assert cp is None


class TestLangSmithTracing:

    def test_tracing_env_set_when_configured(self, monkeypatch):
        """The module-level LangSmith block (orchestrator.py 128-131) only
        runs at import time when both tracing flags are set, so we flip the
        settings and reload the module to exercise it."""
        import importlib
        import os

        from app.core import orchestrator

        monkeypatch.setattr(orchestrator.settings, "langchain_tracing_v2", True)
        monkeypatch.setattr(orchestrator.settings, "langchain_api_key", "test-key")
        monkeypatch.setattr(orchestrator.settings, "langchain_project", "proj")
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        try:
            importlib.reload(orchestrator)
            assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
            assert os.environ.get("LANGCHAIN_API_KEY") == "test-key"
            assert os.environ.get("LANGCHAIN_PROJECT") == "proj"
        finally:
            # Restore the original (tracing-off) module state.
            monkeypatch.undo()
            importlib.reload(orchestrator)


class TestStreamQuery:

    @pytest.mark.asyncio
    async def test_stream_query_yields_serialized_events(self):
        """stream_query() (orchestrator.py 662-677) wraps query_graph.stream()
        and applies _safe_serialize to every node update.

        Note: LangGraph's sync ``.stream()`` returns a regular (sync) generator,
        which stream_query iterates with a plain ``for`` — so the mock must be a
        sync generator too, while stream_query *itself* is an async generator.
        """
        from app.core.orchestrator import ARAPOrchestrator

        orc = ARAPOrchestrator()
        orc._checkpointer = None  # skip Redis for the cached graph
        # LangGraph stream_mode="updates" yields raw events of shape
        # {node_name: state_update_dict}; stream_query iterates event.items()
        # to split node_name / node_output and re-emits {"node", "data"}.
        raw_events = [
            {"router": {"query_type": "single"}},
            {"judge": {
                "answer": "A",
                "faithfulness_score": 0.9,
                "raw_bytes": b"should-be-stripped",
            }},
        ]

        def _fake_stream(*args, **kwargs):
            yield from raw_events

        orc._query_graph = MagicMock()
        orc._query_graph.stream.side_effect = _fake_stream

        collected = [ev async for ev in orc.stream_query("Question?", "sess", "user")]

        assert collected[0]["node"] == "router"
        assert "raw_bytes" not in collected[1]["data"]      # stripped by _safe_serialize
        assert collected[1]["data"]["answer"] == "A"


class TestIngestStatusEndpoint:

    @pytest.mark.asyncio
    async def test_status_success_returns_result(self):
        from app.api.main import get_ingest_status
        fake = MagicMock()
        fake.status = "SUCCESS"
        fake.result = {"doc_id": "d", "chunk_count": 2, "kg_triples": 1}
        with patch("app.api.main.AsyncResult", return_value=fake):
            resp = await get_ingest_status("t1")
        assert resp.status == "SUCCESS"
        assert resp.result["doc_id"] == "d"

    @pytest.mark.asyncio
    async def test_status_failure_returns_error(self):
        from app.api.main import get_ingest_status
        fake = MagicMock()
        fake.status = "FAILURE"
        fake.info = RuntimeError("boom")
        with patch("app.api.main.AsyncResult", return_value=fake):
            resp = await get_ingest_status("t2")
        assert resp.status == "FAILURE"
        assert resp.error is not None

    @pytest.mark.asyncio
    async def test_status_pending_has_no_extra_fields(self):
        from app.api.main import get_ingest_status
        fake = MagicMock()
        fake.status = "PENDING"
        with patch("app.api.main.AsyncResult", return_value=fake):
            resp = await get_ingest_status("t3")
        assert resp.status == "PENDING"
        assert resp.result is None
        assert resp.error is None


class TestEvalEndpoint:

    @pytest.mark.asyncio
    async def test_eval_returns_report(self):
        from app.api.main import EvalRequest, eval_endpoint
        report = {"run_id": 1, "faithfulness": 0.9}
        with patch("app.api.main.run_ragas_evaluation", new=AsyncMock(return_value=report)):
            resp = await eval_endpoint(EvalRequest(limit=5, no_seed=False))
        assert resp["run_id"] == 1

    @pytest.mark.asyncio
    async def test_eval_500_on_failure(self):
        from fastapi import HTTPException

        from app.api.main import EvalRequest, eval_endpoint
        with patch("app.api.main.run_ragas_evaluation",
                   new=AsyncMock(side_effect=RuntimeError("eval died"))):
            with pytest.raises(HTTPException) as exc:
                await eval_endpoint(EvalRequest())
        assert exc.value.status_code == 500


class TestAnalyticsEndpoints:

    @pytest.mark.asyncio
    async def test_analytics_summary(self):
        with patch("app.api.main.analytics_service.summary",
                   return_value={"total_queries": 5}):
            from app.api.main import analytics_summary
            resp = await analytics_summary()
        assert resp["total_queries"] == 5

    @pytest.mark.asyncio
    async def test_analytics_documents(self):
        with patch("app.api.main.analytics_service.top_documents",
                   return_value=[{"filename": "a.pdf"}]):
            from app.api.main import analytics_documents
            resp = await analytics_documents(limit=3)
        assert resp[0]["filename"] == "a.pdf"

    @pytest.mark.asyncio
    async def test_analytics_eval_trend(self):
        with patch("app.api.main.analytics_service.eval_trend",
                   return_value=[{"run_id": 1}]):
            from app.api.main import analytics_eval_trend
            resp = await analytics_eval_trend(limit=10)
        assert resp[0]["run_id"] == 1

    @pytest.mark.asyncio
    async def test_analytics_dashboard_returns_html(self):
        from app.api.main import analytics_dashboard
        resp = await analytics_dashboard()
        assert b"ARAP Analytics" in resp.body


class TestLifespan:

    @pytest.mark.asyncio
    async def test_startup_runs_full_warmup_and_load(self):
        """Covers lifespan.py 162 (warmup), 168-170 (graph compile),
        178-182 (BM25 load + success log)."""
        from app.api.main import app, lifespan

        emb = MagicMock()
        emb.warmup = MagicMock()
        orc = MagicMock()
        orc.ingest_graph = MagicMock()
        orc.query_graph = MagicMock()
        vs = MagicMock()
        vs.client = MagicMock()
        bm25 = MagicMock()
        bm25.size = 7

        with patch("app.services.embedder.embedder", emb), \
             patch("app.api.main.orchestrator", orc), \
             patch("app.services.vector_store.vector_store", vs), \
             patch("app.services.bm25_index.bm25_index", bm25), \
             patch("app.api.main._bm25_reload_listener"):
            async with lifespan(app):
                pass

        emb.warmup.assert_called_once()
        bm25.load_from_qdrant.assert_called_once()

    @pytest.mark.asyncio
    async def test_startup_warmup_failure_is_non_fatal(self):
        """Embedding warmup failure must be caught (lifespan.py 163-164)
        and startup must continue."""
        from app.api.main import app, lifespan

        emb = MagicMock()
        emb.warmup.side_effect = RuntimeError("no model")
        orc = MagicMock()
        orc.ingest_graph = MagicMock()
        orc.query_graph = MagicMock()
        vs = MagicMock()
        vs.client = MagicMock()
        bm25 = MagicMock()
        bm25.size = 7

        with patch("app.services.embedder.embedder", emb), \
             patch("app.api.main.orchestrator", orc), \
             patch("app.services.vector_store.vector_store", vs), \
             patch("app.services.bm25_index.bm25_index", bm25), \
             patch("app.api.main._bm25_reload_listener"):
            async with lifespan(app):
                pass
        emb.warmup.assert_called_once()

    @pytest.mark.asyncio
    async def test_startup_graph_compile_failure_is_logged(self):
        """Graph compilation failure must be caught (lifespan.py 171-172)."""
        from app.api.main import app, lifespan

        emb = MagicMock()
        emb.warmup = MagicMock()
        orc = MagicMock()
        # ingest_graph is a property on the real orchestrator (it compiles the
        # graph lazily), so accessing it can raise. A plain MagicMock attribute
        # would return itself on access without raising — make it raise on
        # *access* so the lifespan's except branch (171-172) is actually hit.
        def _raise_on_access(*args, **kwargs):
            raise RuntimeError("bad graph")
        type(orc).ingest_graph = property(_raise_on_access)
        orc.query_graph = MagicMock()
        vs = MagicMock()
        vs.client = MagicMock()
        bm25 = MagicMock()
        bm25.size = 7

        with patch("app.services.embedder.embedder", emb), \
             patch("app.api.main.orchestrator", orc), \
             patch("app.services.vector_store.vector_store", vs), \
             patch("app.services.bm25_index.bm25_index", bm25), \
             patch("app.api.main._bm25_reload_listener"):
            async with lifespan(app):
                pass


class TestBM25ReloadListener:

    def test_processes_reload_signal(self):
        """_bm25_reload_listener (main.py 123-132) must rebuild BM25 from
        Qdrant when it receives a pubsub 'message'."""
        from app.api.main import _bm25_reload_listener

        fake_pubsub = MagicMock()
        fake_pubsub.listen.return_value = iter([{"type": "message", "data": "reload"}])
        fake_redis = MagicMock()
        fake_redis.pubsub.return_value = fake_pubsub

        with patch("redis.from_url", return_value=fake_redis), \
             patch("app.services.bm25_index.bm25_index") as bm25, \
             patch("app.services.vector_store.vector_store") as vs:
            bm25.size = 10
            _bm25_reload_listener()

        bm25.load_from_qdrant.assert_called_once()
        # First positional arg must be the Qdrant client.
        assert bm25.load_from_qdrant.call_args.args[0] is vs.client


class TestWebSocketDirect:

    @pytest.mark.asyncio
    async def test_collects_sources_and_emits_done(self):
        """WebSocket final_sources accumulation (main.py 745) + consolidated
        'done' message (752-758)."""
        from fastapi import WebSocketDisconnect

        from app.api.main import websocket_query

        ws = MagicMock()
        ws.accept = AsyncMock()
        # First call delivers the question; second call ends the connection so
        # the `while True` receive loop terminates cleanly.
        ws.receive_text = AsyncMock(side_effect=[
            json.dumps({"question": "What?"}),
            WebSocketDisconnect(),
        ])
        ws.send_json = AsyncMock()

        async def _gen(*args, **kwargs):
            yield {"node": "generate", "data": {
                "answer": "A", "sources": [{"index": 1, "text": "t"}]}}
            yield {"node": "judge", "data": {
                "faithfulness_score": 0.9, "query_type": "single"}}

        with patch("app.api.main.orchestrator.stream_query", side_effect=_gen):
            await websocket_query(ws, "sess")

        sent = [call.args[0] for call in ws.send_json.call_args_list]
        done = next(m for m in sent if m.get("type") == "done")
        assert done["sources"] == [{"index": 1, "text": "t"}]
        assert done["faithfulness_score"] == 0.9

    @pytest.mark.asyncio
    async def test_fatal_exception_caught_by_outer_handler(self):
        """A non-WebSocketDisconnect error outside the per-message loop is
        caught by the top-level except (main.py 773-774) and does not
        propagate out of the endpoint."""
        from app.api.main import websocket_query

        ws = MagicMock()
        ws.accept = AsyncMock()
        ws.receive_text = AsyncMock(side_effect=RuntimeError("kaboom"))
        ws.send_json = AsyncMock()

        with patch("app.api.main.orchestrator"):
            # Must not raise — the outer except swallows it.
            await websocket_query(ws, "sess")


class TestMainEntrypoint:

    def test_main_launches_uvicorn(self):
        """main() (main.py 782-789) launches uvicorn with the app target."""
        from app.api.main import main
        with patch("uvicorn.run") as mock_run:
            main()
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == "app.api.main:app"
        assert kwargs.get("port") == 8000
