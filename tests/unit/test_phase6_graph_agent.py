"""
tests/unit/test_phase6_graph_agent.py
=======================================
Unit tests for the Knowledge Graph Agent (Phase 6).

Coverage:
  1.  Triple / TripleExtractionResult / EntityExtractionResult — Pydantic validation
  2.  _validate_cypher() — the safety layer, all forbidden keywords, missing LIMIT/MATCH
  3.  _extract_triples_from_text() — LLM call, JSON parsing, graceful failure
  4.  _batch_upsert_triples() — Neo4j write, empty input, exception handling
  5.  _extract_query_entities() — NER call, fallback on failure
  6.  _generate_cypher() — markdown fence stripping, fallback to empty string
  7.  _execute_cypher() — Neo4j read, result serialization, exception handling
  8.  extract_and_store_node() — full ingest node state contract
  9.  graph_retrieve() — full query node state contract, rejected Cypher path
  10. Downstream compatibility — output matches generator (Phase 7) expectations

All Neo4j and LLM calls are mocked. Tests run offline in < 2 seconds.

Run:
    pytest tests/unit/test_phase6_graph_agent.py -v
"""

import pytest
import json
from unittest.mock import MagicMock, patch


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures & Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_chunk(text: str, doc_id: str = "doc_abc", page: int = 1) -> dict:
    """Build a chunk dict matching Phase 2/3 output contract."""
    return {
        "text": text,
        "original_text": text,
        "page": page,
        "filename": "test.pdf",
        "doc_id": doc_id,
        "chunk_index": 0,
        "word_count": len(text.split()),
        "context_prepended": False,
    }


def make_state(
    question: str = "Which authors collaborated with MIT?",
    doc_id: str | None = "doc_xyz",
    chunks: list | None = None,
    latency_ms: dict | None = None,
) -> dict:
    return {
        "question":    question,
        "doc_id":      doc_id,
        "chunks":      chunks if chunks is not None else [],
        "user_id":     "user_123",
        "session_id":  "session_abc",
        "retry_count": 0,
        "latency_ms":  latency_ms or {},
    }


def make_extraction_json(triples: list[dict]) -> MagicMock:
    """Mock LLM response for triple extraction."""
    return MagicMock(content=json.dumps({"triples": triples}))


def make_entity_json(entities: list[str]) -> MagicMock:
    """Mock LLM response for query entity extraction."""
    return MagicMock(content=json.dumps({"entities": entities}))


def make_agent_with_mocks():
    """Build a KnowledgeGraphAgent with both LLMs and Neo4j driver mocked."""
    from app.agents.graph_agent import KnowledgeGraphAgent
    agent = KnowledgeGraphAgent()
    agent.extraction_llm = MagicMock()
    agent.cypher_llm = MagicMock()
    agent._driver = MagicMock()
    return agent


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Triple / Pydantic Model Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestTripleModel:

    def test_valid_triple_accepted(self):
        from app.agents.graph_agent import Triple
        t = Triple(head="FloodNet", relation="developed_by", tail="MIT", confidence=0.9)
        assert t.head == "FloodNet"
        assert t.confidence == 0.9

    def test_empty_head_rejected(self):
        from app.agents.graph_agent import Triple
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Triple(head="", relation="cites", tail="X", confidence=0.8)

    def test_confidence_above_one_rejected(self):
        """Unlike RouterOutput (which clamps), Triple uses strict ge/le bounds."""
        from app.agents.graph_agent import Triple
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Triple(head="A", relation="r", tail="B", confidence=1.5)

    def test_confidence_below_zero_rejected(self):
        from app.agents.graph_agent import Triple
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Triple(head="A", relation="r", tail="B", confidence=-0.1)

    def test_whitespace_is_stripped(self):
        from app.agents.graph_agent import Triple
        t = Triple(head="  FloodNet  ", relation=" developed_by ", tail=" MIT ", confidence=0.9)
        assert t.head == "FloodNet"
        assert t.relation == "developed_by"
        assert t.tail == "MIT"

    def test_extraction_result_defaults_to_empty_list(self):
        from app.agents.graph_agent import TripleExtractionResult
        result = TripleExtractionResult()
        assert result.triples == []

    def test_extraction_result_parses_nested_triples(self):
        from app.agents.graph_agent import TripleExtractionResult
        result = TripleExtractionResult(triples=[
            {"head": "A", "relation": "r1", "tail": "B", "confidence": 0.8},
            {"head": "C", "relation": "r2", "tail": "D", "confidence": 0.6},
        ])
        assert len(result.triples) == 2
        assert result.triples[0].head == "A"

    def test_entity_extraction_result_defaults_to_empty(self):
        from app.agents.graph_agent import EntityExtractionResult
        result = EntityExtractionResult()
        assert result.entities == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cypher Safety Validation Tests — the most critical security boundary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestValidateCypher:

    def test_valid_read_query_passes(self):
        from app.agents.graph_agent import _validate_cypher
        cypher = "MATCH (h:Entity)-[r:RELATES_TO]->(t:Entity) RETURN h.name, r.relation, t.name LIMIT 20"
        is_safe, reason = _validate_cypher(cypher)
        assert is_safe is True

    def test_empty_query_rejected(self):
        from app.agents.graph_agent import _validate_cypher
        is_safe, reason = _validate_cypher("")
        assert is_safe is False
        assert "Empty" in reason

    def test_whitespace_only_query_rejected(self):
        from app.agents.graph_agent import _validate_cypher
        is_safe, reason = _validate_cypher("   \n  ")
        assert is_safe is False

    @pytest.mark.parametrize("keyword", [
        "CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DETACH", "DROP",
    ])
    def test_each_forbidden_keyword_rejected(self, keyword):
        """Every single forbidden write keyword must be caught independently."""
        from app.agents.graph_agent import _validate_cypher
        cypher = f"MATCH (n) {keyword} (n) RETURN n LIMIT 10"
        is_safe, reason = _validate_cypher(cypher)
        assert is_safe is False
        assert keyword in reason

    def test_lowercase_forbidden_keyword_still_rejected(self):
        """Case-insensitivity: 'delete' (lowercase) must be caught too."""
        from app.agents.graph_agent import _validate_cypher
        cypher = "MATCH (n) delete n RETURN n LIMIT 10"
        is_safe, reason = _validate_cypher(cypher)
        assert is_safe is False

    def test_load_csv_rejected(self):
        from app.agents.graph_agent import _validate_cypher
        cypher = "LOAD CSV FROM 'file:///etc/passwd' AS row RETURN row LIMIT 10"
        is_safe, reason = _validate_cypher(cypher)
        assert is_safe is False

    def test_missing_limit_rejected(self):
        from app.agents.graph_agent import _validate_cypher
        cypher = "MATCH (h:Entity)-[r]->(t:Entity) RETURN h.name, t.name"
        is_safe, reason = _validate_cypher(cypher)
        assert is_safe is False
        assert "LIMIT" in reason

    def test_missing_match_rejected(self):
        from app.agents.graph_agent import _validate_cypher
        cypher = "RETURN 1 LIMIT 10"
        is_safe, reason = _validate_cypher(cypher)
        assert is_safe is False
        assert "MATCH" in reason

    def test_word_boundary_prevents_false_positive(self):
        """
        'Setting' contains 'SET' as a substring but is NOT the SET keyword.
        The regex word-boundary check must not flag this as forbidden.
        """
        from app.agents.graph_agent import _validate_cypher
        cypher = (
            "MATCH (h:Entity) WHERE h.name CONTAINS 'Setting' "
            "RETURN h.name LIMIT 10"
        )
        is_safe, reason = _validate_cypher(cypher)
        assert is_safe is True, f"False positive rejection: {reason}"

    def test_optional_match_with_relates_to_passes(self):
        from app.agents.graph_agent import _validate_cypher
        cypher = (
            "MATCH (h:Entity {name: 'X'}) "
            "OPTIONAL MATCH (h)-[r:RELATES_TO]->(t:Entity) "
            "RETURN h.name, r.relation, t.name LIMIT 15"
        )
        is_safe, _ = _validate_cypher(cypher)
        assert is_safe is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Triple Extraction (Ingestion) Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestExtractTriplesFromText:

    def test_returns_validated_triples_on_success(self):
        agent = make_agent_with_mocks()
        agent.extraction_llm.invoke.return_value = make_extraction_json([
            {"head": "FloodNet", "relation": "developed_by", "tail": "MIT", "confidence": 0.9},
        ])
        result = agent._extract_triples_from_text(
            "FloodNet was developed by researchers at MIT using deep learning."
        )
        assert len(result) == 1
        assert result[0].head == "FloodNet"

    def test_returns_empty_list_for_short_text(self):
        """Text under 10 words is too short to contain meaningful relationships."""
        agent = make_agent_with_mocks()
        result = agent._extract_triples_from_text("Too short.")
        assert result == []
        agent.extraction_llm.invoke.assert_not_called()

    def test_returns_empty_list_for_empty_text(self):
        agent = make_agent_with_mocks()
        result = agent._extract_triples_from_text("")
        assert result == []

    def test_returns_empty_list_on_json_parse_error(self):
        agent = make_agent_with_mocks()
        agent.extraction_llm.invoke.return_value = MagicMock(content="not valid json at all")
        result = agent._extract_triples_from_text("A valid long text passage about something important here.")
        assert result == []

    def test_returns_empty_list_on_llm_exception(self):
        agent = make_agent_with_mocks()
        agent.extraction_llm.invoke.side_effect = ConnectionError("API timeout")
        result = agent._extract_triples_from_text("A valid long text passage about something important here.")
        assert result == []

    def test_empty_triples_array_handled_gracefully(self):
        """LLM correctly identifying no relationships must return [] without error."""
        agent = make_agent_with_mocks()
        agent.extraction_llm.invoke.return_value = make_extraction_json([])
        result = agent._extract_triples_from_text("A long passage with no clear entity relationships at all here.")
        assert result == []

    def test_truncates_long_text_before_extraction(self):
        """Text over 1500 words must be truncated to control cost."""
        agent = make_agent_with_mocks()
        agent.extraction_llm.invoke.return_value = make_extraction_json([])
        long_text = " ".join([f"word{i}" for i in range(3000)])
        agent._call_extraction_llm(long_text)
        sent_content = agent.extraction_llm.invoke.call_args[0][0][1].content
        word_count_sent = len(sent_content.replace("Text:\n", "").split())
        assert word_count_sent <= 1500


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Neo4j Batch Write Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBatchUpsertTriples:

    def test_empty_triples_returns_zero_without_neo4j_call(self):
        agent = make_agent_with_mocks()
        result = agent._batch_upsert_triples([], doc_id="doc_1", chunks=[])
        assert result == 0
        agent._driver.session.assert_not_called()

    def test_returns_count_on_successful_write(self):
        from app.agents.graph_agent import Triple
        agent = make_agent_with_mocks()
        mock_session = MagicMock()
        agent._driver.session.return_value.__enter__.return_value = mock_session

        triples = [
            Triple(head="A", relation="r1", tail="B", confidence=0.8),
            Triple(head="C", relation="r2", tail="D", confidence=0.7),
        ]
        result = agent._batch_upsert_triples(triples, doc_id="doc_1", chunks=[])
        assert result == 2

    def test_calls_execute_write_not_execute_read(self):
        """Writing triples must use execute_write (Neo4j enforces write permission)."""
        from app.agents.graph_agent import Triple
        agent = make_agent_with_mocks()
        mock_session = MagicMock()
        agent._driver.session.return_value.__enter__.return_value = mock_session

        triples = [Triple(head="A", relation="r", tail="B", confidence=0.8)]
        agent._batch_upsert_triples(triples, doc_id="doc_1", chunks=[])
        mock_session.execute_write.assert_called_once()

    def test_neo4j_exception_returns_zero_gracefully(self):
        """Neo4j write failures must not raise — ingestion continues."""
        from app.agents.graph_agent import Triple
        agent = make_agent_with_mocks()
        agent._driver.session.side_effect = ConnectionError("Neo4j unreachable")

        triples = [Triple(head="A", relation="r", tail="B", confidence=0.8)]
        result = agent._batch_upsert_triples(triples, doc_id="doc_1", chunks=[])
        assert result == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Query-Time Entity Extraction Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestExtractQueryEntities:

    def test_returns_entities_on_success(self):
        agent = make_agent_with_mocks()
        agent.extraction_llm.invoke.return_value = make_entity_json(["FloodNet", "MIT"])
        result = agent._extract_query_entities("Which organizations is FloodNet associated with?")
        assert result == ["FloodNet", "MIT"]

    def test_returns_empty_list_when_no_entities_found(self):
        agent = make_agent_with_mocks()
        agent.extraction_llm.invoke.return_value = make_entity_json([])
        result = agent._extract_query_entities("What is the weather like?")
        assert result == []

    def test_returns_empty_list_on_exception(self):
        agent = make_agent_with_mocks()
        agent.extraction_llm.invoke.side_effect = RuntimeError("API down")
        result = agent._extract_query_entities("Some question?")
        assert result == []

    def test_returns_empty_list_on_invalid_json(self):
        agent = make_agent_with_mocks()
        agent.extraction_llm.invoke.return_value = MagicMock(content="not json")
        result = agent._extract_query_entities("Some question?")
        assert result == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cypher Generation Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGenerateCypher:

    def test_returns_llm_content_stripped(self):
        agent = make_agent_with_mocks()
        agent.cypher_llm.invoke.return_value = MagicMock(
            content="  MATCH (n) RETURN n LIMIT 10  "
        )
        result = agent._generate_cypher("Question?", ["X"])
        assert result == "MATCH (n) RETURN n LIMIT 10"

    def test_strips_markdown_cypher_fence(self):
        """LLM sometimes wraps output in ```cypher despite instructions not to."""
        agent = make_agent_with_mocks()
        agent.cypher_llm.invoke.return_value = MagicMock(
            content="```cypher\nMATCH (n) RETURN n LIMIT 10\n```"
        )
        result = agent._generate_cypher("Question?", ["X"])
        assert "```" not in result
        assert result == "MATCH (n) RETURN n LIMIT 10"

    def test_strips_plain_markdown_fence(self):
        agent = make_agent_with_mocks()
        agent.cypher_llm.invoke.return_value = MagicMock(
            content="```\nMATCH (n) RETURN n LIMIT 10\n```"
        )
        result = agent._generate_cypher("Question?", ["X"])
        assert "```" not in result

    def test_returns_empty_string_on_llm_failure(self):
        """On failure, must return empty string — _validate_cypher() rejects it downstream."""
        agent = make_agent_with_mocks()
        agent.cypher_llm.invoke.side_effect = TimeoutError("LLM timeout")
        result = agent._generate_cypher("Question?", ["X"])
        assert result == ""

    def test_handles_empty_entity_list(self):
        """No entities found must not crash Cypher generation."""
        agent = make_agent_with_mocks()
        agent.cypher_llm.invoke.return_value = MagicMock(content="MATCH (n) RETURN n LIMIT 5")
        result = agent._generate_cypher("Vague question?", [])
        assert result == "MATCH (n) RETURN n LIMIT 5"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cypher Execution Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestExecuteCypher:

    def _make_record(self, **kwargs) -> MagicMock:
        """Mock a Neo4j Record object with .items() support."""
        record = MagicMock()
        record.items.return_value = list(kwargs.items())
        return record

    def test_returns_serialized_string_dicts(self):
        agent = make_agent_with_mocks()
        mock_session = MagicMock()
        mock_session.execute_read.return_value = [
            self._make_record(head="FloodNet", relation="developed_by", tail="MIT"),
        ]
        agent._driver.session.return_value.__enter__.return_value = mock_session

        result = agent._execute_cypher("MATCH (n) RETURN n LIMIT 10", doc_id="doc_1")
        assert len(result) == 1
        assert result[0]["head"] == "FloodNet"
        assert isinstance(result[0]["head"], str)

    def test_calls_execute_read_not_execute_write(self):
        """Query execution must use execute_read — the driver-level read-only guarantee."""
        agent = make_agent_with_mocks()
        mock_session = MagicMock()
        mock_session.execute_read.return_value = []
        agent._driver.session.return_value.__enter__.return_value = mock_session

        agent._execute_cypher("MATCH (n) RETURN n LIMIT 10", doc_id="doc_1")
        mock_session.execute_read.assert_called_once()

    def test_caps_results_at_20(self):
        """Even if Cypher LIMIT is bypassed somehow, we cap at 20 defensively."""
        agent = make_agent_with_mocks()
        mock_session = MagicMock()
        mock_session.execute_read.return_value = [
            self._make_record(name=f"entity_{i}") for i in range(30)
        ]
        agent._driver.session.return_value.__enter__.return_value = mock_session

        result = agent._execute_cypher("MATCH (n) RETURN n LIMIT 30", doc_id=None)
        assert len(result) <= 20

    def test_returns_empty_list_on_neo4j_exception(self):
        agent = make_agent_with_mocks()
        agent._driver.session.side_effect = ConnectionError("Neo4j unreachable")
        result = agent._execute_cypher("MATCH (n) RETURN n LIMIT 10", doc_id="doc_1")
        assert result == []

    def test_works_without_doc_id(self):
        """doc_id=None must not crash — params dict simply omits it."""
        agent = make_agent_with_mocks()
        mock_session = MagicMock()
        mock_session.execute_read.return_value = []
        agent._driver.session.return_value.__enter__.return_value = mock_session

        result = agent._execute_cypher("MATCH (n) RETURN n LIMIT 10", doc_id=None)
        assert result == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# extract_and_store_node() — Ingest Graph Node Contract Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestExtractAndStoreNode:

    def test_returns_kg_entities_key(self):
        agent = make_agent_with_mocks()
        agent.extraction_llm.invoke.return_value = make_extraction_json([
            {"head": "A", "relation": "r", "tail": "B", "confidence": 0.8},
        ])
        mock_session = MagicMock()
        agent._driver.session.return_value.__enter__.return_value = mock_session

        chunks = [make_chunk("FloodNet was developed by researchers at MIT in 2024.")]
        result = agent.extract_and_store_node(make_state(chunks=chunks))
        assert "kg_entities" in result

    def test_empty_chunks_returns_empty_kg_entities(self):
        agent = make_agent_with_mocks()
        result = agent.extract_and_store_node(make_state(chunks=[]))
        assert result == {"kg_entities": []}
        agent.extraction_llm.invoke.assert_not_called()

    def test_no_triples_found_returns_empty_list(self):
        """When LLM finds no relationships across all chunks, must not crash."""
        agent = make_agent_with_mocks()
        agent.extraction_llm.invoke.return_value = make_extraction_json([])
        chunks = [make_chunk("A passage with absolutely no entity relationships present.")]
        result = agent.extract_and_store_node(make_state(chunks=chunks))
        assert result["kg_entities"] == []

    def test_kg_entities_are_plain_dicts_not_pydantic_models(self):
        """
        AgentState is a TypedDict requiring JSON-serializable values.
        kg_entities must contain plain dicts, not Triple objects.
        """
        agent = make_agent_with_mocks()
        agent.extraction_llm.invoke.return_value = make_extraction_json([
            {"head": "X", "relation": "uses", "tail": "Y", "confidence": 0.9},
        ])
        mock_session = MagicMock()
        agent._driver.session.return_value.__enter__.return_value = mock_session

        chunks = [make_chunk("X uses Y as its primary underlying technology stack here.")]
        result = agent.extract_and_store_node(make_state(chunks=chunks))
        for entity in result["kg_entities"]:
            assert isinstance(entity, dict)
            assert not hasattr(entity, "model_dump")  # not a Pydantic model

    def test_one_failed_chunk_does_not_block_others(self):
        """
        If chunk 1's extraction fails (exception), chunk 2's triples must
        still be extracted and included in the result.
        """
        agent = make_agent_with_mocks()
        call_count = [0]

        def side_effect(messages):
            call_count[0] += 1
            if call_count[0] == 1:
                raise TimeoutError("Simulated failure on first chunk")
            return make_extraction_json([
                {"head": "C", "relation": "r", "tail": "D", "confidence": 0.7}
            ])

        agent.extraction_llm.invoke.side_effect = side_effect
        mock_session = MagicMock()
        agent._driver.session.return_value.__enter__.return_value = mock_session

        chunks = [
            make_chunk("First chunk with enough words to pass the minimum length check."),
            make_chunk("Second chunk with enough words to pass the minimum length check."),
        ]
        result = agent.extract_and_store_node(make_state(chunks=chunks))
        assert len(result["kg_entities"]) == 1
        assert result["kg_entities"][0]["head"] == "C"

    def test_output_keys_valid_in_agent_state(self):
        """kg_entities must be a valid AgentState field."""
        from app.core.state import AgentState
        valid_keys = set(AgentState.__annotations__.keys())
        agent = make_agent_with_mocks()
        result = agent.extract_and_store_node(make_state(chunks=[]))
        for key in result:
            assert key in valid_keys


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# graph_retrieve() — Query Graph Node Contract Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGraphRetrieveNode:

    def _make_record(self, **kwargs) -> MagicMock:
        record = MagicMock()
        record.items.return_value = list(kwargs.items())
        return record

    def test_returns_kg_paths_key(self):
        agent = make_agent_with_mocks()
        agent.extraction_llm.invoke.return_value = make_entity_json(["FloodNet"])
        agent.cypher_llm.invoke.return_value = MagicMock(
            content="MATCH (h:Entity)-[r:RELATES_TO]->(t:Entity) WHERE toLower(h.name) CONTAINS toLower('FloodNet') RETURN h.name, r.relation, t.name LIMIT 20"
        )
        mock_session = MagicMock()
        mock_session.execute_read.return_value = [
            self._make_record(head="FloodNet", relation="developed_by", tail="MIT")
        ]
        agent._driver.session.return_value.__enter__.return_value = mock_session

        result = agent.graph_retrieve(make_state())
        assert "kg_paths" in result
        assert len(result["kg_paths"]) == 1

    def test_returns_latency_ms_with_graph_key(self):
        agent = make_agent_with_mocks()
        agent.extraction_llm.invoke.return_value = make_entity_json([])
        agent.cypher_llm.invoke.return_value = MagicMock(content="MATCH (n) RETURN n LIMIT 5")
        mock_session = MagicMock()
        mock_session.execute_read.return_value = []
        agent._driver.session.return_value.__enter__.return_value = mock_session

        result = agent.graph_retrieve(make_state())
        assert "graph" in result["latency_ms"]

    def test_preserves_existing_latency_keys(self):
        """graph_retrieve() must add 'graph' to latency_ms, not replace existing keys."""
        agent = make_agent_with_mocks()
        agent.extraction_llm.invoke.return_value = make_entity_json([])
        agent.cypher_llm.invoke.return_value = MagicMock(content="MATCH (n) RETURN n LIMIT 5")
        mock_session = MagicMock()
        mock_session.execute_read.return_value = []
        agent._driver.session.return_value.__enter__.return_value = mock_session

        state = make_state(latency_ms={"router": 200.0})
        result = agent.graph_retrieve(state)
        assert result["latency_ms"]["router"] == 200.0
        assert "graph" in result["latency_ms"]

    def test_empty_question_returns_empty_paths(self):
        agent = make_agent_with_mocks()
        result = agent.graph_retrieve(make_state(question=""))
        assert result["kg_paths"] == []
        agent.extraction_llm.invoke.assert_not_called()

    def test_rejected_cypher_returns_empty_paths_not_exception(self):
        """
        If the LLM generates unsafe Cypher (e.g. via prompt injection),
        graph_retrieve() must return empty kg_paths, NOT execute the query
        or raise an exception. This is the critical security test.
        """
        agent = make_agent_with_mocks()
        agent.extraction_llm.invoke.return_value = make_entity_json(["X"])
        agent.cypher_llm.invoke.return_value = MagicMock(
            content="MATCH (n) DETACH DELETE n RETURN n LIMIT 10"
        )
        result = agent.graph_retrieve(make_state())
        assert result["kg_paths"] == []
        # Critically: Neo4j session must NEVER have been opened for execution
        agent._driver.session.assert_not_called()

    def test_rejected_cypher_missing_limit_blocks_execution(self):
        agent = make_agent_with_mocks()
        agent.extraction_llm.invoke.return_value = make_entity_json(["X"])
        agent.cypher_llm.invoke.return_value = MagicMock(
            content="MATCH (n) RETURN n"   # no LIMIT
        )
        result = agent.graph_retrieve(make_state())
        assert result["kg_paths"] == []
        agent._driver.session.assert_not_called()

    def test_output_keys_valid_in_agent_state(self):
        from app.core.state import AgentState
        valid_keys = set(AgentState.__annotations__.keys())
        agent = make_agent_with_mocks()
        agent.extraction_llm.invoke.return_value = make_entity_json([])
        agent.cypher_llm.invoke.return_value = MagicMock(content="MATCH (n) RETURN n LIMIT 5")
        mock_session = MagicMock()
        mock_session.execute_read.return_value = []
        agent._driver.session.return_value.__enter__.return_value = mock_session

        result = agent.graph_retrieve(make_state())
        for key in result:
            assert key in valid_keys, f"graph_retrieve() returned invalid key '{key}'"

    def test_doc_id_passed_through_to_execution(self):
        agent = make_agent_with_mocks()
        agent.extraction_llm.invoke.return_value = make_entity_json(["X"])
        agent.cypher_llm.invoke.return_value = MagicMock(
            content="MATCH (h:Entity) WHERE h.doc_id = $doc_id RETURN h.name LIMIT 10"
        )
        mock_session = MagicMock()
        mock_session.execute_read.return_value = []
        agent._driver.session.return_value.__enter__.return_value = mock_session

        agent.graph_retrieve(make_state(doc_id="doc_specific_123"))
        # Verify session.execute_read was called (doc_id flows into _execute_cypher)
        mock_session.execute_read.assert_called_once()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Downstream Compatibility — Phase 7 Generator Expectations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDownstreamCompatibility:
    """
    Verify graph_retrieve() output matches what Phase 7 (generator) expects.

    Generator._build_context() iterates kg_paths and renders them in the
    "[Knowledge Graph Paths]" section of the LLM prompt — it expects a
    list of dicts (any shape; generator just str()-renders each one).
    """

    def _make_record(self, **kwargs) -> MagicMock:
        record = MagicMock()
        record.items.return_value = list(kwargs.items())
        return record

    def test_kg_paths_is_always_a_list(self):
        agent = make_agent_with_mocks()
        agent.extraction_llm.invoke.return_value = make_entity_json([])
        agent.cypher_llm.invoke.return_value = MagicMock(content="MATCH (n) RETURN n LIMIT 5")
        mock_session = MagicMock()
        mock_session.execute_read.return_value = []
        agent._driver.session.return_value.__enter__.return_value = mock_session

        result = agent.graph_retrieve(make_state())
        assert isinstance(result["kg_paths"], list)

    def test_kg_paths_entries_are_json_serializable(self):
        """
        Every value in kg_paths entries must be a plain string (not a Neo4j
        Node/Relationship object), since AgentState requires serializable values
        for LangGraph's Redis checkpointing (Phase 8).
        """
        import json as json_module
        agent = make_agent_with_mocks()
        agent.extraction_llm.invoke.return_value = make_entity_json(["X"])
        agent.cypher_llm.invoke.return_value = MagicMock(
            content="MATCH (h:Entity) RETURN h.name AS head LIMIT 10"
        )
        mock_session = MagicMock()
        mock_session.execute_read.return_value = [self._make_record(head="EntityX")]
        agent._driver.session.return_value.__enter__.return_value = mock_session

        result = agent.graph_retrieve(make_state())
        # This must not raise — proves all values are JSON-serializable
        json_module.dumps(result["kg_paths"])

    def test_singleton_is_knowledge_graph_agent_instance(self):
        from app.agents.graph_agent import kg_agent, KnowledgeGraphAgent
        assert isinstance(kg_agent, KnowledgeGraphAgent)
