"""
tests/unit/test_phase4_router.py
==================================
Unit tests for the Adaptive Query Router (Phase 4).

What we test:
  1. RouterOutput model — validation, clamping, invalid type rejection
  2. RouterAgent.route() — state read/write contract with AgentState
  3. RouterAgent._classify() — LLM call, JSON parsing, fallback behavior
  4. RouterAgent.get_route() — conditional edge function correctness
  5. RouterAgent._fetch_memories() — Mem0 integration, error handling
  6. End-to-end state flow — route() output feeds correctly into downstream nodes

What we do NOT test:
  - Actual LLM classification quality (that's an eval/prompt engineering concern)
  - Actual Mem0 connectivity (integration test, not unit test)
  - LangGraph graph wiring (tested in orchestrator integration tests)

All LLM and Mem0 calls are mocked. Tests run offline in < 1 second.

Run:
    pytest tests/unit/test_phase4_router.py -v
"""

import pytest
import json
import time
from unittest.mock import MagicMock, patch, PropertyMock


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_state(
    question: str = "What is the main finding?",
    user_id: str = "user_123",
    session_id: str = "session_abc",
    doc_id: str | None = "doc_xyz",
    latency_ms: dict | None = None,
) -> dict:
    """
    Build a minimal AgentState dict for router input.
    Matches the fields router.route() reads from state.
    """
    return {
        "question":   question,
        "user_id":    user_id,
        "session_id": session_id,
        "doc_id":     doc_id,
        "top_k":      5,
        "retry_count": 0,
        "latency_ms": latency_ms or {},
    }


def make_llm_response(
    query_type: str = "single",
    confidence: float = 0.92,
    reason: str = "Simple factual question requiring one retrieval round.",
) -> MagicMock:
    """
    Build a mock ChatOpenAI response matching what the router LLM returns.
    The router expects response.content to be a valid JSON string.
    """
    payload = json.dumps({
        "type":       query_type,
        "confidence": confidence,
        "reason":     reason,
    })
    mock_response = MagicMock()
    mock_response.content = payload
    return mock_response


def make_router(llm_response: MagicMock | None = None, mem0_memories: list | None = None):
    """
    Create a RouterAgent with all external dependencies mocked.

    Args:
        llm_response:  What the mock LLM returns. None → default single response.
        mem0_memories: What mock Mem0 returns. None → empty list.
    """
    from app.agents.router import RouterAgent

    agent = RouterAgent()

    # Mock LLM
    agent.llm = MagicMock()
    agent.llm.invoke.return_value = llm_response or make_llm_response()

    # Mock Mem0 client
    mock_mem0 = MagicMock()
    mock_mem0.search.return_value = mem0_memories or []
    agent._mem0_client = mock_mem0

    return agent


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RouterOutput Model Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRouterOutput:
    """
    Tests for the Pydantic RouterOutput model.
    This model validates and normalizes LLM JSON output before it touches state.
    """

    def test_valid_direct_type_is_accepted(self):
        from app.agents.router import RouterOutput
        out = RouterOutput(type="direct", confidence=0.95, reason="General knowledge.")
        assert out.type == "direct"

    def test_valid_single_type_is_accepted(self):
        from app.agents.router import RouterOutput
        out = RouterOutput(type="single", confidence=0.88, reason="One retrieval.")
        assert out.type == "single"

    def test_valid_multi_hop_type_is_accepted(self):
        from app.agents.router import RouterOutput
        out = RouterOutput(type="multi_hop", confidence=0.80, reason="Multi-step.")
        assert out.type == "multi_hop"

    def test_valid_graph_type_is_accepted(self):
        from app.agents.router import RouterOutput
        out = RouterOutput(type="graph", confidence=0.75, reason="Entity relationship.")
        assert out.type == "graph"

    def test_invalid_type_raises_validation_error(self):
        """
        Any type not in ('direct', 'single', 'multi_hop', 'graph') must fail.
        This catches LLM hallucinations like "complex" or "vector_search".
        """
        from app.agents.router import RouterOutput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            RouterOutput(type="complex", confidence=0.9, reason="Invalid type.")

    def test_confidence_below_zero_is_clamped_to_zero(self):
        """LLM sometimes returns -0.1 or other negative values. Must clamp to 0."""
        from app.agents.router import RouterOutput
        out = RouterOutput(type="single", confidence=-0.5, reason="test")
        assert out.confidence == 0.0

    def test_confidence_above_one_is_clamped_to_one(self):
        """LLM sometimes returns 1.05 or 2.0. Must clamp to 1.0."""
        from app.agents.router import RouterOutput
        out = RouterOutput(type="single", confidence=1.5, reason="test")
        assert out.confidence == 1.0

    def test_confidence_boundary_zero_is_valid(self):
        from app.agents.router import RouterOutput
        out = RouterOutput(type="direct", confidence=0.0, reason="Uncertain.")
        assert out.confidence == 0.0

    def test_confidence_boundary_one_is_valid(self):
        from app.agents.router import RouterOutput
        out = RouterOutput(type="graph", confidence=1.0, reason="Certain.")
        assert out.confidence == 1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# route() Node — State Contract Tests
# These verify the exact fields written to AgentState.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRouteNodeStateContract:
    """
    Tests for RouterAgent.route() — the LangGraph node function.

    The contract:
      INPUT:  AgentState with {question, user_id, latency_ms}
      OUTPUT: partial dict with {query_type, routing_confidence,
                                  long_term_memories, latency_ms}

    The output dict is what LangGraph merges into the running state.
    Every key in the output must be a valid AgentState field.
    """

    def test_route_returns_query_type(self):
        agent = make_router(make_llm_response("multi_hop", 0.85, "Multi-step reasoning."))
        result = agent.route(make_state("How does section 3 relate to section 7?"))
        assert result["query_type"] == "multi_hop"

    def test_route_returns_routing_confidence(self):
        agent = make_router(make_llm_response("single", 0.93, "Simple factual."))
        result = agent.route(make_state("What is the paper's main contribution?"))
        assert abs(result["routing_confidence"] - 0.93) < 0.001

    def test_route_returns_long_term_memories_list(self):
        """long_term_memories must always be a list (possibly empty)."""
        agent = make_router()
        result = agent.route(make_state())
        assert isinstance(result["long_term_memories"], list)

    def test_route_returns_latency_ms_dict(self):
        """latency_ms must be a dict with a 'router' key."""
        agent = make_router()
        result = agent.route(make_state())
        assert isinstance(result["latency_ms"], dict)
        assert "router" in result["latency_ms"]

    def test_route_router_latency_is_positive_float(self):
        """Router latency must be a positive number (execution time in ms)."""
        agent = make_router()
        result = agent.route(make_state())
        assert result["latency_ms"]["router"] >= 0.0

    def test_route_preserves_existing_latency_keys(self):
        """
        route() must ADD 'router' to existing latency_ms, not replace it.
        Previous nodes may have already written their latency values.
        This is a critical contract — losing latency data breaks observability.
        """
        state = make_state(latency_ms={"some_previous_node": 123.4})
        agent = make_router()
        result = agent.route(state)
        # Both the previous key and the new 'router' key must be present
        assert "some_previous_node" in result["latency_ms"]
        assert "router" in result["latency_ms"]

    def test_route_output_keys_are_valid_agent_state_fields(self):
        """
        All keys returned by route() must be valid AgentState TypedDict fields.
        Extra keys would be silently ignored by LangGraph, causing subtle bugs.
        """
        from app.core.state import AgentState
        valid_keys = set(AgentState.__annotations__.keys())

        agent = make_router()
        result = agent.route(make_state())

        for key in result.keys():
            assert key in valid_keys, (
                f"route() returned '{key}' which is not a field in AgentState. "
                f"Add it to state.py or remove it from the return dict."
            )

    def test_route_returns_only_four_keys(self):
        """
        route() must return EXACTLY four keys — no more, no less.
        Additional keys would be unexpected mutations of state.
        """
        agent = make_router()
        result = agent.route(make_state())
        expected_keys = {"query_type", "routing_confidence", "long_term_memories", "latency_ms"}
        assert set(result.keys()) == expected_keys


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _classify() — LLM Call and Error Handling Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestClassify:

    def test_classify_returns_router_output_on_valid_json(self):
        from app.agents.router import RouterAgent, RouterOutput
        agent = RouterAgent()
        agent.llm = MagicMock()
        agent.llm.invoke.return_value = make_llm_response("graph", 0.88, "Entity relation.")
        result = agent._classify("Which authors co-appear?")
        assert isinstance(result, RouterOutput)
        assert result.type == "graph"
        assert result.confidence == 0.88

    def test_classify_falls_back_to_single_on_json_error(self):
        """
        If LLM returns non-JSON (prompt injection, markdown wrapping, etc.),
        classifier must fall back to 'single' — never raise.
        """
        from app.agents.router import RouterAgent
        agent = RouterAgent()
        agent.llm = MagicMock()
        agent.llm.invoke.return_value = MagicMock(content="Sorry, I cannot classify this.")
        result = agent._classify("Some question")
        assert result.type == "single"
        assert result.confidence == 0.5

    def test_classify_falls_back_to_single_on_llm_exception(self):
        """If LLM raises (network error, timeout), must return safe fallback."""
        from app.agents.router import RouterAgent
        agent = RouterAgent()
        agent.llm = MagicMock()
        agent.llm.invoke.side_effect = ConnectionError("API unreachable")
        result = agent._classify("Some question")
        assert result.type == "single"
        assert result.confidence == 0.5

    def test_classify_falls_back_on_invalid_type_in_json(self):
        """If LLM returns valid JSON but invalid type, Pydantic catches it."""
        from app.agents.router import RouterAgent
        agent = RouterAgent()
        agent.llm = MagicMock()
        agent.llm.invoke.return_value = MagicMock(
            content=json.dumps({
                "type": "complex_hybrid",  # not a valid QueryType
                "confidence": 0.9,
                "reason": "Hallucinated type."
            })
        )
        result = agent._classify("Some question")
        assert result.type == "single"

    def test_classify_calls_llm_once_per_question(self):
        """LLM must be called exactly once per _classify() invocation."""
        from app.agents.router import RouterAgent
        agent = RouterAgent()
        agent.llm = MagicMock()
        agent.llm.invoke.return_value = make_llm_response()
        agent._classify("Test question")
        assert agent.llm.invoke.call_count == 1

    def test_classify_passes_question_in_human_message(self):
        """The user's question must appear in the HumanMessage sent to LLM."""
        from app.agents.router import RouterAgent
        from langchain_core.messages import HumanMessage
        agent = RouterAgent()
        agent.llm = MagicMock()
        agent.llm.invoke.return_value = make_llm_response()

        test_question = "What is the impact of climate change on flood frequency?"
        agent._classify(test_question)

        call_args = agent.llm.invoke.call_args[0][0]
        human_messages = [m for m in call_args if isinstance(m, HumanMessage)]
        assert any(test_question in m.content for m in human_messages)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# get_route() — Conditional Edge Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGetRoute:
    """
    Tests for RouterAgent.get_route() — the LangGraph conditional edge.

    This function is called by LangGraph AFTER the router node.
    It reads query_type from state and returns the next node's name.
    The return value MUST match a key in the edge map in orchestrator.py.
    """

    def setup_method(self):
        from app.agents.router import RouterAgent
        self.agent = RouterAgent()

    def test_direct_routes_to_direct(self):
        result = self.agent.get_route({"query_type": "direct"})
        assert result == "direct"

    def test_single_routes_to_single(self):
        result = self.agent.get_route({"query_type": "single"})
        assert result == "single"

    def test_multi_hop_routes_to_multi_hop(self):
        result = self.agent.get_route({"query_type": "multi_hop"})
        assert result == "multi_hop"

    def test_graph_routes_to_graph(self):
        result = self.agent.get_route({"query_type": "graph"})
        assert result == "graph"

    def test_missing_query_type_falls_back_to_single(self):
        """If state has no query_type (shouldn't happen but defensive), use single."""
        result = self.agent.get_route({})
        assert result == "single"

    def test_invalid_query_type_falls_back_to_single(self):
        """Unknown type value must not crash — fall back to single."""
        result = self.agent.get_route({"query_type": "unknown_type"})
        assert result == "single"

    def test_all_valid_types_return_string(self):
        """get_route() must always return a string, never raise."""
        for qt in ["direct", "single", "multi_hop", "graph"]:
            result = self.agent.get_route({"query_type": qt})
            assert isinstance(result, str)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _fetch_memories() — Mem0 Integration Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFetchMemories:

    def test_returns_memories_from_mem0(self):
        from app.agents.router import RouterAgent
        agent = RouterAgent()
        agent._mem0_client = MagicMock()
        agent._mem0_client.search.return_value = [
            {"memory": "User works on flood prediction models.", "score": 0.91},
            {"memory": "User prefers concise answers.", "score": 0.82},
        ]
        memories = agent._fetch_memories("What is the best model?", "user_123")
        assert len(memories) == 2
        assert memories[0]["memory"] == "User works on flood prediction models."
        assert memories[0]["score"] == 0.91

    def test_returns_empty_list_for_anonymous_user(self):
        """
        Anonymous users have no stored memories.
        Must return [] without calling Mem0 (avoids unnecessary API call).
        """
        from app.agents.router import RouterAgent
        agent = RouterAgent()
        agent._mem0_client = MagicMock()
        memories = agent._fetch_memories("What is X?", user_id="anonymous")
        assert memories == []
        agent._mem0_client.search.assert_not_called()

    def test_returns_empty_list_for_empty_user_id(self):
        from app.agents.router import RouterAgent
        agent = RouterAgent()
        agent._mem0_client = MagicMock()
        memories = agent._fetch_memories("What is X?", user_id="")
        assert memories == []
        agent._mem0_client.search.assert_not_called()

    def test_returns_empty_list_when_mem0_is_none(self):
        """If Mem0 client failed to initialize, must return [] gracefully."""
        from app.agents.router import RouterAgent
        agent = RouterAgent()
        agent._mem0_client = None
        memories = agent._fetch_memories("Some question", "user_123")
        assert memories == []

    def test_returns_empty_list_on_mem0_exception(self):
        """Mem0 network errors must not propagate — return [] silently."""
        from app.agents.router import RouterAgent
        agent = RouterAgent()
        agent._mem0_client = MagicMock()
        agent._mem0_client.search.side_effect = ConnectionError("Mem0 unreachable")
        memories = agent._fetch_memories("Some question", "user_123")
        assert memories == []

    def test_memories_have_required_keys(self):
        """Each memory dict must have 'memory' and 'score' keys."""
        from app.agents.router import RouterAgent
        agent = RouterAgent()
        agent._mem0_client = MagicMock()
        agent._mem0_client.search.return_value = [
            {"memory": "User is an ML engineer.", "score": 0.9, "extra_key": "ignored"},
        ]
        memories = agent._fetch_memories("question", "user_1")
        for m in memories:
            assert "memory" in m
            assert "score" in m

    def test_empty_memory_strings_are_filtered_out(self):
        """Mem0 sometimes returns records with empty memory strings. Filter them."""
        from app.agents.router import RouterAgent
        agent = RouterAgent()
        agent._mem0_client = MagicMock()
        agent._mem0_client.search.return_value = [
            {"memory": "Valid memory.", "score": 0.9},
            {"memory": "",             "score": 0.5},  # empty — should be filtered
            {"memory": "Another one.", "score": 0.7},
        ]
        memories = agent._fetch_memories("question", "user_1")
        assert len(memories) == 2
        assert all(m["memory"] for m in memories)

    def test_mem0_search_called_with_user_id(self):
        """Mem0 must be searched with the correct user_id for isolation."""
        from app.agents.router import RouterAgent
        agent = RouterAgent()
        agent._mem0_client = MagicMock()
        agent._mem0_client.search.return_value = []
        agent._fetch_memories("What is X?", "user_abc")
        agent._mem0_client.search.assert_called_once()
        call_kwargs = agent._mem0_client.search.call_args.kwargs
        assert call_kwargs.get("user_id") == "user_abc"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Empty / Edge Case Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEdgeCases:

    def test_route_with_empty_question_returns_single_fallback(self):
        """Empty question must fall back gracefully without calling LLM."""
        agent = make_router()
        state = make_state(question="")
        result = agent.route(state)
        assert result["query_type"] == "single"
        agent.llm.invoke.assert_not_called()

    def test_route_with_missing_latency_ms_initializes_dict(self):
        """
        If latency_ms is absent from state, route() must create it.
        This happens on the very first node execution.
        """
        agent = make_router()
        state = make_state()
        del state["latency_ms"]    # simulate missing key
        result = agent.route(state)
        assert isinstance(result["latency_ms"], dict)
        assert "router" in result["latency_ms"]

    def test_route_with_no_user_id_skips_mem0(self):
        """If state has no user_id, Mem0 must not be called."""
        agent = make_router()
        state = make_state(user_id="")
        result = agent.route(state)
        agent._mem0_client.search.assert_not_called()
        assert result["long_term_memories"] == []

    def test_singleton_is_router_agent_instance(self):
        """Module-level router_agent must be a RouterAgent instance."""
        from app.agents.router import router_agent, RouterAgent
        assert isinstance(router_agent, RouterAgent)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Downstream Compatibility Test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDownstreamCompatibility:
    """
    Verifies that route() output is compatible with what downstream agents expect.

    In the LangGraph pipeline, route() output is merged into state before the
    retrieval agents (Phase 5) read from it. These tests ensure the contract
    between Phase 4 and Phase 5 is maintained.
    """

    def test_query_type_is_valid_literal(self):
        """
        query_type must be one of the four valid Literal values.
        Phase 5 agents use query_type to decide retrieval strategy.
        An invalid value would cause Phase 5 to fall back silently.
        """
        from app.core.state import QueryType
        valid = {"direct", "single", "multi_hop", "graph"}

        for qt in valid:
            agent = make_router(make_llm_response(qt))
            result = agent.route(make_state(f"Test question for {qt}"))
            assert result["query_type"] in valid

    def test_memories_format_matches_generator_expectation(self):
        """
        long_term_memories must be list[dict] with 'memory' and 'score' keys.
        The generator (Phase 7) iterates over memories and accesses m['memory'].
        """
        agent = make_router(mem0_memories=[
            {"memory": "User is an ML engineer.", "score": 0.92},
        ])
        result = agent.route(make_state(user_id="user_real"))
        memories = result["long_term_memories"]
        assert isinstance(memories, list)
        if memories:
            assert "memory" in memories[0]
            assert "score" in memories[0]

    def test_latency_ms_is_additive(self):
        """
        latency_ms update must be additive, not destructive.
        This test simulates multiple nodes having already written latency data.
        Phase 5 must also see 'router' in the accumulated latency dict.
        """
        existing_latency = {"some_init_node": 5.2}
        agent = make_router()
        state = make_state(latency_ms=existing_latency)
        result = agent.route(state)

        # Both pre-existing and new key must be present
        assert result["latency_ms"]["some_init_node"] == 5.2
        assert "router" in result["latency_ms"]
