"""Agent package exports and lazy-loading interface.

This package exposes the core agent classes and singleton instances used
throughout the application. Imports are resolved lazily so that importing
app.agents does not eagerly load every agent module and its heavy
dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.agents.generator import AnswerGenerator
    from app.agents.graph_agent import (
        EntityExtractionResult,
        KnowledgeGraphAgent,
        Triple,
        TripleExtractionResult,
        kg_agent,
    )
    from app.agents.retrieval_agent import RetrievalAgent, retrieval_agent
    from app.agents.router import RouterAgent, RouterOutput, router_agent

# Module paths for the lazily-loaded agent submodules. Centralised here so the
# dotted paths are not duplicated across the export table below.
_GENERATOR_MODULE = "app.agents.generator"
_GRAPH_AGENT_MODULE = "app.agents.graph_agent"
_RETRIEVAL_AGENT_MODULE = "app.agents.retrieval_agent"
_ROUTER_MODULE = "app.agents.router"

__all__ = [
    "AnswerGenerator",
    "EntityExtractionResult",
    "KnowledgeGraphAgent",
    "RetrievalAgent",
    "RouterAgent",
    "RouterOutput",
    "Triple",
    "TripleExtractionResult",
    "generator",
    "kg_agent",
    "retrieval_agent",
    "router_agent",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "AnswerGenerator": (_GENERATOR_MODULE, "AnswerGenerator"),
    "EntityExtractionResult": (_GRAPH_AGENT_MODULE, "EntityExtractionResult"),
    "KnowledgeGraphAgent": (_GRAPH_AGENT_MODULE, "KnowledgeGraphAgent"),
    "Triple": (_GRAPH_AGENT_MODULE, "Triple"),
    "TripleExtractionResult": (_GRAPH_AGENT_MODULE, "TripleExtractionResult"),
    "RetrievalAgent": (_RETRIEVAL_AGENT_MODULE, "RetrievalAgent"),
    "RouterAgent": (_ROUTER_MODULE, "RouterAgent"),
    "RouterOutput": (_ROUTER_MODULE, "RouterOutput"),
    "generator": (_GENERATOR_MODULE, "generator"),
    "kg_agent": (_GRAPH_AGENT_MODULE, "kg_agent"),
    "retrieval_agent": (_RETRIEVAL_AGENT_MODULE, "retrieval_agent"),
    "router_agent": (_ROUTER_MODULE, "router_agent"),
}


def __getattr__(name: str) -> Any:
    """Resolve package exports on first use."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
