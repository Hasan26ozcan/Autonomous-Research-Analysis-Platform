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
    "AnswerGenerator": ("app.agents.generator", "AnswerGenerator"),
    "EntityExtractionResult": ("app.agents.graph_agent", "EntityExtractionResult"),
    "KnowledgeGraphAgent": ("app.agents.graph_agent", "KnowledgeGraphAgent"),
    "Triple": ("app.agents.graph_agent", "Triple"),
    "TripleExtractionResult": ("app.agents.graph_agent", "TripleExtractionResult"),
    "RetrievalAgent": ("app.agents.retrieval_agent", "RetrievalAgent"),
    "RouterAgent": ("app.agents.router", "RouterAgent"),
    "RouterOutput": ("app.agents.router", "RouterOutput"),
    "generator": ("app.agents.generator", "generator"),
    "kg_agent": ("app.agents.graph_agent", "kg_agent"),
    "retrieval_agent": ("app.agents.retrieval_agent", "retrieval_agent"),
    "router_agent": ("app.agents.router", "router_agent"),
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
