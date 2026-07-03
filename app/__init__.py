"""Top-level package for the Autonomous Research Analysis Platform.

This package exposes the main application layers in a single, convenient
namespace for imports such as ``app.agents`` or ``app.core``.
"""

from __future__ import annotations

from app.agents import AnswerGenerator, KnowledgeGraphAgent, RetrievalAgent, RouterAgent
from app.core import ARAPOrchestrator, AgentState, QueryType, Settings, settings

__all__ = [
    "AnswerGenerator",
    "AgentState",
    "ARAPOrchestrator",
    "KnowledgeGraphAgent",
    "QueryType",
    "RetrievalAgent",
    "RouterAgent",
    "Settings",
    "settings",
]
