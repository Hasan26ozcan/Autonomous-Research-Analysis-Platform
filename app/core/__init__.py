"""Core package exports for the autonomous research analysis platform.

This package exposes the central configuration, shared state schema, and the
orchestrator used to compose the LangGraph workflow.
"""

from __future__ import annotations

from app.core.config import Settings, settings
from app.core.orchestrator import ARAPOrchestrator, orchestrator
from app.core.state import AgentState, QueryType

__all__ = [
    "AgentState",
    "ARAPOrchestrator",
    "QueryType",
    "Settings",
    "orchestrator",
    "settings",
]
