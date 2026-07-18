"""Core package exports for the autonomous research analysis platform.

This package exposes the central configuration, shared state schema, and the
orchestrator used to compose the LangGraph workflow.

Imports are resolved lazily (mirroring app.agents) so that importing a single
submodule such as ``app.core.config`` does not eagerly trigger the import of
``app.core.orchestrator``, which in turn imports back from ``app.agents``.
That eager chain is what causes:

    ImportError: cannot import name 'generator' from partially initialized
    module 'app.agents.generator' (most likely due to a circular import)

when app.agents.generator (or any other agent module) does
``from app.core.config import settings`` while it is still mid-import.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentState": ("app.core.state", "AgentState"),
    "ARAPOrchestrator": ("app.core.orchestrator", "ARAPOrchestrator"),
    "QueryType": ("app.core.state", "QueryType"),
    "Settings": ("app.core.config", "Settings"),
    "orchestrator": ("app.core.orchestrator", "orchestrator"),
    "settings": ("app.core.config", "settings"),
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
