"""API package exports for the autonomous research analysis platform.

This package exposes the FastAPI application entrypoint and the main request
schemas used by the HTTP and WebSocket interfaces.
"""

from __future__ import annotations

from app.api.main import (
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceItem,
    app,
)

__all__ = [
    "HealthResponse",
    "IngestResponse",
    "QueryRequest",
    "QueryResponse",
    "SourceItem",
    "app",
]
