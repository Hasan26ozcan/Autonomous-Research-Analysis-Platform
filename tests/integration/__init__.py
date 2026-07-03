"""Integration test package for ARAP.

This package marks the integration-test directory as a proper Python package
and provides a small, explicit namespace for broader test discovery.

Integration tests are intended for end-to-end workflows that exercise the
application with its surrounding services, such as FastAPI, Redis, Qdrant,
and Neo4j, where applicable.
"""

from __future__ import annotations

__all__ = [
    "__doc__",
]
