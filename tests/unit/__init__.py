"""Unit test package for ARAP.

This module keeps the unit-test package explicit and lightweight so that
pytest, IDEs, and local tooling can import the test suite consistently.

It intentionally does not import the full application stack at module import
time; instead, it exposes a small, stable namespace for shared test utilities
and acts as a clear marker for the unit-test package.
"""

from __future__ import annotations

__all__ = [
    "__doc__",
]
