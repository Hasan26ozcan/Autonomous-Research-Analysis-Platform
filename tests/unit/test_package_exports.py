"""
tests/unit/test_package_exports.py
===================================
Coverage for the lazy-loading package ``__init__.py`` files:

  * app/agents/__init__.py
  * app/core/__init__.py

Both packages resolve their public names on first access via ``__getattr__``
and advertise their exports via ``__dir__``. The two branches that are easy
to leave uncovered are:

  1. ``__getattr__`` raising ``AttributeError`` for a name that is NOT
     part of the package's export table.
  2. ``__dir__`` being invoked (e.g. via the builtin ``dir()``).
"""

import pytest


def test_agents_getattr_unknown_name_raises():
    """Accessing a non-exported name must raise AttributeError (line 50)."""
    import app.agents

    with pytest.raises(AttributeError):
        _ = app.agents.this_name_does_not_exist


def test_agents_dir_includes_exports():
    """dir() on the package must surface the advertised exports (line 60)."""
    import app.agents

    names = dir(app.agents)
    assert "AnswerGenerator" in names
    assert "generator" in names
    assert "kg_agent" in names


def test_core_getattr_unknown_name_raises():
    """Accessing a non-exported name must raise AttributeError (line 50)."""
    import app.core

    with pytest.raises(AttributeError):
        _ = app.core.this_name_does_not_exist


def test_core_dir_includes_exports():
    """dir() on the package must surface the advertised exports (line 60)."""
    import app.core

    names = dir(app.core)
    assert "settings" in names
    assert "ARAPOrchestrator" in names
