"""
tests/pg_helpers.py
====================
Reusable in-memory fakes for the PostgreSQL persistence layer.

The persistence modules (postgres_store, eval_store, log_store, analytics)
all obtain a connection through ``get_pool()`` and run synchronous
``psycopg2`` calls against it. None of them should ever be exercised against
a real database in unit tests, so we replace the pool with a fake that records
the SQL it was asked to run and returns scripted rows.

Design notes
------------
* ``FakeRow`` behaves like a ``RealDictCursor`` row: it is a dict keyed by
  column name (so ``row["doc_id"]`` works) but ALSO supports integer indexing
  (so ``row[0]`` works the same as a positional record tuple). This lets a
  single fake satisfy both ``_fetchval`` (which does ``row[0]``) and the
  analytics readers (which do ``r["id"]`` etc.).
* ``FakePGPool`` owns a single scripted result list. Every cursor created from
  any connection of the pool advances a shared pointer through that list, so a
  sequence of independent ``conn.cursor()`` calls (as ``analytics.summary``
  makes) returns the scripted values in order.
* Setting ``raise_on`` makes every execute/fetch raise, which exercises the
  best-effort ``except`` branches in the production code.
"""

from __future__ import annotations

from typing import Any


class FakeRow(dict):
    """dict that additionally supports integer indexing like a record tuple."""

    def __getitem__(self, key: Any) -> Any:  # type: ignore[override]
        if isinstance(key, int):
            values = list(self.values())
            return values[key] if values else None
        return super().__getitem__(key)


class FakePGCursor:
    """Records executed SQL and returns scripted rows from the owning pool."""

    def __init__(self, pool: FakePGPool):
        self._pool = pool
        self.executed: list[tuple] = []

    def __enter__(self) -> FakePGCursor:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def execute(self, sql: str, params: Any = ()) -> None:
        self.executed.append((sql, params))
        self._pool.all_executed.append((sql, params))
        if self._pool.raise_on is not None:
            raise self._pool.raise_on

    def executemany(self, sql: str, params: Any = None) -> None:
        # Batch variant: the whole parameter set is recorded as a single
        # scripted call. Delegate to execute() so the recording/raise-on
        # behavior stays identical to a single-statement call.
        self.execute(sql, params)

    def fetchone(self) -> Any:
        if self._pool.raise_on is not None:
            raise self._pool.raise_on
        script = self._pool.script
        if self._pool.ptr < len(script):
            value = script[self._pool.ptr]
            self._pool.ptr += 1
            return value
        return None

    def fetchall(self) -> list:
        if self._pool.raise_on is not None:
            raise self._pool.raise_on
        script = self._pool.script
        remaining = script[self._pool.ptr :]
        self._pool.ptr += len(remaining)
        return remaining


class FakePGConn:
    """A fake psycopg2 connection backed by a FakePGPool."""

    def __init__(self, pool: FakePGPool):
        self._pool = pool
        self.committed = False

    def cursor(self, cursor_factory: Any = None) -> FakePGCursor:
        # cursor_factory is accepted for psycopg2 API compatibility (callers
        # such as analytics pass cursor_factory=RealDictCursor) but is ignored
        # by the fake, which always returns a FakePGCursor.
        self._cursor_factory = cursor_factory
        return FakePGCursor(self._pool)

    def commit(self) -> None:
        self.committed = True
        self._pool.committed = True

    def close(self) -> None:
        # intentionally empty: the fake connection owns no real socket to tear
        # down, so there is nothing to release on close().
        pass


class FakePGPool:
    """
    Drop-in replacement for ``psycopg2.pool.SimpleConnectionPool``.

    Args:
        script:    Ordered list of values returned by ``fetchone``/``fetchall``.
        raise_on:  If set, every execute/fetch raises this exception (to test
                   best-effort error handling).
    """

    def __init__(self, script: list | None = None, raise_on: BaseException | None = None):
        self.script: list = list(script or [])
        self.ptr = 0
        self.raise_on = raise_on
        self.getconn_calls = 0
        self.all_executed: list[tuple] = []
        self.committed = False

    def getconn(self) -> FakePGConn:
        self.getconn_calls += 1
        return FakePGConn(self)

    def putconn(self, conn: FakePGConn) -> None:  # noqa: D401 - pool API
        pass


def make_pool(script: list | None = None, raise_on: BaseException | None = None) -> FakePGPool:
    """Convenience constructor for a ``FakePGPool``."""
    return FakePGPool(script=script, raise_on=raise_on)
