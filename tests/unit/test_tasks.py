"""
tests/unit/test_tasks.py
========================
Unit tests for app/services/tasks.py.

``ping`` is a trivial health-check task. ``ingest_document_task`` is the Celery
worker entry point that delegates to ``run_ingest_pipeline``; we monkeypatch
that delegate so the task's own control flow (success return, retry-on-error)
is exercised without any real ingest or broker.
"""

import pytest


@pytest.fixture
def tasks():
    from app.services import tasks as module
    return module


def test_ping_returns_pong(tasks):
    assert tasks.ping() == "pong"
    assert tasks.ping.name == "app.services.tasks.ping"


def test_ingest_document_task_runs_pipeline(tasks, monkeypatch):
    fake_result = {"status": "success", "doc_id": "d1", "chunks": 3, "kg_triples": 1}
    monkeypatch.setattr(
        tasks, "run_ingest_pipeline",
        lambda file_content, filename, user_id="default": fake_result,
    )
    # .run() executes the task body directly (Celery binds `self` internally).
    result = tasks.ingest_document_task.run(b"pdf-bytes", "f.pdf", "u1")
    assert result == fake_result


def test_ingest_document_task_propagates_for_retry(tasks, monkeypatch):
    """On error the task must raise so Celery can retry it."""
    def _boom(file_content, filename, user_id="default"):
        raise RuntimeError("ingest exploded")

    monkeypatch.setattr(tasks, "run_ingest_pipeline", _boom)
    with pytest.raises(RuntimeError):
        tasks.ingest_document_task.run(b"pdf-bytes", "f.pdf")
