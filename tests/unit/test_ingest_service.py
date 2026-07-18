"""
tests/unit/test_ingest_service.py
=================================
Unit tests for app/services/ingest_service.py (the synchronous Celery ingest
pipeline).

``run_ingest_pipeline`` orchestrates many sibling services (chunker, enricher,
embedder, vector store, BM25, KG agent, Postgres, Redis state, worker logs).
Every one of those is monkey-patched so the test exercises the *orchestration*
logic — the control flow, the doc_id extraction, the success/error contracts,
and the KG-triple counting — without touching any real infrastructure.

Three scenarios are covered:
  * happy path → success dict with chunk + triple counts
  * empty extraction → structured error dict
  * exception mid-pipeline → error dict + status/state/log side effects
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def pipeline(monkeypatch):
    from app.services import ingest_service as svc

    chunks = [
        {
            "text": "original one",
            "doc_id": "docX",
            "page": 1,
            "chunk_index": 0,
            "filename": "f.pdf",
        },
        {
            "text": "original two",
            "doc_id": "docX",
            "page": 2,
            "chunk_index": 1,
            "filename": "f.pdf",
        },
    ]

    monkeypatch.setattr(svc, "chunk_pdf", lambda p: chunks)
    monkeypatch.setattr(svc, "pdf_page_count", lambda p: 5)
    # Enrichment appends a marker so text != original (success_count > 0).
    monkeypatch.setattr(svc, "enrich_chunk", lambda text, *a, **k: text + " [ctx]")

    embedder = MagicMock()
    embedder.embed.return_value = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    monkeypatch.setattr(svc, "embedder", embedder)

    vector_store = MagicMock()
    vector_store.upsert.return_value = 2
    monkeypatch.setattr(svc, "vector_store", vector_store)

    bm25 = MagicMock()
    monkeypatch.setattr(svc, "bm25_index", bm25)

    kg = MagicMock()
    kg.extract_and_store_node.return_value = {"kg_entities": [1, 2]}
    monkeypatch.setattr(svc, "kg_agent", kg)

    record_document = MagicMock()
    monkeypatch.setattr(svc, "record_document", record_document)
    record_chunk_metadata = MagicMock()
    monkeypatch.setattr(svc, "record_chunk_metadata", record_chunk_metadata)
    update_document_status = MagicMock()
    monkeypatch.setattr(svc, "update_document_status", update_document_status)
    pipeline_state_set = MagicMock()
    monkeypatch.setattr(svc, "pipeline_state_set", pipeline_state_set)
    pipeline_state_clear = MagicMock()
    monkeypatch.setattr(svc, "pipeline_state_clear", pipeline_state_clear)
    log_worker = MagicMock()
    monkeypatch.setattr(svc, "log_worker", log_worker)
    notify = MagicMock()
    monkeypatch.setattr(svc, "_notify_bm25_reload", notify)

    return {
        "svc": svc, "chunks": chunks, "embedder": embedder,
        "vector_store": vector_store, "bm25": bm25, "kg": kg,
        "record_document": record_document, "record_chunk_metadata": record_chunk_metadata,
        "update_document_status": update_document_status, "pipeline_state_set": pipeline_state_set,
        "pipeline_state_clear": pipeline_state_clear, "log_worker": log_worker, "notify": notify,
    }


def test_run_ingest_pipeline_success(pipeline):
    result = pipeline["svc"].run_ingest_pipeline(b"fake pdf", "f.pdf")
    assert result["status"] == "success"
    assert result["doc_id"] == "docX"
    assert result["chunks"] == 2
    assert result["kg_triples"] == 2

    # Key side effects must have fired.
    pipeline["kg"].extract_and_store_node.assert_called_once()
    pipeline["vector_store"].upsert.assert_called_once()
    pipeline["embedder"].embed.assert_called_once()
    pipeline["bm25"].remove_by_doc.assert_called_once_with("docX")
    pipeline["bm25"].add_chunks.assert_called_once()
    pipeline["notify"].assert_called_once()

    # Postgres metadata saved with the right counts.
    pipeline["record_document"].assert_called_once()
    args, kwargs = pipeline["record_document"].call_args
    assert kwargs["status"] == "ready"
    assert kwargs["total_pages"] == 5
    # call signature: (doc_id, filename, chunk_count, kg_triples, ...)
    assert args[2] == 2
    assert args[3] == 2


def test_run_ingest_pipeline_empty_extraction(pipeline, monkeypatch):
    pipeline["svc"].chunk_pdf = lambda p: []  # no chunks extracted
    result = pipeline["svc"].run_ingest_pipeline(b"blank", "blank.pdf")
    assert result["status"] == "error"
    assert result["doc_id"] == "unknown"
    assert result["chunks"] == 0
    assert result["kg_triples"] == 0
    assert "No text extracted" in result["error"]
    # No downstream work should have happened.
    pipeline["kg"].extract_and_store_node.assert_not_called()
    pipeline["vector_store"].upsert.assert_not_called()


def test_run_ingest_pipeline_handles_enrichment_error(pipeline):
    def _boom(*a, **k):
        raise RuntimeError("enrich failed")

    pipeline["svc"].enrich_chunk = _boom

    result = pipeline["svc"].run_ingest_pipeline(b"fake pdf", "f.pdf")
    assert result["status"] == "error"
    assert result["doc_id"] == "docX"
    assert result["kg_triples"] == 0

    # The document lifecycle must be flipped to error and the in-flight flag
    # cleared, and a worker error log written.
    assert any(
        call.args[1] == "error"
        for call in pipeline["update_document_status"].call_args_list
    )
    pipeline["pipeline_state_clear"].assert_called_once_with("docX")
    # Worker error log carries the exception text.
    error_log_calls = [
        c for c in pipeline["log_worker"].call_args_list
        if c.args[2] == "error"
    ]
    assert error_log_calls, "expected an error-level worker log"
    assert "enrich failed" in error_log_calls[0].args[3]


def test_notify_bm25_reload_publishes(monkeypatch):
    """_notify_bm25_reload must publish a reload signal on the Redis channel."""
    from app.services import ingest_service as svc

    fake_redis = MagicMock()
    with patch("redis.from_url", return_value=fake_redis) as mock_from_url:
        svc._notify_bm25_reload()

    mock_from_url.assert_called_once_with(svc.settings.redis_url)
    fake_redis.publish.assert_called_once_with("arap:bm25:reload", "1")
