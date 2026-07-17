"""
Shared pytest configuration for the ARAP test suite.

Disables the real Groq rate limiter across every test. The limiter paces
real LLM calls via time.sleep() to stay under RPM/TPM caps; in tests all LLM
responses are mocked, so the pacing is pure dead time (and, accumulated across
many calls, can exceed the test timeout and hang the run). Mocking `acquire`
to a no-op keeps LLM-calling code paths fast and deterministic without
touching the production limiter.
"""

import os

# Make Celery use in-memory broker/result backends for tests so that
# ingest endpoints (.delay()) and task-status lookups (AsyncResult) never
# touch a real Redis server — otherwise the suite hangs on connection retries.
os.environ["CELERY_BROKER_URL"] = "memory://"
os.environ["CELERY_RESULT_BACKEND"] = "rpc://"

import pytest


@pytest.fixture(autouse=True)
def _disable_rate_limiter(monkeypatch):
    from app.services.rate_limiter import groq_rate_limiter

    monkeypatch.setattr(groq_rate_limiter, "acquire", lambda *args, **kwargs: None)
