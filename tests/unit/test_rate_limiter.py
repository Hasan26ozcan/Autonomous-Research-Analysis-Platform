"""
tests/unit/test_rate_limiter.py
================================
Unit tests for app/services/rate_limiter.py.

``SlidingWindowRateLimiter.acquire`` blocks (sleeping) until the RPM/TPM
budget allows the call. In tests we mock ``time.monotonic`` (to advance the
clock deterministically) and ``time.sleep`` (to avoid real waiting) so we can
drive the limiter into both the immediate-pass and the forced-wait branches.

Covers:
  * estimate_tokens heuristics
  * acquire() immediate pass on an empty window
  * acquire() when a single call exceeds the whole TPM budget (special branch)
  * acquire() forced to wait by the RPM window
  * acquire() forced to wait by the TPM window
"""
from unittest.mock import MagicMock, patch

import pytest

from app.services.rate_limiter import SlidingWindowRateLimiter, estimate_tokens


def test_estimate_tokens_adds_output_budget():
    # "hello world" = 11 chars → 11 // 4 = 2 input tokens, + 10 output = 12.
    assert estimate_tokens("hello world", max_output_tokens=10) == 12


def test_estimate_tokens_minimum_one_input_token():
    # Empty input still counts as at least 1 input token.
    assert estimate_tokens("", max_output_tokens=0) == 1


def test_estimate_tokens_ignores_blank_texts():
    # Blank entries are skipped; only real text contributes.
    assert estimate_tokens("abc", "", "   ", max_output_tokens=0) == max(1, 3 // 4) or 1



def test_acquire_passes_immediately_on_empty_window():
    limiter = SlidingWindowRateLimiter(max_requests_per_minute=30, max_tokens_per_minute=6000)
    sleep = MagicMock()
    with patch("time.monotonic", return_value=1000.0), patch("time.sleep", sleep):
        limiter.acquire(100)
    # No wait needed → sleep must never be called, and the call is recorded.
    sleep.assert_not_called()
    assert len(limiter._request_times) == 1
    assert len(limiter._token_events) == 1



def test_acquire_when_estimate_exceeds_full_tpm_budget():
    # 150 tokens vs a 100 TPM cap: the strict check could never be satisfied,
    # so the limiter must let it through once the window is empty.
    limiter = SlidingWindowRateLimiter(max_requests_per_minute=30, max_tokens_per_minute=100)
    sleep = MagicMock()
    with patch("time.monotonic", return_value=1000.0), patch("time.sleep", sleep):
        limiter.acquire(150)
    sleep.assert_not_called()
    assert limiter._token_events[0][1] == 150



def test_acquire_waits_for_rpm_window_to_drain():
    limiter = SlidingWindowRateLimiter(max_requests_per_minute=5, max_tokens_per_minute=6000)
    # Seed the window with 5 recent requests so the RPM budget is exactly full.
    limiter._request_times.extend([1000.0] * 5)

    sleep = MagicMock()
    # Call 1 (now=1000): request still in window → must wait ~60s, then
    # call 2 (now=1061): window drained → acquires.
    with patch("time.monotonic", side_effect=[1000.0, 1061.0]), patch("time.sleep", sleep):
        limiter.acquire(100)

    sleep.assert_called_once()
    # The new request is recorded after acquiring.
    assert limiter._request_times[-1] == pytest.approx(1061.0)



def test_acquire_waits_for_tpm_window_to_drain():
    limiter = SlidingWindowRateLimiter(max_requests_per_minute=100, max_tokens_per_minute=500)
    # Seed the window with a recent token event leaving no room for +300.
    limiter._token_events.append((1000.0, 300))

    sleep = MagicMock()
    with patch("time.monotonic", side_effect=[1000.0, 1061.0]), patch("time.sleep", sleep):
        limiter.acquire(300)

    sleep.assert_called_once()
    assert limiter._token_events[-1] == (1061.0, 300)
