"""
Proactive rate limiter for Groq (or any OpenAI-compatible) LLM calls.

WHY THIS EXISTS
----------------
Groq's free tier enforces both a requests-per-minute (RPM) and a
tokens-per-minute (TPM) cap per model. The reactive approach - fire requests
as fast as possible and let tenacity/openai's client retry on 429 - works,
but it means every burst of chunk-level LLM calls (contextual enrichment, KG
extraction, etc.) slams into the per-minute wall repeatedly before finally
squeezing through, which is slow AND wastes part of the daily token budget
on rejected attempts.

This module takes the opposite approach: before every LLM call, ask the
shared limiter "is it safe to send ~N tokens right now?". If sending would
exceed the configured RPM or TPM budget, the limiter sleeps just long enough
for the oldest request/token usage in the trailing 60-second window to age
out, then lets the call through. Done correctly, this means the app almost
never sees a 429 from the per-minute limits at all - it simply paces itself
to stay under them.

This does NOT help with Groq's separate tokens-per-day (TPD) limit - that is
a hard daily budget with no workaround except waiting for it to roll off
(see settings.llm_base_url / README for details). This limiter only handles
the per-minute axis.

USAGE
-----
    from app.services.rate_limiter import groq_rate_limiter

    groq_rate_limiter.acquire(estimated_tokens=800)
    response = some_chat_openai_client.invoke([...])
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

from app.core.config import settings

logger = logging.getLogger(__name__)


class SlidingWindowRateLimiter:
    """
    Thread-safe limiter tracking both request count and token usage over a
    trailing 60-second window, sleeping as needed to stay under both caps.
    """

    def __init__(self, max_requests_per_minute: int, max_tokens_per_minute: int):
        self.max_rpm = max(1, max_requests_per_minute)
        self.max_tpm = max(1, max_tokens_per_minute)
        self._lock = threading.Lock()
        self._request_times: deque[float] = deque()
        self._token_events: deque[tuple[float, int]] = deque()

    def _prune(self, now: float) -> None:
        while self._request_times and now - self._request_times[0] > 60:
            self._request_times.popleft()
        while self._token_events and now - self._token_events[0][0] > 60:
            self._token_events.popleft()

    def acquire(self, estimated_tokens: int) -> None:
        """
        Block (sleeping if necessary) until a call using ~estimated_tokens
        tokens can be made without exceeding the configured RPM/TPM budget.
        """
        estimated_tokens = max(1, estimated_tokens)

        while True:
            with self._lock:
                now = time.monotonic()
                self._prune(now)

                requests_in_window = len(self._request_times)
                tokens_in_window = sum(t for _, t in self._token_events)

                rpm_ok = requests_in_window < self.max_rpm

                if estimated_tokens >= self.max_tpm:
                    # This single call's own token estimate already meets or
                    # exceeds the whole per-minute budget - the strict check
                    # below (tokens_in_window + estimated_tokens <= max_tpm)
                    # could NEVER be true in this case, even against a
                    # completely empty window, causing acquire() to sleep
                    # forever (a real hang observed with generate(): 5
                    # retrieved chunks + max_output_tokens routinely adds up
                    # to more than the 5000 TPM default). Best effort
                    # instead: let it through once the window has fully
                    # drained of other traffic, rather than blocking forever
                    # on an unsatisfiable condition.
                    tpm_ok = tokens_in_window == 0
                else:
                    tpm_ok = tokens_in_window + estimated_tokens <= self.max_tpm

                if rpm_ok and tpm_ok:
                    self._request_times.append(now)
                    self._token_events.append((now, estimated_tokens))
                    return

                # Not safe yet - figure out the minimum wait to free up
                # enough room, release the lock, then sleep and retry.
                wait_candidates = []
                if not rpm_ok and self._request_times:
                    wait_candidates.append(60 - (now - self._request_times[0]))
                if not tpm_ok and self._token_events:
                    wait_candidates.append(60 - (now - self._token_events[0][0]))
                wait_s = max(wait_candidates) if wait_candidates else 1.0
                wait_s = max(wait_s, 0.1) + 0.05   # small safety margin

            logger.debug(
                "Rate limiter: pacing for %.2fs to stay under %d RPM / %d TPM",
                wait_s, self.max_rpm, self.max_tpm,
            )
            time.sleep(wait_s)


def estimate_tokens(*texts: str, max_output_tokens: int = 0) -> int:
    """
    Rough token estimate for pacing purposes only (does not need to be exact
    - the limiter just needs a conservative-enough number to avoid
    overshooting the real per-minute cap). Uses the common ~4 chars/token
    heuristic for English/mixed text, plus the expected output budget.
    """
    input_chars = sum(len(t) for t in texts if t)
    input_tokens = max(1, input_chars // 4)
    return input_tokens + max_output_tokens


# Shared singleton used by every Groq/OpenAI-compatible call site in the app.
# Conservative defaults matching Groq's published free-tier per-minute caps
# (most free-tier models: 30 RPM / 6,000-8,000 TPM) - deliberately a bit
# under the documented ceiling to leave margin for clock drift and the fact
# that several agents share this one budget.
groq_rate_limiter = SlidingWindowRateLimiter(
    max_requests_per_minute=settings.llm_rpm_limit,
    max_tokens_per_minute=settings.llm_tpm_limit,
)
