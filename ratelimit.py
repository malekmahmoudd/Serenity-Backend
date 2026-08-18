"""
ratelimit.py
============
Token-bucket rate limiting, in process.

WHY THIS MATTERS MORE THAN USUAL HERE
-------------------------------------
Groq's free tier is a shared, exhaustible resource. One script hammering /chat
drains the daily token quota in minutes, and from then on EVERY user gets
hand-written template fallbacks instead of a real reply. The failure is not
"the attacker gets rate limited" -- it is "the product quietly degrades for
everyone else".

There is also a cost asymmetry worth naming: each /chat call runs an LSTM
forward pass and an LLM request. It is one of the more expensive endpoints you
could leave unguarded.

WHY A TOKEN BUCKET
------------------
Fixed windows let someone fire the whole allowance in the last second of one
window and again in the first second of the next -- double the intended rate at
the boundary. A bucket refills continuously, so bursts are bounded by capacity
and the sustained rate is exactly the refill rate.

Capacity above the refill rate is deliberate: a real person typing three quick
messages should not be throttled, while a loop still settles to the sustained
rate.

LIMITS ARE PER IDENTITY, NOT PER ENDPOINT
-----------------------------------------
Signed-in users are keyed by user id, anonymous ones by IP. That means a shared
office NAT shares an anonymous bucket -- imperfect, and the reason the
anonymous limit is not tight.

IN-PROCESS, WHICH HAS A REAL LIMIT
----------------------------------
Buckets live in memory, so with N uvicorn workers the effective limit is N
times what is configured, and a restart forgets everything. Fine for one
process. Move to Redis when you scale out.

THE CRISIS EXEMPTION
--------------------
`crisis_exempt` skips the limit when a message trips the safety rule. Someone
in distress sending messages quickly must never be told to slow down. This is
abusable in principle -- send crisis words to bypass the limiter -- and that is
an acceptable trade: the failure mode of the alternative is refusing to talk to
someone in crisis.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

# Sustained requests per minute, and how much burst is tolerated.
USER_RATE = float(os.environ.get("RATE_USER_PER_MIN", "20"))
USER_BURST = float(os.environ.get("RATE_USER_BURST", "8"))
ANON_RATE = float(os.environ.get("RATE_ANON_PER_MIN", "10"))
ANON_BURST = float(os.environ.get("RATE_ANON_BURST", "5"))
# Auth endpoints are guarded separately and much harder: this is where
# credential stuffing lands.
AUTH_RATE = float(os.environ.get("RATE_AUTH_PER_MIN", "5"))
AUTH_BURST = float(os.environ.get("RATE_AUTH_BURST", "5"))

MAX_BUCKETS = 20000
IDLE_EVICT_S = 900


@dataclass
class _Bucket:
    tokens: float
    last: float


class TokenBucket:
    def __init__(self, rate_per_min: float, burst: float, name: str):
        self.rate = rate_per_min / 60.0        # tokens per second
        self.burst = burst
        self.name = name
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def _evict(self, now: float) -> None:
        if len(self._buckets) <= MAX_BUCKETS:
            return
        for k in [k for k, b in self._buckets.items()
                  if now - b.last > IDLE_EVICT_S]:
            self._buckets.pop(k, None)

    def check(self, key: str) -> tuple[bool, float]:
        """
        Consume one token. Returns (allowed, retry_after_seconds).

        retry_after is how long until one token exists again, so the caller can
        send a truthful Retry-After header rather than a guess.
        """
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            b = self._buckets.get(key)
            if b is None:
                b = _Bucket(tokens=self.burst, last=now)
                self._buckets[key] = b
            else:
                b.tokens = min(self.burst, b.tokens + (now - b.last) * self.rate)
                b.last = now

            if b.tokens >= 1.0:
                b.tokens -= 1.0
                return True, 0.0
            return False, (1.0 - b.tokens) / self.rate if self.rate > 0 else 60.0

    def peek(self, key: str) -> float:
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                return self.burst
            return min(self.burst, b.tokens + (time.monotonic() - b.last) * self.rate)


chat_user = TokenBucket(USER_RATE, USER_BURST, "chat_user")
chat_anon = TokenBucket(ANON_RATE, ANON_BURST, "chat_anon")
auth_ip = TokenBucket(AUTH_RATE, AUTH_BURST, "auth")


def client_ip(request) -> str:
    """
    Best-effort client address.

    X-Forwarded-For is only trustworthy behind a proxy you control -- anyone can
    send the header directly. Set TRUST_PROXY=true only when something in front
    of this service overwrites it.
    """
    if os.environ.get("TRUST_PROXY", "false").lower() == "true":
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def stats() -> dict:
    return {
        "chat_user": {"per_min": USER_RATE, "burst": USER_BURST,
                      "tracked": len(chat_user._buckets)},
        "chat_anon": {"per_min": ANON_RATE, "burst": ANON_BURST,
                      "tracked": len(chat_anon._buckets)},
        "auth": {"per_min": AUTH_RATE, "burst": AUTH_BURST,
                 "tracked": len(auth_ip._buckets)},
        "scope": "in-process (multiply by worker count)",
    }
