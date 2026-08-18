"""
metrics.py
==========
Structured, queryable counters -- the numbers /health needs that print()
cannot give you.

WHY THIS EXISTS SEPARATELY FROM THE CONSOLE LOG
--------------------------------------------------
main.py already prints one line per turn:

    [a3f2b1c0] t3 RESOLVE tok=24 labels=['depression'] top3=[...] | text

That's fine for watching a live terminal. It is useless once this runs
unattended -- "is the crisis-fire rate normal today" is not a question you can
answer by reading scrollback. This file keeps the same events as running
counters you can query over HTTP, in one process, with no new infrastructure.

WHAT IT DELIBERATELY IS NOT
----------------------------
Not Prometheus, not a metrics pipeline, not anything requiring a sidecar. In
one process it's a dict with a lock. If you outgrow single-process (multiple
uvicorn workers, containers behind a load balancer), these counters become
per-process shards and /metrics only reflects whichever process answered the
request -- point a real metrics backend (Prometheus, Datadog) at that scale
instead of extending this file.

THE THREE NUMBERS THE ESCALATION POLICY ASKS FOR
---------------------------------------------------
ESCALATION_POLICY.md section 4 says crisis volume should get a manual look if
it triples week over week. You cannot act on that without a number to watch.
crisis_fire_rate, abstention_rate and llm_failure_rate are exactly the ones
named in the review process that document describes -- this file is what makes
that policy checkable rather than aspirational.

RETENTION
---------
In-memory, resets on restart. That's fine for "is today anomalous" (the actual
use case) and wrong for "what happened last month" (query db.Turn for that --
it has the same action/timestamp data, durably, per §db.py).
"""

from __future__ import annotations

import collections
import threading
import time

_lock = threading.Lock()
_start = time.time()

_action_counts: dict[str, int] = collections.Counter()
_llm_ok = 0
_llm_failed = 0
_llm_skipped = 0            # no key configured -- distinct from a failed call
_crisis_fires = 0
_turns_total = 0
_latency_ms: collections.deque = collections.deque(maxlen=500)

# Rolling per-hour buckets for the last 24h, so a spike is visible without a
# separate time-series database. Key = hour-since-epoch.
_hourly_turns: dict[int, int] = collections.OrderedDict()
_hourly_crisis: dict[int, int] = collections.OrderedDict()
_MAX_HOURS = 48


def _hour_bucket() -> int:
    return int(time.time() // 3600)


def _trim_hourly() -> None:
    cutoff = _hour_bucket() - _MAX_HOURS
    for d in (_hourly_turns, _hourly_crisis):
        for k in [k for k in d if k < cutoff]:
            del d[k]


def record_turn(action: str, llm_used: bool, llm_attempted: bool,
                latency_ms: float | None = None) -> None:
    """
    Called once per /chat request, after the decision and reply are known.

    llm_attempted vs llm_used distinguishes "we tried Groq and it failed"
    from "we never tried because no key is configured" -- the second is a
    deployment fact, the first is an incident.
    """
    global _llm_ok, _llm_failed, _llm_skipped, _crisis_fires, _turns_total
    with _lock:
        _turns_total += 1
        _action_counts[action] += 1
        if action == "SAFETY":
            _crisis_fires += 1
        if llm_attempted:
            if llm_used:
                _llm_ok += 1
            else:
                _llm_failed += 1
        else:
            _llm_skipped += 1
        if latency_ms is not None:
            _latency_ms.append(latency_ms)

        h = _hour_bucket()
        _hourly_turns[h] = _hourly_turns.get(h, 0) + 1
        if action == "SAFETY":
            _hourly_crisis[h] = _hourly_crisis.get(h, 0) + 1
        _trim_hourly()


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = min(int(len(s) * p), len(s) - 1)
    return round(s[idx], 1)


def snapshot() -> dict:
    """
    What /health and /metrics expose.

    week_over_week_crisis is the specific number ESCALATION_POLICY.md §4 asks
    someone to watch for a 3x jump -- computed here so nobody has to derive it
    from raw counts by hand.
    """
    with _lock:
        _trim_hourly()
        latencies = list(_latency_ms)

        this_week = sum(v for k, v in _hourly_crisis.items()
                        if k >= _hour_bucket() - 24 * 7)
        prior_week = sum(v for k, v in _hourly_crisis.items()
                         if _hour_bucket() - 24 * 14 <= k < _hour_bucket() - 24 * 7)
        ratio = (round(this_week / prior_week, 2) if prior_week
                else (None if this_week == 0 else float("inf")))

        return {
            "uptime_seconds": round(time.time() - _start),
            "turns_total": _turns_total,
            "actions": dict(_action_counts),
            "abstention_rate": _rate(_action_counts.get("ABSTAIN", 0), _turns_total),
            "crisis_fire_rate": _rate(_crisis_fires, _turns_total),
            "ood_rate": _rate(_action_counts.get("OOD", 0), _turns_total),
            "resolve_rate": _rate(_action_counts.get("RESOLVE", 0), _turns_total),
            "llm": {
                "ok": _llm_ok,
                "failed": _llm_failed,
                "skipped_no_key": _llm_skipped,
                "failure_rate_of_attempts": _rate(_llm_failed, _llm_ok + _llm_failed),
            },
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "p99": _percentile(latencies, 0.99),
                "samples": len(latencies),
            },
            "crisis_week_over_week": {
                "this_week": this_week,
                "prior_week": prior_week,
                "ratio": ratio,
                # matches ESCALATION_POLICY.md section 4 verbatim
                "needs_manual_review": bool(ratio and ratio >= 3.0),
            },
            "scope": "in-process, resets on restart. See db.Turn for durable "
                    "history.",
        }
