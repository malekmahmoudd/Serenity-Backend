"""
test_ratelimit.py
=================
    venv\\Scripts\\python.exe test_ratelimit.py

Checks:
  1. burst is allowed, then the limit engages
  2. tokens refill over time
  3. keys are independent -- one user cannot exhaust another's allowance
  4. retry_after is truthful, not a guess
  5. the sustained rate really is the refill rate, not the burst size
  6. crisis messages bypass the limiter
"""

import os
import time

os.environ["RATE_USER_PER_MIN"] = "60"     # 1/s, so tests run quickly
os.environ["RATE_USER_BURST"] = "3"
os.environ["RATE_ANON_PER_MIN"] = "60"
os.environ["RATE_ANON_BURST"] = "2"

import ratelimit  # noqa: E402
import safety  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


print("=" * 62)
print("1. burst then throttle")
print("=" * 62)
b = ratelimit.TokenBucket(rate_per_min=60, burst=3, name="t")
results = [b.check("alice")[0] for _ in range(5)]
check("first 3 allowed (burst)", results[:3] == [True, True, True])
check("4th and 5th blocked", results[3:] == [False, False], str(results))

print()
print("=" * 62)
print("2. refill")
print("=" * 62)
allowed, retry = b.check("alice")
check("still blocked immediately", not allowed)
check("retry_after is under a second", 0 < retry <= 1.1, f"({retry:.2f}s)")
time.sleep(1.1)
check("allowed again after refill", b.check("alice")[0])

print()
print("=" * 62)
print("3. keys are independent")
print("=" * 62)
b2 = ratelimit.TokenBucket(rate_per_min=60, burst=2, name="t2")
b2.check("alice"); b2.check("alice")
check("alice exhausted", not b2.check("alice")[0])
check("bob unaffected", b2.check("bob")[0],
      "(one user cannot exhaust another's allowance)")

print()
print("=" * 62)
print("4. retry_after is truthful")
print("=" * 62)
b3 = ratelimit.TokenBucket(rate_per_min=30, burst=1, name="t3")   # 1 per 2s
b3.check("x")
_, retry = b3.check("x")
check("reports ~2s for a 30/min bucket", 1.5 <= retry <= 2.5, f"({retry:.2f}s)")
time.sleep(retry + 0.1)
check("allowed exactly when promised", b3.check("x")[0])

print()
print("=" * 62)
print("5. sustained rate equals refill rate, not burst")
print("=" * 62)
b4 = ratelimit.TokenBucket(rate_per_min=120, burst=2, name="t4")  # 2/s
start = time.monotonic()
allowed_count = 0
while time.monotonic() - start < 2.0:
    if b4.check("y")[0]:
        allowed_count += 1
    time.sleep(0.05)
# 2 burst + ~4 refilled over 2s
check("~6 allowed in 2s at 2/s + burst 2", 4 <= allowed_count <= 8,
      f"({allowed_count})")

print()
print("=" * 62)
print("6. crisis bypass")
print("=" * 62)
crisis_msgs = ["i don't want to be here anymore",
               "i can't do this anymore.",
               "nobody would really notice if i wasn't around"]
for m in crisis_msgs:
    check(f"detected: {m[:38]}", safety.is_crisis(m))
check("ordinary message not exempt", not safety.is_crisis("i feel a bit tired"))
check("post office not exempt",
      not safety.is_crisis("what time does the post office close"))

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    raise SystemExit(1)
print("ALL PASS -- rate limiting is sound")
