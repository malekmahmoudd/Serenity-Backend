"""
test_reply_integrity.py
=======================
Two defects observed in a live conversation on 2026-08-20, both fixed here.

    venv\\Scripts\\python.exe test_reply_integrity.py

1. THE SAME QUESTION, TWICE
   A user was asked "when your mood shifts, does it usually follow something
   happening with someone close to you, or does it come on by itself?" on turn
   1, answered it, and was asked it again verbatim on turn 2.

   pick_question indexed candidates[(turn - 1) % len(candidates)] and its
   docstring claimed that prevented repeats. It does not: the candidate list is
   ordered by which class pair is closest to tied, so it REORDERS as scores
   move between turns, and index 1 on turn 2 can be the question index 0
   returned on turn 1.

2. A REPLY DELIVERED CUT IN HALF
   The practical-tips turn came back as "...keeping a simple sleep routine by"
   and stopped. gpt-oss bills reasoning as completion tokens, so at the default
   reasoning effort the 400-token ceiling was spent before the sentence
   finished. Nothing inspected finish_reason, so the fragment went to the user.

No network: the classifier is stubbed and the HTTP call is monkeypatched.
"""

import json
import os
import sys
import types

os.environ["SERENITY_ENC_KEY"] = "sVh0kQ2Zp7XcJ9nR4tB6yU8wE1oA3iL5mF7gD0hK2vM="
os.environ["SERENITY_JWT_SECRET"] = "test-secret"

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


# ---- stub classifier -------------------------------------------------------
# bipolar and bpd deliberately near-tied, which is the shape that produced the
# repeat: their QUESTION_BANK entry wins on both turns.
NAMES = ["adhd", "anxiety", "autism", "bipolar", "bpd", "depression", "ocd",
         "ptsd", "schizophrenia", "suicidal"]

TURN_SCORES = [
    {**{n: 0.02 for n in NAMES}, "bipolar": 0.31, "bpd": 0.29, "depression": 0.20},
    # turn 2: same pair still closest, slightly reordered -- the exact
    # condition under which the old index rotation repeated itself
    {**{n: 0.02 for n in NAMES}, "bpd": 0.32, "bipolar": 0.30, "depression": 0.22},
]
_call_count = {"n": 0}

stub = types.ModuleType("classifier")
stub.available = lambda: True
stub.score = lambda text: dict(TURN_SCORES[min(_call_count["n"], len(TURN_SCORES) - 1)])
stub.fired = lambda scores: []
stub.THRESHOLDS = {n: 0.40 for n in NAMES}
stub.LABELS = NAMES
sys.modules["classifier"] = stub

import decision  # noqa: E402
import llm       # noqa: E402

print("\n=== 1. the same question is never asked twice ===")

asked = []
questions = []
for turn in (1, 2, 3):
    _call_count["n"] = turn - 1
    d = decision.decide(accumulated="x " * (turn * 30), latest="x " * 30,
                        turn=turn, asked=asked)
    if d.question:
        questions.append(d.question)
        asked.append(d.question)

check("three turns produced three questions", len(questions) == 3,
      f"got {len(questions)}")
check("no question repeated", len(set(questions)) == len(questions),
      f"{len(set(questions))} unique of {len(questions)}")
for i, q in enumerate(questions, start=1):
    print(f"      turn {i}: {q[:72]}...")

print("\n=== 2. pick_question skips anything already asked ===")
first = decision.pick_question(TURN_SCORES[0], turn=1, asked=())
second = decision.pick_question(TURN_SCORES[0], turn=1, asked=(first,))
check("same scores + same turn yields a different question when excluded",
      first != second)
check("the excluded question is not returned", second != first)

everything = list(decision.QUESTION_BANK.values()) + decision.GENERIC_QUESTIONS
fallback = decision.pick_question(TURN_SCORES[0], turn=1, asked=everything)
check("still returns a question when all are exhausted", bool(fallback))

print("\n=== 3. a truncated reply never reaches the user ===")


def fake_urlopen_factory(finish_reason, content):
    class FakeResponse:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *a):
            return False

        def read(self_inner):
            return json.dumps({
                "choices": [{"message": {"content": content},
                             "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 400},
            }).encode()

    return lambda req, timeout=30: FakeResponse()


llm._get_api_key = lambda: "test-key"
state = {
    "classification": {"leading_categories": [], "evidence_sufficiency": "insufficient",
                       "already_introduced": False, "has_signal": True},
    "safety": {"crisis_signal": False, "crisis_patterns": [],
               "crisis_earlier_in_conversation": False},
    "domain": {"in_domain": True},
    "context": {"situational_causes": []},
    "conversation": {"turn": 2, "evidence_tokens": 40, "informational_request": False},
    "decision": {"action": "ASK", "reason": "test"},
}
history = [{"role": "user", "content": "hello"}]

import urllib.request  # noqa: E402
real_urlopen = urllib.request.urlopen

# truncated mid-sentence
urllib.request.urlopen = fake_urlopen_factory(
    "length", "keeping a simple sleep routine by")
out = llm.respond(state, history, question="How long has this been going on for?")
check("truncated reply replaced by the template",
      "keeping a simple sleep routine by" not in out, repr(out[:70]))
check("template is a complete sentence", out.strip().endswith(("?", ".", "!")),
      repr(out[-40:]))

# empty content
urllib.request.urlopen = fake_urlopen_factory("stop", "")
out = llm.respond(state, history, question="How long has this been going on for?")
check("empty reply replaced by the template", bool(out.strip()))

# a normal complete reply still passes through untouched
urllib.request.urlopen = fake_urlopen_factory(
    "stop", "That sounds difficult. How long has this been going on for?")
out = llm.respond(state, history, question="How long has this been going on for?")
check("a complete reply is passed through unchanged",
      out == "That sounds difficult. How long has this been going on for?", repr(out))

urllib.request.urlopen = real_urlopen

print("\n=== 4. the request asks for low reasoning effort ===")
captured = {}


class CaptureResponse:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps({
            "choices": [{"message": {"content": "ok."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }).encode()


def capture_urlopen(req, timeout=30):
    captured["body"] = json.loads(req.data.decode())
    return CaptureResponse()


urllib.request.urlopen = capture_urlopen
llm._call("system", "user")
urllib.request.urlopen = real_urlopen

check("reasoning_effort is set to low",
      captured["body"].get("reasoning_effort") == "low",
      str(captured["body"].get("reasoning_effort")))
check("max_tokens still capped", captured["body"].get("max_tokens") == 400)

print("\n" + "=" * 70)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("ALL PASS -- no repeated questions, no half-sentences delivered")
