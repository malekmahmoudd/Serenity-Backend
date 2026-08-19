"""
test_informational.py
=====================
Verifies the fix for: "tel me about ptsd symptoms" was answered in one line and
then followed with "What does it look like when it's at its worst?" -- an
invitation to narrate trauma, aimed at someone who had asked a library
question.

    venv\\Scripts\\python.exe test_informational.py

The underlying cause was that SIGNAL_FLOOR only screens out the NO-signal case
(a greeting). Naming a condition gives the classifier a genuine above-floor
signal, so has_signal stayed true and the canned GENERIC_QUESTIONS entry was
handed to the LLM to recite verbatim -- while llm.py's own prompt separately
forbids that exact phrasing by name. The missing distinction was not signal
strength but whether the person is talking about THEMSELVES.

No network and no real model: the classifier is stubbed so scenarios are exact.
Checks:
  1. explicit information requests are detected, including the reported typo
  2. first-person disclosure is NEVER treated as an information request
  3. mixed messages ("i think i have adhd, what are the symptoms") stay on the
     support path -- the asymmetry is deliberate
  4. crisis language still routes to SAFETY ahead of this gate
  5. an information request attaches no canned question at all
  6. to_state() carries informational_request through to the LLM
  7. llm.py builds an answer-it instruction, not a probe instruction
  8. the offline fallback for an info request does not ask anything back
"""

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


# ---- stub classifier ------------------------------------------------------
# Scores chosen so ptsd sits well above SIGNAL_FLOOR: this reproduces the real
# failure, where there WAS signal and the old code therefore recited a
# question. A stub that returned noise would pass for the wrong reason.
NAMES = ["adhd", "anxiety", "autism", "bipolar", "bpd", "depression", "ocd",
         "ptsd", "schizophrenia", "suicidal"]
PTSD_SIGNAL = {n: 0.02 for n in NAMES}
PTSD_SIGNAL.update({"ptsd": 0.31, "anxiety": 0.24})

stub = types.ModuleType("classifier")
stub.available = lambda: True
stub.score = lambda text: dict(PTSD_SIGNAL)
stub.fired = lambda scores: []
stub.LABELS = NAMES
sys.modules["classifier"] = stub

import decision  # noqa: E402  (must follow the stub)

print("\n=== 1. information requests are detected ===")
INFO = [
    "tel me about ptsd symptoms",          # the exact reported message
    "tell me about PTSD symptoms",
    "what are the symptoms of ptsd",
    "what is depression",
    "what causes anxiety",
    "explain ocd",
    "define bipolar",
    "signs of adhd",
    "what's the difference between anxiety and panic",
    "how do you know if someone has autism",
]
for text in INFO:
    check(f"info: {text!r}", decision.informational_request(text) is True)

print("\n=== 2. first-person disclosure is never an information request ===")
DISCLOSURE = [
    "i feel anxious all the time",
    "i've been struggling with flashbacks",
    "my ptsd is getting worse",
    "i can't sleep and i keep panicking",
    "i was diagnosed with bipolar last year",
    "i'm scared of what's happening to me",
    "my mum died in march",
    "hi",
    "",
    "   ",
]
for text in DISCLOSURE:
    check(f"not info: {text!r}", decision.informational_request(text) is False)

print("\n=== 3. mixed messages stay on the support path ===")
# Deliberate: dropping support for someone who disclosed is far worse than
# leaving today's behaviour in place for an ambiguous message.
MIXED = [
    "i think i might have adhd, what are the symptoms",
    "i feel awful, what is depression",
]
for text in MIXED:
    check(f"mixed stays support: {text!r}",
          decision.informational_request(text) is False)

print("\n=== 4. crisis still wins ===")
d = decision.decide(accumulated="i want to kill myself",
                    latest="i want to kill myself", turn=1)
check("crisis routes to SAFETY", d.action == "SAFETY", f"got {d.action}")
d = decision.decide(accumulated="what are the signs someone is suicidal",
                    latest="what are the signs someone is suicidal", turn=1)
check("crisis phrasing in an info question still routes to SAFETY",
      d.action == "SAFETY", f"got {d.action}")

print("\n=== 5-6. the reported case attaches no question ===")
msg = "tel me about ptsd symptoms"
d = decision.decide(accumulated=msg, latest=msg, turn=2)
check("informational flag set", d.informational is True)
check("no canned question attached", d.question is None, f"got {d.question!r}")
check("action is ASK", d.action == "ASK", f"got {d.action}")

state = d.to_state()
check("state carries informational_request",
      state["conversation"].get("informational_request") is True)
check("has_signal is false so nothing is recited",
      state["classification"]["has_signal"] is False)
check("no category is claimed about them",
      state["classification"]["leading_categories"] == [])

print("\n=== 7. llm.py builds an answer-it instruction ===")
import llm  # noqa: E402

captured = {}


def fake_call(system, user, timeout=30):
    captured["system"] = system
    captured["user"] = user
    return "PTSD can involve flashbacks, avoidance and feeling on edge."


llm._call = fake_call
llm._get_api_key = lambda: "test-key"

llm.respond(state, [{"role": "user", "content": msg}], question=d.question)
user_prompt = captured.get("user", "")
check("prompt says this is an information request",
      "INFORMATION REQUEST" in user_prompt)
check("prompt forbids a follow-up question",
      "Do NOT end with a question" in user_prompt)
check("prompt does not force a verbatim probe",
      "END YOUR REPLY WITH A QUESTION" not in user_prompt)
check("system prompt carries the informational rule",
      "informational_request" in captured.get("system", ""))

print("\n=== 8. offline fallback does not ask anything back ===")
llm._get_api_key = lambda: ""
offline = llm.respond(state, [{"role": "user", "content": msg}],
                      question=d.question)
check("fallback asks no question", "?" not in offline, repr(offline))
check("fallback does not say 'thank you for telling me that'",
      "telling me that" not in offline.lower())

print("\n" + "=" * 70)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("ALL PASS -- information requests are answered, not interrogated")
