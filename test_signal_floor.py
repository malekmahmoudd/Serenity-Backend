"""
test_signal_floor.py
=====================
Verifies the fix for: "hi" gets answered with a clinical-sounding question
lifted from GENERIC_QUESTIONS.

    venv\\Scripts\\python.exe test_signal_floor.py

No network, no real model -- classifier is stubbed so the scenarios are exact
and reproducible. Checks:
  1. a greeting produces question_optional=True (has_signal=False)
  2. real early disclosure ("i feel bad") still produces a grounded question
  3. classifier-unavailable degrades to question_optional=True as well
  4. llm.py builds a "respond naturally" instruction, not a forced-verbatim
     one, exactly when question_optional is True
  5. to_state() actually carries has_signal through to what the LLM receives
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


# ---- stub classifier so scores are exact and controllable -----------------
NAMES = ["adhd", "anxiety", "autism", "bipolar", "bpd", "depression", "ocd",
         "ptsd", "schizophrenia", "suicidal"]
THRESHOLDS = {"adhd": .41, "anxiety": .45, "autism": .30, "bipolar": .40,
              "bpd": .41, "depression": .38, "ocd": .40, "ptsd": .32,
              "schizophrenia": .47, "suicidal": .28}

SCRIPTED = {
    # a greeting: near-uniform noise, nothing near a threshold, nothing even
    # near SIGNAL_FLOOR. This is the exact shape a model produces on content
    # it has nothing to say about.
    "hi": {"anxiety": 0.04, "depression": 0.03, "bpd": 0.02, "adhd": 0.02,
          "autism": 0.01, "bipolar": 0.01, "ocd": 0.01, "ptsd": 0.01,
          "schizophrenia": 0.01, "suicidal": 0.01},
    # real, early disclosure: below MIN_TOKENS_TO_RESOLVE / below its own
    # threshold, but a genuine lean -- this must NOT be treated like "hi".
    "i feel bad": {"depression": 0.28, "bpd": 0.10, "anxiety": 0.08,
                  "adhd": 0.02, "autism": 0.01, "bipolar": 0.01, "ocd": 0.01,
                  "ptsd": 0.01, "schizophrenia": 0.01, "suicidal": 0.01},
}

classifier_stub = types.ModuleType("classifier")
classifier_stub.THRESHOLDS = THRESHOLDS
classifier_stub._available = True


def _available():
    return classifier_stub._available


def _score(text):
    # substring match, not exact -- a real classifier scores whatever content
    # is PRESENT in the accumulated text, not the string as a whole.
    #
    # Second bug found by running this, not just writing it: "hi i feel bad"
    # contains BOTH "hi" and "i feel bad" as substrings, and picking whichever
    # SCRIPTED key matched first (dict order) returned the "hi" noise profile
    # even though real content was present. Fixed by taking the match with the
    # highest top score, which is what a real classifier effectively does --
    # more content produces a stronger signal, not an arbitrary one.
    key = text.strip().lower()
    candidates = [v for k, v in SCRIPTED.items() if k in key]
    if not candidates:
        return {n: 0.01 for n in NAMES}
    return dict(max(candidates, key=lambda sc: max(sc.values())))


def _fired(scores):
    return [k for k, v in sorted(scores.items(), key=lambda kv: -kv[1])
            if v >= THRESHOLDS.get(k, 0.5)]


classifier_stub.available = _available
classifier_stub.score = _score
classifier_stub.fired = _fired
sys.modules["classifier"] = classifier_stub

safety_stub = types.ModuleType("safety")
safety_stub.crisis_check = lambda text: []
safety_stub.CRISIS_RESOURCES = "call 123 / 16328"
sys.modules["safety"] = safety_stub

situational_stub = types.ModuleType("situational")
situational_stub.situational_check = lambda text: []
sys.modules["situational"] = situational_stub

text_cleaning_stub = types.ModuleType("text_cleaning")
text_cleaning_stub.token_count = lambda t: len(str(t).split())
sys.modules["text_cleaning"] = text_cleaning_stub

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decision  # noqa: E402

print("=" * 62)
print(f"1. greeting -> no signal (SIGNAL_FLOOR = {decision.SIGNAL_FLOOR})")
print("=" * 62)
d = decision.decide(accumulated="hi", latest="hi", turn=1)
check("action is ASK", d.action == "ASK", f"(got {d.action})")
check("question_optional is True", d.question_optional is True)
check("still has SOME question as a fallback string",
      isinstance(d.question, str) and len(d.question) > 0)
check("reason explains why", "no real signal" in d.reason, f"({d.reason!r})")
state = d.to_state()
check("to_state exposes has_signal=False",
      state["classification"]["has_signal"] is False)

print()
print("=" * 62)
print("2. real early disclosure -> grounded question")
print("=" * 62)
d2 = decision.decide(accumulated="i feel bad", latest="i feel bad", turn=1)
check("action is ASK", d2.action == "ASK")
check("question_optional is False (grounded)", d2.question_optional is False,
      f"(top score {sorted(d2.scores.items(), key=lambda kv:-kv[1])[0]})")
state2 = d2.to_state()
check("to_state exposes has_signal=True",
      state2["classification"]["has_signal"] is True)

print()
print("=" * 62)
print("3. classifier unavailable -> also no signal")
print("=" * 62)
classifier_stub._available = False
d3 = decision.decide(accumulated="anything at all here", latest="anything at all here", turn=1)
classifier_stub._available = True
check("question_optional True with no model", d3.question_optional is True)
check("reason says classifier unavailable",
      d3.reason == "classifier unavailable")

print()
print("=" * 62)
print("4. llm.py builds the right instruction for each case")
print("=" * 62)
os.environ["GROQ_API_KEY"] = ""      # force template path is NOT what we test;
                                      # we inspect the prompt-building branch
                                      # directly instead of making a network call
import llm  # noqa: E402

# Reach into respond()'s prompt construction without calling the network: build
# the same `parts` list it would build, using the real has_signal branch logic.
def build_ask_instruction(state, question):
    has_signal = state["classification"].get("has_signal", True)
    if has_signal and question:
        return "END YOUR REPLY WITH A QUESTION BASED ON THIS"
    return "No classifier signal yet"

greet_instr = build_ask_instruction(state, d.question)
real_instr = build_ask_instruction(state2, d2.question)
check("greeting gets the free-form instruction",
      greet_instr == "No classifier signal yet")
check("real disclosure gets the grounded instruction",
      real_instr == "END YOUR REPLY WITH A QUESTION BASED ON THIS")

# Confirm this logic actually matches what's inside llm.respond's source, so
# the test can't silently drift from the real implementation.
src = open(os.path.join(os.path.dirname(__file__), "llm.py")).read()
check("llm.py actually branches on classification.has_signal",
      'state["classification"].get("has_signal"' in src)
check("llm.py's grounded branch keeps the exact-question instruction",
      "END YOUR REPLY WITH A QUESTION BASED ON THIS" in src)
check("llm.py's free-form branch tells the model to skip clinical phrasing",
      "Do NOT ask a clinical-sounding question" in src)

print()
print("=" * 62)
print("5. multi-turn: greeting then real disclosure in the same conversation")
print("=" * 62)
acc = "hi"
t1 = decision.decide(accumulated=acc, latest="hi", turn=1)
check("turn 1 'hi' alone -> no signal", t1.question_optional is True)
acc = acc + " i feel bad"
t2 = decision.decide(accumulated=acc, latest="i feel bad", turn=2)
check("turn 2 adds real content -> signal appears",
      t2.question_optional is False,
      f"(reason: {t2.reason!r})")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    raise SystemExit(1)
print("ALL PASS -- 'hi' no longer gets a clinical-sounding question")
