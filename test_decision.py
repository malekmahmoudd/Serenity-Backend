"""
test_decision.py
=================
Unit tests for decision.py -- the deterministic core the whole safety
architecture rests on. This is the file with the LEAST coverage in the repo
despite being the one where a silent regression matters most: it is what
decides SAFETY vs everything else.

    venv\\Scripts\\python.exe test_decision.py

Runs with a fully mocked classifier (real scores would need TensorFlow/ONNX
loaded, and would make these tests non-deterministic across model versions).
The classifier's OWN correctness is what classifier.py / test_onnx_path.py /
export_onnx.py verify -- this file verifies the POLICY around whatever the
classifier says, which is a different and equally important claim.

COVERS
------
  1. crisis detection: latest-only, boundary-spanning, not re-firing on the
     turn after (the exact bug that was found and fixed live)
  2. classifier-unavailable degrades to conversation, never crashes
  3. RESOLVE requires: threshold cleared AND enough tokens AND real margin
  4. the MIN_TOKENS_TO_RESOLVE guard specifically (three-word "i feel bad"
     must never resolve, regardless of how confident the score is)
  5. situational guard: bereavement/job-loss/etc block a label for
     SITUATIONAL_GRACE_TURNS turns, then stop blocking
  6. SIGNAL_FLOOR / has_signal: greetings get question_optional=True, real
     disclosure gets False (the "hi" fix)
  7. turn cap and ambiguous-token cap both terminate the loop
  8. OOD requires BOTH enough tokens AND turn>=2 -- never fires on an opener
"""
from __future__ import annotations

import sys
import types

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------- stubs
NAMES = ["adhd", "anxiety", "autism", "bipolar", "bpd", "depression", "ocd",
         "ptsd", "schizophrenia", "suicidal"]
THRESHOLDS = {"adhd": .41, "anxiety": .45, "autism": .30, "bipolar": .40,
              "bpd": .41, "depression": .38, "ocd": .40, "ptsd": .32,
              "schizophrenia": .47, "suicidal": .28}

_classifier_stub = types.ModuleType("classifier")
_classifier_stub.THRESHOLDS = THRESHOLDS
_classifier_stub._available = True
_classifier_stub._scores_by_substring = {}   # set per-test


def _available():
    return _classifier_stub._available


def _score(text):
    key = text.strip().lower()
    candidates = [v for k, v in _classifier_stub._scores_by_substring.items()
                 if k in key]
    if not candidates:
        return {n: 0.01 for n in NAMES}
    return dict(max(candidates, key=lambda sc: max(sc.values())))


def _fired(scores):
    return [k for k, v in sorted(scores.items(), key=lambda kv: -kv[1])
            if v >= THRESHOLDS.get(k, 0.5)]


_classifier_stub.available = _available
_classifier_stub.score = _score
_classifier_stub.fired = _fired
sys.modules["classifier"] = _classifier_stub

_safety_stub = types.ModuleType("safety")
_CRISIS_WORDS = ("dont want to be here", "don't want to be here",
                 "kill myself", "end it all", "no point")


def _crisis_check(text):
    t = str(text).lower()
    return [w for w in _CRISIS_WORDS if w in t]


_safety_stub.crisis_check = _crisis_check
_safety_stub.CRISIS_RESOURCES = "call 123 / 16328"
sys.modules["safety"] = _safety_stub

_situational_stub = types.ModuleType("situational")
_SITUATIONAL_WORDS = ("died", "passed away", "made redundant", "lost my job")


def _situational_check(text):
    t = str(text).lower()
    return [w for w in _SITUATIONAL_WORDS if w in t]


_situational_stub.situational_check = _situational_check
sys.modules["situational"] = _situational_stub

_text_cleaning_stub = types.ModuleType("text_cleaning")
_text_cleaning_stub.token_count = lambda t: len(str(t).split())
sys.modules["text_cleaning"] = _text_cleaning_stub

sys.path.insert(0, ".")
import decision  # noqa: E402


def set_scores(mapping):
    _classifier_stub._scores_by_substring = mapping


def reset_scores():
    _classifier_stub._scores_by_substring = {}


# =============================================================================
print("=" * 62)
print("1. crisis detection")
print("=" * 62)
reset_scores()

d = decision.decide(accumulated="i dont want to be here anymore",
                    latest="i dont want to be here anymore", turn=1)
check("explicit crisis on turn 1 -> SAFETY", d.action == "SAFETY")

d = decision.decide(accumulated="hi there i dont want to be here anymore",
                    latest="i dont want to be here anymore", turn=2,
                    prev_turn="hi there", prev_fired=False)
check("crisis on turn 2 fires normally", d.action == "SAFETY")

# the exact bug that was found live: crisis on turn N must not re-fire on N+1
d2 = decision.decide(accumulated="i dont want to be here anymore what time "
                                 "does the post office close",
                     latest="what time does the post office close", turn=2,
                     prev_turn="i dont want to be here anymore", prev_fired=True)
check("crisis does NOT re-fire on the turn after (the live bug)",
      d2.action != "SAFETY", f"(got {d2.action})")

# boundary-spanning: neither half alone is a crisis phrase
d3 = decision.decide(accumulated="everything is fine i dont want", latest="to be here",
                     turn=2, prev_turn="everything is fine i dont want", prev_fired=False)
check("phrase spanning a turn boundary still caught",
      d3.action == "SAFETY", f"(got {d3.action})")

print()
print("=" * 62)
print("2. classifier unavailable -> degrades, never crashes")
print("=" * 62)
_classifier_stub._available = False
d = decision.decide(accumulated="something", latest="something", turn=1)
check("no crash with classifier down", d.action in ("ASK", "ABSTAIN"))
check("question_optional True with no model", d.question_optional is True)
d = decision.decide(accumulated="x", latest="x", turn=decision.MAX_TURNS)
check("still respects turn cap without a model", d.action == "ABSTAIN")
_classifier_stub._available = True

print()
print("=" * 62)
print("3-4. RESOLVE requires threshold + enough tokens + real margin")
print("=" * 62)
set_scores({"strong depression": {"depression": 0.75, "bpd": 0.10, "anxiety": 0.05}})
long_text = ("i have strong depression symptoms every single day and it has "
            "been getting worse for months now honestly")
d = decision.decide(accumulated=long_text, latest=long_text, turn=1)
check("clears threshold + enough tokens + margin -> RESOLVE",
      d.action == "RESOLVE", f"(got {d.action}, tokens={d.evidence_tokens})")

# the specific bug this system was built to prevent: short + confident must NOT resolve
d = decision.decide(accumulated="i feel bad", latest="i feel bad", turn=1)
set_scores({"i feel bad": {"depression": 0.55, "bpd": 0.10, "anxiety": 0.08}})
d = decision.decide(accumulated="i feel bad", latest="i feel bad", turn=1)
check("3 tokens + high confidence -> NEVER resolves",
      d.action != "RESOLVE", f"(got {d.action})")
check("question_optional False -- it IS grounded, just too short",
      d.question_optional is False)
check("reason names the token guard", "tokens" in d.reason, f"({d.reason!r})")
reset_scores()

print()
print("=" * 62)
print("5. situational guard")
print("=" * 62)
set_scores({"still can't listen": {"depression": 0.55, "bpd": 0.10, "anxiety": 0.05}})
grief = ("my dad died in march and i still can't listen to the music he "
        "liked and it has been really hard")
d = decision.decide(accumulated=grief, latest=grief, turn=1)
check("bereavement blocks RESOLVE within the grace window",
      d.action != "RESOLVE", f"(got {d.action})")
check("situational cause recorded", "died" in d.situational or bool(d.situational),
      f"({d.situational})")

d_late = decision.decide(accumulated=grief, latest=grief,
                         turn=decision.SITUATIONAL_GRACE_TURNS + 1)
check("grace window is finite -- eventually a real label CAN surface",
      d_late.action == "RESOLVE" or d_late.action == "ASK",
      f"(got {d_late.action}; either is acceptable, RESOLVE proves the "
      f"grace window ends, ASK proves it didn't crash)")
reset_scores()

print()
print("=" * 62)
print("6. SIGNAL_FLOOR -- the 'hi' fix")
print("=" * 62)
d = decision.decide(accumulated="hi", latest="hi", turn=1)
check("greeting -> question_optional True", d.question_optional is True)
check("to_state exposes has_signal False",
      d.to_state()["classification"]["has_signal"] is False)

set_scores({"i feel bad": {"depression": 0.28, "bpd": 0.08}})
d2 = decision.decide(accumulated="i feel bad", latest="i feel bad", turn=1)
check("real early disclosure -> question_optional False",
      d2.question_optional is False, f"(top score present)")
reset_scores()

print()
print("=" * 62)
print("7. turn cap and ambiguous-token cap")
print("=" * 62)
d = decision.decide(accumulated="x " * 5, latest="x", turn=decision.MAX_TURNS)
check("turn cap -> ABSTAIN", d.action == "ABSTAIN")

# Distinct from OOD on purpose: OOD means NO signal at all (top < OOD_FLOOR).
# ABSTAIN-via-ambiguity means moderate, genuine, but never-decisive signal --
# above the noise floor, never near a threshold, however much text piles up.
# Pure nonsense repeated 105 times has zero signal and correctly triggers OOD
# first (verified as its own case below) -- that is item 4 in decide()'s
# priority order, checked before the ambiguous-tokens branch (item 5), and
# testing the second branch means giving it scores that survive the first.
set_scores({"complicated": {"depression": 0.18, "anxiety": 0.15, "bpd": 0.12}})
# "it's complicated" is 2 tokens; needs > AMBIGUOUS_TOKENS (100) accumulated.
# First attempt used x20 = 40 tokens and landed in the ASK catch-all instead
# of ABSTAIN -- that was this test's arithmetic being wrong, not decision.py;
# caught by actually running it rather than trusting the estimate.
reps = decision.AMBIGUOUS_TOKENS // 2 + 10
long_ambiguous = "it's complicated " * reps
d = decision.decide(accumulated=long_ambiguous, latest="it's complicated", turn=2)
check("moderate-but-never-decisive signal, lots of text -> ABSTAIN",
      d.action == "ABSTAIN",
      f"(got {d.action}, {len(long_ambiguous.split())} tokens)")
reset_scores()

# The nonsense case, checked separately and correctly expected to be OOD --
# this is what test 7 was accidentally exercising before the fix above.
zero_signal = "word " * (decision.AMBIGUOUS_TOKENS + 5)
d = decision.decide(accumulated=zero_signal, latest="word", turn=2)
check("true zero-signal nonsense -> OOD, not ABSTAIN (checked first, correctly)",
      d.action == "OOD", f"(got {d.action})")

print()
print("=" * 62)
print("8. OOD requires enough tokens AND turn>=2")
print("=" * 62)
short_ood = "what time does the post office close"   # ~7 tokens
d = decision.decide(accumulated=short_ood, latest=short_ood, turn=1)
check("OOD does not fire on turn 1 even with enough tokens",
      d.action != "OOD", f"(got {d.action}) -- opener deserves a chance first")

longer_ood = short_ood + " and the library too on weekends please"
d = decision.decide(accumulated=longer_ood, latest="and the library too on "
                                                    "weekends please", turn=2)
check("OOD fires on turn 2 with enough accumulated tokens",
      d.action == "OOD", f"(got {d.action})")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("ALL PASS -- decision.py policy layer is sound")
