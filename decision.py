"""
decision.py
===========
The decision layer. Entirely deterministic -- no model, no LLM, no randomness.

That is the point. Every branch here can be unit-tested exhaustively, which is
not true of anything that involves a network. The models produce signals; this
file decides what to do with them.

    SAFETY   crisis language present -> resources, stop classifying
    OOD      no signal at all and enough words to judge -> not a support query
    ASK      not enough evidence yet -> one targeted follow-up
    RESOLVE  a label cleared its threshold -> hand to the LLM
    ABSTAIN  turn cap reached or genuinely ambiguous -> general support

THE CORE MECHANIC
-----------------
Classification runs on the ACCUMULATED conversation, not the latest message.
The model was trained on ~114-token posts; a four-word opener is out of
distribution, which is why it abstains. Concatenating turns moves the input
back toward what the model knows, so confidence rises for the right reason.

CRISIS DETECTION USES A SLIDING WINDOW, NOT THE FULL TRANSCRIPT
---------------------------------------------------------------
An earlier version checked `latest OR accumulated`. That pinned a session in
SAFETY permanently: once any turn contained crisis language, every later turn
re-matched it from the accumulated text, and a question about post office
opening hours came back with crisis resources attached.

The window covers the latest message plus the one before it, so an expression
split across a turn boundary ("i just" / "don't want to be here anymore") is
still caught -- without turn 12 being held hostage by turn 2.

The session still REMEMBERS that crisis was flagged (see main.py). Memory and
routing are different things: someone who expressed hopelessness and then talks
about their weekend should get a normal reply, not resources every turn.

TWO KINDS OF ABSTENTION, OPPOSITE CORRECT RESPONSES
---------------------------------------------------
Below MIN_TOKENS_TO_RESOLVE, low confidence means "not enough text yet" -> ask.
Above AMBIGUOUS_TOKENS, it means "genuinely unclear" -> stop asking. Treating
them the same is how a support tool turns into an interrogation.

MAX_TURNS IS NOT NEGOTIABLE
---------------------------
An unbounded loop that keeps probing a distressed person to raise a confidence
score has quietly replaced "help this person" with "satisfy the classifier".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

import classifier
import safety
import situational
from text_cleaning import token_count

MAX_TURNS = 4                 # hard cap on follow-up questions
# Raised from 12 after a live failure: "my dad died in march and i still can't
# listen to the music he liked" resolved to depression on turn one at 15
# tokens. Naming a condition from a single opener skips the elicitation loop
# entirely, which is the thing that makes a 0.42-accuracy classifier usable.
# The successful test run resolved at 21 tokens over three turns -- that is the
# shape we want.
MIN_TOKENS_TO_RESOLVE = 15
AMBIGUOUS_TOKENS = 100        # above this, stop asking and route to general
OOD_FLOOR = 0.08              # every class below this = no signal at all
OOD_MIN_TOKENS = 8            # never judge out-of-domain on an opener
# Below this, the classifier has essentially nothing to say -- a greeting like
# "hi" scores near-uniform noise across all ten classes. Below SIGNAL_FLOOR the
# decision layer stops handing the LLM a discriminating question to recite --
# that was the reported bug: "hi" came back with "what does it look like at
# its worst". Above it, even a low score is a real lean worth building a
# question around. Sits between OOD_FLOOR (nothing at all) and the lowest
# calibrated class threshold (~0.23), so it screens out noise without
# screening out genuine low-confidence disclosures. Tune from /dashboard top3
# scores on real greetings if these still slip through.
SIGNAL_FLOOR = 0.15
MARGIN = 0.08                 # near-tie -> treat as unresolved
CRISIS_WINDOW = 2             # how many recent turns the crisis rule sees
# For how many turns a stated external cause (bereavement, redundancy, divorce)
# blocks a clinical label. Not forever: grief and depression can coexist, and
# someone whose grief has become something heavier should still get there --
# via more conversation, not via a label on turn one.
SITUATIONAL_GRACE_TURNS = 3

# ---------------------------------------------------------------------------
# ASKING FOR INFORMATION IS NOT THE SAME AS DISCLOSING
# ---------------------------------------------------------------------------
# Reported failure: "tell me about ptsd symptoms" came back with a short answer
# and then "What does it look like when it's at its worst?" -- a request to
# narrate trauma, aimed at someone who had asked a library question.
#
# SIGNAL_FLOOR already covers the no-signal case ("hi"). It cannot cover this
# one: naming a condition gives the classifier a real, above-floor signal, so
# has_signal stays true and the canned question gets recited. The missing
# distinction is not how much signal there is, but whether the person is
# talking about THEMSELVES.
#
# The test is deliberately asymmetric, because the two error directions are not
# equally bad. Mistaking a disclosure for a question would drop support from
# someone who needs it. Mistaking a question for a disclosure just reproduces
# today's behaviour. So this fires only on an explicit information request with
# no first-person experience anywhere in the message; anything ambiguous falls
# through to the normal path.
#
# Typo tolerance is not a nicety here. The message that prompted this fix was
# "tel me about ptsd symptoms" -- one dropped letter. People type badly when
# they are upset, and a gate that only fires on correct spelling would have
# missed the exact case it was written for.
_INFO_REQUEST = re.compile(
    r"\b("
    r"what (is|are|causes?|does|do)\b"
    r"|what'?s\b"
    r"|(can you )?tel{1,2} me( more)?( about| what)?\b"
    r"|explain\b"
    r"|define\b"
    r"|(symptoms?|sympto?ms?|signs?|causes?|types?|kinds?|effects?|"
    r"treatments?) (of|for)\b"
    r"|difference between\b"
    r"|how (do|does|can) (you|i|one|someone|people) (know|tell|diagnose)\b"
    r"|is .* a (symptom|sign|type|kind) of\b"
    r"|(info|information) (on|about)\b"
    r")",
    re.IGNORECASE,
)

# First-person experience, not bare pronouns: "how do i know if x is a symptom"
# is still a library question and must not match here.
_FIRST_PERSON_EXPERIENCE = re.compile(
    r"\b(i|we)\s*('m|’m|m|ve|'ve|’ve|d|'d|’d)?\s*"
    r"(feel|felt|feeling|am|was|were|been|have been|having|get|got|getting|"
    r"keep|kept|can'?t|cannot|couldn'?t|struggle|struggling|struggled|"
    r"suffer|suffering|experience|experiencing|think|thought|worry|worried|"
    r"cry|crying|panic|panicking|hurt|hurting|need|want|hate|fear|scared|"
    r"tired|exhausted|numb|lost|alone|anxious|depressed|angry|sad)\b"
    r"|\bmy\s+(anxiety|depression|ptsd|ocd|adhd|bipolar|bpd|autism|mood|"
    r"sleep|panic|trauma|flashbacks?|therapist|meds|medication|head|mind|"
    r"brain|life|partner|mum|mom|dad|family|job|work)\b"
    r"|\b(happened|happens) to me\b"
    r"|\bi\b.*\b(diagnosed|diagnosis)\b",
    re.IGNORECASE,
)


def informational_request(latest: str) -> bool:
    """
    True when this turn is a request for general information rather than the
    person talking about their own experience.

    Callers must run the safety check FIRST. "what are the signs someone is
    suicidal" is an information request by this test, and it is still a
    conversation where crisis routing takes precedence.
    """
    if not latest or not latest.strip():
        return False
    if _FIRST_PERSON_EXPERIENCE.search(latest):
        return False
    return bool(_INFO_REQUEST.search(latest))


@dataclass
class Decision:
    action: str                       # SAFETY | OOD | ASK | RESOLVE | ABSTAIN
    labels: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)
    question: Optional[str] = None
    crisis_patterns: list = field(default_factory=list)
    turn: int = 0
    evidence_tokens: int = 0
    reason: str = ""
    crisis_earlier: bool = False      # fired before, but not on this turn
    already_introduced: bool = False  # category was named on an earlier turn
    situational: list = field(default_factory=list)  # stated external causes
    # True when `question` (if set) is a low-confidence guess the LLM should
    # IGNORE rather than recite -- e.g. a greeting or a message with no real
    # classifier signal. False means it's grounded in a class that cleared or
    # nearly cleared threshold, and should actually shape the reply.
    question_optional: bool = False
    # This turn asked for general information rather than describing anything
    # personal. Answer it; do not turn it into an elicitation opportunity.
    informational: bool = False

    def to_state(self) -> dict:
        """
        The structured state handed to the LLM.

        Note what is NOT here: raw probabilities. If the LLM sees "bipolar:
        0.63" it will reason about 0.63 and quietly become the diagnostician.
        It gets a sufficiency LABEL instead.
        """
        return {
            "classification": {
                "leading_categories": self.labels,
                "evidence_sufficiency": (
                    "sufficient" if self.action == "RESOLVE"
                    else "insufficient" if self.action in ("ASK", "ABSTAIN")
                    else "not_applicable"),
                "already_introduced": self.already_introduced,
                # The new flag: is `question` grounded in something real, or
                # a low-confidence guess the LLM should discard? This is what
                # stops "hi" being answered with a clinical-sounding question.
                "has_signal": not self.question_optional,
            },
            "safety": {
                "crisis_signal": self.action == "SAFETY",
                "crisis_patterns": self.crisis_patterns,
                # Context for tone, NOT a routing instruction. The LLM should
                # stay gentle with someone who said something worrying earlier
                # without re-surfacing resources on every subsequent turn.
                "crisis_earlier_in_conversation": self.crisis_earlier,
            },
            "domain": {"in_domain": self.action != "OOD"},
            # Named external causes -- bereavement, redundancy, divorce. When
            # present the LLM must respond to the EVENT, not to a category.
            "context": {"situational_causes": self.situational},
            "conversation": {"turn": self.turn,
                             "evidence_tokens": self.evidence_tokens,
                             # True -> the person asked for information, not
                             # support. Answer them; do not ask how they feel.
                             "informational_request": self.informational},
            "decision": {"action": self.action, "reason": self.reason},
        }


# Follow-up questions keyed on the two classes closest to firing, so the
# question probes the DISCRIMINATING axis. Selecting on raw top-2 instead made
# anxiety/depression fire in 5 of 6 test conversations, because those two
# dominate by prior in early turns.
QUESTION_BANK = {
    ("bipolar", "depression"):
        "Have there been stretches where you needed a lot less sleep than usual "
        "and felt unusually driven or full of ideas?",
    ("anxiety", "ocd"):
        "Are there particular thoughts that keep coming back, where you feel you "
        "have to do something to settle them?",
    ("anxiety", "depression"):
        "When you notice it most, does it feel more like restlessness and dread, "
        "or more like heaviness and not wanting to do much?",
    ("bipolar", "bpd"):
        "When your mood shifts, does it usually follow something happening with "
        "someone close to you, or does it come on by itself?",
    ("adhd", "autism"):
        "Is it more that focus and follow-through slip away, or more that social "
        "situations and changes in routine feel draining?",
    ("anxiety", "ptsd"):
        "Is there a particular experience your mind keeps returning to, or does "
        "the worry attach itself to whatever is nearby?",
    ("depression", "suicidal"):
        "That sounds really heavy. When it's at its worst, where do your "
        "thoughts tend to go?",
    ("bpd", "depression"):
        "Does the low feeling tend to shift with what is happening between you "
        "and other people, or does it sit there regardless?",
    ("depression", "ptsd"):
        "Did this start around a particular time or event, or has it built up "
        "gradually?",
    ("adhd", "anxiety"):
        "When you can't settle to something, is it more that your attention "
        "slides away, or more that worry gets in the way?",
}

GENERIC_QUESTIONS = [
    "Can you tell me a bit more about what that's been like day to day?",
    "What does it look like when it's at its worst?",
    "How long has this been going on for?",
    "Is there anything that makes it better or worse?",
]


def pick_question(scores: dict, turn: int, asked: Sequence[str] = ()) -> str:
    """
    Choose by which pair is closest to TIED, not by which two score highest.
    A near-tie is the axis a question can actually resolve.

    `asked` is the questions already put to this person. Skip them.

    WHY ROTATING ON `turn` WAS NOT ENOUGH
    -------------------------------------
    The previous version indexed `candidates[(turn - 1) % len(candidates)]` and
    claimed that stopped repeats. It does not. The candidate list is ordered by
    how close each pair is to tied, so it REORDERS between turns as scores move.
    Index 1 on turn 2 can be the same question index 0 returned on turn 1.

    Observed live: a user got "when your mood shifts, does it usually follow
    something happening with someone close to you, or does it come on by
    itself?" twice in a row, having answered it the first time. Asking someone
    the same question after they answered it reads as not listening, which is a
    worse failure here than a slightly less discriminating question.

    So repeats are now excluded explicitly, and `turn` only breaks ties among
    what is left.
    """
    candidates = []

    if scores:
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:4]
        best_pairs = []

        for i in range(len(ranked)):
            for j in range(i + 1, len(ranked)):
                a, b = ranked[i], ranked[j]
                if b[1] < 0.05:          # too weak to be a real competitor
                    continue
                key = tuple(sorted([a[0], b[0]]))
                if key in QUESTION_BANK:
                    best_pairs.append((a[1] - b[1], QUESTION_BANK[key]))

        # Preserve the original behavior: closest tied pair first.
        best_pairs.sort(key=lambda item: item[0])
        candidates.extend(question for _, question in best_pairs)

    # Generic questions remain the fallback, but are also candidates so that
    # a question is not repeated when the same classifier pair remains tied.
    candidates.extend(GENERIC_QUESTIONS)

    already = set(asked or ())
    fresh = [q for q in candidates if q not in already]

    if fresh:
        # `turn` only varies the choice among questions not yet used; it can no
        # longer land back on one that was.
        return fresh[(turn - 1) % len(fresh)]

    # Everything has been asked. MAX_TURNS caps the loop long before this is
    # reachable in practice, but returning a repeat is still better than
    # returning nothing, so fall back to the original rotation.
    if candidates:
        return candidates[(turn - 1) % len(candidates)]

    return GENERIC_QUESTIONS[min(turn, len(GENERIC_QUESTIONS) - 1)]


def decide(accumulated: str, latest: str, turn: int,
           prev_turn: Optional[str] = None,
           prev_fired: bool = False,
           crisis_flagged: bool = False,
           asked: Sequence[str] = ()) -> Decision:
    """
    accumulated    every user message so far, joined. What gets classified.
    latest         this turn's message.
    turn           number of user messages received, including `latest`.
    prev_turn      the message before this one, or None on the first turn.
    prev_fired     whether prev_turn ITSELF triggered the safety route. If it
                   did, it is not re-scanned -- otherwise a single crisis
                   disclosure would fire again on the following turn.
    crisis_flagged whether crisis fired earlier in this conversation. Used for
                   tone only -- it must not re-trigger the safety route.
    asked          follow-up questions already put to this person, so the same
                   one is not asked twice.
    """
    n_tok = token_count(accumulated)

    # 1. SAFETY FIRST, every turn, before anything else. You never ask someone
    #    expressing suicidal intent to elaborate so you can classify them.
    #
    #    Scan `latest` alone, and additionally `prev + latest` when the
    #    previous turn did NOT already fire. That catches an expression split
    #    across a boundary ("i just" / "don't want to be here") without a
    #    single disclosure re-triggering on the message after it.
    #
    #    Scanning the full transcript here was the original bug: once any turn
    #    contained crisis language every later turn re-matched it, and a
    #    question about post office hours came back with crisis resources.
    hits = safety.crisis_check(latest)
    if not hits and prev_turn and not prev_fired:
        spanning = safety.crisis_check(f"{prev_turn} {latest}")
        # only accept it if the phrase genuinely straddles the boundary --
        # i.e. neither message contains it on its own
        if spanning and not safety.crisis_check(prev_turn):
            hits = spanning
    if hits:
        return Decision(action="SAFETY", crisis_patterns=hits, turn=turn,
                        evidence_tokens=n_tok, crisis_earlier=crisis_flagged,
                        reason="crisis language detected by rule")

    # 1b. They asked a question ABOUT something, not about themselves. Answer
    #     it. Classifying a library question produces a category the person
    #     never claimed, and appending a follow-up turns "what are the
    #     symptoms" into "describe your worst moment" -- which is what was
    #     reported. Deliberately after safety and before classification: crisis
    #     routing still wins, and nothing here is scored.
    if informational_request(latest):
        return Decision(action="ASK", question=None, question_optional=True,
                        informational=True, turn=turn, evidence_tokens=n_tok,
                        crisis_earlier=crisis_flagged,
                        reason="information request -- answer it without a "
                               "personal follow-up")

    # 2. No model? Degrade to conversation rather than failing the request.
    if not classifier.available():
        if turn >= MAX_TURNS or n_tok >= AMBIGUOUS_TOKENS:
            return Decision(action="ABSTAIN", turn=turn, evidence_tokens=n_tok,
                            crisis_earlier=crisis_flagged,
                            reason="classifier unavailable")
        return Decision(action="ASK", question=pick_question({}, turn, asked),
                        question_optional=True,   # no model at all -> no signal
                        turn=turn, evidence_tokens=n_tok,
                        crisis_earlier=crisis_flagged,
                        reason="classifier unavailable")

    scores = classifier.score(accumulated)
    hits_labels = classifier.fired(scores)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top = ranked[0][1] if ranked else 0.0
    margin = (ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else top

    # 3. Enough text AND something cleared threshold AND it is not a coin flip.
    #    One more gate first: if the person has named an external cause and the
    #    conversation is still young, do not attach a clinical label to it.
    #    Grief, redundancy and divorce look like depression to a model trained
    #    on subreddit posts, and telling someone their bereavement resembles a
    #    disorder is worse than saying nothing.
    sit = situational.situational_check(accumulated)
    if hits_labels and sit and turn <= SITUATIONAL_GRACE_TURNS:
        return Decision(action="ASK", scores=scores,
                        question=pick_question(scores, turn, asked),
                        turn=turn, evidence_tokens=n_tok,
                        crisis_earlier=crisis_flagged,
                        situational=sit,
                        reason=f"{hits_labels[0]} above threshold but a "
                               f"situational cause is present ({sit[0]})")

    if hits_labels and n_tok >= MIN_TOKENS_TO_RESOLVE and margin >= MARGIN:
        return Decision(action="RESOLVE", labels=hits_labels, scores=scores,
                        turn=turn, evidence_tokens=n_tok,
                        crisis_earlier=crisis_flagged, situational=sit,
                        reason=f"{hits_labels[0]} cleared threshold")

    # 3b. Something DID clear its threshold, but a guard held. Naming which
    #     guard matters: "i feel bad" scores depression 0.41 against a 0.38
    #     threshold on three tokens, and a system that diagnoses from three
    #     words is the failure mode, not a feature. Without this distinction
    #     the log just says "insufficient evidence" and you cannot tell a
    #     confident-but-too-short turn from a genuinely unclear one.
    if hits_labels and n_tok < MIN_TOKENS_TO_RESOLVE:
        held = Decision(action="ASK", scores=scores,
                        question=pick_question(scores, turn, asked),
                        turn=turn, evidence_tokens=n_tok,
                        crisis_earlier=crisis_flagged,
                        reason=f"{hits_labels[0]} above threshold but only "
                               f"{n_tok} tokens (need {MIN_TOKENS_TO_RESOLVE})")
        if turn >= MAX_TURNS:
            held.action = "ABSTAIN"
            held.question = None
            held.reason = f"turn cap; only {n_tok} tokens of evidence"
        return held

    if hits_labels and margin < MARGIN:
        near = sorted(scores.items(), key=lambda kv: -kv[1])[:2]
        return Decision(action="ASK", scores=scores,
                        question=pick_question(scores, turn, asked),
                        turn=turn, evidence_tokens=n_tok,
                        crisis_earlier=crisis_flagged,
                        reason=f"near-tie {near[0][0]}/{near[1][0]} "
                               f"(margin {margin:.2f})")

    # 4. Nothing registers at all, and there is enough text to judge that.
    #    Distinct from ambiguity: this is "not a support query".
    if top < OOD_FLOOR and n_tok >= OOD_MIN_TOKENS and turn >= 2:
        return Decision(action="OOD", scores=scores, turn=turn,
                        evidence_tokens=n_tok, crisis_earlier=crisis_flagged,
                        reason="no category registers above the floor")

    # 5. Plenty of words, still nothing clear. Asking again will not help and
    #    starts to feel like being interviewed.
    if n_tok >= AMBIGUOUS_TOKENS:
        return Decision(action="ABSTAIN", scores=scores, turn=turn,
                        evidence_tokens=n_tok, crisis_earlier=crisis_flagged,
                        reason="sufficient text, no clear category")

    if turn >= MAX_TURNS:
        return Decision(action="ABSTAIN", scores=scores, turn=turn,
                        evidence_tokens=n_tok, crisis_earlier=crisis_flagged,
                        reason="turn cap reached")

    # Nothing crossed any threshold at all. This is the ambiguous zone that
    # catches everything from "i feel bad" (real, just early) to "hi" (no
    # content whatsoever) -- both land here because neither clears a class
    # threshold. SIGNAL_FLOOR is what tells them apart: below it, top is
    # near-uniform noise across all ten classes, which is what a greeting
    # looks like to this model. Above it, there's a real if quiet lean.
    has_signal = bool(ranked) and top >= SIGNAL_FLOOR
    if has_signal and ranked and classifier.THRESHOLDS:
        reason = (f"nothing above threshold (top {ranked[0][0]} "
                  f"{top:.2f}/{classifier.THRESHOLDS.get(ranked[0][0], 0):.2f})")
    elif ranked:
        reason = (f"no real signal (top {ranked[0][0]} {top:.2f} < "
                  f"{SIGNAL_FLOOR} floor) -- likely a greeting or too little "
                  f"to go on; letting the LLM respond naturally")
    else:
        reason = "insufficient evidence"

    return Decision(action="ASK", scores=scores,
                    question=pick_question(scores, turn, asked),
                    question_optional=not has_signal,
                    turn=turn, evidence_tokens=n_tok,
                    crisis_earlier=crisis_flagged,
                    reason=reason)