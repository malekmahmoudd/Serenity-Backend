"""
llm.py
======
The communication layer. Turns a structured decision state into something a
person wants to read.

WHAT THE LLM IS NOT ALLOWED TO DO
---------------------------------
It does not classify. It does not override the safety signal. It does not see
raw probabilities -- if it saw "bipolar: 0.63" it would reason about 0.63 and
quietly become the diagnostician, which is exactly the architecture this
project exists to avoid.

It receives leading categories, a sufficiency LABEL, a decision, and the
conversation. It writes prose.

The system prompt below is the contract. If you loosen it, the specialist
models stop being the thing that decides.

FALLBACK
--------
If no API key is configured, or the call fails, every action still has a
hand-written response. The product works without the LLM -- less warmly, but it
works. That matters more than it sounds: the crisis path in particular must
never depend on a third-party API being up.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import safety


def _get_api_key() -> str:
    """
    Read fresh every call, never cached.

    Previously this file also had `API_KEY = os.environ.get(...)` evaluated
    once at import time, and every check in respond()/available() used THAT
    frozen constant instead of this function. `.env` is loaded by main.py --
    see the comment there -- and if that load happens even one import after
    this module's top level runs, API_KEY froze on an empty string forever,
    for the entire lifetime of the process, no matter what the key actually
    was afterward. That was the whole bug: `.env` had the key, `_call()` used
    the fresh read and worked, but every gate around it used the stale one and
    fell through to templates regardless.
    """
    return os.environ.get("GROQ_API_KEY", "")


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Groq deprecated llama-3.3-70b-versatile on 2026-06-17 (free/developer tier).
# This is what the 404s were: not an auth problem, not a dotenv problem --
# the model name itself stopped existing mid-project. openai/gpt-oss-120b is
# Groq's own recommended replacement.
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Content-Type": "application/json",
}

# Set by respond() on every call: True if the reply came from the API, False if
# it fell back to a hand-written template. The /dashboard turn log reads this,
# because you cannot tell the difference from the chat bubble alone.
last_call_used_api = False

SYSTEM = """You are Serenity, a supportive listener in a mental-health support \
app. You are NOT a clinician.

BE BRIEF. 2-3 sentences normally. Never more than 5, and only on RESOLVE. \
People in distress do not want paragraphs. Warm and short beats warm and long.

HARD RULES
1. Never state a diagnosis as fact -- no "you have X", no "this is X". The \
classifier behind you is right roughly half the time on conversational input.
2. Never contradict the safety signal. If crisis_signal is true: acknowledge, \
express care, give the resources. Nothing else -- no advice, no questions, no \
categories.
3. If conversation.informational_request is true, they asked you ABOUT \
something rather than about themselves. Answer the question plainly and \
usefully in 2-4 sentences. Do NOT ask how they feel, do NOT ask them to \
describe their own experience, and do NOT end with a follow-up question of \
any kind. If they want to talk about themselves they will; wait to be invited. \
You may close by noting you're here if they do want to talk, but only as a \
statement, never as a question. This rule outranks rule 4 below.
4. If action is ASK and informational_request is false: check \
classification.has_signal. If true, lead in with one short sentence responding \
to what they actually said, then ask the supplied question (verbatim or a \
light natural rewording -- keep its meaning intact). If false, ignore the \
supplied question completely -- this is usually a greeting or too little to go \
on, so just respond naturally to what they wrote and ask something short, \
warm, and open instead. Never ask a clinical-sounding question (mood, sleep, \
"what does it look like at its worst") when there is nothing to go on yet.
5. If evidence_sufficiency is "insufficient", mention no category at all.
6. If context.situational_causes is non-empty, the person has named a real \
event -- a death, a job loss, a breakup. Respond to THAT. Never suggest their \
reaction to it might be a disorder. Grief is not a symptom.
7. Plain warm language. No jargon, no lists, no headings, no emoji.
8. If in_domain is false, say you are here for emotional support and ask what \
is on their mind.

WHEN action IS "RESOLVE" (max 5 sentences)
  a) One sentence reflecting what they described.
  b) Say the pattern often goes along with <category> -- hedged, never as \
fact. Add that you cannot know, and a GP or counsellor could.
  c) Ask if they would like a few practical things that often help. Ask only; \
do not list them yet.

IF THEY ACCEPT
3-4 small concrete suggestions in prose, not a list. Things doable this week \
without money or a referral. Specific -- "ten minutes outside before you look \
at your phone", not "practise self-care". One short closing line that these \
help people cope but are not a substitute for someone qualified.

IF THEY DECLINE
Accept it in one sentence. Stay available.

You are a supportive space, not a treatment."""

FALLBACKS = {
    "ASK": "Thank you for telling me that. {question}",
    # Used only when the API is unreachable. Hedged the same way the system
    # prompt asks for, because the fallback is what a user sees on a bad day.
    "RESOLVE": ("Thank you for sharing that with me — it sounds like a lot to be "
                "carrying. Some of what you've described overlaps with what "
                "people often call {label}, though I really can't know that, and "
                "someone qualified like a GP or a counsellor is the person who "
                "could actually tell you. Would it help if I shared a few "
                "practical things people in a similar position often find useful?"),
    "ABSTAIN": ("Thank you for telling me. I'm not sure I have the full picture "
                "yet, but I don't want that to stop you talking. What's been "
                "weighing on you most?"),
    "OOD": ("I'm here for emotional support rather than general questions — but "
            "if something's on your mind, I'm listening."),
    # Information was asked for and the API that would answer it is down.
    # Notably this does NOT ask them anything back.
    "INFO_UNAVAILABLE": (
        "I can't look that up properly just now — something on my end isn't "
        "responding. For anything about symptoms or conditions, a GP or a site "
        "like the NHS or Mind will be more reliable than me anyway. I'm here "
        "if you want to talk."),
    "SAFETY": ("Thank you for telling me. What you're describing sounds "
               "genuinely painful, and you shouldn't have to sit with it "
               "alone.\n\n" + safety.CRISIS_RESOURCES),
}

# Plain-English names for the ten classes. The raw label is a subreddit slug;
# saying "bpd" at someone is worse than saying nothing.
LABEL_PHRASES = {
    "adhd": "ADHD",
    "anxiety": "anxiety",
    "autism": "autism",
    "bipolar": "bipolar disorder",
    "bpd": "borderline personality disorder",
    "depression": "depression",
    "ocd": "OCD",
    "ptsd": "PTSD",
    "schizophrenia": "psychosis or schizophrenia",
    # Never surfaced: a `suicidal` RESOLVE is pre-empted by the crisis rule,
    # which routes to SAFETY before classification is consulted. Present only
    # so the lookup never falls through to a raw slug.
    "suicidal": "what you're going through",
}


def _phrase(labels: list) -> str:
    """Human-readable category name, or a neutral phrase if we have none."""
    if not labels:
        return "what you're going through"
    names = [LABEL_PHRASES.get(l, l) for l in labels[:2]]
    return names[0] if len(names) == 1 else f"{names[0]} or {names[1]}"


def _fallback(state: dict, question: str | None) -> str:
    action = state["decision"]["action"]
    # Without the API there is nothing that can actually answer an information
    # request, so say that plainly instead of substituting a personal question
    # -- the template would otherwise reply to "what are the symptoms of ptsd"
    # with "thank you for telling me that, what's been going on?".
    if action == "ASK" and (state.get("conversation") or {}).get(
            "informational_request", False):
        return FALLBACKS["INFO_UNAVAILABLE"]
    text = FALLBACKS.get(action, FALLBACKS["ABSTAIN"])
    if action == "ASK":
        text = text.format(question=question or
                           "Can you tell me a bit more about what's going on?")
    elif action == "RESOLVE":
        labels = state["classification"].get("leading_categories") or []
        text = text.format(label=_phrase(labels))
    return text


class TruncatedReply(RuntimeError):
    """
    The model hit the completion ceiling before finishing, or returned nothing.

    Raised rather than returned so respond()'s existing except-branch sends the
    hand-written template instead. A reply that stops mid-sentence is worse than
    a template, and this is the one place a user is most likely to be reading
    carefully -- the turn where they said yes to help.
    """


def _call(system: str, user: str, timeout: int = 30) -> str:
    body = json.dumps({
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": 400,
        "temperature": 0.7,
        # gpt-oss bills reasoning tokens as completion tokens. At the default
        # effort the longest reply in a conversation -- the practical-tips turn
        # -- spent the entire 400-token ceiling reasoning and was delivered cut
        # off at "keeping a simple sleep routine by". Observed live, not
        # theorised.
        #
        # This layer is only phrasing a decision the deterministic layer already
        # made. It has nothing to reason about, so "low" costs nothing in
        # quality and returns the budget to the actual sentence. It also roughly
        # halves output spend, which bills at 5x the input rate.
        "reasoning_effort": "low",
    }).encode()
    h = dict(HEADERS)
    h["Authorization"] = f"Bearer {_get_api_key()}"
    req = urllib.request.Request(GROQ_URL, data=body, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())

    choice = payload["choices"][0]
    text = (choice.get("message", {}).get("content") or "").strip()
    usage = payload.get("usage", {})

    # Nothing downstream inspected finish_reason before, so a truncated reply
    # was passed to the user verbatim, half-sentence and all.
    if choice.get("finish_reason") == "length":
        raise TruncatedReply(
            f"hit max_tokens (completion={usage.get('completion_tokens')})")
    if not text:
        raise TruncatedReply(
            f"empty content (completion={usage.get('completion_tokens')})")
    return text


def respond(state: dict, history: list, question: str | None = None) -> str:
    """
    state    -- Decision.to_state(), no raw probabilities
    history  -- [{"role": "user"|"assistant", "content": str}, ...]
    """
    global last_call_used_api
    last_call_used_api = False
    action = state["decision"]["action"]

    # The crisis path never depends on a third-party API being reachable.
    # Compose it locally, then let the LLM add warmth only if it is available.
    key = _get_api_key()
    if action == "SAFETY" and not key:
        print("    LLM skipped (no key) -> crisis template", flush=True)
        return FALLBACKS["SAFETY"]

    if not key:
        print("    LLM skipped (no GROQ_API_KEY set) -> template", flush=True)
        return _fallback(state, question)

    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-8:])
    parts = [f"STRUCTURED STATE\n{json.dumps(state, indent=2)}",
             f"\nCONVERSATION SO FAR\n{convo}"]
    if action == "ASK":
        # This is the actual fix for the "hi" bug: only hand the LLM a
        # discriminating question to recite when has_signal says it's grounded
        # in something real. Otherwise the model was reciting a clinical-
        # sounding fallback question onto plain small talk.
        informational = (state.get("conversation") or {}).get(
            "informational_request", False)
        has_signal = state["classification"].get("has_signal", True)
        if informational:
            # They asked about a topic, not about themselves. Answering and
            # then probing is what turned "tell me about ptsd symptoms" into
            # "what does it look like at its worst".
            parts.append(
                "\nTHIS IS AN INFORMATION REQUEST, NOT A DISCLOSURE. Answer "
                "the question they actually asked, plainly and accurately, in "
                "2-4 sentences. Do NOT ask how they feel. Do NOT ask them to "
                "describe their own experience. Do NOT end with a question of "
                "any kind. You may end with a plain statement that you're "
                "here if they want to talk, but it must not be phrased as a "
                "question. Mention no category as applying to them -- they "
                "have not claimed anything about themselves.")
        elif has_signal and question:
            parts.append(
                "\nEND YOUR REPLY WITH A QUESTION BASED ON THIS -- verbatim is "
                "fine, or a light natural rewording, but keep its meaning "
                f"intact:\n{question}")
        else:
            parts.append(
                "\nNo classifier signal yet -- this is likely a greeting, "
                "small talk, or just too brief to tell anything from. Respond "
                "naturally and specifically to what they actually wrote. Ask a "
                "short, warm, open question inviting them to share more, in "
                "your own words. Do NOT ask a clinical-sounding question "
                "(mood, sleep, 'what does it look like at its worst', etc.) "
                "when there is nothing to go on yet.")
    if action == "SAFETY":
        parts.append(f"\nINCLUDE THESE RESOURCES VERBATIM:\n{safety.CRISIS_RESOURCES}")
    if action == "RESOLVE":
        labels = state["classification"].get("leading_categories") or []
        parts.append(f"\nCATEGORY TO MENTION (hedged, never as fact):\n{_phrase(labels)}")
        if state["classification"].get("already_introduced"):
            parts.append(
                "\nYou have ALREADY named this category earlier in the "
                "conversation. Do not reintroduce it and do not repeat the "
                "offer of tips. Respond to what they just said. If they are "
                "accepting the offer of tips, give the tips now.")

    sit = (state.get("context") or {}).get("situational_causes") or []
    if sit:
        parts.append(
            f"\nSITUATIONAL CAUSE NAMED BY THE USER: {', '.join(sit)}\n"
            "Respond to the event itself. Do NOT mention any mental-health "
            "category, and do not imply their reaction might be a disorder.")
    parts.append("\nWrite your reply now. Prose only, and keep it short.")

    try:
        out = _call(SYSTEM, "\n".join(parts))
        last_call_used_api = True
        print(f"    LLM ok via {LLM_MODEL} ({len(out)} chars)", flush=True)
        # Belt and braces: if the model dropped the resources on a crisis turn,
        # append them. This is not a path where we accept a model's judgement.
        if action == "SAFETY" and "16328" not in out:
            print("    LLM omitted resources on a SAFETY turn -> appended",
                  flush=True)
            out = out + "\n\n" + safety.CRISIS_RESOURCES
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"    LLM FAILED -> template. {type(exc).__name__}: "
              f"{str(exc)[:200]}", flush=True)
        return _fallback(state, question)


def available() -> bool:
    return bool(_get_api_key())
