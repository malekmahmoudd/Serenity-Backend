# Serenity — Crisis Escalation Policy

**Status:** operative. This document is the specification `safety.py`,
`decision.py`, and the frontend crisis UI implement. Code that disagrees with
this document has a bug; update one or the other, never let them silently
diverge.

**Jurisdiction:** Egypt. Numbers and legal framing below are Egypt-specific.
If Serenity launches elsewhere, this document is rewritten first, code second.

---

## 1. What "crisis" means here

A message trips the crisis path when `safety.CRISIS_RE` matches: explicit
suicidal language, expressions of being a burden, loss of will to live,
self-harm references, or a stated plan. The pattern list is deliberately
broad — see §5.

This is **not** a clinical assessment. It is a keyword-and-phrase trigger
tuned for recall over precision, because the cost of a false negative (missing
a real crisis) is categorically worse than the cost of a false positive
(showing resources to someone who didn't need them right now).

## 2. What happens when it fires

1. `decision.py` returns `action = "SAFETY"` and stops. No classification, no
   follow-up question, no attempt to continue the conversation as normal.
2. The reply always includes `safety.CRISIS_RESOURCES` verbatim — the ambulance
   number, the Ministry of Health hotline, Befrienders Cairo. `llm.py` checks
   for `"16328"` in the LLM's output and appends the block if it's missing.
   **The resources are never contingent on the LLM behaving correctly.**
3. If the LLM is unreachable, `FALLBACKS["SAFETY"]` in `llm.py` is used
   instead — a hand-written message with the same resources. The crisis path
   has zero dependency on Groq being up.
4. The turn is logged (`dashboard.record`, `sessions.persist_turn`) with
   `crisis_patterns` recorded, and `crisis_flagged = True` is set on the
   conversation and persists for its lifetime.
5. Rate limiting is bypassed for this message (`ratelimit` is skipped when
   `safety.is_crisis(text)` is true) — see §5 for the trade this makes.

## 3. What does **not** happen

This is the part that has to be said plainly, because it's easy to assume
otherwise:

- **Nobody is notified.** No email, no SMS, no alert to a human operator, no
  emergency services contact initiated by the system. The consent modal and
  privacy page both say "nobody is monitoring these conversations" — that
  claim must stay true, or it must be changed and the user re-consented.
- **No account action is taken.** The user is not logged out, blocked, or
  flagged to any third party.
- **No law-enforcement contact.** Serenity does not, and currently has no
  mechanism to, contact Egyptian police or emergency services on a user's
  behalf.

If this changes — if you add human review, automated escalation, or mandatory
reporting — **the consent modal must change in the same release.** Silently
adding monitoring after telling users there is none is the single worst thing
this document can be used to justify.

## 4. Who reviews crisis events, and how often

**As of this writing: nobody, on no schedule.** This is a genuine gap, not an
oversight being described politely. Recorded in `db.stats()["crisis_conversations"]`
and visible on `/dashboard`, but nobody is contractually or operationally
obligated to look.

**Minimum viable review process**, to be implemented before this product has
real users beyond testing:

- A named person (you, until there's a team) checks the crisis count on
  `/dashboard` **daily**.
- Weekly, read a sample of `top_errors`-equivalent crisis transcripts (once
  decrypted with proper authorization — see §6) to check the rule is still
  firing appropriately and the LLM's tone is landing right.
- Any month where crisis volume triples week-over-week gets a manual look —
  that's a signal something changed, either in the user base or in a rule
  regression.

This section should be rewritten with real names and real cadences before
launch. A policy with "nobody" in it is a placeholder, not a policy.

## 5. Known trade-offs, stated rather than hidden

**Recall over precision.** The pattern list will produce false positives —
someone quoting a movie line, discussing the topic academically, or using dark
humor may trip it. The cost of that (an unwanted but harmless supportive
message) is accepted in exchange for catching real disclosures phrased in
ways a narrower rule would miss. This was tested explicitly: the phrase
"nobody would really notice if I wasn't around" broke an earlier, tighter
version of the rule.

**Rate-limit bypass is exploitable.** Including crisis language defeats the
`/chat` rate limiter by design (§2.5). Someone could abuse this to send
unlimited messages by including trigger words in each one. Accepted trade:
the alternative is rate-limiting someone mid-disclosure, which is worse.

**The classifier is not the safety net.** On the frozen evaluation set the
LSTM's own `suicidal` class scored 0.000 recall on conversational-register
text — it essentially never fires on natural phrasing. `safety.py`'s
rule-based check is the entire safety mechanism. This is intentional
architecture, not a fallback: rules are auditable, the classifier is not, and
recall matters more than adaptability here. Do not "improve" this by routing
crisis detection through the classifier without re-deriving this entire
document.

**No geographic detection.** Every user gets Egypt-specific numbers regardless
of where they're actually located. If Serenity's user base becomes
international, this becomes actively harmful (a US user being told to call
123, which is not a US emergency number) and must be fixed before it matters,
not after.

## 6. Data handling for crisis conversations

Crisis-flagged conversations are stored identically to any other conversation:
encrypted at rest (`crypto.py`), 90-day retention (`db.RETENTION_DAYS`), same
access controls. They are **not** held longer, reviewed automatically, or
treated as a special data category beyond the `crisis_flagged` boolean and
`crisis_patterns` string used for the dashboard.

Decrypting a crisis conversation for review (§4) requires the same
`SERENITY_ENC_KEY` as any other data access. There is currently no
audit log of *who* decrypted *what* — that's a gap worth closing if review
becomes a routine practice rather than an ad hoc one.

## 7. Legal context (informational, not legal advice)

Egypt does not currently have telehealth-specific legislation directly
governing AI mental-health chat tools. This product does not claim to provide
medical or psychiatric treatment (see the consent modal and privacy page,
which state this explicitly), and the crisis path routes to the Ministry of
Health's official channel (16328) rather than attempting to provide crisis
counseling itself.

**This section is not a substitute for actual legal review**, and one should
happen before this product is available to the public, not after.

## 8. Change log

| date | change |
|---|---|
| 2026-08-17 | Initial policy written, alongside `safety.py` v2 (natural-register crisis patterns) |

Any change to `safety.CRISIS_PATTERNS`, `safety.CRISIS_RESOURCES`, or the
`SAFETY` branch in `decision.py` should add a row here.
