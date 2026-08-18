"""
safety.py
=========
Crisis detection. Rule-based, and deliberately independent of the classifier.

WHY THIS IS NOT THE MODEL'S JOB
-------------------------------
On the frozen conversational evaluation set the classifier's crisis recall was
0.000 -- it never fired `suicidal` on a single conversational example. The rule
below scored 1.000 on the same set. That is not an accident of tuning: the
training corpus is people who already use clinical vocabulary, and real users
write "everyone would be better off".

So crisis routing runs on the rule, checked on every turn, before any
classification happens. The classifier's suicidal score is used only to
*escalate*, never to gate.

The patterns are deliberately over-broad. A false positive costs one
unnecessary message with support resources attached. A false negative costs
someone in crisis being handed generic advice about managing their anxiety.
Those are not symmetric, so this is not tuned for balance.

Keep this file under review against real logs. A hand-written pattern list is
brittle to phrasing nobody thought of -- two rows got past an earlier version
because it required "nobody would MISS me" and the user wrote "nobody would
really notice".
"""

from __future__ import annotations

import re

CRISIS_PATTERNS = {
    "explicit":
        r"kill(ing)?\s+(my\s?self|meself)|end(ing)?\s+(my\s+life|it\s+all)"
        r"|take\s+my\s+own\s+life|commit\s+suicide|suicid(e|al)"
        r"|\bkms\b|\bktms\b|unalive",

    "burden":
        r"better\s+off\s+(with\s?out\s+me|dead|if\s+i\s+(was|were)\s+(gone|dead))"
        r"|everyone\s+would\s+be\s+better|(they|world)\s+would\s+be\s+better"
        r"|burden\s+(to|on)\s+(everyone|them|my)"
        # adverbs may sit between "would" and the verb
        r"|(no\s?one|nobody)\s+would\s+(\w+\s+){0,2}(miss|notice|care)"
        r"|if\s+i\s+(was|were)\s+(gone|not\s+here|dead)"
        r"|(carry\s+on|go\s+on|manage|be\s+fine|do\s+fine)\s+(just\s+)?fine\s+without\s+me"
        r"|(carry\s+on|go\s+on|manage)\s+without\s+me"
        r"|they'?d\s+(manage|be\s+fine|cope)",

    "no_point":
        r"(no|what'?s\s+the)\s+point\s+(in|to|of)?\s*(it|anything|living|life|going\s+on|trying|anymore)"
        r"|don'?t\s+see\s+the\s+point|nothing\s+(left|to\s+live\s+for)"
        r"|no\s+(reason|future)\s+(to\s+live|left)|why\s+(bother|keep\s+going)",

    "not_here":
        r"(don'?t|do\s+not)\s+want\s+to\s+(be\s+here|exist|live|wake\s+up|be\s+alive)"
        r"|wish\s+i\s+(was|were)\s+(dead|gone|never\s+born)"
        r"|want\s+to\s+(die|disappear|not\s+exist)|ready\s+to\s+(go|give\s+up)"
        r"|hope\s+i\s+don'?t\s+wake",

    "exhausted":
        r"can'?t\s+(do\s+this|take\s+it|keep\s+going|go\s+on)"
        r"(\s+(anymore|any\s+longer|like\s+this))?\s*(?=[.,!?;]|$)"
        r"|tired\s+of\s+(living|being\s+alive|existing|fighting)"
        r"|giving?\s+up\s+on\s+(life|everything)|at\s+the\s+end\s+of\s+my\s+rope",

    "self_harm":
        r"hurt(ing)?\s+my\s?self|self[\s-]?harm|cutting\s+my\s?self"
        r"|\bod\b\s+on|overdos(e|ing)",

    "plan":
        r"(wrote|writing|written)\s+(a\s+)?(note|letter)\s+to"
        r"|saying\s+good\s?bye|last\s+(message|post|night\s+alive)"
        r"|have\s+a\s+plan|made\s+my\s+decision",
}

CRISIS_RE = re.compile("|".join(f"(?:{p})" for p in CRISIS_PATTERNS.values()),
                       re.IGNORECASE)

# Shown when the crisis path fires.
#
# Egypt-specific, because generic international numbers are useless to someone
# in Cairo at 3am. 16328 is the Ministry of Health's General Secretariat of
# Mental Health and Addiction Treatment hotline -- 24/7, free, confidential,
# and it handles emergency referrals rather than only giving advice.
#
# If you launch outside Egypt, this MUST change. A crisis message pointing at a
# number that does not work in the user's country is worse than no number.
# ASCII only, deliberately. These strings pass through consoles, logs, SMS
# gateways and JSON encoders of varying quality; a bullet character that
# renders as a replacement glyph in one of them is a needless risk on the one
# message that has to be readable.
CRISIS_RESOURCES = (
    "If you are in immediate danger, call 123 for an ambulance.\n\n"
    "You can also talk to someone right now:\n"
    "  - Mental health support (24/7, free): 16328\n"
    "  - Ambulance: 123   |   Police: 122\n"
    "  - Befrienders Cairo: 762 1602 or 762 2381\n"
)


def crisis_check(text: str) -> list[str]:
    """Return the names of every pattern that fired. Empty list = clear."""
    t = str(text)
    return [name for name, pat in CRISIS_PATTERNS.items()
            if re.search(pat, t, re.IGNORECASE)]


def is_crisis(text: str) -> bool:
    return bool(CRISIS_RE.search(str(text)))
