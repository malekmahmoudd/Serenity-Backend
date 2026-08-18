"""
situational.py
==============
Detects distress that has an obvious external cause: bereavement, job loss,
divorce, eviction, a serious diagnosis, a breakup.

WHY THIS EXISTS
---------------
The classifier has ten classes and no concept of ordinary human suffering.
Grief looks like depression in text -- low mood, loss of interest, withdrawal --
because clinically it overlaps. So the model happily returns depression 0.55 for

    "my dad died in march and i still can't listen to the music he liked"

which is a normal reaction to a real event, and telling that person it overlaps
with depression is both wrong and unkind.

This was measured, not guessed: on the frozen Phase 6 evaluation set the
`distress-but-not-target` stratum resolved to a clinical label 48.5% of the
time. Those 33 rows exist specifically to catch this.

WHY A RULE AND NOT A MODEL
--------------------------
The signal is not in the score -- raising depression's threshold to 0.56 fixes
this one example and breaks genuine cases. What distinguishes grief from
depression is the presence of a *stated cause*, which is a surface feature a
regex reads perfectly well and a bag-of-tokens classifier ignores entirely.

WHAT IT DOES NOT DO
-------------------
It does not suppress support, and it does not stop the crisis rule -- safety is
checked first and is unaffected. It only stops the system from attaching a
clinical label to a situation that has an obvious explanation. The user still
gets a warm reply and a follow-up question.

It also does not claim these people are fine. Bereavement and depression can
coexist, and someone whose grief has become something heavier should still get
there -- via more conversation, not via a label on turn one.
"""

from __future__ import annotations

import re

SITUATIONAL_PATTERNS = {
    "bereavement":
        r"\b(died|passed\s+away|passing|funeral|buried|cremat)"
        # "my grandmother passed last month" -- `passed` on its own, without
        # "away". Common phrasing and it was missed by the first version.
        r"|\b(grand(ma|pa|mother|father)|nan|nana|granny|dad|mum|mom|father"
        r"|mother|son|daughter|wife|husband|partner|brother|sister|uncle|aunt)"
        r"\s+(passed|died|is\s+gone)\b"
        r"|\blost\s+(my|our)\s+(dad|mum|mom|father|mother|son|daughter|wife"
        r"|husband|partner|brother|sister|friend|grandma|grandad|grandmother"
        r"|grandfather|nan|baby|dog|cat|pet)"
        r"|\bput\s+(down|to\s+sleep)\b"
        r"|\bmiscarriage|\blost\s+the\s+baby\b",

    "job_money":
        r"\b(made\s+redundant|redundancy|laid\s+off|got\s+fired|lost\s+my\s+job"
        r"|out\s+of\s+work|unemployed\s+for)"
        # "my landlord is evicting us" -- the verb form, not just the noun.
        r"|\b(evict(ed|ing|ion)?|repossess|can'?t\s+(afford|cover)\s+(the\s+)?rent)"
        r"|\bnowhere\s+to\s+(go|live)\b"
        r"|\bin\s+debt\s+i\s+can'?t|\bbankrupt",

    "relationship":
        r"\b(divorce|divorcing|separat(ed|ing)\s+from)"
        r"|\b(broke\s+up|breakup|break-up|dumped\s+me|left\s+me\s+for)"
        r"|\basked\s+for\s+a\s+divorce"
        r"|\bcheated\s+on\s+me",

    "health_event":
        r"\b(diagnosed\s+with\s+(cancer|ms|als|parkinson|dementia|alzheimer))"
        r"|\b(cancer|terminal|palliative|hospice)\b"
        r"|\b(in\s+hospital|had\s+a\s+stroke|heart\s+attack)"
        r"|\bcaring\s+for\s+my\s+(mum|mom|dad|father|mother|husband|wife)",

    "upheaval":
        r"\b(moved\s+(country|abroad|away)|had\s+to\s+leave\s+the\s+country)"
        r"|\bvisa\s+(was\s+)?(refused|denied|expired)"
        r"|\b(house|flat|home)\s+(flooded|burned|burnt)"
        r"|\bfailed\s+my\s+(final|exam|degree|year)",
}

SITUATIONAL_RE = re.compile(
    "|".join(f"(?:{p})" for p in SITUATIONAL_PATTERNS.values()), re.IGNORECASE)


def situational_check(text: str) -> list[str]:
    """Return the names of every category that fired. Empty list = none."""
    t = str(text)
    return [name for name, pat in SITUATIONAL_PATTERNS.items()
            if re.search(pat, t, re.IGNORECASE)]


def is_situational(text: str) -> bool:
    return bool(SITUATIONAL_RE.search(str(text)))
