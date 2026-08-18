"""
text_cleaning.py
================
THE single source of truth for text normalisation.

This is a verbatim copy of the cleaning used when the model was trained.
Do not "improve" it here. If serving cleans text even slightly differently
from training, the model degrades silently -- no exception, no log line, just
worse predictions. That is why CLEANING_VERSION is checked against the value
recorded in artifacts/preprocessing_config.json at startup.

If you change anything in this file, you must retrain.
"""

from __future__ import annotations

import html
import re
import unicodedata

CLEANING_VERSION = "1.0.0"

TOMBSTONES = {"[deleted]", "[removed]", "[deleted by user]", "deleted", "removed"}
CONTENTLESS = {".", "?", "!", "^", ":(", ":)", "-", "_", "n/a", "na", "none",
               "title", "help", "nan", "null", ""}

BOILERPLATE_RE = re.compile(
    r"weekly\s+(thread|discussion|check.?in|post)"
    r"|this\s+is\s+a\s+place\s+for\s+off\s+topic"
    r"|welcome\s+to\s+our\s+(weekly|history|book)"
    r"|\*\*dae\s*=|automoderator|i\s+am\s+a\s+bot"
    r"|rules\s+regarding\s+|megathread"
    r"|bookclub\s+(and\s+sources\s+)?wednesday", re.IGNORECASE)

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
SUBREDDIT_RE = re.compile(r"(?:^|(?<=[\s(\[]))/?r/[A-Za-z0-9_]+", re.IGNORECASE)
USER_RE = re.compile(r"(?:^|(?<=[\s(\[]))/?u/[A-Za-z0-9_\-]+", re.IGNORECASE)
MD_QUOTE_RE = re.compile(r"^\s*&?gt;+\s*|^\s*>+\s*", re.MULTILINE)
MD_HEADER_RE = re.compile(r"^\s*#{1,6}\s*", re.MULTILINE)
MD_EMPH_RE = re.compile(r"(\*{1,3}|_{1,3})(?=\S)(.*?)(?<=\S)\1", re.DOTALL)
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
MD_RULE_RE = re.compile(r"^\s*([-*_]\s*){3,}$", re.MULTILINE)
MD_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2028\u2029\ufeff\u00ad]")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
NUM_RE = re.compile(r"\b\d[\d,.]*\b")
REPEAT_RE = re.compile(r"(.)\1{2,}")
WS_RE = re.compile(r"\s+")

# Used only for the leakage ablation, kept so the module matches training.
DIAGNOSTIC_RE = re.compile(
    r"\b(bpd|borderline(\s+personality(\s+disorder)?)?"
    r"|adhd|adderall|vyvanse|concerta|ritalin|elvanse"
    r"|ocd|obsessive[\s-]?compulsive|compulsions?"
    r"|ptsd|cptsd|post[\s-]?traumatic"
    r"|autis(m|tic)|asperger'?s?|aspie|neurodiverg(ent|ence)|neurotypical"
    r"|bipolar|mania|manic|hypomani(a|c)"
    r"|schizophreni(a|c)|schizoaffective|psychosis|psychotic|antipsychotics?"
    r"|anxiet(y|ies)|anxious|panic\s+(attack|disorder)s?"
    r"|depress(ion|ed|ive)|mdd|dysthymi(a|c)"
    r"|suicid(e|al)|kill\s+myself|end\s+my\s+life|kms)\b", re.IGNORECASE)
DX_TOKEN = " dxterm "


def is_junk(raw: str) -> bool:
    if raw is None:
        return True
    s = str(raw).strip()
    if not s:
        return True
    if s.lower() in TOMBSTONES or s.lower() in CONTENTLESS:
        return True
    if len(s) < 3:
        return True
    if BOILERPLATE_RE.search(s):
        return True
    return False


def clean_text(raw: str, lowercase: bool = True,
               mask_diagnostics: bool = False,
               collapse_numbers: bool = True) -> str:
    """
    Order matters:
      1. unescape HTML  -- before entity-sensitive regexes
      2. NFKC normalise -- folds smart quotes so "don't" and "don't" are ONE
                           token rather than two
      3. strip markup   -- URLs, r/subreddit, markdown
      4. collapse       -- numbers, repeated chars, whitespace
    """
    if raw is None:
        return ""
    s = html.unescape(html.unescape(str(raw)))
    s = unicodedata.normalize("NFKC", s)
    s = ZERO_WIDTH_RE.sub("", s)
    s = CTRL_RE.sub(" ", s)
    s = (s.replace("\u2018", "'").replace("\u2019", "'")
           .replace("\u201c", '"').replace("\u201d", '"')
           .replace("\u2013", "-").replace("\u2014", "-").replace("\u2026", "..."))
    s = MD_LINK_RE.sub(r"\1", s)
    s = URL_RE.sub(" ", s)
    s = SUBREDDIT_RE.sub(" ", s)
    s = USER_RE.sub(" ", s)
    s = MD_QUOTE_RE.sub(" ", s)
    s = MD_HEADER_RE.sub(" ", s)
    s = MD_RULE_RE.sub(" ", s)
    s = MD_BULLET_RE.sub(" ", s)
    s = MD_EMPH_RE.sub(r"\2", s)
    if lowercase:
        s = s.lower()
    if mask_diagnostics:
        s = DIAGNOSTIC_RE.sub(DX_TOKEN, s)
    if collapse_numbers:
        s = NUM_RE.sub(" num ", s)
    s = REPEAT_RE.sub(r"\1\1", s)
    return WS_RE.sub(" ", s).strip()


def token_count(text: str) -> int:
    return len(str(text).split())
