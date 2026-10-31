"""Rule-based detector with negation and history handling.

Rules are written from general home-care knowledge (what a care
coordinator would look for), not copied from the test notes. Each rule
returns the exact sentence it fired on, so every flag can be checked.
"""
from __future__ import annotations

import re

from .schema import Finding

# (category, severity, pattern)
RULES = [
    # safeguarding
    ("safeguarding", "high", r"\b(bruis\w*|finger ?marks?|marks? on (her|his|the) )"),
    ("safeguarding", "high", r"\b(taking|took|takes|has) (her |his )?(money|pension|bank card|cards?)\b|money (gone|missing)|pension money"),
    ("safeguarding", "high", r"\b(not|doesn'?t|does not|don'?t) feel safe\b|\bfrightened\b|\bscared of\b|\bafraid of\b"),
    ("safeguarding", "high", r"\b(shout\w*|swear\w*|push\w*|hit|hits|slapp\w*|grabb\w*) (at )?(her|him|\[client\])\b"),
    ("safeguarding", "high", r"\block(ed)? in\b|would not let me see|not (let|allow) (me|us) (see|speak)"),
    ("safeguarding", "high", r"\b(no food|nothing to eat|fridge (was |is )?empty|heating (was |is )?off)\b"),
    ("safeguarding", "high", r"\bsoiled\b|\bbed was wet through\b"),
    ("safeguarding", "high", r"\b(hid|hiding|hides|withhold\w*) .{0,25}(tablets|meds|medication)\b|extra .{0,15}tablets"),
    # missed tasks
    ("missed_task", "medium", r"\b(not given|did not give|didn'?t give|unable to (give|administer)|missed (dose|meds))\b"),
    ("missed_task", "medium", r"\b(no time to|ran out of time|did not (prepare|make|help|finish|complete)|not (done|made|completed|changed|applied))\b"),
    ("missed_task", "medium", r"\b(cut short|only stayed|left early)\b"),
    ("missed_task", "low", r"\b(forgot|forgotten) to\b|\bnot signed\b"),
    # deterioration
    ("deterioration", "high", r"\b(fell|fall|found (her |him |\[client\] )?on the floor|slipped)\b"),
    ("deterioration", "high", r"\b(more confused|very confused|muddled|sudden confusion|did not recognise)\b"),
    ("deterioration", "high", r"\b(breathless|short of breath|struggling to breathe|rattly|blue lips|lips (looked )?(a bit )?blue)\b"),
    ("deterioration", "high", r"\b(red (patch|area|mark)|sore|broken skin|skin .{0,15}broken|weeping|pressure (sore|ulcer|damage))\b"),
    ("deterioration", "high", r"\b(struggl\w* to (stand|walk)|legs (very )?weak|mobility (much )?worse|could not (stand|walk))\b"),
    ("deterioration", "medium", r"\b(barely (ate|eaten|drunk|drinking)|not (eating|drinking)|eating (very )?little|only ate|untouched|poor appetite)\b"),
    ("deterioration", "medium", r"\b(urine .{0,30}(dark|cloudy|smell\w*|strong)|stings? when|burning when)\b"),
    # quality risks
    ("quality_risk", "low", r"\b(\d+ (minutes|mins) late|very late|arrived late|running late)\b"),
    ("quality_risk", "medium", r"\b(complain\w*|unhappy (that|with|about))\b"),
    ("quality_risk", "medium", r"\b(hoist|sling|commode|wheelchair|frame|bed rail)\b.{0,30}\b(broken|faulty|frayed|not (working|charging))\b"),
    ("quality_risk", "low", r"\b(trip hazard|loose rug|not working|beeping|no light|very dark)\b"),
    ("quality_risk", "low", r"\b(could not get in|no answer|key ?safe .{0,20}(not|didn'?t) work)\b"),
]
COMPILED = [(c, s, re.compile(p, re.I)) for c, s, p in RULES]

NEGATION = re.compile(r"\b(no|not|nil|never|denies|without|none|no signs? of|no sign of)\b[^,.;]{0,30}$", re.I)
HISTORY = re.compile(r"\b(last (year|month)|years ago|in the past|has now gone|now healed|healing well|previously|when she was)\b", re.I)
# phrases where a "not" belongs to the event itself, so it must not be read as negation
NEG_SAFE = re.compile(r"\b(not given|not done|not made|not changed|not applied|not signed|not feel safe|not let|not working|not charging|not eating|not drinking|not complet|not stand|not walk)", re.I)

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def detect(text: str) -> list[Finding]:
    found: dict[str, Finding] = {}
    for sent in sentences(text):
        if HISTORY.search(sent):
            continue
        for cat, sev, pat in COMPILED:
            m = pat.search(sent)
            if not m:
                continue
            before = sent[: m.start()]
            if NEGATION.search(before) and not NEG_SAFE.search(sent):
                continue
            prev = found.get(cat)
            rank = {"low": 1, "medium": 2, "high": 3}
            if prev is None or rank[sev] > rank[prev.severity]:
                found[cat] = Finding(category=cat, severity=sev, evidence=sent, confidence=0.9, source="rules")
    return list(found.values())
