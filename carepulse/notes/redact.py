"""Remove personal identifiers from a note before any model sees it.

Names known from the care record are swapped for [CLIENT] / [CARER], and
phone numbers, NHS numbers, emails and postcodes are masked. This is data
minimisation: the model only needs the events, not who they happened to.
"""
from __future__ import annotations

import re

PATTERNS = [
    ("[NHS_NUMBER]", re.compile(r"\b\d{3}[ -]?\d{3}[ -]?\d{4}\b")),
    ("[PHONE]", re.compile(r"(?:\+44\s?|\b0)\d{2,4}[ ]?\d{3,4}[ ]?\d{3,4}\b")),
    ("[EMAIL]", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("[POSTCODE]", re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b", re.I)),
]


def redact(text: str, names: dict[str, str] | None = None) -> tuple[str, list[str]]:
    """Return the redacted text and the list of placeholder types used.

    `names` maps a real name to its placeholder, e.g. {"Mrs Begum": "[CLIENT]"}.
    Phone numbers are matched before NHS numbers would be, by order of checks
    on the masked text, so a phone is never left half masked.
    """
    used = []
    out = text
    for name, tag in sorted((names or {}).items(), key=lambda kv: -len(kv[0])):
        pat = re.compile(re.escape(name), re.I)
        if pat.search(out):
            out = pat.sub(tag, out)
            used.append(tag)
    for tag, pat in [PATTERNS[1], PATTERNS[0], PATTERNS[2], PATTERNS[3]]:
        if pat.search(out):
            out = pat.sub(tag, out)
            used.append(tag)
    return out, used
