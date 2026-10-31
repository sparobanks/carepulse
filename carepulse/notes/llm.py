"""Open-source LLM backend, run locally through Ollama.

The model runs on the provider's own machine or private cloud, so notes
never go to an outside AI service. Notes are redacted before they reach the
model anyway. The model must return JSON, and every finding must quote
words that really appear in the note. Anything that fails these checks is
dropped and logged, never shown as a finding.
"""
from __future__ import annotations

import json
import urllib.request

from .schema import CATEGORIES, DEFINITIONS, SEVERITIES, Finding

PROMPT = """You review home-care visit notes for a care provider in England.
Read the note and list any of these concerns:
{definitions}

Rules:
- Only report something the note actually says. Do not guess.
- Ignore things that are negated ("no bruising seen") or in the past and resolved ("old bruise has gone").
- For each concern give "evidence": a short quote copied exactly from the note.
- severity is one of: low, medium, high. Possible abuse or neglect is always high.
- If there are no concerns, return an empty list.

Answer with JSON only, in this shape:
{{"findings": [{{"category": "...", "severity": "...", "evidence": "...", "confidence": 0.0}}]}}

Note:
\"\"\"{note}\"\"\"
"""


def build_prompt(note: str) -> str:
    defs = "\n".join(f"- {c}: {d}" for c, d in DEFINITIONS.items())
    return PROMPT.format(definitions=defs, note=note)


def validate(raw: str, note: str) -> tuple[list[Finding], list[str]]:
    """Parse the model's answer and keep only findings that pass every check."""
    problems: list[str] = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [], ["answer was not valid JSON"]
    items = data.get("findings", []) if isinstance(data, dict) else []
    keep = []
    low_note = " ".join(note.lower().split())
    for it in items:
        if not isinstance(it, dict):
            problems.append("finding was not an object")
            continue
        cat, sev, ev = it.get("category"), it.get("severity"), str(it.get("evidence", "")).strip()
        if cat not in CATEGORIES:
            problems.append(f"unknown category {cat!r}")
            continue
        if sev not in SEVERITIES:
            problems.append(f"unknown severity {sev!r}, set to high for review")
            sev = "high"
        ev_norm = " ".join(ev.lower().split()).strip(" .\"'")
        if len(ev_norm) < 4 or ev_norm not in low_note:
            problems.append(f"evidence not found in note: {ev!r}")
            continue
        try:
            conf = float(it.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        keep.append(Finding(category=cat, severity=sev, evidence=ev, confidence=max(0.0, min(conf, 1.0)), source="llm"))
    return keep, problems


class OllamaBackend:
    def __init__(self, model: str = "llama3.1:8b", url: str = "http://localhost:11434", timeout: int = 60):
        self.model, self.url, self.timeout = model, url.rstrip("/"), timeout
        self.last_problems: list[str] = []

    def generate(self, prompt: str) -> str:
        body = json.dumps({"model": self.model, "prompt": prompt, "format": "json", "stream": False,
                           "options": {"temperature": 0}}).encode()
        req = urllib.request.Request(f"{self.url}/api/generate", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["response"]

    def detect(self, note: str) -> list[Finding]:
        findings, self.last_problems = validate(self.generate(build_prompt(note)), note)
        return findings
