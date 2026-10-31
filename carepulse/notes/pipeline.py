"""Care-notes pipeline: redact, detect, combine, route, log.

Routing rules (simple on purpose, so staff can understand them):
  * any possible safeguarding concern  -> "urgent_review" (a person always decides)
  * any high-severity finding          -> "review"
  * any finding the system is unsure of (confidence < 0.6) -> "review"
  * other findings                     -> "record" (logged for the coordinator)
  * nothing found                      -> "no_action"
The system never closes a safeguarding concern on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .redact import redact
from .rules import detect as rules_detect
from .schema import Finding

RANK = {"low": 1, "medium": 2, "high": 3}
REVIEW_CONFIDENCE = 0.6


def combine(*lists: list[Finding]) -> list[Finding]:
    """One finding per category: keep the most severe, then the most confident.
    If two backends agree on a category, confidence goes up."""
    by_cat: dict[str, list[Finding]] = {}
    for lst in lists:
        for f in lst:
            by_cat.setdefault(f.category, []).append(f)
    out = []
    for cat, fs in by_cat.items():
        best = max(fs, key=lambda f: (RANK[f.severity], f.confidence))
        sources = sorted({f.source for f in fs})
        conf = best.confidence if len(sources) == 1 else min(1.0, max(f.confidence for f in fs) + 0.1)
        out.append(Finding(cat, best.severity, best.evidence, round(conf, 3), "+".join(sources)))
    return out


def route(findings: list[Finding]) -> str:
    if not findings:
        return "no_action"
    if any(f.category == "safeguarding" for f in findings):
        return "urgent_review"
    if any(f.severity == "high" for f in findings) or any(f.confidence < REVIEW_CONFIDENCE for f in findings):
        return "review"
    return "record"


@dataclass
class Result:
    note_id: str
    redacted_text: str
    findings: list[Finding]
    route: str
    backends: list[str]
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(note_id=self.note_id, redacted_text=self.redacted_text, route=self.route,
                    backends=self.backends, problems=self.problems,
                    findings=[f.to_dict() for f in self.findings])


class NotesPipeline:
    """backends: any objects with .detect(text) -> list[Finding]. Rules always run as a safety net."""

    def __init__(self, backends: list | None = None, version: str = "0.1.0"):
        self.backends = backends or []
        self.version = version

    def analyse(self, note_id: str, text: str, names: dict[str, str] | None = None) -> Result:
        clean, _ = redact(text, names)
        lists = [rules_detect(clean)]
        used = ["rules"]
        problems: list[str] = []
        for b in self.backends:
            try:
                lists.append(b.detect(clean))
                used.append(type(b).__name__)
                problems += getattr(b, "last_problems", [])
            except Exception as e:  # a failed model must never hide the rules' findings
                problems.append(f"{type(b).__name__} failed: {e}")
        findings = combine(*lists)
        r = route(findings)
        if problems and r in ("no_action", "record"):
            r = "review"  # if a model misbehaved, a person takes a look
        return Result(note_id, clean, findings, r, used, problems)
