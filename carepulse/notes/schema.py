from __future__ import annotations

from dataclasses import dataclass, asdict

CATEGORIES = ["safeguarding", "missed_task", "deterioration", "quality_risk"]
SEVERITIES = ["low", "medium", "high"]

# What the category means, in the words used in prompts and the user guide.
DEFINITIONS = {
    "safeguarding": "possible abuse or neglect: unexplained injuries, money being taken, the person feeling unsafe, "
                    "being shouted at, restricted, or left without food, heat or care",
    "missed_task": "a planned care task that did not happen: medication not given, meal not made, personal care not done, "
                   "visit cut short, creams not applied, records not signed",
    "deterioration": "a change in the person's health: falls, new confusion, breathing problems, skin damage, "
                     "weaker mobility, eating or drinking less, signs of infection",
    "quality_risk": "a risk to the quality or safety of the service: late visits, complaints, faulty equipment, "
                    "hazards in the home, problems getting in",
}


@dataclass
class Finding:
    category: str
    severity: str
    evidence: str          # exact words from the note that support the flag
    confidence: float
    source: str            # rules | classifier | llm

    def to_dict(self) -> dict:
        return asdict(self)
