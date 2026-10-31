"""Test a local open-source LLM (through Ollama) on the same held-out notes.

Run on a machine with Ollama installed:

    ollama pull llama3.1:8b
    python scripts/eval_llm.py --model llama3.1:8b --limit 200

Writes reports/llm_results.json with scores for the LLM alone and for
rules plus LLM, plus how often the model's answers failed the checks
(bad JSON, unknown category, or evidence not found in the note).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from carepulse.data.notes_synth import make_notes  # noqa: E402
from carepulse.notes.evaluate import scores  # noqa: E402
from carepulse.notes.llm import OllamaBackend  # noqa: E402
from carepulse.notes.pipeline import NotesPipeline  # noqa: E402
from carepulse.notes.redact import redact  # noqa: E402
from carepulse.notes.schema import CATEGORIES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args()

    notes = make_notes()
    test = notes[notes.split == "test"].head(a.limit)
    llm = OllamaBackend(a.model, a.url)
    pipe = NotesPipeline([llm])
    only, both, problems, secs = {c: [] for c in CATEGORIES}, {c: [] for c in CATEGORIES}, 0, []
    for r in test.itertuples():
        t = time.time()
        clean, _ = redact(r.text, {r.client: "[CLIENT]"})
        try:
            found = {f.category for f in llm.detect(clean)}
            problems += bool(llm.last_problems)
        except Exception as e:
            print("model error:", e)
            found, problems = set(), problems + 1
        secs.append(time.time() - t)
        res = pipe.analyse(r.note_id, r.text, {r.client: "[CLIENT]"})
        for c in CATEGORIES:
            only[c].append(int(c in found))
            both[c].append(int(any(f.category == c for f in res.findings)))
    out = dict(model=a.model, notes=len(test), seconds_per_note=round(sum(secs) / len(secs), 2),
               notes_with_rejected_output=problems,
               llm_only=scores(test, only).to_dict("records"),
               rules_plus_llm=scores(test, both).to_dict("records"))
    (ROOT / "reports" / "llm_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(pd.DataFrame(out["llm_only"]).to_string(index=False))


if __name__ == "__main__":
    main()
