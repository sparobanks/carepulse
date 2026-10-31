import json

from carepulse.notes.llm import OllamaBackend, build_prompt, validate
from carepulse.notes.pipeline import NotesPipeline, route
from carepulse.notes.redact import redact
from carepulse.notes.rules import detect
from carepulse.notes.schema import Finding


def cats(text):
    return {f.category for f in detect(text)}


def test_rules_find_clear_concerns():
    assert "safeguarding" in cats("Noticed bruising on her arm, could not say how.")
    assert "missed_task" in cats("Did not give her morning meds as the box was empty.")
    assert "deterioration" in cats("Found her on the floor in the kitchen.")


def test_rules_ignore_negated_and_past_events():
    assert cats("Checked skin, no redness or bruising seen.") == set()
    assert cats("Daughter mentioned the fall last year but no concerns today.") == set()
    assert cats("Old bruise on arm has now gone.") == set()


def test_negation_does_not_hide_a_real_missed_task():
    assert "missed_task" in cats("Meds not given, could not find them.")


def test_redaction_removes_names_phones_and_nhs_numbers():
    text = "Mrs Begum seen. NHS 943 476 5919. Called GP on 0121 496 0123. Email a.b@nhs.net, B27 6AA."
    out, used = redact(text, {"Mrs Begum": "[CLIENT]"})
    for secret in ["Begum", "943 476 5919", "0121 496 0123", "a.b@nhs.net", "B27 6AA"]:
        assert secret not in out
    assert "[CLIENT]" in out and "[PHONE]" in out


def test_llm_answers_must_be_grounded_in_the_note():
    note = "Noticed bruising on her arm. Ate all her lunch."
    raw = json.dumps({"findings": [
        {"category": "safeguarding", "severity": "high", "evidence": "bruising on her arm", "confidence": 0.9},
        {"category": "deterioration", "severity": "high", "evidence": "she looked pale", "confidence": 0.8},
        {"category": "made_up", "severity": "low", "evidence": "Ate all", "confidence": 0.5},
    ]})
    kept, problems = validate(raw, note)
    assert [f.category for f in kept] == ["safeguarding"]
    assert len(problems) == 2


def test_llm_bad_json_is_rejected():
    kept, problems = validate("not json at all", "anything")
    assert kept == [] and problems


def test_prompt_contains_the_redacted_note_only():
    p = build_prompt("[CLIENT] had a fall.")
    assert "[CLIENT] had a fall." in p


def test_safeguarding_always_goes_to_urgent_review():
    assert route([Finding("safeguarding", "low", "x", 0.99, "rules")]) == "urgent_review"
    assert route([Finding("quality_risk", "low", "x", 0.95, "rules")]) == "record"
    assert route([Finding("quality_risk", "low", "x", 0.40, "llm")]) == "review"
    assert route([]) == "no_action"


def test_pipeline_with_a_mocked_llm(monkeypatch):
    llm = OllamaBackend()
    monkeypatch.setattr(llm, "generate", lambda prompt: json.dumps({"findings": [
        {"category": "deterioration", "severity": "medium", "evidence": "barely touched her tea", "confidence": 0.7}]}))
    res = NotesPipeline([llm]).analyse("n1", "Mrs Kaur barely touched her tea today.", {"Mrs Kaur": "[CLIENT]"})
    assert "Kaur" not in res.redacted_text
    assert any(f.category == "deterioration" and "llm" in f.source for f in res.findings)


def test_failed_model_sends_note_to_a_person():
    class Broken:
        def detect(self, text):
            raise TimeoutError("model did not answer")
    res = NotesPipeline([Broken()]).analyse("n2", "Made tea, all fine.")
    assert res.route == "review" and res.problems
