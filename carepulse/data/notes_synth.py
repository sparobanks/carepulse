"""Synthetic home-care visit notes with labels.

Each note is built from short sentences. Sentences come from "families":
a family is one kind of event (for example "unexplained bruising") written
in several ways. Families are split between train and test, so the test set
uses phrasings the models never saw. This stops a model scoring well just by
memorising templates.
"""
from __future__ import annotations

import random

import pandas as pd

CATEGORIES = ["safeguarding", "missed_task", "deterioration", "quality_risk"]

# Templates are written with she/her, so all generated clients are women.
CLIENTS = ["Mrs Ahmed", "Mrs Patel", "Edna", "Mrs O'Brien", "Mrs Kaur", "Joan", "Mrs Clarke",
           "Doris", "Mrs Hussain", "Mrs Walsh", "Irene", "Mrs Begum", "Mrs Lewis", "Winifred"]
RELATIVES = ["son", "daughter", "nephew", "grandson", "husband", "lodger", "neighbour"]
MEDS = ["morning meds", "metformin", "the blister pack", "her inhaler", "the warfarin", "paracetamol", "eye drops"]
BODY = ["upper arm", "wrist", "forearm", "thigh", "shoulder", "inner arm"]

# family id -> (category or None, severity, list of sentence templates)
FAMILIES: dict[str, tuple[str | None, str, list[str]]] = {
    # ---------------- safeguarding
    "S_bruise": ("safeguarding", "high", [
        "Noticed fresh bruising on {client}'s {body}, {client} could not say how it happened.",
        "Found a large bruise on the {body} that was not there yesterday, no explanation given.",
        "Marks on {body} look like finger marks, {client} went quiet when I asked.",
    ]),
    "S_money": ("safeguarding", "high", [
        "{client} said her {relative} keeps taking money from her purse.",
        "{client} told me the {relative} has her bank card and she has no money for shopping.",
        "Pension money gone again, {client} says the {relative} took it.",
    ]),
    "S_neglect": ("safeguarding", "high", [
        "No food in the house again and heating off, {client} says nobody has been shopping for days.",
        "{client} was wearing the same soiled clothes as two days ago and the bed was wet through.",
        "Fridge empty and milk out of date, the {relative} who does the shopping has not been.",
    ]),
    "S_fear": ("safeguarding", "high", [
        "{client} told me she does not feel safe when the {relative} is around.",
        "Heard the {relative} shouting at {client}, she looked frightened and was crying.",
        "{client} asked me not to tell anyone but said the {relative} pushes her.",
    ]),
    "S_restrict": ("safeguarding", "high", [
        "{client} had been locked in the bedroom when I arrived, {relative} had the key.",
        "{relative} would not let me see {client} alone and answered every question for her.",
    ]),
    "S_meds_family": ("safeguarding", "high", [
        "{relative} has been hiding {client}'s tablets and says she does not need them.",
        "{client} says the {relative} gives her extra sleeping tablets to keep her quiet.",
    ]),
    # ---------------- missed tasks
    "M_meds": ("missed_task", "medium", [
        "Did not give {med} as the box was empty.",
        "{med} not given, could not find them anywhere.",
        "Unable to administer {med}, pharmacy delivery has not arrived.",
    ]),
    "M_meal": ("missed_task", "medium", [
        "No time to make lunch so left a sandwich and a drink.",
        "Did not prepare breakfast as ran out of time.",
        "Meal not made, there was nothing in to cook.",
    ]),
    "M_personal": ("missed_task", "medium", [
        "Shower not done today as visit was running behind.",
        "Ran out of time so did not help with the strip wash.",
        "Pad not changed, {client} was asleep and I had to leave for next call.",
    ]),
    "M_short": ("missed_task", "medium", [
        "Visit cut short by 20 minutes because of traffic, did not finish all tasks.",
        "Only stayed 10 minutes of a 30 minute call due to being double booked.",
    ]),
    "M_cream": ("missed_task", "low", [
        "Forgot to apply the barrier cream to legs.",
        "MAR chart not signed for the evening dose, unsure if it was given.",
        "Creams were not applied as the tube had run out.",
    ]),
    # ---------------- deterioration
    "D_mobility": ("deterioration", "high", [
        "{client} struggled to stand today and needed much more help than usual.",
        "Legs very weak this morning, could not walk to the bathroom with the frame.",
        "Mobility much worse, needed two attempts to get out of the chair.",
    ]),
    "D_intake": ("deterioration", "medium", [
        "Only ate a couple of spoonfuls of porridge and has barely drunk anything.",
        "Eating very little this week and leaving most of the meals.",
        "Not drinking, cup of tea untouched from earlier visit.",
    ]),
    "D_confusion": ("deterioration", "high", [
        "{client} much more confused than normal and did not recognise me.",
        "Very muddled today, thought it was night time and kept asking for her mum.",
        "Sudden confusion today, which is new for {client}.",
    ]),
    "D_skin": ("deterioration", "high", [
        "Red patch on the sacrum that did not fade when pressed.",
        "Skin on heel is broken and weeping.",
        "New sore on bottom, area looks red and broken.",
    ]),
    "D_breathing": ("deterioration", "high", [
        "{client} was breathless just talking and had a chesty cough.",
        "Breathing sounded rattly and fast, lips looked a bit blue.",
    ]),
    "D_fall": ("deterioration", "high", [
        "Found {client} on the floor in the kitchen, said she slipped in the night.",
        "{client} had a fall getting out of bed, small cut on head.",
        "Neighbour said {client} fell in the garden yesterday.",
    ]),
    "D_uti": ("deterioration", "medium", [
        "Urine dark and very strong smelling, {client} says it stings.",
        "Going to the toilet a lot and urine looks cloudy.",
    ]),
    # ---------------- quality risks
    "Q_late": ("quality_risk", "low", [
        "Arrived 50 minutes late as previous call overran, {client} was upset.",
        "Very late to this call due to rota change, client had been waiting.",
    ]),
    "Q_complaint": ("quality_risk", "medium", [
        "{relative} unhappy that a different carer comes every day and wants to complain.",
        "{client} complained that carers keep rushing her.",
        "{relative} rang the office to complain about missed calls last week.",
    ]),
    "Q_equipment": ("quality_risk", "medium", [
        "Hoist battery not charging, used the spare.",
        "Sling is frayed at the edge and needs replacing.",
        "Commode wheel broken, reported to office.",
    ]),
    "Q_environment": ("quality_risk", "low", [
        "Loose rug in the hallway is a trip hazard.",
        "Stair light not working, very dark on the landing.",
        "Smoke alarm beeping, battery seems low.",
    ]),
    "Q_access": ("quality_risk", "low", [
        "Key safe code did not work, waited 15 minutes for the {relative}.",
        "Could not get in, no answer at the door for 10 minutes.",
    ]),
    # ---------------- routine (no flag)
    "R_care": (None, "none", [
        "Assisted with wash and dress, {client} chose her own clothes.",
        "Helped {client} with a full strip wash and applied cream as per care plan.",
        "Supported with personal care, skin looked healthy.",
    ]),
    "R_food": (None, "none", [
        "Made porridge and a cup of tea, ate it all.",
        "Prepared a jacket potato for lunch and left a drink within reach.",
        "Breakfast of toast and jam, good appetite today.",
    ]),
    "R_meds": (None, "none", [
        "Meds given as per MAR and chart signed.",
        "Prompted {med}, taken with water.",
        "All medication given from the blister pack.",
    ]),
    "R_mood": (None, "none", [
        "{client} in good spirits and chatting about the football.",
        "Watched the news together, {client} was cheerful.",
        "Settled and comfortable in her chair when I left.",
    ]),
    "R_tasks": (None, "none", [
        "All tasks completed, house left tidy and secure.",
        "Bins taken out and kitchen wiped down.",
        "Emptied the commode and left the room tidy.",
    ]),
    # ---------------- hard negatives: mention risk words but nothing wrong
    "N_negated": (None, "none", [
        "Checked skin, no redness or bruising seen.",
        "No signs of any marks on arms or legs.",
        "Denies any pain and no new marks seen.",
    ]),
    "N_history": (None, "none", [
        "{relative} mentioned the fall last year but no concerns today.",
        "Talked about when she was in hospital with a chest infection years ago.",
        "Old bruise on {body} has now gone.",
    ]),
    "N_baseline": (None, "none", [
        "Usual mild forgetfulness, settled and chatty.",
        "Walks with frame as normal, steady on feet.",
        "Eating and drinking well, finished a full plate.",
    ]),
    "N_ontime": (None, "none", [
        "Arrived on time, {client} was ready and waiting.",
        "Key safe worked fine today.",
        "Hoist checked and working well.",
    ]),
}

TEST_FAMILIES = {
    "S_restrict", "S_meds_family", "M_short", "M_cream", "D_uti", "D_breathing",
    "Q_access", "Q_environment", "R_tasks", "N_history", "N_ontime",
}


def _fill(t: str, rng: random.Random, client: str) -> str:
    return t.format(client=client, relative=rng.choice(RELATIVES), med=rng.choice(MEDS),
                    body=rng.choice(BODY))


def _noise(s: str, rng: random.Random) -> str:
    """Make sentences look like real quick notes: lower case, shorthand, typos."""
    if rng.random() < 0.25:
        s = s.lower()
    if rng.random() < 0.2:
        s = s.replace("client", "pt").replace("medication", "meds")
    if rng.random() < 0.08 and len(s) > 20:
        i = rng.randrange(5, len(s) - 5)
        s = s[:i] + s[i + 1:]  # dropped letter
    if rng.random() < 0.1:
        s = s.rstrip(".")
    return s


def make_notes(n: int = 3000, seed: int = 5) -> pd.DataFrame:
    rng = random.Random(seed)
    pos = {f: v for f, v in FAMILIES.items() if v[0] is not None}
    routine = [f for f, v in FAMILIES.items() if f.startswith("R_")]
    hard = [f for f, v in FAMILIES.items() if f.startswith("N_")]
    rows = []
    for i in range(n):
        split = "test" if rng.random() < 0.3 else "train"

        def pick(fams):
            allowed = [f for f in fams if (f in TEST_FAMILIES) == (split == "test")]
            return rng.choice(allowed or fams)

        client = rng.choice(CLIENTS)
        sents, fams = [], []
        for _ in range(rng.randint(1, 3)):
            fams.append(pick(routine))
        k = rng.choices([0, 1, 2], weights=[0.5, 0.4, 0.1])[0]
        for _ in range(k):
            fams.append(pick(list(pos)))
        if rng.random() < 0.35:
            fams.append(pick(hard))
        fams = list(dict.fromkeys(fams))  # no repeated sentence kinds
        rng.shuffle(fams)
        labels = {c: 0 for c in CATEGORIES}
        sev = "none"
        order = {"none": 0, "low": 1, "medium": 2, "high": 3}
        for f in fams:
            cat, s, temps = FAMILIES[f]
            sents.append(_noise(_fill(rng.choice(temps), rng, client), rng))
            if cat:
                labels[cat] = 1
                if order[s] > order[sev]:
                    sev = s
        text = " ".join(x if x.endswith(".") else x + "." for x in sents)
        # some notes carry identifiers, so redaction can be tested
        if rng.random() < 0.05:
            # 0121 496 0000 to 0999 is Ofcom's range reserved for fiction
            text += f" Called GP on 0121 496 0{rng.randint(0, 999):03d}."
        rows.append(dict(note_id=f"N{i:05d}", split=split, client=client, text=text,
                         families="|".join(fams), severity=sev, **labels))
    return pd.DataFrame(rows)
