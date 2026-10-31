"""Train and test the care-notes detectors on synthetic notes."""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from ..data.notes_synth import FAMILIES, TEST_FAMILIES
from .classifier import NoteClassifier
from .pipeline import NotesPipeline
from .rules import detect as rules_detect
from .schema import CATEGORIES


def _val_families(seed: int = 0) -> set[str]:
    """Hold out about a quarter of the training families (at least one per category) for tuning."""
    rng = random.Random(seed)
    train_fams = [f for f in FAMILIES if f not in TEST_FAMILIES]
    out = set()
    for cat in CATEGORIES + [None]:
        fams = [f for f in train_fams if FAMILIES[f][0] == cat]
        k = max(1, round(len(fams) * 0.25))
        out |= set(rng.sample(fams, k))
    return out


def tune_thresholds(train: pd.DataFrame, safeguarding_recall: float = 0.95) -> dict[str, float]:
    val_f = _val_families()
    is_val = train.families.str.split("|").apply(lambda fs: any(f in val_f for f in fs))
    fit, val = train[~is_val], train[is_val]
    clf = NoteClassifier().fit(fit.text, fit)
    p = clf.proba(val.text)
    th = {}
    for c in CATEGORIES:
        y = val[c].values
        grid = np.linspace(0.05, 0.95, 91)
        if c == "safeguarding":
            ok = [t for t in grid if y.sum() and (p[c].values[y == 1] >= t).mean() >= safeguarding_recall]
            th[c] = round(float(max(ok)) if ok else 0.05, 2)
        else:
            f1 = [precision_recall_fscore_support(y, p[c] >= t, average="binary", zero_division=0)[2] for t in grid]
            th[c] = round(float(grid[int(np.argmax(f1))]), 2)
    return th


def scores(df: pd.DataFrame, preds: dict[str, list[int]]) -> pd.DataFrame:
    rows = []
    for c in CATEGORIES:
        p, r, f, _ = precision_recall_fscore_support(df[c], preds[c], average="binary", zero_division=0)
        rows.append(dict(category=c, positives=int(df[c].sum()), precision=round(p, 3), recall=round(r, 3), f1=round(f, 3)))
    return pd.DataFrame(rows)


def run(notes: pd.DataFrame) -> dict:
    train, test = notes[notes.split == "train"], notes[notes.split == "test"]
    th = tune_thresholds(train)
    clf = NoteClassifier(th).fit(train.text, train)

    def collect(detector):
        preds = {c: [] for c in CATEGORIES}
        for t in test.text:
            cats = {f.category for f in detector(t)}
            for c in CATEGORIES:
                preds[c].append(int(c in cats))
        return preds

    def pipeline_stats(backends):
        pipe = NotesPipeline(backends)
        res = [pipe.analyse(r.note_id, r.text, {r.client: "[CLIENT]"}) for r in test.itertuples()]
        preds = {c: [int(any(f.category == c for f in x.findings)) for x in res] for c in CATEGORIES}
        routes = pd.Series([x.route for x in res], index=test.index)
        sg = test.safeguarding == 1
        any_c = test[CATEGORIES].sum(axis=1) > 0
        to_person = routes.isin(["urgent_review", "review"])
        return res, scores(test, preds).to_dict("records"), dict(
            share_by_route={k: round(v, 3) for k, v in routes.value_counts(normalize=True).items()},
            safeguarding_notes_sent_to_urgent_review=round(float((routes[sg] == "urgent_review").mean()), 3),
            notes_with_a_concern_seen_by_a_person_or_recorded=round(float(routes[any_c].ne("no_action").mean()), 3),
            routine_notes_sent_to_a_person=round(float(to_person[~any_c].mean()), 3),
        )

    res_rules, rules_scores, rules_routing = pipeline_stats([])
    _, hybrid_scores, hybrid_routing = pipeline_stats([clf])
    return dict(
        thresholds=th,
        n_train=int(len(train)), n_test=int(len(test)),
        rules=rules_scores, rules_routing=rules_routing,
        classifier=scores(test, collect(clf.detect)).to_dict("records"),
        rules_plus_classifier=hybrid_scores, rules_plus_classifier_routing=hybrid_routing,
        redaction=dict(notes_with_phone=int(test.text.str.contains("0121").sum()),
                       phones_left_after_redaction=int(sum("0121" in r.redacted_text for r in res_rules)),
                       client_names_left=int(sum(c.lower() in r.redacted_text.lower()
                                                 for c, r in zip(test.client, res_rules)))),
    ), clf
