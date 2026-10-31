"""A small learned model: TF-IDF features with one logistic regression per category.

It learns from labelled notes rather than hand-written rules. To show which
words drove a flag, each sentence is scored on its own and the highest
scoring sentence is returned as evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from .rules import sentences
from .schema import CATEGORIES, Finding

DEFAULT_SEVERITY = {"safeguarding": "high", "deterioration": "medium", "missed_task": "medium", "quality_risk": "low"}


class NoteClassifier:
    def __init__(self, thresholds: dict[str, float] | None = None):
        self.vec = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
        ])
        self.models: dict[str, LogisticRegression] = {}
        self.thresholds = thresholds or {c: 0.5 for c in CATEGORIES}

    def fit(self, texts: pd.Series, labels: pd.DataFrame):
        X = self.vec.fit_transform(texts)
        for c in CATEGORIES:
            self.models[c] = LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced").fit(X, labels[c])
        return self

    def proba(self, texts) -> pd.DataFrame:
        X = self.vec.transform(list(texts))
        return pd.DataFrame({c: m.predict_proba(X)[:, 1] for c, m in self.models.items()})

    def detect(self, text: str) -> list[Finding]:
        p = self.proba([text]).iloc[0]
        out = []
        sents = sentences(text) or [text]
        sp = self.proba(sents)
        for c in CATEGORIES:
            if p[c] >= self.thresholds[c]:
                best = sents[int(np.argmax(sp[c].values))]
                out.append(Finding(category=c, severity=DEFAULT_SEVERITY[c], evidence=best,
                                   confidence=round(float(p[c]), 3), source="classifier"))
        return out
