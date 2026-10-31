"""Workforce retention: which carers are at risk of leaving in the next 90 days.

Design choices:
* Only work-pattern features the provider controls (hours, rota changes,
  travel, progression, continuity, supervision). Sex and age are NOT model
  inputs; they are used only afterwards to check the model is fair.
* Time-based split with gaps, so the model is always tested on months after
  the ones it learned from, and no label window overlaps the next split.
* Probabilities are calibrated, so "30% risk" means roughly 3 in 10 leave.
* Each score comes with the top reasons and a suggested supportive action.
  The output is meant to start a conversation, never to justify any action
  against a member of staff.
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "tenure_months", "avg_weekly_hours", "overtime_hours", "rota_changes_short_notice",
    "rota_volatility", "unpaid_travel_share", "split_shift_share", "weekend_share",
    "client_continuity", "sickness_days", "months_since_progression", "supervision_overdue",
]
PROTECTED = ["sex", "age_band"]
TARGET = "left_within_90d"

REASONS = {
    "rota_changes_short_notice": ("Many rota changes at short notice", "Keep their runs stable for the next month"),
    "rota_volatility": ("Unsettled rota from week to week", "Keep their runs stable for the next month"),
    "unpaid_travel_share": ("A lot of unpaid travel between visits", "Re-plan their visits closer together"),
    "months_since_progression": ("No training or pay step for a long time", "Talk about training or progression"),
    "client_continuity": ("Rarely sees the same clients", "Give them a steady group of regular clients"),
    "tenure_months": ("New starter", "Book a check-in with a buddy or mentor"),
    "overtime_hours": ("Working a lot of overtime", "Review their hours"),
    "sickness_days": ("Recent sickness absence", "Have a wellbeing conversation"),
    "supervision_overdue": ("Supervision is overdue", "Book their supervision"),
    "split_shift_share": ("Many split shifts", "Reduce split shifts where possible"),
    "weekend_share": ("Lots of weekend work", "Share weekend work more evenly"),
    "avg_weekly_hours": ("Working more hours than most", "Check their hours match what they want"),
}


@dataclass
class Split:
    train: pd.DataFrame
    calib: pd.DataFrame
    test: pd.DataFrame


def time_split(df: pd.DataFrame, train_to: int = 11, calib: tuple = (14, 15), test_from: int = 18) -> Split:
    """Gaps of 2 months between splits, because each label looks 3 months ahead."""
    return Split(df[df.month <= train_to], df[df.month.between(*calib)], df[df.month >= test_from])


def fit_logistic(train: pd.DataFrame):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    m.fit(train[FEATURES], train[TARGET])
    return m


def fit_lgbm(train: pd.DataFrame, seed: int = 0):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=60,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                           random_state=seed, verbose=-1)
    m.fit(train[FEATURES], train[TARGET])
    return m


class CalibratedModel:
    def __init__(self, model, calib: pd.DataFrame):
        self.model = model
        raw = model.predict_proba(calib[FEATURES])[:, 1]
        self.iso = IsotonicRegression(out_of_bounds="clip").fit(raw, calib[TARGET])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.iso.predict(self.model.predict_proba(X[FEATURES])[:, 1])


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error with equal-count bins."""
    order = np.argsort(p)
    err = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            err += len(chunk) / len(p) * abs(p[chunk].mean() - y[chunk].mean())
    return float(err)


def metrics(y: np.ndarray, p: np.ndarray, top: float = 0.1) -> dict:
    k = max(1, int(round(len(p) * top)))
    idx = np.argsort(-p)[:k]
    return dict(
        roc_auc=round(roc_auc_score(y, p), 3),
        pr_auc=round(average_precision_score(y, p), 3),
        brier=round(brier_score_loss(y, p), 4),
        ece=round(ece(y, p), 4),
        base_rate=round(float(y.mean()), 3),
        top10_precision=round(float(y[idx].mean()), 3),
        top10_share_of_leavers=round(float(y[idx].sum() / max(y.sum(), 1)), 3),
    )


def fairness(df: pd.DataFrame, p: np.ndarray, threshold: float) -> pd.DataFrame:
    rows = []
    for col in PROTECTED:
        for g, idx in df.groupby(col).indices.items():
            y, q = df[TARGET].values[idx], p[idx]
            flagged = q >= threshold
            rows.append(dict(
                attribute=col, group=g, n=len(idx), actual_leave_rate=round(float(y.mean()), 3),
                mean_predicted=round(float(q.mean()), 3),
                roc_auc=round(roc_auc_score(y, q), 3) if 0 < y.sum() < len(y) else None,
                flag_rate=round(float(flagged.mean()), 3),
                leavers_caught=round(float(flagged[y == 1].mean()), 3) if y.sum() else None,
            ))
    return pd.DataFrame(rows)


# For these features a LOW value is the risky side; for the rest a HIGH value is.
LOW_IS_RISKY = {"tenure_months", "client_continuity"}


def explain(model, X: pd.DataFrame, reference: pd.DataFrame, top_n: int = 3) -> list[list[dict]]:
    """Top reasons that push each person's risk up, using SHAP on the tree model.

    A reason is only shown when the person's value is on the risky side of the
    workforce median (for example, more short-notice changes than typical).
    This keeps the wording honest: we never say "new starter" about someone
    with five years' service just because the model gave that feature weight.
    """
    import warnings

    import shap
    med = reference[FEATURES].median()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = shap.TreeExplainer(model).shap_values(X[FEATURES])
    if isinstance(sv, list):
        sv = sv[1]
    out = []
    for row, vals in zip(X[FEATURES].itertuples(index=False), sv):
        order = np.argsort(-vals)
        reasons = []
        for j in order:
            if vals[j] <= 0 or len(reasons) == top_n:
                break
            f = FEATURES[j]
            v = getattr(row, f)
            risky = v < med[f] if f in LOW_IS_RISKY else v > med[f]
            if not risky:
                continue
            text, action = REASONS[f]
            reasons.append(dict(feature=f, value=v, typical=float(med[f]), impact=round(float(vals[j]), 3),
                                reason=text, suggested_action=action))
        out.append(reasons)
    return out


def risk_band(p: float) -> str:
    return "high" if p >= 0.30 else "medium" if p >= 0.15 else "low"
