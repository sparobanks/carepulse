import numpy as np

from carepulse.data.synth import make_retention_panel
from carepulse.retention import model as rm


def data():
    return make_retention_panel(n_carers=500, seed=4)


def test_protected_attributes_are_not_features():
    assert not set(rm.PROTECTED) & set(rm.FEATURES)


def test_time_split_has_gaps_so_labels_do_not_leak():
    s = rm.time_split(data())
    assert s.train.month.max() + 3 <= s.calib.month.min()
    assert s.calib.month.max() + 3 <= s.test.month.min()


def test_calibrated_scores_are_probabilities_and_better_than_chance():
    s = rm.time_split(data())
    m = rm.CalibratedModel(rm.fit_lgbm(s.train), s.calib)
    p = m.predict(s.test)
    assert ((p >= 0) & (p <= 1)).all()
    assert rm.metrics(s.test[rm.TARGET].values, p)["roc_auc"] > 0.6


def test_reasons_are_only_on_the_risky_side():
    s = rm.time_split(data())
    model = rm.fit_lgbm(s.train)
    med = s.train[rm.FEATURES].median()
    for reasons in rm.explain(model, s.test.head(50), s.train):
        for r in reasons:
            if r["feature"] in rm.LOW_IS_RISKY:
                assert r["value"] < med[r["feature"]]
            else:
                assert r["value"] > med[r["feature"]]


def test_ece_is_zero_for_perfect_calibration():
    y = np.array([0, 0, 1, 1] * 25)
    assert rm.ece(y, y.astype(float)) == 0
