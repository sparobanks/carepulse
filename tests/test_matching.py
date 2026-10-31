import numpy as np
import pandas as pd

from carepulse.data.synth import make_visits, make_workforce
from carepulse.geo.travel import EstimatedTravel
from carepulse.matching.engine import Weights, check_hard_rules, eligible, evaluate, greedy_plan, solve


def small():
    w = make_workforce(n_carers=10, n_clients=25, seed=1)
    v = make_visits(w.clients, seed=2)
    lat = np.concatenate([w.carers.lat, v.lat])
    lon = np.concatenate([w.carers.lon, v.lon])
    return w.carers, v, EstimatedTravel().matrix(lat, lon)


def test_optimised_plan_keeps_every_hard_rule():
    carers, visits, T = small()
    plan = solve(carers, visits, T, Weights(), time_limit_s=3)
    assert check_hard_rules(plan, carers, visits, T) == []
    assert evaluate(plan, carers, visits, T)["assigned"] > 0


def test_greedy_plan_keeps_every_hard_rule():
    carers, visits, T = small()
    plan = greedy_plan(carers, visits, T)
    assert check_hard_rules(plan, carers, visits, T) == []


def test_optimiser_is_at_least_as_good_as_baseline_on_coverage_and_continuity():
    carers, visits, T = small()
    opt = evaluate(solve(carers, visits, T, Weights(), time_limit_s=3), carers, visits, T)
    base = evaluate(greedy_plan(carers, visits, T), carers, visits, T)
    assert opt["assigned"] >= base["assigned"]
    assert opt["continuity_pct"] >= base["continuity_pct"]


def test_eligibility_respects_skills_and_gender_preference():
    carer = pd.Series(dict(skills=["medication"], sex="M"))
    assert eligible(carer, dict(needs=["medication"], gender_pref=None))
    assert not eligible(carer, dict(needs=["catheter_care"], gender_pref=None))
    assert not eligible(carer, dict(needs=[], gender_pref="F"))


def test_checker_catches_a_broken_plan():
    carers, visits, T = small()
    plan = greedy_plan(carers, visits, T)
    i = plan[plan.carer_id.notna()].index[0]
    plan.loc[i, "start"] = int(visits.set_index("visit_id").loc[plan.visit_id[i], "latest"]) + 60
    assert any("outside window" in e for e in check_hard_rules(plan, carers, visits, T))
