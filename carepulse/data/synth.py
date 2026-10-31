"""Synthetic data for a home-care provider in Birmingham.

Nothing here is real. People, addresses and notes are generated so the
platform can be built and tested without touching personal data.
Locations are random points around real Birmingham neighbourhood centres.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Approximate neighbourhood centres (lat, lon). Public geography, not addresses.
AREAS = {
    "Erdington": (52.524, -1.838),
    "Sutton Coldfield": (52.563, -1.822),
    "Kingstanding": (52.546, -1.886),
    "Handsworth": (52.505, -1.929),
    "Aston": (52.503, -1.884),
    "Hodge Hill": (52.497, -1.793),
    "Yardley": (52.477, -1.806),
    "Acocks Green": (52.448, -1.818),
    "Sparkhill": (52.455, -1.862),
    "Hall Green": (52.433, -1.840),
    "Kings Heath": (52.433, -1.893),
    "Edgbaston": (52.462, -1.920),
    "Selly Oak": (52.437, -1.937),
    "Northfield": (52.415, -1.966),
    "Longbridge": (52.395, -1.982),
}

SKILLS = ["medication", "dementia", "manual_handling", "catheter_care"]


def _jitter(rng, lat, lon, km=1.2):
    # ~0.009 deg lat per km, ~0.0147 deg lon per km at this latitude
    return lat + rng.normal(0, km * 0.009), lon + rng.normal(0, km * 0.0147)


@dataclass
class Workforce:
    carers: pd.DataFrame
    clients: pd.DataFrame
    meta: dict = field(default_factory=dict)


def make_workforce(n_carers: int = 45, n_clients: int = 160, seed: int = 7) -> Workforce:
    rng = np.random.default_rng(seed)
    areas = list(AREAS)

    rows = []
    for i in range(n_carers):
        area = areas[rng.integers(len(areas))]
        lat, lon = _jitter(rng, *AREAS[area], km=1.5)
        skills = [s for s, p in zip(SKILLS, [0.85, 0.55, 0.45, 0.25]) if rng.random() < p]
        rows.append(dict(
            carer_id=f"CR{i + 1:03d}", area=area, lat=lat, lon=lon,
            sex="F" if rng.random() < 0.8 else "M",
            skills=skills,
            shift_start=7 * 60, shift_end=14 * 60,
            max_visits=int(rng.integers(8, 13)),
        ))
    carers = pd.DataFrame(rows)

    crows = []
    for j in range(n_clients):
        area = areas[rng.integers(len(areas))]
        lat, lon = _jitter(rng, *AREAS[area], km=1.0)
        needs = [s for s, p in zip(SKILLS, [0.6, 0.3, 0.2, 0.08]) if rng.random() < p]
        pref = "F" if rng.random() < 0.15 else None
        # regular carers: 2 to 3 nearby carers who are able to meet the needs
        ok = carers[carers.apply(lambda c: set(needs) <= set(c.skills) and (pref is None or c.sex == pref), axis=1)]
        if ok.empty:
            ok = carers
        d = np.hypot(ok.lat - lat, (ok.lon - lon) * 0.61)
        k = min(int(rng.integers(2, 4)), len(ok))
        regular = list(ok.assign(d=d).nsmallest(k + 3, "d").sample(k, random_state=int(rng.integers(1e9))).carer_id)
        crows.append(dict(client_id=f"CL{j + 1:03d}", area=area, lat=lat, lon=lon,
                          needs=needs, gender_pref=pref, regular_carers=regular))
    clients = pd.DataFrame(crows)
    return Workforce(carers, clients, dict(seed=seed))


def make_visits(clients: pd.DataFrame, day: int = 0, seed: int = 11, tolerance: int = 30) -> pd.DataFrame:
    """Morning-shift visits (07:00 to 14:00) for one day.

    Every client gets a morning call; about 70% also get a lunch call.
    Each visit may start up to `tolerance` minutes either side of its planned time.
    """
    rng = np.random.default_rng(seed + day * 101)
    rows = []
    for c in clients.itertuples():
        plans = [(int(rng.integers(7 * 60 + 30, 10 * 60 + 1)), int(rng.choice([30, 45, 60], p=[0.4, 0.4, 0.2])))]
        if rng.random() < 0.7:
            plans.append((int(rng.integers(12 * 60, 13 * 60 + 16)), int(rng.choice([15, 30], p=[0.5, 0.5]))))
        for k, (t0, dur) in enumerate(plans):
            rows.append(dict(visit_id=f"V{day}-{c.client_id}-{k}", client_id=c.client_id,
                             lat=c.lat, lon=c.lon, planned=t0,
                             earliest=t0 - tolerance, latest=t0 + tolerance, duration=dur,
                             needs=c.needs, gender_pref=c.gender_pref,
                             regular_carers=c.regular_carers))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- retention

def make_retention_panel(n_carers: int = 1400, months: int = 24, seed: int = 3) -> pd.DataFrame:
    """Monthly snapshots of care workers with a 'left within 90 days' label.

    Leaving is driven by a hidden hazard built from workload, rota volatility,
    unpaid travel, lack of progression, continuity, tenure and sickness.
    Sex and age band are generated only so fairness can be checked; they are
    not in the hazard and are never used as model features.
    """
    rng = np.random.default_rng(seed)
    recs = []
    next_id = 0

    def new_carer(start_month):
        nonlocal next_id
        next_id += 1
        age_band = rng.choice(["18-24", "25-39", "40-54", "55+"], p=[0.18, 0.37, 0.30, 0.15])
        return dict(
            carer_id=f"W{next_id:05d}", start=start_month,
            sex="F" if rng.random() < 0.8 else "M", age_band=age_band,
            base_hours=float(np.clip(rng.normal(32, 8), 10, 50)),
            base_volatility=float(rng.gamma(2.0, 2.0)),
            base_travel=float(np.clip(rng.beta(2, 6), 0.02, 0.6)),
            base_split=float(np.clip(rng.beta(2, 3), 0, 1)),
            last_progression=start_month - int(rng.integers(0, 24)),
            tenure0=int(rng.integers(0, 60)) if start_month == 0 else 0,
            left=None,
        )

    active = [new_carer(0) for _ in range(n_carers)]
    everyone = list(active)
    for m in range(months):
        for c in active:
            if c["left"] is not None:
                continue
            tenure = c["tenure0"] + (m - c["start"])
            hours = c["base_hours"] + rng.normal(0, 4)
            overtime = max(0.0, hours - 37.5) + max(0.0, rng.normal(0, 2))
            volatility = max(0.0, c["base_volatility"] + rng.normal(0, 1.2))  # avg changes notified <48h per month
            short_notice = rng.poisson(volatility)
            travel_share = float(np.clip(c["base_travel"] + rng.normal(0, 0.04), 0, 0.8))
            split = float(np.clip(c["base_split"] + rng.normal(0, 0.08), 0, 1))
            weekend = float(np.clip(rng.beta(2, 4) + 0.1 * split, 0, 1))
            continuity = float(np.clip(rng.beta(5, 3) - 0.3 * c["base_volatility"] / 10, 0, 1))
            sickness = rng.poisson(0.6 + 0.4 * (overtime > 5))
            if rng.random() < 0.04:
                c["last_progression"] = m
            since_prog = m - c["last_progression"]
            supervision_overdue = int(rng.random() < 0.25)
            z = (-4.35
                 + 0.30 * (volatility - 4) + 0.25 * (short_notice - 4) / 2
                 + 4.0 * (travel_share - 0.25)
                 + 0.55 * (since_prog > 12)
                 + 0.05 * overtime
                 - 1.2 * (continuity - 0.55)
                 + 0.6 * (tenure < 6)
                 + 0.20 * sickness
                 + 0.35 * supervision_overdue
                 + 0.8 * split * (travel_share > 0.3)
                 + rng.normal(0, 0.5))
            p_leave = 1 / (1 + np.exp(-z))
            recs.append(dict(
                carer_id=c["carer_id"], month=m, sex=c["sex"], age_band=c["age_band"],
                tenure_months=tenure, avg_weekly_hours=round(hours, 1), overtime_hours=round(overtime, 1),
                rota_changes_short_notice=int(short_notice), rota_volatility=round(volatility, 2),
                unpaid_travel_share=round(travel_share, 3), split_shift_share=round(split, 3),
                weekend_share=round(weekend, 3), client_continuity=round(continuity, 3),
                sickness_days=int(sickness), months_since_progression=int(since_prog),
                supervision_overdue=supervision_overdue,
            ))
            if rng.random() < p_leave:
                c["left"] = m
        # recruit replacements
        leavers = sum(1 for c in active if c["left"] == m)
        fresh = [new_carer(m + 1) for _ in range(leavers)]
        active = [c for c in active if c["left"] is None] + fresh
        everyone += fresh

    df = pd.DataFrame(recs)
    left_at = {c["carer_id"]: c["left"] for c in everyone}
    df["left_month"] = df.carer_id.map(left_at)
    df["left_within_90d"] = (df.left_month.notna() & (df.left_month >= df.month)
                             & (df.left_month < df.month + 3)).astype(int)
    # snapshots whose 3-month outcome window runs past the data cannot be labelled
    df = df[df.month <= months - 3].drop(columns="left_month").reset_index(drop=True)
    return df
