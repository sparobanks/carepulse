"""Carer-to-visit matching.

The problem: give every visit in a shift to a carer, so that
  * hard rules always hold: the carer has the skills the client needs, meets
    any carer-gender preference, arrives inside the visit's time window, and
    finishes within their shift;
  * and a weighted mix of goals is as good as possible:
      - less travel time,
      - more visits by the client's regular carers (continuity of care),
      - fairer spread of visits between carers.

It is a vehicle routing problem with time windows, solved with Google OR-Tools.
Each carer is a "vehicle" that starts and ends at home. The weights are
explicit, so a manager can see the trade-off rather than trust a black box.

`greedy_plan` is a simple baseline that copies how rotas are often built by
hand: take visits in time order and give each one to the nearest free carer.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

UNASSIGNED_PENALTY = 100_000


@dataclass
class Weights:
    travel: int = 1          # cost per minute of travel
    continuity: int = 15     # cost of a visit by a carer who is not a regular (in "minutes of travel")
    balance: int = 10        # cost per visit on the busiest carer's list
    punctuality: int = 1     # cost per minute a visit starts more than 10 minutes from its planned time


def eligible(carer, visit) -> bool:
    if not set(visit["needs"]) <= set(carer["skills"]):
        return False
    pref = visit.get("gender_pref")
    if isinstance(pref, str) and pref and carer["sex"] != pref:
        return False
    return True


def _points(carers: pd.DataFrame, visits: pd.DataFrame):
    lat = np.concatenate([carers.lat.values, visits.lat.values])
    lon = np.concatenate([carers.lon.values, visits.lon.values])
    return lat, lon


def solve(carers: pd.DataFrame, visits: pd.DataFrame, travel_min: np.ndarray,
          weights: Weights = Weights(), time_limit_s: int = 20) -> pd.DataFrame:
    """Return one row per visit: carer_id (or None), start time, travel to reach it."""
    nc, nv = len(carers), len(visits)
    carers = carers.reset_index(drop=True)
    visits = visits.reset_index(drop=True)
    # node layout: 0..nc-1 carer homes, nc..nc+nv-1 visits
    n = nc + nv
    service = np.concatenate([np.zeros(nc, int), visits.duration.values.astype(int)])

    manager = pywrapcp.RoutingIndexManager(n, nc, list(range(nc)), list(range(nc)))
    routing = pywrapcp.RoutingModel(manager)

    def time_cb(fi, ti):
        i, j = manager.IndexToNode(fi), manager.IndexToNode(ti)
        return int(service[i] + travel_min[i, j])
    t_idx = routing.RegisterTransitCallback(time_cb)

    regular = [set(r) for r in visits.regular_carers]
    cost_idx = []
    for v in range(nc):
        cid = carers.carer_id[v]

        def cost_cb(fi, ti, cid=cid):
            i, j = manager.IndexToNode(fi), manager.IndexToNode(ti)
            c = weights.travel * int(travel_min[i, j])
            if j >= nc and cid not in regular[j - nc]:
                c += weights.continuity
            return c
        k = routing.RegisterTransitCallback(cost_cb)
        cost_idx.append(k)
        routing.SetArcCostEvaluatorOfVehicle(k, v)

    horizon = 24 * 60
    routing.AddDimension(t_idx, 120, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for v in range(nc):
        time_dim.CumulVar(routing.Start(v)).SetRange(int(carers.shift_start[v]), int(carers.shift_end[v]))
        time_dim.CumulVar(routing.End(v)).SetRange(int(carers.shift_start[v]), int(carers.shift_end[v]))
    for k in range(nv):
        idx = manager.NodeToIndex(nc + k)
        time_dim.CumulVar(idx).SetRange(int(visits.earliest[k]), int(visits.latest[k]))
        allowed = [v for v in range(nc) if eligible(carers.loc[v], visits.loc[k])]
        routing.VehicleVar(idx).SetValues([-1] + allowed)  # -1 = left unassigned
        routing.AddDisjunction([idx], UNASSIGNED_PENALTY)
        if weights.punctuality:
            time_dim.SetCumulVarSoftUpperBound(idx, int(visits.planned[k]) + 10, weights.punctuality)
            time_dim.SetCumulVarSoftLowerBound(idx, int(visits.planned[k]) - 10, weights.punctuality)

    count_idx = routing.RegisterUnaryTransitCallback(lambda fi: 1 if manager.IndexToNode(fi) >= nc else 0)
    routing.AddDimensionWithVehicleCapacity(count_idx, 0, [int(x) for x in carers.max_visits], True, "Visits")
    routing.GetDimensionOrDie("Visits").SetGlobalSpanCostCoefficient(weights.balance)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit_s
    sol = routing.SolveWithParameters(params)

    out = [dict(visit_id=visits.visit_id[k], carer_id=None, start=None, travel_in=None) for k in range(nv)]
    if sol is None:
        return pd.DataFrame(out)
    for v in range(nc):
        idx = routing.Start(v)
        prev = manager.IndexToNode(idx)
        while not routing.IsEnd(idx):
            idx = sol.Value(routing.NextVar(idx))
            node = manager.IndexToNode(idx)
            if node >= nc:
                k = node - nc
                out[k].update(carer_id=carers.carer_id[v], start=sol.Min(time_dim.CumulVar(idx)),
                              travel_in=int(travel_min[prev, node]))
            prev = node
    return pd.DataFrame(out)


def greedy_plan(carers: pd.DataFrame, visits: pd.DataFrame, travel_min: np.ndarray,
                prefer_regular: bool = False) -> pd.DataFrame:
    """Baseline: visits in time order, each to the nearest carer who can reach it in time.

    With prefer_regular=True a regular carer is chosen whenever one can make it,
    which is how a careful coordinator usually works.
    """
    carers = carers.reset_index(drop=True)
    visits = visits.reset_index(drop=True)
    nc = len(carers)
    pos = list(range(nc))
    free_at = carers.shift_start.astype(int).tolist()
    count = [0] * nc
    out = []
    for k in visits.sort_values("planned").index:
        vis = visits.loc[k]
        node = nc + k
        best = None
        for v in range(nc):
            if count[v] >= carers.max_visits[v] or not eligible(carers.loc[v], vis):
                continue
            arrive = max(free_at[v] + travel_min[pos[v], node], vis.earliest)
            back_home = arrive + vis.duration + travel_min[node, v]
            if arrive > vis.latest or back_home > carers.shift_end[v]:
                continue
            not_regular = carers.carer_id[v] not in vis.regular_carers
            key = ((not_regular if prefer_regular else 0), travel_min[pos[v], node], arrive)
            if best is None or key < best[0]:
                best = (key, v, arrive)
        if best is None:
            out.append(dict(visit_id=vis.visit_id, carer_id=None, start=None, travel_in=None))
            continue
        _, v, arrive = best
        out.append(dict(visit_id=vis.visit_id, carer_id=carers.carer_id[v], start=int(arrive),
                        travel_in=int(travel_min[pos[v], node])))
        free_at[v] = int(arrive + vis.duration)
        pos[v] = node
        count[v] += 1
    return pd.DataFrame(out)


def evaluate(plan: pd.DataFrame, carers: pd.DataFrame, visits: pd.DataFrame, travel_min: np.ndarray) -> dict:
    """Score a plan. Travel includes each carer's drive home at the end of the shift."""
    carers = carers.reset_index(drop=True)
    visits = visits.reset_index(drop=True)
    nc = len(carers)
    m = plan.merge(visits[["visit_id", "planned", "regular_carers"]], on="visit_id")
    done = m[m.carer_id.notna()].copy()
    home = {c: i for i, c in enumerate(carers.carer_id)}
    vnode = {vid: nc + i for i, vid in enumerate(visits.visit_id)}
    travel_home = 0
    for cid, g in done.sort_values("start").groupby("carer_id"):
        travel_home += int(travel_min[vnode[g.visit_id.iloc[-1]], home[cid]])
    per_carer = done.groupby("carer_id").size().reindex(carers.carer_id, fill_value=0)
    cont = done.apply(lambda r: r.carer_id in r.regular_carers, axis=1)
    return dict(
        visits=len(m), assigned=int(len(done)), unassigned=int(m.carer_id.isna().sum()),
        travel_minutes=int(done.travel_in.sum() + travel_home),
        travel_per_visit=round(float((done.travel_in.sum() + travel_home) / max(len(done), 1)), 1),
        continuity_pct=round(100 * float(cont.mean()), 1),
        mean_minutes_from_planned=round(float((done.start - done.planned).abs().mean()), 1),
        within_15_min_pct=round(100 * float(((done.start - done.planned).abs() <= 15).mean()), 1),
        carers_used=int((per_carer > 0).sum()),
        max_visits_per_carer=int(per_carer.max()),
        visits_per_carer_sd=round(float(per_carer[per_carer > 0].std()), 2),
    )


def check_hard_rules(plan: pd.DataFrame, carers: pd.DataFrame, visits: pd.DataFrame, travel_min: np.ndarray) -> list[str]:
    """Independent check that no plan breaks skills, preferences, time windows or overlaps."""
    errors = []
    carers = carers.reset_index(drop=True)
    visits = visits.reset_index(drop=True)
    nc = len(carers)
    cidx = {c: i for i, c in enumerate(carers.carer_id)}
    vidx = {v: i for i, v in enumerate(visits.visit_id)}
    done = plan[plan.carer_id.notna()]
    for cid, g in done.groupby("carer_id"):
        c = carers.loc[cidx[cid]]
        g = g.sort_values("start")
        prev_node, prev_end = cidx[cid], int(c.shift_start)
        for r in g.itertuples():
            k = vidx[r.visit_id]
            vis = visits.loc[k]
            if not eligible(c, vis):
                errors.append(f"{r.visit_id}: {cid} not eligible")
            if not (vis.earliest <= r.start <= vis.latest):
                errors.append(f"{r.visit_id}: start {r.start} outside window")
            if r.start < prev_end + travel_min[prev_node, nc + k]:
                errors.append(f"{r.visit_id}: {cid} cannot get there in time")
            prev_node, prev_end = nc + k, r.start + vis.duration
        if prev_end + travel_min[prev_node, cidx[cid]] > c.shift_end:
            errors.append(f"{cid}: finishes after shift end")
        if len(g) > c.max_visits:
            errors.append(f"{cid}: too many visits")
    return errors


def weights_dict(w: Weights) -> dict:
    return asdict(w)
