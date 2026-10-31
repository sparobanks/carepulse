"""Run every experiment and write reports/results.json, reports/RESULTS.md and charts.

    python scripts/run_all.py            # full run (about 4 minutes)
    python scripts/run_all.py --quick    # short solver limits, for a fast check
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from carepulse.data.notes_synth import make_notes  # noqa: E402
from carepulse.data.synth import make_retention_panel, make_visits, make_workforce  # noqa: E402
from carepulse.geo.travel import default_provider  # noqa: E402
from carepulse.matching.engine import Weights, check_hard_rules, evaluate, greedy_plan, solve  # noqa: E402
from carepulse.notes.evaluate import run as run_notes  # noqa: E402
from carepulse.retention import model as rm  # noqa: E402

OUT = ROOT / "reports"
INK, MUTED, GRID = "#1d2321", "#5d6763", "#dde2df"
C1, C2, C3 = "#1f7a5c", "#b8641a", "#3b5ba5"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def matching(quick: bool) -> dict:
    limit = 4 if quick else 20
    wf = make_workforce()
    travel = default_provider(ROOT / "data" / "birmingham_drive.graphml")
    days = []
    for day in range(2 if quick else 5):
        v = make_visits(wf.clients, day=day)
        lat = np.concatenate([wf.carers.lat, v.lat])
        lon = np.concatenate([wf.carers.lon, v.lon])
        T = travel.matrix(lat, lon)
        res = {}
        for name, plan in [("nearest_free_carer", greedy_plan(wf.carers, v, T)),
                           ("regular_carer_first", greedy_plan(wf.carers, v, T, prefer_regular=True)),
                           ("optimised", solve(wf.carers, v, T, Weights(), limit))]:
            errs = check_hard_rules(plan, wf.carers, v, T)
            assert not errs, errs[:3]
            res[name] = evaluate(plan, wf.carers, v, T)
        days.append(res)
        print(f"  day {day}: {res['optimised']}")
    mean = {k: {m: round(float(np.mean([d[k][m] for d in days])), 1) for m in days[0][k]} for k in days[0]}

    # trade-off: how much travel does each extra point of continuity cost?
    v = make_visits(wf.clients, day=0)
    T = travel.matrix(np.concatenate([wf.carers.lat, v.lat]), np.concatenate([wf.carers.lon, v.lon]))
    sweep = []
    for cw in [0, 5, 15, 30, 60]:
        e = evaluate(solve(wf.carers, v, T, Weights(continuity=cw), limit), wf.carers, v, T)
        sweep.append(dict(continuity_weight=cw, **e))
        print(f"  continuity weight {cw}: travel {e['travel_minutes']}, continuity {e['continuity_pct']}%")

    fig, ax = plt.subplots(figsize=(6.4, 4))
    style(ax)
    ax.plot([s["continuity_pct"] for s in sweep], [s["travel_minutes"] for s in sweep], "-o", color=C1, lw=2)
    for s in sweep:
        ax.annotate(f"weight {s['continuity_weight']}", (s["continuity_pct"], s["travel_minutes"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8, color=MUTED)
    b = days[0]["nearest_free_carer"]
    ax.plot(b["continuity_pct"], b["travel_minutes"], "s", color=C2, ms=8)
    ax.annotate("nearest free carer\n(baseline, misses visits)", (b["continuity_pct"], b["travel_minutes"]),
                textcoords="offset points", xytext=(8, -18), fontsize=8, color=C2)
    ax.set_xlabel("Visits by one of the client's regular carers (%)", color=INK)
    ax.set_ylabel("Total travel, morning shift (minutes)", color=INK)
    ax.set_title("More continuity costs more travel, with little gain past weight 30", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "matching_tradeoff.png", dpi=150)
    plt.close(fig)
    return dict(travel_times=travel.name, days=len(days), visits_per_day=int(len(v)), carers=int(len(wf.carers)),
                solver_seconds=limit, mean=mean, continuity_sweep=sweep)


def retention() -> dict:
    df = make_retention_panel()
    s = rm.time_split(df)
    y = s.test[rm.TARGET].values
    out = dict(snapshots=dict(train=len(s.train), calibration=len(s.calib), test=len(s.test)),
               carers=int(df.carer_id.nunique()))
    models = {"logistic_regression": rm.fit_logistic(s.train), "lightgbm": rm.fit_lgbm(s.train)}
    preds = {}
    for name, m in models.items():
        cal = rm.CalibratedModel(m, s.calib)
        preds[name] = cal.predict(s.test)
        out[name] = rm.metrics(y, preds[name])
        print(f"  {name}: {out[name]}")
    p = preds["lightgbm"]
    thr = float(np.quantile(p, 0.9))
    out["fairness_top10pct"] = rm.fairness(s.test, p, thr).to_dict("records")

    import warnings

    import shap
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = shap.TreeExplainer(models["lightgbm"]).shap_values(s.test[rm.FEATURES])
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(axis=0), index=rm.FEATURES).sort_values(ascending=False)
    out["mean_abs_shap"] = {k: round(float(v), 4) for k, v in imp.items()}
    out["example"] = dict(risk=round(float(p[np.argmax(p)]), 3),
                          reasons=rm.explain(models["lightgbm"], s.test.iloc[[int(np.argmax(p))]], s.train)[0])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    style(ax)
    top_bin = 0.0
    for name, col in [("lightgbm", C1), ("logistic_regression", C3)]:
        q = preds[name]
        bins = np.array_split(np.argsort(q), 10)
        xs, ys = [q[b].mean() for b in bins], [y[b].mean() for b in bins]
        top_bin = max(top_bin, max(xs), max(ys))
        ax.plot(xs, ys, "-o", color=col, ms=4, label=name.replace("_", " "))
    lim = round(top_bin * 1.15, 2)
    ax.plot([0, lim], [0, lim], "--", color=MUTED, lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Predicted chance of leaving in 90 days", color=INK)
    ax.set_ylabel("Share who actually left", color=INK)
    ax.set_title("Calibration on later months (test), 10 equal-size groups", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    style(ax)
    top = imp.head(8)[::-1]
    ax.barh([t.replace("_", " ") for t in top.index], top.values, color=C1)
    ax.set_xlabel("Average impact on risk (mean |SHAP|)", color=INK)
    ax.set_title("What drives the risk score", fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "retention.png", dpi=150)
    plt.close(fig)
    return out


def table(rows, cols, names=None):
    names = names or cols
    head = "| " + " | ".join(names) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    return head + "".join("| " + " | ".join(str(r[c]) for c in cols) + " |\n" for r in rows)


def write_markdown(r: dict):
    m = r["matching"]
    lines = [f"# Results\n\nGenerated by `scripts/run_all.py` on synthetic data. Re-run it to reproduce every number.\n",
             "## 1. Visit matching\n",
             f"Travel times: **{m['travel_times']}**. {m['carers']} carers, about {m['visits_per_day']} visits per morning shift, "
             f"averaged over {m['days']} days. Solver time limit {m['solver_seconds']} s.\n"]
    rows = [dict(plan=k.replace("_", " "), **v) for k, v in m["mean"].items()]
    lines.append(table(rows, ["plan", "assigned", "unassigned", "travel_minutes", "travel_per_visit", "continuity_pct",
                              "within_15_min_pct", "max_visits_per_carer"],
                       ["Plan", "Visits covered", "Not covered", "Travel (min)", "Travel per visit", "Regular carer %",
                        "Within 15 min %", "Most visits, one carer"]))
    lines.append("\n![trade-off](matching_tradeoff.png)\n")
    lines.append(table(m["continuity_sweep"], ["continuity_weight", "travel_minutes", "continuity_pct", "within_15_min_pct"],
                       ["Continuity weight", "Travel (min)", "Regular carer %", "Within 15 min %"]))
    t = r["retention"]
    lines += ["\n## 2. Retention risk\n",
              f"Train on months 0 to 11, calibrate on 14 to 15, test on 18 to 21 (gaps stop label leakage). "
              f"{t['snapshots']['test']} test snapshots.\n"]
    lines.append(table([dict(model=k.replace("_", " "), **t[k]) for k in ["logistic_regression", "lightgbm"]],
                       ["model", "roc_auc", "pr_auc", "brier", "ece", "base_rate", "top10_precision", "top10_share_of_leavers"],
                       ["Model", "ROC AUC", "PR AUC", "Brier", "Calibration error", "Leave rate", "Precision in top 10%",
                        "Leavers caught in top 10%"]))
    lines.append("\nFairness at the top-10% flag line (sex and age are not model inputs):\n")
    lines.append(table(t["fairness_top10pct"], ["attribute", "group", "n", "actual_leave_rate", "mean_predicted", "roc_auc",
                                                "flag_rate", "leavers_caught"],
                       ["Attribute", "Group", "n", "Actual leave rate", "Mean predicted", "ROC AUC", "Flag rate",
                        "Leavers caught"]))
    lines.append("\n![retention](retention.png)\n")
    n = r["notes"]
    lines += ["\n## 3. Care notes\n",
              f"{n['n_train']} training notes, {n['n_test']} test notes. Test notes use sentence families never seen in training.\n",
              "\n**Rules (default pipeline).** Written with knowledge of the note generator, so read these as a check that "
              "the logic works, not as real-world accuracy.\n"]
    lines.append(table(n["rules"], ["category", "positives", "precision", "recall", "f1"]))
    lines.append("\n**Learned classifier (TF-IDF + logistic regression), on unseen phrasings.**\n")
    lines.append(table(n["classifier"], ["category", "positives", "precision", "recall", "f1"]))
    lines.append(f"\nThresholds tuned on held-out training families: `{n['thresholds']}` "
                 "(safeguarding tuned for at least 95% recall).\n")
    lines.append("\n**Routing with rules only:**\n\n```\n" + json.dumps(n["rules_routing"], indent=2) + "\n```\n")
    lines.append("\n**Routing with rules plus classifier:**\n\n```\n" + json.dumps(n["rules_plus_classifier_routing"], indent=2) + "\n```\n")
    lines.append("\n**Redaction:** `" + json.dumps(n["redaction"]) + "`\n")
    (OUT / "RESULTS.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    print("matching...")
    r = dict(matching=matching(a.quick))
    print("retention...")
    r["retention"] = retention()
    print("notes...")
    r["notes"], _ = run_notes(make_notes())
    r["runtime_seconds"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(r, indent=2, default=str))
    write_markdown(r)
    print(f"done in {r['runtime_seconds']} s")


if __name__ == "__main__":
    main()
