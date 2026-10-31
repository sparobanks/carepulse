"""CarePulse API.

Role-based access with API keys:
  coordinator : plan visits, analyse notes, see the review queue
  manager     : everything a coordinator can, plus retention risk and review decisions
  auditor     : read-only check of the audit log

Keys come from the CAREPULSE_KEYS environment variable ("key1:manager,key2:coordinator").
Demo keys are only switched on when CAREPULSE_DEMO=1.
All data here is synthetic.
"""
from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..common.audit import AuditLog, sha256
from ..data.synth import make_retention_panel, make_visits, make_workforce
from ..geo.travel import default_provider
from ..matching.engine import Weights, evaluate, greedy_plan, solve
from ..notes.pipeline import NotesPipeline
from ..retention import model as rm

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("CAREPULSE_DATA", ROOT / "data"))
DEMO_KEYS = {"demo-coordinator": "coordinator", "demo-manager": "manager", "demo-auditor": "auditor"}


def load_keys() -> dict[str, str]:
    keys = {}
    for pair in filter(None, os.environ.get("CAREPULSE_KEYS", "").split(",")):
        k, _, role = pair.partition(":")
        keys[k.strip()] = role.strip()
    if os.environ.get("CAREPULSE_DEMO") == "1":
        keys.update(DEMO_KEYS)
    return keys


class State:
    pass


S = State()


def _build_state():
    DATA.mkdir(parents=True, exist_ok=True)
    S.keys = load_keys()
    S.audit = AuditLog(DATA / "audit.jsonl")
    S.db = sqlite3.connect(DATA / "reviews.db", check_same_thread=False)
    S.db_lock = threading.Lock()
    S.db.execute("""create table if not exists reviews(
        id integer primary key, note_id text, route text, findings text, redacted_text text,
        status text default 'open', decision text, reviewer_role text)""")
    S.db.commit()
    wf = make_workforce()
    S.carers, S.clients = wf.carers, wf.clients
    S.visits = make_visits(wf.clients, day=0)
    S.travel = default_provider(DATA / "birmingham_drive.graphml")
    lat = np.concatenate([S.carers.lat, S.visits.lat])
    lon = np.concatenate([S.carers.lon, S.visits.lon])
    S.T = S.travel.matrix(lat, lon)
    panel = make_retention_panel()
    split = rm.time_split(panel)
    S.ret_model = rm.fit_lgbm(split.train)
    S.ret_cal = rm.CalibratedModel(S.ret_model, split.calib)
    S.ret_ref = split.train
    S.ret_now = panel[panel.month == panel.month.max()].reset_index(drop=True)
    S.notes = NotesPipeline([])
    if os.environ.get("CAREPULSE_LLM"):
        from ..notes.llm import OllamaBackend
        S.notes = NotesPipeline([OllamaBackend(model=os.environ["CAREPULSE_LLM"])])


@asynccontextmanager
async def lifespan(app):
    _build_state()
    yield
    S.db.close()


app = FastAPI(title="CarePulse", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Cache-Control"] = "no-store"
    return resp


def role(*allowed: str):
    def check(request: Request) -> str:
        key = request.headers.get("X-API-Key", "")
        found = next((r for k, r in S.keys.items() if secrets.compare_digest(k, key)), None)
        if found is None:
            raise HTTPException(401, "missing or unknown API key")
        if found not in allowed:
            S.audit.append("access_denied", found, {"path": request.url.path})
            raise HTTPException(403, f"role '{found}' cannot use this")
        return found
    return check


@app.get("/")
def dashboard():
    return FileResponse(ROOT / "dashboard" / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "travel_times": S.travel.name}


# ------------------------------------------------------------------ matching
class PlanRequest(BaseModel):
    travel: int = Field(1, ge=0, le=10)
    continuity: int = Field(15, ge=0, le=120)
    balance: int = Field(10, ge=0, le=100)
    punctuality: int = Field(1, ge=0, le=10)
    time_limit_s: int = Field(5, ge=1, le=30)


@app.post("/match/plan")
def match_plan(req: PlanRequest, who: str = Depends(role("coordinator", "manager"))):
    w = Weights(req.travel, req.continuity, req.balance, req.punctuality)
    plan = solve(S.carers, S.visits, S.T, w, req.time_limit_s)
    base = greedy_plan(S.carers, S.visits, S.T)
    m = plan.merge(S.visits[["visit_id", "client_id", "lat", "lon", "planned", "duration"]], on="visit_id")
    runs = {}
    for cid, g in m[m.carer_id.notna()].sort_values("start").groupby("carer_id"):
        home = S.carers.set_index("carer_id").loc[cid]
        runs[cid] = dict(home=[float(home.lat), float(home.lon)],
                         stops=[dict(visit_id=r.visit_id, client_id=r.client_id, lat=float(r.lat), lon=float(r.lon),
                                     start=int(r.start), planned=int(r.planned), duration=int(r.duration))
                                for r in g.itertuples()])
    S.audit.append("plan_created", who, {"weights": req.model_dump(), "visits": len(S.visits)})
    return {"travel_times": S.travel.name, "optimised": evaluate(plan, S.carers, S.visits, S.T),
            "baseline_nearest_free_carer": evaluate(base, S.carers, S.visits, S.T), "runs": runs}


# ------------------------------------------------------------------ retention
@app.get("/retention/risk")
def retention_risk(limit: int = 20, who: str = Depends(role("manager"))):
    df = S.ret_now
    p = S.ret_cal.predict(df)
    top = np.argsort(-p)[:max(1, min(limit, 100))]
    sub = df.iloc[top]
    reasons = rm.explain(S.ret_model, sub, S.ret_ref)
    S.audit.append("retention_viewed", who, {"rows": len(top)})
    return {"note": "For supportive conversations only. Not for disciplinary or selection decisions.",
            "people": [dict(carer_id=r.carer_id, risk=round(float(p[i]), 3), band=rm.risk_band(float(p[i])), reasons=rs)
                       for i, r, rs in zip(top, sub.itertuples(), reasons)]}


# ------------------------------------------------------------------ notes
class NoteIn(BaseModel):
    note_id: str = Field(..., max_length=64)
    text: str = Field(..., min_length=1, max_length=5000)
    client_name: str | None = Field(None, max_length=100)


@app.post("/notes/analyse")
def notes_analyse(note: NoteIn, who: str = Depends(role("coordinator", "manager"))):
    names = {note.client_name: "[CLIENT]"} if note.client_name else None
    res = S.notes.analyse(note.note_id, note.text, names)
    out = res.to_dict()
    if res.route in ("urgent_review", "review"):
        with S.db_lock:
            cur = S.db.execute("insert into reviews(note_id, route, findings, redacted_text) values (?,?,?,?)",
                               (note.note_id, res.route, json.dumps(out["findings"]), res.redacted_text))
            S.db.commit()
        out["review_id"] = cur.lastrowid
    S.audit.append("note_analysed", who, {"note_id": note.note_id, "text_sha256": sha256(note.text),
                                          "route": res.route, "categories": [f.category for f in res.findings],
                                          "backends": res.backends, "version": S.notes.version})
    return out


@app.get("/review/queue")
def review_queue(who: str = Depends(role("coordinator", "manager"))):
    rows = S.db.execute("select id, note_id, route, findings, redacted_text from reviews where status='open' "
                        "order by case route when 'urgent_review' then 0 else 1 end, id").fetchall()
    return [dict(id=i, note_id=n, route=r, findings=json.loads(f), redacted_text=t) for i, n, r, f, t in rows]


class Decision(BaseModel):
    decision: str = Field(..., pattern="^(confirmed|not_a_concern|escalated)$")
    comment: str = Field("", max_length=1000)


@app.post("/review/{review_id}/decision")
def review_decision(review_id: int, d: Decision, who: str = Depends(role("manager"))):
    with S.db_lock:
        cur = S.db.execute("update reviews set status='closed', decision=?, reviewer_role=? where id=? and status='open'",
                           (d.decision, who, review_id))
        S.db.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, "no open review with that id")
    S.audit.append("review_decided", who, {"review_id": review_id, "decision": d.decision,
                                           "comment_sha256": sha256(d.comment)})
    return {"ok": True}


@app.get("/audit/verify")
def audit_verify(who: str = Depends(role("auditor", "manager"))):
    return S.audit.verify()
