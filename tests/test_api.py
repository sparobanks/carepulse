import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    os.environ["CAREPULSE_DEMO"] = "1"
    os.environ["CAREPULSE_DATA"] = str(tmp_path_factory.mktemp("data"))
    import importlib
    import carepulse.api.app as app_mod
    importlib.reload(app_mod)
    with TestClient(app_mod.app) as c:
        yield c


C = {"X-API-Key": "demo-coordinator"}
M = {"X-API-Key": "demo-manager"}
A = {"X-API-Key": "demo-auditor"}


def test_needs_a_key(client):
    assert client.get("/review/queue").status_code == 401


def test_roles_are_enforced(client):
    assert client.get("/retention/risk", headers=C).status_code == 403
    assert client.get("/retention/risk", headers=A).status_code == 403
    r = client.get("/retention/risk?limit=5", headers=M)
    assert r.status_code == 200 and len(r.json()["people"]) == 5


def test_note_flow_review_and_audit(client):
    r = client.post("/notes/analyse", headers=C, json={
        "note_id": "t1", "client_name": "Mrs Begum",
        "text": "Mrs Begum told me she does not feel safe when her nephew is around."})
    body = r.json()
    assert body["route"] == "urgent_review" and "Begum" not in body["redacted_text"]
    rid = body["review_id"]
    assert any(x["id"] == rid for x in client.get("/review/queue", headers=C).json())
    assert client.post(f"/review/{rid}/decision", headers=C, json={"decision": "confirmed"}).status_code == 403
    assert client.post(f"/review/{rid}/decision", headers=M, json={"decision": "confirmed"}).status_code == 200
    assert client.post(f"/review/{rid}/decision", headers=M, json={"decision": "confirmed"}).status_code == 404
    v = client.get("/audit/verify", headers=A).json()
    assert v["ok"] and v["entries"] >= 3


def test_plan_endpoint(client):
    r = client.post("/match/plan", headers=C, json={"time_limit_s": 2})
    d = r.json()
    assert r.status_code == 200 and d["optimised"]["assigned"] > 0 and d["runs"]


def test_input_limits(client):
    assert client.post("/notes/analyse", headers=C, json={"note_id": "x", "text": ""}).status_code == 422
    assert client.post("/match/plan", headers=C, json={"continuity": 9999}).status_code == 422
