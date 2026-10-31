import json

from carepulse.common.audit import AuditLog


def test_chain_verifies_and_detects_tampering(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    for i in range(3):
        log.append("event", "manager", {"i": i})
    assert log.verify() == {"ok": True, "entries": 3}
    lines = log.path.read_text().splitlines()
    e = json.loads(lines[1])
    e["payload"]["i"] = 99
    lines[1] = json.dumps(e, sort_keys=True)
    log.path.write_text("\n".join(lines) + "\n")
    assert log.verify()["ok"] is False and log.verify()["broken_at"] == 1
