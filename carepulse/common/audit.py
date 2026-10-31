"""Tamper-evident audit log.

Each entry stores the hash of the entry before it, so changing or deleting
any past entry breaks the chain and `verify()` reports where. The log keeps
hashes of note text rather than the text itself, so it holds no care details.
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "0" * 64


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return GENESIS
        last = self.path.read_text().strip().splitlines()[-1]
        return json.loads(last)["hash"]

    def append(self, event: str, actor: str, payload: dict) -> dict:
        with self._lock:
            entry = dict(time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         event=event, actor=actor, payload=payload, prev=self._last_hash())
            entry["hash"] = sha256(json.dumps(entry, sort_keys=True))
            with self.path.open("a") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
            return entry

    def verify(self) -> dict:
        if not self.path.exists():
            return dict(ok=True, entries=0)
        prev = GENESIS
        lines = self.path.read_text().strip().splitlines()
        for i, line in enumerate(lines):
            e = json.loads(line)
            h = e.pop("hash")
            if e["prev"] != prev or sha256(json.dumps(e, sort_keys=True)) != h:
                return dict(ok=False, entries=len(lines), broken_at=i)
            prev = h
        return dict(ok=True, entries=len(lines))
