"""Persistent diagnostic tickets submitted by a playing browser."""
from __future__ import annotations

import json
import secrets
import threading
import time
from pathlib import Path

MAX_BYTES = 96 * 1024
STATUSES = {"open", "investigating", "resolved"}


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.Lock()

    def submit(self, report: dict, *, peer: str, tier: str, server_state: dict, log_seq: int,
               server_log: list[dict] | None = None) -> dict:
        if not isinstance(report, dict):
            raise ValueError("report must be an object")
        body = json.dumps(report, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(body) > MAX_BYTES:
            raise ValueError("report is too large")
        now = time.time()
        ticket = {"id": f"{int(now)}-{secrets.token_hex(5)}", "created": now, "updated": now,
                  "status": "open", "peer": peer, "tier": tier, "log_seq": log_seq,
                  "server_state": server_state, "server_log": server_log or [], "report": report, "handling_note": ""}
        with self.lock:
            self.path.mkdir(parents=True, exist_ok=True)
            target = self.path / (ticket["id"] + ".json")
            target.write_text(json.dumps(ticket, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        return {"id": ticket["id"], "status": "open"}

    def list(self) -> list[dict]:
        with self.lock:
            out = []
            for path in sorted(self.path.glob("*.json"), reverse=True) if self.path.exists() else []:
                try:
                    ticket = json.loads(path.read_text(encoding="utf-8"))
                    out.append({k: ticket[k] for k in ("id", "created", "updated", "status", "peer", "tier", "log_seq")})
                    out[-1]["note"] = str(ticket.get("report", {}).get("note", ""))[:160]
                except (OSError, ValueError, KeyError, TypeError):
                    continue
            return out

    def get(self, ticket_id: str) -> dict:
        if not self._valid_id(ticket_id):
            raise ValueError("invalid ticket id")
        with self.lock:
            path = self.path / (ticket_id + ".json")
            if not path.is_file():
                raise FileNotFoundError(ticket_id)
            return json.loads(path.read_text(encoding="utf-8"))

    def update(self, ticket_id: str, status: str, note: str) -> dict:
        if not self._valid_id(ticket_id) or status not in STATUSES or not isinstance(note, str) or len(note) > 4000:
            raise ValueError("invalid status update")
        with self.lock:
            path = self.path / (ticket_id + ".json")
            if not path.is_file():
                raise FileNotFoundError(ticket_id)
            ticket = json.loads(path.read_text(encoding="utf-8"))
            ticket.update(status=status, handling_note=note, updated=time.time())
            temp = self.path / (ticket_id + ".tmp")
            temp.write_text(json.dumps(ticket, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            temp.replace(path)
            return {"id": ticket_id, "status": status}

    @staticmethod
    def _valid_id(value: str) -> bool:
        parts = value.split("-", 1) if isinstance(value, str) else []
        return len(parts) == 2 and parts[0].isdigit() and len(parts[1]) == 10 and all(c in "0123456789abcdef" for c in parts[1])
