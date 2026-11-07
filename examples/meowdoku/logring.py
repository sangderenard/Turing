"""[LOG] A size-capped in-memory ring of the server's text log (and the clients' error beacons), fetched incrementally by sequence.

Every line has a ``seq`` (monotonic for the process, continuing across restarts when the ring is persisted), a timestamp and a source:
``client-error`` (a page's uncaught error), ``fountain-diag`` (an owner device's 5 s diagnostics) or ``server`` (everything else the
server prints).  The cap is in BYTES; the oldest lines are dropped first.  ``tee_stdout`` copies what the server prints into the ring
without changing what reaches stdout.  The ring may hold IPs and user agents: ``GET /api/log`` is for the dev tier and above only.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

DEFAULT_CAP = 2 * 1024 * 1024
LINE_CAP = 4000                  # one line's text is cut here (a stack trace stays a line)
OVERHEAD = 48                    # per-entry bytes counted besides the text (seq, timestamp, source)
SOURCES = ("client-error", "fountain-diag", "server")


def classify(text: str) -> str:
    if text.startswith("[client-error]"):
        return "fountain-diag" if ": fountain-diag |" in text[:80] else "client-error"
    return "server"


class Ring:
    def __init__(self, cap_bytes: int = DEFAULT_CAP, path: Path | None = None):
        self.cap, self.path, self.lock = int(cap_bytes), (Path(path) if path else None), threading.Lock()
        self.entries: deque = deque()
        self.bytes, self.seq, self._written = 0, 0, 0
        if self.path is not None:
            self._load()

    @staticmethod
    def _cost(text: str) -> int:
        return len(text.encode("utf-8", "replace")) + OVERHEAD

    def add(self, text: str, source: str | None = None, ts: float | None = None) -> int:
        text = str(text).replace("\r", "")[:LINE_CAP]
        source = source if source in SOURCES else classify(text)
        with self.lock:
            self.seq += 1
            entry = {"seq": self.seq, "ts": round(ts if ts is not None else time.time(), 3), "src": source, "text": text}
            self.entries.append(entry)
            self.bytes += self._cost(text)
            while self.bytes > self.cap and len(self.entries) > 1:
                self.bytes -= self._cost(self.entries.popleft()["text"])
            if self.path is not None:
                self._persist(entry)
            return self.seq

    def fetch(self, since: int = 0, limit: int = 500, sources=None) -> dict:
        """Entries with ``seq > since`` (oldest first, at most ``limit``); ``first`` is the oldest seq still held, ``last`` the newest issued."""
        limit = max(1, min(int(limit), 5000))
        with self.lock:
            out = []
            for e in self.entries:
                if e["seq"] > since and (not sources or e["src"] in sources):
                    out.append(e)
                    if len(out) >= limit:
                        break
            return {"entries": out, "first": self.entries[0]["seq"] if self.entries else self.seq + 1, "last": self.seq,
                    "bytes": self.bytes, "cap": self.cap}

    # ---- persistence (optional, capped): one JSON line per entry, rewritten from the ring when it outgrows twice the cap
    def _load(self) -> None:
        try:
            for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines()[-100000:]:
                e = json.loads(line)
                self.entries.append(e); self.bytes += self._cost(e["text"]); self.seq = max(self.seq, e["seq"])
            while self.bytes > self.cap and len(self.entries) > 1:
                self.bytes -= self._cost(self.entries.popleft()["text"])
        except (OSError, ValueError, KeyError):
            pass
        self._written = self.bytes

    def _persist(self, entry: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self._written > 2 * self.cap:
                self.path.write_text("".join(json.dumps(e) + "\n" for e in self.entries), encoding="utf-8")
                self._written = self.bytes
            else:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
                self._written += self._cost(entry["text"])
        except OSError:
            pass


RING = Ring()


class _Tee:
    """Passes everything through to the real stream and files each complete line into the ring."""
    def __init__(self, real):
        self._real, self._buf, self._lock = real, "", threading.Lock()

    def write(self, s):
        n = self._real.write(s)
        with self._lock:
            self._buf += s
            *lines, self._buf = self._buf.split("\n")
        for line in lines:
            if line.strip():
                RING.add(line)
        return n

    def __getattr__(self, name):
        return getattr(self._real, name)


def tee_stdout() -> None:
    if not isinstance(sys.stdout, _Tee):
        sys.stdout = _Tee(sys.stdout)
