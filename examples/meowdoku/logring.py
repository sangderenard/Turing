"""[LOG] A size-capped in-memory ring of the server's text log (and the clients' error beacons), fetched incrementally by sequence.

Every line has a ``seq`` (monotonic for the process, continuing across restarts when the ring is persisted), a timestamp and a source:
``client-error`` (a page's uncaught error), ``fountain-diag`` (an owner device's 5 s diagnostics), ``login`` (each sign-in outcome and
sign-out, filed by ``note_login``), ``activity`` (an account's first request after ``IDLE_SECONDS`` quiet, filed by ``note_activity``)
or ``server`` (everything else the server prints).  The cap is in BYTES; the oldest lines are dropped first.  ``tee_stdout`` copies what the server prints into the ring
without changing what reaches stdout.  [PRIVATE] Every line passes through ``private_record.redact`` on its way in (and a legacy
persisted line on load), so the ring holds tokens, never an IP or email address (private_record.py).  The persisted ring is
sealed with the records key (secure_records.py), and ``GET /api/log`` needs private_read (operators, admins) over HTTPS.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import sys
import threading
import time
from collections import deque
from pathlib import Path

from meowdoku import private_record
from meowdoku.secure_records import RecordCipher

DEFAULT_CAP = 2 * 1024 * 1024
LINE_CAP = 4000                  # one line's text is cut here (a stack trace stays a line)
OVERHEAD = 48                    # per-entry bytes counted besides the text (seq, timestamp, source)
SOURCES = ("client-error", "fountain-diag", "login", "activity", "server")
IDLE_SECONDS = 600.0             # an account quiet this long is "active" again on its next request (one ``activity`` line, not one per request)


def classify(text: str) -> str:
    if text.startswith("[client-error]"):
        return "fountain-diag" if ": fountain-diag |" in text[:80] else "client-error"
    return "server"


class Ring:
    def __init__(self, cap_bytes: int = DEFAULT_CAP, path: Path | None = None, key_path: Path | None = None):
        self.cap, self.path, self.lock = int(cap_bytes), (Path(path) if path else None), threading.Lock()
        if self.path is not None and key_path is None:
            raise ValueError("persistent log ring requires a records key")
        self.cipher = RecordCipher(key_path, existing_streams=(self.path,)) if key_path is not None else None
        self.entries: deque = deque()
        self.bytes, self.seq, self._written = 0, 0, 0
        if self.path is not None:
            self._load()

    @staticmethod
    def _cost(text: str) -> int:
        return len(text.encode("utf-8", "replace")) + OVERHEAD

    def add(self, text: str, source: str | None = None, ts: float | None = None) -> int:
        text = private_record.redact(str(text).replace("\r", "")[:LINE_CAP])[:LINE_CAP]   # [PRIVATE] tokens, never an address
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
        """Reloads the persisted ring.  A line that does not open (an append cut off by a kill, a damaged rewrite) is skipped and
        counted, never fatal: one bad line must not stop the server from starting.  When any were skipped, the file as found is kept
        beside the ring as ``<name>.bad-<time>`` evidence, the good lines are rewritten atomically, and the ring files one line saying so."""
        self.cipher.migrate(self.path)
        bad, ends_open = 0, False
        try:
            with self.path.open("rb") as saved:
                lines = saved.readlines()[-100000:]
            ends_open = bool(lines) and not lines[-1].endswith(b"\n")
            with private_record.batch():                              # [BATCH] one transaction for the whole reload
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        e = json.loads(self.cipher.open(line))
                        e["text"] = private_record.redact(e["text"])  # a line persisted before [PRIVATE] holds raw addresses
                        seq = int(e["seq"])
                    except (ValueError, KeyError, TypeError):        # ValueError covers a failed cipher.open and bad JSON
                        bad += 1
                        continue
                    self.entries.append(e); self.bytes += self._cost(e["text"]); self.seq = max(self.seq, seq)
            while self.bytes > self.cap and len(self.entries) > 1:
                self.bytes -= self._cost(self.entries.popleft()["text"])
        except FileNotFoundError:
            pass
        self._written = self.bytes
        if bad:
            self._evidence_pending = True
            self._needs_separator = ends_open      # an append after a cut-off tail must not fuse with it into one unreadable line
            evidence = self._keep_evidence()
            if evidence is not None:
                try:
                    self._rewrite()
                except OSError:
                    pass                                              # the bad lines stay on disk (the evidence is kept), skipped at every load
            self.add(f"[logring] skipped {bad} unreadable line(s) of {self.path.name} at load; "
                     + (f"the file as found is kept as {evidence.name}" if evidence is not None
                        else "the evidence copy could not be written, so the file is left as found and only appended to"))

    _evidence_pending = False   # unreadable lines are on disk and no evidence copy of them exists yet: the file must not be rewritten
    _needs_separator = False    # the file ends in a cut-off line: the next append first writes a newline

    def _keep_evidence(self) -> Path | None:
        """Copies the file as found to ``<name>.bad-<time>`` (``-1``, ``-2`` ... when that name is taken).  Each attempt reserves a
        name of its own (O_EXCL), so it never overwrites or removes an earlier snapshot, even one made in the same second.  On failure
        only the name this attempt created is removed and the evidence stays pending, so no rewrite (at load or when the file outgrows
        the cap) may drop the unreadable lines before a copy of them exists."""
        stem = f"{self.path.name}.bad-{int(time.time())}"
        for n in range(1000):
            evidence = self.path.with_name(stem if n == 0 else f"{stem}-{n}")
            try:
                fd = os.open(evidence, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0))
            except FileExistsError:
                continue                                              # an earlier snapshot holds this name: never touch it
            except OSError:
                return None
            try:
                with os.fdopen(fd, "wb") as out, self.path.open("rb") as found:
                    shutil.copyfileobj(found, out)
                    out.flush()
                    os.fsync(out.fileno())
            except OSError:
                try:
                    evidence.unlink(missing_ok=True)                  # only the name this attempt reserved
                except OSError:
                    pass
                return None
            self._evidence_pending = False
            return evidence
        return None

    def _rewrite(self) -> None:
        """Writes the whole ring to a temporary file beside the ring, then replaces the ring with it: never a half-written ring."""
        partial = self.path.with_name(f"{self.path.name}.rewrite-{secrets.token_hex(8)}")
        try:
            with partial.open("xb") as f:
                for e in self.entries:
                    f.write(self.cipher.seal(json.dumps(e).encode("utf-8")))
                f.flush()
                os.fsync(f.fileno())
            os.replace(partial, self.path)
            self._written = self.bytes
            self._needs_separator = False
        finally:
            partial.unlink(missing_ok=True)

    def _persist(self, entry: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self._written > 2 * self.cap and (not self._evidence_pending or self._keep_evidence() is not None):
                self._rewrite()
            else:
                with self.path.open("ab") as f:
                    if self._needs_separator:
                        f.write(b"\n")
                        self._needs_separator = False
                    f.write(self.cipher.seal(json.dumps(entry).encode("utf-8")))
                self._written += self._cost(entry["text"])
        except OSError:
            pass


RING = Ring()


def note_login(outcome: str, username, client) -> int:
    """One ``login`` line: ``<outcome> user=<name> client=<ip token>``; outcome is ok, fail, limited or logout."""
    name = "".join(ch for ch in str(username).strip().lower()[:64] if ch.isprintable() and not ch.isspace())   # a failed name is caller-typed text
    return RING.add(f"{outcome} user={name or '-'} client={private_record.token('ip', client or '-')}", source="login")


_last_seen: dict = {}
_seen_lock = threading.Lock()


def note_activity(user_id, username, client, now: float | None = None) -> int | None:
    """Files ``active user=<name> client=<ip token>`` when this account has been quiet ``IDLE_SECONDS``; otherwise only updates its clock."""
    now = time.time() if now is None else now
    with _seen_lock:
        last = _last_seen.get(user_id)
        _last_seen[user_id] = now
    if last is not None and now - last < IDLE_SECONDS:
        return None
    return RING.add(f"active user={username} client={private_record.token('ip', client or '-')}", source="activity", ts=now)


class _Tee:
    """Passes everything through to the real stream and files each complete line into the ring."""
    def __init__(self, real):
        self._real, self._buf, self._lock = real, "", threading.Lock()

    def write(self, s):
        self._real.write(private_record.redact(s))        # [PRIVATE] what the server prints reaches its log file as tokens too
        n = len(s)
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
