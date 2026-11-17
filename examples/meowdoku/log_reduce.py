"""[REDUCE] Repeat reduction for an append-only log: n-line cycles are counted, not written again.

Ticket 95975791c0e948d7ba96d18eea193275. Owner, 2026-11-17: "an n-line-gram repeat reducer in the logs and a stateful
verbosity for those actions, or ... optional, a log filter on what we record from those messages." The first log is
Apache's access log (apache_supervisor.tail_stdin), where a polling client (the hosted Llama on a 1 s cycle) writes the
same few requests again and again.

Every line gets a KEY by a declared rule (``ACCESS_KEY``): the parts that make two lines "the same action". For the access
log that is the request line and the status; the client, the time and the byte count are masked.

- Cycles: when the last n keys (1 <= n <= MAX_N) equal the n before them, the cycle has been written twice. From then on
  each further full repeat is counted, not written.
- Stepped verbosity (state per cycle): while a cycle continues, one progress line is written when the count reaches 10,
  100, 1000, ... and at least every PROGRESS_SECONDS, so a quiet log still shows life. When anything breaks the cycle, one
  summary line is written (count, time span, pattern), then the lines of the partial repeat in full, then the new line.
- Filter (optional; ``<state>/logs/log_filter.json``, absent by default): {"keep": [regex, ...], "drop": [regex, ...]}
  matched against the key. A kept line is always written in full and never joins a cycle. A dropped line is never
  written; how many were dropped, per pattern, is written every PROGRESS_SECONDS. The file is read again when it changes.

Everything a line held is either written or counted in a line that is written: nothing disappears without a count.
"""
from __future__ import annotations

import json
import re
import time
from collections import deque
from pathlib import Path

MAX_N = 8
PROGRESS_STEPS = 10                   # progress lines at 10, 100, 1000, ... repeats
PROGRESS_SECONDS = 600.0
# Apache "meowdoku_safe": %h %l %u %t "%m %U %H" %>s %b  ->  key = "%m %U %H" %>s
ACCESS_KEY = re.compile(r'^\S+ \S+ \S+ \[[^\]]*\] ("[^"]*") (\d{3}) \S+\s*$')


def access_key(line: str) -> str:
    """The action an access line records: request and status. A line in another shape is its own key, unchanged."""
    m = ACCESS_KEY.match(line)
    return f"{m.group(1)} {m.group(2)}" if m else line.rstrip("\r\n")


def _stamp(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))


class LineFilter:
    """The optional keep/drop rules, read from ``path`` and read again whenever the file changes."""

    def __init__(self, path=None):
        self.path, self.mtime, self.keep, self.drop = (Path(path) if path else None), None, [], []

    def _load(self):
        if self.path is None:
            return
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            self.mtime, self.keep, self.drop = None, [], []
            return
        if mtime == self.mtime:
            return
        self.mtime = mtime
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.keep = [re.compile(p) for p in data.get("keep", []) if isinstance(p, str)]
            self.drop = [re.compile(p) for p in data.get("drop", []) if isinstance(p, str)]
        except (OSError, ValueError, re.error):
            self.keep, self.drop = [], []           # a broken filter file filters nothing: no line is lost to it

    def verdict(self, key: str):
        """("keep" | "drop" | None, the pattern text)."""
        self._load()
        for pattern in self.keep:
            if pattern.search(key):
                return "keep", pattern.pattern
        for pattern in self.drop:
            if pattern.search(key):
                return "drop", pattern.pattern
        return None, None


class Reducer:
    """feed(line) -> the lines to write now (possibly none); flush() -> what is owed at the end."""

    def __init__(self, key=access_key, line_filter=None, max_n=MAX_N, clock=time.time):
        self.key, self.filter, self.max_n, self.clock = key, line_filter or LineFilter(None), max_n, clock
        self.history = deque(maxlen=2 * max_n)
        self.cycle = None                       # tuple of keys while a repeat is being counted
        self.position, self.repeats, self.partial = 0, 0, []
        self.first_at = self.last_at = self.reported_at = 0.0
        self.next_step = PROGRESS_STEPS
        self.dropped, self.dropped_at = {}, None

    # ---- the lines written in place of what was counted
    def _summary(self, final: bool) -> str:
        n = len(self.cycle)
        verb = "repeated" if final else "is repeating:"
        return (f"[log-reduce] the previous {n} line(s) {verb} {self.repeats} more time(s) from {_stamp(self.first_at)} "
                f"to {_stamp(self.last_at)}: " + " / ".join(self.cycle) + "\n")

    def _end_cycle(self) -> list:
        out = []
        if self.repeats:
            out.append(self._summary(final=True))
        out.extend(self.partial)
        for line in self.partial:                    # the partial repeat was written in full: it is history again
            self.history.append(self.key(line))
        self.cycle, self.position, self.repeats, self.partial = None, 0, 0, []
        self.next_step = PROGRESS_STEPS
        return out

    def _dropped_report(self, now: float, force: bool = False) -> list:
        if not self.dropped or (not force and self.dropped_at is not None and now - self.dropped_at < PROGRESS_SECONDS):
            return []
        line = ("[log-filter] dropped since " + _stamp(self.dropped_at or now) + ": " +
                "; ".join(f"{count} matching {pattern}" for pattern, count in sorted(self.dropped.items())) + "\n")
        self.dropped, self.dropped_at = {}, now
        return [line]

    # ---- the stream
    def feed(self, line: str) -> list:
        now = self.clock()
        key = self.key(line)
        verdict, pattern = self.filter.verdict(key)
        if verdict == "drop":
            if self.dropped_at is None:
                self.dropped_at = now
            self.dropped[pattern] = self.dropped.get(pattern, 0) + 1
            return self._dropped_report(now)
        out = self._dropped_report(now)
        if verdict == "keep":                         # always written in full, and never part of a cycle
            return out + (self._end_cycle() if self.cycle else []) + [line]
        if self.cycle is not None:
            if key == self.cycle[self.position]:
                self.partial.append(line)
                self.position += 1
                if self.position == len(self.cycle):  # one more full repeat: counted, not written
                    self.position, self.partial = 0, []
                    self.repeats += 1
                    self.last_at = now
                    if self.repeats >= self.next_step or now - self.reported_at >= PROGRESS_SECONDS:
                        while self.next_step <= self.repeats:
                            self.next_step *= PROGRESS_STEPS
                        self.reported_at = now
                        out.append(self._summary(final=False))
                return out
            out.extend(self._end_cycle())             # broken: the summary, the partial repeat, then this line
        out.append(line)
        self.history.append(key)
        keys = list(self.history)
        for n in range(1, self.max_n + 1):
            if len(keys) >= 2 * n and keys[-n:] == keys[-2 * n:-n]:
                self.cycle, self.position, self.repeats, self.partial = tuple(keys[-n:]), 0, 0, []
                self.first_at = self.last_at = self.reported_at = now
                self.history.clear()
                break
        return out

    def flush(self) -> list:
        """What is owed when the stream ends: the cycle's summary, its partial repeat, and the drop counts."""
        out = self._end_cycle() if self.cycle is not None else []
        return out + self._dropped_report(self.clock(), force=True)
