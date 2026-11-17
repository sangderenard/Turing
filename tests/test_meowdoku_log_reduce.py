"""[REDUCE] Repeating n-line cycles are counted, not written; nothing disappears without a count."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import log_reduce as R  # noqa: E402


def access(path, second, status=200, size=123):
    return f'ip~000000000000 - - [17/Nov/2026:16:00:{second % 60:02d} -0600] "GET {path} HTTP/1.1" {status} {size}\n'


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        self.now += 1.0
        return self.now


def run(reducer, lines):
    out = []
    for line in lines:
        out.extend(reducer.feed(line))
    return out


POLL = ("/api/messages/contacts", "/api/messages/inbox", "/api/chat/rooms/agents?tail=100")


def test_a_three_line_poll_is_written_twice_then_counted_with_stepped_progress():
    reducer = R.Reducer(clock=Clock())
    lines = [access(path, i * 3 + j, size=100 + i) for i in range(52) for j, path in enumerate(POLL)]
    out = run(reducer, lines + [access("/api/profile/me", 59)])
    full = [line for line in out if not line.startswith("[log-reduce]")]
    assert full[:6] == lines[:6] and full[-1].endswith('"GET /api/profile/me HTTP/1.1" 200 123\n')
    assert len(full) == 7, "two full cycles, then only the line that broke it"
    progress = [line for line in out if "is repeating:" in line]
    assert [re.search(r"repeating: (\d+)", p).group(1) for p in progress] == ["10"], "stepped: 10, then 100, ..."
    final = [line for line in out if " repeated " in line]
    assert len(final) == 1 and "the previous 3 line(s) repeated 50 more time(s)" in final[0]
    assert '"GET /api/messages/inbox HTTP/1.1" 200' in final[0], "the pattern is named, by request and status"


def test_a_broken_cycle_writes_its_summary_then_the_partial_repeat_then_the_new_line():
    reducer = R.Reducer(clock=Clock())
    lines = [access(path, i * 3 + j) for i in range(5) for j, path in enumerate(POLL)]
    lines += [access(POLL[0], 50), access(POLL[1], 51), access("/api/log", 52, status=403)]
    out = run(reducer, lines)
    assert "repeated 3 more time(s)" in out[-4]
    assert out[-3:] == lines[-3:], "the two lines of the partial repeat come back in full, then the new one"


def test_a_different_status_is_a_different_action():
    reducer = R.Reducer(clock=Clock())
    lines = [access("/api/x", i) for i in range(4)] + [access("/api/x", 9, status=503)]
    out = run(reducer, lines)
    assert out[-1] == lines[-1] and "repeated 2 more time(s)" in out[-2]


def test_flush_owes_the_summary_and_the_partial_repeat():
    reducer = R.Reducer(clock=Clock())
    lines = [access(path, i * 3 + j) for i in range(4) for j, path in enumerate(POLL)] + [access(POLL[0], 40)]
    run(reducer, lines)
    owed = reducer.flush()
    assert "repeated 2 more time(s)" in owed[0] and owed[1:] == lines[-1:]


def test_the_optional_filter_keeps_and_drops_and_counts_what_it_dropped(tmp_path):
    rules = tmp_path / "log_filter.json"
    rules.write_text(json.dumps({"keep": ["/api/log"], "drop": ["/api/messages/contacts"]}), encoding="utf-8")
    reducer = R.Reducer(line_filter=R.LineFilter(rules), clock=Clock())
    lines = [access("/api/messages/contacts", i) for i in range(5)] + [access("/api/log", i) for i in range(4)]
    out = run(reducer, lines)
    assert [line for line in out if "/api/log" in line and not line.startswith("[")] == lines[5:], "kept lines never reduce"
    owed = reducer.flush()
    assert any("[log-filter]" in line and "5 matching /api/messages/contacts" in line for line in out + owed)
    rules.write_text("not json", encoding="utf-8")
    import os
    os.utime(rules, (rules.stat().st_mtime + 5, rules.stat().st_mtime + 5))
    assert reducer.filter.verdict('"GET /api/messages/contacts HTTP/1.1" 200') == (None, None), "a broken file filters nothing"


def test_the_supervisor_reduces_before_it_seals():
    text = (ROOT / "examples" / "meowdoku" / "deploy" / "apache_supervisor.py").read_text(encoding="utf-8")
    body = text[text.index("def tail_stdin"):text.index("self.stop.set()", text.index("def tail_stdin"))]
    assert "log_reduce.Reducer(" in body and "reducer.feed(" in body and "reducer.flush()" in body
    assert body.index("reducer.feed(") < body.index("cipher.seal(")
