"""Admin-only view of encrypted Apache IP history and request usage."""
from __future__ import annotations

import re
from collections import Counter, deque
from pathlib import Path

from meowdoku.secure_records import RecordCipher

ACCESS = re.compile(r'^(?P<ip>\S+) \S+ \S+ \[(?P<time>[^]]+)\] "(?P<method>\S+) (?P<path>\S+) (?P<protocol>\S+)" (?P<status>\d{3}) (?P<bytes>\d+|-)$')


def read(path: Path, key_path: Path, *, limit: int = 100) -> dict:
    if not 1 <= limit <= 500:
        raise ValueError("limit must be 1 through 500")
    path = Path(path)
    if not path.exists():
        return {"requests": 0, "bytes": 0, "by_status": {}, "by_method": {}, "ip_history": [], "unreadable": 0}
    cipher = RecordCipher(key_path, existing_streams=(path,))
    cipher.migrate(path)
    statuses, methods = Counter(), Counter()
    history = deque(maxlen=limit)
    requests = size = unreadable = 0
    with path.open("rb") as saved:
        for line in saved:
            try:        # ticket f62934c8 hazard 3: one torn or spliced record (or a line still being written) is
                plain = cipher.open(line).decode("utf-8", "replace")     # counted, not allowed to fail the whole view
            except ValueError:
                unreadable += 1
                continue
            match = ACCESS.fullmatch(plain)
            if match is None:
                continue
            item = match.groupdict()
            requests += 1
            size += int(item["bytes"]) if item["bytes"] != "-" else 0
            statuses[item["status"]] += 1
            methods[item["method"]] += 1
            history.append(item)
    return {"requests": requests, "bytes": size, "by_status": dict(statuses), "by_method": dict(methods),
            "ip_history": list(history), "unreadable": unreadable}
