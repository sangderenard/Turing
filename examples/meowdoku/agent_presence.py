"""Observed agent activity for the developer room and repository map.

Only scheduler records and authenticated request timestamps are treated as live
signals. Missing instrumentation is reported as unknown, never as inactivity.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

AGENTS = {"claude": ("agents", 0), "codex": ("agents-codex", 20), "gemini": ("agents-gemini", 40)}
# Hosted Llama has a SpeakToMe account but no Start-AgentDay scheduler slot.
UNSCHEDULED_AGENTS = ("llama", "operator-claude", "operator-codex")
# Installed external API agents (external_agents.py, case 989e507e), read beside the accounts database at each snapshot.
_external: dict[str, tuple[str, int]] = {}
RECENT_SECONDS = 30
REQUEST_CHANNELS = ("api_in", "mcp_in")
_lock = threading.Lock()
_recent: dict[tuple[str, str], float] = {}
_SAFE_PATH = re.compile(r"^[A-Za-z0-9_./-]{1,400}$")


def known_agents(accounts_db: Path | None = None) -> dict[str, tuple[str, int]]:
    """The built-in agents, then the installed external ones (account agents-<name>, their declared minute)."""
    global _external
    if accounts_db is not None:
        found = {}
        path = Path(accounts_db).with_name("admin_control.db")
        try:
            with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=1) as db:
                for name, minute in db.execute("SELECT name, minute FROM external_agents WHERE status='installed'"):
                    if name not in AGENTS:
                        found[name] = ("agents-" + name, int(minute))
        except (sqlite3.Error, OSError, ValueError):
            found = dict(_external)
        _external = found
    return {**AGENTS, **_external}


def note_inbound(username: str, *, mcp: bool, now: float | None = None) -> None:
    """Record an authenticated API request; a call is only 'recent', not 'running'."""
    if username not in AGENTS and username not in _external and username not in UNSCHEDULED_AGENTS:
        return
    with _lock:
        _recent[(username, "mcp_in" if mcp else "api_in")] = time.time() if now is None else now


def _observed(username: str, now: float) -> dict:
    with _lock:
        recent = {channel: _recent.get((username, channel)) for channel in REQUEST_CHANNELS}
    return {channel: {"state": "recent" if at is not None and 0 <= now - at <= RECENT_SECONDS else "unobserved",
                      "at": at} for channel, at in recent.items()}


def _stamp(value) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _process_started(pid: int) -> float | None:
    """Windows process creation time; None means the run cannot be validated."""
    if os.name != "nt" or pid <= 0:
        return None
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x1000, 0, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        created = ctypes.c_uint64()
        exited = ctypes.c_uint64()
        kernel_time = ctypes.c_uint64()
        user_time = ctypes.c_uint64()
        if not kernel.GetProcessTimes(ctypes.c_void_p(handle), ctypes.byref(created), ctypes.byref(exited),
                                      ctypes.byref(kernel_time), ctypes.byref(user_time)):
            return None
        return created.value / 10_000_000 - 11644473600
    finally:
        kernel.CloseHandle(ctypes.c_void_p(handle))


def _run_records(days: Path, now: float, agents: dict | None = None) -> dict[str, list[dict]]:
    agents = AGENTS if agents is None else agents
    found: dict[str, list[dict]] = {name: [] for name in agents}
    directory = days / "running"
    if not directory.is_dir():
        return found
    for file in directory.glob("*.json"):
        try:
            record = json.loads(file.read_text(encoding="utf-8-sig"))
            if not isinstance(record, dict):
                continue
            name = record.get("agent") or next((n for n, (account, _) in agents.items() if record.get("account") == account), None)
            started, pid = _stamp(record.get("started")), record.get("pid")
            if name not in found or started is None or type(pid) is not int:
                continue
            process_started = _process_started(pid)
            if process_started is None or process_started > started + 1 or process_started > now:
                continue
            # The process creation time also rejects a record surviving a reboot with a reused PID.
            if started < process_started - 1:
                continue
            found[name].append({"kind": record.get("kind"), "room": record.get("room"), "started": started})
        except (OSError, ValueError, TypeError):
            continue
    return found


def _schedule(days: Path, name: str, minute: int, now: float) -> dict:
    path = days / "state" / f"{name}.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8-sig"))
        observed_at = path.stat().st_mtime
    except (OSError, ValueError):
        state, observed_at = {}, None
    fresh = observed_at is not None and 0 <= now - observed_at <= 180
    local = datetime.fromtimestamp(now).astimezone()
    slot = local.replace(minute=minute, second=0, microsecond=0)
    if slot > local:
        slot -= timedelta(hours=1)
    last = _stamp(state.get("last_hourly"))
    next_due = slot if last is None or last < slot.timestamp() else slot + timedelta(hours=1)
    cooldown = _stamp(state.get("bell_off_until"))
    return {"next_due": next_due.timestamp() if fresh else None, "due": next_due.timestamp() <= now if fresh else None,
            "bell_off_until": cooldown if cooldown and cooldown > now else None,
            "last_work": last, "last_bell": _stamp(state.get("last_bell")), "observed_at": observed_at}


def checkout_locations(accounts_db: Path, now: float) -> list[dict]:
    """Last audited file access, not an assertion about where the agent is now."""
    try:
        with sqlite3.connect(f"file:{Path(accounts_db).as_posix()}?mode=ro", uri=True, timeout=1) as db:
            rows = db.execute("SELECT username, repository, path, at FROM checkout_calls "
                              "WHERE ok=1 AND operation IN ('read','write') ORDER BY id DESC LIMIT 500").fetchall()
    except (sqlite3.Error, OSError):
        return []
    seen, out = set(), []
    for username, repository, path, at in rows:
        if username not in AGENTS and username not in _external and username not in UNSCHEDULED_AGENTS:
            continue
        if username in seen or not isinstance(path, str) or not _SAFE_PATH.fullmatch(path) or ".." in path.split("/"):
            continue
        seen.add(username)
        out.append({"username": username, "repository": repository, "path": path.replace("\\", "/"), "at": at,
                    "age_seconds": max(0, now - at)})
    return out


def snapshot(accounts_db: Path, agent_days: Path | None = None, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    days = Path(agent_days) if agent_days is not None else Path.home() / "AgentDays"
    known = known_agents(accounts_db)
    records = _run_records(days, now, known)
    records_available = (days / "running").is_dir()
    agents = []
    for name, (account, minute) in known.items():
        runs = records[name]
        channels = {"api_in": _observed(name, now)["api_in"], "mcp_in": _observed(name, now)["mcp_in"],
                    "api_out": {"state": "unknown"}, "mcp_out": {"state": "unknown"},
                    "local": {"state": "active" if runs else "unobserved" if records_available else "unknown", "since": min((r["started"] for r in runs), default=None)}}
        agents.append({"username": name, "account": account, "runs": runs, "schedule": _schedule(days, name, minute, now), "channels": channels})
    for name in UNSCHEDULED_AGENTS:
        agents.append({"username": name, "account": None, "runs": [], "schedule": None,
                       "channels": {**_observed(name, now), "api_out": {"state": "unknown"},
                                    "mcp_out": {"state": "unknown"}, "local": {"state": "unknown"}}})
    return {"as_of": now, "agents": agents, "locations": checkout_locations(accounts_db, now),
            "note": "Recent means an authenticated call observed by this server process in the last 30 seconds. MCP in requires the internal OAuth/preview bridge; other API calls are API in. Unobserved does not mean offline. Outbound channels have no shared observation source."}
