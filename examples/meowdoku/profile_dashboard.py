"""Read-only profile dashboard: who I am, which service instances run (with the git state of their code), leases, and instance log tails.

Pure functions over an accounts.db sqlite connection (SELECT only) and a StatePaths; nothing here writes.  The server's ``[PROFILE]`` block
(server.py) gates them: ``me`` for any signed-in account, the rest for ``account_manage`` or ``lifecycle_admin``.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GIT_TIMEOUT = 5.0
GIT_CACHE_SECONDS = 30.0
MAX_LINES = 200
MAX_BYTES = 32 * 1024
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
SCRUB_RE = re.compile(r"(?i)\b([\w.-]*(?:token|secret|password|passwd|api[_-]?key|authorization|cookie)[\w.-]*)(\s*[=:]\s*|\s+)(?:Bearer\s+)?\S+")

_git_cache: dict = {}
_git_lock = threading.Lock()


class ProfileError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def _rows(db, sql: str, args=()) -> list[dict]:
    """SELECT as dicts; a table that does not exist yet is simply no rows."""
    import sqlite3
    try:
        cur = db.execute(sql, args)
    except sqlite3.OperationalError:
        return []
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


# ---- me ---------------------------------------------------------------------------------------------------------------------
def me(db, principal, capabilities=()) -> dict:
    """The signed-in account: username, display name, role, capabilities, stable/dev preference."""
    display = None
    pref = None
    if principal.user_id is not None:
        cols = {r["name"] for r in _rows(db, "PRAGMA table_info(users)")}
        if "display_name" in cols:
            rows = _rows(db, "SELECT display_name FROM users WHERE id = ?", (principal.user_id,))
            display = rows[0]["display_name"] if rows else None
        rows = _rows(db, "SELECT site FROM account_prefs WHERE user_id = ?", (principal.user_id,))
        pref = rows[0]["site"] if rows else None
    return {"username": principal.username, "display_name": display or principal.username, "roles": [principal.role],
            "capabilities": sorted(capabilities), "site_preference": pref}


# ---- git state of the code an instance runs -----------------------------------------------------------------------------------
def _git(repo: str, *args: str) -> str:
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}          # status must not take the index lock: this is a read
    done = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=GIT_TIMEOUT, env=env)
    if done.returncode != 0:
        raise RuntimeError((done.stderr or done.stdout).strip().splitlines()[-1] if (done.stderr or done.stdout).strip() else f"git exit {done.returncode}")
    return done.stdout


def git_state(repo, now: float | None = None) -> dict:
    """{toplevel, branch, commit, dirty, ahead, behind, upstream_known, error}; cached 30 s per path; a failure is the ``error`` field."""
    now = time.time() if now is None else now
    key = str(repo)
    with _git_lock:
        hit = _git_cache.get(key)
        if hit and now - hit[0] < GIT_CACHE_SECONDS:
            return hit[1]
    state = {"toplevel": None, "branch": None, "commit": None, "dirty": None, "ahead": None, "behind": None, "upstream_known": False, "error": None}
    try:
        state["toplevel"] = _git(key, "rev-parse", "--show-toplevel").strip()
        dirty = False
        for line in _git(key, "status", "--porcelain=v2", "--branch").splitlines():
            if line.startswith("# branch.oid "):
                oid = line.split(" ", 2)[2]
                state["commit"] = None if oid == "(initial)" else oid[:8]
            elif line.startswith("# branch.head "):
                state["branch"] = line.split(" ", 2)[2]
            elif line.startswith("# branch.ab "):
                ahead, behind = line.split()[2:4]
                state["ahead"], state["behind"], state["upstream_known"] = int(ahead[1:]), int(behind[1:]), True
            elif not line.startswith("#"):
                dirty = True
        state["dirty"] = dirty
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        state["error"] = f"{type(error).__name__}: {error}"[:200]
    with _git_lock:
        _git_cache[key] = (now, state)
    return state


# ---- instances ----------------------------------------------------------------------------------------------------------------
def _config(path) -> dict:
    """name -> spec from deploy/instances.json (a list, or an object with "instances"); unreadable -> {}."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if isinstance(data, dict):
        data = data.get("instances", [])
    return {s["name"]: s for s in data if isinstance(s, dict) and s.get("name")} if isinstance(data, list) else {}


def instances(db, config=None, default_repo=None) -> list[dict]:
    """One row per service instance.  ``config`` is deploy/instances.json (enabled, state_dir basename, optional "repo");
    the code's repo is the entry's "repo" or ``default_repo`` (the repo this module lives in)."""
    specs = _config(config) if config else {}
    repo_default = str(default_repo or HERE)
    out = []
    for row in _rows(db, "SELECT name, kind, port, route, pid, state, started, restarts FROM service_instances ORDER BY name"):
        spec = specs.get(row["name"], {})
        state_dir = spec.get("state_dir")
        out.append({**row, "enabled": spec.get("enabled", True) is not False if spec else None,
                    "state_dir": Path(str(state_dir)).name if state_dir else None,
                    "git": git_state(spec.get("repo") or repo_default)})
    return out


# ---- leases -------------------------------------------------------------------------------------------------------------------
def leases(db, now: float | None = None) -> list[dict]:
    """instance_leases (supervisor -> lowering) and lowering_leases rows; the owner is a username where the key is an account."""
    now = time.time() if now is None else now
    out = []
    for r in _rows(db, "SELECT l.name AS owner, l.lease_id, l.product_sha256, l.generation, i.state AS instance_state "
                       "FROM instance_leases l LEFT JOIN service_instances i ON i.name = l.name ORDER BY l.name, l.lease_id"):
        out.append({"table": "instance_leases", "id": r["lease_id"], "owner": r["owner"], "purpose": None, "taken": None, "expires": None,
                    "state": r["instance_state"] or "unknown", "generation": r["generation"], "product": (r["product_sha256"] or "")[:12]})
    for r in _rows(db, "SELECT lease_id, product_sha256, core_id, backend, account_key, purpose, generation, issued, deadline, returned "
                       "FROM lowering_leases ORDER BY lease_id DESC LIMIT 200"):
        owner = r["account_key"]
        if str(owner).isdigit():
            names = _rows(db, "SELECT username FROM users WHERE id = ?", (int(owner),))
            owner = names[0]["username"] if names else owner
        state = "returned" if r["returned"] else ("expired" if r["deadline"] < now else "active")
        out.append({"table": "lowering_leases", "id": r["lease_id"], "owner": owner, "purpose": r["purpose"], "taken": r["issued"],
                    "expires": r["deadline"], "state": state, "generation": r["generation"], "product": (r["product_sha256"] or "")[:12],
                    "core": r["core_id"], "backend": r["backend"]})
    return out


# ---- diagnostics --------------------------------------------------------------------------------------------------------------
def scrub(line: str) -> str:
    return SCRUB_RE.sub(lambda m: f"{m.group(1)}=[scrubbed]", line)


def diagnostics(db, paths, name, lines: int = 80) -> dict:
    """The tail of the newest ``<name>-*.log`` in the StatePaths logs dir.  ``name`` must be a row of service_instances; no path comes from the request."""
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ProfileError("bad instance name", 400)
    if not _rows(db, "SELECT 1 FROM service_instances WHERE name = ?", (name,)):
        raise ProfileError("no such instance", 404)
    try:
        want = max(1, min(int(lines), MAX_LINES))
    except (TypeError, ValueError):
        want = 80
    logs = Path(paths.logs_dir)
    pattern = re.compile(r"^" + re.escape(name) + r"-[\w.]+\.log$")
    candidates = []
    try:
        candidates = [p for p in logs.iterdir() if pattern.match(p.name) and p.is_file()]
    except OSError:
        pass
    if not candidates:
        return {"name": name, "log": None, "lines": [], "truncated": False}
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 4 * MAX_BYTES))
        data = f.read()
    tail = [scrub(x) for x in data.decode("utf-8", "replace").splitlines()][-want:]
    while tail and sum(len(x.encode("utf-8")) + 1 for x in tail) > MAX_BYTES:
        tail.pop(0)
    return {"name": name, "log": path.name, "lines": tail, "truncated": len(tail) < want and size > 0 and len(tail) < len(data.splitlines())}
