"""Git in the core, regulated: declared read operations, asked for by an account, recorded in SQL.

The core repositories are the server's own copy: what makes this server's repositories this server's. Someone with
direct file access on the host can run Git there themselves. Everyone else reaches Git only through here, over REST
and the tools generated from it, and only for the operations declared below. Nothing here writes: changes reach the
core through reviews (``branch_review.py``) and the gate that lands them (``branch_gate.py``).

    run(db, policy, user_id=, username=, repository=, operation=, arguments={...})
    recent(db)                                    the record of calls
    inside_core(policy, path)                     the repository a path lies in, or None
    exec_refusal(policy, program, working_dir)    why a program started for a row may not run there, or None

An operation is a name and a few typed arguments, never a Git command line: the caller cannot pass a flag, a config
value, a remote or a hook. Every call is one row of ``git_calls``: who, which repository, which operation, with what,
when, and how it ended.

Who may call is the caller's business: the server checks the ``core_git`` capability, which no service account holds.
``exec_refusal`` is the other half of the rule: a program started for a row (an exec row's compressor, say) never
runs Git and never runs with its working directory inside the core.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

MAX_OUTPUT = 512 * 1024
TIMEOUT_S = 30.0
REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/~^-]{0,199}\Z")
PATH_PART = re.compile(r"[^\x00-\x1f\\:*?\"<>|]{1,255}\Z")
DDL = ("CREATE TABLE IF NOT EXISTS git_calls (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, username TEXT, "
       "repository TEXT NOT NULL, operation TEXT NOT NULL, arguments TEXT NOT NULL, at REAL NOT NULL, exit_code INTEGER, "
       "bytes INTEGER, truncated INTEGER NOT NULL DEFAULT 0)")


class GitError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def ensure(db) -> None:
    db.execute(DDL)


def _ref(value, what="ref") -> str:
    if not isinstance(value, str) or not REF.match(value) or ".." in value or value.endswith((".lock", "/")):
        raise GitError(f"{what} must be a branch, tag or commit name")
    return value


def _path(value) -> str:
    """A path inside the repository, forward slashes, never absolute, never climbing out, never an option."""
    if value in (None, ""):
        return ""
    if not isinstance(value, str) or len(value) > 512 or value.startswith(("/", "-")):
        raise GitError("path must be a relative path inside the repository")
    parts = value.split("/")
    if any(part in ("", ".", "..") or not PATH_PART.match(part) for part in parts):
        raise GitError("path must be a relative path inside the repository")
    return value


def _count(value, default, most) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= most:
        raise GitError(f"count must be a whole number from 1 to {most}")
    return value


def _only(arguments, allowed) -> None:
    extra = set(arguments) - set(allowed)
    if extra:
        raise GitError("not an argument of this operation: " + ", ".join(sorted(extra)))


def _status(a):
    _only(a, ())
    return ["status", "--porcelain=v1", "--branch"]


def _branches(a):
    _only(a, ())
    return ["for-each-ref", "--format=%(refname)%09%(objectname)%09%(committerdate:unix)", "refs/heads", "refs/remotes"]


def _log(a):
    _only(a, ("ref", "path", "count"))
    command = ["log", f"-n{_count(a.get('count'), 20, 200)}", "--no-color", "--format=%H%x09%ct%x09%an%x09%s",
               _ref(a.get("ref", "HEAD")), "--"]
    return command + ([_path(a.get("path"))] if a.get("path") else [])


def _commit(a):
    _only(a, ("commit",))
    return ["show", "--no-color", "--no-ext-diff", "--no-textconv", "--stat", "--format=%H%n%an%n%ct%n%B",
            _ref(a.get("commit"), "commit"), "--"]


def _diff(a):
    _only(a, ("base", "head", "path"))
    command = ["diff", "--no-color", "--no-ext-diff", "--no-textconv",
               f"{_ref(a.get('base'), 'base')}..{_ref(a.get('head'), 'head')}", "--"]
    return command + ([_path(a.get("path"))] if a.get("path") else [])


def _file(a):
    _only(a, ("ref", "path"))
    path = _path(a.get("path"))
    if not path:
        raise GitError("path is required")
    return ["show", "--no-color", "--no-ext-diff", "--no-textconv", f"{_ref(a.get('ref', 'HEAD'))}:{path}"]


def _tree(a):
    _only(a, ("ref", "path"))
    command = ["ls-tree", "-r", "--name-only", _ref(a.get("ref", "HEAD")), "--"]
    return command + ([_path(a.get("path"))] if a.get("path") else [])


OPERATIONS = {"status": _status, "branches": _branches, "log": _log, "commit": _commit, "diff": _diff, "file": _file,
              "tree": _tree}


def inside_core(policy, path) -> str | None:
    """The policy repository whose folder holds ``path`` (the most specific one), or None."""
    target = Path(path).resolve()
    found, depth = None, -1
    for name, declared in policy["repositories"].items():
        root = Path(declared["path"]).resolve()
        if target == root or root in target.parents:
            if len(root.parts) > depth:
                found, depth = name, len(root.parts)
    return found


def exec_refusal(policy, program, working_dir) -> str | None:
    """Why a program started for a row may not run as asked; None when it may."""
    name = Path(str(program)).name.casefold()
    if name in ("git", "git.exe") or name.startswith("git-"):
        return "a row never runs Git; Git in the core is asked for through core_git"
    held = inside_core(policy, working_dir)
    if held is not None:
        return f"a row's program never runs with its working directory inside the core ({held})"
    return None


def _execute(root, command) -> tuple:
    """(exit code, output bytes, truncated). Output is cut at MAX_OUTPUT and the process ended."""
    environment = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0", GIT_PAGER="cat", LC_ALL="C")
    process = subprocess.Popen(["git", "-C", str(root), "--no-pager", "--no-optional-locks", "--literal-pathspecs",
                                "-c", "core.fsmonitor=false", *command],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, env=environment)
    try:
        data = process.stdout.read(MAX_OUTPUT + 1)
        truncated = len(data) > MAX_OUTPUT
        if truncated:
            process.kill()
        code = process.wait(timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise GitError("the Git operation took too long", 504) from None
    finally:
        process.stdout.close()
    return code, data[:MAX_OUTPUT], truncated


def run(db, policy, *, user_id, username, repository, operation, arguments=None, now=None) -> dict:
    ensure(db)
    if isinstance(user_id, bool) or not isinstance(user_id, int):
        raise GitError("a signed-in account is required", 401)
    declared = policy["repositories"].get(repository) if isinstance(repository, str) else None
    if declared is None:
        raise GitError("repository must be one the policy names: " + ", ".join(sorted(policy["repositories"])), 404)
    builder = OPERATIONS.get(operation) if isinstance(operation, str) else None
    if builder is None:
        raise GitError("operation must be one of " + ", ".join(sorted(OPERATIONS)), 404)
    arguments = dict(arguments or {})
    command = builder(arguments)
    root = Path(declared["path"])
    if not (root / ".git").exists():
        raise GitError("that repository has no checkout here", 503)
    now = time.time() if now is None else now
    made = db.execute("INSERT INTO git_calls (user_id, username, repository, operation, arguments, at) VALUES (?,?,?,?,?,?)",
                      (user_id, username, repository, operation, json.dumps(arguments, sort_keys=True), now))
    code, data, truncated = _execute(root, command)
    db.execute("UPDATE git_calls SET exit_code = ?, bytes = ?, truncated = ? WHERE id = ?",
               (code, len(data), int(truncated), made.lastrowid))
    if code != 0 and not truncated:
        raise GitError("Git could not do that (no such ref or path)", 404)
    return {"call": made.lastrowid, "repository": repository, "operation": operation, "truncated": truncated,
            "bytes": len(data), "output": data.decode("utf-8", "replace")}


def recent(db, limit: int = 100) -> list:
    ensure(db)
    return [{"id": r[0], "username": r[1], "repository": r[2], "operation": r[3], "arguments": json.loads(r[4]), "at": r[5],
             "exit_code": r[6], "bytes": r[7], "truncated": bool(r[8])} for r in db.execute(
        "SELECT id, username, repository, operation, arguments, at, exit_code, bytes, truncated FROM git_calls "
        "ORDER BY id DESC LIMIT ?", (int(limit),))]


def http_route(self, path, method, *, policy, accounts_db):
    """Git in the core over REST, for accounts that hold ``core_git``.

        GET  /api/git/repositories                 the repositories and their default branches
        GET  /api/git/calls                        the record of calls
        POST /api/git/<operation>  {repository, ...the operation's arguments}
             status | branches | log {ref?, path?, count?} | commit {commit} | diff {base, head, path?} |
             file {ref?, path} | tree {ref?, path?}
    """
    import sqlite3
    self._set_cookie = []
    if method == "POST":
        self.close_connection = True
    principal = self._principal()
    if principal.user_id is None:
        return self._json({"error": "sign in first"}, 401)
    if "core_git" not in self._caps():
        return self._json({"error": "this needs core_git"}, 403)
    db = sqlite3.connect(accounts_db, timeout=10, isolation_level=None)
    try:
        if method == "GET" and path == "/api/git/repositories":
            return self._json({"repositories": {name: policy["repositories"][name]["default"]
                                                for name in sorted(policy["repositories"])},
                               "operations": sorted(OPERATIONS)})
        if method == "GET" and path == "/api/git/calls":
            return self._json({"calls": recent(db)})
        if method != "POST" or not path.startswith("/api/git/"):
            return self._json({"error": "not found"}, 404)
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            return self._json({"error": "JSON only"}, 415)
        body = self._body_json(4096)
        if body is None:
            return
        if not isinstance(body, dict):
            raise GitError("a JSON object is required")
        arguments = {key: value for key, value in body.items() if key != "repository"}
        result = run(db, policy, user_id=principal.user_id, username=principal.username,
                     repository=body.get("repository"), operation=path[len("/api/git/"):], arguments=arguments)
        print(f"[git] {principal.username} {result['operation']} {result['repository']} {result['bytes']} bytes", flush=True)
        return self._json(result)
    except GitError as error:
        return self._json({"error": error.message}, error.status)
    finally:
        db.close()
