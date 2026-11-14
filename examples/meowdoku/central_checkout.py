"""Central checkout: a branch and its working folder, made and kept by the service for an account, over REST and MCP.

An account never runs Git in the core. After the SQL checkout (branch_review.register: the branch is this account's),
the service makes the branch from the remote's default branch in the core repository and a working folder for it in
the state root (``<state>/content/checkouts/<user id>/<repository>/<branch>``), a Git worktree of the core repository.
The account then works through declared operations only, each a name and typed arguments, never a command line:

    open    {repository, branch}            register the branch to yourself, make it and its working folder
    files   {repository, branch, path?}     list one folder of the working folder
    read    {repository, branch, path}      read one text file
    write   {repository, branch, path, text}   create or replace one text file
    remove  {repository, branch, path}      delete one file
    status  {repository, branch}            what differs from the branch's last commit
    diff    {repository, branch, path?}     the change in the tracked files, as a unified diff
    commit  {repository, branch, message}   commit everything, as the author identity this account declared
    close   {repository, branch}            remove the working folder (it must be committed); the branch stays

Submitting is the existing review (POST /api/reviews {repository, branch, head}): the branch is in the core
repository, where the lander looks for it, so landing is unchanged. The cleaner removes a landed branch's worktree.

Only this account's own checkouts are reachable. A path is relative to the working folder; it may not leave it, name
``.git``, pass through a link or junction, or use a name Windows treats specially. Every call is recorded
(``checkout_calls``). The working folders belong to the service account: nobody else's Windows account writes them.
"""
from __future__ import annotations

import os
import stat
import time
from pathlib import Path

from meowdoku import branch_gate, branch_review

MAX_FILE_BYTES = 2_000_000
MAX_LISTING = 2000
MAX_DIFF_BYTES = 400_000
MAX_MESSAGE = 5000
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
OPERATIONS = ("open", "files", "read", "write", "remove", "status", "diff", "commit", "close")

DDL = (
    "CREATE TABLE IF NOT EXISTS checkouts (repository TEXT NOT NULL, branch TEXT NOT NULL, user_id INTEGER NOT NULL, "
    "username TEXT, path TEXT NOT NULL, base TEXT NOT NULL, created_at REAL NOT NULL, PRIMARY KEY (repository, branch))",
    "CREATE TABLE IF NOT EXISTS checkout_calls (id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL, user_id INTEGER, "
    "username TEXT, repository TEXT, branch TEXT, operation TEXT, path TEXT, bytes INTEGER, ok INTEGER NOT NULL, "
    "error TEXT)",
)
CALLS_KEEP = 20000


class CheckoutError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def ensure(db) -> None:
    branch_review.ensure(db)
    for statement in DDL:
        db.execute(statement)


def _git(path, *args, check=True) -> str:
    try:
        return branch_gate.git(path, *args, check=check)
    except branch_gate.GateError as error:
        raise CheckoutError(str(error), 409) from None


def _repository(policy, repository):
    declared = policy["repositories"].get(repository) if isinstance(repository, str) else None
    if declared is None:
        raise CheckoutError("repository must be one the policy names: " + ", ".join(sorted(policy["repositories"])), 404)
    return Path(declared["path"]), declared["default"], policy.get("remote", "origin")


def _row(db, repository, branch):
    row = db.execute("SELECT repository, branch, user_id, username, path, base, created_at FROM checkouts "
                     "WHERE repository = ? AND branch = ?", (repository, branch)).fetchone()
    keys = ("repository", "branch", "user_id", "username", "path", "base", "created_at")
    return None if row is None else dict(zip(keys, row))


def _mine(db, *, user_id, repository, branch) -> dict:
    """This account's open checkout of that branch, or a refusal that says so."""
    row = _row(db, repository, branch) if isinstance(repository, str) and isinstance(branch, str) else None
    if row is None or row["user_id"] != user_id:
        raise CheckoutError("you have no open checkout of that branch; open it first", 404)
    if not Path(row["path"]).is_dir():
        raise CheckoutError("the working folder of that checkout is gone; open it again", 410)
    return row


def _is_link(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def target(folder, relative, *, allow_root=False) -> Path:
    """The file or folder ``relative`` names inside ``folder``, or a refusal. Never outside it, never .git, never through
    a link or junction, never a name Windows reads as a device or a stream."""
    if not isinstance(relative, str) or len(relative) > 400:
        raise CheckoutError("path must be a relative path of at most 400 characters")
    cleaned = relative.replace("\\", "/").strip("/")
    folder = Path(folder)
    if not cleaned:
        if allow_root:
            return folder
        raise CheckoutError("path must name a file")
    parts = cleaned.split("/")
    for part in parts:
        if part in ("", ".", "..") or part.casefold() == ".git" or part != part.rstrip(" .") or ":" in part \
                or part.split(".")[0].upper() in RESERVED or any(ord(c) < 32 or c in '<>"|?*' for c in part):
            raise CheckoutError(f"path {relative!r} is not allowed here")
    current = folder
    for part in parts:
        current = current / part
        if _is_link(current):
            raise CheckoutError(f"path {relative!r} passes through a link")
    resolved, root = current.resolve(strict=False), folder.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise CheckoutError(f"path {relative!r} leaves the working folder")
    return current


def _record(db, who, repository, branch, operation, path=None, size=None, error=None) -> None:
    cursor = db.execute("INSERT INTO checkout_calls (at, user_id, username, repository, branch, operation, path, bytes, ok, "
                        "error) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (time.time(), who.get("user_id"), who.get("username"), repository, branch, operation,
                         None if path is None else str(path)[:400], size, int(error is None),
                         None if error is None else str(error)[:500]))
    db.execute("DELETE FROM checkout_calls WHERE id <= ?", (cursor.lastrowid - CALLS_KEEP,))


def _describe(row, head) -> dict:
    return {"repository": row["repository"], "branch": row["branch"], "base": row["base"], "head": head,
            "opened_at": row["created_at"], "owner": row["username"]}


def open_checkout(db, policy, root, *, user_id, username, repository, branch) -> dict:
    """Register the branch to this account (the SQL checkout) and make the branch and its working folder."""
    repo, default, remote = _repository(policy, repository)
    defaults = {declared["default"] for declared in policy["repositories"].values()}
    if not isinstance(branch, str) or branch in defaults:
        raise CheckoutError("choose a branch name of your own, not a default branch")
    held_before = branch_review.owner(db, repository, branch)
    exists = bool(_git(repo, "rev-parse", "-q", "--verify", f"refs/heads/{branch}", check=False)) or \
        bool(_git(repo, "rev-parse", "-q", "--verify", f"refs/remotes/{remote}/{branch}", check=False))
    if exists and (held_before is None or held_before["user_id"] != user_id):
        raise CheckoutError("that branch already exists and is not registered to you; choose a new name", 409)
    try:
        branch_review.register(db, user_id=user_id, username=username, repository=repository, branch=branch,
                               known=list(policy["repositories"]))
    except branch_review.ReviewError as error:
        raise CheckoutError(error.message, error.status) from None
    row = _row(db, repository, branch)
    if row is not None and Path(row["path"]).is_dir():
        return _describe(row, _git(row["path"], "rev-parse", "HEAD"))
    folder = Path(root) / str(int(user_id)) / repository / branch.replace("/", "__")
    if folder.exists():
        raise CheckoutError("a folder is already at that checkout's place; ask an operator to look at it", 409)
    if not _git(repo, "rev-parse", "-q", "--verify", f"refs/heads/{branch}", check=False):
        _git(repo, "fetch", "--quiet", remote, default)
        _git(repo, "branch", "--no-track", branch, f"refs/remotes/{remote}/{default}")
    folder.parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "worktree", "add", "--quiet", str(folder), branch)
    db.execute("INSERT OR REPLACE INTO checkouts VALUES (?,?,?,?,?,?,?)",
               (repository, branch, int(user_id), username, str(folder), default, time.time()))
    return _describe(_row(db, repository, branch), _git(folder, "rev-parse", "HEAD"))


def files(row, path="") -> dict:
    folder = Path(row["path"])
    place = target(folder, path or "", allow_root=True)
    if not place.is_dir():
        raise CheckoutError("path must name a folder", 404)
    entries = []
    for child in sorted(place.iterdir(), key=lambda p: (not p.is_dir(), p.name.casefold())):
        if child.name.casefold() == ".git" or _is_link(child):
            continue
        entries.append({"name": child.name, "kind": "folder" if child.is_dir() else "file",
                        "bytes": None if child.is_dir() else child.stat().st_size})
        if len(entries) >= MAX_LISTING:
            break
    return {"path": place.relative_to(folder).as_posix() if place != folder else "", "entries": entries,
            "truncated": len(entries) >= MAX_LISTING}


def read(row, path) -> dict:
    place = target(row["path"], path)
    if not place.is_file():
        raise CheckoutError("no such file", 404)
    if place.stat().st_size > MAX_FILE_BYTES:
        raise CheckoutError(f"the file is larger than {MAX_FILE_BYTES} bytes", 413)
    data = place.read_bytes()
    if b"\0" in data:
        raise CheckoutError("that is a binary file; only text is offered", 415)
    try:
        return {"path": path, "text": data.decode("utf-8"), "bytes": len(data)}
    except UnicodeDecodeError:
        raise CheckoutError("that file is not UTF-8 text", 415) from None


def write(row, path, text) -> dict:
    if not isinstance(text, str):
        raise CheckoutError("text must be a string")
    data = text.encode("utf-8")
    if len(data) > MAX_FILE_BYTES:
        raise CheckoutError(f"text is larger than {MAX_FILE_BYTES} bytes", 413)
    place = target(row["path"], path)
    if place.is_dir():
        raise CheckoutError("path names a folder", 409)
    place.parent.mkdir(parents=True, exist_ok=True)
    target(row["path"], path)                     # again, now that the folders exist: none of them may be a link
    with open(place, "wb") as handle:             # the bytes as given: no newline translation
        handle.write(data)
    return {"path": path, "bytes": len(data)}


def remove(row, path) -> dict:
    place = target(row["path"], path)
    if not place.is_file():
        raise CheckoutError("no such file", 404)
    place.unlink()
    return {"path": path, "removed": True}


def _porcelain(folder) -> list:
    """``git status --porcelain`` lines as Git wrote them: the shared helper strips the output, which would cut the
    leading space of the first line's two-letter state (" M base.txt" read as "M base.txt")."""
    import subprocess
    done = subprocess.run(["git", "-C", str(folder), "status", "--porcelain=v1", "--untracked-files=all"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if done.returncode != 0:
        raise CheckoutError(f"git status failed: {(done.stderr or done.stdout).strip()[-300:]}", 409)
    return [line for line in done.stdout.split("\n") if line]


def status(row) -> dict:
    lines = _porcelain(row["path"])
    return {"head": _git(row["path"], "rev-parse", "HEAD"),
            "changes": [{"state": line[:2], "path": line[3:]} for line in lines[:MAX_LISTING]],
            "truncated": len(lines) > MAX_LISTING}


def diff(row, path=None) -> dict:
    arguments = ["diff", "HEAD", "--no-color", "--no-ext-diff"]
    if path:
        target(row["path"], path)
        arguments += ["--", path.replace("\\", "/")]
    text = _git(row["path"], *arguments)
    return {"diff": text[:MAX_DIFF_BYTES], "truncated": len(text) > MAX_DIFF_BYTES,
            "note": "untracked files are listed by status, not here"}


def commit(db, row, *, user_id, message) -> dict:
    if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE:
        raise CheckoutError(f"message must be text of 1 to {MAX_MESSAGE} characters")
    declared = branch_review.identity(db, user_id)
    if not declared:
        raise CheckoutError("declare your Git author identity first (POST /api/reviews/identity, reviews_identity_set)", 409)
    folder = row["path"]
    if not _porcelain(folder):
        raise CheckoutError("nothing to commit", 409)
    _git(folder, "add", "--all")
    _git(folder, "-c", f"user.name={declared['name']}", "-c", f"user.email={declared['email']}", "commit", "--quiet",
         "-m", message)
    return {"head": _git(folder, "rev-parse", "HEAD"), "branch": row["branch"],
            "next": "submit it: reviews_submit (POST /api/reviews) with this repository, branch and head"}


def close(db, policy, row) -> dict:
    if _porcelain(row["path"]):
        raise CheckoutError("the working folder has uncommitted changes: commit them first", 409)
    repo, _, _ = _repository(policy, row["repository"])
    _git(repo, "worktree", "remove", row["path"])
    db.execute("DELETE FROM checkouts WHERE repository = ? AND branch = ?", (row["repository"], row["branch"]))
    return {"repository": row["repository"], "branch": row["branch"], "closed": True,
            "note": "the branch stays, registered to you; open it again to keep working"}


def listing(db, user_id) -> list:
    ensure(db)
    rows = db.execute("SELECT repository, branch, base, created_at FROM checkouts WHERE user_id = ? ORDER BY created_at DESC",
                      (user_id,)).fetchall()
    return [{"repository": r[0], "branch": r[1], "base": r[2], "opened_at": r[3]} for r in rows]


def run(db, policy, root, *, user_id, username, operation, body) -> dict:
    """One declared operation for the signed-in account; recorded whether it succeeds or not."""
    ensure(db)
    if operation not in OPERATIONS:
        raise CheckoutError("operation must be one of: " + ", ".join(OPERATIONS), 404)
    who = {"user_id": user_id, "username": username}
    repository, branch, path = body.get("repository"), body.get("branch"), body.get("path")
    try:
        if operation == "open":
            result = open_checkout(db, policy, root, user_id=user_id, username=username, repository=repository, branch=branch)
        else:
            row = _mine(db, user_id=user_id, repository=repository, branch=branch)
            if operation == "files":
                result = files(row, path or "")
            elif operation == "read":
                result = read(row, path)
            elif operation == "write":
                result = write(row, path, body.get("text"))
            elif operation == "remove":
                result = remove(row, path)
            elif operation == "status":
                result = status(row)
            elif operation == "diff":
                result = diff(row, path)
            elif operation == "commit":
                result = commit(db, row, user_id=user_id, message=body.get("message"))
            else:
                result = close(db, policy, row)
    except CheckoutError as error:
        _record(db, who, repository, branch, operation, path, error=error.message)
        raise
    except OSError as error:
        _record(db, who, repository, branch, operation, path, error=f"{type(error).__name__}: {error}")
        raise CheckoutError(f"the service could not do that: {type(error).__name__}", 500) from None
    _record(db, who, repository, branch, operation, path, size=result.get("bytes"))
    return {"operation": operation, **result}


def http_route(self, path, method, *, policy, accounts_db, root):
    """The central checkout over REST, for signed-in accounts holding dev_ui (the same as the reviews).

        GET  /api/checkouts                   your open checkouts
        POST /api/checkouts/<operation>       {repository, branch, ...}   one of OPERATIONS (see the module text)
    """
    import sqlite3
    self._set_cookie = []
    if method == "POST":
        self.close_connection = True
    principal = self._principal()
    if principal.user_id is None:
        return self._json({"error": "sign in first"}, 401)
    if "dev_ui" not in self._caps():
        return self._json({"error": "this needs dev_ui"}, 403)
    db = sqlite3.connect(accounts_db, timeout=10, isolation_level=None)
    try:
        if method == "GET" and path == "/api/checkouts":
            return self._json({"checkouts": listing(db, principal.user_id), "operations": list(OPERATIONS)})
        if method != "POST" or not path.startswith("/api/checkouts/"):
            return self._json({"error": "not found"}, 404)
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            return self._json({"error": "JSON only"}, 415)
        body = self._body_json(3 * MAX_FILE_BYTES)
        if body is None:
            return
        if not isinstance(body, dict):
            return self._json({"error": "a JSON object is required"}, 400)
        operation = path[len("/api/checkouts/"):]
        result = run(db, policy, root, user_id=principal.user_id, username=principal.username, operation=operation, body=body)
        print(f"[checkout] {principal.username} {operation} {body.get('repository')} {body.get('branch')}", flush=True)
        return self._json(result)
    except CheckoutError as error:
        return self._json({"error": error.message}, error.status)
    finally:
        db.close()
