"""The readable project space (dev pages ticket 2/8, 5e77787b; research case 35c3b249): standalone projects.

"My files" (user_files.py) holds only client-encrypted bytes, which the service can never scan, build or recompile.
A standalone project needs plaintext the service reads, so it lives in its own space:

    <state>/content/workspaces/projects/<user id>/<space name>

under workspace_accounting's own default root (owner, 2026-11-27), so one accounting authority serves the filesystem.
``<space name>`` is the contract's ``placement.space`` without its leading ``projects/``.

    POST /api/projects/space/<operation>  {identity, ...}   one of OPERATIONS, for the account that owns the project

    files   {identity, path?}               list one folder of the space
    read    {identity, path}                read one text file
    write   {identity, path, text}          create or replace one text file
    remove  {identity, path}                delete one file
    edit    {identity, path, old, new, count?} | {identity, patch}
    status  {identity}                      what differs from the space's last commit
    diff    {identity, path?}               the change in the tracked files
    commit  {identity, message}             commit everything, as the account's declared Git identity

A space is made by opening a standalone project (project_open.py); this module's ``create`` is that step.

[HISTORY] Each space is a Git repository the service manages, with no remote (owner, 2026-11-27).

[REUSE] The operations are central_checkout's own functions (target, files, read, write, remove, edit, status, diff,
commit) on the space's folder, with its confinement: no path outside the space, no ``.git``, no link or junction, no
name Windows treats specially, and its size limits. Every call is recorded in ``checkout_calls`` (repository
``project-space``, branch the project's identity), succeeding or not.

[QUOTA] workspace_accounting's ledger: one lease per space (reserved at create), its measured bytes, ``.git``
included, recorded after every change. A change that could grow the account past its quota or the volume past its
capacity is refused before it is written (the ledger's own admission). Capacity, per-account quota, free-space floor
and the reservation per space are operator configuration (``deploy/project_space.json``, keys CONFIG_KEYS); while it
is missing, a standalone open is refused with "the project space is not configured" and nothing is made.

[CLEANER] branch_gate.clean_worktrees touches only worktrees of the branch policy's repositories. A space is its own
repository under content/workspaces, never one of them, so the cleaner never removes it.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path, PurePosixPath

from meowdoku import branch_review, central_checkout, workspace_accounting

OPERATIONS = ("files", "read", "write", "remove", "edit", "status", "diff", "commit")
CHANGES = ("write", "remove", "edit", "commit")
CONFIG_KEYS = ("capacity_bytes", "user_quota_bytes", "free_floor_bytes", "space_reservation_bytes")
NOT_CONFIGURED = "the project space is not configured (the operator writes deploy/project_space.json)"
RECORDED_AS = "project-space"
DDL = (
    "CREATE TABLE IF NOT EXISTS project_spaces (identity TEXT PRIMARY KEY, user_id INTEGER NOT NULL, "
    "username TEXT, path TEXT NOT NULL UNIQUE, workspace_id TEXT NOT NULL, created_at REAL NOT NULL)",
)
_KEYS = ("identity", "user_id", "username", "path", "workspace_id", "created_at")


class SpaceError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def ensure(db) -> None:
    central_checkout.ensure(db)
    for statement in DDL:
        db.execute(statement)


def load_config(path) -> dict | None:
    """The operator's limits, or None while the file is missing. A file that is there but wrong is an error."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(data, dict) or set(data) != set(CONFIG_KEYS) or \
            not all(type(data[key]) is int and data[key] > 0 for key in CONFIG_KEYS):
        raise SpaceError(f"{path}: the project space configuration must give exactly {list(CONFIG_KEYS)}, "
                         "each a positive whole number of bytes", 503)
    return data


class Space:
    """The project space of one state root: its folder, its ledger and its configuration."""

    def __init__(self, accounts_db, content_dir, config):
        self.config = config
        self.root = Path(content_dir) / "workspaces"
        self.accounts_db = accounts_db
        self._store = None

    @property
    def store(self) -> workspace_accounting.Store:
        if self.config is None:
            raise SpaceError(NOT_CONFIGURED, 503)
        if self._store is None:
            self.root.mkdir(parents=True, exist_ok=True)
            try:
                self._store = workspace_accounting.Store(
                    self.accounts_db, self.root, capacity_bytes=self.config["capacity_bytes"],
                    user_quota_bytes=self.config["user_quota_bytes"], free_floor_bytes=self.config["free_floor_bytes"])
            except workspace_accounting.WorkspaceError as error:
                raise SpaceError(f"the project space ledger refused: {error}", 503) from None
        return self._store

    def folder(self, user_id, contract) -> Path:
        name = PurePosixPath(contract.space).parts[1:]
        return self.root.joinpath("projects", str(int(user_id)), *name)

    # ------------------------------------------------------------------------------------------------- [QUOTA]

    def _admit(self, user_id, workspace_id, growth) -> None:
        """Refuse a change that could grow the account past what the ledger admits, before it is written."""
        if growth <= 0:
            return
        store = self.store
        with store._transaction() as db:
            row = store._owned(db, user_id, workspace_id)
            accounted = max(row["reserved_bytes"], row["measured_bytes"])
            extra = max(0, row["measured_bytes"] + growth - accounted)
            if extra:
                try:
                    store._admit(db, user_id, extra)
                except workspace_accounting.Refused as error:
                    raise SpaceError(f"the project space quota refuses this change: {error}", 507) from None

    def _measure(self, user_id, workspace_id, folder) -> int:
        amount = _size(Path(folder))
        store = self.store
        with store._transaction() as db:
            store._record_live_usage(db, user_id, workspace_id, amount)
        return amount

    # ------------------------------------------------------------------------------------------------- create

    def create(self, db, *, user_id, username, contract, text) -> dict:
        """Make the space for a standalone contract: folder, Git repository with no remote, project.yaml at its root,
        the first commit, and its lease. All or nothing: a failure removes what this call made."""
        ensure(db)
        store = self.store
        declared = branch_review.identity(db, user_id)
        if not declared:
            raise SpaceError("declare your Git author identity first (POST /api/reviews/identity, "
                             "reviews_identity_set)", 409)
        folder = self.folder(user_id, contract)
        if folder.exists() or db.execute("SELECT 1 FROM project_spaces WHERE path = ?", (str(folder),)).fetchone():
            raise SpaceError(f"your space {contract.space!r} is already in use", 409)
        data = text.encode("utf-8")
        try:
            workspace_id = store.reserve(user_id, self.config["space_reservation_bytes"],
                                         base_revision=hashlib.sha256(data).hexdigest())
        except workspace_accounting.Refused as error:
            raise SpaceError(f"the project space quota refuses a new space: {error}", 507) from None
        made = False
        try:
            folder.parent.mkdir(parents=True, exist_ok=True)
            folder.mkdir()
            made = True
            _git(folder, "init", "--quiet", "-b", "main")
            (folder / "project.yaml").write_bytes(data)
            _git(folder, "add", "--all")
            _git(folder, "-c", f"user.name={declared['name']}", "-c", f"user.email={declared['email']}",
                 "-c", "commit.gpgsign=false", "commit", "--quiet",
                 "-m", f"Open project {contract.identity} (standalone placement)")
            head = _git(folder, "rev-parse", "HEAD")
            db.execute("INSERT INTO project_spaces VALUES (?,?,?,?,?,?)",
                       (contract.identity, int(user_id), username, str(folder), workspace_id, time.time()))
            measured = self._measure(user_id, workspace_id, folder)
        except BaseException:
            if made:
                shutil.rmtree(folder, onerror=_writable)
            db.execute("DELETE FROM project_spaces WHERE identity = ?", (contract.identity,))
            generation = store.begin_release(user_id, workspace_id)
            store.confirm_removed(user_id, workspace_id, expected_generation=generation)
            raise
        return {"space": contract.space, "head": head, "path": "project.yaml", "bytes": measured}

    # ------------------------------------------------------------------------------------------------- operations

    def _row(self, db, user_id, identity) -> dict:
        found = db.execute(f"SELECT {', '.join(_KEYS)} FROM project_spaces WHERE identity = ?",
                           (identity,)).fetchone() if isinstance(identity, str) else None
        if found is None or found[1] != user_id:
            raise SpaceError("you have no project space with that identity", 404)
        row = dict(zip(_KEYS, found))
        if not Path(row["path"]).is_dir():
            raise SpaceError("that project space's folder is gone; ask an operator to look at it", 410)
        return row

    def run(self, db, *, user_id, username, operation, body) -> dict:
        """One declared operation on the account's own space; recorded whether it succeeds or not. [REUSE]"""
        ensure(db)
        if operation not in OPERATIONS:
            raise SpaceError("operation must be one of: " + ", ".join(OPERATIONS), 404)
        who = {"user_id": user_id, "username": username}
        identity, path = body.get("identity"), body.get("path")
        try:
            space = self._row(db, user_id, identity)
            row = {"path": space["path"], "branch": "main", "repository": RECORDED_AS}
            if operation in CHANGES:
                self._admit(user_id, space["workspace_id"], _growth(row, operation, body))
            result = _dispatch(db, row, operation, body, user_id)
            if operation in CHANGES:
                result["space_bytes"] = self._measure(user_id, space["workspace_id"], space["path"])
        except (SpaceError, central_checkout.CheckoutError) as error:
            central_checkout._record(db, who, RECORDED_AS, identity, operation, path, error=error.message)
            if isinstance(error, central_checkout.CheckoutError):
                raise SpaceError(error.message, error.status) from None
            raise
        except OSError as error:
            central_checkout._record(db, who, RECORDED_AS, identity, operation, path,
                                     error=f"{type(error).__name__}: {error}")
            raise SpaceError(f"the service could not do that: {type(error).__name__}", 500) from None
        central_checkout._record(db, who, RECORDED_AS, identity, operation,
                                 path or ",".join(result.get("files", [])) or None, size=result.get("bytes"))
        return {"operation": operation, "identity": identity, **result}


def spaces_of(db, user_id) -> list:
    ensure(db)
    rows = db.execute(f"SELECT {', '.join(_KEYS)} FROM project_spaces WHERE user_id = ? ORDER BY created_at DESC",
                      (user_id,))
    return [{key: value for key, value in zip(_KEYS, row) if key != "path"} for row in rows]


def _dispatch(db, row, operation, body, user_id) -> dict:
    path = body.get("path")
    if operation == "files":
        return central_checkout.files(row, path or "")
    if operation == "read":
        return central_checkout.read(row, path)
    if operation == "write":
        return central_checkout.write(row, path, body.get("text"))
    if operation == "remove":
        return central_checkout.remove(row, path)
    if operation == "edit":
        return central_checkout.edit(row, body)
    if operation == "status":
        return central_checkout.status(row)
    if operation == "diff":
        return central_checkout.diff(row, path)
    return central_checkout.commit(db, row, user_id=user_id, message=body.get("message"))


def _growth(row, operation, body) -> int:
    """An upper bound on how many bytes this change can add, known before it is made."""
    folder = Path(row["path"])
    if operation == "write":
        text = body.get("text")
        if not isinstance(text, str):
            return 0
        place = central_checkout.target(folder, body.get("path"))
        held = place.stat().st_size if place.is_file() else 0
        return len(text.encode("utf-8")) - held
    if operation == "edit":
        patch = body.get("patch")
        if isinstance(patch, str):
            return len(patch.encode("utf-8"))
        new, count = body.get("new"), body.get("count", 1)
        return len(new.encode("utf-8")) * count if isinstance(new, str) and type(count) is int else 0
    if operation == "commit":                     # the objects a commit adds are at most the changed files' bytes
        total = 0
        for line in central_checkout._porcelain(folder):
            place = folder / line[3:].strip('"')
            total += place.stat().st_size if place.is_file() else 0
        return total
    return 0


def _size(folder: Path) -> int:
    total = 0
    for base, folders, files in os.walk(folder):
        folders[:] = [name for name in folders if not central_checkout._is_link(Path(base, name))]
        for name in files:
            place = Path(base, name)
            if not central_checkout._is_link(place):
                total += place.stat().st_size
    return total


def _git(folder, *args) -> str:
    done = subprocess.run(["git", "-C", str(folder), *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    if done.returncode != 0:
        raise SpaceError(f"git {args[0]} failed: {(done.stderr or done.stdout).strip()[-300:]}", 500)
    return done.stdout.strip()


def _writable(function, path, _info):            # Git marks its object files read-only on Windows
    os.chmod(path, stat.S_IWRITE)
    function(path)


def http_route(self, path, method, *, space, accounts_db):
    """The project space over REST, for signed-in accounts holding dev_ui: POST /api/projects/space/<operation>."""
    import sqlite3
    self._set_cookie = []
    if method == "POST":
        self.close_connection = True
    principal = self._principal()
    if principal.user_id is None:
        return self._json({"error": "sign in first"}, 401)
    if "dev_ui" not in self._caps():
        return self._json({"error": "this needs dev_ui"}, 403)
    if method != "POST" or not path.startswith("/api/projects/space/"):
        return self._json({"error": "not found"}, 404)
    if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
        return self._json({"error": "JSON only"}, 415)
    body = self._body_json(3 * central_checkout.MAX_FILE_BYTES)
    if body is None:
        return
    if not isinstance(body, dict):
        return self._json({"error": "a JSON object is required"}, 400)
    db = sqlite3.connect(accounts_db, timeout=10, isolation_level=None)
    try:
        operation = path[len("/api/projects/space/"):]
        result = space.run(db, user_id=principal.user_id, username=principal.username, operation=operation, body=body)
        print(f"[project-space] {principal.username} {operation} {body.get('identity')}", flush=True)
        return self._json(result)
    except SpaceError as error:
        return self._json({"error": error.message}, error.status)
    finally:
        db.close()
