"""Opening a dev page's project (research case d442be8d, dev pages ticket 9db5ea72): the branch placement.

A project is a folder with a ``project.yaml`` at its root (src/compiler/project_contract.py). Opening one makes
its placement and writes that first file. This module opens the ``branch`` placement only; ``standalone`` and
``branch-copy`` need the account's readable project space (ticket 2/8, 5e77787b) and are refused until it exists.

    GET  /api/projects                 your projects
    POST /api/projects/open  {text}    open a project from the text of its project.yaml

[CAPABILITY] dev_ui is enough to open any placement (owner, 2026-11-27); no other capability is asked.

[BRANCH] All branch work is the central checkout's (central_checkout.run: open, status, write, commit), so the
branch is registered to the account, made from the default branch, recorded call by call and lent a server like
any other checkout. The contract is validated first (project_contract.parse_project_contract, with the branch
policy's repositories), and only then written to ``projects/<account>/<project>/project.yaml`` in the branch
(the place the owner approved, 2026-11-27) and committed as the account's declared identity, naming the project.
The commit then goes up for review like any other change (reviews_submit); branch -> main is the review gate.

[ACCOUNT] The contract's identity is ``<account>/<project>``; its account must be the signed-in account's name.

[ONCE] A project is opened once: a second open of the same identity is refused (table ``projects``), and so is
an open into a branch that already holds that project's file.

[WHOLE] Opening is all or nothing. Everything is checked before anything is made. If a later step fails, what
this open made is removed again, and nothing else: the file it wrote; the checkout, the branch and the branch's
registration when this open made them. A checkout that was already open is only used when it is committed and
has no unfinished merge, so the project's commit holds the project file alone.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import PurePosixPath

from meowdoku import branch_review, central_checkout

MAX_CONTRACT = 64_000
SPACE_MODES_WAIT = "the {mode!r} placement needs the readable project space (dev pages ticket 2/8), which is not built yet"
DDL = (
    "CREATE TABLE IF NOT EXISTS projects (identity TEXT PRIMARY KEY, user_id INTEGER NOT NULL, username TEXT, "
    "mode TEXT NOT NULL, repository TEXT, branch TEXT, path TEXT NOT NULL, head TEXT, opened_at REAL NOT NULL)",
)
_KEYS = ("identity", "user_id", "username", "mode", "repository", "branch", "path", "head", "opened_at")


class ProjectError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def ensure(db) -> None:
    central_checkout.ensure(db)
    for statement in DDL:
        db.execute(statement)


def contract_path(contract) -> str:
    """Where a branch project's project.yaml lives in its repository."""
    return PurePosixPath("projects", contract.account, contract.project, "project.yaml").as_posix()


def projects_of(db, user_id) -> list:
    ensure(db)
    rows = db.execute(f"SELECT {', '.join(_KEYS)} FROM projects WHERE user_id = ? ORDER BY opened_at DESC", (user_id,))
    return [dict(zip(_KEYS, row)) for row in rows]


def _parse(text, policy):
    from src.compiler.project_contract import ProjectContractError, parse_project_contract
    if not isinstance(text, str) or not text.strip():
        raise ProjectError("text must be the project.yaml text")
    if len(text.encode("utf-8")) > MAX_CONTRACT:
        raise ProjectError(f"a project.yaml is at most {MAX_CONTRACT} bytes", 413)
    try:
        return parse_project_contract(text, "project.yaml", repositories=frozenset(policy["repositories"]))
    except ProjectContractError as error:
        raise ProjectError(str(error)) from None


def _checkout(db, policy, root, who, operation, **body):
    return central_checkout.run(db, policy, root, user_id=who["user_id"], username=who["username"],
                                operation=operation, body=body)


def _undo(db, policy, root, who, *, repository, branch, wrote, made_checkout, made_branch, made_owner) -> None:
    """[WHOLE] remove what this open made, and nothing else."""
    db.execute("DELETE FROM projects WHERE repository = ? AND branch = ? AND head IS NULL", (repository, branch))
    if wrote and central_checkout._row(db, repository, branch) is not None:
        try:
            _checkout(db, policy, root, who, "remove", repository=repository, branch=branch, path=wrote)
        except central_checkout.CheckoutError:
            pass
    closed = True
    if made_checkout and central_checkout._row(db, repository, branch) is not None:
        try:
            _checkout(db, policy, root, who, "close", repository=repository, branch=branch)
        except central_checkout.CheckoutError:
            closed = False                         # a commit was made: the branch is no longer only what we made
    if made_branch and closed:
        repo, _, _ = central_checkout._repository(policy, repository)
        try:
            central_checkout._git(repo, "branch", "-D", branch, check=False)
        except central_checkout.CheckoutError:
            pass
    if made_owner and closed:
        db.execute("DELETE FROM branch_owners WHERE repository = ? AND branch = ? AND user_id = ?",
                   (repository, branch, who["user_id"]))


def open_project(db, policy, root, *, user_id, username, text, now=None) -> dict:
    """Open a project from its project.yaml text. Only the branch placement is opened here."""
    ensure(db)
    if user_id is None or not username:
        raise ProjectError("sign in first", 401)
    contract = _parse(text, policy)
    if contract.mode != "branch":
        raise ProjectError(SPACE_MODES_WAIT.format(mode=contract.mode), 409)
    if contract.account != username:                                                   # [ACCOUNT]
        raise ProjectError(f"identity {contract.identity!r} names account {contract.account!r}; "
                           f"you open projects as {username!r}", 403)
    if db.execute("SELECT 1 FROM projects WHERE identity = ?", (contract.identity,)).fetchone():   # [ONCE]
        raise ProjectError(f"project {contract.identity!r} is already open", 409)
    if not branch_review.identity(db, user_id):
        raise ProjectError("declare your Git author identity first (POST /api/reviews/identity, reviews_identity_set)",
                           409)
    repository, branch = contract.branch["repository"], contract.branch["name"]
    held = branch_review.owner(db, repository, branch)
    if held is not None and held["user_id"] != user_id:
        raise ProjectError(f"branch {branch!r} is registered to {held['username']}; choose another name", 409)
    repo, _, remote = central_checkout._repository(policy, repository)
    had_branch = bool(central_checkout._git(repo, "rev-parse", "-q", "--verify", f"refs/heads/{branch}", check=False))
    had_checkout = central_checkout._row(db, repository, branch) is not None
    who = {"user_id": user_id, "username": username}
    path = contract_path(contract)
    undo = {"repository": repository, "branch": branch, "wrote": None, "made_checkout": not had_checkout,
            "made_branch": not had_branch, "made_owner": held is None}
    now = time.time() if now is None else now
    try:
        db.execute("INSERT INTO projects VALUES (?,?,?,?,?,?,?,?,?)",                       # [ONCE] claimed first
                   (contract.identity, int(user_id), username, contract.mode, repository, branch, path, None, now))
    except sqlite3.IntegrityError:
        raise ProjectError(f"project {contract.identity!r} is already open", 409) from None
    try:
        opened = _checkout(db, policy, root, who, "open", repository=repository, branch=branch)
        state = _checkout(db, policy, root, who, "status", repository=repository, branch=branch)
        if state["changes"] or state["merging"]:
            undo = {**undo, "made_checkout": False, "made_branch": False, "made_owner": False}
            raise ProjectError("your checkout of that branch has uncommitted changes or an unfinished merge; "
                               "commit them first, so the project's commit holds its project file alone", 409)
        try:
            _checkout(db, policy, root, who, "read", repository=repository, branch=branch, path=path)
        except central_checkout.CheckoutError as error:
            if error.status != 404:
                raise
        else:
            raise ProjectError(f"branch {branch!r} already holds {path}", 409)                  # [ONCE]
        _checkout(db, policy, root, who, "write", repository=repository, branch=branch, path=path, text=text)
        undo["wrote"] = path
        committed = _checkout(db, policy, root, who, "commit", repository=repository, branch=branch,
                              message=f"Open project {contract.identity} (branch placement)\n\n"
                                      f"{path}: the project contract, validated by project_contract before writing.")
        db.execute("UPDATE projects SET head = ? WHERE identity = ?", (committed["head"], contract.identity))
    except central_checkout.CheckoutError as error:
        _undo(db, policy, root, who, **undo)
        raise ProjectError(error.message, error.status) from None
    except BaseException:
        _undo(db, policy, root, who, **undo)
        raise
    return {"identity": contract.identity, "mode": contract.mode, "repository": repository, "branch": branch,
            "path": path, "head": committed["head"], "base": opened.get("base"), "server": opened.get("server"),
            "contract": contract.to_data(),
            "next": "submit it: reviews_submit (POST /api/reviews) with this repository, branch and head"}


def http_route(self, path, method, *, policy, accounts_db, root):
    """Projects over REST, for signed-in accounts holding dev_ui."""
    import sqlite3
    self._set_cookie = []
    if method == "POST":
        self.close_connection = True
    principal = self._principal()
    if principal.user_id is None:
        return self._json({"error": "sign in first"}, 401)
    if "dev_ui" not in self._caps():                                                  # [CAPABILITY]
        return self._json({"error": "this needs dev_ui"}, 403)
    db = sqlite3.connect(accounts_db, timeout=10, isolation_level=None)
    try:
        if method == "GET" and path == "/api/projects":
            return self._json({"projects": projects_of(db, principal.user_id)})
        if method != "POST" or path != "/api/projects/open":
            return self._json({"error": "not found"}, 404)
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            return self._json({"error": "JSON only"}, 415)
        body = self._body_json(2 * MAX_CONTRACT)
        if body is None:
            return
        if not isinstance(body, dict):
            return self._json({"error": "a JSON object is required"}, 400)
        result = open_project(db, policy, root, user_id=principal.user_id, username=principal.username,
                              text=body.get("text"))
        print(f"[projects] {principal.username} open {result['identity']} {result['repository']} {result['branch']}",
              flush=True)
        return self._json(result)
    except ProjectError as error:
        return self._json({"error": error.message}, error.status)
    finally:
        db.close()
