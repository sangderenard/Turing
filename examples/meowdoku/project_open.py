"""Opening a dev page's project (research case d442be8d, dev pages ticket 9db5ea72): branch and standalone.

A project is a folder with a ``project.yaml`` at its root (src/compiler/project_contract.py). Opening one makes
its placement and writes that first file. This module opens the ``branch`` and ``standalone`` placements;
``branch-copy`` needs refresh and accept between the branch and its copy (case d442be8d) and is refused until then.

    GET  /api/projects                 your projects
    POST /api/projects/open  {text}    open a project from the text of its project.yaml

[CAPABILITY] dev_ui is enough to open any placement (owner, 2026-11-27); no other capability is asked.

[BRANCH] All branch work is the central checkout's (central_checkout.run: open, status, write, commit), so the
branch is registered to the account, made from the default branch, recorded call by call and lent a server like
any other checkout. The contract is validated first (project_contract.parse_project_contract, with the branch
policy's repositories), and only then written to ``projects/<account>/<project>/project.yaml`` in the branch
(the place the owner approved, 2026-11-27) and committed as the account's declared identity, naming the project.
The commit then goes up for review like any other change (reviews_submit); branch -> main is the review gate.

[STANDALONE] The project's own readable space (project_space.py, ticket 2/8, case 35c3b249): a Git repository
the service manages, with no remote, under content/workspaces/projects/<user id>/, accounted by workspace_accounting.
While the operator has not configured the space, a standalone open is refused and nothing is made.

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
COPY_WAITS = ("the 'branch-copy' placement needs refresh and accept between a branch and its copy "
              "(research case d442be8d), which are not built yet")
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


def _undo(db, policy, root, who, *, identity, repository, branch, wrote, made_checkout, made_branch,
          made_owner) -> None:
    """[WHOLE] remove what this open made, and nothing else: only the row this open claimed, never another
    open's in-flight row on the same branch."""
    db.execute("DELETE FROM projects WHERE identity = ? AND head IS NULL", (identity,))
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


def _open_standalone(db, space, contract, text, *, user_id, username, now) -> dict:
    """[STANDALONE] the project's own space (project_space.py): claimed here, made there, all or nothing."""
    from meowdoku import project_space
    if space is None or space.config is None:
        raise ProjectError(project_space.NOT_CONFIGURED, 503)
    now = time.time() if now is None else now
    try:
        db.execute("INSERT INTO projects VALUES (?,?,?,?,?,?,?,?,?)",                       # [ONCE] claimed first
                   (contract.identity, int(user_id), username, contract.mode, None, None, contract.space, None, now))
    except sqlite3.IntegrityError:
        raise ProjectError(f"project {contract.identity!r} is already open", 409) from None
    try:
        made = space.create(db, user_id=user_id, username=username, contract=contract, text=text)
        db.execute("UPDATE projects SET head = ? WHERE identity = ?", (made["head"], contract.identity))
    except project_space.SpaceError as error:
        db.execute("DELETE FROM projects WHERE identity = ? AND head IS NULL", (contract.identity,))
        raise ProjectError(error.message, error.status) from None
    except BaseException:
        db.execute("DELETE FROM projects WHERE identity = ? AND head IS NULL", (contract.identity,))
        raise
    return {"identity": contract.identity, "mode": contract.mode, "space": contract.space, "path": made["path"],
            "head": made["head"], "space_bytes": made["bytes"], "contract": contract.to_data(),
            "next": "work in it with the project_space_* tools (POST /api/projects/space/<operation>)"}


def open_project(db, policy, root, *, user_id, username, text, space=None, now=None) -> dict:
    """Open a project from its project.yaml text: the branch or standalone placement."""
    ensure(db)
    if user_id is None or not username:
        raise ProjectError("sign in first", 401)
    contract = _parse(text, policy)
    if contract.mode == "branch-copy":
        raise ProjectError(COPY_WAITS, 409)
    if contract.account != username:                                                   # [ACCOUNT]
        raise ProjectError(f"identity {contract.identity!r} names account {contract.account!r}; "
                           f"you open projects as {username!r}", 403)
    if db.execute("SELECT 1 FROM projects WHERE identity = ?", (contract.identity,)).fetchone():   # [ONCE]
        raise ProjectError(f"project {contract.identity!r} is already open", 409)
    if not branch_review.identity(db, user_id):
        raise ProjectError("declare your Git author identity first (POST /api/reviews/identity, reviews_identity_set)",
                           409)
    if contract.mode == "standalone":
        return _open_standalone(db, space, contract, text, user_id=user_id, username=username, now=now)
    repository, branch = contract.branch["repository"], contract.branch["name"]
    held = branch_review.owner(db, repository, branch)
    if held is not None and held["user_id"] != user_id:
        raise ProjectError(f"branch {branch!r} is registered to {held['username']}; choose another name", 409)
    repo, _, remote = central_checkout._repository(policy, repository)
    had_branch = bool(central_checkout._git(repo, "rev-parse", "-q", "--verify", f"refs/heads/{branch}", check=False))
    had_checkout = central_checkout._row(db, repository, branch) is not None
    who = {"user_id": user_id, "username": username}
    path = contract_path(contract)
    undo = {"identity": contract.identity, "repository": repository, "branch": branch, "wrote": None,
            "made_checkout": not had_checkout, "made_branch": not had_branch, "made_owner": held is None}
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
        recorded = db.execute("UPDATE projects SET head = ? WHERE identity = ? AND head IS NULL",
                              (committed["head"], contract.identity)).rowcount
    except central_checkout.CheckoutError as error:
        _undo(db, policy, root, who, **undo)
        raise ProjectError(error.message, error.status) from None
    except BaseException:
        _undo(db, policy, root, who, **undo)
        raise
    if recorded != 1:                  # the commit stands in the branch, but this open's catalog row is gone
        raise ProjectError(f"project {contract.identity!r} was committed to {branch!r} at {committed['head']}, "
                           "but its catalog row was removed meanwhile; it is not listed as open", 409)
    return {"identity": contract.identity, "mode": contract.mode, "repository": repository, "branch": branch,
            "path": path, "head": committed["head"], "base": opened.get("base"), "server": opened.get("server"),
            "contract": contract.to_data(),
            "next": "submit it: reviews_submit (POST /api/reviews) with this repository, branch and head"}


def http_route(self, path, method, *, policy, accounts_db, root, space=None):
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
                              text=body.get("text"), space=space)
        print(f"[projects] {principal.username} open {result['identity']} {result['mode']}", flush=True)
        return self._json(result)
    except ProjectError as error:
        return self._json({"error": error.message}, error.status)
    finally:
        db.close()
