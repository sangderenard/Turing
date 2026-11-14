"""Changes put up for review before they reach a default branch: SQL rows, and the rule that reads them.

A change is one pushed branch at one commit. Its author submits it; it is then one row of ``branch_reviews`` in the
accounts database, beside the tickets and the instance requests, and the row is the record of what was asked, who
answered and what was done. Nothing here runs Git: ``branch_gate.py`` is the operator's tool that lands an accepted
change and removes branches, and it does only what these rows allow.

    submit(db, user_id=, username=, repository=, branch=, base=, head=, timeout_s=, delete_branch=, known=[...])
    remark(db, review_id, user_id=, username=, verdict="agree" | "object", note="")     a peer; never the author
    decide(db, review_id, user_id=, username=, head=, decision="approved" | "rejected", note="")   an admin
    withdraw(db, review_id, user_id=)                                                   the author
    standing(row, now)        waiting | objected | accepted | approved | rejected | withdrawn | landed
    landable(db, now)         the rows that may land now
    mark_landed(db, review_id, detail)

The rule (``standing``). An admin's decision settles a change at once. Without one, a change that nobody objects to
is accepted when its time-out passes: silence is consent. One standing objection from a peer stops the clock, and
the change then waits for an admin. A decision and a landing both name the commit they are about; a branch that
moved after it was submitted is a different change and is submitted again, which withdraws the older row.

``delete_branch`` is the author's choice of what happens to the branch after it lands: removed, or kept. A kept
branch is never cleaned later.

``branch_merged_seen`` is the cleaner's ledger: when a branch was first seen fully merged at a commit. The grace
period of the cleaner counts from that moment, so nothing is removed the first time it is looked at.

A branch is registered to an account by checking it out from the server (``register``) or by submitting it
(``branch_owners``). A registered branch is its owner's to keep tidy and is given a long life by the cleaner; a branch
registered to nobody is given a short one. ``branch_removed`` records every branch the cleaner removed and the commit
it was at, so a removal can be undone while Git still holds the commit.

``advice`` is the one text that says how a change is made here. The lock prints it, in mock and in earnest.

Who a caller is comes from the authenticated host, never from a request. Who is an admin is the ``account_roles``
table of the same database.
"""
from __future__ import annotations

import re
import time
import uuid
from pathlib import Path

STATUSES = ("open", "approved", "rejected", "withdrawn")
VERDICTS = ("agree", "object")
MAX_NOTE = 2000
MAX_TIMEOUT_S = 30 * 86400.0
HEAD = re.compile(r"[0-9a-f]{40}\Z")
BRANCH = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}\Z")
DDL = (
    "CREATE TABLE IF NOT EXISTS branch_reviews (id TEXT PRIMARY KEY, repository TEXT NOT NULL, branch TEXT NOT NULL, "
    "base TEXT NOT NULL, head TEXT NOT NULL, owner_id INTEGER NOT NULL, owner_name TEXT, created_at REAL NOT NULL, "
    "accept_after REAL NOT NULL, delete_branch INTEGER NOT NULL CHECK(delete_branch IN (0,1)), "
    "status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','approved','rejected','withdrawn')), "
    "reviewed_at REAL, reviewer_id INTEGER, reviewer_name TEXT, note TEXT NOT NULL DEFAULT '', "
    "landed_at REAL, landed_detail TEXT, land_tried_at REAL, land_result TEXT, UNIQUE(repository, branch, head))",
    "CREATE TABLE IF NOT EXISTS branch_review_remarks (review_id TEXT NOT NULL, user_id INTEGER NOT NULL, username TEXT, "
    "verdict TEXT NOT NULL CHECK(verdict IN ('agree','object')), note TEXT NOT NULL DEFAULT '', at REAL NOT NULL, "
    "PRIMARY KEY(review_id, user_id))",
    "CREATE TABLE IF NOT EXISTS branch_merged_seen (repository TEXT NOT NULL, branch TEXT NOT NULL, head TEXT NOT NULL, "
    "seen_at REAL NOT NULL, PRIMARY KEY(repository, branch))",
    "CREATE TABLE IF NOT EXISTS branch_owners (repository TEXT NOT NULL, branch TEXT NOT NULL, user_id INTEGER NOT NULL, "
    "username TEXT, registered_at REAL NOT NULL, PRIMARY KEY(repository, branch))",
    "CREATE TABLE IF NOT EXISTS branch_removed (repository TEXT NOT NULL, branch TEXT NOT NULL, head TEXT NOT NULL, "
    "removed_at REAL NOT NULL, reason TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS worktree_verdicts (path TEXT PRIMARY KEY, verdict TEXT NOT NULL CHECK(verdict IN "
    "('trash','archive','keep')), note TEXT NOT NULL DEFAULT '', decided_by TEXT NOT NULL, decided_at REAL NOT NULL)",
    # The Git author identity an account declares for its own commits (identity()/set_identity()).
    "CREATE TABLE IF NOT EXISTS account_git_identity (user_id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL, "
    "declared_at REAL NOT NULL)",
)
_COLUMNS = ("id", "repository", "branch", "base", "head", "owner_id", "owner_name", "created_at", "accept_after",
            "delete_branch", "status", "reviewed_at", "reviewer_id", "reviewer_name", "note", "landed_at", "landed_detail",
            "land_tried_at", "land_result")


class ReviewError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def ensure(db) -> None:
    for statement in DDL:
        db.execute(statement)
    if "removed_by" not in {row[1] for row in db.execute("PRAGMA table_info(branch_removed)")}:
        db.execute("ALTER TABLE branch_removed ADD COLUMN removed_by TEXT")      # notices made before the cleanup account
    columns = {row[1] for row in db.execute("PRAGMA table_info(branch_reviews)")}
    for column, kind in (("land_tried_at", "REAL"), ("land_result", "TEXT")):   # the landing queue: reviews made before it
        if column not in columns:
            db.execute(f"ALTER TABLE branch_reviews ADD COLUMN {column} {kind}")


_EMAIL = re.compile(r"^[^@\s<>]+@[^@\s<>]+$")


def set_identity(db, *, user_id, name, email, now=None) -> dict:
    """An account declares the Git author name and email its commits carry. Its own; one row, replaced."""
    _account(user_id)
    if not isinstance(name, str) or not name.strip() or len(name) > 100 or any(c in name for c in "<>\r\n"):
        raise ReviewError("name must be a short author name without < > or line breaks")
    if not isinstance(email, str) or not _EMAIL.match(email) or len(email) > 200:
        raise ReviewError("email must be one address")
    ensure(db)
    db.execute("INSERT OR REPLACE INTO account_git_identity VALUES (?,?,?,?)",
               (user_id, name.strip(), email.strip(), time.time() if now is None else now))
    return identity(db, user_id)


def identity(db, user_id) -> dict | None:
    ensure(db)
    row = db.execute("SELECT name, email, declared_at FROM account_git_identity WHERE user_id = ?", (int(user_id),)).fetchone()
    return None if row is None else {"name": row[0], "email": row[1], "declared_at": row[2]}


def authorship(db, review, authors) -> str:
    """Whether a review's commit authors are its submitter's declared identity: one sentence for the record."""
    declared = identity(db, review["owner_id"])
    if declared is None:
        return f"{review['owner_name']} has declared no Git identity"
    mine = f"<{declared['email']}>".lower()
    others = [a for a in authors if not a.lower().endswith(mine)]
    return (f"every commit is by {review['owner_name']}'s declared identity" if not others
            else f"commits not by {review['owner_name']}'s declared identity: {', '.join(others)}")


def is_service(db, username) -> bool:
    """A service account of this name exists: the only kind of account that may be the cleaner."""
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'account_roles'").fetchone():
        return False
    return db.execute("SELECT 1 FROM users u JOIN account_roles r ON r.user_id = u.id WHERE u.username = ? COLLATE NOCASE "
                      "AND r.role = 'service'", (str(username),)).fetchone() is not None


def removed(db, limit: int = 200) -> list:
    """The notices of removal, newest first."""
    ensure(db)
    return [{"repository": r[0], "branch": r[1], "head": r[2], "removed_at": r[3], "reason": r[4], "removed_by": r[5]}
            for r in db.execute("SELECT repository, branch, head, removed_at, reason, removed_by FROM branch_removed "
                                "ORDER BY removed_at DESC LIMIT ?", (int(limit),))]


def _account(user_id) -> int:
    if isinstance(user_id, bool) or not isinstance(user_id, int):
        raise ReviewError("a signed-in account is required", 401)
    return user_id


def _note(note) -> str:
    if not isinstance(note, str) or len(note) > MAX_NOTE:
        raise ReviewError(f"note must be text of at most {MAX_NOTE} characters")
    return note.strip()


def is_admin(db, user_id) -> bool:
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'account_roles'").fetchone():
        return False
    return db.execute("SELECT 1 FROM account_roles WHERE user_id = ? AND role = 'admin'", (user_id,)).fetchone() is not None


def _record(db, row) -> dict:
    record = dict(zip(_COLUMNS, row))
    record["delete_branch"] = bool(record["delete_branch"])
    record["remarks"] = [{"username": r[0], "verdict": r[1], "note": r[2], "at": r[3]} for r in db.execute(
        "SELECT username, verdict, note, at FROM branch_review_remarks WHERE review_id = ? ORDER BY at", (record["id"],))]
    return record


def get(db, review_id, now=None) -> dict:
    ensure(db)
    row = db.execute(f"SELECT {', '.join(_COLUMNS)} FROM branch_reviews WHERE id = ?", (str(review_id),)).fetchone()
    if row is None:
        raise ReviewError("review not found", 404)
    record = _record(db, row)
    record["standing"] = standing(record, time.time() if now is None else now)
    return record


def listing(db, now=None, *, limit: int = 100) -> list:
    """Every change still in play, then the most recent settled ones."""
    ensure(db)
    now = time.time() if now is None else now
    rows = db.execute(f"SELECT {', '.join(_COLUMNS)} FROM branch_reviews ORDER BY (status = 'open' AND landed_at IS NULL) DESC, "
                      "created_at DESC LIMIT ?", (int(limit),)).fetchall()
    out = []
    for row in rows:
        record = _record(db, row)
        record["standing"] = standing(record, now)
        out.append(record)
    return out


def standing(record: dict, now: float) -> str:
    """Where one change stands. The one rule everything else reads."""
    if record["landed_at"] is not None:
        return "landed"
    if record["status"] != "open":
        return record["status"]
    if any(remark["verdict"] == "object" for remark in record["remarks"]):
        return "objected"
    return "accepted" if now >= record["accept_after"] else "waiting"


def submit(db, *, user_id, username, repository, branch, base, head, timeout_s, delete_branch, known, now=None) -> dict:
    ensure(db)
    _account(user_id)
    if repository not in known:
        raise ReviewError("repository must be one the policy names: " + ", ".join(sorted(known)), 404)
    for value, what in ((branch, "branch"), (base, "base")):
        if not isinstance(value, str) or not BRANCH.match(value) or ".." in value or value.endswith(("/", ".lock")):
            raise ReviewError(f"{what} must be a plain branch name")
    if branch == base:
        raise ReviewError("a default branch is not put up for review against itself")
    if not isinstance(head, str) or not HEAD.match(head):
        raise ReviewError("head must be the full 40-character commit id")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not 0 <= timeout_s <= MAX_TIMEOUT_S:
        raise ReviewError(f"timeout must be between 0 and {int(MAX_TIMEOUT_S)} seconds")
    if not isinstance(delete_branch, bool):
        raise ReviewError("delete_branch must be true or false")
    now = time.time() if now is None else now
    db.execute("BEGIN IMMEDIATE")
    try:
        old = db.execute("SELECT id, owner_id FROM branch_reviews WHERE repository = ? AND branch = ? AND head = ?",
                         (repository, branch, head)).fetchone()
        if old is not None:                      # the same change again: the same row, whoever asks
            db.execute("COMMIT")
            return get(db, old[0], now)
        review_id = uuid.uuid4().hex
        db.execute("UPDATE branch_reviews SET status = 'withdrawn', reviewed_at = ?, note = ? WHERE repository = ? AND branch = ? "
                   "AND status = 'open' AND landed_at IS NULL", (now, f"superseded by {review_id}", repository, branch))
        db.execute("INSERT INTO branch_reviews (id, repository, branch, base, head, owner_id, owner_name, created_at, "
                   "accept_after, delete_branch) VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (review_id, repository, branch, base, head, user_id, username, now, now + float(timeout_s), int(delete_branch)))
        db.execute("INSERT OR IGNORE INTO branch_owners VALUES (?,?,?,?,?)", (repository, branch, user_id, username, now))
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    return get(db, review_id, now)


def remark(db, review_id, *, user_id, username, verdict, note="", now=None) -> dict:
    """A peer agrees or objects. One remark per account; a later one replaces it."""
    _account(user_id)
    if verdict not in VERDICTS:
        raise ReviewError("verdict must be agree or object")
    note = _note(note)
    now = time.time() if now is None else now
    record = get(db, review_id, now)
    if record["owner_id"] == user_id:
        raise ReviewError("the author does not review their own change", 403)
    if record["status"] != "open" or record["landed_at"] is not None:
        raise ReviewError("this change is already settled", 409)
    db.execute("INSERT INTO branch_review_remarks VALUES (?,?,?,?,?,?) ON CONFLICT(review_id, user_id) DO UPDATE SET "
               "verdict = excluded.verdict, note = excluded.note, at = excluded.at, username = excluded.username",
               (record["id"], user_id, username, verdict, note, now))
    return get(db, review_id, now)


def decide(db, review_id, *, user_id, username, head, decision, note="", now=None) -> dict:
    """An admin approves or rejects exactly the commit they read."""
    _account(user_id)
    if decision not in ("approved", "rejected"):
        raise ReviewError("decision must be approved or rejected")
    note = _note(note)
    ensure(db)
    if not is_admin(db, user_id):
        raise ReviewError("an admin decides a review", 403)
    now = time.time() if now is None else now
    done = db.execute("UPDATE branch_reviews SET status = ?, reviewed_at = ?, reviewer_id = ?, reviewer_name = ?, note = ? "
                      "WHERE id = ? AND status = 'open' AND landed_at IS NULL AND head = ?",
                      (decision, now, user_id, username, note, str(review_id), head))
    if done.rowcount != 1:
        get(db, review_id, now)                  # 404 when there is no such review
        raise ReviewError("the review is already settled, or it is about a different commit", 409)
    return get(db, review_id, now)


def withdraw(db, review_id, *, user_id, now=None) -> dict:
    _account(user_id)
    now = time.time() if now is None else now
    done = db.execute("UPDATE branch_reviews SET status = 'withdrawn', reviewed_at = ?, note = 'withdrawn by its author' "
                      "WHERE id = ? AND owner_id = ? AND status = 'open' AND landed_at IS NULL", (now, str(review_id), user_id))
    if done.rowcount != 1:
        get(db, review_id, now)
        raise ReviewError("only the author withdraws a change, and only while it is open", 403)
    return get(db, review_id, now)


def landable(db, now=None) -> list:
    now = time.time() if now is None else now
    return [record for record in listing(db, now, limit=1000) if record["standing"] in ("approved", "accepted")]


def accepted_head(db, repository, base, head, now=None) -> bool:
    """True when ``head`` may be the tip of ``base``: a change at that commit is accepted, approved or already landed."""
    now = time.time() if now is None else now
    ensure(db)
    rows = db.execute(f"SELECT {', '.join(_COLUMNS)} FROM branch_reviews WHERE repository = ? AND base = ? AND head = ?",
                      (repository, base, head)).fetchall()
    return any(standing(_record(db, row), now) in ("approved", "accepted", "landed") for row in rows)


def mark_landed(db, review_id, detail, now=None) -> dict:
    now = time.time() if now is None else now
    db.execute("UPDATE branch_reviews SET landed_at = ?, landed_detail = ? WHERE id = ? AND landed_at IS NULL",
               (now, str(detail)[:500], str(review_id)))
    return get(db, review_id, now)


def mark_tried(db, review_id, result, now=None) -> None:
    """The landing task's last attempt at a change it did not land: when, and why it held (the landing queue shows it)."""
    now = time.time() if now is None else now
    db.execute("UPDATE branch_reviews SET land_tried_at = ?, land_result = ? WHERE id = ? AND landed_at IS NULL",
               (now, str(result)[:500], str(review_id)))


def queue(db, now=None) -> list:
    """The landing queue: every change a review accepted that has not landed, oldest first, with its last attempt."""
    return sorted(landable(db, now), key=lambda record: record["created_at"])


def next_scheduled_landing(table_path, now=None) -> dict | None:
    """Read the installed cron table; report only schedules we can time exactly."""
    if table_path is None:
        return None
    try:
        lines = Path(table_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    matches = [line.split() for line in lines if line.strip() and not line.lstrip().startswith("#")
               and line.split()[0] == "reviews.land"]
    if len(matches) != 1 or len(matches[0]) < 6:
        return None
    fields = matches[0]
    minute = re.fullmatch(r"\*/([1-9][0-9]*)", fields[1])
    if (minute is None or fields[2:6] != ["*", "*", "*", "*"] or
            60 % int(minute.group(1)) != 0):
        return None
    interval = int(minute.group(1)) * 60
    current = time.time() if now is None else now
    return {"next_at": (int(current // interval) + 1) * interval,
            "interval_seconds": interval}


def land_now(db, body, *, user_id, username, land_request) -> dict:
    """An admin asks the updater (the task worker's reviews.land) to land changes of the queue now, instead of at its next
    scheduled run: ``{"ids": [...]}`` for those, ``{"all": true}`` for the whole queue. The request is one task asked of the
    cron table (nodus tools/task_request.py); the updater lands what is a fast-forward and records why it held the rest,
    which the queue then shows. Answers the ids asked for and the updater's outcome (``done``, or ``timeout`` when it is
    still working: the result arrives in the queue either way)."""
    _account(user_id)
    ensure(db)
    if not is_admin(db, user_id):
        raise ReviewError("an admin lands changes", 403)
    if land_request is None:
        raise ReviewError("this server cannot reach the updater: its instance declares no system_events", 503)
    waiting = {record["id"]: record for record in queue(db)}
    if body.get("all") is True:
        ids = list(waiting)
    else:
        ids = body.get("ids")
        if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
            raise ReviewError("ids must be a list of review ids, or all must be true")
        missing = [i for i in ids if i not in waiting]
        if missing:
            raise ReviewError(f"not in the landing queue (not accepted, or already landed): {', '.join(missing)}", 409)
    if not ids:
        return {"asked": [], "outcome": {"outcome": "nothing to land"}}
    outcome = land_request(None if body.get("all") is True else ids)
    print(f"[reviews] {username} land now: {len(ids)} change(s) -> {outcome.get('outcome')}", flush=True)
    return {"asked": ids, "outcome": outcome}


def kept(db, repository, branch) -> bool:
    """A branch whose author asked for it to be kept after landing."""
    ensure(db)
    return db.execute("SELECT 1 FROM branch_reviews WHERE repository = ? AND branch = ? AND delete_branch = 0 AND "
                      "landed_at IS NOT NULL", (repository, branch)).fetchone() is not None


def merged_since(db, repository, branch, head, now=None) -> float:
    """When this branch was first seen fully merged at this commit; recorded now if never, or if the branch moved."""
    ensure(db)
    now = time.time() if now is None else now
    row = db.execute("SELECT head, seen_at FROM branch_merged_seen WHERE repository = ? AND branch = ?", (repository, branch)).fetchone()
    if row is not None and row[0] == head:
        return row[1]
    db.execute("INSERT INTO branch_merged_seen VALUES (?,?,?,?) ON CONFLICT(repository, branch) DO UPDATE SET "
               "head = excluded.head, seen_at = excluded.seen_at", (repository, branch, head, now))
    return now


def forget_merged(db, repository, branch) -> None:
    ensure(db)
    db.execute("DELETE FROM branch_merged_seen WHERE repository = ? AND branch = ?", (repository, branch))


def _branch(value, what="branch") -> str:
    if not isinstance(value, str) or not BRANCH.match(value) or ".." in value or value.endswith(("/", ".lock")):
        raise ReviewError(f"{what} must be a plain branch name")
    return value


def register(db, *, user_id, username, repository, branch, known, now=None) -> dict:
    """Check a branch out from the server: it is this account's from now on. The first to ask holds it."""
    ensure(db)
    _account(user_id)
    if repository not in known:
        raise ReviewError("repository must be one the policy names: " + ", ".join(sorted(known)), 404)
    _branch(branch)
    now = time.time() if now is None else now
    db.execute("INSERT OR IGNORE INTO branch_owners VALUES (?,?,?,?,?)", (repository, branch, user_id, username, now))
    held = owner(db, repository, branch)
    if held["user_id"] != user_id:
        raise ReviewError(f"that branch is registered to {held['username']}; choose another name", 409)
    return held


def owner(db, repository, branch) -> dict | None:
    ensure(db)
    row = db.execute("SELECT user_id, username, registered_at FROM branch_owners WHERE repository = ? AND branch = ?",
                     (repository, branch)).fetchone()
    return None if row is None else {"repository": repository, "branch": branch, "user_id": row[0], "username": row[1],
                                     "registered_at": row[2]}


def branches_of(db, user_id) -> list:
    ensure(db)
    return [{"repository": r[0], "branch": r[1], "registered_at": r[2]} for r in db.execute(
        "SELECT repository, branch, registered_at FROM branch_owners WHERE user_id = ? ORDER BY registered_at DESC", (user_id,))]


def in_review(db, repository, branch) -> bool:
    """A change of this branch is open or approved and has not landed yet."""
    ensure(db)
    return db.execute("SELECT 1 FROM branch_reviews WHERE repository = ? AND branch = ? AND status IN ('open','approved') "
                      "AND landed_at IS NULL", (repository, branch)).fetchone() is not None


TREE_VERDICTS = ("trash", "archive", "keep")


def _tree_key(path) -> str:
    return str(path).replace("\\", "/").rstrip("/").casefold()


def set_verdict(db, path, verdict, *, decided_by, note="", now=None) -> dict:
    """A reviewer's decision about one side worktree that holds uncommitted work.

    trash: its uncommitted work is abandoned and goes with it.  archive: commit that work to its branch (or to a new
    archive branch for a detached tree), push it, then remove the tree.  keep: leave it alone."""
    ensure(db)
    if verdict not in TREE_VERDICTS:
        raise ReviewError("verdict must be trash, archive or keep")
    if not isinstance(decided_by, str) or not decided_by:
        raise ReviewError("a reviewer is required", 401)
    now = time.time() if now is None else now
    db.execute("INSERT INTO worktree_verdicts VALUES (?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET verdict = excluded.verdict, "
               "note = excluded.note, decided_by = excluded.decided_by, decided_at = excluded.decided_at",
               (_tree_key(path), verdict, _note(note), decided_by, now))
    return verdict_for(db, path)


def verdict_for(db, path) -> dict | None:
    ensure(db)
    row = db.execute("SELECT verdict, note, decided_by, decided_at FROM worktree_verdicts WHERE path = ?", (_tree_key(path),)).fetchone()
    return None if row is None else {"path": _tree_key(path), "verdict": row[0], "note": row[1], "decided_by": row[2], "decided_at": row[3]}


def record_removed(db, repository, branch, head, reason, now=None, *, keep_owner=False, removed_by=None) -> None:
    """The notice of removal: what was removed from this computer, the commit it was at, and why. ``keep_owner``
    is for a branch that lives on, on the remote: its registration stays."""
    ensure(db)
    db.execute("INSERT INTO branch_removed (repository, branch, head, removed_at, reason, removed_by) VALUES (?,?,?,?,?,?)",
               (repository, branch, head, time.time() if now is None else now, str(reason)[:500], removed_by))
    if not keep_owner:
        db.execute("DELETE FROM branch_owners WHERE repository = ? AND branch = ?", (repository, branch))


def advice(policy, repository, head=None) -> str:
    """How a change is made here, in a few lines. The lock says this and nothing else."""
    declared = policy["repositories"][repository]
    default, mock = declared["default"], policy.get("lock") != "enforce"
    first = f"review gate{' (mock: allowed this time)' if mock else ''}: {default} of {repository} takes only reviewed commits"
    first += f", and {head[:12]} has no accepted review." if head else "."
    return "\n".join([
        first,
        "Do it through the speaktome-api MCP (python -m speaktome_mcp.api), not by pushing:",
        f"  1. reviews_checkout repository={repository} branch=<your-branch>     the branch is registered to you",
        f"  2. git switch -c <your-branch> {default}     then commit there; do not push",
        f"  3. reviews_submit repository={repository} branch=<your-branch> head=<full commit id>",
        f"An admin approves it, or it is accepted after {policy['review_timeout_hours']:g} h with no objection. "
        "The admin tool then pushes and lands it; nobody else talks to GitHub.",
    ])


def http_route(self, path, method, *, policy, accounts_db, land_request=None, landing_schedule_path=None):
    """The reviews over REST. ``self`` is the host's request handler; ``policy`` the loaded branch policy.

        GET  /api/reviews                 the changes in play and the recent settled ones (dev_ui, signed in), and the
                                          landing queue: accepted changes not yet landed, with the last attempt's result
        POST /api/reviews/land            {ids: [...]} or {all: true}   an admin asks the updater to land them now
        POST /api/reviews                 {repository, branch, head, timeout_hours?, keep?}   submit
        POST /api/reviews/remark          {id, verdict, note?}                                a peer
        POST /api/reviews/decide          {id, head, decision, note?}                         an admin
        POST /api/reviews/withdraw        {id}                                                the author
        POST /api/reviews/checkout        {repository, branch}      register a branch to yourself; answers the steps
        GET  /api/reviews/identity        the Git author identity this account declared
        POST /api/reviews/identity        {name, email}             declare it (each account its own)
        GET  /api/reviews/removed         the notices of removal: what left this computer, at which commit, by whom
        GET  /api/reviews/standing?repository=NAME&head=COMMIT      open: {"accepted", "lock", "advice"}

    ``standing`` is what the required check on GitHub asks. It answers one yes or no about one commit and nothing
    else, so it needs no account. Everything else takes the caller from the authenticated session.
    """
    import sqlite3
    from urllib.parse import parse_qs
    self._set_cookie = []
    if method == "POST":
        self.close_connection = True
    known = policy["repositories"]
    db = sqlite3.connect(accounts_db, timeout=10, isolation_level=None)
    try:
        if method == "GET" and path == "/api/reviews/standing":
            query = parse_qs(self.path.partition("?")[2])
            repository, head = (query.get("repository") or [""])[0], (query.get("head") or [""])[0]
            if repository not in known or not HEAD.match(head):
                return self._json({"error": "repository and a full commit id are required"}, 400)
            accepted = accepted_head(db, repository, known[repository]["default"], head)
            return self._json({"repository": repository, "head": head, "accepted": accepted, "lock": policy.get("lock", "mock"),
                               "advice": "" if accepted else advice(policy, repository, head)})
        principal = self._principal()
        if principal.user_id is None:
            return self._json({"error": "sign in first"}, 401)
        if "dev_ui" not in self._caps():
            return self._json({"error": "this needs dev_ui"}, 403)
        who = {"user_id": principal.user_id, "username": principal.username}
        if method == "GET" and path == "/api/reviews/removed":
            return self._json({"removed": removed(db), "cleaner": policy.get("cleaner")})
        if method == "GET" and path == "/api/reviews/identity":
            return self._json({"identity": identity(db, principal.user_id)})
        if method == "GET" and path == "/api/reviews":
            return self._json({"reviews": listing(db), "repositories": {name: known[name]["default"] for name in sorted(known)},
                               "your_branches": branches_of(db, principal.user_id), "lock": policy.get("lock", "mock"),
                               "queue": queue(db), "updater": land_request is not None, "apply": bool(policy.get("apply")),
                               "server_now": time.time(), "automatic_landing": next_scheduled_landing(landing_schedule_path)})
        if method != "POST":
            return self._json({"error": "not found"}, 404)
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            return self._json({"error": "JSON only"}, 415)
        body = self._body_json(8192)
        if body is None:
            return
        if not isinstance(body, dict):
            return self._json({"error": "a JSON object is required"}, 400)
        if path == "/api/reviews":
            repository = body.get("repository")
            hours = body.get("timeout_hours", policy["review_timeout_hours"])
            if isinstance(hours, bool) or not isinstance(hours, (int, float)):
                raise ReviewError("timeout_hours must be a number")
            keep = body.get("keep", not policy["delete_after_landing"])
            if not isinstance(keep, bool):
                raise ReviewError("keep must be true or false")
            base = known[repository]["default"] if isinstance(repository, str) and repository in known else ""
            made = submit(db, **who, repository=repository, branch=body.get("branch"), base=base, head=body.get("head"),
                          timeout_s=hours * 3600.0, delete_branch=not keep, known=list(known))
        elif path == "/api/reviews/identity":
            return self._json({"identity": set_identity(db, user_id=principal.user_id, name=body.get("name"),
                                                        email=body.get("email"))})
        elif path == "/api/reviews/checkout":
            held = register(db, **who, repository=body.get("repository"), branch=body.get("branch"), known=list(known))
            print(f"[reviews] {principal.username} checkout {held['repository']} {held['branch']}", flush=True)
            return self._json({"branch": held, "base": known[held["repository"]]["default"],
                               "steps": advice(policy, held["repository"]).splitlines()[2:]})
        elif path == "/api/reviews/remark":
            made = remark(db, body.get("id"), **who, verdict=body.get("verdict"), note=body.get("note", ""))
        elif path == "/api/reviews/decide":
            made = decide(db, body.get("id"), **who, head=body.get("head"), decision=body.get("decision"), note=body.get("note", ""))
        elif path == "/api/reviews/withdraw":
            made = withdraw(db, body.get("id"), user_id=principal.user_id)
        elif path == "/api/reviews/land":
            return self._json(land_now(db, body, user_id=principal.user_id, username=principal.username,
                                       land_request=land_request))
        else:
            return self._json({"error": "not found"}, 404)
        print(f"[reviews] {principal.username} {path.rsplit('/', 1)[1]} {made['repository']} {made['branch']} -> {made['standing']}", flush=True)
        return self._json({"review": made})
    except ReviewError as error:
        return self._json({"error": error.message}, error.status)
    finally:
        db.close()
