"""Bundles of flagged branches and redelegated reviews: the consolidation and redelegation queues of the reviews.

Both queues hold only work someone flagged by hand. A branch nobody flagged passes them by exactly as before: its reviews
stand by ``branch_review.standing`` alone. Being held up is the consequence of having been identified, never a default.

**Bundle (the consolidation queue).** An operator or an admin looks at every branch checked out (``checked_out``: the
central checkouts and the registered branches), flags a group of them in one repository and bundles them (``flag``).
Every owner of a bundled branch is told (the notice, ``notice``) that their versions are to be resolved into one merge.
From then on every review of a bundled branch stands ``consolidating``: it is never accepted by silence and never lands.
The bundle is resolved by one meta review (``consolidate``): a review in the same repository whose commit contains the
current tip of every bundled branch, so the branches have been merged across each other. The bundle is then
``merged``; the other members' open reviews are withdrawn as consolidated into it; the meta review goes on through the
ordinary review. An operator or admin may instead dissolve a bundle (``dissolve``): its reviews flow again as before.

**Closing a branch, item by item.** Some bundled work was abandoned on purpose, or only needs a continuation report.
An operator or admin (or the branch's own owner) closes one branch of a bundle with a reason (``close``): it leaves
the consolidation, its reviews in play are withdrawn with that reason, and the meta review no longer has to contain it.
With ``remove`` the branch is also to be removed, and ``work`` says what happens to its unlanded work: ``archive``
commits any uncommitted work and pushes the branch's final commit to ``archive/<branch>`` on the remote first;
``discard`` lets it go. Nothing is removed here: this records the decision, and the cleanup account carries it out at
its next run (``branch_gate.carry_closures``: the worktree, the central checkout and its loan, the local branch and,
where the remote's rules allow, the remote branch), writing the notice of removal with the closer and the reason.
When every branch of a bundle is closed the bundle is dissolved.

**Redelegation (the redelegation queue).** An operator or an admin turns a review, or a bundle, into one issue ticket
that names the separate paths the work should take instead (``redelegate``). The source then stands ``redelegated``:
the review (or every review of the bundle's branches made before the redelegation) never lands. The work comes back as
new branches and new reviews, one per path, each meeting the ordinary review. The ticket is made through the issue
tickets (issue_tickets.py) by the caller's own account: ``open_ticket(title, description, diagnostics)`` is the host's.

Rows (accounts database, beside ``branch_reviews``; the DDL is branch_review's):

    review_bundles          id, repository, created_at, created_by, reason, status, meta_review_id, settled_at,
                            settled_by, settle_note          status consolidating | merged | dissolved | redelegated
    review_bundle_members   bundle_id, repository, branch, owner_name, flagged_at
    review_redelegations    id, source_kind (review | bundle), source_id, repository, created_at, created_by, note,
                            paths (JSON list of {title, detail}), ticket_id
    review_bundle_closures  bundle_id, repository, branch, closed_at, closed_by, reason, remove, work (archive |
                            discard), carried_at, carried      one row per branch closed out of a bundle
"""
from __future__ import annotations

import json
import time
import uuid

from meowdoku import branch_review
from meowdoku.branch_review import ReviewError

MIN_MEMBERS = 2
MAX_MEMBERS = 20
MIN_PATHS = 2
MAX_PATHS = 12
MAX_TITLE = 200
MAX_DETAIL = 2000
BUNDLE_STATUSES = ("consolidating", "merged", "dissolved", "redelegated")
CLOSE_WORK = ("archive", "discard")
_CLOSURE_COLUMNS = ("closed_at", "closed_by", "reason", "remove", "work", "carried_at", "carried")
_BUNDLE_COLUMNS = ("id", "repository", "created_at", "created_by", "reason", "status", "meta_review_id", "settled_at",
                   "settled_by", "settle_note")


def _moderator(db, user_id, what) -> None:
    branch_review._account(user_id)
    branch_review.ensure(db)
    if not branch_review.may_moderate(db, user_id):
        raise ReviewError(f"an operator or admin {what}", 403)


def _reason(value, what="reason") -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > branch_review.MAX_NOTE:
        raise ReviewError(f"a {what} must be text of 1 to {branch_review.MAX_NOTE} characters")
    return value.strip()


def checked_out(db, now=None) -> list:
    """Every branch checked out: each central checkout and each registered branch, once, with its owner, the bundle it
    is in (if it is consolidating) and its review still in play (if any). What a moderator chooses a bundle from."""
    branch_review.ensure(db)
    now = time.time() if now is None else now
    rows: dict = {}
    if db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'checkouts'").fetchone():
        for repository, branch, username, base, opened in db.execute(
                "SELECT repository, branch, username, base, created_at FROM checkouts"):
            rows[(repository, branch)] = {"repository": repository, "branch": branch, "owner": username, "base": base,
                                          "opened_at": opened, "checked_out": True}
    for repository, branch, username, registered in db.execute(
            "SELECT repository, branch, username, registered_at FROM branch_owners"):
        rows.setdefault((repository, branch), {"repository": repository, "branch": branch, "owner": username,
                                               "base": None, "opened_at": registered, "checked_out": False})
    consolidating = {
        (repository, branch): bundle_id for bundle_id, repository, branch in db.execute(
            "SELECT m.bundle_id, m.repository, m.branch FROM review_bundle_members m JOIN review_bundles b "
            "ON b.id = m.bundle_id WHERE b.status = 'consolidating' AND NOT EXISTS (SELECT 1 FROM "
            "review_bundle_closures c WHERE c.bundle_id = m.bundle_id AND c.branch = m.branch)")}
    in_play, latest = {}, {}
    for record in branch_review.listing(db, now, limit=1000):
        key = (record["repository"], record["branch"])
        seen = {"id": record["id"], "head": record["head"], "standing": record["standing"]}
        if key not in latest or record["created_at"] > latest[key][0]:
            latest[key] = (record["created_at"], seen)
        if record["standing"] not in ("landed", "withdrawn", "rejected"):
            in_play.setdefault(key, seen)
    removing = set(db.execute(
        "SELECT repository, branch FROM review_bundle_closures WHERE remove = 1 AND carried_at IS NULL UNION "
        "SELECT repository, branch FROM branch_deletions WHERE carried_at IS NULL"))
    out = []
    for key in sorted(rows, key=lambda k: (k[0], k[1])):
        row = rows[key]
        row["bundle"] = consolidating.get(key)
        row["review"] = in_play.get(key)
        row["latest_review"] = latest[key][1] if key in latest else None     # what "reopen" puts back up
        row["removing"] = key in removing                                    # the cleanup account is still to remove it
        out.append(row)
    return out


def get(db, bundle_id, now=None) -> dict:
    branch_review.ensure(db)
    row = db.execute(f"SELECT {', '.join(_BUNDLE_COLUMNS)} FROM review_bundles WHERE id = ?", (str(bundle_id),)).fetchone()
    if row is None:
        raise ReviewError("bundle not found", 404)
    now = time.time() if now is None else now
    bundle = dict(zip(_BUNDLE_COLUMNS, row))
    members = []
    for branch, owner_name, flagged_at in db.execute(
            "SELECT branch, owner_name, flagged_at FROM review_bundle_members WHERE bundle_id = ? ORDER BY branch",
            (bundle["id"],)):
        review = None
        latest = db.execute("SELECT id FROM branch_reviews WHERE repository = ? AND branch = ? "
                            "ORDER BY created_at DESC LIMIT 1", (bundle["repository"], branch)).fetchone()
        if latest is not None:
            record = branch_review.get(db, latest[0], now)
            review = {"id": record["id"], "head": record["head"], "standing": record["standing"]}
        closed = db.execute(f"SELECT {', '.join(_CLOSURE_COLUMNS)} FROM review_bundle_closures WHERE bundle_id = ? "
                            "AND branch = ?", (bundle["id"], branch)).fetchone()
        if closed is not None:
            closed = dict(zip(_CLOSURE_COLUMNS, closed))
            closed["remove"] = bool(closed["remove"])
        members.append({"branch": branch, "owner": owner_name, "flagged_at": flagged_at, "review": review,
                        "closed": closed})
    bundle["members"] = members
    bundle["owners"] = sorted({member["owner"] for member in members if member["owner"]})
    bundle["notice"] = notice(bundle)
    return bundle


def listing(db, now=None, *, limit: int = 100) -> list:
    """The bundles still consolidating, then the most recent settled ones."""
    branch_review.ensure(db)
    ids = [row[0] for row in db.execute("SELECT id FROM review_bundles ORDER BY (status = 'consolidating') DESC, "
                                        "created_at DESC LIMIT ?", (int(limit),))]
    return [get(db, bundle_id, now) for bundle_id in ids]


def notice(bundle) -> str:
    """The one text that tells a bundled branch's owner (or the conversation working on it) what is asked of it."""
    branches = [member["branch"] for member in bundle["members"]]
    if bundle["status"] != "consolidating":
        settled = {"merged": f"resolved by the meta review {str(bundle['meta_review_id'] or '')[:8]}",
                   "dissolved": "dissolved", "redelegated": "redelegated into a ticket"}[bundle["status"]]
        return (f"Bundle {bundle['id'][:8]} ({bundle['repository']}: {', '.join(branches)}) is {settled}. "
                "Nothing more is asked of its branches by this bundle.")
    open_members = [member for member in bundle["members"] if member["closed"] is None]
    listed = "\n".join(f"  - {member['branch']} ({member['owner'] or 'unregistered'})" for member in open_members)
    closed = [member for member in bundle["members"] if member["closed"] is not None]
    closed_lines = [
        f"  - {member['branch']} ({member['owner'] or 'unregistered'}): closed by {member['closed']['closed_by']}: "
        f"{member['closed']['reason']}"
        + (f"; to be removed, its work {'archived' if member['closed']['work'] == 'archive' else 'discarded'}"
           + (f" ({member['closed']['carried']})" if member["closed"]["carried_at"] else " at the cleanup's next run")
           if member["closed"]["remove"] else "; kept")
        for member in closed]
    return "\n".join([
        f"NOTICE: your work was flagged for consolidation (bundle {bundle['id']}, by {bundle['created_by']}).",
        f"Reason: {bundle['reason']}",
        f"These branches of {bundle['repository']} are to be resolved together into a single merge:",
        listed,
        *(["Closed out of this bundle (no longer part of the merge):", *closed_lines] if closed else []),
        "Until they are, every review of these branches stands 'consolidating': it is not accepted by silence and does "
        "not land.",
        "What to do:",
        "  1. Commit what you have on your branch (checkout_commit) and tell the other owners what it changes.",
        "  2. Together, merge every branch above into one branch, resolving the conflicts so the versions agree.",
        f"  3. Submit that one branch (reviews_submit repository={bundle['repository']} ...): the meta review.",
        f"  4. reviews_bundle_consolidate bundle={bundle['id']} id=<the meta review's id>. It is accepted only when its "
        "commit contains the current tip of every branch above.",
        "The meta review then goes through the ordinary review. Read the bundle with reviews_bundles.",
    ])


def flag(db, *, user_id, username, repository, branches, reason, known, now=None) -> dict:
    """An operator or admin bundles a group of branches of one repository. Answers the bundle (with its notice) and who
    is to be told: the owners of its branches, but the flagger."""
    _moderator(db, user_id, "bundles branches")
    if repository not in known:
        raise ReviewError("repository must be one the policy names: " + ", ".join(sorted(known)), 404)
    reason = _reason(reason)
    if (not isinstance(branches, list) or not all(isinstance(branch, str) for branch in branches)
            or not MIN_MEMBERS <= len(set(branches)) <= MAX_MEMBERS):
        raise ReviewError(f"branches must be a list of {MIN_MEMBERS} to {MAX_MEMBERS} distinct branch names")
    branches = sorted(set(branch_review._branch(branch) for branch in branches))
    now = time.time() if now is None else now
    known_branches = {(row["repository"], row["branch"]): row for row in checked_out(db, now)}
    for branch in branches:
        row = known_branches.get((repository, branch))
        if row is None:
            raise ReviewError(f"{branch} is not checked out or registered in {repository}", 404)
        if row["bundle"] is not None:
            raise ReviewError(f"{branch} is already in bundle {row['bundle'][:8]}, which is still consolidating", 409)
    bundle_id = uuid.uuid4().hex
    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute("INSERT INTO review_bundles (id, repository, created_at, created_by, reason, status) "
                   "VALUES (?,?,?,?,?,'consolidating')", (bundle_id, repository, now, str(username), reason))
        for branch in branches:
            db.execute("INSERT INTO review_bundle_members VALUES (?,?,?,?,?)",
                       (bundle_id, repository, branch, known_branches[(repository, branch)]["owner"], now))
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    bundle = get(db, bundle_id, now)
    return {"bundle": bundle, "tell": sorted(set(bundle["owners"]) - {str(username)})}


def _consolidating(db, bundle_id, now) -> dict:
    bundle = get(db, bundle_id, now)
    if bundle["status"] != "consolidating":
        raise ReviewError(f"bundle {bundle['id'][:8]} is {bundle['status']}, not consolidating", 409)
    return bundle


def consolidate(db, bundle_id, review_id, *, user_id, username, contains, now=None) -> dict:
    """The meta review resolves its bundle. ``contains(repository, branch, head)`` answers whether ``head`` contains the
    current tip of ``branch`` (the host asks Git). The caller is a member's owner, or an operator or admin."""
    branch_review._account(user_id)
    now = time.time() if now is None else now
    bundle = _consolidating(db, bundle_id, now)
    if str(username) not in bundle["owners"] and not branch_review.may_moderate(db, user_id):
        raise ReviewError("an owner of a bundled branch, or an operator or admin, consolidates a bundle", 403)
    meta = branch_review.get(db, review_id, now)
    if meta["repository"] != bundle["repository"]:
        raise ReviewError("the meta review must be in the bundle's repository", 409)
    if meta["standing"] in ("landed", "withdrawn", "rejected", "redelegated"):
        raise ReviewError(f"review {meta['id'][:8]} is {meta['standing']}; only a review in play is a meta review", 409)
    missing = [member["branch"] for member in bundle["members"] if member["closed"] is None
               and not contains(bundle["repository"], member["branch"], meta["head"])]
    if missing:
        raise ReviewError(f"the meta review's commit {meta['head'][:12]} does not contain the current tip of: "
                          + ", ".join(missing), 409)
    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute("UPDATE review_bundles SET status = 'merged', meta_review_id = ?, settled_at = ?, settled_by = ?, "
                   "settle_note = ? WHERE id = ? AND status = 'consolidating'",
                   (meta["id"], now, str(username), f"resolved by {meta['branch']} at {meta['head'][:12]}", bundle["id"]))
        for member in bundle["members"]:
            if member["branch"] == meta["branch"] or member["closed"] is not None:
                continue
            db.execute("UPDATE branch_reviews SET status = 'withdrawn', reviewed_at = ?, note = ? WHERE repository = ? "
                       "AND branch = ? AND status IN ('open', 'approved') AND landed_at IS NULL",
                       (now, f"consolidated into the meta review {meta['id']} (bundle {bundle['id']})",
                        bundle["repository"], member["branch"]))
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    bundle = get(db, bundle["id"], now)
    return {"bundle": bundle, "review": branch_review.get(db, meta["id"], now),
            "tell": sorted(set(bundle["owners"]) - {str(username)})}


def dissolve(db, bundle_id, *, user_id, username, reason, now=None) -> dict:
    """An operator or admin releases a bundle without a meta review: its reviews stand as before."""
    _moderator(db, user_id, "dissolves a bundle")
    reason = _reason(reason)
    now = time.time() if now is None else now
    bundle = _consolidating(db, bundle_id, now)
    db.execute("UPDATE review_bundles SET status = 'dissolved', settled_at = ?, settled_by = ?, settle_note = ? "
               "WHERE id = ? AND status = 'consolidating'", (now, str(username), reason, bundle["id"]))
    bundle = get(db, bundle["id"], now)
    return {"bundle": bundle, "tell": sorted(set(bundle["owners"]) - {str(username)})}


def close(db, bundle_id, branch, *, user_id, username, reason, remove=False, work=None, now=None) -> dict:
    """Close one branch of a bundle with a reason: it leaves the consolidation and its reviews in play are withdrawn.
    With ``remove`` it is also to be removed by the cleanup account (``branch_gate.carry_closures``), its unlanded work
    archived or discarded as ``work`` says. The caller is the branch's owner, or an operator or admin. Answers the
    bundle and who is to be told (the bundle's owners, but the caller)."""
    branch_review._account(user_id)
    reason = _reason(reason)
    if not isinstance(remove, bool):
        raise ReviewError("remove must be true or false")
    if remove and work not in CLOSE_WORK:
        raise ReviewError("a branch to be removed says what happens to its unlanded work: work is archive or discard")
    if not remove and work is not None:
        raise ReviewError("work applies only to a branch that is to be removed")
    now = time.time() if now is None else now
    bundle = _consolidating(db, bundle_id, now)
    member = next((item for item in bundle["members"] if item["branch"] == branch), None)
    if member is None:
        raise ReviewError(f"{branch} is not a branch of bundle {bundle['id'][:8]}", 404)
    if member["closed"] is not None:
        raise ReviewError(f"{branch} was already closed by {member['closed']['closed_by']}", 409)
    if str(username) != member["owner"] and not branch_review.may_moderate(db, user_id):
        raise ReviewError("the branch's owner, or an operator or admin, closes a bundled branch", 403)
    still_open = [item for item in bundle["members"] if item["closed"] is None and item["branch"] != branch]
    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute("INSERT INTO review_bundle_closures (bundle_id, repository, branch, closed_at, closed_by, reason, "
                   "remove, work) VALUES (?,?,?,?,?,?,?,?)",
                   (bundle["id"], bundle["repository"], branch, now, str(username), reason, int(remove), work))
        db.execute("UPDATE branch_reviews SET status = 'withdrawn', reviewed_at = ?, note = ? WHERE repository = ? "
                   "AND branch = ? AND status IN ('open', 'approved') AND landed_at IS NULL",
                   (now, f"closed out of bundle {bundle['id']} by {username}: {reason}", bundle["repository"], branch))
        if not still_open:
            db.execute("UPDATE review_bundles SET status = 'dissolved', settled_at = ?, settled_by = ?, settle_note = ? "
                       "WHERE id = ? AND status = 'consolidating'",
                       (now, str(username), "every branch was closed", bundle["id"]))
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    bundle = get(db, bundle["id"], now)
    return {"bundle": bundle, "tell": sorted(set(bundle["owners"]) - {str(username)})}


def delete_branch(db, *, user_id, username, repository, branch, reason, work, known, now=None) -> dict:
    """Ask the cleanup account to delete one branch that is in no bundle, with a reason: at its next run it removes the
    branch as it removes a branch closed out of a bundle (``branch_gate.carry_closures``), its unlanded work archived or
    discarded as ``work`` says. Its reviews in play are withdrawn with the reason. A bundled branch is closed out of its
    bundle instead (``close`` with remove). The caller is the branch's owner, or an operator or admin. Answers the
    deletion and who is to be told (the owner, unless the owner asked)."""
    branch_review._account(user_id)
    reason = _reason(reason)
    if work not in CLOSE_WORK:
        raise ReviewError("a branch to be deleted says what happens to its unlanded work: work is archive or discard")
    if repository not in known:
        raise ReviewError("repository must be one the policy names: " + ", ".join(sorted(known)), 404)
    now = time.time() if now is None else now
    row = next((item for item in checked_out(db, now)
                if item["repository"] == repository and item["branch"] == branch), None)
    if row is None:
        raise ReviewError(f"{branch} is not checked out or registered in {repository}", 404)
    if row["bundle"] is not None:
        raise ReviewError(f"{branch} is in bundle {row['bundle'][:8]}: close it out of the bundle, to be removed", 409)
    if row["removing"]:
        raise ReviewError(f"{branch} is already to be removed by the cleanup account", 409)
    owner = row["owner"]
    if str(username) != owner and not branch_review.may_moderate(db, user_id):
        raise ReviewError("the branch's owner, or an operator or admin, deletes a branch", 403)
    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute("INSERT INTO branch_deletions (repository, branch, requested_at, requested_by, reason, work) "
                   "VALUES (?,?,?,?,?,?)", (repository, branch, now, str(username), reason, work))
        db.execute("UPDATE branch_reviews SET status = 'withdrawn', reviewed_at = ?, note = ? WHERE repository = ? "
                   "AND branch = ? AND status IN ('open', 'approved') AND landed_at IS NULL",
                   (now, f"deleted by {username}: {reason}", repository, branch))
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    deletion = {"repository": repository, "branch": branch, "owner": owner, "requested_at": now,
                "requested_by": str(username), "reason": reason, "work": work}
    return {"deletion": deletion, "tell": [owner] if owner and owner != str(username) else []}


def removals_pending(db) -> list:
    """The branches the cleanup account is still to remove, oldest first: those closed out of a bundle for removal, and
    those asked to be deleted outside any bundle (``bundle_id`` None)."""
    branch_review.ensure(db)
    keys = ("bundle_id", "repository", "branch", "closed_at", "closed_by", "reason", "work")
    return [dict(zip(keys, row)) for row in db.execute(
        "SELECT bundle_id, repository, branch, closed_at, closed_by, reason, work FROM review_bundle_closures "
        "WHERE remove = 1 AND carried_at IS NULL UNION ALL SELECT NULL, repository, branch, requested_at, requested_by, "
        "reason, work FROM branch_deletions WHERE carried_at IS NULL ORDER BY 4")]


def mark_carried(db, bundle_id, branch, carried, now=None, *, repository=None) -> None:
    """The cleanup account's receipt for one removed branch: what it removed and kept. A deletion outside any bundle
    (``bundle_id`` None) is named by its ``repository``."""
    at, carried = time.time() if now is None else now, str(carried)[:500]
    if bundle_id is None:
        db.execute("UPDATE branch_deletions SET carried_at = ?, carried = ? WHERE repository = ? AND branch = ? "
                   "AND carried_at IS NULL", (at, carried, repository, branch))
        return
    db.execute("UPDATE review_bundle_closures SET carried_at = ?, carried = ? WHERE bundle_id = ? AND branch = ? "
               "AND carried_at IS NULL", (at, carried, bundle_id, branch))


def _paths(paths) -> list:
    if not isinstance(paths, list) or not MIN_PATHS <= len(paths) <= MAX_PATHS:
        raise ReviewError(f"paths must be a list of {MIN_PATHS} to {MAX_PATHS} separate paths")
    out = []
    for path in paths:
        if not isinstance(path, dict) or set(path) - {"title", "detail"}:
            raise ReviewError("each path is an object with a title and a detail")
        title, detail = path.get("title"), path.get("detail", "")
        if not isinstance(title, str) or not title.strip() or len(title) > MAX_TITLE:
            raise ReviewError(f"each path needs a title of 1 to {MAX_TITLE} characters")
        if not isinstance(detail, str) or len(detail) > MAX_DETAIL:
            raise ReviewError(f"a path's detail is text of at most {MAX_DETAIL} characters")
        out.append({"title": title.strip(), "detail": detail.strip()})
    if len({path["title"].casefold() for path in out}) != len(out):
        raise ReviewError("each path needs its own title")
    return out


def redelegate(db, *, kind, source_id, paths, note, user_id, username, open_ticket, now=None) -> dict:
    """An operator or admin turns a review or a bundle into one issue ticket that names its separate paths. Answers the
    redelegation, its ticket and who is to be told (the source's author or owners, but the caller)."""
    _moderator(db, user_id, "redelegates work")
    if kind not in ("review", "bundle"):
        raise ReviewError("kind must be review or bundle")
    paths = _paths(paths)
    note = branch_review._note(note or "")
    if open_ticket is None:
        raise ReviewError("this server cannot open tickets", 503)
    now = time.time() if now is None else now
    if kind == "review":
        record = branch_review.get(db, source_id, now)
        if record["standing"] in ("landed", "withdrawn", "rejected", "redelegated"):
            raise ReviewError(f"review {record['id'][:8]} is {record['standing']}; only a review in play is redelegated", 409)
        repository, source = record["repository"], record["id"]
        subject = f"review {record['id'][:8]} of {record['repository']} {record['branch']} ({record['head'][:12]})"
        owners = {record["owner_name"]}
    else:
        bundle = _consolidating(db, source_id, now)
        repository, source = bundle["repository"], bundle["id"]
        subject = (f"bundle {bundle['id'][:8]} of {bundle['repository']}: "
                   + ", ".join(member["branch"] for member in bundle["members"]))
        owners = set(bundle["owners"])
    description = "\n".join([
        f"Redelegated by {username} from {subject}.",
        *([f"Why: {note}"] if note else []),
        "",
        "What had been one review is to be taken as these separate paths, each its own branch and its own review:",
        *(f"{index}. {path['title']}" + (f"\n   {path['detail']}" if path["detail"] else "")
          for index, path in enumerate(paths, 1)),
        "",
        "The source stands 'redelegated' in the reviews and does not land.",
    ])
    redelegation_id = uuid.uuid4().hex
    ticket = open_ticket(f"Redelegate {subject}"[:200], description,
                         {"redelegation": redelegation_id, "source_kind": kind, "source_id": source,
                          "repository": repository, "paths": paths})
    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute("INSERT INTO review_redelegations VALUES (?,?,?,?,?,?,?,?,?)",
                   (redelegation_id, kind, source, repository, now, str(username), note,
                    json.dumps(paths, ensure_ascii=True), str(ticket)))
        if kind == "bundle":
            db.execute("UPDATE review_bundles SET status = 'redelegated', settled_at = ?, settled_by = ?, settle_note = ? "
                       "WHERE id = ? AND status = 'consolidating'",
                       (now, str(username), f"redelegated into ticket {ticket}", source))
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    made = redelegation(db, redelegation_id)
    return {"redelegation": made, "tell": sorted({name for name in owners if name} - {str(username)})}


_REDELEGATION_COLUMNS = ("id", "source_kind", "source_id", "repository", "created_at", "created_by", "note", "paths",
                         "ticket_id")


def redelegation(db, redelegation_id) -> dict:
    branch_review.ensure(db)
    row = db.execute(f"SELECT {', '.join(_REDELEGATION_COLUMNS)} FROM review_redelegations WHERE id = ?",
                     (str(redelegation_id),)).fetchone()
    if row is None:
        raise ReviewError("redelegation not found", 404)
    made = dict(zip(_REDELEGATION_COLUMNS, row))
    made["paths"] = json.loads(made["paths"])
    return made


def redelegations(db, *, limit: int = 100) -> list:
    branch_review.ensure(db)
    return [redelegation(db, row[0]) for row in db.execute(
        "SELECT id FROM review_redelegations ORDER BY created_at DESC LIMIT ?", (int(limit),))]
