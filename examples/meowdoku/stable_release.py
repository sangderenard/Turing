"""[STABLE] Stable is a nightly copy that an admin signs off (Stage 2 of Research case 8bbe0de7).

Stable runs from its own LOCKED, detached worktrees (instances.json ``code_root``/``speaktome_root`` on the stable
instance, made like the dev trees). The flow uses the existing pieces:

1. ``stable.candidate`` (nightly cron): for each of the stable instance's trees, the dev instance's tree's commit
   (what dev has been running) becomes a candidate when it differs from what stable runs. Candidates are rows in
   ``stable_candidates``.
2. **Sign-off:** an admin signs a candidate off: ``POST /api/stable/signoff {id}`` (account_manage, HTTPS). Nothing else
   marks it signed, and silence never does.
3. ``stable.follow`` (cron): moves each stable tree to its newest signed-off commit, with the same checks as
   dev.follow (``dev_follow.follow_tree``: linked worktree, no tracked change, a fast-forward only). It then asks for a
   restart of the stable instance with a countdown (``COUNTDOWN_S``), which the supervisor announces to every admin
   and any of them can interrupt.

The commit stable ran before is kept on the sign-off row (``replaced``), so going back is a sign-off of that commit.
"""
from __future__ import annotations

import time
from pathlib import Path

from meowdoku import dev_follow, instance_control
from meowdoku.branch_gate import GateError, git

COUNTDOWN_S = 300
DDL = ("CREATE TABLE IF NOT EXISTS stable_candidates (id INTEGER PRIMARY KEY AUTOINCREMENT, repository TEXT NOT NULL, "
       "commit_id TEXT NOT NULL, proposed_at REAL NOT NULL, summary TEXT, signed_by INTEGER, signed_name TEXT, signed_at REAL, "
       "applied_at REAL, replaced TEXT, UNIQUE (repository, commit_id))")


class StableError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


def ensure(db) -> None:
    db.execute(DDL)


def _trees(spec) -> dict:
    return {repo: spec[key] for key, repo in dev_follow.TREES if spec.get(key)}


def propose(db, *, instances_path, stable, dev, now=None) -> list:
    """One candidate per repository whose dev tree runs a commit stable does not run yet (and not proposed before)."""
    ensure(db)
    now = time.time() if now is None else now
    stable_trees = _trees(dev_follow.instance_spec(instances_path, stable))
    dev_trees = _trees(dev_follow.instance_spec(instances_path, dev))
    if not stable_trees:
        raise GateError(f"{stable} declares no code_root or speaktome_root: stable still runs the live checkout")
    made = []
    for repo, path in stable_trees.items():
        if repo not in dev_trees:
            continue
        running, offered = git(path, "rev-parse", "HEAD"), git(dev_trees[repo], "rev-parse", "HEAD")
        if not offered or offered == running:
            continue
        log = git(dev_trees[repo], "log", "--format=%h %s", f"{running}..{offered}", check=False).splitlines()
        summary = f"{len(log)} commit(s) since stable's {running[:12]}:\n" + "\n".join(log[:60])
        cur = db.execute("INSERT OR IGNORE INTO stable_candidates (repository, commit_id, proposed_at, summary) VALUES (?,?,?,?)",
                         (repo, offered, now, summary[:20000]))
        if cur.rowcount:
            made.append({"id": cur.lastrowid, "repository": repo, "commit": offered, "commits": len(log)})
    return made


def listing(db, limit=20) -> list:
    ensure(db)
    keys = ("id", "repository", "commit", "proposed_at", "summary", "signed_name", "signed_at", "applied_at", "replaced")
    rows = db.execute("SELECT id, repository, commit_id, proposed_at, summary, signed_name, signed_at, applied_at, replaced "
                      "FROM stable_candidates ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    return [dict(zip(keys, r)) for r in rows]


def sign_off(db, candidate_id, *, user_id, username, now=None) -> dict:
    """The admin's sign-off (the caller has checked account_manage and HTTPS)."""
    ensure(db)
    if type(candidate_id) is not int:
        raise StableError("give the candidate's id")
    row = db.execute("SELECT signed_at FROM stable_candidates WHERE id = ?", (candidate_id,)).fetchone()
    if row is None:
        raise StableError("no such candidate", 404)
    if row[0] is not None:
        raise StableError("already signed off", 409)
    db.execute("UPDATE stable_candidates SET signed_by = ?, signed_name = ?, signed_at = ? WHERE id = ?",
               (int(user_id), username, time.time() if now is None else now, candidate_id))
    return next(c for c in listing(db, 200) if c["id"] == candidate_id)


def follow(db, *, instances_path, stable, policy, requester, apply, now=None, log=print) -> dict:
    """Move each stable tree to its newest signed-off commit, then ask for the counted-down stable restart."""
    ensure(db)
    now = time.time() if now is None else now
    trees = _trees(dev_follow.instance_spec(instances_path, stable))
    if not trees:
        raise GateError(f"{stable} declares no code_root or speaktome_root: stable still runs the live checkout")
    out = {"instance": stable, "trees": []}
    for repo, path in trees.items():
        row = db.execute("SELECT id, commit_id FROM stable_candidates WHERE repository = ? AND signed_at IS NOT NULL "
                         "ORDER BY signed_at DESC, id DESC LIMIT 1", (repo,)).fetchone()
        if row is None:
            out["trees"].append({"repository": repo, "action": "unchanged", "reason": "nothing signed off yet"})
            continue
        entry = dict(dev_follow.follow_tree(path, remote=policy["remote"], default=None, apply=apply, commit=row[1]),
                     repository=repo)
        if entry["action"] == "moved":
            db.execute("UPDATE stable_candidates SET applied_at = ?, replaced = ? WHERE id = ?", (now, entry["head"], row[0]))
        out["trees"].append(entry)
    moved = [t for t in out["trees"] if t["action"] == "moved"]
    if moved:
        try:
            made = instance_control.request(db, instance=stable, action="restart", user_id=requester[0], username=requester[1],
                                            reason=("stable moves to signed-off " + ", ".join(
                                                f"{t['repository']} {t['to'][:12]}" for t in moved))[:instance_control.MAX_REASON],
                                            delay_s=COUNTDOWN_S, known=[s.get("name") for s in dev_follow.instance_specs(instances_path)])
            out["restart"] = f"restart request {made['id']} in {COUNTDOWN_S} s; any admin may interrupt it"
        except instance_control.ControlError as error:
            out["restart"] = f"restart not filed: {error}"
    for t in out["trees"]:
        if t["action"] != "unchanged":
            log(f"[stable] {stable} {t['repository']}: {t['action']} ({t['reason']})")
    return out
