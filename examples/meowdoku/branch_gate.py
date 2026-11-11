"""The Git side of reviewed changes: land what the reviews accept, clean merged branches after a grace period.

``branch_review.py`` holds the rows and the rule; this module is the only place that runs Git for them. It runs as
whoever holds the Git credential (an operator, or a worker started by one), never inside the server.

    python branch_gate.py --policy deploy/branch_policy.json --db <accounts.db> status
    ... submit <repository> <branch> --as <username> [--timeout-hours N] [--keep]
    ... land  [--apply]        fast-forward each accepted change onto its default branch, then remove or keep it
    ... clean [--apply]        remove branches that have been fully merged for the grace period
    ... check-push <repository>    the pre-push hook: refuse a push to a default branch that no review accepts
    ... install-hook / remove-hook

Without ``--apply`` (and with the policy's ``apply`` false) ``land`` and ``clean`` only say what they would do.

What ``clean`` removes: a branch whose every commit is already in the live source, that is not protected by the
policy, not checked out in any worktree (a checked-out branch is in use, here and on the remote), and not kept by
its review. The live source is what the repository's own checkout has checked out (the branch the service runs
from), or its default branch on the remote. The deadline is ``grace_days`` after the cleaner first saw the branch
in that state; a branch whose last commit is already older than ``stale_days`` has no deadline left to wait for
and is removed at the first look. The remote copy goes only when its commits are on the remote too (on the default
branch or on the pushed live branch); otherwise it is kept and said so. A branch with a commit of its own is never removed, whatever its age;
it is reported as unmerged. Removal is by expected commit, locally and on the remote, so a branch that moved
between the look and the removal is left alone.

What ``land`` does: only a fast-forward. A change that is not a descendant of its default branch is reported, and
its author merges the default branch into it and submits the new commit. No merge commit is made here and nothing
is ever forced.

Parts are named by the policy file (repository names and paths, protected patterns); nothing is inferred from a
branch's name beyond those declared patterns.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

try:
    from meowdoku import branch_review
except ImportError:  # run as a plain script, or loaded by path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import branch_review  # type: ignore

HOOK_MARK = "# speaktome branch gate"
DEFAULTS = {"grace_days": 14, "stale_days": 30, "review_timeout_hours": 24, "delete_after_landing": True, "delete_remote": True,
            "apply": False, "remote": "origin", "protected": [], "repositories": {}}


class GateError(Exception):
    pass


def git(path, *args, check=True, stdin=None) -> str:
    done = subprocess.run(["git", "-C", str(path), *args], capture_output=True, input=stdin, text=True,
                          encoding="utf-8", errors="replace")
    if check and done.returncode != 0:
        raise GateError(f"git {' '.join(args[:3])} failed in {path}: {(done.stderr or done.stdout).strip()[-300:]}")
    return done.stdout.strip() if done.returncode == 0 else ""


def load_policy(path) -> dict:
    with open(path, encoding="utf-8") as handle:
        given = json.load(handle)
    unknown = set(given) - set(DEFAULTS) - {"about"}
    if unknown:
        raise GateError("the policy has keys this tool does not know: " + ", ".join(sorted(unknown)))
    policy = {**DEFAULTS, **{k: v for k, v in given.items() if k != "about"}}
    for name, repository in policy["repositories"].items():
        if not isinstance(repository, dict) or not {"path", "default"} <= set(repository):
            raise GateError(f"repository {name} needs a path and a default branch")
    return policy


def protected(policy, repository, branch) -> bool:
    declared = policy["repositories"][repository]
    patterns = list(policy["protected"]) + list(declared.get("protected", [])) + [declared["default"]]
    return any(fnmatch.fnmatchcase(branch, pattern) for pattern in patterns)


def branches(path, default, remote="origin") -> list:
    """Every local branch: its commit and that commit's time, where it is checked out, and where all of it already is:
    ``merged`` on the remote default branch, ``live`` in what this checkout has checked out, ``on_remote`` on the
    remote default branch or on the remote copy of the checked-out branch."""
    trees, at = {}, None
    for line in git(path, "worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            at = line[9:]
        elif line.startswith("branch refs/heads/"):
            trees[line[18:]] = at
    target = f"refs/remotes/{remote}/{default}"
    working = git(path, "symbolic-ref", "-q", "--short", "HEAD", check=False)
    pushed = f"refs/remotes/{remote}/{working}" if working else None
    if pushed and not git(path, "rev-parse", "-q", "--verify", pushed, check=False):
        pushed = None

    def within(commit, ref) -> bool:
        return subprocess.run(["git", "-C", str(path), "merge-base", "--is-ancestor", commit, ref],
                              capture_output=True).returncode == 0

    out = []
    for line in git(path, "for-each-ref", "refs/heads",
                    "--format=%(refname:short)\t%(objectname)\t%(committerdate:unix)").splitlines():
        name, head, stamp = line.split("\t")
        merged = within(head, target)
        out.append({"branch": name, "head": head, "merged": merged, "worktree": trees.get(name),
                    "tip_time": float(stamp), "live": within(head, "HEAD"),
                    "on_remote": merged or bool(pushed and within(head, pushed)),
                    "remote_head": git(path, "rev-parse", "-q", "--verify", f"refs/remotes/{remote}/{name}", check=False) or None})
    return out


def remove_branch(path, branch, head, *, remote, delete_remote, local) -> str:
    """Remove one branch at exactly ``head``; each side is left alone if it has moved.

    ``local`` is what is known of the local branch: "free" (at ``head``, checked out nowhere), "checked out", or
    "absent" (no local branch at ``head``).
    """
    done = []
    remote_head = git(path, "rev-parse", "-q", "--verify", f"refs/remotes/{remote}/{branch}", check=False)
    if delete_remote and remote_head:
        if remote_head != head:
            done.append("remote copy kept: it is at a different commit")
        else:
            git(path, "push", f"--force-with-lease=refs/heads/{branch}:{head}", remote, f":refs/heads/{branch}")
            done.append("removed on the remote")
    if local == "checked out":
        done.append("local branch kept: it is checked out")
    elif local == "absent":
        done.append("no local branch at that commit")
    else:
        git(path, "update-ref", "-d", f"refs/heads/{branch}", head)
        git(path, "config", "--remove-section", f"branch.{branch}", check=False)
        done.append("removed locally")
    return "; ".join(done)


def clean(policy, db, *, apply, now=None, log=print) -> list:
    now = time.time() if now is None else now
    grace, stale = float(policy["grace_days"]) * 86400.0, float(policy["stale_days"]) * 86400.0
    actions = []
    for name, declared in sorted(policy["repositories"].items()):
        path = declared["path"]
        if not Path(path, ".git").exists():
            actions.append({"repository": name, "branch": None, "action": "skip", "reason": "no checkout at its path"})
            continue
        git(path, "fetch", "--quiet", policy["remote"], check=False)      # a stale view only delays a removal
        for row in branches(path, declared["default"], policy["remote"]):
            entry = {"repository": name, "branch": row["branch"], "head": row["head"]}
            if protected(policy, name, row["branch"]):
                entry.update(action="keep", reason="protected by the policy")
            elif not (row["merged"] or row["live"]):
                branch_review.forget_merged(db, name, row["branch"])
                entry.update(action="keep", reason="has commits that are not in the live source")
            elif branch_review.kept(db, name, row["branch"]):
                entry.update(action="keep", reason="kept by its review")
            elif row["worktree"] is not None:
                entry.update(action="keep", reason="checked out in a worktree, so in use")
            else:
                since = branch_review.merged_since(db, name, row["branch"], row["head"], now)
                age = now - row["tip_time"]
                left = 0.0 if age >= stale else since + grace - now
                why = (f"in the live source and its last commit is {age / 86400.0:.0f} days old" if age >= stale
                       else "in the live source for the whole grace period")
                if left > 0:
                    entry.update(action="wait", reason=f"in the live source; {left / 86400.0:.1f} days of grace left")
                elif not apply:
                    entry.update(action="would remove", reason=why if row["on_remote"] else why + "; the remote copy would be kept")
                else:
                    try:
                        detail = remove_branch(path, row["branch"], row["head"], remote=policy["remote"],
                                               delete_remote=policy["delete_remote"] and row["on_remote"], local="free")
                        if not row["on_remote"] and row["remote_head"]:
                            detail += "; remote copy kept: the live branch holding it is not on the remote"
                        branch_review.forget_merged(db, name, row["branch"])
                        entry.update(action="removed", reason=detail)
                    except GateError as error:
                        entry.update(action="failed", reason=str(error))
            actions.append(entry)
    for entry in actions:
        if entry["action"] not in ("keep", "wait"):
            log(f"[branches] {entry['repository']} {entry['branch']}: {entry['action']} ({entry['reason']})")
    return actions


def land(policy, db, *, apply, now=None, log=print) -> list:
    now = time.time() if now is None else now
    results = []
    for review in branch_review.landable(db, now):
        entry = {"review": review["id"], "repository": review["repository"], "branch": review["branch"], "head": review["head"]}
        declared = policy["repositories"].get(review["repository"])
        try:
            if declared is None:
                raise GateError("the policy no longer names this repository")
            path, remote = declared["path"], policy["remote"]
            if review["base"] != declared["default"]:
                raise GateError("the review's base is not the repository's default branch")
            git(path, "fetch", "--quiet", remote)
            if git(path, "rev-parse", "-q", "--verify", f"refs/remotes/{remote}/{review['branch']}", check=False) != review["head"]:
                raise GateError("the branch on the remote is not at the reviewed commit; submit the commit it is at")
            base = f"refs/remotes/{remote}/{review['base']}"
            if subprocess.run(["git", "-C", str(path), "merge-base", "--is-ancestor", base, review["head"]],
                              capture_output=True).returncode != 0:
                raise GateError(f"not a fast-forward of {review['base']}; merge {review['base']} into the branch and submit again")
            if not apply:
                entry.update(action="would land", reason=f"{review['standing']}; a fast-forward of {review['base']}")
            else:
                git(path, "push", remote, f"{review['head']}:refs/heads/{review['base']}")
                detail = f"{review['base']} is at {review['head'][:12]}"
                branch_review.mark_landed(db, review["id"], detail, now)
                if not review["delete_branch"]:
                    detail += "; branch kept as its author asked"
                else:
                    row = {r["branch"]: r for r in branches(path, review["base"], remote)}.get(review["branch"])
                    state = ("absent" if row is None or row["head"] != review["head"]
                             else "checked out" if row["worktree"] is not None else "free")
                    try:
                        detail += "; " + remove_branch(path, review["branch"], review["head"], remote=remote,
                                                       delete_remote=policy["delete_remote"], local=state)
                    except GateError as error:
                        detail += f"; the branch was not removed ({error})"
                db.execute("UPDATE branch_reviews SET landed_detail = ? WHERE id = ?", (detail[:500], review["id"]))
                entry.update(action="landed", reason=detail)
        except GateError as error:
            entry.update(action="held", reason=str(error))
        log(f"[reviews] {entry['repository']} {entry['branch']}: {entry['action']} ({entry['reason']})")
        results.append(entry)
    return results


def check_push(policy, db, repository, lines, now=None) -> list:
    """The refusals for one push (empty when it may go): a default branch moves only to a commit a review accepts."""
    declared = policy["repositories"][repository]
    target = f"refs/heads/{declared['default']}"
    refused = []
    for line in lines:
        parts = line.split()
        if len(parts) != 4 or parts[2] != target:
            continue
        sha = parts[1]
        if set(sha) == {"0"}:
            refused.append(f"{declared['default']} is never deleted")
        elif not branch_review.accepted_head(db, repository, declared["default"], sha, now):
            refused.append(f"{declared['default']} moves only to a reviewed commit, and {sha[:12]} has no accepted review. "
                           "Push your branch and submit it for review instead.")
    return refused


def repository_of(policy, path) -> str | None:
    common = Path(git(path, "rev-parse", "--path-format=absolute", "--git-common-dir")).resolve()
    for name, declared in policy["repositories"].items():
        if (Path(declared["path"]) / ".git").resolve() == common:
            return name
    return None


def hook_text(policy_path, db_path) -> str:
    here = Path(__file__).resolve().as_posix()
    return (f"#!/bin/sh\n{HOOK_MARK}: a default branch moves only to a commit a review accepts.\n"
            f"exec \"{Path(sys.executable).as_posix()}\" \"{here}\" --policy \"{Path(policy_path).resolve().as_posix()}\" "
            f"--db \"{Path(db_path).resolve().as_posix()}\" check-push\n")


def install_hooks(policy, policy_path, db_path, *, remove=False, log=print) -> list:
    out = []
    for name, declared in sorted(policy["repositories"].items()):
        hook = Path(declared["path"], ".git", "hooks", "pre-push")
        if not hook.parent.is_dir():
            out.append((name, "no checkout"))
            continue
        current = hook.read_text(encoding="utf-8", errors="replace") if hook.exists() else None
        if current is not None and HOOK_MARK not in current:
            out.append((name, "left alone: another pre-push hook is installed there"))
        elif remove:
            if current is not None:
                hook.unlink()
            out.append((name, "removed" if current is not None else "none installed"))
        else:
            hook.write_text(hook_text(policy_path, db_path), encoding="utf-8", newline="\n")
            out.append((name, "installed"))
    for name, what in out:
        log(f"[hook] {name}: {what}")
    return out


def _user(db, username):
    row = db.execute("SELECT id, username FROM users WHERE username = ? COLLATE NOCASE", (username,)).fetchone()
    if row is None:
        raise GateError("no such account")
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", required=True)
    parser.add_argument("--db", required=True, help="the accounts database")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    for name in ("land", "clean"):
        sub.add_parser(name).add_argument("--apply", action="store_true")
    submit = sub.add_parser("submit")
    submit.add_argument("repository")
    submit.add_argument("branch")
    submit.add_argument("--as", dest="username", required=True)
    submit.add_argument("--timeout-hours", type=float)
    submit.add_argument("--keep", action="store_true", help="keep the branch after it lands")
    sub.add_parser("check-push").add_argument("repository", nargs="?")
    sub.add_parser("install-hook")
    sub.add_parser("remove-hook")
    args = parser.parse_args(argv)
    try:
        policy = load_policy(args.policy)
        if args.command in ("install-hook", "remove-hook"):
            install_hooks(policy, args.policy, args.db, remove=args.command == "remove-hook")
            return 0
        db = sqlite3.connect(args.db, timeout=10, isolation_level=None)
        try:
            if args.command == "check-push":
                name = args.repository or repository_of(policy, os.getcwd())
                refused = check_push(policy, db, name, sys.stdin.read().splitlines()) if name else []
                for line in refused:
                    print(f"refused: {line}", file=sys.stderr)
                return 1 if refused else 0
            if args.command == "submit":
                declared = policy["repositories"].get(args.repository)
                if declared is None:
                    raise GateError("the policy does not name that repository")
                head = git(declared["path"], "rev-parse", "--verify", f"refs/remotes/{policy['remote']}/{args.branch}^{{commit}}")
                user = _user(db, args.username)
                hours = policy["review_timeout_hours"] if args.timeout_hours is None else args.timeout_hours
                made = branch_review.submit(db, user_id=user[0], username=user[1], repository=args.repository, branch=args.branch,
                                            base=declared["default"], head=head, timeout_s=hours * 3600.0,
                                            delete_branch=policy["delete_after_landing"] and not args.keep,
                                            known=list(policy["repositories"]))
                print(json.dumps({k: made[k] for k in ("id", "repository", "branch", "head", "standing", "accept_after")}))
                return 0
            if args.command == "status":
                for review in branch_review.listing(db):
                    print(f"{review['standing']:10} {review['repository']:18} {review['branch']} {review['head'][:12]} by {review['owner_name']}")
                return 0
            apply = bool(args.apply or policy["apply"])
            done = (land if args.command == "land" else clean)(policy, db, apply=apply)
            print(json.dumps({"applied": apply, "count": len(done),
                              "by_action": {a: sum(1 for e in done if e["action"] == a) for a in sorted({e["action"] for e in done})}}))
            return 0
        finally:
            db.close()
    except (GateError, branch_review.ReviewError, OSError, ValueError) as error:
        print(f"branch_gate: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
