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
branch or on the pushed live branch); otherwise it is kept and said so.

A branch with commits of its own is removed by age alone, and the age depends on whose it is: one registered to an
account (``branch_review.register``, or by submitting it) lives ``registered_days`` after its last commit, one
registered to nobody ``unregistered_days``. A branch under review is never removed. Every removal is written to
``branch_removed`` with the commit the branch was at.

Before a branch is removed on the remote, the tool asks GitHub which rules apply to it (``gh api
repos/<owner>/<name>/rules/branches/<branch>``). Where a rule forbids deletion the remote copy is left, the local
branch is still removed, and the entry carries ``remote_skipped`` with the rule; the summary counts those. A remote
that is not GitHub, or a question GitHub does not answer, is not a refusal: the removal is tried and any refusal is
reported as a failure.

``clean_worktrees`` keeps side worktrees off this computer. A repository's own checkout is the reference core; every
other worktree of it is removed once it has been idle for ``worktree_idle_days`` (counted from when its HEAD last
really moved, read from the reflog entry, never from file times), holds no uncommitted or untracked
file, is not locked, and its commit is on the remote. Its local branch goes with it when the remote holds the same
branch at the same commit. Nothing is lost: the notice of removal (``branch_removed``) names the path, the branch and
the commit, and the remote keeps the branch. A worktree that fails any of those tests is kept and the reason said.
A worktree that holds uncommitted work is never judged by a rule: it waits for a reviewer's verdict
(``branch_review.set_verdict``; the ``verdict`` command): trash discards that work with the tree, archive commits and
pushes it first, keep leaves it. The notice records the reviewer and what was discarded or where it was archived.
Ignored files in a removed worktree (caches, build output) are deleted with it.

One account removes from the core, and no other: the policy's ``cleaner`` (a service account, shown to people under
``cleaner_display_name``). ``clean`` and ``clean_worktrees`` remove nothing unless they are told to act as that account
and it exists in SQL as a service account; every notice of removal is signed with it. Looking, without ``apply``, is
open to anyone. ``cleaner`` (the command) makes the account through the supervisor's own ``ensure_service_user``: a
random password that is stored nowhere, so nobody signs in as it. Until the core's files are closed to every other
account by the operating system this is a declared rule, not a wall.

``publish_notices`` keeps the record of removals in Git as well as in SQL: it writes every notice to the policy's
``notices_file`` (one line each: when, repository, branch, commit, by whom, why) and, when asked, commits and pushes
that file in the repository that holds it. The task worker does this after any run that removed something, so
GitHub's history of that file is the history of what was removed.

``improper_saves`` is the cleaner's other job: it reads each repository's own checkout and reports what was saved there
outside a commit (changed and untracked files, and the large untracked ones that look like temporary binaries or
data). It removes nothing.

``lock`` is "mock" or "enforce". In mock the push check says what it would have refused, prints the advice
(``branch_review.advice``) and lets the push through. A branch with a commit of its own is never removed, whatever its age;
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
import csv
import fnmatch
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

try:
    from meowdoku import branch_review
except ImportError:  # run as a plain script, or loaded by path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import branch_review  # type: ignore

HOOK_MARK = "# speaktome branch gate"
DEFAULTS = {"grace_days": 14, "stale_days": 30, "unregistered_days": 30, "registered_days": 365, "lock": "mock",
            "worktree_idle_days": 1, "cleaner": "svc-pale-rider", "cleaner_display_name": "Pale Rider",
            "notices_file": "C:/dev/Powershell/REMOVAL_NOTICES.csv",
            "review_timeout_hours": 24, "delete_after_landing": True, "delete_remote": True,
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
    # A key this version does not know is carried in "unknown" and otherwise ignored: a running server must keep
    # reading a policy file written for a newer version (refusing it took the review routes down once).
    unknown = sorted(set(given) - set(DEFAULTS) - {"about"})
    policy = {**DEFAULTS, **{k: v for k, v in given.items() if k in DEFAULTS}, "unknown": unknown}
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


def github_slug(path, remote="origin") -> str | None:
    """``owner/name`` when the remote is a GitHub repository, else None."""
    match = re.search(r"github\.com[:/]([A-Za-z0-9._-]+/[A-Za-z0-9._-]+?)(?:\.git)?/?$", git(path, "remote", "get-url", remote, check=False))
    return match[1] if match else None


def github_rules(slug, branch) -> list | None:
    """The rule types GitHub applies to one branch (with their ruleset ids), or None when it cannot be asked."""
    try:
        done = subprocess.run(["gh", "api", f"repos/{slug}/rules/branches/{urllib.parse.quote(branch, safe='')}"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        rules = json.loads(done.stdout) if done.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return rules if isinstance(rules, list) else None


def run_gh(*args) -> tuple[int, str]:
    """One ``gh`` call: (exit code, output). The only place this tool asks GitHub for anything but rules."""
    try:
        done = subprocess.run(["gh", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        return 1, str(error)
    return done.returncode, (done.stdout if done.returncode == 0 else done.stderr or done.stdout).strip()


def pull_request_body(review, authors, authorship="") -> str:
    """What GitHub keeps about one landed review: who submitted it, who accepted it, at which commit, by whom written."""
    decided = (f"approved by {review['reviewer_name']}" if review.get("reviewer_name")
               else "accepted after its time-out with no objection")
    lines = [f"Review `{review['id']}` of `{review['repository']}`, submitted by **{review['owner_name']}**, {decided}.",
             f"Commit `{review['head']}` lands on `{review['base']}` as a fast-forward; GitHub marks this pull request merged",
             "when that push arrives. Nothing here is rebased or squashed.", "", "Commit authors:"]
    lines += [f"- {author}" for author in authors] or ["- (none recorded)"]
    if authorship:
        lines.append(f"- {authorship}")
    for remark in review.get("remarks") or ():
        lines.append(f"- remark by {remark['username']}: {remark['verdict']}" + (f" ({remark['note']})" if remark.get("note") else ""))
    if review.get("note"):
        lines += ["", f"Decision note: {review['note']}"]
    return "\n".join(lines)


def record_pull_request(slug, review, authors, *, authorship="", gh=run_gh) -> str:
    """Open (or update) the pull request that records one review on GitHub, before its commit lands. Returns its URL,
    or a sentence saying why there is none; never raises, so a GitHub that cannot answer does not hold a landing."""
    body = pull_request_body(review, authors, authorship)
    code, out = gh("pr", "list", "--repo", slug, "--head", review["branch"], "--state", "open", "--json", "number,url")
    if code != 0:
        return f"no pull request (GitHub did not answer: {out[-200:]})"
    try:
        found = json.loads(out or "[]")
    except ValueError:
        found = []
    if found:
        number, url = found[0]["number"], found[0]["url"]
        code, out = gh("pr", "edit", str(number), "--repo", slug, "--body", body)
        return url if code == 0 else f"pull request {url} not updated ({out[-200:]})"
    code, out = gh("pr", "create", "--repo", slug, "--base", review["base"], "--head", review["branch"],
                   "--title", f"{review['branch']} (review {review['id'][:8]})", "--body", body)
    return out.splitlines()[-1] if code == 0 and out else f"no pull request ({out[-200:]})"


def deletion_forbidden(path, remote, branch) -> str | None:
    """The reason GitHub would refuse to delete this branch, or None when no rule says so (or nothing could be asked)."""
    slug = github_slug(path, remote)
    for rule in (github_rules(slug, branch) or []) if slug else []:
        if isinstance(rule, dict) and rule.get("type") == "deletion":
            return f"a GitHub rule of {slug} forbids deleting it (ruleset {rule.get('ruleset_id')})"
    return None


def remove_branch(path, branch, head, *, remote, delete_remote, local, skipped=None) -> str:
    """Remove one branch at exactly ``head``; each side is left alone if it has moved.

    ``local`` is what is known of the local branch: "free" (at ``head``, checked out nowhere), "checked out", or
    "absent" (no local branch at ``head``). ``skipped`` is a list that receives the reason when a rule of the remote
    forbids the removal there.
    """
    done = []
    remote_head = git(path, "rev-parse", "-q", "--verify", f"refs/remotes/{remote}/{branch}", check=False)
    if delete_remote and remote_head:
        forbidden = deletion_forbidden(path, remote, branch) if remote_head == head else None
        if remote_head != head:
            done.append("remote copy kept: it is at a different commit")
        elif forbidden:
            done.append(f"remote copy kept: {forbidden}")
            if skipped is not None:
                skipped.append(forbidden)
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


def side_worktrees(path, remote="origin") -> list:
    """Every worktree of a repository except its own checkout, with what removing it would lose."""
    entries, current = [], None
    for line in git(path, "worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            current = {"path": line[9:], "branch": None, "head": None, "locked": False}
            entries.append(current)
        elif line.startswith("HEAD "):
            current["head"] = line[5:]
        elif line.startswith("branch refs/heads/"):
            current["branch"] = line[18:]
        elif line.startswith("locked"):
            current["locked"] = True
    out = []
    for entry in entries[1:]:
        entry["exists"] = Path(entry["path"]).is_dir()
        entry["on_remote"] = bool(entry["head"]) and bool(git(path, "branch", "-r", "--contains", entry["head"], check=False))
        entry["same_on_remote"] = bool(entry["branch"]) and git(
            path, "rev-parse", "-q", "--verify", f"refs/remotes/{remote}/{entry['branch']}", check=False) == entry["head"]
        entry["dirty"], entry["touched"] = 0, None
        if entry["exists"]:
            # Looking must not count as use: no optional locks, so this status never rewrites the index.
            entry["dirty"] = len(git(entry["path"], "--no-optional-locks", "status", "--porcelain", check=False).splitlines())
            # When HEAD last really moved (a checkout, a commit, a reset), from the reflog entry itself. File times
            # are not evidence: Git's housekeeping rewrites every worktree's reflog and made 49 idle worktrees look
            # used within the day (2026-11-12).
            moved = re.search(r"\{(\d+)\}", git(entry["path"], "log", "-g", "-1", "--date=unix", "--format=%gd", "HEAD", check=False))
            if moved:
                entry["touched"] = float(moved[1])
            else:                                # no reflog: the commit's own time
                stamp = git(entry["path"], "log", "-1", "--format=%ct", "HEAD", check=False)
                entry["touched"] = float(stamp) if stamp.isdigit() else None
        out.append(entry)
    return out


def cleaner(policy, db, actor, apply) -> str | None:
    """The account a removal is signed with. Refuses an ``apply`` by anyone but the policy's cleaner."""
    if not apply:
        return None
    declared = policy["cleaner"]
    if actor != declared:
        raise GateError(f"only {declared} removes from the core; this run acts as {actor or 'nobody'}")
    if not branch_review.is_service(db, declared):
        raise GateError(f"the cleanup account {declared} does not exist as a service account; run the cleaner command first")
    return declared


def publish_notices(policy, db, *, commit, log=print) -> str:
    """Write every notice of removal to the notices file; with ``commit``, commit and push it. Returns a sentence."""
    file = Path(policy["notices_file"])
    notices = sorted(branch_review.removed(db, limit=1000000), key=lambda n: (n["removed_at"], n["repository"], n["branch"]))
    with open(file, "w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["removed_at_utc", "repository", "branch", "head", "removed_by", "reason"])
        for n in notices:
            writer.writerow([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(n["removed_at"])), n["repository"], n["branch"],
                             n["head"], n["removed_by"] or "", n["reason"]])
    said = f"{len(notices)} notices written to {file.name}"
    if not commit:
        return said
    home = file.parent
    if not git(home, "status", "--porcelain", "--", file.name, check=False):
        return said + "; already recorded in Git"
    git(home, "add", "--", file.name)
    git(home, "commit", "-q", "-m", f"Removal notices: {len(notices)} records", "--", file.name)
    branch = git(home, "branch", "--show-current")
    git(home, "push", policy["remote"], branch)
    default = next((d["default"] for d in policy["repositories"].values() if Path(d["path"]).resolve() == home.resolve()), None)
    if default and default != branch:
        git(home, "push", policy["remote"], f"HEAD:refs/heads/{default}")
    log(f"[notices] {said}; committed and pushed")
    return said + "; committed and pushed"


def improper_saves(policy, *, large_bytes=1 << 20) -> list:
    """What each repository's own checkout holds outside a commit. Read-only."""
    out = []
    for name, declared in sorted(policy["repositories"].items()):
        path = Path(declared["path"])
        if not (path / ".git").exists():
            continue
        changed, untracked, large = [], [], []
        status = subprocess.run(["git", "-C", str(path), "status", "--porcelain"], capture_output=True, text=True,
                                encoding="utf-8", errors="replace").stdout          # unstripped: column one may be a space
        for line in status.splitlines():
            file = line[3:].strip('"')
            if line.startswith("??"):
                untracked.append(file)
                target = path / file
                size = target.stat().st_size if target.is_file() else 0
                if size >= large_bytes:
                    large.append({"file": file, "bytes": size})
            else:
                changed.append(file)
        out.append({"repository": name, "branch": git(path, "branch", "--show-current", check=False),
                    "changed": changed, "untracked": untracked, "large_untracked": large})
    return out


def _reviewed_dirty_tree(policy, db, path, name, label, tree, *, apply, signer, now) -> dict:
    """What happens to a side worktree that holds uncommitted work: nothing, unless a reviewer has said so."""
    held = f"holds {tree['dirty']} uncommitted or untracked file(s)"
    verdict = branch_review.verdict_for(db, tree["path"])
    if verdict is None:
        return {"action": "keep", "reason": held + "; waiting for a reviewer's verdict"}
    who = f"reviewed by {verdict['decided_by']}" + (f": {verdict['note']}" if verdict["note"] else "")
    if verdict["verdict"] == "keep":
        return {"action": "keep", "reason": f"{held}; kept, {who}"}
    if not apply:
        return {"action": "would remove", "reason": f"{held}; {verdict['verdict']}, {who}"}
    try:
        detail = f"worktree {tree['path']} removed"
        head = tree["head"]
        if verdict["verdict"] == "archive":
            branch = tree["branch"] or f"archive/{Path(tree['path']).name}-{(tree['head'] or '')[:8]}"
            if not tree["branch"]:
                git(tree["path"], "switch", "-q", "-c", branch)
            git(tree["path"], "add", "-A")
            git(tree["path"], "commit", "-q", "--no-verify", "-m",
                f"Archive of uncommitted work at removal ({verdict['decided_by']})")
            head = git(tree["path"], "rev-parse", "HEAD")
            git(tree["path"], "push", policy["remote"], f"{head}:refs/heads/{branch}")
            detail += f"; its uncommitted work is commit {head[:12]} on {policy['remote']}/{branch}"
        else:
            stat = git(tree["path"], "--no-optional-locks", "diff", "--shortstat", check=False) or "untracked files only"
            detail += f"; abandoned uncommitted work discarded ({stat.strip()})"
        git(path, "worktree", "remove", "--force", tree["path"])
        if tree["branch"] and not protected(policy, name, tree["branch"]) and not branch_review.in_review(db, name, tree["branch"]):
            remote_head = git(path, "rev-parse", "-q", "--verify", f"refs/remotes/{policy['remote']}/{tree['branch']}", check=False)
            if remote_head == head or verdict["verdict"] == "archive":
                git(path, "update-ref", "-d", f"refs/heads/{tree['branch']}", check=False)
                detail += "; local branch removed"
        branch_review.record_removed(db, name, label, head, f"{detail}; {who}", now, keep_owner=True, removed_by=signer)
        return {"action": "removed", "reason": f"{detail}; {who}"}
    except GateError as error:
        return {"action": "failed", "reason": str(error)}


def clean_worktrees(policy, db, *, apply, actor=None, now=None, log=print) -> list:
    signer = cleaner(policy, db, actor, apply)
    now = time.time() if now is None else now
    idle = float(policy["worktree_idle_days"]) * 86400.0
    actions = []
    for name, declared in sorted(policy["repositories"].items()):
        path = declared["path"]
        if not Path(path, ".git").exists():
            continue
        for tree in side_worktrees(path, policy["remote"]):
            label = tree["branch"] or f"(detached at {(tree['head'] or '')[:12]})"
            entry = {"repository": name, "branch": label, "head": tree["head"], "worktree": tree["path"]}
            if not tree["exists"]:
                if apply:
                    git(path, "worktree", "prune", check=False)
                entry.update(action="pruned" if apply else "would prune", reason="its folder is gone; only the record was left")
            elif tree["locked"]:
                entry.update(action="keep", reason="locked")
            elif tree["touched"] is None or now - tree["touched"] < idle:
                entry.update(action="keep", reason=f"used within the last {policy['worktree_idle_days']:g} day(s)")
            elif tree["dirty"]:
                entry.update(_reviewed_dirty_tree(policy, db, path, name, label, tree, apply=apply, signer=signer, now=now))
            elif not tree["on_remote"]:
                entry.update(action="keep", reason="its commit is not on the remote")
            elif not apply:
                entry.update(action="would remove", reason="idle, clean, and its commit is on the remote")
            else:
                try:
                    git(path, "worktree", "remove", tree["path"])
                    detail = f"worktree {tree['path']} removed; the commit is on the remote"
                    drop = (tree["same_on_remote"] and not protected(policy, name, tree["branch"])
                            and not branch_review.in_review(db, name, tree["branch"]))
                    if drop:
                        git(path, "update-ref", "-d", f"refs/heads/{tree['branch']}", tree["head"])
                        git(path, "config", "--remove-section", f"branch.{tree['branch']}", check=False)
                        detail += f" as {policy['remote']}/{tree['branch']}; local branch removed"
                    branch_review.record_removed(db, name, label, tree["head"], detail, now, keep_owner=True, removed_by=signer)
                    entry.update(action="removed", reason=detail)
                except GateError as error:
                    entry.update(action="failed", reason=str(error))
            actions.append(entry)
    for entry in actions:
        if entry["action"] != "keep":
            log(f"[worktrees] {entry['repository']} {entry['branch']}: {entry['action']} ({entry['reason']})")
    return actions


def clean(policy, db, *, apply, actor=None, now=None, log=print) -> list:
    signer = cleaner(policy, db, actor, apply)
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
            contained = row["merged"] or row["live"]
            age = now - row["tip_time"]
            if not contained:
                branch_review.forget_merged(db, name, row["branch"])
            if protected(policy, name, row["branch"]):
                entry.update(action="keep", reason="protected by the policy")
            elif row["worktree"] is not None:
                entry.update(action="keep", reason="checked out in a worktree, so in use")
            elif branch_review.kept(db, name, row["branch"]):
                entry.update(action="keep", reason="kept by its review")
            elif branch_review.in_review(db, name, row["branch"]):
                entry.update(action="keep", reason="under review")
            else:
                if contained:
                    since = branch_review.merged_since(db, name, row["branch"], row["head"], now)
                    left = 0.0 if age >= stale else since + grace - now
                    why = (f"in the live source and its last commit is {age / 86400.0:.0f} days old" if age >= stale
                           else "in the live source for the whole grace period")
                    waiting, remote_too = f"in the live source; {left / 86400.0:.1f} days of grace left", row["on_remote"]
                else:
                    holder = branch_review.owner(db, name, row["branch"])
                    left = float(policy["registered_days" if holder else "unregistered_days"]) * 86400.0 - age
                    whose = f"registered to {holder['username']}" if holder else "registered to nobody"
                    why = f"has commits of its own, is {whose}, and its last commit is {age / 86400.0:.0f} days old"
                    waiting, remote_too = f"has commits of its own; {whose}; {left / 86400.0:.0f} days left", True
                if left > 0:
                    entry.update(action="wait" if contained else "keep", reason=waiting)
                elif not apply:
                    forbidden = (deletion_forbidden(path, policy["remote"], row["branch"])
                                 if policy["delete_remote"] and remote_too and row["remote_head"] == row["head"] else None)
                    entry.update(action="would remove", reason=why if remote_too else why + "; the remote copy would be kept")
                    if forbidden:
                        entry.update(remote_skipped=forbidden, reason=f"{why}; the remote copy would be kept: {forbidden}")
                else:
                    try:
                        skips = []
                        detail = remove_branch(path, row["branch"], row["head"], remote=policy["remote"],
                                               delete_remote=policy["delete_remote"] and remote_too, local="free", skipped=skips)
                        if skips:
                            entry["remote_skipped"] = skips[0]
                        if not remote_too and row["remote_head"]:
                            detail += "; remote copy kept: the live branch holding it is not on the remote"
                        branch_review.forget_merged(db, name, row["branch"])
                        branch_review.record_removed(db, name, row["branch"], row["head"], why, now, removed_by=signer)
                        entry.update(action="removed", reason=f"{why}; {detail}")
                    except GateError as error:
                        entry.update(action="failed", reason=str(error))
            actions.append(entry)
    for entry in actions:
        if entry["action"] not in ("keep", "wait"):
            log(f"[branches] {entry['repository']} {entry['branch']}: {entry['action']} ({entry['reason']})")
    return actions


def land(policy, db, *, apply, now=None, log=print, gh=run_gh, only=None) -> list:
    """Land each accepted change that is a fast-forward of its base. ``only`` (review ids) limits the run to those changes:
    the landing queue's "land now". Every change tried and not landed keeps why (branch_review.mark_tried)."""
    now = time.time() if now is None else now
    wanted = None if only is None else {str(review_id) for review_id in only}
    results = []
    for review in branch_review.landable(db, now):
        if wanted is not None and review["id"] not in wanted:
            continue
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
                if git(path, "rev-parse", "-q", "--verify", f"refs/heads/{review['branch']}", check=False) != review["head"]:
                    raise GateError("the branch is not at the reviewed commit, here or on the remote; submit the commit it is at")
                if apply:                        # the author never pushed: this tool does the talking to GitHub
                    git(path, "push", remote, f"{review['head']}:refs/heads/{review['branch']}")
            base = f"refs/remotes/{remote}/{review['base']}"
            if subprocess.run(["git", "-C", str(path), "merge-base", "--is-ancestor", base, review["head"]],
                              capture_output=True).returncode != 0:
                raise GateError(f"not a fast-forward of {review['base']}; merge {review['base']} into the branch and submit again")
            authors = [line for line in git(path, "log", "--format=%an <%ae>", f"{base}..{review['head']}").splitlines() if line]
            authors = list(dict.fromkeys(authors))
            authorship = branch_review.authorship(db, review, authors)
            if not apply:
                entry.update(action="would land", reason=f"{review['standing']}; a fast-forward of {review['base']}")
            else:
                # The record on GitHub: a pull request naming who submitted and who accepted, opened before the push
                # so that GitHub marks it merged when the fast-forward arrives. A remote that is not GitHub has none.
                slug = github_slug(path, remote)
                pull = record_pull_request(slug, review, authors, authorship=authorship, gh=gh) if slug else None
                git(path, "push", remote, f"{review['head']}:refs/heads/{review['base']}")
                detail = f"{review['base']} is at {review['head'][:12]}"
                if pull:
                    detail += f"; {pull}"
                detail += f"; authors {', '.join(authors) or 'none'}; {authorship}"
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
        if entry["action"] != "landed":
            branch_review.mark_tried(db, review["id"], f"{entry['action']}: {entry['reason']}", now)
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
            refused.append(f"{sha[:12]} has no accepted review")
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
    parser.add_argument("--as", dest="actor", default=None, help="the account this run acts as (removals need the policy's cleaner)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    for name in ("land", "clean", "worktrees"):
        sub.add_parser(name).add_argument("--apply", action="store_true")
    submit = sub.add_parser("submit")
    submit.add_argument("repository")
    submit.add_argument("branch")
    submit.add_argument("--as", dest="username", required=True)
    submit.add_argument("--timeout-hours", type=float)
    submit.add_argument("--keep", action="store_true", help="keep the branch after it lands")
    sub.add_parser("check-push").add_argument("repository", nargs="?")
    sub.add_parser("cleaner", help="make the policy's cleanup account, as the supervisor makes service accounts")
    sub.add_parser("inspect", help="report what the core holds outside a commit")
    decide = sub.add_parser("verdict", help="record a reviewer's verdict on a worktree that holds uncommitted work")
    decide.add_argument("path")
    decide.add_argument("decision", choices=("trash", "archive", "keep"))
    decide.add_argument("--note", default="")
    sub.add_parser("notices", help="write the notices of removal to the notices file").add_argument("--commit", action="store_true")
    sub.add_parser("install-hook")
    sub.add_parser("remove-hook")
    args = parser.parse_args(argv)
    try:
        policy = load_policy(args.policy)
        if args.command in ("install-hook", "remove-hook"):
            install_hooks(policy, args.policy, args.db, remove=args.command == "remove-hook")
            return 0
        if args.command == "inspect":
            print(json.dumps(improper_saves(policy), indent=1))
            return 0
        if args.command == "cleaner":
            import importlib.util
            spec = importlib.util.spec_from_file_location("apache_supervisor", Path(__file__).resolve().parent / "deploy" / "apache_supervisor.py")
            supervisor = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(supervisor)
            uid = supervisor.ensure_service_user(policy["cleaner"], args.db)
            with sqlite3.connect(args.db, timeout=10) as made:
                made.execute("UPDATE users SET display_name = ? WHERE id = ?", (policy["cleaner_display_name"], uid))
            print(json.dumps({"cleaner": policy["cleaner"], "shown_as": policy["cleaner_display_name"], "role": "service"}))
            return 0
        db = sqlite3.connect(args.db, timeout=10, isolation_level=None)
        try:
            if args.command == "verdict":
                if not args.actor:
                    raise GateError("a verdict needs --as <reviewer>")
                _user(db, args.actor)
                print(json.dumps(branch_review.set_verdict(db, args.path, args.decision, decided_by=args.actor, note=args.note)))
                return 0
            if args.command == "notices":
                print(publish_notices(policy, db, commit=args.commit))
                return 0
            if args.command == "check-push":
                name = args.repository or repository_of(policy, os.getcwd())
                refused = check_push(policy, db, name, sys.stdin.read().splitlines()) if name else []
                mock = policy["lock"] != "enforce"
                for line in refused:
                    print(f"{'would be refused' if mock else 'refused'}: {line}", file=sys.stderr)
                if refused:
                    print(branch_review.advice(policy, name), file=sys.stderr)
                return 0 if mock or not refused else 1
            if args.command == "submit":
                declared = policy["repositories"].get(args.repository)
                if declared is None:
                    raise GateError("the policy does not name that repository")
                head = git(declared["path"], "rev-parse", "--verify", f"refs/heads/{args.branch}^{{commit}}")
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
            extra = {} if args.command == "land" else {"actor": args.actor}
            done = {"land": land, "clean": clean, "worktrees": clean_worktrees}[args.command](policy, db, apply=apply, **extra)
            print(json.dumps({"applied": apply, "count": len(done),
                              "remote_skipped": sum(1 for e in done if e.get("remote_skipped")),
                              "by_action": {a: sum(1 for e in done if e["action"] == a) for a in sorted({e["action"] for e in done})}}))
            return 0
        finally:
            db.close()
    except (GateError, branch_review.ReviewError, OSError, ValueError) as error:
        print(f"branch_gate: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
