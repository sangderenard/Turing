"""[DEV] Keep the dev instance on what landed (Stage 1.3 of Research case 8bbe0de7).

The dev instance declares its trees in instances.json: ``"code_root"`` (turing) and ``"speaktome_root"``. Each is a
LOCKED, detached worktree of the live repository::

    git -C C:/dev/Powershell/turing worktree add --lock --reason "dev deploy tree" --detach <dev tree> origin/main

The lock keeps it from the branch cleaner, and being detached means it never holds a branch or a commit.
``follow`` does the following:

1. It fetches, then fast-forwards each tree to ``<remote>/<default>`` (the policy's default for that repository).
2. It holds a tree that has tracked changes, one whose commit is not behind the target, and anything that is not a
   linked worktree. The repository's own checkout, which stable runs, is never moved.
3. When a tree moved, it exports the dev pages from the dev turing tree into the instance's ``--web-dir``.
4. It asks for a restart of that one instance through ``instance_requests``, which waits for the instance's running
   jobs.

Nothing here touches another instance or another tree.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from meowdoku import instance_control
from meowdoku.branch_gate import GateError, git

TREES = (("code_root", "turing"), ("speaktome_root", "speaktome"))   # instance key -> repository in the branch policy


def instance_specs(instances_path) -> list:
    data = json.loads(Path(instances_path).read_text(encoding="utf-8"))
    return list(data["instances"] if isinstance(data, dict) else data)


def instance_spec(instances_path, name) -> dict:
    for spec in instance_specs(instances_path):
        if spec.get("name") == name:
            return spec
    raise GateError(f"no instance named {name!r} in {instances_path}")


def _linked_worktree(path) -> bool:
    """True for a worktree added with ``git worktree add``; False for a repository's own checkout."""
    inside = git(path, "rev-parse", "--is-inside-work-tree", check=False)
    if inside != "true":
        return False
    own, common = git(path, "rev-parse", "--absolute-git-dir"), git(path, "rev-parse", "--path-format=absolute", "--git-common-dir")
    return Path(own).resolve() != Path(common).resolve()


def follow_tree(path, *, remote, default, apply, commit=None) -> dict:
    """Fast-forward one linked worktree to ``<remote>/<default>``, or to ``commit`` when given (stable_release.py)."""
    entry = {"tree": str(path), "target": commit[:12] if commit else f"{remote}/{default}"}
    if not Path(path).is_dir():
        return dict(entry, action="held", reason="the tree does not exist")
    if not _linked_worktree(path):
        return dict(entry, action="held", reason="not a linked worktree; a repository's own checkout is never moved")
    fetched = subprocess.run(["git", "-C", str(path), "fetch", "--quiet", remote], capture_output=True, text=True)
    target = git(path, "rev-parse", "-q", "--verify", (commit + "^{commit}") if commit else f"refs/remotes/{remote}/{default}",
                 check=False)
    head = git(path, "rev-parse", "HEAD")
    entry.update(head=head, to=target or None)
    if fetched.returncode != 0:
        entry["fetch"] = (fetched.stderr or fetched.stdout).strip()[-200:]
    if not target:
        return dict(entry, action="held", reason=f"{entry['target']} is unknown here")
    if head == target:
        return dict(entry, action="unchanged", reason="already at what landed")
    if git(path, "status", "--porcelain", "--untracked-files=no"):
        return dict(entry, action="held", reason="tracked changes in the dev tree; nothing is ever edited there")
    if subprocess.run(["git", "-C", str(path), "merge-base", "--is-ancestor", head, target], capture_output=True).returncode != 0:
        return dict(entry, action="held", reason=f"its commit is not behind {entry['target']} (diverged)")
    if not apply:
        return dict(entry, action="would move", reason=f"{head[:12]} -> {target[:12]}")
    git(path, "checkout", "--detach", "--quiet", target)
    return dict(entry, action="moved", reason=f"{head[:12]} -> {target[:12]}")


def _arg(spec, flag):
    args = list(spec.get("args") or [])
    return args[args.index(flag) + 1] if flag in args[:-1] else None


def export_pages(code_root, web_dir, *, python=sys.executable, timeout=1800) -> str:
    script = Path(code_root) / "examples" / "meowdoku" / "web_export.py"
    if not script.is_file():
        return f"pages not exported: {script} is missing"
    done = subprocess.run([python, str(script), "--skip-boards", "--out", str(web_dir)], cwd=str(code_root),
                          capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0:
        return f"pages not exported: web_export exited {done.returncode}: {(done.stderr or done.stdout).strip()[-200:]}"
    return f"pages exported into {web_dir}"


def follow(*, instances_path, instance, policy, db, requester, apply, python=sys.executable, log=print) -> dict:
    """``requester`` is the (user_id, username) the restart request is filed as (the policy's cleanup account)."""
    spec = instance_spec(instances_path, instance)
    trees = []
    for key, repository in TREES:
        if spec.get(key):
            declared = policy["repositories"].get(repository)
            if declared is None:
                trees.append({"tree": spec[key], "action": "held", "reason": f"the policy names no {repository}"})
                continue
            trees.append(dict(follow_tree(spec[key], remote=policy["remote"], default=declared["default"], apply=apply),
                              repository=repository))
    if not trees:
        raise GateError(f"{instance} declares no code_root or speaktome_root: it runs the live checkout, which dev.follow never moves")
    out = {"instance": instance, "trees": trees}
    moved = [t for t in trees if t["action"] == "moved"]
    if moved:
        web_dir = _arg(spec, "--web-dir")
        if any(t["repository"] == "turing" for t in moved) and spec.get("code_root"):
            out["pages"] = export_pages(spec["code_root"], web_dir, python=python) if web_dir else \
                "pages not exported: the instance declares no --web-dir"
        user_id, username = requester
        reason = "dev follows what landed: " + "; ".join(f"{t['repository']} {t['reason']}" for t in moved)
        try:
            known = [s.get("name") for s in instance_specs(instances_path)]
            row = instance_control.request(db, instance=instance, action="restart", user_id=user_id, username=username,
                                           reason=reason[:instance_control.MAX_REASON], known=known)
            out["restart"] = f"restart request {row['id']} filed"
        except instance_control.ControlError as error:
            out["restart"] = f"restart not filed: {error}"
    for t in trees:
        if t["action"] != "unchanged":
            log(f"[dev] {instance} {t.get('repository', '?')}: {t['action']} ({t['reason']})")
    return out
