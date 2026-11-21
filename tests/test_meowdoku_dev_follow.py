"""[DEV] dev.follow moves only the dev instance's own locked worktrees to what landed, and asks for that one restart."""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import dev_follow, instance_control  # noqa: E402


def _git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout.strip()


def _commit(path, name, text):
    (Path(path) / name).write_text(text, encoding="utf-8")
    _git(path, "add", name)
    _git(path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", name)
    return _git(path, "rev-parse", "HEAD")


def _world(tmp_path):
    """origin (bare), the live checkout on main (stable), a locked detached dev worktree, and an author's clone."""
    origin, live, dev, author = (tmp_path / n for n in ("origin.git", "live", "dev-turing", "author"))
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    subprocess.run(["git", "clone", "-q", str(origin), str(live)], check=True, capture_output=True)
    first = _commit(live, "a.txt", "one\n")
    _git(live, "push", "-q", "origin", "main")
    _git(live, "worktree", "add", "--lock", "--detach", "-q", str(dev), "origin/main")
    subprocess.run(["git", "clone", "-q", str(origin), str(author)], check=True, capture_output=True)
    landed = _commit(author, "b.txt", "two\n")
    _git(author, "push", "-q", "origin", "main")
    instances = tmp_path / "instances.json"
    instances.write_text(json.dumps({"instances": [
        {"name": "m-a", "kind": "meowdoku", "port": 1, "site": "stable"},
        {"name": "m-dev", "kind": "meowdoku", "port": 2, "site": "dev", "code_root": str(dev), "args": []}]}), encoding="utf-8")
    policy = {"remote": "origin", "apply": True, "cleaner": "svc-clean", "repositories": {"turing": {"path": str(live), "default": "main"}}}
    db = sqlite3.connect(tmp_path / "accounts.db", isolation_level=None)
    db.row_factory = sqlite3.Row
    instance_control.ensure(db)
    return {"live": live, "dev": dev, "first": first, "landed": landed, "instances": instances, "policy": policy, "db": db}


def _follow(w, apply=True):
    return dev_follow.follow(instances_path=w["instances"], instance="m-dev", policy=w["policy"], db=w["db"],
                             requester=(9, "svc-clean"), apply=apply, log=lambda line: None)


def test_dev_moves_to_what_landed_and_stable_stays(tmp_path):
    w = _world(tmp_path)
    dry = _follow(w, apply=False)
    assert dry["trees"][0]["action"] == "would move" and _git(w["dev"], "rev-parse", "HEAD") == w["first"]
    assert instance_control.pending(w["db"]) == [], "a dry run files no request"
    done = _follow(w)
    assert done["trees"][0]["action"] == "moved" and _git(w["dev"], "rev-parse", "HEAD") == w["landed"]
    assert _git(w["live"], "rev-parse", "HEAD") == w["first"], "the live checkout (stable) is never moved"
    assert done["pages"].startswith("pages not exported") and "restart request" in done["restart"]
    [row] = instance_control.pending(w["db"])
    assert row["instance"] == "m-dev" and row["action"] == "restart" and row["requested_name"] == "svc-clean"
    again = _follow(w)
    assert again["trees"][0]["action"] == "unchanged" and "restart" not in again
    assert len(instance_control.pending(w["db"])) == 1, "nothing moved, so no second restart"


def test_a_dirty_or_diverged_dev_tree_is_held(tmp_path):
    w = _world(tmp_path)
    (w["dev"] / "a.txt").write_text("edited\n", encoding="utf-8")
    assert _follow(w)["trees"][0]["action"] == "held"
    _git(w["dev"], "checkout", "-q", "--", "a.txt")
    _commit(w["dev"], "c.txt", "local\n")                      # a commit made in the dev tree diverges from origin/main
    held = _follow(w)["trees"][0]
    assert held["action"] == "held" and "diverged" in held["reason"]
    assert instance_control.pending(w["db"]) == []


def test_the_repositorys_own_checkout_is_never_moved(tmp_path):
    w = _world(tmp_path)
    data = json.loads(w["instances"].read_text(encoding="utf-8"))
    data["instances"][1]["code_root"] = str(w["live"])
    w["instances"].write_text(json.dumps(data), encoding="utf-8")
    held = _follow(w)["trees"][0]
    assert held["action"] == "held" and "linked worktree" in held["reason"]
    assert _git(w["live"], "rev-parse", "HEAD") == w["first"]


def test_an_instance_without_its_own_tree_is_refused(tmp_path):
    w = _world(tmp_path)
    try:
        dev_follow.follow(instances_path=w["instances"], instance="m-a", policy=w["policy"], db=w["db"],
                          requester=(9, "svc-clean"), apply=True, log=lambda line: None)
    except dev_follow.GateError as error:
        assert "never moves" in str(error)
    else:
        raise AssertionError("an instance on the live checkout must be refused")
