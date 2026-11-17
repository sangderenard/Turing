"""[STABLE] Stable moves only to a commit an admin signed off, and its restart counts down."""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import dev_follow, instance_control, stable_release as S  # noqa: E402


def _git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout.strip()


def _commit(path, name, text):
    (Path(path) / name).write_text(text, encoding="utf-8")
    _git(path, "add", name)
    _git(path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", name)
    return _git(path, "rev-parse", "HEAD")


@pytest.fixture
def world(tmp_path):
    origin, live, dev, stable, author = (tmp_path / n for n in ("origin.git", "live", "dev", "stable", "author"))
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    subprocess.run(["git", "clone", "-q", str(origin), str(live)], check=True, capture_output=True)
    first = _commit(live, "a.txt", "one\n")
    _git(live, "push", "-q", "origin", "main")
    for tree in (dev, stable):
        _git(live, "worktree", "add", "--lock", "--detach", "-q", str(tree), "origin/main")
    subprocess.run(["git", "clone", "-q", str(origin), str(author)], check=True, capture_output=True)
    landed = _commit(author, "b.txt", "two\n")
    _git(author, "push", "-q", "origin", "main")
    instances = tmp_path / "instances.json"
    instances.write_text(json.dumps({"instances": [
        {"name": "m-a", "kind": "meowdoku", "port": 1, "site": "stable", "code_root": str(stable)},
        {"name": "m-dev", "kind": "meowdoku", "port": 2, "site": "dev", "code_root": str(dev)}]}), encoding="utf-8")
    policy = {"remote": "origin", "apply": True, "cleaner": "svc-clean", "repositories": {"turing": {"path": str(live), "default": "main"}}}
    db = sqlite3.connect(tmp_path / "accounts.db", isolation_level=None)
    db.row_factory = sqlite3.Row
    instance_control.ensure(db)
    dev_follow.follow(instances_path=instances, instance="m-dev", policy=policy, db=db, requester=(9, "svc-clean"),
                      apply=True, log=lambda line: None)
    db.execute("DELETE FROM instance_requests")
    return {"live": live, "dev": dev, "stable": stable, "first": first, "landed": landed, "instances": instances,
            "policy": policy, "db": db}


def _follow(w):
    return S.follow(w["db"], instances_path=w["instances"], stable="m-a", policy=w["policy"], requester=(9, "svc-clean"),
                    apply=True, log=lambda line: None)


def test_stable_moves_only_after_an_admin_signs_off(world):
    w = world
    [made] = S.propose(w["db"], instances_path=w["instances"], stable="m-a", dev="m-dev")
    assert made["commit"] == w["landed"] and made["commits"] == 1
    assert S.propose(w["db"], instances_path=w["instances"], stable="m-a", dev="m-dev") == [], "proposed once"
    assert _follow(w)["trees"][0]["reason"] == "nothing signed off yet"
    assert _git(w["stable"], "rev-parse", "HEAD") == w["first"] and instance_control.pending(w["db"]) == []
    signed = S.sign_off(w["db"], made["id"], user_id=1, username="admin")
    assert signed["signed_name"] == "admin"
    with pytest.raises(S.StableError):
        S.sign_off(w["db"], made["id"], user_id=1, username="admin")
    done = _follow(w)
    assert done["trees"][0]["action"] == "moved" and _git(w["stable"], "rev-parse", "HEAD") == w["landed"]
    assert _git(w["live"], "rev-parse", "HEAD") == w["first"], "the live checkout is never moved"
    [row] = instance_control.pending(w["db"])
    assert row["instance"] == "m-a" and row["due_at"] - row["requested_at"] == pytest.approx(S.COUNTDOWN_S)
    [listed] = S.listing(w["db"])
    assert listed["applied_at"] and listed["replaced"] == w["first"], "what stable ran before is kept for going back"
    assert _follow(w)["trees"][0]["action"] == "unchanged"


def test_a_sign_off_never_moves_stable_backwards_or_sideways(world):
    w = world
    [made] = S.propose(w["db"], instances_path=w["instances"], stable="m-a", dev="m-dev")
    _commit(w["stable"], "local.txt", "x\n")                  # stable's tree diverged: a sign-off cannot fast-forward it
    S.sign_off(w["db"], made["id"], user_id=1, username="admin")
    held = _follow(w)["trees"][0]
    assert held["action"] == "held" and "diverged" in held["reason"] and instance_control.pending(w["db"]) == []
