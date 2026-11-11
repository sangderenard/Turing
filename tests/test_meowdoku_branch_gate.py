"""Reviewed changes and branch cleaning, against real temporary Git repositories and a real SQLite database."""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from meowdoku import branch_gate, branch_review  # noqa: E402

DAY = 86400.0
ADMIN, ANN, BOB = 1, 2, 3


def run(path, *args):
    done = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def commit(work, name, text="x"):
    (work / name).write_text(text, encoding="utf-8")
    run(work, "add", name)
    run(work, "commit", "-q", "-m", f"add {name}")
    return run(work, "rev-parse", "HEAD")


@pytest.fixture
def world(tmp_path):
    remote, work = tmp_path / "remote.git", tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(work)], check=True)
    for key, value in (("user.name", "t"), ("user.email", "t@example.invalid"), ("commit.gpgsign", "false")):
        run(work, "config", key, value)
    run(work, "remote", "add", "origin", str(remote))
    commit(work, "base.txt")
    run(work, "push", "-q", "origin", "main")
    db = sqlite3.connect(tmp_path / "accounts.db", isolation_level=None)
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL)")
    db.execute("CREATE TABLE account_roles (user_id INTEGER PRIMARY KEY, role TEXT NOT NULL)")
    db.executemany("INSERT INTO users VALUES (?,?)", [(ADMIN, "admin"), (ANN, "ann"), (BOB, "bob")])
    db.execute("INSERT INTO account_roles VALUES (?, 'admin')", (ADMIN,))
    policy = {**branch_gate.DEFAULTS, "grace_days": 14, "protected": ["snapshot/*"],
              "repositories": {"work": {"path": str(work), "default": "main"}}}
    policy_file = tmp_path / "policy.json"
    policy_file.write_text(json.dumps({k: v for k, v in policy.items()}), encoding="utf-8")
    return {"remote": remote, "work": work, "db": db, "policy": policy, "policy_file": policy_file, "db_file": tmp_path / "accounts.db"}


def branch(world, name, file=None, push=True):
    """A branch with one commit of its own, pushed; the working tree returns to main."""
    work = world["work"]
    run(work, "checkout", "-q", "-b", name, "main")
    head = commit(work, file or name.replace("/", "_") + ".txt")
    if push:
        run(work, "push", "-q", "origin", name)
    run(work, "checkout", "-q", "main")
    return head


def submit(world, name, head, user=ANN, username="ann", timeout_s=DAY, delete_branch=True, now=1000.0):
    return branch_review.submit(world["db"], user_id=user, username=username, repository="work", branch=name, base="main",
                                head=head, timeout_s=timeout_s, delete_branch=delete_branch, known=["work"], now=now)


def remote_has(world, name):
    return subprocess.run(["git", "-C", str(world["remote"]), "rev-parse", "-q", "--verify", f"refs/heads/{name}"],
                          capture_output=True, text=True).stdout.strip() or None


def actions(done):
    return {entry["branch"]: entry["action"] for entry in done}


def test_silence_is_consent_and_an_objection_stops_the_clock(world):
    db = world["db"]
    made = submit(world, "feature", "a" * 40)
    assert made["standing"] == "waiting" and made["accept_after"] == 1000.0 + DAY
    assert branch_review.get(db, made["id"], 1000.0 + DAY)["standing"] == "accepted"
    with pytest.raises(branch_review.ReviewError) as own:
        branch_review.remark(db, made["id"], user_id=ANN, username="ann", verdict="agree")
    assert own.value.status == 403
    branch_review.remark(db, made["id"], user_id=BOB, username="bob", verdict="object", note="the name is wrong")
    assert branch_review.get(db, made["id"], 1000.0 + 9 * DAY)["standing"] == "objected"
    assert branch_review.landable(db, 1000.0 + 9 * DAY) == []
    branch_review.remark(db, made["id"], user_id=BOB, username="bob", verdict="agree")
    assert branch_review.get(db, made["id"], 1000.0 + 9 * DAY)["standing"] == "accepted"


def test_an_admin_decides_exactly_the_commit_they_read(world):
    db = world["db"]
    made = submit(world, "feature", "a" * 40)
    with pytest.raises(branch_review.ReviewError) as not_admin:
        branch_review.decide(db, made["id"], user_id=BOB, username="bob", head="a" * 40, decision="approved")
    assert not_admin.value.status == 403
    with pytest.raises(branch_review.ReviewError) as stale:
        branch_review.decide(db, made["id"], user_id=ADMIN, username="admin", head="b" * 40, decision="approved")
    assert stale.value.status == 409
    done = branch_review.decide(db, made["id"], user_id=ADMIN, username="admin", head="a" * 40, decision="rejected", note="no")
    assert done["standing"] == "rejected" and done["reviewer_name"] == "admin"
    assert not branch_review.accepted_head(db, "work", "main", "a" * 40, 1000.0 + 9 * DAY)
    with pytest.raises(branch_review.ReviewError):
        branch_review.decide(db, made["id"], user_id=ADMIN, username="admin", head="a" * 40, decision="approved")


def test_a_moved_branch_is_a_new_change_and_identity_is_never_text(world):
    db = world["db"]
    first = submit(world, "feature", "a" * 40)
    again = submit(world, "feature", "a" * 40, user=BOB, username="bob")
    assert again["id"] == first["id"] and again["owner_name"] == "ann"
    second = submit(world, "feature", "b" * 40)
    assert branch_review.get(db, first["id"])["standing"] == "withdrawn" and second["standing"] == "waiting"
    for bad in ({"user": "1"}, {"user": True}, {"head": "abc"}, {"name": "main"}, {"name": "a..b"}, {"timeout_s": -1}):
        with pytest.raises(branch_review.ReviewError):
            submit(world, bad.get("name", "other"), bad.get("head", "c" * 40), user=bad.get("user", ANN),
                   timeout_s=bad.get("timeout_s", DAY))
    with pytest.raises(branch_review.ReviewError):
        branch_review.submit(db, user_id=ANN, username="ann", repository="elsewhere", branch="x", base="main", head="c" * 40,
                             timeout_s=0, delete_branch=True, known=["work"])


def test_a_merged_branch_is_removed_only_after_the_grace_period(world):
    work, db, policy = world["work"], world["db"], world["policy"]
    merged = branch(world, "done-work")
    run(work, "merge", "-q", "--ff-only", "done-work")
    run(work, "push", "-q", "origin", "main")
    unmerged = branch(world, "in-progress")
    run(work, "branch", "snapshot/keep", "main")
    run(work, "worktree", "add", "-q", "-b", "elsewhere", str(work.parent / "tree"), "main")
    first = actions(branch_gate.clean(policy, db, apply=True, now=5000.0, log=lambda line: None))
    assert first == {"main": "keep", "done-work": "wait", "in-progress": "keep", "snapshot/keep": "keep", "elsewhere": "keep"}
    assert remote_has(world, "done-work") == merged, "nothing is removed the first time it is looked at"
    dry = actions(branch_gate.clean(policy, db, apply=False, now=5000.0 + 15 * DAY, log=lambda line: None))
    assert dry["done-work"] == "would remove" and remote_has(world, "done-work") == merged
    done = branch_gate.clean(policy, db, apply=True, now=5000.0 + 15 * DAY, log=lambda line: None)
    assert actions(done)["done-work"] == "removed" and actions(done)["in-progress"] == "keep"
    assert remote_has(world, "done-work") is None and remote_has(world, "in-progress") == unmerged
    names = run(work, "for-each-ref", "refs/heads", "--format=%(refname:short)").split()
    assert "done-work" not in names and {"in-progress", "snapshot/keep", "elsewhere", "main"} <= set(names)
    assert "checked out" in next(e for e in done if e["branch"] == "elsewhere")["reason"]
    assert remote_has(world, "elsewhere") is None, "it was never pushed, and a checked-out branch is left alone"


def test_an_old_branch_already_in_the_live_source_has_no_deadline_left(world):
    import time
    work, db, policy = world["work"], world["db"], world["policy"]
    merged = branch(world, "long-done")
    run(work, "merge", "-q", "--ff-only", "long-done")
    run(work, "push", "-q", "origin", "main")
    local_only = branch(world, "in-live-only")
    run(work, "merge", "-q", "--ff-only", "in-live-only")          # in what is checked out here, not on the remote
    unmerged = branch(world, "old-but-its-own")
    fresh = actions(branch_gate.clean(policy, db, apply=True, log=lambda line: None))
    assert fresh["long-done"] == "wait" and fresh["in-live-only"] == "wait", "a recent commit waits out the grace period"
    later = time.time() + 31 * DAY                                  # the same commits, now older than stale_days
    dry = {e["branch"]: e for e in branch_gate.clean(policy, db, apply=False, now=later, log=lambda line: None)}
    assert dry["long-done"]["action"] == "would remove" and "days old" in dry["long-done"]["reason"]
    assert "remote copy would be kept" in dry["in-live-only"]["reason"]
    done = {e["branch"]: e for e in branch_gate.clean(policy, db, apply=True, now=later, log=lambda line: None)}
    assert done["long-done"]["action"] == "removed" and remote_has(world, "long-done") is None
    assert done["in-live-only"]["action"] == "removed" and "remote copy kept" in done["in-live-only"]["reason"]
    assert remote_has(world, "in-live-only") == local_only, "its commits are not on the remote anywhere else"
    assert done["old-but-its-own"]["action"] == "keep" and remote_has(world, "old-but-its-own") == unmerged
    names = run(work, "for-each-ref", "refs/heads", "--format=%(refname:short)").split()
    assert "long-done" not in names and "in-live-only" not in names and "old-but-its-own" in names
    assert merged != local_only


def test_a_branch_that_moved_after_it_was_seen_merged_starts_its_grace_again(world):
    work, db, policy = world["work"], world["db"], world["policy"]
    branch(world, "done-work")
    run(work, "merge", "-q", "--ff-only", "done-work")
    run(work, "push", "-q", "origin", "main")
    branch_gate.clean(policy, db, apply=True, now=5000.0, log=lambda line: None)
    run(work, "checkout", "-q", "done-work")
    commit(work, "more.txt")
    run(work, "checkout", "-q", "main")
    later = actions(branch_gate.clean(policy, db, apply=True, now=5000.0 + 15 * DAY, log=lambda line: None))
    assert later["done-work"] == "keep", "it has a commit of its own again"


def test_an_accepted_change_lands_as_a_fast_forward_and_its_branch_goes(world):
    work, db, policy = world["work"], world["db"], world["policy"]
    head = branch(world, "feature")
    made = submit(world, "feature", head)
    assert branch_gate.land(policy, db, apply=True, now=1000.0 + 60, log=lambda line: None) == [], "still waiting"
    dry = branch_gate.land(policy, db, apply=False, now=1000.0 + DAY, log=lambda line: None)
    assert dry[0]["action"] == "would land" and remote_has(world, "main") != head
    done = branch_gate.land(policy, db, apply=True, now=1000.0 + DAY, log=lambda line: None)
    assert done[0]["action"] == "landed" and remote_has(world, "main") == head
    assert remote_has(world, "feature") is None
    assert "feature" not in run(work, "for-each-ref", "refs/heads", "--format=%(refname:short)").split()
    record = branch_review.get(db, made["id"])
    assert record["standing"] == "landed" and "removed on the remote" in record["landed_detail"]
    assert branch_gate.land(policy, db, apply=True, now=1000.0 + 2 * DAY, log=lambda line: None) == []


def test_a_kept_branch_stays_and_is_never_cleaned(world):
    db, policy = world["db"], world["policy"]
    head = branch(world, "feature")
    made = submit(world, "feature", head, delete_branch=False, timeout_s=0)
    done = branch_gate.land(policy, db, apply=True, now=1000.0, log=lambda line: None)
    assert done[0]["action"] == "landed" and remote_has(world, "feature") == head
    assert "kept" in branch_review.get(db, made["id"])["landed_detail"]
    for now in (2000.0, 2000.0 + 99 * DAY):
        assert actions(branch_gate.clean(policy, db, apply=True, now=now, log=lambda line: None))["feature"] == "keep"
    assert remote_has(world, "feature") == head


def test_a_change_that_is_not_a_fast_forward_or_has_moved_is_held(world):
    work, db, policy = world["work"], world["db"], world["policy"]
    head = branch(world, "feature")
    commit(work, "meanwhile.txt")
    run(work, "push", "-q", "origin", "main")
    submit(world, "feature", head, timeout_s=0)
    held = branch_gate.land(policy, db, apply=True, now=1000.0, log=lambda line: None)
    assert held[0]["action"] == "held" and "not a fast-forward" in held[0]["reason"]
    assert remote_has(world, "feature") == head and remote_has(world, "main") != head
    other = branch(world, "second", push=True)
    submit(world, "second", "f" * 40, timeout_s=0)
    held = {e["branch"]: e for e in branch_gate.land(policy, db, apply=True, now=1000.0, log=lambda line: None)}
    assert held["second"]["action"] == "held" and "not at the reviewed commit" in held["second"]["reason"]
    assert remote_has(world, "second") == other


def test_the_push_check_lets_a_default_branch_move_only_to_a_reviewed_commit(world):
    db, policy = world["db"], world["policy"]
    head = branch(world, "feature")
    line = f"refs/heads/main {head} refs/heads/main {'1' * 40}"
    assert branch_gate.check_push(policy, db, "work", [line], 1000.0)
    assert branch_gate.check_push(policy, db, "work", [f"refs/heads/feature {head} refs/heads/feature {'0' * 40}"], 1000.0) == []
    assert branch_gate.check_push(policy, db, "work", [f"(delete) {'0' * 40} refs/heads/main {head}"], 1000.0)
    made = submit(world, "feature", head)
    assert branch_gate.check_push(policy, db, "work", [line], 1000.0 + 60), "still waiting"
    branch_review.decide(db, made["id"], user_id=ADMIN, username="admin", head=head, decision="approved")
    assert branch_gate.check_push(policy, db, "work", [line], 1000.0 + 60) == []


def test_the_installed_hook_refuses_a_real_push_until_the_review_accepts(world):
    work, db, policy = world["work"], world["db"], world["policy"]
    assert branch_gate.install_hooks(policy, world["policy_file"], world["db_file"], log=lambda line: None) == [("work", "installed")]
    assert branch_gate.repository_of(policy, work) == "work"
    head = commit(work, "direct.txt")
    refused = subprocess.run(["git", "-C", str(work), "push", "origin", "main"], capture_output=True, text=True)
    assert refused.returncode != 0 and "has no accepted review" in refused.stderr
    assert remote_has(world, "main") != head
    run(work, "push", "-q", "origin", "main:feature")
    made = submit(world, "feature", head)
    branch_review.decide(db, made["id"], user_id=ADMIN, username="admin", head=head, decision="approved")
    done = branch_gate.land(policy, db, apply=True, log=lambda line: None)
    assert done[0]["action"] == "landed" and remote_has(world, "main") == head
    hook = work / ".git" / "hooks" / "pre-push"
    hook.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    assert branch_gate.install_hooks(policy, world["policy_file"], world["db_file"], log=lambda line: None)[0][1].startswith("left alone")
    hook.write_text(branch_gate.hook_text(world["policy_file"], world["db_file"]), encoding="utf-8")
    assert branch_gate.install_hooks(policy, world["policy_file"], world["db_file"], remove=True, log=lambda line: None) == [("work", "removed")]
    assert not hook.exists()


def test_command_line_reports_without_applying(world, capsys):
    head = branch(world, "feature")
    base = ["--policy", str(world["policy_file"]), "--db", str(world["db_file"])]
    assert branch_gate.main(base + ["submit", "work", "feature", "--as", "ann", "--timeout-hours", "0"]) == 0
    assert json.loads(capsys.readouterr().out)["head"] == head
    assert branch_gate.main(base + ["land"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {"applied": False, "count": 1, "by_action": {"would land": 1}}
    assert remote_has(world, "feature") == head
    assert branch_gate.main(base + ["submit", "work", "feature", "--as", "nobody"]) == 2
    assert branch_gate.main(base + ["status"]) == 0
    assert "accepted" in capsys.readouterr().out


class Handler:
    """The parts of the server's request handler the reviews route uses."""

    def __init__(self, path, user=None, username=None, capabilities=("dev_ui",), body=None):
        from types import SimpleNamespace
        self.path, self._body, self.sent = path, body, None
        self._who = SimpleNamespace(user_id=user, username=username)
        self._capabilities = set(capabilities)
        self.headers = {"Content-Type": "application/json"}

    def _principal(self):
        return self._who

    def _caps(self):
        return self._capabilities

    def _body_json(self, limit):
        return self._body

    def _json(self, document, status=200):
        self.sent = (int(status), json.loads(json.dumps(document)))


def test_reviews_over_rest_take_identity_from_the_session_only(world):
    policy, db_file = world["policy"], world["db_file"]

    def ask(method, path, **kw):
        handler = Handler(path, **kw)
        branch_review.http_route(handler, path.split("?")[0], method, policy=policy, accounts_db=db_file)
        return handler.sent

    head = "a" * 40
    standing = f"/api/reviews/standing?repository=work&head={head}"
    assert ask("GET", standing) == (200, {"repository": "work", "head": head, "accepted": False}), "open: no account"
    assert ask("GET", "/api/reviews/standing?repository=work&head=abc")[0] == 400
    assert ask("GET", "/api/reviews")[0] == 401
    assert ask("GET", "/api/reviews", user=ANN, username="ann", capabilities=())[0] == 403
    status, made = ask("POST", "/api/reviews", user=ANN, username="ann",
                       body={"repository": "work", "branch": "feature", "head": head, "user_id": ADMIN, "username": "admin"})
    review = made["review"]
    assert status == 200 and review["owner_name"] == "ann" and review["standing"] == "waiting" and review["delete_branch"] is True
    assert ask("POST", "/api/reviews/remark", user=ANN, username="ann", body={"id": review["id"], "verdict": "agree"})[0] == 403
    assert ask("POST", "/api/reviews/decide", user=BOB, username="bob",
               body={"id": review["id"], "head": head, "decision": "approved"})[0] == 403
    assert ask("GET", standing)[1]["accepted"] is False
    status, done = ask("POST", "/api/reviews/decide", user=ADMIN, username="admin",
                       body={"id": review["id"], "head": head, "decision": "approved", "note": "read it"})
    assert status == 200 and done["review"]["standing"] == "approved"
    assert ask("GET", standing)[1]["accepted"] is True
    listed = ask("GET", "/api/reviews", user=BOB, username="bob")[1]
    assert listed["repositories"] == {"work": "main"} and listed["reviews"][0]["id"] == review["id"]
    assert ask("POST", "/api/reviews", user=ANN, username="ann", body={"repository": "nowhere", "branch": "x", "head": head})[0] == 404
    assert ask("POST", "/api/reviews/nothing", user=ANN, username="ann", body={})[0] == 404


def test_the_task_worker_offers_land_and_clean_and_reports_while_apply_is_off(world):
    from types import SimpleNamespace
    from meowdoku import scheduled_tasks
    head = branch(world, "feature")
    submit(world, "feature", head, timeout_s=0)
    cron_host = SimpleNamespace(DONE=2, UNCHANGED=3)
    table = scheduled_tasks.handlers(cron_host, branches={"policy": world["policy_file"], "accounts_db": world["db_file"],
                                                         "module": branch_gate})
    assert set(table) == {"reviews.land", "branches.clean"} <= set(scheduled_tasks.PREFIX_TASKS["cron"]) | set(table)
    status, said, detail = table["reviews.land"]({})
    assert status == 3 and said == "1 would land" and detail[0]["branch"] == "feature" and remote_has(world, "feature") == head
    status, said, detail = table["branches.clean"]({})
    assert status == 3 and detail == []


def test_the_shipped_policy_names_real_repositories_and_reports_only():
    shipped = branch_gate.load_policy(Path(__file__).resolve().parents[1] / "examples" / "meowdoku" / "deploy" / "branch_policy.json")
    assert shipped["apply"] is False and {"turing", "speaktome", "nodus"} <= set(shipped["repositories"])
    assert branch_gate.protected(shipped, "turing", "main") and branch_gate.protected(shipped, "turing", "snapshot/x")
    assert not branch_gate.protected(shipped, "turing", "codex/anything")
