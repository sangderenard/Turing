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
CLEANER = branch_gate.DEFAULTS["cleaner"]


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
    db.execute("INSERT INTO users VALUES (9, ?)", (CLEANER,))
    db.execute("INSERT INTO account_roles VALUES (9, 'service')")
    policy = {**branch_gate.DEFAULTS, "grace_days": 14, "protected": ["snapshot/*"], "lock": "enforce",
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
    first = actions(branch_gate.clean(policy, db, apply=True, actor=CLEANER, now=5000.0, log=lambda line: None))
    assert first == {"main": "keep", "done-work": "wait", "in-progress": "keep", "snapshot/keep": "keep", "elsewhere": "keep"}
    assert remote_has(world, "done-work") == merged, "nothing is removed the first time it is looked at"
    dry = actions(branch_gate.clean(policy, db, apply=False, now=5000.0 + 15 * DAY, log=lambda line: None))
    assert dry["done-work"] == "would remove" and remote_has(world, "done-work") == merged
    done = branch_gate.clean(policy, db, apply=True, actor=CLEANER, now=5000.0 + 15 * DAY, log=lambda line: None)
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
    nobodys = branch(world, "old-and-nobodys")
    reviewed = branch(world, "old-under-review")
    submit(world, "old-under-review", reviewed, timeout_s=30 * DAY, now=time.time())
    held = branch_review.register(db, user_id=ANN, username="ann", repository="work", branch="old-but-its-own", known=["work"])
    assert held["username"] == "ann" and branch_review.branches_of(db, ANN)[0]["branch"] in ("old-but-its-own", "old-under-review")
    with pytest.raises(branch_review.ReviewError) as taken:
        branch_review.register(db, user_id=BOB, username="bob", repository="work", branch="old-but-its-own", known=["work"])
    assert taken.value.status == 409
    fresh = actions(branch_gate.clean(policy, db, apply=True, actor=CLEANER, log=lambda line: None))
    assert fresh["long-done"] == "wait" and fresh["in-live-only"] == "wait", "a recent commit waits out the grace period"
    later = time.time() + 31 * DAY                                  # the same commits, now older than stale_days
    dry = {e["branch"]: e for e in branch_gate.clean(policy, db, apply=False, now=later, log=lambda line: None)}
    assert dry["long-done"]["action"] == "would remove" and "days old" in dry["long-done"]["reason"]
    assert "remote copy would be kept" in dry["in-live-only"]["reason"]
    done = {e["branch"]: e for e in branch_gate.clean(policy, db, apply=True, actor=CLEANER, now=later, log=lambda line: None)}
    assert done["long-done"]["action"] == "removed" and remote_has(world, "long-done") is None
    assert done["in-live-only"]["action"] == "removed" and "remote copy kept" in done["in-live-only"]["reason"]
    assert remote_has(world, "in-live-only") == local_only, "its commits are not on the remote anywhere else"
    assert done["old-but-its-own"]["action"] == "keep" and remote_has(world, "old-but-its-own") == unmerged
    assert "registered to ann" in done["old-but-its-own"]["reason"], "a registered branch lives registered_days"
    assert done["old-and-nobodys"]["action"] == "removed" and "registered to nobody" in done["old-and-nobodys"]["reason"]
    assert remote_has(world, "old-and-nobodys") is None, "a month without an owner and it is gone, here and on the remote"
    assert done["old-under-review"]["action"] == "keep" and done["old-under-review"]["reason"] == "under review"
    names = run(work, "for-each-ref", "refs/heads", "--format=%(refname:short)").split()
    assert "long-done" not in names and "in-live-only" not in names and "old-but-its-own" in names
    assert merged != local_only
    record = db.execute("SELECT head, reason FROM branch_removed WHERE branch = 'old-and-nobodys'").fetchone()
    assert record[0] == nobodys and "registered to nobody" in record[1], "the removal names the commit, so it can be undone"
    year = {e["branch"]: e for e in branch_gate.clean(policy, db, apply=True, actor=CLEANER, now=time.time() + 366 * DAY, log=lambda line: None)}
    assert year["old-but-its-own"]["action"] == "removed" and remote_has(world, "old-but-its-own") is None
    assert branch_review.owner(db, "work", "old-but-its-own") is None


def test_a_branch_that_moved_after_it_was_seen_merged_starts_its_grace_again(world):
    work, db, policy = world["work"], world["db"], world["policy"]
    branch(world, "done-work")
    run(work, "merge", "-q", "--ff-only", "done-work")
    run(work, "push", "-q", "origin", "main")
    branch_gate.clean(policy, db, apply=True, actor=CLEANER, now=5000.0, log=lambda line: None)
    run(work, "checkout", "-q", "done-work")
    commit(work, "more.txt")
    run(work, "checkout", "-q", "main")
    later = actions(branch_gate.clean(policy, db, apply=True, actor=CLEANER, now=5000.0 + 15 * DAY, log=lambda line: None))
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
        assert actions(branch_gate.clean(policy, db, apply=True, actor=CLEANER, now=now, log=lambda line: None))["feature"] == "keep"
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
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {"applied": False, "count": 1, "remote_skipped": 0, "by_action": {"would land": 1}}
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
    status, answer = ask("GET", standing)
    assert status == 200 and answer["accepted"] is False and answer["lock"] == "enforce", "open: no account"
    assert "reviews_checkout repository=work" in answer["advice"] and "speaktome_mcp.api" in answer["advice"]
    assert len(answer["advice"].splitlines()) <= 7, "the lock explains itself in a few lines"
    status, mine = ask("POST", "/api/reviews/checkout", user=ANN, username="ann", body={"repository": "work", "branch": "feature"})
    assert status == 200 and mine["branch"]["username"] == "ann" and mine["base"] == "main" and len(mine["steps"]) == 4
    assert ask("POST", "/api/reviews/checkout", user=BOB, username="bob", body={"repository": "work", "branch": "feature"})[0] == 409
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
    assert ask("GET", standing)[1]["accepted"] is True and ask("GET", standing)[1]["advice"] == ""
    listed = ask("GET", "/api/reviews", user=BOB, username="bob")[1]
    assert listed["repositories"] == {"work": "main"} and listed["reviews"][0]["id"] == review["id"]
    assert listed["your_branches"] == [] and ask("GET", "/api/reviews", user=ANN, username="ann")[1]["your_branches"][0]["branch"] == "feature"
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
    assert set(table) == {"reviews.land", "branches.clean", "core.index"} <= set(scheduled_tasks.PREFIX_TASKS["cron"])
    status, said, detail = table["reviews.land"]({})
    assert status == 3 and said == "1 would land" and detail[0]["branch"] == "feature" and remote_has(world, "feature") == head
    status, said, detail = table["branches.clean"]({})
    assert status == 3 and detail == []


def test_the_mock_lock_says_what_it_would_refuse_and_lets_the_push_through(world, capfd):
    work, policy = world["work"], dict(world["policy"], lock="mock")
    world["policy_file"].write_text(json.dumps(policy), encoding="utf-8")
    branch_gate.install_hooks(policy, world["policy_file"], world["db_file"], log=lambda line: None)
    head = commit(work, "direct.txt")
    pushed = subprocess.run(["git", "-C", str(work), "push", "origin", "main"], capture_output=True, text=True)
    assert pushed.returncode == 0 and remote_has(world, "main") == head
    assert "would be refused" in pushed.stderr and "mock: allowed this time" in pushed.stderr
    assert "reviews_checkout repository=work" in pushed.stderr and "reviews_submit" in pushed.stderr


def test_a_remote_rule_against_deletion_is_asked_first_and_the_skip_is_reported(world, monkeypatch):
    import time
    work, db, policy = world["work"], world["db"], world["policy"]
    assert branch_gate.github_slug(work) is None and branch_gate.deletion_forbidden(work, "origin", "x") is None, "not GitHub: nothing to ask"
    merged = branch(world, "ruled")
    run(work, "merge", "-q", "--ff-only", "ruled")
    run(work, "push", "-q", "origin", "main")
    asked = []

    def rules(slug, name):
        asked.append((slug, name))
        return [{"type": "deletion", "ruleset_id": 7}, {"type": "non_fast_forward", "ruleset_id": 7}] if name == "ruled" else []

    monkeypatch.setattr(branch_gate, "github_slug", lambda path, remote="origin": "owner/name")
    monkeypatch.setattr(branch_gate, "github_rules", rules)
    later = time.time() + 31 * DAY
    dry = {e["branch"]: e for e in branch_gate.clean(policy, db, apply=False, now=later, log=lambda line: None)}
    assert dry["ruled"]["action"] == "would remove" and "ruleset 7" in dry["ruled"]["remote_skipped"]
    done = {e["branch"]: e for e in branch_gate.clean(policy, db, apply=True, actor=CLEANER, now=later, log=lambda line: None)}
    assert done["ruled"]["action"] == "removed" and "forbids deleting it" in done["ruled"]["remote_skipped"]
    assert remote_has(world, "ruled") == merged, "the remote copy is left where the rule forbids removing it"
    assert "ruled" not in run(work, "for-each-ref", "refs/heads", "--format=%(refname:short)").split()
    assert ("owner/name", "ruled") in asked
    monkeypatch.setattr(branch_gate, "github_rules", lambda slug, name: None)
    assert branch_gate.deletion_forbidden(work, "origin", "anything") is None, "no answer is not a refusal"


def test_side_worktrees_leave_this_computer_with_a_notice_when_nothing_would_be_lost(world):
    import time
    work, db, policy = world["work"], world["db"], world["policy"]
    trees = work.parent / "trees"
    heads = {}
    for name in ("pushed", "dirty", "unpushed"):
        run(work, "worktree", "add", "-q", "-b", name, str(trees / name), "main")
        heads[name] = commit(trees / name, f"{name}.txt")
    run(work, "push", "-q", "origin", "pushed", "dirty")
    (trees / "dirty" / "scratch.txt").write_text("not committed", encoding="utf-8")
    run(work, "worktree", "add", "-q", "--detach", str(trees / "detached"), "main")
    today = {e["branch"]: e for e in branch_gate.clean_worktrees(policy, db, apply=True, actor=CLEANER, log=lambda line: None)}
    assert {e["action"] for e in today.values()} == {"keep"} and "used within" in today["pushed"]["reason"]
    later = time.time() + 2 * DAY
    import os
    for name in ("index", "HEAD", "logs/HEAD"):      # Git's housekeeping rewrites these; that is not use
        target = Path(run(trees / "pushed", "rev-parse", "--absolute-git-dir")) / name
        if target.exists():
            os.utime(target, (later, later))
    dry = {e["branch"]: e["action"] for e in branch_gate.clean_worktrees(policy, db, apply=False, now=later, log=lambda line: None)}
    assert dry["pushed"] == "would remove" and (trees / "pushed").is_dir()
    done = {e["branch"]: e for e in branch_gate.clean_worktrees(policy, db, apply=True, actor=CLEANER, now=later, log=lambda line: None)}
    assert done["pushed"]["action"] == "removed" and not (trees / "pushed").exists()
    assert remote_has(world, "pushed") == heads["pushed"], "the remote keeps the branch"
    assert "pushed" not in run(work, "for-each-ref", "refs/heads", "--format=%(refname:short)").split()
    assert done["dirty"]["action"] == "keep" and "uncommitted" in done["dirty"]["reason"] and (trees / "dirty" / "scratch.txt").exists()
    assert done["unpushed"]["action"] == "keep" and "not on the remote" in done["unpushed"]["reason"]
    detached = next(e for name, e in done.items() if name.startswith("(detached"))
    assert detached["action"] == "removed" and not (trees / "detached").exists()
    notice = db.execute("SELECT head, reason FROM branch_removed WHERE branch = 'pushed'").fetchone()
    assert notice[0] == heads["pushed"] and "origin/pushed" in notice[1] and str(trees / "pushed").replace("\\", "/") in notice[1].replace("\\", "/")
    assert (work / "base.txt").exists(), "the repository's own checkout is never touched"


def test_only_the_cleanup_account_removes_from_the_core_and_it_signs_the_notice(world):
    import time
    work, db, policy = world["work"], world["db"], world["policy"]
    merged = branch(world, "finished")
    run(work, "merge", "-q", "--ff-only", "finished")
    run(work, "push", "-q", "origin", "main")
    later = time.time() + 31 * DAY
    for actor in (None, "admin", "ann", "svc-task-worker"):
        with pytest.raises(branch_gate.GateError) as refused:
            branch_gate.clean(policy, db, apply=True, actor=actor, now=later, log=lambda line: None)
        assert CLEANER in str(refused.value)
        with pytest.raises(branch_gate.GateError):
            branch_gate.clean_worktrees(policy, db, apply=True, actor=actor, now=later, log=lambda line: None)
    assert remote_has(world, "finished") == merged, "a refused run removed nothing"
    looked = actions(branch_gate.clean(policy, db, apply=False, actor="ann", now=later, log=lambda line: None))
    assert looked["finished"] == "would remove", "anyone may look"
    db.execute("DELETE FROM account_roles WHERE user_id = 9")
    with pytest.raises(branch_gate.GateError) as missing:
        branch_gate.clean(policy, db, apply=True, actor=CLEANER, now=later, log=lambda line: None)
    assert "does not exist as a service account" in str(missing.value)
    db.execute("INSERT INTO account_roles VALUES (9, 'service')")
    done = actions(branch_gate.clean(policy, db, apply=True, actor=CLEANER, now=later, log=lambda line: None))
    assert done["finished"] == "removed" and remote_has(world, "finished") is None
    notice = branch_review.removed(db)[0]
    assert notice["branch"] == "finished" and notice["head"] == merged and notice["removed_by"] == CLEANER


def test_improper_saves_are_noticed_and_nothing_is_removed(world):
    work, policy = world["work"], world["policy"]
    (work / "base.txt").write_text("edited in place", encoding="utf-8")
    (work / "scratch.bin").write_bytes(b"x" * 2048)
    (work / "note.txt").write_text("left behind", encoding="utf-8")
    found = branch_gate.improper_saves(policy, large_bytes=1024)[0]
    assert found["repository"] == "work" and found["branch"] == "main" and found["changed"] == ["base.txt"]
    assert sorted(found["untracked"]) == ["note.txt", "scratch.bin"]
    assert found["large_untracked"] == [{"file": "scratch.bin", "bytes": 2048}]
    assert (work / "scratch.bin").exists() and (work / "note.txt").exists()


def test_a_policy_written_for_a_newer_version_is_still_read(world):
    newer = json.loads(world["policy_file"].read_text(encoding="utf-8"))
    newer["a_key_from_the_future"] = True
    world["policy_file"].write_text(json.dumps(newer), encoding="utf-8")
    read = branch_gate.load_policy(world["policy_file"])
    assert read["unknown"] == ["a_key_from_the_future"] and read["repositories"] and "a_key_from_the_future" not in read


def test_a_dirty_worktree_waits_for_a_reviewers_verdict(world):
    import time
    work, db, policy = world["work"], world["db"], world["policy"]
    trees = work.parent / "trees"
    heads = {}
    for name in ("abandoned", "valuable", "undecided", "spared"):
        run(work, "worktree", "add", "-q", "-b", name, str(trees / name), "main")
        heads[name] = commit(trees / name, f"{name}.txt")
        run(work, "push", "-q", "origin", name)
        (trees / name / "scratch.txt").write_text(f"uncommitted in {name}", encoding="utf-8")
    later = time.time() + 2 * DAY
    branch_review.set_verdict(db, trees / "abandoned", "trash", decided_by="admin", note="old probe")
    branch_review.set_verdict(db, trees / "valuable", "archive", decided_by="admin")
    branch_review.set_verdict(db, trees / "spared", "keep", decided_by="admin", note="still in use")
    with pytest.raises(branch_review.ReviewError):
        branch_review.set_verdict(db, trees / "undecided", "burn", decided_by="admin")
    dry = {e["branch"]: e for e in branch_gate.clean_worktrees(policy, db, apply=False, now=later, log=lambda line: None)}
    assert dry["abandoned"]["action"] == "would remove" and dry["valuable"]["action"] == "would remove"
    assert dry["undecided"]["action"] == "keep" and "waiting for a reviewer" in dry["undecided"]["reason"]
    assert dry["spared"]["action"] == "keep" and "still in use" in dry["spared"]["reason"]
    done = {e["branch"]: e for e in branch_gate.clean_worktrees(policy, db, apply=True, actor=CLEANER, now=later, log=lambda line: None)}
    assert done["abandoned"]["action"] == "removed" and not (trees / "abandoned").exists()
    assert "discarded" in done["abandoned"]["reason"] and "reviewed by admin: old probe" in done["abandoned"]["reason"]
    assert remote_has(world, "abandoned") == heads["abandoned"], "only the uncommitted work went; the branch is on the remote"
    assert done["valuable"]["action"] == "removed" and not (trees / "valuable").exists()
    archived = remote_has(world, "valuable")
    assert archived != heads["valuable"] and "scratch.txt" in run(world["remote"], "show", "--stat", "--format=", archived)
    assert (trees / "undecided" / "scratch.txt").exists() and (trees / "spared" / "scratch.txt").exists()
    notices = {n["branch"]: n for n in branch_review.removed(db)}
    assert notices["abandoned"]["removed_by"] == CLEANER and "reviewed by admin" in notices["abandoned"]["reason"]
    assert notices["valuable"]["head"] == archived


def test_the_gate_pushes_a_branch_its_author_never_pushed(world):
    db, policy = world["db"], world["policy"]
    head = branch(world, "local-only", push=False)
    assert remote_has(world, "local-only") is None
    made = submit(world, "local-only", head, timeout_s=0, delete_branch=False)
    dry = branch_gate.land(policy, db, apply=False, now=1000.0, log=lambda line: None)
    assert dry[0]["action"] == "would land" and remote_has(world, "local-only") is None
    done = branch_gate.land(policy, db, apply=True, now=1000.0, log=lambda line: None)
    assert done[0]["action"] == "landed" and remote_has(world, "main") == head and remote_has(world, "local-only") == head
    assert branch_review.owner(db, "work", "local-only")["username"] == "ann", "submitting registers the branch to its author"
    assert branch_review.get(db, made["id"])["standing"] == "landed"


def test_the_shipped_policy_names_real_repositories_and_mocks_the_lock():
    shipped = branch_gate.load_policy(Path(__file__).resolve().parents[1] / "examples" / "meowdoku" / "deploy" / "branch_policy.json")
    assert shipped["lock"] == "mock" and {"turing", "speaktome", "nodus"} <= set(shipped["repositories"])
    assert shipped["unregistered_days"] == 30 and shipped["registered_days"] == 365 and shipped["grace_days"] == 1
    assert shipped["worktree_idle_days"] == 1
    assert branch_gate.protected(shipped, "turing", "main") and branch_gate.protected(shipped, "turing", "snapshot/x")
    assert not branch_gate.protected(shipped, "turing", "codex/anything")


def test_the_notices_of_removal_are_kept_in_git_as_well_as_sql(world):
    import csv
    import time
    work, db = world["work"], world["db"]
    policy = dict(world["policy"], notices_file=str(work / "REMOVAL_NOTICES.csv"))
    merged = branch(world, "finished")
    run(work, "merge", "-q", "--ff-only", "finished")
    run(work, "push", "-q", "origin", "main")
    branch_gate.clean(policy, db, apply=True, actor=CLEANER, now=time.time() + 31 * DAY, log=lambda line: None)
    assert branch_gate.publish_notices(policy, db, commit=False).startswith("1 notices written")
    with open(work / "REMOVAL_NOTICES.csv", encoding="utf-8", newline="") as handle:
        lines = list(csv.reader(handle))
    assert lines[0] == ["removed_at_utc", "repository", "branch", "head", "removed_by", "reason"]
    assert lines[1][1:5] == ["work", "finished", merged, CLEANER] and lines[1][0].endswith("Z")
    policy["lock"] = "mock"
    said = branch_gate.publish_notices(policy, db, commit=True, log=lambda line: None)
    assert said.endswith("committed and pushed")
    assert "Removal notices: 1 records" in run(work, "log", "-1", "--format=%s")
    assert run(world["remote"], "rev-parse", "refs/heads/main") == run(work, "rev-parse", "HEAD"), "the record is on the remote"
    assert branch_gate.publish_notices(policy, db, commit=True, log=lambda line: None).endswith("already recorded in Git")


# ---- the record on GitHub: a pull request per landed review, and each account's declared Git identity ----

def test_a_landing_on_github_opens_a_pull_request_naming_submitter_approver_and_authors(world, monkeypatch):
    db, policy = world["db"], world["policy"]
    calls = []

    def gh(*args):
        calls.append(args)
        if args[:2] == ("pr", "list"):
            return 0, "[]"
        if args[:2] == ("pr", "create"):
            return 0, "https://github.com/owner/work/pull/7"
        return 1, "unexpected"

    monkeypatch.setattr(branch_gate, "github_slug", lambda path, remote="origin": "owner/work")
    branch_review.set_identity(db, user_id=ANN, name="t", email="t@example.invalid", now=1.0)
    head = branch(world, "feature")
    made = submit(world, "feature", head, timeout_s=0)
    branch_review.decide(db, made["id"], user_id=ADMIN, username="admin", head=head, decision="approved", note="fine")
    done = branch_gate.land(policy, db, apply=True, now=1000.0, log=lambda line: None, gh=gh)
    assert done[0]["action"] == "landed" and remote_has(world, "main") == head
    create = next(args for args in calls if args[:2] == ("pr", "create"))
    body = create[create.index("--body") + 1]
    assert create[create.index("--head") + 1] == "feature" and create[create.index("--base") + 1] == "main"
    assert "submitted by **ann**, approved by admin" in body and head in body and "- t <t@example.invalid>" in body
    assert "every commit is by ann's declared identity" in body and "Decision note: fine" in body
    detail = branch_review.get(db, made["id"])["landed_detail"]
    assert "https://github.com/owner/work/pull/7" in detail and "every commit is by ann's declared identity" in detail


def test_a_github_that_cannot_answer_never_holds_a_landing_and_an_open_pull_request_is_updated(world, monkeypatch):
    db, policy = world["db"], world["policy"]
    monkeypatch.setattr(branch_gate, "github_slug", lambda path, remote="origin": "owner/work")
    head = branch(world, "feature")
    made = submit(world, "feature", head, timeout_s=0)
    done = branch_gate.land(policy, db, apply=True, now=1000.0, log=lambda line: None, gh=lambda *a: (1, "not logged in"))
    assert done[0]["action"] == "landed" and remote_has(world, "main") == head
    record = branch_review.get(db, made["id"])["landed_detail"]
    assert "no pull request (GitHub did not answer: not logged in)" in record and "ann has declared no Git identity" in record
    edits = []

    def gh(*args):
        edits.append(args)
        return (0, '[{"number": 3, "url": "https://github.com/owner/work/pull/3"}]') if args[1] == "list" else (0, "")

    assert branch_gate.record_pull_request("owner/work", {**branch_review.get(db, made["id"]), "remarks": []}, [], gh=gh) \
        == "https://github.com/owner/work/pull/3"
    assert edits[-1][:3] == ("pr", "edit", "3")


def test_an_account_declares_its_own_git_identity_and_mismatches_are_named(world):
    db = world["db"]
    with pytest.raises(branch_review.ReviewError):
        branch_review.set_identity(db, user_id=ANN, name="a <b>", email="a@b")
    with pytest.raises(branch_review.ReviewError):
        branch_review.set_identity(db, user_id=ANN, name="ann", email="not an address")
    branch_review.set_identity(db, user_id=ANN, name="Ann", email="ann@example.invalid", now=2.0)
    assert branch_review.identity(db, ANN) == {"name": "Ann", "email": "ann@example.invalid", "declared_at": 2.0}
    review = {"owner_id": ANN, "owner_name": "ann"}
    assert branch_review.authorship(db, review, ["Ann <ann@example.invalid>"]) == "every commit is by ann's declared identity"
    assert branch_review.authorship(db, review, ["Albert <albert@example.invalid>"]) == \
        "commits not by ann's declared identity: Albert <albert@example.invalid>"


def test_the_landing_queue_lands_only_what_is_asked_and_keeps_why_the_rest_held(world):
    db, policy = world["db"], world["policy"]
    first, second = branch(world, "first"), branch(world, "second")
    one, two = submit(world, "first", first, timeout_s=0), submit(world, "second", second, timeout_s=0)
    assert [r["id"] for r in branch_review.queue(db, now=1000.0)] == [one["id"], two["id"]]
    done = branch_gate.land(policy, db, apply=True, now=1000.0, log=lambda line: None, only=[two["id"]])
    assert actions(done) == {"second": "landed"} and remote_has(world, "main") == second
    queued = branch_review.queue(db, now=1000.0)
    assert [r["id"] for r in queued] == [one["id"]] and queued[0]["land_tried_at"] is None, "only the asked change was tried"
    done = branch_gate.land(policy, db, apply=True, now=1001.0, log=lambda line: None, only=[one["id"]])
    assert actions(done) == {"first": "held"}
    held = branch_review.queue(db, now=1001.0)[0]
    assert held["land_tried_at"] == 1001.0 and held["land_result"].startswith("held: not a fast-forward of main")
    assert branch_review.get(db, two["id"])["land_result"] is None, "a landed change keeps its landing record, not a hold"


def test_the_queue_columns_are_added_to_a_database_made_before_them(tmp_path):
    db = sqlite3.connect(tmp_path / "old.db", isolation_level=None)
    db.execute(branch_review.DDL[0].replace(", land_tried_at REAL, land_result TEXT", ""))
    branch_review.ensure(db)
    columns = {row[1] for row in db.execute("PRAGMA table_info(branch_reviews)")}
    assert {"land_tried_at", "land_result"} <= columns
    branch_review.ensure(db)                     # idempotent


def test_land_now_over_rest_is_an_admins_and_asks_the_updater_for_exactly_the_queue(world):
    policy, db_file = world["policy"], world["db_file"]
    head = branch(world, "feature")
    made = submit(world, "feature", head, timeout_s=0)
    asked = []

    def updater(ids):
        asked.append(ids)
        return {"outcome": "done", "status": 2}

    def ask(path, land_request=updater, **kw):
        handler = Handler(path, **kw)
        branch_review.http_route(handler, path, "POST" if "body" in kw else "GET", policy=policy, accounts_db=db_file,
                                 land_request=land_request)
        return handler.sent

    status, listing = ask("/api/reviews", user=ANN, username="ann")
    assert status == 200 and listing["updater"] is True and [r["id"] for r in listing["queue"]] == [made["id"]]
    assert ask("/api/reviews", user=ANN, username="ann", land_request=None)[1]["updater"] is False
    assert ask("/api/reviews/land", user=ANN, username="ann", body={"all": True})[0] == 403, "only an admin lands"
    assert ask("/api/reviews/land", user=ADMIN, username="admin", body={"all": True}, land_request=None)[0] == 503
    assert ask("/api/reviews/land", user=ADMIN, username="admin", body={"ids": ["nope"]})[0] == 409
    assert ask("/api/reviews/land", user=ADMIN, username="admin", body={"ids": []})[0] == 400
    assert asked == [], "nothing refused reached the updater"
    status, answer = ask("/api/reviews/land", user=ADMIN, username="admin", body={"ids": [made["id"]]})
    assert status == 200 and answer == {"asked": [made["id"]], "outcome": {"outcome": "done", "status": 2}}
    status, answer = ask("/api/reviews/land", user=ADMIN, username="admin", body={"all": True})
    assert status == 200 and answer["asked"] == [made["id"]]
    assert asked == [[made["id"]], None], "ids for those, none (the whole queue) for all"


def test_a_one_off_land_request_names_its_reviews_to_the_task_worker(world):
    from types import SimpleNamespace
    from meowdoku import scheduled_tasks
    first, second = branch(world, "first"), branch(world, "second")
    one = submit(world, "first", first, timeout_s=0)
    submit(world, "second", second, timeout_s=0)
    table = scheduled_tasks.handlers(SimpleNamespace(DONE=2, UNCHANGED=3), branches={
        "policy": world["policy_file"], "accounts_db": world["db_file"], "module": branch_gate})
    status, said, detail = table["reviews.land"]({"task": "reviews.land", "reviews": [one["id"]]})
    assert said == "1 would land" and [e["branch"] for e in detail] == ["first"]
    status, said, detail = table["reviews.land"]({"task": "reviews.land"})
    assert said == "2 would land", "a request that names none is the whole queue, as the schedule's"
