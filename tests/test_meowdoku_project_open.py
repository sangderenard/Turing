"""Opening a project in the branch placement (project_open.py): through the central checkout, all or nothing, once."""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT))

from meowdoku import branch_gate, branch_review, central_checkout, project_open  # noqa: E402

ANN, BOB = 2, 3


def run(path, *args, check=True):
    return subprocess.run(["git", "-C", str(path), *args], check=check, capture_output=True, text=True).stdout.strip()


def contract(identity="ann/garden", branch="ann/garden", mode="branch", repository="work"):
    placement = {"mode": mode}
    if mode in ("branch", "branch-copy"):
        placement["branch"] = {"repository": repository, "name": branch}
    if mode in ("standalone", "branch-copy"):
        placement["space"] = "projects/garden"
    return yaml.safe_dump({"schema": "project-contract-v0", "identity": identity, "placement": placement,
                           "scope": {"roots": ["examples/garden"], "include": {"languages": ["python"]}},
                           "view": {"name": "sources"}}, sort_keys=False)


@pytest.fixture
def world(tmp_path):
    remote, work = tmp_path / "remote.git", tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(work)], check=True)
    for key, value in (("user.name", "t"), ("user.email", "t@example.invalid"), ("commit.gpgsign", "false")):
        run(work, "config", key, value)
    run(work, "remote", "add", "origin", str(remote))
    (work / "base.txt").write_text("base\n", encoding="utf-8")
    run(work, "add", "base.txt")
    run(work, "commit", "-q", "-m", "base")
    run(work, "push", "-q", "origin", "main")
    db = sqlite3.connect(tmp_path / "accounts.db", isolation_level=None)
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL)")
    db.execute("CREATE TABLE account_roles (user_id INTEGER PRIMARY KEY, role TEXT NOT NULL)")
    db.executemany("INSERT INTO users VALUES (?,?)", [(ANN, "ann"), (BOB, "bob")])
    branch_review.set_identity(db, user_id=ANN, name="Ann", email="ann@example.invalid")
    policy = {**branch_gate.DEFAULTS, "lock": "enforce", "repositories": {"work": {"path": str(work), "default": "main"}}}
    return {"work": work, "db": db, "policy": policy, "root": tmp_path / "state" / "checkouts"}


def open_(world, text, user=ANN, username="ann"):
    return project_open.open_project(world["db"], world["policy"], world["root"], user_id=user, username=username,
                                     text=text)


def nothing_made(world, branch="ann/garden"):
    db = world["db"]
    project_open.ensure(db)
    return (not db.execute("SELECT 1 FROM checkouts WHERE branch = ?", (branch,)).fetchone()
            and branch_review.owner(db, "work", branch) is None
            and not run(world["work"], "rev-parse", "-q", "--verify", f"refs/heads/{branch}", check=False)
            and not db.execute("SELECT 1 FROM projects").fetchone())


def test_a_branch_project_is_opened_through_the_checkout_and_committed(world):
    text = contract()
    opened = open_(world, text)
    assert opened["path"] == "projects/ann/garden/project.yaml" and opened["branch"] == "ann/garden"
    assert run(world["work"], "rev-parse", "refs/heads/ann/garden") == opened["head"], "the branch is in the core"
    assert run(world["work"], "show", f"{opened['head']}:projects/ann/garden/project.yaml") == text.rstrip("\n")
    assert run(world["work"], "log", "-1", "--format=%an|%s", opened["head"]) == \
        "Ann|Open project ann/garden (branch placement)"
    assert run(world["work"], "diff", "--name-only", "origin/main", opened["head"]) == "projects/ann/garden/project.yaml"
    assert branch_review.owner(world["db"], "work", "ann/garden")["user_id"] == ANN
    listed = project_open.projects_of(world["db"], ANN)
    assert [(p["identity"], p["head"]) for p in listed] == [("ann/garden", opened["head"])]
    calls = [r[0] for r in world["db"].execute("SELECT operation FROM checkout_calls WHERE ok = 1 ORDER BY id")]
    assert calls == ["open", "status", "write", "commit"], "every step is a recorded checkout operation"
    assert opened["contract"]["identity"] == "ann/garden" and "reviews_submit" in opened["next"]

    with pytest.raises(project_open.ProjectError, match="already open"):
        open_(world, contract(branch="ann/garden-2"))


@pytest.mark.parametrize("text, user, username, message", [
    ("schema: project-contract-v9\n", ANN, "ann", "schema must be"),
    (contract(mode="standalone"), ANN, "ann", "needs the readable project space"),
    (contract(mode="branch-copy"), ANN, "ann", "needs the readable project space"),
    (contract(identity="bob/garden"), ANN, "ann", "names account 'bob'"),
    (contract(repository="elsewhere"), ANN, "ann", "is not one of \\['work'\\]"),
    (contract(identity="bob/garden"), BOB, "bob", "declare your Git author identity"),
    ("", ANN, "ann", "must be the project.yaml text"),
])
def test_a_refusal_comes_before_anything_is_made(world, text, user, username, message):
    with pytest.raises(project_open.ProjectError, match=message):
        open_(world, text, user=user, username=username)
    assert nothing_made(world) and nothing_made(world, "bob/garden") and not world["root"].exists()


def test_a_branch_held_by_another_account_is_refused(world):
    branch_review.register(world["db"], user_id=BOB, username="bob", repository="work", branch="ann/garden",
                           known=["work"])
    with pytest.raises(project_open.ProjectError, match="registered to bob"):
        open_(world, contract())
    assert branch_review.owner(world["db"], "work", "ann/garden")["user_id"] == BOB


def test_an_open_checkout_is_used_only_when_committed_and_is_left_as_it_was(world):
    call = lambda op, **b: central_checkout.run(world["db"], world["policy"], world["root"], user_id=ANN,  # noqa: E731
                                                username="ann", operation=op,
                                                body={"repository": "work", "branch": "ann/garden", **b})
    call("open")
    call("write", path="notes.txt", text="mine\n")
    with pytest.raises(project_open.ProjectError, match="uncommitted changes"):
        open_(world, contract())
    assert [c["path"] for c in call("status")["changes"]] == ["notes.txt"], "the account's work is untouched"
    assert not world["db"].execute("SELECT 1 FROM projects").fetchone()
    call("commit", message="notes")
    opened = open_(world, contract())
    assert run(world["work"], "diff", "--name-only", f"{opened['head']}~1", opened["head"]) == \
        "projects/ann/garden/project.yaml", "the project's commit holds its file alone"
    assert call("status")["changes"] == [] and world["db"].execute("SELECT 1 FROM checkouts").fetchone()


def test_a_branch_already_holding_the_file_is_refused(world):
    call = lambda op, **b: central_checkout.run(world["db"], world["policy"], world["root"], user_id=ANN,  # noqa: E731
                                                username="ann", operation=op,
                                                body={"repository": "work", "branch": "ann/garden", **b})
    call("open")
    call("write", path="projects/ann/garden/project.yaml", text="old\n")
    call("commit", message="by hand")
    with pytest.raises(project_open.ProjectError, match="already holds projects/ann/garden/project.yaml"):
        open_(world, contract())
    assert call("read", path="projects/ann/garden/project.yaml")["text"] == "old\n"
    assert world["db"].execute("SELECT 1 FROM checkouts").fetchone(), "a checkout that was open stays open"


def test_a_failure_after_writing_removes_what_this_open_made(world, monkeypatch):
    real = central_checkout.commit

    def failing(db, row, *, user_id, message):
        raise central_checkout.CheckoutError("the disk is full", 507)

    monkeypatch.setattr(central_checkout, "commit", failing)
    with pytest.raises(project_open.ProjectError, match="the disk is full"):
        open_(world, contract())
    assert nothing_made(world), "no checkout, branch, registration or project row is left"
    assert not list(world["root"].rglob("project.yaml"))
    monkeypatch.setattr(central_checkout, "commit", real)
    assert open_(world, contract())["identity"] == "ann/garden", "and the same open then succeeds"


def test_a_refused_open_removes_only_its_own_row_not_another_open_in_flight(world):
    """[WHOLE] two opens on one branch at once: the refused one must not delete the other's head-NULL row."""
    call = lambda op, **b: central_checkout.run(world["db"], world["policy"], world["root"], user_id=ANN,  # noqa: E731
                                                username="ann", operation=op,
                                                body={"repository": "work", "branch": "ann/garden", **b})
    project_open.ensure(world["db"])
    world["db"].execute("INSERT INTO projects VALUES (?,?,?,?,?,?,?,?,?)",           # the other open, mid-flight
                        ("ann/other", ANN, "ann", "branch", "work", "ann/garden", "projects/ann/other/project.yaml",
                         None, 1.0))
    call("open")
    call("write", path="projects/ann/other/project.yaml", text="in flight\n")      # its file, not yet committed
    with pytest.raises(project_open.ProjectError, match="uncommitted changes"):
        open_(world, contract())
    rows = [r[0] for r in world["db"].execute("SELECT identity FROM projects")]
    assert rows == ["ann/other"], "the other open's in-flight row survives this open's undo"


def test_an_open_whose_row_is_lost_before_recording_its_head_is_not_reported_as_open(world, monkeypatch):
    real = central_checkout.commit

    def commit_then_lose_row(db, row, *, user_id, message):
        done = real(db, row, user_id=user_id, message=message)
        db.execute("DELETE FROM projects WHERE identity = ?", ("ann/garden",))
        return done

    monkeypatch.setattr(central_checkout, "commit", commit_then_lose_row)
    with pytest.raises(project_open.ProjectError, match="catalog row was removed meanwhile") as refused:
        open_(world, contract())
    assert refused.value.status == 409 and project_open.projects_of(world["db"], ANN) == []
    head = run(world["work"], "rev-parse", "refs/heads/ann/garden")
    assert head in refused.value.message, "the commit that stands is named"


def test_rest_needs_dev_ui_and_takes_the_account_from_the_session(world):
    class Handler:
        def __init__(self, path, body, user, username, caps=("dev_ui",)):
            raw = json.dumps(body).encode("utf-8")
            self.path, self.headers, self._raw, self.sent = path, {"Content-Type": "application/json"}, raw, None
            self._user, self._username, self._capset = user, username, set(caps)

        def _principal(self):
            from types import SimpleNamespace
            return SimpleNamespace(user_id=self._user, username=self._username)

        def _caps(self):
            return self._capset

        def _body_json(self, limit):
            return json.loads(self._raw)

        def _json(self, payload, status=200):
            self.sent = (status, payload)

    db_file = world["db"].execute("PRAGMA database_list").fetchone()[2]

    def send(path, method="POST", body=None, user=ANN, username="ann", caps=("dev_ui",)):
        handler = Handler(path, body or {}, user, username, caps)
        project_open.http_route(handler, path, method, policy=world["policy"], accounts_db=db_file, root=world["root"])
        return handler.sent

    assert send("/api/projects/open", body={"text": contract()}, caps=())[0] == 403
    assert send("/api/projects/open", body={"text": contract()}, user=None, username=None)[0] == 401
    assert send("/api/projects/open", body={"text": contract(identity="bob/garden")})[0] == 403
    status, answer = send("/api/projects/open", body={"text": contract(), "user_id": BOB})
    assert status == 200 and answer["identity"] == "ann/garden"
    status, answer = send("/api/projects", "GET")
    assert status == 200 and [p["identity"] for p in answer["projects"]] == ["ann/garden"]
    assert send("/api/projects/elsewhere")[0] == 404
