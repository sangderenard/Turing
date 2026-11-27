"""The readable project space (project_space.py): confined, recorded, accounted, and left alone by the cleaner."""
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT))

from meowdoku import branch_gate, branch_review, central_checkout, project_open, project_space  # noqa: E402

ANN, BOB = 2, 3
CLEANER = branch_gate.DEFAULTS["cleaner"]
LIMITS = {"capacity_bytes": 50_000_000, "user_quota_bytes": 20_000_000, "free_floor_bytes": 1,
          "space_reservation_bytes": 1_000_000}


def run(path, *args, check=True):
    return subprocess.run(["git", "-C", str(path), *args], check=check, capture_output=True, text=True).stdout.strip()


def contract(identity="ann/garden", space="projects/garden"):
    return yaml.safe_dump({"schema": "project-contract-v0", "identity": identity,
                           "placement": {"mode": "standalone", "space": space},
                           "scope": {"roots": ["src"], "include": {"languages": ["python"]}},
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
    db_file = tmp_path / "accounts.db"
    db = sqlite3.connect(db_file, isolation_level=None)
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL)")
    db.execute("CREATE TABLE account_roles (user_id INTEGER PRIMARY KEY, role TEXT NOT NULL)")
    db.executemany("INSERT INTO users VALUES (?,?)", [(ANN, "ann"), (BOB, "bob"), (9, CLEANER)])
    db.execute("INSERT INTO account_roles VALUES (9, 'service')")
    for user, name in ((ANN, "Ann"), (BOB, "Bob")):
        branch_review.set_identity(db, user_id=user, name=name, email=f"{name.lower()}@example.invalid")
    policy = {**branch_gate.DEFAULTS, "lock": "enforce", "repositories": {"work": {"path": str(work), "default": "main"}}}
    content = tmp_path / "state" / "content"
    content.mkdir(parents=True)
    return {"work": work, "db": db, "db_file": db_file, "policy": policy, "content": content,
            "space": project_space.Space(db_file, content, dict(LIMITS)), "root": content / "checkouts"}


def open_(world, text=None, user=ANN, username="ann", space=None):
    return project_open.open_project(world["db"], world["policy"], world["root"], user_id=user, username=username,
                                     text=text or contract(), space=space or world["space"])


def op(world, operation, user=ANN, username="ann", identity="ann/garden", **body):
    return world["space"].run(world["db"], user_id=user, username=username, operation=operation,
                              body={"identity": identity, **body})


def folder(world, user=ANN, name="garden"):
    return world["content"] / "workspaces" / "projects" / str(user) / name


def lease(world, user=ANN):
    db = sqlite3.connect(world["db_file"])
    try:
        return db.execute("SELECT state, reserved_bytes, measured_bytes FROM workspace_leases WHERE owner_id = ?",
                          (user,)).fetchall()
    finally:
        db.close()


def test_a_standalone_project_opens_into_its_own_git_space_with_a_lease(world):
    text = contract()
    opened = open_(world, text)
    place = folder(world)
    assert opened["mode"] == "standalone" and opened["space"] == "projects/garden" and place.is_dir()
    assert (place / "project.yaml").read_text(encoding="utf-8") == text
    assert run(place, "remote") == "", "a space has no remote"
    assert run(place, "log", "--format=%an|%s") == "Ann|Open project ann/garden (standalone placement)"
    assert run(place, "rev-parse", "HEAD") == opened["head"]
    [(state, reserved, measured)] = lease(world)
    assert state == "active" and reserved == LIMITS["space_reservation_bytes"] and measured == opened["space_bytes"] > 0
    assert [p["identity"] for p in project_open.projects_of(world["db"], ANN)] == ["ann/garden"]
    assert [s["identity"] for s in project_space.spaces_of(world["db"], ANN)] == ["ann/garden"]
    with pytest.raises(project_open.ProjectError, match="already open"):
        open_(world, contract(space="projects/garden-2"))


def test_unconfigured_the_space_refuses_and_makes_nothing(world, tmp_path):
    bare = project_space.Space(world["db_file"], world["content"], None)
    with pytest.raises(project_open.ProjectError, match="not configured"):
        open_(world, space=bare)
    with pytest.raises(project_open.ProjectError, match="not configured"):
        project_open.open_project(world["db"], world["policy"], world["root"], user_id=ANN, username="ann",
                                  text=contract(), space=None)
    assert not (world["content"] / "workspaces").exists()
    assert not world["db"].execute("SELECT 1 FROM projects").fetchone()
    assert project_space.load_config(tmp_path / "missing.json") is None
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({**LIMITS, "space_reservation_bytes": 0}), encoding="utf-8")
    with pytest.raises(project_space.SpaceError, match="positive whole number"):
        project_space.load_config(wrong)
    right = tmp_path / "right.json"
    right.write_text(json.dumps(LIMITS), encoding="utf-8")
    assert project_space.load_config(right) == LIMITS


def test_the_checkout_operations_work_on_the_space_and_every_call_is_recorded(world):
    open_(world)
    place = folder(world)
    op(world, "write", path="src/main.py", text="print('hi')\r\n")
    assert (place / "src" / "main.py").read_bytes() == b"print('hi')\r\n", "bytes as given"
    assert op(world, "read", path="src/main.py")["text"] == "print('hi')\r\n"
    assert [e["name"] for e in op(world, "files")["entries"]] == ["src", "project.yaml"]
    assert [c["path"] for c in op(world, "status")["changes"]] == ["src/main.py"]
    op(world, "edit", path="project.yaml", old="sources", new="garden-view")
    assert "+  name: garden-view" in op(world, "diff")["diff"]
    committed = op(world, "commit", message="first sources")
    assert run(place, "log", "-1", "--format=%an|%s") == "Ann|first sources" and committed["space_bytes"] > 0
    op(world, "remove", path="src/main.py")
    assert not (place / "src" / "main.py").exists()
    with pytest.raises(project_space.SpaceError, match="operation must be one of"):
        op(world, "merge")
    with pytest.raises(project_space.SpaceError, match="no such file"):
        op(world, "read", path="missing.py")
    rows = world["db"].execute("SELECT repository, branch, operation, ok FROM checkout_calls "
                               "WHERE repository = 'project-space' ORDER BY id").fetchall()
    assert [(r[2], r[3]) for r in rows] == [("write", 1), ("read", 1), ("files", 1), ("status", 1), ("edit", 1),
                                           ("diff", 1), ("commit", 1), ("remove", 1), ("read", 0)]
    assert {r[1] for r in rows} == {"ann/garden"}
    [(_, _, measured)] = lease(world)
    assert measured == project_space._size(place), "measured after every change, .git included"


@pytest.mark.parametrize("path", ["../escape.txt", "..\\escape.txt", ".git/config", "src/.GIT/x", "CON", "nul.txt",
                                  "a:b", "C:/x", "src/x?.py", "trailing. "])
def test_a_path_stays_inside_the_space(world, path):
    open_(world)
    with pytest.raises(project_space.SpaceError, match="not allowed|leaves|must be"):
        op(world, "write", path=path, text="x")
    assert not (folder(world).parent / "escape.txt").exists()


def test_a_link_inside_the_space_is_not_followed(world, tmp_path):
    open_(world)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = folder(world) / "linked"
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], check=True, capture_output=True)
    else:
        link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(project_space.SpaceError, match="passes through a link"):
        op(world, "write", path="linked/x.txt", text="x")
    assert not (outside / "x.txt").exists()


def test_size_limits_and_quota_refuse_before_writing(world, monkeypatch):
    open_(world)
    monkeypatch.setattr(central_checkout, "MAX_FILE_BYTES", 100)
    with pytest.raises(project_space.SpaceError, match="larger than 100 bytes"):
        op(world, "write", path="big.txt", text="x" * 101)
    monkeypatch.undo()
    world["space"] = project_space.Space(world["db_file"], world["content"], dict(LIMITS))
    over = LIMITS["user_quota_bytes"] - LIMITS["space_reservation_bytes"] + 10
    with pytest.raises(project_space.SpaceError, match="quota refuses this change") as refused:
        op(world, "write", path="huge.txt", text="x" * (LIMITS["space_reservation_bytes"] + over))
    assert refused.value.status == 507 and not (folder(world) / "huge.txt").exists(), "refused before it is written"
    assert op(world, "write", path="small.txt", text="fits")["bytes"] == 4, "within the reservation it is admitted"


def test_quota_refuses_a_new_space_and_nothing_is_made(world):
    tight = project_space.Space(world["db_file"], world["content"],
                                {**LIMITS, "space_reservation_bytes": LIMITS["user_quota_bytes"] + 1})
    with pytest.raises(project_open.ProjectError, match="quota refuses a new space"):
        open_(world, space=tight)
    assert not folder(world).exists() and not lease(world)
    assert not world["db"].execute("SELECT 1 FROM projects").fetchone()


def test_a_failure_while_making_the_space_removes_it_and_releases_its_lease(world, monkeypatch):
    real = project_space._git

    def failing(where, *args):
        if "commit" in args:
            raise project_space.SpaceError("git commit failed: disk", 500)
        return real(where, *args)

    monkeypatch.setattr(project_space, "_git", failing)
    with pytest.raises(project_open.ProjectError, match="disk"):
        open_(world)
    assert not folder(world).exists()
    assert lease(world) == [("released", 0, 0)]
    assert not world["db"].execute("SELECT 1 FROM projects").fetchone()
    assert not world["db"].execute("SELECT 1 FROM project_spaces").fetchone()
    monkeypatch.setattr(project_space, "_git", real)
    assert open_(world)["identity"] == "ann/garden", "and the same open then succeeds"


def test_one_account_never_reaches_anothers_space(world):
    open_(world)
    with pytest.raises(project_space.SpaceError, match="no project space with that identity"):
        op(world, "read", user=BOB, username="bob", path="project.yaml")
    with pytest.raises(project_open.ProjectError, match="names account 'ann'"):
        open_(world, user=BOB, username="bob")
    bob = open_(world, contract(identity="bob/garden"), user=BOB, username="bob")
    assert folder(world, BOB).is_dir() and bob["identity"] == "bob/garden", "the same space name, another account"


def test_the_cleaner_leaves_spaces_alone(world):
    open_(world)
    place = folder(world)
    trees = world["work"].parent / "trees"
    run(world["work"], "worktree", "add", "-q", "-b", "idle", str(trees / "idle"), "main")
    later = time.time() + 400 * 86400
    done = branch_gate.clean_worktrees(world["policy"], world["db"], apply=True, actor=CLEANER, now=later,
                                       log=lambda line: None)
    assert all(not str(e["worktree"]).startswith(str(world["content"])) for e in done), "a space is never considered"
    assert place.is_dir() and (place / "project.yaml").is_file() and run(place, "rev-parse", "HEAD")


def test_rest_needs_dev_ui_and_takes_the_account_from_the_session(world):
    open_(world)

    class Handler:
        def __init__(self, path, body, user, username, caps=("dev_ui",)):
            self.path, self.headers, self._body, self.sent = path, {"Content-Type": "application/json"}, body, None
            self._user, self._username, self._capset = user, username, set(caps)

        def _principal(self):
            from types import SimpleNamespace
            return SimpleNamespace(user_id=self._user, username=self._username)

        def _caps(self):
            return self._capset

        def _body_json(self, limit):
            if not 0 < len(json.dumps(self._body).encode("utf-8")) <= limit:
                self._json({"error": "body size is invalid"}, 400)
                return None
            return self._body

        def _json(self, payload, status=200):
            self.sent = (status, payload)

    def post(operation, body, user=ANN, username="ann", caps=("dev_ui",)):
        handler = Handler(f"/api/projects/space/{operation}", body, user, username, caps)
        project_space.http_route(handler, handler.path, "POST", space=world["space"], accounts_db=world["db_file"])
        return handler.sent

    assert post("read", {"identity": "ann/garden", "path": "project.yaml"}, caps=())[0] == 403
    assert post("read", {"identity": "ann/garden", "path": "project.yaml"}, user=None, username=None)[0] == 401
    status, answer = post("read", {"identity": "ann/garden", "path": "project.yaml", "user_id": BOB})
    assert status == 200 and answer["text"].startswith("schema: project-contract-v0")
    assert post("read", {"identity": "ann/garden", "path": "project.yaml"}, user=BOB, username="bob")[0] == 404
    assert post("write", {"identity": "ann/garden", "path": "../x", "text": "x"})[0] == 400
    text = "\x01" * 1000                                   # issue 6d7b113d: six bytes each in JSON
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(central_checkout, "MAX_FILE_BYTES", 1000)
        assert len(json.dumps({"text": text}).encode("utf-8")) > 3 * 1000, "the old limit refused it"
        status, answer = post("write", {"identity": "ann/garden", "path": "src/control.txt", "text": text})
        assert status == 200 and answer["bytes"] == 1000, answer
        assert post("write", {"identity": "ann/garden", "path": "src/big.txt", "text": "a" * 1001})[0] == 413
        patch.setattr(central_checkout, "ENVELOPE_BYTES", 1000)         # review d16933d0: edit carries two texts
        body = {"identity": "ann/garden", "path": "src/control.txt", "old": text, "new": "\x02" * 1000}
        assert len(json.dumps(body).encode("utf-8")) > central_checkout.body_limit("write")
        status, answer = post("edit", body)
        assert status == 200, answer
        assert post("read", {"identity": "ann/garden", "path": "src/control.txt"})[1]["text"] == "\x02" * 1000
