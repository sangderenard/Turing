"""The central checkout: the service makes an account's branch and working folder and does its Git; the lander lands it."""
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from meowdoku import branch_gate, branch_review, central_checkout  # noqa: E402

ADMIN, ANN, BOB = 1, 2, 3


def run(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True).stdout.strip()


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
    db.executemany("INSERT INTO users VALUES (?,?)", [(ADMIN, "admin"), (ANN, "ann"), (BOB, "bob")])
    db.execute("INSERT INTO account_roles VALUES (?, 'admin')", (ADMIN,))
    policy = {**branch_gate.DEFAULTS, "lock": "enforce", "repositories": {"work": {"path": str(work), "default": "main"}}}
    return {"remote": remote, "work": work, "db": db, "policy": policy, "root": tmp_path / "state" / "checkouts"}


def call(world, operation, user=ANN, username="ann", **body):
    return central_checkout.run(world["db"], world["policy"], world["root"], user_id=user, username=username,
                                operation=operation, body={"repository": "work", **body})


def test_an_account_edits_commits_and_lands_without_running_git(world):
    db = world["db"]
    opened = call(world, "open", branch="ann/first")
    assert opened["head"] == run(world["work"], "rev-parse", "origin/main"), "made from the remote's default branch"
    assert branch_review.owner(db, "work", "ann/first")["user_id"] == ANN, "the SQL checkout is part of opening"
    folder = Path(db.execute("SELECT path FROM checkouts").fetchone()[0])
    assert folder.parent.parent.parent == world["root"] and folder.is_dir(), "the working folder is in the state root"

    call(world, "write", branch="ann/first", path="notes/hello.txt", text="hello\r\nworld\n")
    assert (folder / "notes" / "hello.txt").read_bytes() == b"hello\r\nworld\n", "bytes as given, no newline translation"
    assert call(world, "read", branch="ann/first", path="notes/hello.txt")["text"] == "hello\r\nworld\n"
    call(world, "write", branch="ann/first", path="base.txt", text="base, changed\n")
    listed = call(world, "files", branch="ann/first")["entries"]
    assert [e["name"] for e in listed] == ["notes", "base.txt"] and ".git" not in [e["name"] for e in listed]
    assert {c["path"] for c in call(world, "status", branch="ann/first")["changes"]} == {"base.txt", "notes/hello.txt"}
    assert "+base, changed" in call(world, "diff", branch="ann/first")["diff"]

    with pytest.raises(central_checkout.CheckoutError, match="declare your Git author identity"):
        call(world, "commit", branch="ann/first", message="first")
    branch_review.set_identity(db, user_id=ANN, name="Ann", email="ann@example.invalid")
    head = call(world, "commit", branch="ann/first", message="Ann's first change")["head"]
    assert run(world["work"], "rev-parse", "refs/heads/ann/first") == head, "the branch is in the core repository"
    assert run(world["work"], "log", "-1", "--format=%an <%ae>", head) == "Ann <ann@example.invalid>"
    assert not call(world, "status", branch="ann/first")["changes"]

    branch_review.submit(db, user_id=ANN, username="ann", repository="work", branch="ann/first", base="main", head=head,
                         timeout_s=0, delete_branch=False, known=["work"], now=1000.0)
    done = branch_gate.land(world["policy"], db, apply=True, now=2000.0, log=lambda line: None)
    assert [entry["action"] for entry in done] == ["landed"], "the lander finds and lands it unchanged"
    assert run(world["remote"], "rev-parse", "main") == head

    closed = call(world, "close", branch="ann/first")
    assert closed["closed"] and not folder.exists()
    assert run(world["work"], "rev-parse", "refs/heads/ann/first") == head, "closing keeps the branch"
    calls = [r[0] for r in db.execute("SELECT operation FROM checkout_calls WHERE ok = 1 ORDER BY id")]
    assert calls[0] == "open" and calls[-1] == "close" and "commit" in calls, "every call is recorded"


@pytest.mark.parametrize("path", ["../escape.txt", "..\\escape.txt", ".git/config", "a/.GIT/x", "C:/x.txt", "con.txt",
                                  "aux", "a:stream", "dot.", "space ", "q?.txt", "a/../../b"])
def test_a_path_stays_inside_the_working_folder(world, path):
    call(world, "open", branch="ann/paths")
    with pytest.raises(central_checkout.CheckoutError):
        call(world, "write", branch="ann/paths", path=path, text="x")
    assert db_failures(world) >= 1, "a refusal is recorded too"


def db_failures(world):
    return world["db"].execute("SELECT COUNT(*) FROM checkout_calls WHERE ok = 0").fetchone()[0]


@pytest.mark.skipif(os.name != "nt", reason="junctions are a Windows thing")
def test_a_junction_inside_the_working_folder_is_not_followed(world, tmp_path):
    call(world, "open", branch="ann/junction")
    folder = Path(world["db"].execute("SELECT path FROM checkouts").fetchone()[0])
    outside = tmp_path / "outside"
    outside.mkdir()
    subprocess.run(["cmd", "/c", "mklink", "/J", str(folder / "door"), str(outside)], check=True, capture_output=True)
    for operation, extra in (("write", {"text": "x"}), ("read", {}), ("files", {})):
        with pytest.raises(central_checkout.CheckoutError, match="link"):
            call(world, operation, branch="ann/junction", path="door/secret.txt" if operation != "files" else "door", **extra)
    assert not any(outside.iterdir())
    assert "door" not in [e["name"] for e in call(world, "files", branch="ann/junction")["entries"]]


def test_one_account_never_reaches_anothers_checkout_or_branch(world):
    call(world, "open", branch="ann/mine")
    call(world, "write", branch="ann/mine", path="a.txt", text="ann's")
    for operation in ("read", "write", "status", "commit", "close"):
        with pytest.raises(central_checkout.CheckoutError) as refused:
            call(world, operation, user=BOB, username="bob", branch="ann/mine", path="a.txt", text="bob's", message="m")
        assert refused.value.status == 404
    with pytest.raises(central_checkout.CheckoutError) as taken:
        call(world, "open", user=BOB, username="bob", branch="ann/mine")
    assert taken.value.status == 409


def test_an_existing_or_default_branch_is_not_taken_over(world):
    run(world["work"], "branch", "someone/else", "main")
    with pytest.raises(central_checkout.CheckoutError, match="already exists"):
        call(world, "open", branch="someone/else")
    assert branch_review.owner(world["db"], "work", "someone/else") is None, "a refused open registers nothing"
    with pytest.raises(central_checkout.CheckoutError, match="default branch"):
        call(world, "open", branch="main")
    with pytest.raises(central_checkout.CheckoutError):
        call(world, "open", branch="../main")
    with pytest.raises(central_checkout.CheckoutError) as unknown:
        call(world, "open", branch="ann/x", repository="elsewhere")
    assert unknown.value.status == 404


def test_reopening_returns_the_same_checkout_and_closing_needs_a_commit(world):
    first = call(world, "open", branch="ann/again")
    assert call(world, "open", branch="ann/again")["head"] == first["head"]
    assert world["db"].execute("SELECT COUNT(*) FROM checkouts").fetchone()[0] == 1
    call(world, "write", branch="ann/again", path="x.txt", text="x")
    with pytest.raises(central_checkout.CheckoutError, match="uncommitted"):
        call(world, "close", branch="ann/again")
    call(world, "remove", branch="ann/again", path="x.txt")
    assert call(world, "close", branch="ann/again")["closed"]
    reopened = call(world, "open", branch="ann/again")
    assert reopened["head"] == first["head"], "the branch was kept; opening it again makes a new working folder"


def test_large_binary_and_unknown_operations_are_refused(world):
    call(world, "open", branch="ann/limits")
    folder = Path(world["db"].execute("SELECT path FROM checkouts").fetchone()[0])
    (folder / "blob.bin").write_bytes(b"\0\1\2")
    with pytest.raises(central_checkout.CheckoutError, match="binary"):
        call(world, "read", branch="ann/limits", path="blob.bin")
    with pytest.raises(central_checkout.CheckoutError) as large:
        call(world, "write", branch="ann/limits", path="big.txt", text="x" * (central_checkout.MAX_FILE_BYTES + 1))
    assert large.value.status == 413
    with pytest.raises(central_checkout.CheckoutError) as unknown:
        call(world, "push", branch="ann/limits")
    assert unknown.value.status == 404


def test_rest_takes_the_account_from_the_session_only(world):
    class Handler:
        def __init__(self, path, body, user, username, caps=("dev_ui",)):
            raw = json.dumps(body).encode("utf-8")
            self.path, self.headers, self._raw, self.sent = path, {"Content-Type": "application/json",
                                                                 "Content-Length": str(len(raw))}, raw, None
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

    def post(operation, body, user=ANN, username="ann", caps=("dev_ui",)):
        handler = Handler(f"/api/checkouts/{operation}", body, user, username, caps)
        central_checkout.http_route(handler, handler.path, "POST", policy=world["policy"], accounts_db=db_file,
                                    root=world["root"])
        return handler.sent

    status, answer = post("open", {"repository": "work", "branch": "ann/rest", "user_id": BOB})
    assert status == 200 and branch_review.owner(world["db"], "work", "ann/rest")["user_id"] == ANN, "a body's user is ignored"
    assert post("open", {"repository": "work", "branch": "ann/rest2"}, user=None, username=None)[0] == 401
    assert post("open", {"repository": "work", "branch": "ann/rest2"}, caps=())[0] == 403
    assert post("read", {"repository": "work", "branch": "ann/rest", "path": "../x"})[0] == 400


def folder_of(world):
    return Path(world["db"].execute("SELECT path FROM checkouts").fetchone()[0])


def test_exact_edit_replaces_known_text_and_refuses_ambiguity(world):
    call(world, "open", branch="ann/exact")
    folder = folder_of(world)
    call(world, "write", branch="ann/exact", path="a.txt", text="one\ntwo\ntwo\n")
    done = call(world, "edit", branch="ann/exact", path="a.txt", old="one", new="ONE")
    assert done["format"] == "exact" and (folder / "a.txt").read_text(encoding="utf-8") == "ONE\ntwo\ntwo\n"
    with pytest.raises(central_checkout.CheckoutError, match="occurs 2 time"):
        call(world, "edit", branch="ann/exact", path="a.txt", old="two", new="2")
    call(world, "edit", branch="ann/exact", path="a.txt", old="two", new="2", count=2)
    assert (folder / "a.txt").read_text(encoding="utf-8") == "ONE\n2\n2\n"
    with pytest.raises(central_checkout.CheckoutError, match="occurs 0 time"):
        call(world, "edit", branch="ann/exact", path="a.txt", old="three", new="3")
    call(world, "write", branch="ann/exact", path="w.txt", text="x\r\ny\r\n")
    call(world, "edit", branch="ann/exact", path="w.txt", old="x\ny", new="x\nz")
    assert (folder / "w.txt").read_bytes() == b"x\r\nz\r\n", "Windows line endings in the file are kept"
    with pytest.raises(central_checkout.CheckoutError):
        call(world, "edit", branch="ann/exact", path="../a.txt", old="ONE", new="1")


UNIFIED = """--- a/base.txt
+++ b/base.txt
@@ -1 +1 @@
-base
+base, patched
diff --git a/new.txt b/new.txt
new file mode 100644
--- /dev/null
+++ b/new.txt
@@ -0,0 +1 @@
+hello
"""


def test_a_unified_diff_is_checked_then_applied(world):
    call(world, "open", branch="ann/unified")
    folder = folder_of(world)
    done = call(world, "edit", branch="ann/unified", patch=UNIFIED)
    assert done["format"] == "unified" and done["files"] == ["base.txt", "new.txt"]
    assert (folder / "base.txt").read_text(encoding="utf-8") == "base, patched\n"
    assert (folder / "new.txt").read_text(encoding="utf-8") == "hello\n"
    stale = UNIFIED.split("diff --git")[0]                # base.txt no longer reads "base": the hunk cannot apply
    with pytest.raises(central_checkout.CheckoutError, match="nothing was changed"):
        call(world, "edit", branch="ann/unified", patch=stale)
    assert (folder / "base.txt").read_text(encoding="utf-8") == "base, patched\n"
    for refused in ("--- a/../x.txt\n+++ b/../x.txt\n@@ -0,0 +1 @@\n+x\n",
                    "--- a/.git/config\n+++ b/.git/config\n@@ -0,0 +1 @@\n+x\n",
                    "diff --git a/b.bin b/b.bin\nGIT binary patch\nliteral 3\n",
                    "diff --git a/l b/l\nnew file mode 120000\n--- /dev/null\n+++ b/l\n@@ -0,0 +1 @@\n+..\n",
                    "just some text\n"):
        with pytest.raises(central_checkout.CheckoutError):
            call(world, "edit", branch="ann/unified", patch=refused)
    assert not (folder.parent / "x.txt").exists()


CODEX = """*** Begin Patch
*** Add File: docs/notes.md
+# Notes
+first
*** Update File: base.txt
@@
-base
+base, by codex
*** End Patch
"""


def test_a_codex_patch_applies_whole_or_not_at_all(world):
    call(world, "open", branch="ann/codex")
    folder = folder_of(world)
    done = call(world, "edit", branch="ann/codex", patch=CODEX)
    assert done["format"] == "codex" and done["files"] == ["base.txt", "docs/notes.md"]
    assert (folder / "docs" / "notes.md").read_text(encoding="utf-8") == "# Notes\nfirst\n"
    assert (folder / "base.txt").read_text(encoding="utf-8") == "base, by codex\n"
    moved = """*** Begin Patch
*** Update File: docs/notes.md
*** Move to: docs/renamed.md
@@ # Notes
-first
+second
*** End Patch"""
    call(world, "edit", branch="ann/codex", patch=moved)
    assert not (folder / "docs" / "notes.md").exists()
    assert (folder / "docs" / "renamed.md").read_text(encoding="utf-8") == "# Notes\nsecond\n"
    broken = """*** Begin Patch
*** Add File: should-not-exist.txt
+x
*** Update File: base.txt
@@
-this line is not there
+y
*** End Patch"""
    with pytest.raises(central_checkout.CheckoutError, match="not in the file"):
        call(world, "edit", branch="ann/codex", patch=broken)
    assert not (folder / "should-not-exist.txt").exists(), "a patch that fails part-way writes nothing"
    call(world, "edit", branch="ann/codex", patch="*** Begin Patch\n*** Delete File: docs/renamed.md\n*** End Patch\n")
    assert not (folder / "docs" / "renamed.md").exists()
    with pytest.raises(central_checkout.CheckoutError):
        call(world, "edit", branch="ann/codex", patch="*** Begin Patch\n*** Add File: ../out.txt\n+x\n*** End Patch\n")
    recorded = [r[0] for r in world["db"].execute("SELECT path FROM checkout_calls WHERE operation = 'edit' AND ok = 1")]
    assert "base.txt,docs/notes.md" in recorded, "a patch's call records the files it changed"
