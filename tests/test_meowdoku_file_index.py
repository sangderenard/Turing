"""The flat file index, against a real temporary Git repository."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from meowdoku import branch_gate, file_index  # noqa: E402

CLEANER = branch_gate.DEFAULTS["cleaner"]


def run(path, *args):
    done = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


@pytest.fixture
def core(tmp_path):
    work = tmp_path / "work"
    subprocess.run(["git", "init", "-q", "-b", "main", str(work)], check=True)
    for key, value in (("user.name", "t"), ("user.email", "t@example.invalid"), ("commit.gpgsign", "false")):
        run(work, "config", key, value)
    (work / "src").mkdir()
    (work / "src" / "law.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    (work / "src" / "kernel.glsl").write_text("void main() {}\n", encoding="utf-8")
    (work / "picture.bin").write_bytes(bytes(range(256)) * 16)
    (work / ".gitignore").write_text("build/\n*.tmp\n", encoding="utf-8")
    run(work, "add", ".")
    run(work, "commit", "-q", "-m", "first")
    (work / "src" / "law.py").write_text("a = 1\nb = 2\nc = 3\nd = 4\n", encoding="utf-8")
    run(work, "commit", "-q", "-am", "second")
    (work / "src" / "kernel.glsl").write_text("void main() { }\n", encoding="utf-8")          # changed, not committed
    (work / "notes.txt").write_text("left here\n", encoding="utf-8")                          # untracked
    (work / "build").mkdir()
    (work / "build" / "out.o").write_bytes(b"x" * 10)                                         # inside an ignored folder
    (work / "scratch.tmp").write_bytes(b"y" * 7)                                              # an ignored file
    policy = {**branch_gate.DEFAULTS, "repositories": {"work": {"path": str(work), "default": "main"}}}
    return {"work": work, "policy": policy, "db": tmp_path / "file_index.db"}


def rows(core, **filters):
    db = file_index.connect(core["db"])
    try:
        return file_index.search(db, **filters)
    finally:
        db.close()


def test_only_the_cleanup_account_builds_the_index(core):
    for actor in (None, "admin", "svc-task-worker"):
        with pytest.raises(file_index.IndexError_) as refused:
            file_index.build(core["policy"], core["db"], actor=actor, log=lambda line: None)
        assert refused.value.status == 403 and CLEANER in refused.value.message
    made = file_index.build(core["policy"], core["db"], actor=CLEANER, log=lambda line: None)
    assert made["indexed_by"] == CLEANER and made["repositories"] == {"work": 7} and made["entries"] == 7


def test_one_row_per_file_with_every_metric(core):
    file_index.build(core["policy"], core["db"], actor=CLEANER, log=lambda line: None)
    found = {row["path"]: row for row in rows(core)["files"]}
    assert set(found) == {".gitignore", "picture.bin", "src/law.py", "src/kernel.glsl", "notes.txt", "build", "scratch.tmp"}
    law = found["src/law.py"]
    assert (law["status"], law["ext"], law["dir"], law["depth"], law["name"]) == ("tracked", "py", "src", 1, "law.py")
    assert law["lines"] == 4 and law["binary"] == 0 and law["commits"] == 2 and len(law["blob"]) == 40
    assert law["first_commit_at"] <= law["last_commit_at"] and len(law["last_commit"]) == 40 and law["bytes"] == 28
    assert found["src/kernel.glsl"]["status"] == "modified" and found["src/kernel.glsl"]["commits"] == 1
    assert found["picture.bin"]["binary"] == 1 and found["picture.bin"]["lines"] is None and found["picture.bin"]["bytes"] == 4096
    assert found["notes.txt"]["status"] == "untracked" and found["notes.txt"]["commits"] is None and found["notes.txt"]["lines"] == 1
    folder = found["build"]
    assert folder["status"] == "ignored" and folder["is_dir"] == 1 and folder["bytes"] is None, "an ignored folder is one row"
    assert found["scratch.tmp"]["status"] == "ignored" and found["scratch.tmp"]["bytes"] == 7


def test_the_flat_set_is_filtered_and_ordered(core):
    file_index.build(core["policy"], core["db"], actor=CLEANER, log=lambda line: None)
    assert [r["path"] for r in rows(core, q="law")["files"]] == ["src/law.py"]
    assert [r["path"] for r in rows(core, ext=".GLSL")["files"]] == ["src/kernel.glsl"]
    assert {r["path"] for r in rows(core, under="src")["files"]} == {"src/law.py", "src/kernel.glsl"}
    assert [r["path"] for r in rows(core, status="untracked")["files"]] == ["notes.txt"]
    assert [r["path"] for r in rows(core, binary="true")["files"]] == ["picture.bin"]
    largest = rows(core, order="-bytes", limit=2)
    assert [r["path"] for r in largest["files"]] == ["picture.bin", "src/law.py"] and largest["total"] == 7 and largest["next_offset"] == 2
    assert [r["path"] for r in rows(core, min_bytes=4000)["files"]] == ["picture.bin"]
    assert rows(core, q="100%_")["total"] == 0, "a search word is text, never a pattern"
    for bad in ({"order": "path; DROP TABLE files"}, {"status": "weird"}, {"min_bytes": "many"}):
        with pytest.raises(file_index.IndexError_):
            rows(core, **bad)
    db = file_index.connect(core["db"])
    summary = file_index.summary(db)
    db.close()
    assert summary["last"]["indexed_by"] == CLEANER
    assert {(s["status"], s["entries"]) for s in summary["by_status"]} == {("tracked", 3), ("modified", 1), ("untracked", 1), ("ignored", 2)}


def test_a_second_build_follows_the_core(core, monkeypatch):
    file_index.build(core["policy"], core["db"], actor=CLEANER, log=lambda line: None)
    counted = []
    real = file_index._count_lines
    monkeypatch.setattr(file_index, "_count_lines", lambda file, size: counted.append(Path(file).name) or real(file, size))
    (core["work"] / "notes.txt").unlink()
    (core["work"] / "new.md").write_text("one\ntwo\n", encoding="utf-8")
    file_index.build(core["policy"], core["db"], actor=CLEANER, log=lambda line: None)
    found = {row["path"]: row for row in rows(core)["files"]}
    assert "notes.txt" not in found and found["new.md"]["lines"] == 2
    assert counted == ["new.md"], "an unchanged file keeps its counted lines"


def test_the_index_over_rest_is_read_only_and_needs_an_account(core):
    def ask(path, user=7, capabilities=("dev_ui",)):
        sent = []
        handler = SimpleNamespace(path=path, _principal=lambda: SimpleNamespace(user_id=user), _caps=lambda: set(capabilities),
                                  _json=lambda document, status=200: sent.append((int(status), json.loads(json.dumps(document)))))
        file_index.http_route(handler, path.split("?")[0], db_path=core["db"])
        return sent[0]

    assert ask("/api/files/search")[0] == 503, "not built yet"
    file_index.build(core["policy"], core["db"], actor=CLEANER, log=lambda line: None)
    assert ask("/api/files/search", user=None)[0] == 401 and ask("/api/files/search", capabilities=())[0] == 403
    status, found = ask("/api/files/search?ext=py&order=-bytes")
    assert status == 200 and [r["path"] for r in found["files"]] == ["src/law.py"]
    assert ask("/api/files/summary")[1]["last"]["entries"] == 7
    assert ask("/api/files/search?order=nothing")[0] == 400 and ask("/api/files/search?drop=1")[0] == 400
    assert ask("/api/files/elsewhere")[0] == 404
