"""Git in the core, regulated: against a real temporary repository and a real SQLite database."""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from meowdoku import authz, core_git  # noqa: E402

ANN = 2


def git(path, *args):
    done = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


@pytest.fixture
def world(tmp_path):
    work = tmp_path / "core" / "work"
    work.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(work)], check=True)
    for key, value in (("user.name", "t"), ("user.email", "t@example.invalid"), ("commit.gpgsign", "false")):
        git(work, "config", key, value)
    (work / "src").mkdir()
    (work / "src" / "law.py").write_text("a = 1\n", encoding="utf-8")
    git(work, "add", ".")
    git(work, "commit", "-q", "-m", "first")
    first = git(work, "rev-parse", "HEAD")
    (work / "src" / "law.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
    git(work, "commit", "-q", "-am", "second")
    db = sqlite3.connect(tmp_path / "accounts.db", isolation_level=None)
    policy = {"repositories": {"work": {"path": str(work), "default": "main"}}}
    return {"work": work, "db": db, "policy": policy, "first": first, "db_file": tmp_path / "accounts.db", "tmp": tmp_path}


def call(world, operation, **arguments):
    return core_git.run(world["db"], world["policy"], user_id=ANN, username="ann", repository="work",
                        operation=operation, arguments=arguments)


def test_declared_operations_read_the_repository(world):
    assert call(world, "status")["output"].startswith("## main")
    assert "refs/heads/main" in call(world, "branches")["output"]
    log = call(world, "log", count=5)["output"].splitlines()
    assert len(log) == 2 and log[0].split("\t")[3] == "second"
    assert len(call(world, "log", path="src/law.py", count=1)["output"].splitlines()) == 1
    assert "first" in call(world, "commit", commit=world["first"])["output"]
    assert "+b = 2" in call(world, "diff", base=world["first"], head="main")["output"]
    assert call(world, "file", ref="main", path="src/law.py")["output"] == "a = 1\nb = 2\n"
    assert call(world, "file", ref=world["first"], path="src/law.py")["output"] == "a = 1\n"
    assert call(world, "tree", ref="main")["output"].split() == ["src/law.py"]


def test_a_caller_never_passes_a_flag_a_config_or_a_way_out(world):
    refused = [
        ("log", {"ref": "--output=x"}), ("log", {"ref": "main..other"}), ("log", {"count": 100000}),
        ("log", {"extra": 1}), ("file", {"path": "../outside"}), ("file", {"path": "/etc/passwd"}),
        ("file", {"path": "-p"}), ("file", {"path": ""}), ("file", {"ref": "-c core.pager=x", "path": "src/law.py"}),
        ("diff", {"base": "main", "head": "--no-index"}), ("commit", {"commit": "main --"}),
        ("tree", {"path": "src/../.."}), ("status", {"anything": 1}),
    ]
    before = world["db"].execute("SELECT COUNT(*) FROM sqlite_master WHERE name = 'git_calls'").fetchone()[0]
    for operation, arguments in refused:
        with pytest.raises(core_git.GitError):
            call(world, operation, **arguments)
    for operation in ("push", "config", "fetch", "commit --amend", ""):
        with pytest.raises(core_git.GitError) as unknown:
            call(world, operation)
        assert unknown.value.status == 404
    with pytest.raises(core_git.GitError) as elsewhere:
        core_git.run(world["db"], world["policy"], user_id=ANN, username="ann", repository="other", operation="status")
    assert elsewhere.value.status == 404
    for user in (None, True, "2"):
        with pytest.raises(core_git.GitError):
            core_git.run(world["db"], world["policy"], user_id=user, username="x", repository="work", operation="status")
    assert before == 0 or world["db"].execute("SELECT COUNT(*) FROM git_calls").fetchone()[0] == 0, "a refused call ran nothing"
    assert git(world["work"], "status", "--porcelain") == "", "nothing was written to the repository"


def test_every_call_is_recorded_and_a_missing_thing_is_a_plain_refusal(world):
    call(world, "status")
    with pytest.raises(core_git.GitError) as missing:
        call(world, "file", ref="main", path="src/nothing.py")
    assert missing.value.status == 404
    rows = core_git.recent(world["db"])
    assert [(r["username"], r["repository"], r["operation"]) for r in rows] == [("ann", "work", "file"), ("ann", "work", "status")]
    assert rows[0]["arguments"] == {"path": "src/nothing.py", "ref": "main"} and rows[0]["exit_code"] != 0
    assert rows[1]["exit_code"] == 0 and rows[1]["bytes"] > 0


def test_large_output_is_cut_and_said_so(world, monkeypatch):
    monkeypatch.setattr(core_git, "MAX_OUTPUT", 4)
    result = call(world, "file", ref="main", path="src/law.py")
    assert result["truncated"] is True and result["output"] == "a = " and result["bytes"] == 4


def test_a_rows_program_never_runs_git_or_inside_the_core(world):
    policy, work, outside = world["policy"], world["work"], world["tmp"] / "scratch"
    outside.mkdir()
    assert core_git.inside_core(policy, work) == "work" and core_git.inside_core(policy, work / "src") == "work"
    assert core_git.inside_core(policy, outside) is None
    assert core_git.exec_refusal(policy, "C:/tools/zstd.exe", outside) is None
    assert "inside the core" in core_git.exec_refusal(policy, "C:/tools/zstd.exe", work / "src")
    for program in ("git", "C:/Program Files/Git/cmd/git.exe", "GIT.EXE", "git-upload-pack"):
        assert "never runs Git" in core_git.exec_refusal(policy, program, outside)


def test_no_service_account_and_no_tier_holds_the_capability():
    assert "core_git" in authz.CAPS
    roles, tiers = authz.DEFAULT_POLICY["role_caps"], authz.DEFAULT_POLICY["tier_caps"]
    assert "core_git" in roles["admin"] and "core_git" in roles["operator"]
    assert "core_git" not in roles["service"] and "core_git" not in roles["member"] and "core_git" not in roles["guest"]
    assert not any("core_git" in caps for caps in tiers.values()), "being on the LAN grants nothing here"


def test_over_rest_the_caller_comes_from_the_session(world):
    def ask(method, path, user=ANN, capabilities=("core_git",), payload=None):
        sent = []
        handler = SimpleNamespace(path=path, headers={"Content-Type": "application/json"},
                                  _principal=lambda: SimpleNamespace(user_id=user, username="ann"),
                                  _caps=lambda: set(capabilities), _body_json=lambda limit: payload,
                                  _json=lambda document, status=200: sent.append((int(status), json.loads(json.dumps(document)))))
        core_git.http_route(handler, path, method, policy=world["policy"], accounts_db=world["db_file"])
        return sent[0]

    assert ask("GET", "/api/git/repositories", user=None)[0] == 401
    assert ask("GET", "/api/git/repositories", capabilities=("dev_ui",))[0] == 403
    assert ask("GET", "/api/git/repositories")[1]["repositories"] == {"work": "main"}
    status, result = ask("POST", "/api/git/file", payload={"repository": "work", "ref": "main", "path": "src/law.py",
                                                           })
    assert status == 200 and result["output"] == "a = 1\nb = 2\n"
    assert ask("POST", "/api/git/push", payload={"repository": "work"})[0] == 404
    assert ask("POST", "/api/git/log", payload={"repository": "work", "ref": "--all"})[0] == 400
    assert ask("POST", "/api/git/status", payload={"repository": "work", "user_id": 1})[0] == 400, "identity is not an argument"
    assert ask("GET", "/api/git/calls")[1]["calls"][0]["username"] == "ann"
