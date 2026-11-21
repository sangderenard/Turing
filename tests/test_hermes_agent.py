"""[HERMES] generic launcher: prepare output shape and process-tree kill at timeout."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "examples" / "meowdoku" / "deploy")]

import hermes_agent as A  # noqa: E402


def _prepare(tmp_path, account="fake"):
    package = tmp_path / "mem" / "speaktome_hermes"
    package.mkdir(parents=True, exist_ok=True)
    argv = ["prepare", "--account", account, "--provider", "custom", "--model", "m-1", "--base-url", "http://127.0.0.1:1/v1",
            "--hermes-root", str(tmp_path / "h"), "--speaktome-hermes", str(package),
            "--secrets-dir", str(tmp_path / "sec"), "--accounts-db", str(tmp_path / "none.db"),
            "--mcp-python", "PY", "--mcp-api", "API"]
    assert A.main(argv) == 0
    return tmp_path / "h" / account, package


def test_prepare_shape_and_regeneration(tmp_path):
    _prepare(tmp_path)
    home, package = _prepare(tmp_path)  # second run replaces the link and files
    config = (home / "config.yaml").read_text(encoding="utf-8")
    assert "model:\n  provider: custom\n  base_url: http://127.0.0.1:1/v1\n  default: m-1\n" in config
    assert "custom_providers:\n  - name: local-llama\n    base_url: http://127.0.0.1:1/v1\n    model: m-1\n" in config
    assert "memory:\n  provider: speaktome\nplatform_toolsets:\n  cli:\n    - memory\n    - todo\n" in config
    assert "\ntoolsets:" not in config  # Hermes CLI ignores a top-level toolsets key (review 2058da11)
    assert "      SPEAKTOME_USER: fake\n" in config and str(tmp_path / "sec" / "fake.secret") in config
    assert "include: [whoami, where, look, enter, go, leave, examine, man, say, pending, acknowledge, contacts]" in config
    for banned in ("terminal", "file", "web", "browser", "shell", "code_execution"):
        assert f"  - {banned}" not in config
    assert A.GENERATED in config
    env = dict(line.split("=", 1) for line in (home / ".env").read_text(encoding="utf-8").splitlines() if "=" in line)
    assert env["SPEAKTOME_USER"] == "fake" and env["SPEAKTOME_ROOT"] == str(package.resolve().parent)
    assert env["SPEAKTOME_SECRET_FILE"].endswith("fake.secret") and env["SPEAKTOME_BASE"].startswith("https://")
    assert (home / "plugins" / "speaktome").resolve() == package.resolve()
    assert not (home / "SOUL.md").exists()  # no accounts db: warned, continued


@pytest.mark.skipif(not (A.DEFAULT_PROGRAM / ("Scripts" if os.name == "nt" else "bin")).is_dir(), reason="needs the installed Hermes")
def test_hermes_itself_enables_only_memory_and_todo(tmp_path):
    """Ask the installed Hermes (not our own reading of the config) which CLI toolsets a prepared home enables."""
    home, _ = _prepare(tmp_path)
    probe = ("import sys, yaml; from hermes_cli.tools_config import _get_platform_tools; "
             "cfg = yaml.safe_load(open(sys.argv[1], encoding='utf-8')); "
             "print(','.join(sorted(_get_platform_tools(cfg, 'cli', include_default_mcp_servers=False))))")
    python = A.DEFAULT_PROGRAM / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    done = subprocess.run([str(python), "-c", probe, str(home / "config.yaml")], capture_output=True, text=True,
                          env=dict(os.environ, HERMES_HOME=str(home)), cwd=str(home), timeout=120)
    assert done.returncode == 0, done.stderr
    enabled = set(done.stdout.strip().split(","))
    # Hermes 0.19 adds its own `kanban` plugin toolset on its own; its tools register only in kanban mode
    # (check_fn _check_kanban_mode returns False otherwise), so it is the one known extra.
    assert enabled - {"kanban"} == set(A.TOOLSETS), enabled
    assert not enabled & set(A.FORBIDDEN_TOOLSETS), enabled


def test_run_kills_orphan_children(tmp_path):
    _prepare(tmp_path)
    marker = tmp_path / "child.pid"
    (tmp_path / "fake.py").write_text(
        "import subprocess,sys,time\n"
        "c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(300)'])\n"
        f"open({str(marker)!r},'w').write(str(c.pid))\n"
        "time.sleep(300)\n", encoding="utf-8")
    if os.name == "nt":
        fake = tmp_path / "hermes.cmd"
        fake.write_text(f'@"{sys.executable}" "{tmp_path / "fake.py"}" %*\n', encoding="utf-8")
    else:
        fake = tmp_path / "hermes"
        fake.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{tmp_path / "fake.py"}" "$@"\n', encoding="utf-8")
        fake.chmod(0o755)
    started = time.time()
    code = A.main(["run", "--account", "fake", "--hermes-root", str(tmp_path / "h"), "--prompt", "hi",
                   "--timeout-seconds", "4", "--hermes-exe", str(fake)])
    assert code == A.TIMEOUT_EXIT and time.time() - started < 30
    pid = int(marker.read_text())

    def alive():
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
            return str(pid) in out
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    for _ in range(20):
        if not alive():
            break
        time.sleep(0.25)
    assert not alive()


# ---- the sleep cycle ----
def _fake_hermes(tmp_path, hang_unless_wind_down=False):
    """A fake hermes that logs its argv (one JSON line) and optionally hangs unless it is the wind-down turn."""
    log = tmp_path / "args.log"
    (tmp_path / "fake_cycle.py").write_text(
        "import json,sys,time\n"
        f"open({str(log)!r},'a',encoding='utf-8').write(json.dumps(sys.argv[1:])+chr(10))\n"
        + ("if not sys.argv[3].startswith('You will sleep soon'):\n    time.sleep(300)\n" if hang_unless_wind_down else ""),
        encoding="utf-8")
    if os.name == "nt":
        fake = tmp_path / "hermes_cycle.cmd"
        fake.write_text(f'@"{sys.executable}" "{tmp_path / "fake_cycle.py"}" %*\n', encoding="utf-8")
    else:
        fake = tmp_path / "hermes_cycle"
        fake.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{tmp_path / "fake_cycle.py"}" "$@"\n', encoding="utf-8")
        fake.chmod(0o755)
    return fake, log


def _calls(log):
    import json
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def _cycle_argv(tmp_path, fake, *extra):
    return ["cycle", "--account", "fake", "--hermes-root", str(tmp_path / "h"), "--hermes-exe", str(fake),
            "--work-seconds", "6", "--wind-down-seconds", "3", "--min-rest", "10", "--max-rest", "100", "--poll-seconds", "5",
            *extra]


def test_rest_seconds_proportional_and_clamped():
    assert A.rest_seconds(100, 1.0, 60, 3600) == 100
    assert A.rest_seconds(100, 2.5, 60, 3600) == 250
    assert A.rest_seconds(10, 1.0, 60, 3600) == 60       # little work: the minimum
    assert A.rest_seconds(5000, 1.0, 60, 3600) == 3600   # much work: the maximum
    assert A.rest_seconds(0, 0.0, 0, 5) == 0


def test_second_cycle_refused_while_lock_is_held(tmp_path, capsys):
    home, _ = _prepare(tmp_path)
    fake, log = _fake_hermes(tmp_path)
    held = A.acquire_lock(home / "cycle.lock")
    assert held is not None and A.acquire_lock(home / "cycle.lock") is None
    try:
        assert A.main(_cycle_argv(tmp_path, fake, "--cycles", "1"), pending_check=lambda: (0, None)) == 0
    finally:
        A.release_lock(held)
    assert "already running" in capsys.readouterr().out and not log.exists()  # nothing was started
    again = A.acquire_lock(home / "cycle.lock")  # and the lock is free once released
    assert again is not None
    A.release_lock(again)


def test_work_limit_kills_turn_and_wind_down_resumes_same_session(tmp_path):
    home, _ = _prepare(tmp_path)
    fake, log = _fake_hermes(tmp_path, hang_unless_wind_down=True)
    started = time.time()
    code = A.main(_cycle_argv(tmp_path, fake, "--cycles", "1"), session_probe=lambda *_: ("sess-1", 4),
                  pending_check=lambda: (0, None))
    assert code == 0 and time.time() - started < 40
    work, wind = _calls(log)
    assert work[:2] == ["chat", "-q"] and "--resume" not in work           # first wake: a new session
    assert wind[2].startswith("You will sleep soon") and wind[-2:] == ["--resume", "sess-1"]
    state = __import__("json").loads((home / "cycle_state.json").read_text(encoding="utf-8"))
    assert state["session_id"] == "sess-1" and state["cycles"] == 1 and state["tool_calls"] == 4
    assert state["last_exit_codes"] == {"work": A.TIMEOUT_EXIT, "wind_down": 0}
    assert 10 <= state["rest_seconds"] <= 100 and state["work_seconds"] >= 3
    lines = (home / "cycle.log").read_text(encoding="utf-8").splitlines()
    assert [word.split()[1] for word in lines] == ["wake", "wind-down", "rest"]


def test_session_id_persists_across_cycles(tmp_path):
    home, _ = _prepare(tmp_path)
    fake, log = _fake_hermes(tmp_path)
    sleeps = []
    totals = iter([3, 3, 9, 9])  # tool_call_count after each probe: cycle 1 work+wind, cycle 2 work+wind
    now = [0.0]

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
    assert A.main(_cycle_argv(tmp_path, fake, "--cycles", "2"), session_probe=lambda *_: ("sess-7", next(totals)),
                  pending_check=lambda: (0, None), sleep=sleep, clock=lambda: now[0]) == 0
    first_work, first_wind, second_work, second_wind = _calls(log)
    assert "--resume" not in first_work and first_wind[-2:] == ["--resume", "sess-7"]
    assert second_work[2].startswith("You are awake again") and second_work[-2:] == ["--resume", "sess-7"]
    assert second_wind[-2:] == ["--resume", "sess-7"]
    state = __import__("json").loads((home / "cycle_state.json").read_text(encoding="utf-8"))
    assert state["session_id"] == "sess-7" and state["cycles"] == 2 and state["tool_calls"] == 6  # delta of the 2nd wake
    assert sum(sleeps) >= 10 and max(sleeps) <= 5  # rested in poll-sized slices between the cycles


def test_rest_wakes_early_on_new_message_and_survives_check_failure():
    now = [0.0]
    answers = iter([(1, 5), RuntimeError("down"), (1, 5), (2, 6), (2, 6)])
    logged = []

    def check():
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def sleep(seconds):
        now[0] += seconds
    outcome = A.rest(1000, 15, check, sleep, lambda: now[0], logged.append)
    assert outcome == "early" and now[0] == 45                    # 3 sleeps, then the 4th check saw message 6
    assert len(logged) == 1 and "pending check failed: RuntimeError: down" in logged[0]
    now[0] = 0.0  # nothing new: the full rest, one check per poll, never a tight loop
    assert A.rest(40, 15, lambda: (1, 5), sleep, lambda: now[0], logged.append) == "rested" and now[0] == 40


def test_cycle_ends_rest_early_when_pending_reports_message(tmp_path):
    _prepare(tmp_path)
    fake, log = _fake_hermes(tmp_path)
    seen = iter([(0, None), (1, 9)])
    assert A.main(_cycle_argv(tmp_path, fake, "--cycles", "2"), session_probe=lambda *_: ("s", 1),
                  pending_check=lambda: next(seen), sleep=lambda s: None, clock=iter(range(0, 1000, 1)).__next__) == 0
    assert len(_calls(log)) == 4  # the second wake started at once, after one poll slice
    assert "woke early" in (tmp_path / "h" / "fake" / "cycle.log").read_text(encoding="utf-8")


# ---- serve: children tied to a job, port check, order, llama-server always killed ----
class _FakeServer:
    returncode = None

    def poll(self):
        return None


def _serve_args(tmp_path, *extra):
    model = tmp_path / "m.gguf"
    exe = tmp_path / "llama-server.exe"
    model.write_bytes(b"x")
    exe.write_bytes(b"x")
    return A.build_parser().parse_args(["serve", "--account", "fake", "--hermes-root", str(tmp_path / "h"), "--llama-server", str(exe),
                                        "--model-file", str(model), "--alias", "m-1", "--state-dir", str(tmp_path / "s"), *extra])


def test_serve_ties_children_starts_server_prepares_cycles_and_kills_server(tmp_path, monkeypatch):
    events, popen_calls = [], []
    monkeypatch.setattr(A, "prepare", lambda a: events.append(("prepare", a.base_url, a.model)) or 0)
    monkeypatch.setattr(A, "cycle", lambda a: events.append(("cycle", a.account)) or 0)
    monkeypatch.setattr(A, "kill_tree", lambda p: events.append("kill"))

    def popen(command, **kw):
        popen_calls.append(command)
        events.append("popen")
        return _FakeServer()

    code = A.serve(_serve_args(tmp_path), tie=lambda: events.append("tie") or "children are tied", listening=lambda port: False,
                   healthy=lambda port: True, popen=popen)
    assert code == 0
    assert events == ["tie", "popen", ("prepare", "http://127.0.0.1:11436/v1", "m-1"), ("cycle", "fake"), "kill"]
    assert "--load-mode" not in popen_calls[0] and popen_calls[0][popen_calls[0].index("--port") + 1] == "11436"


def test_serve_refuses_a_listening_port_and_kills_server_when_cycle_raises(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="already listening"):
        A.serve(_serve_args(tmp_path), tie=lambda: "children are tied", listening=lambda port: True,
                popen=lambda *a, **k: pytest.fail("must not start"))
    killed = []
    monkeypatch.setattr(A, "prepare", lambda a: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(A, "kill_tree", lambda p: killed.append(p))
    with pytest.raises(RuntimeError):
        A.serve(_serve_args(tmp_path), tie=lambda: "children are tied", listening=lambda port: False, healthy=lambda port: True,
                popen=lambda *a, **k: _FakeServer())
    assert len(killed) == 1


def test_serve_stops_when_the_job_cannot_be_made(tmp_path):
    if os.name != "nt":
        pytest.skip("kill-on-close job is Windows")
    with pytest.raises(SystemExit, match="NOT tied"):
        A.serve(_serve_args(tmp_path), tie=lambda: "children are NOT tied to the supervisor", listening=lambda p: False)
