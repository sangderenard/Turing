"""[HERMES] generic launcher: prepare output shape and process-tree kill at timeout."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

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
    assert "memory:\n  provider: speaktome\ntoolsets:\n  - memory\n  - todo\n  - mcp-speaktome\n" in config
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
