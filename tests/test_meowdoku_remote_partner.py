"""[REMOTE] Loopback is the default, never the only option: a server may believe one more named proxy, and nothing else."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import proxy_trust  # noqa: E402

SERVER = ROOT / "examples" / "meowdoku" / "server.py"


def _run(tmp_path, *args):
    return subprocess.run([sys.executable, str(SERVER), "--web-dir", str(tmp_path / "web"), *args],
                          capture_output=True, text=True, timeout=120)


def test_a_trusted_proxy_must_be_an_address(tmp_path):
    done = _run(tmp_path, "--trusted-proxy", "dev.example")
    assert done.returncode == 2 and "--trusted-proxy expects an IP address" in done.stderr
    done = _run(tmp_path, "--trusted-proxy", "192.168.1.232")
    assert done.returncode == 1 and "no export" in done.stdout, "a valid address passes on to the next check"


def test_a_named_proxy_is_believed_and_any_other_is_not():
    saved = set(proxy_trust.TRUSTED_PROXIES)
    try:
        headers = {"X-Forwarded-For": "203.0.113.9", "X-Forwarded-Proto": "https"}
        assert proxy_trust.client_origin("192.168.1.232", headers)[0] == "192.168.1.232", "by default a network proxy is the client"
        proxy_trust.TRUSTED_PROXIES.add("192.168.1.232")                 # what --trusted-proxy 192.168.1.232 does
        assert proxy_trust.client_origin("192.168.1.232", headers)[:2] == ("203.0.113.9", "https")
        assert proxy_trust.client_origin("192.168.1.99", headers)[0] == "192.168.1.99", "only the named proxy"
        assert proxy_trust.client_origin("127.0.0.1", headers)[0] == "203.0.113.9", "loopback stays trusted"
    finally:
        proxy_trust.TRUSTED_PROXIES.clear()
        proxy_trust.TRUSTED_PROXIES.update(saved)
