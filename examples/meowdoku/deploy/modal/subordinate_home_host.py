"""Home side of the temporary Modal subordinate test (ticket cc439864).

Starts a hosting canvas (tools/run_canvas.py, Windows build-service nodus_headless.dll) that accepts links on
127.0.0.1:47616, the loopback port Apache's /ws/link/<prefix>/ route proxies to; issues one short-lived ticket pair for the
Modal side from srv/state/secrets/nodus_link.key into tickets.json; then reports the link every few seconds until stopped
or until --seconds pass.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import secrets
import signal
import socket
import subprocess
import sys
import time

WS = pathlib.Path("C:/dev/Powershell")
NODUS = WS / "nodus"
HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(NODUS), str(WS / "turing" / "examples")]
os.environ.setdefault("NODUS_HEADLESS_LIB", str(NODUS / "build-service" / "Release" / "nodus_headless.dll"))

from nodus_graph import Graph  # noqa: E402
from meowdoku import link_tickets  # noqa: E402


class Console:
    def __init__(self, port, token):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=10.0)
        self.f = self.sock.makefile("rb")
        self.sock.sendall(f"HELLO nodus-console/2 {token}\n".encode())
        assert self.f.readline().decode().startswith("OK")

    def action(self, **request):
        self.sock.sendall(b"SNAPSHOT\n")
        head = self.f.readline().decode().split()
        self.f.read(int(head[2]))
        self.f.readline()
        revision = json.loads(self.f.readline().decode().split(" ", 1)[1])["revision"]
        body = json.dumps(dict(request, expected_revision=revision)).encode()
        self.sock.sendall(f"ACTION {len(body)}\n".encode() + body)
        head = self.f.readline().decode().split()
        return json.loads(self.f.read(int(head[2])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="modaltest")
    ap.add_argument("--port", type=int, default=47616)
    ap.add_argument("--account", type=int, default=17)          # operator-claude
    ap.add_argument("--life", type=int, default=300)            # ticket life, seconds (link_tickets.MAX_LIFE)
    ap.add_argument("--seconds", type=int, default=600)
    args = ap.parse_args()

    work = HERE / "home-run"
    work.mkdir(exist_ok=True)
    canvas = work / "home_host.nodus.txt"
    with Graph() as graph:
        graph.add_module("start")
        graph.save(canvas)
    token = secrets.token_urlsafe(24)
    (work / "console.token").write_text(token, encoding="utf-8")
    err = open(work / "run_canvas.stderr.txt", "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(NODUS / "tools" / "run_canvas.py"), "--canvas", str(canvas), "--rate", "200",
                             "--console-port", "0", "--console-token-file", str(work / "console.token")],
                            stdout=subprocess.PIPE, stderr=err, text=True,
                            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    try:
        console = Console(json.loads(proc.stdout.readline())["console_port"], token)
        import threading
        threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()   # keep the pipe drained
        key_file = WS / "srv" / "state" / "secrets" / "nodus_link.key"
        print("link_host:", console.action(op="link_host", id="subordinate", prefix=args.prefix, port=args.port,
                                           key_file=str(key_file)), flush=True)
        tickets = link_tickets.issue(key_file.read_bytes(), args.prefix, args.account, life=args.life)
        tickets["prefix"] = args.prefix
        (HERE / "tickets.json").write_text(json.dumps(tickets), encoding="utf-8")
        print(f"tickets.json written (expires in {args.life} s)", flush=True)
        end = time.monotonic() + args.seconds
        last = None
        while time.monotonic() < end and proc.poll() is None:
            report = console.action(op="link_report", id="subordinate").get("link")
            line = json.dumps(report, sort_keys=True)
            if line != last:
                print(time.strftime("%H:%M:%S"), line, flush=True)
                last = line
            time.sleep(1.0)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.CTRL_BREAK_EVENT) if sys.platform == "win32" else proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        err.close()
        (HERE / "tickets.json").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
