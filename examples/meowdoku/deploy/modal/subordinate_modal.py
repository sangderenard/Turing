"""Temporary Modal subordinate (ticket cc439864): the existing x64-linux Nodus build, dialing home over the public link.

Not the real subordinate yet: no per-node keys, contracts or sealing exist. It carries one short-lived link ticket pair
issued at home (passed as arguments) and test bytes only. What it measures: image build, cold start, time to link, and a
bulk send's arrival at home.

Run (from this folder):  modal run subordinate_modal.py --tickets-file tickets.json
"""
from __future__ import annotations

import json
import pathlib

import modal

HERE = pathlib.Path(__file__).resolve().parent

image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    # The build's system libraries (ldd of libnodus_headless / tensor_core / runtime on Ubuntu 24.04).
    .apt_install("libpython3.12t64", "libssl3t64", "libvulkan1", "libexpat1", "zlib1g", "ca-certificates")
    .pip_install("pyyaml")          # nodus_graph/spirv.py; the only import outside the standard library
    .add_local_file(HERE / "nodus-x64-linux.tar.gz", "/opt/nodus.tar.gz", copy=True)
    .run_commands("mkdir -p /opt/nodus && tar -xzf /opt/nodus.tar.gz -C /opt/nodus && rm /opt/nodus.tar.gz")
    .env({"NODUS_HEADLESS_LIB": "/opt/nodus/lib/libnodus_headless.so", "LD_LIBRARY_PATH": "/opt/nodus/lib",
          "PYTHONPATH": "/opt/nodus"})
)

app = modal.App("nodus-subordinate-test", image=image)


@app.function(timeout=600, cpu=1.0, memory=1024)
def subordinate(tickets: dict, host: str, port: int, payload_bytes: int) -> dict:
    import secrets
    import socket
    import subprocess
    import sys
    import time

    t0 = time.monotonic()
    sys.path.insert(0, "/opt/nodus")
    from nodus_graph import Graph

    canvas = pathlib.Path("/tmp/subordinate.nodus.txt")
    with Graph() as graph:
        graph.add_module("start")
        graph.save(canvas)
    token = secrets.token_urlsafe(24)
    pathlib.Path("/tmp/console.token").write_text(token)
    proc = subprocess.Popen([sys.executable, "/opt/nodus/tools/run_canvas.py", "--canvas", str(canvas), "--rate", "200",
                             "--console-port", "0", "--console-token-file", "/tmp/console.token"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    first = json.loads(proc.stdout.readline())
    started = time.monotonic() - t0
    import threading
    stderr_lines: list[str] = []
    threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()       # keep the pipes drained
    threading.Thread(target=lambda: [stderr_lines.append(l) for l in proc.stderr], daemon=True).start()

    sock = socket.create_connection(("127.0.0.1", first["console_port"]), timeout=10.0)
    f = sock.makefile("rb")
    sock.sendall(f"HELLO nodus-console/2 {token}\n".encode())
    assert f.readline().decode().startswith("OK")

    def action(**request):
        sock.sendall(b"SNAPSHOT\n")
        head = f.readline().decode().split()
        f.read(int(head[2]))
        f.readline()
        revision = json.loads(f.readline().decode().split(" ", 1)[1])["revision"]
        body = json.dumps(dict(request, expected_revision=revision)).encode()
        sock.sendall(f"ACTION {len(body)}\n".encode() + body)
        head = f.readline().decode().split()
        return json.loads(f.read(int(head[2])))

    answer = action(op="link_dial", id="home", host=host, port=port, prefix=tickets["prefix"],
                    control_ticket=tickets["control_ticket"], bulk_ticket=tickets["bulk_ticket"], secure=True)
    report, linked_at = None, None
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        report = action(op="link_report", id="home").get("link")
        if report and report.get("state") == "linked":
            linked_at = time.monotonic() - t0
            break
        time.sleep(0.2)
    sent = None
    if linked_at is not None:
        sent = action(op="link_send", id="home", bytes=payload_bytes, fill=7)
        time.sleep(3.0)
        report = action(op="link_report", id="home").get("link")
    sock.close()
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    return {"dial": answer, "started_s": round(started, 3), "linked_s": linked_at and round(linked_at, 3),
            "sent": sent, "report": report, "stderr_tail": "".join(stderr_lines[-15:])[-2000:]}


@app.local_entrypoint()
def main(tickets_file: str, host: str = "stichka.duckdns.org", port: int = 8766, payload_bytes: int = 1 << 20):
    import time
    tickets = json.loads(pathlib.Path(tickets_file).read_text())
    t = time.monotonic()
    result = subordinate.remote(tickets, host, port, payload_bytes)
    result["round_trip_s"] = round(time.monotonic() - t, 3)
    print(json.dumps(result, indent=2))
