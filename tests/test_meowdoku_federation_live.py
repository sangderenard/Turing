"""Federation live shape: the /ws/contract/<prefix> bridge in front of a REAL Nodus canvas host (nodus/tools/run_canvas.py).

No stub port: the messenger canvas is built and hosted by the Nodus tools and DLL; one contract (Submit -> Admitted -> Offer/Accept ->
Result -> Route -> Decision -> Receipt) is driven by an engine peer and a host peer that both enter through the Meowdoku bridge.
Everything is on ephemeral ports and a temp state root; skipped (not faked) when the Nodus DLL or tools are missing."""
from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import authz  # noqa: E402
from meowdoku import federation_peer as fp  # noqa: E402
from meowdoku import server as srv  # noqa: E402

NODUS = ROOT.parent / "nodus"
DLL = NODUS / "build-service" / "Release" / "nodus_headless.dll"
RUN_CANVAS = NODUS / "tools" / "run_canvas.py"
PEER_HELPER = NODUS / "tools" / "contract_service_peer.py"
CANVAS_TEST = NODUS / "tests" / "test_run_canvas.py"
CAPACITY = 4

_SAVE = ("import importlib.util, sys\n"
         "spec = importlib.util.spec_from_file_location('nodus_test_run_canvas', sys.argv[1])\n"
         "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
         "from pathlib import Path\nmod.save_messenger_canvas(Path(sys.argv[2])); print(mod.PREFIX)\n")


def _nodus_env():
    env = dict(os.environ, PYTHONUTF8="1", NODUS_HEADLESS_TEST_DLL=str(DLL))
    env["PATH"] = os.pathsep.join([str(DLL.parent), env.get("PATH", "")])
    return env


@pytest.fixture()
def live(tmp_path):
    missing = [str(p) for p in (DLL, RUN_CANVAS, PEER_HELPER, CANVAS_TEST) if not p.exists()]
    if missing:
        pytest.skip("Nodus pieces missing: " + ", ".join(missing))
    env = _nodus_env()
    canvas = tmp_path / "messenger.nodus.txt"
    saved = subprocess.run([sys.executable, "-I", "-c", _SAVE, str(CANVAS_TEST), str(canvas)], env=env, cwd=str(NODUS),
                           capture_output=True, text=True, timeout=60)
    if saved.returncode != 0 or not canvas.exists():
        pytest.skip("could not build the messenger canvas with the Nodus DLL: " + (saved.stderr or saved.stdout)[-500:])
    prefix = saved.stdout.strip().splitlines()[-1]
    host_proc = subprocess.Popen([sys.executable, str(RUN_CANVAS), "--canvas", str(canvas), "--rate", "500",
                                  "--acquire", f"{prefix}:{CAPACITY}"], env=env, cwd=str(NODUS), stdout=subprocess.PIPE,
                                 stderr=open(tmp_path / "run_canvas.err", "w", encoding="utf-8"), text=True,
                                 creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    httpd = None
    try:
        first = json.loads(host_proc.stdout.readline())              # the host's first JSON line (the supervisor reads the same one)
        entry = first["contract_ports"][0]
        assert entry["prefix"] == prefix and entry["listen_port"] > 0

        web = tmp_path / "build" / "meowdoku" / "web"
        web.mkdir(parents=True)
        (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}), encoding="utf-8")
        (web / "lowerings").mkdir()
        (web / "lowerings" / "demo.api.yaml").write_text("module: demo\nentry: run\nlanguage: wasm\n", encoding="utf-8")
        sessions = srv.Sessions(web, None)
        sessions.lifecycle = SimpleNamespace(token="t" * 40, restore_for=lambda token: None)
        sessions.access.lookup = lambda url: None
        sessions.federation_port, sessions.federation_canvas = entry["listen_port"], str(canvas)   # what --federation-port/-canvas set
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        db = sessions.accounts_db
        uids = {}
        for name in ("lab-engine", "lab-host"):
            uids[name] = sessions.accounts.ensure_user(name, "correcthorse1")
            conn = sqlite3.connect(db)
            conn.executescript(authz._DDL)
            conn.execute("INSERT OR REPLACE INTO account_roles (user_id, role, granted_by, created_at) VALUES (?, 'service', 'test', 0)",
                         (uids[name],))
            conn.commit()
            conn.close()
        yield SimpleNamespace(prefix=prefix, port=httpd.server_address[1], db=db, uids=uids, digests=fp.identity_digests(canvas, web),
                              contract_port=entry["listen_port"])
    finally:
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if host_proc.poll() is None:
            host_proc.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGTERM)
            try:
                host_proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                host_proc.kill()
                host_proc.communicate()
        host_proc.stdout.close()


def test_contract_round_trip_through_bridge_and_real_canvas_host(live):
    sys.path.insert(0, str(NODUS / "tools"))
    try:
        import contract_service_peer as csp
    finally:
        sys.path.pop(0)
    url = f"ws://127.0.0.1:{live.port}/ws/contract/{live.prefix}"
    e_id, e_secret = authz.grant_peer(live.db, "lab-engine", live.prefix, "engine", "submit,decision", 1, "test")
    h_id, h_secret = authz.grant_peer(live.db, "lab-host", live.prefix, "host", "accept,result", 1, "test")
    engine_sock = fp.connect(url, e_id, e_secret, "engine", live.digests)
    host_sock = fp.connect(url, h_id, h_secret, "host", live.digests)
    events = {"admitted": [], "routed": [], "receipts": []}

    class Producer:
        def admitted(self, identifier):
            events["admitted"].append(identifier)

        def route(self, result):
            events["routed"].append(result)
            return 1                                                  # Decision 1 = commit

    started = time.monotonic()
    try:
        engine = csp.EnginePeer(engine_sock, Producer())
        with ThreadPoolExecutor(max_workers=1) as executor:
            host = csp.ServicePeer(host_sock, executor, lambda loan: 555,
                                   lambda loan, accepted: events["receipts"].append(accepted), announce=False)
            assert engine.submit((100, live.uids["lab-engine"], 3, 4))      # field 1 is the engine's Hello id: the bridge forces it to the service account id
            deadline = started + 20.0
            while time.monotonic() < deadline:
                engine.poll()
                host.poll()
                if events["admitted"] and events["routed"] and events["receipts"]:
                    break
                time.sleep(0.002)
            elapsed = time.monotonic() - started
            print(f"federation live round trip: {elapsed:.3f}s through /ws/contract/{live.prefix} -> real run_canvas port {live.contract_port}")
            assert engine.error is None and host.error is None
            assert events["admitted"] == [100]
            assert len(events["routed"]) == 1 and events["routed"][0][7] == 555
            assert events["receipts"] == [True]
            assert elapsed < 20.0
    finally:
        engine_sock.close()
        host_sock.close()
