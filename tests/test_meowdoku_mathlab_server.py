"""Math lab endpoints (examples/meowdoku/server.py [MATHLAB], math_jobs.py) over an ephemeral loopback port and a temp state root.

One real job (targets numpy + wasm; native is skipped to keep it fast) runs through the single worker; the limit and permission
tests use a server with no worker, so their jobs stay queued."""
from __future__ import annotations

import http.client
import json
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import math_jobs  # noqa: E402
from meowdoku import server as srv  # noqa: E402

LAW = "r = sqrt(x*x + y*y)\ne = x*y + 2*z"
LOCAL = {"Host": "127.0.0.1"}


def _call(port, path, method="GET", headers=None, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    data = None if body is None else json.dumps(body).encode()
    hdrs = {**LOCAL, **(headers or {})}
    if data is not None:
        hdrs.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=data, headers=hdrs)
    r = conn.getresponse()
    raw = r.read()
    info = SimpleNamespace(status=r.status, ctype=r.getheader("Content-Type"), disposition=r.getheader("Content-Disposition"),
                           cookies=r.msg.get_all("Set-Cookie") or [], raw=raw)
    conn.close()
    try:
        info.json = json.loads(raw)
    except ValueError:
        info.json = None
    return info


def _start(tmp_path, worker):
    web = Path(tmp_path) / "build" / "meowdoku" / "web"
    web.mkdir(parents=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}), encoding="utf-8")
    sessions = srv.Sessions(web, None)
    sessions.lifecycle = SimpleNamespace(token="t" * 40, restore_for=lambda token: None)
    sessions.access.lookup = lambda url: None
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    if worker:
        sessions.mathlab_worker = math_jobs.Worker(sessions.accounts_db, sessions.state, poll_s=0.2).start()
    return httpd, sessions


def _fixture(tmp_path, worker):
    httpd, sessions = _start(tmp_path, worker)
    try:
        yield SimpleNamespace(port=httpd.server_address[1], sessions=sessions)
    finally:
        if sessions.mathlab_worker is not None:
            sessions.mathlab_worker.stop()
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture()
def live(tmp_path):
    yield from _fixture(tmp_path, True)


@pytest.fixture()
def idle(tmp_path):
    yield from _fixture(tmp_path, False)


def _signup(port, name):
    r = _call(port, "/api/account/signup", "POST", body={"username": name, "password": "correcthorse1"})
    assert r.status == 201
    return {"Cookie": next(c.split(";")[0] for c in r.cookies if c.startswith(srv.ACCOUNT_COOKIE))}


def test_guest_is_refused_naming_compile_submit(idle):
    r = _call(idle.port, "/api/mathlab/jobs", "POST", body={"text": LAW})
    assert r.status == 403 and "compile_submit" in r.json["error"]
    assert _call(idle.port, "/api/mathlab/jobs").status == 403
    who = _call(idle.port, "/api/whoami").json
    assert "compile_submit" not in who["capabilities"]            # the LAN edit tier does not grant it


def test_syntax_error_is_rejected_with_position_and_spends_no_worker(idle):
    member = _signup(idle.port, "cat1")
    r = _call(idle.port, "/api/mathlab/jobs", "POST", member, {"text": "r = x + __import__('os')"})
    assert r.status == 422 and r.json["state"] == "rejected"
    err = r.json["error"]
    assert err["line"] == 1 and isinstance(err["col"], int) and err["construct"]
    job = _call(idle.port, f"/api/mathlab/jobs/{r.json['job_id']}", headers=member).json
    assert job["state"] == "rejected" and job["queue_position"] == 0


def test_per_owner_queue_limit_and_privacy(idle):
    a, b = _signup(idle.port, "cat2"), _signup(idle.port, "cat3")
    ids = []
    for i in range(3):
        r = _call(idle.port, "/api/mathlab/jobs", "POST", a, {"text": LAW, "targets": ["numpy"]})
        assert r.status == 200 and r.json["state"] == "queued" and r.json["queue_position"] == i + 1
        ids.append(r.json["job_id"])
    r = _call(idle.port, "/api/mathlab/jobs", "POST", a, {"text": LAW, "targets": ["numpy"]})
    assert r.status == 429 and "3" in r.json["error"]
    assert _call(idle.port, "/api/mathlab/jobs", "POST", b, {"text": LAW, "targets": ["numpy"]}).status == 200   # another owner is unaffected
    assert _call(idle.port, f"/api/mathlab/jobs/{ids[0]}", headers=b).status == 404
    assert _call(idle.port, f"/api/mathlab/jobs/{ids[0]}/file/law.numpy.py", headers=b).status == 404
    assert [j["job_id"] for j in _call(idle.port, "/api/mathlab/jobs", headers=a).json["jobs"]] == ids[::-1]
    r = _call(idle.port, "/api/mathlab/jobs", "POST", a, {"text": LAW, "targets": ["bogus"]})
    assert r.status == 400


def test_job_runs_through_the_worker_and_serves_its_products(live):
    port = live.port
    member, other = _signup(port, "cat4"), _signup(port, "cat5")
    r = _call(port, "/api/mathlab/jobs", "POST", member, {"text": LAW, "targets": ["numpy", "wasm"]})
    assert r.status == 200, r.json
    job_id = r.json["job_id"]
    deadline, job = time.time() + 80, None
    while time.time() < deadline:
        job = _call(port, f"/api/mathlab/jobs/{job_id}", headers=member).json
        if job["state"] not in ("queued", "running"):
            break
        time.sleep(0.5)
    assert job["state"] == "complete", job
    targets = job["products"]["targets"]
    assert targets["numpy"]["state"] == "complete" and targets["wasm"]["state"] == "complete", targets
    assert job["products"]["schema"] == "turing.math-lab-products.v1"
    assert job["profile"]["reference"]["rows"] == 64 and set(job["profile"]["reference"]["inputs"]) == {"x", "y", "z"}
    base = f"/api/mathlab/jobs/{job_id}/file/"
    wasm = _call(port, base + "law.wasm", headers=member)
    assert wasm.status == 200 and wasm.ctype == "application/wasm" and wasm.raw[:4] == b"\0asm"
    py = _call(port, base + "law.numpy.py", headers=member)
    assert py.status == 200 and "attachment" in py.disposition and b"def user_law" in py.raw
    assert _call(port, base + "law.abi.json", headers=member).ctype == "application/json"
    # the native library and piece are never served, nor anything outside the job's listed files
    for name in ("law.native.piece", "native/user_law.dll", "..%2Fproducts.json", "law.math.txt"):
        assert _call(port, base + name, headers=member).status == 404, name
    assert _call(port, base + "law.wasm", headers=other).status == 404
    assert _call(port, f"/api/mathlab/jobs/{job_id}", headers=other).status == 404
    # the browser posts its own timings
    prof = f"/api/mathlab/jobs/{job_id}/browser_profile"
    good = {"target": "wasm", "rows": 4096, "ns_per_row": 3.5, "max_abs_err": 1e-16, "user_agent_hint": "Chrome 1 / Windows"}
    assert _call(port, prof, "POST", member, good).status == 200
    assert _call(port, prof, "POST", other, good).status == 404
    assert _call(port, prof, "POST", member, {**good, "ns_per_row": float("inf")}).status == 400
    assert _call(port, prof, "POST", member, {**good, "target": "numpy"}).status == 400
    assert _call(port, prof, "POST", member, {**good, "rows": -1}).status == 400
    stored = _call(port, f"/api/mathlab/jobs/{job_id}", headers=member).json["browser_profile"]
    assert stored["results"][0]["target"] == "wasm" and stored["results"][0]["ns_per_row"] == 3.5
