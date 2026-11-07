"""[LOADING] Start-up: the server binds before any prepare step finishes, reports per-product readiness, gates requests for files that are
still building, and a failed step settles (nothing waits on it for ever)."""

import http.client
import json
import socket
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from meowdoku import server as srv  # noqa: E402
from meowdoku import startup  # noqa: E402


def test_readiness_states_and_waiting():
    r = startup.Readiness()
    r.declare("a", "b")
    assert r.state("a") == startup.PENDING and not r.settled(["a"]) and r.settled(["unknown"])
    r.building("a", progress=0.5)
    threading.Timer(0.2, lambda: r.ready("a")).start()
    assert not r.wait(["a"], 0.05)
    assert r.wait(["a"], 5)
    r.failed("b", "boom")
    snap = r.snapshot()
    assert snap["products"]["a"]["state"] == "ready" and snap["products"]["b"] == {"state": "failed", "label": "b", "detail": "boom", "seconds": snap["products"]["b"]["seconds"]}
    assert snap["all_settled"] and snap["compiling"] == []


def test_snapshot_names_what_is_compiling():
    r = startup.Readiness()
    r.declare("predictor", "fountain", "graph")
    r.building("fountain")
    snap = r.snapshot()
    assert [c["name"] for c in snap["compiling"]] == ["fountain", "graph", "predictor"] and snap["compiling"][0]["label"] == "fountain"
    assert {c["label"] for c in snap["compiling"]} >= {"engine laws", "page law graph"}
    assert not snap["all_settled"]


def test_files_map_to_their_products():
    stems = ("motion", "travel")
    f = lambda name: startup.product_for_file(name, stems)  # noqa: E731
    assert f("motion.wasm") == "predictor" and f("motion.abi.json") == "predictor" and f("index.html") is None and f("app.js") is None
    assert f("policy.wasm") == "policy" and f("law.abi.json") == "graph" and f("pilot.weights.json") == "graph"
    assert f("fountain.manifest.json") == "fountain" and f("fountain_4k.wgsl") == "fountain" and f("fountain_4k.wasm") == "fountain_cpu"
    assert f("fountain_wheel_4k.wgsl") == "fountain_wheel" and f("fountain_purrline.manifest.json") == "fountain_purrline"
    assert f("fountain3d_flat_small.abi.json") == "fountain3d_flat_cpu" and f("fountain3d_wheel.manifest.json") == "fountain3d_wheel"
    assert f("fountain3d_4k.wgsl") == "fountain3d" and f("fountain.html") is None and f("fountain.js") is None


def test_the_puzzle_needs_no_product():
    assert "meowdoku" not in startup.GAME_NEEDS and "chess" not in startup.GAME_NEEDS
    assert startup.GAME_NEEDS["fountain"]["gpu"] == ["fountain"] and startup.GAME_NEEDS["space_invaders"] == ["predictor"]


def test_lanes_settle_failures_and_partial_readiness():
    r = startup.Readiness()
    log = []

    def partial(ctx):
        ctx.ready("gpu")
        assert r.state("gpu") == startup.READY and r.state("cpu") == startup.BUILDING
        time.sleep(0.05)

    def boom(ctx):
        raise RuntimeError("compile failed")

    lanes = {"one": [(("gpu", "cpu"), partial), (("bad",), boom)], "two": [(("other",), lambda ctx: None)]}
    for t in startup.Lanes(r, log.append).start(lanes):
        t.join(5)
    assert [r.state(n) for n in ("gpu", "cpu", "bad", "other")] == ["ready", "ready", "failed", "ready"]
    assert any("compile failed" in line for line in log)


# ---------------------------------------------------------------- the server itself
def _free_port():
    for port in range(8880, 8990):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port")


def _get(port, path, timeout=30):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("GET", path)
    r = conn.getresponse()
    body = r.read()
    conn.close()
    return r.status, body


@pytest.fixture()
def web(tmp_path):
    web = tmp_path / "build" / "meowdoku" / "web"
    web.mkdir(parents=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}), encoding="utf-8")
    (web / "index.html").write_text("<html>page</html>", encoding="utf-8")
    return web


def test_server_binds_before_a_slow_prepare_finishes(web, monkeypatch):
    release = threading.Event()
    started = threading.Event()
    servers = []

    class Capture(ThreadingHTTPServer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            servers.append(self)

    def slow(ctx):
        started.set()
        release.wait(60)
        (web / "motion.wasm").write_bytes(b"\0asm")

    lanes = {"laws": [(("predictor",), slow)]}
    monkeypatch.setattr(srv, "ThreadingHTTPServer", Capture)
    monkeypatch.setattr(srv, "prepare_lanes", lambda web_dir, args: lanes)
    monkeypatch.setattr(srv, "predictor_stems", lambda: ("motion",))
    monkeypatch.setattr(srv, "PRODUCT_WAIT_SECONDS", 0.3)
    port = _free_port()
    done = []
    thread = threading.Thread(target=lambda: done.append(srv.main(["--port", str(port), "--web-dir", str(web), "--profiles", str(web.parent / "p.json"),
                                                                   "--no-idle-train"])), daemon=True)
    thread.start()
    try:
        assert started.wait(30), "the prepare step never started"
        # the prepare step is still running (blocked), yet the page and the readiness endpoint answer
        assert _get(port, "/index.html") == (200, b"<html>page</html>")
        status, body = _get(port, "/api/ready")
        ready = json.loads(body)
        assert status == 200 and ready["products"]["predictor"]["state"] == "building" and not ready["all_settled"]
        assert ready["compiling"][0]["name"] == "predictor" and ready["loading_anim"] == {"present": False, "upload": True}
        assert ready["needs"]["space_invaders"] == ["predictor"]
        # a file of the building product waits, then answers "try again" rather than 404
        t0 = time.time()
        assert _get(port, "/motion.wasm")[0] == 503 and time.time() - t0 >= 0.25
        # the catalog and a game that needs nothing are served now
        assert _get(port, "/api/catalog")[0] == 200
        release.set()
        for _ in range(100):
            if json.loads(_get(port, "/api/ready")[1])["all_settled"]:
                break
            time.sleep(0.05)
        ready = json.loads(_get(port, "/api/ready")[1])
        assert ready["products"]["predictor"]["state"] == "ready" and ready["all_settled"]
        assert _get(port, "/motion.wasm") == (200, b"\0asm")
    finally:
        release.set()
        for s in servers:
            s.shutdown()
        thread.join(10)
