"""[LOADING] The loading animation: the MLA1 stream, its octet-codec packet (Python encode, JS decode bit-exact), the loop seam, and the
store / upload rules (size cap, format, write-once)."""

import http.client
import json
import shutil
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from meowdoku import loading_anim as L  # noqa: E402
from meowdoku import server as srv  # noqa: E402

NODE = shutil.which("node")
CHECK = ROOT / "tests" / "meowdoku_loading_anim_check.mjs"


def _anim(frames=60, grains=24, seed=3, seam=8):
    rng = np.random.default_rng(seed)
    x = (np.cumsum(rng.integers(-40, 41, (frames, grains)), axis=0) + 30000).astype(np.uint16)
    y = (np.cumsum(rng.integers(-40, 41, (frames, grains)), axis=0) + 20000).astype(np.uint16)
    x[30:], y[30:] = x[30], y[30]                                   # at rest from frame 30: the deltas are zero
    return {"fps": 24, "seam": seam, "bbox": (-0.5, 0.5, 0.0, 1.25), "radius": rng.integers(100, 900, grains).astype(np.uint16),
            "angle": (np.arange(frames) * 700).astype(np.uint16), "outline": [np.stack([np.arange(5) * 1000, np.arange(5) * 3000], axis=1).astype(np.uint16)],
            "x": x, "y": y}


def _same(a, b):
    assert a["fps"] == b["fps"] and a["seam"] == b["seam"] and tuple(a["bbox"]) == tuple(b["bbox"])
    for key in ("radius", "angle", "x", "y"):
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key
    assert len(a["outline"]) == len(b["outline"]) and all(np.array_equal(p, q) for p, q in zip(a["outline"], b["outline"]))


def test_stream_round_trips():
    anim = _anim()
    raw = L.encode(anim)
    _same(L.decode(raw), anim)
    assert raw.endswith(bytes(2 * 2 * 24 * 10))                      # the frames at rest are all-zero delta planes (the codec makes them free)


def test_octet_codec_packet_round_trips_in_python():
    raw = L.encode(_anim())
    packet = L.pack(raw)
    assert L.unpack(packet) == raw
    assert len(packet) < len(raw)


def test_malformed_streams_are_refused():
    raw = L.encode(_anim())
    for bad in (b"", raw[:-1], raw + b"\0", b"XXXX" + raw[4:], raw[:6] + bytes([0]) + raw[7:]):     # truncated, trailing, magic, fps 0
        with pytest.raises(L.AnimError):
            L.decode(bad)
    with pytest.raises(L.AnimError):
        L.decode(bytes(L.MAX_BYTES + 1))


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_decodes_the_python_packet_bit_exact(tmp_path):
    anim = _anim()
    packet = L.pack(L.encode(anim))
    (tmp_path / "a.mla").write_bytes(packet)
    subprocess.run([NODE, str(CHECK), "decode", str(tmp_path / "a.mla"), str(tmp_path / "a.json")], check=True, capture_output=True, text=True)
    got = json.loads((tmp_path / "a.json").read_text())
    want = L.decode(L.unpack(packet))
    assert got["fps"] == want["fps"] and got["seam"] == want["seam"] and np.allclose(got["bbox"], want["bbox"])
    for key in ("radius", "angle", "x", "y"):
        assert np.array_equal(np.asarray(got[key]), want[key]), key
    assert all(np.array_equal(np.asarray(p), q.reshape(-1)) or np.array_equal(np.asarray(p).reshape(-1, 2), q) for p, q in zip(got["outline"], want["outline"]))


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_encoded_stream_decodes_in_python(tmp_path):
    anim = _anim(seed=9)
    spec = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in anim.items() if k != "outline"}
    spec["outline"] = [p.reshape(-1).tolist() for p in anim["outline"]]
    spec["bbox"] = list(anim["bbox"])
    (tmp_path / "s.json").write_text(json.dumps(spec))
    subprocess.run([NODE, str(CHECK), "encode", str(tmp_path / "s.json"), str(tmp_path / "s.bin")], check=True, capture_output=True, text=True)
    raw = (tmp_path / "s.bin").read_bytes()
    assert raw == L.encode(anim)                                     # the two encoders write the same bytes
    _same(L.decode(raw), anim)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_loop_seam_matches_in_js_and_is_continuous():
    frames, overlap, count = 60, 12, 96
    out = subprocess.run([NODE, str(CHECK), "seam", str(frames), str(overlap), str(count)], check=True, capture_output=True, text=True).stdout
    js = json.loads(out)
    py = [L.seam(p, frames, overlap) for p in range(count)]
    assert [tuple(r) for r in js] == [tuple(r) for r in py]
    period = frames - overlap
    # outside the overlap one frame plays; inside it the tail frame fades out as the head frame fades in
    assert py[overlap] == (overlap, overlap, 0.0) and py[period - 1][2] == 0.0
    assert py[0][0] == period and py[0][1] == 0 and py[overlap - 1][2] > 0.9 and py[0][2] < 0.1
    # the loop is seamless: the frame shown before the wrap (period - 1) is followed by the tail's first frame, which the wrap shows
    assert py[period - 1][0] + 1 == py[period % period][0]
    assert [r[0] for r in py[:period]] == [r[0] for r in py[period:2 * period]]            # periodic


def test_store_keeps_the_first_as_active_and_validates(tmp_path):
    store = L.Store(tmp_path)
    assert store.info() is None
    with pytest.raises(L.AnimError):
        store.put(b"junk")
    assert store.info() is None
    raw = L.encode(_anim())
    meta = store.put(raw)
    assert meta["raw_bytes"] == len(raw) and meta["bytes"] == (store.root / f"{meta['digest']}.mla").stat().st_size and meta["frames"] == 60 and meta["grains"] == 24
    assert L.unpack(store.read()) == raw
    second = store.put(L.encode(_anim(seed=5)))                      # [STUDIO] a later one joins the library; the first stays the one players get
    assert second["active"] is False and store.info()["digest"] == meta["digest"]


# ---------------------------------------------------------------- the endpoint
def _web(tmp_path):
    web = tmp_path / "build" / "meowdoku" / "web"
    web.mkdir(parents=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}), encoding="utf-8")
    return web


@pytest.fixture()
def server(tmp_path):
    web = _web(tmp_path)
    sessions = srv.Sessions(web, None)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1], sessions
    httpd.shutdown()
    httpd.server_close()


def _post(port, body, headers=None, length=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    h = {"X-Meowdoku-Loading": "1", "Content-Type": "application/octet-stream", **(headers or {})}
    if length is not None:
        h["Content-Length"] = str(length)
    conn.request("POST", "/api/loading_anim", body=body, headers=h)
    r = conn.getresponse()
    data = r.read()
    conn.close()
    return r.status, json.loads(data or b"{}")


def test_upload_rules(server):
    port, sessions = server
    raw = L.encode(_anim())
    assert _post(port, raw, {"X-Meowdoku-Loading": "0"})[0] == 403                         # not from the page's own script
    assert _post(port, b"")[0] == 413                                                      # empty
    assert _post(port, b"x", length=L.MAX_BYTES + 1)[0] == 413                             # over the cap (refused before reading)
    assert _post(port, b"not an animation")[0] == 400                                      # wrong format
    assert sessions.anim.info() is None
    status, reply = _post(port, raw)
    assert status == 201 and reply["stored"]["raw_bytes"] == len(raw)
    assert _post(port, L.encode(_anim(seed=7)))[0] == 201                                  # [STUDIO] a later save joins the library (not active)
    assert sessions.anim.info()["digest"] == reply["stored"]["digest"]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request("GET", "/api/ready")
    ready = json.loads(conn.getresponse().read())
    assert ready["loading_anim"]["present"] and ready["loading_anim"]["bytes"] == sessions.anim.info()["bytes"]
    conn.request("GET", ready["loading_anim"]["url"])
    r = conn.getresponse()
    body = r.read()
    assert r.status == 200 and "immutable" in r.getheader("Cache-Control") and L.unpack(body) == raw


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_animation_is_a_filter_of_the_state_memory():
    """filterTape reads a recording.js Recording: dead grains dropped, frames decimated by the memory's snapshot interval, positions
    quantized, and the view angle follows the memory's logged gravity turns -- the same way round each time, ending where it began."""
    out = subprocess.run([NODE, str(CHECK), "filter"], check=True, capture_output=True, text=True).stdout
    r = json.loads(out)
    assert r["grains"] == 2 and r["fps"] == 30 and r["frames"] == 60       # 120 recorded frames of 1/60 s, kept every other one
    angle = np.asarray(r["angle"]) / 65536.0 * 2 * np.pi
    assert angle[0] == 0 and angle[-1] == 0                                  # starts and ends in the same pose: it loops
    assert np.all(np.diff(np.unwrap(angle)) <= 1e-6)                         # the view turns one way (opposite the case), never back
    assert abs(angle[30] - np.pi) < 1e-3                                     # the first half turn (step 240) is complete a moment later
    assert abs(np.unwrap(angle)[-1] + 2 * np.pi) < 1e-3                      # two half turns: one whole turn, the pose it began in
