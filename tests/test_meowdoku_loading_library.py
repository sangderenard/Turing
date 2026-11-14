"""[STUDIO] The loading animation LIBRARY: the store (add, list, cap, delete, activate, migrate the old single file), the owner's token on
activate / delete (missing, wrong, plain http from afar), /api/loading_anim serving the active entry, and the recorder's parameter handling
and the stream at each bit depth (node)."""

import http.client
import json
import shutil
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from meowdoku import loading_anim as L  # noqa: E402
from meowdoku import server as srv  # noqa: E402

NODE = shutil.which("node")
CHECK = ROOT / "tests" / "meowdoku_loading_params_check.mjs"
TOKEN = "t" * 43


def _anim(seed=3, frames=40, grains=16):
    rng = np.random.default_rng(seed)
    x = (np.cumsum(rng.integers(-40, 41, (frames, grains)), axis=0) + 30000).astype(np.uint16)
    y = (np.cumsum(rng.integers(-40, 41, (frames, grains)), axis=0) + 20000).astype(np.uint16)
    return {"fps": 24, "seam": 6, "bbox": (-0.5, 0.5, 0.0, 1.25), "radius": rng.integers(100, 900, grains).astype(np.uint16),
            "angle": (np.arange(frames) * 700).astype(np.uint16), "outline": [], "x": x, "y": y}


def _raw(seed=3, **kw):
    return L.encode(_anim(seed, **kw))


# ---------------------------------------------------------------- the store
def test_add_list_activate_delete(tmp_path):
    lib = L.Library(tmp_path)
    assert lib.info() is None and lib.entries() == [] and lib.active_digest() is None
    a = lib.put(_raw(1), {"chamber": "sdf_hourglass", "bits": 12, "evil": {"nested": 1}})
    assert a["active"] is True and a["params"] == {"chamber": "sdf_hourglass", "bits": 12}          # nested values are not kept
    b = lib.put(_raw(2))
    assert b["active"] is False and lib.active_digest() == a["digest"]                              # a later save is only added
    again = lib.put(_raw(2))
    assert again["duplicate"] is True and len(lib.entries()) == 2                                   # the same animation is one entry
    assert [e["digest"] for e in lib.entries()].count(b["digest"]) == 1 and {e["digest"] for e in lib.entries() if e["active"]} == {a["digest"]}
    assert L.unpack(lib.read(b["digest"])) == _raw(2) and L.unpack(lib.read()) == _raw(1)
    assert (tmp_path / "loading_library" / f"{a['digest']}.mla").exists() and (tmp_path / "loading_library" / f"{a['digest']}.json").exists()
    lib.activate(b["digest"])
    assert lib.info()["digest"] == b["digest"] and L.Library(tmp_path).active_digest() == b["digest"]   # persisted
    gone = lib.delete(a["digest"])
    assert gone["was_active"] is False and [e["digest"] for e in lib.entries()] == [b["digest"]]
    assert lib.delete(b["digest"])["was_active"] is True and lib.info() is None and lib.read() is None
    for bad in ("zz", "../../etc/passwd", "0" * 15, "0" * 16):
        with pytest.raises(KeyError):
            lib.activate(bad)
        with pytest.raises(KeyError):
            lib.delete(bad)
        assert lib.read(bad) is None
    with pytest.raises(L.AnimError):
        lib.put(b"junk")


def test_library_keeps_saves_without_entry_or_byte_quota(tmp_path):
    lib = L.Library(tmp_path)
    for seed in range(5):
        lib.put(_raw(seed))
    assert len(lib.entries()) == 5
    assert lib.caps()["max_entries"] is None and lib.caps()["max_bytes"] is None


def test_temporary_capture_graduation_and_expiry(tmp_path, monkeypatch):
    lib = L.Library(tmp_path)
    candidate = lib.put(_raw(23), {"source": "fountain3d", "projected_3d": True}, temporary=True)
    digest = candidate["digest"]
    assert candidate["expires_at"] and lib.active_digest() is None
    assert lib.read(digest) and lib.entries()[0]["expires_at"] == candidate["expires_at"]
    with pytest.raises(KeyError):
        lib.activate(digest)
    permanent = lib.graduate(digest)
    assert "expires_at" not in permanent and "expires_at" not in L.Library(tmp_path).meta(digest)
    assert lib.activate(digest)["digest"] == digest

    monkeypatch.setattr(L, "TEMP_SECONDS", -1)
    expired = lib.put(_raw(24), temporary=True)["digest"]
    assert expired not in [entry["digest"] for entry in lib.entries()]
    assert not (lib.root / f"{expired}.mla").exists()
    assert not (lib.root / f"{expired}.json").exists()


def test_migrates_the_old_single_file_as_the_active_entry(tmp_path):
    raw = _raw(5)
    packet = L.pack(raw)
    meta = {"digest": L.digest(packet), "raw_bytes": len(raw), "bytes": len(packet), "frames": 40, "grains": 16, "fps": 24}
    (tmp_path / "loading_anim.mla").write_bytes(packet)
    (tmp_path / "loading_anim.json").write_text(json.dumps(meta))
    lib = L.Library(tmp_path)
    assert lib.info()["digest"] == meta["digest"] and lib.read() == packet
    [entry] = lib.entries()
    assert entry["active"] and entry["params"] == {"migrated": True} and entry["date"].endswith("Z")
    assert (tmp_path / "loading_anim.mla").exists()                                                 # the old files are left alone
    lib.delete(meta["digest"])
    assert L.Library(tmp_path).info() is None                                                       # not migrated again once the pointer exists
    # a corrupt old file is not migrated
    other = tmp_path / "other"
    other.mkdir()
    (other / "loading_anim.mla").write_bytes(b"junk")
    (other / "loading_anim.json").write_text(json.dumps({**meta, "bytes": 4}))
    assert L.Library(other).info() is None


def test_owner_authorize_rules():
    ok = lambda **k: L.owner_authorize(**{"tls": False, "forwarded_proto": None, "loopback_direct": False, "supplied": TOKEN, "token": TOKEN, **k})[:2]
    assert ok(tls=True) == (True, 200) and ok(forwarded_proto="https") == (True, 200) and ok(loopback_direct=True) == (True, 200)
    assert ok() == (False, 403) and ok(forwarded_proto="http") == (False, 403)                       # plain http from afar
    assert ok(tls=True, supplied="x" * 43) == (False, 401) and ok(tls=True, supplied=None) == (False, 401) and ok(tls=True, supplied="") == (False, 401)
    assert ok(tls=True, token="") == (False, 401)                                                    # a server with no token admits no one


# ---------------------------------------------------------------- the endpoints
@pytest.fixture()
def server(tmp_path):
    web = tmp_path / "build" / "meowdoku" / "web"
    web.mkdir(parents=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}), encoding="utf-8")
    sessions = srv.Sessions(web, None)
    sessions.lifecycle = SimpleNamespace(token=TOKEN)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    _sign_in(httpd.server_address[1])
    yield httpd.server_address[1], sessions
    httpd.shutdown()
    httpd.server_close()


SIGNED_IN = {}       # port -> account cookie: a visitor with no account is a guest and never holds library_manage (owner, 2026-11-13)


def _sign_in(port):
    status, _, response = _req(port, "POST", "/api/account/signup",
                               json.dumps({"username": "studio_cat", "password": "correcthorse1"}).encode(),
                               {"Content-Type": "application/json", "Cookie": ""})
    assert status == 201
    SIGNED_IN[port] = next(c.split(";")[0] for c in response.msg.get_all("Set-Cookie") if c.startswith(srv.ACCOUNT_COOKIE))


def _req(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    h = {"Cookie": SIGNED_IN.get(port, ""), **(headers or {})}
    if not h["Cookie"]:
        del h["Cookie"]
    if body is not None and "Content-Length" not in h:
        h["Content-Length"] = str(len(body))
    conn.request(method, path, body=body, headers=h)
    r = conn.getresponse()
    data = r.read()
    out = (r.status, data, r)
    conn.close()
    return out


def _save(port, raw, params=None):
    h = {"X-Meowdoku-Loading": "1", "Content-Type": "application/octet-stream"}
    if params is not None:
        from urllib.parse import quote
        h["X-Meowdoku-Params"] = quote(json.dumps(params))
    status, data, _ = _req(port, "POST", "/api/loading_anim", raw, h)
    return status, json.loads(data or b"{}")


def _owner(port, verb, digest, token=TOKEN, extra=None):
    h = {"Content-Type": "application/json", **({"X-Meowdoku-Admin": token} if token is not None else {}), **(extra or {})}
    status, data, _ = _req(port, "POST", f"/api/loading_library/{verb}", json.dumps({"digest": digest}).encode(), h)
    return status, json.loads(data or b"{}")


def test_saves_go_to_the_library_and_the_first_is_active(server):
    port, sessions = server
    s1, r1 = _save(port, _raw(1), {"chamber": "sdf_circle", "bits": 10})
    s2, r2 = _save(port, _raw(2))
    assert (s1, s2) == (201, 201) and r1["stored"]["active"] is True and r2["stored"]["active"] is False
    status, data, _ = _req(port, "GET", "/api/loading_library")
    lib = json.loads(data)
    assert status == 200 and lib["active"] == r1["stored"]["digest"] and len(lib["entries"]) == 2 and lib["caps"]["entries"] == 2
    assert {e["digest"]: e["params"] for e in lib["entries"]}[r1["stored"]["digest"]] == {"chamber": "sdf_circle", "bits": 10}
    assert "token" not in json.dumps(lib).lower()


def test_temporary_upload_requires_graduation_before_activation(server):
    port, sessions = server
    raw = _raw(31)
    headers = {"X-Meowdoku-Loading": "1", "Content-Type": "application/octet-stream"}
    status, body, _ = _req(port, "POST", "/api/loading_candidate", raw, headers)
    assert status == 201
    candidate = json.loads(body)["stored"]
    digest = candidate["digest"]
    assert candidate["expires_at"] and not candidate["active"]
    assert _req(port, "GET", "/api/loading_anim")[0] == 404
    assert _req(port, "GET", f"/api/loading_anim?v={digest}")[0] == 200
    assert _owner(port, "activate", digest)[0] == 404
    status, reply = _owner(port, "graduate", digest)
    assert status == 200 and "expires_at" not in reply["entry"]
    assert _owner(port, "activate", digest)[0] == 200
    assert L.unpack(_req(port, "GET", "/api/loading_anim")[1]) == raw


def test_loading_anim_serves_the_active_entry(server):
    port, sessions = server
    assert _req(port, "GET", "/api/loading_anim")[0] == 404
    _, r1 = _save(port, _raw(1))
    _, r2 = _save(port, _raw(2))
    d1, d2 = r1["stored"]["digest"], r2["stored"]["digest"]
    status, body, resp = _req(port, "GET", "/api/loading_anim")
    assert status == 200 and L.unpack(body) == _raw(1) and resp.getheader("Cache-Control") == "no-cache"        # no digest: whatever is active now
    status, body, resp = _req(port, "GET", f"/api/loading_anim?v={d2}")
    assert status == 200 and L.unpack(body) == _raw(2) and "immutable" in resp.getheader("Cache-Control")      # anyone may view any entry
    assert _req(port, "GET", "/api/loading_anim?v=" + "0" * 16)[0] == 404 and _req(port, "GET", "/api/loading_anim?v=../x")[0] == 404
    ready = json.loads(_req(port, "GET", "/api/ready")[1])["loading_anim"]
    assert ready["present"] and ready["url"].endswith(d1)
    assert _owner(port, "activate", d2)[0] == 200
    assert L.unpack(_req(port, "GET", "/api/loading_anim")[1]) == _raw(2)                                      # now players get the other one
    assert json.loads(_req(port, "GET", "/api/ready")[1])["loading_anim"]["url"].endswith(d2)


def test_activate_and_delete_need_the_owner_token(server):
    port, sessions = server
    sessions.access.decide = lambda ip, host: ("ip", "edit")          # a REMOTE owner (bare IP, edit tier): the token applies
    _, r1 = _save(port, _raw(1))
    _, r2 = _save(port, _raw(2))
    d1, d2 = r1["stored"]["digest"], r2["stored"]["digest"]
    for verb in ("activate", "delete"):
        assert _owner(port, verb, d2, token=None)[0] == 401                                          # no token
        assert _owner(port, verb, d2, token="wrong")[0] == 401                                       # a bad one
        assert _owner(port, verb, d2, token=TOKEN, extra={"X-Forwarded-For": "8.8.8.8"})[0] == 403  # through a proxy that did not say https
    assert sessions.anim.active_digest() == d1 and len(sessions.anim.entries()) == 2                 # nothing changed
    assert _owner(port, "activate", d2, extra={"X-Forwarded-For": "8.8.8.8", "X-Forwarded-Proto": "https"})[0] == 200
    assert sessions.anim.active_digest() == d2
    assert _owner(port, "activate", "0" * 16)[0] == 404 and _owner(port, "delete", "../../x")[0] == 404
    status, reply = _owner(port, "delete", d1)
    assert status == 200 and reply["entry"]["was_active"] is False and [e["digest"] for e in sessions.anim.entries()] == [d2]
    status, reply = _owner(port, "delete", d2)
    assert status == 200 and reply["active"] is None and _req(port, "GET", "/api/loading_anim")[0] == 404


def test_wrong_tokens_are_rate_limited_and_the_token_is_never_sent(server):
    port, sessions = server
    sessions.access.decide = lambda ip, host: ("ip", "edit")          # a REMOTE owner (bare IP, edit tier): the token applies
    _, r1 = _save(port, _raw(1))
    codes = [_owner(port, "delete", r1["stored"]["digest"], token="bad")[0] for _ in range(7)]
    assert codes[:5] == [401] * 5 and codes[5:] == [429, 429]
    assert _owner(port, "delete", r1["stored"]["digest"])[0] == 429                                  # even the right one, from that address, for a while
    assert len(sessions.anim.entries()) == 1
    for path in ("/api/loading_library", "/api/ready", "/api/loading_anim"):
        body = _req(port, "GET", path)[1]
        assert TOKEN.encode() not in body


def test_an_owner_action_without_a_lifecycle_is_unavailable(server):
    port, sessions = server
    sessions.lifecycle = None
    _, r1 = _save(port, _raw(1))
    assert _owner(port, "delete", r1["stored"]["digest"])[0] == 503


def test_upload_rules_still_hold(server):
    port, sessions = server
    raw = _raw(1)
    h = {"X-Meowdoku-Loading": "1"}
    assert _req(port, "POST", "/api/loading_anim", raw, {"X-Meowdoku-Loading": "0"})[0] == 403
    assert _req(port, "POST", "/api/loading_anim", b"", h)[0] == 400
    assert _save(port, b"not an animation")[0] == 400
    assert sessions.anim.entries() == []


def test_the_library_keeps_multiple_saves(server):
    port, sessions = server
    assert _save(port, _raw(1))[0] == 201
    assert _save(port, _raw(2))[0] == 201
    assert len(sessions.anim.entries()) == 2


# ---------------------------------------------------------------- the page side (node)
@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_recorder_parameter_handling():
    out = json.loads(subprocess.run([NODE, str(CHECK), "params"], check=True, capture_output=True, text=True).stdout)
    d = out["defaults"]
    assert (d["chamber"], d["size"], d["fps"], d["bits"], d["settle"], d["turns"], d["flipSeconds"], d["speed"]) == \
        ("sdf_hourglass", "small", 12, 12, 40, 4, 0.4, 0)
    assert out["bare"] == {"on": True, "studio": False, "params": d} and out["off"]["on"] is False
    assert out["bad"]["params"] == d                                                                  # a malformed rp is the defaults
    w = out["wild"]                                                                                   # everything is clamped or dropped
    assert (w["chamber"], w["size"], w["mix"], w["grainScale"], w["fps"], w["bits"], w["turns"], w["flipSeconds"], w["speed"], w["cap"], w["colour"]) == \
        ("sdf_hourglass", "small", 1, 0.5, 2, 12, 4, 2, 0, 10, 2)
    assert "junk" not in w
    assert out["sizes"] == ["one", "small", "4k", "16k", "64k", "256k", "1M", "4M", "10M"]
    assert out["sizeIds"]["one"] == 1
    assert out["sizeIds"]["256k"] == 262144 and out["sizeIds"]["10M"] == 10000000
    back = out["back"]
    assert back["on"] and back["studio"] and back["params"]["chamber"] == "sdf_circle" and back["params"]["bits"] == 10 and back["params"]["mix"] == 0.25 and back["params"]["turns"] == 6
    assert "record_loading=1" in out["query"] and "studio=1" in out["query"] and "rp=" in out["query"]
    assert out["legacy"] == [[12, 256, 12]]
    assert out["fixed"] == [[8, 256, 16]]
    assert out["tight"] == [[8, 256, 11]]
    assert out["meta"]["mix"] == "default" and out["meta"]["capped"] is True and out["meta"]["bits"] == 12
    assert all(isinstance(v, (str, int, float, bool)) for v in out["meta"].values())                  # flat: the server keeps these


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_visual_projection_filters_existing_packed_snapshots():
    out = json.loads(subprocess.run([NODE, str(CHECK), "visual"], check=True, capture_output=True, text=True).stdout)
    assert out == {"frames": 10, "sampled": 256, "step": 24, "grains": 256, "raw": 10806}


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_stream_at_each_bit_depth(tmp_path):
    subprocess.run([NODE, str(CHECK), "bits", str(tmp_path)], check=True, capture_output=True, text=True)
    info = json.loads((tmp_path / "info.json").read_text())
    assert set(info) == {"10", "11", "12", "16"}
    packed = {}
    for bits, i in info.items():
        raw = (tmp_path / f"b{bits}.bin").read_bytes()
        anim = L.decode(raw)                                                                          # Python reads what JS wrote, at every depth
        assert anim["x"].shape == (i["frames"], i["grains"]) and len(raw) == i["bytes"]
        assert i["lowBitsZero"], bits                                                                 # the unused low bits are zero (the codec drops them)
        low = (1 << (16 - int(bits))) - 1
        assert not (anim["x"] & low).any() and not (anim["y"] & low).any()
        assert L.Library(tmp_path / f"lib{bits}").put(raw)["frames"] == i["frames"]                   # and the library accepts it
        packed[int(bits)] = len(L.pack(raw))
    assert len({i["bytes"] for i in info.values()}) == 1                                              # the raw stream is the same size at every depth
    assert packed[10] < packed[11] < packed[12] < packed[16]                                          # fewer bits: the codec packs smaller


def test_a_local_visitor_needs_no_token_for_the_studio(server):
    """Signed-in LAN-origin visitors (the owner's own network) activate and delete without the token; the owner asked for the studio
    open to creative use.  A visitor with no account is a guest and must sign in first (owner, 2026-11-13)."""
    port, sessions = server
    sessions.access.decide = lambda ip, host: ("lan", "edit")
    _, r1 = _save(port, _raw(1))
    _, r2 = _save(port, _raw(2))
    d2 = r2["stored"]["digest"]
    assert _owner(port, "activate", d2, token=None)[0] == 200
    assert _owner(port, "delete", r1["stored"]["digest"], token=None)[0] == 200
