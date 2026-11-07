"""Saved marble recipes survive reload and are served by digest to the simulator."""
import http.client
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
from meowdoku import marble_objects as M  # noqa: E402
from meowdoku import server as srv  # noqa: E402


def example():
    tables = {name: {"stride": 2, "data": [0.1, 0.2] * 5}
              for name in ("pbr", "phong", "enamel", "texstack")}
    return {"v": 1, "name": "Clear D6", "generator": "meowdoku-marble-object-v1",
            "design": {"base-shape": "d6", "kind": "cat_eye", "seed": 42},
            "materials": {"index": {"glass": 0, "dye": 1},
                          "extra": {"coated": 2, "paintA": 3, "paintB": 4}, "tables": tables},
            "physical": {"radius-mm": 10, "shell-density": 2500, "sim-rho": 1,
                         "sim-radius": 0.01, "sim-mu": 0.35, "sim-e": 0.05,
                         "sim-roll": 0, "sim-irr": 0},
            "derived": {"massKg": 0.004, "centerM": [0, 0, 0],
                        "inertiaKgM2": [[1e-7, 0, 0], [0, 1e-7, 0], [0, 0, 1e-7]],
                        "massExact": True},
            "geometry_sha256": "f" * 64}


def test_immutable_store_roundtrip_and_source_digest(tmp_path):
    store = M.Store(tmp_path / "objects")
    entry = store.put(example())
    assert not entry["duplicate"]
    assert store.put(example())["duplicate"]
    assert store.get(entry["digest"])["name"] == "Clear D6"
    assert store.entries()[0]["shape"] == "d6"
    changed = example(); changed["physical"]["sim-mu"] = 0.4
    assert store.put(changed)["digest"] != entry["digest"]
    tampered = store.directory / f"{entry['digest']}.json"
    value = json.loads(tampered.read_text(encoding="utf-8"))
    value["design"]["seed"] = 7
    tampered.write_text(json.dumps(value), encoding="utf-8")
    try:
        store.get(entry["digest"])
    except M.ObjectError as error:
        assert "digest mismatch" in str(error)
    else:
        raise AssertionError("tampered object was accepted")


def test_library_routes_save_list_and_load(tmp_path):
    web = tmp_path / "build" / "meowdoku" / "web"
    web.mkdir(parents=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}), encoding="utf-8")
    sessions = srv.Sessions(web, None)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    def request(method, path, value=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        body = None if value is None else json.dumps(value).encode("utf-8")
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        status, content = response.status, json.loads(response.read() or b"{}")
        conn.close()
        return status, content
    try:
        status, saved = request("POST", "/api/marble_objects", example(),
                                {"Content-Type": "application/json", "X-Meowdoku-Marble": "1"})
        assert status == 201, saved
        status, listing = request("GET", "/api/marble_objects")
        assert status == 200 and listing["entries"][0]["digest"] == saved["digest"]
        status, loaded = request("GET", f"/api/marble_objects/{saved['digest']}")
        assert status == 200 and loaded["design"]["seed"] == 42
        assert request("GET", "/api/marble_objects/garbage")[0] == 404
    finally:
        httpd.shutdown(); httpd.server_close()
