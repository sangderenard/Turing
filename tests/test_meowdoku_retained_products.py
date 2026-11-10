"""A backend-only restart pins the accepted products and never builds physics."""
import builtins
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import retained_products as retained
from meowdoku import server, startup


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def release(tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    write_json(web / "boards.json", {"boards": [], "levels": {}})
    for prefix, family in startup.FOUNTAIN_FAMILIES:
        record = {
            "tag": "one", "n": 1, "pad": 2, "route": "webgpu-compute", "verdict": {"speedup": 1.0},
            "source_digest": "the-original-source-digest", "wgsl": f"{prefix}_one.wgsl",
            "wgsl_abi": {"count": 2, "arguments": ["x", "gravity"], "uniforms": ["gravity"], "outputs": ["x_next"],
                         "slots": 2, "bindings": 1, "slots_per_binding": 16, "arena_elements": 5,
                         "dispatches": 1, "groups": [1, 1, 1], "workgroup_size": [32, 1, 1], "entries": ["p0"],
                         "feed_slots": {"x": 0}, "output_slots": {"x_next": 1}, "uniform_index": {"gravity": 0}},
            "memory": {"arena_bytes": 20},
        }
        if family == "fountain":
            record.update(wasm=f"{prefix}_one.wasm", wasm_abi=f"{prefix}_one.abi.json")
            (web / record["wasm"]).write_bytes(b"\0asm\x01\0\0\0")
            write_json(web / record["wasm_abi"], {
                "schema": "turing.graph-buffer-abi.v1", "wasm": record["wasm"], "count": 2,
                "argument_names": ["x", "gravity"], "order": [0, 1, 2], "shapes": [[2], [], [2]],
                "dtypes": ["double"] * 3, "inputs": {"x": 0, "gravity": 1}, "outputs": [{"name": "x_next", "id": 2}],
            })
        write_json(web / f"{prefix}.manifest.json", {
            "schema": "turing.fountain3d-products.v1" if prefix.startswith("fountain3d") else "turing.fountain-products.v3",
            "cost_model": {"min_speedup": 2.0}, "law": {"arguments": ["x", "gravity"], "outputs": ["x_next"],
                "state": ["x"], "uniforms": ["gravity"], "diagnostics": [], "dt": 0.01}, "sizes": {"one": record},
        })
        (web / record["wgsl"]).write_text("@compute @workgroup_size(32) fn p0() {}", encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    write_json(receipt_path, retained.inspect_products(web))
    return web, receipt_path


def hashes(web):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in web.iterdir() if p.is_file()}


def test_retained_validation_does_not_modify_products_or_source_digests(release):
    web, receipt = release
    before = hashes(web)
    report = retained.validate_receipt(web, receipt)
    assert len(report["families"]) == 8
    assert len(report["files"]) == 18
    assert before == hashes(web)
    assert "the-original-source-digest" in (web / "fountain.manifest.json").read_text()


@pytest.mark.parametrize("filename", ["fountain3d_flat.manifest.json", "fountain3d_toy_one.wgsl", "fountain_one.wasm", "fountain_one.abi.json"])
def test_missing_files_refuse_without_writing(release, filename):
    web, receipt = release
    (web / filename).unlink()
    before = hashes(web)
    with pytest.raises(ValueError, match="missing or unreadable"):
        retained.validate_receipt(web, receipt)
    assert before == hashes(web)


@pytest.mark.parametrize("filename", ["fountain3d.manifest.json", "fountain3d_wheel_one.wgsl", "fountain_one.wasm", "fountain_one.abi.json"])
def test_changed_files_refuse_even_if_nonempty(release, filename):
    web, receipt = release
    path = web / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        retained.validate_receipt(web, receipt)


def alter_manifest(web, change):
    path = web / "fountain3d_toy.manifest.json"
    data = json.loads(path.read_text())
    change(data)
    write_json(path, data)


@pytest.mark.parametrize("bad_path", ["../outside.wgsl", "/tmp/out.wgsl", "C:\\outside.wgsl", "https://example.com/x.wgsl", "subdir/out.wgsl"])
def test_traversal_and_absolute_references_are_rejected(release, bad_path):
    web, _ = release
    alter_manifest(web, lambda d: d["sizes"]["one"].update(wgsl=bad_path))
    with pytest.raises(ValueError, match="unexpected WGSL filename"):
        retained.inspect_products(web)


@pytest.mark.parametrize("change, message", [
    (lambda d: d.update(schema="unknown"), "unsupported manifest schema"),
    (lambda d: d.update(sizes={}), "missing law or sizes"),
    (lambda d: d.pop("cost_model"), "missing cost_model"),
    (lambda d: d["sizes"]["one"].update(pad=0), "invalid particle counts"),
    (lambda d: d["sizes"]["one"]["wgsl_abi"].update(count=3), "WGSL count mismatch"),
    (lambda d: d["sizes"]["one"]["wgsl_abi"].update(arguments=["wrong"]), "argument, output or uniform mismatch"),
    (lambda d: d["sizes"]["one"]["wgsl_abi"].update(output_slots={"x_next": 2}), "invalid output_slots"),
    (lambda d: d["sizes"]["one"]["wgsl_abi"].update(feed_slots={}), "missing state/output slots"),
    (lambda d: d["sizes"]["one"]["wgsl_abi"].update(entries=["missing"]), "missing advertised compute entry"),
    (lambda d: d["sizes"]["one"].update(wasm="fountain3d_toy_one.wasm"), "incomplete CPU fallback reference"),
])
def test_malformed_manifest_contracts_are_rejected_before_inventory(release, change, message):
    web, _ = release
    alter_manifest(web, change)
    with pytest.raises(ValueError, match=message):
        retained.inspect_products(web)


def test_cpu_magic_and_abi_checked_even_when_gpu_exists(release):
    web, _ = release
    (web / "fountain_one.wasm").write_bytes(b"not wasm")
    with pytest.raises(ValueError, match="invalid WebAssembly header"):
        retained.inspect_products(web)
    (web / "fountain_one.wasm").write_bytes(b"\0asm\x01\0\0\0")
    path = web / "fountain_one.abi.json"
    data = json.loads(path.read_text())
    data["inputs"]["x"] = 999
    write_json(path, data)
    with pytest.raises(ValueError, match="invalid input references"):
        retained.inspect_products(web)


def test_retained_lane_no_builder_calls_or_imports_and_truthful_cpu(release, monkeypatch):
    web, receipt = release
    before = hashes(web)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not any(part in name for part in ("fountain_build", "fountain3d_build", "studio_physics_build"))
        fromlist = kwargs.get("fromlist", args[2] if len(args) > 2 else ()) or ()
        assert not set(fromlist) & {"fountain_build", "fountain3d_build", "studio_physics_build"}
        return original_import(name, *args, **kwargs)

    def no_build(*args, **kwargs):
        pytest.fail("retained mode called a fountain builder")

    for name in ("prepare_fountain", "prepare_fountain_wheel", "prepare_fountain_purrline", "prepare_fountain_mobroad",
                 "prepare_fountain3d", "prepare_fountain3d_wheel", "prepare_fountain3d_toy"):
        monkeypatch.setattr(server, name, no_build)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setenv("MEOWDOKU_FOUNTAIN_WASM", "0")
    lanes = server.prepare_lanes(web, SimpleNamespace(retain_fountain_products=str(receipt)))
    ready = startup.Readiness()
    startup.Lanes(ready)._run("fountain", lanes["fountain"])
    snap = ready.snapshot()
    for family in startup.FAMILY_NAMES:
        assert ready.state(family) == "ready"
        assert "source freshness not checked" in snap["products"][family]["detail"]
        assert ready.state(f"{family}_cpu") == ("ready" if family == "fountain" else "skipped")
    assert "one" in snap["products"]["fountain_cpu"]["detail"]
    assert before == hashes(web)


def test_retained_lane_fails_closed_if_product_changes_after_preflight(release):
    web, receipt = release
    retained.validate_receipt(web, receipt)
    (web / "fountain3d_toy_one.wgsl").write_text("changed")
    ready = startup.Readiness()
    lanes = server.prepare_lanes(web, SimpleNamespace(retain_fountain_products=str(receipt)))
    startup.Lanes(ready)._run("fountain", lanes["fountain"])
    assert all(v["state"] == "failed" for v in ready.snapshot()["products"].values())


def test_default_lanes_keep_original_build_order(release, monkeypatch):
    web, _ = release
    called = []
    expected = ["prepare_fountain3d_toy", "prepare_fountain", "prepare_fountain_wheel", "prepare_fountain_purrline",
                "prepare_fountain_mobroad", "prepare_fountain3d", "prepare_fountain3d_wheel"]
    for name in expected:
        monkeypatch.setattr(server, name, lambda *a, name=name: called.append(name))
    lanes = server.prepare_lanes(web, SimpleNamespace())
    startup.Lanes(startup.Readiness())._run("fountain", lanes["fountain"])
    assert called == expected
    assert [step[0] for step in lanes["laws"]] == [("predictor",), ("policy",), ("graph",)]
    assert [step[0] for step in lanes["autopilot"]] == [("autopilot",)]


def test_invalid_retained_startup_stops_before_copy_bind_or_lifecycle(release, monkeypatch, capsys):
    web, receipt = release
    (web / "fountain3d_flat_one.wgsl").unlink()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid retained startup reached side effects")

    monkeypatch.setattr(server, "install_shell", forbidden)
    monkeypatch.setattr(server, "Sessions", forbidden)
    monkeypatch.setattr(server, "ThreadingHTTPServer", forbidden)
    assert server.main(["--web-dir", str(web), "--retain-fountain-products", str(receipt)]) == 1
    assert "no build attempted" in capsys.readouterr().out


def test_receipt_cli_is_create_only_and_checks_without_changes(release, capsys):
    web, receipt = release
    before = hashes(web)
    new_receipt = receipt.with_name("new.json")
    assert retained.main(["--web-dir", str(web), "--write-receipt", str(new_receipt)]) == 0
    assert retained.main(["--web-dir", str(web), "--write-receipt", str(new_receipt)]) == 1
    assert retained.main(["--web-dir", str(web), "--check-receipt", str(new_receipt)]) == 0
    assert retained.main(["--web-dir", str(web), "--write-receipt", str(web / "receipt.json")]) == 1
    assert before == hashes(web)
