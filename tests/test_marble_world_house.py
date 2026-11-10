"""No-bake, additive composition of a real class with the original map cache."""

import base64
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from examples.meowdoku.marble_world_house import (
    WORLD_ID, append_studio_house, load_verified_living_map,
)
from src.compiler.abstract_ui_introspection import build_introspective_world


class CameraPublication:
    """Test introspection must never instantiate the supplied actual type."""

    def __init__(self, scene):
        raise AssertionError("Class inspection instantiated its subject")

    def capture(self, time_s):
        raise AssertionError("Class inspection executed a method")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def packet_for(subject=CameraPublication):
    identity = f"python:{subject.__module__}.{subject.__qualname__}"
    face = [[0, 0, 0], [1, 0, 0], [0, 1, 0], 0,
            [[0, 0, 1], [0, 0, 1], [0, 0, 1]]]
    pbr = [1., .5, .2, 1., 0., 0., 1.5, 1., 0., 0., 0., 0., 0., 0., 0., 0.]
    shaders = {"vertex": "#version 300 es\nvoid main() {}",
               "fragment": "#version 300 es\nvoid main() {}"}
    return {
        "schema": "turing.studio-class-world-import.v1", "revision": 3,
        "classIdentity": identity,
        "placement": {"translationM": [40., 0., -25.], "axes": "Y-up", "scale": 1},
        "scene": {"schema": "turing.studio-visible-scene.v1", "channel": "studio",
                  "frame": {"unitM": .5, "rayOriginM": [0, 0, 100]},
                  "room": {"leftX": -4, "rightX": 4, "backZ": -6, "frontZ": 4,
                           "floorY": -2, "ceilingY": 4},
                  "parts": [{"id": "room/front", "faces": [deepcopy(face)]}],
                  "subject": {"id": "subject", "faces": [deepcopy(face)]},
                  "ray": {"pbr": pbr, "material_ids": [0]},
                  "collisionReady": False},
        "render": {"shaders": shaders,
                   "shaderHashes": {key: digest(value.encode()) for key, value in shaders.items()},
                   "materials": {name: {"layout": {"floats_per_record": size},
                                         "data": pbr if name == "pbr" else [0.] * size}
                                 for name, size in (("pbr", 16), ("phong", 8), ("enamel", 8), ("texstack", 16))},
                   "uniforms": [{"name": "uExposure", "value": .7}]},
        "future_extension": {"preserve": [1, "exact", {"unknown": True}]},
    }


def compose(cache, packet=None, subject=CameraPublication):
    packet = packet or packet_for(subject)
    return append_studio_house(
        cache, subject, packet,
        room_bounds_m={key: value * packet["scene"]["frame"]["unitM"]
                       for key, value in packet["scene"]["room"].items()},
        placement_m=packet["placement"]["translationM"],
    )


@pytest.fixture
def cached(tmp_path):
    # A transport-only fixture, never evidence of engine runtime correctness.
    wasm = b"\0asm\x01\0\0\0"
    key = digest(wasm)
    worker = "self.onmessage = event => self.postMessage(event.data);"
    model = {
        "identity": WORLD_ID, "regions": [{"identity": "region:original", "buildings": []}],
        "tracks": [{"source": "old-a", "target": "old-b", "authored": True}],
        "document_geometry": {"boxes": [{"identity": "track", "center": [1, 2]}]},
        "vehicle_slot": {"untouched": [1, 2, 3]}, "projectiles": {"original": True},
        "physics_program": {"original": True}, "viewer": {"camera": {"scale": .25}},
        "loop_deployment": {"workers": [{"source": worker}]},
        "world": {"identity": WORLD_ID + "/world-registry", "root": WORLD_ID,
                  "objects": [{"identity": "old-a", "parent": WORLD_ID,
                               "semantic_parts": [{"identity": "old-a/surface"}]}],
                  "object_order": ["old-a"], "structural_object_order": ["old-a"],
                  "conceptual_object_order": [], "plugins": [{"entrypoint": "old"}],
                  "wasm_modules": [{"content_key": "wasm:sha256:" + key,
                                    "binary_base64": base64.b64encode(wasm).decode()}],
                  "identity_specialization": {"authority": "authored-string-identity", "missing_runtime_id": 0,
                      "objects": [{"identity": "old-a", "runtime_id": 7}],
                      "semantic_parts": [{"identity": "old-a/surface", "runtime_id": 11, "object_runtime_id": 7}]}},
    }
    encoded = json.dumps(model).encode()
    files = {"model.json": encoded, "physics-worker.js": worker.encode(),
             "behavior.js": b"/* original behavior */", "presentation.css": b"/* original style */",
             f"wasm/{key}.wasm": wasm,
             "index.html": b'<script id="abstract-ui-model" type="application/json">' + encoded + b"</script>"}
    receipt = {"schema": "living-map-matched-extraction-v1", "source_world": WORLD_ID,
               "source_sha256": digest(files["index.html"]), "source": "fixture-only",
               "artifacts": [{"path": name, "bytes": len(data), "sha256": digest(data)}
                             for name, data in files.items()]}
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    return load_verified_living_map(tmp_path, expected_html_sha256=receipt["source_sha256"]), tmp_path


def test_preserves_original_world_and_compiled_products(cached, monkeypatch):
    cache, _ = cached
    before = deepcopy(cache.model)
    import src.compiler.abstract_ui_div_map as projector

    def forbidden(*args, **kwargs):
        raise AssertionError("Composition must not bake or regenerate the original lot")

    for name in ("world_div_map_model", "document_geometry", "symbolic_world_physics_wasm_plugin",
                 "symbolic_vehicle_physics_wasm_plugin", "symbolic_wheel_contact_wasm_plugin"):
        monkeypatch.setattr(projector, name, forbidden)
    result = compose(cache)
    assert cache.model == before
    for key, value in before.items():
        if key not in {"regions", "world"}:
            assert result.model[key] == value
    assert result.model["regions"][:-1] == before["regions"]
    old_world, new_world = before["world"], result.model["world"]
    for key, value in old_world.items():
        if key in {"objects", "object_order", "conceptual_object_order"}:
            assert new_world[key][:len(value)] == value
        elif key == "identity_specialization":
            for lane, old in value.items():
                if isinstance(old, list):
                    assert new_world[key][lane][:len(old)] == old
                else:
                    assert new_world[key][lane] == old
        else:
            assert new_world[key] == value
    assert result.receipt["ready"] is False
    assert result.receipt["compiled"] is False
    assert "front wall" in " ".join(result.receipt["remaining"])
    assert new_world["identity_specialization"]["objects"][1]["runtime_id"] == 8
    assert new_world["identity_specialization"]["semantic_parts"][1]["runtime_id"] == 12


def test_real_introspector_rooms_and_exact_detached_packet(cached):
    from src.compiler.abstract_ui_div_map import _room_model
    cache, _ = cached
    packet = packet_for()
    result = compose(cache, packet)
    building = build_introspective_world(CameraPublication).regions[0].buildings[0]
    added = result.model["regions"][-1]["buildings"][0]
    assert added["rooms"] == [_room_model(room, building.identity) for room in building.rooms]
    house = next(item for item in result.model["world"]["objects"] if item["identity"] == result.house_identity)
    attachment = house["extensions"]["turing.studio_world_import"]
    assert attachment["packet"] == packet
    assert house["transform"]["position"] == [40., 0., -25.]
    assert attachment["room_bounds_m"]["floorY"] == -1
    assert house["physics"]["enabled"] is False
    assert "old-a" in result.model["world"]["structural_object_order"]
    assert result.house_identity not in result.model["world"]["structural_object_order"]
    packet["future_extension"]["preserve"].append("edited later")
    assert attachment["packet"] != packet


def test_rejects_changed_cache_and_wrong_source_digest(cached):
    cache, directory = cached
    with pytest.raises(ValueError, match="selected original"):
        load_verified_living_map(directory, expected_html_sha256="0" * 64)
    (directory / "physics-worker.js").write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        load_verified_living_map(directory, expected_html_sha256=cache.source_receipt["source_sha256"])
    cache.model["vehicle_slot"]["untouched"].append(4)
    with pytest.raises(ValueError, match="mutated"):
        compose(cache)


@pytest.mark.parametrize("change", [
    lambda p: p.update(classIdentity="python:unrelated.Class"),
    lambda p: p["scene"].update(channel="exposure"),
    lambda p: p["scene"]["parts"].append(deepcopy(p["scene"]["parts"][0])),
    lambda p: p["scene"]["parts"][0]["faces"][0].__setitem__(3, 42),
    lambda p: p["render"]["shaderHashes"].update(vertex="0" * 64),
    lambda p: p["render"]["materials"]["phong"]["data"].append(1),
    lambda p: p["render"]["materials"]["pbr"]["data"].__setitem__(0, .2),
    lambda p: p.update(future_extension=float("nan")),
])
def test_rejects_unmatched_scene_or_materials(cached, change):
    cache, _ = cached
    packet = packet_for()
    # PBR fixture has shared lists; detach like actual JSON transport first.
    packet = json.loads(json.dumps(packet))
    change(packet)
    with pytest.raises(ValueError):
        compose(cache, packet)


def test_requires_matching_physical_bounds_and_placement(cached):
    cache, _ = cached
    packet = packet_for()
    bounds = {key: value * .5 for key, value in packet["scene"]["room"].items()}
    with pytest.raises(ValueError, match="placement"):
        append_studio_house(cache, CameraPublication, packet, room_bounds_m=bounds, placement_m=[0, 0, 0])
    bounds["leftX"] -= 1
    with pytest.raises(ValueError, match="bound"):
        append_studio_house(cache, CameraPublication, packet, room_bounds_m=bounds, placement_m=[40, 0, -25])


def test_rejects_existing_region_and_changed_receipt(cached, monkeypatch):
    cache, _ = cached
    monkeypatch.setattr(CameraPublication, "__module__", "original")
    with pytest.raises(ValueError, match="already exists"):
        compose(cache)
    cache.source_receipt["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="receipt was mutated"):
        compose(cache)


def test_real_cached_world_and_intrinsic_camera_class_when_available():
    root = Path(__file__).resolve().parents[1]
    directory = root / "build/dot-living-map-runtime-20261110/matched"
    source = root.parent / "spectral-analyzer/camera_software/exposure_timing.py"
    if not directory.is_dir() or not source.is_file():
        pytest.skip("Optional exact original local artifact and intrinsic camera source are absent")
    # Load the actual stdlib-only source, avoiding unrelated package __init__ UI imports.
    name = "camera_software.exposure_timing"
    existing = sys.modules.get(name)
    try:
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        cache = load_verified_living_map(directory)
        result = compose(cache, subject=module.SceneCameraCoordinator)
        assert result.house_identity == "python:camera_software.exposure_timing.SceneCameraCoordinator"
        assert len(result.model["world"]["wasm_modules"]) == 7
        assert result.model["document_geometry"] == cache.model["document_geometry"]
        assert result.model["loop_deployment"] == cache.model["loop_deployment"]
        assert result.model["vehicle_slot"] == cache.model["vehicle_slot"]
        assert len(result.model["world"]["objects"]) == len(cache.model["world"]["objects"]) + 7
        node = shutil.which("node")
        if node:
            # Run the actual deployed registry, not a replacement validator.
            script = r"""
const fs=require('node:fs'),vm=require('node:vm');
const behavior=fs.readFileSync(process.argv[1],'utf8');
const start=behavior.indexOf('class TuringWorldRegistry {');
const end=behavior.indexOf('function turingCreateWorldRegistry(',start);
if(start<0||end<0)throw Error('Original registry boundary is absent');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const registry=vm.runInNewContext(behavior.slice(start,end)+'\nnew TuringWorldRegistry(world)',{world:input.world});
for(const row of input.original.objects)if(registry.objectRuntimeId(row.identity)!==row.runtime_id)
  throw Error('An original object runtime ID changed');
for(const row of input.original.semantic_parts)if(registry.partRuntimeId(row.identity)!==row.runtime_id)
  throw Error('An original part runtime ID changed');
if(registry.objects.size!==input.world.objects.length)throw Error('An appended object was lost');
process.stdout.write(JSON.stringify({objects:registry.objects.size,accepted:true}));
"""
            checked = subprocess.run(
                [node, "-e", script, str(directory / "behavior.js")],
                input=json.dumps({"world": result.model["world"],
                                  "original": cache.model["world"]["identity_specialization"]}),
                capture_output=True, text=True, check=True, timeout=10,
            )
            assert json.loads(checked.stdout)["accepted"] is True
    finally:
        if existing is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = existing
