"""Append an introspected Studio house to a verified, already compiled world.

This is model preparation, not a second world or a renderer/physics fallback.
The original layout, shell, worker and compiled products remain authoritative.
No compiler, browser, server, class instance or scene realization is started.
"""

from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from src.compiler.abstract_ui_introspection import build_introspective_world
from src.compiler.abstract_ui_world import WorldObject


DEPLOYED_HTML_SHA256 = "2cd5c5e23f1491243346dd46586f5ebc10a4391187a4e991fa8f151f8dfd4ef4"
WORLD_ID = "introspection-world:python:src.compiler.mechanical_creature.MechanicalCreature"
ROOM_KEYS = ("leftX", "rightX", "backZ", "frontZ", "floorY", "ceilingY")
PENDING = (
    "placed Studio material-ID renderer and world camera/light transforms",
    "entry realization: captured front wall has no admitted doorway or portal",
    "original engine contact-shape realization and ownership admission",
    "same-world browser validation",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class VerifiedLivingMap:
    model: dict[str, Any]
    source_receipt: dict[str, Any]
    canonical_model_sha256: str
    canonical_receipt_sha256: str


@dataclass(frozen=True)
class StudioHouseImport:
    model: dict[str, Any]
    receipt: dict[str, Any]
    house_identity: str


def load_verified_living_map(
    directory: Path | str, *, expected_html_sha256: str = DEPLOYED_HTML_SHA256,
) -> VerifiedLivingMap:
    """Read the matched extraction, checking it against an explicit source hash.

    The receipt authenticates no publisher; the caller's expected source digest
    pins the chosen original page. All receipt-listed artifacts must still match.
    """
    root = Path(directory).resolve()
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    if (receipt.get("schema") != "living-map-matched-extraction-v1"
            or receipt.get("source_sha256") != expected_html_sha256
            or receipt.get("source_world") != WORLD_ID):
        raise ValueError("The extraction does not match the selected original world")
    artifacts = {}
    for entry in receipt["artifacts"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root) or entry["path"] in artifacts:
            raise ValueError("Invalid or duplicate extraction artifact path")
        data = path.read_bytes()
        if len(data) != entry["bytes"] or _sha(data) != entry["sha256"]:
            raise ValueError(f"Extraction artifact changed: {entry['path']}")
        artifacts[entry["path"]] = data
    if _sha(artifacts["index.html"]) != expected_html_sha256:
        raise ValueError("Original page hash differs from the selected source")
    # The extracted JSON must be the exact script body of that same full page.
    html = artifacts["index.html"]
    marker = html.index(b"abstract-ui-model")
    start = html.index(b">", marker) + 1
    end = html.index(b"</script>", start)
    if html[start:end] != artifacts["model.json"]:
        raise ValueError("Model and full source page are not the same compiled artifact")
    model = json.loads(artifacts["model.json"])
    if model.get("identity") != WORLD_ID or model["world"].get("root") != WORLD_ID:
        raise ValueError("Unexpected cached world root")
    worker = model["loop_deployment"]["workers"][0]["source"].encode("utf-8")
    if worker != artifacts["physics-worker.js"]:
        raise ValueError("Worker and cached model are not a matched pair")
    for module in model["world"]["wasm_modules"]:
        binary = base64.b64decode(module["binary_base64"], validate=True)
        digest = _sha(binary)
        if (module["content_key"] != f"wasm:sha256:{digest}"
                or binary != artifacts[f"wasm/{digest}.wasm"]):
            raise ValueError("A cached compiled module has lost its source identity")
    return VerifiedLivingMap(model, receipt, _sha(_json_bytes(model)), _sha(_json_bytes(receipt)))


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_packet(packet: Mapping[str, Any], identity: str,
                     room_bounds_m: Mapping[str, float], placement_m: Sequence[float]) -> None:
    if (packet.get("schema") != "turing.studio-class-world-import.v1"
            or packet.get("classIdentity") != identity):
        raise ValueError("Studio packet must name this actual introspected class")
    scene = packet["scene"]
    if scene.get("schema") != "turing.studio-visible-scene.v1" or scene.get("channel") != "studio":
        raise ValueError("A displayed Studio scene is required, not an Exposure scene")
    if (len(placement_m) != 3 or not all(map(_finite, placement_m))
            or packet.get("placement") != {
                "translationM": list(placement_m), "axes": "Y-up", "scale": 1}):
        raise ValueError("Explicit placement must match the retained Studio packet")
    unit = scene["frame"]["unitM"]
    if not _finite(unit) or unit <= 0:
        raise ValueError("Studio unit conversion must be positive and finite")
    for key in ROOM_KEYS:
        value = room_bounds_m.get(key)
        if not _finite(value) or not math.isclose(value, scene["room"][key] * unit,
                                                rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Physical room bound {key} does not match the captured Studio room")
    for low, high in zip(ROOM_KEYS[::2], ROOM_KEYS[1::2]):
        if room_bounds_m[low] >= room_bounds_m[high]:
            raise ValueError("Studio room bounds must have positive spans")
    parts = scene.get("parts")
    if not isinstance(parts, list) or not parts or not scene.get("subject"):
        raise ValueError("The full authored Studio scene and displayed subject are required")
    ids = [part.get("id") for part in parts]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Studio source part identities must be nonempty and unique")
    if any(not part.get("faces") for part in parts) or not scene["subject"].get("faces"):
        raise ValueError("Authored faces must be retained, not replaced with room bounds")
    render = packet["render"]
    count = None
    for name in ("pbr", "phong", "enamel", "texstack"):
        table = render["materials"][name]
        stride = table["layout"]["floats_per_record"]
        data = table["data"]
        if not isinstance(stride, int) or stride <= 0 or not data or len(data) % stride:
            raise ValueError("Studio material table is incomplete")
        rows = len(data) // stride
        if count is not None and count != rows:
            raise ValueError("Studio material row identities do not agree")
        count = rows
    pbr = render["materials"]["pbr"]
    stride = pbr["layout"]["floats_per_record"]
    if stride < 16 or scene["ray"]["pbr"] != [
        value for offset in range(0, len(pbr["data"]), stride)
        for value in pbr["data"][offset:offset + 16]
    ]:
        raise ValueError("Captured materials differ from the actual Studio render state")
    for part in [*parts, scene["subject"]]:
        for face in part["faces"]:
            material = face[3]
            if not isinstance(material, int) or isinstance(material, bool) or not 0 <= material < count:
                raise ValueError("A source triangle references an absent material row")
    for name in ("vertex", "fragment"):
        source = render["shaders"][name]
        if (not source.startswith("#version 300 es")
                or _sha(source.encode("utf-8")) != render["shaderHashes"][name]):
            raise ValueError("Actual Studio shader source hash does not match")
    _json_bytes(packet)  # Reject non-JSON/nonfinite payloads without reinterpreting them.


def append_studio_house(
    cache: VerifiedLivingMap, subject: type, studio_packet: Mapping[str, Any], *,
    room_bounds_m: Mapping[str, float], placement_m: Sequence[float],
) -> StudioHouseImport:
    """Return an additive model; leave every pre-existing subtree unchanged.

    Class member rooms remain semantic inspection records. They do not fabricate
    partitions in the captured Studio. Bounds describe placement, never triangle
    contact admission. The packet is retained verbatim as JSON data in the house's
    ``extensions['turing.studio_world_import']['packet']``.
    """
    if _sha(_json_bytes(cache.model)) != cache.canonical_model_sha256:
        raise ValueError("Verified cached model was mutated before composition")
    if _sha(_json_bytes(cache.source_receipt)) != cache.canonical_receipt_sha256:
        raise ValueError("Verified source receipt was mutated before composition")
    if not isinstance(subject, type):
        raise TypeError("Use a real Python class, never a constructed instance")
    # These canonical serializers are pure. Do not call world_div_map_model:
    # that path also bakes geometry, world physics and vehicle kernels.
    from src.compiler.abstract_ui_div_map import _position, _room_model

    introspection = build_introspective_world(subject, depth_up=0, depth_down=0)
    region = introspection.regions[0]
    building = region.buildings[0]
    _validate_packet(studio_packet, building.identity, room_bounds_m, placement_m)
    original = cache.model
    existing_ids = {item["identity"] for item in original["world"]["objects"]}
    existing_ids.update(item["identity"] for item in original["regions"])
    added_ids = [region.identity, building.identity, *(room.identity for room in building.rooms)]
    if len(set(added_ids)) != len(added_ids) or existing_ids.intersection(added_ids):
        raise ValueError("Class/region/member already exists; refusing to replace any original subtree")
    rooms = [_room_model(room, building.identity) for room in building.rooms]
    building_record = {
        "identity": building.identity, "kind": "building", "name": building.name,
        "module": building.module, "source_kind": building.source_kind,
        "metaphor": building.metaphor, "position": _position(building.position),
        "dependencies": [{"relationship": "contained-by", "target": region.identity}],
        "rooms": rooms,
    }
    region_record = {
        "identity": region.identity, "kind": "region", "name": region.name,
        "position": _position(region.position),
        "dependencies": [{"relationship": "contained-by", "target": original["identity"]}],
        "buildings": [building_record],
    }
    attachment = {
        "schema": "turing.studio-house-attachment.v1", "packet": deepcopy(dict(studio_packet)),
        "room_bounds_m": {key: room_bounds_m[key] for key in ROOM_KEYS},
        "placement_m": list(placement_m),
        "coordinate_contract": "world = placement_m + Studio_position * scene.frame.unitM; Y-up",
        "ray_origin": "exposure-ray origin is retained but is not a world placement transform",
        "ready": False, "remaining": list(PENDING),
        "source_receipt": deepcopy(cache.source_receipt),
    }
    parts = tuple({
        "identity": f"{building.identity}/studio-part:{quote(part['id'], safe='')}",
        "role": "authored-studio-part", "source_part_id": part["id"],
        "material_authority": "retained per-triangle material IDs and Studio material tables",
    } for part in studio_packet["scene"]["parts"])
    parts += ({"identity": f"{building.identity}/studio-subject",
               "role": "displayed-studio-subject", "source_part_id": studio_packet["scene"]["subject"]["id"]},)
    objects = [WorldObject(
        region.identity, "region", original["identity"], region.name, {},
        {"recipe": "introspected-module-scope", "spatial_realization": "pending"},
        capabilities=("inspect",), physics={"enabled": False},
    ), WorldObject(
        building.identity, "building", region.identity, building.name,
        {"position": list(placement_m), "coordinate_space": "data-world", "axes": "Y-up"},
        {"recipe": "studio-visible-scene", "room_bounds_m": attachment["room_bounds_m"],
         "spatial_realization": "pending"},
        capabilities=("inspect",), semantic_parts=parts,
        physics={"enabled": False, "admission": "pending-original-engine-shape-realization"},
        extensions={"turing.studio_world_import": attachment,
                    "abstract_ui.introspection": building_record},
    )]
    objects.extend(WorldObject(
        room["identity"], "introspection-member", building.identity, room["name"], {},
        {"recipe": "introspection-member", "spatial_realization": "unassigned"},
        capabilities=("inspect",), physics={"enabled": False},
        extensions={"abstract_ui.introspection": room},
    ) for room in rooms)
    result = deepcopy(original)
    result["regions"].append(region_record)
    world = result["world"]
    world["objects"].extend(obj.to_data() for obj in objects)
    world["object_order"].extend(added_ids)
    world.setdefault("conceptual_object_order", []).extend(added_ids)
    specialization = world["identity_specialization"]
    if specialization.get("authority") != "authored-string-identity" or specialization.get("missing_runtime_id") != 0:
        raise ValueError("Original world has an unsupported identity specialization")
    object_id = max((item["runtime_id"] for item in specialization["objects"]), default=0)
    part_id = max((item["runtime_id"] for item in specialization["semantic_parts"]), default=0)
    old_parts = {item["identity"] for item in specialization["semantic_parts"]}
    for obj in objects:
        object_id += 1
        specialization["objects"].append({"runtime_id": object_id, "identity": obj.identity})
        for part in obj.semantic_parts:
            if part["identity"] in old_parts:
                raise ValueError("Studio semantic part identity collides with the existing world")
            old_parts.add(part["identity"])
            part_id += 1
            specialization["semantic_parts"].append({
                "runtime_id": part_id, "identity": part["identity"], "object_runtime_id": object_id})
    if object_id >= 2**32 or part_id >= 2**32:
        raise ValueError("World identity specialization exceeds its u32 ABI")
    receipt = {
        "schema": "turing.studio-house-composition.v1", "world_identity": original["identity"],
        "house_identity": building.identity, "added_object_identities": added_ids,
        "source_html_sha256": cache.source_receipt["source_sha256"],
        "source_model_canonical_sha256": cache.canonical_model_sha256,
        "result_model_canonical_sha256": _sha(_json_bytes(result)),
        "studio_packet_canonical_sha256": _sha(_json_bytes(studio_packet)),
        "source_receipt": deepcopy(cache.source_receipt),
        "original_layout_and_compiled_products_unchanged": True,
        "ready": False, "remaining": list(PENDING),
        "compiled": False, "published": False, "browser_started": False,
    }
    return StudioHouseImport(result, receipt, building.identity)
