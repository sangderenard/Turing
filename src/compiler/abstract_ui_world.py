"""Pluck-compatible world objects and WebAssembly plugin manifests.

AbstractUI owns the portable contract.  Pluck and document backends adapt
their authored records into it without surrendering their complete source
payloads or making a renderer authoritative for object identity.
"""

from __future__ import annotations

import base64
import contextlib
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import io
from typing import Any, Iterable, Mapping
import warnings

import numpy as np

from ..common.tensors.accelerator_backends.aot_compile import (
    compile_ast_aot,
    project_public_numerical_program,
)
from .fused_program_wasm_backend import emit_wasm_module


ABSTRACT_UI_WORLD_VERSION = "abstract-ui-world-v0"
WORLD_MESH_PACKET_VERSION = "abstract-ui-world-mesh-packet-v0"
WORLD_PLUGIN_VERSION = "abstract-ui-world-plugin-v0"


WORLD_TRANSFORM_SOURCE = """\
def transform_world_vertex(local_x, local_y, local_z,
                           translate_x, translate_y, translate_z,
                           yaw_cos, yaw_sin):
    world_x = translate_x + local_x * yaw_cos - local_z * yaw_sin
    world_y = translate_y + local_y
    world_z = translate_z + local_x * yaw_sin + local_z * yaw_cos
    return world_x, world_y, world_z
"""

_WORLD_TRANSFORM_PARAMETERS = (
    "local_x", "local_y", "local_z", "translate_x", "translate_y",
    "translate_z", "yaw_cos", "yaw_sin",
)


def _json_value(value: Any) -> Any:
    """Return a JSON-safe copy while retaining extension field structure."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True, slots=True)
class WorldObject:
    """One persistent conceptual object, independent of its realization."""

    identity: str
    kind: str
    parent: str
    label: str
    transform: Mapping[str, Any]
    form: Mapping[str, Any]
    material_bindings: Mapping[str, str] = field(default_factory=dict)
    capabilities: tuple[str, ...] = ()
    semantic_parts: tuple[Mapping[str, Any], ...] = ()
    physics: Mapping[str, Any] = field(default_factory=dict)
    persistence: Mapping[str, Any] = field(default_factory=dict)
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.identity or not self.kind or not self.parent:
            raise ValueError("world objects require identity, kind, and parent")

    def to_data(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "kind": self.kind,
            "parent": self.parent,
            "label": self.label,
            "name": self.label,
            "transform": _json_value(self.transform),
            "form": _json_value(self.form),
            "material_bindings": dict(self.material_bindings),
            "capabilities": list(self.capabilities),
            "semantic_parts": [_json_value(part) for part in self.semantic_parts],
            "physics": _json_value(self.physics),
            "persistence": _json_value(self.persistence),
            "extensions": _json_value(self.extensions),
            "interaction": {"type": "inspect", "destination": self.identity},
            "dependencies": [{"relationship": "contained-by", "target": self.parent}],
        }


@dataclass(frozen=True, slots=True)
class WorldWasmPlugin:
    """A bounded numerical helper deployable beside a living world."""

    identity: str
    operation: str
    binary: bytes
    entrypoint: str
    parameters: tuple[dict[str, str], ...]
    source: str
    source_language: str = "python"
    capability: str = "geometry"
    operation_count: int = 0
    reserved_bytes: int = 0
    abi: Mapping[str, Any] = field(default_factory=dict)

    @property
    def content_key(self) -> str:
        return f"wasm:sha256:{hashlib.sha256(self.binary).hexdigest()}"

    def to_data(self, *, include_binary: bool = True) -> dict[str, Any]:
        result = {
            "schema": WORLD_PLUGIN_VERSION,
            "identity": self.identity,
            "module": self.content_key,
            "operation": self.operation,
            "capability": self.capability,
            "source_language": self.source_language,
            "source": self.source,
            "entrypoint": self.entrypoint,
            "parameters": [dict(parameter) for parameter in self.parameters],
            "binary_bytes": len(self.binary),
            "operation_count": self.operation_count,
            "reserved_bytes": self.reserved_bytes,
            "host_contract": {
                "memory": "plugin-owned-webassembly-linear-memory",
                "invocation": str(self.abi.get(
                    "invocation", "published-parameter-order",
                )),
                "authority": "helper-only-world-object-remains-authoritative",
            },
        }
        if self.abi:
            result["abi"] = _json_value(self.abi)
        if include_binary:
            result["binary_base64"] = base64.b64encode(self.binary).decode("ascii")
        return result


@lru_cache(maxsize=1)
def compile_world_transform_wasm() -> WorldWasmPlugin:
    """Compile Pluck-style position/yaw realization from authored Python."""

    samples = {
        name: np.ones(2, dtype=np.float64) for name in _WORLD_TRANSFORM_PARAMETERS
    }
    captured_output = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(captured_output):
        warnings.simplefilter("ignore")
        compilation = compile_ast_aot(
            WORLD_TRANSFORM_SOURCE,
            "transform_world_vertex",
            samples,
            backend="c",
            precompile_only=True,
            mutable_parameters=_WORLD_TRANSFORM_PARAMETERS,
        )
    program = project_public_numerical_program(compilation)
    module = emit_wasm_module(
        program,
        name="abstract_ui_world_transform",
        function_name="transform_world_vertex",
        dtype="float64",
    )
    if not module.complete or module.binary is None:
        raise RuntimeError(module.shortfall_report())
    entry = module.api.entry_points[0]
    return WorldWasmPlugin(
        "abstract-ui/plugins/world-transform",
        "transform-position-yaw",
        module.binary,
        entry.symbol,
        tuple({
            "name": parameter.name,
            "role": parameter.role,
            "dtype": parameter.dtype,
        } for parameter in entry.parameters),
        WORLD_TRANSFORM_SOURCE,
        operation_count=len(program.steps),
        reserved_bytes=int(module.api.metadata.get("reserved_bytes", 0)),
    )


def document_world_objects(
    system_root: str,
    geometry: Mapping[str, Any],
    *,
    external_owners: Iterable[str] = (),
) -> tuple[WorldObject, ...]:
    """Promote document geometry into editable Pluck-compatible objects."""

    objects: list[WorldObject] = []
    identities: set[str] = set()
    for box in geometry.get("boxes", ()):
        identity = str(box["identity"])
        if identity in identities:
            raise ValueError(f"duplicate world object identity {identity!r}")
        identities.add(identity)
        parent = str(box.get("parent_identity") or system_root)
        if str(box.get("geometry_mode")) == "solid":
            artifact = dict(box.get("artifact") or {})
            objects.append(WorldObject(
                identity=identity,
                kind=str(box.get("kind", "object")),
                parent=parent,
                label=str(box.get("label") or identity.rsplit("/", 1)[-1]),
                transform={
                    "position": [float(box["center"][0]), 0.0, float(box["center"][1])],
                    "yaw_deg": float(box.get("yaw_deg", 0.0)),
                    "coordinate_space": str(geometry.get("coordinate_space", "data-world")),
                    "placed_in": str(box.get("spatial_container") or parent),
                },
                form={
                    "recipe": "solid-box",
                    "half_extent": [float(value) for value in box["half_extent"]],
                    "height": float(box["height"]),
                    "radius": float(box.get("radius", 0.0)),
                },
                material_bindings={
                    "body": str(box.get("palette_role", "artifact-source")),
                },
                capabilities=tuple(artifact.get("capabilities") or (
                    "inspect", "move", "attach", "detach", "publish-mesh",
                )),
                semantic_parts=({
                    "identity": f"{identity}/surface:body",
                    "role": "body",
                    "material_role": str(box.get("palette_role", "artifact-source")),
                },),
                physics={
                    **dict(box.get("physics") or {}),
                    "collision_authority": "world-physics",
                    "enabled": True,
                },
                persistence={
                    "authority": "filesystem-and-living-document",
                    "revision": int(box.get("revision", 0)),
                    "override_store": "browser-cookie-with-local-storage-fallback",
                },
                extensions={
                    "abstract_ui.human_artifact": _json_value(artifact),
                    "abstract_ui.document_geometry": _json_value(box),
                },
            ))
            continue
        wall_parts = tuple({
            "identity": f"{identity}/surface:{side}",
            "role": "boundary-wall",
            "side": side,
            "material_role": str(box.get("wall_palette_role", "line")),
        } for side in ("south", "north", "west", "east"))
        opening_parts = tuple({
            "identity": str(opening["identity"]),
            "role": "opening",
            "opening_kind": str(opening["kind"]),
            "side": str(opening["side"]),
        } for opening in box.get("openings", ()))
        objects.append(WorldObject(
            identity=identity,
            kind=str(box.get("kind", "object")),
            parent=parent,
            label=str(box.get("label") or identity.rsplit("/", 1)[-1]),
            transform={
                "position": [float(box["center"][0]), 0.0, float(box["center"][1])],
                "yaw_deg": float(box.get("yaw_deg", 0.0)),
                "coordinate_space": str(geometry.get("coordinate_space", "data-world")),
            },
            form={
                "recipe": "boundary-floor-with-openings",
                "half_extent": [float(value) for value in box["half_extent"]],
                "height": float(box["height"]),
                "floor_height": float(box.get("floor_height", 0.035)),
                "wall_thickness": float(box.get("wall_thickness", 0.04)),
                "radius": float(box.get("radius", 0.0)),
                "openings": _json_value(box.get("openings", ())),
            },
            material_bindings={
                "floor": str(box.get("palette_role", "room-face")),
                "walls": str(box.get("wall_palette_role", "room-wall")),
            },
            capabilities=(
                "inspect", "edit-form", "contain", "receive-openings",
                "publish-mesh", "collide-static",
            ),
            semantic_parts=(
                {
                    "identity": f"{identity}/surface:floor",
                    "role": "floor",
                    "material_role": str(box.get("palette_role", "room-face")),
                },
                *wall_parts,
                *opening_parts,
            ),
            physics={
                "body": "static",
                "collider": "boundary-shell-plus-floor",
                "collision_authority": "world-physics",
                "enabled": True,
            },
            persistence={
                "authority": "living-document",
                "revision": int(box.get("revision", 0)),
                "override_store": "browser-cookie-with-local-storage-fallback",
            },
            extensions={
                "abstract_ui.document_geometry": _json_value(box),
                "pluck.compatibility": {
                    "placed_object": True,
                    "room_workspace_registry": True,
                    "material_binding_roles": True,
                    "triangle_group_actions": True,
                    "region_authority": parent,
                },
            },
        ))
    known = identities | {system_root, *external_owners}
    missing = sorted({item.parent for item in objects} - known)
    if missing:
        raise ValueError(f"world objects have missing containers: {missing}")
    return tuple(objects)


def pluck_placed_object(
    payload: Mapping[str, Any],
    *,
    parent: str,
) -> WorldObject:
    """Losslessly adapt ``PlacedObject.to_dict()`` data into AbstractUI.

    The complete source record rides in the namespaced extension.  Recognized
    fields additionally populate the portable surface; unknown game metadata
    therefore survives newer Pluck object kinds and revisions.
    """

    source = _json_value(payload)
    identity = str(payload.get("id", ""))
    kind = str(payload.get("type", "object"))
    if not identity:
        raise ValueError("Pluck placed objects require id")
    bindings = {
        str(role): str(material)
        for role, material in dict(payload.get("material_bindings", {})).items()
    }
    form: dict[str, Any] = {"recipe": f"pluck:{kind}"}
    semantic_parts: list[dict[str, Any]] = []
    physics: dict[str, Any] = {
        "body": "static", "collider": "object-bounds", "enabled": True,
    }
    capabilities = ["inspect", "select", "publish-mesh"]
    if kind == "enclosure":
        form.update({"shape": payload.get("shape", "rect"), "parameters": source.get("dims", {})})
        semantic_parts.extend((
            {"identity": f"{identity}/surface:glass", "role": "shell", "material_role": "glass"},
            {"identity": f"{identity}/surface:skirt", "role": "base", "material_role": "skirt"},
        ))
        capabilities.extend(("contain", "collide-static"))
    elif kind == "camera":
        form.update({"mesh_preset": payload.get("mesh_id", "camera_35mm")})
        capabilities.extend(("enter-camera", "aim", "focus"))
    elif kind == "duty_station":
        form.update({"station_type": payload.get("station_type", "fabricator")})
        capabilities.extend(("interact", "host-dialogue", "receive-materials"))
    elif kind == "portal":
        form.update({"target_region": payload.get("target_room_id", "")})
        capabilities.extend(("traverse", "connect-regions"))
        physics["collider"] = "portal-frame"
    elif kind == "light":
        form.update({"light_kind": payload.get("kind", "point")})
        capabilities.extend(("illuminate", "edit-light"))
        physics = {"body": "none", "collider": "none", "enabled": False}
    return WorldObject(
        identity,
        kind,
        str(parent),
        str(payload.get("label", identity)),
        {
            "position": source.get("pos", [0.0, 0.0, 0.0]),
            "yaw_deg": float(payload.get("yaw_deg", 0.0)),
            "coordinate_space": "pluck-world-metres",
        },
        form,
        bindings,
        tuple(dict.fromkeys(capabilities)),
        tuple(semantic_parts),
        physics,
        {"authority": "pluck-room-workspace", "revision": 0},
        {"pluck.placed_object": source},
    )


def performance_observation_objects(
    labels: Iterable[Mapping[str, Any]],
    parent_by_identity: Mapping[str, str],
) -> tuple[WorldObject, ...]:
    """Materialize performance labels as loose objects inside code domains."""

    observations = []
    for ordinal, raw in enumerate(labels):
        label = _json_value(raw)
        label_identity = str(label.get("identity", ""))
        parent = str(parent_by_identity.get(label_identity, ""))
        if not label_identity or not parent:
            continue
        observations.append(WorldObject(
            identity=f"{parent}/observations/performance:{ordinal}",
            kind="performance-observation",
            parent=parent,
            label=f"inline {label.get('inline', 'neutral')}",
            transform={
                "placement": "ambient-domain",
                "ordinal": ordinal,
                "coordinate_space": "conceptual-interior",
            },
            form={
                "recipe": "performance-marker",
                "shape": "loose-token",
                "scale": 0.18,
                "color_role": (
                    "performance-hot" if label.get("hot_path")
                    else f"inline-{label.get('inline', 'neutral')}"
                ),
            },
            capabilities=("inspect", "compare-performance", "trace-source"),
            physics={"body": "none", "collider": "none", "enabled": False},
            persistence={"authority": "derived-emission", "revision": 0},
            extensions={"turing.performance": label},
        ))
    return tuple(observations)


def identity_specialization_table(
    objects: Iterable[WorldObject],
) -> dict[str, Any]:
    """Lower authored identities to compact hot-path IDs without losing meaning.

    Zero is deliberately reserved as the missing/unresolved sentinel used by
    GPU buffers and packed ABI records.  Appearance order is the stable input
    to this lowering, so the table can be rebuilt deterministically for every
    published world revision and reversed at any host boundary.
    """

    object_values = tuple(objects)
    object_entries = [
        {"runtime_id": index, "identity": item.identity}
        for index, item in enumerate(object_values, start=1)
    ]
    part_entries: list[dict[str, Any]] = []
    seen_parts: set[str] = set()
    for object_runtime_id, item in enumerate(object_values, start=1):
        for part in item.semantic_parts:
            identity = str(part["identity"])
            if identity in seen_parts:
                raise ValueError(f"duplicate semantic part identity {identity!r}")
            seen_parts.add(identity)
            part_entries.append({
                "runtime_id": len(part_entries) + 1,
                "identity": identity,
                "object_runtime_id": object_runtime_id,
            })
    return {
        "schema": "abstract-ui-identity-specialization-v0",
        "policy": "stable-appearance-order-to-dense-u32",
        "missing_runtime_id": 0,
        "lifetime": "world-revision-or-mesh-bake",
        "authority": "authored-string-identity",
        "reversible": True,
        "objects": object_entries,
        "semantic_parts": part_entries,
    }


def world_graph_model(
    system_root: str,
    geometry: Mapping[str, Any],
    *,
    plugins: Iterable[WorldWasmPlugin] = (),
    performance_labels: Iterable[Mapping[str, Any]] = (),
    performance_parents: Mapping[str, str] | None = None,
    external_owners: Iterable[str] = (),
) -> dict[str, Any]:
    """Assemble the portable world registry and realization ABI."""

    external_owner_values = tuple(external_owners)
    structural_objects = document_world_objects(
        system_root, geometry, external_owners=external_owner_values,
    )
    observations = performance_observation_objects(
        performance_labels, performance_parents or {},
    )
    objects = structural_objects + observations
    identity_specialization = identity_specialization_table(objects)
    known_identities = {
        system_root, *external_owner_values, *(item.identity for item in objects),
    }
    missing_parents = sorted({item.parent for item in objects} - known_identities)
    if missing_parents:
        raise ValueError(
            f"performance observations have missing domains: {missing_parents}"
        )
    plugin_values = tuple(plugins)
    modules = {}
    for plugin in plugin_values:
        modules.setdefault(plugin.content_key, {
            "content_key": plugin.content_key,
            "format": "webassembly",
            "binary_base64": base64.b64encode(plugin.binary).decode("ascii"),
            "binary_bytes": len(plugin.binary),
        })
    return {
        "schema": ABSTRACT_UI_WORLD_VERSION,
        "identity": f"{system_root}/world-registry",
        "root": system_root,
        "coordinate_space": str(geometry.get("coordinate_space", "data-world")),
        "object_order": [item.identity for item in objects],
        "structural_object_order": [item.identity for item in structural_objects],
        "observation_object_order": [item.identity for item in observations],
        "objects": [item.to_data() for item in objects],
        "identity_specialization": identity_specialization,
        "mesh_packet": {
            "schema": WORLD_MESH_PACKET_VERSION,
            "topology": "triangle-list",
            "vertex_layout": ["position.xyz", "normal.xyz", "color.rgb"],
            "identity_table": "variable-length-object-spans",
            "semantic_part_table": "variable-length-part-spans",
            "material_binding_table": "world-object-material-bindings",
            "revision_source": "living-document-edit-revision",
            "authority": "world-object-recipes-not-renderer-buffers",
        },
        "systems": {
            "containment": "parent-is-authority",
            "actions": "triangle-part-to-action-edge",
            "physics": "reserved-world-physics-lane",
            "persistence": "living-document-plus-representation-overrides",
        },
        "wasm_modules": [modules[key] for key in sorted(modules)],
        "plugins": [plugin.to_data(include_binary=False) for plugin in plugin_values],
        "compatibility": {
            "pluck": {
                "placed_objects": "lossless-namespaced-extension",
                "room_workspace": "object-registry-and-container-authority",
                "procedural_meshes": "form-recipe-plugins",
                "render_assets": "content-addressed-bake-artifacts",
                "triangle_group_actions": "semantic-part-action-edges",
            },
        },
    }


def wasm_artifact_plugin(
    identity: str,
    operation: str,
    artifact: Any,
    *,
    source: str,
    capability: str,
) -> WorldWasmPlugin:
    """Publish an existing compiler artifact through the common plugin ABI."""

    return WorldWasmPlugin(
        str(identity),
        str(operation),
        artifact.binary,
        artifact.entrypoint,
        tuple(dict(parameter) for parameter in artifact.parameters),
        str(source),
        capability=str(capability),
        operation_count=int(artifact.operation_count),
        reserved_bytes=int(artifact.reserved_bytes),
    )


__all__ = [
    "ABSTRACT_UI_WORLD_VERSION", "WORLD_MESH_PACKET_VERSION",
    "WORLD_PLUGIN_VERSION", "WORLD_TRANSFORM_SOURCE", "WorldObject",
    "WorldWasmPlugin", "compile_world_transform_wasm",
    "document_world_objects", "identity_specialization_table",
    "performance_observation_objects",
    "pluck_placed_object", "wasm_artifact_plugin", "world_graph_model",
]
