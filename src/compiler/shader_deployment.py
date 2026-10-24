"""Deployable shader products: exact source, its API sidecar, and the
physical launch contract, published together under one component identity.

``docs/NODUS_SHADER_DEPLOYMENT_REQUEST.md`` asks for a produced directory a
consumer can run without joining unrelated in-memory objects or guessing
declarations from shader text. Every emitter that writes a shader already
builds the logical ``turing.shader-component.v1`` record. That record stays
authoritative. This module adds only what cannot be read from it: where each
logical port physically lives (descriptor set, binding, byte offset and
extent), what launches the shader (local size, group-count rule, the
element-count source and every control value with its derivation), the
target environment, provenance, and structured shortfalls.

The deployment record refers to the component by identity and never copies
its ports, so there is one ABI, not two. Physical facts are read from the
emitter's own artifact (its bindings, its launch plan, the control uniforms
its source declares), never from logical port order: for the desktop GLSL
arena, for example, every logical port is a view of one storage buffer, so
``backend_binding`` is not a descriptor binding.

Runtime extent fields reuse ``turing-runtime-abi-v1``'s spelling
(``src/compiler/runtime_abi.py``): ``rank`` is always present, ``shape`` is a
list or ``null`` when runtime-sized, ``extents_from`` is ``"fed"`` or the
index of the input whose extents an output takes, and ``extents.order`` lists
``{slot, buffer, kind, axis}``.

Turing never names a consumer's handles or includes here. A product is files
and records only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .compiled_program_api import CompiledProgramAPI, EntryPoint
from .shader_component_abi import SCHEMA as COMPONENT_SCHEMA


SCHEMA = "turing.shader-deployment.v1"
#: The request manifest a Nodus shader consumer reads (its own schema name).
NODUS_REQUEST_SCHEMA = "nodus-turing-shader-artifact-v1"
SPIRV_PRODUCT_SCHEMA = "turing.spirv-product.v1"
SHADER_SOURCE = "shader-source"
DEFAULT_SOURCE_TARGETS = ("glsl", "essl", "hlsl", "msl")
DIGESTS_FILE = "SHA256SUMS"

_ELEMENT_BYTES = {
    "f32": 4, "i32": 4, "u32": 4, "bool": 4, "f64": 8, "i64": 8,
}
_STAGE_SUFFIX = {
    "compute": ".comp", "fragment": ".frag", "vertex": ".vert",
    "geometry": ".geom", "tess_control": ".tesc", "tess_evaluation": ".tese",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(member) for key, member in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(member) for member in value]
    return value


def control_ref(control: str, index: int, *, scale: int = 1) -> dict[str, Any]:
    """A byte quantity the host supplies at launch: ``scale * control[index]``."""

    return {"control": str(control), "index": int(index), "scale": int(scale)}


@dataclass(frozen=True)
class Shortfall:
    """One reason the product is not fully deployable, by stable code."""

    code: str
    detail: str

    def to_mapping(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


@dataclass(frozen=True)
class PortView:
    """Where one logical component port lives inside a physical resource."""

    slot: int
    byte_offset: int | Mapping[str, Any]
    byte_extent: int | Mapping[str, Any] | None
    rank: int
    shape: tuple[int, ...] | None
    extents_from: str | int | None = None
    value_id: int | None = None
    #: The static shape of one instance when the resource holds a run of
    #: instances back to back (deployable SPIR-V); None otherwise.
    instance_shape: tuple[int, ...] | None = None

    def to_mapping(self) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "slot": self.slot,
            "byte_offset": _plain(self.byte_offset),
            "byte_extent": _plain(self.byte_extent),
            "rank": self.rank,
            "shape": None if self.shape is None else list(self.shape),
        }
        if self.extents_from is not None:
            mapping["extents_from"] = self.extents_from
        if self.value_id is not None:
            mapping["value_id"] = self.value_id
        if self.instance_shape is not None:
            mapping["instance_shape"] = list(self.instance_shape)
        return mapping


@dataclass(frozen=True)
class PhysicalResource:
    name: str
    kind: str  # storage-buffer | sampled-texture | draw-buffer
    access: str  # read | write | read_write
    layout: str
    dtype: str
    element_bytes: int
    set: int | None
    binding: int | None
    realizes: tuple[PortView, ...]
    byte_offset: int = 0
    byte_extent: int | Mapping[str, Any] | None = None
    location: int | None = None
    uniform: str | None = None
    extensions: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    specialization_constants: tuple[Mapping[str, Any], ...] = ()
    #: The declared members of a block resource, in declaration order.
    members: tuple[Mapping[str, Any], ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "set": self.set,
            "binding": self.binding,
            "access": self.access,
            "layout": self.layout,
            "dtype": self.dtype,
            "element_bytes": self.element_bytes,
            "byte_offset": self.byte_offset,
            "byte_extent": _plain(self.byte_extent),
            "realizes": [view.to_mapping() for view in self.realizes],
            "extensions": list(self.extensions),
            "capabilities": list(self.capabilities),
            "specialization_constants": _plain(self.specialization_constants),
        }
        if self.location is not None:
            mapping["location"] = self.location
        if self.uniform is not None:
            mapping["uniform"] = self.uniform
        if self.members:
            mapping["members"] = _plain(self.members)
        return mapping


@dataclass(frozen=True)
class LaunchControl:
    """A value the host must supply per launch, and how to derive it."""

    name: str
    storage: str  # default-uniform | storage-buffer
    dtype: str
    derivation: tuple[Mapping[str, Any], ...]  # one entry per element
    binding: int | None = None
    byte_offset: int | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "storage": self.storage,
            "dtype": self.dtype,
            "length": len(self.derivation),
            "binding": self.binding,
            "byte_offset": self.byte_offset,
            "derivation": _plain(self.derivation),
        }


@dataclass(frozen=True)
class LaunchContract:
    local_size: tuple[int, int, int]
    group_count: Mapping[str, Any]
    element_count: Mapping[str, Any]
    extents: Mapping[str, Any]
    controls: tuple[LaunchControl, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "local_size": list(self.local_size),
            "group_count": _plain(self.group_count),
            "element_count": _plain(self.element_count),
            "extents": _plain(self.extents),
            "controls": [control.to_mapping() for control in self.controls],
        }


@dataclass(frozen=True)
class ShaderDeployment:
    """The deployment view of one component, derived from its emitter."""

    component_id: str
    product_kind: str
    language: str
    version: str | None
    profile: str | None
    stage: str
    entrypoint: str
    target: Mapping[str, Any]
    resources: tuple[PhysicalResource, ...]
    launch: LaunchContract | None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    shortfalls: tuple[Shortfall, ...] = ()
    #: Whether the API sidecar carries the logical component record. An
    #: extracted host shader has none until Turing derives one.
    component_abi_published: bool = True
    source: Mapping[str, Any] | None = None
    companions: tuple[Mapping[str, Any], ...] = ()

    @property
    def complete(self) -> bool:
        return not self.shortfalls

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "component_id": self.component_id,
            "product_kind": self.product_kind,
            "component_abi": (
                {
                    "schema": COMPONENT_SCHEMA,
                    "component_id": self.component_id,
                    "record": "metadata.component_abi",
                }
                if self.component_abi_published else None
            ),
            "source": _plain(self.source),
            "companions": _plain(self.companions),
            "language": self.language,
            "version": self.version,
            "profile": self.profile,
            "stage": self.stage,
            "entrypoint": self.entrypoint,
            "target": _plain(self.target),
            "resources": [resource.to_mapping() for resource in self.resources],
            "launch": None if self.launch is None else self.launch.to_mapping(),
            "provenance": _plain(self.provenance),
            "complete": self.complete,
            "shortfalls": [item.to_mapping() for item in self.shortfalls],
        }


def glsl_version_directive(source: str) -> tuple[str | None, str | None]:
    """The ``#version`` number and profile the source itself declares."""

    match = re.search(r"^[ \t]*#[ \t]*version[ \t]+(\d+)(?:[ \t]+(\w+))?", source, re.M)
    if match is None:
        return None, None
    return match.group(1), match.group(2)


def _element_bytes(dtype: str) -> int:
    try:
        return _ELEMENT_BYTES[str(dtype)]
    except KeyError:
        raise ValueError(f"no storage element size for logical dtype {dtype!r}") from None


# -- derivations, one per emitter -----------------------------------------


def glsl_compute_deployment(component: Any) -> ShaderDeployment:
    """The arena launch contract of a desktop GLSL compute component.

    ``compose_shader`` declares one std430 ``Arena`` of ``uint`` words and
    two read-only std430 control blocks; ``_bind_launch_controls`` fills
    them: ``u_slot[i]`` (``TuringLaunchSlots``) is the arena word offset of
    port ``i`` (feeds, then outputs), and ``u_count`` then ``u_extent[0]``
    (``TuringLaunchExtents``) are the dispatch's element count. Each port is
    therefore a view at ``4 * u_slot[i]`` bytes.
    """

    abi = component.component_abi
    version, profile = glsl_version_directive(component.source)
    arena_binding = int(abi.decorations.get("arena_binding", 0))
    scalar_feeds = {int(value_id) for value_id in component.scalar_feeds}
    feed_shapes = {
        int(value_id): tuple(int(size) for size in shape)
        for value_id, shape in dict(component.feed_shapes or {}).items()
    }
    output_shape = (
        None if component.output_shape is None
        else tuple(int(size) for size in component.output_shape)
    )
    if output_shape is None and feed_shapes:
        output_shape = tuple(int(size) for size in _broadcast(feed_shapes.values()))
    feeds = [port for port in abi.ports if port.role == "feed"]
    runtime_feed = next(
        (port.slot for port in feeds
         if port.value_id not in scalar_feeds and port.value_id not in feed_shapes),
        None,
    )
    views = []
    for port in abi.ports:
        word = _element_bytes(port.dtype)
        offset = control_ref("u_slot", port.slot, scale=word)
        if port.role == "feed" and port.value_id in scalar_feeds:
            views.append(PortView(port.slot, offset, word, 0, (), value_id=port.value_id))
        elif port.role == "feed" and port.value_id in feed_shapes:
            shape = feed_shapes[port.value_id]
            views.append(PortView(
                port.slot, offset, word * math.prod(shape), len(shape), shape,
                value_id=port.value_id,
            ))
        elif port.role == "output" and output_shape is not None:
            views.append(PortView(
                port.slot, offset, word * math.prod(output_shape),
                len(output_shape), output_shape, value_id=port.value_id,
            ))
        else:
            views.append(PortView(
                port.slot, offset, control_ref("u_extent", 0, scale=word), 1, None,
                extents_from="fed" if port.role == "feed" else runtime_feed,
                value_id=port.value_id,
            ))
    arena = PhysicalResource(
        name="Arena", kind="storage-buffer", access="read_write",
        layout="std430", dtype="u32", element_bytes=4,
        set=0, binding=arena_binding, realizes=tuple(views),
        # The arena is host-sized: it must cover every view above.
        byte_extent=None,
    )
    port_count = len(abi.ports)
    slot_binding = int(abi.decorations["launch_slot_binding"])
    extent_binding = int(abi.decorations["launch_extent_binding"])
    operations = 1  # a component is one composed elementwise operation
    control_resources = (
        PhysicalResource(
            name="TuringLaunchSlots", kind="storage-buffer", access="read",
            layout="std430", dtype="u32", element_bytes=4,
            set=0, binding=slot_binding, realizes=(),
            byte_extent=4 * port_count,
        ),
        PhysicalResource(
            name="TuringLaunchExtents", kind="storage-buffer", access="read",
            layout="std430", dtype="u32", element_bytes=4,
            set=0, binding=extent_binding, realizes=(),
            byte_extent=4 * (1 + operations),
        ),
    )
    controls = (
        LaunchControl(
            "u_count", "storage-buffer", "u32", ({"rule": "element_count"},),
            binding=extent_binding, byte_offset=0,
        ),
        LaunchControl(
            "u_extent", "storage-buffer", "u32",
            ({"rule": "element_count", "operation": 0},),
            binding=extent_binding, byte_offset=4,
        ),
        LaunchControl(
            "u_slot", "storage-buffer", "u32",
            tuple({"rule": "arena_word_offset", "port_slot": slot}
                  for slot in range(port_count)),
            binding=slot_binding, byte_offset=0,
        ),
    )
    local = (int(component.local_size), 1, 1)
    output_slots = [port.slot for port in abi.ports if port.role == "output"]
    launch = LaunchContract(
        local_size=local,
        group_count={
            # turing_linear_gid folds x/y/z into one index space and every
            # invocation at or past the extent is guarded, so any grid that
            # covers the element count is exact.
            "rule": "cover",
            "work": "element_count",
            "invocations_per_group": local[0],
            "minimum_groups": "ceil(element_count / invocations_per_group)",
            "linearization": "x + y*X + z*X*Y over gl_NumWorkGroups * gl_WorkGroupSize",
            "guard": "u_extent[0]",
            "groups": None,
        },
        element_count={
            "source": "control",
            "control": "u_count",
            "kind": "element_count",
            "rank": 1 if output_shape is None else len(output_shape),
            "shape": None if output_shape is None else list(output_shape),
            "value": None if output_shape is None else math.prod(output_shape),
        },
        extents={
            "type": "uint32_t",
            "order": [{
                "slot": 0,
                "buffer": output_slots[0] if output_slots else None,
                "kind": "element_count",
                "axis": None,
            }],
        },
        controls=controls,
    )
    return ShaderDeployment(
        component_id=abi.component_id,
        product_kind=SHADER_SOURCE,
        language=abi.language,
        version=version, profile=profile,
        stage=abi.stage, entrypoint=abi.entrypoint,
        target={
            "client": "opengl", "environment": "opengl4.3", "spirv_version": None,
            # Every resource is a storage block in set 0 at bindings 0..2,
            # so the same source also compiles as GLSL for Vulkan.
            "clients": [
                {"client": "opengl", "environment": "opengl4.3", "spirv_version": None},
                {"client": "vulkan", "environment": "vulkan1.1", "spirv_version": "1.3"},
            ],
        },
        resources=(arena, *control_resources),
        launch=launch,
        provenance={"emitter": "glsl_component_backend.emit_glsl_compute_component"},
    )


def _broadcast(shapes: Iterable[Sequence[int]]) -> tuple[int, ...]:
    result: tuple[int, ...] = ()
    for shape in shapes:
        shape = tuple(int(size) for size in shape)
        width = max(len(result), len(shape))
        left = (1,) * (width - len(result)) + result
        right = (1,) * (width - len(shape)) + shape
        result = tuple(max(a, b) for a, b in zip(left, right))
    return result


def wgsl_deployment(module: Any) -> ShaderDeployment:
    """WebGPU storage bindings and the baked dispatch of a WGSL module."""

    abi = module.component_abi
    plan = module.launch_plan
    count = int(plan.count)
    metadata = dict(module.api.metadata) if module.api is not None else {}
    feed_span = metadata.get("feed_span")
    output_span = metadata.get("output_span")
    resources = []
    for port in abi.ports:
        word = _element_bytes(port.dtype)
        span = feed_span if port.role == "feed" else output_span
        members = len(span) if port.value_id is None and span else 1
        extent = word * count * members
        resources.append(PhysicalResource(
            name=port.name, kind="storage-buffer",
            access="read" if port.role == "feed" else "read_write",
            layout="wgsl-storage", dtype=port.dtype, element_bytes=word,
            set=0, binding=int(port.backend_binding),
            byte_extent=extent,
            realizes=(
                PortView(port.slot, 0, extent, 1 if members == 1 else 2,
                         (count,) if members == 1 else (members, count),
                         value_id=port.value_id),
            ),
        ))
    shortfalls = tuple(
        Shortfall("emission", item.format()) for item in module.shortfalls
    )
    return ShaderDeployment(
        component_id=abi.component_id,
        product_kind=SHADER_SOURCE,
        language=abi.language, version=None, profile=None,
        stage=abi.stage, entrypoint=abi.entrypoint,
        target={"client": "webgpu", "environment": "webgpu", "spirv_version": None},
        resources=tuple(resources),
        launch=LaunchContract(
            local_size=tuple(int(size) for size in plan.workgroup_size),
            group_count={"rule": "exact", "groups": [int(item) for item in plan.groups]},
            element_count={
                # The emitter bakes the count into the bounds guard.
                "source": "literal", "kind": "element_count",
                "rank": 1, "shape": [count], "value": count,
            },
            extents={"type": None, "order": []},
        ),
        provenance={"emitter": "ssa_webgpu_backend.emit_module"},
        shortfalls=shortfalls,
    )


def webgl_fragment_deployment(module: Any) -> ShaderDeployment:
    """Textures, draw buffers and controls of a WebGL fragment component."""

    abi = module.component_abi
    metadata = dict(module.api.metadata)
    version, profile = glsl_version_directive(module.source)
    extension = metadata.get("required_extension")
    resources = []
    for port in abi.ports:
        if port.role == "feed":
            resources.append(PhysicalResource(
                name=port.name, kind="sampled-texture", access="read",
                layout="texture2d-red", dtype=port.dtype, element_bytes=4,
                set=None, binding=None, uniform=port.name,
                realizes=(PortView(port.slot, 0, None, 2, None, "fed", port.value_id),),
            ))
        else:
            resources.append(PhysicalResource(
                name=port.name, kind="draw-buffer", access="write",
                layout=str(metadata.get("output_layout")), dtype=port.dtype,
                element_bytes=4, set=None, binding=None,
                location=int(port.backend_binding),
                extensions=(str(extension),) if extension else (),
                realizes=(PortView(port.slot, 0, None, 2, None, value_id=port.value_id),),
            ))
    controls = ()
    if metadata.get("input_sampling") == "normalized":
        controls = (LaunchControl(
            "turing_resolution", "default-uniform", "vec2",
            ({"rule": "framebuffer_extent"},),
        ),)
    return ShaderDeployment(
        component_id=abi.component_id,
        product_kind=SHADER_SOURCE,
        language=abi.language, version=version, profile=profile,
        stage=abi.stage, entrypoint=abi.entrypoint,
        target={"client": "webgl2", "environment": "opengl-es3.0", "spirv_version": None},
        resources=tuple(resources),
        launch=LaunchContract(
            local_size=(1, 1, 1),
            group_count={"rule": "raster", "draw": "full-screen-triangle",
                         "vertices": 3, "vertex_stage": "companion"},
            element_count={"source": "framebuffer", "kind": "element_count",
                           "rank": 2, "shape": None, "value": None},
            extents={"type": None, "order": []},
            controls=controls,
        ),
        provenance={"emitter": "fused_program_webgl_backend.emit_webgl_fragment_module"},
        shortfalls=(
            Shortfall(
                "graphics-stage",
                "a fragment component runs inside a raster pipeline with the "
                "companion full-screen vertex stage, not as a compute dispatch",
            ),
            *(Shortfall("emission", item.format()) for item in module.shortfalls),
        ),
    )


def _extracted_port(declaration: Any, stage: str) -> tuple[str, str, str, str]:
    """``(role, transport, kind, access)`` of one declared interface item."""

    from .glsl_interface import is_opaque

    storage = declaration.storage
    if storage == "varying":
        storage = "out" if stage == "vertex" else "in"
    qualifiers = set(declaration.qualifiers)
    written = (
        ("output", "write") if "writeonly" in qualifiers
        else ("feed", "read") if "readonly" in qualifiers
        else ("output", "read_write")
    )
    if storage == "in":
        if stage == "vertex":
            return "feed", "vertex-attribute", "vertex-attribute", "read"
        return "feed", "stage-input", "stage-input", "read"
    if storage == "out":
        if stage == "fragment":
            return "output", "draw-buffer", "draw-buffer", "write"
        return "output", "stage-output", "stage-output", "write"
    if storage == "buffer":
        role, access = written
        return role, "storage-buffer", "storage-buffer", access
    if declaration.is_block:
        return "uniform", "uniform-buffer", "uniform-buffer", "read"
    if declaration.glsl_type.startswith(("image", "iimage", "uimage")):
        role, access = written
        return role, "storage-image", "storage-image", access
    if declaration.glsl_type == "atomic_uint":
        return "output", "atomic-counter", "atomic-counter", "read_write"
    if is_opaque(declaration.glsl_type):
        return "feed", "texture", "sampled-texture", "read"
    return "uniform", "default-uniform", "default-uniform", "read"


def _extracted_view(slot: int, array: tuple[int, ...] | None) -> PortView:
    if array is None:
        return PortView(slot, 0, None, 0, ())
    if any(size < 0 for size in array):
        return PortView(slot, 0, None, len(array), None, extents_from="fed")
    return PortView(slot, 0, None, len(array), tuple(array))


def _extracted_interface(shader: Any) -> tuple[Any, tuple[PhysicalResource, ...], tuple[Shortfall, ...]]:
    """The component ABI and resources an extracted shader declares.

    Ports come from the source's own global declarations
    (``glsl_interface.read_glsl_interface``): feeds, then outputs, then
    uniforms, each in declaration order. ``backend_binding`` is the declared
    ``binding``, else the declared ``location``, else ``-1`` when the host or
    the linker assigns it. A resource whose binding or location the source
    leaves to the host is a ``link-assigned`` shortfall, because the record
    alone cannot bind it.
    """

    from .glsl_interface import glsl_dtype, read_glsl_interface
    from .shader_component_abi import (
        ComponentPort,
        ComponentSentinels,
        ShaderComponentABI,
    )

    interface = read_glsl_interface(shader.source)
    stage = shader.stage
    rows = []
    for declaration in interface.declarations:
        role, transport, kind, access = _extracted_port(declaration, stage)
        rows.append((role, transport, kind, access, declaration))
    order = {"feed": 0, "output": 1, "uniform": 2}
    rows.sort(key=lambda row: order[row[0]])
    ports: list[Any] = []
    resources: list[PhysicalResource] = []
    unassigned: list[str] = []
    fragment_outputs = sum(1 for row in rows if row[2] == "draw-buffer")
    for slot, (role, transport, kind, access, declaration) in enumerate(rows):
        binding = declaration.layout_int("binding")
        location = declaration.layout_int("location")
        members = declaration.members
        if len(members) == 1:
            glsl_type, array = members[0].glsl_type, members[0].array
        elif members:
            glsl_type, array = "block", declaration.array
        else:
            glsl_type, array = declaration.glsl_type, declaration.array
        dtype = glsl_dtype(glsl_type)
        ports.append(ComponentPort(
            slot=slot, name=str(declaration.name), role=role, dtype=dtype, value_id=None,
            backend_binding=(
                binding if binding is not None
                else location if location is not None else -1
            ),
            transport=transport,
        ))
        by_binding = kind in {
            "storage-buffer", "uniform-buffer", "sampled-texture",
            "storage-image", "atomic-counter",
        }
        if by_binding and binding is None:
            unassigned.append(f"{declaration.name} ({kind}, binding)")
        if kind == "vertex-attribute" and location is None:
            unassigned.append(f"{declaration.name} ({kind}, location)")
        if kind == "draw-buffer" and location is None and fragment_outputs > 1:
            unassigned.append(f"{declaration.name} ({kind}, location)")
        packing = next(
            (key for key, _value in declaration.layout
             if key in {"std140", "std430", "shared", "packed", "scalar"}),
            None,
        )
        element = dtype.split("x", 1)[0]
        resources.append(PhysicalResource(
            name=str(declaration.name),
            kind=kind,
            access=access,
            layout=(
                packing or ("shared" if kind in {"storage-buffer", "uniform-buffer"} else "glsl")
            ),
            dtype=dtype,
            element_bytes=_ELEMENT_BYTES.get(element, 0) if glsl_type != "block" else 0,
            set=(
                declaration.layout_int("set")
                if declaration.has_layout("set")
                else (0 if by_binding and binding is not None else None)
            ),
            binding=binding,
            location=location,
            uniform=str(declaration.name) if kind == "default-uniform" else None,
            realizes=(_extracted_view(slot, array),),
            members=tuple(member.to_mapping() for member in members),
        ))
    abi = ShaderComponentABI(
        component_id=str(shader.source_name or f"{Path(shader.filename).stem}_{shader.line}"),
        language=shader.language,
        stage=stage,
        entrypoint="main",
        ports=tuple(ports),
        sentinels=ComponentSentinels.for_ports(len(ports)),
        decorations=MappingProxyType({
            "interface": "declared",
            "reader": "glsl_interface.read_glsl_interface",
            **({"local_size": list(interface.local_size)} if interface.local_size else {}),
        }),
    )
    shortfalls: list[Shortfall] = []
    if unassigned:
        shortfalls.append(Shortfall(
            "link-assigned",
            "the source leaves these to the host program or the linker, so "
            "the record alone cannot bind them: " + ", ".join(unassigned),
        ))
    return abi, tuple(resources), tuple(shortfalls)


def extracted_shader_deployment(shader: Any) -> ShaderDeployment:
    """A host-compiled shader: exact source, provenance, declared interface."""

    return extracted_shader_component(shader)[1]


def extracted_shader_component(shader: Any) -> tuple[Any, ShaderDeployment]:
    """The component ABI (or None) and deployment of an extracted shader.

    The component ABI is read from the source's own declarations. A source
    whose interface cannot be read with certainty has none, and keeps the
    ``no-component-abi`` shortfall with the reader's reason.
    """

    from .glsl_interface import GLSLInterfaceRefusal
    from .shader_stages import STAGES

    version, profile = glsl_version_directive(shader.source)
    identity = shader.source_name or f"{Path(shader.filename).stem}_{shader.line}"
    provenance = {
        "emitter": "shader_extractor.extract_shader_compile_calls",
        "filename": shader.filename,
        "line": shader.line,
        "source_symbol": shader.source_name,
        "compiler_call": shader.compiler,
        "source_sha256": shader.sha256,
    }
    target = {
        "client": "opengl-es" if profile == "es" else "opengl",
        "environment": None, "spirv_version": None,
    }
    common = dict(
        component_id=str(identity),
        product_kind=SHADER_SOURCE,
        language=shader.language, version=version, profile=profile,
        stage=shader.stage, entrypoint="main", target=target,
        provenance=provenance,
    )
    reason = None
    if shader.stage not in STAGES:
        reason = f"the component ABI has no {shader.stage!r} stage"
    else:
        try:
            abi, resources, shortfalls = _extracted_interface(shader)
        except GLSLInterfaceRefusal as error:
            reason = f"the declared interface cannot be read: {error}"
    if reason is not None:
        return None, ShaderDeployment(
            **common, resources=(), launch=None,
            shortfalls=(Shortfall("no-component-abi", reason),),
            component_abi_published=False,
        )
    launch = None
    stage_shortfalls: tuple[Shortfall, ...]
    if shader.stage == "compute":
        local_size = tuple(abi.decorations.get("local_size") or (1, 1, 1))
        launch = LaunchContract(
            local_size=local_size,  # type: ignore[arg-type]
            group_count={"rule": "host"},
            element_count={"source": "host", "kind": "element_count",
                           "rank": None, "shape": None, "value": None},
            extents={"type": None, "order": []},
        )
        stage_shortfalls = (Shortfall(
            "host-launch",
            "the host program chooses the group count at its dispatch call; "
            "Turing has not read that call",
        ),)
    else:
        stage_shortfalls = (Shortfall(
            "graphics-stage",
            "the host program links this stage into a raster pipeline and "
            "issues the draw; the record describes the stage's interface only",
        ),)
    return abi, ShaderDeployment(
        **common, resources=resources, launch=launch,
        shortfalls=(*stage_shortfalls, *shortfalls),
    )


def spirv_deployment(shader: Any) -> ShaderDeployment:
    """The descriptor ABI a deployable SPIR-V compute module declares."""

    abi = shader.component_abi
    ports = {int(port.value_id): port for port in abi.ports if port.value_id is not None}
    first_feed = next((port.slot for port in abi.ports if port.role == "feed"), None)
    # Element mode (an elementwise function over runtime-sized spans): one
    # element per invocation; outputs take the first span feed's extents.
    element_rank = max(
        (int(getattr(item, "runtime_rank", 0) or 0) for item in shader.bindings),
        default=0,
    )
    if element_rank:
        first_feed = next((
            ports[int(item.value_id)].slot for item in shader.bindings
            if item.role == "feed" and getattr(item, "runtime_rank", 0)
        ), None)
    # Reduction mode: one invocation folds the span feed into element 0 of
    # the one output, so the output is a single element and the launch is
    # one group.
    reduction = getattr(shader, "reduction", None)
    resources = []
    for binding in shader.bindings:
        if binding.role == "control":
            continue
        port = ports[int(binding.value_id)]
        if element_rank:
            broadcast = bool(getattr(binding, "broadcast", False)) or (
                reduction is not None and binding.role == "output"
            )
            extent = (
                binding.element_bytes if broadcast
                else control_ref("element_count", 0, scale=binding.element_bytes)
            )
            resources.append(PhysicalResource(
                name=binding.name, kind="storage-buffer", access=binding.access,
                layout="std430", dtype=binding.dtype,
                element_bytes=binding.element_bytes,
                set=0, binding=binding.binding, byte_extent=extent,
                capabilities=tuple(
                    item for item in shader.capabilities if item != "Shader"
                ),
                realizes=(PortView(
                    port.slot, 0, extent,
                    0 if broadcast else int(binding.runtime_rank),
                    () if broadcast else None,
                    None if broadcast
                    else "fed" if binding.role == "feed" else first_feed,
                    port.value_id,
                ),),
            ))
            continue
        # One instance per invocation: ``element_count`` instances of
        # ``prod(instance_shape)`` elements each, row-major, back to back.
        instance = tuple(getattr(binding, "instance_shape", ()) or ())
        extent = control_ref(
            "element_count", 0,
            scale=binding.element_bytes * math.prod(instance),
        )
        resources.append(PhysicalResource(
            name=binding.name, kind="storage-buffer", access=binding.access,
            layout="std430", dtype=binding.dtype, element_bytes=binding.element_bytes,
            set=0, binding=binding.binding, byte_extent=extent,
            capabilities=tuple(item for item in shader.capabilities if item != "Shader"),
            realizes=(PortView(
                port.slot, 0, extent, 1 + len(instance), None,
                "fed" if binding.role == "feed" else first_feed,
                port.value_id,
                instance_shape=instance,
            ),),
        ))
    control = next(item for item in shader.bindings if item.role == "control")
    output_slot = next((port.slot for port in abi.ports if port.role == "output"), None)
    local = tuple(int(size) for size in shader.local_size)
    return ShaderDeployment(
        component_id=abi.component_id,
        product_kind=shader.product_kind,
        language="spirv", version=shader.spirv_version, profile=None,
        stage=abi.stage, entrypoint=shader.entrypoint,
        target={
            "client": "vulkan",
            "environment": shader.target_env,
            "spirv_version": shader.spirv_version,
        },
        resources=tuple(resources),
        launch=LaunchContract(
            local_size=local,
            group_count=(
                {
                    "rule": "fixed", "work": None,
                    "invocations_per_group": local[0], "groups": 1,
                    "axis": "x",
                } if reduction is not None else {
                    "rule": "ceil_div",
                    "work": "element_count",
                    "invocations_per_group": local[0],
                    "groups": None,
                    "axis": "x",
                }
            ),
            element_count={
                "source": "storage-word", "control": "element_count",
                "kind": "element_count", "rank": element_rank or 1,
                "shape": None, "value": None,
                # Element mode: the product of the span feeds' runtime
                # extents (which must agree; the host checks them).
                **({"elements_of": first_feed} if element_rank else {}),
                # max/min/mean of no elements has no value; the shader
                # then writes nothing.
                **(
                    {"reduction": reduction, "minimum": (
                        1 if reduction in ("max", "min", "mean") else 0
                    )} if reduction is not None else {}
                ),
            },
            extents={
                "type": "uint32_t",
                "order": [{"slot": 0, "buffer": output_slot,
                           "kind": "element_count", "axis": None}],
            },
            controls=(LaunchControl(
                "element_count", "storage-buffer", "u32",
                ({"rule": "element_count"},),
                binding=control.binding, byte_offset=0,
            ),),
        ),
        provenance={
            "emitter": "ssa_spirv_backend.emit_deployable_compute_shader",
            "ssa_function": shader.function,
        },
        shortfalls=tuple(Shortfall("emission", item.format()) for item in shader.shortfalls),
    )


# -- publication ------------------------------------------------------------


@dataclass(frozen=True)
class PublishedShaderProduct:
    manifest_path: Path
    api_path: Path | None
    digests_path: Path
    source_path: Path | None = None
    module_path: Path | None = None
    companion_paths: tuple[Path, ...] = ()
    deployment: Mapping[str, Any] = field(default_factory=dict)


def _write_digests(directory: Path, paths: Sequence[Path]) -> Path:
    lines = [f"{_sha(path.read_bytes())}  {path.name}" for path in paths]
    destination = directory / DIGESTS_FILE
    existing = {}
    if destination.exists():
        for line in destination.read_text(encoding="utf-8").splitlines():
            digest, _, name = line.partition("  ")
            existing[name] = digest
    for line in lines:
        digest, _, name = line.partition("  ")
        existing[name] = digest
    destination.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(existing.items())),
        encoding="utf-8", newline="\n",
    )
    return destination


def _with_deployment(api: CompiledProgramAPI, record: Mapping[str, Any]) -> CompiledProgramAPI:
    return replace(api, metadata={**dict(api.metadata), "shader_deployment": dict(record)})


def publish_shader_source(
    directory: str | Path,
    *,
    name: str,
    source: str,
    suffix: str,
    api: CompiledProgramAPI,
    deployment: ShaderDeployment,
    targets: Sequence[str] = DEFAULT_SOURCE_TARGETS,
    companions: Mapping[str, str] | None = None,
) -> PublishedShaderProduct:
    """Write source, API sidecar, request manifest and digests together.

    The source is written byte for byte as emitted; its digest is recorded in
    the deployment record inside the sidecar, and ``SHA256SUMS`` covers every
    file of the product, the sidecar and manifest included.
    """

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    source_path = output / f"{name}{suffix}"
    source_bytes = source.encode("utf-8")
    source_path.write_bytes(source_bytes)
    companion_paths = []
    companion_rows = []
    for companion_suffix, text in dict(companions or {}).items():
        path = output / f"{name}{companion_suffix}"
        data = text.encode("utf-8")
        path.write_bytes(data)
        companion_paths.append(path)
        companion_rows.append({"file": path.name, "sha256": _sha(data)})
    record = replace(
        deployment,
        source={"file": source_path.name, "sha256": _sha(source_bytes)},
        companions=tuple(companion_rows),
    ).to_mapping()
    api_path = _with_deployment(api, record).write(output / f"{name}.api.yaml")
    manifest_path = output / f"{name}.shader.json"
    manifest_path.write_text(
        json.dumps({
            "schema": NODUS_REQUEST_SCHEMA,
            "source": source_path.name,
            "api": api_path.name,
            "targets": list(targets),
        }, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    digests = _write_digests(
        output, (source_path, *companion_paths, api_path, manifest_path),
    )
    return PublishedShaderProduct(
        manifest_path, api_path, digests, source_path=source_path,
        companion_paths=tuple(companion_paths), deployment=record,
    )


def publish_extracted_shader(
    directory: str | Path, shader: Any, *, name: str | None = None,
    targets: Sequence[str] = DEFAULT_SOURCE_TARGETS,
) -> PublishedShaderProduct:
    """Publish a host-extracted shader with its provenance carried along."""

    abi, deployment = extracted_shader_component(shader)
    name = str(name or deployment.component_id)
    metadata: dict[str, Any] = {
        "stage": shader.stage,
        "extracted_shader": dict(shader.to_mapping()["origin"]),
    }
    if abi is not None:
        metadata["component_abi"] = abi.to_mapping()
    api = CompiledProgramAPI(
        module=name,
        language=shader.language,
        entry="main",
        entry_points=(EntryPoint("main", "main", "numerical"),),
        metadata=metadata,
    )
    return publish_shader_source(
        directory, name=name, source=shader.source,
        suffix=_STAGE_SUFFIX.get(shader.stage, ".glsl"),
        api=api, deployment=deployment, targets=targets,
    )


def _run_tool(command: Sequence[str], *, tool: str) -> dict[str, Any]:
    completed = subprocess.run(list(command), capture_output=True, text=True)
    return {
        # Windows parity receipt (2026-10-24): shutil.which returns an
        # executable path whose basename can be ``spirv-as.EXE``.  The
        # manifest records the selected tool identity, not the host path's
        # spelling, so carry the authored identity from the selection site.
        "tool": tool,
        "arguments": list(command[1:]),
        "returncode": completed.returncode,
        "diagnostics": (completed.stdout + completed.stderr).strip(),
    }


def publish_spirv_product(directory: str | Path, product: Any) -> PublishedShaderProduct:
    """Publish a SPIR-V library or deployable shader under its product kind.

    The assembly is always written. A binary is assembled, and validated for
    the product's target environment, only when ``spirv-as``/``spirv-val``
    are present; their absence is a recorded shortfall, never a skipped claim.
    """

    from .ssa_spirv_backend import DEPLOYABLE_SHADER, spirv_assembler

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    deployable = product.product_kind == DEPLOYABLE_SHADER
    assembly_path = product.write(output)
    binary_path = output / f"{product.name}.spv"
    target_env = getattr(product, "target_env", None)
    tools: list[dict[str, Any]] = []
    shortfalls: list[Shortfall] = []
    assembler = spirv_assembler()
    if not product.complete:
        shortfalls.extend(Shortfall("emission", item.format()) for item in product.shortfalls)
        binary_path = None
    elif assembler is None:
        shortfalls.append(Shortfall("not-assembled", "no spirv-as was found"))
        binary_path = None
    else:
        env = ["--target-env", target_env] if target_env else []
        tools.append(_run_tool(
            [assembler, *env, str(assembly_path), "-o", str(binary_path)],
            tool="spirv-as",
        ))
        if tools[-1]["returncode"] != 0:
            shortfalls.append(Shortfall("assembly-failed", tools[-1]["diagnostics"]))
            binary_path = None
        else:
            validator = shutil.which("spirv-val")
            if validator is None:
                shortfalls.append(Shortfall("not-validated", "no spirv-val was found"))
            else:
                tools.append(_run_tool(
                    [validator, *env, str(binary_path)], tool="spirv-val",
                ))
                if tools[-1]["returncode"] != 0:
                    shortfalls.append(Shortfall("validation-failed", tools[-1]["diagnostics"]))
    files = [assembly_path, *([binary_path] if binary_path is not None else [])]
    api_path = None
    record: dict[str, Any] = {}
    if deployable and product.component_abi is not None:
        deployment = spirv_deployment(product)
        deployment = replace(
            deployment,
            shortfalls=(*deployment.shortfalls, *(
                item for item in shortfalls if item.code != "emission"
            )),
            source={
                "file": (binary_path or assembly_path).name,
                "sha256": _sha((binary_path or assembly_path).read_bytes()),
                "assembly": assembly_path.name,
                "assembly_sha256": _sha(assembly_path.read_bytes()),
            },
            provenance={**dict(deployment.provenance), "tools": tools},
        )
        record = deployment.to_mapping()
        api_path = _with_deployment(product.api, record).write(output / f"{product.name}.api.yaml")
        files.append(api_path)
    manifest_path = output / f"{product.name}.spirv.json"
    manifest_path.write_text(json.dumps({
        "schema": SPIRV_PRODUCT_SCHEMA,
        "product_kind": product.product_kind,
        "assembly": assembly_path.name,
        "module": None if binary_path is None else binary_path.name,
        "api": None if api_path is None else api_path.name,
        "entrypoint": getattr(product, "entrypoint", None) if deployable else None,
        "target_env": target_env,
        "spirv_version": getattr(product, "spirv_version", None),
        "complete": not shortfalls,
        "shortfalls": [item.to_mapping() for item in shortfalls],
        "tools": tools,
        "sha256": {path.name: _sha(path.read_bytes()) for path in files},
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    digests = _write_digests(output, (*files, manifest_path))
    return PublishedShaderProduct(
        manifest_path, api_path, digests,
        source_path=assembly_path, module_path=binary_path, deployment=record,
    )


__all__ = [
    "DEFAULT_SOURCE_TARGETS", "DIGESTS_FILE", "LaunchContract", "LaunchControl",
    "NODUS_REQUEST_SCHEMA", "PhysicalResource", "PortView", "PublishedShaderProduct",
    "SCHEMA", "SHADER_SOURCE", "SPIRV_PRODUCT_SCHEMA", "ShaderDeployment", "Shortfall",
    "control_ref", "extracted_shader_deployment", "glsl_compute_deployment",
    "glsl_version_directive", "publish_extracted_shader", "publish_shader_source",
    "publish_spirv_product", "spirv_deployment", "webgl_fragment_deployment",
    "wgsl_deployment",
]
