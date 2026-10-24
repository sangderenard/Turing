"""Python-facing desktop GLSL compute and GLSL fragment component emitters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..common.tensors.accelerator_backends.glsl_backend import (
    _normalize_dtype,
    emit_multi_output_program_source,
)
from ..common.tensors.fused_ir import FusedProgram
from .compiled_program_api import CompiledProgramAPI, EntryPoint
from .fused_program_webgl_backend import (
    WebGLFragmentModule,
    emit_webgl_fragment_module,
)
from .shader_component_abi import ShaderComponentABI, component_abi_from_layout
from .shader_stages import COMPUTE, BufferBinding, ShaderIOLayout


@dataclass(frozen=True, slots=True)
class GLSLComputeComponent:
    name: str
    source: str
    api: CompiledProgramAPI
    io_layout: ShaderIOLayout
    component_abi: ShaderComponentABI
    local_size: int
    # The emission's own extent facts, kept so the deployment view reads
    # them instead of re-deriving extents from port order.
    scalar_feeds: tuple[int, ...] = ()
    feed_shapes: Mapping[int, tuple[int, ...]] | None = None
    output_shape: tuple[int, ...] | None = None

    def deployment(self):
        from .shader_deployment import glsl_compute_deployment
        return glsl_compute_deployment(self)

    def write(self, directory: str | Path) -> Path:
        """Publish source, API sidecar, request manifest and digests."""

        from .shader_deployment import publish_shader_source
        return publish_shader_source(
            directory, name=self.name, source=self.source, suffix=".comp",
            api=self.api, deployment=self.deployment(),
        ).source_path


def emit_glsl_compute_component(
    program: FusedProgram,
    *,
    name: str = "turing_compute",
    local_size: int = 256,
    scalar_feeds: Iterable[int] = (),
    feed_shapes: Mapping[int, Sequence[int]] | None = None,
    output_shape: Sequence[int] | None = None,
) -> GLSLComputeComponent:
    """Emit desktop GLSL compute plus the same logical ABI used by WGSL."""

    source = emit_multi_output_program_source(
        program,
        local_size=local_size,
        scalar_feeds=scalar_feeds,
        feed_shapes=feed_shapes,
        output_shape=output_shape,
    )
    # A port's dtype is the word interpretation the emitted source applies
    # (program_snippet reads each value's metadata dtype through the same
    # normalization), not a blanket f32: an int32 feed is read as int(word).
    def port_dtype(value_id: int) -> str:
        meta = (program.meta or {}).get(value_id)
        declared = getattr(meta, "dtype", None) if meta is not None else None
        kind = _normalize_dtype(None if declared in (None, "None") else declared).kind
        return {"f": "f32", "i": "i32", "u": "u32", "b": "bool"}[kind]

    feeds = tuple(
        BufferBinding(f"feed_{value_id}", "feed", port_dtype(value_id), index, value_id)
        for index, value_id in enumerate(sorted(program.feeds))
    )
    outputs = tuple(
        BufferBinding(
            f"output_{index}", "output", port_dtype(value_id),
            len(feeds) + index, value_id,
        )
        for index, value_id in enumerate(program.outputs.values())
    )
    layout = ShaderIOLayout(COMPUTE.name, feeds=feeds, outputs=outputs)
    component = component_abi_from_layout(
        name,
        "glsl-430",
        layout,
        decorations={
            "execution_model": "compute",
            "workgroup_size": (int(local_size), 1, 1),
            "arena_binding": 0,
            "sentinel_policy": "validate-before-dispatch/publish-after-dispatch",
        },
    )
    api = CompiledProgramAPI(
        module=name,
        language="glsl-430",
        entry="main",
        entry_points=(EntryPoint("main", "main", "numerical"),),
        metadata={
            "stage": "compute",
            "io_layout": layout.to_mapping(),
            "component_abi": component.to_mapping(),
        },
    )
    return GLSLComputeComponent(
        name, source, api, layout, component, int(local_size),
        scalar_feeds=tuple(sorted(int(value_id) for value_id in scalar_feeds)),
        feed_shapes=(
            None if feed_shapes is None else {
                int(value_id): tuple(int(size) for size in shape)
                for value_id, shape in feed_shapes.items()
            }
        ),
        output_shape=(
            None if output_shape is None
            else tuple(int(size) for size in output_shape)
        ),
    )


def emit_glsl_fragment_component(
    program: FusedProgram,
    *,
    name: str = "turing_fragment",
) -> WebGLFragmentModule:
    """Emit GLSL ES fragment with the canonical component ABI attached."""

    return emit_webgl_fragment_module(program, name=name)


__all__ = [
    "GLSLComputeComponent",
    "emit_glsl_compute_component",
    "emit_glsl_fragment_component",
]
