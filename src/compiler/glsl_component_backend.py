"""Python-facing desktop GLSL compute and GLSL fragment component emitters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..common.tensors.accelerator_backends.glsl_backend import (
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
    feeds = tuple(
        BufferBinding(f"feed_{value_id}", "feed", "f32", index, value_id)
        for index, value_id in enumerate(sorted(program.feeds))
    )
    outputs = tuple(
        BufferBinding(f"output_{index}", "output", "f32", len(feeds) + index, value_id)
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
    return GLSLComputeComponent(name, source, api, layout, component, int(local_size))


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
