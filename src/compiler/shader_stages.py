"""Shader-stage taxonomy and the buffer-binding contract shared by every
backend that emits stage-specialized shader text.

Every backend that speaks a real GPU shading language (WebGPU/WGSL, WebGL 2/
GLSL ES, desktop GLSL) targets one of a small number of pipeline stages, and
stage identity is what determines the entry-point attribute, which builtins
are available, and whether the stage can write to storage buffers at all --
that is real, stage-intrinsic behavior, not backend-specific trivia, so it
belongs in one shared table instead of being re-decided inside each backend.

``BufferBinding``/``ShaderIOLayout`` are the matching backend-agnostic
description of what a compiled module reads and writes: a WGSL storage
buffer, a WebGL ``sampler2D`` uniform, and a desktop GLSL SSBO arena slot are
all "a named channel a caller must fill with N scalars of dtype T before the
program runs" -- described once here so a caller (the published-page JS
runtime, or anything else) does not need backend-specific parsing to find
out what to feed a compiled module.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ShaderStage:
    """One pipeline stage's real constraints, stated rather than assumed."""

    name: str
    # The WGSL entry-point attribute for this stage, or None if WebGPU has
    # no such stage at all (true of geometry -- there is no WGSL geometry
    # shader; desktop GLSL is the only backend that can express one).
    wgsl_attribute: str | None
    supports_storage_write: bool
    builtin_input: str


COMPUTE = ShaderStage("compute", "@compute", True, "global_invocation_id")
FRAGMENT = ShaderStage("fragment", "@fragment", False, "position")
VERTEX = ShaderStage("vertex", "@vertex", False, "vertex_index")
GEOMETRY = ShaderStage("geometry", None, False, "primitive_id")

STAGES: dict[str, ShaderStage] = {
    stage.name: stage for stage in (COMPUTE, FRAGMENT, VERTEX, GEOMETRY)
}


@dataclass(frozen=True)
class BufferBinding:
    """One storage/uniform channel a shader module reads or writes."""

    name: str
    role: str  # "feed" | "output" | "uniform"
    dtype: str
    index: int
    value_id: int | None = None

    def to_mapping(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "dtype": self.dtype,
            "index": self.index,
            "value_id": self.value_id,
        }


@dataclass(frozen=True)
class ShaderIOLayout:
    """The full set of buffers one compiled module needs marshalled."""

    stage: str
    feeds: tuple[BufferBinding, ...] = ()
    outputs: tuple[BufferBinding, ...] = ()
    uniforms: tuple[BufferBinding, ...] = field(default_factory=tuple)

    def to_mapping(self) -> dict:
        return {
            "stage": self.stage,
            "feeds": [item.to_mapping() for item in self.feeds],
            "outputs": [item.to_mapping() for item in self.outputs],
            "uniforms": [item.to_mapping() for item in self.uniforms],
        }


__all__ = [
    "BufferBinding",
    "COMPUTE",
    "FRAGMENT",
    "GEOMETRY",
    "STAGES",
    "ShaderIOLayout",
    "ShaderStage",
    "VERTEX",
]
