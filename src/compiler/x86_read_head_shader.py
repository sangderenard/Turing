"""WebGPU register-display kernel for the reversible x86 read head."""

from __future__ import annotations

from dataclasses import dataclass

from .x86_tensor_read_head import X86ReadHeadState


@dataclass(frozen=True, slots=True)
class X86ReadHeadShaderArtifact:
    """A shader and its tightly packed storage-buffer contract."""

    source: str
    register_names: tuple[str, ...]
    workgroup_size: int

    @property
    def register_count(self) -> int:
        return len(self.register_names)


def build_x86_read_head_register_shader(
    *, workgroup_size: int = 64,
) -> X86ReadHeadShaderArtifact:
    """Emit a compute shader that updates one display cell per register.

    Binding 0 is the packed ``core_count * register_count`` signed register
    matrix from :meth:`X86ReadHeadState.register_tensor`. Binding 1 is an RGBA
    float matrix of the same logical size. A single dispatch therefore updates
    every visible register across every virtual core without CPU formatting.
    """

    if workgroup_size <= 0 or workgroup_size > 256:
        raise ValueError("WebGPU workgroup size must be in [1, 256]")
    count = len(X86ReadHeadState.REGISTER_NAMES)
    source = f"""// turing.x86-read-head-registers.v1
struct DisplayUniforms {{ core_count: u32, register_count: u32, history_position: u32, _pad: u32 }};
@group(0) @binding(0) var<storage, read> registers: array<i32>;
@group(0) @binding(1) var<storage, read_write> cells: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> display: DisplayUniforms;

@compute @workgroup_size({int(workgroup_size)})
fn update_read_head_registers(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let index = gid.x;
  let count = display.core_count * display.register_count;
  if (index >= count) {{ return; }}
  let register_index = index % display.register_count;
  let value = registers[index];
  let magnitude = log2(1.0 + abs(f32(value))) / 32.0;
  let sign_color = select(vec3<f32>(0.20, 0.72, 1.0), vec3<f32>(1.0, 0.34, 0.24), value < 0);
  let phase_band = f32((register_index + display.history_position) % {count}u) / f32({count});
  cells[index] = vec4<f32>(mix(sign_color, vec3<f32>(phase_band, 1.0 - phase_band, 0.55), 0.25) * (0.2 + 0.8 * magnitude), 1.0);
}}
"""
    return X86ReadHeadShaderArtifact(
        source=source,
        register_names=X86ReadHeadState.REGISTER_NAMES,
        workgroup_size=int(workgroup_size),
    )


__all__ = ["X86ReadHeadShaderArtifact", "build_x86_read_head_register_shader"]
