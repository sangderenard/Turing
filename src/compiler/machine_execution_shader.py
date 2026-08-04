"""WebGPU presentation shader for complete virtual-machine register files."""

from __future__ import annotations

from dataclasses import dataclass

from .machine_execution import MachineExecutionState


MACHINE_DISPLAY_REGISTERS = (
    *MachineExecutionState.REGISTER_NAMES,
    "rip", "rflags", "steps", "call_depth",
)


@dataclass(frozen=True, slots=True)
class MachineRegisterShaderArtifact:
    source: str
    register_names: tuple[str, ...]
    workgroup_size: int
    register_stride_bytes: int = 8
    register_bank_stride_bytes: int = 256


def build_machine_register_shader(*, workgroup_size: int = 64) -> MachineRegisterShaderArtifact:
    """Emit the lossless-u64-input shader used by the live register display."""

    if not 1 <= workgroup_size <= 256:
        raise ValueError("WebGPU workgroup size must be in [1, 256]")
    register_count = len(MACHINE_DISPLAY_REGISTERS)
    source = f"""// turing.machine-register-display.v1
struct DisplayUniforms {{ core_count: u32, history_position: u32, register_stride_cells: u32, cache_block_count: u32 }};
struct CacheBlock {{ byte_offset: u32, byte_capacity: u32, occupied_bytes: u32, address_low: u32 }};
@group(0) @binding(0) var<storage, read> register_words: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> cells: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> display: DisplayUniforms;
@group(0) @binding(3) var<storage, read> cache_blocks: array<CacheBlock>;
@group(0) @binding(4) var<storage, read_write> cache_cells: array<vec4<f32>>;

@compute @workgroup_size({workgroup_size})
fn update_machine_registers(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let index = gid.x;
  let count = display.core_count * {register_count}u;
  if (index >= count) {{ return; }}
  let core_index = index / {register_count}u;
  let register_index = index % {register_count}u;
  let physical_index = core_index * display.register_stride_cells + register_index;
  let words = register_words[physical_index];
  let nonzero = select(0.0, 1.0, (words.x | words.y) != 0u);
  let high_energy = log2(1.0 + f32(words.y)) / 32.0;
  let low_energy = log2(1.0 + f32(words.x)) / 32.0;
  let scan = f32((register_index + display.history_position) % {register_count}u) / f32({register_count});
  cells[index] = vec4<f32>(0.12 + low_energy, 0.15 + high_energy, 0.25 + 0.65 * scan, 0.35 + 0.65 * nonzero);
}}

@compute @workgroup_size({workgroup_size})
fn update_program_cache(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let index = gid.x;
  if (index >= display.cache_block_count) {{ return; }}
  let block = cache_blocks[index];
  let occupancy = select(0.0, f32(block.occupied_bytes) / f32(block.byte_capacity), block.byte_capacity > 0u);
  let pulse = 0.65 + 0.35 * sin(f32(display.history_position + index) * 0.11);
  cache_cells[index] = vec4<f32>(0.08 + occupancy * pulse, 0.16 + occupancy * 0.68, 0.22 + occupancy, 1.0);
}}
"""
    return MachineRegisterShaderArtifact(
        source, MACHINE_DISPLAY_REGISTERS, int(workgroup_size),
    )


__all__ = [
    "MACHINE_DISPLAY_REGISTERS",
    "MachineRegisterShaderArtifact",
    "build_machine_register_shader",
]
