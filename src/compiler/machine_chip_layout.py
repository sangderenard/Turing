"""Physical-style register banks, program cache blocks, and bounded clocks.

These layouts are observation and deployment ABIs, not claims about the host
CPU's silicon. They make the virtual machine's fixed allocations spatially
honest so the same addresses can drive an executor and an occupancy display.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Sequence

from .machine_execution import MachineExecutionState, MachineVirtualMulticore


def _align(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (int(value) + alignment - 1) & -alignment


@dataclass(frozen=True, slots=True)
class RegisterCell:
    core: int
    name: str
    byte_offset: int
    byte_size: int = 8

    @property
    def word_offsets(self) -> tuple[int, int]:
        """Adjacent little-endian u32 words holding this complete register."""

        return self.byte_offset, self.byte_offset + 4


@dataclass(frozen=True, slots=True)
class RegisterBankLayout:
    base_offset: int
    core_stride: int
    register_names: tuple[str, ...]
    cells: tuple[RegisterCell, ...]

    @property
    def byte_size(self) -> int:
        if not self.cells:
            return 0
        return self.core_stride * (1 + max(cell.core for cell in self.cells))

    def cell(self, core: int, name: str) -> RegisterCell:
        for cell in self.cells:
            if cell.core == core and cell.name == name:
                return cell
        raise KeyError(f"no register cell {core}:{name}")


def build_register_bank_layout(
    core_count: int,
    *,
    base_offset: int = 0,
    bank_alignment: int = 256,
) -> RegisterBankLayout:
    """Allocate fixed contiguous 64-bit cells inside each contiguous core bank."""

    if core_count <= 0:
        raise ValueError("register layout requires at least one core")
    if base_offset < 0 or base_offset % 8:
        raise ValueError("register-bank base must be non-negative and 8-byte aligned")
    names = (
        *MachineExecutionState.REGISTER_NAMES,
        "rip", "rflags", "steps", "call_depth",
    )
    stride = _align(len(names) * 8, bank_alignment)
    cells = tuple(
        RegisterCell(
            core=core,
            name=name,
            byte_offset=base_offset + core * stride + index * 8,
        )
        for core in range(core_count)
        for index, name in enumerate(names)
    )
    return RegisterBankLayout(base_offset, stride, names, cells)


def pack_register_banks(
    machine: MachineVirtualMulticore,
    layout: RegisterBankLayout,
) -> tuple[int, ...]:
    """Pack every register into its fixed adjacent low/high-u32 cells."""

    if len(machine.cores) * len(layout.register_names) != len(layout.cells):
        raise ValueError("register layout does not match virtual core count")
    words_per_bank = layout.core_stride // 4
    words = [0] * (words_per_bank * len(machine.cores))
    for core_index, core in enumerate(machine.cores):
        values = core.state.packed_register_words()
        for register_index, (low, high) in enumerate(values):
            word = core_index * words_per_bank + register_index * 2
            words[word:word + 2] = [low, high]
    return tuple(words)


@dataclass(frozen=True, slots=True)
class ProgramCacheBlock:
    identity: str
    virtual_address: int
    byte_offset: int
    byte_capacity: int
    occupied_bytes: int

    @property
    def occupancy(self) -> float:
        return self.occupied_bytes / self.byte_capacity if self.byte_capacity else 0.0


@dataclass(frozen=True, slots=True)
class FixedProgramCacheLayout:
    base_offset: int
    line_bytes: int
    blocks: tuple[ProgramCacheBlock, ...]

    @property
    def byte_size(self) -> int:
        return sum(block.byte_capacity for block in self.blocks)

    def shader_words(self) -> tuple[tuple[int, int, int, int], ...]:
        """Return offset/capacity/occupied/address-low words for visualization."""

        return tuple(
            (
                block.byte_offset,
                block.byte_capacity,
                block.occupied_bytes,
                block.virtual_address & 0xFFFFFFFF,
            )
            for block in self.blocks
        )


def build_fixed_program_cache_layout(
    program,
    *,
    base_offset: int,
    line_bytes: int = 64,
) -> FixedProgramCacheLayout:
    """Give every decoded function a stable cache-line-aligned allocation."""

    if base_offset < 0:
        raise ValueError("program-cache base must be non-negative")
    _align(0, line_bytes)  # validates the line size
    cursor = _align(base_offset, line_bytes)
    blocks: list[ProgramCacheBlock] = []
    for index, record in enumerate(program.functions):
        instructions = tuple(record.report.instructions)
        if not instructions:
            continue
        begin = min(int(item.address) for item in instructions)
        end = max(int(item.address) + len(item.encoded) for item in instructions)
        occupied = sum(len(item.encoded) for item in instructions)
        capacity = _align(max(1, end - begin), line_bytes)
        identity = str(
            getattr(record, "name", None)
            or getattr(record, "identity", None)
            or f"function_{index}@{begin:#x}"
        )
        blocks.append(ProgramCacheBlock(
            identity, begin, cursor, capacity, occupied,
        ))
        cursor += capacity
    return FixedProgramCacheLayout(_align(base_offset, line_bytes), line_bytes, tuple(blocks))


@dataclass(frozen=True, slots=True)
class ExecutionClockPolicy:
    cycles_per_second: float = 60.0
    time_scale: float = 1.0
    maximum_cycles_per_frame: int = 256
    maximum_total_cycles: int = 1_000_000

    def __post_init__(self) -> None:
        if self.cycles_per_second <= 0 or self.time_scale < 0:
            raise ValueError("execution clock rates must be positive")
        if self.maximum_cycles_per_frame <= 0 or self.maximum_total_cycles <= 0:
            raise ValueError("execution clock bounds must be positive")


@dataclass(slots=True)
class BoundedMachineClock:
    """Wall-clock governor that prevents recursive/self-hosted time runaway."""

    machine: MachineVirtualMulticore
    policy: ExecutionClockPolicy
    total_cycles: int = 0
    fractional_cycles: float = 0.0

    def advance(self, elapsed_seconds: float) -> int:
        if elapsed_seconds < 0:
            raise ValueError("elapsed time cannot be negative")
        requested = (
            elapsed_seconds * self.policy.cycles_per_second * self.policy.time_scale
            + self.fractional_cycles
        )
        cycles = min(floor(requested), self.policy.maximum_cycles_per_frame)
        cycles = min(cycles, self.policy.maximum_total_cycles - self.total_cycles)
        self.fractional_cycles = requested - floor(requested)
        for _ in range(max(0, cycles)):
            self.machine.cycle_forward()
        self.total_cycles += max(0, cycles)
        return max(0, cycles)


__all__ = [
    "BoundedMachineClock",
    "ExecutionClockPolicy",
    "FixedProgramCacheLayout",
    "ProgramCacheBlock",
    "RegisterBankLayout",
    "RegisterCell",
    "build_fixed_program_cache_layout",
    "build_register_bank_layout",
    "pack_register_banks",
]
