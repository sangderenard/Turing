from dataclasses import replace
from types import SimpleNamespace

from src.compiler.machine_chip_layout import (
    BoundedMachineClock,
    ExecutionClockPolicy,
    build_fixed_program_cache_layout,
    build_register_bank_layout,
    pack_register_banks,
)
from src.compiler.machine_execution import (
    MachineExecutionOrchestrator,
    MachineVirtualMulticore,
)
from src.compiler.machine_execution_shader import build_machine_register_shader
from src.compiler.machine_reference_vocabulary import MachineSemanticToken


def _program():
    instructions = (
        SimpleNamespace(address=0x401000, encoded=b"\x90", semantic=MachineSemanticToken.INTEGER_ADD, operands=()),
        SimpleNamespace(address=0x401001, encoded=b"\xc3", semantic=MachineSemanticToken.RETURN, operands=()),
    )
    return SimpleNamespace(
        image=SimpleNamespace(image_base=0x400000, entrypoint_rva=0x1000),
        functions=(SimpleNamespace(name="self_host", report=SimpleNamespace(instructions=instructions)),),
    )


def _machine(core_count=2):
    def increment(state, _instruction):
        registers = list(state.registers)
        registers[0] += 1
        registers[4] = 0x123456789ABCDEF0
        return replace(state, registers=tuple(registers))

    executor = MachineExecutionOrchestrator(
        _program(),
        effect_handlers={int(MachineSemanticToken.INTEGER_ADD): increment},
    )
    return MachineVirtualMulticore.create(executor, core_count=core_count)


def test_every_register_is_an_individual_contiguous_u64_cell_on_each_core():
    layout = build_register_bank_layout(2, base_offset=0x2000)

    assert layout.core_stride == 512
    assert layout.byte_size == 1024
    for core in range(2):
        cells = [layout.cell(core, name) for name in layout.register_names]
        assert all(cell.word_offsets[1] - cell.word_offsets[0] == 4 for cell in cells)
        assert [cell.byte_offset for cell in cells] == list(range(
            0x2000 + core * 512,
            0x2000 + core * 512 + len(cells) * 8,
            8,
        ))


def test_fixed_register_arena_preserves_full_values_and_bank_padding():
    machine = _machine()
    machine.cycle_forward()
    layout = build_register_bank_layout(2)

    words = pack_register_banks(machine, layout)

    assert len(words) == layout.byte_size // 4
    # RSP is the fifth register: adjacent little-endian low/high words.
    assert words[8:10] == (0x9ABCDEF0, 0x12345678)
    second_bank = layout.core_stride // 4
    assert words[second_bank + 8:second_bank + 10] == (0x9ABCDEF0, 0x12345678)


def test_program_cache_uses_fixed_aligned_blocks_with_visible_occupancy():
    cache = build_fixed_program_cache_layout(_program(), base_offset=0x3003)
    block = cache.blocks[0]

    assert cache.base_offset == 0x3040
    assert block.byte_offset == 0x3040
    assert block.byte_capacity == 64
    assert block.occupied_bytes == 2
    assert block.occupancy == 2 / 64
    assert cache.shader_words() == ((0x3040, 64, 2, 0x401000),)


def test_execution_clock_drops_runaway_time_instead_of_catching_up_forever():
    machine = _machine(core_count=1)
    clock = BoundedMachineClock(machine, ExecutionClockPolicy(
        cycles_per_second=1_000_000,
        time_scale=1000,
        maximum_cycles_per_frame=1,
        maximum_total_cycles=1,
    ))

    assert clock.advance(60.0) == 1
    assert clock.advance(60.0) == 0
    assert clock.total_cycles == 1


def test_shader_has_separate_register_and_cache_update_kernels():
    source = build_machine_register_shader().source

    assert "state_snapshot: array<u32>" in source
    assert "core_index * register_stride_words" in source
    assert "fn update_machine_registers" in source
    assert "fn update_program_cache" in source
    assert "occupied_bytes" in source
