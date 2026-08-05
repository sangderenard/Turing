from dataclasses import replace
from types import SimpleNamespace
import struct
import time

import pytest

from src.compiler.machine_chip_layout import build_register_bank_layout
from src.compiler.binary_machine_program import BinaryMachineProgram
from src.compiler.amd64_machine_semantics import PagedByteMemory
from src.compiler.machine_execution import (
    MachineExternalCallRequest,
    MachineExecutionOrchestrator,
    MachineExecutionStatus,
    MachineVirtualMulticore,
)
from src.compiler.machine_reference_vocabulary import MachineSemanticToken, X86InstructionToken
from src.compiler.machine_state_buffer import (
    ExternalMachineClock,
    FreeRunningMachineRunner,
    MachineRunDirection,
    MachineSnapshotLayout,
    MachineSnapshotTripleBuffer,
    MachineSnapshotView,
    SubjectOutputBuffer,
    SubjectOutputFormat,
    SubjectOutputKind,
    build_machine_state_snapshot,
)
from src.compiler.machine_tape_segments import SegmentedMachineTapeStore
from src.compiler.machine_system_tape import MachineSystemTape
from src.compiler.machine_system_ports import deterministic_windows_bootstrap_port


def _machine(core_count=2):
    instruction = SimpleNamespace(
        address=0x401000, encoded=b"\x90",
        token=X86InstructionToken.NOP,
        semantic=MachineSemanticToken.INTEGER_ADD, operands=(),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0x400000, entrypoint_rva=0x1000),
        functions=(SimpleNamespace(report=SimpleNamespace(instructions=(instruction,))),),
    )

    def increment(state, _instruction):
        registers = list(state.registers)
        registers[0] += 1
        return replace(state, pc=0x401000, registers=tuple(registers))

    executor = MachineExecutionOrchestrator(
        program, effect_handlers={int(MachineSemanticToken.INTEGER_ADD): increment},
    )
    return MachineVirtualMulticore.create(executor, core_count=core_count)


def _buffers(machine, *, output_bytes=1024):
    registers = build_register_bank_layout(len(machine.cores))
    layout = MachineSnapshotLayout.build(
        registers, core_count=len(machine.cores), maximum_output_bytes=output_bytes,
    )
    return MachineSnapshotTripleBuffer(layout, registers)


def _minimal_amd64_pe_return():
    """Two-section PE32+ whose one runtime-described function is RET."""

    image = bytearray(0x800)
    image[:2] = b"MZ"
    struct.pack_into("<I", image, 0x3C, 0x80)
    image[0x80:0x84] = b"PE\0\0"
    coff = 0x84
    struct.pack_into("<HHIIIHH", image, coff, 0x8664, 2, 0, 0, 0, 0xF0, 0x22)
    optional = coff + 20
    struct.pack_into("<H", image, optional, 0x20B)
    struct.pack_into("<I", image, optional + 16, 0x1000)
    struct.pack_into("<Q", image, optional + 24, 0x140000000)
    struct.pack_into("<I", image, optional + 108, 16)
    struct.pack_into("<II", image, optional + 112 + 3 * 8, 0x2000, 12)
    sections = optional + 0xF0
    image[sections:sections + 8] = b".text\0\0\0"
    struct.pack_into("<IIII", image, sections + 8, 1, 0x1000, 0x200, 0x400)
    struct.pack_into("<I", image, sections + 36, 0x60000020)
    pdata = sections + 40
    image[pdata:pdata + 8] = b".pdata\0\0"
    struct.pack_into("<IIII", image, pdata + 8, 12, 0x2000, 0x200, 0x600)
    struct.pack_into("<I", image, pdata + 36, 0x40000040)
    image[0x400] = 0xC3
    struct.pack_into("<III", image, 0x600, 0x1000, 0x1001, 0)
    return bytes(image)


def _minimal_relocatable_amd64_pe_return():
    """PE32+ RET image with one DIR64 relocation in a data section."""

    image = bytearray(0xC00)
    image[:2] = b"MZ"
    struct.pack_into("<I", image, 0x3C, 0x80)
    image[0x80:0x84] = b"PE\0\0"
    coff = 0x84
    struct.pack_into("<HHIIIHH", image, coff, 0x8664, 4, 0, 0, 0, 0xF0, 0x22)
    optional = coff + 20
    struct.pack_into("<H", image, optional, 0x20B)
    struct.pack_into("<I", image, optional + 16, 0x1000)
    struct.pack_into("<Q", image, optional + 24, 0x140000000)
    struct.pack_into("<I", image, optional + 108, 16)
    struct.pack_into("<II", image, optional + 112 + 3 * 8, 0x2000, 12)
    struct.pack_into("<II", image, optional + 112 + 5 * 8, 0x4000, 12)
    sections = optional + 0xF0
    rows = (
        (b".text\0\0\0", 1, 0x1000, 0x200, 0x400, 0x60000020),
        (b".pdata\0\0", 12, 0x2000, 0x200, 0x600, 0x40000040),
        (b".data\0\0\0", 8, 0x3000, 0x200, 0x800, 0xC0000040),
        (b".reloc\0\0", 12, 0x4000, 0x200, 0xA00, 0x42000040),
    )
    for index, (name, virtual_size, rva, raw_size, raw_offset, flags) in enumerate(rows):
        section = sections + index * 40
        image[section:section + 8] = name
        struct.pack_into(
            "<IIII", image, section + 8,
            virtual_size, rva, raw_size, raw_offset,
        )
        struct.pack_into("<I", image, section + 36, flags)
    image[0x400] = 0xC3
    struct.pack_into("<III", image, 0x600, 0x1000, 0x1001, 0)
    struct.pack_into("<Q", image, 0x800, 0x140001000)
    struct.pack_into("<IIHH", image, 0xA00, 0x3000, 12, 0xA000, 0)
    return bytes(image)


def _minimal_exporting_amd64_pe_return():
    image = bytearray(_minimal_relocatable_amd64_pe_return())
    struct.pack_into("<H", image, 0x84 + 18, 0x2022)
    optional = 0x84 + 20
    struct.pack_into("<II", image, optional + 112, 0x3100, 0x60)
    export = 0x900  # .data RVA 0x3100
    struct.pack_into(
        "<IIHHIIIIIII",
        image,
        export,
        0, 0, 0, 0,
        0x3180, 1, 2, 2,
        0x3128, 0x3130, 0x3138,
    )
    struct.pack_into("<II", image, 0x928, 0x1000, 0x3140)
    struct.pack_into("<II", image, 0x930, 0x3190, 0x31A0)
    struct.pack_into("<HH", image, 0x938, 0, 1)
    image[0x940:0x940 + len(b"KERNEL32.Sleep\0")] = b"KERNEL32.Sleep\0"
    image[0x980:0x980 + len(b"demo.dll\0")] = b"demo.dll\0"
    image[0x990:0x990 + len(b"Real\0")] = b"Real\0"
    image[0x9A0:0x9A0 + len(b"Forwarded\0")] = b"Forwarded\0"
    return bytes(image)


def _minimal_importing_amd64_pe_call():
    """Main PE that calls demo.dll!Real through its IAT, then returns."""

    image = bytearray(0xA00)
    image[:2] = b"MZ"
    struct.pack_into("<I", image, 0x3C, 0x80)
    image[0x80:0x84] = b"PE\0\0"
    coff = 0x84
    struct.pack_into("<HHIIIHH", image, coff, 0x8664, 3, 0, 0, 0, 0xF0, 0x22)
    optional = coff + 20
    struct.pack_into("<H", image, optional, 0x20B)
    struct.pack_into("<I", image, optional + 16, 0x1000)
    struct.pack_into("<Q", image, optional + 24, 0x400000)
    struct.pack_into("<I", image, optional + 108, 16)
    struct.pack_into("<II", image, optional + 112 + 1 * 8, 0x2100, 40)
    struct.pack_into("<II", image, optional + 112 + 3 * 8, 0x3000, 12)
    sections = optional + 0xF0
    rows = (
        (b".text\0\0\0", 7, 0x1000, 0x200, 0x400, 0x60000020),
        (b".idata\0\0", 0x200, 0x2000, 0x200, 0x600, 0xC0000040),
        (b".pdata\0\0", 12, 0x3000, 0x200, 0x800, 0x40000040),
    )
    for index, (name, virtual_size, rva, raw_size, raw_offset, flags) in enumerate(rows):
        section = sections + index * 40
        image[section:section + 8] = name
        struct.pack_into(
            "<IIII", image, section + 8,
            virtual_size, rva, raw_size, raw_offset,
        )
        struct.pack_into("<I", image, section + 36, flags)
    image[0x400:0x407] = b"\xff\x15\xfa\x0f\x00\x00\xc3"
    struct.pack_into("<QQ", image, 0x600, 0x2040, 0)
    struct.pack_into("<QQ", image, 0x610, 0x2040, 0)
    image[0x640:0x647] = b"\x00\x00Real\x00"
    image[0x660:0x669] = b"demo.dll\x00"
    struct.pack_into("<IIIII", image, 0x700, 0x2010, 0, 0, 0x2060, 0x2000)
    struct.pack_into("<III", image, 0x800, 0x1000, 0x1007, 0)
    return bytes(image)


def _minimal_importing_forwarded_amd64_pe_call():
    image = bytearray(_minimal_importing_amd64_pe_call())
    image[0x640:0x64C] = b"\x00\x00Forwarded\x00"
    return bytes(image)


def _minimal_delay_importing_amd64_pe_call():
    image = bytearray(_minimal_importing_amd64_pe_call())
    optional = 0x84 + 20
    struct.pack_into("<II", image, optional + 112 + 1 * 8, 0, 0)
    struct.pack_into("<II", image, optional + 112 + 13 * 8, 0x2100, 64)
    struct.pack_into(
        "<IIIIIIII", image, 0x700,
        1, 0x2060, 0x2080, 0x2000, 0x2010, 0, 0, 0,
    )
    image[0x720:0x740] = bytes(32)
    return bytes(image)


def _minimal_exporting_amd64_pe_answer():
    image = bytearray(_minimal_exporting_amd64_pe_return())
    image[0x400:0x406] = b"\xb8\x2a\x00\x00\x00\xc3"
    struct.pack_into("<I", image, 0x188 + 8, 6)
    struct.pack_into("<I", image, 0x604, 0x1006)
    return bytes(image)


def _minimal_exporting_kernel32_sleep_answer():
    image = bytearray(_minimal_exporting_amd64_pe_answer())
    image[0x980:0x98D] = b"KERNEL32.dll\x00"
    image[0x990:0x996] = b"Sleep\x00"
    return bytes(image)


def _minimal_tls_amd64_pe_return():
    image = bytearray(_minimal_relocatable_amd64_pe_return())
    optional = 0x84 + 20
    struct.pack_into("<II", image, optional + 112 + 9 * 8, 0x3100, 40)
    text_section = optional + 0xF0
    data_section = text_section + 2 * 40
    struct.pack_into("<I", image, text_section + 8, 0x20)
    struct.pack_into("<I", image, data_section + 8, 0x200)
    image[0x410:0x416] = b"\xb8\x07\x00\x00\x00\xc3"
    struct.pack_into(
        "<QQQQII", image, 0x900,
        0x140003180, 0x140003184,
        0x140003190, 0x1400031A0,
        4, 0,
    )
    image[0x980:0x984] = b"ABCD"
    struct.pack_into("<QQ", image, 0x9A0, 0x140001010, 0)
    return bytes(image)


def _minimal_amd64_pe_thread_loop():
    image = bytearray(_minimal_relocatable_amd64_pe_return())
    optional = 0x84 + 20
    text_section = optional + 0xF0
    struct.pack_into("<I", image, text_section + 8, 0x20)
    image[0x400:0x402] = b"\xeb\xfe"
    image[0x410:0x416] = b"\xb8\x09\x00\x00\x00\xc3"
    struct.pack_into("<II", image, 0x600, 0x1000, 0x1002)
    return bytes(image)


def _minimal_tls_amd64_pe_thread_loop():
    image = bytearray(_minimal_tls_amd64_pe_return())
    optional = 0x84 + 20
    text_section = optional + 0xF0
    struct.pack_into("<I", image, text_section + 8, 0x40)
    image[0x400:0x402] = b"\xeb\xfe"
    image[0x420:0x426] = b"\xb8\x09\x00\x00\x00\xc3"
    image[0x430:0x436] = b"\xb8\x08\x00\x00\x00\xc3"
    struct.pack_into(
        "<QQQ", image, 0x9A0,
        0x140001010, 0x140001030, 0,
    )
    struct.pack_into("<II", image, 0x600, 0x1000, 0x1002)
    return bytes(image)


def test_complete_register_banks_and_subject_output_flip_together():
    machine = _machine()
    machine.cycle_forward()
    buffers = _buffers(machine)
    generation = buffers.publish(
        machine, direction=MachineRunDirection.FORWARD, transitions=1,
        outputs=(SubjectOutputBuffer(
            SubjectOutputKind.FRAMEBUFFER, SubjectOutputFormat.RGBA8,
            b"\x01\x02\x03\xff", width=1, height=1, channels=4,
            row_stride=4, generation=7,
        ),),
    )

    with buffers.lease_latest() as snapshot:
        assert snapshot is not None
        assert snapshot.header.generation == generation
        assert snapshot.header.slot_index in (0, 1, 2)
        assert snapshot.register_words(0, 0) == (1, 0)
        assert snapshot.register_words(1, 0) == (1, 0)
        assert snapshot.core_status(0)["history_position"] == 1
        descriptor = snapshot.output_descriptor(0)
        assert descriptor.kind is SubjectOutputKind.FRAMEBUFFER
        assert descriptor.generation == 7
        assert bytes(snapshot.output_bytes(0)) == b"\x01\x02\x03\xff"


def test_snapshot_carries_colored_tape_annotation_flag_per_core():
    machine = _machine(core_count=1)
    registers = build_register_bank_layout(1)
    layout = MachineSnapshotLayout.build(registers, core_count=1)
    buffers = MachineSnapshotTripleBuffer(
        layout, registers, core_annotation_provider=lambda core, position: 0xFF3030FF,
    )
    buffers.publish(machine, direction=MachineRunDirection.PAUSED, transitions=0)
    with buffers.lease_latest() as snapshot:
        assert snapshot.core_status(0)["annotation_rgba8"] == 0xFF3030FF


def test_missing_instruction_handler_annotates_rip_with_compatibility_note():
    program = _machine(core_count=1).cores[0].executor.program
    runtime = BinaryMachineProgram.from_program(program, effect_handlers={})
    runtime.set_direction(MachineRunDirection.FORWARD)
    runtime.runner.tick(1)

    annotation = runtime.system_tape.annotations[-1]
    assert annotation.feature == "instruction_set_compatibility"
    assert annotation.address == 0x401000
    assert annotation.color == "red"
    assert dict(annotation.metadata) == {
        "architecture": "windows-amd64",
        "compatibility": "unsupported",
        "encoded": "90",
        "instruction_token": "NOP",
        "semantic_token": "INTEGER_ADD",
    }
    with runtime.snapshots.lease_latest() as snapshot:
        assert snapshot.core_status(0)["annotation_rgba8"] == 0xFF3030FF


def test_publisher_never_overwrites_the_slot_leased_by_display():
    machine = _machine(core_count=1)
    buffers = _buffers(machine)
    buffers.publish(machine, direction=MachineRunDirection.FORWARD, transitions=0)

    with buffers.lease_latest() as displayed:
        assert displayed is not None
        held_slot = displayed.header.slot_index
        held_generation = displayed.header.generation
        for transition in range(1, 8):
            machine.cycle_forward()
            buffers.publish(
                machine, direction=MachineRunDirection.FORWARD,
                transitions=transition,
            )
            assert displayed.header.slot_index == held_slot
            assert displayed.header.generation == held_generation
            assert displayed.register_words(0, 0) == (0, 0)


def test_free_runner_outpaces_sampling_and_can_free_spin_backward():
    machine = _machine(core_count=2)
    buffers = _buffers(machine)
    runner = FreeRunningMachineRunner(
        machine, buffers, transitions_per_publication=8,
    )
    runner.start()
    deadline = time.monotonic() + 1.0
    while runner.transitions < 32 and time.monotonic() < deadline:
        time.sleep(0.001)
    runner.set_direction(MachineRunDirection.PAUSED)
    forward_position = machine.cores[0].position
    sampled_generation = buffers.publication[0]
    assert forward_position >= 32
    assert sampled_generation < runner.transitions

    runner.set_direction(MachineRunDirection.BACKWARD)
    deadline = time.monotonic() + 1.0
    while machine.cores[0].position >= forward_position and time.monotonic() < deadline:
        time.sleep(0.001)
    runner.set_direction(MachineRunDirection.PAUSED)
    runner.stop()

    assert runner.failure is None
    assert machine.cores[0].position < forward_position
    with buffers.lease_latest() as snapshot:
        assert snapshot is not None
        assert snapshot.header.generation > sampled_generation
        assert snapshot.header.direction in {
            MachineRunDirection.BACKWARD, MachineRunDirection.PAUSED,
        }


def test_binary_program_bounds_hot_history_and_rehydrates_exact_tape_states():
    program = _machine(core_count=1).cores[0].executor.program

    def increment(state, _instruction):
        registers = list(state.registers)
        registers[0] += 1
        return replace(state, pc=0x401000, registers=tuple(registers))

    runtime = BinaryMachineProgram.from_program(
        program,
        effect_handlers={int(MachineSemanticToken.INTEGER_ADD): increment},
        maximum_hot_history_states=4,
    )
    runtime.set_direction(MachineRunDirection.FORWARD)
    assert runtime.runner.tick(10) == 10
    core = runtime.machine.cores[0]

    assert core.position == 10
    assert core.hot_history_range == (7, 11)
    assert len(core._states) == 4
    assert core.state.register_contents()["rax"] == 10

    runtime.set_direction(MachineRunDirection.BACKWARD)
    assert runtime.runner.tick(10) == 10
    assert core.position == 0
    assert core.state.register_contents()["rax"] == 0
    assert len(core._states) <= 4
    assert any(record["event"] == "backward" for record in runtime.system_tape.records)


def test_multicore_shared_memory_schedule_is_retained_on_exact_tape(tmp_path):
    program = _machine(core_count=1).cores[0].executor.program

    def increment(state, _instruction):
        registers = list(state.registers)
        registers[0] += 1
        return replace(state, pc=0x401000, registers=tuple(registers))

    runtime = BinaryMachineProgram.from_program(
        program, core_count=2,
        effect_handlers={int(MachineSemanticToken.INTEGER_ADD): increment},
    )
    runtime.set_direction(MachineRunDirection.FORWARD)
    assert runtime.runner.tick(1) == 1

    records = runtime.system_tape.records[-2:]
    for record in records:
        commit = record["metadata"]["shared_memory_commit"]
        assert commit["cycle_index"] == 1
        assert commit["core_order"] == (0, 1)
        assert commit["core_positions"] == (1, 1)

    reopened = MachineSystemTape.read(
        runtime.save_system_tape(tmp_path / "shared-memory.tape.jsonl"),
    )
    persisted = reopened.records[-1]["metadata"]["shared_memory_commit"]
    assert persisted["core_order"] == [0, 1]
    assert persisted["core_positions"] == [1, 1]


def test_cold_reverse_streams_from_segmented_tape_without_resident_future(tmp_path):
    program = _machine(core_count=1).cores[0].executor.program

    def increment(state, _instruction):
        registers = list(state.registers)
        registers[0] += 1
        return replace(state, pc=0x401000, registers=tuple(registers))

    runtime = BinaryMachineProgram.from_program(
        program,
        effect_handlers={int(MachineSemanticToken.INTEGER_ADD): increment},
        maximum_hot_history_states=4,
    )
    runtime.set_direction(MachineRunDirection.FORWARD)
    assert runtime.runner.tick(20) == 20
    jsonl = runtime.save_system_tape(tmp_path / "hot.tape.jsonl")
    store = SegmentedMachineTapeStore.import_jsonl(
        jsonl, tmp_path / "hot.segmented-tape", records_per_segment=4,
    )
    store.begin_append()
    runtime.system_tape = store

    runtime.set_direction(MachineRunDirection.BACKWARD)
    assert runtime.runner.tick(20) == 20
    store.flush()

    core = runtime.machine.cores[0]
    assert core.position == 0
    assert core.state.register_contents()["rax"] == 0
    assert len(core._states) <= 4
    reopened = SegmentedMachineTapeStore(store.root)
    assert reopened.record_count == 41
    assert reopened.resume_state().register_contents()["rax"] == 0


def test_shell_clock_sets_speed_and_one_tick_publishes_one_complete_flip():
    machine = _machine(core_count=1)
    buffers = _buffers(machine)
    runner = FreeRunningMachineRunner(machine, buffers)
    runner.set_direction(MachineRunDirection.FORWARD)
    clock = ExternalMachineClock(
        transitions_per_second=100.0, maximum_transitions_per_tick=100,
    )

    assert runner.regulated_tick(clock, 0.025) == 2
    first_generation = buffers.publication[0]
    assert machine.cores[0].position == 2

    clock.set_speed(200.0)
    assert runner.regulated_tick(clock, 0.025) == 5
    assert buffers.publication[0] == first_generation + 1
    assert machine.cores[0].position == 7

    runner.set_direction(MachineRunDirection.BACKWARD)
    assert runner.tick(3) == 3
    assert machine.cores[0].position == 4
    with buffers.lease_latest() as snapshot:
        assert snapshot is not None
        assert snapshot.header.direction is MachineRunDirection.BACKWARD


def test_runtime_coordinator_keeps_subject_machine_out_of_card_ssa_and_publishes_devices():
    source_machine = _machine(core_count=1)
    executor = source_machine.cores[0].executor
    runtime = BinaryMachineProgram.from_program(
        executor.program,
        core_count=1,
        transitions_per_second=10.0,
        effect_handlers=executor.effect_handlers,
    )
    runtime.devices.publish((SubjectOutputBuffer(
        SubjectOutputKind.TERMINAL, SubjectOutputFormat.UTF8, b"hello", generation=1,
    ),))
    runtime.set_direction(MachineRunDirection.FORWARD)

    assert runtime.tick(0.2) == 2
    assert runtime.machine.cores[0].state.register_contents()["rax"] == 2
    with runtime.snapshots.lease_latest() as snapshot:
        assert snapshot is not None
        assert snapshot.header.transitions == 2
        assert snapshot.output_descriptor(0).kind is SubjectOutputKind.TERMINAL
        assert bytes(snapshot.output_bytes(0)) == b"hello"


def test_reversible_console_device_state_reaches_the_shader_snapshot_abi():
    source_machine = _machine(core_count=1)
    executor = source_machine.cores[0].executor
    runtime = BinaryMachineProgram.from_program(
        executor.program, core_count=1, effect_handlers=executor.effect_handlers,
    )
    observed = replace(
        runtime.machine.cores[0].state,
        device_state={"console.output": b"hello\r\n"},
        device_generations={"console.output": 4},
    )
    runtime._sync_devices(observed)
    runtime.snapshots.publish(
        runtime.machine, direction=MachineRunDirection.PAUSED, transitions=0,
        outputs=runtime.devices.snapshot(),
    )

    with runtime.snapshots.lease_latest() as snapshot:
        descriptor = snapshot.output_descriptor(0)
        assert descriptor.kind is SubjectOutputKind.TERMINAL
        assert descriptor.format is SubjectOutputFormat.UTF8
        assert descriptor.generation == 4
        assert bytes(snapshot.output_bytes(0)) == b"hello\r\n"


def test_snapshot_exposes_page_aligned_memory_occupancy_for_the_shader():
    state = replace(
        _machine(1).cores[0].state,
        memory=PagedByteMemory.empty().map_zeroes(0x4000, 4096).map_bytes(
            0x4010, b"four",
        ),
        steps=9,
    )
    snapshot = MachineSnapshotView(memoryview(build_machine_state_snapshot((state,))))
    descriptor = snapshot.output_descriptor(0)
    assert descriptor.kind is SubjectOutputKind.MEMORY_PAGES
    assert descriptor.format is SubjectOutputFormat.PAGE_OCCUPANCY_V1
    assert (descriptor.width, descriptor.row_stride, descriptor.generation) == (1, 16, 9)
    page_index, occupied, flags = struct.unpack("<QII", snapshot.output_bytes(0))
    assert page_index == 4
    assert occupied == 4
    assert flags & 1


def test_shell_console_input_is_a_reversible_taped_machine_effect():
    source_machine = _machine(core_count=1)
    executor = source_machine.cores[0].executor
    runtime = BinaryMachineProgram.from_program(
        executor.program, core_count=1, effect_handlers=executor.effect_handlers,
    )

    runtime.inject_console_input("dir\r\n")

    assert runtime.machine.cores[0].state.device_state["console.input"] == b"dir\r\n"
    assert runtime.system_tape.records[-1]["event"] == "shell_device_input"
    assert runtime.system_tape.records[-1]["metadata"] == {
        "device": "console.input", "bytes": 5, "append": True,
    }
    runtime.machine.cores[0].step_backward()
    assert "console.input" not in runtime.machine.cores[0].state.device_state


def test_runtime_loader_uses_existing_pe_decompiler_before_clocked_execution():
    runtime = BinaryMachineProgram.load_pe(
        _minimal_amd64_pe_return(),
        maximum_file_size=4096,
        transitions_per_second=1.0,
    )
    runtime.set_direction(MachineRunDirection.FORWARD)

    assert len(runtime.program.functions) == 1
    assert runtime.program.functions[0].report.instructions[0].encoded == b"\xc3"
    assert runtime.tick(1.0) == 1
    assert runtime.runner.direction is MachineRunDirection.PAUSED
    with runtime.snapshots.lease_latest() as snapshot:
        assert snapshot is not None
        assert snapshot.core_status(0)["status"] != 0


def test_runtime_loader_relocates_image_and_decoded_instruction_namespace():
    runtime_base = 0x150000000
    runtime = BinaryMachineProgram.load_pe(
        _minimal_relocatable_amd64_pe_return(),
        maximum_file_size=4096,
        load_address=runtime_base,
        transitions_per_second=1.0,
    )
    state = runtime.machine.cores[0].state

    assert [(item.type, item.rva) for item in runtime.program.image.base_relocations] == [
        (10, 0x3000),
    ]
    assert state.pc == runtime_base + 0x1000
    assert state.memory.read_unsigned(runtime_base + 0x3000, 64) == runtime_base + 0x1000
    assert state.system_state["windows.loader.preferred_image_base"] == 0x140000000
    assert state.system_state["windows.loader.image_base"] == runtime_base
    assert state.system_state["windows.loader.load_bias"] == 0x10000000
    assert state.system_state["windows.loader.base_relocation_count"] == 1
    assert runtime_base + 0x1000 in runtime.machine.cores[0].executor.instructions

    runtime.set_direction(MachineRunDirection.FORWARD)
    assert runtime.tick(1.0) == 1
    assert runtime.machine.cores[0].latest_edge.status is MachineExecutionStatus.HALTED
    assert runtime.runner.direction is MachineRunDirection.PAUSED


def test_relocated_runtime_tape_restores_loader_base_without_user_hint(tmp_path):
    runtime_base = 0x150000000
    runtime = BinaryMachineProgram.load_pe(
        _minimal_relocatable_amd64_pe_return(),
        maximum_file_size=4096,
        load_address=runtime_base,
    )
    tape_path = runtime.save_system_tape(tmp_path / "relocated.tape.jsonl")

    resumed = BinaryMachineProgram.load_system_tape(
        tape_path, maximum_file_size=4096,
    )

    assert resumed.machine.cores[0].executor.image_base == runtime_base
    assert resumed.machine.cores[0].state.pc == runtime_base + 0x1000
    assert runtime_base + 0x1000 in resumed.machine.cores[0].executor.instructions
    with pytest.raises(ValueError, match="conflicts with the exact tape"):
        BinaryMachineProgram.load_system_tape(
            tape_path,
            maximum_file_size=4096,
            load_address=0x160000000,
        )


def test_loader_fails_closed_when_moved_image_has_no_relocation_directory():
    with pytest.raises(ValueError, match="no base relocation records"):
        BinaryMachineProgram.load_pe(
            _minimal_amd64_pe_return(),
            maximum_file_size=4096,
            load_address=0x150000000,
        )


def test_loader_fails_closed_on_unsupported_relocation_kind():
    image = bytearray(_minimal_relocatable_amd64_pe_return())
    # Change IMAGE_REL_BASED_DIR64 to an unsupported type while retaining the
    # relocation offset and otherwise valid bounded directory structure.
    struct.pack_into("<H", image, 0xA08, 0x5000)

    with pytest.raises(ValueError, match="unsupported PE base relocation type 5"):
        BinaryMachineProgram.load_pe(
            bytes(image),
            maximum_file_size=4096,
            load_address=0x150000000,
        )


def test_pe_loader_catalogues_named_ordinal_and_forwarded_exports():
    runtime = BinaryMachineProgram.load_pe(
        _minimal_exporting_amd64_pe_return(), maximum_file_size=4096,
    )

    assert runtime.program.image.export_name == "demo.dll"
    assert [
        (item.name, item.ordinal, item.rva, item.forwarder)
        for item in runtime.program.image.exports
    ] == [
        ("Real", 1, 0x1000, None),
        ("Forwarded", 2, None, "KERNEL32.Sleep"),
    ]
    assert runtime.program.image.export_by_name("Real").rva == 0x1000
    assert runtime.program.image.export_by_name("real") is None
    assert runtime.program.image.export_by_ordinal(2).forwarder == "KERNEL32.Sleep"


def test_capability_supplied_dependency_executes_through_bound_guest_iat(tmp_path):
    runtime = BinaryMachineProgram.load_pe(
        _minimal_importing_amd64_pe_call(),
        maximum_file_size=4096,
        dependency_modules={
            "demo.dll": _minimal_exporting_amd64_pe_answer(),
        },
    )
    core = runtime.machine.cores[0]
    dependency_base = 0x140000000

    assert core.state.memory.read_unsigned(0x402000, 64) == dependency_base + 0x1000
    assert not runtime.external_references
    assert runtime.import_bindings[0].resolution_kind == "mapped_export"
    assert runtime.import_bindings[0].resolved_library == "demo.dll"
    assert dependency_base + 0x1000 in core.executor.instructions

    runtime.set_direction(MachineRunDirection.FORWARD)
    assert runtime.runner.tick(64) == 6
    assert core.state.registers[0] == 42
    assert core.latest_edge.status is MachineExecutionStatus.HALTED
    assert runtime.runner.direction is MachineRunDirection.PAUSED
    assert runtime.linked_modules[0].program.image.is_dll
    assert core.state.system_state["windows.loader.startup_calls_complete"] == 1

    # Dependency bytes, address and every IAT witness remain tape authority.
    fresh = BinaryMachineProgram.load_pe(
        _minimal_importing_amd64_pe_call(),
        maximum_file_size=4096,
        dependency_modules={
            "demo.dll": _minimal_exporting_amd64_pe_answer(),
        },
    )
    tape_path = fresh.save_system_tape(tmp_path / "linked.tape.jsonl")
    reopened = MachineSystemTape.read(tape_path)
    assert reopened.linked_modules[0].library == "demo.dll"
    assert reopened.linked_modules[0].digest == fresh.linked_modules[0].digest
    assert reopened.import_bindings == list(fresh.import_bindings)

    resumed = BinaryMachineProgram.load_system_tape(
        tape_path, maximum_file_size=4096,
    )
    resumed.set_direction(MachineRunDirection.FORWARD)
    assert resumed.runner.tick(64) == 6
    assert resumed.machine.cores[0].state.registers[0] == 42

    store = SegmentedMachineTapeStore.import_jsonl(
        tape_path, tmp_path / "linked-segments", records_per_segment=1,
    )
    assert (store.root / "modules" / f"{fresh.linked_modules[0].digest}.bin").is_file()
    segmented = BinaryMachineProgram.load_segmented_system_tape(
        store, maximum_file_size=4096,
    )
    segmented.set_direction(MachineRunDirection.FORWARD)
    assert segmented.runner.tick(64) == 6
    assert segmented.machine.cores[0].state.registers[0] == 42


def test_unmapped_forwarder_leaf_remains_a_capability_request_with_provenance():
    runtime = BinaryMachineProgram.load_pe(
        _minimal_importing_forwarded_amd64_pe_call(),
        maximum_file_size=4096,
        dependency_modules={"demo.dll": _minimal_exporting_amd64_pe_answer()},
    )

    assert len(runtime.external_references) == 1
    reference = runtime.external_references[0]
    assert (reference.library, reference.symbol) == ("KERNEL32", "Sleep")
    binding = runtime.import_bindings[0]
    assert binding.resolution_kind == "capability"
    assert binding.forwarder_chain == (
        "demo.dll!Forwarded->KERNEL32.Sleep",
    )

    runtime.set_direction(MachineRunDirection.FORWARD)
    assert runtime.runner.tick(64) == 3
    request = runtime.pending_external_requests()[0]
    assert request.reference == reference


def test_false_dllmain_process_attach_result_traps_before_subject_entry():
    runtime = BinaryMachineProgram.load_pe(
        _minimal_importing_amd64_pe_call(),
        maximum_file_size=4096,
        dependency_modules={"demo.dll": _minimal_exporting_amd64_pe_return()},
    )
    core = runtime.machine.cores[0]
    assert core.state.pc == 0x140001000
    assert core.state.registers[1:3] == (0x140000000, 1)

    runtime.set_direction(MachineRunDirection.FORWARD)
    assert runtime.runner.tick(64) == 1
    result = runtime.runner._last_results[0]
    assert result.status is MachineExecutionStatus.TRAPPED
    assert "returned false" in result.reason
    assert result.state.system_state["windows.loader.startup_failure_kind"] == 2
    assert result.state.system_state["windows.loader.startup_failure_index"] == 0


def test_dependency_linker_fails_closed_on_overlapping_guest_images():
    with pytest.raises(ValueError, match="linked PE images.*overlap"):
        BinaryMachineProgram.load_pe(
            _minimal_importing_amd64_pe_call(),
            maximum_file_size=4096,
            dependency_modules={"demo.dll": _minimal_exporting_amd64_pe_answer()},
            dependency_load_addresses={"demo.dll": 0x400000},
        )


def test_dependency_linker_cannot_overwrite_reserved_guest_environment():
    with pytest.raises(ValueError, match="guest stack.*PE image|PE image.*guest stack"):
        BinaryMachineProgram.load_pe(
            _minimal_importing_amd64_pe_call(),
            maximum_file_size=4096,
            dependency_modules={"demo.dll": _minimal_exporting_amd64_pe_answer()},
            dependency_load_addresses={"demo.dll": 0x00007FFEFFFFE000},
        )


def test_delay_import_is_lowered_to_witnessed_iat_and_module_handle():
    runtime = BinaryMachineProgram.load_pe(
        _minimal_delay_importing_amd64_pe_call(),
        maximum_file_size=4096,
        dependency_modules={"demo.dll": _minimal_exporting_amd64_pe_answer()},
    )
    dependency_base = 0x140000000

    assert not runtime.program.image.imports
    assert runtime.program.image.delay_imports[0].display_name == "demo.dll!Real"
    assert runtime.import_bindings[0].is_delay
    state = runtime.machine.cores[0].state
    assert state.memory.read_unsigned(0x402000, 64) == dependency_base + 0x1000
    assert state.memory.read_unsigned(0x402080, 64) == dependency_base

    runtime.set_direction(MachineRunDirection.FORWARD)
    assert runtime.runner.tick(64) == 6
    assert runtime.machine.cores[0].state.registers[0] == 42


def test_dependency_provider_recursively_supplies_only_requested_module_bytes():
    requests = []

    def provider(library):
        requests.append(library)
        return (
            _minimal_exporting_amd64_pe_answer()
            if library.casefold() == "demo.dll" else None
        )

    runtime = BinaryMachineProgram.load_pe(
        _minimal_importing_amd64_pe_call(),
        maximum_file_size=4096,
        dependency_provider=provider,
    )

    assert requests == ["demo.dll"]
    assert [module.requested_library for module in runtime.linked_modules] == [
        "demo.dll",
    ]
    runtime.set_direction(MachineRunDirection.FORWARD)
    assert runtime.runner.tick(64) == 6
    assert runtime.machine.cores[0].state.registers[0] == 42

    with pytest.raises(ValueError, match="maximum_dependency_bytes"):
        BinaryMachineProgram.load_pe(
            _minimal_importing_amd64_pe_call(),
            maximum_file_size=4096,
            dependency_provider=provider,
            maximum_dependency_bytes=1,
        )


def test_dependency_provider_follows_forwarder_to_second_approved_module():
    requests = []

    def provider(library):
        requests.append(library)
        if library.casefold() == "demo.dll":
            return _minimal_exporting_amd64_pe_answer()
        if library.casefold() == "kernel32":
            return _minimal_exporting_kernel32_sleep_answer()
        return None

    runtime = BinaryMachineProgram.load_pe(
        _minimal_importing_forwarded_amd64_pe_call(),
        maximum_file_size=4096,
        dependency_provider=provider,
        dependency_load_addresses={"KERNEL32": 0x150000000},
    )

    assert requests == ["demo.dll", "KERNEL32"]
    binding = runtime.import_bindings[0]
    assert binding.resolution_kind == "mapped_export"
    assert binding.resolved_library == "KERNEL32.dll"
    assert binding.resolved_symbol == "Sleep"
    assert binding.forwarder_chain == (
        "demo.dll!Forwarded->KERNEL32.Sleep",
    )
    runtime.set_direction(MachineRunDirection.FORWARD)
    assert runtime.runner.tick(64) == 8
    assert runtime.machine.cores[0].state.registers[0] == 42


def test_tls_template_vector_and_process_attach_callback_are_reversible(tmp_path):
    runtime = BinaryMachineProgram.load_pe(
        _minimal_tls_amd64_pe_return(), maximum_file_size=4096,
    )
    core = runtime.machine.cores[0]
    initial = core.state
    image_base = 0x140000000

    assert runtime.program.image.tls_directory.template == b"ABCD"
    assert initial.pc == image_base + 0x1010
    assert initial.registers[1] == image_base
    assert initial.registers[2] == 1
    assert initial.call_stack
    vector = initial.memory.read_unsigned(initial.gs_base + 0x58, 64)
    tls_base = initial.memory.read_unsigned(vector, 64)
    assert initial.memory.read_unsigned(image_base + 0x3190, 32) == 0
    assert bytes(initial.memory[tls_base + index] for index in range(8)) == b"ABCD\0\0\0\0"

    runtime.set_direction(MachineRunDirection.FORWARD)
    assert runtime.runner.tick(64) == 3
    assert core.latest_edge.status is MachineExecutionStatus.HALTED
    assert core.state.registers[0] == 7
    assert core.state.system_state["windows.loader.tls_callbacks_complete"] == 1
    assert core.state.pc == image_base + 0x1001

    assert core.step_backward().pc == image_base + 0x1000
    assert core.step_backward().pc == image_base + 0x1015
    rewound = core.step_backward()
    assert rewound == initial

    fresh = BinaryMachineProgram.load_pe(
        _minimal_tls_amd64_pe_return(), maximum_file_size=4096,
    )
    tape_path = fresh.save_system_tape(tmp_path / "tls.tape.jsonl")
    resumed = BinaryMachineProgram.load_system_tape(
        tape_path, maximum_file_size=4096,
    )
    resumed.set_direction(MachineRunDirection.FORWARD)
    assert resumed.runner.tick(64) == 3
    assert resumed.machine.cores[0].state.registers[0] == 7


def test_createthread_activates_parked_core_and_thread_return_is_reversible():
    runtime = BinaryMachineProgram.load_pe(
        _minimal_amd64_pe_thread_loop(),
        maximum_file_size=4096,
        core_count=2,
        initial_active_cores=1,
    )
    parent, child = runtime.machine.cores
    assert child.state.system_state["windows.thread.active"] == 0
    reference = runtime.register_external_reference(
        "api-ms-win-core-processthreads-l1-1-0.dll", "CreateThread",
    )
    return_address = 0x140001000
    start_address = 0x140001010
    thread_id_pointer = 0x140003020
    registers = list(parent.state.registers)
    registers[4] -= 8
    memory = parent.state.memory.write_unsigned(registers[4], 64, return_address)
    request = MachineExternalCallRequest(
        55, reference, return_address, return_address,
        (0, 0, start_address, 0x1234),
        registers[4],
        (0, thread_id_pointer),
    )
    parent.commit_shell_effect(replace(
        parent.state,
        pc=reference.target_address,
        registers=tuple(registers),
        memory=memory,
        call_stack=(return_address,),
        external_requests=(request,),
    ))

    port = deterministic_windows_bootstrap_port()
    assert runtime.service_external_requests(port) == 1

    spawned = child.state
    handle = parent.state.registers[0]
    thread_id = 0x30000
    assert spawned.system_state["windows.thread.active"] == 1
    assert spawned.system_state["windows.thread.auxiliary"] == 1
    assert spawned.system_state["windows.thread.id"] == thread_id
    assert spawned.pc == start_address
    assert spawned.registers[1] == 0x1234
    assert spawned.gs_base != parent.state.gs_base
    assert parent.state.memory.read_unsigned(thread_id_pointer, 32) == thread_id
    assert runtime.system_tape.records[-1]["event"] == "thread_spawn"
    assert runtime.system_tape.records[-1]["dependencies"][-1]["kind"] == "thread_spawn_request"
    wait_reference = runtime.register_external_reference(
        "api-ms-win-core-synch-l1-2-0.dll", "WaitForSingleObject",
    )
    zero_wait = MachineExternalCallRequest(
        56, wait_reference, 0, 0, (handle, 0, 0, 0), 0,
    )
    assert port.handle(zero_wait, parent.state).result == 258
    blocking_wait = replace(zero_wait, request_id=57, arguments=(handle, 0xFFFFFFFF, 0, 0))
    assert port.handle(blocking_wait, parent.state) is None

    first = runtime.machine.cycle_forward()
    second = runtime.machine.cycle_forward()
    assert all(result.status is MachineExecutionStatus.RUNNING for result in (*first, *second))
    assert child.state.system_state["windows.thread.active"] == 0
    assert child.state.system_state["windows.thread.exit_code"] == 9
    assert child.state.halted
    assert parent.state.system_state[f"windows.thread.{thread_id}.active"] == 0
    assert port.handle(blocking_wait, parent.state).result == 0
    exit_reference = runtime.register_external_reference(
        "api-ms-win-core-processthreads-l1-1-0.dll", "GetExitCodeThread",
    )
    exit_request = MachineExternalCallRequest(
        58, exit_reference, 0, 0, (handle, thread_id_pointer + 8, 0, 0), 0,
    )
    exit_completion = port.handle(exit_request, parent.state)
    assert exit_completion.result == 1
    assert exit_completion.memory_writes[0].data == (9).to_bytes(4, "little")
    close_reference = runtime.register_external_reference(
        "api-ms-win-core-handle-l1-1-0.dll", "CloseHandle",
    )
    close_completion = port.handle(
        MachineExternalCallRequest(
            59, close_reference, 0, 0, (handle, 0, 0, 0), 0,
        ),
        parent.state,
    )
    assert close_completion.result == 1
    assert close_completion.system_writes[0].key == f"windows.handle.{handle}.kind"

    runtime.machine.cycle_backward()
    runtime.machine.cycle_backward()
    runtime.machine.cycle_backward()
    assert child.state.system_state["windows.thread.active"] == 0
    assert not child.state.halted
    assert parent.state.external_requests == (request,)


def test_createthread_clones_tls_and_runs_attach_then_detach_before_signalling(tmp_path):
    runtime = BinaryMachineProgram.load_pe(
        _minimal_tls_amd64_pe_thread_loop(),
        maximum_file_size=4096,
        core_count=2,
        initial_active_cores=1,
    )
    parent, child = runtime.machine.cores
    # Finish the primary thread's process-attach TLS callback, then remain in
    # its deterministic entry loop while the auxiliary thread runs.
    for _ in range(4):
        runtime.machine.cycle_forward()
    assert parent.state.system_state["windows.loader.startup_calls_complete"] == 1

    reference = runtime.register_external_reference(
        "api-ms-win-core-processthreads-l1-1-0.dll", "CreateThread",
    )
    start_address = 0x140001020
    registers = list(parent.state.registers)
    registers[4] -= 8
    memory = parent.state.memory.write_unsigned(registers[4], 64, 0x140001000)
    request = MachineExternalCallRequest(
        77, reference, parent.state.pc, 0x140001000,
        (0, 0, start_address, 0xCAFE), registers[4], (0, 0),
    )
    parent.commit_shell_effect(replace(
        parent.state,
        pc=reference.target_address,
        registers=tuple(registers),
        memory=memory,
        call_stack=(0x140001000,),
        external_requests=(request,),
    ))
    assert runtime.service_external_requests(
        deterministic_windows_bootstrap_port(),
    ) == 1

    spawned = child.state
    handle = parent.state.registers[0]
    wait_reference = runtime.register_external_reference(
        "api-ms-win-core-synch-l1-2-0.dll", "WaitForSingleObject",
    )
    wait_request = MachineExternalCallRequest(
        78, wait_reference, 0, 0, (handle, 0xFFFFFFFF, 0, 0), 0,
    )
    assert spawned.pc == 0x140001010  # TLS callback, not start routine yet
    assert spawned.registers[2] == 2  # DLL_THREAD_ATTACH
    assert spawned.system_state["windows.loader.startup_reason"] == 2
    parent_vector = parent.state.memory.read_unsigned(parent.state.gs_base + 0x58, 64)
    child_vector = spawned.memory.read_unsigned(spawned.gs_base + 0x58, 64)
    parent_tls = parent.state.memory.read_unsigned(parent_vector, 64)
    child_tls = spawned.memory.read_unsigned(child_vector, 64)
    assert child_vector != parent_vector
    assert child_tls != parent_tls
    assert bytes(spawned.memory[child_tls + index] for index in range(8)) == b"ABCD\0\0\0\0"
    port = deterministic_windows_bootstrap_port()
    assert port.handle(wait_request, parent.state) is None

    runtime.set_direction(MachineRunDirection.FORWARD)
    assert runtime.runner.tick(4) == 4  # two callbacks -> thread start
    assert child.state.pc == start_address
    assert child.state.registers[1] == 0xCAFE
    assert runtime.runner.tick(2) == 2  # thread MOV/RET -> detach callback
    assert child.state.system_state["windows.thread.active"] == 1
    assert child.state.system_state["windows.thread.detach_started"] == 1
    assert child.state.system_state["windows.thread.detach_complete"] == 0
    assert child.state.system_state["windows.thread.pending_exit_code"] == 9
    assert child.state.pc == 0x140001010
    assert child.state.registers[2] == 3  # DLL_THREAD_DETACH
    assert child.state.system_state["windows.loader.detach_call_count"] == 2
    assert child.state.system_state[
        "windows.loader.detach_call.0.startup_index"
    ] == 0
    assert child.state.system_state[
        "windows.loader.detach_call.1.startup_index"
    ] == 1
    assert parent.state.system_state["windows.thread.196608.active"] == 1
    assert port.handle(wait_request, parent.state) is None
    assert runtime.runner.tick(4) == 4  # callbacks retain PE array order -> parked
    assert child.state.system_state["windows.thread.active"] == 0
    assert child.state.system_state["windows.thread.exit_code"] == 9
    assert child.state.system_state["windows.thread.detach_complete"] == 1
    assert parent.state.system_state["windows.thread.196608.active"] == 0
    assert port.handle(wait_request, parent.state).result == 0

    tape_path = runtime.save_system_tape(tmp_path / "thread-detach.tape.jsonl")
    resumed = BinaryMachineProgram.load_system_tape(
        tape_path, maximum_file_size=4096,
    )
    resumed_child = resumed.machine.cores[1].state
    assert resumed_child.halted
    assert resumed_child.exit_code == 9
    assert resumed_child.system_state["windows.thread.detach_complete"] == 1


def test_thread_wait_parks_parent_until_child_exit_then_resumes_exact_call():
    runtime = BinaryMachineProgram.load_pe(
        _minimal_amd64_pe_thread_loop(), maximum_file_size=4096,
        core_count=2, initial_active_cores=1,
    )
    parent, child = runtime.machine.cores
    create_reference = runtime.register_external_reference(
        "api-ms-win-core-processthreads-l1-1-0.dll", "CreateThread",
    )
    return_address = 0x140001000
    start_address = 0x140001010
    registers = list(parent.state.registers)
    registers[4] -= 8
    memory = parent.state.memory.write_unsigned(registers[4], 64, return_address)
    create_request = MachineExternalCallRequest(
        90, create_reference, return_address, return_address,
        (0, 0, start_address, 0xBEEF), registers[4], (0, 0),
    )
    parent.commit_shell_effect(replace(
        parent.state, pc=create_reference.target_address,
        registers=tuple(registers), memory=memory,
        call_stack=(return_address,), external_requests=(create_request,),
    ))
    port = deterministic_windows_bootstrap_port()
    assert runtime.service_external_requests(port) == 1
    handle = parent.state.registers[0]

    wait_reference = runtime.register_external_reference(
        "api-ms-win-core-synch-l1-2-0.dll", "WaitForSingleObject",
    )
    registers = list(parent.state.registers)
    registers[4] -= 8
    memory = parent.state.memory.write_unsigned(registers[4], 64, return_address)
    wait_request = MachineExternalCallRequest(
        91, wait_reference, return_address, return_address,
        (handle, 0xFFFFFFFF, 0, 0), registers[4], (0, 0),
    )
    parent.commit_shell_effect(replace(
        parent.state, pc=wait_reference.target_address,
        registers=tuple(registers), memory=memory,
        call_stack=(return_address,), external_requests=(wait_request,),
    ))

    assert runtime.service_external_requests(port) == 0
    assert parent.state.system_state["windows.thread.waiting_request"] == 91
    assert runtime.system_tape.records[-1]["event"] == "thread_wait"

    runtime.machine.cycle_forward()  # child MOV; parent is parked
    runtime.machine.cycle_forward()  # child RET/exit; parent is still parked
    assert child.state.system_state["windows.thread.active"] == 0
    assert parent.state.external_requests == (wait_request,)
    assert parent.state.system_state["windows.thread.196608.active"] == 0

    assert runtime.service_external_requests(port) == 1
    assert parent.state.registers[0] == 0  # WAIT_OBJECT_0
    assert parent.state.pc == return_address
    assert parent.state.external_requests == ()
    assert parent.state.system_state["windows.thread.waiting_request"] == 0


def test_runtime_resumes_and_appends_directly_on_segmented_tape(tmp_path):
    runtime = BinaryMachineProgram.load_pe(
        _minimal_amd64_pe_return(), maximum_file_size=4096,
    )
    source = runtime.save_system_tape(tmp_path / "source.tape.jsonl")
    store = SegmentedMachineTapeStore.import_jsonl(
        source, tmp_path / "segmented", records_per_segment=1,
    )

    resumed = BinaryMachineProgram.load_segmented_system_tape(
        store, maximum_file_size=4096,
    )
    resumed.set_direction(MachineRunDirection.FORWARD)
    resumed.runner.tick(1)
    resumed.save_system_tape(store.root)

    reopened = SegmentedMachineTapeStore(store.root)
    assert reopened.record_count == 2
    assert reopened.resume_state().steps == 1
    assert reopened.segments[-1].parent_digest == reopened.segments[-2].digest

    cold = BinaryMachineProgram.load_segmented_system_tape(
        reopened, maximum_file_size=4096,
    )
    assert cold.machine.cores[0].position == 1
    assert cold.machine.cores[0].hot_history_range == (1, 2)
    tip = cold.machine.cores[0].state
    assert cold.machine.cores[0].step_backward().steps == 0
    assert cold.machine.cores[0].position == 0
    assert cold.machine.cores[0].step_forward().state == tip
    cold.close()


def test_jsonl_resume_retains_absolute_position_and_reverses_into_tape(tmp_path):
    runtime = BinaryMachineProgram.load_pe(
        _minimal_amd64_pe_return(), maximum_file_size=4096,
    )
    runtime.set_direction(MachineRunDirection.FORWARD)
    runtime.runner.tick(1)
    path = runtime.save_system_tape(tmp_path / "resume.tape.jsonl")

    cold = BinaryMachineProgram.load_system_tape(path, maximum_file_size=4096)

    assert cold.machine.cores[0].position == 1
    assert cold.machine.cores[0].hot_history_range == (1, 2)
    assert cold.machine.cores[0].step_backward().steps == 0
    assert cold.machine.cores[0].position == 0
    cold.close()
    runtime.close()


def test_runtime_bootstraps_live_segmented_tape_without_retaining_jsonl(tmp_path):
    runtime = BinaryMachineProgram.load_pe(
        _minimal_amd64_pe_return(), maximum_file_size=4096,
    )
    root = tmp_path / "live.segmented-tape"

    store = runtime.begin_segmented_system_tape(root, records_per_segment=1)

    assert runtime.system_tape is store
    assert store.record_count == 1
    assert store.resume_state().steps == 0
    assert not tuple(tmp_path.glob("*.bootstrap.jsonl"))
    with pytest.raises(FileExistsError, match="refusing to replace"):
        runtime.begin_segmented_system_tape(root)

    runtime.set_direction(MachineRunDirection.FORWARD)
    runtime.runner.tick(1)
    runtime.save_system_tape(root)
    reopened = SegmentedMachineTapeStore(root)
    assert reopened.record_count == 2
    assert reopened.resume_state().steps == 1
    runtime.close()


def test_retained_state_builds_display_snapshot_without_subject_recompile():
    state = replace(
        _machine(1).cores[0].state,
        registers=(0x1122334455667788, *(0 for _ in range(15))),
        device_state={"console.output": b"card output\r\n"},
        device_generations={"console.output": 7},
        steps=42,
    )
    encoded = build_machine_state_snapshot(
        (state,), positions=(9,), annotation_colors=(0xFF00FFFF,),
        maximum_output_bytes=1024,
    )
    snapshot = MachineSnapshotView(memoryview(encoded))

    assert snapshot.register_words(0, 0) == (0x55667788, 0x11223344)
    assert snapshot.core_status(0)["steps"] == 42
    assert snapshot.core_status(0)["annotation_rgba8"] == 0xFF00FFFF
    descriptor = snapshot.output_descriptor(0)
    assert descriptor.kind is SubjectOutputKind.TERMINAL
    assert descriptor.format is SubjectOutputFormat.UTF8
    assert bytes(snapshot.output_bytes(0)) == b"card output\r\n"


def test_runtime_blocked_rip_becomes_a_validated_resumable_dispatch_plan(tmp_path):
    image = bytearray(_minimal_amd64_pe_return())
    # Keep a second RET file-backed and executable, but outside the sole
    # runtime-function record and therefore outside initial reachability.
    struct.pack_into("<I", image, 0x188 + 8, 0x20)
    image[0x410] = 0xC3
    runtime = BinaryMachineProgram.load_pe(
        bytes(image), maximum_file_size=4096, transitions_per_second=1.0,
    )
    target = 0x140001010
    core = runtime.machine.cores[0]
    core.executor.translated_block(0x140001000)
    assert core.executor.translation_cache_stats["blocks"] == 1
    core._states[0] = replace(core.state, pc=target)

    assert target not in core.executor.instructions
    assert runtime.service_dispatch_frontiers(core_index=0) == 1
    assert core.executor.instructions[target].encoded == b"\xc3"
    assert dict(core.executor.translation_cache_stats) == {
        "generation": 1, "blocks": 0, "hits": 0, "misses": 1,
    }
    assert runtime.dispatch_plans[-1].targets == (target,)
    annotation = runtime.system_tape.annotations[-1]
    assert annotation.feature == "runtime_dispatch"
    assert annotation.address == target
    assert runtime.system_tape.records[-1]["event"] == "runtime_dispatch"

    path = runtime.save_system_tape(tmp_path / "dispatch.tape.jsonl")
    resumed = BinaryMachineProgram.load_system_tape(path, maximum_file_size=4096)
    assert target in resumed.machine.cores[0].executor.instructions
    assert resumed.machine.cores[0].state.pc == target
