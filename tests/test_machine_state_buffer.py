from dataclasses import replace
from types import SimpleNamespace
import struct
import time

from src.compiler.machine_chip_layout import build_register_bank_layout
from src.compiler.binary_machine_program import BinaryMachineProgram
from src.compiler.machine_execution import MachineExecutionOrchestrator, MachineVirtualMulticore
from src.compiler.machine_reference_vocabulary import MachineSemanticToken
from src.compiler.machine_state_buffer import (
    ExternalMachineClock,
    FreeRunningMachineRunner,
    MachineRunDirection,
    MachineSnapshotLayout,
    MachineSnapshotTripleBuffer,
    SubjectOutputBuffer,
    SubjectOutputFormat,
    SubjectOutputKind,
)


def _machine(core_count=2):
    instruction = SimpleNamespace(
        address=0x401000, encoded=b"\x90",
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
