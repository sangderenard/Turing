from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.compiler.machine_execution import (
    MachineExecutionOrchestrator,
    MachineExecutionStatus,
    MachineVirtualMulticore,
    ReversibleMachineExecutor,
)
from src.compiler.machine_execution_shader import (
    MACHINE_DISPLAY_REGISTERS,
    build_machine_register_shader,
)
from src.compiler.machine_reference_vocabulary import MachineSemanticToken


def _program():
    add = SimpleNamespace(
        address=0x1000,
        encoded=b"\x90",
        semantic=MachineSemanticToken.INTEGER_ADD,
        operands=(),
    )
    ret = SimpleNamespace(
        address=0x1001,
        encoded=b"\xc3",
        semantic=MachineSemanticToken.RETURN,
        operands=(),
    )
    return SimpleNamespace(
        image=SimpleNamespace(image_base=0x1000, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(add, ret)),
        ),),
    )


def _executor():
    def add_one(state, _instruction):
        registers = list(state.registers)
        registers[0] += 1
        return replace(state, registers=tuple(registers))

    return MachineExecutionOrchestrator(
        _program(),
        effect_handlers={int(MachineSemanticToken.INTEGER_ADD): add_one},
    )


def test_actual_executor_rewinds_complete_machine_state():
    runtime = ReversibleMachineExecutor.create(_executor())
    initial = runtime.state

    running = runtime.step_forward()
    halted = runtime.step_forward()

    assert running.status is MachineExecutionStatus.RUNNING
    assert running.state.register_contents()["rax"] == 1
    assert halted.status is MachineExecutionStatus.HALTED
    assert runtime.step_backward() == running.state
    assert runtime.step_backward() == initial
    assert len(runtime.edges) == 2


def test_rewind_then_execute_branches_and_forks_remain_independent():
    runtime = ReversibleMachineExecutor.create(_executor())
    runtime.step_forward()
    branch = runtime.fork()
    branch.step_forward()

    assert branch.state.pc == 0x1002
    assert runtime.state.pc == 0x1001
    runtime.step_backward()
    runtime.step_forward()
    assert len(runtime.edges) == 1


def test_virtual_multicore_moves_heads_across_reversible_barrier():
    cores = MachineVirtualMulticore.create(_executor(), core_count=3)

    results = cores.cycle_forward()

    assert [result.state.register_contents()["rax"] for result in results] == [1, 1, 1]
    assert len(cores.register_contents()) == 3
    assert cores.cycle_backward()[0].register_contents()["rax"] == 0
    with pytest.raises(IndexError, match="initial state"):
        cores.cycle_backward()


def test_machine_register_shader_accepts_lossless_64_bit_word_pairs():
    artifact = build_machine_register_shader(workgroup_size=128)

    assert artifact.register_names == MACHINE_DISPLAY_REGISTERS
    assert len(artifact.register_names) == 20
    assert "array<vec2<u32>>" in artifact.source
    assert "@compute @workgroup_size(128)" in artifact.source

