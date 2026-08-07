from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.compiler.machine_execution import (
    MachineExecutionOrchestrator,
    MachineExecutionState,
    MachineExecutionStatus,
    MachineVirtualMulticore,
    ReversibleMachineExecutor,
)
from src.compiler.amd64_machine_semantics import PagedByteMemory, default_effect_handlers
from src.compiler.machine_execution_shader import (
    MACHINE_DISPLAY_REGISTERS,
    build_machine_register_shader,
)
from src.compiler.machine_reference_vocabulary import (
    MachineSemanticToken,
    X86InstructionToken,
)
from src.compiler.virtual_registry import VirtualRegistryEffect, VirtualRegistryState
from src.compiler.virtual_memory import (
    PAGE_EXECUTE_READWRITE, VirtualMemoryEffect, VirtualMemoryState,
)


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


def test_translated_block_cache_retains_instruction_edges_and_exact_rewind():
    executor = _executor()
    first = executor.translated_block(0x1000)
    second = executor.translated_block(0x1000)

    assert first is second
    assert first.instruction_addresses == (0x1000, 0x1001)
    assert len(first.code_digest) == 64
    assert dict(executor.translation_cache_stats) == {
        "generation": 0, "blocks": 1, "hits": 1, "misses": 1,
    }

    runtime = ReversibleMachineExecutor.create(executor)
    observed_positions = []
    results = runtime.step_block_forward(
        64, transition_observer=lambda: observed_positions.append(runtime.position),
    )

    assert [result.status for result in results] == [
        MachineExecutionStatus.RUNNING, MachineExecutionStatus.HALTED,
    ]
    assert observed_positions == [1, 2]
    assert [edge.instruction.address for edge in runtime.edges] == [0x1000, 0x1001]
    assert runtime.step_backward().register_contents()["rax"] == 1
    assert runtime.step_backward().register_contents()["rax"] == 0


def test_executable_page_write_invalidates_block_and_redecodes_guest_bytes():
    first = SimpleNamespace(
        address=0x1000,
        encoded=b"\x90",
        instruction=X86InstructionToken.NOP,
        semantic=MachineSemanticToken.INTEGER_ADD,
        operands=(),
    )
    ret = SimpleNamespace(
        address=0x1001,
        encoded=b"\xc3",
        instruction=X86InstructionToken.RET_NEAR,
        semantic=MachineSemanticToken.RETURN,
        operands=(),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(
            image_base=0x1000,
            entrypoint_rva=0,
            encoded=b"\x90\xc3",
            sections=(SimpleNamespace(
                executable=True,
                virtual_address=0,
                virtual_size=0x1000,
                raw_size=2,
            ),),
        ),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(first, ret)),
        ),),
    )

    def rewrite_successor(state, _instruction):
        return replace(state, memory=state.memory.map_bytes(0x1001, b"\xcc"))

    executor = MachineExecutionOrchestrator(
        program,
        effect_handlers={
            int(MachineSemanticToken.INTEGER_ADD): rewrite_successor,
        },
    )
    initial = replace(
        MachineExecutionState(pc=0x1000),
        memory=PagedByteMemory.empty().map_bytes(0x1000, b"\x90\xc3"),
    )
    runtime = ReversibleMachineExecutor.create(executor, initial)

    # The cached block originally contains NOP + RET. The first instruction
    # rewrites RET to INT3, so execution must stop before using the stale op.
    results = runtime.step_block_forward(64)

    assert len(results) == 1
    assert results[0].status is MachineExecutionStatus.RUNNING
    assert runtime.state.memory[0x1001] == 0xCC
    assert runtime.state.system_state["machine.code_page.0x1.version"] == 1
    assert executor.translation_cache_stats["blocks"] == 0

    trapped = runtime.step_forward()

    assert trapped.status is MachineExecutionStatus.TRAPPED
    assert trapped.instruction.token is X86InstructionToken.INT3
    assert trapped.instruction.encoded == b"\xcc"
    assert runtime.step_backward().memory[0x1001] == 0xCC
    restored = runtime.step_backward()
    assert restored.memory[0x1001] == 0xC3
    assert "machine.code_page.0x1.version" not in restored.system_state


def test_dynamically_allocated_executable_page_decodes_and_versions_code():
    base = 0x10000000000
    program = SimpleNamespace(
        image=SimpleNamespace(
            image_base=base, entrypoint_rva=0, encoded=None, sections=(),
        ),
        functions=(),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    empty = MachineExecutionState(
        pc=base, memory=PagedByteMemory.empty(),
        virtual_memory=VirtualMemoryState.create(),
    )
    allocated = replace(
        empty, memory=empty.memory.map_zeroes(base, 4096).map_bytes(base, b"\x90"),
        virtual_memory=empty.virtual_memory.apply(VirtualMemoryEffect(
            "allocate", base, 4096, PAGE_EXECUTE_READWRITE,
        )),
    )
    reconciled = executor.reconcile_external_state(empty, allocated)
    key = f"machine.code_page.{base // 4096:#x}.version"
    assert reconciled.system_state[key] == 1
    result = executor.step(reconciled)
    assert result.status is MachineExecutionStatus.RUNNING
    assert result.state.pc == base + 1


def test_external_executable_write_versions_page_and_clears_translation_cache():
    executor = _executor()
    source = replace(
        MachineExecutionState(pc=0x1000),
        memory=PagedByteMemory.empty().map_bytes(0x1000, b"\x90\xc3"),
    )
    executor.translated_block(0x1000, source)
    target = replace(source, memory=source.memory.map_bytes(0x1001, b"\xcc"))

    reconciled = executor.reconcile_external_state(source, target)

    assert reconciled.memory[0x1001] == 0xCC
    assert reconciled.system_state["machine.code_page.0x1.version"] == 1
    assert executor.translation_cache_stats["blocks"] == 0
    assert executor.translation_cache_stats["generation"] == 1


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


def test_guest_threads_share_memory_in_deterministic_core_order_and_reverse_barrier():
    instruction = SimpleNamespace(
        address=0x2000, encoded=b"\x90",
        semantic=MachineSemanticToken.INTEGER_ADD, operands=(),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0x2000, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(instruction,)),
        ),),
    )

    def accumulate(state, _instruction):
        value = state.memory[0x3000] + state.registers[0]
        return replace(state, memory=state.memory.map_bytes(0x3000, bytes((value,))))

    executor = MachineExecutionOrchestrator(
        program,
        effect_handlers={int(MachineSemanticToken.INTEGER_ADD): accumulate},
    )
    shared = PagedByteMemory.empty().map_bytes(0x3000, b"\x00")
    registry = VirtualRegistryState.create()
    left = replace(
        MachineExecutionState(pc=0x2000), memory=shared,
        virtual_registry=registry,
    )
    right_registers = list(left.registers)
    right_registers[0] = 2
    left_registers = list(left.registers)
    left_registers[0] = 1
    states = (
        replace(left, registers=tuple(left_registers)),
        replace(left, registers=tuple(right_registers)),
    )
    machine = MachineVirtualMulticore.create(
        executor, core_count=2, initial_states=states,
    )

    results = machine.cycle_forward()

    # Core 0 writes 1; core 1 observes that write and adds 2. Both threads see
    # the final barrier state, and the overlap is retained as race provenance.
    assert [result.state.memory[0x3000] for result in results] == [3, 3]
    commit = machine.last_shared_memory_commit
    assert commit is not None
    assert commit.core_order == (0, 1)
    assert [item.core_index for item in commit.writes] == [0, 1]
    assert commit.conflicts[0].address == 0x3000
    assert dict(commit.to_mapping())["cycle_index"] == 1

    machine.cores[0].commit_shell_effect(replace(
        machine.cores[0].state,
        memory=machine.cores[0].state.memory.map_bytes(0x3000, b"\x09"),
        virtual_registry=registry.apply(VirtualRegistryEffect(
            "create_key", "hkey_current_user\\Software\\Shared",
        )),
    ))
    assert machine.synchronize_shared_memory(0) == (1,)
    assert [core.state.memory[0x3000] for core in machine.cores] == [9, 9]
    assert all(
        "hkey_current_user\\software\\shared" in core.state.virtual_registry.keys
        for core in machine.cores
    )

    synchronized_undo = machine.cycle_backward()
    assert [state.memory[0x3000] for state in synchronized_undo] == [3, 3]
    assert all(
        "hkey_current_user\\software\\shared" not in state.virtual_registry.keys
        for state in synchronized_undo
    )
    assert machine.last_shared_memory_commit is commit

    restored = machine.cycle_backward()
    assert [state.memory[0x3000] for state in restored] == [0, 0]
    assert machine.last_shared_memory_commit is None


def test_machine_register_shader_accepts_lossless_64_bit_word_pairs():
    artifact = build_machine_register_shader(workgroup_size=128)

    assert artifact.register_names == MACHINE_DISPLAY_REGISTERS
    assert len(artifact.register_names) == 54
    assert "state_snapshot: array<u32>" in artifact.source
    assert "register_base_words = state_snapshot[13u] / 4u" in artifact.source
    assert "annotation != 0u && register_index == 16u" in artifact.source
    assert "@compute @workgroup_size(128)" in artifact.source
