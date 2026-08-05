import shutil
import struct
import time
from types import SimpleNamespace

import pytest

from src.compiler.amd64_machine_semantics import PagedByteMemory, default_effect_handlers
from src.compiler.amd64_machine_semantics import indirect_target
from src.compiler.binary_machine_program import BinaryMachineProgram
from src.compiler.machine_execution import (
    MachineExecutionOrchestrator,
    MachineExecutionState,
    MachineExternalReference,
    MachineVirtualMulticore,
    ReversibleMachineExecutor,
)
from src.compiler.machine_block_recompiler import (
    JOURNAL_EFFECT_OFFSET, MachineBlockLoweringError,
)
from src.compiler.machine_reference_vocabulary import (
    EffectiveAddressOperand,
    ImmediateOperand,
    MachineSemanticToken,
    RegisterOperand,
    RelativeAddressOperand,
    VectorRegisterOperand,
    X86ReferenceDecoder,
    X86Register,
    X86VectorRegister,
)
from src.compiler.machine_state_buffer import (
    MachineRunDirection, MachineSnapshotView, build_machine_state_snapshot,
)
from src.compiler.machine_snapshot_host import (
    LiveMachineSnapshotController,
    MachineSnapshotMailbox,
    MachineTerminalInputQueue,
)
from src.compiler.machine_wasm_runtime import (
    MachineWasmBlockDispatcher,
    NodeMachineWasmHost,
)


def _mov_return_program():
    encoded = b"\xb8\x09\x00\x00\x00\xc3"
    decoder = X86ReferenceDecoder()
    instructions = []
    cursor = 0
    while cursor < len(encoded):
        instruction, cursor = decoder.decode_one(
            memoryview(encoded), cursor, base_address=0x1000,
        )
        instructions.append(instruction)
    return SimpleNamespace(
        image=SimpleNamespace(
            image_base=0x1000, entrypoint_rva=0,
            section_for_rva=lambda _rva: None,
            runtime_function_for_rva=lambda _rva: None,
        ),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=tuple(instructions)),
        ),),
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_shell_tick_automatically_dispatches_wasm_then_tapes_reference_fallback():
    runtime = BinaryMachineProgram.from_program(
        _mov_return_program(), effect_handlers=default_effect_handlers(),
        machine_block_backend="node-wasm",
    )
    try:
        runtime.set_direction(MachineRunDirection.FORWARD)
        assert runtime.runner.tick(2) == 2
        core = runtime.machine.cores[0]
        assert core.state.registers[0] == 9
        assert core.position == 2
        assert runtime.runner._last_results[0].status.name == "HALTED"
        assert [item["position"] for item in runtime.system_tape.records[-2:]] == [1, 2]
        stats = runtime.recompilation_statistics
        assert stats["executions"] == 1
        assert stats["committed_instructions"] == 1
        assert stats["fallbacks"] == 1
        assert stats["host_requests"] == 1
        assert stats["host_module_loads"] == 1

        runtime.set_direction(MachineRunDirection.BACKWARD)
        assert runtime.runner.tick(2) == 2
        assert core.position == 0
        assert core.state.registers[0] == 0
    finally:
        runtime.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_live_html_snapshot_controller_drives_the_same_compiled_runner():
    runtime = BinaryMachineProgram.from_program(
        _mov_return_program(), effect_handlers=default_effect_handlers(),
        machine_block_backend="node-wasm",
    )
    mailbox = MachineSnapshotMailbox()
    controller = LiveMachineSnapshotController(
        runtime, object(), mailbox, MachineTerminalInputQueue(),
        transitions_per_cycle=2, idle_seconds=0.001,
    )
    try:
        controller.start()
        deadline = time.monotonic() + 5
        while (
            runtime.recompilation_statistics.get("executions", 0) < 1
            and controller.failure is None and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        controller.stop()
        assert controller.failure is None
        assert runtime.recompilation_statistics["executions"] == 1
        assert mailbox.generation >= 2
        assert runtime.machine.cores[0].state.registers[0] == 9
    finally:
        if controller.running:
            controller.stop()
        runtime.close()


def _call_return_program():
    call = SimpleNamespace(
        address=0xA000, encoded=b"\xe8\xfb\x00\x00\x00",
        semantic=MachineSemanticToken.DIRECT_RELATIVE_CALL,
        token=SimpleNamespace(name="CALL_REL32"),
        operands=(RelativeAddressOperand(0xFB, 32, 0xA100),),
    )
    ret = SimpleNamespace(
        address=0xA100, encoded=b"\xc3",
        semantic=MachineSemanticToken.RETURN,
        token=SimpleNamespace(name="RET_NEAR"), operands=(),
    )
    return SimpleNamespace(
        image=SimpleNamespace(image_base=0xA000, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(call, ret)),
        ),),
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_dispatcher_runs_specialized_call_and_return_as_reversible_edges():
    executor = MachineExecutionOrchestrator(
        _call_return_program(), effect_handlers=default_effect_handlers(),
    )
    registers = [0] * 16
    registers[4] = 0xB008
    initial = MachineExecutionState(
        pc=0xA000, registers=tuple(registers),
        memory=PagedByteMemory.empty().map_zeroes(0xB000, 16),
    )
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    core = machine.cores[0]
    dispatcher = MachineWasmBlockDispatcher(NodeMachineWasmHost())
    observed = []
    try:
        call_results = dispatcher.execute(
            core, 1, transition_observer=lambda: observed.append(core.position),
        )
        assert call_results is not None and len(call_results) == 1
        called = core.state
        assert called.pc == 0xA100
        assert called.call_stack == (0xA005,)
        assert called.memory.read_unsigned(0xB000, 64) == 0xA005

        return_results = dispatcher.execute(
            core, 1, transition_observer=lambda: observed.append(core.position),
        )
        assert return_results is not None and len(return_results) == 1
        assert core.state.pc == 0xA005
        assert core.state.call_stack == ()
        assert observed == [1, 2]
        assert dispatcher.statistics["executions"] == 2

        assert core.step_backward() == called
        assert core.step_backward() == initial
    finally:
        dispatcher.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_persistent_host_reuses_instantiated_module_across_loop_iterations():
    jump = SimpleNamespace(
        address=0xD000, encoded=b"\xeb\xfe",
        semantic=MachineSemanticToken.DIRECT_RELATIVE_JUMP,
        token=SimpleNamespace(name="JMP_REL8"),
        operands=(RelativeAddressOperand(-2, 8, 0xD000),),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xD000, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(jump,)),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    machine = MachineVirtualMulticore.create(
        executor, core_count=1,
        initial_states=(MachineExecutionState(pc=0xD000),),
    )
    dispatcher = MachineWasmBlockDispatcher(NodeMachineWasmHost())
    try:
        assert dispatcher.execute(machine.cores[0], 1) is not None
        assert dispatcher.execute(machine.cores[0], 1) is not None
        assert machine.cores[0].position == 2
        assert machine.cores[0].state.steps == 2
        assert dispatcher.statistics["host_requests"] == 2
        assert dispatcher.statistics["host_module_loads"] == 1
        assert dispatcher.statistics["host_resident_modules"] == 1
        assert dispatcher.statistics["cached_artifacts"] == 1
    finally:
        dispatcher.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_block_entry_dynamic_address_is_specialized_witnessed_and_guarded():
    instruction = SimpleNamespace(
        address=0xB800, encoded=b"\x90",
        semantic=MachineSemanticToken.REGISTER_OR_MEMORY_READ,
        token=SimpleNamespace(name="MOV_R64_RM64"), legacy_prefixes=(0x65,),
        operands=(
            RegisterOperand(X86Register.RAX, 64),
            EffectiveAddressOperand(
                X86Register.RCX, X86Register.RDX, 2, 8, 64, False,
            ),
        ),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xB800, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(instruction,)),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    registers = [0] * 16
    registers[1], registers[2] = 0x200, 4
    address = 0xC000 + 0x200 + 4 * 2 + 8
    memory = PagedByteMemory.empty().map_zeroes(address, 8)
    memory = memory.write_unsigned(address, 64, 0x1122334455667788)
    initial = MachineExecutionState(
        pc=0xB800, registers=tuple(registers), gs_base=0xC000, memory=memory,
    )
    expected = executor.step(initial).state
    artifact = executor.recompile_block_wasm(0xB800, initial, strict=True)
    assert artifact.guest_memory_base == address
    assert artifact.specialization_guard["registers"] == ((1, 0x200), (2, 4))
    assert artifact.specialization_guard["gs_base"] == 0xC000
    host = NodeMachineWasmHost()
    try:
        journal = host.execute(artifact, initial)
    finally:
        host.close()
    assert artifact.states_from_journal(journal, initial) == (expected,)

    changed = MachineExecutionState(
        pc=initial.pc,
        registers=(initial.registers[0], 0x208, *initial.registers[2:]),
        gs_base=initial.gs_base, memory=initial.memory,
    )
    with pytest.raises(ValueError, match="specialization register mismatch"):
        artifact.pack_guest_memory(changed)


def test_recompiled_journal_accepts_exact_dynamically_decoded_guest_code():
    """Runtime-discovered code need not already live in the static decode map."""

    address = 0xB880
    instruction = SimpleNamespace(
        address=address, encoded=b"\x90",
        semantic=MachineSemanticToken.REGISTER_OR_MEMORY_READ,
    )
    memory = PagedByteMemory.empty().map_zeroes(0xB000, 0x1000)
    memory = memory.map_bytes(address, instruction.encoded)
    initial = MachineExecutionState(pc=address, memory=memory)
    expected = MachineExecutionState(pc=address + 1, memory=memory, steps=1)
    executor = SimpleNamespace(
        instructions={},
        _decode_instruction_from_state=lambda state, target: (
            instruction if target == address else None
        ),
        _instruction_bytes_match=lambda state, decoded: (
            bytes(state.memory[decoded.address + index] for index in range(len(decoded.encoded)))
            == decoded.encoded
        ),
        reconcile_external_state=lambda source, target: target,
    )
    core = ReversibleMachineExecutor.create(executor, initial)
    artifact = SimpleNamespace(
        witnesses=(SimpleNamespace(address=address, encoded=b"\x90"),),
        states_from_journal=lambda encoded, state: (expected,),
    )

    results = core.commit_recompiled_journal(artifact, b"journal")

    assert results[0].state == expected
    assert core.step_backward() == initial


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_block_entry_dynamic_store_matches_memory_effect_and_reverses():
    instruction = SimpleNamespace(
        address=0xB900, encoded=b"\x90",
        semantic=MachineSemanticToken.REGISTER_OR_MEMORY_WRITE,
        token=SimpleNamespace(name="MOV_RM64_R64"), legacy_prefixes=(),
        operands=(
            EffectiveAddressOperand(X86Register.RBX, None, 1, 16, 64, False),
            RegisterOperand(X86Register.RAX, 64),
        ),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xB900, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(instruction,)),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    registers = [0] * 16
    registers[0], registers[3] = 0xCAFEBABE11223344, 0xC800
    address = 0xC810
    memory = PagedByteMemory.empty().map_zeroes(address, 8)
    memory = memory.write_unsigned(address, 64, 0xDEADBEEF)
    initial = MachineExecutionState(
        pc=0xB900, registers=tuple(registers), memory=memory,
    )
    expected = executor.step(initial).state
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    dispatcher = MachineWasmBlockDispatcher(NodeMachineWasmHost())
    try:
        results = dispatcher.execute(machine.cores[0], 1)
        assert results is not None and len(results) == 1
        assert machine.cores[0].state == expected
        assert machine.cores[0].state.memory.read_unsigned(address, 64) == 0xCAFEBABE11223344
        assert machine.cores[0].step_backward() == initial
    finally:
        dispatcher.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_lea_uses_register_values_produced_inside_the_same_compiled_block():
    instructions = (
        SimpleNamespace(
            address=0xB980, encoded=b"\x90",
            semantic=MachineSemanticToken.REGISTER_WRITE_IMMEDIATE,
            token=SimpleNamespace(name="MOV_R64_IMM64"), legacy_prefixes=(),
            operands=(
                RegisterOperand(X86Register.RCX, 64),
                ImmediateOperand(0x100, 64, False),
            ),
        ),
        SimpleNamespace(
            address=0xB981, encoded=b"\x90",
            semantic=MachineSemanticToken.EFFECTIVE_ADDRESS,
            token=SimpleNamespace(name="LEA_R64_M"), legacy_prefixes=(),
            operands=(
                RegisterOperand(X86Register.RAX, 64),
                EffectiveAddressOperand(
                    X86Register.RCX, X86Register.RDX, 4, 8, 64, False,
                ),
            ),
        ),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xB980, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=instructions),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    registers = [0] * 16
    registers[1], registers[2] = 0xDEAD, 3
    initial = MachineExecutionState(pc=0xB980, registers=tuple(registers))
    first = executor.step(initial).state
    expected = executor.step(first).state
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    dispatcher = MachineWasmBlockDispatcher(NodeMachineWasmHost())
    try:
        results = dispatcher.execute(machine.cores[0], 2)
        assert results is not None and len(results) == 2
        assert results[0].state == first
        assert machine.cores[0].state == expected
        assert machine.cores[0].state.registers[0] == 0x114
        assert machine.cores[0].step_backward() == first
        assert machine.cores[0].step_backward() == initial
    finally:
        dispatcher.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_static_compare_and_dynamic_test_memory_operands_match_flags_and_witness_reads():
    cases = (
        SimpleNamespace(
            address=0xBF00, encoded=b"\x90",
            semantic=MachineSemanticToken.INTEGER_COMPARE,
            token=SimpleNamespace(name="CMP_RM64_IMM8"), legacy_prefixes=(),
            operands=(
                EffectiveAddressOperand(None, None, 1, 0xE000 - 0xBF01, 64, True),
                ImmediateOperand(0xFF, 8, True),
            ),
            memory_address=0xE000, memory_value=0, registers=(0,) * 16,
        ),
        SimpleNamespace(
            address=0xBF10, encoded=b"\x90",
            semantic=MachineSemanticToken.INTEGER_TEST,
            token=SimpleNamespace(name="TEST_RM64_R64"), legacy_prefixes=(),
            operands=(
                EffectiveAddressOperand(X86Register.RBX, None, 1, 8, 64, False),
                RegisterOperand(X86Register.RAX, 64),
            ),
            memory_address=0xE108, memory_value=0xF0F0,
            registers=(0x0FF0, 0, 0, 0xE100, *(0 for _ in range(12))),
        ),
    )
    host = NodeMachineWasmHost()
    try:
        for instruction in cases:
            program = SimpleNamespace(
                image=SimpleNamespace(image_base=instruction.address, entrypoint_rva=0),
                functions=(SimpleNamespace(
                    report=SimpleNamespace(instructions=(instruction,)),
                ),),
            )
            executor = MachineExecutionOrchestrator(
                program, effect_handlers=default_effect_handlers(),
            )
            memory = PagedByteMemory.empty().map_zeroes(instruction.memory_address, 8)
            memory = memory.write_unsigned(
                instruction.memory_address, 64, instruction.memory_value,
            )
            initial = MachineExecutionState(
                pc=instruction.address,
                registers=tuple(instruction.registers), memory=memory,
            )
            expected = executor.step(initial).state
            artifact = executor.recompile_block_wasm(
                instruction.address, initial, strict=True,
            )
            journal = host.execute(artifact, initial)
            (compiled,) = artifact.states_from_journal(journal, initial)
            assert compiled == expected
            assert compiled.registers == initial.registers

            tampered = bytearray(journal)
            struct.pack_into(
                "<Q", tampered, JOURNAL_EFFECT_OFFSET + 24,
                instruction.memory_value ^ 1,
            )
            with pytest.raises(ValueError, match="memory-read witness mismatch"):
                artifact.states_from_journal(bytes(tampered), initial)
    finally:
        host.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_dynamic_memory_read_modify_write_has_one_exact_before_after_effect():
    instruction = SimpleNamespace(
        address=0xBF40, encoded=b"\x90",
        semantic=MachineSemanticToken.BITWISE_XOR,
        token=SimpleNamespace(name="XOR_RM64_R64"), legacy_prefixes=(),
        operands=(
            EffectiveAddressOperand(X86Register.RBX, None, 1, 8, 64, False),
            RegisterOperand(X86Register.RAX, 64),
        ),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xBF40, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(instruction,)),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    registers = [0] * 16
    registers[0], registers[3] = 0x00FF00FF00FF00FF, 0xE200
    address, before = 0xE208, 0xFFFF0000AAAA5555
    memory = PagedByteMemory.empty().map_zeroes(address, 8)
    memory = memory.write_unsigned(address, 64, before)
    initial = MachineExecutionState(
        pc=0xBF40, registers=tuple(registers), memory=memory, flags=0x202,
    )
    expected = executor.step(initial).state
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    dispatcher = MachineWasmBlockDispatcher(NodeMachineWasmHost())
    try:
        results = dispatcher.execute(machine.cores[0], 1)
        assert results is not None and results[0].state == expected
        assert machine.cores[0].state.memory.read_unsigned(address, 64) == (
            before ^ registers[0]
        )
        assert machine.cores[0].step_backward() == initial
    finally:
        dispatcher.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_dynamic_memory_movsxd_sign_extends_with_a_read_witness():
    instruction = SimpleNamespace(
        address=0xBF60, encoded=b"\x90",
        semantic=MachineSemanticToken.SIGN_EXTEND,
        token=SimpleNamespace(name="MOVSXD_R64_RM32"), legacy_prefixes=(),
        operands=(
            RegisterOperand(X86Register.RAX, 64),
            EffectiveAddressOperand(X86Register.RBX, None, 1, 4, 64, False),
        ),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xBF60, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(instruction,)),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    registers = [0] * 16
    registers[3] = 0xE300
    address = 0xE304
    memory = PagedByteMemory.empty().map_zeroes(address, 4)
    memory = memory.write_unsigned(address, 32, 0x80000001)
    initial = MachineExecutionState(
        pc=0xBF60, registers=tuple(registers), memory=memory,
    )
    expected = executor.step(initial).state
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    dispatcher = MachineWasmBlockDispatcher(NodeMachineWasmHost())
    try:
        results = dispatcher.execute(machine.cores[0], 1)
        assert results is not None and results[0].state == expected
        assert machine.cores[0].state.registers[0] == 0xFFFFFFFF80000001
        assert machine.cores[0].step_backward() == initial
    finally:
        dispatcher.close()


def _decoded_single_instruction_program(encoded: bytes, address: int):
    instruction, end = X86ReferenceDecoder().decode_one(
        memoryview(encoded), 0, base_address=address,
    )
    assert end == len(encoded)
    return instruction, SimpleNamespace(
        image=SimpleNamespace(image_base=address, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(instruction,)),
        ),),
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_cdqe_compiled_journal_matches_reference_boundaries_and_wat():
    instruction, program = _decoded_single_instruction_program(
        bytes.fromhex("48 98"), 0xBF80,
    )
    assert instruction.semantic is MachineSemanticToken.SIGN_EXTEND_ACCUMULATOR
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    host = NodeMachineWasmHost()
    try:
        for source, result in (
            (0xAAAAAAAA80000001, 0xFFFFFFFF80000001),
            (0xFFFFFFFF7FFFFFFF, 0x000000007FFFFFFF),
        ):
            registers = [0] * 16
            registers[0] = source
            initial = MachineExecutionState(
                pc=0xBF80, registers=tuple(registers), flags=0xA57,
            )
            expected = executor.step(initial).state
            artifact = executor.recompile_block_wasm(0xBF80, initial, strict=True)
            assert "SIGN_EXTEND_ACCUMULATOR" in artifact.wat
            states = artifact.states_from_journal(host.execute(artifact, initial), initial)
            assert states == (expected,)
            assert states[0].registers[0] == result
            assert states[0].flags == initial.flags
    finally:
        host.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_cqo_compiled_journal_matches_reference_sign_boundaries():
    instruction, program = _decoded_single_instruction_program(
        bytes.fromhex("48 99"), 0xBF88,
    )
    assert instruction.semantic is MachineSemanticToken.SIGN_EXTEND_ACCUMULATOR
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    host = NodeMachineWasmHost()
    try:
        for source, result in ((1 << 63, (1 << 64) - 1), ((1 << 63) - 1, 0)):
            registers = [0] * 16
            registers[0], registers[2] = source, 0x123456789ABCDEF0
            initial = MachineExecutionState(
                pc=0xBF88, registers=tuple(registers), flags=0xA57,
            )
            expected = executor.step(initial).state
            artifact = executor.recompile_block_wasm(0xBF88, initial, strict=True)
            states = artifact.states_from_journal(host.execute(artifact, initial), initial)
            assert states == (expected,)
            assert states[0].registers[2] == result
            assert states[0].flags == initial.flags
    finally:
        host.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_bt_register_compiled_journal_preserves_every_flag_except_carry():
    instruction, program = _decoded_single_instruction_program(
        bytes.fromhex("48 0f a3 c1"), 0xBF90,
    )
    assert instruction.semantic is MachineSemanticToken.BIT_TEST
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    registers = [0] * 16
    registers[0] = 65
    registers[1] = 2
    initial = MachineExecutionState(
        pc=0xBF90, registers=tuple(registers), flags=0xA56,
    )
    expected = executor.step(initial).state
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    dispatcher = MachineWasmBlockDispatcher(NodeMachineWasmHost())
    try:
        results = dispatcher.execute(machine.cores[0], 1)
        assert results is not None and results[0].state == expected
        assert machine.cores[0].state.flags == (initial.flags | 1)
        assert machine.cores[0].state.registers == initial.registers
        assert machine.cores[0].step_backward() == initial
    finally:
        dispatcher.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_bt_memory_signed_register_index_specializes_adjacent_bit_string():
    instruction, program = _decoded_single_instruction_program(
        bytes.fromhex("48 0f a3 0b"), 0xBF98,
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    address = 0xE8F8
    registers = [0] * 16
    registers[1], registers[3] = (1 << 64) - 1, 0xE900
    memory = PagedByteMemory.empty().map_zeroes(address, 8).write_unsigned(
        address, 64, 1 << 63,
    )
    initial = MachineExecutionState(
        pc=0xBF98, registers=tuple(registers), memory=memory, flags=0xA56,
    )
    expected = executor.step(initial).state
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    dispatcher = MachineWasmBlockDispatcher(NodeMachineWasmHost())
    try:
        artifact = executor.recompile_block_wasm(0xBF98, initial, strict=True)
        assert (artifact.guest_memory_base, artifact.guest_memory_size) == (
            address, 8,
        )
        assert dict(artifact.specialization_guard["registers"])[1] == (1 << 64) - 1
        results = dispatcher.execute(machine.cores[0], 1)
        assert results is not None and results[0].state == expected
        assert machine.cores[0].state.flags == (initial.flags | 1)
        assert machine.cores[0].step_backward() == initial
    finally:
        dispatcher.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_btr_memory_compiled_journal_is_exact_witnessed_rmw_and_reverses():
    instruction, program = _decoded_single_instruction_program(
        bytes.fromhex("41 0f ba 30 1f"), 0xBFA0,
    )
    assert instruction.semantic is MachineSemanticToken.BIT_TEST_RESET
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    address = 0xE800
    registers = [0] * 16
    registers[8] = address
    memory = PagedByteMemory.empty().map_zeroes(address, 4).write_unsigned(
        address, 32, 0xFFFFFFFF,
    )
    initial = MachineExecutionState(
        pc=0xBFA0, registers=tuple(registers), memory=memory, flags=0xA56,
    )
    expected = executor.step(initial).state
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    host = NodeMachineWasmHost()
    dispatcher = MachineWasmBlockDispatcher(host)
    try:
        results = dispatcher.execute(machine.cores[0], 1)
        assert results is not None and results[0].state == expected
        assert machine.cores[0].state.memory.read_unsigned(address, 32) == 0x7FFFFFFF
        artifact = executor.recompile_block_wasm(0xBFA0, initial, strict=True)
        journal = host.execute(artifact, initial)
        tampered = bytearray(journal)
        struct.pack_into("<Q", tampered, JOURNAL_EFFECT_OFFSET + 24, 0xFFFFFFFE)
        with pytest.raises(ValueError, match="memory-read witness mismatch"):
            artifact.states_from_journal(bytes(tampered), initial)
        assert machine.cores[0].step_backward() == initial
    finally:
        dispatcher.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
@pytest.mark.parametrize(("direction_flag", "destination"), ((False, 0xE900), (True, 0xE906)))
def test_rep_stosw_compiled_fill_descriptor_matches_reference_and_tamper_closes(
    direction_flag, destination,
):
    instruction, program = _decoded_single_instruction_program(
        bytes.fromhex("66 f3 ab"), 0xBFB0,
    )
    assert instruction.semantic is MachineSemanticToken.STRING_STORE
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    registers = [0] * 16
    registers[0], registers[1], registers[7] = 0xBEEF, 4, destination
    memory = PagedByteMemory.empty().map_zeroes(0xE900, 8).map_bytes(
        0xE900, bytes.fromhex("00 11 22 33 44 55 66 77"),
    )
    initial = MachineExecutionState(
        pc=0xBFB0, registers=tuple(registers), memory=memory,
        flags=(1 << 10) if direction_flag else 0,
    )
    expected = executor.step(initial).state
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    host = NodeMachineWasmHost()
    dispatcher = MachineWasmBlockDispatcher(host)
    try:
        results = dispatcher.execute(machine.cores[0], 1)
        assert results is not None and results[0].state == expected
        assert machine.cores[0].state.memory.read_unsigned(0xE900, 64) == 0xBEEFBEEFBEEFBEEF
        artifact = executor.recompile_block_wasm(0xBFB0, initial, strict=True)
        journal = host.execute(artifact, initial)
        tampered = bytearray(journal)
        struct.pack_into("<Q", tampered, JOURNAL_EFFECT_OFFSET + 16, 3)
        with pytest.raises(ValueError, match="fill descriptor mismatch"):
            artifact.states_from_journal(bytes(tampered), initial)
        assert machine.cores[0].step_backward() == initial
    finally:
        dispatcher.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_scalar_shift_unary_set_and_register_exchange_match_reference_and_reverse():
    encoded = bytes.fromhex(
        "48 c1 e0 01 "  # shl rax, 1
        "48 f7 d0 "     # not rax
        "48 f7 d8 "     # neg rax
        "0f 95 c0 "     # setne al
        "48 87 c8"      # xchg rax, rcx
    )
    decoder = X86ReferenceDecoder()
    instructions = []
    cursor = 0
    while cursor < len(encoded):
        instruction, cursor = decoder.decode_one(
            memoryview(encoded), cursor, base_address=0xBFA0,
        )
        instructions.append(instruction)
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xBFA0, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=tuple(instructions)),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    registers = [0] * 16
    registers[0], registers[1] = 0x8000000000000003, 0x1122334455667788
    initial = MachineExecutionState(
        pc=0xBFA0, registers=tuple(registers), flags=0x202,
    )
    expected_states = []
    expected = initial
    for _instruction in instructions:
        expected = executor.step(expected).state
        expected_states.append(expected)
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    dispatcher = MachineWasmBlockDispatcher(NodeMachineWasmHost())
    try:
        results = dispatcher.execute(machine.cores[0], len(instructions))
        assert results is not None
        assert [item.state for item in results] == expected_states
        assert machine.cores[0].state == expected_states[-1]
        assert dispatcher.statistics["committed_instructions"] == 5
        for expected_prior in reversed((initial, *expected_states[:-1])):
            assert machine.cores[0].step_backward() == expected_prior
    finally:
        dispatcher.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_dynamic_memory_negate_is_one_exact_tamper_checked_rmw_effect():
    address = 0xE408
    instruction = SimpleNamespace(
        address=0xBFE0, encoded=b"\x48\xf7\x5b\x08",
        semantic=MachineSemanticToken.INTEGER_NEGATE,
        token=SimpleNamespace(name="NEG_RM64"), legacy_prefixes=(),
        operands=(
            EffectiveAddressOperand(X86Register.RBX, None, 1, 8, 64, False),
        ),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xBFE0, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(instruction,)),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    registers = [0] * 16
    registers[3] = 0xE400
    memory = PagedByteMemory.empty().map_zeroes(address, 8)
    memory = memory.write_unsigned(address, 64, 7)
    initial = MachineExecutionState(
        pc=0xBFE0, registers=tuple(registers), memory=memory, flags=0x202,
    )
    expected = executor.step(initial).state
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    host = NodeMachineWasmHost()
    dispatcher = MachineWasmBlockDispatcher(host)
    try:
        results = dispatcher.execute(machine.cores[0], 1)
        assert results is not None and results[0].state == expected
        assert machine.cores[0].state.memory.read_unsigned(address, 64) == ((-7) & ((1 << 64) - 1))
        artifact = executor.recompile_block_wasm(0xBFE0, initial, strict=True)
        journal = host.execute(artifact, initial)
        tampered = bytearray(journal)
        struct.pack_into("<Q", tampered, JOURNAL_EFFECT_OFFSET + 24, 6)
        with pytest.raises(ValueError, match="memory-read witness mismatch"):
            artifact.states_from_journal(bytes(tampered), initial)
        assert machine.cores[0].step_backward() == initial
    finally:
        dispatcher.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_memory_exchange_is_one_atomic_witnessed_reversible_edge():
    address = 0xE508
    instruction = SimpleNamespace(
        address=0xC040, encoded=b"\x48\x87\x43\x08",
        semantic=MachineSemanticToken.EXCHANGE,
        token=SimpleNamespace(name="XCHG_RM64_R64"), legacy_prefixes=(),
        operands=(
            EffectiveAddressOperand(X86Register.RBX, None, 1, 8, 64, False),
            RegisterOperand(X86Register.RAX, 64),
        ),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xC040, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(instruction,)),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    registers = [0] * 16
    registers[0], registers[3] = 0x1111222233334444, 0xE500
    memory = PagedByteMemory.empty().map_zeroes(address, 8).write_unsigned(
        address, 64, 0xAAAABBBBCCCCDDDD,
    )
    initial = MachineExecutionState(
        pc=0xC040, registers=tuple(registers), memory=memory,
    )
    expected = executor.step(initial).state
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    dispatcher = MachineWasmBlockDispatcher(NodeMachineWasmHost())
    try:
        results = dispatcher.execute(machine.cores[0], 1)
        assert results is not None and results[0].state == expected
        assert machine.cores[0].state.registers[0] == 0xAAAABBBBCCCCDDDD
        assert machine.cores[0].state.memory.read_unsigned(address, 64) == 0x1111222233334444
        assert machine.cores[0].step_backward() == initial
    finally:
        dispatcher.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
@pytest.mark.parametrize(
    ("semantic", "token", "value", "flags"),
    (
        (MachineSemanticToken.INTEGER_INCREMENT, "INC_RM64", 0x7FFFFFFFFFFFFFFF, 0x203),
        (MachineSemanticToken.INTEGER_DECREMENT, "DEC_RM64", 0x8000000000000000, 0x202),
    ),
)
def test_increment_decrement_preserve_carry_match_boundary_flags_and_reverse(
    semantic, token, value, flags,
):
    instruction = SimpleNamespace(
        address=0xC080, encoded=b"\x48\xff\xc0",
        semantic=semantic, token=SimpleNamespace(name=token), legacy_prefixes=(),
        operands=(RegisterOperand(X86Register.RAX, 64),),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xC080, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(instruction,)),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    registers = (value, *(0 for _ in range(15)))
    initial = MachineExecutionState(
        pc=0xC080, registers=registers, flags=flags,
    )
    expected = executor.step(initial).state
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    dispatcher = MachineWasmBlockDispatcher(NodeMachineWasmHost())
    try:
        results = dispatcher.execute(machine.cores[0], 1)
        assert results is not None and results[0].state == expected
        assert bool(machine.cores[0].state.flags & 1) == bool(flags & 1)
        assert machine.cores[0].state.flags & (1 << 11)
        assert machine.cores[0].step_backward() == initial
    finally:
        dispatcher.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
@pytest.mark.parametrize("zf", (False, True))
def test_cmovne_r32_matches_true_and_false_write_semantics_and_reverses(zf):
    instruction = SimpleNamespace(
        address=0xC0C0, encoded=b"\x0f\x45\xc1",
        semantic=MachineSemanticToken.CONDITIONAL_MOVE,
        token=SimpleNamespace(name="CMOVNE_R32_RM32"), legacy_prefixes=(),
        operands=(
            RegisterOperand(X86Register.RAX, 32),
            RegisterOperand(X86Register.RCX, 32),
        ),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xC0C0, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(instruction,)),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    registers = [0] * 16
    registers[0], registers[1] = 0xAAAABBBBCCCCDDDD, 0x1111222233334444
    initial = MachineExecutionState(
        pc=0xC0C0, registers=tuple(registers),
        flags=(0x202 | ((1 << 6) if zf else 0)),
    )
    expected = executor.step(initial).state
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    dispatcher = MachineWasmBlockDispatcher(NodeMachineWasmHost())
    try:
        results = dispatcher.execute(machine.cores[0], 1)
        assert results is not None and results[0].state == expected
        assert machine.cores[0].state.registers[0] == (
            0xAAAABBBBCCCCDDDD if zf else 0x33334444
        )
        assert machine.cores[0].step_backward() == initial
    finally:
        dispatcher.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_false_cmov_memory_source_still_has_authenticated_read_provenance():
    address = 0xE608
    instruction = SimpleNamespace(
        address=0xC100, encoded=b"\x48\x0f\x45\x43\x08",
        semantic=MachineSemanticToken.CONDITIONAL_MOVE,
        token=SimpleNamespace(name="CMOVNE_R64_RM64"), legacy_prefixes=(),
        operands=(
            RegisterOperand(X86Register.RAX, 64),
            EffectiveAddressOperand(X86Register.RBX, None, 1, 8, 64, False),
        ),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xC100, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(instruction,)),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    registers = [0] * 16
    registers[0], registers[3] = 0x123456789ABCDEF0, 0xE600
    memory_value = 0x0FEDCBA987654321
    memory = PagedByteMemory.empty().map_zeroes(address, 8).write_unsigned(
        address, 64, memory_value,
    )
    initial = MachineExecutionState(
        pc=0xC100, registers=tuple(registers), memory=memory,
        flags=0x202 | (1 << 6),
    )
    expected = executor.step(initial).state
    artifact = executor.recompile_block_wasm(0xC100, initial, strict=True)
    host = NodeMachineWasmHost()
    try:
        journal = host.execute(artifact, initial)
        assert artifact.states_from_journal(journal, initial) == (expected,)
        tampered = bytearray(journal)
        struct.pack_into(
            "<Q", tampered, JOURNAL_EFFECT_OFFSET + 24, memory_value ^ 1,
        )
        with pytest.raises(ValueError, match="memory-read witness mismatch"):
            artifact.states_from_journal(bytes(tampered), initial)
    finally:
        host.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
@pytest.mark.parametrize(
    ("left", "right", "carry"),
    (
        (9, 3, False),
        (9, 3, True),
        (0x1234, 0xFFFFFFFFFFFFFFFF, True),
    ),
)
def test_sbb_carry_input_and_effective_operand_overflow_match_and_reverse(
    left, right, carry,
):
    instruction = SimpleNamespace(
        address=0xC140, encoded=b"\x48\x1b\xc1",
        semantic=MachineSemanticToken.INTEGER_SUBTRACT_WITH_BORROW,
        token=SimpleNamespace(name="SBB_R64_RM64"), legacy_prefixes=(),
        operands=(
            RegisterOperand(X86Register.RAX, 64),
            RegisterOperand(X86Register.RCX, 64),
        ),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xC140, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(instruction,)),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    registers = [0] * 16
    registers[0], registers[1] = left, right
    initial = MachineExecutionState(
        pc=0xC140, registers=tuple(registers),
        flags=(0x202 | int(carry)),
    )
    expected = executor.step(initial).state
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    dispatcher = MachineWasmBlockDispatcher(NodeMachineWasmHost())
    try:
        results = dispatcher.execute(machine.cores[0], 1)
        assert results is not None and results[0].state == expected
        assert machine.cores[0].step_backward() == initial
    finally:
        dispatcher.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
@pytest.mark.parametrize(
    ("semantic", "token", "value", "count"),
    (
        (MachineSemanticToken.SHIFT_RIGHT_ARITHMETIC, "SAR_RM64_1", 0x8000000000000003, 1),
        (MachineSemanticToken.ROTATE_LEFT, "ROL_RM64_IMM8", 0x8000000000000001, 1),
        (MachineSemanticToken.ROTATE_LEFT, "ROL_RM64_IMM8", 0x0123456789ABCDEF, 9),
    ),
)
def test_sar_and_rol_match_reference_flags_and_reverse(semantic, token, value, count):
    instruction = SimpleNamespace(
        address=0xC180, encoded=b"\x48\xc1\xc0\x01",
        semantic=semantic, token=SimpleNamespace(name=token), legacy_prefixes=(),
        operands=(
            RegisterOperand(X86Register.RAX, 64),
            ImmediateOperand(count, 8, False),
        ),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xC180, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(instruction,)),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    initial = MachineExecutionState(
        pc=0xC180, registers=(value, *(0 for _ in range(15))), flags=0xA93,
    )
    expected = executor.step(initial).state
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    dispatcher = MachineWasmBlockDispatcher(NodeMachineWasmHost())
    try:
        results = dispatcher.execute(machine.cores[0], 1)
        assert results is not None and results[0].state == expected
        assert machine.cores[0].step_backward() == initial
    finally:
        dispatcher.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
@pytest.mark.parametrize(
    ("left", "right"),
    (
        (3, 5),
        (0xFFFFFFFFFFFFFFFF, 2),
        (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF),
    ),
)
def test_mul_r64_produces_exact_rdx_rax_product_flags_and_reverse(left, right):
    instruction = SimpleNamespace(
        address=0xC1C0, encoded=b"\x48\xf7\xe1",
        semantic=MachineSemanticToken.INTEGER_MULTIPLY_UNSIGNED,
        token=SimpleNamespace(name="MUL_RM64"), legacy_prefixes=(),
        operands=(RegisterOperand(X86Register.RCX, 64),),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xC1C0, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(instruction,)),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    registers = [0] * 16
    registers[0], registers[1] = left, right
    initial = MachineExecutionState(
        pc=0xC1C0, registers=tuple(registers), flags=0xA92,
    )
    expected = executor.step(initial).state
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    dispatcher = MachineWasmBlockDispatcher(NodeMachineWasmHost())
    try:
        results = dispatcher.execute(machine.cores[0], 1)
        assert results is not None and results[0].state == expected
        product = left * right
        assert machine.cores[0].state.registers[0] == product & ((1 << 64) - 1)
        assert machine.cores[0].state.registers[2] == (product >> 64) & ((1 << 64) - 1)
        assert machine.cores[0].step_backward() == initial
    finally:
        dispatcher.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_mul_memory_source_is_read_witnessed_and_reversible():
    address = 0xE708
    instruction = SimpleNamespace(
        address=0xC200, encoded=b"\x48\xf7\x63\x08",
        semantic=MachineSemanticToken.INTEGER_MULTIPLY_UNSIGNED,
        token=SimpleNamespace(name="MUL_RM64"), legacy_prefixes=(),
        operands=(
            EffectiveAddressOperand(X86Register.RBX, None, 1, 8, 64, False),
        ),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xC200, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(instruction,)),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    registers = [0] * 16
    registers[0], registers[3] = 0x100000000, 0xE700
    memory = PagedByteMemory.empty().map_zeroes(address, 8).write_unsigned(
        address, 64, 0x100000001,
    )
    initial = MachineExecutionState(
        pc=0xC200, registers=tuple(registers), memory=memory,
    )
    expected = executor.step(initial).state
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    dispatcher = MachineWasmBlockDispatcher(NodeMachineWasmHost())
    try:
        results = dispatcher.execute(machine.cores[0], 1)
        assert results is not None and results[0].state == expected
        assert machine.cores[0].step_backward() == initial
    finally:
        dispatcher.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_vector_xor_round_trips_all_128_bits_through_compiled_checkpoint_and_reverse():
    instruction = SimpleNamespace(
        address=0xC240, encoded=b"\x0f\x57\xc1",
        semantic=MachineSemanticToken.VECTOR_XOR,
        token=SimpleNamespace(name="XORPS_XMM_XMMM128"), legacy_prefixes=(),
        operands=(
            VectorRegisterOperand(X86VectorRegister.XMM0, 128),
            VectorRegisterOperand(X86VectorRegister.XMM1, 128),
        ),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xC240, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(instruction,)),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    vectors = [0] * 16
    vectors[0] = 0xFFEEDDCCBBAA99887766554433221100
    vectors[1] = 0x0123456789ABCDEFFEDCBA9876543210
    initial = MachineExecutionState(
        pc=0xC240, vector_registers=tuple(vectors),
    )
    expected = executor.step(initial).state
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    dispatcher = MachineWasmBlockDispatcher(NodeMachineWasmHost())
    try:
        results = dispatcher.execute(machine.cores[0], 1)
        assert results is not None and results[0].state == expected
        assert machine.cores[0].state.vector_registers[0] == vectors[0] ^ vectors[1]
        artifact = executor.recompile_block_wasm(0xC240, initial, strict=True)
        assert artifact.state_abi["schema"] == "turing.machine-block-state.v2"
        assert artifact.state_abi["vector_register_count"] == 16
        assert len(artifact.pack_state(initial)) == artifact.state_abi["state_size"]
        snapshot = MachineSnapshotView(memoryview(build_machine_state_snapshot(
            (machine.cores[0].state,), maximum_output_bytes=0,
        )))
        vector_value = vectors[0] ^ vectors[1]
        assert snapshot.register_words(0, 20) == (
            vector_value & 0xFFFFFFFF, (vector_value >> 32) & 0xFFFFFFFF,
        )
        assert snapshot.register_words(0, 21) == (
            (vector_value >> 64) & 0xFFFFFFFF,
            (vector_value >> 96) & 0xFFFFFFFF,
        )
        assert machine.cores[0].step_backward() == initial
    finally:
        dispatcher.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_movdqa_memory_load_has_exact_128_bit_read_witness_and_reverses():
    address = 0xE800
    instruction = SimpleNamespace(
        address=0xC280, encoded=b"\x66\x0f\x6f\x13",
        semantic=MachineSemanticToken.VECTOR_MOVE,
        token=SimpleNamespace(name="MOVDQA_XMM_XMMM128"), legacy_prefixes=(0x66,),
        operands=(
            VectorRegisterOperand(X86VectorRegister.XMM2, 128),
            EffectiveAddressOperand(X86Register.RBX, None, 1, 0, 64, False),
        ),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xC280, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(instruction,)),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    value = 0xFEDCBA98765432100123456789ABCDEF
    registers = [0] * 16
    registers[3] = address
    memory = PagedByteMemory.empty().map_zeroes(address, 16).write_unsigned(
        address, 128, value,
    )
    initial = MachineExecutionState(
        pc=0xC280, registers=tuple(registers), memory=memory,
    )
    expected = executor.step(initial).state
    artifact = executor.recompile_block_wasm(0xC280, initial, strict=True)
    host = NodeMachineWasmHost()
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    dispatcher = MachineWasmBlockDispatcher(host)
    try:
        journal = host.execute(artifact, initial)
        assert artifact.states_from_journal(journal, initial) == (expected,)
        tampered = bytearray(journal)
        struct.pack_into(
            "<Q", tampered, JOURNAL_EFFECT_OFFSET + 32,
            ((value >> 64) & ((1 << 64) - 1)) ^ 1,
        )
        with pytest.raises(ValueError, match="memory-read witness mismatch"):
            artifact.states_from_journal(bytes(tampered), initial)

        results = dispatcher.execute(machine.cores[0], 1)
        assert results is not None and results[0].state == expected
        assert machine.cores[0].state.vector_registers[2] == value
        assert machine.cores[0].step_backward() == initial
    finally:
        dispatcher.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_widened_vector_journal_places_guest_mirror_after_long_checkpoint_batch():
    base, data_address = 0xC300, 0xE900
    instructions = [
        SimpleNamespace(
            address=base + index, encoded=b"\x90",
            semantic=MachineSemanticToken.NO_OPERATION,
            token=SimpleNamespace(name="NOP"), legacy_prefixes=(), operands=(),
        )
        for index in range(8)
    ]
    instructions.append(SimpleNamespace(
        address=base + 8, encoded=b"\x90",
        semantic=MachineSemanticToken.REGISTER_OR_MEMORY_READ,
        token=SimpleNamespace(name="MOV_R64_RM64"), legacy_prefixes=(),
        operands=(
            RegisterOperand(X86Register.RAX, 64),
            EffectiveAddressOperand(
                None, None, 1, data_address - (base + 9), 64, True,
            ),
        ),
    ))
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=base, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=tuple(instructions)),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    memory = PagedByteMemory.empty().map_zeroes(data_address, 8).write_unsigned(
        data_address, 64, 0xCAFEBABEDEADBEEF,
    )
    initial = MachineExecutionState(pc=base, memory=memory)
    artifact = executor.recompile_block_wasm(base, initial, strict=True)
    assert artifact.covered_operation_count == 9
    assert artifact.state_abi["guest_buffer_offset"] >= 8192
    host = NodeMachineWasmHost()
    try:
        journal = host.execute(artifact, initial)
        states = artifact.states_from_journal(journal, initial)
        assert len(states) == 9
        assert states[-1].registers[0] == 0xCAFEBABEDEADBEEF
    finally:
        host.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_vector_memory_store_is_one_exact_128_bit_write_and_reverses():
    address = 0xEA08
    instruction = SimpleNamespace(
        address=0xC380, encoded=b"\x0f\x11\x5b\x08",
        semantic=MachineSemanticToken.VECTOR_MOVE,
        token=SimpleNamespace(name="MOVUPS_XMMM128_XMM"), legacy_prefixes=(),
        operands=(
            EffectiveAddressOperand(X86Register.RBX, None, 1, 8, 64, False),
            VectorRegisterOperand(X86VectorRegister.XMM3, 128),
        ),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xC380, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(instruction,)),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    registers = [0] * 16
    registers[3] = 0xEA00
    vectors = [0] * 16
    vectors[3] = 0x00112233445566778899AABBCCDDEEFF
    before = 0xFFEEDDCCBBAA99887766554433221100
    memory = PagedByteMemory.empty().map_zeroes(address, 16).write_unsigned(
        address, 128, before,
    )
    initial = MachineExecutionState(
        pc=0xC380, registers=tuple(registers),
        vector_registers=tuple(vectors), memory=memory,
    )
    expected = executor.step(initial).state
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    dispatcher = MachineWasmBlockDispatcher(NodeMachineWasmHost())
    try:
        results = dispatcher.execute(machine.cores[0], 1)
        assert results is not None and results[0].state == expected
        assert machine.cores[0].state.memory.read_unsigned(address, 128) == vectors[3]
        assert machine.cores[0].step_backward() == initial
    finally:
        dispatcher.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
@pytest.mark.parametrize(
    ("semantic", "address", "target"),
    (
        (MachineSemanticToken.INDIRECT_JUMP, 0xBA00, 0xBA80),
        (MachineSemanticToken.INDIRECT_CALL, 0xBB00, 0xBB80),
    ),
)
def test_register_indirect_internal_control_is_specialized_and_reversible(
    semantic, address, target,
):
    instruction = SimpleNamespace(
        address=address, encoded=b"\xff\xe0",
        semantic=semantic,
        token=SimpleNamespace(name=(
            "CALL_RM64" if semantic is MachineSemanticToken.INDIRECT_CALL
            else "JMP_RM64"
        )),
        operands=(RegisterOperand(X86Register.RAX, 64),),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=address, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(instruction,)),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
        indirect_target_handler=indirect_target,
    )
    registers = [0] * 16
    registers[0], registers[4] = target, 0xD008
    initial = MachineExecutionState(
        pc=address, registers=tuple(registers),
        memory=PagedByteMemory.empty().map_zeroes(0xD000, 16),
    )
    expected = executor.step(initial).state
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    dispatcher = MachineWasmBlockDispatcher(NodeMachineWasmHost())
    try:
        results = dispatcher.execute(machine.cores[0], 1)
        assert results is not None and len(results) == 1
        assert machine.cores[0].state == expected
        assert machine.cores[0].state.pc == target
        if semantic is MachineSemanticToken.INDIRECT_CALL:
            assert machine.cores[0].state.call_stack == (address + 2,)
            assert machine.cores[0].state.memory.read_unsigned(0xD000, 64) == address + 2
        assert machine.cores[0].step_backward() == initial
    finally:
        dispatcher.close()


def test_external_indirect_target_stays_at_capability_request_boundary():
    target = 0xFFFF800000000100
    instruction = SimpleNamespace(
        address=0xBC00, encoded=b"\xff\xd0",
        semantic=MachineSemanticToken.INDIRECT_CALL,
        token=SimpleNamespace(name="CALL_RM64"),
        operands=(RegisterOperand(X86Register.RAX, 64),),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xBC00, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(instruction,)),
        ),),
    )
    reference = MachineExternalReference(1, target, "windows", "demo.dll", "Run")
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
        indirect_target_handler=indirect_target,
        external_target_resolver=lambda candidate: reference if candidate == target else None,
    )
    registers = [0] * 16
    registers[0], registers[4] = target, 0xD008
    initial = MachineExecutionState(
        pc=0xBC00, registers=tuple(registers),
        memory=PagedByteMemory.empty().map_zeroes(0xD000, 64),
    )
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    host = SimpleNamespace(
        statistics={"requests": 0}, close=lambda: None,
        execute=lambda *_args: pytest.fail("external target reached Wasm host"),
    )
    dispatcher = MachineWasmBlockDispatcher(host)
    try:
        assert dispatcher.execute(machine.cores[0], 1) is None
        result = machine.cores[0].step_forward()
        assert result.status.name == "WAITING_EXTERNAL"
        assert result.state.external_requests[0].reference == reference
        assert dispatcher.statistics["host_requests"] == 0
    finally:
        dispatcher.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_specialized_push_and_pop_preserve_stack_memory_and_reverse():
    instructions = (
        SimpleNamespace(
            address=0xBE00, encoded=b"\x50",
            semantic=MachineSemanticToken.STACK_PUSH,
            token=SimpleNamespace(name="PUSH_R64"),
            operands=(RegisterOperand(X86Register.RAX, 64),),
        ),
        SimpleNamespace(
            address=0xBE01, encoded=b"\x5b",
            semantic=MachineSemanticToken.STACK_POP,
            token=SimpleNamespace(name="POP_R64"),
            operands=(RegisterOperand(X86Register.RBX, 64),),
        ),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xBE00, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=instructions),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    registers = [0] * 16
    registers[0], registers[4] = 0x123456789ABCDEF0, 0xD108
    initial = MachineExecutionState(
        pc=0xBE00, registers=tuple(registers),
        memory=PagedByteMemory.empty().map_zeroes(0xD100, 16),
    )
    pushed = executor.step(initial).state
    popped = executor.step(pushed).state
    machine = MachineVirtualMulticore.create(
        executor, core_count=1, initial_states=(initial,),
    )
    dispatcher = MachineWasmBlockDispatcher(NodeMachineWasmHost())
    try:
        assert dispatcher.execute(machine.cores[0], 2)[0].state == pushed
        assert dispatcher.execute(machine.cores[0], 1)[0].state == popped
        assert machine.cores[0].state.registers[3] == 0x123456789ABCDEF0
        assert machine.cores[0].state.registers[4] == 0xD108
        assert machine.cores[0].step_backward() == pushed
        assert machine.cores[0].step_backward() == initial
    finally:
        dispatcher.close()


def test_machine_block_module_sizes_linear_memory_for_the_declared_guest_window():
    # The state and journal live below guest+4096, so a full 64 KiB guest mirror
    # necessarily requires a second Wasm page.
    instruction = SimpleNamespace(
        address=0xC000, encoded=b"\x90",
        semantic=MachineSemanticToken.REGISTER_OR_MEMORY_READ,
        token=SimpleNamespace(name="MOV_R64_RM64"), legacy_prefixes=(),
        operands=(),
    )
    # Exercise the sizing rule directly through a valid artifact whose ABI is
    # then widened to the maximum by lowering two static endpoints.
    instruction.operands = (
        RegisterOperand(X86Register.RAX, 64),
        EffectiveAddressOperand(None, None, 1, 0xD000 - 0xC001, 64, True),
    )
    store = SimpleNamespace(
        address=0xC001, encoded=b"\x90",
        semantic=MachineSemanticToken.REGISTER_OR_MEMORY_WRITE,
        token=SimpleNamespace(name="MOV_RM64_R64"), legacy_prefixes=(),
        operands=(
            EffectiveAddressOperand(None, None, 1, 0x1CFF8 - 0xC002, 64, True),
            RegisterOperand(X86Register.RAX, 64),
        ),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xC000, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(instruction, store)),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    artifact = executor.recompile_block_wasm(0xC000, strict=True)
    assert artifact.guest_memory_size == 64 * 1024
    assert artifact.state_abi["memory_pages"] == 2
    assert '(memory (export "memory") 2)' in artifact.wat


def test_unsupported_single_memory_operand_does_not_poison_safe_prefix_planning():
    from src.compiler.machine_reference_vocabulary import EffectiveAddressOperand

    instructions = (
        SimpleNamespace(
            address=0xE000, encoded=b"\x90",
            semantic=MachineSemanticToken.NO_OPERATION, operands=(),
        ),
        SimpleNamespace(
            address=0xE001, encoded=b"\x90",
            semantic=MachineSemanticToken.STACK_PUSH,
            operands=(EffectiveAddressOperand(None, None, 1, 0x100, 64, False),),
        ),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xE000, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=instructions),
        ),),
    )
    artifact = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    ).recompile_block_wasm(0xE000)
    assert artifact.covered_operation_count == 1
    assert artifact.shortfalls[0].semantic == "STACK_PUSH"
    assert artifact.guest_memory_size == 0


def test_distant_future_memory_access_yields_a_bounded_compilable_prefix():
    instructions = tuple(
        SimpleNamespace(
            address=0xE200 + index, encoded=b"\x90",
            semantic=MachineSemanticToken.REGISTER_OR_MEMORY_READ,
            token=SimpleNamespace(name="MOV_R64_RM64"), legacy_prefixes=(),
            operands=(
                RegisterOperand((X86Register.RAX, X86Register.RCX)[index], 64),
                EffectiveAddressOperand(
                    None, None, 1,
                    memory_address - (0xE201 + index), 64, True,
                ),
            ),
        )
        for index, memory_address in enumerate((0x10000, 0x20000))
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0xE200, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=instructions),
        ),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    artifact = executor.recompile_block_wasm(0xE200)
    assert artifact.covered_operation_count == 1
    assert not artifact.complete
    assert artifact.guest_memory_base == 0x10000
    assert artifact.guest_memory_size == 8
    assert artifact.continuation_address == 0xE201
    assert artifact.shortfalls[0].address == 0xE201
    with pytest.raises(MachineBlockLoweringError, match="bounded capacity"):
        executor.recompile_block_wasm(0xE200, strict=True)
