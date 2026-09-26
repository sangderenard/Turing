import base64
import json
import shutil
import struct
import subprocess
from types import SimpleNamespace

import pytest

from src.compiler.amd64_machine_semantics import PagedByteMemory, default_effect_handlers
from src.compiler.machine_block_recompiler import (
    JOURNAL_EFFECT_OFFSET,
    JOURNAL_STACK_OFFSET,
    JOURNAL_STATE_OFFSET,
    JOURNAL_STRIDE,
    MachineBlockLoweringError,
)
from src.compiler.machine_execution import (
    MachineExecutionOrchestrator,
    MachineExecutionState,
    ReversibleMachineExecutor,
)
from src.compiler.machine_reference_vocabulary import X86ReferenceDecoder
from src.compiler.machine_reference_vocabulary import (
    ImmediateOperand,
    EffectiveAddressOperand,
    MachineSemanticToken,
    RegisterOperand,
    X86Register,
    RelativeAddressOperand,
)
from src.compiler.amd64_machine_semantics import condition_holds


def _mov_then_return_program():
    encoded = b"\xb8\x09\x00\x00\x00\xc3"  # MOV EAX,9; RET
    decoder = X86ReferenceDecoder()
    instructions = []
    cursor = 0
    while cursor < len(encoded):
        instruction, cursor = decoder.decode_one(
            memoryview(encoded), cursor, base_address=0x1000,
        )
        instructions.append(instruction)
    return SimpleNamespace(
        image=SimpleNamespace(image_base=0x1000, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=tuple(instructions)),
        ),),
    )


def test_machine_block_wasm_is_executable_and_retains_instruction_journal(tmp_path):
    executor = MachineExecutionOrchestrator(
        _mov_then_return_program(), effect_handlers=default_effect_handlers(),
    )
    artifact = executor.recompile_block_wasm(0x1000)

    assert artifact.covered_operation_count == 1
    assert not artifact.complete
    assert artifact.continuation_address == 0x1005
    assert artifact.shortfalls[0].address == 0x1005
    assert artifact.witnesses[0].encoded == b"\xb8\x09\x00\x00\x00"
    assert "(i64.store" in artifact.wat
    assert "guest 0x1000 REGISTER_WRITE_IMMEDIATE" in artifact.wat
    with pytest.raises(MachineBlockLoweringError, match="stopped at 0x1005"):
        executor.recompile_block_wasm(0x1000, strict=True)

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the emitted Wasm artifact")
    wasm_path = tmp_path / "machine-block.wasm"
    wasm_path.write_bytes(artifact.binary)
    initial = MachineExecutionState(
        pc=0x1000,
        registers=(0xFFFFFFFFFFFFFFFF, *(0 for _ in range(15))),
        flags=0x202,
        steps=4,
    )
    state_path = tmp_path / "state.bin"
    state_path.write_bytes(artifact.pack_state(initial))
    script_path = tmp_path / "run.mjs"
    script_path.write_text(
        """
import {readFileSync} from 'node:fs';
const [wasmPath, statePath, stride] = process.argv.slice(2);
const {instance} = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const bytes = new Uint8Array(instance.exports.memory.buffer);
bytes.set(readFileSync(statePath), 0);
instance.exports.run(0, 1024, 4096);
const journal = Buffer.from(instance.exports.memory.buffer, 1024, Number(stride));
console.log(JSON.stringify({
  rax: new DataView(instance.exports.memory.buffer).getBigUint64(0, true).toString(),
  journal: journal.toString('base64')
}));
""".strip(),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            node, str(script_path), str(wasm_path), str(state_path),
            str(JOURNAL_STRIDE * artifact.covered_operation_count),
        ],
        check=True, capture_output=True, text=True, timeout=20,
    )
    observed = json.loads(completed.stdout)
    assert observed["rax"] == "9"  # 32-bit MOV zero-extends on AMD64
    journal = base64.b64decode(observed["journal"])
    (compiled_state,) = artifact.states_from_journal(journal, initial)
    assert compiled_state.registers[0] == 9
    assert compiled_state.pc == 0x1005
    assert compiled_state.flags == 0x202
    assert compiled_state.steps == 5

    runtime = ReversibleMachineExecutor.create(executor, initial)
    (committed,) = runtime.commit_recompiled_journal(artifact, journal)
    assert committed.instruction.address == 0x1000
    assert runtime.position == 1
    assert runtime.state == compiled_state
    assert runtime.step_backward() == initial
    tampered = bytearray(journal)
    tampered[16] ^= 1
    with pytest.raises(ValueError, match="digest mismatch"):
        artifact.states_from_journal(bytes(tampered), initial)


def test_machine_block_wasm_fails_closed_when_first_effect_is_not_admitted():
    program = _mov_then_return_program()
    # Enter at RET: the first tier cannot hide a control transition in a
    # supposedly compiled block.
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    with pytest.raises(MachineBlockLoweringError, match="no Wasm-safe prefix"):
        executor.recompile_block_wasm(0x1005)


def _arithmetic_program(semantic, width, right, *, register_source=False):
    instruction = SimpleNamespace(
        address=0x2000,
        encoded=b"\x90",
        semantic=semantic,
        operands=(
            RegisterOperand(X86Register.RAX, width),
            RegisterOperand(X86Register.RCX, width)
            if register_source else ImmediateOperand(right, width, False),
        ),
    )
    ret = SimpleNamespace(
        address=0x2001, encoded=b"\xc3",
        semantic=MachineSemanticToken.RETURN, operands=(),
    )
    return SimpleNamespace(
        image=SimpleNamespace(image_base=0x2000, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(instruction, ret)),
        ),),
    )


def _run_artifact(node, tmp_path, name, artifact, state):
    wasm_path = tmp_path / f"{name}.wasm"
    state_path = tmp_path / f"{name}.state.bin"
    guest_path = tmp_path / f"{name}.guest.bin"
    script_path = tmp_path / "run-arithmetic.mjs"
    wasm_path.write_bytes(artifact.binary)
    state_path.write_bytes(artifact.pack_state(state))
    guest_path.write_bytes(artifact.pack_guest_memory(state))
    if not script_path.exists():
        script_path.write_text(
            """
import {readFileSync} from 'node:fs';
const [wasmPath, statePath, guestPath, total] = process.argv.slice(2);
const {instance} = await WebAssembly.instantiate(readFileSync(wasmPath), {});
new Uint8Array(instance.exports.memory.buffer).set(readFileSync(statePath), 0);
const guest = readFileSync(guestPath);
new Uint8Array(instance.exports.memory.buffer).set(guest, 4096);
instance.exports.run(0, 1024, 4096);
console.log(JSON.stringify({
  journal: Buffer.from(instance.exports.memory.buffer, 1024, Number(total)).toString('base64'),
  guest: Buffer.from(instance.exports.memory.buffer, 4096, guest.length).toString('base64')
}));
""".strip(),
            encoding="utf-8",
        )
    completed = subprocess.run(
        [
            node, str(script_path), str(wasm_path), str(state_path),
            str(guest_path),
            str(JOURNAL_STRIDE * artifact.covered_operation_count),
        ],
        check=True, capture_output=True, text=True, timeout=20,
    )
    observed = json.loads(completed.stdout)
    return base64.b64decode(observed["journal"]), base64.b64decode(observed["guest"])


def test_wasm_arithmetic_matches_reference_flags_across_widths(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for differential Wasm execution")
    mask64 = 0xFFFFFFFFFFFFFFFF
    cases = (
        (MachineSemanticToken.INTEGER_ADD, 64, mask64, 1, False),
        (MachineSemanticToken.INTEGER_ADD, 64, 41, 1, True),
        (MachineSemanticToken.INTEGER_ADD, 32, mask64, 1, False),
        (MachineSemanticToken.INTEGER_ADD, 16, 0x123400000000FFFF, 1, False),
        (MachineSemanticToken.INTEGER_ADD, 8, 0x123400000000007F, 1, False),
        (MachineSemanticToken.INTEGER_SUBTRACT, 64, 0, 1, False),
        (MachineSemanticToken.INTEGER_SUBTRACT, 8, 0x80, 1, False),
        (MachineSemanticToken.BITWISE_AND, 64, mask64, 0, False),
        (MachineSemanticToken.BITWISE_OR, 32, 0xFFFF000000000000, 0, False),
        (MachineSemanticToken.BITWISE_XOR, 16, 0xABCD00000000FFFF, 0xFFFF, True),
        (MachineSemanticToken.INTEGER_COMPARE, 64, 0, 1, False),
        (MachineSemanticToken.INTEGER_COMPARE, 8, 0x80, 1, True),
        (MachineSemanticToken.INTEGER_TEST, 32, 0xFFFF0000AAAA5555, 0x00FF00FF, False),
    )
    for index, (semantic, width, left, right, register_source) in enumerate(cases):
        executor = MachineExecutionOrchestrator(
            _arithmetic_program(
                semantic, width, right, register_source=register_source,
            ),
            effect_handlers=default_effect_handlers(),
        )
        initial = MachineExecutionState(
            pc=0x2000,
            registers=(left, right, *(0 for _ in range(14))),
            flags=0xA55,
            steps=17,
        )
        interpreted = executor.step(initial).state
        artifact = executor.recompile_block_wasm(0x2000)
        journal, _guest = _run_artifact(
            node, tmp_path, f"case-{index}", artifact, initial,
        )
        (compiled,) = artifact.states_from_journal(journal, initial)
        assert compiled.registers == interpreted.registers
        assert compiled.pc == interpreted.pc
        assert compiled.flags == interpreted.flags
        assert compiled.steps == interpreted.steps
        runtime = ReversibleMachineExecutor.create(executor, initial)
        runtime.commit_recompiled_journal(artifact, journal)
        assert runtime.state == compiled
        assert runtime.step_backward() == initial


def test_multi_instruction_wasm_block_commits_and_reverses_each_guest_edge(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for differential Wasm execution")
    instructions = (
        SimpleNamespace(
            address=0x3000, encoded=b"\x90",
            semantic=MachineSemanticToken.REGISTER_WRITE_IMMEDIATE,
            operands=(RegisterOperand(X86Register.RAX, 64), ImmediateOperand(0x7F, 64, False)),
        ),
        SimpleNamespace(
            address=0x3001, encoded=b"\x90",
            semantic=MachineSemanticToken.INTEGER_ADD,
            operands=(RegisterOperand(X86Register.RAX, 8), ImmediateOperand(1, 8, False)),
        ),
        SimpleNamespace(
            address=0x3002, encoded=b"\x90",
            semantic=MachineSemanticToken.BITWISE_XOR,
            operands=(RegisterOperand(X86Register.RAX, 32), RegisterOperand(X86Register.RCX, 32)),
        ),
        SimpleNamespace(
            address=0x3003, encoded=b"\xc3",
            semantic=MachineSemanticToken.RETURN, operands=(),
        ),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0x3000, entrypoint_rva=0),
        functions=(SimpleNamespace(report=SimpleNamespace(instructions=instructions)),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    initial = MachineExecutionState(
        pc=0x3000, registers=(0, 0x55, *(0 for _ in range(14))),
        flags=0x202, steps=100,
    )
    expected = []
    active = initial
    for _ in range(3):
        active = executor.step(active).state
        expected.append(active)
    artifact = executor.recompile_block_wasm(0x3000)
    assert artifact.covered_operation_count == 3
    assert artifact.continuation_address == 0x3003
    journal, _guest = _run_artifact(node, tmp_path, "multi", artifact, initial)
    compiled = artifact.states_from_journal(journal, initial)
    assert compiled == tuple(expected)

    runtime = ReversibleMachineExecutor.create(executor, initial)
    results = runtime.commit_recompiled_journal(artifact, journal)
    assert [item.instruction.address for item in results] == [0x3000, 0x3001, 0x3002]
    assert runtime.position == 3
    assert runtime.step_backward() == expected[1]
    assert runtime.step_backward() == expected[0]
    assert runtime.step_backward() == initial


def test_static_guest_memory_load_store_is_witnessed_and_reversible(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for guest-memory Wasm execution")
    load_address = 0x5000
    store_address = 0x5008
    instructions = (
        SimpleNamespace(
            address=0x4000, encoded=b"\x90",
            semantic=MachineSemanticToken.REGISTER_OR_MEMORY_READ,
            legacy_prefixes=(),
            token=SimpleNamespace(name="MOV_R64_RM64"),
            operands=(
                RegisterOperand(X86Register.RAX, 64),
                EffectiveAddressOperand(None, None, 1, load_address - 0x4001, 64, True),
            ),
        ),
        SimpleNamespace(
            address=0x4001, encoded=b"\x90",
            semantic=MachineSemanticToken.REGISTER_OR_MEMORY_WRITE,
            legacy_prefixes=(),
            token=SimpleNamespace(name="MOV_RM64_R64"),
            operands=(
                EffectiveAddressOperand(None, None, 1, store_address - 0x4002, 64, True),
                RegisterOperand(X86Register.RAX, 64),
            ),
        ),
        SimpleNamespace(
            address=0x4002, encoded=b"\xc3",
            semantic=MachineSemanticToken.RETURN, operands=(),
        ),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0x4000, entrypoint_rva=0),
        functions=(SimpleNamespace(report=SimpleNamespace(instructions=instructions)),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    memory = PagedByteMemory.empty().map_zeroes(0x5000, 16)
    memory = memory.write_unsigned(load_address, 64, 0x1122334455667788)
    memory = memory.write_unsigned(store_address, 64, 0xDEADBEEF)
    initial = MachineExecutionState(pc=0x4000, memory=memory, steps=7)
    first = executor.step(initial).state
    second = executor.step(first).state

    artifact = executor.recompile_block_wasm(0x4000)
    assert artifact.covered_operation_count == 2
    assert artifact.guest_memory_base == load_address
    assert artifact.guest_memory_size == 16
    journal, guest = _run_artifact(node, tmp_path, "guest-memory", artifact, initial)
    compiled = artifact.states_from_journal(journal, initial)
    assert compiled == (first, second)
    assert int.from_bytes(guest[8:16], "little") == 0x1122334455667788

    runtime = ReversibleMachineExecutor.create(executor, initial)
    runtime.commit_recompiled_journal(artifact, journal)
    assert runtime.state.memory.read_unsigned(store_address, 64) == 0x1122334455667788
    assert runtime.step_backward() == first
    assert runtime.step_backward() == initial

    tampered = bytearray(journal)
    tampered[JOURNAL_STRIDE + JOURNAL_EFFECT_OFFSET + 24] ^= 1
    with pytest.raises(ValueError, match="memory-read witness mismatch"):
        artifact.states_from_journal(bytes(tampered), initial)


def test_dynamic_guest_memory_address_fails_closed_before_wasm_execution():
    instruction = SimpleNamespace(
        address=0x6000, encoded=b"\x90",
        semantic=MachineSemanticToken.REGISTER_OR_MEMORY_READ,
        operands=(
            RegisterOperand(X86Register.RAX, 64),
            EffectiveAddressOperand(X86Register.RCX, None, 1, 0, 64, False),
        ),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0x6000, entrypoint_rva=0),
        functions=(SimpleNamespace(report=SimpleNamespace(instructions=(instruction,))),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    with pytest.raises(MachineBlockLoweringError, match="dynamic guest-memory"):
        executor.recompile_block_wasm(0x6000)

    executable_store = SimpleNamespace(
        address=0x7000, encoded=b"\x90",
        semantic=MachineSemanticToken.REGISTER_OR_MEMORY_WRITE,
        operands=(
            EffectiveAddressOperand(None, None, 1, 0x7010 - 0x7001, 64, True),
            RegisterOperand(X86Register.RAX, 64),
        ),
    )
    executable_program = SimpleNamespace(
        image=SimpleNamespace(image_base=0x7000, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(executable_store,)),
        ),),
    )
    executable_executor = MachineExecutionOrchestrator(
        executable_program, effect_handlers=default_effect_handlers(),
    )
    with pytest.raises(MachineBlockLoweringError, match="executable page"):
        executable_executor.recompile_block_wasm(0x7000)


def _relative_jump_program(*, condition=None):
    address = 0x8000
    target = 0x8010
    semantic = (
        MachineSemanticToken.DIRECT_RELATIVE_JUMP
        if condition is None else MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP
    )
    token = "JMP_REL8" if condition is None else f"J{condition}_REL8"
    instruction = SimpleNamespace(
        address=address, encoded=b"\xeb\x0e", semantic=semantic,
        token=SimpleNamespace(name=token),
        operands=(RelativeAddressOperand(target - address - 2, 8, target),),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=address, entrypoint_rva=0),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=(instruction,)),
        ),),
    )
    return program, instruction


def test_direct_relative_jump_wasm_commits_exact_control_edge(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for control-flow Wasm execution")
    program, _instruction = _relative_jump_program()
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    initial = MachineExecutionState(pc=0x8000, steps=3)
    expected = executor.step(initial).state
    artifact = executor.recompile_block_wasm(0x8000, strict=True)
    assert artifact.complete
    assert artifact.continuation_address == 0x8010
    assert artifact.possible_continuations == (0x8010,)
    journal, _guest = _run_artifact(node, tmp_path, "jump", artifact, initial)
    assert artifact.states_from_journal(journal, initial) == (expected,)
    runtime = ReversibleMachineExecutor.create(executor, initial)
    runtime.commit_recompiled_journal(artifact, journal)
    assert runtime.state == expected
    assert runtime.step_backward() == initial


def test_conditional_jump_wasm_matches_reference_predicates_and_rejects_bad_rip(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for control-flow Wasm execution")
    # CF/ZF/SF/OF patterns exercise every canonical predicate formula.
    cases = (
        ("E", 1 << 6), ("NE", 1 << 6),
        ("B", 1 << 0), ("AE", 1 << 0),
        ("BE", 1 << 6), ("A", 0),
        ("S", 1 << 7), ("NS", 1 << 7),
        ("O", 1 << 11), ("NO", 1 << 11),
        ("L", 1 << 7), ("GE", (1 << 7) | (1 << 11)),
        ("LE", 1 << 6), ("G", 0),
    )
    for index, (condition, flags) in enumerate(cases):
        program, instruction = _relative_jump_program(condition=condition)
        executor = MachineExecutionOrchestrator(
            program,
            effect_handlers=default_effect_handlers(),
            predicate_handler=condition_holds,
        )
        initial = MachineExecutionState(pc=0x8000, flags=flags, steps=index)
        expected = executor.step(initial).state
        artifact = executor.recompile_block_wasm(0x8000, strict=True)
        assert artifact.complete
        assert artifact.continuation_address == -1
        assert artifact.possible_continuations == (0x8010, 0x8002)
        journal, _guest = _run_artifact(
            node, tmp_path, f"j{condition.lower()}", artifact, initial,
        )
        (compiled,) = artifact.states_from_journal(journal, initial)
        assert compiled == expected
        assert compiled.pc == (
            0x8010 if condition_holds(initial, instruction) else 0x8002
        )

        runtime = ReversibleMachineExecutor.create(executor, initial)
        runtime.commit_recompiled_journal(artifact, journal)
        assert runtime.step_backward() == initial

        if index == 0:
            tampered = bytearray(journal)
            struct.pack_into(
                "<Q", tampered, JOURNAL_STATE_OFFSET + 16 * 8, 0xDEAD,
            )
            with pytest.raises(ValueError, match="impossible successor"):
                artifact.states_from_journal(bytes(tampered), initial)


def _direct_call_program():
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


def test_specialized_direct_call_and_return_wasm_are_exact_and_reversible(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for call/return Wasm execution")
    executor = MachineExecutionOrchestrator(
        _direct_call_program(), effect_handlers=default_effect_handlers(),
    )
    registers = [0] * 16
    registers[4] = 0xB008
    initial = MachineExecutionState(
        pc=0xA000, registers=tuple(registers),
        memory=PagedByteMemory.empty().map_zeroes(0xB000, 16), steps=10,
    )
    called = executor.step(initial).state
    call_artifact = executor.recompile_block_wasm(0xA000, initial, strict=True)
    assert call_artifact.possible_continuations == (0xA100,)
    assert "(local.set $stack_kind (i64.const 1))" in call_artifact.wat
    call_journal, call_guest = _run_artifact(
        node, tmp_path, "direct-call", call_artifact, initial,
    )
    assert call_artifact.states_from_journal(call_journal, initial) == (called,)
    assert int.from_bytes(call_guest[:8], "little") == 0xA005

    returned = executor.step(called).state
    return_artifact = executor.recompile_block_wasm(0xA100, called, strict=True)
    assert return_artifact.possible_continuations == (0xA005,)
    assert "(local.set $stack_kind (i64.const 2))" in return_artifact.wat
    return_journal, _return_guest = _run_artifact(
        node, tmp_path, "ordinary-return", return_artifact, called,
    )
    assert return_artifact.states_from_journal(return_journal, called) == (returned,)

    runtime = ReversibleMachineExecutor.create(executor, initial)
    runtime.commit_recompiled_journal(call_artifact, call_journal)
    runtime.commit_recompiled_journal(return_artifact, return_journal)
    assert runtime.state == returned
    assert runtime.step_backward() == called
    assert runtime.step_backward() == initial


def test_specialized_call_stack_witnesses_fail_closed():
    executor = MachineExecutionOrchestrator(
        _direct_call_program(), effect_handlers=default_effect_handlers(),
    )
    registers = [0] * 16
    registers[4] = 0xB008
    initial = MachineExecutionState(
        pc=0xA000, registers=tuple(registers),
        memory=PagedByteMemory.empty().map_zeroes(0xB000, 16),
    )
    with pytest.raises(MachineBlockLoweringError, match="specialization state"):
        executor.recompile_block_wasm(0xA000, strict=True)
    with pytest.raises(MachineBlockLoweringError, match="outermost return"):
        executor.recompile_block_wasm(
            0xA100, MachineExecutionState(
                pc=0xA100, registers=tuple(registers), memory=initial.memory,
            ), strict=True,
        )

    # A syntactically valid record cannot silently invent a different shadow
    # return address even if every architectural word still looks plausible.
    artifact = executor.recompile_block_wasm(0xA000, initial, strict=True)
    record = bytearray(JOURNAL_STRIDE)
    witness = artifact.witnesses[0]
    struct.pack_into(
        "<QQQ", record, 0, witness.address, witness.semantic_id,
        int(witness.encoded_digest[:16], 16),
    )
    values = (*initial.registers, 0xA100, initial.flags, initial.steps + 1)
    struct.pack_into("<19Q", record, JOURNAL_STATE_OFFSET, *values)
    struct.pack_into("<5Q", record, JOURNAL_EFFECT_OFFSET, 2, 0xB000, 64, 0, 0xA005)
    struct.pack_into("<3Q", record, JOURNAL_STACK_OFFSET, 1, 0xBAD, 1)
    with pytest.raises(ValueError, match="call-stack witness mismatch"):
        artifact.states_from_journal(bytes(record), initial)

    # Make the depth consistent; the independent shadow-stack provenance must
    # still reject the forged return address before it can become an edge.
    struct.pack_into("<3Q", record, JOURNAL_STACK_OFFSET, 1, 0xBAD, 0)
    with pytest.raises(ValueError, match="call-stack witness mismatch"):
        artifact.states_from_journal(bytes(record), initial)
