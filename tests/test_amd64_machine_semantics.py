from types import SimpleNamespace

from src.compiler.amd64_machine_semantics import (
    PagedByteMemory,
    build_initial_machine_state,
    condition_holds,
    complete_external_call_state,
    default_effect_handlers,
    indirect_target,
    read_operand,
)
from src.compiler.machine_execution import (
    MachineExecutionOrchestrator,
    MachineExecutionState,
    MachineExecutionStatus,
    MachineExternalReference,
    ReversibleMachineExecutor,
)
from src.compiler.machine_reference_vocabulary import (
    DecodedInstruction,
    EffectiveAddressOperand,
    ImmediateOperand,
    MachineSemanticToken,
    RegisterOperand,
    X86InstructionToken,
    X86Register,
)


def _instruction(token, semantic, operands, encoded=b"\x90", address=0x1000):
    return DecodedInstruction(address, token, semantic, tuple(operands), encoded)


def test_paged_memory_is_little_endian_copy_on_write_and_fail_closed():
    original = PagedByteMemory.empty().map_zeroes(0x1000, 0x1000)
    changed = original.write_unsigned(0x1FFC, 64, 0x8877665544332211)

    assert original.read_unsigned(0x1FFC, 32) == 0
    assert changed.read_unsigned(0x1FFC, 64) == 0x8877665544332211
    assert changed[0x1FFC] == 0x11
    try:
        changed[0x3000]
    except KeyError as error:
        assert "unmapped guest address" in str(error)
    else:
        raise AssertionError("unmapped reads must fail closed")


def test_sub_rsp_updates_real_register_and_integer_flags():
    instruction = _instruction(
        X86InstructionToken.SUB_R64_IMM8,
        MachineSemanticToken.INTEGER_SUBTRACT,
        (RegisterOperand(X86Register.RSP, 64), ImmediateOperand(40, 8, True)),
        b"\x48\x83\xec\x28",
    )
    state = MachineExecutionState(pc=instruction.address + 4, registers=(0, 0, 0, 0, 0x1000) + (0,) * 11)
    handler = default_effect_handlers()[int(MachineSemanticToken.INTEGER_SUBTRACT)]

    result = handler(state, instruction)

    assert result.registers[4] == 0xFD8
    assert not (result.flags & (1 << 6))  # ZF
    assert not (result.flags & 1)  # CF


def test_32_bit_register_write_clears_upper_half_and_rip_relative_read_works():
    memory = PagedByteMemory.empty().map_zeroes(0x2000, 0x1000)
    memory = memory.write_unsigned(0x2010, 32, 0xDEADBEEF)
    state = MachineExecutionState(
        pc=0x2004,
        registers=(0xFFFFFFFFFFFFFFFF,) + (0,) * 15,
        memory=memory,
    )
    instruction = _instruction(
        X86InstructionToken.MOV_R32_RM32,
        MachineSemanticToken.REGISTER_OR_MEMORY_READ,
        (
            RegisterOperand(X86Register.RAX, 32),
            EffectiveAddressOperand(None, None, 1, 12, rip_relative=True),
        ),
        b"\x8b\x05\x0c\x00\x00\x00",
        address=0x1FFE,
    )
    handler = default_effect_handlers()[int(MachineSemanticToken.REGISTER_OR_MEMORY_READ)]

    result = handler(state, instruction)

    assert result.registers[0] == 0xDEADBEEF


def test_conditions_consume_flags_by_instruction_condition_name():
    jne = _instruction(
        X86InstructionToken.JNE_REL32,
        MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        (),
    )
    assert condition_holds(MachineExecutionState(pc=0), jne)
    assert not condition_holds(MachineExecutionState(pc=0, flags=1 << 6), jne)


def test_initial_pe_mapping_has_headers_sections_and_windows_entry_stack():
    section = SimpleNamespace(
        raw_offset=0x200,
        raw_size=4,
        virtual_address=0x1000,
        virtual_size=0x20,
    )
    encoded = bytearray(0x204)
    encoded[:2] = b"MZ"
    encoded[0x200:] = b"CODE"
    program = SimpleNamespace(image=SimpleNamespace(
        image_base=0x140000000,
        entrypoint_rva=0x1000,
        sections=(section,),
        encoded=bytes(encoded),
    ))

    state = build_initial_machine_state(program, stack_top=0x800000, stack_size=0x2000)

    assert bytes(state.memory[0x140000000 + i] for i in range(2)) == b"MZ"
    assert bytes(state.memory[0x140001000 + i] for i in range(4)) == b"CODE"
    assert state.registers[4] == 0x7FFFF8
    assert read_operand(
        state,
        _instruction(
            X86InstructionToken.MOV_R64_RM64,
            MachineSemanticToken.REGISTER_OR_MEMORY_READ,
            (RegisterOperand(X86Register.RAX, 64), EffectiveAddressOperand(X86Register.RSP, None, 1, 0)),
        ),
        1,
        width=64,
    ) == 0


def test_external_import_call_pauses_as_request_and_completion_is_reversible():
    target = 0xFFFF800000000000
    instruction = _instruction(
        X86InstructionToken.CALL_RM64,
        MachineSemanticToken.INDIRECT_CALL,
        (RegisterOperand(X86Register.RAX, 64),),
        b"\xff\xd0",
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0x1000, entrypoint_rva=0),
        functions=(SimpleNamespace(report=SimpleNamespace(instructions=(instruction,))),),
    )
    reference = MachineExternalReference(1, target, "guest-binary", "kernel32.dll", "Example")
    executor = MachineExecutionOrchestrator(
        program,
        effect_handlers=default_effect_handlers(),
        indirect_target_handler=indirect_target,
        external_target_resolver=lambda address: reference if address == target else None,
    )
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x1000)
    registers = [0] * 16
    registers[0] = target
    registers[1] = 11
    registers[2] = 22
    registers[4] = 0x8000
    core = ReversibleMachineExecutor.create(
        executor,
        MachineExecutionState(pc=0x1000, registers=tuple(registers), memory=memory),
    )

    waiting = core.step_forward()

    assert waiting.status is MachineExecutionStatus.WAITING_EXTERNAL
    request = waiting.state.external_requests[0]
    assert request.reference.symbol == "Example"
    assert request.arguments[:2] == (11, 22)
    assert waiting.state.memory.read_unsigned(0x7FF8, 64) == 0x1002

    completed = complete_external_call_state(waiting.state, request.request_id, result=73)
    core.commit_external_completion(completed)
    assert core.state.pc == 0x1002
    assert core.state.registers[0] == 73
    assert core.state.registers[4] == 0x8000
    assert core.step_backward() == waiting.state
