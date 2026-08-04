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
    MachineExternalCallRequest,
    MachineExternalReference,
    MachineExternalCallCompletion,
    MachineExternalMemoryWrite,
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
    VectorRegisterOperand,
    X86VectorRegister,
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


def test_gs_override_uses_visible_segment_base_not_literal_low_memory():
    memory = PagedByteMemory.empty().map_zeroes(0x9000, 0x1000)
    memory = memory.write_unsigned(0x9030, 64, 0xCAFEBABE)
    state = MachineExecutionState(pc=0, memory=memory, gs_base=0x9000)
    instruction = _instruction(
        X86InstructionToken.MOV_R64_RM64_FS,
        MachineSemanticToken.REGISTER_OR_MEMORY_READ,
        (RegisterOperand(X86Register.RAX, 64), EffectiveAddressOperand(None, None, 1, 0x30)),
    )
    instruction = DecodedInstruction(
        instruction.address, instruction.token, instruction.semantic,
        instruction.operands, instruction.encoded, legacy_prefixes=(0x65,),
    )
    handler = default_effect_handlers()[int(MachineSemanticToken.REGISTER_OR_MEMORY_READ)]

    assert handler(state, instruction).registers[0] == 0xCAFEBABE


def test_movsxd_uses_memory_source_width_not_destination_width_from_token_name():
    memory = PagedByteMemory.empty().map_zeroes(0x9000, 0x1000)
    memory = memory.write_unsigned(0x9010, 64, 0xAABBCCDDFFFFFFF0)
    state = MachineExecutionState(pc=0, memory=memory)
    instruction = _instruction(
        X86InstructionToken.MOVSXD_R64_RM32,
        MachineSemanticToken.SIGN_EXTEND,
        (RegisterOperand(X86Register.RCX, 64), EffectiveAddressOperand(None, None, 1, 0x9010)),
    )
    result = default_effect_handlers()[int(MachineSemanticToken.SIGN_EXTEND)](state, instruction)
    assert result.registers[1] == 0xFFFFFFFFFFFFFFF0


def test_conditions_consume_flags_by_instruction_condition_name():
    jne = _instruction(
        X86InstructionToken.JNE_REL32,
        MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        (),
    )
    assert condition_holds(MachineExecutionState(pc=0), jne)
    assert not condition_holds(MachineExecutionState(pc=0, flags=1 << 6), jne)


def test_shift_family_writes_value_carry_and_zero_flags():
    instruction = _instruction(
        X86InstructionToken.SHL_R64_IMM8,
        MachineSemanticToken.SHIFT_LEFT,
        (RegisterOperand(X86Register.RAX, 64), ImmediateOperand(1, 8, False)),
    )
    state = MachineExecutionState(pc=0, registers=((1 << 63),) + (0,) * 15)
    result = default_effect_handlers()[int(MachineSemanticToken.SHIFT_LEFT)](state, instruction)

    assert result.registers[0] == 0
    assert result.flags & 1  # CF
    assert result.flags & (1 << 6)  # ZF
    assert result.flags & (1 << 11)  # OF


def test_compare_exchange_updates_destination_or_accumulator_atomically():
    instruction = _instruction(
        X86InstructionToken.CMPXCHG_RM64_R64,
        MachineSemanticToken.ATOMIC_COMPARE_EXCHANGE,
        (RegisterOperand(X86Register.RCX, 64), RegisterOperand(X86Register.RBX, 64)),
    )
    handler = default_effect_handlers()[int(MachineSemanticToken.ATOMIC_COMPARE_EXCHANGE)]
    registers = [0] * 16
    registers[0], registers[1], registers[3] = 5, 5, 9
    equal = handler(MachineExecutionState(pc=0, registers=tuple(registers)), instruction)
    assert equal.registers[1] == 9
    assert equal.registers[0] == 5
    assert equal.flags & (1 << 6)

    registers[0], registers[1] = 4, 5
    unequal = handler(MachineExecutionState(pc=0, registers=tuple(registers)), instruction)
    assert unequal.registers[1] == 5
    assert unequal.registers[0] == 5
    assert not (unequal.flags & (1 << 6))


def test_subtract_with_borrow_consumes_and_recomputes_carry():
    instruction = _instruction(
        X86InstructionToken.SBB_R32_RM32,
        MachineSemanticToken.INTEGER_SUBTRACT_WITH_BORROW,
        (RegisterOperand(X86Register.RAX, 32), RegisterOperand(X86Register.RCX, 32)),
    )
    registers = [0] * 16
    registers[0], registers[1] = 5, 4
    handler = default_effect_handlers()[int(MachineSemanticToken.INTEGER_SUBTRACT_WITH_BORROW)]

    result = handler(
        MachineExecutionState(pc=0, registers=tuple(registers), flags=1),
        instruction,
    )

    assert result.registers[0] == 0
    assert result.flags & (1 << 6)
    assert not (result.flags & 1)


def test_vector_xor_updates_complete_xmm_register_without_integer_flags():
    instruction = _instruction(
        X86InstructionToken.XORPS_XMM_XMMM128,
        MachineSemanticToken.VECTOR_XOR,
        (
            VectorRegisterOperand(X86VectorRegister.XMM0),
            VectorRegisterOperand(X86VectorRegister.XMM1),
        ),
    )
    vectors = [0] * 16
    vectors[0] = (1 << 127) | 0x55
    vectors[1] = (1 << 127) | 0xAA
    state = MachineExecutionState(pc=0, vector_registers=tuple(vectors), flags=0x202)
    result = default_effect_handlers()[int(MachineSemanticToken.VECTOR_XOR)](state, instruction)
    assert result.vector_registers[0] == 0xFF
    assert result.flags == 0x202


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
    assert state.memory.read_unsigned(state.gs_base + 0x30, 64) == state.gs_base
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

    completed = complete_external_call_state(
        waiting.state,
        MachineExternalCallCompletion(
            request.request_id,
            result=73,
            memory_writes=(MachineExternalMemoryWrite(0x7000, b"ok"),),
        ),
    )
    core.commit_external_completion(completed)
    assert core.state.pc == 0x1002
    assert core.state.registers[0] == 73
    assert core.state.registers[4] == 0x8000
    assert bytes((core.state.memory[0x7000], core.state.memory[0x7001])) == b"ok"
    assert core.step_backward() == waiting.state


def test_external_completion_can_dispatch_ordered_guest_callbacks_before_return():
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x1000)
    memory = memory.write_unsigned(0x7FF8, 64, 0x2000)
    registers = [0] * 16
    registers[4] = 0x7FF8
    reference = MachineExternalReference(1, 0, "guest-binary", "msvcrt.dll", "_initterm")
    request = MachineExternalCallRequest(
        7, reference, 0x1000, 0x2000, (0, 0, 0, 0), 0x7FF8,
    )
    state = MachineExecutionState(
        pc=0,
        registers=tuple(registers),
        memory=memory,
        call_stack=(0x2000,),
        external_requests=(request,),
    )

    result = complete_external_call_state(
        state,
        MachineExternalCallCompletion(7, guest_calls=(0x3000, 0x4000, 0x5000)),
    )

    assert result.pc == 0x3000
    assert result.call_stack == (0x2000, 0x5000, 0x4000)
    assert result.memory.read_unsigned(result.registers[4], 64) == 0x4000
    assert result.memory.read_unsigned(result.registers[4] + 8, 64) == 0x5000
    assert result.memory.read_unsigned(result.registers[4] + 16, 64) == 0x2000
