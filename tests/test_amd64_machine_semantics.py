from dataclasses import replace
from types import SimpleNamespace
import struct

import pytest

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
    MACHINE_TERMINATION_RETURN,
    MachineExecutionOrchestrator,
    MachineExecutionState,
    MachineExecutionStatus,
    MachineExternalCallRequest,
    MachineExternalReference,
    MachineExternalCallCompletion,
    MachineExternalMemoryWrite,
    MachineExternalRegisterWrite,
    MachineExternalDeviceWrite,
    ReversibleMachineExecutor,
)
from src.compiler.virtual_registry import VirtualRegistryEffect, VirtualRegistryState
from src.compiler.virtual_memory import (
    PAGE_EXECUTE_READWRITE, VirtualMemoryEffect, VirtualMemoryState,
)
from src.compiler.machine_reference_vocabulary import (
    DecodedInstruction,
    EffectiveAddressOperand,
    ImmediateOperand,
    MachineSemanticToken,
    RegisterOperand,
    X86InstructionToken,
    X86ReferenceDecoder,
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


@pytest.mark.parametrize(("source", "rounding", "expected"), (
    ((1 << 53) + 1, 0, 1 << 53),
    ((1 << 53) + 1, 1, 1 << 53),
    ((1 << 53) + 1, 2, (1 << 53) + 2),
    ((1 << 53) + 1, 3, 1 << 53),
    (-((1 << 53) + 1), 0, -(1 << 53)),
    (-((1 << 53) + 1), 1, -((1 << 53) + 2)),
    (-((1 << 53) + 1), 2, -(1 << 53)),
    (-((1 << 53) + 1), 3, -(1 << 53)),
))
def test_cvtsi2sd_uses_mxcsr_rounding_and_sets_precision_status(
    source, rounding, expected,
):
    instruction = _instruction(
        X86InstructionToken.CVTSI2SD_XMM_RM64,
        MachineSemanticToken.SIGNED_INTEGER_TO_SCALAR_FLOAT64,
        (
            VectorRegisterOperand(X86VectorRegister.XMM0),
            RegisterOperand(X86Register.RCX, 64),
        ),
    )
    registers = [0] * 16
    registers[1] = source & ((1 << 64) - 1)
    vectors = [0] * 16
    vectors[0] = 0xA5A5A5A5A5A5A5A5 << 64
    mxcsr = 0x1F80 | (rounding << 13)
    state = MachineExecutionState(
        pc=0, registers=tuple(registers), vector_registers=tuple(vectors),
        system_state={"amd64.mxcsr": mxcsr},
    )

    result = default_effect_handlers()[
        int(MachineSemanticToken.SIGNED_INTEGER_TO_SCALAR_FLOAT64)
    ](state, instruction)

    expected_bits = int.from_bytes(struct.pack("<d", float(expected)), "little")
    assert result.vector_registers[0] >> 64 == 0xA5A5A5A5A5A5A5A5
    assert result.vector_registers[0] & ((1 << 64) - 1) == expected_bits
    assert result.system_state["amd64.mxcsr"] & (1 << 5)


def test_cvtsi2sd_unmasked_precision_exception_traps_before_destination_write():
    instruction = _instruction(
        X86InstructionToken.CVTSI2SD_XMM_RM64,
        MachineSemanticToken.SIGNED_INTEGER_TO_SCALAR_FLOAT64,
        (
            VectorRegisterOperand(X86VectorRegister.XMM0),
            RegisterOperand(X86Register.RCX, 64),
        ),
    )
    registers = [0] * 16
    registers[1] = (1 << 53) + 1
    state = MachineExecutionState(
        pc=0, registers=tuple(registers),
        system_state={"amd64.mxcsr": 0x1F80 & ~(1 << 12)},
    )

    with pytest.raises(FloatingPointError, match="precision"):
        default_effect_handlers()[
            int(MachineSemanticToken.SIGNED_INTEGER_TO_SCALAR_FLOAT64)
        ](state, instruction)


@pytest.mark.parametrize(("source", "rounding", "expected"), (
    ((1 << 24) + 1, 0, 1 << 24),
    ((1 << 24) + 1, 1, 1 << 24),
    ((1 << 24) + 1, 2, (1 << 24) + 2),
    (-((1 << 24) + 1), 1, -((1 << 24) + 2)),
))
def test_cvtsi2ss_uses_integer_only_mxcsr_rounding_and_preserves_upper_bits(
    source, rounding, expected,
):
    instruction = _instruction(
        X86InstructionToken.CVTSI2SS_XMM_RM64,
        MachineSemanticToken.SIGNED_INTEGER_TO_SCALAR_FLOAT32,
        (
            VectorRegisterOperand(X86VectorRegister.XMM0),
            RegisterOperand(X86Register.RCX, 64),
        ),
    )
    registers = [0] * 16
    registers[1] = source & ((1 << 64) - 1)
    vectors = [0] * 16
    vectors[0] = 0xA5A5A5A5A5A5A5A5A5A5A5A5 << 32
    state = MachineExecutionState(
        pc=0, registers=tuple(registers), vector_registers=tuple(vectors),
        system_state={"amd64.mxcsr": 0x1F80 | (rounding << 13)},
    )
    result = default_effect_handlers()[
        int(MachineSemanticToken.SIGNED_INTEGER_TO_SCALAR_FLOAT32)
    ](state, instruction)
    expected_bits = int.from_bytes(struct.pack("<f", float(expected)), "little")
    assert result.vector_registers[0] >> 32 == vectors[0] >> 32
    assert result.vector_registers[0] & 0xFFFFFFFF == expected_bits
    assert result.system_state["amd64.mxcsr"] & (1 << 5)


def test_unsigned_multiply_writes_implicit_rdx_rax_pair():
    instruction = _instruction(
        X86InstructionToken.MUL_RM64,
        MachineSemanticToken.INTEGER_MULTIPLY_UNSIGNED,
        (RegisterOperand(X86Register.RCX, 64),),
    )
    registers = [0] * 16
    registers[0] = 1 << 63
    registers[1] = 3
    result = default_effect_handlers()[int(MachineSemanticToken.INTEGER_MULTIPLY_UNSIGNED)](
        MachineExecutionState(pc=0, registers=tuple(registers)), instruction,
    )
    assert result.registers[0] == 1 << 63
    assert result.registers[2] == 1
    assert result.flags & 1
    assert result.flags & (1 << 11)


def test_accumulator_sign_extension_supports_cdqe_and_cqo():
    handlers = default_effect_handlers()
    cdqe = _instruction(
        X86InstructionToken.CDQE,
        MachineSemanticToken.SIGN_EXTEND_ACCUMULATOR,
        (
            RegisterOperand(X86Register.RAX, 64),
            RegisterOperand(X86Register.RAX, 32),
        ),
    )
    cqo = _instruction(
        X86InstructionToken.CQO,
        MachineSemanticToken.SIGN_EXTEND_ACCUMULATOR,
        (
            RegisterOperand(X86Register.RDX, 64),
            RegisterOperand(X86Register.RAX, 64),
        ),
    )
    state = MachineExecutionState(
        pc=0, registers=(0xAAAAAAAA80000001, 0, 7) + (0,) * 13,
    )
    extended = handlers[int(MachineSemanticToken.SIGN_EXTEND_ACCUMULATOR)](
        state, cdqe,
    )
    signed_pair = handlers[int(MachineSemanticToken.SIGN_EXTEND_ACCUMULATOR)](
        extended, cqo,
    )
    assert extended.registers[0] == 0xFFFFFFFF80000001
    assert signed_pair.registers[2] == 0xFFFFFFFFFFFFFFFF


def test_signed_divide_uses_rdx_rax_and_truncates_toward_zero():
    instruction = _instruction(
        X86InstructionToken.IDIV_RM64,
        MachineSemanticToken.INTEGER_DIVIDE_SIGNED,
        (RegisterOperand(X86Register.RCX, 64),),
    )
    registers = [0] * 16
    registers[0] = (-17) & ((1 << 64) - 1)
    registers[2] = (1 << 64) - 1
    registers[1] = 5
    result = default_effect_handlers()[int(MachineSemanticToken.INTEGER_DIVIDE_SIGNED)](
        MachineExecutionState(pc=0, registers=tuple(registers)), instruction,
    )
    assert result.registers[0] == ((-3) & ((1 << 64) - 1))
    assert result.registers[2] == ((-2) & ((1 << 64) - 1))


def test_unsigned_divide_rejects_architectural_quotient_overflow():
    instruction = _instruction(
        X86InstructionToken.DIV_RM64,
        MachineSemanticToken.INTEGER_DIVIDE,
        (RegisterOperand(X86Register.RCX, 64),),
    )
    registers = [0] * 16
    registers[2] = 1
    registers[1] = 1
    handler = default_effect_handlers()[int(MachineSemanticToken.INTEGER_DIVIDE)]
    try:
        handler(MachineExecutionState(pc=0, registers=tuple(registers)), instruction)
    except OverflowError as error:
        assert "quotient overflow" in str(error)
    else:
        raise AssertionError("DIV quotient overflow must fail closed")


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
        MachineExecutionState(
            pc=0x1000, registers=tuple(registers), memory=memory,
            virtual_registry=VirtualRegistryState.create(),
            virtual_memory=VirtualMemoryState.create(),
        ),
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
            registry_effects=(VirtualRegistryEffect(
                "create_key", "hkey_current_user\\Software\\Turing",
            ),),
            virtual_memory_effects=(VirtualMemoryEffect(
                "allocate", 0x10000000000, 4096, PAGE_EXECUTE_READWRITE,
            ),),
        ),
    )
    core.commit_external_completion(completed)
    assert core.state.pc == 0x1002
    assert core.state.registers[0] == 73
    assert core.state.registers[4] == 0x8000
    assert bytes((core.state.memory[0x7000], core.state.memory[0x7001])) == b"ok"
    assert "hkey_current_user\\software\\turing" in core.state.virtual_registry.keys
    assert core.state.memory[0x10000000000] == 0
    assert core.state.virtual_memory.is_executable(0x10000000000)
    assert core.step_backward() == waiting.state
    assert "hkey_current_user\\software\\turing" not in core.state.virtual_registry.keys
    assert core.state.virtual_memory.regions == {}


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
        MachineExternalCallCompletion(
            7,
            register_writes=(
                MachineExternalRegisterWrite(1, 1),
                MachineExternalRegisterWrite(2, 0x7FFEFFF0),
            ),
            guest_calls=(0x3000, 0x4000, 0x5000),
        ),
    )

    assert result.pc == 0x3000
    assert result.registers[1:3] == (1, 0x7FFEFFF0)
    assert result.call_stack == (0x2000, 0x5000, 0x4000)
    assert result.memory.read_unsigned(result.registers[4], 64) == 0x4000
    assert result.memory.read_unsigned(result.registers[4] + 8, 64) == 0x5000
    assert result.memory.read_unsigned(result.registers[4] + 16, 64) == 0x2000


def test_terminating_external_completion_replaces_return_with_halt_sentinel():
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x1000)
    memory = memory.write_unsigned(0x7FF8, 64, 0x2000)
    registers = [0] * 16
    registers[4] = 0x7FF8
    reference = MachineExternalReference(1, 0, "guest-binary", "msvcrt.dll", "exit")
    request = MachineExternalCallRequest(
        8, reference, 0x1000, 0x2000, (23, 0, 0, 0), 0x7FF8,
    )
    state = MachineExecutionState(
        pc=0, registers=tuple(registers), memory=memory,
        call_stack=(0x2000,), external_requests=(request,),
    )
    result = complete_external_call_state(
        state,
        MachineExternalCallCompletion(
            8, guest_calls=(0x3000, 0x4000), terminate=True, exit_code=23,
        ),
    )
    assert result.pc == 0x3000
    assert result.call_stack == (MACHINE_TERMINATION_RETURN, 0x4000)
    assert result.memory.read_unsigned(result.registers[4] + 8, 64) == MACHINE_TERMINATION_RETURN
    assert result.termination_requested and result.exit_code == 23

    ret = _instruction(
        X86InstructionToken.RET_NEAR, MachineSemanticToken.RETURN, (),
        encoded=b"\xc3", address=0x3000,
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0x3000, entrypoint_rva=0),
        functions=(SimpleNamespace(report=SimpleNamespace(instructions=(ret,))),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    halted = executor.step(result)
    assert halted.status is MachineExecutionStatus.RUNNING
    halted = executor.step(replace(
        halted.state, pc=0x3000,
    ))
    assert halted.status is MachineExecutionStatus.HALTED
    assert halted.state.halted and halted.state.exit_code == 23


def test_rep_string_store_move_and_scan_obey_count_and_direction_flag():
    memory = PagedByteMemory.empty().map_zeroes(0x1000, 0x2000)
    registers = [0] * 16
    registers[0], registers[1], registers[7] = 0xBEEF, 3, 0x1800
    state = MachineExecutionState(pc=0, registers=tuple(registers), memory=memory)
    handlers = default_effect_handlers()
    stored = handlers[int(MachineSemanticToken.STRING_STORE)](
        state, _instruction(X86InstructionToken.REP_STOSW, MachineSemanticToken.STRING_STORE, ()),
    )
    assert stored.memory.read_unsigned(0x1800, 16) == 0xBEEF
    assert stored.memory.read_unsigned(0x1804, 16) == 0xBEEF
    assert stored.registers[1] == 0 and stored.registers[7] == 0x1806

    memory = stored.memory.write_unsigned(0x1900, 64, 0x1122334455667788)
    registers = list(stored.registers)
    registers[1], registers[6], registers[7] = 1, 0x1900, 0x1A00
    moved = handlers[int(MachineSemanticToken.STRING_MOVE)](
        MachineExecutionState(pc=0, registers=tuple(registers), memory=memory),
        _instruction(X86InstructionToken.REP_MOVSQ, MachineSemanticToken.STRING_MOVE, ()),
    )
    assert moved.memory.read_unsigned(0x1A00, 64) == 0x1122334455667788
    assert moved.registers[6:8] == (0x1908, 0x1A08)

    registers = list(moved.registers)
    registers[0], registers[7] = 0x88, 0x1A00
    scanned = handlers[int(MachineSemanticToken.STRING_COMPARE)](
        MachineExecutionState(pc=0, registers=tuple(registers), memory=moved.memory),
        _instruction(X86InstructionToken.SCASB, MachineSemanticToken.STRING_COMPARE, ()),
    )
    assert scanned.flags & (1 << 6)
    assert scanned.registers[7] == 0x1A01


def test_bit_test_family_updates_carry_and_cross_word_memory_location():
    handlers = default_effect_handlers()
    registers = [0] * 16
    registers[0], registers[1] = 0b1000, 3
    register_state = MachineExecutionState(pc=0, registers=tuple(registers))
    tested = handlers[int(MachineSemanticToken.BIT_TEST)](
        register_state,
        _instruction(
            X86InstructionToken.BT_RM64_R64, MachineSemanticToken.BIT_TEST,
            (RegisterOperand(X86Register.RAX, 64), RegisterOperand(X86Register.RCX, 64)),
        ),
    )
    assert tested.flags & 1
    assert tested.registers[0] == 0b1000

    memory = PagedByteMemory.empty().map_zeroes(0x1000, 0x1000)
    memory = memory.write_unsigned(0x1804, 32, 0b1000)
    registers = [0] * 16
    registers[0], registers[7] = 35, 0x1800
    changed = handlers[int(MachineSemanticToken.BIT_TEST)](
        MachineExecutionState(pc=0, registers=tuple(registers), memory=memory),
        _instruction(
            X86InstructionToken.BTS_RM32_R32, MachineSemanticToken.BIT_TEST,
            (
                EffectiveAddressOperand(X86Register.RDI, None, 1, 0),
                RegisterOperand(X86Register.RAX, 32),
            ),
        ),
    )
    assert changed.flags & 1
    assert changed.memory.read_unsigned(0x1804, 32) == 0b1000


def test_btc_memory_complements_selected_bit_and_preserves_other_flags():
    decoder = X86ReferenceDecoder()
    instruction, end = decoder.decode_one(
        memoryview(b"\x0f\xba\x39\x05"), 0, base_address=0x1000,
    )
    memory = PagedByteMemory.empty().map_zeroes(0x2000, 8)
    memory = memory.write_unsigned(0x2000, 32, 0b100000)
    registers = [0] * 16
    registers[1] = 0x2000
    original_flags = (1 << 6) | (1 << 11)
    changed = default_effect_handlers()[int(instruction.semantic)](
        MachineExecutionState(
            pc=instruction.address, registers=tuple(registers),
            memory=memory, flags=original_flags,
        ),
        instruction,
    )

    assert end == 4
    assert changed.memory.read_unsigned(0x2000, 32) == 0
    assert changed.flags & 1
    assert changed.flags & ~1 == original_flags


def test_locked_xadd_memory_returns_observed_value_and_sets_add_flags():
    decoder = X86ReferenceDecoder()
    instruction, end = decoder.decode_one(
        memoryview(b"\xf0\x0f\xc1\x11"), 0, base_address=0x1000,
    )
    memory = PagedByteMemory.empty().map_zeroes(0x2000, 8)
    memory = memory.write_unsigned(0x2000, 32, 0xFFFFFFFF)
    registers = [0] * 16
    registers[1], registers[2] = 0x2000, 1
    changed = default_effect_handlers()[int(instruction.semantic)](
        MachineExecutionState(
            pc=instruction.address, registers=tuple(registers), memory=memory,
        ),
        instruction,
    )

    assert end == 4
    assert changed.memory.read_unsigned(0x2000, 32) == 0
    assert changed.registers[2] == 0xFFFFFFFF
    assert changed.flags & 1  # CF
    assert changed.flags & (1 << 6)  # ZF


def test_external_device_effect_is_part_of_reversible_machine_state():
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x1000)
    memory = memory.write_unsigned(0x7FF8, 64, 0x2000)
    registers = [0] * 16
    registers[4] = 0x7FF8
    request = MachineExternalCallRequest(
        53, MachineExternalReference(47, 0, "guest-binary", "kernel32.dll", "WriteConsoleW"),
        0x1000, 0x2000, (0, 0, 0, 0), 0x7FF8,
    )
    state = MachineExecutionState(
        pc=0, registers=tuple(registers), memory=memory,
        call_stack=(0x2000,), external_requests=(request,),
        device_state={"console.output": b"before "},
        device_generations={"console.output": 2},
    )
    completed = complete_external_call_state(
        state, MachineExternalCallCompletion(
            53, device_writes=(MachineExternalDeviceWrite("console.output", b"after"),),
        ),
    )
    assert completed.device_state["console.output"] == b"before after"
    assert completed.device_generations["console.output"] == 3
    assert state.device_state["console.output"] == b"before "


def test_byte_not_decodes_and_executes_without_widening_its_operand():
    instruction, end = X86ReferenceDecoder().decode_one(
        memoryview(b"\xf6\xd0"), 0, base_address=0x140018590,
    )
    registers = [0] * 16
    registers[0] = 0x1122334455667788
    changed = default_effect_handlers()[int(instruction.semantic)](
        MachineExecutionState(pc=instruction.address, registers=tuple(registers)),
        instruction,
    )

    assert end == 2
    assert instruction.token is X86InstructionToken.NOT_RM8
    assert changed.registers[0] == 0x1122334455667777


def test_addss_executes_low_binary32_lane_and_preserves_upper_lanes():
    instruction, end = X86ReferenceDecoder().decode_one(
        memoryview(b"\xf3\x0f\x58\xc1"), 0, base_address=0x140018600,
    )
    upper = 0xAABBCCDDEEFF001122334455
    left = int.from_bytes(struct.pack("<f", 1.25), "little")
    right = int.from_bytes(struct.pack("<f", 2.5), "little")
    vectors = [0] * 16
    vectors[0] = (upper << 32) | left
    vectors[1] = right
    changed = default_effect_handlers()[int(instruction.semantic)](
        MachineExecutionState(pc=instruction.address, vector_registers=tuple(vectors)),
        instruction,
    )

    assert end == 4
    assert instruction.token is X86InstructionToken.ADDSS_XMM_XMMM32
    expected = int.from_bytes(struct.pack("<f", 3.75), "little")
    assert changed.vector_registers[0] == (upper << 32) | expected


def test_vinsertf128_decodes_and_inserts_selected_ymm_lane_without_host_avx():
    encoded = bytes.fromhex("c4 e3 7d 18 c0 01")
    instruction, end = X86ReferenceDecoder().decode_one(
        memoryview(encoded), 0, base_address=0x180011822,
    )
    low = 0x00112233445566778899AABBCCDDEEFF
    high = 0xFFEEDDCCBBAA99887766554433221100
    vectors = [0] * 16
    vectors[0] = low | (high << 128)
    changed = default_effect_handlers()[int(instruction.semantic)](
        MachineExecutionState(
            pc=instruction.address, vector_registers=tuple(vectors),
        ),
        instruction,
    )

    assert end == len(encoded)
    assert instruction.token is X86InstructionToken.VINSERTF128_YMM_YMM_XMMM128_IMM8
    assert instruction.semantic is MachineSemanticToken.VECTOR_INSERT_128_LANE
    assert tuple(getattr(item, "width", None) for item in instruction.operands) == (
        256, 256, 128, 8,
    )
    # Source YMM0's low XMM0 lane is copied into its upper lane.
    assert changed.vector_registers[0] == low | (low << 128)


def test_divss_executes_low_binary32_lane_and_records_masked_zero_division():
    instruction, _ = X86ReferenceDecoder().decode_one(
        memoryview(b"\xf3\x0f\x5e\xc1"), 0, base_address=0x140018610,
    )
    upper = 0x112233445566778899AABBCC
    vectors = [0] * 16
    vectors[0] = (upper << 32) | int.from_bytes(struct.pack("<f", 4.0), "little")
    vectors[1] = int.from_bytes(struct.pack("<f", 0.0), "little")
    changed = default_effect_handlers()[int(instruction.semantic)](
        MachineExecutionState(pc=instruction.address, vector_registers=tuple(vectors)),
        instruction,
    )

    assert instruction.token is X86InstructionToken.DIVSS_XMM_XMMM32
    assert changed.vector_registers[0] == (upper << 32) | 0x7F800000
    assert changed.system_state["amd64.mxcsr"] & (1 << 2)


def test_comiss_executes_ordered_binary32_comparison_flags():
    instruction, _ = X86ReferenceDecoder().decode_one(
        memoryview(b"\x0f\x2f\xc1"), 0, base_address=0x140018620,
    )
    vectors = [0] * 16
    vectors[0] = int.from_bytes(struct.pack("<f", 1.0), "little")
    vectors[1] = int.from_bytes(struct.pack("<f", 2.0), "little")
    changed = default_effect_handlers()[int(instruction.semantic)](
        MachineExecutionState(pc=instruction.address, vector_registers=tuple(vectors)),
        instruction,
    )

    assert instruction.token is X86InstructionToken.COMISS_XMM_XMMM32
    assert changed.flags & 1
    assert not (changed.flags & (1 << 2))
    assert not (changed.flags & (1 << 6))
