from types import SimpleNamespace

from src.compiler.amd64_machine_semantics import default_effect_handlers
from src.compiler.machine_execution import (
    MachineExecutionOrchestrator, MachineExecutionState,
    ReversibleMachineExecutor,
)
from src.compiler.machine_reference_vocabulary import (
    EffectiveAddressOperand,
    ImmediateOperand,
    MachineSemanticToken,
    RegisterOperand,
    RelativeAddressOperand,
    X86InstructionToken,
    X86ReferenceDecoder,
    X86Register,
)
from src.compiler.machine_symbolic_effects import (
    symbolic_effect_for_instruction, translated_block_to_symbolic_ssa,
    validate_symbolic_transition,
)


def _decode(encoded: bytes):
    instruction, end = X86ReferenceDecoder().decode_one(
        memoryview(encoded), 0, base_address=0x1000,
    )
    assert end == len(encoded)
    return instruction


def test_encoding_variants_share_semantic_effect_contracts():
    test_register = symbolic_effect_for_instruction(_decode(b"\x48\x85\xc0"))
    test_memory = symbolic_effect_for_instruction(_decode(b"\x48\x85\x01"))

    assert test_register.semantic is MachineSemanticToken.INTEGER_TEST
    assert test_register.reads == ("register.rax",)
    assert test_register.writes == ("flags", "control.rip")
    assert test_memory.reads == ("register.rcx", "memory", "register.rax")
    assert test_memory.writes == ("flags", "control.rip")


def test_memory_destination_contract_retains_address_data_and_state_effects():
    instruction = SimpleNamespace(
        semantic=MachineSemanticToken.BITWISE_XOR,
        token=X86InstructionToken.XOR_RM32_R32,
        operands=(
            EffectiveAddressOperand(X86Register.RCX, X86Register.RDX, 4, 8),
            RegisterOperand(X86Register.R8, 32),
        ),
        legacy_prefixes=(),
    )

    effect = symbolic_effect_for_instruction(instruction)

    assert effect.reads == (
        "register.rcx", "register.rdx", "memory", "register.r8",
    )
    assert effect.writes == ("memory", "flags", "control.rip")
    assert effect.effect_domains == ("control", "flags", "memory", "register")


def test_translated_vm_operation_carries_symbolic_effect_without_redirecting_execution():
    instruction = _decode(b"\x83\xe8\x01")  # sub eax, 1
    ret = SimpleNamespace(
        address=0x1003,
        encoded=b"\xc3",
        token=X86InstructionToken.RET_NEAR,
        semantic=MachineSemanticToken.RETURN,
        operands=(),
        legacy_prefixes=(),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0x1000, entrypoint_rva=0),
        functions=(SimpleNamespace(report=SimpleNamespace(
            instructions=(instruction, ret),
        )),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )

    translated = executor.translated_block(0x1000).operations[0]

    assert translated.execute is not None
    assert translated.symbolic_effect.semantic is MachineSemanticToken.INTEGER_SUBTRACT
    assert translated.symbolic_effect.reads == ("register.rax",)
    assert translated.symbolic_effect.writes == (
        "register.rax", "flags", "control.rip",
    )


def test_control_contract_exposes_call_stack_and_memory_state():
    instruction = SimpleNamespace(
        semantic=MachineSemanticToken.DIRECT_RELATIVE_CALL,
        token=X86InstructionToken.CALL_REL32,
        operands=(RelativeAddressOperand(8, 32, 0x2000),),
        legacy_prefixes=(),
    )

    effect = symbolic_effect_for_instruction(instruction)

    assert effect.reads == ("control.rip", "register.rsp", "memory")
    assert effect.writes == (
        "register.rsp", "memory", "control.call_stack", "control.rip",
    )
    assert effect.effect_domains == ("control", "memory", "register")


def test_translated_block_becomes_resource_versioned_machine_ssa():
    instructions = (
        _decode(b"\x83\xe8\x01"),  # sub eax, 1
        _decode(b"\x48\x85\xc0"),  # test rax, rax
    )
    block = SimpleNamespace(
        entry_address=0x1000,
        operations=tuple(SimpleNamespace(
            address=item.address,
            instruction=item,
            symbolic_effect=symbolic_effect_for_instruction(item),
        ) for item in instructions),
    )

    symbolic = translated_block_to_symbolic_ssa(block)

    assert [item.semantic for item in symbolic.operations] == [
        MachineSemanticToken.INTEGER_SUBTRACT,
        MachineSemanticToken.INTEGER_TEST,
    ]
    assert [item.identity for item in symbolic.operations[0].outputs] == [
        "register.rax@1", "flags@1", "control.rip@1",
    ]
    assert [item.identity for item in symbolic.operations[1].inputs] == [
        "register.rax@1", "flags@1", "control.rip@1",
    ]
    assert symbolic.final_values["flags"].identity == "flags@2"


def test_bidirectional_head_validates_declared_symbolic_writes():
    instruction = _decode(b"\x83\xe8\x01")
    ret = SimpleNamespace(
        address=0x1003, encoded=b"\xc3",
        token=X86InstructionToken.RET_NEAR,
        semantic=MachineSemanticToken.RETURN,
        operands=(), legacy_prefixes=(),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0x1000, entrypoint_rva=0),
        functions=(SimpleNamespace(report=SimpleNamespace(
            instructions=(instruction, ret),
        )),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    initial = MachineExecutionState(
        pc=0x1000, registers=(4, *(0 for _ in range(15))),
    )
    head = ReversibleMachineExecutor.create(executor, initial)

    result = head.step_forward()
    effect = executor.translated_block(0x1000).operations[0].symbolic_effect

    assert validate_symbolic_transition(effect, initial, result.state) == (
        "register.rax", "flags", "control.rip",
    )
    assert head.step_backward() == initial


def test_signed_multiply_contract_distinguishes_two_and_three_operand_forms():
    two_operand = symbolic_effect_for_instruction(
        _decode(b"\x4c\x0f\xaf\x40\x28")
    )
    three_operand = symbolic_effect_for_instruction(
        _decode(b"\x4c\x6b\x40\x28\x03")
    )

    assert two_operand.reads == (
        "register.r8", "register.rax", "memory",
    )
    assert three_operand.reads == (
        "register.rax", "memory",
    )
    assert two_operand.writes == (
        "register.r8", "flags", "control.rip",
    )
    assert three_operand.writes == two_operand.writes
