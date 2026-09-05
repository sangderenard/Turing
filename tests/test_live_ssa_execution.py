from src.compiler.live_ssa_execution import (
    LiveSSAExecutionSession, address_linked_ssa_lines,
    SSAEditTransaction, replace_ssa_operations, ssa_viewport,
)
from src.compiler.machine_dialect_ssa import decoded_function_to_machine_ssa
from src.compiler.machine_execution import MachineExecutionState
from src.compiler.machine_reference_vocabulary import (
    ImmediateOperand, RegisterOperand, X86InstructionToken,
    X86ReferenceDecoder, X86Register,
)
from src.compiler.machine_code_lifting import raise_binary_region_to_ssa
from src.compiler.machine_ssa_execution import machine_ssa_program
from src.compiler.x86_tensor_read_head import X86AllocatedInstruction
from src.transmogrifier.ssa import IRModule


class _Image:
    image_base = 0x1000
    entrypoint_rva = 0
    encoded = None
    sections = ()


def _session(encoded=b"\x48\xff\xc0\x48\x83\xc0\x02\xc3", **viewport):
    decoded = X86ReferenceDecoder().decode_report(
        encoded, base_address=0x1000,
        stop_at_return=False, allow_trailing_after_terminal=True,
    ).instructions
    module = IRModule({
        "subject": decoded_function_to_machine_ssa("subject", decoded),
    })
    program = machine_ssa_program(module, image=_Image())
    state = MachineExecutionState(pc=program.entry_address("subject"))
    return module, LiveSSAExecutionSession(
        module, program.executor(), state, **viewport,
    )


def test_live_step_highlights_all_ssa_at_current_machine_address():
    module, session = _session(before=1, after=1)

    event = session.step()

    assert event.machine_address == 0x1000
    assert event.instruction_token == "INC_RM64"
    assert event.encoded == b"\x48\xff\xc0"
    assert event.changed_registers["rax"] == (0, 1)
    assert event.changed_registers["rip"] == (0x1000, 0x1003)
    assert event.highlighted_line_ids
    assert all(
        line.machine_address == 0x1000
        for line in address_linked_ssa_lines(module)
        if line.line_id in event.highlighted_line_ids
    )
    assert 1 <= len(event.lines) <= len(address_linked_ssa_lines(module))


def test_viewport_search_preserves_highlight_identity_outside_text_match():
    module, _session_value = _session()
    lines = address_linked_ssa_lines(module)

    visible, highlighted = ssa_viewport(lines, 0x1000, search="integer_add")

    assert highlighted
    assert visible
    assert all("integer_add" in item.text for item in visible)


def test_search_replace_targets_stable_ssa_occurrences_but_grants_no_bytes():
    module, _session_value = _session()
    lines = address_linked_ssa_lines(module)
    target = next(item for item in lines if item.machine_address == 0x1003)

    edits = replace_ssa_operations(
        module, "integer_add", "integer_subtract", line_ids=(target.line_id,),
    )

    assert tuple(item.line_id for item in edits) == (target.line_id,)
    assert edits[0].old_operation == "machine.integer_add"
    assert edits[0].new_operation == "machine.integer_subtract"
    changed = next(
        item for item in address_linked_ssa_lines(module)
        if item.line_id == target.line_id
    )
    assert changed.operation == "machine.integer_subtract"
    assert "machine_encoded" not in changed.__dataclass_fields__


def test_edit_transaction_is_non_executable_until_ledger_proves_every_group():
    module, _session_value = _session()
    target = next(
        item for item in address_linked_ssa_lines(module)
        if item.machine_address == 0x1003
    )
    transaction = SSAEditTransaction(module)

    transaction.replace(
        "machine.integer_add", "machine.integer_subtract",
        line_ids=(target.line_id,),
    )
    validation = transaction.validate_pe()

    assert not validation.executable
    assert validation.ledger.unresolved
    transaction.rollback()
    restored = next(
        item for item in address_linked_ssa_lines(module)
        if item.line_id == target.line_id
    )
    assert restored.operation == "machine.integer_add"
    assert transaction.replacements == ()


def test_benign_constant_edit_recompiles_through_bidirectional_head():
    lifting = raise_binary_region_to_ssa(
        b"\x48\x83\xc0\x01\xc3", maximum_file_size=5, size=5,
        base_address=0x1000, name="pixel_add",
        full_vocabulary_report=True, cfg_decode=True,
    )
    module = IRModule({"pixel_add": lifting.function})
    constant = next(
        line for line in address_linked_ssa_lines(module)
        if line.machine_address == 0x1000 and line.operation == "Const"
        and "value=1" in line.text
    )
    transaction = SSAEditTransaction(module)
    transaction.replace_constant(constant.line_id, 2)
    facts = {
        "register-or-memory-destination", "signed-immediate-8",
        "width-64", "modulo-2^64", "all-add-flags-exact",
    }

    validation = transaction.validate_pe(
        proven_facts_by_address={0x1000: facts},
        allocated_instructions_by_address={
            0x1000: X86AllocatedInstruction(
                int(X86InstructionToken.ADD_R64_IMM8),
                (
                    RegisterOperand(X86Register.RAX, 64),
                    ImmediateOperand(2, 8, True),
                ),
                0x1000,
            ),
        },
        allow_new_selection=True,
    )

    assert validation.executable
    edited = next(
        item for item in validation.ledger.occurrences
        if item.machine_address == 0x1000
    )
    assert edited.disposition == "allocated-selection"
    assert edited.encoded == b"\x48\x83\xc0\x02"
    decoded, end = X86ReferenceDecoder().decode_one(
        memoryview(edited.encoded), 0, base_address=0x1000,
    )
    assert end == 4
    assert decoded.operands[1] == ImmediateOperand(2, 8, True)
