from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from src.compiler.machine_dialect_ssa import decoded_function_to_machine_ssa
from src.compiler.binary_ingestion import parse_pe_image, pe_runtime_function_region
from src.compiler.machine_execution import (
    MachineExecutionOrchestrator, MachineExecutionState, MachineExecutionStatus,
)
from src.compiler.machine_reference_vocabulary import X86ReferenceDecoder
from src.compiler.machine_ssa_execution import (
    MachineSSABoundaryError, machine_ssa_program,
)
from src.compiler.native_code_retention import retain_pe_image
from src.transmogrifier.ssa import IRModule


def _decoded(encoded=b"\x48\xff\xc0\xc3"):
    return X86ReferenceDecoder().decode_report(
        encoded, base_address=0x1000,
        stop_at_return=False, allow_trailing_after_terminal=True,
    ).instructions


def _image():
    return SimpleNamespace(
        image_base=0x1000, entrypoint_rva=0, encoded=None, sections=(),
    )


def _decoded_program(instructions):
    return SimpleNamespace(
        image=_image(),
        functions=(SimpleNamespace(
            report=SimpleNamespace(instructions=instructions),
        ),),
    )


def test_machine_ssa_executes_the_same_architectural_transitions_as_decoded_vm():
    instructions = _decoded()
    module = IRModule({
        "increment": decoded_function_to_machine_ssa("increment", instructions),
    })
    from_ssa = machine_ssa_program(module, image=_image())

    initial = MachineExecutionState(pc=from_ssa.entry_address("increment"))
    ssa_executor = from_ssa.executor()
    decoded_executor = MachineExecutionOrchestrator(
        _decoded_program(instructions),
        effect_handlers=ssa_executor.effect_handlers,
        predicate_handler=ssa_executor.predicate_handler,
        indirect_target_handler=ssa_executor.indirect_target_handler,
    )
    ssa_first = ssa_executor.step(initial)
    decoded_first = decoded_executor.step(initial)
    ssa_second = ssa_executor.step(ssa_first.state)
    decoded_second = decoded_executor.step(decoded_first.state)

    assert ssa_first == decoded_first
    assert ssa_first.status is MachineExecutionStatus.RUNNING
    assert ssa_first.state.registers[0] == 1
    assert ssa_second == decoded_second
    assert ssa_second.status is MachineExecutionStatus.HALTED


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("machine_token", "NOP", "token mismatch"),
        ("machine_semantic", "NO_OPERATION", "semantic mismatch"),
        ("machine_encoded", "90", "operation mismatch"),
        ("machine_reads", (), "reads mismatch"),
    ),
)
def test_machine_ssa_boundary_rejects_tampered_transition_context(
    field, replacement, message,
):
    function = decoded_function_to_machine_ssa("increment", _decoded())
    transition = next(
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op.startswith("machine.integer_increment")
    )
    transition.attributes[field] = replacement

    with pytest.raises(MachineSSABoundaryError, match=message):
        machine_ssa_program(IRModule({"increment": function}), image=_image())


def test_machine_ssa_boundary_requires_loader_context_and_exact_dialect():
    function = decoded_function_to_machine_ssa("increment", _decoded())
    module = IRModule({"increment": function})
    with pytest.raises(MachineSSABoundaryError, match="loader/image context"):
        machine_ssa_program(module)

    changed = deepcopy(function)
    changed.metadata["dialect"] = "some-other-machine"
    with pytest.raises(MachineSSABoundaryError, match="contains no"):
        machine_ssa_program(IRModule({"increment": changed}), image=_image())


def test_machine_ssa_boundary_verifies_transitions_against_retained_pe_bytes():
    path = Path(sys.executable).resolve()
    encoded = path.read_bytes()
    image, _statistics = parse_pe_image(
        encoded, maximum_file_size=len(encoded),
    )
    owner = image.runtime_function_for_rva(image.entrypoint_rva)
    assert owner is not None
    _record, _offset, region = pe_runtime_function_region(
        image, owner.begin_rva,
        maximum_function_size=owner.end_rva - owner.begin_rva,
    )
    report = X86ReferenceDecoder().decode_cfg_report(
        region, base_address=image.image_base + owner.begin_rva,
    )
    assert report.instructions
    function = decoded_function_to_machine_ssa("entry", report.instructions)
    module = IRModule({"entry": function})
    retained = retain_pe_image(image, source_identity=str(path))

    program = machine_ssa_program(module, retained_native_module=retained)

    assert program.entry_address("entry") == min(
        item.address for item in report.instructions
    )
    assert tuple(
        instruction
        for record in program.functions
        for instruction in record.report.instructions
    ) == report.instructions
