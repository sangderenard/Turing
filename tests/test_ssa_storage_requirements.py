from src.compiler.ssa_storage_requirements import module_storage_requirements
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue


def test_constant_only_address_use_proves_minimum_storage_span_across_call():
    formal = SSAValue(10, "float64")
    constants = [SSAValue(11 + index, "int64") for index in range(3)]
    addresses = [SSAValue(20 + index, "ptr") for index in range(3)]
    instructions = []
    for index, (constant, address) in enumerate(zip(constants, addresses)):
        instructions.extend((
            Instr("Const", [], constant, attributes={"constant": index}),
            Instr("GetElementPtr", [formal, constant], address),
        ))
    instructions.append(Instr("Ret", [], None))
    callee = Function(
        "constant_projection", [formal],
        {"entry": BasicBlock("entry", instructions, [])},
    )
    actual = SSAValue(0, "float64")
    caller = Function(
        "owner", [actual],
        {"entry": BasicBlock("entry", [
            Instr("Call", [actual], None, attributes={"callee": callee.name}),
            Instr("Ret", [], None),
        ], [])},
    )

    requirements = module_storage_requirements(
        IRModule({caller.name: caller, callee.name: callee})
    )

    assert requirements[callee.name][formal.id].shape == (3,)
    assert requirements[caller.name][actual.id].shape == (3,)


def test_dynamic_address_use_does_not_guess_storage_span():
    values = SSAValue(0, "float64")
    index = SSAValue(1, "int64")
    address = SSAValue(2, "ptr")
    function = Function(
        "dynamic_projection", [values, index],
        {"entry": BasicBlock("entry", [
            Instr("GetElementPtr", [values, index], address),
            Instr("Ret", [], None),
        ], [])},
    )

    requirement = module_storage_requirements(
        IRModule({function.name: function})
    )[function.name][values.id]

    assert requirement.shape == ()
    assert requirement.element_count is None


def test_llvm_literal_indices_are_storage_evidence():
    values = SSAValue(0, "float64")
    index = SSAValue(1, "int64")
    address = SSAValue(2, "ptr")
    function = Function(
        "literal_projection", [values],
        {"entry": BasicBlock("entry", [
            Instr("Const", [], index, attributes={"llvm_literal": "i64 7"}),
            Instr("GetElementPtr", [values, index], address),
            Instr("Ret", [], None),
        ], [])},
    )

    requirement = module_storage_requirements(
        IRModule({function.name: function})
    )[function.name][values.id]

    assert requirement.shape == (8,)
