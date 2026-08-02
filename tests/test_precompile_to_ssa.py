from __future__ import annotations

from src.common.tensors.fused_ir import FusedProgram, Meta, OpStep
from src.compiler.control_source import (
    CallBlock,
    ControlProgram,
    ControlUniform,
    LoopBlock,
    ParallelDeployment,
    RecursionRegion,
    SequenceBlock,
    StatementBlock,
)
from src.compiler.precompile_to_ssa import (
    find_ssa_cycles,
    lower_control_program_to_ssa,
    lower_class_navigation_to_ssa,
    lower_fused_program_to_ssa,
    lower_precompile_and_control_to_ssa,
)
from src.compiler.shell_reference_tables import (
    ClassNavigationMember,
    ClassNavigationRecord,
    ClassNavigationTable,
)


def _program(*steps):
    value_ids = {0}
    value_ids.update(step.result_id for step in steps)
    return FusedProgram(
        version=1,
        feeds={0},
        steps=list(steps),
        outputs={"result": steps[-1].result_id},
        meta={
            value_id: Meta((4,), "float32", "glsl")
            for value_id in value_ids
        },
    )


def test_numerical_tensor_ops_call_real_imported_llvm_algorithms():
    program = _program(
        OpStep(0, "neg", [0], {}, 1),
        OpStep(1, "add", [1], {"right_scalar": 1.0}, 2),
    )

    function, shortfalls = lower_fused_program_to_ssa(program)

    assert shortfalls == ()
    assert [instruction.op for instruction in function.blocks["entry"].instrs] == [
        "Call",
        "Call",
        "Ret",
    ]
    assert function.blocks["entry"].instrs[0].res.id == 1
    assert function.blocks["entry"].instrs[0].attributes["callee"] == (
        "unary_double"
    )
    assert function.blocks["entry"].instrs[1].attributes["callee"] == (
        "binary_scalar_double"
    )
    assert function.blocks["entry"].instrs[1].attributes["right_scalar"] == 1.0


def test_class_navigation_has_general_ssa_semantic_procedures():
    navigation = ClassNavigationTable((ClassNavigationRecord(
        identity="Vault",
        permissions=("vault:enter",),
        members=(ClassNavigationMember(
            name="read",
            identity="Vault.read",
            kind="method",
            storage=None,
            function_reference=7,
            permissions=("vault:read",),
        ),),
        instantiation_functions=(),
    ),))

    module = lower_class_navigation_to_ssa(navigation)

    assert set(module.functions) == {
        "turing.class.lookup",
        "turing.class.instantiate",
        "turing.class.resolve_member",
        "turing.class.evaluate_permission",
    }
    operations = {
        instruction.op
        for function in module.functions.values()
        for instruction in function.blocks["entry"].instrs
    }
    assert operations <= {"Const", "Eq", "And", "LAnd", "Select", "Ret"}
    assert {"ClassLookup", "ClassInstantiate", "ResolveMember", "EvaluatePermission"}.isdisjoint(operations)
    resolve = module.functions["turing.class.resolve_member"]
    assert resolve.blocks["entry"].instrs[0].attributes[
        "class_navigation_lut"
    ]["classes"][0]["members"][0]["function_reference"] == 7
    assert [value.dtype for value in resolve.blocks["entry"].instrs[-1].args] == [
        "i32", "i32", "bool",
    ]


def test_numerical_lowering_routes_scatter_through_real_llvm_algorithm():
    program = _program(
        OpStep(0, "scatter", [0], {}, 1),
        OpStep(1, "add", [1], {"right_scalar": 1.0}, 2),
    )

    function, shortfalls = lower_fused_program_to_ssa(program)

    assert shortfalls == ()
    assert [instruction.op for instruction in function.blocks["entry"].instrs] == [
        "Call",
        "Call",
        "Ret"
    ]
    assert function.blocks["entry"].instrs[0].attributes["callee"] == (
        "index_assign_double"
    )


def test_planner_loop_becomes_phi_cfg_cycle_with_region_call():
    control = ControlProgram(
        LoopBlock(
            "iteration",
            "0",
            "frame_count",
            "1",
            StatementBlock(("__scheduled_region_7__",)),
        ),
        region_indices=(7,),
        uniforms=(ControlUniform("frame_count", 40, "int"),),
    )

    function, shortfalls = lower_control_program_to_ssa(
        control,
        first_value_id=100,
    )
    cycles = find_ssa_cycles(function)

    assert shortfalls == ()
    assert len(cycles) == 1
    assert cycles[0].represented_by_phi
    assert cycles[0].phi_blocks == ("loop_header",)
    assert [
        instruction.op
        for instruction in function.blocks["loop_header"].instrs
    ] == ["Phi", "Lt", "CondBr"]
    assert function.blocks["loop_body"].instrs[0].op == "Call"
    assert function.blocks["loop_body"].instrs[0].attributes[
        "region_index"
    ] == 7


def test_combined_lowering_retains_sequence_order_and_cycle_report():
    program = _program(OpStep(0, "mul", [0], {"right_scalar": 2}, 1))
    control = ControlProgram(
        SequenceBlock((
            StatementBlock(("__scheduled_region_0__",)),
            LoopBlock(
                "i",
                "0",
                "count",
                "1",
                StatementBlock(("__scheduled_region_1__",)),
            ),
        )),
        region_indices=(0, 1),
        uniforms=(ControlUniform("count", 9, "int"),),
    )

    result = lower_precompile_and_control_to_ssa(program, control)

    assert {
        "numerical_precompile",
        "planned_control",
        "binary_scalar_double",
    } <= set(result.module.functions)
    assert len(result.cycles) == 1
    assert result.cycles[0].represented_by_phi
    assert result.shortfalls == ()


def test_control_region_call_wires_feeds_and_explicit_output_producers():
    control = ControlProgram(
        StatementBlock(("__scheduled_region_3__",)),
        region_indices=(3,),
    )

    function, shortfalls = lower_control_program_to_ssa(
        control,
        first_value_id=100,
        region_callees={3: "numerical_region_3"},
        region_signatures={3: ((10, 11), (12, 13))},
    )
    instructions = function.blocks["entry"].instrs

    assert shortfalls == ()
    call = next(item for item in instructions if item.op == "Call")
    assert [value.id for value in call.args] == [10, 11]
    assert call.attributes["output_ids"] == (12, 13)
    loads = [item for item in instructions if item.op == "Load"]
    assert [item.res.id for item in loads] == [12, 13]


def test_control_ssa_returns_declared_region_output_for_shell_binding():
    control = ControlProgram(
        StatementBlock(("__scheduled_region_3__",)),
        region_indices=(3,),
    )

    function, shortfalls = lower_control_program_to_ssa(
        control,
        first_value_id=100,
        region_callees={3: "numerical_region_3"},
        region_signatures={3: ((), (12,))},
        output_value_ids=(12,),
    )

    assert shortfalls == ()
    returned = function.blocks["entry"].instrs[-1]
    assert returned.op == "Ret"
    assert [value.id for value in returned.args] == [12]


def test_control_ssa_resolves_named_output_from_retained_id_history():
    control = ControlProgram(
        StatementBlock(("__scheduled_region_3__",)),
        region_indices=(3,),
    )

    function, shortfalls = lower_control_program_to_ssa(
        control,
        first_value_id=100,
        region_callees={3: "numerical_region_3"},
        region_signatures={3: ((), (7,))},
        named_output_histories={"count": (4, 7, 8)},
        value_name_histories={"x": (0,), "count": (4, 7, 8)},
        parameter_names=("x",),
    )

    assert shortfalls == ()
    assert function.metadata["named_outputs"] == (("count", 7),)
    assert function.metadata["value_names"] == (("count", 7),)
    assert function.metadata["parameter_names"] == ()
    assert function.metadata["control_ir"] is True
    assert [value.id for value in function.blocks["entry"].instrs[-1].args] == [7]


def test_loop_collection_binding_is_indexed_store_after_region_publication():
    control = ControlProgram(
        LoopBlock(
            "iteration",
            "0",
            "count",
            "1",
            StatementBlock(("__scheduled_region_4__",)),
        ),
        region_indices=(4,),
        uniforms=(ControlUniform("count", 40, "int"),),
        collection_bindings=((12, 20, "iteration", 0),),
    )

    function, shortfalls = lower_control_program_to_ssa(
        control,
        first_value_id=100,
        region_callees={4: "numerical_region_4"},
        region_signatures={4: ((), (12,))},
    )
    body = function.blocks["loop_body"].instrs

    assert shortfalls == ()
    call_index = next(
        index for index, item in enumerate(body) if item.op == "Call"
    )
    store_index = next(
        index for index, item in enumerate(body) if item.op == "Store"
    )
    assert call_index < store_index
    assert body[store_index].args[0].id == 12
    assert body[store_index].attributes["binding"] == (
        "collection_publication"
    )


def test_loop_carried_value_is_a_phi_with_the_region_update():
    control = ControlProgram(
        LoopBlock(
            "iteration",
            "0",
            "4",
            "1",
            StatementBlock(("__scheduled_region_3__",)),
            carried_aliases=((20, 10),),
        ),
        region_indices=(3,),
    )

    function, shortfalls = lower_control_program_to_ssa(
        control,
        first_value_id=100,
        region_signatures={3: ((10,), (20,))},
    )

    assert shortfalls == ()
    phis = [
        instruction
        for instruction in function.blocks["loop_header"].instrs
        if instruction.attributes.get("binding") == "loop_carried"
    ]
    assert len(phis) == 1
    assert phis[0].attributes["initial_value_id"] == 10
    assert phis[0].attributes["updated_value_id"] == 20
    assert phis[0].args[0].id == 10
    assert phis[0].args[1].id == 20
    assert any(
        instruction.res is phis[0].args[1]
        for instruction in function.blocks["loop_body"].instrs
    )


def test_recursion_table_lowers_to_phi_and_llvm_shaped_backedge():
    region = RecursionRegion(
        region_id=7,
        kind="irreducible_recursion",
        lower_as="while",
        members=(10, 20),
        feedback=((20, 10, "carried"),),
    )
    control = ControlProgram(
        LoopBlock(
            "iteration",
            "0",
            "4",
            "1",
            StatementBlock(("__scheduled_region_3__",)),
            carried_aliases=((20, 10),),
            recursion_region_id=7,
        ),
        region_indices=(3,),
        recursion_regions=(region,),
    )

    function, shortfalls = lower_control_program_to_ssa(
        control,
        first_value_id=100,
        region_signatures={3: ((10,), (20,))},
    )

    assert shortfalls == ()
    lowered = function.metadata["recursion_table"][7]
    assert lowered["lower_as"] == "while"
    loop, = lowered["loops"]
    assert loop["header"] == "loop_header"
    assert loop["latch"] == "loop_latch"
    assert loop["backedge"] == ("loop_latch", "loop_header")
    assert len(loop["phi_value_ids"]) == 2
    assert function.blocks["loop_latch"].successors == ["loop_header"]
    assert all(
        instruction.attributes.get("recursion_region_id") == 7
        for instruction in function.blocks["loop_header"].instrs
        if instruction.op == "Phi"
    )


def test_callblock_evaporates_and_parallel_lanes_linearize_without_fake_call():
    control = ControlProgram(
        CallBlock(
            7,
            ParallelDeployment((
                StatementBlock(("__scheduled_region_1__",)),
                StatementBlock(("__scheduled_region_2__",)),
            )),
            argument_bindings=((10, 10),),
            result_bindings=((30, 30),),
        ),
        region_indices=(1, 2),
    )

    function, shortfalls = lower_control_program_to_ssa(
        control,
        region_signatures={
            1: ((10,), (20,)),
            2: ((20,), (30,)),
        },
    )

    assert shortfalls == ()
    calls = [
        instruction
        for instruction in function.blocks["entry"].instrs
        if instruction.op == "Call"
    ]
    assert [call.attributes["callee"] for call in calls] == [
        "numerical_region_1",
        "numerical_region_2",
    ]
