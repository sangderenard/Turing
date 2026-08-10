from __future__ import annotations

from src.common.tensors.fused_ir import FusedProgram, Meta, OpStep
from src.compiler.control_source import (
    ControlDeploymentLane,
    ControlDeploymentRegion,
    CallBlock,
    ControlProgram,
    ControlUniform,
    LoopBlock,
    LoopControlBlock,
    ParallelDeployment,
    RecursionRegion,
    SequenceBlock,
    StatementBlock,
    StateMachineTick,
    WhileBlock,
)
from src.compiler.precompile_to_ssa import (
    find_ssa_cycles,
    lower_control_program_to_ssa,
    lower_class_navigation_to_ssa,
    lower_fused_program_to_ssa,
    lower_precompile_and_control_to_ssa,
)
from src.compiler.ssa_fortran_backend import emit_module
from src.compiler.shell_reference_tables import (
    ClassNavigationMember,
    ClassNavigationRecord,
    ClassNavigationTable,
)
from src.transmogrifier.ssa import IRModule


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


def test_repeat_lowers_as_native_fortran_axis_tiling():
    program = FusedProgram(
        version=1,
        feeds={0},
        steps=[
            OpStep(
                0,
                "slice",
                [0],
                {
                    "slice_kind": "axis",
                    "dim": 1,
                    "start": 0,
                    "step": 1,
                    "count": 1,
                },
                1,
            ),
            OpStep(
                1,
                "repeat",
                [1],
                {"repeats": 2, "dim": 0},
                2,
            ),
        ],
        outputs={"result": 2},
        meta={
            0: Meta((3, 1, 4), "float32", "cpu"),
            1: Meta((3, 1, 4), "float32", "cpu"),
            2: Meta((6, 1, 4), "float32", "cpu"),
        },
    )

    function, shortfalls = lower_fused_program_to_ssa(program)
    output = next(
        instruction.res
        for instruction in function.blocks["entry"].instrs
        if instruction.res is not None and instruction.res.id == 2
    )
    module = emit_module(
        {function.name: function},
        outputs={function.name: [output]},
    )

    assert shortfalls == ()
    assert module.complete, [item.format() for item in module.shortfalls]
    assert "mod(" in module.source
    assert "= 1, 6" in module.source
    assert "1:1, :)([" not in module.source


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


def test_random_source_lowers_to_repository_ssa_feature_call():
    program = FusedProgram(
        version=1,
        feeds=set(),
        steps=[
            OpStep(
                0,
                "random_source",
                [],
                {"shape": (4,), "seed": 7, "seed0": 11, "seed1": 13},
                0,
            ),
        ],
        outputs={"result": 0},
        meta={0: Meta((4,), "float64", "cpu")},
    )

    function, shortfalls = lower_fused_program_to_ssa(program)
    call = function.blocks["entry"].instrs[0]

    assert shortfalls == ()
    assert call.op == "Call"
    assert call.attributes["callee"] == "xoroshiro128ss_fill_double"
    assert call.attributes["seed"] == 7


def test_native_fortran_ops_keep_mean_and_span_fill_in_ssa():
    program = _program(
        OpStep(0, "mean", [0], {}, 1),
        OpStep(1, "zeros_like", [0], {}, 2),
        OpStep(2, "add", [1, 2], {}, 3),
    )

    function, shortfalls = lower_fused_program_to_ssa(program)

    assert shortfalls == ()
    instructions = function.blocks["entry"].instrs
    assert [instruction.op for instruction in instructions] == [
        "Call",
        "Fill",
        "Call",
        "Ret",
    ]
    assert instructions[0].attributes["tensor_operation"] == "mean"


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


def test_combined_lowering_uses_graph_region_without_fused_kernel():
    from src.compiler.hierarchical_plan import PlanClosure, PlanLine

    program = _program(OpStep(0, "mul", [0], {"right_scalar": 2}, 1))
    control = ControlProgram(
        StatementBlock(("__scheduled_region_4__",)),
        region_indices=(4,),
    )
    hierarchy = PlanClosure(
        "root",
        (10,),
        (
            PlanClosure(
                "region_4",
                (10,),
                (PlanLine.create("Store", inputs=(10,), outputs=(11,)),),
            ),
        ),
    )

    result = lower_precompile_and_control_to_ssa(
        program,
        control,
        hierarchy_plan=hierarchy,
    )

    assert "planned_region_4" in result.module.functions
    assert result.module.functions["planned_region_4"].blocks[
        "entry"
    ].instrs[0].op == "Store"
    assert not any(
        shortfall.domain == "planned-region"
        for shortfall in result.shortfalls
    )


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


def test_projected_enumerate_binding_lowers_extent_counter_and_element():
    control = ControlProgram(
        LoopBlock(
            "iteration_9",
            "0",
            "__iterable_extent_40__",
            "1",
            StatementBlock(("__scheduled_region_4__",)),
        ),
        region_indices=(4,),
        projected_iterable_bindings=(
            (40, 41, "iteration_9", "induction"),
            (40, 42, "iteration_9", None),
        ),
    )

    function, shortfalls = lower_control_program_to_ssa(
        control,
        first_value_id=100,
        region_callees={4: "numerical_region_4"},
        region_signatures={4: ((41, 42), ())},
        region_value_meta={
            40: Meta((4,), "float64", "cpu"),
            41: Meta((), "int32", "cpu"),
            42: Meta((), "float64", "cpu"),
        },
    )

    assert shortfalls == ()
    assert any(
        instruction.attributes.get("tensor_operation") == "extent"
        for instruction in function.blocks["entry"].instrs
    )
    body = function.blocks["loop_body"].instrs
    assert any(
        instruction.attributes.get("binding") == "projected_iterable"
        and instruction.op == "Load"
        for instruction in body
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
    assert [instruction.op for instruction in function.blocks["entry"].instrs][
        :1
    ] == ["Deploy"]
    assert function.blocks["entry"].instrs[-2].op == "Join"
    deployment, = function.metadata["deployment_regions"]
    assert deployment.schedule == "independent_lanes"
    assert deployment.deploy_site == ("entry", 0)
    assert deployment.join_site == (
        "entry",
        next(
            index
            for index, instruction in enumerate(
                function.blocks["entry"].instrs
            )
            if instruction.op == "Join"
        ),
    )
    assert [lane.callees for lane in deployment.lanes] == [
        ("numerical_region_1",),
        ("numerical_region_2",),
    ]
    assert [lane.source_region_indices for lane in deployment.lanes] == [
        (1,),
        (2,),
    ]
    module = IRModule({function.name: function})
    assert module.deployment_table[function.name] == (deployment,)


def test_parallel_loop_retains_iteration_deployment_region_in_ssa():
    control = ControlProgram(
        LoopBlock(
            "iteration",
            "0",
            "8",
            "1",
            StatementBlock(("__scheduled_region_7__",)),
            parallel_iterations=True,
        ),
        region_indices=(7,),
    )

    function, shortfalls = lower_control_program_to_ssa(
        control,
        first_value_id=100,
    )

    assert shortfalls == ()
    deployment, = function.metadata["deployment_regions"]
    assert deployment.schedule == "independent_iterations"
    assert deployment.iteration_space == ("0", "8", "1")
    assert deployment.lanes[0].source_region_indices == (7,)
    call = next(
        instruction
        for instruction in function.blocks["loop_body"].instrs
        if instruction.op == "Call"
    )
    assert call.attributes["deployment_memberships"] == ((0, 0),)


def test_control_deployment_table_maps_scheduled_subgraphs_into_ssa():
    control = ControlProgram(
        SequenceBlock((
            StatementBlock(("__scheduled_region_3__",)),
            StatementBlock(("__scheduled_region_4__",)),
        )),
        region_indices=(3, 4),
        deployment_regions=(ControlDeploymentRegion(
            region_id=12,
            kind="parallel_candidate",
            schedule="independent_lanes",
            schedule_preference="asap",
            lanes=(
                ControlDeploymentLane(
                    0, (3,), (30,), (300, 301)
                ),
                ControlDeploymentLane(
                    1, (4,), (40,), (400, 401)
                ),
            ),
            origin="unrolled_loop",
            source_loop_node_id=22,
        ),),
    )

    function, shortfalls = lower_control_program_to_ssa(control)

    assert shortfalls == ()
    deployment, = function.metadata["deployment_regions"]
    assert deployment.region_id == 12
    assert deployment.origin == "unrolled_loop"
    assert deployment.schedule_preference == "asap"
    assert deployment.source_loop_node_id == 22
    assert deployment.lanes[0].instruction_sites == (("entry", 0),)
    assert deployment.lanes[0].source_region_indices == (3,)
    assert deployment.lanes[0].source_value_ids == (30,)
    assert deployment.lanes[0].source_node_ids == (300, 301)
    assert deployment.lanes[1].instruction_sites == (("entry", 1),)
    assert [
        instruction.attributes["deployment_memberships"]
        for instruction in function.blocks["entry"].instrs[:2]
    ] == [((12, 0),), ((12, 1),)]


def test_condition_loop_break_and_switch_default_lower_to_cfg_ssa():
    control = ControlProgram(
        WhileBlock(
            predicate_value_id=10,
            condition=StatementBlock(("__scheduled_region_0__",)),
            body=SequenceBlock((
                StatementBlock(("__scheduled_region_1__",)),
                StateMachineTick(
                    state="value_20",
                    cases=(("1", LoopControlBlock("continue")),),
                    default=LoopControlBlock("break", predicate_value_id=11),
                ),
            )),
            recursion_region_id=4,
        ),
        region_indices=(0, 1),
        recursion_regions=(RecursionRegion(
            4, "cycle", "while", (10, 11, 20)
        ),),
    )

    function, shortfalls = lower_control_program_to_ssa(
        control,
        region_signatures={
            0: ((), (10,)),
            1: ((10,), (11, 20)),
        },
    )

    assert shortfalls == ()
    assert function.blocks["while_header"].successors == [
        "while_body", "while_exit"
    ]
    assert function.blocks["while_latch"].successors == ["while_header"]
    assert any(
        instruction.op == "Phi"
        and instruction.attributes.get("binding") == "while_condition"
        for instruction in function.blocks["while_header"].instrs
    )
    assert function.metadata["recursion_table"][4]["loops"][0][
        "domain"
    ] == "condition"
