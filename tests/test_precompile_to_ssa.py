from __future__ import annotations

from src.common.tensors.fused_ir import FusedProgram, Meta, OpStep
from src.compiler.control_source import (
    ControlDeploymentLane,
    ControlDeploymentRegion,
    CallBlock,
    ConditionalBlock,
    ControlProgram,
    ControlExpression,
    ControlSequenceMutation,
    ControlUniform,
    LoopBlock,
    LoopControlBlock,
    ParallelDeployment,
    RecursionRegion,
    SequenceBlock,
    SequenceMutationBlock,
    SequenceQueryBlock,
    StatementBlock,
    StateMachineTick,
    WhileBlock,
    overlay_scheduled_control,
)
from src.compiler.precompile_to_ssa import (
    _materialize_control_constants,
    ResolvedSequenceSchema,
    find_ssa_cycles,
    lower_control_program_to_ssa,
    lower_class_navigation_to_ssa,
    lower_fused_integral_to_repository_ssa,
    lower_fused_program_to_ssa,
    lower_precompile_and_control_to_ssa,
    lower_control_sections_to_ssa,
    link_verified_source_region_integrals,
    merge_repository_ssa_modules,
    resolve_sequence_schemas,
)
from src.compiler.ssa_fortran_backend import emit_module
from src.compiler.shell_reference_tables import (
    ClassNavigationMember,
    ClassNavigationRecord,
    ClassNavigationTable,
)
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue
from src.transmogrifier.function_table import ParameterContract


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


def test_structural_integral_lowers_resident_mapping_store_not_numeric_slots():
    program = FusedProgram(
        version=1,
        feeds={13, 19, 289},
        steps=[OpStep(0, "IndexedStore", [289, 13, 19], {}, 290)],
        outputs={"mutated": 290},
        meta={},
        extras={
            "structural_resident_table_contract": {
                "schema": "turing.structural-resident-table-integral.v1",
                "sequences": [{
                    "sequence_id": 289,
                    "policy": "unique",
                    "column_count": 2,
                    "writable": True,
                    "column_dtypes": ["int64", "int64"],
                    "storage_identity": "Builder.external_values",
                    "value_record": "SSAValue",
                    "value_optional": True,
                }],
                "stores": [{
                    "effect_value_id": 290,
                    "key_value_id": 13,
                    "stored_value_id": 19,
                    "sequence_value_id": 289,
                }],
            },
        },
    )

    module, outputs, exports, shortfalls = (
        lower_fused_integral_to_repository_ssa(
            program, function_name="restore_external_value",
        )
    )

    assert shortfalls == ()
    assert exports == ("restore_external_value",)
    assert outputs == {"restore_external_value": ()}
    function = module.functions["restore_external_value"]
    calls = [
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
    ]
    assert calls[0].attributes["ssa_sequence_operation"] == "table_store"
    assert calls[0].attributes["callee"] in module.functions
    assert module.sequence_tables["restore_external_value"].sequences[
        289
    ].column_dtypes == ("int64", "int64")
    previous = next(value for value in function.args if value.id == 19)
    assert previous.accounting == {
        "structural_record_identity": "SSAValue",
        "structural_record_handle": True,
    }



def test_control_wrapper_materializes_authored_constants_outside_its_abi():
    dynamic = SSAValue(0, dtype="int")
    literal = SSAValue(1, dtype="int")
    result = SSAValue(2, dtype="int")
    function = Function(
        "control",
        [dynamic, literal],
        {"entry": BasicBlock("entry", [Instr("Add", [dynamic, literal], result)])},
    )

    _materialize_control_constants(
        function, {1: 8}, value_dtypes={0: "int", 1: "int"},
    )

    assert [argument.id for argument in function.args] == [0]
    instruction = function.blocks["entry"].instrs[0]
    assert instruction.op == "Const"
    assert instruction.res.id == 1
    assert instruction.res.dtype == "int"
    assert instruction.attributes == {"value": 8}


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
    emitted = emit_module(module, name="class_navigation_probe")
    assert emitted.api.metadata["class_table_schema"] == (
        "turing.repository-ssa-class-table.v1"
    )
    assert emitted.api.metadata["class_table"][0]["identity"] == "Vault"
    reference = module.function_table.declare(
        "read",
        qualified_name="Vault.read",
        parameter_contracts=(ParameterContract(
            "self", transfer="alias", access="inout", storage="record",
            scope="caller",
        ),),
    )
    emitted = emit_module(module, name="class_navigation_function_probe")
    assert emitted.api.metadata["function_table_schema"] == (
        "turing.repository-ssa-function-table.v1"
    )
    assert emitted.api.metadata["function_table"] == [{
        "reference": reference.address,
        "name": "read",
        "qualified_name": "Vault.read",
        "state": "declared",
        "recursive": False,
        "parameter_contracts": [{
            "name": "self",
            "transfer": "alias",
            "access": "inout",
            "storage": "record",
            "scope": "caller",
        }],
    }]


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


def test_structural_region_ops_are_ssa_values_and_indexes_are_legalized():
    root = _program(OpStep(0, "mul", [0], {"right_scalar": 2}, 1))
    region = FusedProgram(
        version=1,
        feeds={10, 11, 12},
        steps=[
            OpStep(0, "getattr", [10], {"attribute": "items"}, 13),
            OpStep(1, "Indexed", [13, 11], {}, 14),
            OpStep(2, "IndexedStore", [13, 11, 12], {}, 15),
            OpStep(3, "tolist", [14], {}, 16),
        ],
        outputs={"read": 14, "stored": 15, "host_view": 16},
        meta={
            value_id: Meta((), None, None)
            for value_id in range(10, 17)
        },
    )
    control = ControlProgram(
        StatementBlock(("__scheduled_region_0__",)),
        region_indices=(0,),
    )

    result = lower_precompile_and_control_to_ssa(
        root, control, region_programs={0: region},
    )

    assert result.shortfalls == ()
    operations = [
        instruction.op
        for block in result.module.functions["numerical_region_0"].blocks.values()
        for instruction in block.instrs
    ]
    assert "GetAttr" in operations
    assert "Call" in operations
    assert "Indexed" not in operations
    assert "IndexedStore" not in operations
    assert operations.count("GetElementPtr") == 2
    assert "Load" in operations
    assert "Store" in operations


def test_index_dtype_propagation_is_scoped_per_function_identity():
    from src.compiler.ir_indexing import lower_indexing_to_ssa_addressing

    base = SSAValue(1, dtype="int64", shape=(3,))
    index = SSAValue(2, dtype="int64")
    loaded = SSAValue(3, dtype="float64")
    region = Function(
        "region", [base, index],
        {"entry": BasicBlock("entry", [Instr("Indexed", [base, index], loaded)])},
    )
    aggregate = SSAValue(3, dtype="ssa.aggregate")
    root = Function("root", [], {"entry": BasicBlock("entry", [])})
    root.args.append(aggregate)

    lower_indexing_to_ssa_addressing({"region": region, "root": root})

    region_result = next(
        instruction.res
        for instruction in region.blocks["entry"].instrs
        if instruction.op == "Load"
    )
    assert region_result.dtype == "int64"
    assert aggregate.dtype == "ssa.aggregate"


def test_record_field_getattr_becomes_a_loaded_region_capture():
    from src.compiler.hierarchical_plan import PlanClosure, PlanLine

    region = PlanClosure(
        "region_0",
        captures=(5, 7),
        items=(
            PlanLine.create(
                "getattr",
                inputs=(5,),
                outputs=(6,),
                attributes={"attribute": "gain"},
            ),
            PlanLine.create("Add", inputs=(6, 7), outputs=(8,)),
        ),
        value_shapes=tuple(
            (value_id, (), "float64") for value_id in (5, 6, 7, 8)
        ),
    )
    hierarchy = PlanClosure("root", (), (region,))
    control = ControlProgram(
        StatementBlock(("__scheduled_region_0__",)),
        region_indices=(0,),
    )

    module, shortfalls, _outputs = lower_control_sections_to_ssa(
        control,
        hierarchy_plan=hierarchy,
        identity_table={"self": (5,), "input": (7,), "result": (8,)},
        function_outputs=("result",),
        function_parameters=("self", "input"),
        self_value_id=5,
        field_ops=(("read", 6, 0),),
        field_count=1,
        field_names=("gain",),
        record_identity="Gain",
    )

    assert shortfalls == ()
    region_function = module.functions["planned_control__planned_region_0"]
    assert [value.id for value in region_function.args] == [7, 6]
    assert region_function.metadata["source_region_integral"] == {
        "schema": "turing.source-region-integral.v1",
        "owner": "planned_control",
        "plan_name": "region_0",
        "region_index": 0,
        "closure_id": -1,
        "identity_token_chain": (
            "source-region", "planned_control", "closure:-1", "region_0",
        ),
        "capture_value_ids": (7, 6),
        "output_value_ids": (8,),
    }
    assert [instruction.op for instruction in region_function.blocks["entry"].instrs] == [
        "Add"
    ]
    control_operations = [
        instruction.op
        for block in module.functions["planned_control"].blocks.values()
        for instruction in block.instrs
    ]
    assert "GetAttr" not in control_operations
    assert "getattr" not in control_operations
    assert "GetElementPtr" in control_operations
    assert "Load" in control_operations


def test_verified_source_region_link_replaces_only_an_exact_structural_abi():
    token_chain = (
        "source-region", "planned_control", "closure:1", "region_0",
    )
    argument = SSAValue(1, dtype="float64")
    result = SSAValue(2, dtype="float64")
    metadata = {
        "source_region_integral": {
            "schema": "turing.source-region-integral.v1",
            "identity_token_chain": token_chain,
        },
    }
    current = Function(
        "planned_control__planned_region_0", [argument],
        {"entry": BasicBlock("entry", [Instr("Neg", [argument], result)])},
        metadata=dict(metadata),
    )
    linked = Function(
        current.name, [argument],
        {"entry": BasicBlock("entry", [Instr("Abs", [argument], result)])},
        metadata=dict(metadata),
    )
    module = IRModule({current.name: current})
    outputs = {current.name: (result,)}
    linked_module = IRModule({linked.name: linked})

    receipts = link_verified_source_region_integrals(
        module,
        outputs,
        {token_chain: (
            linked_module,
            {linked.name: (result,)},
            {
                "status": "verified",
                "identity_token_chain": token_chain,
                "probe_count": 3,
            },
        )},
    )

    assert receipts == ({
        "ssa_function": current.name,
        "identity_token_chain": list(token_chain),
        "status": "linked",
        "probe_count": 3,
    },)
    assert module.functions[current.name].blocks["entry"].instrs[0].op == "Abs"


def test_stale_source_region_link_falls_back_to_current_source_lowering():
    token_chain = (
        "source-region", "planned_control", "closure:1", "region_0",
    )
    argument = SSAValue(1, dtype="float64")
    result = SSAValue(2, dtype="float64")
    current = Function(
        "planned_control__planned_region_0", [argument],
        {"entry": BasicBlock("entry", [Instr("Neg", [argument], result)])},
        metadata={"source_region_integral": {
            "identity_token_chain": token_chain,
        }},
    )
    stale_argument = SSAValue(99, dtype="float64")
    linked = Function(
        current.name, [stale_argument], current.blocks,
        metadata=current.metadata,
    )
    module = IRModule({current.name: current})

    receipt, = link_verified_source_region_integrals(
        module,
        {current.name: (result,)},
        {token_chain: (
            IRModule({linked.name: linked}),
            {linked.name: (result,)},
            {
                "status": "verified",
                "identity_token_chain": token_chain,
                "probe_count": 3,
            },
        )},
    )

    assert receipt["status"] == "fallback"
    assert receipt["reason"] == "input-abi-mismatch"
    assert module.functions[current.name] is current


def test_repository_module_merge_retains_numerical_and_object_surfaces():
    from src.transmogrifier.ssa import (
        BasicBlock,
        Function,
        SSAClassDefinition,
        SSAClassField,
        SSAClassTable,
        SSARecordDescriptor,
        SSARecordFieldDescriptor,
        SSARecordFieldStorage,
        SSARecordTable,
    )

    numerical = Function("numeric", [], {"entry": BasicBlock("entry", [])})
    method = Function("Thing__run", [], {"entry": BasicBlock("entry", [])})
    record = SSARecordDescriptor(
        10,
        "Thing",
        (SSARecordFieldDescriptor(
            "value",
            SSARecordFieldStorage.SCALAR,
            value_ids=(10,),
            offset=0,
        ),),
    )
    primary = IRModule({numerical.name: numerical})
    objects = IRModule(
        {method.name: method},
        class_table=SSAClassTable((SSAClassDefinition(
            "Thing", (SSAClassField("value", 0),), (),
        ),)),
        record_tables={method.name: SSARecordTable({10: record})},
    )

    merged = merge_repository_ssa_modules(primary, objects)

    assert set(merged.functions) == {"numeric", "Thing__run"}
    assert merged.class_table.by_identity("Thing") is not None
    assert merged.record_tables["Thing__run"].records[10] == record


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


def test_whole_object_region_signature_preserves_planner_value_shapes():
    from src.compiler.hierarchical_plan import PlanClosure, PlanLine

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
                (PlanLine.create("transpose", inputs=(10,), outputs=(11,)),),
                value_shapes=(
                    (10, (2, 3), "float64"),
                    (11, (3, 2), "float64"),
                ),
            ),
        ),
    )

    module, shortfalls, outputs = lower_control_sections_to_ssa(
        control,
        hierarchy_plan=hierarchy,
    )

    assert shortfalls == ()
    region = module.functions["planned_control__planned_region_4"]
    assert region.args[0].shape == (2, 3)
    assert outputs[region.name][0].shape == (3, 2)


def test_string_tokens_remain_typed_i64_without_numeric_projection():
    from src.compiler.hierarchical_plan import PlanClosure, PlanLine
    from src.compiler.string_table import string_token

    control = ControlProgram(
        StatementBlock(("__scheduled_region_0__",)),
        region_indices=(0,),
    )
    hierarchy = PlanClosure(
        "root",
        (),
        (
            PlanClosure(
                "region_0",
                (),
                (
                    PlanLine.create(
                        "Const",
                        outputs=(3,),
                        attributes={"value": "node-wasm"},
                    ),
                ),
            ),
        ),
    )

    module, shortfalls, outputs = lower_control_sections_to_ssa(
        control,
        hierarchy_plan=hierarchy,
    )
    token_instruction = module.functions[
        "planned_control__planned_region_0"
    ].blocks["entry"].instrs[0]
    emitted = emit_module(
        module,
        name="typed_string_record",
        outputs=outputs,
        extra_roots=tuple(module.functions),
    )

    assert shortfalls == ()
    assert token_instruction.op == "string_token"
    assert token_instruction.res.dtype == "int64"
    assert f"{string_token('node-wasm')}_c_int64_t" in emitted.source
    assert "transfer(" not in emitted.source


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


def test_nested_conditionals_overlay_and_lower_each_region_exactly_once():
    def markers(*regions):
        return SequenceBlock(tuple(
            StatementBlock((f"__scheduled_region_{region}__",))
            for region in regions
        ))

    def conditional(predicate, body, orelse, regions):
        return ControlProgram(
            ConditionalBlock(
                predicate,
                markers(*body),
                markers(*orelse),
                predicate_expression=ControlExpression(
                    "value", value_id=predicate
                ),
            ),
            region_indices=regions,
        )

    flat = ControlProgram(markers(0, 1, 2, 3), region_indices=(0, 1, 2, 3))
    outer = conditional(10, (0, 1), (2, 3), (0, 1, 2, 3))
    nested_true = conditional(11, (0,), (1,), (0, 1))
    nested_else = conditional(12, (2,), (3,), (2, 3))
    control = overlay_scheduled_control(
        (0, 1, 2, 3),
        (flat, outer, nested_true, nested_else),
        known_nesting={0: (1,), 1: (2, 3)},
    )

    function, shortfalls = lower_control_program_to_ssa(
        control,
        first_value_id=100,
        region_callees={index: f"structural_region_{index}" for index in range(4)},
        region_signatures={index: ((), ()) for index in range(4)},
    )
    instructions = [
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
    ]

    assert shortfalls == ()
    assert sum(item.op == "CondBr" for item in instructions) == 3
    calls = [
        item.attributes.get("callee")
        for item in instructions if item.op == "Call"
    ]
    assert calls == [
        "structural_region_0",
        "structural_region_1",
        "structural_region_2",
        "structural_region_3",
    ]


def test_cross_region_live_out_survives_local_consumption():
    from src.compiler.hierarchical_plan import PlanCall, PlanClosure, PlanLine

    control = ControlProgram(
        SequenceBlock((
            StatementBlock(("__scheduled_region_0__",)),
            StatementBlock(("__scheduled_region_1__",)),
        )),
        region_indices=(0, 1),
    )
    hierarchy = PlanClosure(
        "root",
        (10,),
        (
            PlanClosure(
                "region_0",
                (10,),
                (
                    PlanLine.create("Add", inputs=(10, 10), outputs=(11,)),
                    PlanLine.create("Mul", inputs=(11, 10), outputs=(12,)),
                    PlanLine.create("Sub", inputs=(10, 10), outputs=(14,)),
                    PlanLine.create("Add", inputs=(10, 10), outputs=(15,)),
                ),
            ),
            PlanClosure(
                "region_1",
                (10, 11),
                (PlanLine.create("Sub", inputs=(11, 10), outputs=(13,)),),
            ),
            PlanCall(
                99,
                PlanClosure("callee", (100,), ()),
                argument_value_ids=(999,),
                argument_bindings=((14, 100),),
            ),
        ),
    )

    module, shortfalls, outputs = lower_control_sections_to_ssa(
        control,
        hierarchy_plan=hierarchy,
        identity_table={"result": (12,)},
        function_outputs=("result",),
    )

    region_0 = module.functions["planned_control__planned_region_0"]
    control_function = module.functions["planned_control"]
    calls = [
        instruction
        for block in control_function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
    ]
    assert shortfalls == ()
    assert [value.id for value in outputs[region_0.name]] == [11, 12, 14]
    assert calls[0].attributes["output_ids"] == (11, 12, 14)
    assert [value.id for value in calls[1].args] == [10, 11]


def test_generator_sequence_plan_call_is_scheduled_before_result_consumer():
    from src.compiler.control_source import (
        ControlProgram, ControlSequenceMutation, LoopBlock, SequenceBlock,
        StatementBlock,
    )
    from src.compiler.hierarchical_plan import PlanCall, PlanClosure, PlanLine
    from src.compiler.precompile_to_ssa import _schedule_loop_callsites

    producer = LoopBlock(
        "item", "0", "n", "1", SequenceBlock(()),
        source_loop_node_id=204,
        sequence_mutations=(ControlSequenceMutation(
            215, "append", (108,), 205, policy="duplicates",
        ),),
    )
    consumer = StatementBlock(("__scheduled_region_29__",))
    hierarchy = PlanClosure("root", (), (
        PlanCall(
            206,
            PlanClosure("vector", (0,), ()),
            argument_bindings=((215, 0),),
            result_bindings=((9, 206),),
        ),
        PlanClosure(
            "region_29", (206,),
            (PlanLine.create("Add", inputs=(206, 206), outputs=(207,)),),
        ),
    ))

    scheduled, bindings = _schedule_loop_callsites(
        ControlProgram(SequenceBlock((consumer, producer))),
        hierarchy,
        {29: ((206,), (207,))},
        {29: ((206,), (207,))},
    )

    assert bindings[206] == ((215,), (206,))
    assert scheduled.root.blocks == (
        producer,
        StatementBlock(("__plan_callsite_206__",)),
        consumer,
    )


def test_mapping_update_lowers_to_existing_table_store_identity():
    control = ControlProgram(LoopBlock(
        "item", "0", "1", "1", SequenceBlock(()),
        sequence_mutations=(ControlSequenceMutation(
            20,
            "update",
            (21, 22),
            23,
            policy="unique",
            argument_kind="mapping_items",
        ),),
    ))

    function, shortfalls = lower_control_program_to_ssa(
        control,
        function_name="mapping_update",
        first_value_id=1000,
        sequence_declarations=((20, "unique", 2, True),),
        sequence_column_dtypes={20: ("int64", "float64")},
    )
    calls = [
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if str(instruction.op).casefold() == "call"
    ]

    assert shortfalls == ()
    assert any(
        instruction.attributes.get("ssa_sequence_operation") == "table_store"
        for instruction in calls
    )


def test_mapping_setdefault_returns_initialized_child_table_handle():
    control = ControlProgram(LoopBlock(
        "item", "0", "1", "1", SequenceBlock(()),
        sequence_mutations=(ControlSequenceMutation(
            20,
            "setdefault",
            (21, 22),
            23,
            policy="unique",
            argument_kind="mapping_setdefault",
        ),),
    ))

    function, shortfalls = lower_control_program_to_ssa(
        control,
        function_name="mapping_setdefault",
        first_value_id=1000,
        sequence_declarations=((20, "unique", 2, True),),
        sequence_column_dtypes={20: ("int64", "int")},
        resolved_sequence_schemas={20: ResolvedSequenceSchema(
            column_count=2,
            policy="unique",
            writable=True,
            retains_deleted_rows=True,
            nested_table=True,
            nested_value_dtype="int64",
        )},
    )
    instructions = [
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
    ]

    assert shortfalls == ()
    assert any(
        instruction.attributes.get("ssa_sequence_operation")
        == "setdefault_lookup"
        for instruction in instructions
    )
    assert any(
        instruction.attributes.get("ssa_sequence_operation") == "table_store"
        for instruction in instructions
    )
    assert any(
        instruction.res is not None
        and instruction.res.id == 23
        and str(instruction.op).casefold() == "phi"
        for instruction in instructions
    )


def test_mapping_pop_none_lowers_to_typed_optional_lookup_then_delete():
    control = ControlProgram(LoopBlock(
        "item", "0", "1", "1", SequenceBlock(()),
        sequence_mutations=(ControlSequenceMutation(
            20,
            "pop",
            (21, 22),
            23,
            policy="unique",
            argument_kind="mapping_pop_default_none",
        ),),
    ))

    function, shortfalls = lower_control_program_to_ssa(
        control,
        function_name="mapping_pop_none",
        first_value_id=1000,
        sequence_declarations=((20, "unique", 2, True),),
        sequence_column_dtypes={20: ("int64", "int64")},
        resolved_sequence_schemas={20: ResolvedSequenceSchema(
            column_count=2,
            policy="unique",
            writable=True,
            retains_deleted_rows=True,
            nested_table=False,
        )},
    )
    instructions = [
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
    ]
    calls = [
        instruction for instruction in instructions
        if str(instruction.op).casefold() == "call"
    ]
    defaults = [
        instruction for instruction in instructions
        if str(instruction.op).casefold() == "const"
        and instruction.attributes.get("value") == -1
    ]

    assert shortfalls == ()
    assert len(defaults) == 1
    assert defaults[0].res.dtype == "int64"
    assert any(
        instruction.attributes.get("ssa_sequence_operation") == "lookup"
        for instruction in calls
    )
    assert any(
        instruction.attributes.get("ssa_sequence_operation") == "table_delete"
        for instruction in calls
    )


def test_structural_dependency_chain_delays_early_sequence_mutation():
    from src.compiler.control_source import (
        ControlProgram, ControlSequenceMutation, LoopBlock, SequenceBlock,
        SequenceMutationBlock, StatementBlock,
    )
    from src.compiler.hierarchical_plan import PlanCall, PlanClosure, PlanLine
    from src.compiler.precompile_to_ssa import _schedule_loop_callsites

    producer = LoopBlock(
        "item", "0", "n", "1", SequenceBlock(()),
        source_loop_node_id=204,
        sequence_mutations=(ControlSequenceMutation(
            215, "append", (108,), 205, policy="duplicates",
        ),),
    )
    final_mutation = SequenceMutationBlock(ControlSequenceMutation(
        188, "append", (208,), 209, policy="duplicates",
    ))
    region_29 = StatementBlock(("__scheduled_region_29__",))
    region_30 = StatementBlock(("__scheduled_region_30__",))
    hierarchy = PlanClosure("root", (), (
        PlanCall(
            206,
            PlanClosure("vector", (0,), ()),
            argument_bindings=((215, 0),),
            result_bindings=((9, 206),),
        ),
        PlanClosure(
            "region_29", (206,),
            (PlanLine.create("Add", inputs=(206, 206), outputs=(207,)),),
        ),
        PlanClosure(
            "region_30", (207,),
            (PlanLine.create("Add", inputs=(207, 207), outputs=(208,)),),
        ),
    ))

    scheduled, _bindings = _schedule_loop_callsites(
        ControlProgram(SequenceBlock((
            final_mutation, region_29, region_30, producer,
        ))),
        hierarchy,
        {},
        {
            29: ((206,), (207,)),
            30: ((207,), (208,)),
        },
    )

    assert scheduled.root.blocks == (
        producer,
        StatementBlock(("__plan_callsite_206__",)),
        region_29,
        region_30,
        final_mutation,
    )


def test_joined_generator_singleton_appends_scalar_without_temporary_sequence():
    from src.compiler.control_source import (
        ControlProgram, ControlSequenceMutation, LoopBlock, SequenceBlock,
    )
    from src.compiler.precompile_to_ssa import lower_control_program_to_ssa

    control = ControlProgram(LoopBlock(
        "item", "0", "1", "1", SequenceBlock(()),
        sequence_mutations=(ControlSequenceMutation(
            215, "append", (108,), 205, policy="duplicates",
        ),),
    ))

    function, shortfalls = lower_control_program_to_ssa(
        control,
        function_name="joined_singleton",
        first_value_id=1000,
        sequence_declarations=((215, "duplicates", 1, True),),
        joined_sequence_ids=(215,),
        joined_singleton_values={108: 102},
    )

    calls = [
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if str(instruction.op).casefold() == "call"
    ]
    singleton = next(
        instruction for instruction in calls
        if instruction.attributes.get("ssa_sequence_operation")
        == "append_joined_singleton"
    )
    assert shortfalls == ()
    assert singleton.attributes["joined_source_sequence_id"] == 108
    assert singleton.attributes["joined_source_value_id"] == 102
    assert 108 not in function.metadata["sequence_table"].sequences


def test_region_outputs_consume_exact_dispatch_boundary():
    from src.compiler.hierarchical_plan import PlanClosure, PlanLine

    control = ControlProgram(
        StatementBlock(("__scheduled_region_0__",)),
        region_indices=(0,),
    )
    hierarchy = PlanClosure(
        "root",
        (10,),
        (PlanClosure(
            "region_0",
            (10,),
            (
                PlanLine.create("Add", inputs=(10, 10), outputs=(11,)),
                PlanLine.create("Mul", inputs=(11, 10), outputs=(12,)),
            ),
        ),),
    )

    module, shortfalls, outputs = lower_control_sections_to_ssa(
        control,
        hierarchy_plan=hierarchy,
        region_output_value_ids={0: (11,)},
    )

    assert shortfalls == ()
    region = module.functions["planned_control__planned_region_0"]
    assert [value.id for value in outputs[region.name]] == [11]

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


def test_control_ssa_names_authored_parameter_used_by_structured_predicate():
    control = ControlProgram(ConditionalBlock(
        0,
        SequenceBlock(()),
        SequenceBlock(()),
        predicate_expression=ControlExpression("value", value_id=0),
    ))

    function, shortfalls = lower_control_program_to_ssa(
        control,
        first_value_id=10,
        value_name_histories={"flag": (0,)},
        parameter_names=("flag",),
    )

    assert shortfalls == ()
    assert function.metadata["parameter_names"] == (("flag", 0),)


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


def test_nested_iterable_row_is_handle_stride_and_child_address():
    control = ControlProgram(
        LoopBlock(
            "iteration_9",
            "0",
            "__iterable_extent_40__",
            "1",
            StatementBlock(("__scheduled_region_4__",)),
        ),
        region_indices=(4,),
        iterable_bindings=((40, 42, "iteration_9"),),
    )

    function, shortfalls = lower_control_program_to_ssa(
        control,
        first_value_id=100,
        region_callees={4: "numerical_region_4"},
        region_signatures={4: ((42,), ())},
        region_value_meta={
            40: Meta((4,), "int", "cpu"),
            42: Meta((3,), "int", "cpu"),
        },
        nested_row_target_ids=(42,),
    )

    assert shortfalls == ()
    bindings = [
        instruction.attributes.get("binding")
        for block in function.blocks.values()
        for instruction in block.instrs
    ]
    assert bindings.count("nested_row_handle") == 2
    assert "nested_row_offset" in bindings
    assert "nested_row_base" in bindings
    child_table = function.metadata["nested_child_tables"]
    assert len(child_table) == 1
    assert child_table[0][:3] == ("iterable", 40, -1)
    assert len(child_table[0]) == 7
    assert set(child_table[0][3:]) <= {value.id for value in function.args}
    call = next(
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and instruction.attributes.get("callee") == "numerical_region_4"
    )
    row_address = next(
        instruction.res
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("binding") == "nested_row_base"
    )
    assert call.args == [row_address]


def test_explicit_iterable_binding_precedes_duplicate_closure_fallback():
    control = ControlProgram(
        LoopBlock(
            "iteration_9",
            "0",
            "__iterable_extent_40__",
            "1",
            StatementBlock(("__scheduled_region_4__",)),
        ),
        region_indices=(4,),
        iterable_bindings=((40, 42, "iteration_9"),),
        closure_iterable_bindings=((40, 42, "iteration_9", (41,)),),
    )

    function, shortfalls = lower_control_program_to_ssa(
        control,
        first_value_id=100,
        region_callees={4: "numerical_region_4"},
        region_signatures={4: ((42,), ())},
        region_value_meta={
            40: Meta((4,), "float64", "cpu"),
            41: Meta((), "float64", "cpu"),
            42: Meta((), "float64", "cpu"),
        },
    )

    assert shortfalls == ()
    producers = [
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None and instruction.res.id == 42
    ]
    assert len(producers) == 1
    assert producers[0].attributes["binding"] == "iterable"
    assert not any(
        instruction.attributes.get("binding") == "closure_iterable"
        for block in function.blocks.values()
        for instruction in block.instrs
    )


def test_empty_iterable_storage_inherits_exact_loop_target_dtype():
    control = ControlProgram(
        LoopBlock(
            "iteration_9",
            "0",
            "__iterable_extent_40__",
            "1",
            StatementBlock(("__scheduled_region_4__",)),
        ),
        region_indices=(4,),
        iterable_bindings=((40, 42, "iteration_9"),),
    )

    function, shortfalls = lower_control_program_to_ssa(
        control,
        first_value_id=100,
        region_callees={4: "numerical_region_4"},
        region_signatures={4: ((42,), ())},
        region_value_meta={
            40: Meta((0,), "bool", "cpu"),
            42: Meta((), "float64", "cpu"),
        },
    )

    assert shortfalls == ()
    iterable = next(value for value in function.args if value.id == 40)
    loaded = next(
        instruction.res
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Load"
        and instruction.attributes.get("binding") == "iterable"
    )
    assert iterable.dtype == "float64"
    assert iterable.accounting["iterable_target_value_id"] == 42
    assert loaded.dtype == "float64"


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


def test_nested_loop_final_value_drives_enclosing_carried_phi():
    inner = LoopBlock(
        "inner",
        "0",
        "4",
        "1",
        StatementBlock(("__scheduled_region_3__",)),
        carried_aliases=((20, 10),),
    )
    control = ControlProgram(
        LoopBlock(
            "outer",
            "0",
            "4",
            "1",
            inner,
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
    outer_phi = next(
        instruction
        for instruction in function.blocks["loop_header"].instrs
        if instruction.attributes.get("binding") == "loop_carried"
    )
    inner_phi = next(
        instruction
        for instruction in function.blocks["loop_header.1"].instrs
        if instruction.attributes.get("binding") == "loop_carried"
    )
    assert outer_phi.args[1] is inner_phi.res
    assert any(
        instruction.res is outer_phi.args[1]
        for block in function.blocks.values()
        for instruction in block.instrs
    )


def test_dynamic_arithmetic_loop_bound_lowers_to_ssa_values():
    control = ControlProgram(
        LoopBlock(
            "iteration",
            "0",
            "(((u_control_66 - u_control_80) / u_control_48) + 1)",
            "1",
            StatementBlock(("__scheduled_region_3__",)),
        ),
        region_indices=(3,),
    )

    function, shortfalls = lower_control_program_to_ssa(
        control,
        first_value_id=100,
        region_signatures={3: ((), ())},
    )

    assert shortfalls == ()
    entry_ops = [instruction.op for instruction in function.blocks["entry"].instrs]
    assert "Sub" in entry_ops
    assert "Div" in entry_ops
    assert "Add" in entry_ops


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
            source_loop_node_id=178,
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
        and instruction.attributes.get("source_loop_node_id") == 178
        for instruction in function.blocks["while_header"].instrs
    )
    assert function.metadata["recursion_table"][4]["loops"][0][
        "domain"
    ] == "condition"


def test_terminal_loop_return_runs_after_predicated_sequence_effects():
    predicate = ControlExpression("value", value_id=10)
    argument = ControlExpression(
        "bitand",
        (
            ControlExpression("value", value_id=0),
            ControlExpression("const", value_id=11, literal=127),
        ),
        value_id=12,
    )
    control = ControlProgram(WhileBlock(
        predicate_value_id=20,
        condition=SequenceBlock(()),
        body=SequenceBlock(()),
        predicate_expression=ControlExpression(
            "const", value_id=20, literal=True
        ),
        sequence_mutations=(ControlSequenceMutation(
            sequence_value_id=30,
            operator="append",
            argument_value_ids=(12,),
            effect_node_id=40,
            policy="duplicates",
            predicate_expression=predicate,
            argument_expressions=(argument,),
        ),),
        terminal_controls=(LoopControlBlock(
            "break",
            predicate_value_id=10,
            expect_true=False,
            predicate_expression=predicate,
            source_action="loop-return",
        ),),
    ))

    function, shortfalls = lower_control_program_to_ssa(control)

    assert shortfalls == ()
    append_block = next(
        block for block in function.blocks.values()
        if any(
            instruction.attributes.get("ssa_sequence_operation") == "append"
            for instruction in block.instrs
        )
    )
    return_block = next(
        block for block in function.blocks.values()
        if any(
            instruction.attributes.get("source_control") == "loop-return"
            for instruction in block.instrs
        )
    )
    assert return_block.name.startswith("sequence_mutation_merge")
    assert return_block.name in append_block.successors
    assert 12 not in {value.id for value in function.args}


def test_lexical_sequence_mutation_block_stays_inside_conditional_arm():
    mutation = ControlSequenceMutation(
        sequence_value_id=30,
        operator="append",
        argument_value_ids=(12,),
        effect_node_id=40,
        policy="duplicates",
    )
    control = ControlProgram(ConditionalBlock(
        predicate_value_id=10,
        body=SequenceBlock((SequenceMutationBlock(mutation),)),
        orelse=SequenceBlock(()),
        predicate_expression=ControlExpression(
            "value", value_id=10
        ),
        source_node_id=50,
    ))

    function, shortfalls = lower_control_program_to_ssa(control)

    assert shortfalls == ()
    append_site = next(
        (block.name, instruction)
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("source_effect_node_id") == 40
    )
    assert append_site[0].startswith("if_true")
    assert append_site[1].attributes["ssa_sequence_operation"] == "append"


def test_conditional_sequence_assignment_replaces_one_resident_arena():
    control = ControlProgram(ConditionalBlock(
        predicate_value_id=10,
        body=SequenceBlock(()),
        orelse=SequenceBlock(()),
        predicate_expression=ControlExpression("value", value_id=10),
        carried_sequence_aliases=((31, 30, 30, 32),),
        source_node_id=50,
    ))

    function, shortfalls = lower_control_program_to_ssa(
        control,
        sequence_initializations=(
            (30, "duplicates", 1),
            (31, "duplicates", 1),
        ),
        sequence_declarations=(
            (30, "duplicates", 1, True),
            (31, "duplicates", 1, False),
        ),
        value_aliases={32: 30},
        plan_callsite_bindings={99: ((12,), (31,))},
    )

    assert shortfalls == ()
    replacements = [
        (block.name, instruction)
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("ssa_sequence_operation") == "replace"
    ]
    assert len(replacements) == 1
    assert replacements[0][0].startswith("if_true")
    planned_call = next(
        (block.name, instruction)
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("plan_callsite_id") == 99
    )
    assert planned_call[0].startswith("if_true")
    assert any(
        instruction.attributes.get("binding")
        == "ssa_sequence_replace_clear"
        for block in function.blocks.values()
        for instruction in block.instrs
    )
    assert not any(
        instruction.op == "Phi"
        and instruction.attributes.get("binding") == "conditional_carried"
        for block in function.blocks.values()
        for instruction in block.instrs
    )


def test_sequence_first_or_default_query_is_resident_and_receipted():
    control = ControlProgram(SequenceBlock((
        SequenceMutationBlock(ControlSequenceMutation(
            sequence_value_id=30,
            operator="append",
            argument_value_ids=(12,),
            effect_node_id=40,
            policy="duplicates",
        )),
        SequenceQueryBlock(
            result_value_id=50,
            sequence_value_id=30,
            operation="first_or_default",
            default_value_id=13,
            source_call_node_id=41,
            extraction_identity="builtins.next",
            result_alias_ids=(51,),
        ),
    )))

    function, shortfalls = lower_control_program_to_ssa(
        control,
        sequence_initializations=((30, "duplicates", 1),),
        sequence_declarations=((30, "duplicates", 1, True),),
    )

    assert shortfalls == ()
    query_phi = next(
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None and instruction.res.id == 50
    )
    assert query_phi.op == "Phi"
    assert query_phi.attributes["extraction_identity"] == "builtins.next"
    assert 50 not in {value.id for value in function.args}
    assert 51 not in {value.id for value in function.args}


def test_fixed_width_sequence_append_passes_every_row_column():
    control = ControlProgram(SequenceBlock((
        SequenceMutationBlock(ControlSequenceMutation(
            sequence_value_id=30,
            operator="append",
            argument_value_ids=(12, 13),
            effect_node_id=40,
            policy="duplicates",
            argument_kind="row",
        )),
    )))

    function, shortfalls = lower_control_program_to_ssa(
        control,
        sequence_initializations=((30, "duplicates", 2),),
        sequence_declarations=((30, "duplicates", 2, True),),
    )

    assert shortfalls == ()
    append = next(
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("ssa_sequence_operation") == "append"
    )
    assert tuple(value.id for value in append.args[-2:]) == (12, 13)


def test_local_sequence_lifetime_resets_without_erasing_source_sequence():
    function, shortfalls = lower_control_program_to_ssa(
        ControlProgram(SequenceBlock(())),
        sequence_declarations=(
            (30, "duplicates", 1, True),
            (40, "duplicates", 1, True),
        ),
        source_sequence_ids=(30,),
    )

    assert shortfalls == ()
    initialized = {
        int(instruction.attributes["sequence_id"])
        for instruction in function.blocks["entry"].instrs
        if instruction.attributes.get("binding")
        == "ssa_local_sequence_initialize"
    }
    assert initialized == {40}


def test_compile_time_mapping_initializes_a_typed_local_table():
    function, shortfalls = lower_control_program_to_ssa(
        ControlProgram(SequenceBlock(())),
        sequence_declarations=((30, "unique", 2, False),),
        sequence_initializations=((
            30, "literal_table=((101, 127), (202, 126))", 2,
        ),),
        sequence_column_dtypes={30: ("int64", "int64")},
    )

    assert shortfalls == ()
    entry = function.blocks["entry"].instrs
    literal_stores = [
        instruction for instruction in entry
        if instruction.attributes.get("binding")
        == "ssa_sequence_literal_table"
        and instruction.op == "Store"
    ]
    assert len(literal_stores) == 4
    assert all(instruction.args[0].dtype == "int64"
               for instruction in literal_stores)
    assert any(
        instruction.attributes.get("binding")
        == "ssa_sequence_literal_table_length"
        for instruction in entry
    )


def test_literal_table_initialization_agrees_with_unique_declaration():
    schemas, shortfalls = resolve_sequence_schemas(({
        "sequence_declarations": ((30, "unique", 2, False),),
        "sequence_initializations": ((
            30, "literal_table=((101, 127),)", 2,
        ),),
    },))

    assert shortfalls == ()
    assert schemas[30].policy == "unique"
    assert schemas[30].column_count == 2


def test_resident_iterable_loop_uses_logical_length_not_storage_extent():
    control = ControlProgram(LoopBlock(
        induction="iteration_9",
        start="0",
        stop="__iterable_extent_30__",
        step="1",
        body=SequenceBlock(()),
    ))

    function, shortfalls = lower_control_program_to_ssa(
        control,
        sequence_declarations=((30, "duplicates", 1, False),),
        source_sequence_ids=(30,),
    )

    assert shortfalls == ()
    length_loads = [
        instruction
        for instruction in function.blocks["entry"].instrs
        if instruction.op == "Load"
        and instruction.attributes.get("binding")
        == "resident_iterable_length"
    ]
    assert len(length_loads) == 1
    assert length_loads[0].attributes["sequence_id"] == 30
    assert not any(
        instruction.op == "Call"
        and instruction.attributes.get("binding") == "iterable_extent"
        for block in function.blocks.values()
        for instruction in block.instrs
    )


def test_sequence_while_condition_reloads_length_at_latch():
    predicate = ControlExpression(
        "sequence_nonempty",
        (ControlExpression("value", value_id=10),),
        value_id=10,
        literal=False,
    )
    control = ControlProgram(
        WhileBlock(
            predicate_value_id=10,
            condition=SequenceBlock(()),
            body=SequenceBlock(()),
            predicate_expression=predicate,
        )
    )

    function, shortfalls = lower_control_program_to_ssa(control)

    assert shortfalls == ()
    length_loads = [
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Load"
        and instruction.attributes.get("binding") == "ssa_sequence_length"
    ]
    predicates = [
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Gt"
        and instruction.attributes.get("binding") == "sequence_nonempty"
    ]
    assert len(length_loads) == 2
    assert len(predicates) == 2
    assert predicates[1].res is function.blocks["while_header"].instrs[0].args[1]
