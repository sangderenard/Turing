import ast
import inspect
import textwrap
from types import SimpleNamespace

import networkx as nx

from src.compiler.hierarchical_plan import (
    PlanCall,
    PlanClosure,
    PlanLine,
    assign_hierarchy_ids,
    plan_region_to_ssa_instrs,
    reduce_hierarchy_identities,
    render_plan_ascii,
)
from src.compiler.glsl_deployment_strategy import (
    PlannedOperatorImplementation,
    _build_planned_operator_implementations,
    _build_shell_hierarchy_plan,
    _planned_operator_node_ids,
)
from src.common.tensors.accelerator_backends.c_primitive_program import (
    CapturedFusedProgram,
)
from src.common.tensors.accelerator_backends.aot_compile import (
    compile_ast_aot,
)
from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.control_source import (
    CallBlock,
    ControlExpression,
    ControlTarget,
    StatementBlock,
    StreamPublishBlock,
    WhileBlock,
    render_control_block,
    ControlProgram,
    LoopBlock,
    SequenceBlock,
)
from src.compiler.hierarchical_control import compose_hierarchical_control


def test_shell_hierarchy_plan_lists_mixed_operator_without_capture():
    process = nx.DiGraph()
    process.add_node(
        0,
        type="Input",
        attributes={"binding_name": "subject"},
        tensor={"shape": (1, 4), "dtype": "float32", "device": "glsl"},
    )
    process.add_node(
        1,
        type="GetAttr",
        op="GetAttr",
        parents=((0, "base"),),
        attributes={"attribute": "header"},
        tensor={"shape": (1, 2), "dtype": "float32", "device": "glsl"},
    )
    process.add_edge(0, 1)
    region = nx.DiGraph()
    region.graph.update({
        "deployment_nodes": (1,),
        "deployment_inputs": (0,),
    })
    shell = SimpleNamespace(
        process_graph=SimpleNamespace(G=process),
        dispatch_subgraphs=(SimpleNamespace(G=region),),
        callsite_function_shells={},
        loop_plans=(),
        shell_control_program=None,
    )

    plan = _build_shell_hierarchy_plan(shell)
    planned_region = plan.items[0]
    line = planned_region.items[0]

    assert line.opcode == "GetAttr"
    assert line.inputs == (0,)
    assert line.input_roles == ("base",)
    assert line.outputs == (1,)
    assert dict(line.attributes) == {
        "attribute": "header",
        "region": 0,
    }
    assert _planned_operator_node_ids(plan) == (1,)
    assert planned_region.value_shapes == (
        (1, (1, 2), "float32"),
        (0, (1, 4), "float32"),
    )


def test_control_loop_backedges_are_exact_hierarchy_dependencies():
    process = nx.DiGraph()
    process.add_node(
        1,
        type="Input",
        attributes={"binding_name": "initial"},
    )
    process.add_node(
        2,
        type="Add",
        op="Add",
        parents=((1, "lhs"), (1, "rhs")),
        attributes={},
    )
    process.add_edge(1, 2)
    region = nx.DiGraph()
    region.graph.update({
        "deployment_nodes": (2,),
        "deployment_inputs": (1,),
    })
    control = ControlProgram(
        LoopBlock(
            "iteration_0",
            "0",
            "1",
            "1",
            StatementBlock(("__scheduled_region_0__",)),
            carried_aliases=((2, 1),),
        ),
        region_indices=(0,),
    )
    shell = SimpleNamespace(
        process_graph=SimpleNamespace(G=process),
        dispatch_subgraphs=(SimpleNamespace(G=region),),
        callsite_function_shells={},
        loop_plans=(),
        shell_control_program=control,
    )

    plan = _build_shell_hierarchy_plan(shell)

    assert plan.captures == (1, 2)


def test_loop_carried_initializer_is_region_capture_not_body_constant():
    process = nx.DiGraph()
    process.add_node(
        1,
        type="Constant",
        op="const",
        parents=(),
        attributes={"value": 0},
    )
    process.add_node(
        2,
        type="Add",
        op="Add",
        parents=((1, "lhs"), (1, "rhs")),
        attributes={},
    )
    process.add_edge(1, 2)
    region = nx.DiGraph()
    region.add_nodes_from(process.nodes(data=True))
    region.add_edge(1, 2)
    region.graph.update({
        "deployment_nodes": (1, 2),
        "deployment_inputs": (),
        "deployment_outputs": (1, 2),
    })
    loop = SimpleNamespace(
        loop=SimpleNamespace(carried_bindings=(("iters", 1, 2),)),
    )
    shell = SimpleNamespace(
        process_graph=SimpleNamespace(G=process),
        dispatch_subgraphs=(SimpleNamespace(G=region),),
        callsite_function_shells={},
        loop_plans=(loop,),
        shell_control_program=ControlProgram(
            LoopBlock(
                "iteration_0", "0", "1", "1",
                StatementBlock(("__scheduled_region_0__",)),
                carried_aliases=((2, 1),),
            ),
            region_indices=(0,),
        ),
    )

    plan = _build_shell_hierarchy_plan(shell)
    planned_region = next(
        item for item in plan.items
        if isinstance(item, PlanClosure) and item.name == "region_0"
    )

    assert planned_region.captures == (1,)
    assert [line.outputs for line in planned_region.items] == [(2,)]

def test_region_pursues_structural_constant_expression_captures_into_source():
    process = nx.DiGraph()
    process.add_node(0, type="Constant", op="const", attributes={"value": 2})
    process.add_node(
        1, type="neg", op="neg", parents=((0, "operand"),), attributes={}
    )
    process.add_node(2, type="Input", op="input", attributes={"binding_name": "x"})
    process.add_node(
        3,
        type="transpose",
        op="transpose",
        parents=((2, "operand"), (1, "arg0")),
        attributes={"tensor": "transpose"},
    )
    process.add_edges_from(((0, 1), (1, 3), (2, 3)))
    region = nx.DiGraph()
    region.graph.update({"deployment_nodes": (3,), "deployment_inputs": (2, 1)})
    shell = SimpleNamespace(
        process_graph=SimpleNamespace(G=process),
        dispatch_subgraphs=(SimpleNamespace(G=region),),
        callsite_function_shells={},
        loop_plans=(),
        shell_control_program=None,
    )

    planned_region = _build_shell_hierarchy_plan(shell).items[0]

    assert planned_region.captures == (2,)
    assert planned_region.items[0].opcode == "Const"
    assert planned_region.items[0].outputs == (1,)
    assert dict(planned_region.items[0].attributes)["value"] == -2


def test_structural_constant_state_does_not_use_python_object_sentinel():
    source = textwrap.dedent(inspect.getsource(_build_shell_hierarchy_plan))
    tree = ast.parse(source)

    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "object"
        for node in ast.walk(tree)
    )


def test_structural_none_constant_is_distinct_from_unavailable_state():
    process = nx.DiGraph()
    process.add_node(0, type="Constant", op="const", attributes={"value": None})
    process.add_node(1, type="Constant", op="const", attributes={"value": 2})
    process.add_node(
        2,
        type="Tuple",
        op="Tuple",
        parents=((0, "item0"), (1, "item1")),
        attributes={},
    )
    process.add_node(3, type="Input", op="input", attributes={"binding_name": "x"})
    process.add_node(
        4,
        type="reshape",
        op="reshape",
        parents=((3, "operand"), (2, "arg0")),
        attributes={"tensor": "reshape"},
    )
    process.add_edges_from(((0, 2), (1, 2), (2, 4), (3, 4)))
    region = nx.DiGraph()
    region.graph.update({"deployment_nodes": (4,), "deployment_inputs": (3, 2)})
    shell = SimpleNamespace(
        process_graph=SimpleNamespace(G=process),
        dispatch_subgraphs=(SimpleNamespace(G=region),),
        callsite_function_shells={},
        loop_plans=(),
        shell_control_program=None,
    )

    planned_region = _build_shell_hierarchy_plan(shell).items[0]

    assert planned_region.captures == (3,)
    assert dict(planned_region.items[0].attributes)["value"] == (None, 2)


def test_structural_constant_capture_walk_is_iterative_for_deep_source():
    process = nx.DiGraph()
    process.add_node(0, type="Constant", op="const", attributes={"value": 7})
    # Above CPython's default recursion limit: the former recursive evaluator
    # fails here, while the explicit work stack remains bounded by heap memory.
    depth = 1050
    for node_id in range(1, depth + 1):
        process.add_node(
            node_id,
            type="pos",
            op="pos",
            parents=((node_id - 1, "operand"),),
            attributes={},
        )
        process.add_edge(node_id - 1, node_id)
    data_id = depth + 1
    result_id = depth + 2
    process.add_node(
        data_id, type="Input", op="input", attributes={"binding_name": "x"}
    )
    process.add_node(
        result_id,
        type="reshape",
        op="reshape",
        parents=((data_id, "operand"), (depth, "arg0")),
        attributes={"tensor": "reshape"},
    )
    process.add_edges_from(((data_id, result_id), (depth, result_id)))
    region = nx.DiGraph()
    region.graph.update({
        "deployment_nodes": (result_id,),
        "deployment_inputs": (data_id, depth),
    })
    shell = SimpleNamespace(
        process_graph=SimpleNamespace(G=process),
        dispatch_subgraphs=(SimpleNamespace(G=region),),
        callsite_function_shells={},
        loop_plans=(),
        shell_control_program=None,
    )

    planned_region = _build_shell_hierarchy_plan(shell).items[0]

    assert planned_region.captures == (data_id,)
    assert dict(planned_region.items[0].attributes)["value"] == 7


def test_feedback_capture_is_not_misclassified_as_constant_source():
    process = nx.DiGraph()
    process.add_node(0, type="pos", op="pos", parents=((1, "operand"),), attributes={})
    process.add_node(1, type="pos", op="pos", parents=((0, "operand"),), attributes={})
    process.add_node(2, type="Input", op="input", attributes={"binding_name": "x"})
    process.add_node(
        3,
        type="add",
        op="add",
        parents=((2, "lhs"), (0, "rhs")),
        attributes={},
    )
    process.add_edges_from(((0, 1), (1, 0), (2, 3), (0, 3)))
    region = nx.DiGraph()
    region.graph.update({"deployment_nodes": (3,), "deployment_inputs": (2, 0)})
    shell = SimpleNamespace(
        process_graph=SimpleNamespace(G=process),
        dispatch_subgraphs=(SimpleNamespace(G=region),),
        callsite_function_shells={},
        loop_plans=(),
        shell_control_program=None,
    )

    # Retained-loop planning supplies a stable condensation level table. This
    # is the same contract _dependency_order uses for real feedback graphs.
    shell.process_graph.levels = {0: 0, 1: 0, 2: 0, 3: 1}
    process.graph["recursion_table"] = {
        0: {"members": (0, 1), "kind": "irreducible_recursion"}
    }
    planned_region = _build_shell_hierarchy_plan(shell).items[0]

    assert planned_region.captures == (2, 0)


def test_planned_operators_attach_existing_tensor_kernel_and_plain_lowering():
    hierarchy = PlanClosure(
        "mixed",
        (0,),
        (
            PlanClosure(
                "region_0",
                (0,),
                (
                    PlanLine.create("Add", inputs=(0,), outputs=(1,)),
                    PlanLine.create(
                        "GetAttr", inputs=(1,), outputs=(2,)
                    ),
                ),
            ),
        ),
    )
    captured = CapturedFusedProgram(
        FusedProgram(
            version=1,
            feeds={0},
            steps=[OpStep(7, "add", [0], result_id=1)],
            outputs={"value_1": 1},
        ),
        {},
    )

    implementations = _build_planned_operator_implementations(
        hierarchy,
        {0: captured},
        (2,),
    )

    assert implementations == {
        0: (
            PlannedOperatorImplementation(1, "fused", (7,)),
            PlannedOperatorImplementation(2, "plain"),
        ),
    }

    instructions = plan_region_to_ssa_instrs(hierarchy.items[0])
    assert [instruction.op for instruction in instructions] == [
        "Add",
        "GetAttr",
    ]
    assert instructions[1].args[0].id == 1
    assert instructions[1].res.id == 2


def test_python_identity_callables_lower_to_typed_ssa_value_flow():
    region = PlanClosure(
        "region_0",
        (0, 1),
        (
            PlanLine.create(
                "float",
                inputs=(0, 1),
                outputs=(2,),
                input_roles=("callee", "arg:0"),
            ),
            PlanLine.create(
                "tensor",
                inputs=(2,),
                outputs=(3,),
                input_roles=("arg:0",),
                attributes={"ensures_schema_type": "AbstractTensor"},
            ),
        ),
        value_shapes=(
            (0, (), "opaque_ref"),
            (1, (), "int"),
            (2, (), "float64"),
            (3, (), "float64"),
        ),
    )

    instructions = plan_region_to_ssa_instrs(region)

    assert [instruction.op for instruction in instructions] == ["Cast", "Cast"]
    assert [argument.id for argument in instructions[0].args] == [1]
    assert instructions[0].arg_roles == ["arg:0"]
    assert instructions[0].res.dtype == "float64"
    assert instructions[0].attributes["source_operator"] == "float"
    assert [argument.id for argument in instructions[1].args] == [2]
    assert instructions[1].attributes["ensures_schema_type"] == "AbstractTensor"


def test_python_variadic_max_and_clamp_decompose_to_binary_ssa_flow():
    region = PlanClosure(
        "region_0",
        (0, 1, 2, 3, 4, 5),
        (
            PlanLine.create(
                "max", inputs=(0, 1, 2), outputs=(6,),
                input_roles=("arg:0", "arg:1", "arg:2"),
            ),
            PlanLine.create(
                "clamp", inputs=(6, 4, 5), outputs=(7,),
                input_roles=("operand", "kw:min", "kw:max"),
            ),
        ),
        value_shapes=tuple(
            (value_id, (), "float64") for value_id in range(8)
        ),
    )

    instructions = plan_region_to_ssa_instrs(region)

    assert [instruction.op for instruction in instructions] == [
        "Max", "Max", "Max", "Min",
    ]
    assert [argument.id for argument in instructions[1].args][1] == 2
    assert instructions[1].res.id == 6
    assert instructions[-1].res.id == 7
    temporary_ids = {
        instructions[0].res.id,
        instructions[2].res.id,
    }
    assert temporary_ids.isdisjoint(range(8))


def test_graph_only_aot_does_not_require_mutable_runtime_feed():
    compilation = compile_ast_aot(
        "def page(subject):\n"
        "    value = subject + 1\n"
        "    return value\n",
        "page",
        {},
        precompile_only=True,
        mutable_parameters=("subject",),
        checkpoint=False,
    )

    assert compilation.compiled_shell_program.program.steps == []
    assert compilation.shell_control_program.region_indices == (0,)
    assert compilation.public_input_value_ids == {"subject": 0}
    assert tuple(compilation.planned_operator_implementations) == (0,)
    assert compilation.hierarchy_plan.name == "page"


def test_hierarchy_ids_separate_equal_local_value_ids_across_calls():
    child = PlanClosure(
        "child",
        (0,),
        (PlanLine.create("Mul", inputs=(0, 1), outputs=(2,)),),
    )
    root = PlanClosure(
        "root",
        (0,),
        (
            PlanLine.create("Add", inputs=(0, 1), outputs=(2,)),
            PlanCall(
                7,
                child,
                (2,),
                (3,),
                argument_bindings=((2, 0),),
                result_bindings=((2, 3),),
            ),
        ),
    )

    planned, values = assign_hierarchy_ids(root)
    call = planned.items[1]

    assert isinstance(call, PlanCall)
    assert planned.closure_id != call.callee.closure_id
    assert call.argument_bindings == ((2, 0),)
    assert call.result_bindings == ((2, 3),)
    assert values.global_id(planned.closure_id, 0) != values.global_id(
        call.callee.closure_id, 0
    )
    assert values.global_id(planned.closure_id, 2) == values.global_id(
        call.callee.closure_id, 0
    )
    assert values.global_id(call.callee.closure_id, 2) == values.global_id(
        planned.closure_id, 3
    )
    rendered = render_plan_ascii(planned)
    assert "call #7" in rendered
    assert "arg-bind=[2->0]" in rendered
    assert "result-bind=[2->3]" in rendered
    assert "closure child id=1" in rendered


def test_post_hierarchy_identity_reduction_removes_only_proven_call_boundary():
    identity = PlanClosure("require_tensor", (0,), (), 1)
    compute = PlanClosure(
        "compute",
        (0,),
        (PlanLine.create("Mul", inputs=(0, 0), outputs=(1,)),),
        2,
    )
    root = PlanClosure(
        "root",
        (0,),
        (
            PlanCall(
                10,
                identity,
                (0,),
                (1,),
                argument_bindings=((0, 0),),
                result_bindings=((0, 1),),
            ),
            PlanCall(
                11,
                compute,
                (1,),
                (2,),
                argument_bindings=((1, 0),),
                result_bindings=((1, 2),),
            ),
        ),
        0,
    )

    reduced = reduce_hierarchy_identities(root, {1})

    assert reduced.collapsed_callsites == (10,)
    assert reduced.rounds == 1
    assert tuple(
        item.callsite_id
        for item in reduced.root.items
        if isinstance(item, PlanCall)
    ) == (11,)


def test_typed_call_block_lowers_as_nested_compiled_control():
    block = CallBlock(
        9,
        StatementBlock(("value += 1;",)),
        argument_bindings=((2, 0),),
        result_bindings=((1, 3),),
    )
    assert render_control_block(block, ControlTarget.GLSL) == (
        "value += 1;",
    )


def test_direct_return_expression_receives_public_output_identity():
    module = ast.parse(
        "def direct(x):\n"
        "    return x + 1\n"
    )
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)
    direct = next(
        entry.graph
        for entry in graph.function_table
        if entry.name == "direct"
    )
    assert direct.G.graph["function_outputs"] == ("result_0",)
    output_id = direct.G.graph["identity_table"]["result_0"][-1]
    assert direct.G.nodes[output_id]["op"].lower() == "add"


def test_hierarchical_control_inserts_call_inside_owner_loop_scope():
    child = PlanClosure(
        "child",
        (0,),
        (PlanClosure("region_0", (0,), (), -1),),
    )
    root = PlanClosure(
        "root",
        (0,),
        (
            PlanClosure("region_0", (0,), (), -1),
            PlanCall(
                8,
                child,
                (0,),
                (2,),
                argument_bindings=((0, 0),),
                result_bindings=((1, 2),),
            ),
            PlanClosure("region_1", (2,), (), -1),
        ),
    )
    planned, table = assign_hierarchy_ids(root)
    child_plan = planned.items[1].callee
    controls = {
        planned.closure_id: ControlProgram(
            LoopBlock(
                "i",
                "0",
                "4",
                "1",
                SequenceBlock((
                    StatementBlock(("__scheduled_region_0__",)),
                    StatementBlock(("__scheduled_region_1__",)),
                )),
            ),
            (0, 1),
        ),
        child_plan.closure_id: ControlProgram(
            StatementBlock(("__scheduled_region_0__",)),
            (0,),
        ),
    }
    composed = compose_hierarchical_control(planned, controls, table)
    loop = composed.program.root
    assert isinstance(loop, LoopBlock)
    second = loop.body.blocks[1]
    assert isinstance(second, SequenceBlock)
    assert isinstance(second.blocks[0], CallBlock)
    assert second.blocks[0].argument_bindings[0][0] == (
        second.blocks[0].argument_bindings[0][1]
    )
    assert composed.region_correlations == (
        (planned.closure_id, 0, 0),
        (planned.closure_id, 1, 1),
        (child_plan.closure_id, 0, 2),
    )


def test_hierarchical_call_precedes_trailing_publication_in_loop_scope():
    child = PlanClosure(
        "encode",
        (0,),
        (PlanClosure("region_0", (0,), (), -1),),
    )
    root = PlanClosure(
        "record",
        (0,),
        (
            PlanClosure("region_0", (0,), (), -1),
            PlanCall(
                8,
                child,
                (0,),
                (2,),
                argument_bindings=((0, 0),),
                result_bindings=((1, 2),),
            ),
        ),
    )
    planned, table = assign_hierarchy_ids(root)
    child_plan = planned.items[1].callee
    controls = {
        planned.closure_id: ControlProgram(
            LoopBlock(
                "i",
                "0",
                "4",
                "1",
                SequenceBlock((
                    StatementBlock(("__scheduled_region_0__",)),
                    StreamPublishBlock(0, 2, None),
                )),
            ),
            (0,),
        ),
        child_plan.closure_id: ControlProgram(
            StatementBlock(("__scheduled_region_0__",)),
            (0,),
        ),
    }

    composed = compose_hierarchical_control(planned, controls, table)
    loop = composed.program.root
    assert isinstance(loop, LoopBlock)
    trailing = loop.body.blocks[1]
    assert isinstance(trailing, SequenceBlock)
    assert isinstance(trailing.blocks[0], CallBlock)
    assert isinstance(trailing.blocks[1], StreamPublishBlock)


def test_hierarchical_call_without_later_region_stays_in_enclosing_loop():
    child = PlanClosure(
        "encode",
        (0,),
        (PlanClosure("region_0", (0,), (), -1),),
    )
    root = PlanClosure(
        "record",
        (0,),
        (
            PlanClosure("region_0", (0,), (), -1),
            PlanCall(
                8,
                child,
                (0,),
                (2,),
                argument_bindings=((0, 0),),
                result_bindings=((1, 2),),
                enclosing_loop_ids=(34,),
            ),
        ),
    )
    planned, table = assign_hierarchy_ids(root)
    child_plan = planned.items[1].callee
    controls = {
        planned.closure_id: ControlProgram(
            LoopBlock(
                "iteration_34",
                "0",
                "4",
                "1",
                StatementBlock(("__scheduled_region_0__",)),
            ),
            (0,),
        ),
        child_plan.closure_id: ControlProgram(
            StatementBlock(("__scheduled_region_0__",)),
            (0,),
        ),
    }

    composed = compose_hierarchical_control(planned, controls, table)
    loop = composed.program.root
    assert isinstance(loop, LoopBlock)
    assert isinstance(loop.body, SequenceBlock)
    assert isinstance(loop.body.blocks[-1], CallBlock)


def test_hierarchical_call_without_later_region_stays_in_enclosing_while():
    child = PlanClosure(
        "advance",
        (0,),
        (PlanClosure("region_0", (0,), (), -1),),
    )
    root = PlanClosure(
        "superstep",
        (0,),
        (
            PlanClosure("region_0", (0,), (), -1),
            PlanCall(
                8,
                child,
                (0,),
                (2,),
                argument_bindings=((0, 0),),
                result_bindings=((1, 2),),
                enclosing_loop_ids=(178,),
            ),
        ),
    )
    planned, table = assign_hierarchy_ids(root)
    child_plan = planned.items[1].callee
    controls = {
        planned.closure_id: ControlProgram(
            WhileBlock(
                predicate_value_id=5,
                condition=SequenceBlock(()),
                body=StatementBlock(("__scheduled_region_0__",)),
                predicate_expression=ControlExpression(
                    "literal", literal=True
                ),
                source_loop_node_id=178,
            ),
            (0,),
        ),
        child_plan.closure_id: ControlProgram(
            StatementBlock(("__scheduled_region_0__",)),
            (0,),
        ),
    }

    composed = compose_hierarchical_control(planned, controls, table)
    loop = composed.program.root
    assert isinstance(loop, WhileBlock)
    assert isinstance(loop.body, SequenceBlock)
    assert isinstance(loop.body.blocks[-1], CallBlock)


def test_hierarchical_call_anchors_to_surviving_outer_loop_when_inner_loop_fused_away():
    # GLSL scheduling can fuse an inner loop's iteration space into a
    # surviving outer loop without marking it evaporated, so a call's
    # recorded innermost enclosing loop (34) can name a LoopBlock this
    # closure's control program no longer has -- only the outer loop (12)
    # survived. The call must still land inside the outer loop, not raise
    # and not fall out to the closure boundary.
    child = PlanClosure(
        "encode",
        (0,),
        (PlanClosure("region_0", (0,), (), -1),),
    )
    root = PlanClosure(
        "record",
        (0,),
        (
            PlanClosure("region_0", (0,), (), -1),
            PlanCall(
                8,
                child,
                (0,),
                (2,),
                argument_bindings=((0, 0),),
                result_bindings=((1, 2),),
                enclosing_loop_ids=(12, 34),
            ),
        ),
    )
    planned, table = assign_hierarchy_ids(root)
    child_plan = planned.items[1].callee
    controls = {
        planned.closure_id: ControlProgram(
            LoopBlock(
                "iteration_12",
                "0",
                "4",
                "1",
                StatementBlock(("__scheduled_region_0__",)),
            ),
            (0,),
        ),
        child_plan.closure_id: ControlProgram(
            StatementBlock(("__scheduled_region_0__",)),
            (0,),
        ),
    }

    composed = compose_hierarchical_control(planned, controls, table)
    loop = composed.program.root
    assert isinstance(loop, LoopBlock)
    assert isinstance(loop.body, SequenceBlock)
    assert isinstance(loop.body.blocks[-1], CallBlock)


def test_hierarchical_control_globalizes_iterable_extent_marker():
    child = PlanClosure(
        "child",
        (7,),
        (PlanLine.create("Identity", inputs=(7,), outputs=(8,)),),
        8,
    )
    root = PlanClosure(
        "root",
        (0,),
        (
            PlanCall(
                4,
                child,
                (0,),
                (1,),
                argument_bindings=((0, 7),),
                result_bindings=((8, 1),),
            ),
        ),
        1,
    )
    planned, table = assign_hierarchy_ids(root)
    child_plan = planned.items[0].callee
    controls = {
        planned.closure_id: ControlProgram(
            StatementBlock(("__scheduled_region_0__",)),
            (0,),
        ),
        child_plan.closure_id: ControlProgram(
            LoopBlock(
                "i",
                "0",
                "__iterable_extent_7__",
                "1",
                StatementBlock(("__scheduled_region_0__",)),
            ),
            (0,),
            iterable_bindings=((7, 8, "i"),),
        ),
    }

    composed = compose_hierarchical_control(planned, controls, table)
    global_iterable = table.global_id(child_plan.closure_id, 7)
    rendered = "\n".join(
        render_control_block(composed.program.root, ControlTarget.GLSL)
    )

    assert f"__iterable_extent_{global_iterable}__" in rendered
    assert "__iterable_extent_7__" not in rendered


def test_hierarchical_control_survives_value_alias_missing_from_table():
    # ``assign_hierarchy_ids`` only harvests (closure, local) keys from
    # PlanLine/PlanCall value ids. A control-flow-only endpoint discovered
    # later -- here, a loop-carried alias referencing local id 99, which was
    # never a PlanLine input/output or PlanCall binding -- must not crash
    # composition; it gets its own stable synthetic global identity instead.
    root = PlanClosure(
        "root",
        (0,),
        (PlanLine.create("Identity", inputs=(0,), outputs=(1,)),),
        0,
    )
    planned, table = assign_hierarchy_ids(root)
    controls = {
        planned.closure_id: ControlProgram(
            StatementBlock(("__scheduled_region_0__",)),
            (0,),
            value_aliases=((99, 0),),
        ),
    }

    composed = compose_hierarchical_control(planned, controls, table)

    assert len(composed.program.value_aliases) == 1
    updated, initial = composed.program.value_aliases[0]
    assert initial == table.global_id(planned.closure_id, 0)
    assert updated != initial


import ast
import contextlib
import io

from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.common.tensors.topological_reducer import (
    reduce_abstract_tensor_topology,
)
