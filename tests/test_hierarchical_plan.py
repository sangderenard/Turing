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
    ControlTarget,
    StatementBlock,
    StreamPublishBlock,
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
    )
    process.add_node(
        1,
        type="GetAttr",
        op="GetAttr",
        parents=((0, "base"),),
        attributes={"attribute": "header"},
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
