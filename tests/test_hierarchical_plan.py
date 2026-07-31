from src.compiler.hierarchical_plan import (
    PlanCall,
    PlanClosure,
    PlanLine,
    assign_hierarchy_ids,
    reduce_hierarchy_identities,
    render_plan_ascii,
)
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
import ast
import contextlib
import io

from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.common.tensors.topological_reducer import (
    reduce_abstract_tensor_topology,
)
