import ast

import networkx as nx
import pytest

from src.compiler.fortran_c_shell import (
    _recover_late_source_pure_expressions,
    _recover_late_source_unary_operations,
)
from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue


@pytest.mark.parametrize("operation", ["item", "mul"])
def test_multiple_recoveries_follow_live_producer_positions(operation):
    graph = nx.DiGraph()
    sources = [SSAValue(10, "float64"), SSAValue(20, "float64")]
    results = [SSAValue(11, "float64"), SSAValue(21, "float64")]
    for source, result in zip(sources, results):
        graph.add_node(source.id, type="tensor", op="tensor", parents=[])
        graph.add_node(result.id, type=operation, op=operation, parents=(
            [(source.id, "operand")] if operation == "item"
            else [(source.id, "lhs"), (source.id, "rhs")]
        ))
    block = BasicBlock("entry", [
        Instr("Const", [], sources[0], attributes={"value": 2.0}),
        Instr("Const", [], sources[1], attributes={"value": 3.0}),
        Instr("Ret", results, None),
    ])
    function = Function("root", list(results), {"entry": block})
    recover = (_recover_late_source_unary_operations if operation == "item"
               else _recover_late_source_pure_expressions)
    assert len(recover(function, graph)) == 2
    available = set()
    for instruction in block.instrs:
        assert all(value.id in available for value in instruction.args)
        if instruction.res is not None:
            available.add(instruction.res.id)


def test_scalar_storage_handle_does_not_make_list_replication_numeric():
    graph = nx.DiGraph()
    graph.add_node(1, type="Constant", op="const", parents=[],
                   attributes={"aggregate_kind": "list", "value": [2.0]})
    graph.add_node(2, type="Constant", op="const", parents=[],
                   attributes={"value": 3})
    graph.add_node(3, type="Mul", op="mul", parents=[(1, "lhs"), (2, "rhs")])
    arena, count, result = (SSAValue(i, "float64") for i in (1, 2, 3))
    block = BasicBlock("entry", [
        Instr("Load", [SSAValue(4, "ptr")], arena),
        Instr("Const", [], count, attributes={"value": 3}),
        Instr("Ret", [result], None),
    ])
    function = Function("root", [result], {"entry": block})
    assert _recover_late_source_pure_expressions(function, graph) == ()
    assert function.args == [result]
    assert all(instruction.op != "Mul" for instruction in block.instrs)


def _source_graph():
    graph = nx.DiGraph()
    graph.add_node(130, type="isfinite", op="isfinite", parents=[])
    graph.add_node(
        132,
        type="all",
        op="all",
        parents=[(130, "operand")],
    )
    return graph


def test_late_entry_tensor_reduction_replaces_provisional_formal():
    source = SSAValue(130, "float64", shape=(8, 4, 14))
    missing = SSAValue(132)
    combined = SSAValue(133, "bool")
    function = Function(
        "root",
        [missing],
        {
            "entry": BasicBlock("entry", [
                Instr("Load", [SSAValue(900, "ptr")], source),
                Instr("LAnd", [SSAValue(127, "bool"), missing], combined),
                Instr("Ret", [combined], None),
            ]),
        },
    )

    recovered = _recover_late_source_unary_operations(function, _source_graph())

    assert recovered == ((132, 130, "all"),)
    assert function.args == []
    reduction = function.blocks["entry"].instrs[1]
    assert reduction.op == "all"
    assert reduction.args == [source]
    assert reduction.res is missing
    assert missing.dtype == "bool"
    assert function.blocks["entry"].instrs[2].args[1] is missing


def test_late_branch_local_reduction_is_not_hoisted():
    source = SSAValue(130, "float64", shape=(4,))
    missing = SSAValue(132)
    function = Function(
        "root",
        [missing],
        {
            "entry": BasicBlock("entry", [
                Instr("CondBr", [SSAValue(1, "bool")], None,
                      attributes={"true_target": "if_true",
                                  "false_target": "if_false"}),
            ]),
            "if_true": BasicBlock("if_true", [
                Instr("Load", [SSAValue(900, "ptr")], source),
                Instr("Ret", [missing], None),
            ]),
            "if_false": BasicBlock("if_false", [Instr("Ret", [], None)]),
        },
    )

    assert _recover_late_source_unary_operations(function, _source_graph()) == ()
    assert function.args == [missing]


def test_late_loop_item_consumes_current_carried_phi():
    graph = nx.DiGraph()
    graph.add_node(43, type="tensor", op="tensor", parents=[])
    graph.add_node(169, type="item", op="item", parents=[(43, "operand")])
    initial = SSAValue(43, "float64", shape=())
    updated = SSAValue(568, "float64", shape=())
    current = SSAValue(713, "float64", shape=())
    missing = SSAValue(169)
    entry = BasicBlock("entry", [
        Instr("Load", [SSAValue(900, "ptr")], initial),
        Instr("Br", [], None, attributes={"target": "while_header"}),
    ], successors=["while_header"])
    header = BasicBlock("while_header", [
        Instr("Phi", [initial, updated], current, attributes={
            "binding": "loop_carried",
            "initial_value_id": 43,
            "updated_value_id": 568,
            "incoming_blocks": ("entry", "while_latch"),
        }),
        Instr("CondBr", [SSAValue(1, "bool")], None, attributes={
            "true_target": "while_body", "false_target": "while_exit",
        }),
    ], successors=["while_body", "while_exit"])
    body = BasicBlock("while_body", [
        Instr("Call", [missing], SSAValue(800), attributes={"callee": "use"}),
        Instr("Br", [], None, attributes={"target": "while_latch"}),
    ], successors=["while_latch"])
    latch = BasicBlock("while_latch", [
        Instr("Br", [], None, attributes={"target": "while_header"}),
    ], successors=["while_header"])
    exit_block = BasicBlock("while_exit", [Instr("Ret", [], None)])
    function = Function(
        "root", [missing],
        {block.name: block for block in (entry, header, body, latch, exit_block)},
    )

    recovered = _recover_late_source_unary_operations(function, graph)

    assert recovered == ((169, 43, "item"),)
    assert function.args == []
    scalar = header.instrs[1]
    assert scalar.op == "Cast"
    assert scalar.args == [current]
    assert scalar.res is missing
    assert body.instrs[0].args[0] is missing


def test_late_scalar_expression_closure_consumes_current_carried_phi():
    graph = nx.DiGraph()
    graph.add_node(10, type="tensor", op="tensor", parents=[])
    graph.add_node(11, type="item", op="item", parents=[(10, "operand")])
    graph.add_node(
        12, type="Constant", op="const", parents=[],
        expr_obj=ast.Constant(value=0.5),
        attributes={"value": 0.5}, constant=0.5,
    )
    graph.add_node(
        13, type="Mul", op="Mul", parents=[(11, "lhs"), (12, "rhs")],
    )
    initial = SSAValue(10, "float64", shape=())
    updated = SSAValue(20, "float64", shape=())
    current = SSAValue(21, "float64", shape=())
    missing = SSAValue(13, "float64", shape=())
    entry = BasicBlock("entry", [
        Instr("Load", [SSAValue(900, "ptr")], initial),
        Instr("Br", [], None, attributes={"target": "while_header"}),
    ], successors=["while_header"])
    header = BasicBlock("while_header", [
        Instr("Phi", [initial, updated], current, attributes={
            "binding": "loop_carried",
            "initial_value_id": 10,
            "updated_value_id": 20,
            "incoming_blocks": ("entry", "while_latch"),
        }),
        Instr("CondBr", [SSAValue(1, "bool")], None, attributes={
            "true_target": "while_body", "false_target": "while_exit",
        }),
    ], successors=["while_body", "while_exit"])
    body = BasicBlock("while_body", [
        Instr("Call", [missing], SSAValue(30), attributes={"callee": "use"}),
        Instr("Br", [], None, attributes={"target": "while_latch"}),
    ], successors=["while_latch"])
    latch = BasicBlock("while_latch", [
        Instr("Br", [], None, attributes={"target": "while_header"}),
    ], successors=["while_header"])
    exit_block = BasicBlock("while_exit", [Instr("Ret", [], None)])
    function = Function(
        "root", [missing],
        {block.name: block for block in (entry, header, body, latch, exit_block)},
    )

    recovered = _recover_late_source_pure_expressions(function, graph)

    assert recovered == ((13, 10, "mul"),)
    assert function.args == []
    item, constant, multiply = body.instrs[:3]
    assert (item.op, [value.id for value in item.args], item.res.id) == (
        "Cast", [21], 11,
    )
    assert (constant.op, constant.attributes["value"], constant.res.id) == (
        "Const", 0.5, 12,
    )
    assert (multiply.op, [value.id for value in multiply.args]) == (
        "Mul", [11, 12],
    )
    assert multiply.res is missing
    assert body.instrs[3].args[0] is missing


def test_late_scalar_items_use_unique_dominating_resident_without_loop_phi():
    graph = nx.DiGraph()
    graph.add_node(10, type="tensor", op="tensor", parents=[])
    graph.add_node(11, type="item", op="item", parents=[(10, "operand")])
    graph.add_node(
        12, type="Constant", op="const", parents=[],
        expr_obj=ast.Constant(value=0.5),
        attributes={"value": 0.5}, constant=0.5,
    )
    graph.add_node(
        13, type="Mul", op="Mul", parents=[(11, "lhs"), (12, "rhs")],
    )
    graph.add_node(14, type="item", op="item", parents=[(10, "operand")])
    source = SSAValue(10, "float64", shape=())
    half = SSAValue(13, "float64", shape=())
    direct = SSAValue(14, "float64", shape=())
    function = Function(
        "root", [half, direct],
        {
            "entry": BasicBlock("entry", [
                Instr("Load", [SSAValue(900, "ptr")], source),
                Instr("Br", [], None, attributes={"target": "body"}),
            ], successors=["body"]),
            "body": BasicBlock("body", [
                Instr("Call", [half, direct], SSAValue(30),
                      attributes={"callee": "use"}),
                Instr("Ret", [], None),
            ]),
        },
    )

    unary = _recover_late_source_unary_operations(function, graph)
    expressions = _recover_late_source_pure_expressions(function, graph)

    assert unary == ((14, 10, "item"),)
    assert expressions == ((13, 10, "mul"),)
    assert function.args == []
    resident_item = function.blocks["entry"].instrs[1]
    assert (resident_item.op, [value.id for value in resident_item.args]) == (
        "Cast", [10],
    )
    item, constant, multiply = function.blocks["body"].instrs[:3]
    assert (item.op, [value.id for value in item.args], item.res.id) == (
        "Cast", [10], 11,
    )
    assert (constant.op, constant.attributes["value"], constant.res.id) == (
        "Const", 0.5, 12,
    )
    assert (multiply.op, [value.id for value in multiply.args], multiply.res) == (
        "Mul", [11, 12], half,
    )
    assert [value.id for value in function.blocks["body"].instrs[3].args] == (
        [13, 14]
    )
