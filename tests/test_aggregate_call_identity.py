"""Exact identity of authored aggregates across call, loop, and return boundaries.

These tests pin the rules that let one authored Python program cross a
function call with a tuple of tensors (``tire_history``), unpack a
multi-result call inside a retained loop, and return a nested tuple, without
the compiler manufacturing, dropping, or recycling identities:

* a tuple of ProgramABI span parameters is dataflow, never a folded Constant;
* every aggregate member has a scoped callee formal bound by exact index;
* a multi-result call inside a loop publishes every carried result at its
  scheduled position;
* call results and returned aggregates correlate by structural path;
* value ids are never recycled after a removal.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler import glsl_deployment_strategy as planner
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.hierarchical_plan import PlanCall, PlanClosure
from src.compiler.process_graph_value_ids import next_process_value_id

CONTRACTS = Path(__file__).resolve().parents[1] / "extraction_contracts"

HISTORY_SOURCE = """
def inner_recurrence(inputs, history, valid, count):
    previous_hub, previous_angle = history
    current_hub = inputs[:, :, 0:3]
    current_angle = inputs[:, :, 3]
    previous_hub = AbstractTensor.where(valid.reshape((-1, 1, 1)), previous_hub, current_hub)
    previous_angle = AbstractTensor.where(valid.reshape((-1, 1)), previous_angle, current_angle)
    total = inputs[:, :, 0:3] * 0.0
    for step in range(count):
        alpha = (step + 1.0) / count
        total = total + previous_hub + alpha * (current_hub - previous_hub)
        total = total + previous_angle.reshape((-1, 4, 1))
    return total, (current_hub, current_angle), valid * 0.0 + 1.0


def tick(inputs, previous_hub, previous_angle, valid, enabled, count):
    history = (previous_hub, previous_angle)
    total, next_history, next_valid = inner_recurrence(inputs, history, valid, count)
    history = (
        AbstractTensor.where(enabled.reshape((-1, 1, 1)), next_history[0], history[0]),
        AbstractTensor.where(enabled.reshape((-1, 1)), next_history[1], history[1]),
    )
    valid = AbstractTensor.where(enabled, next_valid, valid * 0.0)
    return total, history, valid
"""

LOOP_CALL_SOURCE = """
def leaf_step(inputs, state, output, rest):
    state = state + inputs[:, :, 0:3] * rest.reshape((1, 1, 3))
    output = output * 0.0 + state.sum(dim=2).reshape((-1, 4, 1))
    return state, output


def recurrence(inputs, state, output, history, valid, constants, count):
    rest = constants[0]
    previous_hub, previous_angle = history
    current_hub = inputs[:, :, 0:3]
    current_angle = inputs[:, :, 3]
    previous_hub = AbstractTensor.where(valid.reshape((-1, 1, 1)), previous_hub, current_hub)
    previous_angle = AbstractTensor.where(valid.reshape((-1, 1)), previous_angle, current_angle)
    wrench = output * 0.0
    for step in range(count):
        alpha = (step + 1.0) / count
        inputs[:, :, 0:3] = previous_hub + alpha * (current_hub - previous_hub)
        inputs[:, :, 3] = previous_angle + alpha * (current_angle - previous_angle)
        state, output = leaf_step(inputs, state, output, *constants)
        wrench = wrench + output
    output = wrench / count
    return state, output, (current_hub, current_angle), valid * 0.0 + 1.0


def tick(inputs, state, output, previous_hub, previous_angle, valid, enabled, rest, count):
    history = (previous_hub, previous_angle)
    constants = (rest,)
    next_state, next_output, next_history, next_valid = recurrence(
        inputs, state, output, history, valid, constants, count)
    state = AbstractTensor.where(enabled.reshape((-1, 1, 1)), next_state, state)
    output = AbstractTensor.where(enabled.reshape((-1, 1, 1)), next_output, output)
    history = (
        AbstractTensor.where(enabled.reshape((-1, 1, 1)), next_history[0], history[0]),
        AbstractTensor.where(enabled.reshape((-1, 1)), next_history[1], history[1]),
    )
    valid = AbstractTensor.where(enabled, next_valid, valid * 0.0)
    return state, output, history, valid
"""


def _contract(entry: str, feeds: dict):
    values = []
    for name, value in feeds.items():
        if isinstance(value, np.ndarray):
            values.append({
                "function": entry, "parameter": name, "storage": "span",
                "dtype": "float64", "rank": value.ndim,
                "shape": list(value.shape),
                "python_type": "src.common.tensors.abstraction.AbstractTensor",
            })
        else:
            values.append({
                "function": entry, "parameter": name, "storage": "scalar",
                "dtype": "int64", "rank": 0, "python_type": "builtins.int",
            })
    return ExtractionContract(
        CONTRACTS / "program_extraction.yaml"
    ).with_program_abi(
        {"records": {}, "bindings": [], "values": values}
    ).with_execution_file(CONTRACTS / "vehicle_full_native_execution.yaml")


def _lower(source: str, name: str, feeds: dict, **kwargs):
    return lower_ast_source_to_ssa(
        source, "tick",
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        name=name, runtime_closure_only=True,
        extraction_contract=_contract("tick", feeds),
        **kwargs,
    )


def _flat_calls(items):
    for item in items:
        if isinstance(item, PlanCall):
            yield item
        elif isinstance(item, PlanClosure):
            yield from _flat_calls(item.items)


def _capture_plans(monkeypatch):
    captured = []
    original = planner._build_shell_hierarchy_plan

    def capture(shell):
        plan = original(shell)
        captured.append((shell, plan))
        return plan

    monkeypatch.setattr(planner, "_build_shell_hierarchy_plan", capture)
    return captured


def _ret_args(function):
    return [
        instruction.args
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    ][-1]


HISTORY_FEEDS = {
    "inputs": np.zeros((8, 4, 4)),
    "previous_hub": np.zeros((8, 4, 3)),
    "previous_angle": np.zeros((8, 4)),
    "valid": np.zeros((8,)),
    "enabled": np.zeros((8,)),
    "count": 1,
}


def test_history_tuple_members_bind_by_exact_identity(monkeypatch):
    captured = _capture_plans(monkeypatch)

    module, outputs, _exports = _lower(
        HISTORY_SOURCE, "history_identity", HISTORY_FEEDS,
    )

    root = module.functions["history_identity__tick"]
    declared = dict(root.metadata["parameter_names"])
    # The authored span parameters that only feed the tuple remain public
    # formals; the tuple was never folded into a Constant.
    assert {"previous_hub", "previous_angle"} <= set(declared)
    for shell, plan in captured:
        graph = shell.process_graph.G
        if graph.graph.get("function_name") != "tick":
            continue
        calls = list(_flat_calls(plan.items))
        if not calls:
            continue
        call = calls[0]
        identities = graph.graph["identity_table"]
        hub_id = int(identities["previous_hub"][0])
        angle_id = int(identities["previous_angle"][0])
        tuple_id = int(identities["history"][0])
        tuple_data = graph.nodes[tuple_id]
        assert tuple_data.get("type") == "Tuple"
        assert tuple(tuple_data["attributes"]["aggregate_leaf_value_ids"]) == (
            hub_id, angle_id,
        )
        child = shell.callsite_function_shells[call.callsite_id].process_graph.G
        bound = dict(call.argument_bindings)
        for caller_id, index, shape in (
            (hub_id, 0, (8, 4, 3)), (angle_id, 1, (8, 4)),
        ):
            member = child.nodes[bound[caller_id]]
            assert member["type"] == "Input"
            assert member["attributes"]["aggregate_parent_binding"] == "history"
            assert member["attributes"]["aggregate_index"] == index
            assert tuple(member["tensor"]["shape"]) == shape
        # The aggregate itself is not a physical argument.
        assert tuple_id not in bound
        # Results correlate by structural path: (0,), (1,0), (1,1), (2,).
        assert len(call.result_bindings) == 4
        break
    else:
        pytest.fail("tick plan with the inner_recurrence call was not built")

    callee = next(
        function for name, function in module.functions.items()
        if "inner_recurrence__specialized" in name
        and "planned_region" not in name
    )
    member_shapes = sorted(
        tuple(value.shape)
        for value in callee.args
        if tuple(value.shape or ()) in {(8, 4, 3), (8, 4)}
    )
    assert member_shapes == [(8, 4), (8, 4, 3)]
    # Nested tuple return: leaves are physical outputs in path order.
    assert [tuple(value.shape) for value in _ret_args(callee)] == [
        (8, 4, 3), (8, 4, 3), (8, 4), (8,),
    ]
    assert [tuple(value.shape) for value in _ret_args(root)] == [
        (8, 4, 3), (8, 4, 3), (8, 4), (8,),
    ]
    assert len(outputs["history_identity__tick"]) == 4


def test_nested_call_result_projections_are_typed_call_outputs():
    module, _outputs, _exports = _lower(
        HISTORY_SOURCE, "history_projection", HISTORY_FEEDS,
    )

    root = module.functions["history_projection__tick"]
    linked = next(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and "inner_recurrence__specialized" in str(
            instruction.attributes.get("callee")
        )
    )
    assert len(linked.attributes["output_ids"]) == 4
    loads = {
        int(instruction.res.id): tuple(instruction.res.shape or ())
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Load"
        and int(instruction.res.id) in set(linked.attributes["output_ids"])
    }
    assert sorted(loads.values()) == [(8,), (8, 4), (8, 4, 3), (8, 4, 3)]
    # No caller region receives an undeclared/untyped argument for the
    # projections ``next_history[0]``/``[1]``.
    for name, function in module.functions.items():
        if not name.startswith("history_projection__tick__planned_region"):
            continue
        for value in function.args:
            assert value.dtype is not None, (name, value.id)


LOOP_FEEDS = {
    "inputs": np.zeros((8, 4, 4)),
    "state": np.zeros((8, 4, 3)),
    "output": np.zeros((8, 4, 1)),
    "previous_hub": np.zeros((8, 4, 3)),
    "previous_angle": np.zeros((8, 4)),
    "valid": np.zeros((8,)),
    "enabled": np.zeros((8,)),
    "rest": np.zeros((3,)),
    "count": 1,
}


def test_unpacked_call_results_carry_a_retained_loop(monkeypatch):
    captured = _capture_plans(monkeypatch)

    module, _outputs, _exports = _lower(
        LOOP_CALL_SOURCE, "loop_call_carry", LOOP_FEEDS,
    )

    shell, plan, call = next(
        (shell, plan, calls[0]) for shell, plan in captured
        if shell.process_graph.G.graph.get("function_name") == "recurrence"
        for calls in (list(_flat_calls(plan.items)),)
        if calls
    )
    graph = shell.process_graph.G
    identities = graph.graph["identity_table"]
    bound = dict(call.argument_bindings)
    # ``state, output = leaf_step(inputs, state, output, *constants)``: the
    # arguments are the loop-entry values, never the call's own unpack
    # targets spelled left of it on the same line.
    assert int(identities["state"][0]) in bound
    assert int(identities["output"][0]) in bound
    for _callee_id, caller_id in call.result_bindings:
        assert caller_id not in bound
    assert call.enclosing_loop_ids
    # The starred aggregate binds its member formal, not a recycled port id.
    rest_member = next(
        (member_id, data) for member_id, data in graph.nodes(data=True)
        if (data.get("attributes") or {}).get("aggregate_parent_binding")
        == "constants"
    )
    assert graph.nodes[rest_member[0]]["type"] == "Input"

    callee = next(
        function for name, function in module.functions.items()
        if "recurrence__specialized" in name and "planned_region" not in name
    )
    callee_returns = _ret_args(callee)
    # Five physical outputs in structural path order.  The first two are the
    # loop-carried results (their carried phi values carry no shape receipt
    # of their own today); the nested aggregate leaves and the flag follow.
    assert len(callee_returns) == 5
    assert [tuple(value.shape) for value in callee_returns[2:]] == [
        (8, 4, 3), (8, 4), (8,),
    ]
    root = module.functions["loop_call_carry__tick"]
    assert [tuple(value.shape) for value in _ret_args(root)] == [
        (8, 4, 3), (8, 4, 1), (8, 4, 3), (8, 4), (8,),
    ]


def test_process_value_ids_are_never_recycled():
    graph = nx.DiGraph()
    graph.add_node(0)
    graph.add_node(7)
    first = next_process_value_id(graph)
    assert first == 8
    graph.add_node(first)
    graph.remove_node(first)
    graph.remove_node(7)
    second = next_process_value_id(graph)
    assert second > first
    assert graph.graph["value_id_watermark"] == second


def test_dependency_order_cache_keys_on_node_identity():
    class Graph:
        def __init__(self, ids):
            self.G = nx.DiGraph()
            for node_id in ids:
                self.G.add_node(node_id, parents=[])
            self.levels = {}

    first = Graph((0, 1, 2))
    order = planner._dependency_order(first)
    assert order == (0, 1, 2)
    # Same counts and (empty) semantic edges, different identities: the
    # cached order must not be reused from an inherited metadata dict.
    second = Graph((0, 1, 5))
    second.G.graph.update(dict(first.G.graph))
    assert planner._dependency_order(second) == (0, 1, 5)
