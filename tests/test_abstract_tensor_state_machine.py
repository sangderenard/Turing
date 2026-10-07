import ast
import contextlib
import io

import pytest

from src.common import (
    AbstractTensorStateMachine,
    TensorStateField,
    is_abstract_tensor_state_machine,
)
from src.common.dt_system.dt_scaler import Metrics
from src.common.dt_system.state_table import StateTable
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.computational_world import ComputationalWorldState
from src.compiler.control_source import StateMachineTick
from src.compiler.precompile_to_ssa import lower_control_program_to_ssa


class ExampleMachine(AbstractTensorStateMachine):
    state_fields = (
        TensorStateField("position", ("N", 3), "float32", scope="world"),
    )

    def __init__(self):
        self.value = 0.0

    def transition(self, state, dt, *, state_table):
        self.value += dt
        return True, Metrics(0.0, 0.0, 0.0, 0.0, advanced_dt=dt), state

    def get_state(self, state=None):
        return state

    def snapshot(self):
        return self.value

    def restore(self, snapshot):
        self.value = float(snapshot)


def test_marker_is_a_dt_engine_contract_without_own_time_manager():
    machine = ExampleMachine()
    state = object()
    table = StateTable()

    ok, metrics, returned = machine.step(0.125, state, table)

    assert ok
    assert returned is state
    assert metrics.advanced_dt == pytest.approx(0.125)
    assert machine.value == pytest.approx(0.125)
    assert is_abstract_tensor_state_machine(ExampleMachine)
    assert ExampleMachine.tensor_state_schema()[0].shape == ("N", 3)


def test_marker_rejects_unadmitted_or_unaccounted_transitions():
    machine = ExampleMachine()
    with pytest.raises(ValueError, match="finite and positive"):
        machine.step(0.0, object(), StateTable())
    with pytest.raises(ValueError, match="StateTable"):
        machine.step(0.1, object(), None)


def test_ast_map_recognizes_only_explicit_state_machine_base():
    source = """
from src.common import AbstractTensorStateMachine

class OrdinaryWorld:
    def transition(self, state, dt):
        return state

class ComputationalWorld(AbstractTensorStateMachine):
    def transition(self, state, dt, *, state_table):
        return state
"""
    tree = ast.parse(source)
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(tree)

    assert graph.G.graph["map_ir"]["state_machines"] == (
        {
            "class_name": "ComputationalWorld",
            "identity": "ComputationalWorld",
            "marker": "AbstractTensorStateMachine",
            "bases": ("AbstractTensorStateMachine",),
            "transition_identity": "ComputationalWorld.transition",
            "ast_node_id": next(
                id(node)
                for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef)
                and node.name == "ComputationalWorld"
            ),
        },
    )


def test_sparse_world_state_checkpoint_restores_all_authoritative_tensors():
    state = ComputationalWorldState.empty()
    state.validate_sparse_shapes()
    checkpoint = state.copy_shallow()

    state.player_intent = state.player_intent + 3.0
    state.provenance_cursor = state.provenance_cursor + 8
    state.pending_status = (("artifact", "compiler:7"),)
    state.restore(checkpoint)

    assert state.player_intent.tolist() == [[0.0, 0.0, 0.0]]
    assert state.provenance_cursor.tolist() == [-1]
    assert state.pending_status == ()
    state.validate_sparse_shapes()


def test_marked_match_dispatch_reduces_to_existing_state_machine_tick_and_ssa():
    source = """
from src.common import AbstractTensorStateMachine

class SpringWorld(AbstractTensorStateMachine):
    def transition(self, state, dt, *, state_table):
        match int(state.phase.item()):
            case 0:
                return self.growing(state, dt, state_table=state_table)
            case 1:
                return self.settled(state, dt, state_table=state_table)

    def growing(self, state, dt, *, state_table):
        return state

    def settled(self, state, dt, *, state_table):
        return state
"""
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(ast.parse(source))

    (plan,) = graph.G.graph["state_machine_controls"]
    assert graph.G.graph["state_machine_control_shortfalls"] == ()
    assert isinstance(plan.control.root, StateMachineTick)
    assert plan.state_field == "phase"
    assert plan.case_methods == ((0, "growing"), (1, "settled"))

    function, shortfalls = lower_control_program_to_ssa(
        plan.control,
        function_name="spring_world_tick",
        first_value_id=10,
        region_callees={0: "SpringWorld.growing", 1: "SpringWorld.settled"},
        region_signatures={0: ((), ()), 1: ((), ())},
    )
    assert shortfalls == ()
    op_names = [
        getattr(instruction.op, "name", str(instruction.op))
        for block in function.blocks.values()
        for instruction in block.instrs
    ]
    assert "Eq" in op_names
    assert op_names.count("Call") == 2


# -- the reducer ingests a planned dispatch --------------------------------

_DISPATCH_SOURCE = """
from src.common import AbstractTensorStateMachine

class Ramp(AbstractTensorStateMachine):
    def transition(self, state, dt, *, state_table):
        match self.phase:
            case 0:
                {first}
            case 1:
                {second}

    def grow(self, state, dt, *, state_table):
        self.x = self.x + self.rate * dt

    def hold(self, state, dt, *, state_table):
        pass

def root(machine, dt):
    machine.transition(machine, dt, state_table=None)
"""


def _reduced_transition_graph(first, second):
    from pathlib import Path

    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    contracts = Path(__file__).resolve().parents[1] / "extraction_contracts"

    def scalar(dtype="float64"):
        return {"storage": "scalar", "dtype": dtype, "rank": 0, "mutable": True}

    policy = ExtractionContract(
        contracts / "program_extraction.yaml"
    ).with_program_abi({
        "records": {"Ramp": {"identity": "ramp.Ramp", "fields": {
            "x": scalar(), "rate": scalar(), "phase": scalar("int64"),
        }}},
        "bindings": [
            {"function": "*", "parameter": "machine", "record": "Ramp"},
        ],
        "values": [{
            "function": "root", "parameter": "dt", "storage": "scalar",
            "dtype": "float64", "rank": 0, "python_type": "builtins.float",
        }],
    })
    graphs = []
    lower_ast_source_to_ssa(
        _DISPATCH_SOURCE.format(first=first, second=second), "root",
        name="dispatch", extraction_contract=policy,
        resolved_process_graph_sink=graphs.append,
        stop_after_compilation_unit_plan=True, progress=lambda message: None,
    )
    return next(
        entry.graph for entry in graphs[0].function_table
        if entry.name == "transition"
    )


def test_planned_dispatch_is_one_node_with_state_and_case_operands():
    import ast

    graph = _reduced_transition_graph(
        "self.grow(state, dt, state_table=state_table)",
        "self.hold(state, dt, state_table=state_table)",
    )
    (dispatch,) = [
        (node, data) for node, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Match)
    ]
    node, data = dispatch
    roles = {role: parent for parent, role in data["parents"]}
    assert data["attributes"]["state_machine_dispatch"] is True
    assert set(roles) == {"state", "case:0", "case:1"}
    state = graph.G.nodes[roles["state"]]
    assert state["type"] == "GetAttr"
    assert state["expr_obj"].attr == "phase"
    assert [
        (graph.G.nodes[roles[f"case:{index}"]]["expr_obj"].value)
        for index in (0, 1)
    ] == [0, 1]
    # The arms' calls are ordinary call nodes of the same function.
    calls = [
        data for _node, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
    ]
    assert len(calls) == 2
    assert not graph.G.graph.get("translation_shortfalls")


def test_a_returning_arm_is_not_ingested_and_says_so():
    import ast

    graph = _reduced_transition_graph(
        "return self.grow(state, dt, state_table=state_table)",
        "return self.hold(state, dt, state_table=state_table)",
    )
    dispatch = [
        data for _node, data in graph.G.nodes(data=True)
        if (data.get("attributes") or {}).get("state_machine_dispatch")
    ]
    assert dispatch == []
    shortfalls = graph.G.graph.get("translation_shortfalls") or ()
    assert [item.pass_name for item in shortfalls] == ["state_dispatch"]
    assert "return" in shortfalls[0].reason
