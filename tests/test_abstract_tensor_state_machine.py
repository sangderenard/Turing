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
