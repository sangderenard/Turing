"""Adaptive growth travels through the real graph, piece state and native ABI."""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

from llvm_dt_system import (  # noqa: E402
    NativeSystem, RoundPiece, Subcycle, advance_round, dt_system_contract, dt_system_from_graph,
    instantiate_system, piece_leaf,
)
from src.common.dt_system.dt import SuperstepPlan  # noqa: E402
from src.common.dt_system.dt_controller import STController, Targets, run_superstep  # noqa: E402
from src.common.dt_system.dt_graph import ControllerNode, GraphBuilder, RoundNode  # noqa: E402
from src.common.dt_system.engine_api import DtCompatibleEngine, EngineRegistration  # noqa: E402
from src.common.dt_system.error_channels import DT_CHANNEL_NAMES, channel_fields  # noqa: E402
from src.common.dt_system.time_runtime import TimeWindowRequest  # noqa: E402
from src.common.tensors import AbstractTensor  # noqa: E402
from src.compiler.identity_concordance import concordance_report  # noqa: E402
from src.compiler.native_package import piece_from_law  # noqa: E402
from src.compiler.symbolic_equation_compiler import compile_sympy_equations  # noqa: E402

ERROR = "adaptive_growth_error"
CHANNEL_NAMES = (*DT_CHANNEL_NAMES, ERROR)


@pytest.fixture(scope="module")
def native_piece(tmp_path_factory):
    x, dt = sp.symbols("x dt")
    error = sp.Piecewise((dt * dt, x < sp.Rational(1, 32)), (0, True))
    law = compile_sympy_equations((
        sp.Eq(sp.Symbol("x_next"), x + dt, evaluate=False),
        sp.Eq(sp.Symbol(ERROR), error, evaluate=False),
        sp.Eq(sp.Symbol("max_vel"), 1, evaluate=False),
    ), name="adaptive_growth_native_law")
    piece = piece_from_law(law, "adaptive_growth_native_law", 1,
                           directory=tmp_path_factory.mktemp("adaptive_growth"))
    print("\n".join(concordance_report(piece.module).splitlines()[:4]), flush=True)
    return piece.for_runtime()


def graph(piece, *, dt_initial=0.03125, growth=None):
    targets = Targets(0.5, 1.0, 1.0, **channel_fields(
        {ERROR: 0.01}, names=CHANNEL_NAMES, limits=True))
    options = {} if growth is None else {"allow_increase_mid_round": growth}
    return RoundNode(
        SuperstepPlan(0.25, dt_initial, **options),
        ControllerNode(STController(Kp=1.0, Ki=0.0), targets, 1.0),
        children=[piece_leaf(piece)], **options,
    )


def observe_state(state):
    observations = []
    advance = state.program["advance_pieces"]

    def observe(actual_state, dt):
        before = actual_state.x.copy()
        ok, metrics = advance(actual_state, dt)
        observations.append((float(dt), before, actual_state.x.copy(),
                             float(metrics.error_channels[-1])))
        return ok, metrics

    state.program["advance_pieces"] = observe
    return observations


def test_graph_default_recovers_after_transient_restriction(native_piece):
    state = instantiate_system(graph(native_piece), {"x": np.zeros(1)},
                               channel_names=CHANNEL_NAMES)
    assert state.allow_increase_mid_round is True
    owned_x = state.x
    observations = observe_state(state)
    advanced, continuation, _ = advance_round(state)
    assert [row[0] for row in observations] == [0.03125, 0.21875]
    assert observations[0][3] == 0.03125 ** 2
    assert observations[1][3] == 0.0
    assert advanced == 0.25 and continuation > 0.03125
    assert state.x is owned_x and np.shares_memory(owned_x, state.span)
    np.testing.assert_array_equal(state.x, [0.25])
    print("adaptive graph attempts:", [row[0] for row in observations], flush=True)


def test_adaptive_graph_restores_rejections_and_lands_outer_window(native_piece):
    state = instantiate_system(graph(native_piece, dt_initial=0.25),
                               {"x": np.zeros(1)}, channel_names=CHANNEL_NAMES)
    owned_x = state.x
    observations = observe_state(state)
    advanced, continuation, _ = advance_round(state)
    assert [row[0] for row in observations] == [0.25, 0.125, 0.0625, 0.1875]
    for _, before, _, _ in observations[:3]:
        np.testing.assert_array_equal(before, [0.0])
    np.testing.assert_array_equal(observations[3][1], [0.0625])
    assert observations[3][3] == 0.0
    assert state.x is owned_x and advanced == 0.25 and continuation > 0.0625
    np.testing.assert_array_equal(state.x, [0.25])


def test_explicit_legacy_optout_is_preserved_as_a_declared_choice(native_piece):
    state = instantiate_system(graph(native_piece, growth=False),
                               {"x": np.zeros(1)}, channel_names=CHANNEL_NAMES)
    assert state.allow_increase_mid_round is False
    observations = observe_state(state)
    advanced, _, _ = advance_round(state)
    assert advanced == 0.25
    assert [row[0] for row in observations] == [0.03125] * 8


@pytest.mark.parametrize("growth", (True, False))
def test_direct_graph_path_forwards_growth_choice(native_piece, growth):
    root = graph(native_piece, growth=growth)
    columns = {"x": np.zeros(1)}
    state, _, results = dt_system_from_graph(
        root, columns, rounds=1, channel_names=CHANNEL_NAMES)
    assert state.allow_increase_mid_round is growth
    assert results[0][0] == 0.25
    np.testing.assert_array_equal(columns["x"], [0.25])


def test_nested_round_uses_actual_graph_growth_choice(native_piece):
    nested = RoundPiece(graph(native_piece), channel_names=CHANNEL_NAMES)
    observations = []
    advance = nested._advance

    def observe(state, dt):
        observations.append(float(dt))
        return advance(state, dt)

    nested._advance = observe
    columns = {"x": np.zeros(1)}
    nested.instantiate(columns)
    result, = nested(columns["x"], np.array([0.25]))
    assert observations == [0.03125, 0.21875]
    np.testing.assert_array_equal(result, [0.25])
    assert nested.tau == 1.0


def test_pinned_interior_ignores_adaptive_growth(native_piece):
    state = instantiate_system(graph(native_piece), {"x": np.zeros(1)},
                               channel_names=CHANNEL_NAMES)
    observations = observe_state(state)
    advanced, _, _ = run_superstep(
        state, 0.25, 0.25, 1.0, state.targets, state.controller,
        state.program["advance_pieces"], substep="pinned", substep_dt=0.03125,
        rollback=True, allow_unresolved=False,
    )
    assert advanced == 0.25
    assert [row[0] for row in observations] == [0.03125] * 8
    np.testing.assert_array_equal(state.x, [0.25])


def test_declared_growth_bool_reaches_actual_native_record(native_piece, tmp_path):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )

    state = instantiate_system(graph(native_piece), {"x": np.zeros(1)},
                               channel_names=CHANNEL_NAMES)
    contract = dt_system_contract("root", ("x",), 1, channel_names=CHANNEL_NAMES)
    field = contract.program_abi.receipt()["records"]["PieceState"]["fields"][
        "allow_increase_mid_round"]
    assert field["dtype"] == "bool" and field["storage"] == "scalar"
    module, _, exports = lower_ast_source_to_ssa(
        "def root(state):\n"
        "    state.x[...] = state.x + 1.0 * state.allow_increase_mid_round\n"
        "    return state.x\n",
        "root", name="adaptive_growth_bool", extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    root = module.functions[exports[0]]
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path, optimization="O0")
    system = NativeSystem(artifact, module, root.name, (), columns=("x",),
                          channel_names=CHANNEL_NAMES)
    output_id = system.state_field_ids()["x"]
    for growth in (True, False):
        state.allow_increase_mid_round = growth
        state.x[...] = 0.0
        feeds = system.feeds(state, state.targets, state.controller, 0.25, 0.03125, 1.0)
        execution = artifact.prepare_execution(feeds)
        execution.run()
        np.testing.assert_array_equal(execution.buffers[output_id], [float(growth)])
    print("native growth bool:\n" + "\n".join(
        concordance_report(module).splitlines()[:4]), flush=True)


def test_adaptive_defaults_and_graphbuilder_local_choice():
    from src.cells.bath.adapter import HybridAdapter, MACAdapter, SPHAdapter
    from src.cells.cellsim.api.saline import SalinePressureAPI

    assert SuperstepPlan(1.0, 0.1).allow_increase_mid_round
    assert TimeWindowRequest(1, 0, 0.0, 1.0, 0.1).allow_increase_mid_window
    for method in (run_superstep, GraphBuilder.round, SalinePressureAPI.step_super,
                   SPHAdapter.step_super, MACAdapter.step_super, HybridAdapter.step_super,
                   Subcycle.__init__):
        assert inspect.signature(method).parameters["allow_increase_mid_round"].default is True

    class Engine(DtCompatibleEngine):
        def step(self, dt, state=None, state_table=None):
            raise AssertionError("this regression inspects graph construction only")

    targets = Targets(0.5, 1.0, 1.0)
    for solver in (None, object()):
        registration = EngineRegistration(
            "local", Engine(), targets=targets, dx=1.0, solver_config=solver)
        for growth in (True, False):
            root = GraphBuilder(STController(), targets, 1.0).round(
                0.25, [registration], allow_increase_mid_round=growth)
            assert root.allow_increase_mid_round is growth
            assert root.plan.allow_increase_mid_round is growth
            nested = next(child for child in root.children if isinstance(child, RoundNode))
            assert nested.allow_increase_mid_round is growth
            assert nested.plan.allow_increase_mid_round is growth
