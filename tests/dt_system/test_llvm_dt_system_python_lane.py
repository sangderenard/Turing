"""llvm_dt_system's Python lane: time vocabulary and independent subcycles.

Pieces here are plain positional callables that declare ``argument_names``
and ``output_names`` -- the same shape an ``LLVMPiece`` has -- so the dt
system's own Python path runs without building anything native.

* ``pub_exchange_time`` (the dt system's scheduling slot) carries ``exchange_time_s``
  = E/P; the energy and transport Courant numbers are tracked beside it.
* telemetry ``tau`` is time velocity: advanced / asked.
* the wall-cost ledger records every call and is read by nothing.
* a ``Subcycle`` runs on its own thread and is consulted without waiting;
  its ``tau`` and ``slip_s`` are measured against the lockstep world time.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from llvm_dt_system import Subcycle, TELEMETRY_FIELDS, dt_system  # noqa: E402

BATCH = 4
TAU = TELEMETRY_FIELDS.index("tau")


class Piece:
    def __init__(self, entry, argument_names, output_names, fn, batch=BATCH):
        self.entry = entry
        self.argument_names = tuple(argument_names)
        self.output_names = tuple(output_names)
        self.batch = batch
        self._fn = fn

    def __call__(self, *columns):
        return tuple(np.asarray(value, dtype=np.float64) * np.ones(self.batch)
                     for value in self._fn(*columns))


def drift(dt_limit=0.01, energy=10.0):
    return Piece(
        "drift", ("x", "v", "dt"),
        ("x_next", "dt_limit", "max_vel", "energy_j", "power_w"),
        lambda x, v, dt: (x + dt * v, dt_limit, np.abs(v), energy, np.abs(v)),
    )


def test_exchange_time_courant_numbers_and_time_velocity():
    x0 = np.array([0.0, 1.0, -2.0, 10.0])
    v = np.array([1.0, 2.0, 3.0, -4.0])
    columns = {"x": x0.copy(), "v": v.copy()}

    state, _ctrl, results = dt_system([drift()], columns, rounds=3, round_dt=0.05, dx=1.0)

    np.testing.assert_allclose(columns["x"], x0 + 0.15 * v, rtol=0, atol=1e-12)
    for _total, _dt, telemetry in results:
        assert telemetry[TAU] == pytest.approx(1.0)          # every window landed
    # exchange_time_s = E / P with the batch's reduced power (max |v| = 4)
    assert float(state.pub_exchange_time[0]) == pytest.approx(10.0 / 4.0)
    assert float(state.pub_exchange_time_present[0]) == 1.0
    dt_last = float(state.dt[0])
    assert float(state.pub_energy_courant[0]) == pytest.approx(4.0 * dt_last / 10.0)
    assert float(state.pub_dt_courant[0]) == pytest.approx(dt_last / 0.01)
    assert float(state.pub_dt_courant[0]) <= 1.0 + 1e-12      # never past the floor

    report = state.wall_cost_ledger.report()
    assert report["participants"]["drift"]["calls"] >= 1 + 5 + 5
    assert report["world_s"] == pytest.approx(0.15)
    assert report["wall_cost"] > 0.0


def lockstep_reader():
    # y relaxes toward the subcycle-owned z, read lagged
    return Piece("reader", ("y", "z", "dt"), ("y_next", "dt_limit"),
                 lambda y, z, dt: (y + dt * (z - y), 0.01))


def clock(delay_s=0.0):
    def step(z, dt):
        if delay_s:
            time.sleep(delay_s)
        return z + dt, 0.005
    return Piece("clock", ("z", "dt"), ("z_next", "dt_limit"), step)


def test_subcycle_runs_independently_and_publishes_its_own_clock():
    columns = {"y": np.zeros(BATCH), "z": np.zeros(BATCH)}
    sub = Subcycle([clock()], round_dt=0.02, dx=1.0, name="clock")

    state, _ctrl, _results = dt_system([lockstep_reader()], columns, rounds=20,
                                       round_dt=0.02, dx=1.0, subcycles=[sub])

    (status,) = state.subcycle_status
    assert status["windows"] >= 1
    # the owner's value comes back: z is exactly its own world time
    np.testing.assert_allclose(columns["z"], status["world_s"], rtol=0, atol=1e-9)
    # never more than lead_windows ahead of what it was offered
    assert status["world_s"] <= status["reference_s"] + 0.02 + 1e-9
    assert status["tau"] == pytest.approx(status["world_s"] / status["reference_s"])
    assert status["slip_s"] == pytest.approx(status["reference_s"] - status["published_stamp_s"])
    assert status["wall_cost"]["participants"]["clock"]["calls"] >= 1


def test_slow_subcycle_falls_behind_and_the_lockstep_side_never_waits():
    columns = {"y": np.zeros(BATCH), "z": np.zeros(BATCH)}
    # 4 substeps per 0.02 s window at 20 ms each: ~80 ms of wall per window
    sub = Subcycle([clock(delay_s=0.02)], round_dt=0.02, dx=1.0, name="slow")

    start = time.perf_counter()
    state, _ctrl, results = dt_system([lockstep_reader()], columns, rounds=50,
                                      round_dt=0.02, dx=1.0, subcycles=[sub])
    elapsed = time.perf_counter() - start

    (status,) = state.subcycle_status
    assert sum(float(total) for total, _dt, _t in results) == pytest.approx(1.0)
    assert status["reference_s"] == pytest.approx(1.0)
    assert status["tau"] < 1.0                   # it could not keep up...
    assert status["slip_s"] > 0.0                # ...and the gap is recorded
    # Had the lockstep side waited, the subcycle would have finished one of
    # its windows per lockstep window -- 50.  It finished far fewer: the 50
    # lockstep windows ran without joining on ~4 s of subcycle work.
    assert status["windows"] < 25, (status["windows"], elapsed)


def test_a_column_has_one_owner():
    columns = {"z": np.zeros(BATCH)}
    sub = Subcycle([clock()], round_dt=0.02, dx=1.0)
    with pytest.raises(ValueError, match="one owner"):
        dt_system([clock()], columns, rounds=1, round_dt=0.02, dx=1.0, subcycles=[sub])


# --------------------------------------------------------------------------
# the time-velocity mechanics: engine_toy's TimeField + StoreLedger
# --------------------------------------------------------------------------

def test_every_scope_is_a_time_field_node_nested_like_the_windows():
    columns = {"y": np.zeros(BATCH), "z": np.zeros(BATCH)}
    sub = Subcycle([clock()], round_dt=0.02, dx=1.0, name="clock",
                   coupling="differential")

    state, _ctrl, _results = dt_system([lockstep_reader()], columns, rounds=10,
                                       round_dt=0.02, dx=1.0, subcycles=[sub])

    record = state.time_velocity
    field_ = record.field
    assert set(field_.nodes) == {"lockstep", "lockstep/reader", "clock", "clock/clock"}
    assert len(record.log) == 10
    # the lockstep root landed every window: time velocity 1.0, measured
    assert field_.velocity("lockstep") == pytest.approx(1.0)
    assert all(entry["tau"] == pytest.approx(1.0) for entry in record.log)
    # a piece runs at its scope's rate; nesting composes by adding logs
    assert field_.effective_velocity("clock/clock") == pytest.approx(
        field_.velocity("clock") * field_.velocity("lockstep"))


def test_slow_subcycle_is_dilated_in_the_field_and_its_store_charges():
    from dataclasses import replace

    from src.common.dt_system import time_field

    geared = replace(time_field.ADAPTORS["differential"], store_inertia_kg_m2=0.5)
    columns = {"y": np.zeros(BATCH), "z": np.zeros(BATCH)}
    sub = Subcycle([clock(delay_s=0.02)], round_dt=0.02, dx=1.0, name="slow",
                   coupling=geared, omega_ref_rad_s=100.0)

    state, _ctrl, _results = dt_system([lockstep_reader()], columns, rounds=40,
                                       round_dt=0.02, dx=1.0, subcycles=[sub])

    log = [entry["subcycles"]["slow"] for entry in state.time_velocity.log]
    last = log[-1]
    assert last["velocity"] < 1.0                   # dilated relative to its reference
    assert last["gradient"] < 0.0
    assert last["admits"] is True                   # a differential carries it
    assert last["adaptor"] == "differential"
    assert last["shortfall_ema"] > 0.0              # leading edge: asked more than it got
    # the gradient changed, so the store had to absorb reaction * slip
    assert any(entry["reaction_nm"] > 0.0 for entry in log)
    assert max(entry["stored_j"] for entry in log) > 0.0
    assert last["persistence"] >= 3


def test_an_undeclared_coupling_is_rigid_and_reports_the_shear():
    columns = {"y": np.zeros(BATCH), "z": np.zeros(BATCH)}
    sub = Subcycle([clock(delay_s=0.02)], round_dt=0.02, dx=1.0, name="rigid",
                   omega_ref_rad_s=100.0)

    state, _ctrl, _results = dt_system([lockstep_reader()], columns, rounds=20,
                                       round_dt=0.02, dx=1.0, subcycles=[sub])

    log = [entry["subcycles"]["rigid"] for entry in state.time_velocity.log]
    assert log[-1]["adaptor"] == "torque-shaft"      # adaptor_for's conservative default
    assert log[-1]["admits"] is False
    assert any(entry["reaction_nm"] == float("inf") for entry in log)


def test_exchange_time_prefers_the_law_s_exchangeable_energy():
    # stored energy C*T from absolute zero, exchangeable C*|T - T_eq|: a cell
    # one kelvin from its neighbour relaxes in C/(hA) whatever T it sits at
    C, hA = 1000.0, 5.0

    def cell(T, T_n, dt):
        Q = hA * (T_n - T) * dt
        return (T + Q / C, 1.0, C * T, np.abs(Q) / dt, C * np.abs(Q) / (hA * dt))

    piece = Piece("cell", ("T", "T_n", "dt"),
                  ("T_next", "dt_limit", "energy_j", "power_w", "exchangeable_energy_j"), cell)
    columns = {"T": np.full(BATCH, 301.0), "T_n": np.full(BATCH, 300.0)}

    state, _ctrl, _results = dt_system([piece], columns, rounds=1, round_dt=0.01, dx=1.0)

    assert float(state.pub_exchange_time[0]) == pytest.approx(C / hA)       # 200 s, not 60 200 s


# --------------------------------------------------------------------------
# dt_graph defines how llvm_dt_system interprets its pieces
# --------------------------------------------------------------------------

def _round(children, *, label, round_dt=0.05, schedule="sequential"):
    from src.common.dt_system.dt import SuperstepPlan
    from src.common.dt_system.dt_controller import STController, Targets
    from src.common.dt_system.dt_graph import ControllerNode, RoundNode

    return RoundNode(
        plan=SuperstepPlan(round_max=round_dt, dt_init=round_dt),
        controller=ControllerNode(
            ctrl=STController(dt_min=round_dt * 1e-6),
            targets=Targets(cfl=0.5, div_max=1e9, mass_max=1e-3, energy_exchange_fraction=0.2),
            dx=1.0),
        children=list(children), schedule=schedule, label=label)


def _writer():
    return Piece("writer", ("a", "dt"), ("a_next", "dt_limit"), lambda a, dt: (a + 1.0, 1.0))


def _copier():
    return Piece("copier", ("a", "b", "dt"), ("b_next", "dt_limit"), lambda a, b, dt: (a, 1.0))


@pytest.mark.parametrize("schedule, expected_b", [("sequential", 1.0), ("parallel", 0.0)])
def test_the_round_schedule_is_the_read_discipline(schedule, expected_b):
    from llvm_dt_system import dt_system_from_graph, piece_leaf

    root = _round([piece_leaf(_writer()), piece_leaf(_copier())], label="root",
                  schedule=schedule)
    columns = {"a": np.zeros(BATCH), "b": np.zeros(BATCH)}
    dt_system_from_graph(root, columns, rounds=1)
    # one step: sequential -> the copier read the writer's new a; parallel -> the old a
    np.testing.assert_allclose(columns["a"], 1.0)
    np.testing.assert_allclose(columns["b"], expected_b)


def test_a_nested_round_is_the_dt_systems_own_subdivision():
    from llvm_dt_system import dt_system_from_graph, piece_leaf

    stiff = Piece("stiff", ("x", "v", "dt"), ("x_next", "dt_limit"),
                  lambda x, v, dt: (x + dt * v, 0.001))
    inner = _round([piece_leaf(stiff)], label="inner")
    root = _round([inner], label="root")
    x0, v = np.zeros(BATCH), np.array([1.0, 2.0, 3.0, 4.0])
    columns = {"x": x0.copy(), "v": v.copy()}

    state, _ctrl, results = dt_system_from_graph(root, columns, rounds=3)

    np.testing.assert_allclose(columns["x"], x0 + 0.15 * v, rtol=0, atol=1e-12)
    # the inner round subdivided; the parent was never capped by its 1 ms --
    # nothing binds the parent, so its proposal is free and each window lands
    for total, dt_next, _telemetry in results:
        assert float(total) == pytest.approx(0.05)
        assert float(dt_next) >= 0.05
    field_ = state.time_velocity.field
    assert "root/inner/stiff" in field_.nodes
    assert field_.velocity("root/inner") == pytest.approx(1.0)       # it landed every window


def test_a_piece_declares_its_contract():
    from src.common.dt_system.time_contracts import DILATE

    piece = drift()
    piece.contract = DILATE
    columns = {"x": np.zeros(BATCH), "v": np.ones(BATCH)}
    state, _ctrl, _results = dt_system([piece], columns, rounds=1, round_dt=0.05, dx=1.0)
    assert float(state.pub_contract[0]) == float(DILATE)


def test_a_graph_leaf_refuses_any_other_runner():
    from llvm_dt_system import piece_leaf

    leaf = piece_leaf(drift())
    assert leaf.state.state.entry == "drift"
    with pytest.raises(TypeError, match="interpreted by llvm_dt_system"):
        leaf.advance(None, 0.01)
