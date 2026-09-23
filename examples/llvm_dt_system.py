"""The dt system with LLVM pieces as its steps -- the common path.

    state, ctrl, results = dt_system(piece_files, columns, rounds=..., round_dt=..., dx=...)

This is ``run_superstep`` used directly, the same way it is used everywhere
else in the tree that already lowers (vehicle_python_compilation.py,
native_voxel_fluid.py, symbolic_fluid_dt.py, symbolic_fluid_native_runtime.py,
and this repo's own chamber_dt_join.py/chamber_sim.py): one ``state`` object
(with ``copy_shallow``/``restore`` for in-round rollback), one
``STController`` kept alive across frames, and one ``advance(state, dt)``
function that calls the pieces and returns ``(ok, metrics)``.  Nothing else
of the dt system is needed -- no ``DtCompatibleEngine``, no ``StateTable``,
no ``GraphBuilder``/``MetaLoopRunner`` (that layer is optional composition on
top and has never been lowered; see llvm_dt_system.py.bak for the discovery).

dt_graph's NODE TREE is how this module interprets its pieces
(``dt_system_from_graph``): a ``RoundNode``'s children in order are the
causal order, its ``schedule`` is the read discipline (``sequential`` --
same-step reads; ``parallel`` -- start-of-step reads), an ``AdvanceNode``
holds a piece in its ``StateNode`` (``piece_leaf``), and a nested
``RoundNode`` is the dt system's own subdivision (``RoundPiece``).  The tree
is read here; ``MetaLoopRunner`` does not run it.  Independent participants
(``Subcycle``) are consulted without waiting, on their own threads.

Python does the lowering.  The pieces decide the columns, so the state class
(one span field per column, exactly the tire's ``BalloonTireManagedState``
shape) and the advance function (one named call per piece) are spelled by
Python for the piece set at hand -- ``state_source``/``piece_source`` -- and
that same text is what runs in Python and what is handed to the compiler.
The emitted unit is the run-many-times part.

TIME VOCABULARY (decided 2026-09-22).  Three quantities that used to share
the word "tau" are kept apart here:

``exchange_time_s``
    A law's exchangeable energy over its exchange power, E/P: the seconds it
    would take its exchange to run its course at the rate it runs now.  E is
    the law's ``exchangeable_energy_j`` -- measured from where the exchange
    is going, so E/P is the true relaxation time -- falling back to its stored
    ``energy_j`` for a law that has not declared one.  This is
    the energy-stability metric that SCHEDULES dt, as
    ``dt <= Targets.energy_exchange_fraction * exchange_time_s``.  It is
    written into the dt system's per-participant scheduling slot,
    ``pub_exchange_time``.
Courant numbers
    Tracked per law, information only: the energy Courant number
    ``P * dt / E`` (the quantity the exchange fraction caps) and the transport
    Courant number ``dt / dt_limit``.  Nothing schedules on them.
``tau``
    Time velocity: the world time a system actually advanced, over the world
    time its reference asked for.  Measured, never set, and never an input to
    scheduling.  In a lockstep window it is ``advanced / round_dt`` (telemetry
    ``tau``).  For an independent ``Subcycle`` it is the participant's own world
    time over the world time of the system consulting it.

Wall cost (host seconds per world second) is a fourth, separate quantity:
``WallCostLedger``, measured by the host around each call, always on, and
information only.  It is never read by anything that chooses dt.
"""

import math
import sys
import threading
import time

import numpy as np

from src.common.dt_system.dt_controller import STController, Targets, run_superstep
from src.common.dt_system.dt_scaler import Metrics
from src.common.dt_system.participants import StepSpans
from src.common.dt_system.error_channels import DT_CHANNEL_NAMES
from src.common.tensors import AbstractTensor
from src.common.dt_system.time_contracts import BIND, DILATE, HOLD, SUBCYCLE, ParticipantRegistry
from src.compiler.native_law_kernels import LLVMPiece

C_BACKEND = "c"
LLVM_BACKEND = "llvm"
FORTRAN_BACKEND = "fortran"

METRIC_FIELDS = ("max_vel", "max_flux", "div_inf", "mass_err", "dt_limit",
                 "energy_j", "power_w", "exchangeable_energy_j")
TELEMETRY_FIELDS = ("advanced", "dt_next", "max_vel", "max_flux", "div_inf",
                    "mass_err", "dt_limit", "hard_failure", "tau")
#: The dt system's own per-participant publication spans (the controller reads these).
PUBLICATION_FIELDS = ("pub_exchange_time", "pub_exchange_time_present", "pub_contract", "pub_dt_limit",
                      "pub_dt_limit_present")
#: Per-law Courant numbers, tracked for information only; never passed to
#: ``Metrics``, so nothing that chooses dt can read them.
COURANT_FIELDS = ("pub_energy_courant", "pub_energy_courant_present",
                  "pub_dt_courant", "pub_dt_courant_present")


def column_names_of(pieces):
    """Every column some piece reads, in first-appearance order (``dt`` is
    the step's own column and is not one of them)."""

    names = []
    for piece in pieces:
        for name in piece.argument_names:
            if name != "dt" and name not in names:
                names.append(name)
    return tuple(names)


def state_source(columns, participants=1):
    """Spell ``PieceState`` for these columns: one span field per column,
    the ``dt`` column the step fills, and the window's telemetry span.
    ``copy_shallow``/``restore`` are the tire's: copies out, in-place back."""

    fields = (*columns, "dt")
    lines = ["class PieceState:"]
    lines.append(f"    def __init__(self, {', '.join(fields)}, telemetry):")
    for name in fields:
        lines.append(f"        self.{name} = {name}")
    lines.append("        self.telemetry = telemetry")
    lines.append(f"        self.channel_names = {DT_CHANNEL_NAMES!r}")
    for field in (*PUBLICATION_FIELDS, *COURANT_FIELDS):
        lines.append(f"        self.{field} = AbstractTensor.zeros(({participants},))")
    for field in ("pub_values", "pub_present", "pub_limits", "pub_limits_present"):
        lines.append(f"        self.{field} = AbstractTensor.zeros(({participants * len(DT_CHANNEL_NAMES)},))")
    lines.append("")
    lines.append("    def copy_shallow(self):")
    lines.append("        return (")
    for name in fields:
        lines.append(f"            self.{name}.copy(),")
    lines.append("        )")
    lines.append("")
    lines.append("    def restore(self, snapshot):")
    lines.append(f"        {', '.join(fields)}, = snapshot")
    for name in fields:
        lines.append(f"        self.{name}[...] = {name}")
    return "\n".join(lines) + "\n"


SCHEDULES = ("sequential", "parallel")


def piece_source(pieces, schedule="sequential"):
    """Spell ``advance_pieces(state, dt)`` for exactly these pieces.

    ``schedule`` is the round's read discipline, as ``dt_graph.RoundNode``
    spells it: ``"sequential"`` -- piece i reads what pieces before it wrote
    this step (each write lands before the next call); ``"parallel"`` -- every
    piece reads the state the step started from, and the writes land after the
    last call.  Which one is right is the modeller's choice at the coupling
    (same-step versus lagged), stated in the graph rather than implied by list
    position.  A piece may declare ``contract`` (``BIND``/``HOLD``/``DILATE``/
    ``SUBCYCLE``); undeclared, it binds when it exchanges and holds when it
    does not.

    Every piece is called by its own bound name (``step_0``, ``step_1``,
    ...) with its columns spelled out, every ``<name>_next`` output is
    assigned to its column, and the published metrics are folded
    explicitly -- maxima of the error measures, the minimum published
    ``dt_limit`` (+inf when none is published; the controller's minimum
    against it is a no-op), the energy/power channels summed.
    """

    if schedule not in SCHEDULES:
        raise NotImplementedError(
            f"schedule={schedule!r}: llvm_dt_system interprets {SCHEDULES}")
    columns = column_names_of(pieces)
    lines = ["def advance_pieces(state, dt):"]
    lines.append("    state.dt[...] = dt")
    folds = {name: [] for name in METRIC_FIELDS}
    deferred = []
    for index, piece in enumerate(pieces):
        arguments = ", ".join(f"state.{name}" for name in piece.argument_names)
        outputs = [f"o{index}_{name}" for name in piece.output_names]
        lines.append(f"    {', '.join(outputs)}, = step_{index}({arguments})")
        for name, output in zip(piece.output_names, outputs):
            if name.endswith("_next") and name[:-5] in columns:
                # In place: the column is the caller's buffer (the same
                # write ``restore`` makes), never a rebinding of the field.
                write = f"    state.{name[:-5]}[...] = {output}"
                if schedule == "parallel":
                    deferred.append(write)
                else:
                    lines.append(write)
            if name in folds:
                folds[name].append(output)
    # parallel: every piece read the start-of-step state; the writes land now
    lines.extend(deferred)

    # Each law publishes what IT measured, keyed by its own identity, in the
    # order the laws ran -- which is the causal order, because law i may consume
    # what law i-1 just published.  dt_system amalgamates what is system-wide
    # and gates what is individual; this step's job is to report honestly, not
    # to decide.
    lines.append("")
    lines.append("    # what each law measured, in causal order")
    for index, piece in enumerate(pieces):
        published = set(piece.output_names)
        lines.append(f"    # -- {piece.entry}")
        for name in METRIC_FIELDS:
            if name not in published:
                continue
            reducer = "min" if name == "dt_limit" else "max"
            # Keep the reduced 0-d tensor in the tensorized report. The dt
            # boundary's ``coerce_metrics`` returns canonical Metrics records
            # unchanged; extracting here would split the tensorized handoff
            # with a structural scalar call result.
            lines.append(f"    m{index}_{name} = o{index}_{name}.{reducer}()")
        # exchange_time_s is this law's own energy over its own power: the time
        # it would take to exchange its stored energy at the rate it is
        # exchanging now.  It is the energy-stability metric that schedules dt,
        # so it goes into the dt system's scheduling slot
        # (``pub_exchange_time``) and stays attached to the law that measured it.
        # It is NOT this module's tau, which is time velocity (see the module
        # docstring).  Write declared buffers directly. No Publication
        # constructor or keyed table result crosses this call boundary. Every
        # attempt writes its masks.
        # The energy a law's exchange can actually move is measured from where
        # that exchange is going (``exchangeable_energy_j``, derived by the law
        # from its own Jacobian), not from absolute zero; a law that has not
        # declared it falls back to its stored ``energy_j``.
        energy = "exchangeable_energy_j" if "exchangeable_energy_j" in published else "energy_j"
        if energy in published and "power_w" in published:
            lines.append(f"    m{index}_exchange_time_s = m{index}_{energy} / m{index}_power_w if m{index}_power_w > 0.0 else 0.0")
            lines.append(f"    state.pub_exchange_time_present[{index}] = float(m{index}_power_w > 0.0)")
            lines.append(f"    state.pub_exchange_time[{index}] = m{index}_exchange_time_s")
            declared = getattr(piece, "contract", None)
            lines.append(f"    state.pub_contract[{index}] = "
                         + (f"{float(declared)}" if declared is not None
                            else f"BIND if m{index}_power_w > 0.0 else HOLD"))
            # Energy Courant number, P*dt/E: the fraction of the stored energy
            # exchanged in this step -- what energy_exchange_fraction caps.
            # Tracked only; absent (not zero) when there is no stored energy.
            lines.append(f"    state.pub_energy_courant_present[{index}] = float(m{index}_{energy} > 0.0)")
            lines.append(f"    state.pub_energy_courant[{index}] = m{index}_power_w * dt / m{index}_{energy} if m{index}_{energy} > 0.0 else 0.0")
        else:
            lines.append(f"    state.pub_exchange_time[{index}] = 0.0")
            lines.append(f"    state.pub_exchange_time_present[{index}] = 0.0")
            declared = getattr(piece, "contract", None)
            lines.append(f"    state.pub_contract[{index}] = "
                         + (f"{float(declared)}" if declared is not None else "HOLD"))
            lines.append(f"    state.pub_energy_courant[{index}] = 0.0")
            lines.append(f"    state.pub_energy_courant_present[{index}] = 0.0")
        floor = f"m{index}_dt_limit" if "dt_limit" in published else "0.0"
        lines.append(f"    state.pub_dt_limit[{index}] = {floor}")
        lines.append(f"    state.pub_dt_limit_present[{index}] = {float('dt_limit' in published)}")
        # Transport Courant number, dt/dt_limit: 1.0 is the law's own
        # stability floor.  Tracked only.
        if "dt_limit" in published:
            lines.append(f"    state.pub_dt_courant_present[{index}] = float(m{index}_dt_limit > 0.0)")
            lines.append(f"    state.pub_dt_courant[{index}] = dt / m{index}_dt_limit if m{index}_dt_limit > 0.0 else 0.0")
        else:
            lines.append(f"    state.pub_dt_courant[{index}] = 0.0")
            lines.append(f"    state.pub_dt_courant_present[{index}] = 0.0")
        for channel, name in enumerate(DT_CHANNEL_NAMES):
            slot = index * len(DT_CHANNEL_NAMES) + channel
            measured = name in published and name in METRIC_FIELDS
            value = f"m{index}_{name}" if measured else "0.0"
            lines.append(f"    state.pub_values[{slot}] = {value}")
            lines.append(f"    state.pub_present[{slot}] = {float(measured)}")

    def fold(operator, terms, empty):
        if not terms:
            return empty
        if len(terms) == 1:
            return terms[0]
        return f"{operator}({', '.join(terms)})"

    # The returned Metrics is the amalgamated REPORT, not the decision: the
    # per-law rows above are what the controller gates on.  Extensives are
    # summed because energy and power are extensive; the error measures are
    # maxed, which is a worst-offender report and says nothing about WHICH law
    # it came from -- the rows do.
    lines.append("")
    lines.append("    # the amalgamated report; the rows above are the decision")
    for name in ("max_vel", "max_flux", "div_inf", "mass_err"):
        terms = [f"m{index}_{name}" for index, piece in enumerate(pieces)
                 if name in set(piece.output_names)]
        lines.append(f"    {name} = " + fold("max", terms, "0.0"))
    terms = [f"m{index}_dt_limit" for index, piece in enumerate(pieces)
             if "dt_limit" in set(piece.output_names)]
    lines.append("    dt_limit = " + fold("min", terms, 'float("inf")'))
    for name in ("energy_j", "power_w"):
        terms = " + ".join(f"m{index}_{name}" for index, piece in enumerate(pieces)
                           if name in set(piece.output_names))
        lines.append(f"    {name} = {terms or '0.0'}")
    lines.append("    metrics = Metrics(")
    lines.append("        max_vel=max_vel, max_flux=max_flux, div_inf=div_inf,")
    lines.append("        mass_err=mass_err, dt_limit=dt_limit,")
    report = [name if any(name in p.output_names for p in pieces) else "0.0"
              for name in DT_CHANNEL_NAMES]
    # Only fields reduced above are part of the aggregate report.
    report = [value if name in METRIC_FIELDS else "0.0"
              for name, value in zip(DT_CHANNEL_NAMES, report)]
    flags = [float(name in METRIC_FIELDS and any(name in p.output_names for p in pieces))
             for name in DT_CHANNEL_NAMES]
    lines.append(f"        error_channels=AbstractTensor.tensor([{', '.join(report)}]),")
    lines.append(f"        error_present=AbstractTensor.tensor({flags}),")
    for field in (*PUBLICATION_FIELDS,
                  "pub_values", "pub_present", "pub_limits", "pub_limits_present"):
        lines.append(f"        {field}=state.{field},")
    lines.append("    )")
    lines.append("    return True, metrics")
    return "\n".join(lines) + "\n"


def generated_source(pieces, schedule="sequential"):
    return state_source(column_names_of(pieces), len(pieces)) + "\n\n" + piece_source(pieces, schedule)


def bind_namespace(pieces, *, wrap=None, schedule="sequential"):
    """Exec the generated ``PieceState``/``advance_pieces`` for these pieces
    into a fresh namespace and return it, touching no module global.

    ``wrap(index, piece) -> callable`` lets the Python host put a measurement
    around each call (the wall-cost ledger).  The generated text is the same
    either way; only what ``step_i`` names differs, and only in Python.
    """

    bindings = {
        f"step_{index}": piece if wrap is None else wrap(index, piece)
        for index, piece in enumerate(pieces)
    }
    namespace = {
        "np": np, "Metrics": Metrics,
        # the dt system's own publication vocabulary: a law states what it
        # measured and how its exchange time participates, and nothing here
        # decides
        "StepSpans": StepSpans, "AbstractTensor": AbstractTensor, "BIND": BIND, "HOLD": HOLD,
        "DILATE": DILATE, "SUBCYCLE": SUBCYCLE,
        **bindings,
    }
    exec(generated_source(pieces, schedule), namespace)
    return namespace


def bind_pieces(pieces, *, wrap=None, schedule="sequential"):
    """Bind ``PieceState`` and ``advance_pieces`` for these pieces in this
    module: the Python path runs the very text the lowering is given."""

    namespace = bind_namespace(pieces, wrap=wrap, schedule=schedule)
    globals()["PieceState"] = namespace["PieceState"]
    globals()["advance_pieces"] = namespace["advance_pieces"]
    return {f"step_{index}": piece for index, piece in enumerate(pieces)}


def participant_registry(pieces):
    """Declare each law as a participant, once, in causal order.

    Identity is assigned at declaration and the id IS the index into every span
    the dt system builds, so this is a build-time act and the order is the order
    the laws run in.
    """

    registry = ParticipantRegistry()
    for piece in pieces:
        registry.declare(str(piece.entry))
    return registry


def dt_system_over(state, targets, controller, round_dt, dt_initial, dx):
    """One round of the dt system over the bound pieces: the function that
    lowers, shaped exactly like the tire's ``balloon_tire_managed_window``.
    The caller owns ``state``/``targets``/``controller`` across rounds; the
    round's results are published on ``state.telemetry``."""

    advanced, dt_next, metrics = run_superstep(
        state, round_dt, dt_initial, dx, targets, controller, advance_pieces)
    publish_window(state, advanced, dt_next, metrics, round_dt)
    return advanced, dt_next


def publish_window(state, advanced, dt_next, metrics, round_dt):
    """Write one window's results to ``state.telemetry``.

    ``tau`` is the window's time velocity: world time advanced over world
    time asked for.  It is 1.0 whenever the window landed, and less only when
    ``run_superstep`` stopped short (``max_iters`` exhausted); it is measured
    here and read by nothing that chooses dt.
    """

    state.telemetry[0] = advanced
    state.telemetry[1] = dt_next
    state.telemetry[2] = metrics.max_vel
    state.telemetry[3] = metrics.max_flux
    state.telemetry[4] = metrics.div_inf
    state.telemetry[5] = metrics.mass_err
    state.telemetry[6] = metrics.dt_limit if metrics.dt_limit is not None else 0.0
    state.telemetry[7] = float(metrics.hard_failure)
    state.telemetry[8] = advanced / round_dt if round_dt > 0.0 else 0.0


def load_pieces(piece_files):
    """Pieces from ``.piece`` files; an item that already is a piece (it
    declares ``argument_names``/``output_names``) is taken as it is."""

    return [
        item if hasattr(item, "argument_names") and hasattr(item, "output_names")
        else LLVMPiece.load(item)
        for item in piece_files
    ]


def owned_columns(pieces, columns=None):
    """Columns these pieces write: every ``<name>_next`` output whose
    ``<name>`` is one of the columns."""

    columns = column_names_of(pieces) if columns is None else tuple(columns)
    owned = []
    for piece in pieces:
        for name in piece.output_names:
            if name.endswith("_next") and name[:-5] in columns and name[:-5] not in owned:
                owned.append(name[:-5])
    return tuple(owned)


class WallCostLedger:
    """Host seconds spent against world seconds advanced. INFORMATION ONLY.

    Measured by the host around each piece call and each window, outside
    anything that lowers: the compiled step is deterministic and has no wall
    clock in it.  Always on, because a profiler that has to be switched on
    measures a different program.  Nothing that chooses dt, fidelity or a
    window reads this; it is a record, keyed by participant (the same causal
    order ``participant_registry`` declares).
    """

    def __init__(self, names):
        self.names = tuple(names)
        self.calls = [0] * len(self.names)
        self.piece_wall_s = [0.0] * len(self.names)
        self.windows = 0
        self.wall_s = 0.0
        self.world_s = 0.0

    def wrap(self, index, piece):
        def measured(*columns):
            start = time.perf_counter()
            try:
                return piece(*columns)
            finally:
                self.piece_wall_s[index] += time.perf_counter() - start
                self.calls[index] += 1
        measured.__name__ = f"measured_step_{index}"
        return measured

    def window(self, wall_s, world_s):
        self.windows += 1
        self.wall_s += float(wall_s)
        self.world_s += float(world_s)

    @property
    def wall_cost(self):
        """Host seconds per world second (above 1.0: slower than realtime)."""
        return self.wall_s / self.world_s if self.world_s > 0.0 else float("inf")

    def report(self):
        return {
            "windows": self.windows, "wall_s": self.wall_s, "world_s": self.world_s,
            "wall_cost": self.wall_cost,
            "participants": {
                name: {"calls": calls, "wall_s": wall,
                       "wall_cost": wall / self.world_s if self.world_s > 0.0 else float("inf")}
                for name, calls, wall in zip(self.names, self.calls, self.piece_wall_s)
            },
        }


class Subcycle:
    """An INDEPENDENT participant: its own pieces, its own dt system, its own
    clock, running on its own ``threading.Thread``.

    This is what the ``SUBCYCLE`` contract means.  The system that consults
    it is never lockstepped with it and never waits for it:

    * ``consult()`` takes the latest publication under the condition and
      returns at once -- the participant's owned columns, stamped with the
      world time they are true at.  Whatever is not fresh is read lagged.
    * ``offer(values, world_s)`` hands it the consulting system's current
      columns and world time, and notifies it.  It reads those lagged too.
    * It never runs more than ``lead_windows`` of its own windows ahead of the
      world time it was last offered, so a fast participant waits for its
      reference and a slow one simply falls behind.

    ``tau`` is its time velocity: its world time over its reference's world
    time.  ``slip_s`` is how far behind the reference its publication is.
    Both are measured, never set.  Threading is the stdlib ``Thread``/
    ``Condition``/``Event`` the compiler already recognises as dispatcher
    operations (``python_special_cases.lower_python_threading``).
    """

    def __init__(self, piece_files, *, round_dt, dx, targets=None, controller=None,
                 name="subcycle", lead_windows=1.0, wait_timeout_s=0.05,
                 coupling=None, omega_ref_rad_s=0.0):
        self.name = str(name)
        #: What sits across the boundary to the consulting system: a joint
        #: kind or a declared ``time_field.TimeAdaptor``.  Undeclared is
        #: treated as RIGID by ``time_field.adaptor_for`` -- the conservative
        #: answer -- so an independent participant coupled without a declared
        #: rate freedom reports a shear whenever its gradient changes.
        self.coupling = coupling
        #: Reference angular rate of the coupling's freedom, for the shear
        #: reaction and the slip rate.  0.0: not a rotating coupling, so the
        #: store sees no power (the energy build-up stays at zero, honestly).
        self.omega_ref_rad_s = float(omega_ref_rad_s)
        self.pieces = load_pieces(piece_files)
        self.round_dt = float(round_dt)
        self.dx = float(dx)
        self.targets = targets or Targets(cfl=0.5, div_max=1e9, mass_max=1e-3,
                                          energy_exchange_fraction=0.2)
        self.controller = controller or STController(dt_min=self.round_dt * 1e-6)
        self.lead_windows = float(lead_windows)
        self.wait_timeout_s = float(wait_timeout_s)
        self.names = column_names_of(self.pieces)
        self.owned = owned_columns(self.pieces, self.names)
        self.ledger = WallCostLedger(str(piece.entry) for piece in self.pieces)
        namespace = bind_namespace(self.pieces, wrap=self.ledger.wrap)
        self._state_class = namespace["PieceState"]
        self._advance = namespace["advance_pieces"]
        self.state = None
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.thread = None
        self.world_s = 0.0
        self.reference_s = 0.0
        self.windows = 0
        self.error = None
        self._inbox = {}
        self._outbox = {}
        self._stamp = 0.0

    def attach(self, columns):
        """Build this participant's own state from the shared columns."""
        self.state = self._state_class(
            *(np.array(columns[name], dtype=np.float64) for name in self.names),
            np.zeros((self.pieces[0].batch,), dtype=np.float64),
            np.zeros((len(TELEMETRY_FIELDS),), dtype=np.float64),
        )
        self.state.participants = participant_registry(self.pieces)
        configure_publication_limits(self.state, self.targets)
        self._outbox = {name: getattr(self.state, name).copy() for name in self.owned}
        self._stamp = 0.0

    def start(self):
        if self.state is None:
            raise RuntimeError(f"{self.name}: attach() before start()")
        self.thread = threading.Thread(target=self._run, args=(self.stop_event,),
                                       name=f"subcycle-{self.name}", daemon=True)
        self.thread.start()

    def _run(self, stop):
        dt = self.round_dt
        try:
            while True:
                with self.condition:
                    while (not stop.is_set() and self.world_s
                           >= self.reference_s + self.lead_windows * self.round_dt):
                        self.condition.wait(timeout=self.wait_timeout_s)
                    if stop.is_set():
                        return
                    for name, value in self._inbox.items():
                        if name not in self.owned:
                            getattr(self.state, name)[...] = value
                start = time.perf_counter()
                advanced, dt, metrics = run_superstep(
                    self.state, self.round_dt, dt, self.dx, self.targets,
                    self.controller, self._advance)
                publish_window(self.state, advanced, dt, metrics, self.round_dt)
                self.ledger.window(time.perf_counter() - start, float(advanced))
                with self.condition:
                    self.world_s += float(advanced)
                    self.windows += 1
                    self._outbox = {name: getattr(self.state, name).copy()
                                    for name in self.owned}
                    self._stamp = self.world_s
                    self.condition.notify_all()
        except BaseException as error:  # reported by consult()/stop(), never swallowed
            with self.condition:
                self.error = error
                self.condition.notify_all()

    def consult(self):
        """``(publication, stamp_s)`` -- never waits for a window to finish."""
        with self.condition:
            if self.error is not None:
                raise RuntimeError(f"subcycle {self.name!r} failed") from self.error
            return dict(self._outbox), self._stamp

    def offer(self, values, world_s):
        with self.condition:
            self._inbox = {name: np.array(value, dtype=np.float64, copy=True)
                           for name, value in values.items()}
            self.reference_s = float(world_s)
            self.condition.notify_all()

    def status(self):
        with self.condition:
            reference = self.reference_s
            return {
                "name": self.name, "windows": self.windows, "world_s": self.world_s,
                "reference_s": reference, "published_stamp_s": self._stamp,
                "tau": self.world_s / reference if reference > 0.0 else float("nan"),
                "slip_s": reference - self._stamp,
                "wall_cost": self.ledger.report(),
            }

    def stop(self, timeout_s=10.0):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=timeout_s)
        if self.error is not None:
            raise RuntimeError(f"subcycle {self.name!r} failed") from self.error


def time_mechanics():
    """The dt system's time-velocity mechanics: ``TimeField`` (log time
    velocity and its derivative per node, nested scopes, adaptors, shear
    reaction) and ``StoreLedger`` (the energy the time force store has had to
    absorb), both in ``src/common/dt_system/time_field.py``."""

    from src.common.dt_system import time_field

    return time_field


class TimeVelocityRecord:
    """``llvm_dt_system``'s time-velocity record, kept in the time field.

    Every scope is a ``TimeField`` node, nested the way the windows nest:
    the lockstep system is the root, each lockstep piece runs in it, each
    ``Subcycle`` is a child scope with its own pieces inside it.  After every
    lockstep window the MEASURED time velocity of every scope goes into the
    field -- never a target, and not rate limited on the way in (the same
    choice ``time_trials/race.py`` makes): the lockstep root at
    ``advanced / round_dt``, a subcycle at its published world time over the
    reference world time.  The field then derives ``dlog_tau_dt``.

    Each subcycle boundary is a coupling with a declared adaptor.  Per window
    it reports the ratio and gradient across the joint, whether the joint
    admits that gradient, the shear reaction while the gradient CHANGES, and
    folds reaction * slip into a ``StoreLedger`` -- the energy build-up the
    time force store is holding, with the asked/got shortfall as its leading
    edge.  All of it is information: nothing here moves dt or any column.
    """

    def __init__(self, root, pieces, subcycles):
        time_field = time_mechanics()
        self._time_field = time_field
        self.root = str(root)
        # (node, parent, piece) for every scope, nested the way the windows
        # nest: a RoundPiece's own pieces sit inside it, to any depth.
        self._scopes = []

        def declare(scope, owner, members):
            for piece in members:
                node = f"{scope}/{piece.entry}"
                self._scopes.append((node, owner, piece))
                inner = getattr(piece, "pieces", None)
                if isinstance(piece, RoundPiece) and inner:
                    declare(node, node, inner)

        declare(self.root, self.root, pieces)
        for sub in subcycles:
            self._scopes.append((sub.name, self.root, sub))
            declare(sub.name, sub.name, sub.pieces)
        nodes = [self.root, *(node for node, _owner, _piece in self._scopes)]
        if len(set(nodes)) != len(nodes):
            raise ValueError(f"time scopes must be uniquely named: {nodes}")
        self.field = time_field.TimeField.flat(nodes)
        for node, owner, _piece in self._scopes:
            self.field.set_parent(node, owner)
        self.subcycles = tuple(subcycles)
        self.ledgers = {sub.name: time_field.StoreLedger() for sub in self.subcycles}
        self._seen_stamp = {sub.name: 0.0 for sub in self.subcycles}
        self.log = []

    def _measure(self, node, velocity, dt):
        self.field.set_target(node, math.log(max(float(velocity), 1e-9)), dt, math.inf)

    def window(self, advanced, round_dt, world_s):
        """Fold one lockstep window into the field and the ledgers."""
        dt = float(round_dt)
        velocity = float(advanced) / dt if dt > 0.0 else 0.0
        self._measure(self.root, velocity, dt)
        for node, _owner, piece in self._scopes:
            if isinstance(piece, Subcycle):
                continue                       # measured against the reference below
            # a piece runs at its scope's rate; a nested round at its own
            # measured advanced/asked
            self._measure(node, getattr(piece, "tau", 1.0) if isinstance(piece, RoundPiece) else 1.0, dt)
        record = {"world_s": float(world_s), "tau": velocity, "subcycles": {}}
        for sub in self.subcycles:
            status = sub.status()
            stamp = float(status["published_stamp_s"])
            reference = float(status["reference_s"])
            local = stamp / reference if reference > 0.0 else 1.0
            self._measure(sub.name, local, dt)
            kind = sub.coupling
            ratio = self.field.ratio(sub.name, self.root)
            reaction = self.field.shear_reaction_nm(sub.name, self.root, kind,
                                                    sub.omega_ref_rad_s)
            slip = sub.omega_ref_rad_s * abs(ratio - 1.0)
            got = max(0.0, stamp - self._seen_stamp[sub.name])
            self._seen_stamp[sub.name] = stamp
            ledger = self.ledgers[sub.name]
            ledger.observe(reaction_nm=reaction, slip_rad_s=slip, dt_s=dt,
                           asked_s=float(advanced), got_s=got)
            ledger.relax(dt)
            adaptor = self._time_field.adaptor_for(kind)
            record["subcycles"][sub.name] = {
                "velocity": self.field.velocity(sub.name),
                "effective_velocity": self.field.effective_velocity(sub.name),
                "dlog_tau_dt": self.field.dlog_tau_dt[self.field.nodes.index(sub.name)],
                "gradient": self.field.gradient(sub.name, self.root),
                "gradient_rate": self.field.gradient_rate(sub.name, self.root),
                "adaptor": adaptor.kind,
                "admits": self.field.admits(sub.name, self.root, kind),
                "reaction_nm": reaction,
                "slip_rad_s": slip,
                "stored_j": ledger.stored_j,
                "persistence": ledger.persistence,
                "shortfall_ema": ledger.shortfall_ema,
                "slip_s": float(status["slip_s"]),
            }
        self.log.append(record)
        return record


# --------------------------------------------------------------------------
# dt_graph: how llvm_dt_system interprets the pieces it is given
# --------------------------------------------------------------------------

def _interpreted_only(_state, _dt):
    raise TypeError(
        "this AdvanceNode names an llvm_dt_system piece; the tree is interpreted "
        "by llvm_dt_system.dt_system_from_graph, not run by dt_graph.MetaLoopRunner")


def piece_leaf(piece, label=None):
    """A ``dt_graph.AdvanceNode`` naming ``piece``.

    The piece is the leaf's ``StateNode.state`` -- the simulator subset it
    advances -- and the node is read by ``dt_system_from_graph``; its
    ``advance`` refuses to be called by any other runner.
    """
    from src.common.dt_system.dt_graph import AdvanceNode, StateNode

    name = str(label or piece.entry)
    return AdvanceNode(advance=_interpreted_only, state=StateNode(piece, label=name), label=name)


class RoundPiece:
    """A nested ``dt_graph.RoundNode`` as one piece of its parent's step.

    This is subdivision BY THE DT SYSTEM -- not ``SUBCYCLE``.  Given the
    parent's attempt ``dt``, it runs its own ``run_superstep`` over exactly
    that window with its own controller (the node's ``ControllerNode``),
    subdividing as its pieces' exchange times and ``dt_limit`` demand, and
    must land it: a window it cannot land is refused, the dt_graph rule
    ("restore its own checkpoint and raise; the parent attempt fails").  It
    publishes no exchange time and no ``dt_limit`` -- it lands whatever the
    parent asks -- so it binds nobody, and it declares ``BIND`` so that
    absence reads as "no bound" rather than ``HOLD``'s "do not grow".
    """

    contract = BIND

    def __init__(self, node, *, wrap=None):
        self.node = node
        self.entry = str(node.label)
        self.pieces, self.schedule = interpret_round(node, wrap=wrap)
        self.batch = self.pieces[0].batch
        self.names = column_names_of(self.pieces)
        self.owned = owned_columns(self.pieces, self.names)
        self.argument_names = (*self.names, "dt")
        self.output_names = tuple(f"{name}_next" for name in self.owned)
        control = node.controller
        self.targets = control.targets
        self.controller = control.ctrl
        self.dx = float(control.dx)
        self.dt_inner = float(node.plan.dt_init)
        namespace = bind_namespace(self.pieces, wrap=wrap, schedule=self.schedule)
        self._advance = namespace["advance_pieces"]
        self.state = namespace["PieceState"](
            *(np.zeros(self.batch, dtype=np.float64) for _ in self.names),
            np.zeros((self.batch,), dtype=np.float64),
            np.zeros((len(TELEMETRY_FIELDS),), dtype=np.float64),
        )
        self.state.participants = participant_registry(self.pieces)
        configure_publication_limits(self.state, self.targets)
        #: time velocity of the last window: advanced / asked (1.0 when landed)
        self.tau = 1.0

    def __call__(self, *columns):
        *values, dt = columns
        window = float(np.asarray(dt).reshape(-1)[0])
        for name, value in zip(self.names, values):
            getattr(self.state, name)[...] = value
        advanced, self.dt_inner, metrics = run_superstep(
            self.state, window, self.dt_inner, self.dx, self.targets,
            self.controller, self._advance)
        publish_window(self.state, advanced, self.dt_inner, metrics, window)
        self.tau = float(advanced) / window if window > 0.0 else 1.0
        if abs(float(advanced) - window) > 1e-12 * max(1.0, window):
            raise RuntimeError(
                f"nested round {self.entry!r} advanced {float(advanced)!r} of "
                f"{window!r}: a nested round must land its parent's window")
        return tuple(getattr(self.state, name).copy() for name in self.owned)


def interpret_round(node, *, wrap=None):
    """``(pieces, schedule)`` for one ``dt_graph.RoundNode``.

    Children in order are the causal order.  An ``AdvanceNode`` leaf is the
    piece held in its ``StateNode``; a nested ``RoundNode`` is a
    ``RoundPiece``.  ``schedule`` is the node's own.
    """
    from src.common.dt_system.dt_graph import AdvanceNode, RoundNode

    if node.schedule not in SCHEDULES:
        raise NotImplementedError(
            f"RoundNode {node.label!r}: schedule={node.schedule!r}; "
            f"llvm_dt_system interprets {SCHEDULES}")
    pieces = []
    for child in node.children:
        if isinstance(child, RoundNode):
            pieces.append(RoundPiece(child, wrap=wrap))
        elif isinstance(child, AdvanceNode):
            piece = child.state.state
            if not (hasattr(piece, "argument_names") and hasattr(piece, "output_names")):
                raise TypeError(f"AdvanceNode {child.label!r} does not hold a piece")
            pieces.append(piece)
        else:
            raise TypeError(f"RoundNode {node.label!r}: unknown child {type(child).__name__}")
    if not pieces:
        raise ValueError(f"RoundNode {node.label!r} has no pieces")
    return pieces, node.schedule


def dt_system_from_graph(root, columns, *, rounds, subcycles=()):
    """Run ``dt_system`` as the ``dt_graph.RoundNode`` tree ``root`` defines.

    The root's ``plan.round_max`` is the window, its ``plan.dt_init`` the
    first attempt, its ``ControllerNode`` the controller, targets and ``dx``,
    its ``schedule`` the read discipline, its children the causal order and
    its nested rounds the dt system's own subdivision.  ``subcycles`` are the
    independent participants, consulted without waiting as in ``dt_system``.
    """

    pieces, schedule = interpret_round(root)
    control = root.controller
    return dt_system(pieces, columns, rounds=rounds, round_dt=float(root.plan.round_max),
                     dx=float(control.dx), targets=control.targets, controller=control.ctrl,
                     subcycles=subcycles, scope=str(root.label), schedule=schedule,
                     dt_initial=float(root.plan.dt_init))


def dt_system(piece_files, columns, *, rounds, round_dt, dx, targets=None, controller=None,
              subcycles=(), scope="lockstep", schedule="sequential", dt_initial=None):
    """Load the pieces from their files and run ``rounds`` rounds of the dt
    system over them in Python, the way the native unit will be driven.

    ``subcycles`` are independent participants (``Subcycle``).  Each round
    the lockstep system consults them without waiting, reads their owned
    columns as published (lagged by ``slip_s``), steps, and offers them its
    columns and world time.  A column has one owner: a lockstep piece may not
    write a column a subcycle owns.

    ``scope`` names the lockstep system's node in the time field.  After every
    window the measured time velocities go into ``state.time_velocity``
    (a ``TimeVelocityRecord``): the ``TimeField`` itself, a per-window log, and
    one ``StoreLedger`` per subcycle coupling tracking the energy build-up.
    """

    pieces = load_pieces(piece_files)
    ledger = WallCostLedger(str(piece.entry) for piece in pieces)
    bind_pieces(pieces, wrap=ledger.wrap, schedule=schedule)
    batch = pieces[0].batch
    names = column_names_of(pieces)
    subcycles = tuple(subcycles)
    mine = set(owned_columns(pieces, names))
    for sub in subcycles:
        clash = mine.intersection(sub.owned)
        if clash:
            raise ValueError(f"columns {sorted(clash)} are written by both the lockstep "
                             f"pieces and subcycle {sub.name!r}; a column has one owner")
    targets = targets or Targets(cfl=0.5, div_max=1e9, mass_max=1e-3, energy_exchange_fraction=0.2)
    # A floor, always.  Without ``dt_min`` a step that keeps being rejected
    # halves until halving a float64 stops changing it -- about 1074 times, and
    # ``run_superstep`` may do that for up to ``max_iters`` substeps, each
    # halving calling every law again.  The result is not an error but a run
    # that appears to hang, which is the worst of the three outcomes.
    #
    # The floor is a fraction of the window rather than an invented constant,
    # because this is supposed to work for any simulation and an absolute
    # number cannot be right for all of them.  A step smaller than a millionth
    # of the window means over a million substeps to cross it once: that is a
    # failure to report, not progress to keep making.  A caller who knows its
    # own physical floor passes its own controller and that wins.
    controller = controller or STController(dt_min=float(round_dt) * 1e-6)
    state = PieceState(
        *(np.array(columns[name], dtype=np.float64) for name in names),
        np.zeros((batch,), dtype=np.float64),
        np.zeros((len(TELEMETRY_FIELDS),), dtype=np.float64),
    )
    # The laws declare themselves once, in causal order, and the state carries
    # the registry so the controller can index the rows the step publishes.
    state.participants = participant_registry(pieces)
    state.wall_cost_ledger = ledger
    configure_publication_limits(state, targets)
    for sub in subcycles:
        sub.attach(columns)
    for sub in subcycles:
        sub.start()
    # The time-velocity record: every scope a node of the time field, the
    # measured velocities folded in after every window.  Information only.
    time_record = TimeVelocityRecord(scope, pieces, subcycles)
    state.time_velocity = time_record
    dt = round_dt if dt_initial is None else float(dt_initial)
    world_s = 0.0
    results = []
    try:
        for _round in range(rounds):
            for sub in subcycles:
                publication, _stamp = sub.consult()
                for name, value in publication.items():
                    if name in names:
                        getattr(state, name)[...] = value
            start = time.perf_counter()
            total, dt = dt_system_over(state, targets, controller, round_dt, dt, dx)
            ledger.window(time.perf_counter() - start, float(total))
            world_s += float(total)
            for sub in subcycles:
                sub.offer({column: getattr(state, column) for column in sub.names
                           if column in names}, world_s)
            time_record.window(total, round_dt, world_s)
            results.append((total, dt, state.telemetry.copy()))
    finally:
        for sub in subcycles:
            sub.stop()
    for name in names:
        columns[name] = getattr(state, name)
    for sub in subcycles:
        # the owner's own latest value, not the lagged copy the lockstep side read
        for name in sub.owned:
            columns[name] = getattr(sub.state, name)
    state.subcycle_status = tuple(sub.status() for sub in subcycles)
    return state, controller, results


def configure_publication_limits(state, targets):
    """Build-time defaults for the declared participant/channel buffers."""
    for index in range(int(state.pub_exchange_time.shape[0])):
        start = index * len(DT_CHANNEL_NAMES)
        stop = start + len(DT_CHANNEL_NAMES)
        state.pub_limits[start:stop] = targets.error_limits
        state.pub_limits_present[start:stop] = targets.error_limits_present


def dt_system_contract(entry, columns, batch, participants=1):
    """The state's span fields plus the dt system's own records.

    ``Targets``, ``STController`` and ``Metrics`` are retained exactly as
    ``extraction_contracts/program_extraction.yaml`` declares them, the same
    way the managed vehicle contract retains them; ``PieceState`` is
    declared the way ``BalloonTireManagedState`` is: one mutable span per
    field.
    """
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.native_law_kernels import _CONTRACTS

    policy = ExtractionContract(_CONTRACTS / "program_extraction.yaml")
    base = policy.program_abi.receipt()
    records = {name: base["records"][name] for name in ("Targets", "STController", "Metrics")}
    bindings = [b for b in base["bindings"] if b["record"] in records]
    span = lambda length: {  # noqa: E731
        "storage": "span", "dtype": "float64", "rank": 1,
        "shape": [int(length)], "mutable": True,
    }
    records["PieceState"] = {
        "identity": "llvm_dt_system.PieceState",
        "fields": {
            **{name: span(batch) for name in (*columns, "dt")},
            "telemetry": span(len(TELEMETRY_FIELDS)),
            **{name: span(participants) for name in (*PUBLICATION_FIELDS, *COURANT_FIELDS)},
            **{name: span(participants * len(DT_CHANNEL_NAMES)) for name in
               ("pub_values", "pub_present", "pub_limits", "pub_limits_present")},
        },
    }
    records["StepSpans"] = {
        "identity": "src.common.dt_system.participants.StepSpans",
        "fields": {
            **{name: span(participants) for name in PUBLICATION_FIELDS},
            **{name: span(participants * len(DT_CHANNEL_NAMES)) for name in
               ("pub_values", "pub_present", "pub_limits", "pub_limits_present")},
        },
    }
    records["Metrics"]["fields"].update(records["StepSpans"]["fields"])
    bindings.append({"function": "*", "parameter": "spans", "record": "StepSpans"})
    bindings.append({"function": "*", "parameter": "state", "record": "PieceState"})
    values = [
        {"function": entry, "parameter": name, "storage": "scalar",
         "dtype": "float64", "rank": 0, "python_type": "builtins.float"}
        for name in ("round_dt", "dt_initial", "dx")
    ]
    return policy.with_program_abi(
        {"records": records, "bindings": bindings, "values": values}
    ).with_execution_file(_CONTRACTS / "vehicle_full_native_execution.yaml")


def lowered_system(piece_files, *, backend=C_BACKEND, directory=None, optimization="O2",
                   link="static", piece_mode="link", progress=None):
    """Lower ``dt_system_over`` to ``backend`` and return the compiled artifact.

    The pieces are bound by name (``step_i``) and called by name in the
    generated ``advance_pieces``, so the lowering notices each call as an
    LLVM piece and the backend calls the piece's entry and links its module.

    ``piece_mode`` chooses what the laws are to this program:

    ``"link"``
        Each law stays the artifact it was built as.  Its LLVM is linked in
        and the call site is a call, so the law is compiled once and reused,
        and this program only has to know its ABI.

    ``"inline"``
        Each law becomes part of this program.  The lowering has already put
        the law's authored AbstractTensor source in the module and lowered it
        -- that is where its SSA comes from -- and the ``llvm_piece`` receipt
        on the law's root is the only thing that then tells a backend to call
        the prebuilt LLVM instead of emitting that body.  Dropping the receipt
        replaces the call with the law itself, so the result is one program
        with no external symbol and nothing that is already machine code.
        That is what a target which cannot link LLVM needs -- a shader
        dispatch has no linker -- and it is also the only form in which a
        whole-program pass can see through a law rather than around it.
    """
    import inspect
    from pathlib import Path

    from src.common.tensors import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    if progress is None:
        progress = lambda message: print(
            f"[llvm_dt_system] {message}", file=sys.stderr, flush=True,
        )

    progress(f"loading {len(piece_files)} compiled law piece(s)")
    pieces = [LLVMPiece.load(path) for path in piece_files]
    bindings = bind_pieces(pieces)
    batch = pieces[0].batch
    entry = "dt_system_over"
    source = inspect.getsource(sys.modules[__name__]) + "\n\n" + generated_source(pieces)
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, entry,
        python_bindings={"AbstractTensor": AbstractTensor, **bindings},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        runtime_closure_only=True, name="llvm_dt_system",
        extraction_contract=dt_system_contract(entry, column_names_of(pieces), batch, len(pieces)),
        progress=progress,
    )
    if piece_mode not in {"link", "inline"}:
        raise ValueError(f"piece_mode={piece_mode!r}: 'link' or 'inline'")
    if piece_mode == "inline":
        # The law's body is already here: `linked_repository_ssa` merged the SSA
        # that its authored AbstractTensor source lowered to.  The receipt on
        # the law's root is what makes a backend call the prebuilt LLVM instead
        # of emitting that body, so removing it substitutes the law for the
        # call.  Nothing is re-lowered and no source is re-parsed -- the choice
        # is only whether the emitter looks through the law or at its symbol.
        inlined = [
            name for name, function in module.functions.items()
            if function.metadata.pop("llvm_piece", None) is not None
        ]
        if not inlined:
            raise RuntimeError(
                "piece_mode='inline' but no law carried an llvm_piece receipt; "
                "the pieces were not recognised as pieces"
            )
        print(f"[llvm_dt_system] inlined {len(inlined)} law(s): {inlined}",
              flush=True)
    build = Path(directory) if directory is not None else Path.cwd() / "build" / "llvm_dt_system"
    if link != "static" and backend != C_BACKEND:
        raise NotImplementedError(f"link={link!r}: only the C lane links pieces dynamically")
    if backend == C_BACKEND:
        from src.compiler.ssa_c_backend import emit_ssa_module_to_c

        progress("emitting linked repository SSA to C")
        artifact = emit_ssa_module_to_c(module, exports[0])
        if not artifact.complete:
            raise RuntimeError("C emission shortfalls: " + "; ".join(
                f"{s.operation}: {s.reason}" for s in artifact.shortfalls[:6]))
        # link="static": each piece's LLVM IR is compiled into this module.
        # link="dynamic": the module calls the pieces' own DLLs by symbol.
        progress(f"compiling native artifact in {build}")
        compiled = artifact.compile(build, optimization=optimization, link=link)
        progress("native artifact compiled")
    elif backend == LLVM_BACKEND:
        from src.compiler.ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm

        compiled = compile_artifact(emit_ssa_function_to_llvm(module, exports[0]),
                                    directory=build, optimization=optimization)
    elif backend == FORTRAN_BACKEND:
        raise NotImplementedError("the Fortran deployment route is not built for this program yet")
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return NativeSystem(compiled, module, exports[0], pieces,
                        columns=column_names_of(pieces), batch=batch)


class NativeSystem:
    """The compiled dt-system window, the repository SSA it was emitted from,
    and the accounting a caller needs to actually drive it.

    The module's root formals carry the ABI accounting that maps authored feeds
    onto the artifact's buffers, and for a while this class held only the module
    and left every caller to dig that out again.  Three different hosts then
    rediscovered the same four facts -- which formal is which state field, which
    buffer index that formal is, what the round's own scalars are, and how long
    each buffer is -- and each rediscovery was a chance to get it subtly
    different.  They are published here instead, because this window is meant to
    be driven by many things.
    """

    def __init__(self, artifact, module, entry, pieces, *,
                 columns=(), batch=1):
        self.artifact = artifact
        self.module = module
        self.entry = entry
        self.pieces = pieces
        #: the state's span fields, in the order ``PieceState`` takes them
        self.columns = tuple(columns)
        #: cells per column; the batch the pieces were built at
        self.batch = int(batch)

    # ``_managed_native_feeds_by_id`` reads these two names.
    @property
    def root_name(self):
        return self.entry

    @property
    def root(self):
        """The root function, whose formals hold the ABI accounting."""
        return self.module.functions[self.entry]

    # ------------------------------------------------------------------ ABI
    def state_field_ids(self):
        """``state`` field name -> root formal value id.

        This is the mapping that says which physical buffer carries ``T``, or
        ``telemetry``, or any other declared field.
        """
        found = {}
        for argument in self.root.args:
            accounting = dict(argument.accounting or {})
            if accounting.get("program_abi_parameter") == "state":
                found[str(accounting.get("program_abi_field"))] = int(argument.id)
        return found

    def scalar_ids(self):
        """``round_dt``/``dt_initial``/``dx`` -> root formal value id.

        A host that wants to hand the next round the dt this one proposed has to
        write into the right buffer, and nothing else records which that is.
        """
        names = {
            int(value_id): str(name)
            for name, value_id in self.root.metadata.get("parameter_names", ())
        }
        wanted = {"round_dt", "dt_initial", "dx"}
        found = {}
        for argument in self.root.args:
            accounting = dict(argument.accounting or {})
            if accounting.get("program_abi_field") is not None:
                continue
            name = accounting.get("program_abi_parameter") or names.get(int(argument.id))
            if name in wanted:
                found[str(name)] = int(argument.id)
        return found

    def buffer_index_of(self, value_id):
        """Where a value id sits in the artifact's ``void **buffers`` table."""
        for index, held in enumerate(self.artifact.buffer_order):
            if int(held) == int(value_id):
                return index
        return None

    def feeds(self, state, targets, controller, round_dt, dt_initial, dx):
        """The physical feed mapping, flattened onto root formals."""
        from src.compiler.vehicle_python_compilation import (
            _managed_native_feeds_by_id,
        )

        return _managed_native_feeds_by_id(self, {
            "state": state, "targets": targets, "controller": controller,
            "round_dt": round_dt, "dt_initial": dt_initial, "dx": dx,
        })

    def prepare(self, state, targets, controller, round_dt, dt_initial, dx):
        """Allocate the public buffers from real values, ready to step."""
        configure_publication_limits(state, targets)
        return self.artifact.prepare_execution(
            self.feeds(state, targets, controller, round_dt, dt_initial, dx)
        )

    def layout(self, execution=None):
        """Everything a host needs, as plain data.

        Buffer lengths are only truthful once something has been fed -- a
        region formal's declared shape is ``()`` whether it is a scalar or the
        base of a million-element array -- so pass the ``execution`` from
        :meth:`prepare` to have the real counts included.
        """
        fields = self.state_field_ids()
        scalars = self.scalar_ids()
        table = []
        for index, value_id in enumerate(self.artifact.buffer_order):
            entry = {"index": index, "value_id": int(value_id),
                     "dtype": str(self.artifact.buffer_dtypes[index])}
            if execution is not None:
                held = execution.buffers[int(value_id)]
                entry["count"] = int(held.size)
                entry["itemsize"] = int(held.dtype.itemsize)
            table.append(entry)
        return {
            "entry": self.artifact.name,
            "module_source": f"{self.artifact.name}.c",
            "library": (str(self.artifact.library_path)
                        if getattr(self.artifact, "library_path", None) else None),
            "batch": self.batch,
            "columns": list(self.columns),
            "channel_names": list(DT_CHANNEL_NAMES),
            "laws": [str(piece.entry) for piece in self.pieces],
            "linked_llvm": [symbol for symbol, _ir
                            in getattr(self.artifact, "linked_llvm", ())],
            "buffers": table,
            "state_fields": {name: self.buffer_index_of(value_id)
                             for name, value_id in fields.items()},
            "state_field_value_ids": fields,
            "scalars": {name: self.buffer_index_of(value_id)
                        for name, value_id in scalars.items()},
            "total_bytes": (sum(item["count"] * item["itemsize"]
                                for item in table)
                            if execution is not None else None),
        }

    def write_layout(self, directory, execution=None):
        """Write ``layout.json`` beside the artifact."""
        import json
        from pathlib import Path

        path = Path(directory) / "layout.json"
        path.write_text(json.dumps(self.layout(execution), indent=2),
                        encoding="utf-8")
        return path

    def write_state(self, directory, execution):
        """Write ``initial-state.bin``: every buffer, in ``buffer_order``.

        The same order and packing the tree's own generated standalone host
        reads back, so a host can allocate one buffer per entry and read
        sequentially.
        """
        import numpy as np
        from pathlib import Path

        path = Path(directory) / "initial-state.bin"
        with path.open("wb") as stream:
            for value_id in self.artifact.buffer_order:
                stream.write(np.ascontiguousarray(
                    execution.buffers[int(value_id)]).tobytes(order="C"))
        return path
