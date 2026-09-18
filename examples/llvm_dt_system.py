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

Python does the lowering.  The pieces decide the columns, so the state class
(one span field per column, exactly the tire's ``BalloonTireManagedState``
shape) and the advance function (one named call per piece) are spelled by
Python for the piece set at hand -- ``state_source``/``piece_source`` -- and
that same text is what runs in Python and what is handed to the compiler.
The emitted unit is the run-many-times part.
"""

import sys

import numpy as np

from src.common.dt_system.dt_controller import STController, Targets, run_superstep
from src.common.dt_system.dt_scaler import Metrics
from src.compiler.native_law_kernels import LLVMPiece

C_BACKEND = "c"
LLVM_BACKEND = "llvm"
FORTRAN_BACKEND = "fortran"

METRIC_FIELDS = ("max_vel", "max_flux", "div_inf", "mass_err", "dt_limit",
                 "energy_j", "power_w")
TELEMETRY_FIELDS = ("advanced", "dt_next", "max_vel", "max_flux", "div_inf",
                    "mass_err", "dt_limit", "hard_failure")


def column_names_of(pieces):
    """Every column some piece reads, in first-appearance order (``dt`` is
    the step's own column and is not one of them)."""

    names = []
    for piece in pieces:
        for name in piece.argument_names:
            if name != "dt" and name not in names:
                names.append(name)
    return tuple(names)


def state_source(columns):
    """Spell ``PieceState`` for these columns: one span field per column,
    the ``dt`` column the step fills, and the window's telemetry span.
    ``copy_shallow``/``restore`` are the tire's: copies out, in-place back."""

    fields = (*columns, "dt")
    lines = ["class PieceState:"]
    lines.append(f"    def __init__(self, {', '.join(fields)}, telemetry):")
    for name in fields:
        lines.append(f"        self.{name} = {name}")
    lines.append("        self.telemetry = telemetry")
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


def piece_source(pieces):
    """Spell ``advance_pieces(state, dt)`` for exactly these pieces.

    Every piece is called by its own bound name (``step_0``, ``step_1``,
    ...) with its columns spelled out, every ``<name>_next`` output is
    assigned to its column, and the published metrics are folded
    explicitly -- maxima of the error measures, the minimum published
    ``dt_limit`` (+inf when none is published; the controller's minimum
    against it is a no-op), the energy/power channels summed.
    """

    columns = column_names_of(pieces)
    lines = ["def advance_pieces(state, dt):"]
    lines.append("    state.dt[...] = dt")
    folds = {name: [] for name in METRIC_FIELDS}
    for index, piece in enumerate(pieces):
        arguments = ", ".join(f"state.{name}" for name in piece.argument_names)
        outputs = [f"o{index}_{name}" for name in piece.output_names]
        lines.append(f"    {', '.join(outputs)}, = step_{index}({arguments})")
        for name, output in zip(piece.output_names, outputs):
            if name.endswith("_next") and name[:-5] in columns:
                # In place: the column is the caller's buffer (the same
                # write ``restore`` makes), never a rebinding of the field.
                lines.append(f"    state.{name[:-5]}[...] = {output}")
            if name in folds:
                folds[name].append(output)

    def fold(operator, terms, empty):
        if not terms:
            return empty
        if len(terms) == 1:
            return terms[0]
        return f"{operator}({', '.join(terms)})"

    for name in ("max_vel", "max_flux", "div_inf", "mass_err"):
        terms = [f"float({o}.max())" for o in folds[name]]
        lines.append(f"    {name} = " + fold("max", terms, "0.0"))
    terms = [f"float({o}.min())" for o in folds["dt_limit"]]
    lines.append("    dt_limit = " + fold("min", terms, 'float("inf")'))
    for name in ("energy_j", "power_w"):
        terms = " + ".join(f"float({o}.max())" for o in folds[name])
        lines.append(f"    {name} = {terms or '0.0'}")
    lines.append("    metrics = Metrics(")
    lines.append("        max_vel=max_vel, max_flux=max_flux, div_inf=div_inf,")
    lines.append("        mass_err=mass_err, dt_limit=dt_limit,")
    lines.append('        error_channels={"energy_j": energy_j, "power_w": power_w},')
    lines.append("    )")
    lines.append("    return True, metrics")
    return "\n".join(lines) + "\n"


def generated_source(pieces):
    return state_source(column_names_of(pieces)) + "\n\n" + piece_source(pieces)


def bind_pieces(pieces):
    """Bind ``PieceState`` and ``advance_pieces`` for these pieces in this
    module: the Python path runs the very text the lowering is given."""

    bindings = {f"step_{index}": piece for index, piece in enumerate(pieces)}
    namespace = {"np": np, "Metrics": Metrics, **bindings}
    exec(generated_source(pieces), namespace)
    globals()["PieceState"] = namespace["PieceState"]
    globals()["advance_pieces"] = namespace["advance_pieces"]
    return bindings


def dt_system_over(state, targets, controller, round_dt, dt_initial, dx):
    """One round of the dt system over the bound pieces: the function that
    lowers, shaped exactly like the tire's ``balloon_tire_managed_window``.
    The caller owns ``state``/``targets``/``controller`` across rounds; the
    round's results are published on ``state.telemetry``."""

    advanced, dt_next, metrics = run_superstep(
        state, round_dt, dt_initial, dx, targets, controller, advance_pieces)
    state.telemetry[0] = advanced
    state.telemetry[1] = dt_next
    state.telemetry[2] = metrics.max_vel
    state.telemetry[3] = metrics.max_flux
    state.telemetry[4] = metrics.div_inf
    state.telemetry[5] = metrics.mass_err
    state.telemetry[6] = metrics.dt_limit if metrics.dt_limit is not None else 0.0
    state.telemetry[7] = float(metrics.hard_failure)
    return advanced, dt_next


def dt_system(piece_files, columns, *, rounds, round_dt, dx, targets=None, controller=None):
    """Load the pieces from their files and run ``rounds`` rounds of the dt
    system over them in Python, the way the native unit will be driven."""

    pieces = [LLVMPiece.load(path) for path in piece_files]
    bind_pieces(pieces)
    batch = pieces[0].batch
    names = column_names_of(pieces)
    targets = targets or Targets(cfl=0.5, div_max=1e9, mass_max=1e-3, energy_exchange_fraction=0.2)
    controller = controller or STController()
    state = PieceState(
        *(np.array(columns[name], dtype=np.float64) for name in names),
        np.zeros((batch,), dtype=np.float64),
        np.zeros((len(TELEMETRY_FIELDS),), dtype=np.float64),
    )
    dt = round_dt
    results = []
    for _round in range(rounds):
        total, dt = dt_system_over(state, targets, controller, round_dt, dt, dx)
        results.append((total, dt, state.telemetry.copy()))
    for name in names:
        columns[name] = getattr(state, name)
    return state, controller, results


def dt_system_contract(entry, columns, batch):
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
        },
    }
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
                   link="static"):
    """Lower ``dt_system_over`` to ``backend`` and return the compiled artifact.

    The pieces are bound by name (``step_i``) and called by name in the
    generated ``advance_pieces``, so the lowering notices each call as an
    LLVM piece and the backend calls the piece's entry and links its module.
    """
    import inspect
    from pathlib import Path

    from src.common.tensors import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

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
        extraction_contract=dt_system_contract(entry, column_names_of(pieces), batch),
    )
    build = Path(directory) if directory is not None else Path.cwd() / "build" / "llvm_dt_system"
    if link != "static" and backend != C_BACKEND:
        raise NotImplementedError(f"link={link!r}: only the C lane links pieces dynamically")
    if backend == C_BACKEND:
        from src.compiler.ssa_c_backend import emit_ssa_module_to_c

        artifact = emit_ssa_module_to_c(module, exports[0])
        if not artifact.complete:
            raise RuntimeError("C emission shortfalls: " + "; ".join(
                f"{s.operation}: {s.reason}" for s in artifact.shortfalls[:6]))
        # link="static": each piece's LLVM IR is compiled into this module.
        # link="dynamic": the module calls the pieces' own DLLs by symbol.
        compiled = artifact.compile(build, optimization=optimization, link=link)
    elif backend == LLVM_BACKEND:
        from src.compiler.ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm

        compiled = compile_artifact(emit_ssa_function_to_llvm(module, exports[0]),
                                    directory=build, optimization=optimization)
    elif backend == FORTRAN_BACKEND:
        raise NotImplementedError("the Fortran deployment route is not built for this program yet")
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return NativeSystem(compiled, module, exports[0], pieces)


class NativeSystem:
    """The compiled dt-system window plus the repository SSA it was emitted
    from (the module's root formals carry the ABI accounting that maps
    authored feeds onto the artifact's buffers)."""

    def __init__(self, artifact, module, entry, pieces):
        self.artifact = artifact
        self.module = module
        self.entry = entry
        self.pieces = pieces

    # ``_managed_native_feeds_by_id`` reads these two names.
    @property
    def root_name(self):
        return self.entry
