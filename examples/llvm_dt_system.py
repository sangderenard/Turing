"""The dt system with LLVM pieces as its steps -- the common path.

    state, ctrl, results = dt_system(piece_files, columns, rounds=..., round_dt=..., dx=...)

This is ``run_superstep`` used directly, the same way it is used everywhere
else in the tree that already lowers (vehicle_python_compilation.py,
native_voxel_fluid.py, symbolic_fluid_dt.py, symbolic_fluid_native_runtime.py,
and this repo's own chamber_dt_join.py/chamber_sim.py): one ``state`` object
(``PieceState``, with ``copy_shallow``/``restore`` for in-round rollback),
one ``STController`` kept alive across frames, and one ``advance(state, dt)``
function that calls the pieces and returns ``(ok, metrics)``.  Nothing else
of the dt system is needed -- no ``DtCompatibleEngine``, no ``StateTable``,
no ``GraphBuilder``/``MetaLoopRunner`` (that layer is optional composition on
top and has never been lowered; see llvm_dt_system.py.bak for the discovery).
"""

import sys

import numpy as np

from src.common.dt_system.dt_controller import STController, Targets, run_superstep
from src.common.dt_system.dt_scaler import Metrics
from src.compiler.native_law_kernels import LLVMPiece

C_BACKEND = "c"
LLVM_BACKEND = "llvm"
FORTRAN_BACKEND = "fortran"

CHANNEL_FIELDS = ("energy_j", "power_w")


class PieceState:
    """The columns every step reads and writes, by name."""

    def __init__(self, columns):
        self.columns = dict(columns)

    def copy_shallow(self):
        return PieceState(self.columns)

    def restore(self, snapshot):
        self.columns = dict(snapshot.columns)


def step_piece(state, piece, dt):
    """Run one piece on the state's columns; feed its ``<name>_next`` outputs
    back into ``<name>``.  Returns the piece's outputs, by name."""

    dt_column = np.full(piece.batch, float(dt))
    arguments = [dt_column if name == "dt" else state.columns[name]
                 for name in piece.argument_names]
    outputs = dict(zip(piece.output_names, piece(*arguments)))
    for name, value in outputs.items():
        if name.endswith("_next") and name[:-5] in state.columns:
            state.columns[name[:-5]] = value
    return outputs


def metrics_of(outputs):
    return Metrics(
        max_vel=float(outputs["max_vel"].max()) if "max_vel" in outputs else 0.0,
        max_flux=float(outputs["max_flux"].max()) if "max_flux" in outputs else 0.0,
        div_inf=float(outputs["div_inf"].max()) if "div_inf" in outputs else 0.0,
        mass_err=float(outputs["mass_err"].max()) if "mass_err" in outputs else 0.0,
        dt_limit=float(outputs["dt_limit"].min()) if "dt_limit" in outputs else None,
        error_channels={name: float(outputs[name].max())
                        for name in CHANNEL_FIELDS if name in outputs},
    )


def merge_metrics(rows):
    limits = [row.dt_limit for row in rows if row.dt_limit is not None]
    channels = {}
    for row in rows:
        for key, value in (row.error_channels or {}).items():
            channels[key] = channels.get(key, 0.0) + value
    return Metrics(
        max_vel=max(row.max_vel for row in rows),
        max_flux=max(row.max_flux for row in rows),
        div_inf=max(row.div_inf for row in rows),
        mass_err=max(row.mass_err for row in rows),
        dt_limit=min(limits) if limits else None,
        error_channels=channels,
    )


def advance(steps, state, dt):
    rows = [metrics_of(step_piece(state, piece, dt)) for piece in steps]
    return True, merge_metrics(rows)


def dt_system_over(steps, columns, rounds, round_dt, dx, targets=None, controller=None):
    """The dt system over already-loaded pieces.  This is the function that lowers."""

    targets = targets or Targets(cfl=0.5, div_max=1e9, mass_max=1e-3, energy_exchange_fraction=0.2)
    controller = controller or STController()
    state = PieceState(columns)
    dt = round_dt
    results = []
    for _round in range(rounds):
        total, dt, metrics = run_superstep(
            state, round_dt, dt, dx, targets, controller,
            lambda s, step_dt: advance(steps, s, step_dt))
        results.append((total, dt, metrics))
    for name in columns:
        columns[name] = state.columns[name]
    return state, controller, results


def dt_system(piece_files, columns, *, rounds, round_dt, dx, targets=None, controller=None):
    """Load the pieces from their files and run the dt system over them in Python."""

    steps = [LLVMPiece.load(path) for path in piece_files]
    return dt_system_over(steps, columns, rounds, round_dt, dx, targets, controller)


def dt_system_contract(entry, column_names, batch):
    """The columns as spans plus the dt system's own records.

    ``Targets``, ``STController`` and ``Metrics`` are retained exactly as
    ``extraction_contracts/program_extraction.yaml`` declares them, the same
    way the managed vehicle contract retains them; the columns are the
    program's span parameters.
    """
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.native_law_kernels import _CONTRACTS

    policy = ExtractionContract(_CONTRACTS / "program_extraction.yaml")
    base = policy.program_abi.receipt()
    records = {name: base["records"][name] for name in ("Targets", "STController", "Metrics")}
    bindings = [b for b in base["bindings"] if b["record"] in records]
    values = [{
        "function": entry, "parameter": name, "storage": "span",
        "dtype": "float64", "rank": 1, "shape": [int(batch)],
        "python_type": "src.common.tensors.abstraction.AbstractTensor",
    } for name in column_names]
    return policy.with_program_abi(
        {"records": records, "bindings": bindings, "values": values}
    ).with_execution_file(_CONTRACTS / "vehicle_full_native_execution.yaml")


def lowered_system(piece_files, columns, *, rounds, round_dt, dx,
                   targets=None, controller=None,
                   backend=C_BACKEND, parameterized=None,
                   directory=None, optimization="O2"):
    """Lower ``dt_system_over`` to ``backend`` and return the compiled artifact.

    ``parameterized`` names the columns that are the program's parameters
    (spans); by default every column is.  The pieces are bound by name, so
    the lowering notices each ``steps[i](...)`` as an LLVM call and the
    backend calls the piece's entry and links its module.
    """
    import inspect
    from pathlib import Path

    from src.common.tensors import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    steps = [LLVMPiece.load(path) for path in piece_files]
    batch = steps[0].batch
    names = tuple(parameterized) if parameterized is not None else tuple(columns)
    entry = "dt_system_over"
    source = inspect.getsource(sys.modules[__name__])
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, entry,
        python_bindings={"AbstractTensor": AbstractTensor, "steps": steps},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        runtime_closure_only=True, name="llvm_dt_system",
        extraction_contract=dt_system_contract(entry, names, batch),
    )
    build = Path(directory) if directory is not None else Path.cwd() / "build" / "llvm_dt_system"
    if backend == C_BACKEND:
        from src.compiler.ssa_c_backend import emit_ssa_module_to_c

        artifact = emit_ssa_module_to_c(module, exports[0])
        if not artifact.complete:
            raise RuntimeError("C emission shortfalls: " + "; ".join(
                f"{s.operation}: {s.reason}" for s in artifact.shortfalls[:6]))
        return artifact.compile(build, optimization=optimization)
    if backend == LLVM_BACKEND:
        from src.compiler.ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm

        return compile_artifact(emit_ssa_function_to_llvm(module, exports[0]),
                                directory=build, optimization=optimization)
    if backend == FORTRAN_BACKEND:
        raise NotImplementedError("the Fortran deployment route is not built for this program yet")
    raise ValueError(f"unknown backend {backend!r}")
