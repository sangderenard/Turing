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
from src.common.dt_system.participants import Publication
from src.common.dt_system.time_contracts import BIND, HOLD, ParticipantRegistry
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
            lines.append(f"    m{index}_{name} = o{index}_{name}.{reducer}().item()")
        # tau is this law's own energy over its own power: the time it would
        # take to exchange its stored energy at the rate it is exchanging now.
        # That is the quantity the blended energy/power pin always computed --
        # here it stays attached to the law that measured it.
        if "energy_j" in published and "power_w" in published:
            lines.append(
                f"    t{index}_tau = (m{index}_energy_j / m{index}_power_w"
                f" if m{index}_power_w > 0.0 else None)")
        else:
            lines.append(f"    t{index}_tau = None")

    lines.append("")
    lines.append("    state.publications = {")
    for index, piece in enumerate(pieces):
        published = set(piece.output_names)
        channels = [name for name in ("energy_j", "power_w", "div_inf", "mass_err")
                    if name in published]
        channel_text = ", ".join(
            f'"{name}": m{index}_{name}' for name in channels)
        floor = (f"m{index}_dt_limit" if "dt_limit" in published else "None")
        lines.append(f'        "{piece.entry}": Publication(')
        lines.append(f"            channels={{{channel_text}}},")
        lines.append(f"            dt_limit={floor},")
        lines.append(f"            tau_s=t{index}_tau,")
        lines.append(f"            contract=(BIND if t{index}_tau is not None"
                     f" else HOLD),")
        lines.append("        ),")
    lines.append("    }")

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
    namespace = {
        "np": np, "Metrics": Metrics,
        # the dt system's own publication vocabulary: a law states what it
        # measured and how its tau participates, and nothing here decides
        "Publication": Publication, "BIND": BIND, "HOLD": HOLD,
        **bindings,
    }
    exec(generated_source(pieces), namespace)
    globals()["PieceState"] = namespace["PieceState"]
    globals()["advance_pieces"] = namespace["advance_pieces"]
    return bindings


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
    state.participant_limits = dict(targets.error_limits or {})
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
                   link="static", piece_mode="link"):
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
