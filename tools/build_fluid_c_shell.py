"""Build the symbolic fluid frame program as a basic C-shell executable.

The canonical chain, nothing bespoke:

    lower_ast_source_to_ssa (already run by symbolic_fluid_direct_control,
    extraction contract applied)  ->  repository SSA pickle
    emit_module                   ->  Fortran module + api contract
    compile_fortran_module_c_shell -> standalone C-shell executable

Inputs are resolved through the api contract's own dotted source names
(``state.height``, ``targets.cfl``, ...) against the same feed objects the
capture declares.  The dt controller runs compiled inside the executable;
the C shell only loops frames and swaps the declared state feedback.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pickle
import sys as _sys

from src.compiler.ssa_fortran_backend import emit_module
from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
from src.compiler.symbolic_fluid_dt import SymbolicFluidGridState
from src.common.dt_system.dt_controller import STController, Targets
from src.compiler.string_table import string_token


def _resolve(source: str, roots: dict) -> object:
    head, _, rest = source.partition(".")
    value = roots[head]
    for part in rest.split(".") if rest else ():
        value = getattr(value, part)
    return value


#: Source trees that feed the SSA lowering this pickle caches. Broader than
#: ``symbolic_fluid_native_runtime.py``'s own ``_cache_is_stale`` (which only
#: scans ``src/compiler``): a fix can land in ``src/common`` (the dt
#: controller) or ``src/transmogrifier`` (AST ingestion) just as easily, and
#: missing one of those was exactly the trap that made a real source fix look
#: like it "didn't work" -- the pickle was silently reused unchanged.
_LOWERING_SOURCE_ROOTS = ("src/compiler", "src/common", "src/transmogrifier")

#: Non-Python inputs that also feed the lowering: the program ABI contract
#: (declared record fields crossing the Python/native boundary) changes the
#: compiled output just as much as a ``.py`` edit does. A field missing here
#: is silently dropped during lowering with no error -- the same "fix looks
#: like it didn't work" trap ``_LOWERING_SOURCE_ROOTS`` exists to avoid.
_LOWERING_SOURCE_FILES = ("extraction_contracts/program_extraction.yaml",)


def _pickle_is_stale(pkl_path: Path) -> tuple[bool, str | None]:
    try:
        stamp = pkl_path.stat().st_mtime
    except OSError:
        return True, None
    newest = 0.0
    newest_path: Path | None = None
    for relative_root in _LOWERING_SOURCE_ROOTS:
        root = ROOT / relative_root
        if not root.is_dir():
            continue
        for source in root.rglob("*.py"):
            if "__pycache__" in source.parts:
                continue
            try:
                when = source.stat().st_mtime
            except OSError:
                continue
            if when > newest:
                newest, newest_path = when, source
    for relative_file in _LOWERING_SOURCE_FILES:
        source = ROOT / relative_file
        try:
            when = source.stat().st_mtime
        except OSError:
            continue
        if when > newest:
            newest, newest_path = when, source
    if newest <= stamp:
        return False, None
    return True, (str(newest_path) if newest_path is not None else None)


def _regenerate_pickle(pkl_path: Path, *, max_memory_gb: float = 6.0) -> None:
    from src.compiler.symbolic_fluid_direct_control import bounded_compile

    report = bounded_compile(pkl_path.parent, max_memory_gb=max_memory_gb)
    print(
        "[stale-cache] regenerated "
        f"{pkl_path}: {report.get('function_count')} function(s), "
        f"completed={report.get('completed')}",
        file=_sys.stderr,
    )
    if not report.get("completed"):
        raise SystemExit(
            "auto-regeneration of the SSA pickle failed: "
            f"{report!r}"
        )


def build(pkl_path: Path, directory: Path, *, grid: int = 32,
          frame_duration: float = 1.0 / 30.0, dt_initial: float = 1.0e-3,
          trace: bool = False, auto_regenerate: bool = True):
    if auto_regenerate:
        stale, newer_than = _pickle_is_stale(pkl_path)
        if stale:
            reason = (
                f"predates {newer_than}" if newer_than is not None
                else "is missing"
            )
            print(
                f"[stale-cache] {pkl_path} {reason}; re-lowering before "
                "compiling Fortran. A cached lowering is not trusted once "
                "the compiler that produced it has changed.",
                file=_sys.stderr,
            )
            _regenerate_pickle(pkl_path)

    with pkl_path.open("rb") as stream:
        module, outputs, exports = pickle.load(stream)

    fortran = emit_module(module, name="symbolic_fluid", outputs=outputs)
    if not fortran.complete:
        for shortfall in fortran.shortfalls:
            print("shortfall:", str(shortfall)[:200])
        raise SystemExit("Fortran emission incomplete")

    state = SymbolicFluidGridState.initial(grid, grid)
    targets = Targets(
        cfl=0.45,
        div_max=1.0,
        mass_max=1.0e-8,
        error_limits={
            "height_positivity": 0.0,
            "tracer_bounds": 0.0,
        },
    )
    controller = STController(dt_min=1.0e-8, dt_max=frame_duration)
    from types import SimpleNamespace

    metrics_seed = SimpleNamespace(
        error_channels={
            "height_positivity": 0.0,
            "tracer_bounds": 0.0,
        },
        advanced_dt=0.0,
        max_vel=0.0,
        max_flux=0.0,
        div_inf=0.0,
        mass_err=0.0,
        dt_limit=0.0,
        osc_flag=False,
    )
    roots = {
        "state": state,
        "targets": targets,
        "controller": controller,
        # Contract-named record roles inside pursued helpers: ``self`` is
        # the controller instance, ``value`` the metrics record seed.
        "self": controller,
        "value": metrics_seed,
        "frame_duration": frame_duration,
        "dt_initial": dt_initial,
    }

    entry = fortran.api.entry_point(fortran.api.entry)
    inputs: dict[str, object] = {}
    extent_overrides: dict[str, int] = {}
    unresolved: list[str] = []
    for parameter in entry.parameters:
        if parameter.role not in {"input", "inout"}:
            continue
        source = str(parameter.source_name or parameter.name)
        base, _, leaf = source.rpartition(".")
        if leaf in {"length", "keys", "values"} and base:
            mapping = _resolve(base, roots)
            if isinstance(mapping, dict):
                if leaf == "length":
                    value = np.int64(len(mapping))
                elif leaf == "keys":
                    value = np.array(
                        [string_token(key) for key in mapping],
                        dtype=np.int64,
                    )
                else:
                    value = np.array(list(mapping.values()), dtype=np.float64)
                inputs[source] = value
                _bind_extents(parameter, value, extent_overrides)
                continue
        try:
            value = _resolve(source, roots)
        except (KeyError, AttributeError):
            unresolved.append(source)
            continue
        if isinstance(value, dict):
            # The keyed container BASE slot is a declared placeholder cell;
            # the real data crosses through the length/keys/values slots.
            value = np.float64(0.0)
        value = np.asarray(value)
        inputs[source] = value
        _bind_extents(parameter, value, extent_overrides)

    if unresolved:
        raise SystemExit(f"unresolved input sources: {unresolved}")

    # Workspace extents.  Every 2-D dynamic temporary in this program is
    # grid-shaped (state fields and their stencil shifts), and there the
    # declared leading extent is the indexing stride, so it must equal the
    # actual grid dimension.  Lone 1-D extents size capacity-only buffers
    # (sequence histories, rows); the grid cell count bounds all of them.
    # A scalar riding a declared 1-slot arena: its arena extent is 1.
    for parameter in entry.parameters:
        arena = getattr(parameter, "extent", None)
        if (
            arena is not None
            and not tuple(parameter.shape or ())
            and not tuple(getattr(parameter, "extents", ()) or ())
        ):
            extent_overrides.setdefault(str(arena), 1)

    extent_names = [
        str(parameter.name)
        for parameter in entry.parameters
        if parameter.role == "extent"
    ]
    families: dict[str, set[str]] = {}
    for name in extent_names:
        base, _, dim = name.rpartition("_")
        if dim.isdigit():
            families.setdefault(base, set()).add(dim)
    for name in extent_names:
        if name in extent_overrides or not name.startswith("extent_dynamic"):
            continue
        base, _, dim = name.rpartition("_")
        if len(families.get(base, ())) > 1:
            extent_overrides[name] = int(grid)
        else:
            extent_overrides[name] = int(grid) * int(grid)

    executable = compile_fortran_module_c_shell(
        fortran,
        inputs,
        directory,
        trace=trace,
        state_feedback={"dt_initial": "t14"},
        extent_overrides=extent_overrides,
        name="symbolic_fluid_frame_shell",
    )
    return executable


def _bind_extents(parameter, value, extent_overrides):
    names = tuple(getattr(parameter, "extents", ()) or ())
    array = np.asarray(value)
    if not names:
        return
    if array.ndim == 0 and len(names) == 1:
        # A scalar bound through a declared 1-slot arena.
        extent_overrides[str(names[0])] = 1
        return
    if array.ndim != len(names):
        # A flat arena bound to a multi-extent declaration: bind the
        # leading extent to the element count only when 1-D.
        if array.ndim == 1 and len(names) == 1:
            extent_overrides[str(names[0])] = int(array.shape[0])
        return
    for name, size in zip(names, array.shape):
        extent_overrides[str(name)] = int(size)


if __name__ == "__main__":
    pkl = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "build" / "sfdc-cshell" / "control_repository_ssa.pkl"
    )
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        ROOT / "build" / "fluid-c-shell"
    )
    # `--trace` compiles the launch digest into the executable. It is a
    # build-time choice, not a runtime switch: without it the ring and its
    # logger are not in the binary at all.
    trace = "--trace" in sys.argv[3:]
    executable = build(pkl, out, trace=trace)
    print("built:", executable.executable_path, "(trace)" if trace else "")
