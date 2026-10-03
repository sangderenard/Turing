"""Orbital work item 3, seconds-long: an applied undefined Function as an external.

One law ``y = 2*f(t) + t`` with ``f`` an undefined sympy Function of ``t``.
Route: ``compile_sympy_equations`` -> ``piece_from_law`` (LLVM) -> C emission
and compile -> native, called with a Python-supplied ``f`` and compared with
``2*f(t) + t`` evaluated in numpy.  Two variants: ``f`` a Python callable
(wrapped once as a C-callable piece over the buffer ABI) and ``f`` an
``LLVMPiece`` (the law ``fy = cos(t) + 1/4``) whose own entry fills the slot.  Each stage prints ok or the failure
verbatim (stage, raising frame, message); exit 1 on any failure.

    python -u tools/compiler_probes/probe_external_function.py
"""
from __future__ import annotations

import pathlib
import sys
import traceback
import warnings

import numpy as np
import sympy as sp

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

BUILD = REPO / "build" / "probe_external_function"
BATCH = 4
LAW = "external_probe"


def host_f(t):
    """The host's f: what the craft's r()/F() seam supplies at runtime."""
    return np.cos(np.asarray(t, dtype=np.float64)) + 0.25


def _frame(error: BaseException) -> str:
    frames = traceback.extract_tb(error.__traceback__)
    inside = [f for f in frames if "src" in pathlib.Path(f.filename).parts]
    frame = (inside or frames)[-1]
    return f"{pathlib.Path(frame.filename).name}:{frame.lineno} {frame.name}"


def _fail(stage: str, error: BaseException) -> int:
    print(f"FAIL {stage}: raised at {_frame(error)}")
    print(f"     {type(error).__name__}: {str(error)[:1600]}")
    return 1


def run_variant(label, compilation, arguments, columns, expected, implementation,
                supplied) -> int:
    """Lower, emit and run one variant; ``supplied`` goes to piece_from_law,
    ``implementation`` fills the slots at load."""
    from src.compiler.external_functions import bind_external_slots
    from src.compiler.native_package import piece_from_law
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c

    directory = BUILD / label
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            piece = piece_from_law(compilation, LAW, BATCH, directory=directory,
                                   externals=supplied)
    except Exception as error:  # noqa: BLE001
        return _fail(f"[{label}] piece_from_law", error)
    print(f"ok   [{label}] piece_from_law: entry={piece.entry} "
          f"slots={[row['external'] for row in piece.artifact.external_slots]}")
    failures = 0
    try:
        bind_external_slots(piece.artifact, {"f": implementation})
        got = dict(zip(piece.output_names, piece(*(columns[name] for name in arguments))))
        error = float(np.max(np.abs(got["y"] - expected)))
        print(f"{'ok  ' if error <= 1e-12 else 'FAIL'} [{label}] llvm: max abs error {error:.3e}")
        failures += error > 1e-12
    except Exception as error:  # noqa: BLE001
        failures += _fail(f"[{label}] llvm run", error)
    try:
        artifact = emit_ssa_module_to_c(piece.module, piece.entry)
        if not artifact.complete:
            raise RuntimeError(f"C emission shortfalls: {artifact.shortfalls[:3]}")
        artifact = artifact.compile(directory / "c")
        bind_external_slots(artifact, {"f": implementation})
        execution = artifact.prepare_execution(
            {value_id: columns[name] for name, value_id in zip(arguments, piece.argument_ids)})
        execution.run()
        error = float(np.max(np.abs(execution.buffers[piece.output_ids["y"]] - expected)))
        print(f"{'ok  ' if error <= 1e-12 else 'FAIL'} [{label}] c: max abs error {error:.3e}")
        failures += error > 1e-12
    except Exception as error:  # noqa: BLE001
        failures += _fail(f"[{label}] c run", error)
    # An unfilled slot is a loud error: empty both lanes' slot and run again.
    import ctypes
    for lane, compiled, run in (
            ("llvm", piece.artifact,
             lambda: piece(*(columns[name] for name in arguments))),
            ("c", artifact, lambda: execution.run())):
        library = ctypes.CDLL(str(compiled.library_path))
        library[f"{compiled.name}__bind_external"](ctypes.c_int32(0), ctypes.c_void_p(None))
        try:
            run()
        except RuntimeError as error:
            print(f"ok   [{label}] {lane}: unfilled slot refused: {error}")
        else:
            print(f"FAIL [{label}] {lane}: unfilled slot ran without an error")
            failures += 1
    return failures


def host_piece():
    """The host's f as an LLVM piece: the law ``fy = cos(t) + 1/4``."""
    from src.compiler.native_package import piece_from_law
    from src.compiler.symbolic_equation_compiler import compile_sympy_equations

    t = sp.Symbol("t", real=True)
    law = [sp.Eq(sp.Symbol("fy"), sp.cos(t) + sp.Rational(1, 4), evaluate=False)]
    compilation = compile_sympy_equations(law, name="host_f")
    directory = BUILD / "host_f"
    directory.mkdir(parents=True, exist_ok=True)
    return piece_from_law(compilation, "host_f", BATCH, directory=directory)


def main() -> int:
    import faulthandler

    faulthandler.dump_traceback_later(900, repeat=True, file=sys.stderr)
    from src.compiler.symbolic_equation_compiler import compile_sympy_equations

    t = sp.Symbol("t", real=True)
    f = sp.Function("f")
    law = [sp.Eq(sp.Symbol("y"), 2 * f(t) + t, evaluate=False)]
    columns = {"t": np.linspace(0.1, 1.3, BATCH)}
    expected = 2 * host_f(columns["t"]) + columns["t"]

    try:
        compilation = compile_sympy_equations(law, name=LAW)
    except Exception as error:  # noqa: BLE001 -- recorded verbatim
        return _fail("compile_sympy_equations", error)
    metadata = compilation.function.metadata
    arguments = tuple(metadata["argument_names"])
    print(f"ok   compile_sympy_equations: arguments={arguments} "
          f"externals={metadata.get('external_functions')}")

    failures = run_variant("python_f", compilation, arguments, columns, expected,
                           host_f, None)
    try:
        piece = host_piece()
    except Exception as error:  # noqa: BLE001
        return failures + _fail("host piece", error)
    failures += run_variant("llvm_piece_f", compilation, arguments, columns, expected,
                            piece, {"f": piece})
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
