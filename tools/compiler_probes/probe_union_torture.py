"""Union torture: one program, eager ctypes versus each native backend.

The program declares two ``ctypes.Structure`` types and one ``ctypes.Union``
of them, then writes through one member and reads through the other (type
punning), so any disagreement about size, alignment, offset or width shows up
in the bytes.  Eagerly, CPython and ctypes run it.  Natively, the same source
is lowered by ``lower_ast_source_to_ssa`` with the classes bound through
``python_bindings``; the compiler intercepts them into struct/union rows and
each backend spells those rows its own way.  The judge is byte equality of
the union payload plus the returned value.

    python -u tools/compiler_probes/probe_union_torture.py [c|llvm|fortran ...]
"""
from __future__ import annotations

import ctypes
import inspect
import pathlib
import sys
import warnings

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402

CONTRACTS = REPO / "extraction_contracts"
BUILD = REPO / "build" / "union_torture"


# --------------------------------------------------------------------- program

class Pair(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("n", ctypes.c_int64)]


class Words(ctypes.Structure):
    _fields_ = [("lo", ctypes.c_int32), ("hi", ctypes.c_int32),
                ("tail", ctypes.c_float)]


class Cell(ctypes.Union):
    _fields_ = [("pair", Pair), ("words", Words)]


def torture(cell: Cell, k: int) -> int:
    cell.pair.x = 1.5 * k
    cell.pair.n = k
    lo = cell.words.lo
    hi = cell.words.hi
    cell.words.tail = 2.0
    return lo + hi + cell.pair.n


BINDINGS = {"Cell": Cell, "Pair": Pair, "Words": Words}
K = 7


# --------------------------------------------------------------------- eager

def eager() -> tuple[int, bytes]:
    cell = Cell()
    result = torture(cell, K)
    return int(result), bytes(cell)


# --------------------------------------------------------------------- lowering

def lower():
    contract = (
        ExtractionContract(CONTRACTS / "program_extraction.yaml")
        .with_program_abi({"records": {}, "bindings": [], "values": []})
        .with_execution_file(CONTRACTS / "vehicle_full_native_execution.yaml")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, outputs, exports = lower_ast_source_to_ssa(
            inspect.getsource(torture), "torture", name="union_torture",
            python_bindings=dict(BINDINGS), extraction_contract=contract,
        )
    return module, outputs, exports


def layout_formal_ids(function) -> dict[int, dict]:
    """Root formals that are struct/union bases, by value id."""

    return {
        int(value.id): dict(value.accounting or {})
        for value in function.args
        if (value.accounting or {}).get("ssa_layout_kind") is not None
    }


# --------------------------------------------------------------------- pointer table

def run_pointer_table(entry, buffer_order, buffer_dtypes, layout_ids, result_ids,
                      cell, extents_ctype, extent_values=()):
    """Call an artifact's ``entry(void **buffers, extents)`` with our own
    table: the union base is the ctypes object's own address, every other
    slot a one-cell numpy array."""

    keep = []
    table = (ctypes.c_void_p * max(1, len(buffer_order)))()
    results = {}
    for index, value_id in enumerate(buffer_order):
        value_id = int(value_id)
        if value_id in layout_ids:
            table[index] = ctypes.addressof(cell)
            continue
        dtype = str(buffer_dtypes[index])
        numpy_dtype = {
            "float64": np.float64, "double": np.float64,
            "int64": np.int64, "i64": np.int64,
            "int32": np.int32, "i32": np.int32,
            "bool": np.bool_, "i1": np.bool_, "ptr": np.uintp,
        }[dtype]
        array = np.zeros((1,), dtype=numpy_dtype)
        if value_id in result_ids:
            results[value_id] = array
        else:
            array[0] = K  # the only scalar input is k
        keep.append(array)
        table[index] = array.ctypes.data
    extents = (extents_ctype * max(1, len(extent_values)))(*extent_values)
    entry(table, extents)
    return results


def judge(name, native_result, native_bytes, eager_result, eager_bytes) -> bool:
    ok = (native_result == eager_result) and (native_bytes == eager_bytes)
    print(f"[{name}] result native={native_result} eager={eager_result}")
    print(f"[{name}] bytes  native={native_bytes.hex()}")
    print(f"[{name}] bytes  eager ={eager_bytes.hex()}")
    print(f"[{name}] {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------- lanes

def lane_c(module, exports, eager_result, eager_bytes) -> bool:
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c

    root = module.functions[exports[0]]
    artifact = emit_ssa_module_to_c(module, exports[0])
    if not artifact.complete:
        raise RuntimeError("C emission shortfalls: " + "; ".join(
            f"{s.operation}: {s.reason}" for s in artifact.shortfalls[:6]))
    (BUILD / "c").mkdir(parents=True, exist_ok=True)
    (BUILD / "c" / "module.c").write_text(artifact.source, encoding="utf-8")
    artifact.compile(BUILD / "c", optimization="O2")
    cell = Cell()
    result_ids = {int(v.id) for v in module.functions[exports[0]].blocks["entry"].instrs[-1].args} \
        if False else set(int(v) for v in artifact.buffer_order) - set(int(a.id) for a in root.args)
    results = run_pointer_table(
        artifact.entry(), artifact.buffer_order, artifact.buffer_dtypes,
        layout_formal_ids(root), result_ids, cell, ctypes.c_longlong,
    )
    native_result = int(next(iter(results.values()))[0]) if results else None
    return judge("c", native_result, bytes(cell), eager_result, eager_bytes)


def lane_llvm(module, exports, eager_result, eager_bytes) -> bool:
    from src.compiler.ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm

    root = module.functions[exports[0]]
    artifact = emit_ssa_function_to_llvm(module, exports[0])
    if not artifact.complete:
        raise RuntimeError("LLVM emission shortfalls: " + "; ".join(
            f"{s.function}: {s.operation}: {s.reason}" for s in artifact.shortfalls[:6]))
    (BUILD / "llvm").mkdir(parents=True, exist_ok=True)
    (BUILD / "llvm" / "module.ll").write_text(artifact.llvm_ir, encoding="utf-8")
    compile_artifact(artifact, directory=BUILD / "llvm", optimization="O2")
    cell = Cell()
    result_ids = set(int(v) for v in artifact.buffer_order) - set(int(a.id) for a in root.args)
    results = run_pointer_table(
        artifact.entry(), artifact.buffer_order, artifact.buffer_dtypes,
        layout_formal_ids(root), result_ids, cell, ctypes.c_int32,
    )
    native_result = int(next(iter(results.values()))[0]) if results else None
    return judge("llvm", native_result, bytes(cell), eager_result, eager_bytes)


def lane_fortran(module, outputs, exports, eager_result, eager_bytes) -> bool:
    from src.compiler.ssa_fortran_backend import compile_module, emit_module

    fortran = emit_module(module, name="union_torture", outputs=outputs)
    (BUILD / "fortran").mkdir(parents=True, exist_ok=True)
    (BUILD / "fortran" / "module.f90").write_text(fortran.source, encoding="utf-8")
    library = compile_module(fortran, directory=BUILD / "fortran")
    print("[fortran] library", library)
    print("[fortran] api parameters:")
    entry = fortran.api.entry_point(exports[0]) if hasattr(fortran.api, "entry_point") else None
    for parameter in getattr(entry, "parameters", ()):
        print("   ", parameter)
    raise NotImplementedError("fortran host call: filled in once the lane emits the derived types")


# --------------------------------------------------------------------- main

def main(argv) -> int:
    lanes = tuple(argv[1:]) or ("c", "llvm", "fortran")
    eager_result, eager_bytes = eager()
    print(f"[eager] result={eager_result} bytes={eager_bytes.hex()} "
          f"sizeof={ctypes.sizeof(Cell)} align={ctypes.alignment(Cell)}")
    module, outputs, exports = lower()
    root = module.functions[exports[0]]
    print("[lowered] root formals:",
          [(int(a.id), a.dtype, (a.accounting or {}).get("ssa_layout_identity")) for a in root.args])
    print("[lowered] struct rows:", [r.to_mapping() for r in module.struct_table.structs.values()])
    print("[lowered] union rows:", [r.to_mapping() for r in module.union_table.unions.values()])
    verdicts = {}
    for lane in lanes:
        try:
            if lane == "c":
                verdicts[lane] = lane_c(module, exports, eager_result, eager_bytes)
            elif lane == "llvm":
                verdicts[lane] = lane_llvm(module, exports, eager_result, eager_bytes)
            elif lane == "fortran":
                verdicts[lane] = lane_fortran(module, outputs, exports, eager_result, eager_bytes)
        except Exception as error:  # report every lane, then fail
            import traceback
            traceback.print_exc()
            print(f"[{lane}] ERROR {type(error).__name__}: {str(error)[:400]}")
            verdicts[lane] = False
    print("VERDICTS", verdicts)
    return 0 if all(verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
