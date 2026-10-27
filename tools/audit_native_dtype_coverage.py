"""Observe native dtype mappings and emit tiny real SSA operations.

This is a diagnostic, not a passing coverage test. Exit 1 means observed
representation mismatches. --native-dir additionally compiles/runs the uint64
precision-loss repro. No whole-program lowering or corpus run is performed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.transmogrifier import dtype_layout as layouts
from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock, IRModule


def addition(dtype: str) -> IRModule:
    left, right, result = (SSAValue(index, dtype) for index in range(3))
    return IRModule({"audit": Function("audit", [left, right], {
        "entry": BasicBlock("entry", [
            Instr("Add", [left, right], result), Instr("Ret", [result], None),
        ]),
    })})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path)
    args = parser.parse_args()
    from src.compiler.ssa_c_backend import _scalar_c_type, emit_ssa_to_c
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
    from src.compiler.ssa_fortran_backend import _declaration
    from llvmlite import binding

    mismatches = []
    print("dtype | bytes declared/lookup | LLVM declared/lookup | C declared/scalar | C buffer | public ABI | Fortran", flush=True)
    for name, layout in layouts.DTYPES.items():
        if layout.kind == "ptr":
            continue
        actual_bytes = layouts.dtype_byte_size(name)
        llvm = layouts.llvm_type_for_dtype(name)
        c_scalar = _scalar_c_type(name)
        try:
            fortran = _declaration(SSAValue(0, name), elemental=True)
        except Exception as error:
            fortran = type(error).__name__
        print(f"{name} | {layout.byte_size}/{actual_bytes} | "
              f"{layout.llvm_type}/{llvm} | {layout.c_storage_type}/{c_scalar} | "
              f"{layouts.c_lane_numpy_dtype(name)} | {layouts.c_abi_type(name)} | {fortran}", flush=True)
        if actual_bytes != layout.byte_size:
            mismatches.append(f"{name}: byte count {actual_bytes} != {layout.byte_size}")
        if llvm != layout.llvm_type:
            mismatches.append(f"{name}: LLVM lookup {llvm} != {layout.llvm_type}")
        # float32 scalar widening is an explicitly documented lane policy.
        if name != "float32" and c_scalar != layout.c_storage_type:
            mismatches.append(f"{name}: C scalar {c_scalar} != {layout.c_storage_type}")
        if layouts.c_lane_numpy_dtype(name) != layout.numpy_dtype:
            mismatches.append(f"{name}: C buffer {layouts.c_lane_numpy_dtype(name)} != {layout.numpy_dtype}")

    print("SSA Add emission (valid LLVM does not establish dtype correctness):", flush=True)
    for name in ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float32", "float64"):
        artifact = emit_ssa_function_to_llvm(addition(name), "audit")
        binding.parse_assembly(artifact.llvm_ir).verify()
        operation = next(line.strip() for line in artifact.llvm_ir.splitlines()
                         if " = fadd " in line or " = add " in line)
        c_artifact = emit_ssa_to_c(addition(name), "audit")
        print(f"{name}: LLVM complete={artifact.complete} verified=True {operation}; C complete={c_artifact.complete}", flush=True)

    print("Unsigned operation probes:", flush=True)
    for dtype in ("uint8", "uint16", "uint32", "uint64"):
        for operation in ("Mod", "FloorDiv", "Lt", "ULt", "Shr", "BitAnd", "Cast"):
            left, right = SSAValue(0, dtype), SSAValue(1, dtype)
            result = SSAValue(2, "bool" if operation in {"Lt", "ULt"} else dtype)
            operands = [SSAValue(0, "int64")] if operation == "Cast" else [left, right]
            formals = operands if operation == "Cast" else [left, right]
            module = IRModule({"audit": Function("audit", formals, {
                "entry": BasicBlock("entry", [
                    Instr(operation, operands, result,
                          attributes={"target_dtype": dtype} if operation == "Cast" else {}),
                    Instr("Ret", [result], None),
                ]),
            })})
            artifact = emit_ssa_function_to_llvm(module, "audit")
            verified = True
            try:
                binding.parse_assembly(artifact.llvm_ir).verify()
            except RuntimeError:
                verified = False
                if artifact.complete:
                    mismatches.append(f"{dtype} {operation}: complete artifact has invalid LLVM")
            selected = [line.strip() for line in artifact.llvm_ir.splitlines()
                        if any(word in line for word in
                               ("frem ", "fdiv ", "icmp ", "fcmp ", "sitofp ", "lshr ", "and i"))]
            print(f"{dtype} {operation}: complete={artifact.complete} verified={verified} "
                  f"{selected[:2]}", flush=True)

    if args.native_dir:
        import numpy as np
        artifact = emit_ssa_to_c(addition("uint64"), "audit")
        artifact.compile(args.native_dir)
        expected = 2**53 + 1
        execution = artifact.prepare_execution({
            0: np.array([expected], dtype=np.uint64),
            1: np.array([0], dtype=np.uint64),
        }).run()
        result = execution.buffers[2]
        actual = result.item()
        print(f"NATIVE uint64: expected={expected} actual={actual!r} buffer_dtype={result.dtype}", flush=True)
        if int(actual) != expected or result.dtype != np.dtype("uint64"):
            mismatches.append("uint64 native addition loses value or storage dtype")

    print(f"OBSERVED REPRESENTATION MISMATCHES: {len(mismatches)}", flush=True)
    for finding in mismatches:
        print("  " + finding, flush=True)
    return int(bool(mismatches))


if __name__ == "__main__":
    raise SystemExit(main())
