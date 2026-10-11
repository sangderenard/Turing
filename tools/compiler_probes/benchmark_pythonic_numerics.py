"""Phase and steady-call timing through OUR public compiler, no deadlines."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa, compile_fortran_module_c_shell
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative
from src.compiler.work_contract import PRESETS, set_active_contract


def median_call(call, repeats):
    for _ in range(30):
        call()
    values = []
    for _ in range(7):
        start = time.perf_counter()
        for _ in range(repeats):
            call()
        values.append((time.perf_counter() - start) / repeats)
    return statistics.median(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 256])
    parser.add_argument("--backends", nargs="+", choices=("c", "llvm", "fortran"), default=["c", "llvm", "fortran"])
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--flags", action="append", default=[])
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 1 for size in args.sizes):
        parser.error("sizes and repeats must be positive")
    set_active_contract(replace(PRESETS["develop"], compiler_flags=(*PRESETS["develop"].compiler_flags, *args.flags)))
    directory = Path(tempfile.mkdtemp(prefix="turing_pythonic_numeric_"))
    for size in args.sizes:
        source = f"""
def root(seed, guard):
    values = []
    total = seed
    for i in range({size}):
        total = total + i * 0.125
        if guard:
            values.append(total)
    for value in values:
        total = total + value * 0.5
    return total
"""
        eager_namespace = {}
        exec(compile(source, "authored_timing_reference", "exec"), eager_namespace)
        eager = eager_namespace["root"]
        policy = ExtractionContract("extraction_contracts/program_extraction.yaml").with_execution_file(
            "extraction_contracts/vehicle_full_native_execution.yaml").with_program_abi({"bindings": [], "values": [
            {"function": "root", "parameter": "seed", "storage": "scalar", "dtype": "float64", "rank": 0, "python_type": "builtins.float"},
            {"function": "root", "parameter": "guard", "storage": "scalar", "dtype": "bool", "rank": 0, "python_type": "builtins.bool"}]})
        # Lower the identical authored source separately for every backend:
        # emitters may publish derived type views. No lane inherits another's.
        for backend in args.backends:
            timings = {}
            start = time.perf_counter()
            module, outputs, exports = lower_ast_source_to_ssa(source, "root",
                name=f"mixed_{size}_{backend}", extraction_contract=policy,
                progress=lambda _: None, progress_bars=False)
            timings["lower_s"] = time.perf_counter() - start
            name = exports[0]
            parameters = dict(module.functions[name].metadata["parameter_names"])
            feeds = {parameters["seed"]: np.asarray([0.125]), parameters["guard"]: np.asarray([True])}
            start = time.perf_counter()
            if backend == "c":
                artifact = emit_ssa_module_to_c(module, name)
            elif backend == "llvm":
                artifact = emit_ssa_function_to_llvm(module, name)
            else:
                artifact = emit_module(module, name=f"mixed_{size}", outputs=outputs, progress=lambda _: None)
            if not artifact.complete:
                raise AssertionError(artifact.shortfalls)
            timings["emit_s"] = time.perf_counter() - start
            start = time.perf_counter()
            build_dir = directory / str(size) / backend
            ordinary_run = subprocess.run
            def observe_build(command, *positional, **keywords):
                print("BUILD_COMMAND", backend, command, flush=True)
                return ordinary_run(command, *positional, **keywords)
            subprocess.run = observe_build
            try:
                if backend == "c":
                    artifact.compile(build_dir, optimization="O3")
                elif backend == "llvm":
                    compile_artifact(artifact, directory=build_dir, optimization="O3")
                else:
                    built = compile_fortran_module_c_shell(artifact, {}, build_dir, entrypoint=name, library=True)
            finally:
                subprocess.run = ordinary_run
            timings["build_s"] = time.perf_counter() - start
            start = time.perf_counter()
            if backend == "c":
                execution = artifact.prepare_execution(feeds)
            elif backend == "llvm":
                execution = prepare_artifact_execution(artifact, feeds)
            else:
                native_core = FortranCoreNative(built.executable_path, artifact.api.entry_point(name), artifact.api)
                execution = native_core.prepare_execution(feeds)
            timings["prepare_s"] = time.perf_counter() - start
            result = outputs[name][0].id
            for guard in (False, True):
                feeds[parameters["guard"]][0] = guard
                execution.buffers[parameters["guard"]][0] = guard
                execution.run()
                expected = 0.125 + 0.125 * size * (size - 1) / 2
                if guard:
                    expected += 0.5 * (size * 0.125 + 0.125 * size * (size - 1) * (size + 1) / 6)
                actual = execution.buffers[result].item()
                assert actual == expected, (backend, size, guard, actual, expected)
                assert eager(0.125, guard) == expected
                timings[f"native_{guard}_us"] = 1e6 * median_call(execution.run, args.repeats)
                timings[f"python_{guard}_us"] = 1e6 * median_call(lambda: eager(0.125, guard), args.repeats)
                def boundary():
                    if backend == "c":
                        prepared = artifact.prepare_execution(feeds)
                    elif backend == "llvm":
                        prepared = prepare_artifact_execution(artifact, feeds)
                    else:
                        prepared = native_core.prepare_execution(feeds)
                    prepared.run()
                timings[f"prepare_and_run_{guard}_us"] = 1e6 * median_call(boundary, min(args.repeats, 200))
            print("MIXED_TIMING", json.dumps({"backend": backend, "rows": size, **timings}), flush=True)
    print("BUILD_DIRECTORY", directory, flush=True)


if __name__ == "__main__":
    main()
