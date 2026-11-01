#!/usr/bin/env python3
"""Compile authored Python through repository SSA and the standalone C shell.

No C-written benchmark loop, ctypes execution, or Python fallback is substituted.
An unsupported source program fails compilation and produces no native timings.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PYTHON_RUN = r'''
import json, pathlib, sys, time
source = pathlib.Path(sys.argv[1])
namespace = {'__name__': 'authored_queue_benchmark', '__file__': str(source)}
exec(compile(source.read_text(), str(source), 'exec'), namespace)
inputs = json.loads(sys.argv[2])
start = time.perf_counter()
value = namespace['root'](**inputs)
elapsed = time.perf_counter() - start
print(json.dumps({'implementation': 'authored_python_reference',
                  'seconds': elapsed, 'result': value}), flush=True)
'''


def python_run(source, inputs, timeout):
    process = subprocess.run([sys.executable, "-c", PYTHON_RUN, str(source), json.dumps(inputs)],
                             capture_output=True, text=True, check=True, timeout=timeout)
    # The fixture performs validation inside every worker; unexpected exceptions
    # must not masquerade as a faster run with fewer records.
    if process.stderr:
        raise RuntimeError(process.stderr)
    return json.loads(process.stdout)


def compile_source(source_path, inputs, directory):
    import numpy as np
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import concordance_report
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c

    module, _, exports = lower_ast_source_to_ssa(
        source_path.read_text(), "root",
        extraction_contract=ExtractionContract(ROOT / "extraction_contracts/program_extraction.yaml"),
        progress=lambda _: None, progress_bars=False,
    )
    report = concordance_report(module)
    print(report, flush=True)
    if "0 finding(s)" not in report:
        raise RuntimeError("SSA concordance audit did not pass")
    entry, = exports
    parameters = dict(module.functions[entry].metadata["parameter_names"])
    feeds = {parameters[name]: np.array([value], dtype=np.int64) for name, value in inputs.items()}
    artifact = emit_ssa_module_to_c(module, entry)
    if not artifact.complete:
        raise RuntimeError(f"C emission refused: {artifact.shortfalls}")
    executable = artifact.compile_standalone(directory, feeds, optimization="O3")
    print(concordance_report(module), flush=True)
    return artifact, executable, feeds


def native_run(artifact, executable, feeds, timeout):
    import numpy as np
    from src.transmogrifier.dtype_layout import c_lane_numpy_dtype
    process = subprocess.run([str(executable.executable_path), "1", "0"],
                             cwd=executable.directory, capture_output=True,
                             text=True, check=True, timeout=timeout)
    print(process.stdout, end="", flush=True)
    match = re.search(r"\belapsed_s=([0-9.eE+-]+)", process.stdout)
    if match is None:
        raise RuntimeError("standalone C shell did not report elapsed_s")
    raw = executable.final_outputs_path.read_bytes()
    offset, outputs = 0, []
    for value_id, dtype, shape, role in zip(artifact.buffer_order, artifact.buffer_dtypes,
                                           artifact.buffer_shapes, artifact.buffer_roles):
        count = np.asarray(feeds[value_id]).size if value_id in feeds else math.prod(shape or (1,))
        held = np.frombuffer(raw, dtype=c_lane_numpy_dtype(dtype), count=count, offset=offset)
        offset += held.nbytes
        if role == "output":
            if held.size != 1:
                raise RuntimeError("benchmark expects one scalar result")
            outputs.append(held.item())
    if offset != len(raw) or len(outputs) != 1:
        raise RuntimeError("standalone result buffer layout mismatch")
    return {"implementation": "python_to_ssa_to_c_shell",
            "seconds": float(match.group(1)), "result": outputs[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "examples/byte_queue/threaded_pipeline.py")
    parser.add_argument("--inputs", default='{"lanes":4,"hops":2,"items":5000,"capacity":64}')
    parser.add_argument("--directory", type=Path, default=ROOT / "scratch/python-queue-c-shell")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--python-only", action="store_true", help="measure the reference workload without claiming native support")
    args = parser.parse_args()
    inputs = json.loads(args.inputs)
    if not isinstance(inputs, dict) or not all(isinstance(v, int) and not isinstance(v, bool) for v in inputs.values()):
        parser.error("inputs must be a JSON object of integer scalars")
    if args.rounds < 1 or args.timeout <= 0:
        parser.error("rounds and timeout must be positive")
    source = args.source.resolve()
    compiled = None
    if not args.python_only:
        # Compile first: a refusal must never be followed by timings labeled as
        # native, and a handwritten C workload is never an alternate path.
        compiled = compile_source(source, inputs, args.directory)
    for index in range(args.rounds):
        reference = python_run(source, inputs, args.timeout)
        print(json.dumps({"round": index + 1, **reference}), flush=True)
        if compiled is not None:
            native = native_run(*compiled, args.timeout)
            if native["result"] != reference["result"]:
                raise AssertionError((native, reference))
            print(json.dumps({"round": index + 1, **native,
                              "speedup": reference["seconds"] / native["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
