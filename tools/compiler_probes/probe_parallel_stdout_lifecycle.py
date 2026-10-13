"""Reproduce source workers restoring another worker's closed stdout.

The gates force the two real ingestion scopes to overlap; they are solely a
race regression, not part of the worker-throughput or overlap measurements.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
import threading


def observe_parallel_ingestion_stdout():
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.work_contract import active_contract, scoped_work_contract
    from src.common.tensors import topological_reducer
    from src.transmogrifier.graph.graph_express2 import ProcessGraph

    original_stream = sys.stdout
    original_build = ProcessGraph.build_from_ast
    original_reduce = topological_reducer.reduce_abstract_tensor_topology
    lock, local = threading.Lock(), threading.local()
    second_inside, first_restored = threading.Event(), threading.Event()
    ordinal = [0]

    def build(graph, *args, **kwargs):
        with lock:
            local.ordinal = ordinal[0]
            ordinal[0] += 1
        if local.ordinal == 0:
            assert second_inside.wait(10), "second ingestion did not enter"
        else:
            second_inside.set()
            assert first_restored.wait(10), "first ingestion did not exit"
        try:
            return original_build(graph, *args, **kwargs)
        except BaseException:
            first_restored.set()
            raise

    def reduce(graph):
        if local.ordinal == 0:
            # Reduction follows ingestion's stdout context exit exactly.
            first_restored.set()
        return original_reduce(graph)

    contract = ExtractionContract("extraction_contracts/program_extraction.yaml").with_program_abi({
        "records": {}, "bindings": [], "values": [
            {"function": name, "parameter": "x", "storage": "scalar", "dtype": "float64",
             "rank": 0, "shape": [], "python_type": "builtins.float"}
            for name in ("left", "right")]})
    ProcessGraph.build_from_ast = build
    topological_reducer.reduce_abstract_tensor_topology = reduce
    try:
        with scoped_work_contract(dataclasses.replace(active_contract(),
                memory_budget_bytes=1 << 30, memory_elastic_available_percent=0)):
            results = lower_ast_source_to_ssa([
                {"label": name, "source": f"def {name}(x):\n    return x + 1.0\n",
                 "entrypoint": name, "options": {"extraction_contract": contract,
                                                   "progress": lambda _message: None}}
                for name in ("left", "right")], source_compilation="parallel", progress_bars=False,
                partitioned_options={"jobs": 2, "max_total_resident_bytes": 1 << 30,
                    "max_worker_memory_bytes": 1 << 30, "worker_resident_reservation_bytes": 1 << 20})
        observed = sys.stdout
        return {"sources_completed": len(results), "stdout_preserved": observed is original_stream,
                "stdout_closed": observed.closed, "stdout_name": str(getattr(observed, "name", ""))}
    finally:
        # Restore the diagnostic process's stream even when exposing the bug.
        # The compiler itself must preserve it; this is not a compiler repair.
        sys.stdout = original_stream
        ProcessGraph.build_from_ast = original_build
        topological_reducer.reduce_abstract_tensor_topology = original_reduce


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    receipt = observe_parallel_ingestion_stdout()
    print("PARALLEL_STDOUT=" + json.dumps(receipt), flush=True)
    return int(not receipt["stdout_preserved"] or receipt["stdout_closed"])


if __name__ == "__main__":
    raise SystemExit(main())
