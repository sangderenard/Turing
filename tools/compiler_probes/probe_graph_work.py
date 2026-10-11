"""Bounded public-compiler scaling: workers, residency, provenance and native parity.

python -u tools/compiler_probes/probe_graph_work.py --matrix
python -u tools/compiler_probes/probe_graph_work.py --calls 16 --workers 4 --resident-regions 4

Not an orbital benchmark. No allocation tracing. Foundation setup is timed
separately and compiled through OUR public entry. Each runtime check executes
in the existing crash-isolated child, with a 10-second limit.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.graph_work import ensure_graph_foundation
from src.compiler.identity_concordance import concordance_report
from src.compiler.ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm
from src.compiler.work_contract import PRESETS, set_active_contract


def _progress(message):
    if message.startswith(("ssa-source:", "graph-work:")):
        print(message, flush=True)


def run(calls, workers, resident_regions):
    set_active_contract(dataclasses.replace(PRESETS["develop"], graph_workers=workers,
                                           graph_resident_regions=resident_regions))
    source = "def law(x):\n    return (x + 1.0) * (x - 1.0)\n\ndef root(x):\n"
    for index in range(calls):
        source += f"    y{index} = law(x + {float(index)!r})\n"
    source += "    return " + " + ".join(f"y{i}" for i in range(calls)) + "\n"
    contracts = REPO / "extraction_contracts"
    contract = ExtractionContract(contracts / "program_extraction.yaml").with_program_abi({
        "records": {}, "bindings": [], "values": [{
            "function": "root", "parameter": "x", "storage": "span", "dtype": "float64",
            "rank": 1, "shape": [4], "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
    }).with_execution_file(contracts / "vehicle_full_native_execution.yaml")
    started = time.perf_counter()
    module, outputs, _exports = lower_ast_source_to_ssa(
        source, "root", name="graph_work_scaling", extraction_contract=contract,
        runtime_closure_only=True, python_bindings={}, progress=_progress,
    )
    lowered = time.perf_counter() - started
    report = concordance_report(module)
    if not report.splitlines()[0].endswith(", 0 finding(s)"):
        raise AssertionError(report)
    root = module.functions["graph_work_scaling__root"]
    artifact = emit_ssa_function_to_llvm(module, root.name)
    if not artifact.complete:
        raise AssertionError(artifact.shortfalls)
    directory = REPO / "build" / "graph_work_scaling" / f"calls_{calls}_workers_{workers}_resident_{resident_regions}"
    build_started = time.perf_counter()
    artifact = compile_artifact(artifact, directory=directory)
    build_seconds = time.perf_counter() - build_started
    values = np.array([1.5, -2.0, 3.25, 4.0], dtype=np.float64)
    parameters = dict(root.metadata["parameter_names"])
    runtime_artifact = dataclasses.replace(artifact, emission=None, _entry=None, _library=None,
                                          _validation_error_reset=None, _validation_error_take=None)
    payload = directory / "native_parity.pkl"
    payload.write_bytes(pickle.dumps((runtime_artifact, parameters, ("program_outputs",), {
        "inputs": {int(parameters["x"]): values},
        "outputs": tuple(int(value.id) for value in outputs[root.name]),
    })))
    completed = subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("graph_kernel_native_worker.py")),
                                str(payload)], capture_output=True, text=True, timeout=10)
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr[-2000:])
    output = next(line.removeprefix("GRAPH_PROGRAM_OUTPUTS=") for line in completed.stdout.splitlines()
                  if line.startswith("GRAPH_PROGRAM_OUTPUTS="))
    actual = np.asarray(json.loads(output)[0]).reshape(-1)
    expected = sum((values + index + 1.0) * (values + index - 1.0) for index in range(calls))
    np.testing.assert_array_equal(actual, expected)
    stats = module.metadata["graph_work"]
    assert stats["live_topology_peak"] <= resident_regions
    assert stats["outstanding_peak"] <= min(workers, resident_regions)
    assert stats["admitted"] == stats["published"]
    receipt = dict(calls=calls, workers=workers, functions=len(module.functions),
                   lower_seconds=round(lowered, 4), native_build_seconds=round(build_seconds, 4),
                   total_seconds=round(time.perf_counter() - started, 4),
                   exact_native_parity=True, native=actual.tolist(), graph_work=stats)
    print("GRAPH_WORK_RESULT=" + json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def run_mechanics(unit_nodes):
    """Larger DAG units through the SAME admitted worker operation, not a source benchmark."""
    import networkx as nx
    from src.compiler.graph_work import dependency_snapshot
    from src.compiler.native_compiler_accelerators import TOPOLOGICAL_ORDER, DEPENDENCY_LEVELS, compiler_accelerators
    from src.transmogrifier.ilpscheduler import ILPScheduler

    graph = nx.DiGraph()
    width, depth = unit_nodes // 32, 32
    graph.add_nodes_from(range(width * depth))
    for layer in range(1, depth):
        for column in range(width):
            node = layer * width + column
            graph.add_edges_from(((layer - 1) * width + ((column + step) % width), node) for step in (0, 1, 7))
    for node in graph:
        graph.nodes[node].update(parents=[(parent, "value") for parent in graph.predecessors(node)],
                                 children=[(child, "value") for child in graph.successors(node)])
    snapshots, expected = [], []
    for _index in range(16):
        preference = "asap" if _index % 2 == 0 else "alap"
        snapshots.append(dependency_snapshot(SimpleNamespace(G=graph), preference))
        expected.append({str(node): level for node, level in
                         ILPScheduler(SimpleNamespace(G=graph)).compute_levels(preference, "dependency").items()})
    providers = [compiler_accelerators.resolve(semantic) for semantic in (TOPOLOGICAL_ORDER, DEPENDENCY_LEVELS)]
    artifacts = tuple(dataclasses.replace(provider.metadata["artifact"], emission=None, _entry=None, _library=None,
                                         _validation_error_reset=None, _validation_error_take=None) for provider in providers)
    parameters = tuple(provider.metadata["abi"]["parameters"] for provider in providers)
    directory = REPO / "build" / "graph_work_scaling"
    directory.mkdir(parents=True, exist_ok=True)
    for workers in (1, 2, 4, 8):
        payload = directory / f"mechanics_workers_{workers}.pkl"
        payload.write_bytes(pickle.dumps((artifacts, parameters, ("graph_work_batch",),
                                         {"workers": workers, "resident_regions": 8, "snapshots": snapshots})))
        completed = subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("graph_kernel_native_worker.py")),
                                   str(payload)], capture_output=True, text=True, timeout=15)
        if completed.returncode:
            raise RuntimeError(completed.stdout + completed.stderr[-2000:])
        output = next(line.removeprefix("GRAPH_WORK_BATCH=") for line in completed.stdout.splitlines()
                      if line.startswith("GRAPH_WORK_BATCH="))
        receipt = json.loads(output)
        assert receipt.pop("levels") == expected
        assert receipt["stats"]["live_topology_peak"] <= 8
        print("GRAPH_MECHANICS_RESULT=" + json.dumps({"workers": workers, "nodes_per_unit": len(graph),
            "edges_per_unit": graph.number_of_edges(), "units": len(snapshots),
            "exact_scheduler_parity": True, **receipt}, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, choices=(2, 4, 8, 16, 32), default=8)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--resident-regions", type=int, default=4)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--mechanics", action="store_true", help="16 bounded 1024-node DAG units; not an orbital/source timing")
    parser.add_argument("--unit-nodes", type=int, choices=(1024, 4096), default=1024)
    args = parser.parse_args()
    if not 1 <= args.resident_regions <= 16:
        parser.error("resident-regions must be between 1 and 16 for this tiny probe")
    # Same startup for every matrix point; do not attribute foundation build
    # time to the first ordinary program or compare warm and cold samples.
    set_active_contract(dataclasses.replace(PRESETS["develop"], graph_workers=max(2, args.workers),
                                           graph_resident_regions=args.resident_regions))
    started = time.perf_counter()
    ensure_graph_foundation(_progress)
    print(f"GRAPH_FOUNDATION_SETUP_SECONDS={time.perf_counter() - started:.4f}", flush=True)
    try:
        if args.mechanics:
            run_mechanics(args.unit_nodes)
            return
        results = []
        points = ((calls, workers) for calls in (4, 8, 16) for workers in (2, 4, 8)) if args.matrix else ((args.calls, args.workers),)
        for calls, workers in points:
            results.append(run(calls, workers, max(args.resident_regions, 8) if args.matrix else args.resident_regions))
        print("GRAPH_WORK_MATRIX=" + json.dumps(results, sort_keys=True), flush=True)
    finally:
        set_active_contract(None)


if __name__ == "__main__":
    main()
