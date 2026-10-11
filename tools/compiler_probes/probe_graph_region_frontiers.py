"""Tiny public-entry compilation proving dispatch frontier work and parity.

    python -u tools/compiler_probes/probe_graph_region_frontiers.py [2|4|8]

Counts source adjacency queries during region carving without allocation
tracing. Compiles the authored program through the repository compiler and
executes its artifact. All evidence is printed inline.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import time
import weakref

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.compiler import glsl_deployment_strategy as strategy
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)


def main() -> int:
    calls = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    if calls not in {2, 4, 8}:
        raise ValueError("this proof is deliberately limited to 2, 4, or 8 calls")
    source = "def law(x):\n    return (x + 1.0) * (x - 1.0)\n\ndef root(x):\n"
    for index in range(calls):
        source += f"    y{index} = law(x + {float(index)!r})\n"
    source += "    return " + " + ".join(f"y{i}" for i in range(calls)) + "\n"
    contracts = REPO / "extraction_contracts"
    contract = ExtractionContract(contracts / "program_extraction.yaml").with_program_abi({
        "records": {}, "bindings": [], "values": [{
            "function": "root", "parameter": "x", "storage": "span",
            "dtype": "float64", "rank": 1, "shape": [4],
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
    }).with_execution_file(contracts / "vehicle_full_native_execution.yaml")
    counts = {"regions": 0, "source_adjacency_queries": 0,
              "region_members": 0, "index_builds": 0, "indexed_nodes": 0,
              "index_adjacency_queries": 0}
    index_refs = []
    original_dispatch = strategy._dispatch_subgraph

    def counted_dispatch(graph, node_ids, **kwargs):
        counts["regions"] += 1
        counts["region_members"] += len(node_ids)
        original_predecessors = graph.G.predecessors
        original_successors = graph.G.successors

        def predecessors(node):
            counts["source_adjacency_queries"] += 1
            return original_predecessors(node)

        def successors(node):
            counts["source_adjacency_queries"] += 1
            return original_successors(node)

        graph.G.predecessors = predecessors
        graph.G.successors = successors
        try:
            return original_dispatch(graph, node_ids, **kwargs)
        finally:
            del graph.G.predecessors
            del graph.G.successors

    strategy._dispatch_subgraph = counted_dispatch
    original_index = getattr(strategy, "_dispatch_graph_index", None)
    if original_index is not None:
        def counted_index(graph):
            counts["index_builds"] += 1
            counts["indexed_nodes"] += len(graph.G)
            original_predecessors = graph.G.predecessors

            def predecessors(node):
                counts["index_adjacency_queries"] += 1
                return original_predecessors(node)

            graph.G.predecessors = predecessors
            try:
                index = original_index(graph)
            finally:
                graph.G.predecessors = original_predecessors
            index_refs.append(weakref.ref(index))
            return index
        strategy._dispatch_graph_index = counted_index
    started = time.perf_counter()
    try:
        module, outputs, _exports = lower_ast_source_to_ssa(
            source, "root", name="graph_frontier_proof",
            extraction_contract=contract, runtime_closure_only=True,
            python_bindings={}, tensor_ssa_reference=c_backend_repository_ssa_reference(),
        )
    finally:
        strategy._dispatch_subgraph = original_dispatch
        if original_index is not None:
            strategy._dispatch_graph_index = original_index
    lowered = time.perf_counter() - started
    report = concordance_report(module)
    print(report, flush=True)
    if not report.splitlines()[0].endswith(", 0 finding(s)"):
        raise AssertionError("identity concordance has structural findings")
    if original_index is not None:
        assert counts["source_adjacency_queries"] == 0, counts
        assert counts["index_adjacency_queries"] == counts["indexed_nodes"], counts
        assert all(reference() is None for reference in index_refs), "phase index retained after lowering"
    root = module.functions["graph_frontier_proof__root"]
    artifact = emit_ssa_module_to_c(module, root.name)
    if not artifact.complete:
        raise RuntimeError(artifact.shortfalls)
    print(f"calls={calls} lower_seconds={lowered:.3f} functions={len(module.functions)}", flush=True)
    print(f"graph_work={counts}", flush=True)
    if original_index is not None:
        print("phase_indexes_released=True", flush=True)
    print(f"emitted_sha256={hashlib.sha256(artifact.source.encode()).hexdigest()}", flush=True)
    artifact.compile(REPO / "build" / "graph_region_frontiers" / f"calls_{calls}")
    parameter = dict(root.metadata["parameter_names"])["x"]
    values = np.array([1.5, -2.0, 3.25, 4.0], dtype=np.float64)
    execution = artifact.prepare_execution({int(parameter): values.copy()}).run()
    actual = np.asarray(execution.buffers[outputs[root.name][0].id]).reshape(-1)
    expected = sum((values + i + 1.0) * (values + i - 1.0) for i in range(calls))
    np.testing.assert_array_equal(actual, expected)
    print(f"native={actual.tolist()} exact_parity=True total_seconds={time.perf_counter() - started:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
