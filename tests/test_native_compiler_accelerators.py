from __future__ import annotations

import ast
import contextlib
import io

import networkx as nx
import pytest

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.compiler.native_compiler_accelerators import (
    TOPOLOGICAL_ORDER,
    CompilerAcceleratorRegistry,
    compiler_accelerators,
    compile_and_register_topology_accelerator,
    lexicographical_topological_order,
)
from src.compiler.opportunistic_pipeline import (
    ArtifactCheckpointStore,
    Pipeline,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def _registry() -> CompilerAcceleratorRegistry:
    registry = CompilerAcceleratorRegistry()
    registry.register_fallback(
        TOPOLOGICAL_ORDER,
        lambda graph, *, key=None: tuple(
            nx.lexicographical_topological_sort(graph, key=key)
        ),
        name="test:networkx",
    )
    return registry


def test_topological_accelerator_has_an_exact_python_fallback():
    graph = nx.DiGraph([(8, 3), (5, 3), (3, 1), (5, 2)])
    key = lambda node: (node % 3, -node)
    registry = _registry()

    assert lexicographical_topological_order(
        graph, key=key, registry=registry
    ) == tuple(nx.lexicographical_topological_sort(graph, key=key))
    assert registry.resolve(TOPOLOGICAL_ORDER).tier == "python"


def test_registry_refuses_native_semantics_without_a_python_oracle(tmp_path):
    library = tmp_path / "unrelated.dll"
    library.write_bytes(b"not loaded by this registration-only check")
    with pytest.raises(KeyError, match="no correctness fallback"):
        CompilerAcceleratorRegistry().register_native(
            "compiler.unknown",
            lambda value: value,
            name="native:unknown",
            library_path=library,
        )


def test_topology_provider_is_built_by_the_public_source_compiler(tmp_path, monkeypatch):
    from src.compiler import fortran_c_shell, ssa_fortran_backend
    calls = []
    canonical = fortran_c_shell.lower_ast_source_to_ssa

    def observed(source, entrypoint, **kwargs):
        calls.append((source, entrypoint))
        return canonical(source, entrypoint, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("compiler mechanics must not be a handwritten foreign-language module")

    monkeypatch.setattr(fortran_c_shell, "lower_ast_source_to_ssa", observed)
    monkeypatch.setattr(ssa_fortran_backend, "compile_module", forbidden)
    registry = _registry()
    provider = compile_and_register_topology_accelerator(tmp_path, registry=registry)
    assert len(calls) == 1
    assert calls[0][1] == "topological_order"
    graph = nx.DiGraph([(0, 3), (1, 3), (1, 2), (2, 4), (3, 4)])
    assert provider.run(graph) == tuple(nx.lexicographical_topological_sort(graph))


def test_own_compiler_topology_has_exact_parallel_results_across_small_dags(tmp_path):
    import dataclasses
    import json
    import pickle
    import random
    import subprocess
    import sys
    from pathlib import Path
    provider = compile_and_register_topology_accelerator(tmp_path, registry=_registry())
    generator = random.Random(34701)
    graphs = []
    for trial in range(100):
        count = trial % 13
        graph = nx.MultiDiGraph() if trial % 3 == 0 else nx.DiGraph()
        nodes = list(range(count))
        generator.shuffle(nodes)
        graph.add_nodes_from(nodes)
        for left in range(count):
            for right in range(left + 1, count):
                if generator.random() < 0.25:
                    graph.add_edge(nodes[left], nodes[right])
                    if graph.is_multigraph() and generator.random() < 0.5:
                        graph.add_edge(nodes[left], nodes[right])
        graphs.append(graph)
    expected = [list(nx.lexicographical_topological_sort(graph, key=lambda node: node % 2)) for graph in graphs]
    artifact = dataclasses.replace(provider.metadata["artifact"], emission=None,
        _entry=None, _library=None, _validation_error_reset=None, _validation_error_take=None)
    payload = tmp_path / "parallel_topology.pkl"
    payload.write_bytes(pickle.dumps((artifact, provider.metadata["abi"]["parameters"], ("topology_graphs",), graphs)))
    worker = Path(__file__).resolve().parents[1] / "tools/compiler_probes/graph_kernel_native_worker.py"
    completed = subprocess.run([sys.executable, "-u", str(worker), str(payload)],
        capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stdout + completed.stderr[-2000:]
    orders = next(line.removeprefix("GRAPH_KERNEL_ORDERS=") for line in completed.stdout.splitlines()
                  if line.startswith("GRAPH_KERNEL_ORDERS="))
    assert json.loads(orders) == expected


def test_own_compiler_topological_module_registers_and_matches_networkx(tmp_path):
    registry = _registry()
    pipeline = Pipeline(
        ("source", "result"),
        checkpoints=ArtifactCheckpointStore(tmp_path / "checkpoints"),
    )
    provider = compile_and_register_topology_accelerator(
        tmp_path / "native", registry=registry, pipeline=pipeline
    )

    graphs = (
        nx.DiGraph(),
        nx.DiGraph([(0, 3), (1, 3), (1, 2), (2, 4), (3, 4)]),
        nx.DiGraph([(9, 2), (7, 2), (9, 5), (2, 1), (5, 1)]),
    )
    for graph in graphs:
        key = lambda node: (node % 2, -node)
        assert lexicographical_topological_order(
            graph, key=key, registry=registry
        ) == tuple(nx.lexicographical_topological_sort(graph, key=key))

    assert provider.tier == "native"
    assert provider.library_path is not None and provider.library_path.is_file()
    assert provider.library_path.with_suffix(".ll").is_file()
    assert provider.exports == ("compiler_graph_topology__topological_order",)
    assert provider.metadata["abi"]["metadata"]["tensor_semantics"] is False
    assert registry.resolve(TOPOLOGICAL_ORDER) is provider
    foundation, = pipeline.foundations
    assert foundation.loaded
    assert foundation.accelerates == (TOPOLOGICAL_ORDER,)

    # Install the already-loaded function in the process-wide registry and
    # prove that the canonical ProcessGraph relabeler actually consults it.
    calls = []

    def observed(graph, *, key=None):
        calls.append(len(graph))
        return provider.run(graph, key=key)

    compiler_accelerators.register_native(
        TOPOLOGICAL_ORDER,
        observed,
        name=provider.name,
        library_path=provider.library_path,
        exports=provider.exports,
    )
    try:
        module = ast.parse("def kernel(left, right):\n    return left + right\n")
        process_graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            process_graph.build_from_ast(module)
        reduce_abstract_tensor_topology(process_graph)
    finally:
        compiler_accelerators.unregister_native(TOPOLOGICAL_ORDER)

    assert calls
    assert (
        process_graph.function_table.entry("kernel").graph.G.graph[
            "canonical_value_ids"
        ]
        is True
    )


def test_own_compiler_topological_module_reports_cycles_like_networkx(tmp_path):
    registry = _registry()
    compile_and_register_topology_accelerator(
        tmp_path / "native", registry=registry
    )
    graph = nx.DiGraph([(0, 1), (1, 2), (2, 0)])

    with pytest.raises(nx.NetworkXUnfeasible):
        lexicographical_topological_order(graph, registry=registry)


def test_self_hosted_dependency_levels_match_authoritative_scheduler_in_two_native_workers(tmp_path):
    import dataclasses
    import json
    import pickle
    import random
    import subprocess
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from src.transmogrifier.ilpscheduler import ILPScheduler
    from src.compiler.native_compiler_accelerators import (
        DEPENDENCY_LEVELS, compile_and_register_dependency_levels_accelerator,
    )
    registry = CompilerAcceleratorRegistry()
    registry.register_fallback(DEPENDENCY_LEVELS, lambda *_args, **_kwargs: None, name="unused:oracle-registration")
    provider = compile_and_register_dependency_levels_accelerator(tmp_path, registry=registry)
    randomizer = random.Random(88031)
    work, expected = [], []
    for trial in range(30):
        count = 1 + trial % 15
        nodes = list(range(count))
        randomizer.shuffle(nodes)
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        for left in range(count):
            for right in range(left + 1, count):
                if randomizer.random() < 0.2:
                    graph.add_edge(nodes[left], nodes[right])
        for node in graph:
            graph.nodes[node].update(parents=[(parent, "value") for parent in graph.predecessors(node)],
                                     children=[(child, "value") for child in graph.successors(node)])
        order = tuple(nx.lexicographical_topological_sort(graph))
        for preference in ("asap", "alap"):
            work.append((graph, order, preference))
            expected.append({str(node): level for node, level in
                ILPScheduler(SimpleNamespace(G=graph)).compute_levels(preference, "dependency").items()})
    artifact = dataclasses.replace(provider.metadata["artifact"], emission=None,
        _entry=None, _library=None, _validation_error_reset=None, _validation_error_take=None)
    payload = tmp_path / "native_levels.pkl"
    payload.write_bytes(pickle.dumps((artifact, provider.metadata["abi"]["parameters"], ("dependency_graphs",), work)))
    worker = Path(__file__).resolve().parents[1] / "tools/compiler_probes/graph_kernel_native_worker.py"
    completed = subprocess.run([sys.executable, "-u", str(worker), str(payload)],
        capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stdout + completed.stderr[-2000:]
    levels = next(line.removeprefix("GRAPH_KERNEL_LEVELS=") for line in completed.stdout.splitlines()
                  if line.startswith("GRAPH_KERNEL_LEVELS="))
    assert json.loads(levels) == expected
