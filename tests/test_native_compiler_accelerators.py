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
from src.compiler.ssa_fortran_backend import fortran_compiler
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


@pytest.mark.skipif(fortran_compiler() is None, reason="no Fortran compiler installed")
def test_fortran_topological_module_compiles_registers_and_matches_networkx(tmp_path):
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
    assert provider.library_path.with_suffix(".api.yaml").is_file()
    assert provider.exports == ("turing_lexicographical_topological_order",)
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


@pytest.mark.skipif(fortran_compiler() is None, reason="no Fortran compiler installed")
def test_fortran_topological_module_reports_cycles_like_networkx(tmp_path):
    registry = _registry()
    compile_and_register_topology_accelerator(
        tmp_path / "native", registry=registry
    )
    graph = nx.DiGraph([(0, 1), (1, 2), (2, 0)])

    with pytest.raises(nx.NetworkXUnfeasible):
        lexicographical_topological_order(graph, registry=registry)
