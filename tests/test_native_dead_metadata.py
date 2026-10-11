"""Whole compiler graph worklist, executed in one native call per graph."""
import dataclasses
import json
from pathlib import Path
import pickle
import subprocess
import sys

import networkx as nx

from src.compiler.native_compiler_accelerators import (
    DEAD_METADATA_ORDER, CompilerAcceleratorRegistry, _python_dead_metadata_order,
    compile_and_register_dead_metadata_accelerator,
)
from tests.test_structural_fold_dead_metadata import _WORKER, _restart_scan, _run, _synthetic_graph

ROOT = Path(__file__).resolve().parents[1]


def test_native_whole_worklist_and_real_compiler_publication(tmp_path, monkeypatch):
    from src.compiler import fortran_c_shell
    canonical = fortran_c_shell.lower_ast_source_to_ssa
    entries = []
    def observed(source, entrypoint, **kwargs):
        entries.append(entrypoint)
        return canonical(source, entrypoint, **kwargs)
    monkeypatch.setattr(fortran_c_shell, "lower_ast_source_to_ssa", observed)
    registry = CompilerAcceleratorRegistry()
    registry.register_fallback(DEAD_METADATA_ORDER, _python_dead_metadata_order, name="test:eager-worklist")
    provider = compile_and_register_dead_metadata_accelerator(
        tmp_path / "native", registry=registry, progress=lambda _message: None,
    )
    assert entries == ["dead_metadata_order"]
    artifact = dataclasses.replace(provider.metadata["artifact"], emission=None, _entry=None,
        _library=None, _validation_error_reset=None, _validation_error_take=None)
    cases = []
    for seed in range(80):
        graph, protected = _synthetic_graph(seed, 25 + seed * 3)
        if seed % 3 == 0:
            graph.add_edge(1, 1)  # Live cycle is not a topological-order error here.
        if seed % 5 == 0:
            graph = nx.MultiDiGraph(graph)
            graph.add_edge(0, 1)
            graph.add_edge(0, 1)
        cases.append((graph, protected, _run(_restart_scan, graph, protected)))
    graph = nx.DiGraph()
    cases.append((graph, set(), _run(_restart_scan, graph, set())))
    payload = tmp_path / "native-worklist.pkl"
    payload.write_bytes(pickle.dumps((artifact, provider.metadata["abi"]["parameters"],
                                    ("dead_metadata_graphs",), cases)))
    child = subprocess.run([sys.executable, "-u", str(ROOT / "tools/compiler_probes/graph_kernel_native_worker.py"),
                            str(payload)], capture_output=True, text=True, timeout=15)
    print(child.stdout, end="", flush=True)
    assert child.returncode == 0, child.stdout + child.stderr[-3000:]
    receipt = json.loads(next(line.split("=", 1)[1] for line in child.stdout.splitlines()
                             if line.startswith("DEAD_METADATA_NATIVE=")))
    assert receipt == {"graphs": 81, "native_calls": 80, "exact_removal_order": True}

    # Reuse the existing exact removal/graph/book-row observer. Only the
    # provider changes; the original removal callback and ledger remain.
    implementation = '''implementation = (
    _restart_scan if mode == "restart" else strategy._remove_dead_metadata_nodes
)'''
    native_install = '''import pickle
if mode == "native":
    from src.compiler.native_compiler_accelerators import (
        DEAD_METADATA_ORDER, compiler_accelerators, _native_dead_metadata_runner,
    )
    with open(sys.argv[4], "rb") as stream:
        compiled, parameters, _names, _arrays = pickle.load(stream)
    compiler_accelerators.register_native(DEAD_METADATA_ORDER,
        _native_dead_metadata_runner(compiled, parameters), name="test:whole-native-worklist",
        library_path=compiled.library_path)
implementation = strategy._remove_dead_metadata_nodes'''
    code = _WORKER.format(root=str(ROOT), tools=str(ROOT / "tools"), tests=str(ROOT / "tests"))
    assert implementation in code
    code = code.replace(implementation, native_install)
    observer = tmp_path / "book_observer.py"
    observer.write_text(code)
    observed = []
    for mode in ("python", "native"):
        out = tmp_path / f"{mode}-energy.json"
        completed = subprocess.run([sys.executable, "-u", str(observer), mode, "energy", str(out), str(payload)],
                                   capture_output=True, text=True, timeout=20)
        print(completed.stdout, end="", flush=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr[-3000:]
        observed.append(json.loads(out.read_text()))
    assert observed[0]["calls"]
    assert observed[1] == observed[0]
    print("DEAD_METADATA_REAL_BOOK=" + json.dumps({"passes": len(observed[0]["calls"]),
                                                 "exact_graphs_and_rows": True}), flush=True)
