"""The dead-metadata worklist removes exactly what the restart scan removed.

``_fold_callsite_structural_values`` removes unconsumed metadata nodes after
its fixed point.  The original loop restarted a whole-graph scan after every
single removal (quadratic); ``_remove_dead_metadata_nodes`` seeds one scan and
afterwards revisits only the predecessors of each removed node, popping the
lowest original node position first.  This file keeps the ORIGINAL loop
(``_restart_scan``) and requires, node for node, row for row:

* synthetic graphs (chains, fans, diamonds, protected and kept nodes): the
  identical removal sequence and the identical remaining graph;
* every audit lowering (``tools/audit_identity_concordance.py CASES``): the
  identical removal sequence and graph at every call, and the identical
  identity-book rows -- page, row, column, fact and write stamp, in write
  order -- each lowering run in a fresh process under one implementation.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

CASES = (
    "view", "toplevel", "energy", "controller", "controller_untyped",
    "mapping", "oscillator",
)


def _restart_scan(graph, protected_values, remove_node):
    """The original loop, verbatim: restart the whole-graph scan per removal."""
    dead_metadata = True
    while dead_metadata:
        dead_metadata = False
        for node_id, data in tuple(graph.G.nodes(data=True)):
            attributes = data.get("attributes") or {}
            dead_pure_tensor_call = bool(
                attributes.get("tensor_candidate") is not None
                or str(attributes.get("static_python_reference") or "").startswith(
                    "AbstractTensor."
                )
            )
            if (
                int(node_id) not in protected_values
                and graph.G.out_degree(int(node_id)) == 0
                and not (
                    str(data.get("type")) in {"Constant", "Const", "const"}
                    and attributes.get("structural_specialization")
                )
                and (
                    str(data.get("type")) in {
                        "GetAttr", "Attribute", "StaticReference",
                        "Constant", "Const", "const",
                        "Tuple", "List", "Set",
                    }
                    or dead_pure_tensor_call
                )
            ):
                remove_node(int(node_id))
                dead_metadata = True
                break


def _synthetic_graph(seed: int, size: int):
    rng = random.Random(seed)
    types = ("Tuple", "GetAttr", "Const", "Call", "Add", "Input", "List")
    G = nx.DiGraph()
    order = list(range(size))
    rng.shuffle(order)  # node insertion order is not topological order
    for node_id in order:
        kind = rng.choice(types)
        attributes = {}
        if kind == "Call" and rng.random() < 0.5:
            attributes["tensor_candidate"] = object()
        if kind == "Add" and rng.random() < 0.3:
            attributes["static_python_reference"] = "AbstractTensor.add"
        if kind == "Const" and rng.random() < 0.4:
            attributes["structural_specialization"] = True
        G.add_node(node_id, type=kind, attributes=attributes)
    for child in range(size):
        for parent in rng.sample(range(child), min(child, rng.randint(0, 3))):
            G.add_edge(parent, child)
    protected = {n for n in range(size) if rng.random() < 0.1}
    return G, protected


def _run(implementation, G, protected):
    graph = SimpleNamespace(G=G.copy())
    removed: list[int] = []

    def remove_node(node_id):
        node_id = int(node_id)
        if node_id not in graph.G:
            return
        removed.append(node_id)
        graph.G.remove_node(node_id)

    implementation(graph, protected, remove_node)
    return removed, tuple(graph.G.nodes), sorted(graph.G.edges)


@pytest.mark.parametrize("seed", range(40))
def test_worklist_equals_restart_scan_on_synthetic_graphs(seed):
    from src.compiler.glsl_deployment_strategy import _remove_dead_metadata_nodes

    G, protected = _synthetic_graph(seed, 30 + seed * 7)
    old = _run(_restart_scan, G, protected)
    new = _run(_remove_dead_metadata_nodes, G, protected)
    assert new == old
    assert old[0], "the generator should produce removals"


def test_worklist_is_linear_where_restart_scan_is_quadratic():
    from src.compiler.glsl_deployment_strategy import _remove_dead_metadata_nodes

    # A chain of Tuples consumed head-to-tail: each removal exposes the next,
    # so the restart scan pays a scan per node.
    size = 1500
    G = nx.DiGraph()
    for node_id in range(size):
        G.add_node(node_id, type="Tuple", attributes={})
    for node_id in range(size - 1):
        G.add_edge(node_id, node_id + 1)
    timings = {}
    results = {}
    for name, implementation in (
        ("restart", _restart_scan), ("worklist", _remove_dead_metadata_nodes),
    ):
        started = time.perf_counter()
        results[name] = _run(implementation, G, set())
        timings[name] = time.perf_counter() - started
    assert results["worklist"] == results["restart"]
    assert len(results["worklist"][0]) == size
    assert timings["worklist"] < timings["restart"]


_WORKER = r'''
import json, sys
sys.path.insert(0, {root!r})
sys.path.insert(0, {tools!r})
sys.path.insert(0, {tests!r})
mode, case, out = sys.argv[1:4]
import audit_identity_concordance as audit
from src.compiler import glsl_deployment_strategy as strategy
from test_structural_fold_dead_metadata import _restart_scan

implementation = (
    _restart_scan if mode == "restart" else strategy._remove_dead_metadata_nodes
)
calls = []
books = []

def recording(graph, protected_values, remove_node):
    removed = []
    before = [int(n) for n in graph.G.nodes]
    # Some node ids are object addresses (ingestion values) that differ
    # between processes: name those by position in the call's node order.
    names = {{
        node_id: node_id if node_id < 10 ** 12 else -(10 ** 6 + position)
        for position, node_id in enumerate(before)
    }}

    def tracked(node_id):
        if int(node_id) in graph.G:
            removed.append(names[int(node_id)])
        remove_node(node_id)

    from src.compiler.identity_concordance import current_identity_book
    book = current_identity_book()
    clock_before = book.clock[0]
    implementation(graph, protected_values, tracked)
    books.append(book)
    calls.append([
        [names[n] for n in before], removed,
        [names[int(n)] for n in graph.G.nodes],
        [clock_before, book.clock[0]],
    ])

strategy._remove_dead_metadata_nodes = recording
audit.CASES[case]()
book = books[-1]
rows = []
# Rows posted INSIDE each removal call's clock window, in write order: the
# book's own record of what the removals did.  (Later stages iterate
# address-keyed sets, so rows outside the windows differ even between two
# runs of the ORIGINAL loop.)
windows = [tuple(call[3]) for call in calls]
for page_name, page in book.pages.items():
    for (row, column), fact in page.cells.items():
        stamp = page.stamps.get((row, column))
        if stamp is None or not any(lo <= stamp < hi for lo, hi in windows):
            continue
        rows.append([
            stamp, page_name, repr(row), column, repr(fact),
        ])
rows.sort(key=lambda item: (item[0] is None, item[0] or 0))
text = json.dumps({{"calls": calls, "rows": rows}})
# Object addresses, and digests of address-keyed state, differ between
# processes (even between two runs of the ORIGINAL loop); only their place in
# the text is compared.
import re
text = re.sub(
    r"(?<!\d)\d{{12,}}(?!\d)|0x[0-9A-Fa-f]{{8,}}|'[0-9a-f]{{16}}'", "#", text,
)
open(out, "w").write(text)
'''


def _lower_in_fresh_process(mode, case, tmp_path):
    worker = tmp_path / f"worker_{mode}_{case}.py"
    worker.write_text(_WORKER.format(
        root=str(ROOT), tools=str(ROOT / "tools"),
        tests=str(ROOT / "tests"),
    ))
    out = tmp_path / f"{mode}_{case}.json"
    completed = subprocess.run(
        [sys.executable, str(worker), mode, case, str(out)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=900,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return json.loads(out.read_text())


@pytest.mark.parametrize("case", CASES)
def test_audit_lowering_identical_removals_graphs_and_book_rows(case, tmp_path):
    old = _lower_in_fresh_process("restart", case, tmp_path)
    new = _lower_in_fresh_process("worklist", case, tmp_path)
    assert old["calls"], "the fold must reach the dead-metadata removal"
    assert [c[1] for c in new["calls"]] == [c[1] for c in old["calls"]]
    assert new["calls"] == old["calls"]
    if any(call[1] for call in old["calls"]):
        assert old["rows"], "removals must post book rows"
    assert new["rows"] == old["rows"]
