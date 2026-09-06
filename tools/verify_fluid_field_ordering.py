"""Read-only verification that the write->read dependency edge fix
(topological_reducer.py, attribute_effect_nodes) closes the gap documented
in tools/HANDOFF_2026-08-17_CRASH.md and tools/DIFFERENTIAL_PHASES.md.

Safe by construction: uses only ``ProcessGraph.build_from_ast`` with its
defaults (``resolve_unresolved_parents=False``), which never reaches
``_expand_unresolved_ast_parents`` / the x86 read-head machine-decompiler.
No ``extraction_contract`` is touched, no native code is executed.
"""
from __future__ import annotations

import ast
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.compiler.symbolic_fluid_dt import SYMBOLIC_FLUID_DT_SOURCE

module = ast.parse(SYMBOLIC_FLUID_DT_SOURCE)
graph = ProcessGraph(materialize_memory=False)
with contextlib.redirect_stdout(io.StringIO()):
    graph.build_from_ast(module)
reduce_abstract_tensor_topology(graph)

func_name = module.body[0].name
function_graph = graph.function_table.entry(func_name).graph

fields = ["height", "momentum_x", "momentum_y", "tracer"]
print(f"function: {func_name}")
for field in fields:
    next_field = f"next_{field}"
    indexed_stores = [
        node_id
        for node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "IndexedStore"
        and any(
            (function_graph.G.nodes[p].get("attributes") or {}).get("attribute")
            == next_field
            for p, role in data["parents"]
            if role == "base" and p in function_graph.G
        )
    ]
    reads = [
        node_id
        for node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "GetAttr"
        and (data.get("attributes") or {}).get("attribute") == next_field
    ]
    base_parents = {
        p
        for store_id in indexed_stores
        for p, role in function_graph.G.nodes[store_id]["parents"]
        if role == "base"
    }
    post_loop_reads = [r for r in reads if r not in base_parents]
    ok = bool(indexed_stores) and bool(post_loop_reads) and all(
        any(nx.has_path(function_graph.G, store_id, read_id) for store_id in indexed_stores)
        for read_id in post_loop_reads
    )
    print(
        f"  {field:12s} stores={len(indexed_stores)} "
        f"post_loop_reads={len(post_loop_reads)} ordered_ok={ok}"
    )
