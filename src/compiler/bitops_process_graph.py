"""Expand ProcessGraph integer operations through Turing provenance.

Derived arithmetic remains defined by :class:`BitOpsTranslator` and
:class:`Turing`.  This module only splices the recorded primitive provenance
subgraph into the caller's ProcessGraph.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict

import networkx as nx

from .bitops_translator import BitOpsTranslator
from .process_graph_helper import provenance_to_process_graph
from ..transmogrifier.graph.graph_express2 import ProcessGraph


_EXPANDABLE = frozenset(
    {"bitand", "bitor", "bitxor", "invert", "add", "sub", "mul"}
)


def _next_free_id(graph: nx.DiGraph, start: int) -> int:
    while start in graph:
        start += 1
    return start


def _bits_for_node(data: dict, translator: BitOpsTranslator) -> Any:
    if data.get("op") == "const" and isinstance(data.get("constant"), int):
        mask = (1 << translator.bit_width) - 1
        return translator.bits_from_int(int(data["constant"]) & mask)
    return translator.bits_from_int(0)


def _remove_child_reference(data: dict, child: int) -> None:
    data["children"] = [
        item for item in data.get("children", ()) if item[0] != child
    ]


def _append_child_reference(data: dict, child: int, role: str) -> None:
    children = data.setdefault("children", [])
    if (child, role) not in children:
        children.append((child, role))


def expand_bitops_process_graph(
    source: ProcessGraph,
    *,
    bit_width: int,
    translator_factory: Callable[[int], BitOpsTranslator] = BitOpsTranslator,
) -> ProcessGraph:
    """Replace supported integer nodes with recorded Turing primitives.

    Every replacement is obtained by executing the existing instrumented
    Turing implementation. Unsupported operations stay in place and are marked
    explicitly. No derived arithmetic is implemented in this pass.
    """

    if bit_width <= 0:
        raise ValueError("bit_width must be positive")

    target = ProcessGraph(materialize_memory=False)
    target.G = copy.deepcopy(source.G)
    target.scheduler = type(target.scheduler)(target)
    target.roots = list(source.roots)
    target.domain_shape = source.domain_shape
    target.node_map = dict(source.node_map)
    original_order = list(nx.topological_sort(source.G))
    next_id = _next_free_id(
        target.G,
        max((node for node in target.G if isinstance(node, int)), default=-1) + 1,
    )

    for old_id in original_order:
        if old_id not in target.G:
            continue
        old_data = target.G.nodes[old_id]
        op = old_data.get("op") or old_data.get("label")
        if op not in _EXPANDABLE:
            if op not in {"input", "const", "return", "select"}:
                attributes = old_data.setdefault("attributes", {})
                attributes["bitops_status"] = "unexpanded"
                old_data["extra_args"] = dict(attributes)
            continue

        parents = list(old_data.get("parents", ()))
        translator = translator_factory(bit_width)
        operands = []
        provenance_inputs: Dict[int, int] = {}
        for parent, _role in parents:
            bits = _bits_for_node(target.G.nodes[parent], translator)
            provenance_id = translator.graph.bind_input(
                bits,
                name=f"process_node_{parent}",
                metadata={
                    "source_node": parent,
                    "result_length": bit_width,
                },
            )
            provenance_inputs[provenance_id] = parent
            operands.append(bits)

        result = translator.apply_bits(op, *operands)
        result_provenance_id = translator.graph.producer_index(result)
        if result_provenance_id is None:
            raise RuntimeError(f"Turing provenance did not record the result of {op}")
        recorded = provenance_to_process_graph(translator.graph)
        live_nodes = nx.ancestors(recorded.G, result_provenance_id)
        live_nodes.add(result_provenance_id)
        recorded_order = [
            node
            for node in nx.topological_sort(recorded.G)
            if node in live_nodes
        ]
        mapping: Dict[int, int] = {
            node: parent
            for node, parent in provenance_inputs.items()
            if node in live_nodes
        }

        for provenance_id in recorded_order:
            if provenance_id in mapping:
                continue
            next_id = _next_free_id(target.G, next_id)
            mapping[provenance_id] = next_id
            node_data = copy.deepcopy(recorded.G.nodes[provenance_id])
            node_data["control"] = {
                **dict(node_data.get("control") or {}),
                "lowered_by": "bitops",
                "source_operation": op,
            }
            target.G.add_node(next_id, **node_data)
            next_id += 1

        for provenance_id in recorded_order:
            mapped = mapping[provenance_id]
            if provenance_id in provenance_inputs:
                continue
            data = target.G.nodes[mapped]
            mapped_parents = [
                (mapping[parent], role)
                for parent, role in recorded.G.nodes[provenance_id].get(
                    "parents", ()
                )
            ]
            data["parents"] = mapped_parents
            data["children"] = []
            data["input_roles"] = tuple(role for _, role in mapped_parents)
            quanta = int(
                (data.get("tensor") or {}).get("shape", (bit_width,))[0]
            )
            data["bit_quanta"] = {
                "quanta": quanta,
                "bits_per_quantum": 1,
                "pid_domains": (),
                "source_nodes": tuple(parent for parent, _ in mapped_parents),
            }
            for parent, role in mapped_parents:
                target.G.add_edge(parent, mapped, role=role)
                _append_child_reference(target.G.nodes[parent], mapped, role)

        replacement = mapping[result_provenance_id]
        children = list(old_data.get("children", ()))
        for parent, _role in parents:
            if parent in target.G:
                _remove_child_reference(target.G.nodes[parent], old_id)
            if target.G.has_edge(parent, old_id):
                target.G.remove_edge(parent, old_id)
        for child, role in children:
            if child not in target.G:
                continue
            child_data = target.G.nodes[child]
            child_data["parents"] = [
                (replacement if parent == old_id else parent, parent_role)
                for parent, parent_role in child_data.get("parents", ())
            ]
            if target.G.has_edge(old_id, child):
                target.G.remove_edge(old_id, child)
            target.G.add_edge(replacement, child, role=role)
            _append_child_reference(target.G.nodes[replacement], child, role)
        target.roots = [
            replacement if root == old_id else root for root in target.roots
        ]
        target.G.remove_node(old_id)

    return target
