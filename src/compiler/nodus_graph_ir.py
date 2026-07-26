"""Serialize a semantic ProcessGraph into Nodus's executable GraphIR."""

from __future__ import annotations

import json
from typing import Any, Dict

import networkx as nx


def _quote(value: str) -> str:
    # JSON string escaping is compatible with GraphIR's quoted-string subset
    # for quotes and backslashes.
    return json.dumps(value, ensure_ascii=True)


def _scalar(value: Any) -> str | None:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return _quote(value)
    if value is None:
        return None
    return _quote(json.dumps(value, sort_keys=True, default=str))


def process_graph_to_nodus_graph_ir(graph) -> str:
    """Emit a Nodus graph of typed AbstractTensor operation tools.

    Every ProcessGraph node becomes an ``abstract_tensor_tool`` node with one
    output port and an ordered input port per dependency. Edges preserve roles.
    Canonical operation validation occurs in Nodus when ``tensor_node`` runs.
    """

    lines = ["# Generated from Turing ProcessGraph schema 1."]
    output_vars: Dict[int, str] = {}

    for ordinal, nid in enumerate(nx.topological_sort(graph.G)):
        data = graph.G.nodes[nid]
        op = str(data.get("op") or data.get("label") or data.get("type") or "opaque")
        attributes = dict(data.get("attributes") or data.get("extra_args") or {})
        constant = data.get("constant")
        tensor = data.get("tensor") or {}
        bit_quanta = data.get("bit_quanta") or {}
        output_roles = tuple(data.get("output_roles") or ("result",))
        schema_version = int(data.get("schema_version", 1))

        node_var = f"n{ordinal}"
        output_var = f"o{ordinal}"
        lines.append(f"{node_var} = tensor_node({_quote(op)});")
        lines.append(
            f"attr({node_var}, \"process.schema_version\", {schema_version});"
        )
        for key, value in sorted(attributes.items()):
            encoded = _scalar(value)
            if encoded is not None:
                lines.append(
                    f"attr({node_var}, {_quote('process.' + key)}, {encoded});"
                )
        if constant is not None:
            encoded = _scalar(constant)
            if encoded is not None:
                lines.append(f"attr({node_var}, \"process.constant\", {encoded});")
        if tensor:
            if tensor.get("dtype"):
                lines.append(
                    f"attr({node_var}, \"tensor.dtype\", {_quote(tensor['dtype'])});"
                )
            if tensor.get("shape"):
                lines.append(
                    f"attr({node_var}, \"tensor.shape\", "
                    f"{_quote(json.dumps(tensor['shape']))});"
                )
            if tensor.get("device"):
                lines.append(
                    f"attr({node_var}, \"tensor.device\", {_quote(tensor['device'])});"
                )
        if bit_quanta:
            lines.append(
                f"attr({node_var}, \"bitbit.quanta\", {bit_quanta['quanta']});"
            )
            lines.append(
                f"attr({node_var}, \"bitbit.bitsforbits\", "
                f"{bit_quanta.get('bits_per_quantum', 1)});"
            )
            if bit_quanta.get("pid_domains"):
                lines.append(
                    f"attr({node_var}, \"bitbit.pid_domains\", "
                    f"{_quote(json.dumps(bit_quanta['pid_domains']))});"
                )
        output_role = output_roles[0] if output_roles else "control"
        lines.append(f"{output_var} = tensor_output({node_var}, {_quote(output_role)});")
        output_vars[nid] = output_var

        for input_index, (parent, role) in enumerate(data.get("parents", ())):
            port_var = f"i{ordinal}_{input_index}"
            lines.append(f"{port_var} = tensor_input({node_var}, {_quote(role)});")
            lines.append(f"connect({output_vars[parent]}, {port_var});")

    return "\n".join(lines) + "\n"
