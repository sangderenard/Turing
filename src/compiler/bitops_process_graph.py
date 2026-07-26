"""Rewrite canonical integer ProcessGraph operations into Turing primitives."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Tuple

import networkx as nx

from ..transmogrifier.graph.graph_express2 import ProcessGraph
from ..transmogrifier.process_op import BitQuantaSpec, ProcessOp, TensorSpec
from ..transmogrifier.solver_types import DomainNode
from ..turing_machine.turing import Hooks, Turing


@dataclass(frozen=True)
class GraphBits:
    """Symbolic bitstring carrier used by the ordinary :class:`Turing` algebra."""

    node_id: int
    accounting: BitQuantaSpec

    @property
    def width(self) -> int:
        return self.accounting.quanta

    def copy(self) -> "GraphBits":
        """Satisfy the generic bitstring carrier protocol without mutation."""

        return self


class _PrimitiveEmitter:
    def __init__(self, graph: ProcessGraph, start_id: int = 0):
        self.graph = graph
        self.next_id = start_id

    def emit(
        self,
        op: str,
        inputs: Iterable[Tuple[GraphBits, str]] = (),
        *,
        width: int,
        attributes=None,
    ) -> GraphBits:
        nid = self.next_id
        self.next_id += 1
        parents = [(value.node_id, role) for value, role in inputs]
        payload = ProcessOp(
            op,
            tuple(role for _, role in parents),
            attributes=attributes or {},
            tensor=TensorSpec(dtype="bit", shape=(width,)),
            bit_quanta=BitQuantaSpec(
                quanta=width,
                bits_per_quantum=1,
                source_nodes=tuple(parent for parent, _ in parents),
            ),
            control={"lowered_by": "bitops"},
        )
        domain_node = DomainNode(shape=(1, 1, 1), unit_size=1)
        domain_node.id = id(domain_node)
        self.graph.G.add_node(
            nid,
            label=op,
            type=op,
            expr_obj=None,
            process_op=payload,
            extra_args=dict(payload.attributes),
            domain_node=domain_node,
            store_id=None,
            parents=parents,
            children=[],
        )
        for src, role in parents:
            self.graph.G.add_edge(src, nid, role=role)
            self.graph.G.nodes[src]["children"].append((nid, role))
        return GraphBits(nid, payload.bit_quanta)

    def hooks(self) -> Hooks:
        def nand(a, b):
            if a.width != b.width:
                raise ValueError("nand operands must have equal bit width")
            return self.emit("nand", ((a, "lhs"), (b, "rhs")), width=a.width)

        def sigma_l(a, amount):
            return self.emit(
                "sigma_L",
                ((a, "value"),),
                width=a.width + int(amount),
                attributes={"amount": int(amount)},
            )

        def sigma_r(a, amount):
            return self.emit(
                "sigma_R",
                ((a, "value"),),
                width=max(a.width - int(amount), 0),
                attributes={"amount": int(amount)},
            )

        def concat(a, b):
            return self.emit(
                "concat",
                ((a, "lhs"), (b, "rhs")),
                width=a.width + b.width,
            )

        def slice_(a, start, stop):
            start, stop = int(start), int(stop)
            return self.emit(
                "slice",
                ((a, "value"),),
                width=max(stop - start, 0),
                attributes={"start": start, "stop": stop},
            )

        def mu(a, b, selector):
            if not (a.width == b.width == selector.width):
                raise ValueError("mu operands must have equal bit width")
            return self.emit(
                "mu",
                ((a, "if_false"), (b, "if_true"), (selector, "selector")),
                width=a.width,
            )

        def length(a):
            return a.width

        def zeros(width):
            width = int(width)
            return self.emit("zeros", width=width, attributes={"length": width})

        return Hooks(
            nand=nand,
            sigma_L=sigma_l,
            sigma_R=sigma_r,
            concat=concat,
            slice=slice_,
            mu=mu,
            length=length,
            zeros=zeros,
        )


def _expand(tm: Turing, op: str, args, width: int) -> GraphBits | None:
    if op == "bitand":
        return tm.AND(*args)
    if op == "bitor":
        return tm.OR(*args)
    if op == "bitxor":
        return tm.XOR(*args)
    if op == "invert":
        return tm.NOT(*args)
    if op == "add":
        return tm.slc(tm.ripple_add(*args), 1, width + 1)
    if op == "sub":
        inverted = tm.NOT(args[1])
        negated = tm.slc(tm.succ(inverted), 1, width + 1)
        return tm.slc(tm.ripple_add(args[0], negated), 1, width + 1)
    if op == "mul":
        product = tm.zeros(width * 2)
        for i in range(width):
            selector_bit = tm.slc(args[1], width - 1 - i, width - i)
            shifted = tm.sigma_L(args[0], i)
            padded = tm.concat(tm.zeros(width - i), shifted)
            selector = tm.zeros(width * 2)
            for j in range(width * 2):
                selector = tm.write_bit(selector, j, selector_bit)
            addend = tm.mu(tm.zeros(width * 2), padded, selector)
            product = tm.slc(tm.ripple_add(product, addend), 1, width * 2 + 1)
        return tm.slc(product, width, width * 2)
    return None


def expand_bitops_process_graph(
    source: ProcessGraph,
    *,
    bit_width: int,
) -> ProcessGraph:
    """Return a graph with supported integer operations expanded to primitives.

    Unsupported operations remain in the graph with an explicit
    ``bitops_status=unexpanded`` attribute. This makes partial lowering
    inspectable and prevents accidental claims of backend completeness.
    """

    if bit_width <= 0:
        raise ValueError("bit_width must be positive")

    target = ProcessGraph(materialize_memory=False)
    emitter = _PrimitiveEmitter(target)
    tm = Turing(emitter.hooks())
    values: Dict[int, GraphBits] = {}

    for old_id in nx.topological_sort(source.G):
        data = source.G.nodes[old_id]
        payload = data.get("process_op")
        if not isinstance(payload, ProcessOp):
            payload = ProcessOp(str(data.get("label") or data.get("type") or "opaque"))
        parent_items = list(data.get("parents", ()))
        args = [values[parent] for parent, _ in parent_items]
        expanded = _expand(tm, payload.op, args, bit_width)
        if expanded is not None:
            values[old_id] = expanded
            continue

        attrs = dict(payload.attributes)
        if payload.op not in {"input", "const", "return", "select"}:
            attrs["bitops_status"] = "unexpanded"
        cloned = replace(payload, attributes=attrs)
        width = (
            cloned.tensor.shape[0]
            if cloned.tensor and cloned.tensor.shape and cloned.tensor.shape[0] is not None
            else bit_width
        )
        values[old_id] = emitter.emit(
            cloned.op,
            ((values[parent], role) for parent, role in parent_items),
            width=int(width),
            attributes={
                **dict(cloned.attributes),
                **({"value": cloned.constant} if cloned.constant is not None else {}),
            },
        )
        new_data = target.G.nodes[values[old_id].node_id]
        if cloned.bit_quanta is None:
            cloned = replace(
                cloned,
                bit_quanta=BitQuantaSpec(
                    quanta=int(width),
                    bits_per_quantum=1,
                    source_nodes=tuple(values[parent].node_id for parent, _ in parent_items),
                ),
            )
        new_data["process_op"] = cloned
        new_data["label"] = data.get("label", cloned.op)
        new_data["type"] = data.get("type", cloned.op)

    target.roots = [values[root].node_id for root in source.roots]
    target.domain_shape = source.domain_shape
    return target
