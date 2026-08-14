"""Reverse-mode differentiation of semantic :class:`ProcessGraph` objects.

The forward ``ProcessGraph`` is the authority.  This module constructs a new
``ProcessGraph`` whose nodes are ordinary, inspectable numerical operations;
it does not execute Python backward callables and it does not discover a
backward program by observing a tape traversal.

This first tranche covers acyclic numerical graphs.  Logical control is
rejected explicitly until the planner-owned ``ControlProgram`` adjoint is
attached: silently differentiating one observed branch would be a false
whole-program result.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import networkx as nx

from ..common.tensors.backward_registry import BACKWARD_RULES
from ..transmogrifier.graph.graph_express2 import ProcessGraph


class ProcessGraphAutogradError(ValueError):
    """The semantic forward graph cannot yet be differentiated faithfully."""


@dataclass(frozen=True)
class ProcessGraphAdjoint:
    """A parametric backward graph and its explicit forward-value contract."""

    forward: ProcessGraph
    backward: ProcessGraph
    output_value_ids: tuple[int, ...]
    wrt_value_ids: tuple[int, ...]
    seed_value_ids: Mapping[int, int]
    saved_value_ids: Mapping[int, int]
    gradient_value_ids: Mapping[int, int]


@dataclass(frozen=True)
class ForwardLossBackwardMotion:
    """One semantic graph containing forward, loss, and reverse motion."""

    graph: ProcessGraph
    loss_value_ids: tuple[int, ...]
    gradient_value_ids: Mapping[int, int]
    seed_value_ids: Mapping[int, int]


@dataclass(frozen=True)
class TrainingMotionSSALowering:
    """Direct ProcessGraph-to-repository-SSA lowering result."""

    module: Any
    function_name: str
    outputs: Mapping[str, int]
    shortfalls: tuple[Any, ...]


def _operation(data: Mapping[str, Any]) -> str:
    return str(data.get("op") or data.get("type") or "").casefold()


def _parents(graph: ProcessGraph, node_id: int) -> tuple[int, ...]:
    data = graph.G.nodes[node_id]
    declared = tuple(int(parent) for parent, _role in data.get("parents", ()))
    if declared:
        return declared
    return tuple(int(parent) for parent in graph.G.predecessors(node_id))


def _returned_value(graph: ProcessGraph, node_id: int) -> int:
    if _operation(graph.G.nodes[node_id]) != "return":
        return int(node_id)
    parents = _parents(graph, node_id)
    if len(parents) != 1:
        raise ProcessGraphAutogradError(
            f"return node {node_id} has {len(parents)} values; "
            "name each differentiated output explicitly"
        )
    return parents[0]


class _AdjointBuilder:
    def __init__(self, forward: ProcessGraph) -> None:
        self.forward = forward
        self.backward = ProcessGraph(materialize_memory=False)
        self.next_id = 0
        self.saved: dict[int, int] = {}

    def add(
        self,
        op: str,
        parents: Iterable[tuple[int, str]] = (),
        *,
        label: str | None = None,
        attributes: Mapping[str, Any] | None = None,
        source_forward_id: int | None = None,
    ) -> int:
        node_id = self.next_id
        self.next_id += 1
        parent_items = tuple((int(parent), str(role)) for parent, role in parents)
        attrs = dict(attributes or {})
        if source_forward_id is not None:
            attrs.setdefault("source_forward_id", int(source_forward_id))
        self.backward.G.add_node(
            node_id,
            label=label or op,
            type=op,
            op=op,
            parents=list(parent_items),
            children=[],
            attributes=attrs,
            extra_args=copy.deepcopy(attrs),
            tensor={},
            control={},
            constant=None,
            expr_obj=None,
            store_id=None,
            schema_version=1,
        )
        for parent, role in parent_items:
            self.backward.G.add_edge(parent, node_id, role=role)
            self.backward.G.nodes[parent]["children"].append((node_id, role))
        return node_id

    def input(
        self, name: str, *, kind: str, source_forward_id: int,
    ) -> int:
        node_id = self.add(
            "input",
            label=name,
            attributes={
                "name": name,
                "binding_kind": kind,
                "source_forward_id": int(source_forward_id),
            },
            source_forward_id=source_forward_id,
        )
        self.backward.G.nodes[node_id]["tensor"] = copy.deepcopy(
            self.forward.G.nodes[source_forward_id].get("tensor") or {}
        )
        return node_id

    def saved_value(self, forward_id: int) -> int:
        forward_id = int(forward_id)
        if forward_id not in self.saved:
            self.saved[forward_id] = self.input(
                f"saved_{forward_id}",
                kind="saved_forward",
                source_forward_id=forward_id,
            )
        return self.saved[forward_id]

    def unary(self, op: str, value: int, source: int) -> int:
        return self.add(
            op, ((value, "operand"),), source_forward_id=source,
        )

    def binary(self, op: str, left: int, right: int, source: int) -> int:
        return self.add(
            op,
            ((left, "lhs"), (right, "rhs")),
            source_forward_id=source,
        )

    def reduce_to_shape(
        self,
        gradient: int,
        forward_id: int,
        source: int,
        *,
        gradient_shape: tuple[int, ...] | None = None,
    ) -> int:
        """Expand the registry's ``unbroadcast`` helper into existing ops."""

        source_shape = tuple(gradient_shape or (
            (self.forward.G.nodes[source].get("tensor") or {}).get("shape") or ()
        ))
        target_shape = tuple(
            (self.forward.G.nodes[forward_id].get("tensor") or {}).get("shape") or ()
        )
        if source_shape == target_shape:
            return gradient
        if not source_shape or not target_shape:
            raise ProcessGraphAutogradError(
                "canonical backward helper unbroadcast requires symbolic-shape "
                "ProcessGraph lowering before this region can dispatch natively; "
                f"source={source} shape={source_shape}, target={forward_id} "
                f"shape={target_shape}"
            )
        current = gradient
        current_shape = list(source_shape)
        if len(current_shape) > len(target_shape):
            for _ in range(len(current_shape) - len(target_shape)):
                current = self.add(
                    "sum",
                    ((current, "operand"),),
                    attributes={"dim": 0, "keepdim": False},
                    source_forward_id=source,
                )
                current_shape.pop(0)
        reduce_axes = tuple(
            axis
            for axis, (actual, target) in enumerate(
                zip(current_shape, target_shape)
            )
            if target == 1 and actual != 1
        )
        for axis in reduce_axes:
            current = self.add(
                "sum",
                ((current, "operand"),),
                attributes={"dim": int(axis), "keepdim": True},
                source_forward_id=source,
            )
        return self.add(
            "reshape",
            ((current, "operand"),),
            attributes={"shape": target_shape},
            source_forward_id=source,
        )

    def expand_reduction(
        self,
        gradient: int,
        forward_id: int,
        source: int,
        attributes: Mapping[str, Any],
    ) -> int:
        """Expand the registry's ``expand_reduction`` using canonical ops."""

        source_shape = tuple(
            (self.forward.G.nodes[source].get("tensor") or {}).get("shape") or ()
        )
        target_shape = tuple(
            (self.forward.G.nodes[forward_id].get("tensor") or {}).get("shape") or ()
        )
        if not target_shape:
            raise ProcessGraphAutogradError(
                "canonical backward helper expand_reduction requires a known "
                f"target shape for forward value {forward_id}"
            )
        axis = attributes.get("axis", attributes.get("dim"))
        axes = (
            tuple(range(len(target_shape))) if axis is None
            else tuple(int(item) for item in axis)
            if isinstance(axis, (tuple, list))
            else (int(axis),)
        )
        axes = tuple(sorted(item % len(target_shape) for item in axes))
        current = gradient
        if not bool(attributes.get("keepdim", False)):
            restored = list(source_shape)
            for item in axes:
                restored.insert(item, 1)
            current = self.add(
                "reshape",
                ((current, "operand"),),
                attributes={"shape": tuple(restored)},
                source_forward_id=source,
            )
        reference = self.saved_value(forward_id)
        zero = self.binary("sub", reference, reference, source)
        return self.binary(
            "add",
            zero,
            current,
            source,
        )


def _broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    result: list[int] = []
    for dimensions in zip(*(
        (1,) * (max(map(len, shapes), default=0) - len(shape)) + tuple(shape)
        for shape in shapes
    )):
        concrete = {int(value) for value in dimensions if int(value) != 1}
        if len(concrete) > 1:
            raise ProcessGraphAutogradError(
                f"incompatible tensor broadcast shapes: {shapes!r}"
            )
        result.append(next(iter(concrete), 1))
    return tuple(result)


def _annotate_numeric_metadata(graph: ProcessGraph) -> None:
    """Propagate AbstractTensor shape/dtype facts through generated nodes."""

    for node_id in nx.topological_sort(graph.G):
        data = graph.G.nodes[node_id]
        tensor = dict(data.get("tensor") or {})
        parents = _parents(graph, int(node_id))
        parent_tensors = [graph.G.nodes[parent].get("tensor") or {} for parent in parents]
        parent_shapes = [tuple(item.get("shape") or ()) for item in parent_tensors]
        dtype = tensor.get("dtype") or next(
            (item.get("dtype") for item in parent_tensors if item.get("dtype")),
            "float64",
        )
        op = _operation(data)
        shape = tuple(tensor.get("shape") or ())
        attrs = data.get("attributes") or {}
        if op == "const":
            shape = tuple(tensor.get("shape") or ())
        elif op == "reshape":
            shape = tuple(attrs.get("shape") or ())
        elif op in {"transpose", "swapaxes", "permute"} and parent_shapes:
            shape = parent_shapes[0]
            if len(shape) >= 2:
                shape = (*shape[:-2], shape[-1], shape[-2])
        elif op in {"matmul", "mm"} and len(parent_shapes) == 2:
            left, right = parent_shapes
            if len(left) < 2 or len(right) < 2:
                raise ProcessGraphAutogradError(
                    f"matmul node {node_id} requires rank-two tensor metadata"
                )
            batch = _broadcast_shape(left[:-2], right[:-2])
            shape = (*batch, left[-2], right[-1])
        elif op in {"sum", "mean"} and parent_shapes:
            source_shape = parent_shapes[0]
            axis = attrs.get("axis", attrs.get("dim"))
            axes = (
                tuple(range(len(source_shape))) if axis is None
                else tuple(int(item) % len(source_shape) for item in axis)
                if isinstance(axis, (tuple, list))
                else (int(axis) % len(source_shape),)
            )
            if bool(attrs.get("keepdim", False)):
                shape = tuple(
                    1 if index in axes else extent
                    for index, extent in enumerate(source_shape)
                )
            else:
                shape = tuple(
                    extent for index, extent in enumerate(source_shape)
                    if index not in axes
                )
        elif parent_shapes and op not in {"input", "return"}:
            shape = _broadcast_shape(*parent_shapes)
        data["tensor"] = {
            **tensor,
            "shape": tuple(shape),
            "dtype": dtype,
        }


def differentiate_process_graph(
    forward: ProcessGraph,
    *,
    outputs: Iterable[int] | None = None,
    wrt: Iterable[int] | None = None,
) -> ProcessGraphAdjoint:
    """Construct a first-class parametric backward ``ProcessGraph``.

    ``outputs`` names differentiated forward values (``return`` nodes are
    resolved to their value).  Each output receives an explicit upstream
    gradient input. ``wrt`` defaults to every forward ``input`` node.

    The pass is fail-closed.  It accepts no cycles and no reachable operation
    without a graph-native rule.  This prevents an observed branch, a Python
    function object, or a missing derivative from masquerading as compiled
    logical backward code.
    """

    if not nx.is_directed_acyclic_graph(forward.G):
        raise ProcessGraphAutogradError(
            "logical/cyclic ProcessGraph differentiation requires the "
            "planner-owned ControlProgram adjoint; refusing tape-like unrolling"
        )

    raw_outputs = tuple(outputs or getattr(forward, "roots", ()) or ())
    if not raw_outputs:
        raw_outputs = tuple(
            int(node_id) for node_id in forward.G
            if forward.G.out_degree(node_id) == 0
        )
    output_ids = tuple(_returned_value(forward, int(node)) for node in raw_outputs)
    if not output_ids:
        raise ProcessGraphAutogradError("forward graph has no differentiated output")

    wrt_ids = tuple(
        int(node_id) for node_id in (
            wrt if wrt is not None else (
                node for node, data in forward.G.nodes(data=True)
                if _operation(data) == "input"
            )
        )
    )
    unknown = (set(output_ids) | set(wrt_ids)) - set(forward.G)
    if unknown:
        raise ProcessGraphAutogradError(
            "unknown ProcessGraph value ids: " + ", ".join(map(str, sorted(unknown)))
        )

    reachable: set[int] = set()
    for output_id in output_ids:
        reachable |= nx.ancestors(forward.G, output_id) | {output_id}
    irrelevant_wrt = tuple(node for node in wrt_ids if node not in reachable)
    if irrelevant_wrt:
        raise ProcessGraphAutogradError(
            "requested gradients are disconnected from the outputs: "
            + ", ".join(map(str, irrelevant_wrt))
        )

    builder = _AdjointBuilder(forward)
    contributions: dict[int, list[int]] = {}
    seeds: dict[int, int] = {}
    gradients: dict[int, int] = {}
    registry_rules: dict[int, str] = {}

    for output_id in output_ids:
        seed = builder.input(
            f"grad_seed_{output_id}",
            kind="gradient_seed",
            source_forward_id=output_id,
        )
        seeds[output_id] = seed
        contributions.setdefault(output_id, []).append(seed)

    def total_gradient(forward_id: int) -> int | None:
        terms = contributions.get(forward_id, ())
        if not terms:
            return None
        result = terms[0]
        for term in terms[1:]:
            result = builder.binary("add", result, term, forward_id)
        gradients[forward_id] = result
        return result

    def contribute(parent_id: int, value_id: int) -> None:
        contributions.setdefault(int(parent_id), []).append(int(value_id))

    unsupported: list[tuple[int, str]] = []
    order = tuple(nx.topological_sort(forward.G.subgraph(reachable)))
    for node_id in reversed(order):
        gradient = total_gradient(int(node_id))
        if gradient is None:
            continue
        data = forward.G.nodes[node_id]
        op = _operation(data)
        parents = _parents(forward, int(node_id))
        if op in {"input", "const"}:
            continue
        registry_op = {
            "truediv": "div",
            "mm": "matmul",
            "select": "where",
            "flatten": "reshape",
            "identity": "clone",
        }.get(op, op)
        if registry_op not in BACKWARD_RULES:
            unsupported.append((int(node_id), op))
            continue
        registry_rules[int(node_id)] = registry_op
        if registry_op in {"clone", "reshape"} and len(parents) == 1:
            contribute(
                parents[0],
                builder.reduce_to_shape(gradient, parents[0], int(node_id)),
            )
            continue
        if registry_op in {"add", "sub", "mul", "div"} and len(parents) == 2:
            left, right = parents
            if registry_op == "add":
                left_term, right_term = gradient, gradient
            elif registry_op == "sub":
                left_term = gradient
                right_term = builder.unary("neg", gradient, int(node_id))
            elif registry_op == "mul":
                left_term = builder.binary(
                    "mul", gradient, builder.saved_value(right), int(node_id)
                )
                right_term = builder.binary(
                    "mul", gradient, builder.saved_value(left), int(node_id)
                )
            else:
                saved_left = builder.saved_value(left)
                saved_right = builder.saved_value(right)
                left_term = builder.binary(
                    "truediv", gradient, saved_right, int(node_id)
                )
                denominator = builder.binary(
                    "mul", saved_right, saved_right, int(node_id)
                )
                numerator = builder.binary(
                    "mul", gradient, saved_left, int(node_id)
                )
                right_term = builder.unary(
                    "neg",
                    builder.binary(
                        "truediv", numerator, denominator, int(node_id)
                    ),
                    int(node_id),
                )
            contribute(
                left,
                builder.reduce_to_shape(left_term, left, int(node_id)),
            )
            contribute(
                right,
                builder.reduce_to_shape(right_term, right, int(node_id)),
            )
            continue
        if registry_op == "neg" and len(parents) == 1:
            term = builder.unary("neg", gradient, int(node_id))
            contribute(
                parents[0],
                builder.reduce_to_shape(term, parents[0], int(node_id)),
            )
            continue
        if registry_op in {"sin", "cos", "exp", "log", "tanh"} and len(parents) == 1:
            parent = parents[0]
            saved_parent = builder.saved_value(parent)
            if registry_op == "sin":
                local = builder.unary("cos", saved_parent, int(node_id))
            elif registry_op == "cos":
                local = builder.unary(
                    "neg",
                    builder.unary("sin", saved_parent, int(node_id)),
                    int(node_id),
                )
            elif registry_op == "exp":
                local = builder.saved_value(int(node_id))
            elif registry_op == "log":
                term = builder.binary(
                    "truediv", gradient, saved_parent, int(node_id)
                )
                contribute(
                    parent,
                    builder.reduce_to_shape(term, parent, int(node_id)),
                )
                continue
            else:
                saved_result = builder.saved_value(int(node_id))
                square = builder.binary(
                    "mul", saved_result, saved_result, int(node_id)
                )
                one = builder.add(
                    "const",
                    label="1.0",
                    attributes={"values": 1.0},
                    source_forward_id=int(node_id),
                )
                builder.backward.G.nodes[one]["constant"] = 1.0
                local = builder.binary("sub", one, square, int(node_id))
            term = builder.binary("mul", gradient, local, int(node_id))
            contribute(
                parent,
                builder.reduce_to_shape(term, parent, int(node_id)),
            )
            continue
        if registry_op in {"sum", "mean"} and len(parents) == 1:
            parent = parents[0]
            reduction_attrs = {
                key: copy.deepcopy(value)
                for key, value in (data.get("attributes") or {}).items()
                if key in {"axis", "dim", "keepdim"}
            }
            expanded = builder.expand_reduction(
                gradient,
                parent,
                int(node_id),
                reduction_attrs,
            )
            if registry_op == "mean":
                target_shape = tuple(
                    (forward.G.nodes[parent].get("tensor") or {}).get("shape")
                    or ()
                )
                axis = reduction_attrs.get("axis", reduction_attrs.get("dim"))
                axes = (
                    tuple(range(len(target_shape))) if axis is None
                    else tuple(int(item) for item in axis)
                    if isinstance(axis, (tuple, list))
                    else (int(axis),)
                )
                count_value = float(
                    __import__("math").prod(
                        target_shape[item % len(target_shape)] for item in axes
                    )
                )
                count = builder.add(
                    "const",
                    label=repr(count_value),
                    attributes={"values": count_value},
                    source_forward_id=int(node_id),
                )
                builder.backward.G.nodes[count]["constant"] = count_value
                expanded = builder.binary(
                    "truediv", expanded, count, int(node_id)
                )
            contribute(parent, expanded)
            continue
        if registry_op == "matmul" and len(parents) == 2:
            left, right = parents
            saved_left = builder.saved_value(left)
            saved_right = builder.saved_value(right)
            right_t = builder.add(
                "transpose",
                ((saved_right, "operand"),),
                attributes={"dim0": -2, "dim1": -1},
                source_forward_id=int(node_id),
            )
            left_t = builder.add(
                "transpose",
                ((saved_left, "operand"),),
                attributes={"dim0": -2, "dim1": -1},
                source_forward_id=int(node_id),
            )
            left_term = builder.binary(
                "matmul", gradient, right_t, int(node_id)
            )
            right_term = builder.binary(
                "matmul", left_t, gradient, int(node_id)
            )
            contribute(
                left,
                builder.reduce_to_shape(
                    left_term,
                    left,
                    int(node_id),
                    gradient_shape=tuple(
                        (forward.G.nodes[left].get("tensor") or {}).get("shape")
                        or ()
                    ),
                ),
            )
            contribute(
                right,
                builder.reduce_to_shape(
                    right_term,
                    right,
                    int(node_id),
                    gradient_shape=tuple(
                        (forward.G.nodes[right].get("tensor") or {}).get("shape")
                        or ()
                    ),
                ),
            )
            continue
        if registry_op == "where" and len(parents) == 3:
            condition, if_true, if_false = parents
            saved_condition = builder.saved_value(condition)
            zero = builder.add(
                "const",
                label="0.0",
                attributes={"values": 0.0},
                source_forward_id=int(node_id),
            )
            builder.backward.G.nodes[zero]["constant"] = 0.0
            true_term = builder.add(
                "where",
                (
                    (saved_condition, "condition"),
                    (gradient, "if_true"),
                    (zero, "if_false"),
                ),
                source_forward_id=int(node_id),
            )
            false_term = builder.add(
                "where",
                (
                    (saved_condition, "condition"),
                    (zero, "if_true"),
                    (gradient, "if_false"),
                ),
                source_forward_id=int(node_id),
            )
            contribute(
                if_true,
                builder.reduce_to_shape(true_term, if_true, int(node_id)),
            )
            contribute(
                if_false,
                builder.reduce_to_shape(false_term, if_false, int(node_id)),
            )
            continue
        unsupported.append((int(node_id), op))

    if unsupported:
        detail = ", ".join(f"{node}:{op or '?'}" for node, op in unsupported)
        raise ProcessGraphAutogradError(
            "ProcessGraph has no graph-native adjoint rule for " + detail
        )

    for wrt_id in wrt_ids:
        if wrt_id not in gradients:
            total_gradient(wrt_id)
    missing = tuple(node for node in wrt_ids if node not in gradients)
    if missing:
        raise ProcessGraphAutogradError(
            "backward graph produced no gradient for "
            + ", ".join(map(str, missing))
        )

    builder.backward.roots = [gradients[node] for node in wrt_ids]
    # ``ProcessGraph.compute_levels`` is an execution scheduler and may
    # materialize Store nodes at roots. Differentiation is still constructing
    # semantic IR here, so record dependency levels without mutating it into a
    # deployment graph.
    builder.backward.levels = {
        int(node_id): int(level)
        for level, generation in enumerate(
            nx.topological_generations(builder.backward.G)
        )
        for node_id in generation
    }
    builder.backward.G.graph.update({
        "graph_kind": "parametric_backward",
        "schema_version": 1,
        "forward_output_ids": output_ids,
        "wrt_value_ids": wrt_ids,
        "gradient_outputs": {
            str(forward_id): gradients[forward_id] for forward_id in wrt_ids
        },
        "saved_forward_values": dict(builder.saved),
        "gradient_seeds": dict(seeds),
        "python_backward_callbacks": False,
        "backward_rule_registry": "src.common.tensors.backward_registry.BACKWARD_RULES",
        "backward_rule_nodes": dict(registry_rules),
    })
    execution_contract = forward.G.graph.get("execution_contract")
    if isinstance(execution_contract, Mapping):
        builder.backward.G.graph["execution_contract"] = copy.deepcopy(
            dict(execution_contract)
        )
        builder.backward.G.graph["deployment_role"] = (
            "opportunistic_numeric_dispatch"
            if execution_contract.get("native_lowering") == "opportunistic"
            else "required_numeric_program"
        )
    _annotate_numeric_metadata(builder.backward)
    return ProcessGraphAdjoint(
        forward=forward,
        backward=builder.backward,
        output_value_ids=output_ids,
        wrt_value_ids=wrt_ids,
        seed_value_ids=dict(seeds),
        saved_value_ids=dict(builder.saved),
        gradient_value_ids={node: gradients[node] for node in wrt_ids},
    )


def fuse_forward_loss_backward(
    adjoint: ProcessGraphAdjoint,
    *,
    unit_loss_seed: bool = True,
) -> ForwardLossBackwardMotion:
    """Compose a ProcessGraph-derived adjoint with its forward graph.

    Saved-value inputs disappear: each is wired directly to its authoritative
    forward producer. A scalar loss may use a graph constant seed of one; set
    ``unit_loss_seed=False`` to retain explicit upstream-gradient ABI inputs.
    The optimizer is intentionally absent from this graph.
    """

    forward = adjoint.forward
    backward = adjoint.backward
    motion = ProcessGraph(materialize_memory=False)
    forward_keep: set[int] = set()
    for loss_id in adjoint.output_value_ids:
        forward_keep |= nx.ancestors(forward.G, loss_id) | {loss_id}

    for node_id in nx.topological_sort(forward.G.subgraph(forward_keep)):
        data = copy.deepcopy(dict(forward.G.nodes[node_id]))
        data["parents"] = [
            (int(parent), str(role))
            for parent, role in data.get("parents", ())
            if int(parent) in forward_keep
        ]
        data["children"] = []
        attrs = dict(data.get("attributes") or {})
        attrs.setdefault("training_motion_phase", "forward")
        data["attributes"] = attrs
        data["extra_args"] = copy.deepcopy(attrs)
        motion.G.add_node(int(node_id), **data)
    for left, right, edge in forward.G.subgraph(forward_keep).edges(data=True):
        motion.G.add_edge(int(left), int(right), **copy.deepcopy(dict(edge)))
        role = str(edge.get("role") or "value")
        motion.G.nodes[int(left)]["children"].append((int(right), role))

    next_id = max((int(node) for node in motion.G), default=-1) + 1
    remap: dict[int, int] = {}
    seed_ids: dict[int, int] = {}
    saved_by_backward = {
        int(backward_id): int(forward_id)
        for forward_id, backward_id in adjoint.saved_value_ids.items()
    }
    output_by_seed = {
        int(seed_id): int(output_id)
        for output_id, seed_id in adjoint.seed_value_ids.items()
    }
    for node_id in nx.topological_sort(backward.G):
        node_id = int(node_id)
        if node_id in saved_by_backward:
            remap[node_id] = saved_by_backward[node_id]
            continue
        new_id = next_id
        next_id += 1
        remap[node_id] = new_id
        data = copy.deepcopy(dict(backward.G.nodes[node_id]))
        attrs = dict(data.get("attributes") or {})
        attrs["training_motion_phase"] = "backward"
        if node_id in output_by_seed and unit_loss_seed:
            data.update({
                "op": "const",
                "type": "const",
                "label": "1.0",
                "constant": 1.0,
            })
            attrs = {
                "values": 1.0,
                "gradient_seed_for": output_by_seed[node_id],
                "training_motion_phase": "backward",
            }
            data["parents"] = []
        else:
            data["parents"] = [
                (remap[int(parent)], str(role))
                for parent, role in data.get("parents", ())
            ]
        data["children"] = []
        data["attributes"] = attrs
        data["extra_args"] = copy.deepcopy(attrs)
        motion.G.add_node(new_id, **data)
        if node_id in output_by_seed:
            seed_ids[output_by_seed[node_id]] = new_id

    for node_id in nx.topological_sort(backward.G):
        node_id = int(node_id)
        if node_id in saved_by_backward:
            continue
        new_id = remap[node_id]
        for parent, role in backward.G.nodes[node_id].get("parents", ()):
            new_parent = remap[int(parent)]
            motion.G.add_edge(new_parent, new_id, role=str(role))
            motion.G.nodes[new_parent]["children"].append((new_id, str(role)))

    gradient_ids = {
        int(forward_id): remap[int(backward_id)]
        for forward_id, backward_id in adjoint.gradient_value_ids.items()
    }
    motion.roots = [*adjoint.output_value_ids, *gradient_ids.values()]
    motion.levels = {
        int(node_id): int(level)
        for level, generation in enumerate(nx.topological_generations(motion.G))
        for node_id in generation
    }
    execution_contract = forward.G.graph.get("execution_contract")
    motion.G.graph.update({
        "graph_kind": "forward_loss_backward_motion",
        "schema_version": 1,
        "loss_outputs": tuple(adjoint.output_value_ids),
        "gradient_outputs": dict(gradient_ids),
        "gradient_seeds": dict(seed_ids),
        "unit_loss_seed": bool(unit_loss_seed),
        "optimizer_included": False,
        "backward_rule_registry": backward.G.graph.get(
            "backward_rule_registry"
        ),
        "backward_rule_nodes": copy.deepcopy(
            backward.G.graph.get("backward_rule_nodes", {})
        ),
    })
    if isinstance(execution_contract, Mapping):
        motion.G.graph["execution_contract"] = copy.deepcopy(
            dict(execution_contract)
        )
    _annotate_numeric_metadata(motion)
    return ForwardLossBackwardMotion(
        graph=motion,
        loss_value_ids=tuple(adjoint.output_value_ids),
        gradient_value_ids=gradient_ids,
        seed_value_ids=seed_ids,
    )


def lower_training_motion_to_repository_ssa(
    motion: ForwardLossBackwardMotion,
    *,
    function_name: str = "forward_loss_backward",
    tensor_ssa_reference: Any | None = None,
) -> TrainingMotionSSALowering:
    """Lower the semantic training motion directly to repository SSA.

    This path never constructs a ``FusedProgram``. Inputs become SSA function
    arguments; every other numeric ProcessGraph node becomes one canonical
    tensor instruction before the shared tensor-SSA lowering expands it into
    authored kernel calls.
    """

    from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from ..transmogrifier.ssa import (
        BasicBlock,
        Function,
        IRModule,
        Instr,
        SSAValue,
    )
    from ..transmogrifier.ssa_registry import Handler
    from .tensor_ssa_lowering import lower_tensor_calls_to_repository_ssa

    graph = motion.graph
    _annotate_numeric_metadata(graph)
    values: dict[int, SSAValue] = {}

    def value(node_id: int) -> SSAValue:
        node_id = int(node_id)
        if node_id not in values:
            tensor = graph.G.nodes[node_id].get("tensor") or {}
            values[node_id] = SSAValue(
                node_id,
                dtype=tensor.get("dtype") or "float64",
                shape=tuple(tensor.get("shape") or ()),
                device=tensor.get("device"),
            )
        return values[node_id]

    args = [
        value(int(node_id))
        for node_id in nx.topological_sort(graph.G)
        if _operation(graph.G.nodes[node_id]) == "input"
    ]
    instructions = []
    for node_id in nx.topological_sort(graph.G):
        node_id = int(node_id)
        data = graph.G.nodes[node_id]
        op = _operation(data)
        if op == "input":
            continue
        parents = _parents(graph, node_id)
        attrs = copy.deepcopy(dict(data.get("attributes") or {}))
        if op == "const":
            opcode = Handler.Const.value
            attrs.setdefault("constant", data.get("constant"))
            attrs.setdefault("values", data.get("constant"))
        else:
            opcode = op
            attrs["tensor_operation"] = op
        instructions.append(Instr(
            opcode,
            [value(parent) for parent in parents],
            value(node_id),
            arg_roles=[
                str(role) for _parent, role in data.get("parents", ())
            ],
            attributes=attrs,
            source_span=data.get("source_span"),
        ))

    outputs = {
        **{f"loss_{index}": int(value_id)
           for index, value_id in enumerate(motion.loss_value_ids)},
        **{f"grad_{forward_id}": int(value_id)
           for forward_id, value_id in motion.gradient_value_ids.items()},
    }
    # Outputs are ordinary return operands, not metadata-only labels.  This
    # keeps their identities live through tensor expansion and gives native
    # emitters an explicit result ABI (including scalar loss registers).
    instructions.append(Instr(
        Handler.Ret.value,
        [value(value_id) for value_id in outputs.values()],
        None,
        arg_roles=list(outputs),
        attributes={"training_motion_outputs": tuple(outputs)},
    ))

    function = Function(
        function_name,
        args,
        {"entry": BasicBlock("entry", instructions)},
        metadata={
            "graph_kind": graph.G.graph.get("graph_kind"),
            "loss_outputs": tuple(motion.loss_value_ids),
            "gradient_outputs": dict(motion.gradient_value_ids),
            "optimizer_included": False,
        },
    )
    module = IRModule({function_name: function})
    reference = tensor_ssa_reference or c_backend_repository_ssa_reference()
    shortfalls = lower_tensor_calls_to_repository_ssa(module, reference)
    lowered_return = next(
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op in {Handler.Ret.value, "ret", "Return", "return"}
        and instruction.attributes.get("training_motion_outputs")
    )
    outputs = {
        str(role): int(argument.id)
        for role, argument in zip(lowered_return.arg_roles, lowered_return.args)
    }
    return TrainingMotionSSALowering(
        module=module,
        function_name=function_name,
        outputs=outputs,
        shortfalls=tuple(shortfalls),
    )


__all__ = [
    "ForwardLossBackwardMotion",
    "TrainingMotionSSALowering",
    "ProcessGraphAdjoint",
    "ProcessGraphAutogradError",
    "differentiate_process_graph",
    "fuse_forward_loss_backward",
    "lower_training_motion_to_repository_ssa",
]
