"""Callsite descriptor application: writing exact callsite tensor and aggregate descriptors onto callee graphs, aliasing projections to their members and expanding specialized unbroadcast identities."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from .process_graph_value_ids import next_process_value_id
from ..common.tensors.topological_reducer import _set_operands
from .concordance_declarations import (
    AGGREGATE_FORMAL_MEMBERS as _AGGREGATE_FORMAL_MEMBERS,
    PROJECTION_TO_LEAF as _PROJECTION_TO_LEAF,
    UNBROADCAST_CHAIN as _UNBROADCAST_CHAIN,
)
from .planner_concordance_posts import _post_identity_table_mutation
from .tensor_descriptor_query import (
    _invalidate_tensor_descriptor_dependents,
)
from .structural_fold import (
    _follow_declared_value_source,
    _retarget_all_cached_value_ids,
)


def _apply_callsite_tensor_descriptors(
    graph: Any,
    descriptors: Mapping[str, Mapping[str, Any]],
) -> None:
    graph.G.graph["callsite_tensor_descriptor_names"] = tuple(sorted({
        *map(
            str,
            graph.G.graph.get("callsite_tensor_descriptor_names") or (),
        ),
        *map(str, descriptors),
    }))
    authoritative = dict(
        graph.G.graph.get("planner_tensor_descriptors") or {}
    )
    authoritative.update({
        str(name): copy.deepcopy(dict(descriptor))
        for name, descriptor in descriptors.items()
    })
    graph.G.graph["planner_tensor_descriptors"] = authoritative
    identities = graph.G.graph.get("identity_table") or {}
    changed_inputs = set()
    for name, descriptor in descriptors.items():
        candidates = tuple(dict.fromkeys((
            *tuple(identities.get(str(name), ())),
            *(
                int(node_id)
                for node_id, data in graph.G.nodes(data=True)
                if data.get("type") == "Input"
                and (data.get("attributes") or {}).get("binding_name") == name
            ),
        )))
        for node_id in candidates:
            if int(node_id) not in graph.G:
                continue
            data = graph.G.nodes[int(node_id)]
            if data.get("type") != "Input":
                continue
            if data.get("tensor") != dict(descriptor):
                changed_inputs.add(int(node_id))
            data["tensor"] = copy.deepcopy(dict(descriptor))

    # Cached intermediate shapes were derived before these exact argument
    # descriptors arrived. Invalidate their dependency closure, including the
    # concordance proof that would otherwise immediately repopulate the cache.
    _invalidate_tensor_descriptor_dependents(
        graph, changed_inputs, "callsite-input-specialization-changed",
    )


def _alias_projection_to_member(
    graph: Any, projection: int, leaf_id: int,
) -> None:
    """Give one authored ``name[index]`` projection its member's identity."""

    projection = int(projection)
    leaf_id = int(leaf_id)
    if projection not in graph.G or leaf_id not in graph.G:
        return
    projection_data = graph.G.nodes[projection]
    for successor in tuple(graph.G.successors(projection)):
        successor_data = graph.G.nodes[int(successor)]
        _set_operands(graph, int(successor), [
            (
                leaf_id if int(parent) == projection else int(parent),
                str(role),
            )
            for parent, role in successor_data.get("parents") or ()
        ], cause=_PROJECTION_TO_LEAF, same={projection: leaf_id})
        _follow_declared_value_source(successor_data, projection, leaf_id)
    index_ids = tuple(
        int(parent)
        for parent, role in projection_data.get("parents") or ()
        if str(role) in {"index", "slice", "subscript"}
    )
    # The projection's own positions retire (its parents' ``children``
    # entries and edges with them) before the node is removed below.
    _set_operands(graph, projection, [], cause=_PROJECTION_TO_LEAF)
    # A loop that carries this projection (``a, b = history`` bound as its
    # initial) caches the projection id without being its successor; move
    # every cached copy before the node is gone.
    _retarget_all_cached_value_ids(graph, projection, leaf_id)
    graph.roots = [leaf_id if int(root) == projection else int(root)
                   for root in graph.roots]
    graph.G.graph["return_value_nodes"] = {
        site: leaf_id if value == projection else value
        for site, value in (graph.G.graph.get("return_value_nodes") or {}).items()
    }
    graph.G.graph["return_slot_values"] = {
        site: tuple(leaf_id if value == projection else value for value in values)
        for site, values in (graph.G.graph.get("return_slot_values") or {}).items()
    }
    graph.G.remove_node(projection)
    for index_id in index_ids:
        if (
            index_id in graph.G
            and graph.G.out_degree(index_id) == 0
            and graph.G.nodes[index_id].get("type") in {
                "Constant", "Const", "const",
            }
        ):
            graph.G.remove_node(index_id)
    identities = graph.G.graph.get("identity_table") or {}
    # The projection's ``name_binding`` versions (and the return slots that
    # named it) revise to the member's identity (plan 80, A2.7).
    from .concordance_declarations import PLANNER_PROJECTION_ALIAS

    _post_identity_table_mutation(
        graph, aliases={projection: leaf_id}, stage=PLANNER_PROJECTION_ALIAS,
    )
    graph.G.graph["identity_table"] = {
        str(key): tuple(
            leaf_id if int(value_id) == projection else int(value_id)
            for value_id in history
        )
        for key, history in identities.items()
    }


def _apply_callsite_aggregate_descriptors(
    graph: Any,
    descriptors: Mapping[str, tuple[Any, ...]],
) -> None:
    """Bind an aggregate formal to its exact authored projection identities."""

    identities = graph.G.graph.get("identity_table") or {}
    for name, root_members in descriptors.items():
        inputs = tuple(dict.fromkeys((
            *tuple(identities.get(str(name), ())),
            *(
                int(node_id)
                for node_id, data in graph.G.nodes(data=True)
                if data.get("type") == "Input"
                and (data.get("attributes") or {}).get("binding_name") == name
            ),
        )))
        inputs = tuple(
            int(node_id) for node_id in inputs
            if int(node_id) in graph.G
            and graph.G.nodes[int(node_id)].get("type") == "Input"
        )
        if len(inputs) != 1:
            raise ValueError(
                f"aggregate parameter {name!r} does not have one input identity: "
                f"{inputs!r}"
            )
        def materialize(
            input_id: int,
            binding_path: str,
            members: tuple[Any, ...],
            *,
            root_binding: str,
            aggregate_path: tuple[int, ...] = (),
        ) -> None:
            """Materialize one level, retaining nested member ledgers."""

            projections: dict[int, list[int]] = {}
            for node_id, data in tuple(graph.G.nodes(data=True)):
                if str(
                    data.get("op") or data.get("type") or ""
                ).casefold() not in {"indexed", "getitem", "subscript"}:
                    continue
                base = tuple(
                    int(parent)
                    for parent, role in data.get("parents") or ()
                    if str(role) in {"base", "value", "object"}
                )
                if base != (int(input_id),):
                    continue
                index_ids = tuple(
                    int(parent)
                    for parent, role in data.get("parents") or ()
                    if str(role) in {"index", "slice", "subscript"}
                )
                if len(index_ids) != 1 or index_ids[0] not in graph.G:
                    continue
                index_data = graph.G.nodes[index_ids[0]]
                index = index_data.get("constant")
                if index is None:
                    index = (index_data.get("attributes") or {}).get("value")
                if (
                    not isinstance(index, int) or isinstance(index, bool)
                    or not members
                ):
                    continue
                normalized = int(index) % len(members)
                projections.setdefault(normalized, []).append(int(node_id))

            leaves = []
            nested = []
            for index, descriptor in enumerate(members):
                leaf_id = next_process_value_id(graph)
                member_path = f"{binding_path}[{index}]"
                full_path = (*aggregate_path, int(index))
                is_nested = isinstance(descriptor, tuple)
                graph.G.add_node(
                    leaf_id,
                    type="Input", op="input", label=member_path,
                    expr_obj=None, value_id=leaf_id,
                    parents=[], children=[],
                    attributes={
                        "binding_name": member_path,
                        "binding_kind": "parameter",
                        "aggregate_parent_binding": str(root_binding),
                        "aggregate_path": tuple(full_path),
                        "aggregate_index": int(index),
                    },
                    tensor=(
                        copy.deepcopy(dict(descriptor))
                        if isinstance(descriptor, Mapping) else {}
                    ),
                )
                leaves.append(leaf_id)
                for projection in projections.get(index, ()):
                    _alias_projection_to_member(graph, projection, leaf_id)
                if is_nested:
                    nested.append((
                        leaf_id, member_path, tuple(descriptor), full_path,
                    ))

            leaves = tuple(leaves)
            input_attributes = dict(
                graph.G.nodes[input_id].get("attributes") or {}
            )
            input_attributes.update({
                "producer_kind": "aggregate",
                "aggregate_kind": "tuple",
                "aggregate_leaf_value_ids": leaves,
                "tensor_output_descriptors": copy.deepcopy(tuple(members)),
                "sequence_key_columns": (),
                "sequence_column_count": 1,
                "sequence_writable": False,
            })
            formal = graph.G.nodes[input_id]
            formal["attributes"] = input_attributes
            formal["type"] = "Tuple"
            formal["op"] = None
            formal["tensor"] = {}
            _set_operands(graph, int(input_id), [
                (int(leaf), "elts") for leaf in leaves
            ], cause=_AGGREGATE_FORMAL_MEMBERS)
            for leaf_id, member_path, nested_members, nested_path in nested:
                materialize(
                    leaf_id,
                    member_path,
                    nested_members,
                    root_binding=root_binding,
                    aggregate_path=tuple(nested_path),
                )

        materialize(
            int(inputs[0]),
            str(name),
            tuple(root_members),
            root_binding=str(name),
        )


def _expand_specialized_unbroadcast_identity(graph: Any) -> bool:
    """Lower the authored ``unbroadcast`` helper to canonical tensor nodes.

    The helper's Python loop is only structural shape logic.  Once a callsite
    supplies both shapes, retain its exact BACKWARD_RULES meaning as a finite
    sequence of existing ``sum`` and ``reshape`` operators instead of asking
    the control planner to carry a tensor through a Python ``for`` loop.
    """

    if str(graph.G.graph.get("function_name") or "") != "unbroadcast":
        return False
    descriptors = graph.G.graph.get("planner_tensor_descriptors") or {}
    specializations = graph.G.graph.get("planner_specializations") or {}
    source_descriptor = descriptors.get("G")
    target_shape = specializations.get("target_shape")
    if source_descriptor is None or not isinstance(target_shape, (tuple, list)):
        return False
    source_shape = tuple(map(int, source_descriptor.get("shape") or ()))
    target_shape = tuple(map(int, target_shape))
    input_id = next((
        int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        if data.get("type") == "Input"
        and (data.get("attributes") or {}).get("binding_name") == "G"
    ), None)
    if input_id is None:
        return False
    metadata = copy.deepcopy(dict(graph.G.graph))
    input_data = copy.deepcopy(dict(graph.G.nodes[input_id]))
    graph.G.clear()
    graph.G.graph.update(metadata)
    input_data["parents"] = []
    input_data["children"] = []
    input_data["tensor"] = copy.deepcopy(dict(source_descriptor))
    graph.G.add_node(input_id, **input_data)
    next_id = input_id + 1
    current = input_id
    current_shape = list(source_shape)

    def append(op: str, attributes: Mapping[str, Any], shape: tuple[int, ...]):
        nonlocal next_id, current
        node_id = next_id
        next_id += 1
        role = "operand"
        graph.G.add_node(
            node_id, op=op, type=op, label=op,
            parents=[], children=[],
            attributes={
                **dict(attributes),
                "tensor_candidate": op,
                "authored_identity": "backward_registry.unbroadcast",
            },
            extra_args=dict(attributes),
            tensor={
                "shape": tuple(shape),
                "dtype": str(source_descriptor.get("dtype") or "float64"),
            },
            control={}, constant=None, expr_obj=None, store_id=None,
        )
        _set_operands(graph, node_id, [(current, role)], cause=_UNBROADCAST_CHAIN)
        current = node_id

    while len(current_shape) > len(target_shape):
        current_shape.pop(0)
        append(
            "sum", {"dim": 0, "keepdim": False}, tuple(current_shape),
        )
    for axis, (actual, target) in enumerate(zip(current_shape, target_shape)):
        if target == 1 and actual != 1:
            current_shape[axis] = 1
            append(
                "sum", {"dim": axis, "keepdim": True},
                tuple(current_shape),
            )
    append("reshape", {"shape": target_shape}, target_shape)
    graph.roots = [current]
    identities = copy.deepcopy(metadata.get("identity_table") or {})
    identities["G"] = (input_id,)
    identities["result_0"] = (current,)
    graph.G.graph.update({
        "identity_table": identities,
        "function_outputs": ("result_0",),
        "function_parameters": ("G",),
        "positional_parameters": ("G",),
        "keyword_only_parameters": (),
        "authored_identity_expansion": "backward_registry.unbroadcast",
    })
    return True
