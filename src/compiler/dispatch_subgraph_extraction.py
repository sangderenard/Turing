"""Dispatch subgraph extraction: the clean process subgraph of a dispatch region, its reduction digest and the structural region program built from a subgraph."""

from __future__ import annotations

import ast
import copy
import hashlib
import networkx as nx
from typing import Any

from ..common.tensors.topological_reducer import _set_operands
from .concordance_declarations import (
    DISPATCH_STORE as _DISPATCH_STORE_TRANSFORM,
    REGION_BOUNDARY_INPUT as _REGION_BOUNDARY_INPUT,
)
from .process_graph_fusion import (
    DispatchRegion,
    dispatch_region_to_fused_program,
    extract_clean_process_subgraph,
)
from ..common.tensors.accelerator_backends.c_primitive_program import CapturedFusedProgram
from .planner_concordance_posts import _post_deployment_region
from .tensor_descriptor_query import (
    descriptor_states_a_shape,
    _tensor_descriptor,
)
from .dispatch_metadata_classifier import _is_ast_metadata_node


def _subgraph_reduction_digest(subgraph: Any) -> str:
    """Content key for one region's structural lowering.

    ``_structural_region_program_from_subgraph`` is a pure function of the
    dispatch subgraph, so hashing the subgraph identifies the region program it
    will produce -- letting the incremental backup reuse an already-lowered
    region whenever its subgraph is unchanged.
    """

    from joblib.externals import cloudpickle

    graph_digest = hashlib.sha256(cloudpickle.dumps(subgraph)).hexdigest()
    # Region artifacts depend on the transcription contract as well as graph
    # topology.  Version that contract so a provenance/lowering fix cannot
    # keep resurrecting an older region program from the reduction cache.
    return hashlib.sha256(
        (
            "structural-region-v7-source-value-receipts|" + graph_digest
        ).encode(
            "utf-8"
        )
    ).hexdigest()


def _structural_region_program_from_subgraph(
    subgraph: Any,
) -> CapturedFusedProgram:
    """Build one region's numeric program from its dispatch subgraph structure.

    This is the value-free counterpart to a captured region program: the same
    ``dispatch_region_to_fused_program`` the JIT/tape path uses, driven by the
    subgraph the planner already isolated rather than by an observed tape. It
    is how the graph-only precompile (an unfed mutable parameter left symbolic,
    the exact case an autograd tape cannot observe) still produces real region
    programs -- transcribing the ops faithfully, no hand-synthesis. Layout/cast
    ops flow through under their own names for the backend to lower.
    """

    graph_data = subgraph.G.graph
    region = DispatchRegion(
        tuple(int(n) for n in graph_data.get("deployment_nodes", ())),
        tuple(int(n) for n in graph_data.get("deployment_inputs", ())),
        tuple(
            (f"value_{int(v)}", int(v))
            for v in graph_data.get("deployment_outputs", ())
        ),
        0.0,
    )
    # This is the handoff from the compiler's provenance backbone into the
    # backend-neutral numeric IR.  Several descriptors are proven recursively
    # by the concordance and therefore do not live in the source node's
    # shallow ``tensor`` snapshot.  Materialize those proven facts onto the
    # dispatch subgraph before transcription; otherwise a later region seam
    # forgets the shape even though the concordance still knows it.
    for value_id in (*region.input_ids, *region.node_ids):
        descriptor = _tensor_descriptor(subgraph, int(value_id))
        if descriptor is None:
            continue
        node = subgraph.G.nodes[int(value_id)]
        tensor = dict(node.get("tensor") or {})
        tensor.update({
            key: copy.deepcopy(descriptor[key])
            for key in ("shape", "dtype", "device", "rank")
            if descriptor.get(key) is not None
        })
        node["tensor"] = tensor
    program = dispatch_region_to_fused_program(subgraph, region)
    # Region-local integers are useful execution slots, but they are not the
    # identity a diagnostic or later compiler pass should have to remember.
    # Preserve the source graph's deterministic structural token chain for
    # every exposed value while the graph still owns that correlation.
    identity_tokens = dict(graph_data.get("ssa_identity_tokens") or {})
    exposed_values = {
        *map(int, program.feeds),
        *map(int, program.outputs.values()),
    }
    from .identity_concordance import shape_scope_of

    program.extras = {
        **dict(program.extras or {}),
        "source_function": str(graph_data.get("function_name") or ""),
        "source_shape_scope": shape_scope_of(graph_data),
        "source_value_ids": {
            int(value_id): int(value_id)
            for value_id in sorted(exposed_values)
        },
    }
    exposed_identity_tokens = {
        int(value_id): tuple(map(str, identity_tokens[value_id]))
        for value_id in sorted(exposed_values)
        if value_id in identity_tokens
    }
    if exposed_identity_tokens:
        program.extras = {
            **dict(program.extras or {}),
            "ssa_identity_tokens": exposed_identity_tokens,
        }
    structural_table_stores = []
    structural_sequences: dict[int, dict[str, Any]] = {}
    for node_id in graph_data.get("deployment_nodes", ()):
        data = subgraph.G.nodes[int(node_id)]
        operation = str(data.get("op") or data.get("type") or "").casefold()
        if operation != "indexedstore":
            continue
        by_role = {
            str(role): int(parent)
            for parent, role in data.get("parents") or ()
            if int(parent) in subgraph.G
        }
        base_id = by_role.get("base")
        key_id = by_role.get("index")
        value_id = by_role.get("value")
        if None in {base_id, key_id, value_id}:
            continue
        base_data = subgraph.G.nodes[int(base_id)]
        attributes = dict(base_data.get("attributes") or {})
        if attributes.get("aggregate_kind") != "dict":
            continue
        key_dtype = attributes.get("mapping_key_dtype")
        value_dtype = attributes.get("mapping_value_dtype")
        if key_dtype is None or value_dtype is None:
            # The mutation is structural, but its physical ABI is not stated.
            # Preserve that fact for diagnostics; never guess scalar widths.
            program.extras = {
                **dict(program.extras or {}),
                "structural_boundary_shortfalls": ({
                    "kind": "unresolved-resident-table-contract",
                    "sequence_value_id": int(base_id),
                    "effect_value_id": int(node_id),
                    "record_field": attributes.get("record_field"),
                },),
            }
            continue
        sequence_id = int(base_data.get("value_id", base_id))
        structural_sequences[sequence_id] = {
            "sequence_id": sequence_id,
            "policy": "unique",
            "column_count": int(
                attributes.get("sequence_column_count") or 2
            ),
            "writable": bool(attributes.get("sequence_writable", True)),
            "column_dtypes": [str(key_dtype), str(value_dtype)],
            "storage_identity": ".".join(map(
                str, attributes.get("record_field") or ("mapping", sequence_id),
            )),
            "value_record": attributes.get("mapping_value_record"),
            "value_optional": bool(
                attributes.get("mapping_value_optional", False)
            ),
        }
        structural_table_stores.append({
            "effect_value_id": int(data.get("value_id", node_id)),
            "key_value_id": int(
                subgraph.G.nodes[key_id].get("value_id", key_id)
            ),
            "stored_value_id": int(
                subgraph.G.nodes[value_id].get("value_id", value_id)
            ),
            "sequence_value_id": sequence_id,
        })
    if structural_table_stores:
        program.extras = {
            **dict(program.extras or {}),
            "structural_resident_table_contract": {
                "schema": "turing.structural-resident-table-integral.v1",
                "sequences": [
                    structural_sequences[key]
                    for key in sorted(structural_sequences)
                ],
                "stores": structural_table_stores,
            },
        }
    return CapturedFusedProgram(program, {})


def _dispatch_subgraph(
    graph: Any,
    node_ids: tuple[int, ...],
    *,
    required_outputs: frozenset[int] = frozenset(),
    inert_nodes: frozenset[int] = frozenset(),
    schedule_preference: str = "asap",
    region_index: int | None = None,
) -> Any:
    """Return the planned dispatch as an independent ProcessGraph subgraph.

    ``required_outputs`` names values the coordinator observes by identity
    rather than through a dataflow edge.  Such a value is a compartment
    boundary even when every graph-visible consumer lives inside this region,
    so it must be published instead of staying shader-local.

    ``inert_nodes`` names lookups nobody reads.  They are not consumers, so
    they never turn a shader-local intermediate into a published output.

    ``region_index`` is the planner's ordinal for this subgraph.  With it
    the region is posted on the book under the graph's planning scope
    (plan 80, A2.4): ``deployment_region`` DERIVED from every member's
    ``executable_node`` cell, one ``deployment_region_member`` per node
    (INPUT / NODE / OUTPUT) and one ``dispatch_store`` per Store node,
    NOVEL from the OUTPUT member it stores.  The five ``subgraph.G.graph``
    tuples stay as the read views the emitters consume.
    """

    selected = set(node_ids)
    semantic_parents = {
        node_id: {
            *graph.G.predecessors(node_id),
            *(
                parent
                for parent, _role in graph.G.nodes[node_id].get(
                    "parents", ()
                )
                if parent in graph.G
            ),
        }
        for node_id in graph.G
    }
    semantic_children = {
        node_id: set(graph.G.successors(node_id))
        for node_id in graph.G
    }
    for child, parents in semantic_parents.items():
        for parent in parents:
            semantic_children[parent].add(child)
    # Container literals are routing, not dispatches of their own.  When one
    # feeds a numerical region, inline the container and expose its leaves as
    # the region boundary so a captured tensor operation retains each feed's
    # independent identity.
    pending = list(selected)
    while pending:
        child = pending.pop()
        for parent in semantic_parents[child]:
            if parent in selected:
                continue
            expression = graph.G.nodes[parent].get("expr_obj")
            if isinstance(expression, (ast.Tuple, ast.List)):
                selected.add(parent)
                pending.append(parent)
    boundary = {
        parent
        for node_id in selected
        for parent in semantic_parents[node_id]
        if parent not in selected
        and not _is_ast_metadata_node(graph, parent)
    }
    included = selected | boundary
    subgraph = extract_clean_process_subgraph(graph, included)
    subgraph.python_bindings = dict(
        getattr(graph, "python_bindings", {}) or {}
    )
    for child in included:
        roles = {
            parent: role
            for parent, role in graph.G.nodes[child].get("parents", ())
            if parent in included
        }
        for parent in semantic_parents[child]:
            if parent in included and not subgraph.G.has_edge(parent, child):
                subgraph.G.add_edge(
                    parent,
                    child,
                    role=roles.get(parent, "dependency"),
                )

    carried_initials = {
        int(initial)
        for _, data in graph.G.nodes(data=True)
        for initial, _updated in (data.get("attributes") or {}).get(
            "loop_carried_bindings", {}
        ).values()
    }
    for node_id in boundary:
        data = subgraph.G.nodes[node_id]
        # Ask the concordance while the source graph still owns this value's
        # complete causal parents.  Turning a boundary into an Input below is
        # intentionally lossy compartmentalization; without materializing the
        # proven descriptor first, a chained layout operation (for example
        # ``literal.unsqueeze(1).unsqueeze(2)``) loses the first view's shape
        # exactly at the region seam.
        descriptor = _tensor_descriptor(graph, int(node_id))
        if descriptor_states_a_shape(descriptor):
            tensor = dict(data.get("tensor") or {})
            tensor.update({
                key: copy.deepcopy(descriptor[key])
                for key in ("shape", "dtype", "device", "rank")
                if descriptor.get(key) is not None
            })
            data["tensor"] = tensor
        if str(data.get("type")) in {"Const", "const", "Constant"} and node_id not in carried_initials:
            continue
        data["type"] = "Input"
        if node_id in carried_initials:
            data["constant"] = None
            data["attributes"] = dict(data.get("attributes") or {})
            data["attributes"].pop("value", None)
        data["op"] = "input"
        data["label"] = f"value_{node_id}"
        _set_operands(subgraph, node_id, [], cause=_REGION_BOUNDARY_INPUT)
        # Semantic ``dependency`` edges written above are not operand
        # positions; they go with the boundary too.
        for parent in tuple(subgraph.G.predecessors(node_id)):
            subgraph.G.remove_edge(parent, node_id)

    observed_children = {
        node_id: children - inert_nodes
        for node_id, children in semantic_children.items()
    }
    subgraph.roots = [
        node_id
        for node_id in node_ids
        if (
            node_id in required_outputs
            or not observed_children[node_id]
            or any(
                child not in selected
                for child in observed_children[node_id]
            )
        )
    ]
    deployment_outputs = tuple(subgraph.roots)
    next_node_id = max(
        (node_id for node_id in graph.G if isinstance(node_id, int)),
        default=0,
    ) + 1
    store_nodes = []
    for output_id in deployment_outputs:
        while next_node_id in subgraph.G:
            next_node_id += 1
        store_id = next_node_id
        next_node_id += 1
        subgraph.G.add_node(
            store_id,
            type="Store",
            op="store",
            label=f"value_{output_id}",
            parents=[],
            children=[],
        )
        # The Store publishes ``output_id``: its row derives from the stored
        # value's cell (edges lane C).  Its destination is the region's
        # output publication (``dispatch_store``, posted with the region);
        # no ``storage_root`` row exists yet at planning time.
        from ..common.tensors.topological_reducer import (
            node_identity_cell as _node_identity_cell,
            post_derived_node_identity as _post_derived_node_identity,
        )
        from .concordance_declarations import (
            PLANNER_REGION_CARVE as _PLANNER_REGION_CARVE,
        )

        _post_derived_node_identity(
            subgraph, store_id,
            (_node_identity_cell(subgraph, int(output_id)),),
            stage=_PLANNER_REGION_CARVE,
        )
        _set_operands(
            subgraph, store_id, [(output_id, "value")],
            cause=_DISPATCH_STORE_TRANSFORM,
        )
        store_nodes.append(store_id)
    subgraph.roots = store_nodes
    # Compartmentalization changes both the region boundary and its available
    # inputs, so inherited levels from the source graph are no longer an
    # execution schedule for this subgraph.  Schedule every compartment from
    # its reconstructed internal dependencies.  The source schedule guided
    # legal region formation; this local schedule governs emitted instruction
    # order inside the resulting shader.
    schedule_preference = str(schedule_preference).lower()
    if schedule_preference not in {"asap", "alap"}:
        raise ValueError(
            "dispatch schedule preference must be 'asap' or 'alap'"
        )
    computed_levels = subgraph.compute_levels(
        method=schedule_preference,
        order="dependency",
    )
    if computed_levels is not None:
        subgraph.levels = computed_levels
    subgraph.G.graph["compartment_schedule_preference"] = (
        schedule_preference
    )
    # One topological sort, then bucket by level -- not one full sort per
    # level. The order within a level is identical (it is the single sort's
    # order, filtered), so the schedule is unchanged; only the redundant
    # O(levels x sort) work is removed.
    _compartment_order = nx.lexicographical_topological_sort(
        subgraph.G, key=lambda value_id: int(value_id)
    )
    _nodes_by_level: dict[int, list[int]] = {}
    for node_id in _compartment_order:
        _nodes_by_level.setdefault(
            int(subgraph.levels.get(node_id, 0)), []
        ).append(node_id)
    subgraph.G.graph["compartment_schedule"] = tuple(
        (level, tuple(_nodes_by_level.get(level, ())))
        for level in sorted(set(subgraph.levels.values()))
    )
    subgraph.G.graph["deployment_inputs"] = tuple(
        node_id
        for node_id, data in subgraph.G.nodes(data=True)
        if data.get("type") == "Input"
    )
    subgraph.G.graph["deployment_outputs"] = deployment_outputs
    subgraph.G.graph["deployment_store_nodes"] = tuple(store_nodes)
    subgraph.G.graph["deployment_nodes"] = tuple(
        node_id for node_id in node_ids if node_id in subgraph.G
    )
    if region_index is not None:
        _post_deployment_region(graph, subgraph, int(region_index))
    return subgraph
