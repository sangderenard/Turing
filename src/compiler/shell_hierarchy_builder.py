"""Shell hierarchy builder: builds the hierarchical plan of a shell (regions, calls, closures and result bindings) from its process graph, with the consumer/item/call-argument operand concordance and the shape-only view operation set."""

from __future__ import annotations

import ast
import networkx as nx
import numpy as np
from typing import Any, Mapping

from .hierarchical_plan import (
    PlanCall,
    PlanClosure,
    PlanLine,
)
from ..common.tensors.topological_reducer import _widest_element
from ..transmogrifier.operator_defs import abstract_tensor_funcs
from .source_node_facts import _ast_source_signature, _constant_value
from .call_argument_binding import (
    _positional_argument_index,
    _method_parameter_layout,
    _authored_aggregate_leaves,
    _expanded_callsite_argument_edges,
)
from .region_scheduling import (
    _dependency_order,
    _control_dependency_value_ids,
)
from .tensor_descriptor_query import (
    descriptor_states_a_shape,
    _tensor_descriptor,
)
from .dispatch_region_partition import _atomic_region_node_order
from .structural_fold import _fold_callsite_structural_values


def _concord_consumer_operands(
    graph: Any, consumer_nodes: Any, values: Any,
) -> None:
    """Commit, per consumer node and value, which operands read it.

    Page ``consumer_operand`` row ``(read scope, node, value)`` holds the
    ``(role, ordinal)`` positions of ``node``'s operands that read
    ``value``.  Structure only: the binding each operand read is the base
    fact ``lexical_read_binding``, at ``(read scope, node, role, ordinal)``.
    """

    from ..common.tensors.topological_reducer import _operand_positions
    from .identity_concordance import current_identity_book

    scope = graph.G.graph.get("lexical_read_scope")
    if scope is None:
        return
    wanted = set(map(int, values))
    book = current_identity_book()
    page = book.page("consumer_operand")
    # A read row naming a position its consumer no longer has was orphaned
    # by an operand rewrite that bypassed ``_set_operands``: the binding the
    # operand read is lost.  Record it where the live graph and the rows
    # meet; ``CorrelationTable`` reports page ``operand_position_orphan``.
    read_page = book.page("lexical_read_binding")
    orphan_page = book.page("operand_position_orphan")
    read_rows: dict[int, list[tuple[Any, int]]] = {}
    for row in read_page.scope_rows(tuple(scope)):
        if (
            len(row) == 4 and isinstance(row[1], int)
            and read_page.latest(row) is not None
        ):
            read_rows.setdefault(int(row[1]), []).append((row[2], row[3]))
    for node_id in consumer_nodes:
        if int(node_id) not in graph.G:
            continue
        positions: dict[int, list[tuple[Any, int]]] = {}
        present: set[tuple[Any, int]] = set()
        for role, ordinal, parent in _operand_positions(
            graph.G.nodes[int(node_id)].get("parents") or ()
        ):
            present.add((role, ordinal))
            if int(parent) in wanted:
                positions.setdefault(int(parent), []).append((role, ordinal))
        for position in read_rows.get(int(node_id), ()):
            if position not in present:
                orphan_page.concord(
                    (tuple(scope), int(node_id), *position),
                    read_page.latest((tuple(scope), int(node_id), *position)),
                )
        for parent, operand_positions in positions.items():
            page.concord(
                (tuple(scope), int(node_id), parent), tuple(operand_positions),
            )


def _concord_item_operands(graph: Any, captures: Any) -> None:
    """Record, per captured ``item()`` node, the operand position it reads.

    ``float(total.item())`` in a loop body: no region computes the ``item``,
    so the region captures the item's id.  The loop carries ``total``, not
    the item id; a later id-keyed identity rewrite
    (``_rebind_recorded_scalar_identities``) replaced the capture with
    ``total``'s only definition -- the pre-loop one -- and ``total_value``
    froze at its first value (silent wrong answer).

    Page ``item_operand`` row ``(read scope, item id)`` holds ``(operand id,
    role, ordinal)``: structure only.  The control lowering reads the
    operand by the binding at that position (``lexical_read_binding``) in
    each read site's loop generation, and decides the rank-0 identity from
    the value it reads.
    """

    from ..common.tensors.topological_reducer import _operand_positions
    from .identity_concordance import current_identity_book

    scope = graph.G.graph.get("lexical_read_scope")
    if scope is None:
        return
    page = current_identity_book().page("item_operand")
    for value_id in captures:
        if int(value_id) not in graph.G:
            continue
        data = graph.G.nodes[int(value_id)]
        if str(data.get("op") or data.get("type") or "").casefold() != "item":
            continue
        operands = [
            (role, ordinal, parent)
            for role, ordinal, parent in _operand_positions(
                data.get("parents") or ()
            )
            if str(role) in {"operand", "arg:0", "value"}
        ]
        if len(operands) != 1:
            continue
        role, ordinal, operand = operands[0]
        page.concord(
            (tuple(scope), int(value_id)),
            (int(operand), str(role), int(ordinal)),
        )


def _concord_call_argument_operands(
    graph: Any, callsite_id: int, operand_edges: Mapping[int, tuple[int, str]],
) -> None:
    """Commit, per argument position of one call, which operand it is.

    Page ``call_argument_operand`` row ``(read scope, callsite, position)``
    holds the ``(role, ordinal)`` of the call node's operand that produced
    that argument binding.  One value passed twice is two operands; the
    binding each read is the base fact ``lexical_read_binding``.
    """

    from ..common.tensors.topological_reducer import _operand_positions
    from .identity_concordance import current_identity_book

    scope = graph.G.graph.get("lexical_read_scope")
    if scope is None or int(callsite_id) not in graph.G:
        return
    positions = list(_operand_positions(
        graph.G.nodes[int(callsite_id)].get("parents") or ()
    ))
    page = current_identity_book().page("call_argument_operand")
    for index, (parent, role) in operand_edges.items():
        ordinal = next((
            edge_ordinal for edge_role, edge_ordinal, edge_parent in positions
            if str(edge_role) == str(role) and int(edge_parent) == int(parent)
        ), None)
        if ordinal is not None:
            page.concord(
                (tuple(scope), int(callsite_id), int(index)), (role, ordinal),
            )


def _build_shell_hierarchy_plan(
    shell: Any,
    _memo: dict[int, PlanClosure] | None = None,
) -> PlanClosure:
    """Freeze call/region ownership before backend source composition."""

    memo = {} if _memo is None else _memo
    incumbent_plan = memo.get(id(shell))
    if incumbent_plan is not None:
        return incumbent_plan

    graph = shell.process_graph
    # The plan freezes tensor descriptors into ``PlanClosure.value_shapes``.
    # Some catalogue and callsite shells are constructed before their
    # ProgramABI descriptors arrive; freezing them at that point permanently
    # records reshape results as scalars even though the graph can now resolve
    # their structural shape tuples. Make the documented ordering invariant
    # local to the freeze operation itself. The fold is structural,
    # idempotent, and executes no numerical source code.
    if graph.G.graph.get("function_name"):
        _fold_callsite_structural_values(graph)
    # An Indexed successor used to unpack a multi-result source call is a
    # call-boundary projection, not a numerical gather owned by a region.
    # Dispatch regions can be carved before callsite shells are attached, so
    # remove these exact projection nodes when freezing final ownership. If
    # both the PlanCall and a stale region produce the same caller value, the
    # repository-SSA type linker sees two physical definitions and can
    # oscillate between their shapes indefinitely.
    def call_result_projections(value_id: int) -> set[int]:
        # Every literal-index projection below a call result is a structural
        # path onto that call's outputs: ``r[1]`` (an intermediate aggregate
        # with no physical value of its own) as much as ``r[1][0]`` (a bound
        # leaf).  None of them is an operator a region may compute.
        found: set[int] = set()
        pending = [int(value_id)]
        while pending:
            current = pending.pop()
            for successor in graph.G.successors(current):
                successor_data = graph.G.nodes[successor]
                if str(
                    successor_data.get("op")
                    or successor_data.get("type")
                    or ""
                ).casefold() != "indexed":
                    continue
                if not any(
                    str(role) in {"value", "base", "operand", "object"}
                    and int(parent) == current
                    for parent, role in successor_data.get("parents") or ()
                ):
                    continue
                # Only a literal integer index is a structural path onto a
                # call's outputs.  A slice or a tuple of slices over a tensor
                # call result (``moment[:, :, 1]``, ``m[:, :, 0:2]``) is a
                # numerical view an owning region must compute; treating it
                # as a projection left it with no producer at all.
                index_parents = tuple(
                    int(parent)
                    for parent, role in successor_data.get("parents") or ()
                    if str(role) == "index"
                )
                if len(index_parents) != 1 or index_parents[0] not in graph.G:
                    continue
                index_data = graph.G.nodes[index_parents[0]]
                if str(index_data.get("type")) not in {
                    "Const", "const", "Constant",
                }:
                    continue
                try:
                    index_value = _constant_value(index_data)
                except KeyError:
                    continue
                if not isinstance(index_value, int) or isinstance(
                    index_value, bool
                ):
                    continue
                if int(successor) in found:
                    continue
                found.add(int(successor))
                # Keep descending only while the projection is itself an
                # intermediate aggregate.  ``r[1][0]`` is a structural path
                # when ``r[1]`` is another tuple, and an ordinary element read
                # when ``r[1]`` is a tensor.  Walking through a tensor removed
                # that read from every region, so nothing computed it and the
                # caller kept a formal for it.
                if _tensor_descriptor(graph, int(successor)) is None:
                    pending.append(int(successor))
        return found

    call_result_projection_ids = {
        projection
        for callsite_id in shell.callsite_function_shells
        if int(callsite_id) in graph.G
        for projection in call_result_projections(int(callsite_id))
    }
    region_by_node = {
        int(node_id): int(region_index)
        for region_index, subgraph in enumerate(shell.dispatch_subgraphs)
        for node_id in subgraph.G.graph.get("deployment_nodes", ())
        if int(node_id) not in call_result_projection_ids
    }
    emitted_regions = set()
    items = []
    topological_nodes = _dependency_order(graph)
    order_index = {
        int(node_id): index
        for index, node_id in enumerate(topological_nodes)
    }
    def source_position(node_id: int) -> tuple[int, int] | None:
        expression = graph.G.nodes[int(node_id)].get("expr_obj")
        line = getattr(expression, "lineno", None)
        column = getattr(expression, "col_offset", None)
        if line is None:
            return None
        return int(line), int(column or 0)

    caller_identities = graph.G.graph.get("identity_table") or {}
    # Regions are emitted atomically at the position of their first member,
    # so the walk must already respect region-level dependencies.
    walk_order = _atomic_region_node_order(
        graph, topological_nodes, region_by_node,
    )

    def direct_return_value_ids(child_graph: Any) -> tuple[int, ...]:
        """Resolve one direct return by deterministic source coordinates.

        ProcessGraph return_value_nodes historically held ``id(ast_node)``
        correlations, which do not survive graph reduction/reconstruction.
        The reduced nodes retain exact source spans, so correlate the authored
        Return.value to its graph value without restoring transient ids.
        """

        return_signatures = set()
        for _candidate_id, candidate_data in child_graph.nodes(data=True):
            expression = candidate_data.get("expr_obj")
            if not isinstance(expression, ast.AST):
                continue
            for nested in ast.walk(expression):
                if not isinstance(nested, ast.Return) or nested.value is None:
                    continue
                return_signatures.add(_ast_source_signature(nested.value))
        matched = []
        for candidate_id, candidate_data in child_graph.nodes(data=True):
            expression = candidate_data.get("expr_obj")
            if not isinstance(expression, ast.AST):
                continue
            if _ast_source_signature(expression) not in return_signatures:
                continue
            value_id = int(candidate_data.get("value_id", candidate_id))
            attributes = candidate_data.get("attributes") or {}
            if (
                str(attributes.get("extraction_identity") or "") in {
                    "builtins.bytes", "builtins.bytearray", "builtins.list",
                    "builtins.tuple", "builtins.frozenset", "builtins.set",
                }
                or str(attributes.get("producer_kind") or "")
                == "aggregate_materialization"
            ):
                backing = tuple(
                    int(parent)
                    for parent, role in candidate_data.get("parents") or ()
                    if str(role) in {"arg:0", "operand", "value"}
                )
                if len(backing) == 1:
                    value_id = backing[0]
            matched.append(value_id)
        unique = tuple(dict.fromkeys(matched))
        # Multiple lexical returns need control-aware result merging; never
        # guess that they are a tuple-valued return surface.
        return unique if len(unique) == 1 else ()

    for node_id in walk_order:
        child = shell.callsite_function_shells.get(node_id)
        if child is not None:
            call_attributes = graph.G.nodes[node_id].get("attributes") or {}
            constructor_call = call_attributes.get("constructor_ref") is not None
            call_parents = _expanded_callsite_argument_edges(graph, node_id)
            parents = tuple(parent for parent, _role in call_parents)
            child_graph = child.process_graph.G
            child_identities = (
                child_graph.graph.get("identity_table") or {}
            )
            receiver_parameter, positional_parameters, _parameters = (
                _method_parameter_layout(child_graph)
            )
            argument_bindings = []
            if constructor_call and receiver_parameter is not None:
                receiver_identities = child_identities.get(
                    receiver_parameter, ()
                )
                if receiver_identities:
                    # The class-call result is the caller-owned record identity
                    # initialized by __init__/__new__.  It is both the
                    # constructor callsite and the concrete ``self`` binding;
                    # its fields remain ordinary caller-owned arena values.
                    argument_bindings.append((
                        int(node_id), int(receiver_identities[0])
                    ))
            # Which call edge produced each argument binding, whatever path
            # the body below takes: the operand's authored read is the edge's.
            operand_edges: dict[int, tuple[int, str]] = {}

            def attributed_call_parents():
                for edge_parent, edge_role in call_parents:
                    start = len(argument_bindings)
                    yield edge_parent, edge_role
                    for index in range(start, len(argument_bindings)):
                        operand_edges[index] = (int(edge_parent), str(edge_role))

            for parent, role in attributed_call_parents():
                position = _positional_argument_index(role)
                # The call's edge is the argument's identity.  (A former
                # name-history override replaced it with the source-latest
                # definition of the argument's name, which chose a continue
                # arm's value over the edge to the pre-branch value.)
                if position is not None:
                    name = (
                        positional_parameters[position]
                        if position < len(positional_parameters)
                        else None
                    )
                elif (
                    role == "operand"
                    and receiver_parameter is not None
                ):
                    name = receiver_parameter
                elif role.startswith("kw:"):
                    name = role.split(":", 1)[1]
                elif role.startswith("closure:"):
                    name = role.split(":", 1)[1]
                else:
                    name = None
                identities = child_identities.get(name, ())
                caller_leaves = _authored_aggregate_leaves(
                    graph, int(parent),
                )
                # Member formals materialized for an aggregate parameter
                # carry their exact member path.  They outlive the
                # shapeless aggregate formal, which is dropped once every
                # authored projection reads its member directly, so the
                # member binding must not depend on that formal's identity.
                member_inputs = {
                    tuple(map(int, member_attributes.get(
                        "aggregate_path",
                        (member_attributes["aggregate_index"],),
                    ))): int(member_id)
                    for member_id, member_data in child_graph.nodes(data=True)
                    if member_data.get("type") == "Input"
                    and str(
                        (member_data.get("attributes") or {}).get(
                            "aggregate_parent_binding"
                        )
                    ) == str(name)
                    for member_attributes in (
                        member_data.get("attributes") or {},
                    )
                    if member_attributes.get("aggregate_index") is not None
                }
                if member_inputs:
                    caller_member_ids: dict[tuple[int, ...], int] = {}

                    def record_caller_members(
                        value_id: int, path: tuple[int, ...] = (),
                    ) -> None:
                        member_leaves = _authored_aggregate_leaves(
                            graph, int(value_id),
                        )
                        if not member_leaves:
                            if path:
                                caller_member_ids[path] = int(value_id)
                            return
                        for member_index, member_id in enumerate(member_leaves):
                            record_caller_members(
                                int(member_id), (*path, int(member_index)),
                            )

                    record_caller_members(int(parent))
                    missing_paths = tuple(
                        path for path in member_inputs
                        if path not in caller_member_ids
                    )
                    if missing_paths:
                        raise ValueError(
                            f"aggregate call binding for {name!r} names "
                            f"paths {missing_paths!r} absent from caller "
                            f"paths {tuple(caller_member_ids)!r}"
                        )
                    for path in sorted(member_inputs):
                        argument_bindings.append((
                            int(caller_member_ids[path]), member_inputs[path],
                        ))
                if identities and not member_inputs:
                    # With member formals bound above, the aggregate itself
                    # has no physical storage to pass: the callee's formal is
                    # the tuple of its members.  Binding it too would turn an
                    # authored tuple node into an untyped caller argument.
                    child_input = int(identities[0])
                    child_input_data = child_graph.nodes.get(child_input, {})
                    if (child_input_data.get("attributes") or {}).get(
                        "structural_specialization"
                    ):
                        # This exact formal has already become a callee-owned
                        # literal. Its source identity remains in the ledger,
                        # but it no longer accepts a runtime argument frame.
                        continue
                    argument_bindings.append((parent, child_input))
                    child_leaves = _authored_aggregate_leaves(
                        child.process_graph, child_input,
                    )
                    if caller_leaves or child_leaves:
                        if len(caller_leaves) != len(child_leaves):
                            raise ValueError(
                                f"aggregate call binding for {name!r} has "
                                "different caller/callee arity: "
                                f"{len(caller_leaves)} != {len(child_leaves)}"
                            )
                        argument_bindings.extend(zip(
                            caller_leaves, child_leaves, strict=True,
                        ))
            bound_child_inputs = {
                int(child_input)
                for _caller, child_input in argument_bindings
            }
            for child_input, child_data in child_graph.nodes(data=True):
                child_attributes = child_data.get("attributes") or {}
                if (
                    child_data.get("type") != "Input"
                    or child_attributes.get("binding_kind") != "external"
                    or int(child_input) in bound_child_inputs
                ):
                    continue
                name = child_attributes.get("binding_name")
                call_position = source_position(int(node_id))
                candidates = [
                    int(definition)
                    for definition in caller_identities.get(name, ())
                    if (
                        int(definition) in order_index
                        and not nx.has_path(
                            graph.G, int(node_id), int(definition)
                        )
                        and (
                            (
                                call_position is not None
                                and source_position(int(definition))
                                is not None
                                and source_position(int(definition))
                                < call_position
                            )
                            or (
                                (
                                    call_position is None
                                    or source_position(int(definition))
                                    is None
                                )
                                and order_index[int(definition)]
                                < order_index[int(node_id)]
                            )
                        )
                    )
                ]
                if candidates:
                    argument_bindings.append((
                        max(
                            candidates,
                            key=lambda definition: (
                                source_position(definition)
                                or (-1, order_index[definition])
                            ),
                        ),
                        int(child_input),
                    ))
            child_outputs = tuple(
                int(child_identities[name][-1])
                for name in child_graph.graph.get(
                    "function_outputs", ()
                )
                if child_identities.get(name)
            )
            if not child_outputs:
                # ``return bytes(out)`` and other direct aggregate-return
                # expressions have no authored assignment name, so
                # function_outputs is legitimately empty.  Capture analysis
                # already records their exact ProcessGraph value ids; use
                # that structural return surface for the call binding instead
                # of silently turning a value-returning call into void.
                child_outputs = tuple(map(int, getattr(
                    child, "_captured_return_value_ids", (),
                )))
            if not child_outputs:
                child_outputs = direct_return_value_ids(child_graph)
            # A returned aggregate is a structural path, not one position in
            # a flat list.  ``return total, (hub, angle), valid`` has leaf
            # paths (0,), (1, 0), (1, 1), (2,); the caller's ``r[1][0]`` is
            # the projection path (1, 0).  Expanding the callee leaves and
            # zipping them against the caller's first-level projections
            # shifts every member after the aggregate onto the wrong result.
            child_output_paths: dict[tuple[int, ...], int] = {}

            def expand_child_output(
                path: tuple[int, ...], value_id: int,
            ) -> None:
                leaves = _authored_aggregate_leaves(
                    child.process_graph, int(value_id),
                )
                if not leaves:
                    child_output_paths[path] = int(value_id)
                    return
                for index, leaf in enumerate(leaves):
                    expand_child_output((*path, int(index)), int(leaf))

            for position, child_output in enumerate(child_outputs):
                expand_child_output((int(position),), int(child_output))

            def literal_projection_index(successor: int) -> int | None:
                successor_data = graph.G.nodes[int(successor)]
                if str(successor_data.get("op", "")).lower() != "indexed":
                    return None
                index_parents = tuple(
                    int(parent)
                    for parent, role in (
                        successor_data.get("parents") or ()
                    )
                    if str(role) == "index"
                )
                if len(index_parents) != 1:
                    return None
                index_data = graph.G.nodes[index_parents[0]]
                index_value = index_data.get("constant")
                if index_value is None:
                    index_value = (
                        index_data.get("attributes") or {}
                    ).get("value")
                if isinstance(index_value, bool) or not isinstance(
                    index_value, int
                ):
                    return None
                return int(index_value)

            caller_projection_paths: dict[tuple[int, ...], int] = {}

            def collect_caller_projections(
                path: tuple[int, ...], value_id: int,
            ) -> None:
                for successor in graph.G.successors(int(value_id)):
                    index = literal_projection_index(int(successor))
                    if index is None:
                        continue
                    projection_path = (*path, index)
                    caller_projection_paths[projection_path] = int(successor)
                    collect_caller_projections(projection_path, int(successor))

            collect_caller_projections((), int(node_id))
            # This is the operation that owns the exact relation between a
            # source call's authored result path and the caller projection
            # value.  Publish it now.  Later SSA stages must not reconstruct
            # that relation from target names or from dependency enumeration,
            # either of which can change when a projection becomes loop state.
            from .identity_concordance import current_identity_book

            call_result_projection_page = current_identity_book().page(
                "call_result_projection_concordance"
            )
            for projection_path, caller_projection_id in (
                caller_projection_paths.items()
            ):
                call_result_projection_page.concord(
                    (int(node_id), int(caller_projection_id)),
                    tuple(map(int, projection_path)),
                )
            if tuple(child_output_paths) == ((0,),) and not (
                graph.G.nodes[int(node_id)].get("attributes") or {}
            ).get("authored_return_container"):
                result_bindings = (
                    (child_output_paths[(0,)], int(node_id)),
                )
            else:
                result_bindings = tuple(
                    (child_output_paths[path], caller_projection_paths[path])
                    for path in child_output_paths
                    if path in caller_projection_paths
                )
            child_outputs = tuple(child_output_paths.values())
            _concord_call_argument_operands(graph, int(node_id), operand_edges)
            items.append(PlanCall(
                int(node_id),
                _build_shell_hierarchy_plan(child, memo),
                parents,
                tuple(caller for _callee, caller in result_bindings),
                tuple(argument_bindings),
                tuple(result_bindings),
                tuple(
                    int(plan.loop.node_id)
                    for plan in sorted(
                        (
                            plan
                            for plan in shell.loop_plans
                            if int(node_id) in plan.loop.body_nodes
                        ),
                        key=lambda plan: -len(plan.loop.body_nodes),
                    )
                ),
            ))
            continue
        region_index = region_by_node.get(int(node_id))
        if (
            region_index is None
            or region_index in emitted_regions
        ):
            continue
        emitted_regions.add(region_index)
        subgraph = shell.dispatch_subgraphs[region_index]
        # The dispatch subgraph was carved before loop evaporation could
        # replace the nodes it named.  The live graph is the authority for
        # what this region still contains, so drop members it no longer
        # holds instead of indexing them and raising a bare KeyError.
        region_nodes = tuple(
            int(value)
            for value in subgraph.G.graph.get("deployment_nodes", ())
            if int(value) not in call_result_projection_ids
            and int(value) in graph.G
        )
        original_region_node_set = set(region_nodes)
        carried_initial_boundaries = {
            int(initial)
            for loop_plan in getattr(shell, "loop_plans", ())
            for _name, initial, updated in loop_plan.loop.carried_bindings
            if int(updated) in original_region_node_set
            and int(initial) != int(updated)
        }
        # A carried initializer is the entry arm of the coordinator-owned Phi.
        # If the numerical compartment also owns that initializer, its body
        # recomputes from the iteration-zero value on every trip.  Cut it from
        # this region so the ordinary capture calculation below binds the
        # current Phi value into the update operation.
        region_nodes = tuple(
            value_id for value_id in region_nodes
            if value_id not in carried_initial_boundaries
        )
        region_node_set = set(region_nodes)

        def numerical_region_parents(
            node_id: int,
        ) -> tuple[tuple[int, str], ...]:
            """Expose structural tensor-call leaves at the numerical ABI.

            A Python list/tuple is one source argument, but stack/cat consume
            each tensor element as a separate numerical operand.  The graph's
            aggregate leaf ledger is the exact authored correspondence; the
            container identity itself is coordinator storage and must not
            replace those producers at a numerical region cut.
            """

            node_data = graph.G.nodes.get(int(node_id))
            if node_data is None:
                # Loop evaporation removes the nodes it replaced, and a region
                # view carved before that still names them.  The graph is the
                # authority this recomputation exists to consult, so a node it
                # no longer holds contributes no parents rather than raising.
                return ()
            operation = str(
                node_data.get("op") or node_data.get("type") or ""
            )
            expanded: list[tuple[int, str]] = []
            for parent, role in node_data.get("parents") or ():
                parent = int(parent)
                parent_data = graph.G.nodes.get(parent, {})
                attributes = parent_data.get("attributes") or {}
                leaves = tuple(map(
                    int,
                    attributes.get("aggregate_leaf_value_ids", ()),
                ))
                if (
                    operation in {"stack", "cat", "concat"}
                    and str(role).startswith("arg")
                    and attributes.get("producer_kind") == "aggregate"
                    and attributes.get("aggregate_kind") in {"list", "tuple"}
                    and leaves
                ):
                    expanded.extend(
                        (leaf, f"{role}:element:{index}")
                        for index, leaf in enumerate(leaves)
                    )
                else:
                    expanded.append((parent, str(role)))
            return tuple(expanded)

        # Loop-port materialization can rewrite a region's live parents after
        # its dispatch subgraph was first carved. Recompute region captures
        # from the authoritative current graph instead of
        # reusing stale deployment_inputs from the numeric view.
        region_captures = tuple(dict.fromkeys((
            *(
                int(parent)
                for node_id in region_nodes
                for parent, _role in numerical_region_parents(node_id)
                if int(parent) not in region_node_set
            ),
        )))

        # A region capture is not necessarily a runtime input. Structural
        # arguments such as ``transpose(-2, -1)`` often sit just outside the
        # tensor region as a tiny constant-expression chain (Const -> neg, or
        # several constants -> Tuple). Preserve that source closure in the
        # region instead of promoting its terminal to an unexplained ABI
        # parameter. This is graph-source pursuit, not value specialization:
        # only expressions whose complete producer closure is literal are
        # admitted here.
        # Keep evaluator state separate from its values.  A Python ``object()``
        # sentinel used to be written into this runtime-indexed cache for an
        # unavailable producer.  When this planner itself is ingested, that is
        # a static Python reference flowing through dynamic table storage --
        # something complete SSA correctly refuses to pretend is data.
        _CONSTANT_KNOWN = 1
        _CONSTANT_UNAVAILABLE = 2
        _constant_status: dict[int, int] = {}
        _constant_values: dict[int, Any] = {}
        # A loop-carried initial is the entry arm of a runtime Phi, not a
        # literal every consumer may fold: inside the loop the name holds
        # the previous iteration's update.  Region extraction already
        # captures such a seed as an Input; a capture derived from it
        # (``i + 1`` over the seed ``0``) is likewise a runtime input, never
        # a region-local constant.  Seeding the evaluator state here keeps
        # every closure through a carried seed unavailable.
        for _node_id, _node_data in graph.G.nodes(data=True):
            for _initial, _updated in (
                (_node_data.get("attributes") or {}).get(
                    "loop_carried_bindings"
                ) or {}
            ).values():
                _constant_status[int(_initial)] = _CONSTANT_UNAVAILABLE

        def _constant_expression(value_id: int):
            """Evaluate a literal producer closure without Python recursion.

            A retained loop can expose feedback in the ProcessGraph, and a
            large whole-object compile can contain a perfectly acyclic closure
            deeper than Python's call-stack limit. Neither makes the capture a
            constant. Use an explicit post-order walk, marking every active
            frame nonconstant if a feedback edge is encountered.
            """

            root = int(value_id)
            if root in _constant_status:
                return (
                    _constant_status[root] == _CONSTANT_KNOWN,
                    _constant_values.get(root),
                )

            # 0/absent = unseen, 1 = active DFS frame, 2 = evaluated.
            states: dict[int, int] = {}
            stack: list[tuple[int, bool]] = [(root, False)]
            while stack:
                current, expanded = stack.pop()
                if current in _constant_status:
                    states[current] = 2
                    continue
                node = graph.G.nodes.get(current, {})
                opcode = str(node.get("op") or node.get("type") or "")
                attributes = dict(node.get("attributes") or {})
                parents = tuple(
                    int(parent)
                    for parent, _role in (node.get("parents") or ())
                )

                if not expanded:
                    if states.get(current) == 1:
                        # This node is already on the active producer path: a
                        # feedback closure cannot be a literal expression.
                        for active, state in tuple(states.items()):
                            if state == 1:
                                _constant_status[active] = (
                                    _CONSTANT_UNAVAILABLE
                                )
                                states[active] = 2
                        continue
                    states[current] = 1
                    if opcode in {"const", "Const", "Constant"}:
                        expression = node.get("expr_obj")
                        if "value" in attributes:
                            value = attributes["value"]
                            known = True
                        elif "constant" in node:
                            value = node["constant"]
                            known = True
                        elif isinstance(expression, ast.Constant):
                            value = expression.value
                            known = True
                        else:
                            value = None
                            known = False
                        if known:
                            _constant_values[current] = value
                            _constant_status[current] = _CONSTANT_KNOWN
                        else:
                            _constant_status[current] = (
                                _CONSTANT_UNAVAILABLE
                            )
                        states[current] = 2
                        continue
                    stack.append((current, True))
                    for parent in reversed(parents):
                        if parent in _constant_status:
                            continue
                        if states.get(parent) == 1:
                            for active, state in tuple(states.items()):
                                if state == 1:
                                    _constant_status[active] = (
                                        _CONSTANT_UNAVAILABLE
                                    )
                                    states[active] = 2
                            break
                        stack.append((parent, False))
                    continue

                if states.get(current) == 2:
                    continue
                if any(
                    _constant_status.get(parent) != _CONSTANT_KNOWN
                    for parent in parents
                ):
                    known = False
                elif opcode in {"neg", "USub"} and len(parents) == 1:
                    values = tuple(_constant_values[parent] for parent in parents)
                    try:
                        value = -values[0]
                    except (ArithmeticError, TypeError, ValueError):
                        known = False
                    else:
                        known = True
                elif opcode in {"pos", "UAdd"} and len(parents) == 1:
                    values = tuple(_constant_values[parent] for parent in parents)
                    try:
                        value = +values[0]
                    except (ArithmeticError, TypeError, ValueError):
                        known = False
                    else:
                        known = True
                elif opcode in {"Tuple", "tuple"}:
                    values = tuple(_constant_values[parent] for parent in parents)
                    value = tuple(values)
                    known = True
                elif opcode in {"List", "list"}:
                    values = tuple(_constant_values[parent] for parent in parents)
                    value = list(values)
                    known = True
                elif opcode in {"add", "Add"} and len(parents) == 2:
                    values = tuple(_constant_values[parent] for parent in parents)
                    try:
                        value = values[0] + values[1]
                    except (ArithmeticError, TypeError, ValueError):
                        known = False
                    else:
                        known = True
                elif opcode in {"sub", "Sub"} and len(parents) == 2:
                    values = tuple(_constant_values[parent] for parent in parents)
                    try:
                        value = values[0] - values[1]
                    except (ArithmeticError, TypeError, ValueError):
                        known = False
                    else:
                        known = True
                elif opcode in {"mul", "Mul"} and len(parents) == 2:
                    values = tuple(_constant_values[parent] for parent in parents)
                    try:
                        value = values[0] * values[1]
                    except (ArithmeticError, TypeError, ValueError):
                        known = False
                    else:
                        known = True
                else:
                    known = False
                if known:
                    _constant_values[current] = value
                    _constant_status[current] = _CONSTANT_KNOWN
                else:
                    _constant_status[current] = _CONSTANT_UNAVAILABLE
                states[current] = 2
            return (
                _constant_status.get(root) == _CONSTANT_KNOWN,
                _constant_values.get(root),
            )

        # A loop target's value is per-iteration even though its INITIALIZER
        # is a literal; folding it froze ``row`` at 0 inside every region a
        # nested body formed, so value = row*4+col compiled as value = col.
        loop_target_value_ids = {
            int(bound_target)
            for _loop_id, loop_data in graph.G.nodes(data=True)
            for bound_target in (
                (loop_data.get("attributes") or {}).get(
                    "loop_target_bindings", {}
                ) or {}
            ).values()
        }
        capture_constants = {}
        for value_id in region_captures:
            if value_id in carried_initial_boundaries:
                continue
            if int(value_id) in loop_target_value_ids:
                continue
            known, value = _constant_expression(value_id)
            if known:
                capture_constants[value_id] = value
        region_captures = tuple(
            value_id
            for value_id in region_captures
            if value_id not in capture_constants
        )
        compute_lines = []
        for value in region_nodes:
            node_data = graph.G.nodes[value]
            expression = node_data.get("expr_obj")
            parents = numerical_region_parents(value)
            opcode = str(node_data.get("op") or node_data.get("type"))
            line_attributes = dict(node_data.get("attributes") or {})
            if (
                isinstance(expression, ast.Attribute)
                or (
                    str(node_data.get("op") or node_data.get("type") or "")
                    .casefold() == "getattr"
                    and line_attributes.get("tensor") is not None
                )
            ):
                tensor_property = line_attributes.get("tensor")
                if tensor_property in abstract_tensor_funcs:
                    # NumPy/Torch expose ``real``/``imag`` as properties;
                    # after receiver grounding they are the same abstract
                    # tensor operations as ``AbstractTensor.real(x)`` and
                    # ``AbstractTensor.imag(x)``.
                    opcode = str(tensor_property)
                else:
                    # The graph may retain the frontend spelling ``Attribute``;
                    # repository SSA uses the semantic memory operation and must
                    # retain the selected field name in its record ABI.
                    opcode = "GetAttr"
                    line_attributes.setdefault("attribute", expression.attr)
            elif opcode in {"Call", "call"} and str(
                line_attributes.get("tensor") or ""
            ) in abstract_tensor_funcs:
                # A builtin arriving as a Call names its catalogued operator
                # in ``tensor`` (``max``/``min``); the operator IS the
                # semantics, exactly as ``float`` respells to a cast.
                opcode = str(line_attributes["tensor"])
            compute_lines.append(PlanLine.create(
                opcode,
                inputs=tuple(parent for parent, _role in parents),
                outputs=(int(value),),
                attributes={
                    **line_attributes,
                    "region": region_index,
                },
                input_roles=tuple(role for _parent, role in parents),
            ))
        compute_lines = tuple(compute_lines)
        # A constant operand (``self.n + 1``) is a leaf the region consumes but
        # neither produces nor captures -- it is not a runtime input, so it never
        # entered ``deployment_inputs``. The fused capture folds it into a scalar
        # attribute; the plan lines reference it by value id, so without a
        # defining line it is a dangling SSA value the backend emits as an
        # undeclared variable. Materialise each such constant as its own ``const``
        # line, ahead of the consumers, so the region is self-contained.
        produced = set(region_nodes)
        captured = set(region_captures)
        const_lines = []
        materialised: set[int] = set()
        for value_id, value in capture_constants.items():
            materialised.add(int(value_id))
            const_lines.append(PlanLine.create(
                "Const",
                inputs=(),
                outputs=(int(value_id),),
                attributes={"value": value, "region": region_index},
            ))
        for value in region_nodes:
            for parent, _role in (graph.G.nodes[value].get("parents") or ()):
                parent = int(parent)
                if (
                    parent in produced
                    or parent in captured
                    or parent in materialised
                ):
                    continue
                parent_data = graph.G.nodes.get(parent, {})
                if (
                    parent_data.get("op") or parent_data.get("type")
                ) not in ("const", "Constant"):
                    continue
                materialised.add(parent)
                const_lines.append(PlanLine.create(
                    "Const",
                    inputs=(),
                    outputs=(parent,),
                    attributes={
                        **dict(parent_data.get("attributes") or {}),
                        "region": region_index,
                    },
                ))
        # Carry each region value's shape/dtype from its process-graph node's
        # domain so the lowered SSA is shaped: a value that is an array lowers
        # as an array, not a scalar. A domain padded with size-1 dims (a scalar
        # is (1, 1, 1)) squeezes to its logical shape -- () for a scalar.
        _shape_dtype_cache: dict[int, tuple[tuple[int, ...], str]] = {}

        def _value_shape_dtype(value_id):
            """Infer one value record with a bounded explicit producer stack.

            Feedback is a retained-loop boundary.  When the work stack sees a
            parent already active, it seeds that loop-carried record from the
            parent's declared tensor/domain metadata and does not expand the
            edge again.  This is finite even for irreducible recursion and does
            not turn a loop into an unbounded source expansion.
            """

            root = int(value_id)

            def declared(current: int):
                node = graph.G.nodes.get(int(current), {})
                tensor = node.get("tensor") or {}
                domain = node.get("domain_node")
                attributes = dict(node.get("attributes") or {})
                binding_name = attributes.get("binding_name")
                specialization = (
                    (graph.G.graph.get("planner_specializations") or {}).get(
                        str(binding_name)
                    )
                    if binding_name is not None else None
                )
                specialized_shape = tuple(
                    int(extent)
                    for extent in (
                        getattr(specialization, "shape", ()) or ()
                    )
                )
                # Extents the source-program boundary states for this
                # parameter. Rank alone cannot serve here: the scalar-versus-
                # tensor decision reads EXTENTS (an operation is spelled with
                # its scalar opcode when result and operands all have empty
                # shape), so a parameter that declares `storage: span,
                # rank: 1` and nothing else still arrives shapeless and its
                # arithmetic is compiled as scalar arithmetic. A declared
                # shape is authoritative the same way a planner
                # specialization is -- it is a statement by the boundary, not
                # an inference from a default domain.
                abi_shape: tuple[int, ...] = ()
                if binding_name is not None:
                    abi_entry = (
                        graph.G.graph.get("parameter_value_abi") or {}
                    ).get(str(binding_name))
                    if isinstance(abi_entry, Mapping):
                        abi_shape = tuple(
                            int(extent)
                            for extent in (abi_entry.get("shape") or ())
                        )
                tensor_shape_declared = "shape" in tensor
                if tensor_shape_declared:
                    # An empty static extent tuple can be the exact companion
                    # to a nonzero dynamic rank.  It means "runtime extents",
                    # not "consult the padded DomainNode default".
                    shape = tuple(tensor.get("shape") or ())
                elif specialized_shape:
                    shape = specialized_shape
                elif abi_shape:
                    shape = abi_shape
                else:
                    shape = tuple(getattr(domain, "shape", ()) or ())
                logical = (
                    tuple(map(int, shape))
                    if tensor_shape_declared
                    or specialized_shape
                    or abi_shape
                    else tuple(int(dim) for dim in shape if int(dim) != 1)
                )
                dtype = str(
                    tensor.get("dtype")
                    or getattr(specialization, "dtype", None)
                    or node.get("dtype")
                    or getattr(domain, "dtype", None)
                    or "float64"
                )
                field_name = attributes.get("attribute")
                if (
                    not logical
                    and field_name is not None
                    and str(node.get("type")) in {"GetAttr", "Attribute"}
                ):
                    # A record field read through a mutable, non-array-typed
                    # receiver (``state.next_height``) has no tensor/domain
                    # shape of its own -- ``specialized_shape`` above only
                    # covers a receiver bound *directly* to an array. By this
                    # pass, ``expr_obj`` is gone (nodes created during SSA
                    # normalization carry none), so the receiver is found via
                    # the graph edge instead of the AST. The receiver's own
                    # bound specialization (a real object, set by
                    # ``propagate_bound_planner_specializations`` before this
                    # pass runs) still has the real field on it; read it
                    # structurally, the same safe, side-effect-free
                    # ``getattr`` this module already uses for call-argument
                    # specialization. Without this, every whole-array read of
                    # a record field silently collapses to ``()`` and a
                    # downstream array op is emitted as a scalar op on one
                    # element (see tools/HANDOFF_fluid_c_shell.md, Defect 1).
                    receiver_id = next((
                        int(parent)
                        for parent, role in (node.get("parents") or ())
                        if str(role) in {"value", "base", "object"}
                    ), None)
                    receiver_binding = (
                        (
                            graph.G.nodes.get(receiver_id, {}).get(
                                "attributes"
                            ) or {}
                        ).get("binding_name")
                        if receiver_id is not None else None
                    )
                    receiver_value = (
                        (graph.G.graph.get("planner_specializations") or {})
                        .get(str(receiver_binding))
                        if receiver_binding is not None else None
                    )
                    if receiver_value is not None:
                        try:
                            field_value = getattr(
                                receiver_value, str(field_name)
                            )
                        except AttributeError:
                            field_value = None
                        field_shape = getattr(field_value, "shape", None)
                        if field_shape:
                            logical = tuple(int(dim) for dim in field_shape)
                            field_dtype = getattr(field_value, "dtype", None)
                            if field_dtype is not None:
                                dtype = str(field_dtype)
                operation = str(
                    attributes.get("tensor")
                    or attributes.get("tensor_operation")
                    or attributes.get("tensor_candidate")
                    or node.get("op")
                    or node.get("type")
                    or ""
                ).casefold()
                if operation in {"const", "constant"}:
                    literal = attributes.get("value", node.get("constant"))
                    if isinstance(literal, bool):
                        dtype = "bool"
                    elif isinstance(literal, int):
                        dtype = "int"
                    elif isinstance(literal, float):
                        dtype = "float64"
                # Callable/operator-reference edges are provenance, not
                # numeric operands.  Letting their default domain dtype enter
                # scalar promotion made an all-integer ``min`` or arithmetic
                # chain appear float-valued merely because ``builtins.min``
                # itself was represented by a graph node.
                parents = tuple(
                    int(parent)
                    for parent, role in (node.get("parents") or ())
                    if int(parent) in graph.G
                    and str(role).casefold() not in {
                        "callee", "func", "function", "definition",
                        "operator", "operator_reference",
                    }
                )
                return node, tensor, attributes, logical, dtype, operation, parents

            states: dict[int, int] = {}
            stack: list[tuple[int, bool]] = [(root, False)]
            while stack:
                current, expanded = stack.pop()
                if current in _shape_dtype_cache:
                    states[current] = 2
                    continue
                (
                    _node,
                    tensor,
                    attributes,
                    logical,
                    dtype,
                    operation,
                    parents,
                ) = declared(current)
                if not expanded:
                    states[current] = 1
                    stack.append((current, True))
                    for parent in reversed(parents):
                        if parent in _shape_dtype_cache:
                            continue
                        if states.get(parent) == 1:
                            # The edge closes a retained loop. Its declared
                            # record is the finite boundary condition.
                            (_n, _t, _a, parent_shape, parent_dtype, _o, _p) = (
                                declared(parent)
                            )
                            _shape_dtype_cache[parent] = (
                                parent_shape, parent_dtype
                            )
                            states[parent] = 2
                            continue
                        stack.append((parent, False))
                    continue
                if states.get(current) == 2:
                    continue
                parent_records = [
                    (parent, *_shape_dtype_cache[parent])
                    for parent in parents
                    if parent in _shape_dtype_cache
                ]
                parent_shapes = [record[1] for record in parent_records]
                parent_dtypes = [record[2] for record in parent_records]
                if operation == "cast_like" and len(parents) >= 2:
                    # ``value.cast_like(reference)`` has the REFERENCE's
                    # element dtype (AbstractTensor.cast_like: the destination
                    # dtype is the reference operand's), not the domain's
                    # default.  Left at the default every float32 solve typed
                    # its one-hot selectors float64 while the formals they
                    # feed were float32.
                    reference_record = _shape_dtype_cache.get(parents[1])
                    if reference_record is not None and reference_record[1]:
                        dtype = reference_record[1]
                if (
                    operation in {
                        "unsqueeze", "squeeze", "expand", "broadcast_to",
                        "sum", "prod", "min", "max", "cumsum", "cumprod",
                    }
                    and not tensor.get("dtype")
                ):
                    # These keep their VALUE operand's element dtype (the
                    # descriptor rules above read it from the operand).  With
                    # no dtype posted on the node, the domain default
                    # (float64) is not a declaration: take the posted
                    # descriptor's dtype, else the operand's own record.
                    posted = _tensor_descriptor(graph, int(current))
                    value_parent = next((
                        int(parent)
                        for parent, role in (_node.get("parents") or ())
                        if str(role).casefold() in {
                            "operand", "value", "base", "input", "self",
                            "receiver",
                        }
                        and int(parent) in _shape_dtype_cache
                    ), next((p for p in parents[:1]), None))
                    if posted is not None and posted.get("dtype"):
                        dtype = str(posted["dtype"])
                    elif value_parent in _shape_dtype_cache:
                        dtype = _shape_dtype_cache[value_parent][1]
                # DomainNode historically pads scalars with unit axes. If it
                # says scalar but a canonical tensor op consumes a shaped
                # tensor, carry that shape through.
                if operation in {
                    "neg", "abs", "sqrt", "exp", "log", "round", "trunc",
                    "floor", "ceil", "isfinite", "isnan", "isinf",
                    "logical_not", "tanh", "sin", "cos", "tan", "asin",
                    "acos", "atan", "sinh", "cosh", "asinh", "acosh",
                    "atanh", "sign", "add", "sub", "mul", "truediv",
                    "floordiv", "mod", "pow", "less", "less_equal",
                    "greater", "greater_equal", "equal", "not_equal",
                    "maximum", "minimum", "where", "float", "double",
                    "long", "int", "to", "to_dtype", "astype", "cbrt",
                    "zeros_like", "empty_like", "ones_like", "full_like",
                }:
                    shaped = [candidate for candidate in parent_shapes if candidate]
                    if shaped:
                        try:
                            logical = tuple(np.broadcast_shapes(*shaped))
                        except ValueError:
                            # Preserve the declared descriptor for an invalid
                            # expression.  Kernel lowering will report the
                            # incompatible operands precisely; choosing one
                            # tuple here would fabricate a region ABI.
                            pass
                    if parent_dtypes and not tensor.get("dtype"):
                        # A literal Python float is a weak operand: it names
                        # no element width, so beside a typed float operand
                        # it does not widen the result (``float32 * 2.0`` is
                        # float32, as in the eager lane).  The float rank of
                        # the result is the widest the operands DECLARE; an
                        # explicit ``float``/``double`` cast is float64 by
                        # its own source.
                        weak_literals = {
                            parent for parent, _s, _d in parent_records
                            if declared(parent)[5] in {"const", "constant"}
                            and isinstance(
                                declared(parent)[2].get(
                                    "value", declared(parent)[0].get("constant")
                                ),
                                float,
                            )
                        }
                        numeric_dtypes = {
                            "int" if candidate in {"i32", "int32"}
                            else "int64" if candidate == "i64"
                            else "float64" if candidate in {
                                "float", "double", "f64",
                            }
                            else str(candidate)
                            for parent, _s, candidate in parent_records
                            if parent not in weak_literals
                        }
                        # The declared promotion rule is the precision
                        # layer's: the wider element type wins.
                        declared_float = _widest_element(*numeric_dtypes)
                        if declared_float is None and weak_literals:
                            # Only literal floats are float: the literal's
                            # own default width is all the source says.
                            declared_float = "float64"
                        if declared_float is not None:
                            dtype = (
                                "float64"
                                if operation in {"float", "double"}
                                else declared_float
                            )
                        elif "int64" in numeric_dtypes:
                            dtype = "int64"
                        elif "int" in numeric_dtypes:
                            dtype = "int"
                        elif "bool" in numeric_dtypes:
                            dtype = "bool"
                if operation in {
                    "sum", "prod", "min", "max", "any", "all",
                    "unsqueeze", "squeeze",
                    # ``cast_like`` keeps its VALUE's extents (its reference
                    # operand only names the dtype), so broadcasting the
                    # parents' shapes would answer the reference's; a scan
                    # keeps its operand's.  Both are settled by the
                    # descriptor.
                    "cast_like", "cumsum", "cumprod",
                }:
                    # The result extents of a reduction or an axis insertion
                    # are settled by the graph's tensor descriptor
                    # (``_tensor_descriptor``, which also posts them on the
                    # book) -- the same record ``value_ranks`` below reads.
                    # Re-deriving them here read only an ``axis``/``dim``
                    # ATTRIBUTE: the orbital quadrature's positional
                    # ``t26.sum(t3)`` became rank 0 while the descriptor said
                    # (4,), and ``t1.unsqueeze(t2)`` kept its source's (5,)
                    # while the descriptor said (5, 1).
                    settled = _tensor_descriptor(graph, int(current))
                    if settled is not None and descriptor_states_a_shape(settled):
                        logical = tuple(int(e) for e in (settled.get("shape") or ()))
                        _shape_dtype_cache[current] = (logical, dtype)
                        states[current] = 2
                        continue
                if operation in {"sum", "prod", "min", "max", "any", "all"}:
                    axis = attributes.get("axis", attributes.get("dim"))
                    source_shape = next(
                        (candidate for candidate in parent_shapes if candidate),
                        (),
                    )
                    if axis is None:
                        logical = ()
                    elif source_shape:
                        normalized = int(axis) % len(source_shape)
                        logical = (
                            source_shape[:normalized]
                            + source_shape[normalized + 1:]
                        )
                if operation == "matmul" and len(parent_shapes) >= 2:
                    left_shape, right_shape = parent_shapes[:2]
                    if (
                        len(left_shape) >= 2
                        and len(right_shape) >= 2
                        and left_shape[-1] == right_shape[-2]
                    ):
                        try:
                            batch_shape = tuple(np.broadcast_shapes(
                                left_shape[:-2], right_shape[:-2]
                            ))
                        except ValueError:
                            pass
                        else:
                            logical = (
                                *batch_shape, left_shape[-2], right_shape[-1]
                            )
                _shape_dtype_cache[current] = (logical, dtype)
                states[current] = 2
            logical, dtype = _shape_dtype_cache.get(
                root, declared(root)[3:5]
            )
            return (root, logical, dtype)

        value_shapes = tuple(
            _value_shape_dtype(value_id)
            for value_id in (*region_nodes, *region_captures)
        )
        value_ranks = tuple(
            (
                int(value_id),
                int((descriptor or {}).get(
                    "rank", len(tuple((descriptor or {}).get("shape") or ()))
                )),
            )
            for value_id in (*region_nodes, *region_captures)
            for descriptor in (_tensor_descriptor(graph, int(value_id)),)
            if descriptor is not None
        )
        # A capture produced by a shape-only operation outside the region is
        # fed its source's storage (the operation has no runtime existence);
        # the formal is a view of it (``PlanClosure.value_views``).
        value_views = []
        for value_id in region_captures:
            capture = graph.G.nodes.get(int(value_id), {})
            capture_operation = str(
                (capture.get("attributes") or {}).get("tensor_operation")
                or capture.get("op") or capture.get("type") or ""
            ).casefold()
            if capture_operation not in _SHAPE_ONLY_VIEW_OPERATIONS:
                continue
            storage = next((
                int(parent) for parent, role in capture.get("parents") or ()
                if str(role).casefold() in {
                    "operand", "value", "base", "input", "self", "receiver",
                } and int(parent) in graph.G
            ), None)
            descriptor = _tensor_descriptor(graph, int(value_id))
            if storage is None or descriptor is None or not descriptor_states_a_shape(descriptor):
                continue
            value_views.append((
                int(value_id), storage,
                tuple(int(e) for e in descriptor.get("shape") or ()),
                capture_operation,
            ))
        _concord_consumer_operands(graph, region_nodes, region_captures)
        _concord_item_operands(graph, region_captures)
        items.append(PlanClosure(
            name=f"region_{region_index}",
            captures=region_captures,
            items=(*const_lines, *compute_lines),
            value_shapes=value_shapes,
            value_ranks=value_ranks,
            value_views=tuple(value_views),
        ))
    control_values = set(_control_dependency_value_ids(
        getattr(shell, "shell_control_program", None)
    ))

    planned = PlanClosure(
        name=str(
            graph.G.graph.get("function_name")
            or type(shell).__name__
        ),
        captures=tuple(dict.fromkeys((
            *sorted(control_values),
            *(
            int(node_id)
            for node_id, data in graph.G.nodes(data=True)
            if data.get("type") == "Input"
            and graph.G.out_degree(node_id)
            ),
        ))),
        items=tuple(items),
    )
    memo[id(shell)] = planned
    return planned


#: Shape-only operations (no runtime existence, alias in every target): the
#: likeness table's ``_SHAPE_ONLY`` set (``ssa_llvm_backend``).
def _shape_only_view_operations() -> frozenset[str]:
    from .ssa_llvm_backend import _SHAPE_ONLY

    return frozenset(_SHAPE_ONLY)


_SHAPE_ONLY_VIEW_OPERATIONS = _shape_only_view_operations()
