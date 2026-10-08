"""Callsite tensor specialization: propagating exact tensor descriptors from callsites into callee graphs (per-copy and cross-book), with the conditional-tuple and call-return member publishers and the child-signature memo."""

from __future__ import annotations

import ast
import copy
import hashlib
import os
import sys
import time
from typing import Any, Mapping

from .process_graph_value_ids import next_process_value_id
from ..common.tensors.topological_reducer import _append_operand, _set_operands
from .concordance_declarations import AGGREGATE_MEMBER as _AGGREGATE_MEMBER
from .process_graph_fusion import extract_clean_process_subgraph
from .source_node_facts import _ast_source_signature, _constant_value
from .call_argument_binding import (
    _method_parameter_layout,
    _callsite_parameter_name,
    _authored_aggregate_leaves,
    _expanded_callsite_argument_edges,
)
from .region_scheduling import _dependency_order, _dependency_levels
from .formal_shape_ledger import _publish_formal_shape
from .planner_concordance_posts import (
    _argument_identity_cells,
    _post_copy_planner_specializations,
    _callsite_descriptor_receipt,
    _post_callsite_return_member,
)
from .source_control_retention import _source_control_records
from .tensor_descriptor_query import (
    _authored_aggregate_descriptor_tree,
    _invalidate_tensor_descriptor_dependents,
    _structured_output_descriptor,
    descriptor_states_a_shape,
    _tensor_descriptor,
)
from .callsite_literal_specialization import (
    _source_static_value,
    _source_static_literal,
)
from .conditional_control_programs import _branch_compartments
from .structural_fold import (
    _fold_callsite_structural_values,
    _retarget_all_cached_value_ids,
)
from .callsite_descriptor_application import (
    _apply_callsite_tensor_descriptors,
    _apply_callsite_aggregate_descriptors,
    _expand_specialized_unbroadcast_identity,
)


def _source_fallthrough_predicate_facts(graph: Any, controls, predicate_fact):
    """Facts reaching source expressions, excluding paths which already exit.

    Predicates are graph value identities. Loop exits and exception handlers
    do not export newly inferred facts; nested function bodies are separate
    scopes. Unknown control remains conservative rather than guessing a path.
    """
    body = graph.G.graph.get("function_body")
    if not body:
        return None
    expressions = {}
    for node_id, data in graph.G.nodes(data=True):
        expression = data.get("expr_obj")
        if isinstance(expression, ast.AST):
            expressions.setdefault(_ast_source_signature(expression), []).append(int(node_id))
    predicates = {}
    for record in controls.values():
        if record.get("predicate_id") in graph.G:
            predicates[_ast_source_signature(record["expression"])] = int(record["predicate_id"])
    reached = {}

    def observed(node, facts):
        for node_id in expressions.get(_ast_source_signature(node), ()):
            reached[node_id] = facts if node_id not in reached else reached[node_id] & facts

    def branch_facts(node, facts, truth):
        predicate = predicates.get(_ast_source_signature(node))
        return facts if predicate is None else facts | {predicate_fact(predicate, truth)}

    def expression(node, facts):
        if node is None or isinstance(node, (ast.Lambda, ast.GeneratorExp)):
            return
        observed(node, facts)
        if isinstance(node, ast.IfExp):
            expression(node.test, facts)
            expression(node.body, branch_facts(node, facts, True))
            expression(node.orelse, branch_facts(node, facts, False))
        else:
            for child in ast.iter_child_nodes(node):
                if not isinstance(child, ast.stmt):
                    expression(child, facts)

    def sequence(statements, facts):
        for statement in statements:
            if facts is None:
                break
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            observed(statement, facts)
            if isinstance(statement, ast.If):
                expression(statement.test, facts)
                left = sequence(statement.body, branch_facts(statement, facts, True))
                right = sequence(statement.orelse, branch_facts(statement, facts, False))
                facts = right if left is None else left if right is None else left & right
            elif isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                for child in ast.iter_child_nodes(statement):
                    expression(child, facts)
                facts = None
            elif isinstance(statement, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)):
                for child in ast.iter_child_nodes(statement):
                    if not isinstance(child, ast.stmt):
                        expression(child, facts)
                sequence(statement.body, facts)
                sequence(getattr(statement, "orelse", ()), facts)
            elif isinstance(statement, (ast.Try, ast.TryStar)):
                sequence(statement.body, facts)
                sequence(statement.orelse, facts)
                sequence(statement.finalbody, facts)
                for handler in statement.handlers:
                    sequence(handler.body, facts)
            else:
                expression(statement, facts)
        return facts

    sequence(body, frozenset())
    return reached


def _publish_conditional_tuple_members(graph: Any) -> bool:
    """Publish position-matched tuple members as control-owned merge values."""
    changed = False
    memberships = _branch_compartments(graph)
    controls = _source_control_records(graph.G)

    def predicate_fact(value_id, truth):
        data = graph.G.nodes.get(int(value_id), {})
        if isinstance(data.get("expr_obj"), ast.UnaryOp) and isinstance(data["expr_obj"].op, ast.Not):
            operand = next((int(parent) for parent, role in data.get("parents", ())
                            if str(role) == "operand"), None)
            if operand is not None:
                return predicate_fact(operand, not truth)
        return int(value_id), bool(truth)

    fallthrough_facts = _source_fallthrough_predicate_facts(graph, controls, predicate_fact)
    for node_id, data in tuple(graph.G.nodes(data=True)):
        if not isinstance(data.get("expr_obj"), ast.IfExp):
            continue
        attributes = dict(data.get("attributes") or {})
        if attributes.get("conditional_member_bindings"):
            continue
        parents = {str(role): int(parent) for parent, role in data.get("parents", ())}
        if not all(parents.get(role) in graph.G for role in ("test", "body", "orelse")):
            continue
        arms = [graph.G.nodes[parents[role]].get("attributes") or {}
                for role in ("body", "orelse")]
        leaves = [_authored_aggregate_leaves(graph, parents[role]) for role in ("body", "orelse")]
        optional = None
        absent_arms = [
            graph.G.nodes[parents[role]].get("type") in {"Constant", "Const", "const"}
            and _constant_value(graph.G.nodes[parents[role]]) is None
            for role in ("body", "orelse")
        ]
        if sum(absent_arms) == 1:
            present_arm = 1 if absent_arms[0] else 0
            if arms[present_arm].get("aggregate_kind") != "tuple" or not leaves[present_arm]:
                continue
            descriptors = [_tensor_descriptor(graph, leaf) for leaf in leaves[present_arm]]
            if any(item is None for item in descriptors):
                continue
            required = predicate_fact(parents["test"], present_arm == 0)
            consumers = tuple(graph.G.successors(node_id))
            guarded = []
            for consumer in consumers:
                facts = {
                    predicate_fact(controls[control]["predicate_id"], arm == "body")
                    for control, arm in memberships.get(int(consumer), ())
                    if control in controls and controls[control].get("predicate_id") in graph.G
                    and arm in {"body", "orelse"}
                }
                if fallthrough_facts is not None:
                    facts = fallthrough_facts.get(int(consumer), frozenset())
                if required not in facts:
                    raise ValueError(
                        f"optional tuple {node_id} consumer {consumer} has no "
                        "proven presence guard in function "
                        f"{graph.G.graph.get('function_name')!r}"
                    )
                guarded.append(int(consumer))
            if not consumers or any(node_id in values for values in
                                    (graph.G.graph.get("return_slot_values") or {}).values()):
                raise ValueError(f"optional tuple {node_id} escapes without a presence ABI")
            inactive = []
            for descriptor in descriptors:
                value_id = next_process_value_id(graph)
                graph.G.add_node(value_id, type="Constant", op="const", value_id=value_id,
                                 constant=0, parents=[], children=[],
                                 tensor={"shape": (), "dtype": "ptr" if descriptor["shape"] else descriptor["dtype"]},
                                 attributes={"value": 0, "conditional_result_of": int(node_id),
                                             "optional_inactive_payload": True})
                inactive.append(value_id)
            leaves[1 - present_arm] = tuple(inactive)
            optional = {"predicate_id": required[0], "present_when": required[1],
                        "guarded_consumers": tuple(guarded), "inactive_value_ids": tuple(inactive)}
            descriptors = [descriptors, descriptors]
        else:
            if any(arm.get("aggregate_kind") != "tuple" for arm in arms):
                continue
            descriptors = [[_tensor_descriptor(graph, leaf) for leaf in arm] for arm in leaves]
        if not leaves[0] or len(leaves[0]) != len(leaves[1]):
            continue
        if descriptors[0] != descriptors[1] or any(item is None for item in descriptors[0]):
            continue
        bindings = []
        for left, right, descriptor in zip(*leaves, descriptors[0]):
            member = next_process_value_id(graph)
            edges = [(left, "body"), (right, "orelse"), (parents["test"], "test")]
            graph.G.add_node(member, type="Phi", op="Phi", value_id=member,
                             parents=[], children=[], tensor=copy.deepcopy(descriptor),
                             attributes={"conditional_result_of": int(node_id)})
            _set_operands(graph, member, edges, cause=_AGGREGATE_MEMBER)
            _append_operand(graph, int(node_id), member, "elts")
            bindings.append((int(left), int(right), member))
        attributes.update({
            "producer_kind": "aggregate", "aggregate_kind": "tuple",
            "aggregate_leaf_value_ids": tuple(item[2] for item in bindings),
            "tensor_output_descriptors": tuple(copy.deepcopy(descriptors[0])),
            "conditional_member_bindings": tuple(bindings),
        })
        if optional is not None:
            attributes["optional_presence"] = optional
        data["attributes"] = attributes
        data.pop("tensor", None)
        changed = True
    return changed


def _publish_callsite_return_members(
    caller: Any, node_id: int, descriptors: Any, kind: str,
    return_cells: tuple = (),
) -> bool:
    """Materialize or enrich the exact projections of one aggregate call.

    Projection ids are stable identities.  Their descriptors are fixed-point
    facts and may become more precise after those ids are first materialized.
    Existing ids therefore receive the newer fact in place; node existence is
    not evidence that their metadata has settled.
    """

    node_id = int(node_id)
    data = caller.G.nodes[node_id]
    attributes = dict(data.get("attributes") or {})
    descriptors = tuple(copy.deepcopy(tuple(descriptors)))
    incumbent_leaves = tuple(map(
        int, attributes.get("aggregate_leaf_value_ids") or (),
    ))

    from .concordance_declarations import PLANNER_TENSOR_SPECIALIZATION
    from .identity_concordance import current_identity_book

    caller_name = str(caller.G.graph.get("function_name") or "")
    return_cells = tuple(return_cells)

    def record(index: int, leaf_id: int, action: str, descriptor: Any) -> None:
        # DERIVED from the callee return value's cells for this slot (its
        # identity cell and its shape-state cell): a refined member is a
        # changed source; a changed member from an unchanged source is not.
        _post_callsite_return_member(
            caller, node_id, int(index), int(leaf_id), descriptor,
            tuple(return_cells[index]) if index < len(return_cells) else (),
            stage=PLANNER_TENSOR_SPECIALIZATION,
        )

    if (
        len(incumbent_leaves) == len(descriptors)
        and incumbent_leaves
        and all(leaf_id in caller.G for leaf_id in incumbent_leaves)
    ):
        changed = False
        for index, (leaf_id, descriptor) in enumerate(zip(
            incumbent_leaves, descriptors,
        )):
            if not isinstance(descriptor, Mapping):
                continue
            replacement = copy.deepcopy(dict(descriptor))
            leaf = caller.G.nodes[leaf_id]
            if dict(leaf.get("tensor") or {}) == replacement:
                continue
            leaf["tensor"] = replacement
            record(index, leaf_id, "enriched", descriptor)
            _invalidate_tensor_descriptor_dependents(
                caller, (leaf_id,), "aggregate-return-projection-enriched",
            )
            changed = True
        if tuple(attributes.get("tensor_output_descriptors") or ()) != descriptors:
            attributes["tensor_output_descriptors"] = descriptors
            data["attributes"] = attributes
            changed = True
        return changed

    stale_leaves = tuple(
        leaf_id for leaf_id in incumbent_leaves if leaf_id not in caller.G
    )

    def resident_direct_projection(index: int) -> int | None:
        """Return the one authored ``call[index]`` value, when exact.

        Lexical tuple unpacking creates these projections before callsite
        return descriptors settle.  Reusing that identity is essential:
        loop/effect metadata already names it, so minting a parallel leaf
        makes the call publish one value while the effect consumes another.
        """

        candidates = []
        for successor in caller.G.successors(node_id):
            projection = caller.G.nodes[int(successor)]
            if str(
                projection.get("op") or projection.get("type") or ""
            ).casefold() != "indexed":
                continue
            if not any(
                int(parent) == node_id
                and str(role) in {"base", "value", "operand", "object"}
                for parent, role in projection.get("parents") or ()
            ):
                continue
            indices = tuple(
                int(parent)
                for parent, role in projection.get("parents") or ()
                if str(role) == "index" and int(parent) in caller.G
            )
            if len(indices) != 1:
                continue
            try:
                projected_index = _constant_value(
                    caller.G.nodes[indices[0]]
                )
            except KeyError:
                continue
            if (
                isinstance(projected_index, int)
                and not isinstance(projected_index, bool)
                and int(projected_index) == int(index)
            ):
                candidates.append(int(successor))
        unique = tuple(dict.fromkeys(candidates))
        return unique[0] if len(unique) == 1 else None

    # Materialize exact member projections of the authored return. These are
    # call results, never extra caller inputs or copies of the payload.
    leaves = []
    for index, descriptor in enumerate(descriptors):
        resident_projection = resident_direct_projection(index)
        if resident_projection is not None:
            member = caller.G.nodes[int(resident_projection)]
            member.setdefault("attributes", {})[
                "authored_call_result_projection"
            ] = True
            member["tensor"] = (
                {} if descriptor is None or isinstance(descriptor, tuple)
                else copy.deepcopy(dict(descriptor))
            )
            leaves.append(int(resident_projection))
            record(index, int(resident_projection), "reused", descriptor)
            continue
        index_id = next_process_value_id(caller)
        caller.G.add_node(index_id, type="Constant", op="const",
                          value_id=index_id, constant=index,
                          attributes={"value": index}, parents=[], children=[])
        member_id = next_process_value_id(caller)
        caller.G.add_node(
            member_id, type="Indexed", op="Indexed", value_id=member_id,
            parents=[], children=[],
            attributes={"authored_call_result_projection": True},
            # A member that is itself an aggregate has no tensor fact of its
            # own; its structure stays in the ordered output descriptors,
            # where the nested projection path reads it.
            tensor=(
                {} if descriptor is None or isinstance(descriptor, tuple)
                else copy.deepcopy(dict(descriptor))
            ),
        )
        _set_operands(
            caller, member_id, [(node_id, "base"), (index_id, "index")],
            cause=_AGGREGATE_MEMBER,
        )
        leaves.append(member_id)
        record(index, member_id, "materialized", descriptor)
    attributes.update({
        "authored_return_container": kind,
        "producer_kind": "aggregate", "aggregate_kind": kind,
        "aggregate_leaf_value_ids": tuple(leaves),
        "tensor_output_descriptors": descriptors,
    })
    if stale_leaves:
        # The old leaf ids are not merely descriptor cache entries.  Source
        # control can name them from cached loop/effect receipts even after a
        # specialized call republishes the same return slots under fresh
        # resident ids.  Concord the exact slot transition, then move every
        # cached occurrence through that recorded identity before exposing
        # the replacement ledger.  Otherwise ``append(result_record)`` can
        # retain the dead pre-specialization projection while the call itself
        # publishes the new one.
        projection_identity = current_identity_book().page(
            "callsite_projection_identity_concordance"
        )
        projection_scope = (
            caller_name,
            int(node_id),
            id(caller.G),
        )
        for index, (stale_id, replacement_id) in enumerate(zip(
            incumbent_leaves, leaves,
        )):
            if int(stale_id) == int(replacement_id):
                continue
            row = (projection_scope, int(index), int(stale_id))
            fact = int(replacement_id)
            incumbent = projection_identity.latest(row)
            if incumbent is not None and int(incumbent) != fact:
                raise ValueError(
                    "callsite projection identity concordance disagreement "
                    f"for {row!r}: recorded={incumbent!r}, "
                    f"replacement={fact!r}"
                )
            if incumbent is None:
                projection_identity.set(row, 0, fact)
            _retarget_all_cached_value_ids(
                caller, int(stale_id),
                int(projection_identity.latest(row)),
            )
            if os.environ.get("TURING_DEBUG_SEQUENCE_EFFECT_IDENTITY"):
                print(
                    "DEBUG-CALLSITE-PROJECTION-IDENTITY "
                    f"function={caller_name!r} callsite={int(node_id)} "
                    f"slot={int(index)} stale={int(stale_id)} "
                    f"resident={int(replacement_id)}",
                    file=sys.stderr,
                    flush=True,
                )
        attributes["aggregate_leaf_republication"] = {
            "replaced_missing_value_ids": stale_leaves,
            "replacement_value_ids": tuple(map(int, leaves)),
            "rule": "exact-stored-output-descriptors",
            "tie_policy": "incumbent",
        }
    data["attributes"] = attributes
    data.pop("tensor", None)
    return True


#: Per compile book: the (callee graph, signature) answers a callsite-
#: specialized copy's propagation has computed, and the callees whose answer
#: is being computed now.
_CHILD_SIGNATURE_RESULTS: dict[int, tuple[Any, dict]] = {}


_CHILD_CALLS_IN_PROGRESS: set[tuple] = set()


def _child_signature_results(book: Any) -> dict:
    held = _CHILD_SIGNATURE_RESULTS.get(id(book))
    if held is None or held[0] is not book:
        _CHILD_SIGNATURE_RESULTS.clear()
        held = _CHILD_SIGNATURE_RESULTS[id(book)] = (book, {})
    return held[1]


def _propagate_callsite_tensor_specializations(
    graph: Any,
    *,
    runtime_references: frozenset[int] | None = None,
    publish_to_function_table: bool = True,
    _progress: Any = None,
) -> None:
    """Carry consistent tensor descriptors through the function table.

    This is the descriptor analogue of literal planner specialization.  It
    settles to a fixed point so a root call can shape ``bw_matmul`` and that
    shaped wrapper can in turn shape ``matmul_vjp`` before either function's
    logical control is partitioned. Conflicting callsite shapes remain
    parametric and are left for per-callsite specialization.
    """

    function_table = getattr(graph, "function_table", None)
    if function_table is None:
        return
    # A source-catalogue graph owns the table but is not itself an executable
    # caller.  Its per-definition graphs contain the actual parameter ABI,
    # call edges, and numerical dependencies.  Including the catalogue here
    # both duplicates all callsites and lets tuple/return publication rewrite
    # a map container which the structural-fold stage explicitly excludes.
    graphs = (
        [graph] if graph.G.graph.get("function_name") else []
    )
    graphs.extend(
        entry.graph
        for entry in function_table
        if entry.graph is not None
        # A root that is itself an executable function is registered in its
        # own table: it is already ``graph`` above, and a second visit of
        # the same callsite in one round would post its decision row twice.
        and entry.graph is not graph
        and (
            runtime_references is None
            or int(entry.reference.address) in runtime_references
        )
    )

    #: (caller graph, call node) -> per-slot callee return value cells of
    #: the latest ``call_result_descriptor`` answer for that call.
    return_cells_by_call: dict[tuple[int, int], tuple] = {}

    from .concordance_declarations import (
        CALLSITE_DESCRIPTOR_REUSE, COMPILE_POLICY, POLICY_DECLARATION,
        PLANNER_TENSOR_SPECIALIZATION,
    )
    from .identity_concordance import (
        Derived, Mode, Novel, current_identity_book, fork_shape_scope,
        shape_scope_of,
    )
    from .work_contract import active_contract

    # The work contract's callsite_descriptor_reuse policy, declared on the
    # book as a root row before the first decision reads it: a different
    # policy arriving mid-compile is a disagreement the book refuses.
    reuse_mode = str(active_contract().callsite_descriptor_reuse)
    _book = current_identity_book()
    _book.post(
        COMPILE_POLICY, ("callsite_descriptor_reuse",), reuse_mode,
        stage=PLANNER_TENSOR_SPECIALIZATION,
        provenance=Novel(POLICY_DECLARATION, ()), mode=Mode.CONCORD,
    )
    reuse_policy_cell = _book.latest_ref(
        COMPILE_POLICY, ("callsite_descriptor_reuse",),
    )
    #: This round's (callee graph id, signature) -> (callee graph, output
    #: descriptors, output cells, the computing callsite's decision cell).
    #: Cleared at every round: the callee graphs change between rounds.
    #: A callsite-specialized copy's call edges (``publish_to_function_table``
    #: False) reach callees whose own call edges are not settled on the
    #: function table either, so each propagation copy settles its own
    #: (below); that nested answer is a function of the same inputs, so the
    #: memo outlives the round and the invocation, per book.
    signature_results: dict[tuple, tuple] = (
        {} if publish_to_function_table
        else _child_signature_results(_book)
    )

    def call_result_descriptor(
        caller: Any,
        node_id: int,
        callee: Any,
    ) -> tuple[dict[str, Any] | None, ...]:
        """Infer the ordered tensor returns of an exact callsite."""

        receiver, positional, _all = _method_parameter_layout(callee.G)
        descriptors: dict[str, dict[str, Any]] = {}
        aggregate_descriptors: dict[str, tuple[Any, ...]] = {}
        specializations: dict[str, Any] = {}
        specialization_cells: dict[str, tuple] = {}
        bound_parameters: set[str] = set()
        for parent, role_value in _expanded_callsite_argument_edges(
            caller, int(node_id),
        ):
            role = str(role_value)
            parameter = _callsite_parameter_name(
                role, receiver, positional,
            )
            if parameter is None:
                continue
            parameter = str(parameter)
            bound_parameters.add(parameter)
            descriptor = _tensor_descriptor(caller, int(parent))
            if descriptor is not None:
                descriptors[parameter] = descriptor
            static_argument = _source_static_value(caller, int(parent))
            leaves = _authored_aggregate_leaves(caller, int(parent))
            if leaves and not static_argument and all(
                int(leaf) in caller.G for leaf in leaves
            ):
                # Same member-identity rule as callsite specialization: the
                # callee's return descriptors may depend on aggregate members
                # (``previous_hub, ... = history``), so the propagation copy
                # must see those members bound, not an opaque formal.
                aggregate_descriptors[parameter] = tuple(
                    _authored_aggregate_descriptor_tree(caller, int(leaf))
                    for leaf in leaves
                )
            if static_argument:
                try:
                    specializations[parameter] = _source_static_literal(
                        caller, int(parent)
                    )
                except ValueError:
                    pass
                else:
                    specialization_cells[parameter] = (
                        _argument_identity_cells(caller, int(parent))
                    )
        for parameter, default in (
            callee.G.graph.get("parameter_defaults") or {}
        ).items():
            if str(parameter) not in bound_parameters:
                specializations[str(parameter)] = copy.deepcopy(default)
        if not descriptors and not aggregate_descriptors and not callee.G.graph.get("parameter_record_abi"):
            return ()

        def any_descriptor(item: Any) -> bool:
            if isinstance(item, tuple):
                return any(any_descriptor(member) for member in item)
            return item is not None

        def record_and_return(
            output_descriptors: Any, output_cells: Any,
        ) -> tuple[dict[str, Any] | None, ...]:
            # Record this round's decision in the concordance.  The
            # specialization fixed point below has no bound of its own: it
            # repeats until nothing changes.  Writing each round's published
            # return shape to its own page means
            # ``IdentityPage.oscillating_rows`` can name a row that leaves a
            # shape and comes back to it -- a round trip, which a settling
            # fixed point never makes -- instead of the whole compile merely
            # looking slow from outside.
            _page = _book.page("callsite_return_specialization")
            _row = (
                shape_scope_of(caller), shape_scope_of(callee), int(node_id),
            )
            _page.set(_row, len(_page.history(_row)), tuple(
                None if item is None or not isinstance(item, Mapping)
                else (tuple(item.get("shape") or ()), str(item.get("dtype") or ""))
                for item in output_descriptors
            ))
            return_cells_by_call[(id(caller.G), int(node_id))] = tuple(
                output_cells
            )
            return (
                tuple(copy.deepcopy(output_descriptors))
                if any(any_descriptor(item) for item in output_descriptors)
                else ()
            )

        # The callee copy below is a function of exactly these inputs: the
        # callee graph, the argument descriptors, the aggregate member
        # descriptors, the literal specializations and the callee's declared
        # record ABI.  Under ``callsite_descriptor_reuse="signature"`` (the
        # work contract) the first callsite of a round with this signature
        # derives the copy and every later one in the same round reads its
        # answer: the same descriptors, and the same callee return cells for
        # the member publication to derive from.  ``"callsite"`` derives a
        # copy per callsite, as every round did before.  Either way the
        # decision is a row: ``callsite_descriptor_reuse`` (caller, callee,
        # call, round) = ("computed"|"reused", signature digest), DERIVED
        # from the policy cell (and, for a reuse, the computing callsite's
        # row).  A generator callee publishes its yield rows on the caller's
        # call value while deriving, so it is always derived per callsite.
        signature = (
            _stable_signature_value(descriptors),
            _stable_signature_value(aggregate_descriptors),
            _stable_signature_value(specializations),
            _stable_signature_value(
                callee.G.graph.get("parameter_record_abi") or {}
            ),
        )
        reuse_key = (id(callee.G), signature)
        signature_digest = hashlib.sha256(
            repr(signature).encode("utf-8")
        ).hexdigest()[:12]
        reuse_row = (
            shape_scope_of(caller), shape_scope_of(callee),
            int(node_id), int(round_index),
        )
        held = (
            signature_results.get(reuse_key)
            if reuse_mode == "signature"
            and not callee.G.graph.get("generator_stream")
            else None
        )
        if held is not None and held[0] is callee.G:
            _book.post(
                CALLSITE_DESCRIPTOR_REUSE, reuse_row,
                ("reused", signature_digest),
                stage=PLANNER_TENSOR_SPECIALIZATION,
                provenance=Derived((reuse_policy_cell, held[3])),
                mode=Mode.CONCORD,
            )
            return record_and_return(copy.deepcopy(held[1]), tuple(held[2]))
        specialized = extract_clean_process_subgraph(callee, callee.G)
        # A propagation copy states the shapes of ITS callsite: its own shape
        # scope, never the callee's (or another callsite's).
        fork_shape_scope(specialized, "callsite_tensor_specialization")
        specialized.G.graph["planner_specializations"] = copy.deepcopy(
            specializations
        )
        # This propagation copy's own rows, under its forked scope (7.1).
        _post_copy_planner_specializations(
            specialized, specializations, specialization_cells,
        )
        specialized.G.graph["planner_tensor_descriptors"] = copy.deepcopy(
            descriptors
        )
        _apply_callsite_tensor_descriptors(specialized, descriptors)
        _apply_callsite_aggregate_descriptors(
            specialized, aggregate_descriptors,
        )
        if not _expand_specialized_unbroadcast_identity(specialized):
            _fold_callsite_structural_values(specialized)
        if not publish_to_function_table and reuse_key not in (
            _CHILD_CALLS_IN_PROGRESS
        ):
            # The copy's own call edges are settled the same way (a
            # recursive callee is entered once: its re-entry answers
            # nothing, never loops).
            _CHILD_CALLS_IN_PROGRESS.add(reuse_key)
            try:
                _propagate_callsite_tensor_specializations(
                    specialized,
                    runtime_references=frozenset(),
                    publish_to_function_table=False,
                )
            finally:
                _CHILD_CALLS_IN_PROGRESS.discard(reuse_key)
        output_descriptors: list[dict[str, Any] | None] = []
        # Per published slot, the callee return value's cells: its identity
        # cell and its shape-state cell, read after the descriptor query
        # that (re)states it.  The member publication derives from these.
        output_cells: list[tuple] = []

        def return_value_cells(value_id: int) -> tuple:
            from ..common.tensors.topological_reducer import node_identity_cell
            from .identity_concordance import (
                SHAPE_STATE_PAGE, current_identity_book, shape_scope_of,
            )

            cells = []
            try:
                cells.append(node_identity_cell(specialized, int(value_id)))
            except ValueError:
                pass
            state = current_identity_book().latest_ref(SHAPE_STATE_PAGE, (
                shape_scope_of(specialized),
                int(specialized.G.nodes[int(value_id)].get(
                    "value_id", value_id,
                )),
            ))
            if state is not None:
                cells.append(state)
            return tuple(cells)
        identities = specialized.G.graph.get("identity_table") or {}
        for output_name in specialized.G.graph.get("function_outputs") or ():
            output_ids = tuple(identities.get(str(output_name), ()))
            value_id = next((
                int(candidate)
                for candidate in reversed(output_ids)
                if int(candidate) in specialized.G
            ), None)
            if value_id is None:
                output_descriptors.append(None)
                output_cells.append(())
                continue
            nested = (
                specialized.G.nodes[value_id].get("attributes") or {}
            ).get("tensor_output_descriptors")
            if nested:
                output_descriptors.extend(
                    copy.deepcopy(tuple(nested))
                )
                nested_cells = return_value_cells(value_id)
                output_cells.extend(nested_cells for _item in tuple(nested))
            else:
                described_id = value_id
                descriptor = _structured_output_descriptor(
                    specialized, value_id,
                )
                if not descriptor_states_a_shape(descriptor):
                    # The identity table lists every value that IS this
                    # binding; they are one storage by construction.  The last
                    # resident one is what the return names, but a store or a
                    # loop-carried port among them can answer the descriptor
                    # query with nothing at all.  Read the shape from whichever
                    # identity states it rather than publish a factless return
                    # that every caller installs as a rank-0 scalar.
                    for candidate in reversed(output_ids):
                        if int(candidate) == value_id:
                            continue
                        alternative = _structured_output_descriptor(
                            specialized, int(candidate),
                        )
                        if isinstance(alternative, Mapping) and (
                            descriptor_states_a_shape(alternative)
                        ):
                            descriptor = alternative
                            described_id = int(candidate)
                            break
                output_descriptors.append(descriptor)
                output_cells.append(return_value_cells(described_id))

        # A generator returns no value; its product is the row it yields.
        # Describe every yield site's tuple members on this exact callsite
        # specialization, require the sites to agree per column, and publish
        # the row on the caller's call value through the concordance, so a
        # ``for a, b, c in gen(...)`` target reads its column in the graph
        # phase instead of acquiring a shape only after SSA linking.
        generator_stream = (
            specialized.G.graph.get("generator_stream")
            or callee.G.graph.get("generator_stream")
            or {}
        )
        yield_rows: list[tuple[dict[str, Any] | None, ...]] = []
        for yield_node in generator_stream.get("yield_nodes") or ():
            if int(yield_node) not in specialized.G or not isinstance(
                specialized.G.nodes[int(yield_node)].get("expr_obj"),
                ast.Yield,
            ):
                continue
            yielded = next((
                int(parent)
                for parent, role in (
                    specialized.G.nodes[int(yield_node)].get("parents") or ()
                )
                if str(role) in {"value", "operand"}
                and int(parent) in specialized.G
            ), None)
            if yielded is None:
                continue
            yielded_data = specialized.G.nodes[yielded]
            members = (
                tuple(
                    int(parent)
                    for parent, _role in yielded_data.get("parents") or ()
                    if int(parent) in specialized.G
                )
                if isinstance(yielded_data.get("expr_obj"), ast.Tuple)
                else (yielded,)
            )
            yield_rows.append(tuple(
                _tensor_descriptor(specialized, member) for member in members
            ))
        if yield_rows and len({len(row) for row in yield_rows}) == 1:
            from .identity_concordance import record_shape_transformation

            caller_scope = shape_scope_of(caller)
            callee_scope = shape_scope_of(specialized)
            call_value = int(
                caller.G.nodes[int(node_id)].get("value_id", node_id)
            )
            for column, members in enumerate(zip(*yield_rows)):
                # Only a column every yield site states is a fact; a record
                # handle or a not-yet-described member claims nothing (an
                # empty shape would read as a proven scalar downstream).
                if not all(
                    isinstance(member, Mapping)
                    and descriptor_states_a_shape(member)
                    for member in members
                ):
                    continue
                facts = {
                    (
                        tuple(map(int, member.get("shape") or ())),
                        str(member.get("dtype") or "unknown"),
                    )
                    for member in members
                }
                if len(facts) != 1:
                    continue
                shape, dtype = next(iter(facts))
                state = {"shape": shape, "dtype": dtype, "rank": len(shape)}
                record_shape_transformation(
                    callee_scope, ("yield", column),
                    caller_scope, ("yield_row", call_value, column),
                    stage="callsite_yield_observation",
                    operation="yield_row",
                    source_state=state,
                    target_state=state,
                    role=f"column:{column}",
                )

        computing_cell = _book.post(
            CALLSITE_DESCRIPTOR_REUSE, reuse_row,
            ("computed", signature_digest),
            stage=PLANNER_TENSOR_SPECIALIZATION,
            provenance=Derived((reuse_policy_cell,)),
            mode=Mode.CONCORD,
        )
        if reuse_mode == "signature" and not callee.G.graph.get(
            "generator_stream"
        ):
            signature_results[reuse_key] = (
                callee.G, copy.deepcopy(output_descriptors),
                tuple(output_cells), computing_cell,
            )
        return record_and_return(output_descriptors, output_cells)

    round_index = 0
    seen_round_states: dict[str, int] = {}

    def specialization_state() -> tuple[str, int]:
        facts = []
        tensor_fact_count = 0
        for item in graphs:
            function_name = str(item.G.graph.get("function_name") or "")
            planner = item.G.graph.get("planner_tensor_descriptors") or {}
            facts.append((function_name, "planner", repr(planner)))
            for value_id, node in item.G.nodes(data=True):
                tensor = node.get("tensor")
                if tensor:
                    tensor_fact_count += 1
                    facts.append((
                        function_name,
                        int(value_id),
                        repr(_callsite_descriptor_receipt(tensor)),
                    ))
        digest = hashlib.sha256(repr(tuple(facts)).encode("utf-8")).hexdigest()
        return digest[:16], tensor_fact_count

    def _planner_progress(message: str) -> None:
        if _progress is not None:
            _progress({
                "specialization_state": "fixed-point-guard",
                "message": message,
            })

    from .bounded_fixed_point import BoundedFixedPoint, bound_for

    specialization_guard = BoundedFixedPoint(
        "callsite-specialization",
        bound_for(
            "planner",
            nodes=sum(item.G.number_of_nodes() for item in graphs),
            callsites=sum(
                1
                for item in graphs
                for _node_id, data in item.G.nodes(data=True)
                if (data.get("attributes") or {}).get("callee_ref")
                is not None
                or (data.get("attributes") or {}).get("method_ref")
                is not None
            ),
        ),
        scope=("whole-program",), progress=_planner_progress,
        state=specialization_state()[0],
    )
    changed = True
    while changed:
        round_index += 1
        # A round's reuse table is that round's: the callee graphs it was
        # derived on are mutated by the publications below.  (A specialized
        # copy's memo is not: it publishes nowhere.)
        if publish_to_function_table:
            signature_results.clear()
        round_started = time.monotonic()
        callsites = 0
        mutation_counts = {
            "conditional_members": 0,
            "return_members": 0,
            "single_returns": 0,
            "structured_returns": 0,
            "formal_shapes": 0,
            "callee_descriptors": 0,
            "settled_graph_shapes": 0,
        }
        if _progress is not None:
            _progress({
                "specialization_state": "round-begin",
                "round": round_index,
                "graphs": len(graphs),
                "graph_nodes": sum(item.G.number_of_nodes() for item in graphs),
            })
        changed = False
        # The outer catalogue graph owns the function table, but its entry
        # graphs own the parameter ABI and numerical dependencies.  Settle
        # each caller locally before reading its call arguments; otherwise a
        # shaped ``material_state[:, :, 3]`` still looks scalar here and only
        # a later backend happens to rediscover its extent after the call ABI
        # has already been frozen.
        for caller_index, caller in enumerate(graphs):
            if caller.G.graph.get("function_name"):
                fold_started = time.monotonic()
                if _progress is not None:
                    _progress({
                        "specialization_state": "prefold-begin",
                        "round": round_index,
                        "caller_index": int(caller_index),
                        "caller": str(caller.G.graph.get("function_name")),
                        "caller_nodes": caller.G.number_of_nodes(),
                    })
                _fold_callsite_structural_values(
                    caller,
                    _progress=(
                        None if _progress is None else
                        lambda facts, caller=caller: _progress({
                            "specialization_state": "prefold-fixed-point",
                            "round": round_index,
                            "caller": str(
                                caller.G.graph.get("function_name") or ""
                            ),
                            **facts,
                        })
                    ),
                )
                if _progress is not None:
                    _progress({
                        "specialization_state": "prefold-end",
                        "round": round_index,
                        "caller_index": int(caller_index),
                        "caller": str(caller.G.graph.get("function_name")),
                        "caller_nodes": caller.G.number_of_nodes(),
                        "duration_seconds": round(
                            time.monotonic() - fold_started, 6
                        ),
                    })
        candidates: dict[tuple[int, str], list[dict[str, Any]]] = {}
        for caller in graphs:
            conditional_changed = _publish_conditional_tuple_members(caller)
            changed |= conditional_changed
            mutation_counts["conditional_members"] += int(conditional_changed)
            for _node_id, data in tuple(caller.G.nodes(data=True)):
                attributes = data.get("attributes") or {}
                reference = attributes.get("callee_ref")
                if reference is None:
                    reference = attributes.get("method_ref")
                if reference is None:
                    continue
                try:
                    callee = function_table.entry(int(reference)).graph
                except (KeyError, TypeError, ValueError):
                    continue
                if callee is None:
                    continue
                callsites += 1
                if _progress is not None and (
                    callsites == 1 or callsites % 16 == 0
                ):
                    _progress({
                        "specialization_state": "callsite-progress",
                        "round": round_index,
                        "callsites": callsites,
                        "caller": str(
                            caller.G.graph.get("function_name") or ""
                        ),
                        "callee": str(
                            callee.G.graph.get("function_name") or reference
                        ),
                        "round_elapsed_seconds": round(
                            time.monotonic() - round_started, 6
                        ),
                    })
                # A call node can carry an early catalogue descriptor that is
                # no longer exact after its operands are specialized.  Calls
                # are equality edges, so recompute their return contract on
                # every fixed-point round and replace only when the exact
                # callee result differs.  The decision itself is recorded by
                # ``call_result_descriptor`` in the concordance.
                result_descriptors = call_result_descriptor(
                    caller, int(_node_id), callee
                )
                if result_descriptors:
                    if (
                        len(result_descriptors) == 1
                        and isinstance(result_descriptors[0], Mapping)
                    ):
                        exact_result = result_descriptors[0]
                        exact_shape = tuple(
                            exact_result.get("shape") or ()
                        )
                        if exact_shape:
                            from .identity_concordance import (
                                record_proven_shape,
                                record_shape_transformation,
                                shape_scope_of,
                            )

                            # Publish on every fixed-point observation, not
                            # only when the graph-node dictionary changes.
                            # Another callsite later in the same round may
                            # invalidate a dependent proof while leaving this
                            # exact call descriptor textually unchanged; the
                            # final stable round must restore the call-edge
                            # fact to the concordance.
                            record_proven_shape(
                                shape_scope_of(caller),
                                int(data.get("value_id", _node_id)),
                                exact_shape,
                                exact_result.get("dtype"),
                                int(_dependency_levels(caller).get(
                                    int(_node_id), 0,
                                )),
                            )
                            # The restoration is itself a morph of the call
                            # value: record it as the callee-return edge so
                            # anything derived from the call value while its
                            # proof was withdrawn is superseded along the
                            # transformation graph instead of keeping the
                            # gap's answer.
                            record_shape_transformation(
                                shape_scope_of(callee),
                                ("return",),
                                shape_scope_of(caller),
                                int(data.get("value_id", _node_id)),
                                stage="callsite_return_observation",
                                operation="call_result",
                                source_state=exact_result,
                                target_state=exact_result,
                                role="return",
                            )
                    return_kinds = set((callee.G.graph.get("return_container_kinds") or {}).values())
                    if (
                        len(return_kinds) == 1
                        and return_kinds <= {"tuple", "list"}
                        and result_descriptors
                        # An authored return may mix tensors with nested
                        # aggregates -- the vehicle tick returns fifteen
                        # tensors and one four-member history tuple.  Refusing
                        # the whole publication because one member is nested
                        # left every sibling projection unresolved, so each
                        # ``result[k]`` became a formal no caller could fill.
                        and all(item is None
                                or isinstance(item, (Mapping, tuple))
                                for item in result_descriptors)
                    ):
                        return_members_changed = _publish_callsite_return_members(
                            caller, int(_node_id), result_descriptors,
                            next(iter(return_kinds)),
                            return_cells=return_cells_by_call.get(
                                (id(caller.G), int(_node_id)), (),
                            ),
                        )
                        changed |= return_members_changed
                        mutation_counts["return_members"] += int(
                            return_members_changed
                        )
                    elif len(result_descriptors) == 1 and isinstance(
                        result_descriptors[0], Mapping
                    ):
                        # A single returned aggregate still has a nested
                        # descriptor tree. Only a tensor leaf belongs in
                        # the node's tensor slot; preserve aggregate returns
                        # in the structured output ledger below.
                        replacement = copy.deepcopy(dict(
                            result_descriptors[0]
                        ))
                        if dict(data.get("tensor") or {}) != replacement:
                            previous = copy.deepcopy(data.get("tensor"))
                            data["tensor"] = replacement
                            _invalidate_tensor_descriptor_dependents(
                                caller, (int(_node_id),),
                                "call-result-specialization-changed",
                            )
                            from .identity_concordance import (
                                current_identity_book,
                                record_shape_transformation,
                                record_proven_shape,
                                shape_scope_of,
                            )

                            # The call edge has just replaced a provisional
                            # descriptor with the exact specialized return.
                            # Invalidation withdraws facts derived from the
                            # old result; publish the replacement on the call
                            # value itself in the same causal operation.  A
                            # retained comprehension may use that exact value
                            # as its appended row, and sequence physicalization
                            # consults the shared proof rather than a private
                            # graph-node cache.
                            replacement_shape = tuple(
                                replacement.get("shape") or ()
                            )
                            if replacement_shape:
                                record_proven_shape(
                                    shape_scope_of(caller),
                                    int(data.get("value_id", _node_id)),
                                    replacement_shape,
                                    replacement.get("dtype"),
                                    int(_dependency_levels(caller).get(
                                        int(_node_id), 0,
                                    )),
                                )
                            record_shape_transformation(
                                shape_scope_of(callee),
                                ("return",),
                                shape_scope_of(caller),
                                int(data.get("value_id", _node_id)),
                                stage="callsite_return_specialization",
                                operation="call_result",
                                source_state=replacement,
                                target_state=replacement,
                                role="return",
                            )

                            mutation_page = current_identity_book().page(
                                "callsite_tensor_result_specialization"
                            )
                            mutation_row = (
                                shape_scope_of(caller), int(_node_id),
                            )
                            mutation_page.set(
                                mutation_row,
                                len(mutation_page.history(mutation_row)),
                                (
                                    _callsite_descriptor_receipt(previous),
                                    _callsite_descriptor_receipt(replacement),
                                ),
                            )
                            # Invalidation above withdrew every descriptor
                            # derived from the old call result.  Re-evaluate
                            # those exact transformation descendants only
                            # after installing the replacement, so a
                            # comprehension materializer reads the new row
                            # shape rather than immediately republishing the
                            # provisional scalar that was just retired.
                            def descriptor_dependents(source_id: int):
                                return tuple(
                                    int(candidate_id)
                                    for candidate_id, candidate
                                    in caller.G.nodes(data=True)
                                    if any(
                                        int(parent) == int(source_id)
                                        and str(role) not in {
                                            "callee", "func", "definition",
                                        }
                                        for parent, role in candidate.get(
                                            "parents", ()
                                        )
                                    )
                                )

                            pending_descriptor_nodes = list(
                                descriptor_dependents(int(_node_id))
                            )
                            seen_descriptor_nodes = {int(_node_id)}
                            while pending_descriptor_nodes:
                                dependent_id = int(
                                    pending_descriptor_nodes.pop(0)
                                )
                                if dependent_id in seen_descriptor_nodes:
                                    continue
                                seen_descriptor_nodes.add(dependent_id)
                                _tensor_descriptor(caller, dependent_id)
                                pending_descriptor_nodes.extend(
                                    descriptor_dependents(dependent_id)
                                )
                            changed = True
                            mutation_counts["single_returns"] += 1
                    elif result_descriptors:
                        attributes = dict(data.get("attributes") or {})
                        if attributes.get(
                            "tensor_output_descriptors"
                        ) != result_descriptors:
                            attributes["tensor_output_descriptors"] = (
                                result_descriptors
                            )
                            data["attributes"] = attributes
                            changed = True
                            mutation_counts["structured_returns"] += 1
                receiver, positional, _all = _method_parameter_layout(callee.G)
                for parent, role_value in _expanded_callsite_argument_edges(
                    caller, int(_node_id),
                ):
                    role = str(role_value)
                    parameter = _callsite_parameter_name(
                        role, receiver, positional,
                    )
                    descriptor = _tensor_descriptor(caller, int(parent))
                    if parameter is not None and descriptor is not None:
                        # This fixed point is the first stage that knows the
                        # exact caller value and the callee formal together.
                        # Publish that agreement now, under the authored
                        # identity, so recursive return specialization can
                        # recover the formal shape from every clean or hashed
                        # graph copy on its next round.  Keeping the fact only
                        # in ``planner_tensor_descriptors`` made it local to
                        # one catalogue graph; a fresh specialization then
                        # reverted the same formal to an unknown scalar.
                        # A rank-0 descriptor with a known dtype is a
                        # stated shape (``descriptor_states_a_shape``), not
                        # "recovery stopped".  Publishing only non-empty
                        # extents hid the disagreement between unbroadcast's
                        # G = () (bw_add) and G = (2, 3) (bw_mul) callsites:
                        # no FORMAL_SHAPE_CONFLICT, so every expansion copy
                        # read the shared ('unbroadcast', 1) row and the
                        # (2, 3) reshape answered (), which spun the
                        # return-member fixed point forever.
                        if publish_to_function_table and (
                            descriptor_states_a_shape(descriptor)
                        ):
                            formal_shape_changed = _publish_formal_shape(
                                str(callee.G.graph.get("function_name")),
                                str(parameter),
                                descriptor,
                                str(caller.G.graph.get("function_name")),
                            )
                            changed |= formal_shape_changed
                            mutation_counts["formal_shapes"] += int(
                                formal_shape_changed
                            )
                        candidates.setdefault(
                            (int(reference), str(parameter)), []
                        ).append(copy.deepcopy(dict(descriptor)))
                # A nested function reads bindings its caller owns, and those
                # captures are ordinary values of this exact caller.  The
                # callee had no contract for them, so every shape expression
                # over a capture -- ``wrench_k.shape[:2]`` inside
                # ``_wrench_force`` -- stayed unresolved and escaped as an
                # anonymous formal.  Offer the enclosing value's descriptor
                # under the capture's own name; the agreement check below
                # applies to it exactly as it does to a bound argument, so a
                # shared catalogue graph is only specialized when every
                # callsite names the same contract.
                caller_identities = caller.G.graph.get("identity_table") or {}
                for _capture_id, capture in callee.G.nodes(data=True):
                    capture_attributes = capture.get("attributes") or {}
                    if str(capture.get("type") or "") != "Input":
                        continue
                    if str(
                        capture_attributes.get("binding_kind") or ""
                    ) not in {"closure", "external"}:
                        continue
                    capture_name = str(
                        capture_attributes.get("binding_name") or ""
                    )
                    if not capture_name:
                        continue
                    capture_values = tuple(dict.fromkeys(
                        int(value_id)
                        for value_id in caller_identities.get(capture_name, ())
                        if int(value_id) in caller.G
                    ))
                    capture_descriptor = (
                        None if len(capture_values) != 1
                        else _tensor_descriptor(caller, capture_values[0])
                    )
                    if os.environ.get("TURING_DEBUG_CAPTURE_DESCRIPTOR"):
                        print(
                            "DEBUG-CAPTURE-DESCRIPTOR "
                            f"caller={caller.G.graph.get('function_name')!r} "
                            f"callee_ref={int(reference)} name={capture_name!r} "
                            f"values={capture_values!r} "
                            f"descriptor={capture_descriptor!r}",
                            file=sys.stderr, flush=True,
                        )
                    if capture_descriptor is None:
                        continue
                    candidates.setdefault(
                        (int(reference), capture_name), []
                    ).append(copy.deepcopy(dict(capture_descriptor)))

        # Calls are processed in source-graph order, but that order must not
        # decide which descendant proof survives.  Replacing one exact call
        # result invalidates every shape downstream from that identity.  A
        # later call in the same round can therefore withdraw a proof that an
        # earlier call already republished.  Once all call edges for the round
        # are known, walk each transformation graph in dependency order and
        # let the one concordance re-prove every derivable value.  This is a
        # settlement pass over the graph itself, not a parallel descriptor
        # cache; a newly recovered proof requests another callsite round so
        # callers and callees can observe it symmetrically.
        from .identity_concordance import (
            current_identity_book,
            proven_shape_contract_of,
            shape_scope_of,
        )

        settlement_page = current_identity_book().page(
            "tensor_shape_settlement_concordance"
        )

        for caller in graphs:
            function_name = shape_scope_of(caller)
            for value_id in _dependency_order(caller):
                data = caller.G.nodes[int(value_id)]
                semantic_id = int(data.get("value_id", value_id))
                before = proven_shape_contract_of(
                    function_name, semantic_id,
                )
                _tensor_descriptor(caller, int(value_id))
                after = proven_shape_contract_of(
                    function_name, semantic_id,
                )
                if before is None and after is not None:
                    # A later callsite in this round may invalidate a proof
                    # that this graph walk then restores. Restoration is
                    # necessary, but after the first settled receipt it is
                    # not a new fixed-point mutation. Keep that distinction
                    # on the one concordance: a local ``seen`` cache would
                    # fork identity authority and made identical graph state
                    # spin forever with ``changed=True``.
                    settlement_row = (function_name, int(semantic_id))
                    settlement_fact = (
                        tuple(map(int, after[0])), str(after[1]),
                    )
                    incumbent = settlement_page.latest(settlement_row)
                    if incumbent is None:
                        settlement_page.concord(
                            settlement_row, settlement_fact,
                        )
                        changed = True
                        mutation_counts["settled_graph_shapes"] += 1
                    elif tuple(incumbent) != settlement_fact:
                        raise ValueError(
                            "tensor shape settlement disagreement for "
                            f"{settlement_row!r}: recorded={incumbent!r}, "
                            f"proposed={settlement_fact!r}"
                        )
        by_reference: dict[int, dict[str, dict[str, Any]]] = {}
        for (reference, parameter), descriptors in candidates.items():
            if not descriptors or any(
                descriptor != descriptors[0] for descriptor in descriptors[1:]
            ):
                continue
            by_reference.setdefault(reference, {})[parameter] = descriptors[0]
        if not publish_to_function_table:
            # A callsite-specialized copy settles only its own call edges;
            # the shared function-table graphs belong to every callsite and
            # are not narrowed to one copy's arguments.
            by_reference = {}
        for reference, descriptors in by_reference.items():
            callee = function_table.entry(reference).graph
            if callee is None:
                continue
            existing = callee.G.graph.setdefault(
                "planner_tensor_descriptors", {}
            )
            additions = {
                name: descriptor
                for name, descriptor in descriptors.items()
                if existing.get(name) != descriptor
            }
            if not additions:
                continue
            existing.update(copy.deepcopy(additions))
            _apply_callsite_tensor_descriptors(callee, additions)
            # The catalogue graph is shared by every callsite.  A tensor
            # descriptor alone is not a complete structural specialization:
            # helpers such as ``unbroadcast(G, target_shape)`` also require a
            # callsite-local shape value before their source control can be
            # selected.  Folding the shared graph here permanently discarded
            # branches before that value was known.  The copied graph is
            # folded below by ``_callsite_specialized_shell_type`` after both
            # descriptor and literal arguments have been collected.
            changed = True
            mutation_counts["callee_descriptors"] += len(additions)

        state_digest, tensor_fact_count = specialization_state()
        repeated_round = seen_round_states.get(state_digest)
        seen_round_states.setdefault(state_digest, round_index)
        from .identity_concordance import current_identity_book

        return_page = current_identity_book().page(
            "callsite_return_specialization"
        )
        oscillating_rows = return_page.oscillating_rows()
        if _progress is not None:
            _progress({
                "specialization_state": "round-end",
                "round": round_index,
                "duration_seconds": round(
                    time.monotonic() - round_started, 6
                ),
                "changed": bool(changed),
                "callsites": callsites,
                "tensor_facts": tensor_fact_count,
                "state_digest": state_digest,
                "repeats_round": repeated_round,
                "oscillating_rows": len(oscillating_rows),
                **mutation_counts,
            })
        specialization_guard.round(changed, state=state_digest)


def _stable_signature_value(value: Any) -> object:
    """A hashable, order-independent spelling of a descriptor/specialization
    mapping, for keying one callsite signature against another (the same
    spelling ``_callsite_specialized_shell_type`` keys planned shells by)."""

    if isinstance(value, dict):
        return tuple(sorted(
            (_stable_signature_value(key), _stable_signature_value(item))
            for key, item in value.items()
        ))
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(_stable_signature_value(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value
