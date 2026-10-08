"""Structural fold: folding structural callsite values into the callee graph, with the program-ABI value facts, structural aliases and leaf-projection repairs it uses."""

from __future__ import annotations

import ast
import copy
import hashlib
import operator
import os
import sys
import numpy as np
from dataclasses import dataclass
from typing import Any, Mapping

from .process_graph_value_ids import next_process_value_id
from ..common.tensors.topological_reducer import _set_operands
from .concordance_declarations import (
    AGGREGATE_MEMBER as _AGGREGATE_MEMBER,
    CALLSITE_FOLD_LITERAL as _CALLSITE_FOLD_LITERAL,
    CALLSITE_FOLD_REMOVE_NODE as _CALLSITE_FOLD_REMOVE_NODE,
    CALLSITE_FOLD_REPLACE_ALIAS as _CALLSITE_FOLD_REPLACE_ALIAS,
)
from .loop_composer import _retarget_cached_value_ids
from .source_node_facts import _ast_source_signature, _constant_value
from .call_argument_binding import _positional_argument_index
from .region_scheduling import _dependency_order, _dependency_levels
from .planner_concordance_posts import (
    _post_if_changed,
    _post_control_specialization,
    _post_pruned_return_sites,
    _fold_literal_source_cells,
    _post_identity_table_mutation,
    _callsite_descriptor_receipt,
    _post_callsite_return_member,
)
from .receiver_resolution_prepasses import _resolve_grounded_method_references
from .source_control_retention import (
    _retain_source_control_records,
    _retain_source_sequence_mutation_records,
    _source_control_records,
)
from .tensor_descriptor_query import (
    _operand_descriptor,
    _DTYPE_CAST_OPERATIONS,
    descriptor_states_a_shape,
    _tensor_descriptor,
    _dtype_argument_name,
)
from .callsite_literal_specialization import _source_static_literal
from .conditional_control_programs import _branch_compartments


def _repair_missing_aggregate_leaf_projections(graph: Any) -> int:
    """Restore only aggregate leaves removed by structural branch folding.

    Aggregate leaf IDs are metadata edges.  A selected producer can survive
    while a projection associated with the rejected source arm is removed,
    leaving the producer's exact output descriptor but a dangling leaf ID.
    Keep every resident incumbent and recreate only missing positions from
    those stored descriptors.  A second visit is therefore a no-op.
    """

    repaired = 0
    for node_id, data in tuple(graph.G.nodes(data=True)):
        attributes = dict(data.get("attributes") or {})
        leaves = tuple(map(
            int, attributes.get("aggregate_leaf_value_ids") or (),
        ))
        if not leaves or all(leaf_id in graph.G for leaf_id in leaves):
            continue
        descriptors = tuple(
            attributes.get("tensor_output_descriptors") or (),
        )
        if len(descriptors) != len(leaves):
            continue
        replacements = []
        replaced = []
        for index, (leaf_id, descriptor) in enumerate(zip(
            leaves, descriptors,
        )):
            if leaf_id in graph.G:
                replacements.append(int(leaf_id))
                continue
            index_id = next_process_value_id(graph)
            graph.G.add_node(
                index_id, type="Constant", op="const", value_id=index_id,
                constant=index, attributes={"value": index},
                parents=[], children=[],
            )
            member_id = next_process_value_id(graph)
            graph.G.add_node(
                member_id,
                type="Indexed", op="Indexed", value_id=member_id,
                parents=[],
                children=[],
                attributes={
                    "authored_call_result_projection": True,
                    "aggregate_leaf_republication": True,
                },
                tensor=(
                    {} if descriptor is None
                    else copy.deepcopy(dict(descriptor))
                ),
            )
            _set_operands(
                graph, member_id, [(int(node_id), "base"), (index_id, "index")],
                cause=_AGGREGATE_MEMBER,
            )
            replacements.append(member_id)
            replaced.append((int(leaf_id), int(member_id)))
            repaired += 1
        if replaced:
            from .identity_concordance import current_identity_book

            projection_identity = current_identity_book().page(
                "callsite_projection_identity_concordance"
            )
            projection_scope = (
                str(graph.G.graph.get("function_name") or ""),
                int(node_id),
                id(graph.G),
            )
            replacement_by_stale = dict(replaced)
            for index, stale_id in enumerate(leaves):
                replacement_id = replacement_by_stale.get(int(stale_id))
                if replacement_id is None:
                    continue
                row = (projection_scope, int(index), int(stale_id))
                incumbent = projection_identity.latest(row)
                if (
                    incumbent is not None
                    and int(incumbent) != int(replacement_id)
                ):
                    raise ValueError(
                        "callsite projection identity concordance "
                        f"disagreement for {row!r}: "
                        f"recorded={incumbent!r}, "
                        f"replacement={replacement_id!r}"
                    )
                if incumbent is None:
                    projection_identity.set(
                        row, 0, int(replacement_id)
                    )
                _retarget_all_cached_value_ids(
                    graph, int(stale_id),
                    int(projection_identity.latest(row)),
                )
                if os.environ.get(
                    "TURING_DEBUG_SEQUENCE_EFFECT_IDENTITY"
                ):
                    print(
                        "DEBUG-CALLSITE-PROJECTION-IDENTITY "
                        f"function={projection_scope[0]!r} "
                        f"callsite={int(node_id)} slot={int(index)} "
                        f"stale={int(stale_id)} "
                        f"resident={int(replacement_id)}",
                        file=sys.stderr,
                        flush=True,
                    )
        attributes["aggregate_leaf_value_ids"] = tuple(replacements)
        attributes["aggregate_leaf_republication"] = {
            "replacements": tuple(replaced),
            "rule": "exact-stored-output-descriptors",
            "tie_policy": "incumbent",
        }
        data["attributes"] = attributes
    return repaired


_UNRESOLVED_STRUCTURAL_VALUE = object()


@dataclass(frozen=True)
class _StructuralValueAlias:
    source_id: int


@dataclass(frozen=True)
class _ProgramABIValueFact:
    """Declared source type attached to a physical native parameter."""

    python_type: str
    storage: str
    dtype: str | None
    # A keyed container can return a schema-known record row.  This is a type
    # fact about the lookup result, not a second identity for the lookup or its
    # deterministic key.
    value_python_type: str | None = None
    value_storage: str | None = None
    token_vocabulary: tuple[str, ...] | None = None
    shape: tuple[int, ...] | None = None
    # Logical tensor rank is distinct from the flat span used to carry its
    # elements.  ``None`` means the annotation did not state a rank; tensor
    # operations may constrain it without inventing extents.
    rank: int | None = None
    value_rank: int | None = None
    # Optional storage always has a physical payload, but that payload is only
    # meaningful while its separately owned presence value is true.  Keeping
    # that distinction on the structural fact prevents ``payload is not
    # None`` from being folded merely because native storage exists.
    optional: bool = False


def _contains_program_abi_fact(value: Any) -> bool:
    """Return whether a structural value carries a ProgramABI fact anywhere.

    A fact describes runtime storage/type; it is not a literal.  An authored
    aggregate whose members are facts (``history = (hub, angle)`` over span
    parameters) is therefore a dataflow aggregate, never a foldable constant.
    """

    if isinstance(value, _ProgramABIValueFact):
        return True
    if isinstance(value, Mapping):
        return any(
            _contains_program_abi_fact(key) or _contains_program_abi_fact(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_contains_program_abi_fact(item) for item in value)
    return False


def _fold_callsite_structural_values(
    graph: Any, *, _progress: Any = None,
) -> None:
    """Fold only whitelisted structural Python identities from graph facts.

    This pass is deliberately smaller than Python evaluation.  It consumes
    constants and compiler-owned tensor descriptors, and recognizes the
    structural builtins used to express shapes and source control.  It never
    invokes a retained Python callable or reads a runtime tensor payload.
    """

    # Source conditionals are authored control, not disposable dataflow.
    # Preserve their AST identity and predicate correlation before this pass
    # folds/removes structural nodes so later region overlay can reconstruct
    # the exact branch tree in each function shell.
    _retain_source_control_records(graph.G)
    _retain_source_sequence_mutation_records(graph.G)
    # Resolve capability before getattr(..., default) can select its default.
    _resolve_grounded_method_references(graph)

    unresolved = _UNRESOLVED_STRUCTURAL_VALUE
    known: dict[int, Any] = {}
    topology_changed = False
    mutated_sequence_ids = {
        int(record["sequence_value_id"])
        for record in (
            graph.G.graph.get("source_sequence_mutation_records") or {}
        ).values()
    }
    loop_carried_initial_ids = {
        int(initial)
        for _loop_id, data in graph.G.nodes(data=True)
        for initial, _updated in (
            (data.get("attributes") or {}).get(
                "loop_carried_bindings", {}
            ).values()
        )
    }

    def constant(data: Mapping[str, Any]) -> Any:
        try:
            return _constant_value(data)
        except KeyError:
            return unresolved

    def positional(data: Mapping[str, Any]) -> tuple[int, ...]:
        indexed = []
        for parent, role_value in data.get("parents") or ():
            position = _positional_argument_index(str(role_value))
            if position is not None:
                indexed.append((position, int(parent)))
        return tuple(parent for _position, parent in sorted(indexed))

    def exact_return_formal_actual(
        node_id: int, data: Mapping[str, Any],
    ) -> int | None:
        """Map a one-result call returning a formal to its exact actual.

        Return-slot identity is available before SSA linking.  Reading it here
        lets structural facts attached to an argument (in particular a
        ProgramABI record and its declared span fields) cross a source helper
        which returns that argument unchanged.  The call itself remains in the
        graph: returning a formal does not prove that the callee has no side
        effects on the referenced storage.
        """

        attributes = data.get("attributes") or {}
        reference = attributes.get("callee_ref", attributes.get("method_ref"))
        function_table = getattr(graph, "function_table", None)
        if reference is None or function_table is None:
            return None
        try:
            child = function_table.entry(int(reference)).graph
        except (KeyError, TypeError, ValueError):
            return None
        child_graph = getattr(child, "G", None)
        if child_graph is None:
            return None
        sites = tuple(
            tuple(slots)
            for slots in (child_graph.graph.get("return_slot_values") or {}).values()
        )
        if (
            not sites
            or any(len(slots) != 1 for slots in sites)
            or any(slots[0] is None for slots in sites)
            or len({int(slots[0]) for slots in sites}) != 1
        ):
            return None
        formal_id = int(sites[0][0])
        formal = child_graph.nodes.get(formal_id, {})
        formal_attributes = formal.get("attributes") or {}
        formal_name = formal_attributes.get("binding_name")
        parameters = tuple(map(
            str, child_graph.graph.get("function_parameters") or (),
        ))
        if (
            formal_attributes.get("binding_kind") != "parameter"
            or formal_name is None
            or str(formal_name) not in parameters
        ):
            return None
        parameter_index = parameters.index(str(formal_name))
        actuals = {
            str(role): int(parent)
            for parent, role in data.get("parents") or ()
        }
        return actuals.get(
            f"arg:{parameter_index}", actuals.get(f"kw:{formal_name}"),
        )

    def structural_call_argument(
        data: Mapping[str, Any],
        *,
        roles: set[str],
        keywords: set[str],
        default: Any,
    ) -> Any:
        """Read authored call metadata without evaluating a Python callable.

        Dynamic arguments are represented by dependency edges and therefore
        take precedence.  The graph extractor intentionally does not create a
        value node for a literal keyword such as ``dim=1``; retain that exact
        source literal as normalized operation metadata instead of silently
        substituting the operator's default axis.
        """

        for parent, role_value in data.get("parents") or ():
            if str(role_value) not in roles:
                continue
            value = known.get(int(parent), unresolved)
            # ProgramABI facts describe runtime storage/type, not the value of
            # a structural keyword.  Treat them as unresolved here so a
            # literal ``dim=``/``axis=`` retained in the authored call can be
            # recovered below instead of being coerced with ``int(fact)``.
            if value is not unresolved and not isinstance(
                value, _ProgramABIValueFact
            ):
                return value
        expression = data.get("expr_obj")
        if isinstance(expression, ast.Call):
            for keyword in expression.keywords:
                if keyword.arg not in keywords:
                    continue
                try:
                    return ast.literal_eval(keyword.value)
                except (TypeError, ValueError):
                    return unresolved
        return default

    def _record_field_descriptor(node_id: int) -> dict[str, Any] | None:
        """Real shape/dtype for a record-field read, from its bound value.

        A record field (``state.height``) never gets a ``tensor`` fact of
        its own -- that inference path is for tensor-typed values flowing
        through tensor ops, and a mutable dataclass field is neither. Its
        receiver's own bound specialization (a real object, set by
        ``propagate_bound_planner_specializations`` before this pass runs)
        still has the real field on it; read it structurally, the same
        safe, side-effect-free ``getattr`` this module already uses
        elsewhere for call-argument specialization and region-shape
        inference (see ``declared()`` in ``_build_shell_hierarchy_plan``,
        which fixed the analogous gap for whole-array region shapes).
        Without this, ``state.height.shape[0]`` -- a loop bound -- silently
        resolves to nothing and the per-cell loop it bounds never runs.
        """

        node = graph.G.nodes.get(int(node_id), {})
        if str(node.get("type")) not in {"GetAttr", "Attribute"}:
            return None
        field_name = (node.get("attributes") or {}).get("attribute")
        if field_name is None:
            return None
        receiver_id = next((
            int(candidate)
            for candidate, role in (node.get("parents") or ())
            if str(role) in {"value", "base", "object"}
        ), None)
        if receiver_id is None:
            return None
        receiver_binding = (
            graph.G.nodes.get(receiver_id, {}).get("attributes") or {}
        ).get("binding_name")
        if receiver_binding is None:
            return None
        receiver_value = (
            graph.G.graph.get("planner_specializations") or {}
        ).get(str(receiver_binding))
        if receiver_value is None:
            return None
        try:
            field_value = getattr(receiver_value, str(field_name))
        except AttributeError:
            return None
        field_shape = getattr(field_value, "shape", None)
        if not field_shape:
            return None
        field_dtype = getattr(field_value, "dtype", None)
        return {
            "shape": tuple(int(dim) for dim in field_shape),
            "dtype": "float64" if field_dtype is None else str(field_dtype),
        }

    def _declared_record_span_descriptor(node_id: int) -> dict[str, Any] | None:
        """Shape/dtype of a record-field read from the ProgramABI declaration.

        The same declared-field lookup ``_tensor_descriptor`` performs for a
        ``GetAttr`` on a record receiver, restated here so the shape walk can
        recognise a declared span as a boundary without accepting an
        inferred (possibly provisional) ``tensor`` fact on the node.
        """

        node = graph.G.nodes.get(int(node_id), {})
        if str(node.get("op") or node.get("type") or "").casefold() != "getattr":
            return None
        field_name = str((node.get("attributes") or {}).get("attribute") or "")
        receiver = next((
            int(parent)
            for parent, role in node.get("parents") or ()
            if str(role) in {"value", "base", "object", "receiver"}
            and int(parent) in graph.G
        ), None)
        if receiver is None:
            return None
        receiver_binding = str(
            (graph.G.nodes[receiver].get("attributes") or {}).get("binding_name") or ""
        )
        if not receiver_binding:
            return None
        record = (graph.G.graph.get("parameter_record_abi") or {}).get(receiver_binding)
        field = (
            (record.get("fields") or {}).get(field_name)
            if isinstance(record, Mapping) else None
        )
        if not (isinstance(field, Mapping) and field.get("shape") is not None):
            return None
        return {
            "shape": tuple(map(int, field["shape"])),
            "dtype": str(field.get("dtype") or "float64"),
        }

    def descriptor_attribute(parent: int, attribute: str) -> Any:
        descriptor = _tensor_descriptor(graph, parent)
        if descriptor is None:
            descriptor = _record_field_descriptor(parent)
        if attribute in {"shape", "ndim", "ndims"}:
            pending_inputs = [int(parent)]
            seen_inputs = set()
            while pending_inputs:
                current = pending_inputs.pop()
                if current in seen_inputs or current not in graph.G:
                    continue
                seen_inputs.add(current)
                source = graph.G.nodes[current]
                operation = str(source.get("op") or source.get("type") or "").casefold()
                if operation == "input":
                    if _tensor_descriptor(graph, current) is None:
                        return unresolved
                    continue
                # A declared record span supplies its own shape boundary;
                # the opaque receiver is not an untyped tensor parameter.
                # That boundary is stated two ways: a bound planner
                # specialization (a real object supplied at the entry) or
                # the ProgramABI record declaration itself.  A method on
                # ``self`` in a contract-only lowering has no bound object,
                # so only the declaration can answer -- and without it this
                # walk reached the record receiver, found no tensor
                # descriptor on it, and returned unresolved for a field whose
                # shape was declared.  ``self.memory.ndim`` then never
                # folded, the ingestion expansion of ``self.memory[...]``
                # (``tuple([slice(None)] * (self.memory.ndim - 0))``) stayed
                # a live expression, and the store lowered as a
                # resident-sequence replace with hidden length formals plus
                # a one-element write (PieceState.restore in the dt system:
                # 416 storage formals, 104 unnamed).
                if operation == "getattr" and (
                    _record_field_descriptor(current) is not None
                    or _declared_record_span_descriptor(current) is not None
                ):
                    continue
                pending_inputs.extend(
                    int(value) for value, role in source.get("parents") or ()
                    if str(role) not in {"callee", "func", "definition"}
                )
            planner_descriptors = (
                graph.G.graph.get("planner_tensor_descriptors") or {}
            )
            callsite_descriptor_names = set(map(
                str,
                graph.G.graph.get("callsite_tensor_descriptor_names") or (),
            ))
            input_bindings = tuple(
                (data.get("attributes") or {}).get("binding_name")
                for _node_id, data in graph.G.nodes(data=True)
                if data.get("type") == "Input"
            )
            unsettled_tensor_parameter = (
                bool(input_bindings) and not planner_descriptors
            ) or any(
                bool(data.get("tensor"))
                and str(binding_name) not in callsite_descriptor_names
                and not tuple((
                    planner_descriptors.get(str(binding_name)) or {}
                ).get("shape") or ())
                for _node_id, data in graph.G.nodes(data=True)
                if data.get("type") == "Input"
                for binding_name in (
                    (data.get("attributes") or {}).get("binding_name"),
                )
            )
            if descriptor is None and unsettled_tensor_parameter:
                # Shape-derived constants are irreversible graph rewrites.
                # Before planner/callsite descriptors arrive, input nodes may
                # still be untyped while their internal consumers acquire
                # padded scalar domain defaults such as (1,1,1). Never
                # fold `.shape` from that provisional graph domain. An exact
                # descriptor for the queried value is sufficient even when an
                # unrelated parameter remains unsettled; otherwise parametric
                # functions could never use a declared peer's shape.
                return unresolved
        if descriptor is None:
            return unresolved
        if attribute == "shape":
            return descriptor["shape"]
        if attribute in {"ndim", "ndims"}:
            return len(descriptor["shape"])
        if attribute == "dtype":
            return descriptor["dtype"]
        if attribute == "device" and "device" in descriptor:
            return descriptor["device"]
        return unresolved

    def safe_types(expression: ast.AST | None):
        if isinstance(expression, ast.Name):
            return {
                "bool": (bool,), "int": (int,), "float": (float,),
                "str": (str,), "bytes": (bytes,), "tuple": (tuple,),
                "list": (list,), "dict": (dict,), "set": (set,),
                "range": (range,),
            }.get(expression.id)
        if isinstance(expression, ast.Tuple):
            result = tuple(
                item
                for element in expression.elts
                for item in (safe_types(element) or ())
            )
            return result or None
        return None

    def type_identities(expression: ast.AST | None) -> tuple[str, ...]:
        if isinstance(expression, ast.Name):
            return (str(expression.id),)
        if isinstance(expression, ast.Attribute):
            parts = [str(expression.attr)]
            owner = expression.value
            while isinstance(owner, ast.Attribute):
                parts.append(str(owner.attr))
                owner = owner.value
            if isinstance(owner, ast.Name):
                parts.append(str(owner.id))
                return (".".join(reversed(parts)),)
            return (str(expression.attr),)
        if isinstance(expression, ast.Tuple):
            return tuple(
                identity
                for item in expression.elts
                for identity in type_identities(item)
            )
        return ()

    def declared_record_field(
        owner_fact: Any, attribute: str,
    ) -> Any:
        if not (
            isinstance(owner_fact, _ProgramABIValueFact)
            and owner_fact.storage == "record"
        ):
            return unresolved
        candidate_records = [
            *(graph.G.graph.get("parameter_record_abi") or {}).values(),
            *dict(
                (graph.G.graph.get("program_abi") or {}).get("records") or {}
            ).values(),
        ]
        matching_records = tuple({
            str(record.get("identity") or ""): record
            for record in candidate_records
            if str(record.get("identity") or "") == owner_fact.python_type
        }.values())
        if len(matching_records) != 1:
            return unresolved
        field = dict(matching_records[0].get("fields") or {}).get(attribute)
        if field is None:
            return unresolved
        dtype = field.get("dtype")
        nested_records = dict(
            (graph.G.graph.get("program_abi") or {}).get("records") or {}
        )
        nested_record = nested_records.get(str(field.get("record") or ""))
        value_record = nested_records.get(
            str(field.get("value_record") or "")
        )
        python_type = (
            str((nested_record or {}).get("identity") or field["record"])
            if str(field.get("storage") or "") == "record"
            else {
                "bool": "builtins.bool",
                "int": "builtins.int",
                "int32": "builtins.int",
                "int64": "builtins.int",
                "float": "builtins.float",
                "float32": "builtins.float",
                "float64": "builtins.float",
            }.get(str(dtype), str(dtype or "unknown"))
        )
        value_python_type = (
            str(field.get("value_python_type") or "") or None
            if field.get("value_tensor")
            else (
                None
                if value_record is None
                else str(
                    value_record.get("identity")
                    or field.get("value_record")
                )
            )
        )
        return _ProgramABIValueFact(
            python_type,
            str(field.get("storage") or "unknown"),
            None if dtype is None else str(dtype),
            value_python_type,
            (
                "span"
                if field.get("value_tensor")
                else "record" if value_record is not None else None
            ),
            tuple(map(str, field.get("token_vocabulary") or ())) or None,
            (
                None
                if field.get("shape") is None
                else tuple(map(int, field["shape"]))
            ),
            (
                None
                if field.get("rank") is None
                else int(field["rank"])
            ),
            (
                None
                if field.get("value_rank") is None
                else int(field["value_rank"])
            ),
            bool(field.get("optional", False)),
        )

    def evaluate(node_id: int, data: Mapping[str, Any]) -> Any:
        if (data.get("attributes") or {}).get("bound_method_ref") is not None:
            return unresolved
        node_type = str(data.get("type") or "")
        operation = str(data.get("op") or node_type).casefold()
        expression = data.get("expr_obj")
        if node_type in {"Constant", "Const", "const"}:
            return constant(data)
        if node_type in {"Input", "input"}:
            binding_name = (data.get("attributes") or {}).get(
                "binding_name"
            )
            if binding_name is None:
                identities = graph.G.graph.get("identity_table") or {}
                parameters = set(map(
                    str, graph.G.graph.get("function_parameters") or ()
                ))
                binding_name = next((
                    str(name)
                    for name, history in identities.items()
                    if str(name) in parameters
                    and int(node_id) in set(map(int, history or ()))
                ), None)
            specializations = graph.G.graph.get(
                "planner_specializations"
            ) or {}
            specialized = specializations.get(str(binding_name), unresolved)
            if specialized is not unresolved:
                return specialized
            # A signature default is not a fact about a parametric entrypoint:
            # callers may still supply this input.  Omitted arguments at real
            # call sites are copied into ``planner_specializations`` by
            # _propagate_callsite_planner_specializations, above.  Folding the
            # bare default here erased public inputs (and left already-carved
            # deployment regions referring to the erased nodes).
            value_abi = (
                graph.G.graph.get("parameter_value_abi") or {}
            ).get(str(binding_name))
            if value_abi is not None:
                linked_abi = (
                    graph.G.graph.get("linked_value_abi") or {}
                ).get(int(data.get("value_id", node_id)))
                effective_abi = {
                    **dict(value_abi),
                    **(
                        dict(linked_abi)
                        if isinstance(linked_abi, Mapping) else {}
                    ),
                }
                effective_shape = (
                    None
                    if effective_abi.get("shape") is None
                    else tuple(map(int, effective_abi["shape"]))
                )
                return _ProgramABIValueFact(
                    str(value_abi["python_type"]),
                    str(effective_abi["storage"]),
                    effective_abi.get("dtype"),
                    shape=effective_shape,
                    rank=(
                        len(effective_shape)
                        if effective_shape is not None
                        else (
                            None
                            if effective_abi.get("rank") is None
                            else int(effective_abi["rank"])
                        )
                    ),
                )
            record_abi = (
                graph.G.graph.get("parameter_record_abi") or {}
            ).get(str(binding_name))
            if record_abi is not None:
                return _ProgramABIValueFact(
                    str(record_abi["identity"]), "record", None
                )
            descriptor = _tensor_descriptor(graph, int(node_id))
            if descriptor is not None:
                dtype = str(descriptor.get("dtype") or "unknown")
                # A descriptor that carries the declared Python type (a
                # caller's declared scalar forwarded across the call) is
                # the fact; the dtype map is only the fallback for
                # inferred tensor values.
                python_type = str(descriptor.get("python_type") or "") or {
                    "bool": "builtins.bool",
                    "int": "builtins.int",
                    "int32": "builtins.int",
                    "int64": "builtins.int",
                    "float": "builtins.float",
                    "float32": "builtins.float",
                    "float64": "builtins.float",
                }.get(dtype, dtype)
                return _ProgramABIValueFact(
                    python_type,
                    "span" if tuple(descriptor.get("shape") or ()) else "scalar",
                    dtype,
                )
            return unresolved
        if node_type in {"Slice", "slice"} or isinstance(
            expression, ast.Slice,
        ):
            # A slice is an ordinary structural value, the same currency the
            # basic-index reader already builds.  Without it a shape tuple
            # could be indexed (``x.shape[0]``) and concatenated
            # (``x.shape + (1,)``) but never sliced, so ``x.shape[:2] + (1,)``
            # left its reshape unresolved: the reshape kept the source extents
            # and the consuming operation saw incompatible operands.
            bounds: dict[str, Any] = {}
            for parent, role in (data.get("parents") or ()):
                role = str(role)
                if role not in {"lower", "upper", "step"}:
                    continue
                bound = known.get(int(parent), unresolved)
                if bound is unresolved or isinstance(
                    bound, _ProgramABIValueFact
                ):
                    return unresolved
                bounds[role] = bound
            return slice(
                bounds.get("lower"), bounds.get("upper"), bounds.get("step"),
            )
        if node_type in {"Tuple", "List", "Set"} or isinstance(
            expression, (ast.Tuple, ast.List, ast.Set),
        ):
            values = tuple(
                known.get(int(parent), unresolved)
                for parent, _role in (data.get("parents") or ())
            )
            if any(value is unresolved for value in values):
                return unresolved
            if node_type == "List" or isinstance(expression, ast.List):
                return list(values)
            if node_type == "Set" or isinstance(expression, ast.Set):
                return set(values)
            return tuple(values)
        if node_type in {"GetAttr", "Attribute"} or isinstance(
            expression, ast.Attribute,
        ):
            parent = next((
                int(parent)
                for parent, role in (data.get("parents") or ())
                if str(role) in {"value", "base", "operand", "object"}
            ), None)
            attribute = str(
                (data.get("attributes") or {}).get("attribute")
                or getattr(expression, "attr", "")
            )
            if parent is not None:
                owner_fact = known.get(parent, unresolved)
                if (
                    isinstance(owner_fact, _ProgramABIValueFact)
                    and owner_fact.shape is not None
                    and attribute in {"shape", "ndim", "ndims"}
                ):
                    return (
                        owner_fact.shape
                        if attribute == "shape"
                        else len(owner_fact.shape)
                    )
                fixed = descriptor_attribute(parent, attribute)
                if fixed is not unresolved:
                    return fixed
                field_fact = declared_record_field(owner_fact, attribute)
                if field_fact is not unresolved:
                    return field_fact
        if node_type in {"Indexed", "indexed"} or isinstance(
            expression, ast.Subscript,
        ):
            base = next((
                int(parent)
                for parent, role in (data.get("parents") or ())
                if str(role) in {"value", "base", "operand", "object"}
            ), None)
            base_fact = known.get(base, unresolved)
            if (
                isinstance(base_fact, _ProgramABIValueFact)
                and base_fact.storage == "keyed"
                and base_fact.value_python_type is not None
            ):
                return _ProgramABIValueFact(
                    base_fact.value_python_type,
                    base_fact.value_storage or "record",
                    base_fact.dtype,
                    rank=(
                        int((data.get("attributes") or {}).get(
                            "program_abi_rank"
                        ))
                        if (data.get("attributes") or {}).get(
                            "program_abi_rank"
                        ) is not None
                        else base_fact.value_rank
                    ),
                )
            if (
                isinstance(base_fact, _ProgramABIValueFact)
                and base_fact.storage == "span"
                and isinstance(expression, ast.Subscript)
            ):
                source_rank, result_rank = symbolic_basic_index_ranks(
                    expression.slice, base_fact.rank
                )
                if base is not None and int(base) in graph.G:
                    base_data = graph.G.nodes[int(base)]
                    base_attributes = dict(base_data.get("attributes") or {})
                    base_attributes.update({
                        "program_abi_storage": "span",
                        "program_abi_rank": int(source_rank),
                    })
                    base_data["attributes"] = base_attributes
                    base_tensor = dict(base_data.get("tensor") or {})
                    base_tensor.update({
                        "shape": tuple(base_fact.shape or ()),
                        "dtype": str(base_fact.dtype or "unknown"),
                        "rank": int(source_rank),
                        "metadata_state": "dynamic",
                    })
                    base_data["tensor"] = base_tensor
                return _ProgramABIValueFact(
                    base_fact.python_type,
                    "span",
                    base_fact.dtype,
                    shape=None,
                    rank=int(result_rank),
                )
            index_facts = tuple(
                known.get(int(parent), unresolved)
                for parent, role in (data.get("parents") or ())
                if str(role) == "index"
            )
            if (
                base_fact is not unresolved
                and not isinstance(base_fact, _ProgramABIValueFact)
                and index_facts
                and all(index is not unresolved for index in index_facts)
            ):
                index = (
                    index_facts[0]
                    if len(index_facts) == 1 else tuple(index_facts)
                )
                try:
                    return base_fact[index]
                except (IndexError, KeyError, TypeError):
                    return unresolved
        if operation in {"get_shape", "dim", "numel", "ndim", "ndims"}:
            parent = next((
                int(parent)
                for parent, role in (data.get("parents") or ())
                if str(role) in {"operand", "value", "base", "object"}
            ), None)
            descriptor = None if parent is None else _tensor_descriptor(
                graph, parent
            )
            if descriptor is not None:
                # Read the descriptor for what it states.  ``LU.get_shape()``
                # folded to ``()`` here while LU's descriptor said nothing at
                # all, and ``shp[-1]`` -- the matrix order every loop in the
                # substitution helpers counts with -- then became a formal no
                # caller could fill.
                if not descriptor_states_a_shape(descriptor):
                    return unresolved
                if operation == "get_shape":
                    if descriptor.get("metadata_state") == "dynamic":
                        return unresolved
                    return tuple(descriptor["shape"])
                if operation == "dim":
                    if descriptor.get("metadata_state") == "dynamic":
                        return unresolved
                    return len(descriptor["shape"])
                if operation in {"ndim", "ndims"}:
                    return len(descriptor["shape"])
                count = 1
                for extent in descriptor["shape"]:
                    count *= int(extent)
                return count
        if node_type == "Phi" or isinstance(expression, ast.IfExp):
            parents = {
                str(role): int(parent)
                for parent, role in (data.get("parents") or ())
            }
            predicate = known.get(parents.get("test"), unresolved)
            if predicate is unresolved or isinstance(
                predicate, _ProgramABIValueFact
            ):
                return unresolved
            selected = parents.get("body" if bool(predicate) else "orelse")
            if selected is None:
                return unresolved
            fixed = known.get(selected, unresolved)
            return (
                _StructuralValueAlias(selected)
                if fixed is unresolved
                or isinstance(fixed, _ProgramABIValueFact)
                else fixed
            )
        if isinstance(expression, ast.UnaryOp):
            parent = next((
                int(parent) for parent, role in (data.get("parents") or ())
                if str(role) in {"operand", "value"}
            ), None)
            value = known.get(parent, unresolved)
            if value is unresolved or isinstance(
                value, _ProgramABIValueFact
            ):
                return unresolved
            if isinstance(expression.op, ast.USub):
                return -value
            if isinstance(expression.op, ast.UAdd):
                return +value
            if isinstance(expression.op, ast.Not):
                return not value
        if isinstance(expression, ast.BinOp):
            parents = {
                str(role): int(parent)
                for parent, role in (data.get("parents") or ())
            }
            left_id = parents.get("lhs", parents.get("left"))
            right_id = parents.get("rhs", parents.get("right"))
            left = known.get(left_id, unresolved)
            right = known.get(right_id, unresolved)
            if isinstance(expression.op, ast.Add):
                # Concatenating an EMPTY sequence is an identity on the other
                # sequence, and it stays an identity when that sequence holds
                # a runtime member.  Authored ``LU[..., i, i+1:]`` arrives as
                # ``() + (i, slice(i + 1, None))`` once the ellipsis prefix
                # proves empty; leaving it unresolved kept a Python
                # concatenation in the graph that later lowering read as an
                # arithmetic Add on a float index.
                # Only for TUPLES: an empty list concatenation produces a
                # new, separately mutable list, so aliasing it would merge two
                # storage identities the source keeps apart.
                for empty, other_id in ((left, right_id), (right, left_id)):
                    if (
                        isinstance(empty, tuple)
                        and not empty
                        and other_id is not None
                        and int(other_id) in graph.G
                        and str((
                            graph.G.nodes[int(other_id)].get("attributes")
                            or {}
                        ).get("aggregate_kind") or "") == "tuple"
                        # ...and only where the concatenation is READ as a
                        # subscript.  An index is consumed, never mutated, so
                        # merging it with the sequence it concatenates cannot
                        # merge two mutable storage identities.
                        and bool(data.get("children"))
                        and all(
                            str(role) == "index"
                            for _child, role in data.get("children") or ()
                        )
                    ):
                        other = known.get(int(other_id), unresolved)
                        return (
                            _StructuralValueAlias(int(other_id))
                            if other is unresolved
                            else other
                        )
            if left is unresolved or right is unresolved:
                return unresolved
            # A program-ABI fact states the runtime representation/type; it
            # is not the runtime value.  Dataclass equality against a literal
            # would otherwise evaluate to False here and erase live compiler
            # branches merely because their graph-row schema was discovered.
            if isinstance(left, _ProgramABIValueFact) or isinstance(
                right, _ProgramABIValueFact
            ):
                return unresolved
            try:
                return {
                    ast.Add: operator.add, ast.Sub: operator.sub,
                    ast.Mult: operator.mul, ast.Div: operator.truediv,
                    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
                    ast.Pow: operator.pow,
                }[type(expression.op)](left, right)
            except (KeyError, ArithmeticError, TypeError, ValueError):
                return unresolved
        if isinstance(expression, ast.Compare) and len(expression.ops) == 1:
            parents = {
                str(role): int(parent)
                for parent, role in (data.get("parents") or ())
            }
            left = known.get(parents.get("lhs", parents.get("left")), unresolved)
            right = known.get(parents.get("rhs", parents.get("right")), unresolved)
            if left is unresolved or right is unresolved:
                return unresolved
            comparison_node = expression.ops[0]
            if isinstance(comparison_node, (ast.Is, ast.IsNot)) and (
                (isinstance(left, _ProgramABIValueFact) and right is None)
                or (left is None and isinstance(right, _ProgramABIValueFact))
            ):
                # A program-ABI fact does not reveal a runtime numerical value,
                # but it does prove that physical, nonoptional storage exists.
                # This compiler has no tagged optional scalar/span/record ABI;
                # such a value therefore cannot be Python ``None``.  Restrict
                # the fold to identity-with-None so no equality or truthiness is
                # inferred from the schema fact itself.
                #
                # The one exception is a fact whose declared Python type IS
                # ``NoneType``: the contract states the value is None (a
                # ``distribution=None`` forwarded as a declared scalar).  Then
                # ``is None`` holds and ``is not None`` does not; folding it
                # the other way selected ``return distribution(...)`` over an
                # opaque callable and emptied the specialized callee.
                fact = left if isinstance(left, _ProgramABIValueFact) else right
                if str(fact.python_type) in {"builtins.NoneType", "NoneType"}:
                    return isinstance(comparison_node, ast.Is)
                if fact.optional:
                    return unresolved
                return isinstance(comparison_node, ast.IsNot)
            if isinstance(left, _ProgramABIValueFact) or isinstance(
                right, _ProgramABIValueFact
            ):
                return unresolved
            comparison = {
                ast.Eq: operator.eq, ast.NotEq: operator.ne,
                ast.Lt: operator.lt, ast.LtE: operator.le,
                ast.Gt: operator.gt, ast.GtE: operator.ge,
                ast.Is: operator.is_, ast.IsNot: operator.is_not,
            }.get(type(comparison_node))
            if comparison is None:
                return unresolved
            try:
                return comparison(left, right)
            except (TypeError, ValueError):
                return unresolved
        if isinstance(expression, ast.BoolOp):
            values = [
                known.get(int(parent), unresolved)
                for parent, role in (data.get("parents") or ())
                if str(role).startswith("value")
            ]
            if not values:
                return unresolved
            if not any(
                value is unresolved
                or isinstance(value, _ProgramABIValueFact)
                for value in values
            ):
                # Python returns the selected operand, including non-Boolean
                # values. Preserve source order and stop at the first decision.
                for value in values[:-1]:
                    if bool(value) != isinstance(expression.op, ast.And):
                        return value
                return values[-1]
            # A later constant can determine the branch but cannot erase
            # evaluation of an earlier unknown operand (which may mutate or
            # raise). Fold only a proven constant prefix's selected operand.
            for value in values[:-1]:
                if value is unresolved or isinstance(value, _ProgramABIValueFact):
                    break
                if bool(value) != isinstance(expression.op, ast.And):
                    return value
            return unresolved
        if not isinstance(expression, ast.Call):
            return unresolved
        name = (
            expression.func.id
            if isinstance(expression.func, ast.Name)
            else expression.func.attr
            if isinstance(expression.func, ast.Attribute)
            else ""
        )
        arguments = positional(data)
        values = tuple(known.get(argument, unresolved) for argument in arguments)
        identity_actual = exact_return_formal_actual(node_id, data)
        if identity_actual is not None:
            attributes = data.setdefault("attributes", {})
            attributes["structural_identity_actual_value_id"] = int(
                identity_actual
            )
            receipt = {
                "call_value_id": int(node_id),
                "actual_value_id": int(identity_actual),
                "callee_ref": attributes.get(
                    "callee_ref", attributes.get("method_ref")
                ),
                "priority": "exact_returned_formal",
                "tie_policy": "incumbent",
            }
            concordance = list(graph.G.graph.get(
                "structural_identity_forwarding_concordance", ()
            ))
            if receipt not in concordance:
                concordance.append(receipt)
                graph.G.graph[
                    "structural_identity_forwarding_concordance"
                ] = tuple(concordance)
            identity_fact = known.get(identity_actual, unresolved)
            if isinstance(identity_fact, _ProgramABIValueFact):
                return identity_fact
        result_class = (data.get("attributes") or {}).get("result_class_ref")
        if result_class is not None:
            records = tuple(
                record
                for record in dict(
                    (graph.G.graph.get("program_abi") or {}).get("records") or {}
                ).values()
                if str(record.get("identity") or "") == str(result_class)
                or str(record.get("identity") or "").rsplit(".", 1)[-1]
                == str(result_class).rsplit(".", 1)[-1]
            )
            if len(records) == 1:
                return _ProgramABIValueFact(
                    str(records[0]["identity"]), "record", None,
                )
        if (name == "callable" and len(arguments) == 1
                and (data.get("attributes") or {}).get("extraction_identity") == "builtins.callable"
                and (graph.G.nodes[arguments[0]].get("attributes") or {}).get("bound_method_ref") is not None):
            return True
        if (
            name == "callable"
            and len(values) == 1
            and (data.get("attributes") or {}).get("extraction_identity")
            == "builtins.callable"
            and values[0] is not unresolved
            and not isinstance(values[0], _ProgramABIValueFact)
        ):
            # Capability of an exact structural constant is known without
            # invoking it. In particular callable(None) must remove its dead
            # call arm, not become an unexplained runtime predicate input.
            return callable(values[0])
        if name == "hasattr" and len(arguments) == 2:
            attribute = values[1]
            if isinstance(attribute, str) and not attribute.startswith("_"):
                owner_id = int(arguments[0])
                descriptor = _tensor_descriptor(graph, owner_id)
                if descriptor is not None and attribute in {
                    "item", "shape", "dtype", "ndim", "ndims", "numel",
                }:
                    # Repository tensors, including rank-zero tensors, expose
                    # this fixed structural surface. ``item`` on a scalar is
                    # the native scalar extraction already represented by the
                    # pursued item node; its presence is not runtime data.
                    return True
                if descriptor_attribute(owner_id, attribute) is not unresolved:
                    return True
                if declared_record_field(
                    known.get(owner_id, unresolved), attribute,
                ) is not unresolved:
                    return True
        if name == "get" and values:
            receiver = next((
                int(parent)
                for parent, role in (data.get("parents") or ())
                if str(role) in {"operand", "value", "base", "object"}
            ), None)
            field_name = values[0]
            if receiver is not None and isinstance(field_name, str):
                field_fact = declared_record_field(
                    known.get(receiver, unresolved), field_name,
                )
                if field_fact is not unresolved:
                    return field_fact
        if name == "getattr" and len(arguments) >= 2:
            attribute = values[1]
            if isinstance(attribute, str) and not attribute.startswith("_"):
                fixed = descriptor_attribute(arguments[0], attribute)
                if fixed is not unresolved:
                    return fixed
                # `getattr(record, "field", default)` is the same read as
                # `record.field` (and `record.get("field")` just above): a
                # field the program ABI declares on the receiver is present,
                # so the default never applies. Returning the default
                # unconditionally folded `getattr(targets,
                # "energy_exchange_fraction", None)` to None -- and every
                # `x is not None` guard on it to a constant False -- even
                # when the receiver's record declared the field (verified
                # with tools/audit_structural_boolop_shapes.py
                # --declare-optional-targets); it was only correct while the
                # field happened to be undeclared.
                field_fact = declared_record_field(
                    known.get(arguments[0], unresolved), attribute,
                )
                if field_fact is not unresolved:
                    return field_fact
            if len(values) >= 3 and values[2] is not unresolved:
                return values[2]
        if name == "isinstance" and values:
            if isinstance(values[0], _ProgramABIValueFact):
                if values[0].token_vocabulary:
                    # This is a runtime vocabulary token.  The SSA ABI pass
                    # lowers the accepted source types to positions in the
                    # same vocabulary; the fact is not one particular member.
                    return unresolved
                if (
                    values[0].storage == "reference"
                    or values[0].python_type in {"", "unknown"}
                ):
                    return unresolved
                declared = values[0].python_type
                accepted_identities = type_identities(
                    expression.args[1] if len(expression.args) > 1 else None
                )
                return any(
                    declared == identity
                    or declared.endswith("." + identity)
                    for identity in accepted_identities
                )
            accepted = safe_types(
                expression.args[1] if len(expression.args) > 1 else None
            )
            if values[0] is not unresolved and accepted is not None:
                return isinstance(values[0], accepted)
        if any(value is unresolved or isinstance(value, _ProgramABIValueFact)
               for value in values):
            return unresolved
        try:
            if name in {"bool", "int", "float"}:
                return {"bool": bool, "int": int, "float": float}[name](*values)
            if name in {"tuple", "list", "set"}:
                return {"tuple": tuple, "list": list, "set": set}[name](*values)
            if name == "len":
                return len(*values)
            if name == "range":
                return range(*values)
            if name == "enumerate":
                return tuple(enumerate(*values))
            if name == "zip":
                return tuple(zip(*values))
            if name == "sorted":
                return sorted(*values)
            if name == "max":
                return max(*values)
            if name == "min":
                return min(*values)
            if name == "all":
                return all(*values)
            if name == "any":
                return any(*values)
            if name == "slice":
                return slice(*values)
        except (TypeError, ValueError, OverflowError):
            return unresolved
        return unresolved

    def replace(node_id: int, value: Any) -> None:
        nonlocal topology_changed
        data = graph.G.nodes[int(node_id)]
        topology_changed |= bool(data.get("parents"))
        source_attributes = data.get("attributes") or {}
        if isinstance(data.get("expr_obj"), ast.AST):
            data.setdefault("authored_expr_obj", data["expr_obj"])
        # The cells the fold's ``known`` held for this node: its operands'
        # identity cells and, for an Input fed by the planner, its
        # ``planner_specialization`` cell.  Collected before the parent
        # edges are cut, so the ``proven_literal`` row below can derive from
        # them (plan 80, A1.2).
        literal_sources = _fold_literal_source_cells(graph, int(node_id))
        # The folded node reads nothing any more: its positions retire
        # through the one writer (edges and ``children`` with them).
        _set_operands(graph, int(node_id), [], cause=_CALLSITE_FOLD_LITERAL)
        # A constant fold changes the value-producing operation, not the
        # structural identity of an authored aggregate.  Retain the compact
        # aggregate ledger so later sequence lowering can still distinguish
        # a singleton ``[opcode]`` from an arbitrary scalar-valued constant
        # and recover its original leaf identity.
        # ``tensor_output_descriptors`` travels WITH the ledger or the
        # ledger is unrecoverable.  This fold clears the node's parents and
        # drops every parent edge just above, so the leaf nodes the ledger
        # names are stranded and later pruned; the retained leaf ids then
        # refer to nodes that are no longer resident.
        #
        # ``_repair_missing_aggregate_leaf_projections`` exists for exactly
        # this ("a selected producer can survive while a projection from
        # the rejected arm is removed, leaving ... a dangling leaf ID"),
        # and it rebuilds each missing position from the stored output
        # descriptors -- but it declines when the descriptor count does not
        # match the leaf count, and dropping them here makes that count 0.
        # So the repair silently skipped, the consumer silently rejected a
        # ledger it could not verify as resident, and the members arrived
        # at the far end of the build as formals with no ABI accounting.
        #
        # Measured on the two-piece llvm_dt_system product: the
        # ``copy_shallow()`` result held 52 leaves with 52 resident, and
        # the folded ``saved = ... if rollback else None`` node held the
        # same 52 leaves with 0 resident.  Every writer of this ledger
        # publishes both keys together; only this fold split them.
        attributes = {
            key: copy.deepcopy(source_attributes[key])
            for key in (
                "producer_kind",
                "aggregate_kind",
                "sequence_key_columns",
                "sequence_column_count",
                "sequence_writable",
                "aggregate_leaf_value_ids",
                "tensor_output_descriptors",
            )
            if key in source_attributes
        }
        attributes.update({
            "value": copy.deepcopy(value),
            "structural_specialization": True,
        })
        data.update({
            "type": "Constant", "op": "const", "label": repr(value),
            "attributes": attributes,
            "constant": copy.deepcopy(value), "expr_obj": None,
        })
        # A proven literal belongs to the VALUE, not to the graph instance
        # that happened to fold it.  Several planned shells share one function
        # name, each with its own copy, so a fact folded in one is invisible
        # to a call edge that names another.  The book is keyed by identity,
        # which is what makes it readable from any of them.  The row is
        # DERIVED from the operand cells the fold evaluated (and the node's
        # own identity cell); a fold with no cell to name posts
        # ``Unsourced(RAW_PRIMITIVE)`` under the latch.
        # ``None`` is a proven literal too (``axis=None``): without its row a
        # folded ``bw_sum`` formal ``axis`` left no cause cell, and
        # expand_reduction's ``axis`` row went Unresolved(dynamic) ->
        # LITERAL(None) from the identical cell set ("REVISE without a
        # changed source").
        if value is None or isinstance(value, (int, float, bool, str, tuple)):
            from .concordance_declarations import (
                PLANNER_STRUCTURAL_FOLD, PROVEN_LITERAL,
            )
            from .identity_concordance import (
                Mode, RAW_PRIMITIVE, Unsourced, current_identity_book,
            )

            literal_row = (
                str(graph.G.graph.get("function_name")),
                int(data.get("value_id", node_id)),
            )
            book = current_identity_book()
            literal_page = book.pages.get(PROVEN_LITERAL.name)
            # "No row yet" must not read as "row says None".
            if (
                literal_page is None
                or book.latest_ref(PROVEN_LITERAL, literal_row) is None
                or literal_page.latest(literal_row) != value
            ):
                if literal_sources:
                    round_cells.append(_post_if_changed(
                        PROVEN_LITERAL, literal_row, value,
                        stage=PLANNER_STRUCTURAL_FOLD, cells=literal_sources,
                    ))
                else:
                    round_cells.append(book.post(
                        PROVEN_LITERAL, literal_row, value,
                        stage=PLANNER_STRUCTURAL_FOLD,
                        provenance=Unsourced(RAW_PRIMITIVE), mode=Mode.REVISE,
                    ))

    def remove_node(node_id: int) -> None:
        nonlocal topology_changed
        node_id = int(node_id)
        if node_id not in graph.G:
            return
        topology_changed = True
        for successor in tuple(graph.G.successors(node_id)):
            successor_data = graph.G.nodes[int(successor)]
            _set_operands(graph, int(successor), [
                (int(parent), str(role))
                for parent, role in successor_data.get("parents") or ()
                if int(parent) != node_id
            ], cause=_CALLSITE_FOLD_REMOVE_NODE)
        _set_operands(graph, node_id, [], cause=_CALLSITE_FOLD_REMOVE_NODE)
        graph.G.remove_node(node_id)
        # The identity table must name only values that exist.  A removed
        # aggregate formal left in a name's history is later seeded as an
        # untyped function argument by the SSA builder ("preserve the initial
        # identity"), bound to the caller's tuple node, and reported as a
        # missing caller value.
        identities = graph.G.graph.get("identity_table") or {}
        if any(
            int(value_id) == node_id
            for history in identities.values()
            for value_id in history
        ):
            # Each ``name_binding`` version naming the removed node revises
            # to ``Unresolved(BINDING_VERSION_REMOVED)`` (plan 80, A2.7).
            _post_identity_table_mutation(graph, removed=(node_id,))
            graph.G.graph["identity_table"] = {
                str(name): tuple(
                    int(value_id) for value_id in history
                    if int(value_id) != node_id
                )
                for name, history in identities.items()
            }

    def replace_alias(node_id: int, source_id: int) -> None:
        node_id = int(node_id)
        source_id = int(source_id)
        if node_id == source_id or source_id not in graph.G:
            return
        for successor in tuple(graph.G.successors(node_id)):
            successor_data = graph.G.nodes[int(successor)]
            replacement = tuple(
                (
                    source_id if int(parent) == node_id else int(parent),
                    str(role),
                )
                for parent, role in successor_data.get("parents") or ()
            )
            _set_operands(
                graph, int(successor), list(replacement),
                cause=_CALLSITE_FOLD_REPLACE_ALIAS, same={node_id: source_id},
            )
            _follow_declared_value_source(successor_data, node_id, source_id)
        identities = graph.G.graph.get("identity_table") or {}
        # Each ``name_binding`` version naming the alias revises to the
        # source's identity (plan 80, A2.7); return slots naming it follow.
        _post_identity_table_mutation(graph, aliases={node_id: source_id})
        graph.G.graph["identity_table"] = {
            str(name): tuple(dict.fromkeys(
                source_id if int(value_id) == node_id else int(value_id)
                for value_id in history
            ))
            for name, history in identities.items()
        }
        graph.roots = [
            source_id if int(root) == node_id else int(root)
            for root in graph.roots
        ]
        if graph.G.graph.get("return_slot_values"):
            graph.G.graph["return_slot_values"] = {
                site: tuple(source_id if value is not None and int(value) == node_id else value
                            for value in slots)
                for site, slots in graph.G.graph["return_slot_values"].items()
            }
        # Cached copies of the aliased id live on nodes that are not its
        # successors too (a loop node's carried initial, a tuple's leaf
        # ledger); move every one before the id disappears.
        _retarget_all_cached_value_ids(graph, node_id, source_id)
        remove_node(node_id)

    def same_structural_value(left: Any, right: Any) -> bool:
        if left is right:
            return True
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            try:
                return bool(np.array_equal(left, right))
            except (TypeError, ValueError):
                return False
        try:
            return bool(left == right)
        except (TypeError, ValueError):
            return False

    def tensor_data_descriptors(
        data: Mapping[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        """Return numerical operands, expanding authored tensor aggregates."""

        descriptors = []
        structural_roles = {"dim", "axis", "keepdim", "kw:dim", "kw:axis",
                            "kw:keepdim", "kw:out", "kw:dtype"}
        for parent, role in data.get("parents") or ():
            role = str(role)
            if role in {"callee", "func", "definition"} or role in (
                structural_roles
            ):
                continue
            parent_data = graph.G.nodes.get(int(parent), {})
            attributes = parent_data.get("attributes") or {}
            leaves = tuple(map(
                int, attributes.get("aggregate_leaf_value_ids", ()),
            ))
            candidate_ids = (
                leaves
                if attributes.get("producer_kind") == "aggregate" and leaves
                else (int(parent),)
            )
            for candidate_id in candidate_ids:
                descriptor = _tensor_descriptor(graph, candidate_id)
                if descriptor is None:
                    candidate = graph.G.nodes.get(candidate_id, {})
                    literal = candidate.get("constant")
                    if literal is None:
                        literal = (candidate.get("attributes") or {}).get(
                            "value"
                        )
                    if str(candidate.get("type")) in {
                        "Constant", "Const", "const",
                    } and isinstance(literal, (bool, int, float)):
                        # A scalar literal operand is an exact rank-0 value.
                        descriptor = {
                            "shape": (),
                            "dtype": (
                                "bool" if isinstance(literal, bool)
                                else "int64" if isinstance(literal, int)
                                else "float64"
                            ),
                        }
                if descriptor is None:
                    # One undescribed numerical operand makes the result
                    # undescribed.  Broadcasting only the operands that
                    # happen to be known silently shrinks a (8, 4, 800, ...)
                    # result to the mask's (8, 4, 1, ...) and every later
                    # shape is then wrong while looking specialized.
                    return ()
                descriptors.append(descriptor)
        return tuple(descriptors)

    def basic_index(node: ast.AST) -> Any:
        """Decode only Python's literal basic-index vocabulary."""

        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Slice):
            return slice(
                None if node.lower is None else basic_index(node.lower),
                None if node.upper is None else basic_index(node.upper),
                None if node.step is None else basic_index(node.step),
            )
        if isinstance(node, ast.Tuple):
            return tuple(basic_index(item) for item in node.elts)
        raise ValueError("index is not a basic literal")

    def symbolic_basic_index_ranks(
        selector: ast.AST, declared_rank: int | None,
    ) -> tuple[int, int]:
        """Constrain rank from basic-index syntax without guessing extents."""

        items = tuple(selector.elts) if isinstance(selector, ast.Tuple) else (
            selector,
        )
        consumed = sum(
            1 for item in items
            if not (
                isinstance(item, ast.Constant)
                and item.value in {None, Ellipsis}
            )
        )
        has_ellipsis = any(
            isinstance(item, ast.Constant) and item.value is Ellipsis
            for item in items
        )
        source_rank = max(int(declared_rank or 0), consumed)
        if has_ellipsis and declared_rank is None:
            # Ellipsis alone does not state how many axes it spans.  Preserve
            # the minimum proven by explicit selectors; later operations may
            # strengthen the same graph identity.
            source_rank = consumed
        dropped = sum(
            1 for item in items
            if not isinstance(item, ast.Slice)
            and not (
                isinstance(item, ast.Constant)
                and item.value in {None, Ellipsis}
            )
        )
        added = sum(
            1 for item in items
            if isinstance(item, ast.Constant) and item.value is None
        )
        return source_rank, max(0, source_rank - dropped + added)

    def retained_basic_index(data, expression):
        # Structural reduction may fold an authored ndim-driven index while
        # retaining the original AST. Consume that exact graph constant first.
        indices = [int(parent) for parent, role in data.get("parents", ())
                   if str(role) == "index" and int(parent) in graph.G]
        if len(indices) == 1:
            index = graph.G.nodes[indices[0]]
            if str(index.get("type")) in {"Constant", "Const", "const"}:
                return _constant_value(index)
        return basic_index(expression.slice)

    fixed_point_iteration = 0
    seen_fixed_point_states: dict[str, int] = {}
    #: The book cells this round's folds posted (``proven_literal``,
    #: ``source_control_specialization_concordance``); the round record
    #: derives from them.
    round_cells: list = []
    changed = True
    while changed:
        fixed_point_iteration += 1
        round_cells = []
        iteration_mutations: list[Any] = []
        tensor_state_before = {
            int(node_id): copy.deepcopy(data.get("tensor"))
            for node_id, data in graph.G.nodes(data=True)
        }
        changed = False
        for node_id in _dependency_order(graph):
            node_id = int(node_id)
            data = graph.G.nodes[node_id]
            operation = str(data.get("op") or data.get("type") or "").casefold()
            # ``history[0]`` over an authored aggregate with an exact leaf
            # ledger IS that leaf.  Keeping the projection as a separate
            # value leaves a consumer reading an operator whose base is a
            # structural tuple; region planning then treats it as an
            # undeclared input of the enclosing function.  The alias is exact
            # (ledger position by literal index), so it is an identity, not
            # a shape inference.
            if operation in {"indexed", "getitem", "subscript"}:
                base_ids = tuple(
                    int(parent)
                    for parent, role in data.get("parents") or ()
                    if str(role) in {"base", "value", "object"}
                )
                index_ids = tuple(
                    int(parent)
                    for parent, role in data.get("parents") or ()
                    if str(role) in {"index", "slice", "subscript"}
                )
                if (
                    len(base_ids) == 1 and len(index_ids) == 1
                    and base_ids[0] in graph.G and index_ids[0] in graph.G
                ):
                    base_attributes = (
                        graph.G.nodes[base_ids[0]].get("attributes") or {}
                    )
                    leaves = tuple(map(int, base_attributes.get(
                        "aggregate_leaf_value_ids", ()
                    )))
                    index_value = constant(graph.G.nodes[index_ids[0]])
                    if (
                        base_attributes.get("producer_kind") == "aggregate"
                        and leaves
                        and all(int(leaf) in graph.G for leaf in leaves)
                        and isinstance(index_value, int)
                        and not isinstance(index_value, bool)
                        and -len(leaves) <= index_value < len(leaves)
                        and int(node_id) != leaves[index_value % len(leaves)]
                    ):
                        replace_alias(
                            node_id, leaves[index_value % len(leaves)]
                        )
                        known.pop(node_id, None)
                        iteration_mutations.append((
                            "aggregate-alias", int(node_id),
                            int(leaves[index_value % len(leaves)]),
                        ))
                        changed = True
                        break
            # The graph spelling may remain ``Call`` after the concrete
            # frontend qualifier was stripped.  Descriptor laws and their
            # publication follow the canonical tensor identity carried by
            # the node, not that structural AST spelling.
            semantic_operation = str(
                (data.get("attributes") or {}).get("tensor_operation")
                or (data.get("attributes") or {}).get("tensor_candidate")
                or operation
            ).casefold()
            existing_tensor = data.get("tensor") or {}
            inherited_descriptor = _tensor_descriptor(graph, node_id)
            if (
                inherited_descriptor is not None
                and (
                    semantic_operation == "tensor"
                    or "shape" not in existing_tensor
                    or (
                        not tuple(existing_tensor.get("shape") or ())
                    )
                )
                and (
                    operation in {
                        "neg", "abs", "sin", "cos", "tan", "exp", "log",
                        "sqrt", "tanh", "clone", "copy", "identity", "indexedstore",
                        "index_set", "setitem", "indexed", "getitem",
                        "subscript", "real", "imag", "conj",
                    }
                    or (
                        operation == "getattr"
                        and str((data.get("attributes") or {}).get(
                            "attribute", ""
                        )) in {"real", "imag"}
                    )
                    or operation in _DTYPE_CAST_OPERATIONS
                    or semantic_operation in _DTYPE_CAST_OPERATIONS
                )
            ):
                # These nodes preserve or alias a tensor descriptor.  Store
                # the proven fact on the producer as well as returning it
                # through _tensor_descriptor so later region cuts retain it.
                data["tensor"] = copy.deepcopy(inherited_descriptor)
                # A member of an aggregate call's return is also written by
                # ``_publish_callsite_return_members``.  Post this
                # re-derivation on the same ``callsite_return_member`` row,
                # DERIVED from the member's own identity and shape-state
                # cells, so the two writers meet on the book.
                if (data.get("attributes") or {}).get(
                    "authored_call_result_projection"
                ):
                    base_id = next((
                        int(parent)
                        for parent, role in data.get("parents") or ()
                        if str(role) == "base" and int(parent) in graph.G
                    ), None)
                    base_leaves = () if base_id is None else tuple(map(int, (
                        graph.G.nodes[base_id].get("attributes") or {}
                    ).get("aggregate_leaf_value_ids") or ()))
                    if int(node_id) in base_leaves:
                        from ..common.tensors.topological_reducer import (
                            node_identity_cell,
                        )
                        from .concordance_declarations import (
                            PLANNER_STRUCTURAL_FOLD,
                        )
                        from .identity_concordance import (
                            SHAPE_STATE_PAGE, current_identity_book,
                            shape_scope_of,
                        )

                        member_cells = []
                        try:
                            member_cells.append(
                                node_identity_cell(graph, int(node_id))
                            )
                        except ValueError:
                            pass
                        member_cells.append(
                            current_identity_book().latest_ref(
                                SHAPE_STATE_PAGE, (
                                    shape_scope_of(graph),
                                    int(data.get("value_id", node_id)),
                                ),
                            )
                        )
                        _post_callsite_return_member(
                            graph, base_id,
                            base_leaves.index(int(node_id)), int(node_id),
                            inherited_descriptor, tuple(member_cells),
                            stage=PLANNER_STRUCTURAL_FOLD,
                        )
            # The semantic operator name is the authored tensor operation
            # (``greater_equal`` for ``>=``), not only the AST spelling.
            comparison_operations = {
                "equal", "eq", "not_equal", "ne", "lt", "le", "gt", "ge",
                "less", "less_equal", "greater", "greater_equal",
                "lte", "gte",
            }
            if (
                operation in {
                    "add", "sub", "mult", "mul", "div", "truediv",
                    "floordiv", "mod", "pow", "maximum", "minimum",
                    "max", "min",
                } | comparison_operations
                or semantic_operation in comparison_operations
            ) and not (
                operation in {"max", "min"}
                and isinstance(data.get("expr_obj"), ast.Call)
                and isinstance(data["expr_obj"].func, ast.Attribute)
            ):
                # Every numerical operand must be described; a scalar
                # literal is an exact rank-0 operand.  Broadcasting only the
                # known operands is the partial-shape hazard fixed for
                # ``where`` above.
                numerical_parents = tuple(
                    int(parent)
                    for parent, role in (data.get("parents") or ())
                    if str(role) not in {"callee", "func", "definition"}
                )
                parent_descriptors = tuple(
                    descriptor
                    for parent in numerical_parents
                    if (
                        descriptor := _operand_descriptor(graph, parent)
                    ) is not None
                )
                if len(parent_descriptors) != len(numerical_parents):
                    parent_descriptors = ()
                if parent_descriptors:
                    try:
                        shape = tuple(np.broadcast_shapes(*(
                            descriptor["shape"]
                            for descriptor in parent_descriptors
                        )))
                    except ValueError:
                        shape = None
                    if shape is not None:
                        data["tensor"] = {
                            "shape": shape,
                            "dtype": (
                                "bool" if (
                                    operation in comparison_operations
                                    or semantic_operation
                                    in comparison_operations
                                ) else next((
                                    descriptor["dtype"]
                                    for descriptor in parent_descriptors
                                    if tuple(descriptor["shape"])
                                ), parent_descriptors[0]["dtype"])
                            ),
                        }
            if operation in {"indexed", "getitem", "subscript"}:
                expression = data.get("expr_obj")
                source_descriptor = None
                index_values: list[Any] = []
                for parent, role in data.get("parents") or ():
                    if str(role) == "base":
                        source_descriptor = _tensor_descriptor(
                            graph, int(parent)
                        )
                    elif str(role) == "index":
                        index_values.append(
                            known.get(int(parent), unresolved)
                        )

                if source_descriptor is not None and isinstance(
                    expression, ast.Subscript
                ):
                    try:
                        from ..common.tensors.abstraction_methods.indexing import (
                            normalize_basic_index,
                        )

                        # Tensor topology expansion makes an ellipsis explicit
                        # as a structural tuple dependency (using the source
                        # rank).  That parent is the canonical index value;
                        # the surface AST is only a fallback for indexes that
                        # remained literal through reduction.
                        if index_values and all(
                            value is not unresolved for value in index_values
                        ):
                            normalized_index = (
                                index_values[0]
                                if len(index_values) == 1 else tuple(index_values)
                            )
                        else:
                            normalized_index = retained_basic_index(data, expression)
                        _axes, result_shape = normalize_basic_index(
                            normalized_index,
                            tuple(source_descriptor["shape"]),
                        )
                    except (IndexError, TypeError, ValueError, NotImplementedError):
                        result_shape = None
                    if result_shape is not None:
                        attributes = dict(data.get("attributes") or {})
                        attributes["basic_index_axes"] = tuple(
                            (tuple(map(int, indices)), bool(drop_axis))
                            for indices, drop_axis in _axes
                        )
                        attributes["basic_index_source_shape"] = tuple(
                            source_descriptor["shape"]
                        )
                        data["attributes"] = attributes
                        data["tensor"] = {
                            "shape": tuple(result_shape),
                            "dtype": source_descriptor["dtype"],
                        }
            if (
                operation in {"indexedstore", "index_set", "setitem"}
                and not (data.get("attributes") or {}).get(
                    "basic_index_axes"
                )
            ):
                expression = data.get("expr_obj")
                base = next((
                    int(parent)
                    for parent, role in data.get("parents") or ()
                    if str(role) == "base"
                ), None)
                source_descriptor = (
                    None if base is None
                    else _tensor_descriptor(graph, base)
                )
                if (
                    source_descriptor is not None
                    and isinstance(expression, ast.Subscript)
                ):
                    try:
                        from ..common.tensors.abstraction_methods.indexing import (
                            normalize_basic_index,
                        )

                        axes, _selection_shape = normalize_basic_index(
                            retained_basic_index(data, expression),
                            tuple(source_descriptor["shape"]),
                        )
                    except (
                        IndexError, TypeError, ValueError,
                        NotImplementedError,
                    ):
                        axes = None
                    if axes is not None:
                        attributes = dict(data.get("attributes") or {})
                        attributes["basic_index_axes"] = tuple(
                            (tuple(map(int, indices)), bool(drop_axis))
                            for indices, drop_axis in axes
                        )
                        attributes["basic_index_source_shape"] = tuple(
                            source_descriptor["shape"]
                        )
                        data["attributes"] = attributes
                        data["tensor"] = copy.deepcopy(source_descriptor)
            attributes = data.get("attributes") or {}
            tensor_candidate = str(
                attributes.get("tensor_candidate") or operation
            ).casefold()
            if tensor_candidate == "arange":
                positional_values = []
                for position, parent in enumerate(positional(data)):
                    value = known.get(int(parent), unresolved)
                    if value is unresolved:
                        # Unresolved HERE only means this copy was not told.
                        # The bound is a formal whose callsites may all have
                        # proved the same literal, and that is a fact about
                        # the formal.
                        try:
                            value = _source_static_literal(graph, int(parent))
                        except (ValueError, KeyError, TypeError):
                            value = unresolved
                    if value is unresolved:
                        positional_values = []
                        break
                    positional_values.append(value)
                expression = data.get("expr_obj")
                if not positional_values and isinstance(expression, ast.Call):
                    try:
                        positional_values = [
                            ast.literal_eval(argument)
                            for argument in expression.args[:3]
                        ]
                    except (TypeError, ValueError):
                        positional_values = []
                if positional_values:
                    if len(positional_values) == 1:
                        start, stop, step = 0, positional_values[0], 1
                    else:
                        start, stop = positional_values[:2]
                        step = (
                            positional_values[2]
                            if len(positional_values) >= 3 else 1
                        )
                    if stop is None:
                        start, stop = 0, start
                    try:
                        count = len(range(int(start), int(stop), int(step)))
                    except (TypeError, ValueError, ZeroDivisionError):
                        count = 0
                    if count > 0:
                        # The element dtype is the constructor's ``dtype=``
                        # argument when it names one (``AbstractTensor.arange(
                        # n, dtype=like.get_dtype())`` is ``like``'s dtype);
                        # float64 only when the source names none.
                        named_dtype = next((
                            named
                            for parent, role in data.get("parents") or ()
                            if str(role).casefold() == "kw:dtype"
                            and int(parent) in graph.G
                            for named in (
                                _dtype_argument_name(graph, int(parent)),
                            )
                            if named is not None
                        ), None)
                        data["tensor"] = {
                            "shape": (count,),
                            "dtype": named_dtype or "float64",
                        }
                        normalized = dict(data.get("attributes") or {})
                        normalized.update({
                            "arange_start": float(start),
                            "arange_step": float(step),
                            "arange_count": count,
                        })
                        data["attributes"] = normalized
            if tensor_candidate in {
                "zeros_like", "empty_like", "ones_like", "full_like",
            }:
                source_descriptor = next((
                    _tensor_descriptor(graph, int(parent))
                    for parent, role in (data.get("parents") or ())
                    if str(role) not in {
                        "callee", "func", "function", "definition",
                    }
                    and _tensor_descriptor(graph, int(parent)) is not None
                ), None)
                if source_descriptor is not None:
                    data["tensor"] = copy.deepcopy(source_descriptor)
            if (
                tensor_candidate in {
                    "stack", "cat", "concat", "concatenate", "where",
                }
            ):
                operand_descriptors = tensor_data_descriptors(data)
                if tensor_candidate == "stack" and operand_descriptors:
                    operand_shape = tuple(operand_descriptors[0]["shape"])
                    if all(
                        tuple(descriptor["shape"]) == operand_shape
                        for descriptor in operand_descriptors
                    ):
                        axis = structural_call_argument(
                            data,
                            roles={
                                "dim", "axis", "kw:dim", "kw:axis",
                                "arg:1",
                            },
                            keywords={"dim", "axis"},
                            default=0,
                        )
                        position = int(axis) % (len(operand_shape) + 1)
                        data["tensor"] = {
                            "shape": (
                                *operand_shape[:position],
                                len(operand_descriptors),
                                *operand_shape[position:],
                            ),
                            "dtype": operand_descriptors[0]["dtype"],
                        }
                elif tensor_candidate in {
                    "cat", "concat", "concatenate",
                } and operand_descriptors:
                    shapes = tuple(
                        tuple(descriptor["shape"])
                        for descriptor in operand_descriptors
                    )
                    rank = len(shapes[0])
                    axis = structural_call_argument(
                        data,
                        roles={
                            "dim", "axis", "kw:dim", "kw:axis", "arg:1",
                        },
                        keywords={"dim", "axis"},
                        default=0,
                    )
                    position = int(axis) % rank if rank else 0
                    if rank and all(
                        len(shape) == rank and all(
                            index == position or extent == shapes[0][index]
                            for index, extent in enumerate(shape)
                        )
                        for shape in shapes
                    ):
                        result_shape = list(shapes[0])
                        result_shape[position] = sum(
                            shape[position] for shape in shapes
                        )
                        data["tensor"] = {
                            "shape": tuple(result_shape),
                            "dtype": operand_descriptors[0]["dtype"],
                        }
                elif tensor_candidate == "where" and operand_descriptors:
                    try:
                        result_shape = tuple(np.broadcast_shapes(*(
                            descriptor["shape"]
                            for descriptor in operand_descriptors
                        )))
                    except ValueError:
                        result_shape = None
                    # The result dtype is the promotion of the two VALUE arms
                    # (``np.where`` semantics, and the elementwise-binary
                    # rule's ``np.result_type`` above), never the first
                    # non-bool operand: BACKWARD_RULES["maximum"]'s
                    # ``where(x<y, 0, 0.5)`` was typed int64 from the literal
                    # 0, cast 0.5 to 0, and the outer where/Mul then made
                    # d Max(0,u)*y / du read 0.0 instead of y.
                    value_arms = [
                        str(descriptor.get("dtype") or "unknown")
                        for descriptor in (
                            operand_descriptors[1:]
                            if len(operand_descriptors) == 3
                            else operand_descriptors
                        )
                    ]
                    known_arms = [
                        dtype for dtype in value_arms if dtype != "unknown"
                    ]
                    if result_shape is not None and known_arms:
                        data["tensor"] = {
                            "shape": result_shape,
                            "dtype": str(np.result_type(
                                *(np.dtype(dtype) for dtype in known_arms)
                            )),
                        }
            if tensor_candidate in {"gather", "index_select"}:
                source_descriptor = next((
                    _tensor_descriptor(graph, int(parent))
                    for parent, role in (data.get("parents") or ())
                    if str(role) in {"operand", "value", "base"}
                    and _tensor_descriptor(graph, int(parent)) is not None
                ), None)
                index_descriptor = next((
                    _tensor_descriptor(graph, int(parent))
                    for parent, role in (data.get("parents") or ())
                    if str(role) in {"indices", "index", "arg:0", "arg:1"}
                    and _tensor_descriptor(graph, int(parent)) is not None
                ), None)
                axis = structural_call_argument(
                    data,
                    roles={
                        "dim", "axis", "kw:dim", "kw:axis", "arg:1",
                    },
                    keywords={"dim", "axis"},
                    default=0,
                )
                if source_descriptor is not None and index_descriptor is not None:
                    source_shape = tuple(source_descriptor["shape"])
                    position = int(axis) % len(source_shape)
                    attributes = dict(data.get("attributes") or {})
                    attributes.setdefault("dim", position)
                    data["attributes"] = attributes
                    data["tensor"] = {
                        "shape": (
                            *source_shape[:position],
                            *tuple(index_descriptor["shape"]),
                            *source_shape[position + 1:],
                        ),
                        "dtype": source_descriptor["dtype"],
                    }
            if tensor_candidate in {
                    "transpose", "swapaxes", "transpose_last2", "t"
                }:
                source_descriptor = next((
                    _tensor_descriptor(graph, int(parent))
                    for parent, role in (data.get("parents") or ())
                    if str(role) in {"operand", "arg:0", "value"}
                    and _tensor_descriptor(graph, int(parent)) is not None
                ), None)
                if source_descriptor is not None:
                    source_shape = tuple(source_descriptor["shape"])
                    result_shape = (
                        (*source_shape[:-2], source_shape[-1], source_shape[-2])
                        if len(source_shape) >= 2 else source_shape
                    )
                    data["tensor"] = {
                        "shape": result_shape,
                        "dtype": source_descriptor["dtype"],
                    }
            if tensor_candidate in {"matmul", "mm"}:
                operand_descriptors = tuple(
                    descriptor
                    for parent, role in (data.get("parents") or ())
                    if str(role) not in {"callee", "func", "definition"}
                    and (
                        descriptor := _tensor_descriptor(graph, int(parent))
                    ) is not None
                )
                if len(operand_descriptors) >= 2:
                    left_shape = tuple(operand_descriptors[0]["shape"])
                    right_shape = tuple(operand_descriptors[1]["shape"])
                    result_shape = None
                    if len(left_shape) == 1 and len(right_shape) == 1:
                        result_shape = ()
                    elif len(left_shape) == 1 and len(right_shape) >= 2:
                        result_shape = (*right_shape[:-2], right_shape[-1])
                    elif len(left_shape) >= 2 and len(right_shape) == 1:
                        result_shape = (*left_shape[:-2], left_shape[-2])
                    elif len(left_shape) >= 2 and len(right_shape) >= 2:
                        try:
                            batch_shape = tuple(np.broadcast_shapes(
                                left_shape[:-2], right_shape[:-2],
                            ))
                        except ValueError:
                            batch_shape = None
                        if batch_shape is not None:
                            result_shape = (
                                *batch_shape, left_shape[-2], right_shape[-1],
                            )
                    if result_shape is not None:
                        data["tensor"] = {
                            "shape": tuple(result_shape),
                            "dtype": operand_descriptors[0]["dtype"],
                        }
            if tensor_candidate in {"reshape", "view"}:
                source_descriptor = next((
                    _tensor_descriptor(graph, int(parent))
                    for parent, role in (data.get("parents") or ())
                    if str(role) in {"operand", "value"}
                    and _tensor_descriptor(graph, int(parent)) is not None
                ), None)
                shape_arguments = sorted(
                    (
                        int(position),
                        known.get(int(parent), unresolved),
                    )
                    for parent, role in (data.get("parents") or ())
                    for position in (_positional_argument_index(str(role)),)
                    if position is not None
                    and known.get(int(parent), unresolved) is not unresolved
                )
                named_shape = next((
                    known.get(int(parent), unresolved)
                    for parent, role in (data.get("parents") or ())
                    if str(role) in {"shape", "new_shape"}
                    and known.get(int(parent), unresolved) is not unresolved
                ), unresolved)
                result_shape = (
                    named_shape
                    if named_shape is not unresolved
                    else tuple(value for _position, value in shape_arguments)
                )
                if (
                    isinstance(result_shape, tuple)
                    and len(result_shape) == 1
                    and isinstance(result_shape[0], (tuple, list))
                ):
                    result_shape = result_shape[0]
                if source_descriptor is not None:
                    if isinstance(result_shape, (tuple, list)) and all(
                        isinstance(extent, int) and not isinstance(extent, bool)
                        for extent in result_shape
                    ):
                        resolved_shape = [int(extent) for extent in result_shape]
                        inferred = [
                            index for index, extent in enumerate(resolved_shape)
                            if extent == -1
                        ]
                        if len(inferred) <= 1 and all(
                            extent > 0 or extent == -1
                            for extent in resolved_shape
                        ):
                            source_count = int(np.prod(
                                tuple(source_descriptor["shape"]), dtype=np.int64
                            ))
                            known_count = int(np.prod(
                                tuple(
                                    extent for extent in resolved_shape
                                    if extent != -1
                                ),
                                dtype=np.int64,
                            ))
                            if inferred and known_count and source_count % known_count == 0:
                                resolved_shape[inferred[0]] = source_count // known_count
                            if not inferred or -1 not in resolved_shape:
                                data["tensor"] = {
                                    "shape": tuple(resolved_shape),
                                    "dtype": source_descriptor["dtype"],
                                }
            if (
                tensor_candidate in {
                    "sum", "mean", "prod", "min", "max", "any", "all",
                    "argmin", "argmax",
                }
            ):
                method_style = any(
                    str(role) in {"operand", "value", "self", "receiver"}
                    for _parent, role in (data.get("parents") or ())
                )
                source_descriptor = next((
                    _tensor_descriptor(graph, int(parent))
                    for parent, role in (data.get("parents") or ())
                    if str(role) in (
                        {"operand", "value", "self", "receiver"}
                        if method_style else {"arg:0"}
                    )
                    and _tensor_descriptor(graph, int(parent)) is not None
                ), None)
                axis = structural_call_argument(
                    data,
                    roles={
                        "dim", "axis", "kw:dim", "kw:axis",
                        "arg:0" if method_style else "arg:1",
                    },
                    keywords={"dim", "axis"},
                    default=unresolved,
                )
                keepdim = structural_call_argument(
                    data,
                    roles={
                        "keepdim", "kw:keepdim",
                        "arg:1" if method_style else "arg:2",
                    },
                    keywords={"keepdim"},
                    default=False,
                )
                if source_descriptor is not None and axis is not unresolved:
                    source_shape = tuple(source_descriptor["shape"])
                    source_row_shape = tuple(map(
                        int,
                        source_descriptor.get("sequence_row_shape") or (),
                    ))
                    source_rank = int(source_descriptor.get(
                        "rank", len(source_shape),
                    ))
                    dynamic_leading_extent = bool(
                        not source_shape
                        and str(source_descriptor.get(
                            "metadata_state") or ""
                        ) == "dynamic"
                        and source_rank == 1 + len(source_row_shape)
                    )
                    raw_axes = (
                        tuple(axis) if isinstance(axis, (tuple, list))
                        else (int(axis),)
                    )
                    axes = tuple(sorted({
                        int(item) % source_rank for item in raw_axes
                    })) if source_rank else ()
                    logical_shape = (
                        (None, *source_row_shape)
                        if dynamic_leading_extent else source_shape
                    )
                    if bool(keepdim):
                        result_shape = tuple(
                            1 if index in axes else extent
                            for index, extent in enumerate(logical_shape)
                        )
                    else:
                        result_shape = tuple(
                            extent for index, extent in enumerate(logical_shape)
                            if index not in axes
                        )
                    if all(extent is not None for extent in result_shape):
                        data["tensor"] = {
                            "shape": tuple(map(int, result_shape)),
                            "dtype": (
                                "int64"
                                if tensor_candidate in {"argmin", "argmax"}
                                else source_descriptor["dtype"]
                            ),
                        }
                    attributes = dict(data.get("attributes") or {})
                    attributes.setdefault("dim", axis)
                    attributes.setdefault("keepdim", bool(keepdim))
                    data["attributes"] = attributes
            if str(data.get("type")) in {"Constant", "Const", "const"}:
                value = constant(data)
            else:
                value = evaluate(node_id, data)
            if value is unresolved:
                continue
            if (
                isinstance(value, _ProgramABIValueFact)
                and value.storage == "span"
            ):
                # Publish the proved rank before hierarchy freezes values.
                # A keyed Tensor lookup can have runtime extents and rank one,
                # while ``-values[i]`` / ``abs(values[i])`` prove rank zero
                # without static extents. Zero is not an absent rank: replacing
                # it with one promotes scalar reads to anonymous dynamic spans.
                rank = (
                    len(value.shape)
                    if value.shape is not None
                    else int(1 if value.rank is None else value.rank)
                )
                attributes = dict(data.get("attributes") or {})
                attributes.update({
                    "program_abi_storage": "span",
                    "program_abi_rank": int(rank),
                    "program_abi_dtype": str(value.dtype or "unknown"),
                })
                data["attributes"] = attributes
                binding_name = str(attributes.get("binding_name") or "")
                callsite_descriptor = (
                    graph.G.graph.get("planner_tensor_descriptors") or {}
                ).get(binding_name)
                if isinstance(callsite_descriptor, Mapping) and (
                    callsite_descriptor.get("shape") is not None
                ):
                    # The rank-only ProgramABI fact is the fallback boundary;
                    # it cannot overwrite the exact descriptor already bound
                    # by this callsite.  Doing so changed ``mask: (2,)`` back
                    # into a dynamic empty shape while leaving the planner
                    # ledger itself correct, splitting one formal identity
                    # into two incompatible facts.
                    data["tensor"] = copy.deepcopy(dict(callsite_descriptor))
                    from .identity_concordance import (
                        record_proven_shape,
                        shape_scope_of,
                    )

                    record_proven_shape(
                        shape_scope_of(graph),
                        int(data.get("value_id", node_id)),
                        tuple(callsite_descriptor.get("shape") or ()),
                        callsite_descriptor.get("dtype"),
                        int(_dependency_levels(graph).get(int(node_id), 0)),
                    )
                else:
                    data["tensor"] = {
                        "shape": tuple(value.shape or ()),
                        "dtype": str(value.dtype or "unknown"),
                        "rank": int(rank),
                        **(
                            {"metadata_state": "dynamic"}
                            if value.shape is None else {}
                        ),
                    }
            if (
                node_id in loop_carried_initial_ids
                and not isinstance(value, _ProgramABIValueFact)
            ) or (
                node_id in mutated_sequence_ids
                and isinstance(value, (
                    list, tuple, set, frozenset, dict, bytes, bytearray,
                ))
            ):
                # A mutated collection's initializer is not its runtime
                # contents. In particular, sorting a list populated by a
                # retained append loop must not sort its initial empty list.
                # A specialization that is NOT a collection (``None`` for a
                # parameter the caller left at its default) protects nothing:
                # the name never holds contents, the mutation guarded by
                # ``name is not None`` is dead, and folding that guard is what
                # removes the dead rows.
                # Likewise, this literal may be the entry arm of a Phi, not an invariant
                # fact for the loop's condition/body.  Keeping it out of the
                # structural-known table prevents the fixed point from
                # replacing expressions such as ``iters < max_iters`` with
                # their iteration-zero value.  A ProgramABI type/storage fact
                # is different: the native loop cannot change a physical
                # scalar/span/record slot into Python None, so that fact stays
                # invariant even when the value in the slot is loop-carried.
                continue
            if isinstance(value, _StructuralValueAlias):
                alias_expression = data.get("expr_obj")
                if isinstance(alias_expression, ast.IfExp):
                    control_signature = _ast_source_signature(alias_expression)
                    retained_control_ids = tuple(
                        int(source_id)
                        for source_id, record in _source_control_records(
                            graph.G
                        ).items()
                        if isinstance(record.get("expression"), ast.IfExp)
                        and _ast_source_signature(record["expression"])
                        == control_signature
                    )
                    specialized = list(graph.G.graph.get(
                        "structurally_specialized_conditional_node_ids", ()
                    ))
                    specialized.extend((
                        int(node_id), *retained_control_ids,
                    ))
                    graph.G.graph[
                        "structurally_specialized_conditional_node_ids"
                    ] = tuple(dict.fromkeys(map(int, specialized)))
                    parents = {
                        str(role): int(parent)
                        for parent, role in data.get("parents") or ()
                    }
                    predicate = known.get(parents.get("test"), unresolved)
                    selected_role = (
                        "body" if bool(predicate) else "orelse"
                    )
                    # The row is the record of this fold: DERIVED from the
                    # test's ``proven_literal`` cell and the control cell.
                    # A test absent from ``known`` is recorded as
                    # ``Unresolved(PREDICATE_NOT_KNOWN)`` -- the truthy
                    # sentinel no longer records ``selected_role = "body"``
                    # as a proof.  Whether that absence should also stop
                    # the fold is held for the user (plan 80, A9.2); the
                    # fold below still selects as before.
                    for retained_control_id in (
                        retained_control_ids or (int(node_id),)
                    ):
                        round_cells.extend(_post_control_specialization(
                            graph, int(retained_control_id),
                            {
                                "graph_control_id": int(node_id),
                                "source_control_id": int(
                                    retained_control_id
                                ),
                                "predicate_value_id": parents.get("test"),
                                "selected_role": selected_role,
                                "proof": "structural_constant_predicate",
                            },
                            parents.get("test"),
                            predicate_known=predicate is not unresolved,
                        ))
                replace_alias(node_id, value.source_id)
                known.pop(node_id, None)
                iteration_mutations.append((
                    "structural-alias", int(node_id), int(value.source_id),
                ))
                changed = True
                break
            if (
                node_id not in known
                or not same_structural_value(known[node_id], value)
            ):
                known[node_id] = value
                iteration_mutations.append((
                    "known", int(node_id), str(operation), repr(value)[:160],
                ))
                changed = True
            # ABI facts prove source type and physical presence; they are not
            # runtime literals.  Keep the authored SSA producer intact so a
            # record field remains a normal dataflow value while comparisons
            # such as ``field is not None`` can still be decided here.
            if isinstance(value, _ProgramABIValueFact):
                continue
            # The same rule holds for an authored aggregate of facts.  Folding
            # ``(hub, angle)`` into a Constant severs the Tuple from its leaf
            # producers, the now-unconsumed span parameters are dropped as
            # unused, and every later call boundary sees a ledger whose leaf
            # identities no longer exist.  The tuple stays a known structural
            # value for projection/isinstance decisions; its producer stays.
            if _contains_program_abi_fact(value):
                continue
            if str(data.get("type")) not in {"Constant", "Const", "const"}:
                replace(node_id, value)

        # Descriptor propagation and structural folding are one fixed-point
        # problem, but progress is a fact about a COMPLETED round.  An indexed
        # Program-ABI span can acquire a provisional tensor descriptor above
        # and then be restored to its authoritative callsite/ABI descriptor
        # later in the same node visit.  Treating that transient intermediate
        # value as progress made an otherwise settled graph repeat forever.
        # Compare the round boundaries, and record only the net descriptor
        # transitions in both the trace and concordance.
        net_tensor_mutations = []
        for changed_node_id, changed_data in graph.G.nodes(data=True):
            changed_node_id = int(changed_node_id)
            before = tensor_state_before.get(changed_node_id)
            after = changed_data.get("tensor")
            if before == after:
                continue
            net_tensor_mutations.append((
                "tensor", changed_node_id,
                str(changed_data.get("op") or changed_data.get("type") or ""),
                _callsite_descriptor_receipt(before),
                _callsite_descriptor_receipt(after),
            ))
        if net_tensor_mutations:
            iteration_mutations.extend(net_tensor_mutations)
            changed = True

        fixed_point_facts = tuple(
            (
                int(node_id),
                str(data.get("type") or ""),
                repr(_callsite_descriptor_receipt(data.get("tensor"))),
                repr(known.get(int(node_id), unresolved))[:240],
            )
            for node_id, data in sorted(graph.G.nodes(data=True))
        )
        fixed_point_digest = hashlib.sha256(
            repr(fixed_point_facts).encode("utf-8")
        ).hexdigest()[:16]
        repeated_iteration = seen_fixed_point_states.get(fixed_point_digest)
        seen_fixed_point_states.setdefault(
            fixed_point_digest, fixed_point_iteration
        )
        # The round record: row (function, round), DERIVED from the
        # ``proven_literal`` / control-specialization cells posted in this
        # round; a round with no posts derives from the previous round's
        # cell, and a first round with none has no source to name.
        from .concordance_declarations import (
            PLANNER_STRUCTURAL_FOLD, STRUCTURAL_SPECIALIZATION_FIXED_POINT,
        )
        from .identity_concordance import (
            Derived, Mode, RAW_PRIMITIVE, Unsourced, current_identity_book,
        )

        fixed_point_book = current_identity_book()
        fixed_point_row = (
            str(graph.G.graph.get("function_name") or ""),
            int(fixed_point_iteration),
        )
        fixed_point_sources = tuple(dict.fromkeys(round_cells))
        if not fixed_point_sources and fixed_point_iteration > 1:
            previous_round = fixed_point_book.latest_ref(
                STRUCTURAL_SPECIALIZATION_FIXED_POINT,
                (fixed_point_row[0], fixed_point_iteration - 1),
            )
            if previous_round is not None:
                fixed_point_sources = (previous_round,)
        if not fixed_point_sources:
            # A first round that folded nothing: the record is about this
            # function's graph and nothing else on the book changed, so it
            # derives from the function's ``function_address`` cell.
            from .concordance_declarations import FUNCTION_ADDRESS

            for candidate in (
                graph.G.graph.get("qualified_name"),
                graph.G.graph.get("function_name"),
            ):
                if candidate is None:
                    continue
                address = fixed_point_book.latest_ref(
                    FUNCTION_ADDRESS, (str(candidate),),
                )
                if address is not None:
                    fixed_point_sources = (address,)
                    break
        fixed_point_fact = (
            fixed_point_digest,
            bool(changed),
            tuple(net_tensor_mutations),
        )
        previous_record = fixed_point_book.latest_ref(
            STRUCTURAL_SPECIALIZATION_FIXED_POINT, fixed_point_row,
        )
        if previous_record is not None and fixed_point_sources:
            previous_sources = {
                source.key
                for source, _stage in fixed_point_book.edges_into(previous_record)
            }
            if previous_sources == {cell.key for cell in fixed_point_sources}:
                # The fold ran again on this graph.  The same digest from the
                # same cells is not a new record; a different digest from the
                # same cells (node data changed with no cell of its own --
                # a raw tensor annotation) has no cause the book can name and
                # is recorded under the latch.
                fixed_point_sources = ()
                if fixed_point_book.pages[
                    STRUCTURAL_SPECIALIZATION_FIXED_POINT.name
                ].latest(fixed_point_row) == fixed_point_fact:
                    previous_record = False
        if previous_record is not False:
            fixed_point_book.post(
                STRUCTURAL_SPECIALIZATION_FIXED_POINT, fixed_point_row,
                fixed_point_fact,
                stage=PLANNER_STRUCTURAL_FOLD,
                provenance=(
                    Derived(fixed_point_sources) if fixed_point_sources
                    else Unsourced(RAW_PRIMITIVE)
                ),
                mode=Mode.REVISE,
            )
        if _progress is not None and (
            fixed_point_iteration <= 3
            or repeated_iteration is not None
            or not changed
        ):
            _progress({
                "fold_state": "fixed-point-iteration",
                "fold_iteration": fixed_point_iteration,
                "changed": bool(changed),
                "state_digest": fixed_point_digest,
                "repeats_iteration": repeated_iteration,
                "mutation_count": len(iteration_mutations),
                "mutations": tuple(iteration_mutations[:8]),
            })
        if changed and repeated_iteration is not None:
            raise RuntimeError(
                "structural specialization fixed point repeated a prior "
                "state while still claiming progress: "
                f"function={graph.G.graph.get('function_name')!r}, "
                f"iteration={fixed_point_iteration}, "
                f"repeats_iteration={repeated_iteration}, "
                f"digest={fixed_point_digest}, "
                f"mutations={tuple(iteration_mutations[:8])!r}"
            )

    # A callsite specialization may make source control decidable even when
    # the selected value remains runtime numerical data.  Remove only the
    # unselected lexical compartment and replace its merge Phi with an alias
    # to the selected producer.  The proof is the graph constant above; no
    # runtime tensor value or Python callable participates.
    branch_memberships = _branch_compartments(graph)
    pruned = True
    while pruned:
        pruned = False
        for control_id, control_data in tuple(graph.G.nodes(data=True)):
            expression = control_data.get("expr_obj")
            if not isinstance(expression, ast.If):
                continue
            predicate_id = next((
                int(parent)
                for parent, role in control_data.get("parents") or ()
                if str(role) == "test"
            ), None)
            if predicate_id is None or predicate_id not in graph.G:
                continue
            predicate = constant(graph.G.nodes[predicate_id])
            if predicate is unresolved:
                continue
            selected_role = "body" if bool(predicate) else "orelse"
            rejected_role = "orelse" if bool(predicate) else "body"
            for phi_id, phi_data in tuple(graph.G.nodes(data=True)):
                if str(phi_data.get("type")) != "Phi":
                    continue
                phi_parents = {
                    str(role): int(parent)
                    for parent, role in phi_data.get("parents") or ()
                }
                source_conditional_id = (
                    phi_data.get("attributes") or {}
                ).get("source_conditional_id")
                if (
                    phi_parents.get("test") != predicate_id
                    and (
                        source_conditional_id is None
                        or int(source_conditional_id) != int(control_id)
                    )
                ):
                    continue
                selected = phi_parents.get(selected_role)
                if selected is not None:
                    replace_alias(int(phi_id), int(selected))
            rejected_statements = (
                expression.orelse if rejected_role == "orelse"
                else expression.body
            )
            rejected_ast_ids = {
                id(member)
                for statement in rejected_statements
                for member in ast.walk(statement)
            }
            rejected_signatures = {
                _ast_source_signature(member)
                for statement in rejected_statements
                for member in ast.walk(statement)
            }
            # ``if predicate: return value`` guards the lexical tail even
            # though Python spells no explicit ``else``.  Function ingestion
            # retains the exact top-level statement list; when the selected
            # arm is terminal, include every following statement in the
            # rejected compartment.  This is source control, not a heuristic
            # over node numbering, and it survives clean-graph AST copies via
            # source signatures.
            function_body = tuple(
                graph.G.graph.get("function_body") or ()
            )
            source_index = next((
                index
                for index, statement in enumerate(function_body)
                if isinstance(statement, ast.If)
                and _ast_source_signature(statement)
                == _ast_source_signature(expression)
            ), None)
            selected_statements = (
                tuple(expression.body)
                if selected_role == "body"
                else tuple(expression.orelse)
            )
            if (
                source_index is not None
                and selected_statements
                and isinstance(selected_statements[-1], (ast.Return, ast.Raise))
            ):
                rejected_signatures.update(
                    _ast_source_signature(member)
                    for statement in function_body[source_index + 1:]
                    for member in ast.walk(statement)
                )
            selected_return_id = None
            if (
                selected_statements
                and isinstance(selected_statements[-1], ast.Return)
                and selected_statements[-1].value is not None
            ):
                returned_expression = selected_statements[-1].value
                if isinstance(returned_expression, ast.Name):
                    history = tuple(map(
                        int,
                        (graph.G.graph.get("identity_table") or {}).get(
                            returned_expression.id, ()
                        ),
                    ))
                    selected_return_id = next((
                        value_id for value_id in reversed(history)
                        if value_id in graph.G
                    ), None)
                if selected_return_id is None:
                    returned_signature = _ast_source_signature(
                        returned_expression
                    )
                    matches = tuple(
                        int(node_id)
                        for node_id, data in graph.G.nodes(data=True)
                        if isinstance(data.get("authored_expr_obj", data.get("expr_obj")), ast.AST)
                        and _ast_source_signature(data.get("authored_expr_obj", data["expr_obj"]))
                        == returned_signature
                    )
                    if len(matches) == 1:
                        selected_return_id = matches[0]
            control_signature = _ast_source_signature(expression)
            retained_control_id = next((
                int(source_id)
                for source_id, record in _source_control_records(
                    graph.G
                ).items()
                if isinstance(record.get("expression"), ast.If)
                and _ast_source_signature(record["expression"])
                == control_signature
            ), int(control_id))
            # The reducer represents a value assigned in only one arm and used
            # later as an external Input placeholder, retaining the real arm
            # producer immediately before it in that name's identity history.
            # Once this exact conditional is structurally selected, that
            # placeholder is no longer external: collapse it only when one
            # surviving predecessor belongs to the selected compartment.  The
            # branch-membership receipt prevents this from guessing across
            # loops, unrelated assignments, or an undecided conditional.
            identities = graph.G.graph.get("identity_table") or {}
            for _name, history in tuple(identities.items()):
                ordered = tuple(map(int, history or ()))
                if len(ordered) < 2:
                    continue
                placeholder_id = ordered[-1]
                placeholder = graph.G.nodes.get(placeholder_id, {})
                if not (
                    placeholder.get("type") == "Input"
                    and (placeholder.get("attributes") or {}).get(
                        "binding_kind"
                    ) == "external"
                ):
                    continue
                selected_candidates = tuple(dict.fromkeys(
                    value_id
                    for value_id in ordered[:-1]
                    if value_id in graph.G
                    and (int(retained_control_id), str(selected_role))
                    in branch_memberships.get(
                        int(value_id), frozenset()
                    )
                ))
                if len(selected_candidates) == 1:
                    replace_alias(
                        int(placeholder_id), int(selected_candidates[0])
                    )
            rejected_nodes = {
                int(node_id)
                for node_id, data in graph.G.nodes(data=True)
                if id(data.get("expr_obj")) in rejected_ast_ids
                or isinstance(data.get("expr_obj"), ast.AST)
                and _ast_source_signature(data["expr_obj"])
                in rejected_signatures
                or (int(retained_control_id), str(rejected_role)) in (
                    branch_memberships.get(int(node_id), frozenset())
                )
            }
            specialized = list(
                graph.G.graph.get(
                    "structurally_specialized_conditional_node_ids", ()
                )
            )
            specialized.extend((
                int(control_id), int(retained_control_id),
            ))
            graph.G.graph[
                "structurally_specialized_conditional_node_ids"
            ] = tuple(dict.fromkeys(map(int, specialized)))
            round_cells.extend(_post_control_specialization(
                graph, int(retained_control_id),
                {
                    "graph_control_id": int(control_id),
                    "source_control_id": int(retained_control_id),
                    "predicate_value_id": int(predicate_id),
                    "selected_role": str(selected_role),
                    "rejected_role": str(rejected_role),
                    "proof": "structural_constant_predicate",
                },
                int(predicate_id), predicate_known=True,
            ))
            for node_id in sorted(rejected_nodes, reverse=True):
                remove_node(node_id)
            remove_node(int(control_id))
            identities = graph.G.graph.get("identity_table") or {}
            if selected_return_id is not None:
                graph.roots = [int(selected_return_id)]
                # ``selected_return_id`` is the return STATEMENT's value: for
                # ``return a, b, c`` that is the Tuple node, not any slot.
                # The per-slot values of every return site are recorded by
                # the reducer in ``return_slot_values``, keyed by the source
                # span of the returned expression; that ledger is the
                # authority for what each output slot carries at this site.
                # The reducer's own convention (its Return handling) is that
                # a positional slot's identity history lists the return-site
                # values, while a name-spelled slot IS a variable whose own
                # binding history already holds the returned value.  Writing
                # the whole Tuple under every output name -- including a
                # variable's name such as ``metrics`` -- rebinds that variable
                # to the 3-tuple in the identity table; the planner's
                # name-history repair then feeds the tuple to
                # ``_propose_dt_pen(metrics, ...)`` and refuses the call with
                # ``aggregate call binding for 'metrics' ... 3 != 0``.
                returned_expression = selected_statements[-1].value
                selected_span = (
                    int(returned_expression.lineno),
                    int(getattr(returned_expression, "col_offset", -1)),
                    int(getattr(returned_expression, "end_lineno", -1)),
                    int(getattr(returned_expression, "end_col_offset", -1)),
                ) if getattr(returned_expression, "lineno", None) is not None else None
                return_slots = dict(graph.G.graph.get("return_slot_values") or {})
                selected_slot_values = tuple(
                    return_slots.get(selected_span, ())
                ) if selected_span is not None else ()
                output_names = tuple(
                    graph.G.graph.get("function_outputs") or ()
                )
                positional_names = set(map(
                    str, graph.G.graph.get("positional_output_names") or ()
                ))
                for index, output_name in enumerate(output_names):
                    if str(output_name) not in positional_names:
                        continue
                    slot_value = (
                        selected_slot_values[index]
                        if index < len(selected_slot_values)
                        else None
                    )
                    if slot_value is None and len(output_names) == 1:
                        # A single-slot return has no container: the
                        # statement's value is the slot's value.
                        slot_value = int(selected_return_id)
                    if slot_value is not None and int(slot_value) in graph.G:
                        identities[str(output_name)] = (int(slot_value),)
                if source_index is not None:
                    # The source return ledger is also an output authority.
                    # A proven top-level terminal arm makes later return
                    # sites unreachable; retaining their slots can otherwise
                    # restore the rejected fallback during SSA publication.
                    # The selected site is identified by its span (the
                    # ledger's own key); comparing slot tuples against the
                    # statement's value id only ever matched single-slot
                    # returns.
                    selected_sites = {
                        site: slots for site, slots in return_slots.items()
                        if site == selected_span
                        or tuple(slots) == (int(selected_return_id),)
                    }
                    if selected_sites:
                        graph.G.graph["return_slot_values"] = selected_sites
                        _post_pruned_return_sites(
                            graph, selected_sites, int(retained_control_id),
                        )
            # Versions whose nodes the arm removal took revise to
            # ``Unresolved(BINDING_VERSION_REMOVED)`` (plan 80, A2.7).
            _post_identity_table_mutation(graph, removed=(
                int(value_id)
                for history in identities.values()
                for value_id in history
                if int(value_id) not in graph.G
            ))
            graph.G.graph["identity_table"] = {
                str(name): tuple(
                    int(value_id) for value_id in history
                    if int(value_id) in graph.G
                )
                for name, history in identities.items()
            }
            pruned = True
            break

    identities = graph.G.graph.get("identity_table") or {}
    protected_values = {
        int(value_id)
        for name in graph.G.graph.get("function_outputs") or ()
        for value_id in identities.get(str(name), ())
        if int(value_id) in graph.G
    } | {
        int(root) for root in graph.roots if int(root) in graph.G
    } | {
        int(value_id)
        for _node_id, node_data in graph.G.nodes(data=True)
        for value_id in (
            (node_data.get("attributes") or {}).get(
                "aggregate_leaf_value_ids", ()
            )
        )
        if int(value_id) in graph.G
    } | {
        # A return-site slot is a metadata edge from authored control.  The
        # value can have no ordinary graph successor when it is returned by
        # only one arm (``if flag: return limit; return None``), but it is
        # still a live callee input.  Dropping it here erased ``limit`` from
        # the callee identity table, so hierarchy planning retained only the
        # first call argument and the linker grew unrelated zeroed storage.
        int(value_id)
        for slots in (
            graph.G.graph.get("return_slot_values") or {}
        ).values()
        for value_id in slots
        if value_id is not None and int(value_id) in graph.G
    }
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
            # A structurally folded constant is a PROVEN literal standing in
            # for an authored producer, and other layers still name that value
            # identity -- a loop's bound expression renders each leaf as the
            # leaf's literal when it is resident and as a uniform reference
            # when it is not.  Pruning it because its only consumer folded too
            # is what made ``n`` (the matrix order) a formal that no caller
            # could fill inside the substitution helpers, while the same fold
            # in _masked_pivot_rows -- whose constant kept a consumer -- read
            # as the literal 2.  An unconsumed Const costs one instruction the
            # backend drops.
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

    unused_parameters = {
        int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        if data.get("type") == "Input"
        and (data.get("attributes") or {}).get("binding_kind") == "parameter"
        and graph.G.out_degree(int(node_id)) == 0
        and int(node_id) not in protected_values
    }
    for node_id in sorted(unused_parameters, reverse=True):
        remove_node(node_id)
    if unused_parameters:
        identities = graph.G.graph.get("identity_table") or {}
        _post_identity_table_mutation(graph, removed=(
            int(value_id)
            for history in identities.values()
            for value_id in history
            if int(value_id) not in graph.G
        ))
        graph.G.graph["identity_table"] = {
            str(name): tuple(
                int(value_id) for value_id in history
                if int(value_id) in graph.G
            )
            for name, history in identities.items()
        }


    # A fold is not complete while a surviving aggregate producer names
    # projections removed with a rejected arm.  Enforce the ledger invariant
    # here so every caller of this fixed-point pass, including recursive
    # callsite specialization, returns a self-contained graph.
    topology_changed |= bool(
        _repair_missing_aggregate_leaf_projections(graph)
    )

    if topology_changed:
        # Folding can run again when hierarchy construction freezes shapes,
        # after retained-loop planning has already computed condensation
        # levels. Publish ordering and recursion metadata for the new graph.
        from .loop_composer import _rebuild_graph_edges
        _rebuild_graph_edges(graph)


def _follow_declared_value_source(
    node_data: Any, replaced_id: int, source_id: int,
) -> None:
    """Keep a port's declared value source equal to its semantic edge.

    Retained-loop ports record ``value_source_id`` as an explicit identity
    check against their ``value`` edge.  Aliasing the edge's producer without
    moving the declaration turns every later descriptor query into an
    identity-conflict error, so the declaration follows the alias exactly.
    """

    # `value_source_id` is one of several cached copies of an id that must
    # follow an alias: the leaf ledgers (`aggregate_leaf_value_ids`,
    # `materialized_source_value_ids`, `materialized_value_ids`) and a loop
    # node's own carried/initial/state-effect records are the same kind of
    # copy, read in preference to the edge by region capture expansion,
    # iterable trip counts and the retained-loop port builder. One helper,
    # shared with loop_composer's edge rewriters, moves all of them.
    attributes = node_data.get("attributes")
    if isinstance(attributes, dict):
        node_data["attributes"] = dict(attributes)
    _retarget_cached_value_ids(node_data, int(replaced_id), (int(source_id),))


def _retarget_all_cached_value_ids(
    graph: Any, replaced_id: int, source_id: int,
) -> None:
    """Move every cached copy of ``replaced_id`` graph-wide, not just successors.

    A loop node caches a carried binding's initial value (a tuple-unpacking
    projection, say) without being that projection's successor, so an alias
    that only visits successors leaves the loop attribute naming a node the
    alias just removed; the evaporator then seeds clones from a dead id.
    """

    if graph.G.graph.get("return_record_field_states"):
        graph.G.graph["return_record_field_states"] = {
            span: tuple(
                (
                    int(source_id) if int(receiver) == int(replaced_id) else receiver,
                    field,
                    int(source_id) if int(value) == int(replaced_id) else value,
                )
                for receiver, field, value in states
            )
            for span, states in graph.G.graph["return_record_field_states"].items()
        }
    for _node_id, data in graph.G.nodes(data=True):
        attributes = data.get("attributes")
        if not isinstance(attributes, dict):
            continue
        if not any(
            key in attributes
            for key in (
                "value_source_id", "aggregate_leaf_value_ids",
                "materialized_source_value_ids", "materialized_value_ids",
                "loop_carried_bindings", "loop_target_initials",
                "loop_state_effects", "loop_iteration_outputs",
            )
        ):
            continue
        data["attributes"] = dict(attributes)
        _retarget_cached_value_ids(data, int(replaced_id), (int(source_id),))
