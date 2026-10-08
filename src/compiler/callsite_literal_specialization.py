"""Callsite literal specialization: static source values and literals, and the propagation of planner specializations from callsite literals into callee graphs."""

from __future__ import annotations

import ast
import copy
from typing import Any

from ..transmogrifier.function_table import FunctionReference
from .source_node_facts import _constant_value
from .call_argument_binding import (
    _method_parameter_layout,
    _callsite_parameter_name,
    _expanded_callsite_argument_edges,
)
from .formal_shape_ledger import (
    _FORMAL_LITERAL_CONFLICT,
    _proven_formal_literal,
)
from .planner_concordance_posts import (
    _argument_identity_cells,
    _formal_identity_cells,
    _post_planner_specialization,
    _materialize_planner_specializations,
)


def _source_static_value(graph: Any, node_id: int, visiting=None) -> bool:
    """Whether a call argument is structural source data, not runtime data."""

    visiting = set() if visiting is None else set(visiting)
    node_id = int(node_id)
    if node_id in visiting or node_id not in graph.G:
        return False
    visiting.add(node_id)
    data = graph.G.nodes[node_id]
    if data.get("type") in {"Constant", "Const", "const", "StaticReference"}:
        # A structurally specialized aggregate is not source data.
        #
        # Deciding WHICH arm of ``x if flag else y`` survives is a decision
        # about structure; it says nothing about the contents.  The fold
        # rewrites the node's type to Constant and records that distinction
        # as ``structural_specialization``, and the aggregate ledger it
        # carries still names runtime values.
        #
        # Read as static, such a node takes the literal-specialization path
        # and never the aggregate-member path, so its members are never
        # given per-index leaves and never earn a ``parameter_member_
        # formals`` receipt -- they reach emission as formals no caller can
        # name.  ``saved = state.copy_shallow() if rollback else None`` in
        # dt_controller is the case: ``rollback=True`` arrives as a literal,
        # the ternary folds, and 52 live tensor copies start reading as a
        # compile-time constant.
        attributes = data.get("attributes") or {}
        if attributes.get("structural_specialization") and tuple(
            attributes.get("aggregate_leaf_value_ids") or ()
        ):
            return False
        return True
    if data.get("type") == "Input":
        name = (data.get("attributes") or {}).get("binding_name")
        if name in (graph.G.graph.get("planner_specializations") or {}):
            return True
        # Not told HERE is not the same as unknown.  Copies of one function
        # share its formals; if every callsite that proved this one agrees,
        # the value is a literal in all of them.
        return _proven_formal_literal(
            graph, name
        ) is not _FORMAL_LITERAL_CONFLICT
    expression = data.get("expr_obj")
    if isinstance(expression, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
        return all(
            _source_static_value(graph, parent, visiting)
            for parent, _role in (data.get("parents") or ())
        )
    return False


def _source_static_literal(
    graph: Any,
    node_id: int,
    visiting: frozenset[int] = frozenset(),
) -> Any:
    """Evaluate only literal aggregate syntax admitted by source planning."""

    node_id = int(node_id)
    if node_id in visiting or node_id not in graph.G:
        raise ValueError("value is not source-static")
    data = graph.G.nodes[node_id]
    if data.get("type") in {"Constant", "Const", "const"}:
        return _constant_value(data)
    if data.get("type") == "StaticReference":
        attributes = data.get("attributes") or {}
        reference = attributes.get(
            "first_class_function_ref",
            attributes.get("function_ref"),
        )
        if reference is not None:
            return FunctionReference(int(reference))
        raise ValueError("static reference is not a graph-backed function")
    if data.get("type") == "Input":
        name = (data.get("attributes") or {}).get("binding_name")
        specializations = graph.G.graph.get("planner_specializations") or {}
        if name in specializations:
            return specializations[name]
        proven = _proven_formal_literal(graph, name)
        if proven is not _FORMAL_LITERAL_CONFLICT:
            return proven
        raise ValueError("input has no source-static planner binding")
    expression = data.get("expr_obj")
    parents = tuple(data.get("parents") or ())
    nested = visiting | {node_id}
    if isinstance(expression, (ast.Tuple, ast.List, ast.Set)):
        values = tuple(
            _source_static_literal(graph, parent, nested)
            for parent, _role in parents
        )
        if isinstance(expression, ast.Tuple):
            return values
        if isinstance(expression, ast.Set):
            return frozenset(values)
        return list(values)
    if isinstance(expression, ast.Dict):
        keys = tuple(expression.keys)
        values = tuple(expression.values)
        if len(keys) != len(values):
            raise ValueError("invalid static dictionary")
        if any(key is None for key in keys):
            # ``{**other}`` occupies a key slot with no key expression, so the
            # mapping's contents are not decidable from this literal alone.
            raise ValueError("dictionary unpacking is not source-static")
        # Parents arrive grouped by the AST field they came from -- every
        # ``keys`` parent, then every ``values`` parent -- not interleaved per
        # entry, so pair them by role rather than by position.
        key_parents = tuple(
            parent for parent, role in parents if str(role) == "keys"
        )
        value_parents = tuple(
            parent for parent, role in parents if str(role) == "values"
        )
        if len(key_parents) != len(keys) or len(value_parents) != len(values):
            raise ValueError("dictionary parents are not literal leaves")
        return {
            _source_static_literal(graph, key_parent, nested):
                _source_static_literal(graph, value_parent, nested)
            for key_parent, value_parent in zip(key_parents, value_parents)
        }
    raise ValueError("value is not source-static")


def _propagate_callsite_planner_specializations(graph: Any) -> None:
    """Resolve consistent literal call arguments before any loop is planned."""

    function_table = getattr(graph, "function_table", None)
    if function_table is None:
        return
    graphs = [graph]
    graphs.extend(
        entry.graph for entry in function_table if entry.graph is not None
    )
    from .concordance_declarations import (
        SPECIALIZATION_CALLSITES_DISAGREE, SPECIALIZATION_DYNAMIC_ARGUMENT,
        SPECIALIZATION_NOT_SOURCE_STATIC, SPECIALIZATION_TENSOR_ARGUMENT,
        SpecializationFact,
        SpecializationSource,
    )
    from .identity_concordance import Unresolved

    # Per candidate: (value, the cells that prove it, omitted-argument?).
    candidates: dict[tuple[int, str], list[tuple[Any, tuple, bool]]] = {}
    dynamic_argument = object()
    not_source_static = object()
    tensor_argument = object()
    for caller in graphs:
        for _node_id, data in caller.G.nodes(data=True):
            attributes = data.get("attributes") or {}
            reference = attributes.get("callee_ref")
            if reference is None:
                continue
            try:
                callee = function_table.entry(int(reference)).graph
            except (KeyError, TypeError, ValueError):
                continue
            if callee is None:
                continue
            receiver, positional, _all = _method_parameter_layout(callee.G)
            bound_parameters: set[str] = set()
            for parent, role_value in _expanded_callsite_argument_edges(
                caller, int(_node_id),
            ):
                role = str(role_value)
                parameter = _callsite_parameter_name(
                    role, receiver, positional,
                )
                if parameter is None:
                    continue
                parameter = str(parameter)
                bound_parameters.add(parameter)
                role_cells: tuple = ()
                if _source_static_value(caller, int(parent)):
                    try:
                        value = _source_static_literal(caller, int(parent))
                    except ValueError:
                        value = not_source_static
                    else:
                        # A literal the caller DECLARES a tensor argument is
                        # two facts about one value.  Folding it here
                        # rewrites the shared definition with the literal
                        # alone: in the graph-reverse motion of ``x**3*y``
                        # the backward Call passes bw_pow's ``p`` as 3 and
                        # declares it an ``operand`` (a forward tensor); the
                        # catalogue fold made ``p`` a bare ``Constant 3``
                        # before any callsite copy existed, so ``p.shape``
                        # never resolved and ``unbroadcast(..., p.shape)``
                        # kept a shape loop whose iterable ``extent`` LLVM
                        # cannot emit.  Rule: a formal whose argument the
                        # call declares GRADIENT/OPERAND (its
                        # ``callsite_argument_role`` row, posted by the
                        # adjoint builder) is specialized only in its
                        # callsite copy (``_callsite_specialized_shell_type``),
                        # which keeps literal and descriptor together.
                        # METADATA arguments and undeclared calls fold as
                        # before.  The row's cell is cited by the
                        # ``planner_specialization`` row either way.
                        from .concordance_declarations import ArgumentRole
                        from .process_graph_autograd import (
                            declared_argument_role,
                        )

                        position = (
                            int(role[4:])
                            if role.startswith("arg:") and role[4:].isdigit()
                            else None
                        )
                        declared = (
                            None if position is None
                            else declared_argument_role(
                                caller, int(_node_id), position,
                            )
                        )
                        if declared is not None:
                            role_cells = (declared[1],)
                            if declared[0] in {
                                ArgumentRole.GRADIENT, ArgumentRole.OPERAND,
                            }:
                                value = tensor_argument
                else:
                    value = dynamic_argument
                candidates.setdefault(
                    (int(reference), parameter), []
                ).append((
                    value,
                    (*_argument_identity_cells(caller, int(parent)), *role_cells),
                    False,
                ))
            # Omitted arguments are just as exact as authored literal
            # arguments: Python binds them to the signature default before
            # the body starts. Include them in the same consistency proof so
            # optional-selection idioms can be resolved without turning
            # ``None`` into a runtime scalar. A dynamic argument at any other
            # callsite vetoes shared specialization below.
            for parameter, default in (
                callee.G.graph.get("parameter_defaults") or {}
            ).items():
                parameter = str(parameter)
                if parameter not in bound_parameters:
                    candidates.setdefault(
                        (int(reference), parameter), []
                    ).append((
                        copy.deepcopy(default),
                        _formal_identity_cells(callee, parameter), True,
                    ))
    # Every candidate is a row on ``planner_specialization`` (callee scope,
    # parameter): the agreed literal / default as a ``SpecializationFact``,
    # or the reason no literal exists -- a dynamic argument, an argument that
    # is source data but not a literal, or callsites that disagree -- so a
    # later ``_source_static_value`` miss is visible on the book instead of
    # a silent ``continue`` (plan 80, A2.1).  Rows are DERIVED from the
    # argument cells of every contributing callsite; the dict the readers
    # consult is the page's read view.
    touched: list[Any] = []
    for (reference, parameter), contributions in candidates.items():
        try:
            callee = function_table.entry(reference).graph
        except (KeyError, TypeError, ValueError):
            continue
        if callee is None or not contributions:
            continue
        values = [value for value, _cells, _omitted in contributions]
        cells = tuple(dict.fromkeys(
            cell for _value, cell_group, _omitted in contributions
            for cell in cell_group
        ))
        if any(value is not_source_static for value in values):
            _post_planner_specialization(callee, parameter, Unresolved(
                SPECIALIZATION_NOT_SOURCE_STATIC, read=cells,
            ), cells)
            continue
        if any(value is dynamic_argument for value in values):
            _post_planner_specialization(callee, parameter, Unresolved(
                SPECIALIZATION_DYNAMIC_ARGUMENT, read=cells,
            ), cells)
            continue
        if any(value is tensor_argument for value in values):
            _post_planner_specialization(callee, parameter, Unresolved(
                SPECIALIZATION_TENSOR_ARGUMENT, read=cells,
            ), cells)
            continue
        first = values[0]
        try:
            consistent = all(value == first for value in values[1:])
        except (TypeError, ValueError):
            consistent = False
        if not consistent:
            _post_planner_specialization(callee, parameter, Unresolved(
                SPECIALIZATION_CALLSITES_DISAGREE, read=cells,
            ), cells)
            continue
        source = (
            SpecializationSource.DEFAULT
            if all(omitted for _value, _cells, omitted in contributions)
            else SpecializationSource.LITERAL
        )
        _post_planner_specialization(
            callee, parameter, SpecializationFact(first, source), cells,
        )
        callee.G.graph.setdefault(
            "planner_specializations", {}
        )[parameter] = first
        touched.append(callee)
    for callee in {id(item): item for item in touched}.values():
        _materialize_planner_specializations(callee)
