"""Validation control: synthetic device-scalar shells, identity-parameter projection and the construction and placement of validation blocks."""

from __future__ import annotations

import ast
from typing import Any, Iterable, Sequence
from types import SimpleNamespace

from .control_source import (
    ControlExpression,
    ControlProgram,
    ValidationBlock,
    place_validations_after_region_producers,
)
from .source_node_facts import _ast_source_signature, _constant_value
from .source_control_retention import (
    _source_control_records,
    _retained_control_value_id,
)


def _identity_parameter_projection(graph) -> tuple[int, str] | None:
    """Recognize a closure that only forwards or scalar-projects a parameter."""

    identities = graph.graph.get("identity_table") or {}
    outputs = tuple(
        int(identities[name][-1])
        for name in graph.graph.get("function_outputs", ())
        if identities.get(name)
    )
    if len(outputs) != 1:
        return None
    current = outputs[0]
    data = graph.nodes[current]
    if (
        data.get("type") == "Input"
        and (data.get("attributes") or {}).get("binding_kind")
        == "parameter"
    ):
        return current, "direct"
    attributes = data.get("attributes") or {}
    if not (
        isinstance(data.get("expr_obj"), ast.Call)
        and str(attributes.get("static_python_reference", "")).rsplit(
            ".", 1
        )[-1] == "bool"
    ):
        return None
    parents = tuple(
        int(parent)
        for parent, role in (data.get("parents") or ())
        if str(role) not in {"callee", "func", "definition"}
    )
    if len(parents) != 1:
        return None
    item_id = parents[0]
    item = graph.nodes[item_id]
    if str(item.get("op", "")).lower() != "item":
        return None
    item_parents = tuple(
        int(parent)
        for parent, role in (item.get("parents") or ())
        if str(role) in {"operand", "value", "base", "arg:0", "arg0"}
    )
    if len(item_parents) != 1:
        return None
    source = item_parents[0]
    source_data = graph.nodes[source]
    if (
        source_data.get("type") == "Input"
        and (source_data.get("attributes") or {}).get("binding_kind")
        == "parameter"
    ):
        return source, "device_scalar"
    return None


def _synthetic_device_scalar_shell(graph: Any, predicate_id: int) -> Any:
    """Add the node chain ``_identity_parameter_projection`` already
    recognizes (``bool(param[item])``) so an ordinary, non-call predicate
    passes the same device-scalar check a real callsite would.  Its own
    node ids are throwaway: the caller only reads the projection's
    verdict, then falls back to ``predicate_id`` itself.

    UNVERIFIED: no passing test exercises this yet. It targets the
    ``if predicate: raise`` guard condition only; re-running
    ``tools/compile_re_probe.py`` after adding it showed no change in the
    "no registered callee" shortfall list, because that failure is in the
    raise *bodies* (constructing/raising the exception), a separate,
    still-unaddressed gap this function does not touch.
    """

    next_id = (max(graph.nodes, default=0)) + 1
    input_id, item_id, call_id = next_id, next_id + 1, next_id + 2
    graph.add_node(
        input_id, type="Input", op="Input", parents=[], children=[],
        attributes={"binding_kind": "parameter"}, expr_obj=None,
    )
    graph.add_node(
        item_id, type="Item", op="item", parents=[(input_id, "operand")],
        children=[], attributes={}, expr_obj=None,
    )
    graph.add_edge(input_id, item_id, role="operand")
    call_expr = ast.Call(func=ast.Name(id="bool", ctx=ast.Load()), args=[], keywords=[])
    graph.add_node(
        call_id, type="Call", op="Call", parents=[(item_id, "value")],
        children=[], attributes={"static_python_reference": "builtins.bool"},
        expr_obj=call_expr,
    )
    graph.add_edge(item_id, call_id, role="value")
    wrapper = graph.__class__()
    wrapper.add_nodes_from(graph.nodes(data=True))
    wrapper.add_edges_from(graph.edges(data=True))
    wrapper.graph["function_outputs"] = ("result",)
    wrapper.graph["identity_table"] = {"result": (call_id,)}
    # The wrapper is a scratch graph: its three nodes get ``canonical_value``
    # rows under a scratch scope, NOVEL from the predicate node's cell, and
    # its one binding a ``name_binding`` row DERIVED from the call's row, so
    # the verdict this wrapper produces is traceable if it ever fires
    # (plan 80, A2.7).
    from ..common.tensors.topological_reducer import node_identity_cell
    from .concordance_declarations import (
        BindingFact, CANONICAL_VALUE, NAME_BINDING, PLANNER_SCOPE,
        PLANNER_STRUCTURAL_FOLD, SYNTHESIZED_NO_SOURCE,
        SYNTHETIC_DEVICE_SCALAR_PREDICATE,
    )
    from .identity_concordance import (
        Derived, Mode, Novel, Unsourced, current_identity_book,
    )

    book = current_identity_book()
    scope = book.mint_scope("synthetic_device_scalar", PLANNER_SCOPE)
    wrapper.graph["lexical_read_scope"] = scope
    wrapper.graph["canonical_value_ids"] = True
    try:
        predicate_cell = node_identity_cell(graph, int(predicate_id))
    except ValueError:
        predicate_cell = None
    rows = {}
    for synthetic_id in (input_id, item_id, call_id):
        node = wrapper.nodes[synthetic_id]
        fact = (str(node.get("type") or ""), str(node.get("op") or ""), "")
        rows[synthetic_id] = book.post(
            CANONICAL_VALUE, (scope, int(synthetic_id)), fact,
            stage=PLANNER_STRUCTURAL_FOLD,
            provenance=(
                Novel(SYNTHETIC_DEVICE_SCALAR_PREDICATE, (predicate_cell,))
                if predicate_cell is not None
                else Unsourced(SYNTHESIZED_NO_SOURCE)
            ),
            mode=Mode.CONCORD,
        )
    book.post(
        NAME_BINDING, (scope, "result", 0),
        BindingFact(int(call_id), False, (None,) * 4, ""),
        stage=PLANNER_STRUCTURAL_FOLD, provenance=Derived((rows[call_id],)),
        mode=Mode.CONCORD,
    )
    return SimpleNamespace(process_graph=SimpleNamespace(G=wrapper))


def _validation_control_blocks(shell: Any) -> tuple[ValidationBlock, ...]:
    """Translate terminal raise guards into typed control.

    A guard may compute locals used only to construct its exception message
    before the final raise. Invalid calls route through authored-source
    fallback before native invocation, so those diagnostics and any prelude
    effects remain exact; the compiled validation needs only the predicate and
    exception identity.
    """

    graph = shell.process_graph.G
    expression_nodes = {
        id(data.get("expr_obj")): int(node_id)
        for node_id, data in graph.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.AST)
    }
    signature_nodes: dict[tuple[Any, ...], set[int]] = {}
    for node_id, data in graph.nodes(data=True):
        expression = data.get("expr_obj")
        if isinstance(expression, ast.AST):
            signature_nodes.setdefault(
                _ast_source_signature(expression), set()
            ).add(int(node_id))

    def node_for_expression(expression: ast.AST) -> int | None:
        exact = expression_nodes.get(id(expression))
        if exact is not None:
            return exact
        candidates = tuple(sorted(signature_nodes.get(
            _ast_source_signature(expression), (),
        )))
        return candidates[0] if candidates else None

    def scalar_expression(
        value_id: int, visiting: frozenset[int] = frozenset(),
    ) -> ControlExpression | None:
        value_id = int(value_id)
        if value_id in visiting or value_id not in graph:
            return None
        data = graph.nodes[value_id]
        node_type = str(data.get("type") or "")
        if node_type in {"Const", "const", "Constant"}:
            return ControlExpression(
                "const", value_id=value_id,
                literal=_constant_value(data),
            )
        if node_type in {"Input", "input"}:
            if (data.get("attributes") or {}).get("value_kind") != "scalar":
                return None
            return ControlExpression("value", value_id=value_id)
        operation = {
            "less": "lt", "lt": "lt",
            "lessequal": "le", "le": "le",
            "greater": "gt", "gt": "gt",
            "greaterequal": "ge", "ge": "ge",
            "equal": "eq", "eq": "eq",
            "notequal": "ne", "ne": "ne",
            "logical_and": "and", "land": "and",
            "logical_or": "or", "lor": "or",
            "logical_not": "not", "lnot": "not",
            "add": "add", "sub": "sub", "mul": "mul",
            "div": "div", "truediv": "div", "neg": "neg",
            "bitand": "bitand", "bitor": "bitor",
            "bitxor": "bitxor", "shl": "shl", "shr": "shr",
        }.get(str(data.get("op") or node_type).lower())
        if operation is None:
            return None
        operands = tuple(
            scalar_expression(int(parent), visiting | {value_id})
            for parent, role in data.get("parents") or ()
            if str(role) not in {"callee", "ops", "operator"}
        )
        arity = 1 if operation in {"not", "neg"} else 2
        if len(operands) < arity or any(
            operand is None for operand in operands[:arity]
        ):
            return None
        return ControlExpression(
            operation, operands[:arity], value_id=value_id,
        )
    nested_in_loop = {
        _ast_source_signature(member)
        for _node_id, node_data in graph.nodes(data=True)
        for loop in (node_data.get("expr_obj"),)
        if isinstance(loop, (ast.For, ast.While))
        for member in ast.walk(loop)
        if isinstance(member, ast.If)
    }
    blocks = []

    def terminal_raise(statements: Sequence[ast.stmt]) -> ast.Raise | None:
        if not statements or not isinstance(statements[-1], ast.Raise):
            return None
        # Keep this a straight-line guard arm. Compound control, return, and
        # loop control could bypass the terminal raise and therefore are not a
        # validation contract.
        if any(not isinstance(statement, (
            ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Pass,
        )) for statement in statements[:-1]):
            return None
        return statements[-1]

    # Planning a scalar predicate may attach a synthetic callsite shell to the
    # same graph.  Iterate a stable snapshot: validation discovery is allowed
    # to enrich planning metadata, but must not invalidate its own traversal.
    for node_id, record in _source_control_records(graph).items():
        statement = record.get("expression")
        if not isinstance(statement, ast.If) or (
            _ast_source_signature(statement) in nested_in_loop
        ):
            continue
        body_raise = terminal_raise(statement.body)
        orelse_raise = terminal_raise(statement.orelse)
        if body_raise is not None and not statement.orelse:
            raised_statement = body_raise
            raises_when_true = True
        elif orelse_raise is not None and body_raise is None:
            raised_statement = orelse_raise
            raises_when_true = False
        else:
            continue
        test = statement.test
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            test = test.operand
            raises_when_true = not raises_when_true
        predicate_id = _retained_control_value_id(
            graph, record.get("predicate_id"), test,
        )
        if predicate_id is None:
            predicate_id = node_for_expression(test)
        if predicate_id is None:
            continue
        predicate_id = int(predicate_id)
        predicate_expression = scalar_expression(predicate_id)
        child = None
        if predicate_expression is None:
            child = shell.callsite_function_shells.get(predicate_id)
        if predicate_expression is None and child is None:
            child = _synthetic_device_scalar_shell(graph, predicate_id)
            # This wrapper exists only to classify the predicate projection.
            # It is not an authored callsite and has no independent roots,
            # lifecycle, or deployment ABI, so registering it as a child shell
            # makes the compilation-tree walker attempt to compile analysis
            # scaffolding as a function.
        if predicate_expression is None and child is not None:
            projection = _identity_parameter_projection(
                child.process_graph.G
            )
            if projection is None or projection[1] != "device_scalar":
                continue
            # The projection call itself has no device storage after hierarchy
            # identity reduction.  Bind validation to the caller value passed
            # into the projected parameter; retaining the callsite ID here
            # creates a fake SSBO operand for a closure that no longer exists.
            predicate_id = next((
                int(parent)
                for parent, role in (
                    graph.nodes[predicate_id].get("parents") or ()
                )
                if str(role) in {"arg:0", "arg0", "operand", "value"}
            ), predicate_id)
        elif predicate_expression is None:
            continue
        raised = raised_statement.exc
        raised_name = (
            raised.func.id
            if isinstance(raised, ast.Call) and isinstance(raised.func, ast.Name)
            else raised.id if isinstance(raised, ast.Name) else None
        )
        blocks.append(ValidationBlock(
            predicate_id,
            int(node_id),
            expect_true=not raises_when_true,
            predicate_expression=predicate_expression,
            extraction_identity=(
                f"builtins.{raised_name}" if raised_name else None
            ),
        ))
    return tuple(blocks)


def _place_validation_blocks(
    control: ControlProgram,
    validations: Iterable[ValidationBlock],
    retained_regions: Iterable[int],
    dispatch_subgraphs: Sequence[Any],
) -> ControlProgram:
    """Correlate guard predicates with their unique numerical producers."""

    validations = tuple(validations)
    if not validations:
        return control
    wanted = {int(block.predicate_value_id) for block in validations}
    producers: dict[int, int] = {}
    for region_index in map(int, retained_regions):
        subgraph = dispatch_subgraphs[region_index]
        produced_here = {
            int(node_id)
            for node_id in subgraph.G.graph.get("deployment_nodes", ())
        } | {
            int(subgraph.G.nodes[node_id].get("value_id", node_id))
            for node_id in subgraph.G.graph.get("deployment_nodes", ())
            if node_id in subgraph.G
        }
        for predicate_id in sorted(wanted.intersection(produced_here)):
            previous = producers.get(predicate_id)
            if previous is not None and previous != region_index:
                raise ValueError(
                    "a validation predicate has more than one scheduled "
                    f"producer region: value={predicate_id}, "
                    f"regions=({previous}, {region_index})"
                )
            producers[predicate_id] = region_index
    return place_validations_after_region_producers(
        control, validations, predicate_regions=producers,
    )
