"""Topological reduction stage for AbstractTensor ProcessGraphs."""

from __future__ import annotations

import ast
import builtins
import copy
from dataclasses import dataclass
import importlib
import logging
import types
from typing import Any

import networkx as nx

from ...transmogrifier.function_table import (
    ExternalFunctionTable,
    FunctionTable,
)
from ...transmogrifier.ssa_registry import ast_ssa_name_map


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _StaticPythonReference:
    """Ephemeral resolved Python object; never emitted as a graph value."""

    value: Any
    path: str


_AST_PROCESS_GRAPH_ALIASES = {
    "Name": ast_ssa_name_map["name"].value,
    "Assign": ast_ssa_name_map["assign"].value,
    "Call": ast_ssa_name_map["call"].value,
}

_BITOPS_TO_EXECUTABLE = {
    "Xor": "bitxor",
    "Neg": "neg",
    "Eq": "equal",
    "Ne": "not_equal",
    "Lt": "less",
    "Le": "less_equal",
    "Gt": "greater",
    "Ge": "greater_equal",
    "LAnd": "logical_and",
    "LOr": "logical_or",
    "LNot": "logical_not",
}


def _qualified_handler(prefix: str, operator: ast.AST) -> str:
    spelling = f"{prefix}:{type(operator).__name__.lower()}"
    handler = ast_ssa_name_map.get(spelling)
    if handler is None:
        raise KeyError(f"no existing operator alias for {spelling!r}")
    return _BITOPS_TO_EXECUTABLE.get(handler.value, handler.value)


def _replace_inputs(
    graph: Any,
    node_id: int,
    inputs: tuple[tuple[int, str], ...],
) -> None:
    """Replace one wrapper's incoming topology with executable operands."""

    for predecessor in tuple(graph.G.predecessors(node_id)):
        graph.G.remove_edge(predecessor, node_id)
        graph.G.nodes[predecessor]["children"] = [
            (child_id, role)
            for child_id, role in graph.G.nodes[predecessor].get(
                "children", ()
            )
            if child_id != node_id
        ]
    graph.G.nodes[node_id]["parents"] = list(inputs)
    for predecessor, role in inputs:
        graph.G.add_edge(predecessor, node_id, role=role)
        children = graph.G.nodes[predecessor].setdefault("children", [])
        if node_id not in {child_id for child_id, _role in children}:
            children.append((node_id, role))


def _remove_node(graph: Any, node_id: int) -> None:
    """Remove one reduced-away node and its cached adjacency metadata."""

    if node_id not in graph.G:
        return
    for predecessor in tuple(graph.G.predecessors(node_id)):
        graph.G.nodes[predecessor]["children"] = [
            (child_id, role)
            for child_id, role in graph.G.nodes[predecessor].get(
                "children", ()
            )
            if child_id != node_id
        ]
    for successor in tuple(graph.G.successors(node_id)):
        graph.G.nodes[successor]["parents"] = [
            (parent_id, role)
            for parent_id, role in graph.G.nodes[successor].get(
                "parents", ()
            )
            if parent_id != node_id
        ]
    graph.roots = [root for root in graph.roots if root != node_id]
    graph.G.remove_node(node_id)


def _redirect_value(
    graph: Any,
    old_id: int,
    producer_id: int,
) -> None:
    """Fan every use of one lexical occurrence out from its value producer."""

    if old_id == producer_id or old_id not in graph.G:
        return
    for successor in tuple(graph.G.successors(old_id)):
        successor_data = graph.G.nodes[successor]
        replacement = []
        for parent_id, role in successor_data.get("parents", ()):
            replacement.append(
                (producer_id if parent_id == old_id else parent_id, role)
            )
        successor_data["parents"] = replacement
        graph.G.add_edge(producer_id, successor, role=graph.G.edges[
            old_id,
            successor,
        ].get("role"))
        children = graph.G.nodes[producer_id].setdefault("children", [])
        for _parent_id, role in replacement:
            if _parent_id == producer_id and (
                successor,
                role,
            ) not in {
                (child_id, child_role)
                for child_id, child_role in children
            }:
                children.append((successor, role))
    graph.roots = [
        producer_id if root == old_id else root for root in graph.roots
    ]
    _remove_node(graph, old_id)


def _normalize_lexical_values(
    function_graph: Any,
    statement: ast.FunctionDef | ast.AsyncFunctionDef,
    static_bindings: dict[str, Any],
) -> None:
    """Resolve unique lexical occurrences into a monotonic value DAG.

    AST ingestion intentionally gives every ``Name`` occurrence its own node.
    At topological reduction, loads are consolidated against the definition
    visible at that program point.  Consumers then fan out directly from the
    defining value; source-language ``Load`` and ``Store`` wrappers disappear.
    """

    graph = function_graph
    environment: dict[str, int] = {}
    identity_bindings: dict[str, list[int]] = {}
    parameter_names = {
        argument.arg
        for argument in (
            *statement.args.posonlyargs,
            *statement.args.args,
            *statement.args.kwonlyargs,
        )
    }
    for body_statement in statement.body:
        if not isinstance(body_statement, ast.Return):
            continue
        returned = body_statement.value
        expressions = (
            tuple(returned.elts)
            if isinstance(returned, (ast.Tuple, ast.List))
            else (returned,)
        )
        graph.G.graph["function_outputs"] = tuple(
            expression.id
            if isinstance(expression, ast.Name)
            else f"result_{index}"
            for index, expression in enumerate(expressions)
        )
        break
    temporary_id = max(graph.G.nodes, default=-1) + 1

    def new_node(
        node_type: str,
        label: str,
        *,
        attributes: dict[str, Any] | None = None,
        parents: tuple[tuple[int, str], ...] = (),
    ) -> int:
        nonlocal temporary_id
        node_id = temporary_id
        temporary_id += 1
        graph.G.add_node(
            node_id,
            label=label,
            type=node_type,
            op=node_type.lower(),
            expr_obj=None,
            extra_args={},
            domain_node=None,
            store_id=None,
            parents=list(parents),
            children=[],
            attributes=dict(attributes or {}),
        )
        if node_type in {"Const", "Constant"}:
            graph.G.nodes[node_id]["constant"] = (
                attributes or {}
            ).get("value")
        for parent_id, role in parents:
            if parent_id not in graph.G:
                continue
            graph.G.add_edge(parent_id, node_id, role=role)
            graph.G.nodes[parent_id].setdefault("children", []).append(
                (node_id, role)
            )
        return node_id

    def input_value(name: str, *, binding_kind: str) -> int:
        value = environment.get(name)
        if value is not None:
            return value
        value = new_node(
            "Input",
            name,
            attributes={
                "binding_name": name,
                "binding_kind": binding_kind,
            },
        )
        environment[name] = value
        identity_bindings.setdefault(name, []).append(value)
        return value

    def bind_loop_target(target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            value = new_node(
                "Input",
                target.id,
                attributes={
                    "binding_name": target.id,
                    "binding_kind": "loop",
                },
            )
            environment[target.id] = value
            identity_bindings.setdefault(target.id, []).append(value)
            _remove_node(graph, id(target))
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                bind_loop_target(element)

    def loop_target_names(target: ast.AST) -> tuple[str, ...]:
        if isinstance(target, ast.Name):
            return (target.id,)
        if isinstance(target, (ast.Tuple, ast.List)):
            return tuple(
                name
                for element in target.elts
                for name in loop_target_names(element)
            )
        return ()

    def resolve_expression(expression: ast.AST | None) -> int | None:
        if expression is None:
            return None
        if isinstance(
            expression,
            (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
        ):
            for generator in expression.generators:
                resolve_expression(generator)
            if isinstance(expression, ast.DictComp):
                resolve_expression(expression.key)
                resolve_expression(expression.value)
            else:
                resolve_expression(expression.elt)
            node_id = id(expression)
            return node_id if node_id in graph.G else None
        if isinstance(expression, ast.comprehension):
            resolve_expression(expression.iter)
            bind_loop_target(expression.target)
            node_id = id(expression)
            if node_id in graph.G:
                graph.G.nodes[node_id].setdefault("attributes", {})[
                    "loop_target_bindings"
                ] = {
                    name: environment[name]
                    for name in loop_target_names(expression.target)
                }
            for condition in expression.ifs:
                resolve_expression(condition)
            return node_id if node_id in graph.G else None
        if isinstance(expression, ast.Name):
            node_id = id(expression)
            if isinstance(expression.ctx, ast.Load):
                producer_id = environment.get(expression.id)
                static_value = static_bindings.get(expression.id)
                if (
                    producer_id is None
                    and static_value is not None
                    and isinstance(
                        static_value,
                        (
                            types.ModuleType,
                            type,
                            types.FunctionType,
                            types.BuiltinFunctionType,
                            types.MethodType,
                        ),
                    )
                ):
                    _remove_node(graph, node_id)
                    return _StaticPythonReference(
                        static_value,
                        expression.id,
                    )
                if producer_id is None:
                    producer_id = input_value(
                        expression.id,
                        binding_kind=(
                            "parameter"
                            if expression.id in parameter_names
                            else "external"
                        ),
                    )
                _redirect_value(graph, node_id, producer_id)
                return producer_id
            return environment.get(expression.id)
        if isinstance(expression, ast.Attribute):
            if (
                isinstance(expression.value, ast.Name)
                and expression.value.id not in environment
                and expression.value.id in static_bindings
            ):
                _remove_node(graph, id(expression.value))
                receiver = _StaticPythonReference(
                    static_bindings[expression.value.id],
                    expression.value.id,
                )
            else:
                receiver = resolve_expression(expression.value)
            if isinstance(receiver, _StaticPythonReference):
                try:
                    value = getattr(receiver.value, expression.attr)
                except AttributeError:
                    pass
                else:
                    node_id = id(expression)
                    _remove_node(graph, node_id)
                    return _StaticPythonReference(
                        value,
                        f"{receiver.path}.{expression.attr}",
                    )

        if isinstance(expression, ast.Call):
            callee = resolve_expression(expression.func)
            node_id = id(expression)
            if isinstance(callee, _StaticPythonReference) and node_id in graph.G:
                attributes = graph.G.nodes[node_id].setdefault(
                    "attributes",
                    {},
                )
                attributes["static_python_reference"] = callee.path
                static_arguments = {}
                for index, argument in enumerate(expression.args):
                    resolved = resolve_expression(argument)
                    if isinstance(resolved, _StaticPythonReference):
                        static_arguments[f"arg:{index}"] = resolved.path
                for keyword in expression.keywords:
                    if keyword.arg is None:
                        continue
                    resolved = resolve_expression(keyword.value)
                    if isinstance(resolved, _StaticPythonReference):
                        static_arguments[f"kw:{keyword.arg}"] = resolved.path
                if static_arguments:
                    attributes["static_call_arguments"] = static_arguments

        # A named callee already represented by a function-table reference is
        # not a runtime value.  Its arguments still are.
        children = tuple(ast.iter_child_nodes(expression))
        if isinstance(expression, ast.Call):
            call_data = (
                graph.G.nodes[id(expression)]
                if id(expression) in graph.G
                else {}
            )
            call_attributes = call_data.get("attributes") or {}
            if (
                isinstance(expression.func, ast.Name)
                and (
                    "callee_ref" in call_attributes
                    or "external_callee_ref" in call_attributes
                )
            ):
                children = tuple(
                    child
                    for child in children
                    if child is not expression.func
                )
            elif isinstance(callee, _StaticPythonReference):
                children = tuple(
                    child
                    for child in children
                    if child is not expression.func
                )

        # Resolve children first so the existing executable node receives
        # producer IDs rather than lexical occurrence IDs.
        for child in children:
            if isinstance(child, ast.expr_context):
                continue
            resolve_expression(child)
        node_id = id(expression)
        return node_id if node_id in graph.G else None

    def bind_target(target: ast.AST, value_id: int | None) -> None:
        if value_id is None:
            return
        if isinstance(target, ast.Name):
            environment[target.id] = value_id
            identity_bindings.setdefault(target.id, []).append(value_id)
            _remove_node(graph, id(target))
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for index, element in enumerate(target.elts):
                index_id = new_node(
                    "Constant",
                    str(index),
                    attributes={"value": index},
                )
                projected_id = new_node(
                    "Indexed",
                    f"unpack[{index}]",
                    parents=(
                        (value_id, "base"),
                        (index_id, "index"),
                    ),
                )
                bind_target(element, projected_id)

    def reduce_statement(body_statement: ast.stmt) -> int | None:
        if isinstance(body_statement, (ast.Assign, ast.AnnAssign)):
            value = resolve_expression(body_statement.value)
            targets = (
                tuple(body_statement.targets)
                if isinstance(body_statement, ast.Assign)
                else (body_statement.target,)
            )
            for target in targets:
                bind_target(target, value)
            _remove_node(graph, id(body_statement))
            return value
        if isinstance(body_statement, ast.AugAssign):
            # The executable BinOp form is handled by the ordinary operator
            # reducer; preserve this structural node until that lowering exists.
            resolve_expression(body_statement.value)
            return id(body_statement)
        if isinstance(body_statement, ast.Return):
            return resolve_expression(body_statement.value)
        if isinstance(body_statement, (ast.With, ast.AsyncWith)):
            static_contexts = []
            for item in body_statement.items:
                context_value = resolve_expression(item.context_expr)
                if context_value is not None and context_value in graph.G:
                    context_data = graph.G.nodes[context_value]
                    reference = (
                        context_data.get("attributes") or {}
                    ).get("static_python_reference")
                    if reference == "autograd.no_grad":
                        static_contexts.append(
                            {
                                "reference": reference,
                                "effect": "disable_backward_recording",
                                "lineno": getattr(
                                    item.context_expr,
                                    "lineno",
                                    None,
                                ),
                                "end_lineno": getattr(
                                    item.context_expr,
                                    "end_lineno",
                                    None,
                                ),
                            }
                        )
                        _remove_node(graph, context_value)
                resolve_expression(item.optional_vars)
            result = None
            for nested in body_statement.body:
                result = reduce_statement(nested)
            if static_contexts:
                recorded = list(
                    graph.G.graph.get("static_contexts", ())
                )
                recorded.extend(static_contexts)
                graph.G.graph["static_contexts"] = tuple(recorded)
                for item in body_statement.items:
                    _remove_node(graph, id(item))
                _remove_node(graph, id(body_statement))
            return result
        if isinstance(body_statement, ast.If):
            resolve_expression(body_statement.test)
            # Control-flow value merging remains a planner responsibility.
            # Reduce lexical occurrences within each arm without pretending
            # that either arm executed unconditionally.
            before = dict(environment)
            body_environment = dict(before)
            environment.clear()
            environment.update(body_environment)
            for nested in body_statement.body:
                reduce_statement(nested)
            body_environment = dict(environment)
            environment.clear()
            environment.update(before)
            for nested in body_statement.orelse:
                reduce_statement(nested)
            else_environment = dict(environment)
            environment.clear()
            environment.update(before)
            for name in set(body_environment) & set(else_environment):
                if body_environment[name] == else_environment[name]:
                    environment[name] = body_environment[name]
            return id(body_statement)
        if isinstance(body_statement, ast.Try):
            before = dict(environment)
            for nested in body_statement.body:
                reduce_statement(nested)
            body_environment = dict(environment)

            handler_environments = []
            for handler in body_statement.handlers:
                environment.clear()
                environment.update(before)
                if handler.name:
                    input_value(
                        handler.name,
                        binding_kind="exception",
                    )
                for nested in handler.body:
                    reduce_statement(nested)
                handler_environments.append(dict(environment))

            environments = [body_environment, *handler_environments]
            environment.clear()
            if environments:
                common_names = set.intersection(
                    *(set(candidate) for candidate in environments)
                )
                for name in common_names:
                    values = {
                        candidate[name] for candidate in environments
                    }
                    if len(values) == 1:
                        environment[name] = values.pop()
            for nested in body_statement.orelse:
                reduce_statement(nested)
            for nested in body_statement.finalbody:
                reduce_statement(nested)
            return id(body_statement)
        if isinstance(body_statement, (ast.For, ast.While)):
            before_loop = dict(environment)
            if isinstance(body_statement, ast.For):
                resolve_expression(body_statement.iter)
                bind_loop_target(body_statement.target)
            else:
                resolve_expression(body_statement.test)
            for nested in body_statement.body:
                reduce_statement(nested)
            for nested in body_statement.orelse:
                reduce_statement(nested)
            loop_id = id(body_statement)
            if loop_id in graph.G:
                loop_attributes = graph.G.nodes[loop_id].setdefault(
                    "attributes",
                    {},
                )
                loop_attributes["loop_carried_bindings"] = {
                    name: (before_loop[name], environment[name])
                    for name in before_loop.keys() & environment.keys()
                    if before_loop[name] != environment[name]
                }
                loop_attributes["loop_target_bindings"] = {
                    name: value_id
                    for name, value_id in environment.items()
                    if (
                        name not in before_loop
                        and (
                            graph.G.nodes[value_id].get("attributes") or {}
                        ).get("binding_kind") == "loop"
                    )
                }
            return id(body_statement)
        if isinstance(body_statement, ast.Expr):
            return resolve_expression(body_statement.value)
        for child in ast.iter_child_nodes(body_statement):
            if isinstance(child, ast.expr):
                resolve_expression(child)
        return id(body_statement) if id(body_statement) in graph.G else None

    returned_values = []
    for body_statement in statement.body:
        value = reduce_statement(body_statement)
        if isinstance(body_statement, ast.Return) and value is not None:
            returned_values.append(value)
    if returned_values:
        graph.roots = list(dict.fromkeys(returned_values))

    # Any surviving lexical occurrence is either unused syntax or an unresolved
    # source label.  It is not executable work in the reduced value graph.
    for node_id, data in list(graph.G.nodes(data=True)):
        expression = data.get("expr_obj")
        if isinstance(expression, ast.Name):
            _remove_node(graph, node_id)
            continue
        if (
            data.get("type") == "Input"
            and graph.G.out_degree(node_id) == 0
            and node_id not in graph.roots
            and (data.get("attributes") or {}).get("binding_kind")
            == "external"
        ):
            # A named Call target initially looks like a lexical external.
            # Once the Call owns a function-table reference, that temporary
            # value has no consumers and must not leak into the public input
            # contract.
            _remove_node(graph, node_id)

    # Stable topological relabeling turns opaque Python object identities into
    # compact monotonic value IDs without changing the faithfully captured AST.
    source_position = {
        node_id: (
            getattr(data.get("expr_obj"), "lineno", -1),
            getattr(data.get("expr_obj"), "col_offset", -1),
            str(data.get("type", "")),
            int(node_id),
        )
        for node_id, data in graph.G.nodes(data=True)
    }
    ordered = list(
        nx.lexicographical_topological_sort(
            graph.G,
            key=lambda node_id: source_position[node_id],
        )
    )
    mapping = {
        node_id: value_id for value_id, node_id in enumerate(ordered)
    }
    relabeled = nx.relabel_nodes(graph.G, mapping, copy=True)
    ordered_graph = nx.DiGraph()
    ordered_graph.graph.update(relabeled.graph)
    for value_id in range(len(mapping)):
        ordered_graph.add_node(value_id, **relabeled.nodes[value_id])
    ordered_graph.add_edges_from(relabeled.edges(data=True))
    graph.G = ordered_graph
    graph.roots = [
        mapping[root] for root in graph.roots if root in mapping
    ]
    graph.levels = {
        mapping[node_id]: level
        for node_id, level in graph.levels.items()
        if node_id in mapping
    }
    graph.G.graph["identity_table"] = {
        name: tuple(
            mapping[value_id]
            for value_id in value_ids
            if value_id in mapping
        )
        for name, value_ids in identity_bindings.items()
    }
    for value_id, data in graph.G.nodes(data=True):
        data["value_id"] = value_id
        attributes = data.get("attributes") or {}
        if "loop_carried_bindings" in attributes:
            attributes["loop_carried_bindings"] = {
                name: (mapping[initial], mapping[updated])
                for name, (initial, updated) in attributes[
                    "loop_carried_bindings"
                ].items()
                if initial in mapping and updated in mapping
            }
        if "loop_target_bindings" in attributes:
            attributes["loop_target_bindings"] = {
                name: mapping[target]
                for name, target in attributes[
                    "loop_target_bindings"
                ].items()
                if target in mapping
            }
        data["parents"] = [
            (mapping[parent_id], role)
            for parent_id, role in data.get("parents", ())
            if parent_id in mapping
        ]
        data["children"] = [
            (mapping[child_id], role)
            for child_id, role in data.get("children", ())
            if child_id in mapping
        ]


def reduce_abstract_tensor_topology(graph: Any) -> Any:
    """Apply existing ProcessGraph names to the three structural AST nodes."""

    function_table = getattr(graph, "function_table", None)
    if function_table is None:
        function_table = FunctionTable()
        graph.function_table = function_table
    external_function_table = getattr(
        graph,
        "external_function_table",
        None,
    )
    if external_function_table is None:
        external_function_table = ExternalFunctionTable()
        graph.external_function_table = external_function_table

    function_nodes: dict[int, Any] = {}
    function_return_values: dict[int, list[int]] = {}
    call_owners: dict[int, Any] = {}
    class _OwnedCallVisitor(ast.NodeVisitor):
        def __init__(self, owner, function_node_id):
            self.owner = owner
            self.function_node_id = function_node_id

        def visit_Call(self, node):
            call_owners[id(node)] = self.owner
            self.generic_visit(node)

        def visit_Return(self, node):
            if node.value is not None:
                function_return_values.setdefault(
                    self.function_node_id,
                    [],
                ).append(id(node.value))
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            # Nested definitions receive their own table entry and ownership
            # walk; do not assign their calls to the enclosing function.
            return None

        visit_AsyncFunctionDef = visit_FunctionDef

    for node_id, data in graph.G.nodes(data=True):
        statement = data.get("expr_obj")
        if not isinstance(
            statement,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue
        reference = function_table.declare(
            statement.name,
            qualified_name=statement.name,
            metadata={
                "source_type": type(statement).__name__,
                "source_node": node_id,
            },
        )
        data.setdefault("attributes", {})[
            "function_ref"
        ] = reference.address
        function_nodes[node_id] = reference
        visitor = _OwnedCallVisitor(reference, node_id)
        for body_statement in statement.body:
            visitor.visit(body_statement)

    contextual_requirements = list(
        graph.G.graph.get("contextual_requirements", ())
    )
    import_bindings: dict[
        str,
        tuple[str, str, dict[str, Any]],
    ] = {}
    for node_id, data in graph.G.nodes(data=True):
        statement = data.get("expr_obj")
        if not isinstance(statement, (ast.Import, ast.ImportFrom)):
            continue
        requirement = {
            "kind": (
                "import_from"
                if isinstance(statement, ast.ImportFrom)
                else "import"
            ),
            "module": (
                statement.module
                if isinstance(statement, ast.ImportFrom)
                else None
            ),
            "level": (
                int(statement.level)
                if isinstance(statement, ast.ImportFrom)
                else 0
            ),
            "names": tuple(
                (imported.name, imported.asname)
                for imported in statement.names
            ),
        }
        if requirement not in contextual_requirements:
            contextual_requirements.append(requirement)
        data.setdefault("attributes", {})[
            "contextual_requirement"
        ] = requirement
        for imported in statement.names:
            local_name = imported.asname or imported.name.split(".")[0]
            if isinstance(statement, ast.ImportFrom):
                module = "." * int(statement.level) + (
                    statement.module or ""
                )
                qualified_name = (
                    f"{module}.{imported.name}"
                    if module
                    else imported.name
                )
            else:
                qualified_name = imported.name
            import_bindings[local_name] = (
                qualified_name,
                imported.name,
                requirement,
            )
            imported_id = id(imported)
            if imported_id in graph.G:
                graph.G.nodes[imported_id].setdefault("attributes", {})[
                    "contextual_requirement"
                ] = requirement
        logger.info(
            "retaining ProcessGraph import as a contextual requirement: %s",
            requirement,
        )
    graph.G.graph["contextual_requirements"] = tuple(
        contextual_requirements
    )
    static_bindings = dict(getattr(graph, "python_bindings", {}) or {})
    python_package = getattr(graph, "python_package", None)
    for local_name, (
        qualified_name,
        imported_name,
        requirement,
    ) in import_bindings.items():
        try:
            if requirement["kind"] == "import_from":
                module_name = (
                    "." * int(requirement.get("level", 0))
                    + str(requirement.get("module") or "")
                )
                module = importlib.import_module(
                    module_name,
                    package=python_package,
                )
                static_bindings[local_name] = getattr(
                    module,
                    imported_name,
                )
            elif requirement["kind"] == "import":
                static_bindings[local_name] = importlib.import_module(
                    qualified_name,
                    package=python_package,
                )
        except (ImportError, AttributeError, TypeError, ValueError):
            # The contextual requirement remains available to deployment even
            # when it cannot be resolved in the compiler's current process.
            continue
    for builtin_name, value in vars(builtins).items():
        static_bindings.setdefault(builtin_name, value)
    graph.python_bindings = static_bindings

    for _node_id, data in graph.G.nodes(data=True):
        source_type = data.get("type")
        process_type = _AST_PROCESS_GRAPH_ALIASES.get(source_type)
        if process_type is None:
            continue
        data["type"] = process_type
        data["op"] = process_type
        data.setdefault("attributes", {})["source_type"] = source_type

    for node_id, data in list(graph.G.nodes(data=True)):
        expression = data.get("expr_obj")
        if isinstance(expression, ast.Constant):
            data["type"] = "Constant"
            data["op"] = "const"
            data["constant"] = expression.value
            data.setdefault("attributes", {})["value"] = expression.value
        elif isinstance(expression, ast.BinOp):
            operation = _qualified_handler("binop", expression.op)
            data["type"] = operation
            data["op"] = operation
            data.setdefault("attributes", {})["source_type"] = "BinOp"
            _replace_inputs(
                graph,
                node_id,
                (
                    (id(expression.left), "lhs"),
                    (id(expression.right), "rhs"),
                ),
            )
        elif isinstance(expression, ast.UnaryOp):
            operation = _qualified_handler("unaryop", expression.op)
            data["type"] = operation
            data["op"] = operation
            data.setdefault("attributes", {})["source_type"] = "UnaryOp"
            _replace_inputs(
                graph,
                node_id,
                ((id(expression.operand), "operand"),),
            )
        elif isinstance(expression, ast.Compare) and len(expression.ops) == 1:
            operation = _qualified_handler("compare", expression.ops[0])
            data["type"] = operation
            data["op"] = operation
            data.setdefault("attributes", {})["source_type"] = "Compare"
            _replace_inputs(
                graph,
                node_id,
                (
                    (id(expression.left), "lhs"),
                    (id(expression.comparators[0]), "rhs"),
                ),
            )
        elif isinstance(expression, ast.Subscript):
            # Indexing is one operation over the parent tensor and the complete
            # index tuple.  In particular, Ellipsis is not an independent
            # scalar operation: AbstractTensor.__getitem__ and each specialized
            # backend resolve it against the parent tensor's rank.
            index_expressions = (
                tuple(expression.slice.elts)
                if isinstance(expression.slice, ast.Tuple)
                else (expression.slice,)
            )
            data["type"] = "Indexed"
            data["op"] = "Indexed"
            data.setdefault("attributes", {})["source_type"] = "Subscript"
            _replace_inputs(
                graph,
                node_id,
                (
                    (id(expression.value), "base"),
                    *(
                        (id(index_expression), "index")
                        for index_expression in index_expressions
                    ),
                ),
            )

            # The AST Tuple only grouped the index components.  Once those
            # components feed Indexed directly, retaining the wrapper would
            # incorrectly schedule a second tuple-producing computation.
            slice_id = id(expression.slice)
            if (
                isinstance(expression.slice, ast.Tuple)
                and slice_id in graph.G
                and graph.G.out_degree(slice_id) == 0
            ):
                for predecessor in tuple(graph.G.predecessors(slice_id)):
                    graph.G.nodes[predecessor]["children"] = [
                        (child_id, role)
                        for child_id, role in graph.G.nodes[
                            predecessor
                        ].get("children", ())
                        if child_id != slice_id
                    ]
                graph.G.remove_node(slice_id)
        elif (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
        ):
            operation = expression.func.attr
            data["type"] = operation
            data["op"] = operation
            data.setdefault("attributes", {})["source_type"] = "Call"
            _replace_inputs(
                graph,
                node_id,
                (
                    (id(expression.func.value), "operand"),
                    *(
                        (id(argument), f"arg{index}")
                        for index, argument in enumerate(expression.args)
                    ),
                    *(
                        (
                            id(keyword.value),
                            (
                                f"kw:{keyword.arg}"
                                if keyword.arg is not None
                                else "kwargs"
                            ),
                        )
                        for keyword in expression.keywords
                    ),
                ),
            )
        elif isinstance(expression, ast.Call):
            call_inputs: list[tuple[int, str]] = [
                (
                    id(argument),
                    f"arg:{index}",
                )
                for index, argument in enumerate(expression.args)
            ]
            call_inputs.extend(
                (
                    id(keyword.value),
                    (
                        f"kw:{keyword.arg}"
                        if keyword.arg is not None
                        else "kwargs"
                    ),
                )
                for keyword in expression.keywords
            )
            if isinstance(expression.func, ast.Name):
                callee_name = expression.func.id
                reference = function_table.reference(callee_name)
                if reference is None and callee_name in import_bindings:
                    (
                        qualified_name,
                        imported_name,
                        requirement,
                    ) = import_bindings[
                        callee_name
                    ]
                    external_reference = external_function_table.declare(
                        callee_name,
                        qualified_name=qualified_name,
                        external=True,
                        metadata={
                            "contextual_requirement": requirement,
                            "imported_name": imported_name,
                        },
                    )
                    data.setdefault("attributes", {})[
                        "external_callee_ref"
                    ] = external_reference.address
                elif reference is not None:
                    data.setdefault("attributes", {})[
                        "callee_ref"
                    ] = reference.address
                else:
                    call_inputs.insert(
                        0,
                        (id(expression.func), "callee"),
                    )
            else:
                call_inputs.insert(0, (id(expression.func), "callee"))
            _replace_inputs(graph, node_id, tuple(call_inputs))

            # Keyword nodes carry only source spelling once their value and
            # name have been transferred to the Call edge role.
            for keyword in expression.keywords:
                keyword_id = id(keyword)
                if (
                    keyword_id not in graph.G
                    or graph.G.out_degree(keyword_id) != 0
                ):
                    continue
                for predecessor in tuple(graph.G.predecessors(keyword_id)):
                    graph.G.nodes[predecessor]["children"] = [
                        (child_id, role)
                        for child_id, role in graph.G.nodes[
                            predecessor
                        ].get("children", ())
                        if child_id != keyword_id
                    ]
                graph.G.remove_node(keyword_id)

    for node_id, data in list(graph.G.nodes(data=True)):
        statement = data.get("expr_obj")
        if not isinstance(statement, ast.For):
            continue
        attributes = data.setdefault("attributes", {})
        attributes["source_type"] = "For"
        attributes["target"] = (
            statement.target.id
            if isinstance(statement.target, ast.Name)
            else ast.dump(statement.target, include_attributes=False)
        )
        loop_inputs: list[tuple[int, str]] = []
        iterator = statement.iter
        if (
            isinstance(iterator, ast.Call)
            and isinstance(iterator.func, ast.Name)
            and iterator.func.id == "range"
        ):
            attributes["iterator_kind"] = "arithmetic_sequence"
            range_arguments = tuple(iterator.args)
            if not 1 <= len(range_arguments) <= 3:
                raise ValueError("range loop requires one to three arguments")
            roles = (
                ("stop",)
                if len(range_arguments) == 1
                else ("start", "stop")
                if len(range_arguments) == 2
                else ("start", "stop", "step")
            )
            if len(range_arguments) == 1:
                attributes["start"] = 0
                attributes["step"] = 1
            elif len(range_arguments) == 2:
                attributes["step"] = 1
            loop_inputs.extend(
                (id(argument), role)
                for argument, role in zip(range_arguments, roles)
            )
        else:
            attributes["iterator_kind"] = "iterable"
            loop_inputs.append((id(iterator), "iterable"))
        loop_inputs.extend(
            (id(body_statement), "body")
            for body_statement in statement.body
        )
        loop_inputs.extend(
            (id(else_statement), "else")
            for else_statement in statement.orelse
        )
        _replace_inputs(graph, node_id, tuple(loop_inputs))

        if isinstance(iterator, ast.Call) and id(iterator) in graph.G:
            iterator_id = id(iterator)
            if graph.G.out_degree(iterator_id) == 0:
                for predecessor in tuple(graph.G.predecessors(iterator_id)):
                    graph.G.nodes[predecessor]["children"] = [
                        (child_id, role)
                        for child_id, role in graph.G.nodes[
                            predecessor
                        ].get("children", ())
                        if child_id != iterator_id
                    ]
                graph.G.remove_node(iterator_id)

    for node_id, data in list(graph.G.nodes(data=True)):
        if data.get("type") != "Expr":
            continue
        predecessors = tuple(graph.G.predecessors(node_id))
        successors = tuple(graph.G.successors(node_id))
        if len(predecessors) > 1:
            raise ValueError("Expr wrapper must contain at most one value")
        predecessor = predecessors[0] if predecessors else None
        for successor in successors:
            successor_data = graph.G.nodes[successor]
            replacement_parents = []
            for parent_id, role in successor_data.get("parents", ()):
                if parent_id != node_id:
                    replacement_parents.append((parent_id, role))
                elif predecessor is not None:
                    replacement_parents.append((predecessor, role))
            successor_data["parents"] = replacement_parents
            if predecessor is not None:
                graph.G.add_edge(predecessor, successor)
                predecessor_children = graph.G.nodes[predecessor].setdefault(
                    "children", []
                )
                if successor not in {
                    child_id for child_id, _role in predecessor_children
                }:
                    predecessor_children.append((successor, "output"))
        if predecessor is not None:
            graph.G.nodes[predecessor]["children"] = [
                (child_id, role)
                for child_id, role in graph.G.nodes[predecessor].get(
                    "children", ()
                )
                if child_id != node_id
            ]
        graph.roots = [
            predecessor if root_id == node_id and predecessor is not None
            else root_id
            for root_id in graph.roots
            if root_id != node_id or predecessor is not None
        ]
        graph.G.remove_node(node_id)

    for node_id, data in list(graph.G.nodes(data=True)):
        if data.get("type") != "Return":
            continue
        predecessors = tuple(graph.G.predecessors(node_id))
        if len(predecessors) > 1:
            raise ValueError("Return wrapper must contain at most one value")
        returned = predecessors[0] if predecessors else None
        for successor in tuple(graph.G.successors(node_id)):
            graph.G.nodes[successor]["parents"] = [
                (parent_id, role)
                for parent_id, role in graph.G.nodes[successor].get(
                    "parents", ()
                )
                if parent_id != node_id
            ]
        if returned is not None:
            graph.G.nodes[returned]["children"] = [
                (child_id, role)
                for child_id, role in graph.G.nodes[returned].get(
                    "children", ()
                )
                if child_id != node_id
            ]
            if returned not in graph.roots:
                graph.roots.append(returned)
        graph.roots = [root for root in graph.roots if root != node_id]
        graph.G.remove_node(node_id)

    call_graph = nx.DiGraph()
    call_graph.add_nodes_from(
        reference.address for reference in function_nodes.values()
    )
    referenced_calls: list[tuple[int, Any, Any]] = []
    for call_id, owner in call_owners.items():
        if call_id not in graph.G:
            continue
        callee_address = (
            graph.G.nodes[call_id].get("attributes") or {}
        ).get("callee_ref")
        if callee_address is None:
            continue
        referenced_calls.append((call_id, owner, callee_address))
        try:
            callee_entry = function_table.entry(callee_address)
        except KeyError:
            continue
        if callee_entry.graph is None and callee_address not in {
            reference.address for reference in function_nodes.values()
        }:
            continue
        call_graph.add_edge(owner.address, callee_address)

    recursive_edges: set[tuple[int, int]] = set()
    for component in nx.strongly_connected_components(call_graph):
        if len(component) > 1:
            recursive_edges.update(
                (source, target)
                for source, target in call_graph.edges(component)
                if target in component
            )
        else:
            address = next(iter(component))
            if call_graph.has_edge(address, address):
                recursive_edges.add((address, address))
    for call_id, owner, callee_address in referenced_calls:
        if (owner.address, callee_address) not in recursive_edges:
            continue
        graph.G.nodes[call_id].setdefault("attributes", {})[
            "recursive_backedge"
        ] = True
        function_table.entry(owner).recursive = True

    for node_id, reference in function_nodes.items():
        if node_id not in graph.G:
            continue
        return_values = [
            value
            for value in function_return_values.get(node_id, ())
            if value in graph.G
        ]
        statement = graph.G.nodes[node_id].get("expr_obj")
        # Function ownership is already exact in the saved Python AST.  Use
        # that ownership directly: graph ancestry from only the return value
        # silently discarded assignments, calls, loops, and side effects.
        owned_members: set[int] = set()

        class _OwnedMemberVisitor(ast.NodeVisitor):
            def generic_visit(self, member):
                if id(member) in graph.G and not isinstance(
                    member,
                    (
                        ast.arguments,
                        ast.arg,
                        ast.expr_context,
                        ast.operator,
                        ast.unaryop,
                        ast.boolop,
                        ast.cmpop,
                        ast.keyword,
                        ast.alias,
                        ast.Import,
                        ast.ImportFrom,
                    ),
                ):
                    owned_members.add(id(member))
                super().generic_visit(member)

            def visit_FunctionDef(self, member):
                # Its definition and body belong to another function-table
                # entry, not to the enclosing shell.
                return None

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_ClassDef(self, member):
                return None

        ownership = _OwnedMemberVisitor()
        for body_member in statement.body:
            ownership.visit(body_member)
        included = owned_members
        included = {
            member
            for member in included
            if (
                member in return_values
                or any(
                    neighbor in included
                    for neighbor in graph.G.predecessors(member)
                )
                or any(
                    neighbor in included
                    for neighbor in graph.G.successors(member)
                )
            )
        }
        function_graph = copy.copy(graph)
        function_graph.G = graph.G.subgraph(included).copy()
        function_graph.levels = {
            member: level
            for member, level in graph.levels.items()
            if member in included
        }
        function_graph.roots = return_values or [node_id]
        function_graph.function_table = function_table
        positional_parameters = ()
        keyword_only_parameters = ()
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            positional_parameters = tuple(
                argument.arg
                for argument in (
                    *statement.args.posonlyargs,
                    *statement.args.args,
                )
            )
            keyword_only_parameters = tuple(
                argument.arg for argument in statement.args.kwonlyargs
            )
        parameter_names = (
            *positional_parameters,
            *keyword_only_parameters,
        )
        parameter_defaults = {}
        scalar_parameter_names = set()
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for argument in (
                *statement.args.posonlyargs,
                *statement.args.args,
                *statement.args.kwonlyargs,
            ):
                if (
                    isinstance(argument.annotation, ast.Name)
                    and argument.annotation.id
                    in {"bool", "bytes", "complex", "float", "int", "str"}
                ):
                    scalar_parameter_names.add(argument.arg)
            positional_defaults = zip(
                positional_parameters[
                    len(positional_parameters) - len(statement.args.defaults):
                ],
                statement.args.defaults,
            )
            keyword_defaults = zip(
                keyword_only_parameters,
                statement.args.kw_defaults,
            )
            for name, default_expression in (
                *positional_defaults,
                *keyword_defaults,
            ):
                if default_expression is None:
                    continue
                try:
                    value = ast.literal_eval(default_expression)
                except (ValueError, TypeError):
                    if isinstance(default_expression, ast.Name):
                        value = static_bindings.get(
                            default_expression.id,
                            default_expression,
                        )
                    else:
                        continue
                parameter_defaults[name] = value
                if isinstance(
                    value,
                    (bool, bytes, complex, float, int, str),
                ):
                    scalar_parameter_names.add(name)
        function_graph.G.graph.update(
            function_ref=reference.address,
            function_name=function_table.entry(reference).name,
            function_parameters=parameter_names,
            positional_parameters=positional_parameters,
            keyword_only_parameters=keyword_only_parameters,
            parameter_defaults=parameter_defaults,
            scalar_parameters=tuple(sorted(scalar_parameter_names)),
        )
        for member in function_graph.G:
            member_data = function_graph.G.nodes[member]
            expression = member_data.get("expr_obj")
            if (
                isinstance(expression, ast.Name)
                and isinstance(expression.ctx, ast.Load)
                and expression.id in parameter_names
            ):
                member_data["type"] = "Input"
                member_data["op"] = "input"
                member_data["label"] = expression.id
                member_data["parents"] = []
            if (
                member_data.get("type") == "Input"
                and (
                    member_data.get("attributes") or {}
                ).get("binding_name") in scalar_parameter_names
            ):
                member_data.setdefault("attributes", {})[
                    "value_kind"
                ] = "scalar"
            member_data["parents"] = [
                (parent, role)
                for parent, role in member_data.get("parents", ())
                if parent in included
            ]
            member_data["children"] = [
                (child, role)
                for child, role in member_data.get("children", ())
                if child in included
            ]
        _normalize_lexical_values(
            function_graph,
            statement,
            static_bindings,
        )
        for _member, member_data in function_graph.G.nodes(data=True):
            if (
                member_data.get("type") == "Input"
                and (
                    member_data.get("attributes") or {}
                ).get("binding_name") in scalar_parameter_names
            ):
                member_data.setdefault("attributes", {})[
                    "value_kind"
                ] = "scalar"
        function_table.resolve_graph(reference, function_graph)
    return graph


__all__ = ["reduce_abstract_tensor_topology"]
