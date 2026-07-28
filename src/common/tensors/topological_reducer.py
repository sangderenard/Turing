"""Topological reduction stage for AbstractTensor ProcessGraphs."""

from __future__ import annotations

import ast
import copy
import logging
from typing import Any

import networkx as nx

from ...transmogrifier.function_table import FunctionTable
from ...transmogrifier.ssa_registry import ast_ssa_name_map


logger = logging.getLogger(__name__)


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


def reduce_abstract_tensor_topology(graph: Any) -> Any:
    """Apply existing ProcessGraph names to the three structural AST nodes."""

    function_table = getattr(graph, "function_table", None)
    if function_table is None:
        function_table = FunctionTable()
        graph.function_table = function_table

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
    import_bindings: dict[str, tuple[str, dict[str, Any]]] = {}
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
            import_bindings[local_name] = (qualified_name, requirement)
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
        if isinstance(expression, ast.BinOp):
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
                    qualified_name, requirement = import_bindings[
                        callee_name
                    ]
                    reference = function_table.declare(
                        callee_name,
                        qualified_name=qualified_name,
                        external=True,
                        metadata={
                            "contextual_requirement": requirement,
                        },
                    )
                if reference is not None:
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
        included: set[int] = set()
        for value in return_values:
            included.update(nx.ancestors(graph.G, value))
            included.add(value)
        if not included:
            included = nx.ancestors(graph.G, node_id)
            included.add(node_id)
        function_graph = copy.copy(graph)
        function_graph.G = graph.G.subgraph(included).copy()
        function_graph.levels = {
            member: level
            for member, level in graph.levels.items()
            if member in included
        }
        function_graph.roots = return_values or [node_id]
        function_graph.function_table = function_table
        statement = graph.G.nodes[node_id].get("expr_obj")
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
        function_graph.G.graph.update(
            function_ref=reference.address,
            function_name=function_table.entry(reference).name,
            function_parameters=parameter_names,
            positional_parameters=positional_parameters,
            keyword_only_parameters=keyword_only_parameters,
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
        function_table.resolve_graph(reference, function_graph)
    return graph


__all__ = ["reduce_abstract_tensor_topology"]
