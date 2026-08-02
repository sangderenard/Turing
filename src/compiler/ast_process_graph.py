"""Semantic Python-source lowering into the existing ProcessGraph vocabulary.

Structural ``ast.AST`` ingestion remains available in ``ProcessGraph``.  This
module is the source-language specialization: it translates Python spellings
to operators ProcessGraph/SSA already know and leaves unknown syntax explicit
as ``opaque_python`` nodes for the existing diagnostics.
"""

from __future__ import annotations

import ast

from ..common.tensors.fused_ir import ELEMENTWISE_BINARY, ELEMENTWISE_UNARY
from ..transmogrifier.solver_types import DomainNode


_BINARY = {
    ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "truediv",
    ast.FloorDiv: "floordiv", ast.Mod: "mod", ast.Pow: "pow",
    ast.MatMult: "matmul", ast.BitAnd: "bitand", ast.BitOr: "bitor",
    ast.BitXor: "bitxor", ast.LShift: "shl", ast.RShift: "shr",
}
_COMPARE = {
    ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "le",
    ast.Gt: "gt", ast.GtE: "ge",
}
_CALLS = {
    **{name: name for name in ELEMENTWISE_UNARY | ELEMENTWISE_BINARY},
    "absolute": "abs", "divide": "truediv", "subtract": "sub",
    "multiply": "mul", "power": "pow", "concat": "cat",
    "concatenate": "cat", "arange": "arange", "broadcast_to": "broadcast_to",
    "cat": "cat", "clone": "clone", "cumsum": "cumsum",
    "flatten": "flatten", "full": "full", "gather": "gather",
    "log_softmax": "log_softmax", "matmul": "matmul", "max": "max",
    "mean": "mean", "pad": "pad", "reshape": "reshape", "stack": "stack",
    "sum": "sum", "topk": "topk", "transpose": "transpose",
}


def build_semantic_ast(graph, tree: ast.AST, *, filename: str | None = None):
    """Lower one Python function/module into ``graph`` using existing ops."""

    env: dict[str, int] = {}
    next_id = 0
    while next_id in graph.G:
        next_id += 1

    def span(node):
        return {
            "filename": filename,
            "line": getattr(node, "lineno", None),
            "column": getattr(node, "col_offset", None),
            "end_line": getattr(node, "end_lineno", None),
            "end_column": getattr(node, "end_col_offset", None),
        }

    def add_node(
        op, inputs=(), *, label=None, attributes=None, constant=None,
        tensor=None, control=None, source=None, output_roles=("result",),
    ):
        nonlocal next_id
        while next_id in graph.G:
            next_id += 1
        nid = next_id
        next_id += 1
        parents = list(inputs)
        domain = DomainNode(shape=(1, 1, 1), unit_size=1)
        domain.id = id(domain)
        attrs = dict(attributes or {})
        mutation = getattr(graph, "graph_mutation", None)
        if mutation is None:
            from contextlib import nullcontext
            mutation = nullcontext
        with mutation():
            graph.G.add_node(
                nid, label=label or op, type=op, op=op, expr_obj=None,
                extra_args=attrs, attributes=attrs, constant=constant,
                tensor=dict(tensor or {}), bit_quanta=None,
                control=dict(control or {}), source_span=source,
                input_roles=tuple(role for _, role in parents),
                output_roles=tuple(output_roles), schema_version=1,
                domain_node=domain, store_id=None, parents=parents, children=[],
            )
            for parent, role in parents:
                graph.G.add_edge(parent, nid, role=role)
                graph.G.nodes[parent]["children"].append((nid, role))
            observer = getattr(graph, "observe_evolution_node", None)
            if observer is not None:
                observer(nid, graph.G.nodes[nid])
            edge_observer = getattr(graph, "observe_evolution_edge", None)
            if edge_observer is not None:
                for parent, role in parents:
                    edge_observer(parent, nid, role)
        return nid

    def opaque(node):
        return add_node(
            "opaque_python",
            attributes={"ast_type": type(node).__name__, "dump": ast.dump(node)},
            source=span(node),
        )

    def expression(node):
        if isinstance(node, ast.Name):
            if node.id not in env:
                env[node.id] = add_node(
                    "input", label=node.id, attributes={"name": node.id},
                    tensor={}, source=span(node), output_roles=("value",),
                )
            return env[node.id]
        if isinstance(node, ast.Constant):
            return add_node(
                "const", label=repr(node.value), constant=node.value,
                source=span(node), output_roles=("value",),
            )
        if isinstance(node, ast.BinOp):
            op = _BINARY.get(type(node.op))
            if op is None:
                return opaque(node)
            return add_node(
                op, ((expression(node.left), "lhs"), (expression(node.right), "rhs")),
                source=span(node),
            )
        if isinstance(node, ast.UnaryOp):
            op = {ast.USub: "neg", ast.UAdd: "identity", ast.Not: "logical_not",
                  ast.Invert: "invert"}.get(type(node.op))
            return (add_node(op, ((expression(node.operand), "operand"),), source=span(node))
                    if op is not None else opaque(node))
        if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
            op = _COMPARE.get(type(node.ops[0]))
            return (add_node(
                op,
                ((expression(node.left), "lhs"),
                 (expression(node.comparators[0]), "rhs")),
                source=span(node),
            ) if op is not None else opaque(node))
        if isinstance(node, ast.Call):
            spelling = (node.func.id if isinstance(node.func, ast.Name)
                        else node.func.attr if isinstance(node.func, ast.Attribute)
                        else None)
            canonical = _CALLS.get(spelling)
            arguments = []
            if canonical is not None and isinstance(node.func, ast.Attribute):
                arguments.append(expression(node.func.value))
            arguments.extend(expression(arg) for arg in node.args)
            if canonical is not None:
                roles = (("operand",) if canonical in ELEMENTWISE_UNARY
                         else ("lhs", "rhs") if canonical in ELEMENTWISE_BINARY
                         else tuple(f"arg{i}" for i in range(len(arguments))))
                if len(roles) != len(arguments):
                    return opaque(node)
                attrs = {}
                for keyword in node.keywords:
                    key = keyword.arg or "**"
                    try:
                        attrs[key] = ast.literal_eval(keyword.value)
                    except (ValueError, TypeError):
                        attrs[key] = ast.unparse(keyword.value)
                return add_node(
                    canonical, zip(arguments, roles), label=canonical,
                    attributes=attrs, source=span(node),
                )
            function = (node.func.id if isinstance(node.func, ast.Name)
                        else ast.unparse(node.func))
            return add_node(
                "call", zip(arguments, (f"arg{i}" for i in range(len(arguments)))),
                label=function,
                attributes={"function": function, "keywords": {
                    keyword.arg or "**": ast.unparse(keyword.value)
                    for keyword in node.keywords
                }}, source=span(node),
            )
        return opaque(node)

    def statement(node):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = (node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1
                      else node.target if isinstance(node, ast.AnnAssign) else None)
            value_node = getattr(node, "value", None)
            if value_node is None:
                return None
            value = expression(value_node)
            if isinstance(target, ast.Name):
                env[target.id] = value
                return value
            return opaque(node)
        if isinstance(node, ast.Expr):
            return expression(node.value)
        if isinstance(node, ast.Return):
            value = expression(node.value) if node.value is not None else None
            result = add_node(
                "return", () if value is None else ((value, "value"),),
                source=span(node), output_roles=(),
            )
            graph.roots.append(result)
            return result
        if isinstance(node, ast.If):
            condition = expression(node.test)
            before = dict(env)
            for child in node.body:
                statement(child)
            then_env = dict(env)
            env.clear(); env.update(before)
            for child in node.orelse:
                statement(child)
            else_env = dict(env)
            merged = dict(before)
            for name in sorted(set(then_env) | set(else_env)):
                yes = then_env.get(name, before.get(name))
                no = else_env.get(name, before.get(name))
                if yes is None or no is None:
                    continue
                merged[name] = yes if yes == no else add_node(
                    "select", ((condition, "condition"), (yes, "if_true"),
                               (no, "if_false")),
                    attributes={"variable": name}, source=span(node),
                )
            env.clear(); env.update(merged)
            return condition
        return opaque(node)

    if isinstance(tree, ast.Module):
        body = tree.body
    elif isinstance(tree, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for argument in tree.args.args:
            env[argument.arg] = add_node(
                "input", label=argument.arg, attributes={"name": argument.arg},
                tensor={}, source=span(argument), output_roles=("value",),
            )
        body = tree.body
    else:
        body = [ast.Expr(value=tree)]

    for child in body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if graph.G.number_of_nodes():
                raise ValueError("semantic AST import accepts one top-level function")
            return build_semantic_ast(graph, child, filename=filename)
        statement(child)
    if not graph.roots:
        graph.roots = [nid for nid in graph.G if graph.G.out_degree(nid) == 0]
    graph.domain_shape = (1,)
    return graph
