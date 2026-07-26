"""Semantic Python-AST to ProcessGraph lowering.

This is intentionally a small, strict compiler front-end rather than another
generic AST visualizer.  Supported syntax becomes dataflow; unsupported syntax
becomes an explicit ``opaque_python`` operation so later passes can report it.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from ..transmogrifier.graph.graph_express2 import ProcessGraph
from ..transmogrifier.process_op import ProcessOp, SourceSpan, TensorSpec
from ..transmogrifier.solver_types import DomainNode


_BINARY_OPS = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "truediv",
    ast.FloorDiv: "floordiv",
    ast.Mod: "mod",
    ast.Pow: "pow",
    ast.BitAnd: "bitand",
    ast.BitOr: "bitor",
    ast.BitXor: "bitxor",
    ast.LShift: "shl",
    ast.RShift: "shr",
}

_COMPARE_OPS = {
    ast.Eq: "eq",
    ast.NotEq: "ne",
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
}


class AstProcessGraphBuilder:
    """Build a compiler-only ProcessGraph with semantic operation payloads."""

    def __init__(self, filename: Optional[str] = None):
        self.filename = filename
        self.graph = ProcessGraph(materialize_memory=False)
        self._next_id = 0
        self._env: Dict[str, int] = {}

    def _span(self, node: ast.AST) -> SourceSpan:
        return SourceSpan(
            filename=self.filename,
            line=getattr(node, "lineno", None),
            column=getattr(node, "col_offset", None),
            end_line=getattr(node, "end_lineno", None),
            end_column=getattr(node, "end_col_offset", None),
        )

    def _add(
        self,
        payload: ProcessOp,
        inputs: Iterable[Tuple[int, str]] = (),
        *,
        label: Optional[str] = None,
    ) -> int:
        nid = self._next_id
        self._next_id += 1
        parents = list(inputs)
        domain_node = DomainNode(shape=(1, 1, 1), unit_size=1)
        domain_node.id = id(domain_node)
        self.graph.G.add_node(
            nid,
            label=label or payload.op,
            type=payload.op,
            expr_obj=None,
            process_op=payload,
            extra_args=dict(payload.attributes),
            domain_node=domain_node,
            store_id=None,
            parents=parents,
            children=[],
        )
        for src, role in parents:
            self.graph.G.add_edge(src, nid, role=role)
            self.graph.G.nodes[src]["children"].append((nid, role))
        return nid

    def _opaque(self, node: ast.AST) -> int:
        return self._add(
            ProcessOp(
                "opaque_python",
                attributes={"ast_type": type(node).__name__, "dump": ast.dump(node)},
                source=self._span(node),
            )
        )

    def expression(self, node: ast.AST) -> int:
        if isinstance(node, ast.Name):
            if node.id not in self._env:
                self._env[node.id] = self._add(
                    ProcessOp(
                        "input",
                        output_roles=("value",),
                        attributes={"name": node.id},
                        tensor=TensorSpec(),
                        source=self._span(node),
                    ),
                    label=node.id,
                )
            return self._env[node.id]

        if isinstance(node, ast.Constant):
            return self._add(
                ProcessOp(
                    "const",
                    output_roles=("value",),
                    constant=node.value,
                    source=self._span(node),
                ),
                label=repr(node.value),
            )

        if isinstance(node, ast.BinOp):
            lhs = self.expression(node.left)
            rhs = self.expression(node.right)
            op = _BINARY_OPS.get(type(node.op))
            if op is None:
                return self._opaque(node)
            return self._add(
                ProcessOp(op, ("lhs", "rhs"), source=self._span(node)),
                ((lhs, "lhs"), (rhs, "rhs")),
            )

        if isinstance(node, ast.UnaryOp):
            operand = self.expression(node.operand)
            op = {
                ast.USub: "neg",
                ast.Not: "logical_not",
                ast.Invert: "invert",
            }.get(type(node.op))
            if op is None:
                return self._opaque(node)
            return self._add(
                ProcessOp(op, ("operand",), source=self._span(node)),
                ((operand, "operand"),),
            )

        if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
            lhs = self.expression(node.left)
            rhs = self.expression(node.comparators[0])
            op = _COMPARE_OPS.get(type(node.ops[0]))
            if op is None:
                return self._opaque(node)
            return self._add(
                ProcessOp(op, ("lhs", "rhs"), source=self._span(node)),
                ((lhs, "lhs"), (rhs, "rhs")),
            )

        if isinstance(node, ast.Call):
            args = [self.expression(arg) for arg in node.args]
            func = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else ast.unparse(node.func)
            )
            roles = tuple(f"arg{i}" for i in range(len(args)))
            return self._add(
                ProcessOp(
                    "call",
                    roles,
                    attributes={
                        "function": func,
                        "keywords": {
                            kw.arg or "**": ast.unparse(kw.value) for kw in node.keywords
                        },
                    },
                    source=self._span(node),
                ),
                zip(args, roles),
                label=func,
            )

        return self._opaque(node)

    def statement(self, node: ast.stmt) -> Optional[int]:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            value = self.expression(node.value)
            target = node.targets[0]
            if isinstance(target, ast.Name):
                self._env[target.id] = value
                return value
            return self._opaque(node)

        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is None:
                return None
            value = self.expression(node.value)
            self._env[node.target.id] = value
            return value

        if isinstance(node, ast.Expr):
            return self.expression(node.value)

        if isinstance(node, ast.Return):
            value = self.expression(node.value) if node.value is not None else None
            inputs = () if value is None else ((value, "value"),)
            result = self._add(
                ProcessOp("return", tuple(role for _, role in inputs), (), source=self._span(node)),
                inputs,
            )
            self.graph.roots.append(result)
            return result

        if isinstance(node, ast.If):
            condition = self.expression(node.test)
            before = dict(self._env)

            self._env = dict(before)
            for stmt in node.body:
                self.statement(stmt)
            then_env = dict(self._env)

            self._env = dict(before)
            for stmt in node.orelse:
                self.statement(stmt)
            else_env = dict(self._env)

            merged = dict(before)
            for name in sorted(set(then_env) | set(else_env)):
                then_value = then_env.get(name, before.get(name))
                else_value = else_env.get(name, before.get(name))
                if then_value is None or else_value is None:
                    continue
                if then_value == else_value:
                    merged[name] = then_value
                else:
                    merged[name] = self._add(
                        ProcessOp(
                            "select",
                            ("condition", "if_true", "if_false"),
                            attributes={"variable": name},
                            source=self._span(node),
                        ),
                        (
                            (condition, "condition"),
                            (then_value, "if_true"),
                            (else_value, "if_false"),
                        ),
                    )
            self._env = merged
            return condition

        return self._opaque(node)

    def build(self, tree: ast.AST) -> ProcessGraph:
        body: Iterable[ast.stmt]
        if isinstance(tree, ast.Module):
            body = tree.body
        elif isinstance(tree, ast.FunctionDef):
            for arg in tree.args.args:
                self._env[arg.arg] = self._add(
                    ProcessOp(
                        "input",
                        output_roles=("value",),
                        attributes={"name": arg.arg},
                        tensor=TensorSpec(),
                        source=self._span(arg),
                    ),
                    label=arg.arg,
                )
            body = tree.body
        else:
            return self.build(ast.Module(body=[ast.Expr(value=tree)], type_ignores=[]))

        for stmt in body:
            if isinstance(stmt, ast.FunctionDef):
                nested = AstProcessGraphBuilder(self.filename)
                return nested.build(stmt)
            self.statement(stmt)
        if not self.graph.roots:
            self.graph.roots = [
                nid for nid in self.graph.G.nodes if self.graph.G.out_degree(nid) == 0
            ]
        return self.graph


def ast_to_process_graph(source, *, filename: Optional[str] = None) -> ProcessGraph:
    """Parse ``source`` and return a semantic, compiler-only ProcessGraph."""

    if isinstance(source, ast.AST):
        tree = source
    else:
        candidate = (
            Path(source)
            if isinstance(source, str) and "\n" not in source and "\r" not in source
            else None
        )
        if candidate is not None and candidate.exists():
            filename = filename or str(candidate)
            tree = ast.parse(candidate.read_text(encoding="utf-8"), filename=filename)
        elif isinstance(source, str):
            tree = ast.parse(source, filename=filename or "<string>")
        else:
            raise TypeError("source must be Python source, a path, or an AST")
    return AstProcessGraphBuilder(filename).build(tree)
