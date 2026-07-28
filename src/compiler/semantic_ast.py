"""Semantic Python AST ingestion for :class:`ProcessGraph`.

This front end preserves two complementary levels of meaning:

* arithmetic and recognized AbstractTensor calls use the canonical tensor
  operation vocabulary;
* Python structure that has not yet been lowered (functions, loops, indexing,
  contexts, exceptions, comprehensions, and containers) remains explicit
  ProcessGraph IR instead of becoming an ``opaque_python`` placeholder.

The builder accepts several source files at once.  Function definitions are
registered before bodies are visited, so calls can carry a real ``callee``
edge even when the definition appears later or in another supplied module.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable
from collections import defaultdict

import networkx as nx

from ..common.tensors.fused_ir import ELEMENTWISE_BINARY, ELEMENTWISE_UNARY
from ..common.tensors.fused_ir import canonical_elementwise_op
from ..transmogrifier.ssa_registry import BITOPS_EXPANDABLE_OPS, SSARegistry
from ..transmogrifier.solver_types import DomainNode


_BINARY_OPS = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "truediv",
    ast.FloorDiv: "floordiv",
    ast.Mod: "mod",
    ast.Pow: "pow",
    ast.MatMult: "matmul",
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
    ast.Is: "is",
    ast.IsNot: "is_not",
    ast.In: "contains",
    ast.NotIn: "not_contains",
}

_TENSOR_CALL_ALIASES = {
    **{name: name for name in ELEMENTWISE_UNARY | ELEMENTWISE_BINARY},
    "absolute": "abs",
    "divide": "truediv",
    "subtract": "sub",
    "multiply": "mul",
    "power": "pow",
    "concatenate": "cat",
    "concat": "cat",
    "arange": "arange",
    "broadcast_to": "broadcast_to",
    "cat": "cat",
    "clone": "clone",
    "cumsum": "cumsum",
    "flatten": "flatten",
    "full": "full",
    "gather": "gather",
    "log_softmax": "log_softmax",
    "matmul": "matmul",
    "max": "max",
    "mean": "mean",
    "min": "min",
    "ones": "ones",
    "pad": "pad",
    "permute": "permute",
    "repeat": "repeat",
    "reshape": "reshape",
    "scatter": "scatter",
    "stack": "stack",
    "sum": "sum",
    "to_dtype": "to_dtype",
    "topk": "topk",
    "transpose": "transpose",
    "unsqueeze": "unsqueeze",
    "zeros": "zeros",
}

_CONTROL_OPS = {
    "async_for",
    "async_with",
    "await",
    "break",
    "continue",
    "for",
    "if",
    "iteration_item",
    "loop_result",
    "return",
    "select",
    "try",
    "while",
    "with",
    "yield",
    "yield_from",
}

_TENSOR_VALUE_OPS = {
    *_TENSOR_CALL_ALIASES.values(),
    *_BINARY_OPS.values(),
    *_COMPARE_OPS.values(),
    "index",
    "index_set",
    "logical_and",
    "logical_not",
    "logical_or",
    "neg",
    "positive",
    "select",
    "tuple_get",
}

_TENSOR_CONTAINER_OPS = {
    "dict",
    "list",
    "list_comp",
    "set",
    "set_comp",
    "tuple",
}

_TENSOR_METADATA_ATTRIBUTES = {
    "device",
    "dtype",
    "ndim",
    "ndims",
    "shape",
}

_HOST_BOUNDARY_METHODS = {
    "item",
    "numpy",
    "to_bytes",
    "tolist",
}


def _snake(name: str) -> str:
    out: list[str] = []
    for index, char in enumerate(name):
        if char.isupper() and index:
            out.append("_")
        out.append(char.lower())
    return "".join(out)


def _read_sources(
    sources: Any,
    *,
    filename: str | None,
) -> list[tuple[ast.AST, str]]:
    if not isinstance(sources, (list, tuple)):
        sources = [sources]
    parsed: list[tuple[ast.AST, str]] = []
    for index, source in enumerate(sources):
        if isinstance(source, ast.AST):
            parsed.append((source, filename or f"<ast:{index}>"))
            continue
        if isinstance(source, Path):
            source = str(source)
        if not isinstance(source, str):
            raise TypeError("AST sources must be nodes, paths, or source strings")
        path = Path(source)
        if path.exists():
            source_name = str(path)
            parsed.append(
                (
                    ast.parse(
                        path.read_text(encoding="utf-8"),
                        filename=source_name,
                    ),
                    source_name,
                )
            )
        else:
            source_name = filename or f"<string:{index}>"
            parsed.append((ast.parse(source, filename=source_name), source_name))
    return parsed


class SemanticAstBuilder:
    """Build serializable semantic/control IR in an existing ProcessGraph."""

    def __init__(self, graph) -> None:
        self.graph = graph
        self.next_id = 0
        while self.next_id in graph.G:
            self.next_id += 1
        self.filename = "<unknown>"
        self.scope = "<module>"
        self.lexical_parent_scope = "<module>"
        self.env: dict[str, int] = {}
        self.globals_by_file: dict[str, dict[str, int]] = defaultdict(dict)
        self.definitions: dict[str, int] = {}
        self.definition_candidates: dict[str, list[int]] = defaultdict(list)
        self.definitions_by_scope: dict[tuple[str, str, str], int] = {}
        self.class_definitions: dict[str, int] = {}
        self.class_methods: dict[tuple[str, str], int] = {}
        self.value_types: dict[int, str] = {}
        self.definition_asts: list[tuple[ast.AST, str, str]] = []
        self.return_nodes: dict[str, list[int]] = {}
        self.structural = True

    def span(self, node: ast.AST) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "line": getattr(node, "lineno", None),
            "column": getattr(node, "col_offset", None),
            "end_line": getattr(node, "end_lineno", None),
            "end_column": getattr(node, "end_col_offset", None),
        }

    def add(
        self,
        op: str,
        inputs: Iterable[tuple[int, str]] = (),
        *,
        label: str | None = None,
        attributes: dict[str, Any] | None = None,
        constant: Any = None,
        source: ast.AST | None = None,
        control: dict[str, Any] | None = None,
        output_roles: tuple[str, ...] = ("result",),
    ) -> int:
        while self.next_id in self.graph.G:
            self.next_id += 1
        node_id = self.next_id
        self.next_id += 1
        parents = list(inputs)
        domain = DomainNode(shape=(1, 1, 1), unit_size=1)
        domain.id = id(domain)
        attrs = dict(attributes or {})
        attrs.setdefault("scope", self.scope)
        try:
            canonical, _ = canonical_elementwise_op(op)
        except KeyError:
            canonical = None
        if canonical is not None:
            attrs.setdefault("canonical_operation", canonical)
        # This is routing metadata, never an alternate implementation. The
        # actual integer expansion remains exclusively in
        # expand_bitops_process_graph/BitOpsTranslator.
        if op in BITOPS_EXPANDABLE_OPS:
            # Capability metadata only. Selection is deferred until dtype
            # inference proves this value belongs to an integer/bit domain.
            attrs.setdefault("bitops_capable", True)
        handler = SSARegistry.name_map.get(op.lower())
        if handler is not None:
            attrs.setdefault("ssa_handler", str(handler))
        self.graph.G.add_node(
            node_id,
            label=label or op,
            type=op,
            op=op,
            expr_obj=None,
            extra_args=attrs,
            attributes=attrs,
            constant=constant,
            tensor={},
            bit_quanta=None,
            control=dict(control or {}),
            source_span=self.span(source) if source is not None else {
                "filename": self.filename,
                "line": None,
                "column": None,
                "end_line": None,
                "end_column": None,
            },
            input_roles=tuple(role for _, role in parents),
            output_roles=output_roles,
            schema_version=1,
            domain_node=domain,
            store_id=None,
            parents=parents,
            children=[],
        )
        for parent, role in parents:
            if parent not in self.graph.G:
                raise ValueError(f"AST node {node_id} reads missing node {parent}")
            self.graph.G.add_edge(parent, node_id, role=role)
            self.graph.G.nodes[parent]["children"].append((node_id, role))
        return node_id

    def constant(self, value: Any, node: ast.AST | None = None) -> int:
        return self.add(
            "const",
            label=repr(value),
            constant=value,
            source=node,
            output_roles=("value",),
        )

    def bind_target(self, target: ast.AST, value: int) -> list[int]:
        if isinstance(target, ast.Name):
            self.env[target.id] = value
            return [value]
        if isinstance(target, (ast.Tuple, ast.List)):
            values = []
            for index, element in enumerate(target.elts):
                selected = self.add(
                    "tuple_get",
                    ((value, "value"), (self.constant(index, target), "index")),
                    attributes={"index": index},
                    source=target,
                )
                values.extend(self.bind_target(element, selected))
            return values
        if isinstance(target, ast.Starred):
            return self.bind_target(target.value, value)
        if isinstance(target, ast.Subscript):
            base = self.expression(target.value)
            index = self.expression(target.slice)
            updated = self.add(
                "index_set",
                ((base, "base"), (index, "index"), (value, "value")),
                source=target,
            )
            if isinstance(target.value, ast.Name):
                self.env[target.value.id] = updated
            return [updated]
        if isinstance(target, ast.Attribute):
            base = self.expression(target.value)
            return [
                self.add(
                    "attribute_set",
                    ((base, "base"), (value, "value")),
                    attributes={"attribute": target.attr},
                    source=target,
                )
            ]
        return [self.generic(target, extra=((value, "value"),))]

    def expression(self, node: ast.AST | None) -> int:
        if node is None:
            return self.constant(None)
        if isinstance(node, ast.Name):
            if node.id in self.env:
                return self.env[node.id]
            module_globals = self.globals_by_file[self.filename]
            if node.id in module_globals:
                return module_globals[node.id]
            previous_scope = self.scope
            self.scope = "<module>"
            value = self.add(
                "global_ref",
                label=node.id,
                attributes={"name": node.id},
                source=node,
                output_roles=("value",),
            )
            self.scope = previous_scope
            module_globals[node.id] = value
            return value
        if isinstance(node, ast.Constant):
            return self.constant(node.value, node)
        if isinstance(node, ast.BinOp):
            return self.add(
                _BINARY_OPS.get(type(node.op), _snake(type(node.op).__name__)),
                (
                    (self.expression(node.left), "lhs"),
                    (self.expression(node.right), "rhs"),
                ),
                source=node,
            )
        if isinstance(node, ast.BoolOp):
            values = [self.expression(value) for value in node.values]
            op = "logical_and" if isinstance(node.op, ast.And) else "logical_or"
            result = values[0]
            for value in values[1:]:
                result = self.add(op, ((result, "lhs"), (value, "rhs")), source=node)
            return result
        if isinstance(node, ast.UnaryOp):
            op = {
                ast.USub: "neg",
                ast.UAdd: "positive",
                ast.Not: "logical_not",
                ast.Invert: "invert",
            }.get(type(node.op), _snake(type(node.op).__name__))
            return self.add(op, ((self.expression(node.operand), "operand"),), source=node)
        if isinstance(node, ast.Compare):
            left = self.expression(node.left)
            comparisons: list[int] = []
            for operator, comparator in zip(node.ops, node.comparators):
                right = self.expression(comparator)
                comparisons.append(
                    self.add(
                        _COMPARE_OPS.get(
                            type(operator), _snake(type(operator).__name__)
                        ),
                        ((left, "lhs"), (right, "rhs")),
                        source=node,
                    )
                )
                left = right
            result = comparisons[0]
            for comparison in comparisons[1:]:
                result = self.add(
                    "logical_and",
                    ((result, "lhs"), (comparison, "rhs")),
                    source=node,
                )
            return result
        if isinstance(node, ast.IfExp):
            return self.add(
                "select",
                (
                    (self.expression(node.test), "condition"),
                    (self.expression(node.body), "if_true"),
                    (self.expression(node.orelse), "if_false"),
                ),
                source=node,
            )
        if isinstance(node, ast.Attribute):
            return self.add(
                "attribute",
                ((self.expression(node.value), "value"),),
                attributes={"attribute": node.attr},
                source=node,
            )
        if isinstance(node, ast.Slice):
            return self.add(
                "slice_spec",
                (
                    (self.expression(node.lower), "lower"),
                    (self.expression(node.upper), "upper"),
                    (self.expression(node.step), "step"),
                ),
                source=node,
            )
        if isinstance(node, ast.Subscript):
            return self.add(
                "index",
                (
                    (self.expression(node.value), "base"),
                    (self.expression(node.slice), "index"),
                ),
                source=node,
            )
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            op = _snake(type(node).__name__)
            return self.add(
                op,
                (
                    (self.expression(value), f"item{index}")
                    for index, value in enumerate(node.elts)
                ),
                attributes={"length": len(node.elts)},
                source=node,
            )
        if isinstance(node, ast.Dict):
            inputs: list[tuple[int, str]] = []
            for index, (key, value) in enumerate(zip(node.keys, node.values)):
                inputs.append((self.expression(key), f"key{index}"))
                inputs.append((self.expression(value), f"value{index}"))
            return self.add(
                "dict",
                inputs,
                attributes={"length": len(node.values)},
                source=node,
            )
        if isinstance(node, ast.Starred):
            return self.add(
                "starred",
                ((self.expression(node.value), "value"),),
                source=node,
            )
        if isinstance(node, ast.Call):
            return self.call(node)
        if isinstance(node, ast.Lambda):
            before = dict(self.env)
            arguments = []
            for argument in node.args.args:
                value = self.add(
                    "lambda_input",
                    label=argument.arg,
                    attributes={"name": argument.arg},
                    source=argument,
                )
                self.env[argument.arg] = value
                arguments.append((value, argument.arg))
            body = self.expression(node.body)
            self.env = before
            return self.add("lambda", (*arguments, (body, "body")), source=node)
        if isinstance(
            node,
            (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp),
        ):
            return self.comprehension(node)
        if isinstance(node, ast.JoinedStr):
            return self.add(
                "joined_string",
                (
                    (self.expression(value), f"part{index}")
                    for index, value in enumerate(node.values)
                ),
                source=node,
            )
        if isinstance(node, ast.FormattedValue):
            inputs = [(self.expression(node.value), "value")]
            if node.format_spec is not None:
                inputs.append((self.expression(node.format_spec), "format"))
            return self.add(
                "formatted_value",
                inputs,
                attributes={"conversion": node.conversion},
                source=node,
            )
        if isinstance(node, ast.NamedExpr):
            value = self.expression(node.value)
            self.bind_target(node.target, value)
            return value
        if isinstance(node, ast.Await):
            return self.add("await", ((self.expression(node.value), "value"),), source=node)
        if isinstance(node, ast.Yield):
            return self.add("yield", ((self.expression(node.value), "value"),), source=node)
        if isinstance(node, ast.YieldFrom):
            return self.add(
                "yield_from", ((self.expression(node.value), "value"),), source=node
            )
        return self.generic(node)

    def call(self, node: ast.Call) -> int:
        function_text = ast.unparse(node.func)
        spelling = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else function_text
        )
        canonical = _TENSOR_CALL_ALIASES.get(spelling)
        inputs: list[tuple[int, str]] = []
        receiver = None
        if isinstance(node.func, ast.Attribute):
            receiver = self.expression(node.func.value)
            inputs.append(
                (receiver, "operand" if canonical is not None else "self")
            )
        for index, argument in enumerate(node.args):
            role = (
                "variadic"
                if isinstance(argument, ast.Starred)
                else f"arg{index}"
            )
            inputs.append((self.expression(argument), role))
        for keyword in node.keywords:
            inputs.append(
                (
                    self.expression(keyword.value),
                    f"kw:{keyword.arg or '**'}",
                )
            )
        if canonical in ELEMENTWISE_UNARY and len(inputs) == 1:
            inputs[0] = (inputs[0][0], "operand")
        elif canonical in ELEMENTWISE_BINARY and len(inputs) == 2:
            inputs[0] = (inputs[0][0], "lhs")
            inputs[1] = (inputs[1][0], "rhs")

        if canonical is not None:
            return self.add(
                canonical,
                inputs,
                label=canonical,
                attributes={"spelling": function_text},
                source=node,
            )

        callee = None
        constructed_type = None
        if isinstance(node.func, ast.Name):
            bound_callable = self.env.get(spelling)
            if bound_callable is not None:
                inputs.insert(0, (bound_callable, "callable"))
                if self.graph.G.nodes[bound_callable].get("op") == "function_def":
                    callee = bound_callable
            else:
                callee = self.definitions_by_scope.get(
                    (self.filename, self.scope, spelling)
                )
                if callee is None:
                    callee = self.definitions_by_scope.get(
                        (
                            self.filename,
                            self.lexical_parent_scope,
                            spelling,
                        )
                    )
                if callee is None:
                    callee = self.definitions_by_scope.get(
                        (self.filename, "<module>", spelling)
                    )
                if callee is None:
                    candidates = self.definition_candidates.get(spelling, ())
                    if len(candidates) == 1:
                        callee = candidates[0]
                if spelling in self.class_definitions:
                    constructed_type = spelling
        elif receiver is not None:
            receiver_type = self.value_types.get(receiver)
            if receiver_type is not None:
                callee = self.class_methods.get((receiver_type, spelling))
        call_id = self.add(
            "call",
            inputs,
            label=function_text,
            attributes={
                "function": function_text,
                "resolved": callee is not None,
                "callee": callee,
                "constructed_type": constructed_type,
            },
            source=node,
        )
        declared_entrypoint = None
        for value_id, role in inputs:
            if role != "kw:entrypoint":
                continue
            value = self.graph.G.nodes[value_id].get("constant")
            if not isinstance(value, str):
                continue
            candidates = self.definition_candidates.get(value, ())
            if len(candidates) == 1:
                declared_entrypoint = candidates[0]
                break
        if declared_entrypoint is not None:
            # A source compiler invocation is itself part of the program.
            # Keep the statically named program it compiles inside the same
            # entrypoint graph instead of hiding numerical work behind the
            # compiler wrapper.
            self.graph.G.nodes[call_id]["attributes"][
                "compiled_entrypoint"
            ] = declared_entrypoint
            self.graph.G.nodes[call_id]["extra_args"][
                "compiled_entrypoint"
            ] = declared_entrypoint
            self._control_edge(
                call_id,
                declared_entrypoint,
                "compiles",
            )
        if callee is not None:
            # Calls own their result value.  The control edge points into the
            # registered function region so an entrypoint traversal reaches
            # the entire implementation without pretending the definition is
            # a numerical operand.
            if nx.has_path(self.graph.G, callee, call_id):
                self.graph.G.nodes[call_id]["attributes"][
                    "recursive_or_reentrant"
                ] = True
            else:
                self._control_edge(call_id, callee, "invokes")
        if constructed_type is not None:
            self.value_types[call_id] = constructed_type
        return call_id

    def comprehension(self, node: ast.AST) -> int:
        before = dict(self.env)
        inputs: list[tuple[int, str]] = []
        for index, generator in enumerate(node.generators):
            iterator = self.expression(generator.iter)
            item = self.add(
                "iteration_item",
                ((iterator, "iterator"),),
                attributes={"generator": index},
                source=generator,
            )
            self.bind_target(generator.target, item)
            inputs.append((iterator, f"iterator{index}"))
            for guard_index, guard in enumerate(generator.ifs):
                inputs.append(
                    (self.expression(guard), f"guard{index}:{guard_index}")
                )
        if isinstance(node, ast.DictComp):
            inputs.append((self.expression(node.key), "key"))
            inputs.append((self.expression(node.value), "value"))
        else:
            inputs.append((self.expression(node.elt), "element"))
        self.env = before
        return self.add(_snake(type(node).__name__), inputs, source=node)

    def generic(
        self,
        node: ast.AST,
        *,
        extra: Iterable[tuple[int, str]] = (),
    ) -> int:
        inputs = list(extra)
        node_type = type(node).__name__
        schema = self.graph.role_schemas.get(node_type)
        attributes: dict[str, Any] = {
            "ast_type": node_type,
            "schema_registered": schema is not None,
        }
        fields = dict(ast.iter_fields(node))
        ordered_fields = (
            tuple(schema.get("up", ())) + tuple(schema.get("down", ()))
            if schema is not None
            else tuple(fields)
        )
        for field in dict.fromkeys(ordered_fields):
            value = fields.get(field)
            if isinstance(value, ast.AST):
                inputs.append((self.expression(value), field))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        inputs.append((self.expression(item), f"{field}{index}"))
                    else:
                        attributes[f"{field}{index}"] = item
            else:
                attributes[field] = value
        return self.add(
            _snake(type(node).__name__),
            inputs,
            attributes=attributes,
            source=node,
        )

    def statement(self, node: ast.stmt) -> int | None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self.function(node)
        if isinstance(node, ast.ClassDef):
            return self.class_definition(node)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return self.import_statement(node)
        if isinstance(node, ast.Assign):
            value = self.expression(node.value)
            results = []
            for target in node.targets:
                results.extend(self.bind_target(target, value))
            if not self.structural:
                return value
            return self.add(
                "assign",
                ((result, f"target{index}") for index, result in enumerate(results)),
                source=node,
                output_roles=(),
            )
        if isinstance(node, ast.AnnAssign):
            value = self.expression(node.value)
            bound = self.bind_target(node.target, value)
            if not self.structural:
                return value
            return self.add(
                "annotated_assign",
                ((item, f"target{index}") for index, item in enumerate(bound)),
                attributes={"annotation": ast.unparse(node.annotation)},
                source=node,
                output_roles=(),
            )
        if isinstance(node, ast.AugAssign):
            previous = self.expression(node.target)
            value = self.expression(node.value)
            result = self.add(
                _BINARY_OPS.get(type(node.op), _snake(type(node.op).__name__)),
                ((previous, "lhs"), (value, "rhs")),
                source=node,
            )
            self.bind_target(node.target, result)
            return result
        if isinstance(node, ast.Expr):
            return self.expression(node.value)
        if isinstance(node, ast.Return):
            value = self.expression(node.value)
            result = self.add(
                "return",
                ((value, "value"),),
                source=node,
                output_roles=(),
            )
            self.return_nodes.setdefault(self.scope, []).append(result)
            return result
        if isinstance(node, ast.If):
            return self.if_statement(node)
        if isinstance(node, (ast.For, ast.AsyncFor)):
            return self.loop_statement(node, kind="async_for" if isinstance(node, ast.AsyncFor) else "for")
        if isinstance(node, ast.While):
            return self.loop_statement(node, kind="while")
        if isinstance(node, (ast.With, ast.AsyncWith)):
            inputs = []
            for index, item in enumerate(node.items):
                inputs.append((self.expression(item.context_expr), f"context{index}"))
                if item.optional_vars is not None:
                    entered = self.add(
                        "context_enter",
                        ((inputs[-1][0], "context"),),
                        source=item,
                    )
                    self.bind_target(item.optional_vars, entered)
            marker = self.add(
                "async_with" if isinstance(node, ast.AsyncWith) else "with",
                inputs,
                source=node,
                control={"region": "body"},
            )
            for child in node.body:
                child_id = self.statement(child)
                if child_id is not None:
                    self.graph.G.add_edge(marker, child_id, role="body")
                    self.graph.G.nodes[marker]["children"].append((child_id, "body"))
                    self.graph.G.nodes[child_id]["parents"].append((marker, "body"))
            return marker
        if isinstance(node, ast.Try):
            marker = self.add("try", source=node, control={"region": "try"})
            for role, body in (
                ("body", node.body),
                ("orelse", node.orelse),
                ("finally", node.finalbody),
            ):
                for child in body:
                    child_id = self.statement(child)
                    if child_id is not None:
                        self._control_edge(marker, child_id, role)
            for handler in node.handlers:
                handler_id = self.generic(handler)
                self._control_edge(marker, handler_id, "handler")
            return marker
        if isinstance(node, ast.Raise):
            inputs = []
            if node.exc is not None:
                inputs.append((self.expression(node.exc), "exception"))
            if node.cause is not None:
                inputs.append((self.expression(node.cause), "cause"))
            return self.add("raise", inputs, source=node, output_roles=())
        if isinstance(node, ast.Assert):
            inputs = [(self.expression(node.test), "condition")]
            if node.msg is not None:
                inputs.append((self.expression(node.msg), "message"))
            return self.add("assert", inputs, source=node, output_roles=())
        if isinstance(node, (ast.Break, ast.Continue, ast.Pass)):
            return self.add(_snake(type(node).__name__), source=node, output_roles=())
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return self.add(
                _snake(type(node).__name__),
                attributes={"names": tuple(node.names)},
                source=node,
                output_roles=(),
            )
        return self.generic(node)

    def _control_edge(self, source: int, target: int, role: str) -> None:
        self.graph.G.add_edge(source, target, role=role)
        self.graph.G.nodes[source]["children"].append((target, role))
        self.graph.G.nodes[target]["parents"].append((source, role))

    def _propagate_common_value_type(
        self,
        target: int,
        *sources: int | None,
    ) -> None:
        """Retain one unambiguous runtime class across control-flow joins."""

        possible = {
            self.value_types[source]
            for source in sources
            if source is not None and source in self.value_types
        }
        if len(possible) == 1:
            self.value_types[target] = possible.pop()

    def if_statement(self, node: ast.If) -> int:
        condition = self.expression(node.test)
        before = dict(self.env)
        self.env = dict(before)
        then_nodes = [self.statement(child) for child in node.body]
        then_env = dict(self.env)
        self.env = dict(before)
        else_nodes = [self.statement(child) for child in node.orelse]
        else_env = dict(self.env)
        marker = condition
        if self.structural:
            marker = self.add(
                "if",
                ((condition, "condition"),),
                source=node,
                control={"then": True, "else": bool(node.orelse)},
            )
            for child in then_nodes:
                if child is not None:
                    self._control_edge(marker, child, "then")
            for child in else_nodes:
                if child is not None:
                    self._control_edge(marker, child, "else")
        merged = dict(before)
        for name in sorted(set(then_env) | set(else_env)):
            then_value = then_env.get(name, before.get(name))
            else_value = else_env.get(name, before.get(name))
            if then_value is None or else_value is None:
                continue
            if then_value == else_value:
                merged[name] = then_value
            else:
                merged[name] = self.add(
                    "select",
                    (
                        (condition, "condition"),
                        (then_value, "if_true"),
                        (else_value, "if_false"),
                    ),
                    attributes={"variable": name},
                    source=node,
                )
                self._propagate_common_value_type(
                    merged[name],
                    then_value,
                    else_value,
                )
        self.env = merged
        return marker

    def loop_statement(self, node: ast.AST, *, kind: str) -> int:
        before = dict(self.env)
        if isinstance(node, (ast.For, ast.AsyncFor)):
            iterator = self.expression(node.iter)
            condition_inputs = ((iterator, "iterator"),)
            item = self.add(
                "iteration_item",
                ((iterator, "iterator"),),
                source=node.target,
            )
            self.bind_target(node.target, item)
            body = node.body
            orelse = node.orelse
        else:
            condition = self.expression(node.test)
            condition_inputs = ((condition, "condition"),)
            body = node.body
            orelse = node.orelse
        body_nodes = [self.statement(child) for child in body]
        body_env = dict(self.env)
        marker_inputs = list(condition_inputs)
        changed = []
        for name in sorted(set(before) | set(body_env)):
            initial = before.get(name)
            updated = body_env.get(name, initial)
            if initial is not None and updated is not None and initial != updated:
                marker_inputs.extend(
                    ((initial, f"initial:{name}"), (updated, f"updated:{name}"))
                )
                changed.append(name)
        marker = self.add(
            kind,
            marker_inputs,
            attributes={"carried": tuple(changed)},
            source=node,
            control={"region": "loop", "kind": kind},
        )
        self.env = dict(before)
        for name in changed:
            result = self.add(
                "loop_result",
                ((marker, "loop"),),
                attributes={"variable": name},
                source=node,
            )
            self.env[name] = result
            self._propagate_common_value_type(
                result,
                before.get(name),
                body_env.get(name),
            )
        for child in orelse:
            child_id = self.statement(child)
            if child_id is not None:
                self._control_edge(marker, child_id, "orelse")
        return marker

    def import_statement(self, node: ast.Import | ast.ImportFrom) -> int:
        if isinstance(node, ast.Import):
            names = tuple(
                (alias.name, alias.asname or alias.name.split(".", 1)[0])
                for alias in node.names
            )
            module = None
            level = 0
        else:
            names = tuple((alias.name, alias.asname or alias.name) for alias in node.names)
            module = node.module
            level = node.level
        return self.add(
            "import",
            attributes={"module": module, "level": level, "names": names},
            source=node,
            output_roles=(),
        )

    def function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
        owner_scope = self.scope
        definition = self.definitions_by_scope.get(
            (self.filename, owner_scope, node.name)
        )
        if definition is None and self.structural:
            definition = self.add(
                "function_def",
                label=node.name,
                attributes={
                    "name": node.name,
                    "async": isinstance(node, ast.AsyncFunctionDef),
                    "nested_in": self.scope,
                },
                source=node,
            )
            self.definitions_by_scope[
                (self.filename, owner_scope, node.name)
            ] = definition
            self.definition_candidates[node.name].append(definition)
            if owner_scope in self.class_definitions:
                self.class_methods[(owner_scope, node.name)] = definition
            else:
                self.definitions.setdefault(node.name, definition)
        region_start = self.next_id
        (
            previous_env,
            previous_scope,
            previous_lexical_parent_scope,
            previous_filename,
        ) = (
            self.env,
            self.scope,
            self.lexical_parent_scope,
            self.filename,
        )
        # Nested functions close over the values already built in their
        # enclosing ProcessGraph region. Top-level functions begin from module
        # globals. This preserves lexical tensor dependencies such as a local
        # palette function reading ``phase`` from its parent.
        self.env = dict(
            previous_env
            if owner_scope != "<module>"
            else self.globals_by_file[self.filename]
        )
        self.scope = node.name
        self.lexical_parent_scope = owner_scope
        # A single extracted program function may contain all of its helper
        # definitions as nested source. Register that complete local surface
        # before visiting any helper body so forward references resolve in the
        # same way as multi-file module ingestion.
        for child in node.body:
            if not isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                continue
            key = (self.filename, node.name, child.name)
            if key in self.definitions_by_scope:
                continue
            nested = self.add(
                (
                    "class_def"
                    if isinstance(child, ast.ClassDef)
                    else "function_def"
                ),
                label=child.name,
                attributes={
                    "name": child.name,
                    "async": isinstance(child, ast.AsyncFunctionDef),
                    "nested_in": node.name,
                },
                source=child,
            )
            self.definitions_by_scope[key] = nested
            self.definition_candidates[child.name].append(nested)
            if isinstance(child, ast.ClassDef):
                self.class_definitions[child.name] = nested
                for method_node in child.body:
                    if not isinstance(
                        method_node,
                        (ast.FunctionDef, ast.AsyncFunctionDef),
                    ):
                        continue
                    method_key = (
                        self.filename,
                        child.name,
                        method_node.name,
                    )
                    if method_key in self.definitions_by_scope:
                        continue
                    method = self.add(
                        "function_def",
                        label=method_node.name,
                        attributes={
                            "name": method_node.name,
                            "async": isinstance(
                                method_node, ast.AsyncFunctionDef
                            ),
                            "nested_in": child.name,
                        },
                        source=method_node,
                    )
                    self.definitions_by_scope[method_key] = method
                    self.definition_candidates[
                        method_node.name
                    ].append(method)
                    self.class_methods[
                        (child.name, method_node.name)
                    ] = method
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            value = self.add(
                "input",
                label=argument.arg,
                attributes={
                    "name": argument.arg,
                    "function": node.name,
                    "annotation": (
                        ast.unparse(argument.annotation)
                        if argument.annotation is not None
                        else None
                    ),
                },
                source=argument,
                output_roles=("value",),
            )
            self.env[argument.arg] = value
            if argument.arg == "self" and owner_scope in self.class_definitions:
                self.value_types[value] = owner_scope
            elif argument.annotation is not None:
                annotation = ast.unparse(argument.annotation)
                annotation_name = annotation.split("[", 1)[0].split("|", 1)[0].strip()
                if annotation_name in self.class_definitions:
                    self.value_types[value] = annotation_name
            if definition is not None:
                self._control_edge(definition, value, "parameter")
        if node.args.vararg is not None:
            self.env[node.args.vararg.arg] = self.add(
                "vararg",
                label=node.args.vararg.arg,
                source=node.args.vararg,
            )
        if node.args.kwarg is not None:
            self.env[node.args.kwarg.arg] = self.add(
                "kwarg",
                label=node.args.kwarg.arg,
                source=node.args.kwarg,
            )
        for child in node.body:
            child_id = self.statement(child)
            if definition is not None and child_id is not None:
                self._control_edge(definition, child_id, "body")
        (
            self.env,
            self.scope,
            self.lexical_parent_scope,
            self.filename,
        ) = (
            previous_env,
            previous_scope,
            previous_lexical_parent_scope,
            previous_filename,
        )
        if definition is not None:
            # Data dependencies point from expressions toward statements, so
            # a definition linked only to its final statement does not make
            # the complete implementation forward-reachable. Record explicit
            # region ownership for every node created while parsing this
            # function. These are control/containment edges, never operands.
            for node_id in range(region_start, self.next_id):
                if node_id not in self.graph.G or node_id == definition:
                    continue
                data = self.graph.G.nodes[node_id]
                attrs = data.get("attributes") or {}
                if (
                    attrs.get("scope") != node.name
                    or str(
                        (data.get("source_span") or {}).get("filename") or ""
                    )
                    != str(previous_filename)
                ):
                    continue
                if not self.graph.G.has_edge(definition, node_id):
                    self._control_edge(
                        definition,
                        node_id,
                        "contains",
                    )
            return definition
        returns = self.return_nodes.get(node.name, ())
        return returns[-1] if returns else next(iter(self.env.values()), -1)

    def class_definition(self, node: ast.ClassDef) -> int:
        definition = self.class_definitions[node.name]
        previous_scope = self.scope
        self.scope = node.name
        region_start = self.next_id
        for child in node.body:
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            key = (self.filename, node.name, child.name)
            if key in self.definitions_by_scope:
                continue
            method = self.add(
                "function_def",
                label=child.name,
                attributes={
                    "name": child.name,
                    "async": isinstance(child, ast.AsyncFunctionDef),
                    "nested_in": node.name,
                },
                source=child,
            )
            self.definitions_by_scope[key] = method
            self.definition_candidates[child.name].append(method)
            self.class_methods[(node.name, child.name)] = method
        for child in node.body:
            child_id = self.statement(child)
            if child_id is not None:
                self._control_edge(definition, child_id, "body")
        for node_id in range(region_start, self.next_id):
            if node_id not in self.graph.G or node_id == definition:
                continue
            data = self.graph.G.nodes[node_id]
            if (data.get("attributes") or {}).get("scope") != node.name:
                continue
            if not self.graph.G.has_edge(definition, node_id):
                self._control_edge(definition, node_id, "contains")
        self.scope = previous_scope
        return definition

    def register_definitions(self, trees: list[tuple[ast.AST, str]]) -> None:
        for tree, filename in trees:
            body = tree.body if isinstance(tree, ast.Module) else [tree]
            for node in body:
                if isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    previous_filename = self.filename
                    self.filename = filename
                    definition = self.add(
                        "class_def" if isinstance(node, ast.ClassDef) else "function_def",
                        label=node.name,
                        attributes={
                            "name": node.name,
                            "async": isinstance(node, ast.AsyncFunctionDef),
                        },
                        source=node,
                    )
                    self.filename = previous_filename
                    self.definitions_by_scope[
                        (filename, "<module>", node.name)
                    ] = definition
                    self.definition_candidates[node.name].append(definition)
                    if isinstance(node, ast.ClassDef):
                        self.class_definitions[node.name] = definition
                        # Register the complete method surface before parsing
                        # any function body. A runner may construct a class
                        # whose source file appears later in a multi-file
                        # bundle; method resolution must not depend on source
                        # ordering.
                        for child in node.body:
                            if not isinstance(
                                child,
                                (ast.FunctionDef, ast.AsyncFunctionDef),
                            ):
                                continue
                            previous_filename = self.filename
                            self.filename = filename
                            method = self.add(
                                "function_def",
                                label=child.name,
                                attributes={
                                    "name": child.name,
                                    "async": isinstance(
                                        child, ast.AsyncFunctionDef
                                    ),
                                    "nested_in": node.name,
                                },
                                source=child,
                            )
                            self.filename = previous_filename
                            self.definitions_by_scope[
                                (filename, node.name, child.name)
                            ] = method
                            self.definition_candidates[child.name].append(
                                method
                            )
                            self.class_methods[
                                (node.name, child.name)
                            ] = method
                    else:
                        self.definitions.setdefault(node.name, definition)
                    self.definition_asts.append((node, filename, node.name))

    def _annotate_semantic_domains(self) -> None:
        """Mark the small compiler surface hidden inside the Python AST.

        Python is the control language here, not the numerical IR.  This pass
        identifies tensor-valued dataflow, control nodes, and explicit host
        boundaries without attempting to assign executable semantics to every
        Python object in the source bundle.
        """

        tensor_values: set[int] = set()
        tensor_containers: set[int] = set()
        tensor_metadata: set[int] = set()

        for node_id, data in self.graph.G.nodes(data=True):
            attrs = data.get("attributes") or {}
            annotation = str(attrs.get("annotation") or "")
            spelling = str(attrs.get("spelling") or "")
            function = str(attrs.get("function") or "")
            if data.get("op") == "input" and "AbstractTensor" in annotation:
                tensor_values.add(node_id)
            if spelling.startswith("AbstractTensor.") or function.startswith(
                "AbstractTensor."
            ):
                tensor_values.add(node_id)

        changed = True
        while changed:
            changed = False

            # A resolved Python call carries tensor arguments into the
            # corresponding function parameters. This is type flow, not call
            # execution or function inlining.
            for call_id, call_data in self.graph.G.nodes(data=True):
                if call_data.get("op") != "call":
                    continue
                attrs = call_data.get("attributes") or {}
                callee = attrs.get("callee")
                if callee not in self.graph.G:
                    continue
                parameters = [
                    child
                    for child in self.graph.G.successors(callee)
                    if self.graph.G.edges[callee, child].get("role")
                    == "parameter"
                ]
                positional = [
                    parent
                    for parent, role in call_data.get("parents") or ()
                    if role.startswith("arg") or role == "self"
                ]
                keywords = {
                    role[3:]: parent
                    for parent, role in call_data.get("parents") or ()
                    if role.startswith("kw:") and role != "kw:**"
                }
                for index, parameter in enumerate(parameters):
                    parameter_name = str(
                        (self.graph.G.nodes[parameter].get("attributes") or {})
                        .get("name", "")
                    )
                    argument = (
                        positional[index]
                        if index < len(positional)
                        else keywords.get(parameter_name)
                    )
                    if argument in tensor_values | tensor_containers:
                        target = (
                            tensor_containers
                            if argument in tensor_containers
                            else tensor_values
                        )
                        if parameter not in target:
                            target.add(parameter)
                            changed = True

            for node_id, data in self.graph.G.nodes(data=True):
                op = str(data.get("op") or "")
                attrs = data.get("attributes") or {}
                parents = [
                    parent for parent, _ in data.get("parents") or ()
                ]
                parent_is_tensor = any(
                    parent in tensor_values or parent in tensor_containers
                    for parent in parents
                )
                if op == "loop_result" and parents:
                    loop = parents[0]
                    parent_is_tensor = parent_is_tensor or any(
                        source in tensor_values
                        or source in tensor_containers
                        for source, _ in self.graph.G.nodes[loop].get(
                            "parents", ()
                        )
                    )

                if op == "attribute" and parents:
                    attribute = str(attrs.get("attribute") or "")
                    if (
                        parents[0] in tensor_values | tensor_containers
                        and attribute in _TENSOR_METADATA_ATTRIBUTES
                        and node_id not in tensor_metadata
                    ):
                        tensor_metadata.add(node_id)
                        changed = True
                    continue

                if op in _TENSOR_CONTAINER_OPS and parent_is_tensor:
                    if node_id not in tensor_containers:
                        tensor_containers.add(node_id)
                        changed = True
                    continue

                if op == "call":
                    function = str(attrs.get("function") or "")
                    method = function.rsplit(".", 1)[-1]
                    if method in _HOST_BOUNDARY_METHODS:
                        continue
                    callee = attrs.get("callee")
                    if callee in self.graph.G:
                        returns_tensor = any(
                            self.graph.G.nodes[return_id].get("op") == "return"
                            and any(
                                parent in tensor_values
                                or parent in tensor_containers
                                for parent, _ in self.graph.G.nodes[
                                    return_id
                                ].get("parents", ())
                            )
                            for return_id in self.graph.G.successors(callee)
                        )
                        if returns_tensor and node_id not in tensor_values:
                            tensor_values.add(node_id)
                            changed = True
                    elif attrs.get("constructed_type") and parent_is_tensor:
                        if node_id not in tensor_containers:
                            tensor_containers.add(node_id)
                            changed = True
                    continue

                if (
                    op in _TENSOR_VALUE_OPS
                    or op in {"iteration_item", "loop_result"}
                ) and parent_is_tensor:
                    if node_id not in tensor_values:
                        tensor_values.add(node_id)
                        changed = True

        for node_id, data in self.graph.G.nodes(data=True):
            op = str(data.get("op") or "")
            attrs = data.get("attributes") or {}
            function = str(attrs.get("function") or "")
            method = function.rsplit(".", 1)[-1]
            if node_id in tensor_values:
                attrs["semantic_kind"] = (
                    "tensor_input"
                    if op in {"input", "lambda_input", "vararg", "kwarg"}
                    else "tensor_operation"
                )
                attrs["execution_domain"] = "abstract_tensor"
                attrs["tensor_value"] = True
            elif node_id in tensor_containers:
                attrs["semantic_kind"] = "tensor_container"
                attrs["execution_domain"] = "abstract_tensor"
                attrs["tensor_value"] = True
            elif node_id in tensor_metadata:
                attrs["semantic_kind"] = "tensor_metadata"
                attrs["execution_domain"] = "scalar"
            elif op in _CONTROL_OPS:
                attrs["semantic_kind"] = "control"
                attrs["execution_domain"] = "python_control"
            elif op == "call" and method in _HOST_BOUNDARY_METHODS:
                attrs["semantic_kind"] = "host_boundary"
                attrs["execution_domain"] = "host"
            elif op == "call" and attrs.get("resolved"):
                attrs["semantic_kind"] = "process_call"
                attrs["execution_domain"] = "python_control"
            else:
                attrs.setdefault("semantic_kind", "python_support")
                attrs.setdefault("execution_domain", "python")

            # BitOps eligibility is a separate, deliberately conservative
            # decision. AbstractTensor arithmetic is not bit arithmetic merely
            # because the operator has a BitOps implementation.
            if attrs.get("bitops_capable"):
                attrs["bitops_candidate"] = bool(
                    attrs.get("dtype_domain") in {"bit", "integer"}
                )

    def build(
        self,
        trees: list[tuple[ast.AST, str]],
        *,
        entrypoint: str | None,
    ):
        definitions = [
            node
            for tree, _ in trees
            for node in (tree.body if isinstance(tree, ast.Module) else [tree])
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            )
        ]
        self.structural = not (
            entrypoint is None
            and len(trees) == 1
            and len(definitions) == 1
            and isinstance(definitions[0], (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        if self.structural:
            self.register_definitions(trees)
        else:
            node = definitions[0]
            self.definition_asts.append((node, trees[0][1], node.name))
        for tree, filename in trees:
            self.filename = filename
            body = tree.body if isinstance(tree, ast.Module) else [tree]
            for node in body:
                if isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    continue
                result = self.statement(node)
                if result is not None:
                    self.graph.roots.append(result)

        for node, filename, name in self.definition_asts:
            self.filename = filename
            self.function(node) if not isinstance(node, ast.ClassDef) else self.class_definition(node)

        if entrypoint is not None:
            candidates = self.definition_candidates.get(entrypoint, ())
            if not candidates:
                raise ValueError(f"AST entrypoint {entrypoint!r} was not found")
            if len(candidates) != 1:
                raise ValueError(
                    f"AST entrypoint {entrypoint!r} is ambiguous across "
                    f"{len(candidates)} definitions"
                )
            self.graph.roots = [candidates[0]]
        else:
            if self.structural:
                self.graph.roots.extend(self.definitions.values())
            else:
                self.graph.roots.extend(
                    node
                    for values in self.return_nodes.values()
                    for node in values
                )
        self.graph.roots = list(dict.fromkeys(self.graph.roots))
        # A constructor or recursive call can point back into the definition
        # region currently containing it. Keep that relationship as explicit
        # call metadata, but do not turn the schedulable ProcessGraph into a
        # cyclic call graph.
        invocation_edges = [
            (source, target)
            for source, target, data in self.graph.G.edges(data=True)
            if data.get("role") == "invokes"
        ]
        for source, target in invocation_edges:
            self.graph.G.remove_edge(source, target)
            if nx.has_path(self.graph.G, target, source):
                self.graph.G.nodes[source]["attributes"][
                    "recursive_or_reentrant"
                ] = True
                self.graph.G.nodes[source]["children"] = [
                    item
                    for item in self.graph.G.nodes[source]["children"]
                    if item != (target, "invokes")
                ]
                self.graph.G.nodes[target]["parents"] = [
                    item
                    for item in self.graph.G.nodes[target]["parents"]
                    if item != (source, "invokes")
                ]
            else:
                self.graph.G.add_edge(source, target, role="invokes")
        self._annotate_semantic_domains()
        self.graph.domain_shape = (1,)
        return self.graph


def build_semantic_process_graph(
    graph,
    sources: Any,
    *,
    filename: str | None = None,
    entrypoint: str | None = None,
    profile: str = "complete",
):
    """Ingest Python control and tensor operations into ``graph``.

    ``complete`` preserves every recognized Python syntax node for auditing.
    ``program`` retains every node in the transitively resolved entrypoint,
    including host I/O and finalization. ``tensor_control`` projects that
    entrypoint to AbstractTensor operations, governing Python control flow,
    dependencies, and explicit host boundaries that a backend compiler needs.
    """

    trees = _read_sources(sources, filename=filename)
    SemanticAstBuilder(graph).build(trees, entrypoint=entrypoint)
    if profile == "complete":
        graph.G.graph["semantic_profile"] = profile
        return graph
    if profile not in {"tensor_control", "program"}:
        raise ValueError(
            "semantic AST profile must be 'complete', 'program', or "
            "'tensor_control'"
        )
    if entrypoint is None:
        raise ValueError(f"{profile} profile requires an entrypoint")
    if profile == "program":
        return project_entrypoint_program_graph(graph)
    return project_tensor_control_graph(graph)


def _reachable_entrypoint_regions(graph):
    if len(graph.roots) != 1:
        raise ValueError("entrypoint projection requires one graph root")
    root = graph.roots[0]
    calls_by_scope: dict[tuple[str, str], list[int]] = defaultdict(list)
    for node_id, data in graph.G.nodes(data=True):
        if data.get("op") != "call":
            continue
        calls_by_scope[
            (
                str((data.get("source_span") or {}).get("filename") or ""),
                str((data.get("attributes") or {}).get("scope") or ""),
            )
        ].append(node_id)
    reachable_definitions = {root}
    pending = [root]
    while pending:
        definition = pending.pop()
        definition_data = graph.G.nodes[definition]
        scope = str(definition_data.get("label") or "")
        filename = str(
            (definition_data.get("source_span") or {}).get("filename") or ""
        )
        for node_id in calls_by_scope.get((filename, scope), ()):
            data = graph.G.nodes[node_id]
            attrs = data.get("attributes") or {}
            for callee in (
                attrs.get("callee"),
                attrs.get("compiled_entrypoint"),
            ):
                if callee in graph.G and callee not in reachable_definitions:
                    reachable_definitions.add(callee)
                    pending.append(callee)

    reachable_scopes = {
        (
            str((graph.G.nodes[node].get("source_span") or {}).get("filename") or ""),
            str(graph.G.nodes[node].get("label") or ""),
        )
        for node in reachable_definitions
    }

    def in_reachable_scope(node_id: int) -> bool:
        data = graph.G.nodes[node_id]
        return (
            str((data.get("source_span") or {}).get("filename") or ""),
            str((data.get("attributes") or {}).get("scope") or ""),
        ) in reachable_scopes

    return root, reachable_definitions, in_reachable_scope


def _finish_entrypoint_projection(
    graph,
    *,
    root: int,
    keep: set[int],
    profile: str,
    complete_nodes: int,
    reachable_function_count: int,
):
    removed = set(graph.G) - keep
    graph.G.remove_nodes_from(removed)
    for node_id, data in graph.G.nodes(data=True):
        data["parents"] = [
            (parent, role)
            for parent, role in data.get("parents", ())
            if parent in graph.G
        ]
        data["children"] = [
            (child, role)
            for child, role in data.get("children", ())
            if child in graph.G
        ]
        data["input_roles"] = tuple(
            role for _, role in data.get("parents", ())
        )
    graph.roots = [root]

    # Module globals and other explicit environmental inputs are data
    # predecessors, not function-body statements. Attach them to the one
    # program root so the complete retained program is forward-traversable
    # from start to finish without changing their operand relationships.
    forward = nx.descendants(graph.G, root) | {root}
    for node_id in sorted(set(graph.G) - forward):
        if nx.has_path(graph.G, node_id, root):
            raise ValueError(
                "entrypoint ownership would create a ProcessGraph cycle"
            )
        graph.G.add_edge(root, node_id, role="environment")
        graph.G.nodes[root]["children"].append((node_id, "environment"))
        graph.G.nodes[node_id]["parents"].append((root, "environment"))
    forward = nx.descendants(graph.G, root) | {root}
    if len(forward) != graph.G.number_of_nodes():
        raise RuntimeError("entrypoint projection left unreachable nodes")

    graph.G.graph.update(
        semantic_profile=profile,
        complete_node_count=complete_nodes,
        filtered_node_count=len(removed),
        reachable_function_count=reachable_function_count,
        entrypoint_expanded=True,
        forward_reachable_node_count=len(forward),
    )
    return graph


def project_entrypoint_program_graph(graph):
    """Retain the complete start-to-finish program under one entrypoint.

    Unlike ``tensor_control``, this profile does not discard Python support,
    validation, UI, container, or I/O nodes inside reachable project
    functions. Resolved calls, statically declared compiled entrypoints, and
    class methods are all part of the same forward-traversable ProcessGraph.
    """

    complete_nodes = graph.G.number_of_nodes()
    root, reachable_definitions, in_reachable_scope = (
        _reachable_entrypoint_regions(graph)
    )
    keep = set(reachable_definitions)
    keep.update(
        node_id for node_id in graph.G if in_reachable_scope(node_id)
    )
    pending = list(keep)
    while pending:
        node_id = pending.pop()
        for parent in graph.G.predecessors(node_id):
            role = graph.G.edges[parent, node_id].get("role")
            if role in {
                "body",
                "compiles",
                "contains",
                "environment",
                "invokes",
                "parameter",
            }:
                continue
            if parent in keep:
                continue
            keep.add(parent)
            pending.append(parent)
    return _finish_entrypoint_projection(
        graph,
        root=root,
        keep=keep,
        profile="program",
        complete_nodes=complete_nodes,
        reachable_function_count=len(reachable_definitions),
    )


def project_tensor_control_graph(graph):
    """Discard Python noise while retaining tensor-governing process flow.

    This is a source projection, not a Python compiler.  It intentionally
    keeps only reachable function regions, tensor dataflow, control constructs,
    required scalar/shape dependencies, and host materialization boundaries.
    """

    complete_nodes = graph.G.number_of_nodes()
    root, reachable_definitions, in_reachable_scope = (
        _reachable_entrypoint_regions(graph)
    )

    def validation_only(node_id: int) -> bool:
        if graph.G.nodes[node_id].get("op") != "if":
            return False
        branches = [
            child
            for child in graph.G.successors(node_id)
            if graph.G.edges[node_id, child].get("role") in {"then", "else"}
        ]
        return bool(branches) and all(
            graph.G.nodes[child].get("op") in {"assert", "raise"}
            for child in branches
        )

    seeds: set[int] = set(reachable_definitions)
    for node_id, data in graph.G.nodes(data=True):
        if not in_reachable_scope(node_id):
            continue
        attrs = data.get("attributes") or {}
        kind = attrs.get("semantic_kind")
        if kind in {
            "tensor_input",
            "tensor_operation",
            "tensor_container",
            "tensor_metadata",
            "host_boundary",
        }:
            seeds.add(node_id)
        elif kind == "control" and not validation_only(node_id):
            seeds.add(node_id)
        elif (
            kind == "process_call"
            and (
                attrs.get("callee") in reachable_definitions
                or attrs.get("compiled_entrypoint")
                in reachable_definitions
            )
        ):
            seeds.add(node_id)

    keep = set(seeds)
    pending = list(seeds)
    while pending:
        node_id = pending.pop()
        for parent in graph.G.predecessors(node_id):
            if parent in keep:
                continue
            if in_reachable_scope(parent) or parent in reachable_definitions:
                keep.add(parent)
                pending.append(parent)

    return _finish_entrypoint_projection(
        graph,
        root=root,
        keep=keep,
        profile="tensor_control",
        complete_nodes=complete_nodes,
        reachable_function_count=len(reachable_definitions),
    )


__all__ = [
    "SemanticAstBuilder",
    "build_semantic_process_graph",
    "project_entrypoint_program_graph",
    "project_tensor_control_graph",
]
