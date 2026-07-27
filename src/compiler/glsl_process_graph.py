"""Structured ProcessGraph lowering to one GLSL elementwise compute shader.

Python AST contributes resolved calls and control structure.  AbstractTensor
nodes contribute the canonical numerical operations.  This backend compiler
inlines resolved function regions and emits ProcessGraph ``for`` nodes as GLSL
loops; it does not unroll the loop or introduce an application-specific IR.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Any, Iterable, Mapping

import numpy as np

from ..common.tensors.fused_ir import Meta
from ..common.tensors.accelerator_backends.glsl_backend import (
    GLChunk,
    _SHADER_HEADER,
    _compile,
    _dispatch_many,
    _glsl_literal,
    _glsl_type,
    _normalize_dtype,
    _result_dtype,
    _typed_expr,
    canonical_op,
    plan_launch,
)


@dataclass(frozen=True)
class _GLValue:
    expression: str
    dtype: np.dtype


@dataclass(frozen=True)
class CompiledGLSLProcessGraph:
    """One structured shader compiled from a tensor/control ProcessGraph."""

    source: str
    input_names: tuple[str, ...]
    scalar_inputs: frozenset[str]
    scalar_input_order: tuple[str, ...]
    scalar_buffer_name: str | None
    output_names: tuple[str, ...]
    output_dtypes: tuple[np.dtype, ...]
    source_node_count: int
    primitive_count: int
    loop_count: int

    def execute(
        self,
        feeds: Mapping[str, GLChunk],
        *,
        outs: Mapping[str, GLChunk] | None = None,
    ) -> Mapping[str, GLChunk]:
        """Execute the compiled shader with resident GL buffers."""

        missing = set(self.input_names) - set(feeds)
        if missing:
            raise ValueError(f"missing ProcessGraph GLSL feeds: {sorted(missing)}")
        ordered_feeds = [feeds[name] for name in self.input_names]
        nonscalar = [
            feeds[name]
            for name in self.input_names
            if name != self.scalar_buffer_name
        ]
        if not nonscalar:
            raise ValueError("structured GLSL needs at least one array feed")
        output_shape = nonscalar[0].shape
        if any(chunk.shape != output_shape for chunk in nonscalar[1:]):
            raise ValueError("non-scalar structured GLSL feeds must share shape")
        if outs is None:
            resolved_outs = {
                name: GLChunk(output_shape, dtype=dtype).to_gpu()
                for name, dtype in zip(self.output_names, self.output_dtypes)
            }
        else:
            missing_outputs = set(self.output_names) - set(outs)
            if missing_outputs:
                raise ValueError(
                    f"missing ProcessGraph GLSL outputs: "
                    f"{sorted(missing_outputs)}"
                )
            resolved_outs = {
                name: outs[name] for name in self.output_names
            }
            if any(
                chunk.shape != output_shape
                for chunk in resolved_outs.values()
            ):
                raise ValueError("structured GLSL outputs must share feed shape")
        ordered_outs = [
            resolved_outs[name] for name in self.output_names
        ]
        count = int(np.prod(output_shape, dtype=np.int64))
        plan = plan_launch(
            count,
            binding_count=len(ordered_feeds) + len(ordered_outs),
        )
        _dispatch_many(
            _compile(self.source),
            ordered_feeds,
            ordered_outs,
            plan,
        )
        return resolved_outs


_SCALAR_BINARY = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "truediv": operator.truediv,
    "floordiv": operator.floordiv,
    "mod": operator.mod,
    "pow": operator.pow,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "eq": operator.eq,
    "ne": operator.ne,
    "less": operator.lt,
    "less_equal": operator.le,
    "greater": operator.gt,
    "greater_equal": operator.ge,
    "equal": operator.eq,
    "not_equal": operator.ne,
    "minimum": min,
    "maximum": max,
}

_SCALAR_UNARY = {
    "abs": abs,
    "logical_not": operator.not_,
    "neg": operator.neg,
    "positive": operator.pos,
}

_NON_VALUE_ROLES = frozenset(
    {
        "body",
        "else",
        "finally",
        "handler",
        "invokes",
        "orelse",
        "parameter",
        "then",
    }
)


class _Compiler:
    def __init__(
        self,
        graph,
        *,
        specializations: Mapping[str, Any],
        input_meta: Mapping[str, Meta],
        scalar_inputs: frozenset[str],
        output_names: tuple[str, ...],
    ) -> None:
        self.graph = graph
        self.specializations = dict(specializations)
        self.input_meta = dict(input_meta)
        self.scalar_inputs = scalar_inputs
        self.output_names = output_names
        self.input_names: list[str] = []
        self.array_input_names: list[str] = []
        self.input_values: dict[str, _GLValue] = {}
        self.scalar_offsets: dict[str, int] = {}
        self.scalar_buffer_name = "__scalar_controls__"
        self.output_dtypes: list[np.dtype] = []
        self.helpers: list[str] = []
        self.lines: list[str] = []
        self.indent = 1
        self.value_index = 0
        self.primitive_count = 0
        self.loop_count = 0
        self.regions: dict[int, set[int]] = {}
        self.parameters: dict[int, list[int]] = {}
        self.returns: dict[int, list[int]] = {}
        self._index_functions()

    def _definition_key(self, definition: int) -> tuple[str, str]:
        data = self.graph.G.nodes[definition]
        return (
            str((data.get("source_span") or {}).get("filename") or ""),
            str(data.get("label") or ""),
        )

    def _node_key(self, node_id: int) -> tuple[str, str]:
        data = self.graph.G.nodes[node_id]
        return (
            str((data.get("source_span") or {}).get("filename") or ""),
            str((data.get("attributes") or {}).get("scope") or ""),
        )

    def _index_functions(self) -> None:
        for definition, data in self.graph.G.nodes(data=True):
            if data.get("op") != "function_def":
                continue
            key = self._definition_key(definition)
            region = {
                node_id
                for node_id in self.graph.G
                if self._node_key(node_id) == key
            }
            self.regions[definition] = region
            self.parameters[definition] = [
                child
                for child in self.graph.G.successors(definition)
                if self.graph.G.edges[definition, child].get("role")
                == "parameter"
            ]
            self.returns[definition] = [
                node_id
                for node_id in region
                if self.graph.G.nodes[node_id].get("op") == "return"
            ]

    def value_parents(self, node_id: int) -> list[tuple[int, str]]:
        return [
            (parent, role)
            for parent, role in self.graph.G.nodes[node_id].get("parents", ())
            if role not in _NON_VALUE_ROLES
        ]

    def append(self, line: str) -> None:
        self.lines.append("    " * self.indent + line)

    def new_value(self, op: str, values: list[Any]) -> _GLValue:
        name, prefix_reverse = canonical_op(op)
        tensors = [
            (index, value)
            for index, value in enumerate(values)
            if isinstance(value, _GLValue)
        ]
        if len(values) == 1 and len(tensors) == 1:
            left = tensors[0][1]
            right_expression = None
            right_dtype = None
            reverse = False
        elif len(values) == 2 and tensors:
            left_value, right_value = values
            if isinstance(left_value, _GLValue):
                left = left_value
            else:
                right_tensor = right_value
                assert isinstance(right_tensor, _GLValue)
                left = _GLValue(
                    _glsl_literal(left_value, right_tensor.dtype),
                    right_tensor.dtype,
                )
            if isinstance(right_value, _GLValue):
                right_expression = right_value.expression
                right_dtype = right_value.dtype
            else:
                right_dtype = _normalize_dtype(
                    np.asarray(right_value).dtype
                )
                right_expression = _glsl_literal(right_value, right_dtype)
            reverse = prefix_reverse
        else:
            raise ValueError(f"GLSL primitive {op!r} has invalid operands")
        out_dtype = _result_dtype(name, left.dtype, right_dtype)
        helper, expression = _typed_expr(
            name,
            left.expression,
            right_expression,
            reverse,
            left.dtype,
            right_dtype,
            out_dtype,
        )
        if helper and helper not in self.helpers:
            self.helpers.append(helper)
        variable = f"v{self.value_index}"
        self.value_index += 1
        self.primitive_count += 1
        self.append(
            f"{_glsl_type(out_dtype)} {variable} = {expression};"
        )
        return _GLValue(variable, out_dtype)

    def scalar_operation(self, op: str, values: list[Any]) -> Any:
        if op == "max":
            return max(values)
        if op == "min":
            return min(values)
        if op in _SCALAR_BINARY and len(values) == 2:
            return _SCALAR_BINARY[op](*values)
        if op in _SCALAR_UNARY and len(values) == 1:
            return _SCALAR_UNARY[op](values[0])
        raise ValueError(f"cannot specialize Python scalar op {op!r}")

    def call_function(
        self,
        definition: int,
        arguments: list[tuple[str, Any]],
        parent: "_Context | None",
    ) -> Any:
        positional = [
            value for role, value in arguments if role.startswith("arg")
        ]
        keywords = {
            role[3:]: value
            for role, value in arguments
            if role.startswith("kw:") and role != "kw:**"
        }
        overrides: dict[int, Any] = {}
        for index, parameter in enumerate(self.parameters[definition]):
            name = str(
                (self.graph.G.nodes[parameter].get("attributes") or {}).get(
                    "name", ""
                )
            )
            if index < len(positional):
                overrides[parameter] = positional[index]
            elif name in keywords:
                overrides[parameter] = keywords[name]
            else:
                raise ValueError(
                    f"resolved call omitted unsupported default {name!r}"
                )
        context = _Context(self, definition, overrides, parent)
        returns = self.returns[definition]
        if len(returns) != 1:
            raise ValueError(
                f"function {self.graph.G.nodes[definition].get('label')!r} "
                f"has {len(returns)} returns; normalize branches first"
            )
        return context.eval(returns[0])

    def compile(self) -> CompiledGLSLProcessGraph:
        if len(self.graph.roots) != 1:
            raise ValueError("structured GLSL compilation needs one entrypoint")
        root = self.graph.roots[0]
        root_tensor_names = [
            str(
                (
                    self.graph.G.nodes[parameter].get("attributes") or {}
                ).get("name") or ""
            )
            for parameter in self.parameters[root]
            if (
                self.graph.G.nodes[parameter].get("attributes") or {}
            ).get("execution_domain") == "abstract_tensor"
        ]
        self.array_input_names = [
            name
            for name in root_tensor_names
            if name not in self.scalar_inputs
        ]
        scalar_names = [
            name for name in root_tensor_names if name in self.scalar_inputs
        ]
        scalar_dtypes = {
            _normalize_dtype(
                self.input_meta.get(
                    name, Meta(dtype="float32", device="glsl")
                ).dtype
            )
            for name in scalar_names
        }
        if len(scalar_dtypes) > 1:
            raise ValueError(
                "packed scalar GLSL feeds must share one storage dtype"
            )
        scalar_dtype = (
            next(iter(scalar_dtypes))
            if scalar_dtypes
            else np.dtype(np.float32)
        )
        self.scalar_offsets = {
            name: index for index, name in enumerate(scalar_names)
        }
        self.input_names = list(self.array_input_names)
        if scalar_names:
            self.input_names.append(self.scalar_buffer_name)

        root_arguments: list[tuple[str, Any]] = []
        for index, parameter in enumerate(self.parameters[root]):
            attrs = self.graph.G.nodes[parameter].get("attributes") or {}
            name = str(attrs.get("name") or "")
            if name in self.specializations:
                value: Any = self.specializations[name]
            elif attrs.get("execution_domain") == "abstract_tensor":
                meta = self.input_meta.get(
                    name, Meta(dtype="float32", device="glsl")
                )
                dtype = _normalize_dtype(meta.dtype)
                if name in self.scalar_offsets:
                    value = _GLValue(
                        f"scalar_feed[{self.scalar_offsets[name]}]",
                        dtype,
                    )
                else:
                    binding = self.array_input_names.index(name)
                    value = _GLValue(f"feed{binding}[gid]", dtype)
                self.input_values[name] = value
            else:
                raise ValueError(
                    f"entrypoint scalar {name!r} needs specialization"
                )
            root_arguments.append((f"arg{index}", value))

        outputs = self.call_function(root, root_arguments, None)
        output_values = outputs if isinstance(outputs, tuple) else (outputs,)
        if len(output_values) != len(self.output_names):
            raise ValueError(
                f"entrypoint returned {len(output_values)} values for "
                f"{len(self.output_names)} outputs"
            )
        if not all(isinstance(value, _GLValue) for value in output_values):
            raise ValueError("every structured GLSL output must be a tensor")

        declarations = [_SHADER_HEADER.format(local_size=256)]
        for binding, name in enumerate(self.array_input_names):
            value = self.input_values[name]
            declarations.append(
                f"layout(std430, binding = {binding}) readonly buffer "
                f"Feed{binding} {{ {_glsl_type(value.dtype)} "
                f"feed{binding}[]; }};"
            )
        if scalar_names:
            declarations.append(
                f"layout(std430, binding = {len(self.array_input_names)}) "
                f"readonly buffer ScalarFeed {{ {_glsl_type(scalar_dtype)} "
                "scalar_feed[]; };"
            )
        for output_index, (name, value) in enumerate(
            zip(self.output_names, output_values)
        ):
            binding = len(self.input_names) + output_index
            declarations.append(
                f"layout(std430, binding = {binding}) writeonly buffer "
                f"Out{output_index} {{ {_glsl_type(value.dtype)} "
                f"out{output_index}[]; }};"
            )
            self.output_dtypes.append(value.dtype)
        body = [
            "",
            "uniform uint u_count;",
            "",
            "void main() {",
            "    uint gid = turing_linear_gid();",
            "    if (gid >= u_count) { return; }",
            *self.lines,
        ]
        for output_index, value in enumerate(output_values):
            body.append(f"    out{output_index}[gid] = {value.expression};")
        body.append("}")
        source = "\n".join(
            declarations
            + ([""] + self.helpers if self.helpers else [])
            + body
        ) + "\n"
        return CompiledGLSLProcessGraph(
            source=source,
            input_names=tuple(self.input_names),
            scalar_inputs=self.scalar_inputs,
            scalar_input_order=tuple(scalar_names),
            scalar_buffer_name=(
                self.scalar_buffer_name if scalar_names else None
            ),
            output_names=self.output_names,
            output_dtypes=tuple(self.output_dtypes),
            source_node_count=self.graph.G.number_of_nodes(),
            primitive_count=self.primitive_count,
            loop_count=self.loop_count,
        )


class _Context:
    def __init__(
        self,
        compiler: _Compiler,
        definition: int,
        overrides: Mapping[int, Any],
        parent: "_Context | None",
    ) -> None:
        self.compiler = compiler
        self.definition = definition
        self.region = compiler.regions[definition]
        self.overrides = dict(overrides)
        self.parent = parent
        self.cache: dict[int, Any] = {}
        self.active_loops: set[int] = set()

    def eval(self, node_id: int) -> Any:
        if node_id in self.overrides:
            return self.overrides[node_id]
        if node_id not in self.region:
            if self.parent is None:
                raise ValueError(f"unbound ProcessGraph value {node_id}")
            return self.parent.eval(node_id)
        if node_id in self.cache:
            return self.cache[node_id]
        data = self.compiler.graph.G.nodes[node_id]
        op = str(data.get("op") or "")
        parents = self.compiler.value_parents(node_id)

        if op == "const":
            value = data.get("constant")
        elif op == "input":
            raise ValueError(f"unbound function input {data.get('label')!r}")
        elif op == "return":
            value = self.eval(
                next(parent for parent, role in parents if role == "value")
            )
        elif op in {"tuple", "list"}:
            value = tuple(self.eval(parent) for parent, _ in parents)
        elif op == "tuple_get":
            base = self.eval(parents[0][0])
            index = int(self.eval(parents[1][0]))
            value = base[index]
        elif op == "call":
            value = self._call(node_id, parents)
        elif op == "loop_result":
            loop = next(parent for parent, role in parents if role == "loop")
            self._loop(loop)
            return self.cache[node_id]
        else:
            values = [self.eval(parent) for parent, _ in parents]
            if any(isinstance(item, _GLValue) for item in values):
                value = self.compiler.new_value(op, values)
            else:
                value = self.compiler.scalar_operation(op, values)
        self.cache[node_id] = value
        return value

    def _call(
        self,
        node_id: int,
        parents: list[tuple[int, str]],
    ) -> Any:
        data = self.compiler.graph.G.nodes[node_id]
        attrs = data.get("attributes") or {}
        arguments = [
            (role, self.eval(parent))
            for parent, role in parents
            if role != "callable"
        ]
        callee = attrs.get("callee")
        if callee in self.compiler.graph.G:
            return self.compiler.call_function(callee, arguments, self)
        function = str(attrs.get("function") or "")
        values = [value for _, value in arguments]
        if function == "range":
            return range(*[int(value) for value in values])
        if function == "max":
            return max(values)
        if function == "min":
            return min(values)
        raise ValueError(f"unresolved Python call {function!r} crosses GLSL")

    def _loop(self, loop_id: int) -> None:
        if loop_id in self.active_loops:
            raise ValueError("recursive ProcessGraph loop")
        self.active_loops.add(loop_id)
        try:
            parents = self.compiler.value_parents(loop_id)
            iterator_node = next(
                parent for parent, role in parents if role == "iterator"
            )
            iterator = tuple(self.eval(iterator_node))
            initial_nodes: dict[str, int] = {}
            carried: dict[str, _GLValue] = {}
            updated_nodes: dict[str, int] = {}
            for parent, role in parents:
                if role.startswith("initial:"):
                    name = role.split(":", 1)[1]
                    initial_nodes[name] = parent
                    initial = self.eval(parent)
                    if not isinstance(initial, _GLValue):
                        raise ValueError("GLSL loop carries must be tensors")
                    carried[name] = initial
                elif role.startswith("updated:"):
                    updated_nodes[role.split(":", 1)[1]] = parent
            if set(initial_nodes) != set(updated_nodes):
                raise ValueError("loop carry inputs and updates do not match")
            used_iteration_items = [
                node_id
                for node_id in self.region
                if self.compiler.graph.G.nodes[node_id].get("op")
                == "iteration_item"
                and self.compiler.graph.G.out_degree(node_id)
            ]
            if used_iteration_items:
                raise ValueError(
                    "GLSL loop iteration values need scalar SSA lowering"
                )

            mutable: dict[str, _GLValue] = {}
            loop_number = self.compiler.loop_count
            for name, initial in carried.items():
                variable = f"loop{loop_number}_{name}"
                self.compiler.append(
                    f"{_glsl_type(initial.dtype)} {variable} = "
                    f"{initial.expression};"
                )
                mutable[name] = _GLValue(variable, initial.dtype)

            self.compiler.append(
                f"for (int loop{loop_number}_i = 0; "
                f"loop{loop_number}_i < {len(iterator)}; "
                f"++loop{loop_number}_i) {{"
            )
            self.compiler.indent += 1
            iteration_overrides = dict(self.overrides)
            iteration_overrides.update(
                {
                    initial_nodes[name]: mutable[name]
                    for name in initial_nodes
                }
            )
            iteration_context = _Context(
                self.compiler,
                self.definition,
                iteration_overrides,
                self.parent,
            )
            iteration_context.cache.update(self.cache)
            updated = {
                name: iteration_context.eval(updated_nodes[name])
                for name in initial_nodes
            }
            for name, value in updated.items():
                if not isinstance(value, _GLValue):
                    raise ValueError("GLSL loop update must be a tensor")
                self.compiler.append(
                    f"{mutable[name].expression} = {value.expression};"
                )
            self.compiler.indent -= 1
            self.compiler.append("}")
            self.compiler.loop_count += 1

            for result_id in self.compiler.graph.G.successors(loop_id):
                result_data = self.compiler.graph.G.nodes[result_id]
                if result_data.get("op") != "loop_result":
                    continue
                name = str(
                    (result_data.get("attributes") or {}).get("variable")
                )
                self.cache[result_id] = mutable[name]
        finally:
            self.active_loops.remove(loop_id)


def compile_process_graph_glsl(
    graph,
    *,
    specializations: Mapping[str, Any],
    input_meta: Mapping[str, Meta] | None = None,
    scalar_tensor_inputs: Iterable[str] = (),
    output_names: Iterable[str] = ("result",),
) -> CompiledGLSLProcessGraph:
    """Compile one same-shape tensor/control entrypoint into one shader."""

    return _Compiler(
        graph,
        specializations=specializations,
        input_meta=input_meta or {},
        scalar_inputs=frozenset(scalar_tensor_inputs),
        output_names=tuple(output_names),
    ).compile()


__all__ = ["CompiledGLSLProcessGraph", "compile_process_graph_glsl"]
