"""GLSL deployment-strategy stage for ProcessGraphs."""

from __future__ import annotations

import ast
import builtins
import operator
import time
import traceback
from collections import deque
from contextlib import contextmanager
from typing import Any

import networkx as nx

from .deployment_fifo import DeploymentFIFO
from .loop_composer import (
    LoopBackendCapabilities,
    LoopComposer,
)
from .process_graph_callable import EphemeralProcessGraphCallable
from .process_graph_fusion import (
    extract_clean_process_subgraph,
    serialize_scheduled_operator_dispatches,
)
from .shell_reference_tables import build_shell_reference_tables
from ..common.tensors.abstraction import AbstractTensor
from ..common.tensors.accelerator_backends.glsl_fused_network import (
    GLSLFusedProgramNetwork,
)
from ..common.tensors.accelerator_backends.c_primitive_program import (
    CapturedFusedProgram,
    compile_recorded_fused_tape,
)
from ..common.tensors.accelerator_backends.glsl_backend import (
    compile_captured_fused_program,
    dispatch_stats,
    emit_multi_output_program_source,
    execute_captured_fused_program,
)
from ..common.tensors.accelerator_backends.glsl_tensor_backend import (
    GLSLTensorOperations,
)
from ..common.tensors.autograd import autograd
from ..common.tensors.fused_ir import (
    FusedProgram,
    OpStep,
)
from ..transmogrifier.graph.graph_deep_compiler import GraphDeepCompiler
from ..transmogrifier.operator_defs import (
    abstract_tensor_funcs,
    abstract_tensor_sigs,
)


class DeploymentErrorBuffer:
    """Root-owned traceback FIFO shared by every shell in a deployment."""

    def __init__(self, capacity: int = 240) -> None:
        self.records = deque(maxlen=max(1, int(capacity)))
        self.sequence = 0
        self._by_exception_id: dict[int, dict[str, Any]] = {}

    def push(
        self,
        exception: BaseException,
        *,
        path: str,
        phase: str,
        node_id: int | None = None,
        handled: bool = False,
    ) -> dict[str, Any]:
        existing = self._by_exception_id.get(id(exception))
        if existing is not None:
            hop = {
                "path": str(path),
                "phase": str(phase),
                "node_id": node_id,
                "handled": bool(handled),
            }
            if hop not in existing["propagation"]:
                existing["propagation"].append(hop)
            return existing
        self.sequence += 1
        record = {
            "sequence": self.sequence,
            "path": str(path),
            "phase": str(phase),
            "node_id": node_id,
            "handled": bool(handled),
            "exception_type": type(exception).__name__,
            "message": str(exception),
            "propagation": [],
            "traceback": "".join(
                traceback.format_exception(
                    type(exception),
                    exception,
                    exception.__traceback__,
                )
            ),
        }
        self.records.append(record)
        self._by_exception_id[id(exception)] = record
        return record

    def snapshot(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.records)

    def clear(self) -> None:
        self.records.clear()
        self._by_exception_id.clear()


class DeploymentProfiler:
    """Single-owner hierarchical profiler shared by a deployment shell tree."""

    def __init__(self, enabled: bool = False, *, history: int = 240) -> None:
        self.enabled = bool(enabled)
        self.history = deque(maxlen=max(1, int(history)))
        self.depth = 0
        self.sequence = 0
        self._events: list[dict[str, Any]] = []
        self._root_started_ns = 0
        self.error_buffer = DeploymentErrorBuffer(history)

    @property
    def exceptions(self):
        """Compatibility view over the root deployment's error buffer."""

        return self.error_buffer.records

    def begin_shell(self, path: str) -> tuple[int, bool] | None:
        if not self.enabled:
            return None
        root = self.depth == 0
        if root:
            self._events = []
            self._root_started_ns = time.perf_counter_ns()
        self.depth += 1
        return time.perf_counter_ns(), root

    def end_shell(
        self,
        path: str,
        token: tuple[int, bool] | None,
    ) -> None:
        if token is None:
            return
        started_ns, root = token
        self._events.append({
            "path": path,
            "section": "shell",
            "label": "total",
            "cpu_ms": (time.perf_counter_ns() - started_ns) / 1e6,
            "gpu_query": None,
            "dispatches": 0,
        })
        self.depth -= 1
        if root:
            self._finish_root()

    def record(
        self,
        *,
        path: str,
        section: str,
        label: str,
        cpu_ms: float,
        dispatches: int = 0,
        gpu_query: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        self._events.append({
            "path": path,
            "section": section,
            "label": label,
            "cpu_ms": float(cpu_ms),
            "gpu_query": gpu_query,
            "dispatches": int(dispatches),
        })

    def record_exception(
        self,
        exception: BaseException,
        *,
        path: str,
        phase: str,
        node_id: int | None = None,
        handled: bool = False,
    ) -> dict[str, Any]:
        """Retain a structured traceback from any graph/shell boundary."""

        return self.error_buffer.push(
            exception,
            path=path,
            phase=phase,
            node_id=node_id,
            handled=handled,
        )

    def _finish_root(self) -> None:
        from OpenGL import GL
        import ctypes

        rows: dict[tuple[str, str, str], dict[str, Any]] = {}
        for event in self._events:
            gpu_ms = 0.0
            query = event.pop("gpu_query")
            if query is not None:
                elapsed_ns = ctypes.c_uint64()
                GL.glGetQueryObjectui64v(
                    query,
                    GL.GL_QUERY_RESULT,
                    ctypes.byref(elapsed_ns),
                )
                gpu_ms = elapsed_ns.value / 1e6
                GL.glDeleteQueries(1, (query,))
            key = (
                event["path"],
                event["section"],
                event["label"],
            )
            row = rows.setdefault(key, {
                "path": event["path"],
                "section": event["section"],
                "label": event["label"],
                "calls": 0,
                "cpu_ms": 0.0,
                "gpu_ms": 0.0,
                "dispatches": 0,
            })
            row["calls"] += 1
            row["cpu_ms"] += event["cpu_ms"]
            row["gpu_ms"] += gpu_ms
            row["dispatches"] += event["dispatches"]
        self.sequence += 1
        self.history.append({
            "sequence": self.sequence,
            "total_ms": (
                time.perf_counter_ns() - self._root_started_ns
            ) / 1e6,
            "rows": tuple(rows.values()),
        })
        self._events = []

    def report(self) -> dict[str, Any]:
        if not self.history:
            return {
                "sequence": 0,
                "total_ms": 0.0,
                "rows": (),
                "exceptions": self.error_buffer.snapshot(),
            }
        return {
            **self.history[-1],
            "exceptions": self.error_buffer.snapshot(),
        }

    def summary(self, *, window: int = 60) -> dict[str, Any]:
        reports = list(self.history)[-max(1, int(window)):]
        if not reports:
            return {
                "frames": 0,
                "total_mean_ms": 0.0,
                "total_p95_ms": 0.0,
                "rows": (),
            }

        def percentile95(values):
            ordered = sorted(values)
            index = max(0, (95 * len(ordered) + 99) // 100 - 1)
            return ordered[min(index, len(ordered) - 1)]

        keys = {
            (row["path"], row["section"], row["label"])
            for report in reports
            for row in report["rows"]
        }
        rows = []
        for path, section, label in keys:
            matches = [
                next(
                    (
                        row for row in report["rows"]
                        if (
                            row["path"],
                            row["section"],
                            row["label"],
                        ) == (path, section, label)
                    ),
                    None,
                )
                for report in reports
            ]
            cpu = [row["cpu_ms"] if row else 0.0 for row in matches]
            gpu = [row["gpu_ms"] if row else 0.0 for row in matches]
            calls = [row["calls"] if row else 0 for row in matches]
            dispatches = [
                row["dispatches"] if row else 0 for row in matches
            ]
            rows.append({
                "path": path,
                "section": section,
                "label": label,
                "cpu_mean_ms": sum(cpu) / len(cpu),
                "cpu_p95_ms": percentile95(cpu),
                "gpu_mean_ms": sum(gpu) / len(gpu),
                "gpu_p95_ms": percentile95(gpu),
                "calls_mean": sum(calls) / len(calls),
                "dispatches_mean": sum(dispatches) / len(dispatches),
            })
        totals = [report["total_ms"] for report in reports]
        return {
            "frames": len(reports),
            "total_mean_ms": sum(totals) / len(totals),
            "total_p95_ms": percentile95(totals),
            "rows": tuple(rows),
        }


def _shell_profile_name(shell: Any) -> str:
    metadata = shell.process_graph.G.graph
    return str(
        metadata.get("function_name")
        or metadata.get("program_name")
        or "module"
    )


def _attach_profiler(
    shell: Any,
    profiler: DeploymentProfiler,
    path: str,
    *,
    visited: set[int] | None = None,
) -> None:
    visited = set() if visited is None else visited
    if id(shell) in visited:
        return
    visited.add(id(shell))
    shell._profiler = profiler
    shell.error_buffer = profiler.error_buffer
    shell.profile_path = path
    for ephemeral in getattr(shell, "ephemeral_callables", ()):
        ephemeral.error_buffer = profiler.error_buffer
        ephemeral.compiler.error_buffer = profiler.error_buffer
    for reference, child in shell.function_shells.items():
        if id(child) in visited:
            continue
        _attach_profiler(
            child,
            profiler,
            f"{path}/{_shell_profile_name(child)}@{reference}",
            visited=visited,
        )


@contextmanager
def _profile_event(
    shell: Any,
    section: str,
    label: str,
    *,
    gpu: bool = False,
):
    profiler = shell._profiler
    if not profiler.enabled:
        yield
        return

    query = None
    if gpu:
        from OpenGL import GL

        generated = GL.glGenQueries(1)
        try:
            query = int(generated[0])
        except (IndexError, TypeError):
            query = int(generated)
        GL.glBeginQuery(GL.GL_TIME_ELAPSED, query)
    before_dispatches = dispatch_stats()["calls"]
    started_ns = time.perf_counter_ns()
    try:
        yield
    finally:
        if query is not None:
            from OpenGL import GL

            GL.glEndQuery(GL.GL_TIME_ELAPSED)
        profiler.record(
            path=shell.profile_path,
            section=section,
            label=label,
            cpu_ms=(time.perf_counter_ns() - started_ns) / 1e6,
            dispatches=(
                dispatch_stats()["calls"] - before_dispatches
            ),
            gpu_query=query,
        )


def _is_ast_metadata_node(graph: Any, node_id: int) -> bool:
    """Return whether a structural AST node carries no runtime value."""

    data = graph.G.nodes[node_id]
    node_type = data.get("type")
    return (
        node_type in {"str", "NoneType"}
        or isinstance(
            data.get("expr_obj"),
            (
                ast.Module,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.arguments,
                ast.arg,
                ast.keyword,
            ),
        )
    )


def _is_dispatch_metadata_node(graph: Any, node_id: int) -> bool:
    """Return whether a node routes syntax but performs no computation."""

    data = graph.G.nodes[node_id]
    node_type = str(data.get("type"))
    expression = data.get("expr_obj")
    parents = tuple(data.get("parents") or ())
    indexed_base = next(
        (
            parent
            for parent, role in parents
            if str(role) == "base"
        ),
        None,
    )
    python_routing_index = (
        node_type == "Indexed"
        and indexed_base in graph.G
        and (
            str(data.get("label", "")).startswith("unpack[")
            or graph.G.nodes[indexed_base].get("type")
            in {"Tuple", "List", "Set", "Dict", "Attribute"}
        )
    )
    compares_none = (
        isinstance(expression, ast.Compare)
        and any(
            parent in graph.G
            and graph.G.nodes[parent].get("type")
            in {"Const", "const", "Constant"}
            and graph.G.nodes[parent].get("constant") is None
            for parent, _role in parents
        )
    )
    def is_scalar_value(candidate: int, visiting=frozenset()) -> bool:
        if candidate in visiting or candidate not in graph.G:
            return False
        candidate_data = graph.G.nodes[candidate]
        candidate_expression = candidate_data.get("expr_obj")
        candidate_type = str(candidate_data.get("type"))
        if candidate_type in {"Const", "const", "Constant"}:
            return True
        if candidate_type in {"Input", "input"}:
            return (
                (candidate_data.get("attributes") or {}).get("value_kind")
                == "scalar"
            )
        if not isinstance(
            candidate_expression,
            (ast.BinOp, ast.UnaryOp, ast.Compare, ast.IfExp),
        ):
            return False
        candidate_parents = tuple(
            candidate_data.get("parents") or ()
        )
        return bool(candidate_parents) and all(
            is_scalar_value(
                parent,
                visiting | {candidate},
            )
            for parent, _role in candidate_parents
        )

    static_scalar_expression = (
        isinstance(expression, (ast.BinOp, ast.UnaryOp, ast.Compare))
        and bool(parents)
        and all(
            is_scalar_value(parent)
            for parent, _role in parents
        )
    )
    coordinator_bit_shift = (
        isinstance(expression, ast.BinOp)
        and isinstance(expression.op, (ast.LShift, ast.RShift))
    )
    chained_comparison = (
        isinstance(expression, ast.Compare)
        and len(expression.ops) > 1
    )
    return (
        node_type in {
            "Input",
            "input",
            "Const",
            "const",
            "Constant",
            "Store",
            "store",
            "Output",
            "output",
            "Return",
            "return",
            "Call",
            "no_grad",
        }
        or
        (
            isinstance(data.get("expr_obj"), ast.Call)
            and isinstance(data["expr_obj"].func, ast.Attribute)
            and node_type not in abstract_tensor_funcs
        )
        or
        bool(
            (data.get("attributes") or {}).get(
                "contextual_requirement"
            )
        )
        or
        _is_ast_metadata_node(graph, node_id)
        or isinstance(
            expression,
            (
                ast.expr_context,
                ast.operator,
                ast.unaryop,
                ast.boolop,
                ast.cmpop,
                ast.Slice,
                ast.With,
                ast.withitem,
                ast.If,
                ast.Raise,
                ast.Try,
                ast.ExceptHandler,
                ast.For,
                ast.While,
                ast.comprehension,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
                ast.JoinedStr,
                ast.FormattedValue,
                ast.IfExp,
                ast.BoolOp,
                ast.Starred,
                ast.Tuple,
                ast.List,
                ast.Set,
                ast.Dict,
            ),
        )
        or isinstance(expression, ast.Attribute)
        or python_routing_index
        or compares_none
        or static_scalar_expression
        or coordinator_bit_shift
        or chained_comparison
        or (
            data.get("type") == "Load"
            and (data.get("attributes") or {}).get("source_type") == "Name"
        )
    )


def _dispatch_subgraph(graph: Any, node_ids: tuple[int, ...]) -> Any:
    """Return the planned dispatch as an independent ProcessGraph subgraph."""

    selected = set(node_ids)
    # Container literals are routing, not dispatches of their own.  When one
    # feeds a numerical region, inline the container and expose its leaves as
    # the region boundary so a captured tensor operation retains each feed's
    # independent identity.
    pending = list(selected)
    while pending:
        child = pending.pop()
        for parent in graph.G.predecessors(child):
            if parent in selected:
                continue
            expression = graph.G.nodes[parent].get("expr_obj")
            if isinstance(expression, (ast.Tuple, ast.List)):
                selected.add(parent)
                pending.append(parent)
    boundary = {
        parent
        for node_id in selected
        for parent in graph.G.predecessors(node_id)
        if parent not in selected
        and not _is_ast_metadata_node(graph, parent)
    }
    included = selected | boundary
    subgraph = extract_clean_process_subgraph(graph, included)

    for node_id in boundary:
        data = subgraph.G.nodes[node_id]
        if str(data.get("type")) in {"Const", "const", "Constant"}:
            continue
        data["type"] = "Input"
        data["op"] = "input"
        data["label"] = f"value_{node_id}"
        data["parents"] = []
        for parent in tuple(subgraph.G.predecessors(node_id)):
            subgraph.G.remove_edge(parent, node_id)

    source_levels = graph.levels
    minimum_level = min(
        (int(source_levels.get(node_id, 0)) for node_id in included),
        default=0,
    )
    subgraph.levels = {
        node_id: int(source_levels.get(node_id, 0)) - minimum_level
        for node_id in included
    }
    subgraph.roots = [
        node_id
        for node_id in node_ids
        if (
            graph.G.out_degree(node_id) == 0
            or any(
                child not in selected
                for child in graph.G.successors(node_id)
            )
        )
    ]
    deployment_outputs = tuple(subgraph.roots)
    next_node_id = max(
        (node_id for node_id in graph.G if isinstance(node_id, int)),
        default=0,
    ) + 1
    output_level = max(subgraph.levels.values(), default=0) + 1
    store_nodes = []
    for output_id in deployment_outputs:
        while next_node_id in subgraph.G:
            next_node_id += 1
        store_id = next_node_id
        next_node_id += 1
        subgraph.G.add_node(
            store_id,
            type="Store",
            op="store",
            label=f"value_{output_id}",
            parents=[(output_id, "value")],
            children=[],
        )
        subgraph.G.add_edge(output_id, store_id)
        subgraph.G.nodes[output_id].setdefault("children", []).append(
            (store_id, "value")
        )
        subgraph.levels[store_id] = output_level
        store_nodes.append(store_id)
    subgraph.roots = store_nodes
    subgraph.G.graph["deployment_inputs"] = tuple(
        node_id
        for node_id, data in subgraph.G.nodes(data=True)
        if data.get("type") == "Input"
    )
    subgraph.G.graph["deployment_outputs"] = deployment_outputs
    subgraph.G.graph["deployment_store_nodes"] = tuple(store_nodes)
    subgraph.G.graph["deployment_nodes"] = tuple(
        node_id for node_id in node_ids if node_id in subgraph.G
    )
    return subgraph


def _compiler_input_name(label: str) -> str:
    """Mirror the established GraphDeepCompiler input naming convention."""

    import re

    name = str(label).strip()
    lowered = name.lower()
    match = re.match(r"([a-zA-Z]+)[_\d]*$", lowered)
    root = match.group(1) if match else lowered
    if root in {"i", "j", "k", "l", "m", "n"}:
        prefix = "int"
    elif (
        lowered.startswith("num")
        or lowered.endswith("idx")
        or lowered.isdigit()
    ):
        prefix = "int"
    elif lowered.startswith("is_") or lowered.startswith("has_"):
        prefix = "bool"
    else:
        prefix = "float"
    return f"{prefix}{name}"


def _bind_capture_tape(value: Any, tape: Any) -> Any:
    """Wrap resident tensor storage in the tape for the current region."""

    if not isinstance(value, AbstractTensor):
        return value
    rebound = type(value)(
        track_time=value.track_time,
        requires_grad=value.requires_grad,
        tape=tape,
    )
    rebound.data = value.data
    return rebound


def _remap_captured_program(
    captured: CapturedFusedProgram,
    *,
    feed_ids: dict[int, int],
    output_ids: tuple[int, ...],
) -> CapturedFusedProgram:
    """Replace transient tensor identities with ProcessGraph boundary IDs."""

    program = captured.program
    captured_outputs = tuple(program.outputs.values())
    if len(captured_outputs) != len(output_ids):
        raise ValueError(
            "captured terminal count does not match planned outputs"
        )
    id_map = dict(feed_ids)
    id_map.update(zip(captured_outputs, output_ids))
    remap = lambda value_id: id_map.get(value_id, value_id)
    remapped = FusedProgram(
        version=program.version,
        feeds={remap(value_id) for value_id in program.feeds},
        steps=[
            OpStep(
                step_id=step.step_id,
                op_name=step.op_name,
                input_ids=[remap(value_id) for value_id in step.input_ids],
                attrs=dict(step.attrs),
                result_id=remap(step.result_id),
                mode_sensitive=step.mode_sensitive,
                level=step.level,
            )
            for step in program.steps
        ],
        outputs={
            f"value_{output_id}": output_id
            for output_id in output_ids
        },
        state_in=(
            None
            if program.state_in is None
            else {remap(value_id) for value_id in program.state_in}
        ),
        meta={
            remap(value_id): meta
            for value_id, meta in (program.meta or {}).items()
        },
        extras=program.extras,
    )
    if hasattr(program, "glsl_linear_output_shape"):
        remapped.glsl_linear_output_shape = tuple(
            program.glsl_linear_output_shape
        )
    return CapturedFusedProgram(
        remapped,
        {
            remap(value_id): value
            for value_id, value in captured.feeds.items()
        },
    )


def _constant_value(data: dict[str, Any]) -> Any:
    """Read one normalized literal without confusing a real ``None``."""

    expression = data.get("expr_obj")
    if isinstance(expression, ast.Constant):
        return expression.value
    if "constant" in data:
        return data["constant"]
    attributes = data.get("attributes") or {}
    if "value" in attributes:
        return attributes["value"]
    raise KeyError("constant ProcessGraph node has no literal payload")


def _call_arguments(
    parents: tuple[tuple[int, str], ...],
    values: dict[int, Any],
    static_arguments: dict[str, Any] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Reconstruct positional and keyword arguments from ProcessGraph roles."""

    positional: dict[int, Any] = {}
    keywords: dict[str, Any] = {}
    fallback_index = 1 << 30
    for parent, role_value in parents:
        role = str(role_value)
        if role.startswith("kw:"):
            keywords[role[3:]] = values[parent]
            continue
        if role in {"operand", "func", "callee"}:
            continue
        index = fallback_index
        if role.startswith("arg:"):
            index = int(role[4:])
        elif role.startswith("arg") and role[3:].isdigit():
            index = int(role[3:])
        elif role in {"arg", "args"}:
            index = len(positional)
        else:
            continue
        positional[index] = values[parent]
        fallback_index += 1
    for role, value in (static_arguments or {}).items():
        if role.startswith("kw:"):
            keywords[role[3:]] = value
        elif role.startswith("arg:"):
            positional[int(role[4:])] = value
    return [positional[index] for index in sorted(positional)], keywords


def _static_python_value(bindings: dict[str, Any], path: str) -> Any:
    """Resolve one reducer-retained Python reference for coordination."""

    parts = str(path).split(".")
    try:
        value = bindings[parts[0]]
    except KeyError as exc:
        raise KeyError(
            f"static Python reference {path!r} has no retained binding"
        ) from exc
    for part in parts[1:]:
        value = getattr(value, part)
    return value


def _tensorize_graph_input(value: Any, *, device: Any) -> Any:
    """Move array-shaped public inputs onto the selected AbstractTensor backend."""

    if isinstance(value, AbstractTensor):
        return value
    shape = getattr(value, "shape", None)
    if shape is None:
        return value
    return AbstractTensor.tensor(value, device=device)


def _coordinate_scheduled_capture_impl(
    shell: Any,
    initial_values: dict[str | int, Any],
    *,
    device: Any = None,
    capture: bool = True,
) -> Any:
    """Execute structural boundaries and capture each planned numeric region.

    This is the first-invocation coordinator for a function deployment shell.
    Python objects remain structural values; array-shaped public inputs become
    resident tensors.  External calls and control/container nodes run between
    topologically closed numerical regions, while those regions alone enter
    forward capture for backend compilation.
    """

    graph = shell.process_graph
    supplied = dict(initial_values)
    values: dict[int, Any] = {
        int(key): value
        for key, value in supplied.items()
        if isinstance(key, int)
    }
    for node_id, data in graph.G.nodes(data=True):
        if str(data.get("type")) not in {"Input", "input"}:
            continue
        name = str(
            (data.get("attributes") or {}).get(
                "binding_name",
                data.get("label", ""),
            )
        )
        if node_id not in values:
            binding_kind = (
                (data.get("attributes") or {}).get("binding_kind")
            )
            if binding_kind == "loop":
                continue
            if (
                binding_kind == "external"
                and name in shell.static_python_bindings
            ):
                values[node_id] = shell.static_python_bindings[name]
                continue
            if name not in supplied:
                raise KeyError(f"missing ProcessGraph input {name!r}")
            values[node_id] = supplied[name]

    regions = tuple(
        zip(
            shell.deep_compilers,
            shell.dispatch_subgraphs,
            shell.ephemeral_callables,
        )
    )
    region_for_node: dict[int, int] = {}
    for region_index, (_compiler, subgraph, _ephemeral) in enumerate(regions):
        if region_index in shell.coordinator_region_indices:
            continue
        for node_id in subgraph.G.graph.get("deployment_nodes", ()):
            if node_id in region_for_node:
                raise RuntimeError(
                    f"ProcessGraph node {node_id} belongs to two dispatches"
                )
            region_for_node[node_id] = region_index

    controlled_nodes = {
        parent
        for _node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.If)
        for parent, role in data.get("parents", ())
        if str(role) in {"body", "orelse"}
    }
    expression_nodes = {
        id(data.get("expr_obj")): node_id
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.AST)
    }
    for _control_id, data in graph.G.nodes(data=True):
        expression = data.get("expr_obj")
        if isinstance(expression, ast.If):
            controlled_statements = (
                *expression.body,
                *expression.orelse,
            )
        elif isinstance(expression, ast.Try):
            controlled_statements = (
                *expression.body,
                *expression.handlers,
                *expression.orelse,
                *expression.finalbody,
            )
        else:
            continue
        for statement in controlled_statements:
            for member in ast.walk(statement):
                controlled = expression_nodes.get(id(member))
                if controlled is not None:
                    controlled_nodes.add(controlled)
    controlled_nodes.update(
        node_id
        for node_id, data in graph.G.nodes(data=True)
        if data.get("type") == "Input"
        and (
            data.get("attributes") or {}
        ).get("binding_kind") == "loop"
    )
    loop_plans_by_node = {
        plan.loop.node_id: plan for plan in shell.loop_plans
    }
    for plan in shell.loop_plans:
        controlled_nodes.add(plan.loop.node_id)
        controlled_nodes.update(plan.loop.body_nodes)
        controlled_nodes.update(
            binding for _name, binding in plan.loop.target_bindings
        )
        for successor in graph.G.successors(plan.loop.node_id):
            if isinstance(
                graph.G.nodes[successor].get("expr_obj"),
                (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
            ):
                controlled_nodes.update(nx.ancestors(graph.G, successor))
    completed_regions: set[int] = set()
    active_nodes: set[int] = set()
    active_exceptions: dict[int, BaseException] = {}
    tapes = []
    feed_maps = []
    captured_subgraphs = []
    captured_compilers = []
    captured_region_indices = []

    def evaluate_region(region_index: int) -> None:
        if region_index in completed_regions:
            return
        compiler, subgraph, ephemeral = regions[region_index]
        inputs: dict[str, Any] = {}
        for input_id in subgraph.G.graph["deployment_inputs"]:
            input_value = evaluate_node(input_id)
            name = _compiler_input_name(
                subgraph.G.nodes[input_id]["label"]
            )
            inputs[name] = input_value

        operations = [
            str(subgraph.G.nodes[node_id].get("op") or
                subgraph.G.nodes[node_id].get("type"))
            for node_id in subgraph.G.graph.get("deployment_nodes", ())
        ]
        label = f"region {region_index}: " + " -> ".join(operations)

        def run_region() -> None:
            captured_program = shell.captured_region_programs.get(region_index)
            if captured_program is not None:
                chunks = execute_captured_fused_program(
                    captured_program,
                    {
                        input_id: evaluate_node(input_id)
                        for input_id in subgraph.G.graph["deployment_inputs"]
                    },
                )
                for output_id in subgraph.G.graph["deployment_outputs"]:
                    output = GLSLTensorOperations()
                    output.data = chunks[f"value_{output_id}"]
                    values[output_id] = output
                completed_regions.add(region_index)
                return

            function = (
                shell.compiled_dispatch_functions[region_index]
                if shell.compiled_dispatch_functions
                else ephemeral
            )
            if capture:
                capture_context = autograd.forward_capture()
            else:
                capture_context = autograd.no_grad()
            with capture_context as tape:
                bound_inputs = {
                    name: _bind_capture_tape(value, tape)
                    for name, value in inputs.items()
                }
                try:
                    results = function(**bound_inputs)
                except Exception as error:
                    raise RuntimeError(
                        "ProcessGraph numerical region failed in "
                        f"{graph.G.graph.get('function_name', '?')} "
                        f"region {region_index}"
                    ) from error
            if not isinstance(results, tuple):
                results = (results,)
            returned = dict(zip(compiler._outs, results))
            for output_id, store_id in zip(
                subgraph.G.graph["deployment_outputs"],
                subgraph.G.graph["deployment_store_nodes"],
            ):
                values[output_id] = returned[store_id]

            completed_regions.add(region_index)
            if capture and tape._nodes:
                tapes.append(tape)
                feed_maps.append(
                    {
                        id(bound_inputs[
                            _compiler_input_name(
                                subgraph.G.nodes[input_id]["label"]
                            )
                        ]): input_id
                        for input_id in subgraph.G.graph[
                            "deployment_inputs"
                        ]
                        if isinstance(
                            bound_inputs[
                                _compiler_input_name(
                                    subgraph.G.nodes[input_id]["label"]
                                )
                            ],
                            AbstractTensor,
                        )
                    }
                )
                captured_subgraphs.append(subgraph)
                captured_compilers.append(compiler)
                captured_region_indices.append(region_index)

        with _profile_event(
            shell,
            "capture" if capture else "dispatch",
            label,
            gpu=True,
        ):
            run_region()

    def evaluate_node(node_id: int) -> Any:
        if node_id in values:
            return values[node_id]
        if node_id in active_nodes:
            raise RuntimeError(
                f"recursive runtime dependency at ProcessGraph node {node_id}"
            )
        active_nodes.add(node_id)
        try:
            region_index = region_for_node.get(node_id)
            if region_index is not None:
                evaluate_region(region_index)
                return values[node_id]

            data = graph.G.nodes[node_id]
            node_type = str(data.get("type"))
            expression = data.get("expr_obj")
            parents = tuple(data.get("parents") or ())

            if node_type in {"Input", "input"}:
                name = str(data.get("label", node_id))
                raise KeyError(
                    f"missing ProcessGraph input {name!r} in "
                    f"{graph.G.graph.get('function_name', '?')}"
                )
            if node_type in {"Const", "const", "Constant"}:
                result = _constant_value(data)
            elif isinstance(expression, (ast.Tuple, ast.List, ast.Set)):
                items = [
                    evaluate_node(parent)
                    for parent, _role in parents
                ]
                if isinstance(expression, ast.Tuple):
                    result = tuple(items)
                elif isinstance(expression, ast.Set):
                    result = set(items)
                else:
                    result = items
            elif isinstance(expression, ast.Dict):
                keys = [
                    evaluate_node(parent)
                    for parent, role in parents
                    if str(role) == "keys"
                ]
                items = [
                    evaluate_node(parent)
                    for parent, role in parents
                    if str(role) == "values"
                ]
                result = dict(zip(keys, items))
            elif isinstance(expression, ast.Attribute):
                parent = next(
                    parent
                    for parent, role in parents
                    if str(role) == "value"
                )
                result = getattr(evaluate_node(parent), expression.attr)
            elif isinstance(expression, ast.Slice):
                parts = {
                    str(role): evaluate_node(parent)
                    for parent, role in parents
                }
                result = slice(
                    parts.get("lower"),
                    parts.get("upper"),
                    parts.get("step"),
                )
            elif isinstance(expression, ast.Starred):
                result = evaluate_node(parents[0][0])
            elif isinstance(expression, ast.FormattedValue):
                value = evaluate_node(parents[0][0])
                conversion = expression.conversion
                if conversion == ord("r"):
                    value = repr(value)
                elif conversion == ord("a"):
                    value = ascii(value)
                elif conversion == ord("s"):
                    value = str(value)
                result = format(value)
            elif isinstance(expression, ast.JoinedStr):
                result = "".join(
                    str(evaluate_node(parent))
                    for parent, _role in parents
                )
            elif node_type == "Indexed":
                base = evaluate_node(
                    next(
                        parent
                        for parent, role in parents
                        if str(role) == "base"
                    )
                )
                indices = [
                    evaluate_node(parent)
                    for parent, role in parents
                    if str(role) == "index"
                ]
                index = indices[0] if len(indices) == 1 else tuple(indices)
                result = base[index]
            elif isinstance(expression, ast.BoolOp):
                if isinstance(expression.op, ast.And):
                    result = True
                    for parent, _role in parents:
                        result = evaluate_node(parent)
                        if not result:
                            break
                else:
                    result = False
                    for parent, _role in parents:
                        result = evaluate_node(parent)
                        if result:
                            break
            elif isinstance(expression, ast.IfExp):
                by_role = {
                    str(role): parent for parent, role in parents
                }
                branch = "body" if bool(
                    evaluate_node(by_role["test"])
                ) else "orelse"
                result = evaluate_node(by_role[branch])
            elif isinstance(
                expression,
                (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
            ):
                generator_nodes = [
                    parent
                    for parent, role in parents
                    if str(role) == "generators"
                ]
                element_nodes = {
                    str(role): parent
                    for parent, role in parents
                    if str(role) in {"elt", "key", "value"}
                }
                collected = []

                def evaluate_generator(index: int) -> None:
                    if index == len(generator_nodes):
                        if isinstance(expression, ast.DictComp):
                            collected.append((
                                evaluate_node(element_nodes["key"]),
                                evaluate_node(element_nodes["value"]),
                            ))
                        else:
                            collected.append(
                                evaluate_node(element_nodes["elt"])
                            )
                        return
                    generator_node = generator_nodes[index]
                    plan = loop_plans_by_node[generator_node]
                    loop = plan.loop
                    iterator = iter(evaluate_node(loop.iterable_node))
                    for item in iterator:
                        targets = loop.target_bindings
                        items = (
                            tuple(item)
                            if len(targets) > 1
                            else (item,)
                        )
                        invalidated = set()
                        for (_name, binding), value in zip(targets, items):
                            values[binding] = value
                            invalidated.update(
                                nx.descendants(graph.G, binding)
                            )
                        invalidated.discard(node_id)
                        for dependent in invalidated:
                            values.pop(dependent, None)
                        for region_index in shell.loop_region_indices.get(
                            generator_node,
                            (),
                        ):
                            completed_regions.discard(region_index)
                        if all(
                            bool(evaluate_node(condition))
                            for condition in loop.condition_nodes
                        ):
                            evaluate_generator(index + 1)

                evaluate_generator(0)
                if isinstance(expression, ast.DictComp):
                    result = dict(collected)
                elif isinstance(expression, ast.SetComp):
                    result = set(collected)
                else:
                    result = collected
            elif isinstance(expression, ast.Compare):
                operands = [
                    evaluate_node(parent)
                    for parent, _role in parents
                ]
                result = True
                for index, comparison in enumerate(expression.ops):
                    comparator = {
                        ast.Is: operator.is_,
                        ast.IsNot: operator.is_not,
                        ast.Eq: operator.eq,
                        ast.NotEq: operator.ne,
                        ast.Lt: operator.lt,
                        ast.LtE: operator.le,
                        ast.Gt: operator.gt,
                        ast.GtE: operator.ge,
                    }.get(type(comparison))
                    if comparator is None:
                        raise NotImplementedError(
                            "unsupported structural comparison "
                            f"{type(comparison).__name__}"
                        )
                    if not comparator(
                        operands[index],
                        operands[index + 1],
                    ):
                        result = False
                        break
            elif isinstance(expression, ast.UnaryOp):
                operand = evaluate_node(parents[0][0])
                unary = {
                    ast.USub: operator.neg,
                    ast.UAdd: operator.pos,
                    ast.Not: operator.not_,
                    ast.Invert: operator.invert,
                }.get(type(expression.op))
                if unary is None:
                    raise NotImplementedError(
                        "unsupported structural unary operator "
                        f"{type(expression.op).__name__}"
                    )
                result = unary(operand)
            elif isinstance(expression, ast.BinOp):
                operands = [
                    evaluate_node(parent)
                    for parent, _role in parents
                ]
                binary = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                    ast.FloorDiv: operator.floordiv,
                    ast.Mod: operator.mod,
                    ast.Pow: operator.pow,
                    ast.LShift: operator.lshift,
                    ast.RShift: operator.rshift,
                    ast.BitOr: operator.or_,
                    ast.BitXor: operator.xor,
                    ast.BitAnd: operator.and_,
                    ast.MatMult: operator.matmul,
                }.get(type(expression.op))
                if binary is None:
                    raise NotImplementedError(
                        "unsupported structural binary operator "
                        f"{type(expression.op).__name__}"
                    )
                result = binary(operands[0], operands[1])
            elif isinstance(expression, ast.Call):
                for parent, _role in parents:
                    evaluate_node(parent)
                attributes = data.get("attributes") or {}
                static_arguments = {
                    role: _static_python_value(
                        shell.static_python_bindings,
                        reference,
                    )
                    for role, reference in (
                        attributes.get("static_call_arguments") or {}
                    ).items()
                }
                args, kwargs = _call_arguments(
                    parents,
                    values,
                    static_arguments,
                )
                external_ref = attributes.get("external_callee_ref")
                callee_ref = attributes.get("callee_ref")
                if external_ref is not None:
                    try:
                        external_label = (
                            shell.external_function_table
                            .entry(external_ref)
                            .qualified_name
                        )
                    except (AttributeError, KeyError):
                        external_label = str(external_ref)
                    with _profile_event(
                        shell,
                        "external",
                        external_label,
                        gpu=True,
                    ):
                        result = shell.call_external(
                            external_ref,
                            *args,
                            **kwargs,
                        )
                elif (
                    callee_ref is not None
                    and not attributes.get("static_python_reference")
                ):
                    nested = shell.function_shells[int(callee_ref)]
                    if nested.process_graph_boundary:
                        if nested.python_callable is None:
                            raise RuntimeError(
                                "host-boundary ProcessGraph function has no "
                                f"Python implementation: {callee_ref}"
                            )
                        result = nested.python_callable(*args, **kwargs)
                        values[node_id] = result
                        return result
                    nested_inputs = (
                        dict(zip(
                            nested.process_graph.G.graph.get(
                                "function_parameters",
                                (),
                            ),
                            args,
                        ))
                        | kwargs
                    )
                    caller_identities = graph.G.graph.get(
                        "identity_table",
                        {},
                    )
                    for _input_id, input_data in (
                        nested.process_graph.G.nodes(data=True)
                    ):
                        input_attributes = (
                            input_data.get("attributes") or {}
                        )
                        if (
                            input_data.get("type") != "Input"
                            or input_attributes.get("binding_kind")
                            != "external"
                        ):
                            continue
                        name = input_attributes.get("binding_name")
                        if name in nested_inputs:
                            continue
                        for identity in reversed(
                            caller_identities.get(name, ())
                        ):
                            try:
                                nested_inputs[name] = evaluate_node(identity)
                            except (KeyError, RuntimeError):
                                continue
                            break
                    for name, value in (
                        nested.process_graph.G.graph.get(
                            "parameter_defaults",
                            {},
                        ).items()
                    ):
                        nested_inputs.setdefault(name, value)
                    if nested.whole_program_compiled:
                        nested.execute_process_graph(nested_inputs)
                    else:
                        nested.coordinate_first_invocation(
                            nested_inputs,
                            device=device,
                        )
                    result = nested.last_result
                elif attributes.get("static_python_reference"):
                    callable_value = _static_python_value(
                        shell.static_python_bindings,
                        attributes["static_python_reference"],
                    )
                    result = callable_value(*args, **kwargs)
                elif isinstance(expression.func, ast.Attribute):
                    callable_parent = next(
                        (
                            parent
                            for parent, role in parents
                            if str(role) in {"func", "callee"}
                        ),
                        None,
                    )
                    if callable_parent is not None:
                        result = evaluate_node(callable_parent)(
                            *args,
                            **kwargs,
                        )
                    else:
                        receiver_parent = next(
                            (
                                parent
                                for parent, role in parents
                                if str(role) == "operand"
                            ),
                            None,
                        )
                        if receiver_parent is None:
                            raise RuntimeError(
                                "attribute call has neither a callable nor "
                                f"receiver edge at ProcessGraph node {node_id}: "
                                f"{ast.dump(expression, include_attributes=False)}; "
                                f"parents={parents!r}"
                            )
                        receiver = evaluate_node(receiver_parent)
                        result = getattr(
                            receiver,
                            expression.func.attr,
                        )(*args, **kwargs)
                else:
                    raise RuntimeError(
                        f"unresolved ProcessGraph Call node {node_id}"
                    )
            elif isinstance(expression, ast.If):
                test_id = next(
                    parent
                    for parent, role in parents
                    if str(role) == "test"
                )
                test = bool(evaluate_node(test_id))
                branch = "body" if test else "orelse"
                branch_nodes = [
                    parent
                    for parent, role in parents
                    if str(role) == branch
                ]
                result = None
                for branch_node in branch_nodes:
                    result = evaluate_node(branch_node)
            elif isinstance(expression, ast.Try):
                by_role: dict[str, list[int]] = {}
                for parent, role in parents:
                    by_role.setdefault(str(role), []).append(parent)
                result = None
                try:
                    for body_node in by_role.get("body", ()):
                        result = evaluate_node(body_node)
                except Exception as error:
                    selected_handler = None
                    for handler_id in by_role.get("handlers", ()):
                        handler_expression = graph.G.nodes[
                            handler_id
                        ].get("expr_obj")
                        if not isinstance(
                            handler_expression,
                            ast.ExceptHandler,
                        ):
                            continue
                        type_expression = handler_expression.type
                        if type_expression is None:
                            selected_handler = handler_id
                            break
                        names = (
                            tuple(type_expression.elts)
                            if isinstance(type_expression, ast.Tuple)
                            else (type_expression,)
                        )
                        exception_types = []
                        for name in names:
                            if isinstance(name, ast.Name):
                                candidate = shell.static_python_bindings.get(
                                    name.id,
                                    getattr(builtins, name.id, None),
                                )
                                if isinstance(candidate, type) and issubclass(
                                    candidate,
                                    BaseException,
                                ):
                                    exception_types.append(candidate)
                        if exception_types and isinstance(
                            error,
                            tuple(exception_types),
                        ):
                            selected_handler = handler_id
                            break
                    shell._profiler.record_exception(
                        error,
                        path=shell.profile_path,
                        phase="graph_try",
                        node_id=node_id,
                        handled=selected_handler is not None,
                    )
                    if selected_handler is None:
                        raise
                    active_exceptions[selected_handler] = error
                    try:
                        result = evaluate_node(selected_handler)
                    finally:
                        active_exceptions.pop(selected_handler, None)
                else:
                    for else_node in by_role.get("orelse", ()):
                        result = evaluate_node(else_node)
                finally:
                    for final_node in by_role.get("finalbody", ()):
                        result = evaluate_node(final_node)
            elif isinstance(expression, ast.ExceptHandler):
                error = active_exceptions.get(node_id)
                if error is None:
                    raise RuntimeError(
                        "ProcessGraph exception handler executed without "
                        f"an active exception at node {node_id}"
                    )
                if expression.name:
                    for binding in graph.G.graph.get(
                        "identity_table",
                        {},
                    ).get(expression.name, ()):
                        values[binding] = error
                result = None
                for parent, role in parents:
                    if str(role) == "body":
                        result = evaluate_node(parent)
            elif isinstance(expression, (ast.For, ast.While)):
                plan = loop_plans_by_node[node_id]
                loop = plan.loop
                if isinstance(expression, ast.For):
                    if loop.iterator_kind == "arithmetic_sequence":
                        by_role = {
                            str(role): parent for parent, role in parents
                        }
                        start = (
                            loop.start
                            if loop.start is not None
                            else evaluate_node(by_role["start"])
                        )
                        stop = (
                            loop.stop
                            if loop.stop is not None
                            else evaluate_node(by_role["stop"])
                        )
                        step = (
                            loop.step
                            if loop.step is not None
                            else evaluate_node(by_role["step"])
                        )
                        iterator = iter(range(
                            int(start),
                            int(stop),
                            int(step),
                        ))
                    else:
                        iterator = iter(evaluate_node(loop.iterable_node))
                else:
                    iterator = None

                result = None
                iterations_completed = 0
                while True:
                    if isinstance(expression, ast.For):
                        try:
                            item = next(iterator)
                        except StopIteration:
                            break
                        targets = loop.target_bindings
                        items = (
                            tuple(item)
                            if len(targets) > 1
                            else (item,)
                        )
                        for (_name, binding), value in zip(targets, items):
                            values[binding] = value
                    else:
                        test_parent = next(
                            parent
                            for parent, role in parents
                            if str(role) == "test"
                        )
                        if not bool(evaluate_node(test_parent)):
                            break

                    for body_node in loop.body_nodes:
                        values.pop(body_node, None)
                    for region_index in shell.loop_region_indices.get(
                        node_id,
                        (),
                    ):
                        completed_regions.discard(region_index)
                    for body_node in nx.topological_sort(
                        graph.G.subgraph(loop.body_nodes)
                    ):
                        result = evaluate_node(body_node)
                    for _name, initial, updated in loop.carried_bindings:
                        values[initial] = evaluate_node(updated)
                    iterations_completed += 1
                    if (
                        isinstance(expression, ast.While)
                        and iterations_completed > 1_000_000
                    ):
                        raise RuntimeError(
                            "ProcessGraph while loop exceeded safety limit"
                        )
            elif isinstance(expression, ast.Raise):
                exception = (
                    evaluate_node(parents[0][0])
                    if parents
                    else RuntimeError("ProcessGraph bare raise")
                )
                raise exception
            elif node_type in {"Return", "return", "Store", "store", "Output", "output"}:
                result = (
                    evaluate_node(parents[0][0])
                    if parents
                    else None
                )
            else:
                raise NotImplementedError(
                    "scheduled coordinator cannot resolve structural "
                    f"node {node_id} ({node_type})"
                )
            values[node_id] = result
            return result
        finally:
            active_nodes.remove(node_id)

    with AbstractTensor.use_backend("glsl", device):
        for node_id in nx.topological_sort(graph.G):
            if node_id in controlled_nodes:
                continue
            if node_id in values:
                values[node_id] = _tensorize_graph_input(
                    values[node_id],
                    device=device,
                )
            else:
                evaluate_node(node_id)

    if capture:
        shell.forward_tapes = tuple(tapes)
        shell.forward_feed_ids = tuple(feed_maps)
        shell.forward_subgraphs = tuple(captured_subgraphs)
        shell.forward_compilers = tuple(captured_compilers)
        shell.forward_region_indices = tuple(captured_region_indices)
    shell.captured_values = values
    roots = tuple(evaluate_node(node_id) for node_id in graph.roots)
    shell.last_result = roots[0] if len(roots) == 1 else roots
    return shell.last_result


def _coordinate_scheduled_capture(
    shell: Any,
    initial_values: dict[str | int, Any],
    *,
    device: Any = None,
    capture: bool = True,
) -> Any:
    """Profile-aware boundary around one scheduled shell invocation."""

    token = shell._profiler.begin_shell(shell.profile_path)
    try:
        return _coordinate_scheduled_capture_impl(
            shell,
            initial_values,
            device=device,
            capture=capture,
        )
    except Exception as error:
        shell._profiler.record_exception(
            error,
            path=shell.profile_path,
            phase="execution",
        )
        raise
    finally:
        shell._profiler.end_shell(shell.profile_path, token)


def _compile_whole_process_graph(
    shell: Any,
    *,
    device: Any = None,
    _visited: set[int] | None = None,
) -> Any:
    if getattr(shell, "process_graph_boundary", None):
        shell.whole_program_compiled = True
        shell.whole_program_device = device
        return shell
    """Prepare every planned dispatch and validate complete graph coverage.

    ``GraphDeepCompiler`` produces Python callables composed from the
    AbstractTensor operator table.  Selecting the GLSL backend therefore
    delegates their numerical operations to GLSL, but this stage does not
    claim that the complete graph has become one fused shader.
    """

    visited = set() if _visited is None else _visited
    identity = id(shell)
    if identity in visited:
        return shell
    visited.add(identity)

    covered = {
        node_id
        for subgraph in shell.dispatch_subgraphs
        for node_id in subgraph.G.graph.get("deployment_nodes", ())
    }
    uncovered = [
        node_id
        for node_id in shell.process_graph.G
        if node_id not in covered
        and not _is_dispatch_metadata_node(shell.process_graph, node_id)
    ]
    if uncovered:
        details = ", ".join(
            f"{node_id}:{shell.process_graph.G.nodes[node_id].get('type')}"
            for node_id in uncovered
        )
        raise RuntimeError(
            "whole ProcessGraph compilation has uncovered runtime nodes: "
            + details
        )

    functions = []
    sources = []
    for ephemeral in shell.ephemeral_callables:
        ephemeral.prepare(device=device)
        functions.append(ephemeral)
        sources.append(ephemeral.generated_source)
    direct_callees = {
        int(callee_ref)
        for _node_id, data in shell.process_graph.G.nodes(data=True)
        for callee_ref in (
            (data.get("attributes") or {}).get("callee_ref"),
        )
        if callee_ref is not None
    }
    for reference in sorted(direct_callees):
        function_shell = shell.function_shells.get(reference)
        if function_shell is None:
            raise RuntimeError(
                "ProcessGraph references a function with no deployment "
                f"shell: {reference}"
            )
        _compile_whole_process_graph(
            function_shell,
            device=device,
            _visited=visited,
        )
    shell.compiled_dispatch_functions = tuple(functions)
    shell.compiled_dispatch_sources = tuple(sources)
    shell.whole_program_compiled = True
    shell.whole_program_device = device
    return shell


def strategize_glsl_deployment(
    graph: Any,
    *,
    max_nodes_per_dispatch: int = 256,
    _function_table_stack: tuple[int, ...] = (),
) -> type:
    """Build a stateful shell around the graph's flat dispatch schedule."""

    loop_plans = LoopComposer(
        LoopBackendCapabilities(
            backend="glsl",
            native_for=True,
            native_while=True,
            dynamic_bounds=True,
            kpn=False,
        )
    ).compose(graph)
    function_entry = None
    function_reference = graph.G.graph.get("function_ref")
    function_table = getattr(graph, "function_table", None)
    if function_reference is not None and function_table is not None:
        function_entry = function_table.entry(function_reference)
    process_graph_boundary = (
        None
        if function_entry is None
        else function_entry.metadata.get("process_graph_boundary")
    )
    python_callable = (
        None if function_entry is None else function_entry.python_callable
    )
    reference_tables = build_shell_reference_tables(graph)
    dispatch_plan = serialize_scheduled_operator_dispatches(
        graph,
        max_nodes_per_dispatch=max_nodes_per_dispatch,
    )
    executable_dispatch_nodes = tuple(
        tuple(
            node_id
            for node_id in dispatch.node_ids
            if not _is_dispatch_metadata_node(graph, node_id)
        )
        for dispatch in dispatch_plan.dispatches
    )
    dispatch_subgraphs = tuple(
        _dispatch_subgraph(graph, node_ids)
        for node_ids in executable_dispatch_nodes
        if node_ids
    )
    loop_region_indices = {
        plan.loop.node_id: tuple(
            region_index
            for region_index, subgraph in enumerate(dispatch_subgraphs)
            if set(
                subgraph.G.graph.get("deployment_nodes", ())
            ).intersection(plan.loop.body_nodes)
        )
        for plan in loop_plans
    }
    deep_compilers = tuple(
        GraphDeepCompiler(
            subgraph,
            dict(abstract_tensor_funcs),
            abstract_tensor_sigs,
        )
        for subgraph in dispatch_subgraphs
    )
    ephemeral_callables = tuple(
        EphemeralProcessGraphCallable(
            subgraph,
            compiler=compiler,
            eager=False,
        )
        for subgraph, compiler in zip(
            dispatch_subgraphs,
            deep_compilers,
        )
    )

    class_ast = ast.parse(
        """
class ProcessGraphGLSLDeployment:
    process_graph = __process_graph__
    external_function_table = __external_function_table__
    static_python_bindings = __static_python_bindings__
    dispatch_plan = __dispatch_plan__
    loop_plans = __loop_plans__
    loop_region_indices = __loop_region_indices__
    process_graph_boundary = __process_graph_boundary__
    python_callable = __python_callable__
    dispatch_subgraphs = __dispatch_subgraphs__
    deep_compilers = __deep_compilers__
    ephemeral_callables = __ephemeral_callables__
    reference_table_template = __reference_tables__
    function_shell_types = {}
    source_node_count = __source_node_count__
    primitive_count = __primitive_count__
    loop_count = __loop_count__
    dispatch_count = __dispatch_count__

    def __init__(
        self,
        *,
        batch_size=None,
        profiling=False,
        input_slots=64,
        output_slots=64,
        **tuning,
    ):
        self.state = {}
        self.tuning = dict(tuning)
        self.batch_size = dict(
            __deployment_batches__
            if batch_size is None
            else batch_size
        )
        self.profiling = profiling
        self._profiler = DeploymentProfiler(profiling)
        self.error_buffer = self._profiler.error_buffer
        self.profile = self._profiler
        self.profile_path = _shell_profile_name(self)
        self.inputs = DeploymentFIFO(input_slots)
        self.outputs = DeploymentFIFO(output_slots)
        self.reference_tables = self.reference_table_template.copy()
        self.function_references = self.reference_tables.functions
        self.constant_references = self.reference_tables.constants
        self.memory_references = self.reference_tables.memory
        self.reference_correlations = self.reference_tables.correlations
        self.function_shells = {
            reference: shell_type(**tuning)
            for reference, shell_type in self.function_shell_types.items()
        }
        self._owns_function_shells = bool(self.function_shells)
        for function_shell in self.function_shells.values():
            function_shell.function_shells = self.function_shells
            function_shell._owns_function_shells = False
        _attach_profiler(
            self,
            self._profiler,
            self.profile_path,
        )
        self.fused_network = None
        self.whole_program_compiled = False
        self.whole_program_device = None
        self.compiled_dispatch_functions = ()
        self.compiled_dispatch_sources = ()
        self.forward_tapes = ()
        self.forward_feed_ids = ()
        self.forward_subgraphs = ()
        self.forward_compilers = ()
        self.forward_region_indices = ()
        self.compiled_tapes = ()
        self.compiled_region_indices = ()
        self.captured_region_programs = {}
        self.coordinator_region_indices = set()
        self.compile_failures = ()
        self.glsl_sources = ()
        self.input_bindings = {}
        self.output_bindings = {}

    @property
    def ready(self):
        return self.whole_program_compiled or self.fused_network is not None

    @property
    def programs(self):
        if self.fused_network is None:
            return ()
        return self.fused_network.programs

    def enable_profiling(self, enabled=True):
        self._profiler.enabled = bool(enabled)
        self.profiling = bool(enabled)
        return self

    def profile_report(self):
        return self._profiler.report()

    def exception_report(self):
        return self.error_buffer.snapshot()

    def exception_lines(self, *, limit=8):
        records = self.error_buffer.snapshot()
        if limit is not None:
            records = records[-max(0, int(limit)):]
        lines = []
        for record in records:
            lines.append(
                f"error {record['sequence']} | {record['path']} | "
                f"{record['phase']} | {record['exception_type']}: "
                f"{record['message']}"
            )
            lines.extend(
                f"  {line}"
                for line in record["traceback"].rstrip().splitlines()
            )
        return tuple(lines)

    def profile_summary(self, *, window=60):
        return self._profiler.summary(window=window)

    def profile_lines(self, *, limit=None, window=1):
        report = self.profile_summary(window=window)
        rows = sorted(
            report["rows"],
            key=lambda row: (
                row["gpu_mean_ms"],
                row["cpu_mean_ms"],
            ),
            reverse=True,
        )
        if limit is not None:
            rows = rows[:max(0, int(limit))]
        lines = [
            f"shell profile {report['frames']} invocation(s) | "
            f"total mean {report['total_mean_ms']:.3f} ms | "
            f"p95 {report['total_p95_ms']:.3f} ms"
        ]
        for row in rows:
            lines.append(
                f"  {row['path']} | {row['section']} | "
                f"{row['label']} | calls {row['calls_mean']:.1f} | "
                f"cpu {row['cpu_mean_ms']:.3f}/"
                f"{row['cpu_p95_ms']:.3f} ms mean/p95 | "
                f"gpu {row['gpu_mean_ms']:.3f}/"
                f"{row['gpu_p95_ms']:.3f} ms mean/p95 | "
                f"dispatches {row['dispatches_mean']:.1f}"
            )
        return tuple(lines)

    def require_ready(self):
        if not self.ready:
            raise RuntimeError(
                "GLSL deployment is planned but not installed or prepared; "
                "call compile_process_graph() for scheduled AbstractTensor "
                "execution or install fused programs"
            )
        return self

    def call_external(self, reference_or_name, *args, **kwargs):
        return self.external_function_table.invoke(
            reference_or_name,
            *args,
            **kwargs,
        )

    def capture_forward_tapes(
        self,
        dispatch_inputs,
        *,
        device=None,
    ):
        if len(dispatch_inputs) != len(self.deep_compilers):
            raise ValueError(
                "dispatch_inputs must provide one input mapping per "
                "planned subgraph"
            )
        tapes = []
        feed_maps = []
        with AbstractTensor.use_backend("glsl", device):
            for compiler, subgraph, inputs in zip(
                self.deep_compilers,
                self.dispatch_subgraphs,
                dispatch_inputs,
            ):
                function = compiler.build_function(device=device)
                with autograd.forward_capture() as tape:
                    bound_inputs = {
                        name: _bind_capture_tape(value, tape)
                        for name, value in dict(inputs).items()
                    }
                    function(**bound_inputs)
                tapes.append(tape)
                feed_maps.append({
                    id(bound_inputs[
                        _compiler_input_name(
                            subgraph.G.nodes[node_id]["label"]
                        )
                    ]): node_id
                    for node_id in subgraph.G.graph["deployment_inputs"]
                })
        self.forward_tapes = tuple(tapes)
        self.forward_feed_ids = tuple(feed_maps)
        self.forward_subgraphs = self.dispatch_subgraphs
        self.forward_compilers = self.deep_compilers
        self.forward_region_indices = tuple(range(len(self.forward_tapes)))
        return self.forward_tapes

    def capture_scheduled_forward_tapes(
        self,
        initial_values,
        *,
        device=None,
    ):
        tapes = []
        feed_maps = []
        with AbstractTensor.use_backend("glsl", device):
            values = {
                node_id: (
                    value
                    if isinstance(value, AbstractTensor)
                    else AbstractTensor.tensor(value, device=device)
                )
                for node_id, value in dict(initial_values).items()
            }
            for compiler, subgraph in zip(
                self.deep_compilers,
                self.dispatch_subgraphs,
            ):
                inputs = {}
                for node_id in subgraph.G.graph["deployment_inputs"]:
                    if node_id not in values:
                        raise KeyError(
                            "scheduled dispatch input has no routed value: "
                            f"{node_id}"
                        )
                    name = _compiler_input_name(
                        subgraph.G.nodes[node_id]["label"]
                    )
                    inputs[name] = values[node_id]

                function = compiler.build_function(device=device)
                with autograd.forward_capture() as tape:
                    bound_inputs = {
                        name: _bind_capture_tape(value, tape)
                        for name, value in inputs.items()
                    }
                    try:
                        results = function(**bound_inputs)
                    except Exception as error:
                        raise RuntimeError(
                            "ProcessGraph numerical region failed in "
                            f"{graph.G.graph.get('function_name', '?')} "
                            f"region {region_index}"
                        ) from error
                if not isinstance(results, tuple):
                    results = (results,)
                returned = dict(zip(compiler._outs, results))
                for output_id, store_id in zip(
                    subgraph.G.graph["deployment_outputs"],
                    subgraph.G.graph["deployment_store_nodes"],
                ):
                    values[output_id] = returned[store_id]

                tapes.append(tape)
                feed_maps.append({
                    id(bound_inputs[
                        _compiler_input_name(
                            subgraph.G.nodes[node_id]["label"]
                        )
                    ]): node_id
                    for node_id in subgraph.G.graph["deployment_inputs"]
                })
        self.forward_tapes = tuple(tapes)
        self.forward_feed_ids = tuple(feed_maps)
        self.forward_subgraphs = self.dispatch_subgraphs
        self.forward_compilers = self.deep_compilers
        self.forward_region_indices = tuple(range(len(self.forward_tapes)))
        self.captured_values = values
        return self.forward_tapes

    def coordinate_first_invocation(self, initial_values, *, device=None):
        return _coordinate_scheduled_capture(
            self,
            initial_values,
            device=device,
            capture=True,
        )

    def capture_fused_programs(self, initial_values, *, device=None):
        if not self.whole_program_compiled:
            self.compile_process_graph(device=device)
        result = _coordinate_scheduled_capture(
            self,
            initial_values,
            device=device,
            capture=True,
        )
        self.compile_forward_tapes(strict=True)
        captured = set(self.compiled_region_indices)
        missing = set(range(len(self.ephemeral_callables))) - captured
        for region_index in missing:
            subgraph = self.dispatch_subgraphs[region_index]
            tensor_outputs = [
                output_id
                for output_id in subgraph.G.graph["deployment_outputs"]
                if isinstance(self.captured_values.get(output_id), AbstractTensor)
            ]
            if tensor_outputs:
                raise RuntimeError(
                    "every tensor-producing ProcessGraph callable must become "
                    "one CapturedFusedProgram; region "
                    f"{region_index} left tensor outputs {tensor_outputs}"
                )
        self.coordinator_region_indices = missing
        self.captured_region_programs = dict(zip(
            self.compiled_region_indices,
            self.compiled_tapes,
        ))
        return result

    def compile_process_graph(self, *, device=None):
        try:
            return _compile_whole_process_graph(self, device=device)
        except Exception as error:
            self._profiler.record_exception(
                error,
                path=self.profile_path,
                phase="compile",
            )
            raise

    def execute_process_graph(self, feeds):
        if not self.whole_program_compiled:
            raise RuntimeError(
                "compile the whole ProcessGraph before executing it"
            )
        return _coordinate_scheduled_capture(
            self,
            feeds,
            device=self.whole_program_device,
            capture=False,
        )

    def compile_forward_tapes(
        self,
        *,
        dynamic_scalar_ids=(),
        strict=False,
    ):
        if not self.forward_tapes:
            raise RuntimeError("capture forward tapes before compiling them")
        if dynamic_scalar_ids and (
            len(dynamic_scalar_ids) != len(self.forward_tapes)
        ):
            raise ValueError(
                "dynamic_scalar_ids must provide one sequence per tape"
            )
        scalar_ids = (
            dynamic_scalar_ids
            if dynamic_scalar_ids
            else ((),) * len(self.forward_tapes)
        )
        compiled = []
        compiled_region_indices = []
        failures = []
        sources = []
        subgraphs = self.forward_subgraphs or self.dispatch_subgraphs
        region_indices = (
            self.forward_region_indices
            or tuple(range(len(self.forward_tapes)))
        )
        for tape, feed_ids, subgraph, dynamic_ids, region_index in zip(
            self.forward_tapes,
            self.forward_feed_ids,
            subgraphs,
            scalar_ids,
            region_indices,
        ):
            try:
                captured = compile_recorded_fused_tape(
                    tape,
                    dynamic_scalar_ids=tuple(dynamic_ids),
                )
            except ValueError as error:
                if strict:
                    raise RuntimeError(
                        "ProcessGraph numerical region "
                        f"{region_index} did not lower to one "
                        "CapturedFusedProgram"
                    ) from error
                failures.append({
                    "region_index": region_index,
                    "outputs": tuple(
                        subgraph.G.graph["deployment_outputs"]
                    ),
                    "unsupported_ops": (),
                    "reason": str(error),
                })
                continue
            captured = _remap_captured_program(
                captured,
                feed_ids=feed_ids,
                output_ids=tuple(
                    subgraph.G.graph["deployment_outputs"]
                ),
            )
            compiled.append(captured)
            compiled_region_indices.append(region_index)
            sources.append(compile_captured_fused_program(captured))
        self.compiled_tapes = tuple(compiled)
        self.compiled_region_indices = tuple(compiled_region_indices)
        self.compile_failures = tuple(failures)
        self.glsl_sources = tuple(sources)
        return self.glsl_sources

    def install_fused_programs(
        self,
        programs,
        *,
        fifo_slots=2,
        input_bindings=None,
        output_bindings=None,
    ):
        normalized = tuple(
            getattr(program, "program", program)
            for program in programs
        )
        if not normalized:
            raise ValueError("a deployment needs at least one fused program")
        if self.fused_network is not None:
            self.fused_network.release()
        self.fused_network = GLSLFusedProgramNetwork(
            normalized,
            fifo_slots=fifo_slots,
        )
        external_ids = {
            route.value_id
            for route in self.fused_network.routes
            if route.producer is None
        }
        inferred_inputs = {
            f"value_{value_id}": value_id
            for value_id in external_ids
        }
        for value_id in external_ids:
            if value_id not in self.process_graph.G:
                continue
            label = str(
                self.process_graph.G.nodes[value_id].get("label", "")
            )
            if label and label not in inferred_inputs:
                inferred_inputs[label] = value_id
        self.input_bindings = dict(
            inferred_inputs
            if input_bindings is None
            else input_bindings
        )
        if set(self.input_bindings.values()) != external_ids:
            self.fused_network.release()
            self.fused_network = None
            raise ValueError(
                "input bindings must cover every external fused value "
                "exactly once by value identity"
            )
        terminal_names = {
            route.output_name
            for route in self.fused_network.routes
            if route.consumer is None and route.output_name is not None
        }
        self.output_bindings = dict(
            {name: name for name in terminal_names}
            if output_bindings is None
            else output_bindings
        )
        unknown_outputs = set(self.output_bindings.values()) - terminal_names
        if unknown_outputs:
            self.fused_network.release()
            self.fused_network = None
            raise ValueError(
                "output bindings name unknown fused terminals: "
                + ", ".join(sorted(map(str, unknown_outputs)))
            )
        return self

    def install_compiled_tapes(
        self,
        *,
        fifo_slots=2,
        input_bindings=None,
        output_bindings=None,
    ):
        if not self.compiled_tapes:
            raise RuntimeError("compile forward tapes before installing them")
        return self.install_fused_programs(
            self.compiled_tapes,
            fifo_slots=fifo_slots,
            input_bindings=input_bindings,
            output_bindings=output_bindings,
        )

    def execute(self, feeds):
        if not self.ready:
            self.compile_process_graph()
        self.require_ready()
        if self.whole_program_compiled:
            return self.execute_process_graph(feeds)
        return self.fused_network.execute(feeds)

    def execute_named(self, feeds):
        if not self.ready:
            self.compile_process_graph()
        self.require_ready()
        if self.whole_program_compiled:
            result = self.execute_process_graph(feeds)
            values = result if isinstance(result, tuple) else (result,)
            names = tuple(
                self.process_graph.G.graph.get(
                    "function_outputs",
                    (),
                )
            )
            if not names:
                names = tuple(
                    f"result_{index}" for index in range(len(values))
                )
            if len(names) != len(values):
                raise RuntimeError(
                    "compiled ProcessGraph output names do not match "
                    "the returned values"
                )
            return dict(zip(names, values))
        missing = set(feeds) - set(self.input_bindings)
        if missing:
            raise KeyError(
                "unknown deployment inputs: "
                + ", ".join(sorted(map(str, missing)))
            )
        outputs = self.execute({
            self.input_bindings[name]: value
            for name, value in feeds.items()
        })
        return {
            public_name: outputs[network_name]
            for public_name, network_name in self.output_bindings.items()
        }

    def submit(self, feeds):
        if not self.inputs.publish(feeds):
            raise BufferError("GLSL deployment input FIFO is full")
        return self

    def run_pending(self):
        if self.outputs.unread >= self.outputs.slots:
            raise BufferError("GLSL deployment output FIFO is full")
        available, feeds = self.inputs.consume()
        if not available:
            return False
        outputs = self.execute_named(feeds)
        if not self.outputs.publish(outputs):
            raise BufferError("GLSL deployment output FIFO is full")
        return True

    def receive(self):
        return self.outputs.consume()

    def __call__(self, feeds):
        return self.execute_named(feeds)

    def release(self):
        if self.fused_network is not None:
            self.fused_network.release()
            self.fused_network = None
        if self._owns_function_shells:
            for shell in self.function_shells.values():
                shell.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.release()
        return False
"""
    )
    executable_node_ids = {
        node_id
        for node_ids in executable_dispatch_nodes
        for node_id in node_ids
    }
    node_locations = {
        node_id: location
        for node_id, location in dispatch_plan.node_locations.items()
        if node_id in executable_node_ids
    }
    namespace = {
        "__process_graph__": graph,
        "__external_function_table__": getattr(
            graph,
            "external_function_table",
            None,
        ),
        "__static_python_bindings__": dict(
            getattr(graph, "python_bindings", {}) or {}
        ),
        "__dispatch_plan__": dispatch_plan,
        "__loop_plans__": loop_plans,
        "__loop_region_indices__": loop_region_indices,
        "__process_graph_boundary__": process_graph_boundary,
        "__python_callable__": python_callable,
        "__dispatch_subgraphs__": dispatch_subgraphs,
        "__deep_compilers__": deep_compilers,
        "__ephemeral_callables__": ephemeral_callables,
        "__reference_tables__": reference_tables,
        "__source_node_count__": graph.G.number_of_nodes(),
        "__primitive_count__": sum(
            len(node_ids)
            for node_ids in executable_dispatch_nodes
        ),
        "__loop_count__": sum(
            1
            for _node_id, data in graph.G.nodes(data=True)
            if str(data.get("type")) in {"For", "AsyncFor", "While"}
        ),
        "__dispatch_count__": len(dispatch_subgraphs),
        "__deployment_batches__": node_locations,
        "DeploymentFIFO": DeploymentFIFO,
        "DeploymentProfiler": DeploymentProfiler,
        "GLSLFusedProgramNetwork": GLSLFusedProgramNetwork,
        "compile_recorded_fused_tape": (
            compile_recorded_fused_tape
        ),
        "emit_multi_output_program_source": (
            emit_multi_output_program_source
        ),
        "_compiler_input_name": _compiler_input_name,
        "_bind_capture_tape": _bind_capture_tape,
        "_remap_captured_program": _remap_captured_program,
        "_coordinate_scheduled_capture": _coordinate_scheduled_capture,
        "_compile_whole_process_graph": _compile_whole_process_graph,
        "_attach_profiler": _attach_profiler,
        "_shell_profile_name": _shell_profile_name,
        "execute_captured_fused_program": execute_captured_fused_program,
        "compile_captured_fused_program": compile_captured_fused_program,
        "GLSLTensorOperations": GLSLTensorOperations,
        "AbstractTensor": AbstractTensor,
        "autograd": autograd,
    }
    exec(
        compile(
            class_ast,
            filename="<glsl-deployment-strategy>",
            mode="exec",
        ),
        namespace,
    )
    deployment_class = namespace["ProcessGraphGLSLDeployment"]
    deployment_class.generated_ast = class_ast
    function_shell_types = {}
    function_table = getattr(graph, "function_table", None)
    table_identity = id(function_table)
    if (
        function_table is not None
        and table_identity not in _function_table_stack
    ):
        nested_stack = (*_function_table_stack, table_identity)
        for entry in function_table:
            if entry.graph is None:
                continue
            function_graph = extract_clean_process_subgraph(
                entry.graph,
                entry.graph.G,
            )
            function_shell_types[entry.reference.address] = (
                strategize_glsl_deployment(
                    function_graph,
                    max_nodes_per_dispatch=max_nodes_per_dispatch,
                    _function_table_stack=nested_stack,
                )
            )
    deployment_class.function_shell_types = function_shell_types
    return deployment_class


__all__ = [
    "DeploymentErrorBuffer",
    "DeploymentProfiler",
    "strategize_glsl_deployment",
]
