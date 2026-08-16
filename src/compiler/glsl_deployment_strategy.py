"""Deployment-strategy stage for ProcessGraphs.

Despite the module's filename, ``strategize_shell_deployment`` (formerly
``strategize_glsl_deployment``) is not GLSL-specific: it is the single
compilation choke point every backend passes through -- c, python, glsl,
fortran-via-precompile_only, webgl, and (once registered) webgpu all
funnel through this one control-planning stage before diverging into
backend-specific emission later in the pipeline. See
``accelerator_backends/aot_compile.py`` for the full pipeline this stage
sits in.
"""

from __future__ import annotations

import ast
import builtins
import concurrent.futures
import copy
import gc
import hashlib
import operator
import sys
import time
import traceback
from collections import deque
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping
from types import ModuleType

import networkx as nx
import numpy as np

from .deployment_fifo import DeploymentFIFO
from .control_source import (
    CallBlock,
    ControlExpression,
    ControlProgram,
    LoopBlock,
    ParallelDeployment,
    SequenceBlock,
    StatementBlock,
    StateMachineTick,
    StreamPublishBlock,
    ValidationBlock,
    WhileBlock,
    overlay_scheduled_control,
    project_control_regions,
)
from .hierarchical_control import compose_hierarchical_control
from .hierarchical_plan import (
    PlanCall,
    PlanClosure,
    PlanLine,
    assign_hierarchy_ids,
    reduce_hierarchy_identities,
)
from .loop_composer import (
    LoopBackendCapabilities,
    LoopComposer,
    LoopStrategy,
    _destructure_loop_target,
    analyze_shader_loop_reductions,
    bind_control_deployments_to_regions,
    evaporate_unrolled_loops,
    materialize_retained_loop_ports,
)
from .loop_ir import LoopStateEffectMode
from .process_graph_callable import EphemeralProcessGraphCallable
from .process_graph_fusion import (
    DispatchRegion,
    dispatch_region_to_fused_program,
    extract_clean_process_subgraph,
    reduce_scheduled_shader_regions,
)
from .shell_reference_tables import build_shell_reference_tables
from ..transmogrifier.function_table import FunctionReference
from ..common.tensors.abstraction import AbstractTensor, tensor_identity
from ..common.tensors.accelerator_backends.glsl_fused_network import (
    GLSLFusedProgramNetwork,
)
from ..common.tensors.accelerator_backends.c_primitive_program import (
    CapturedFusedProgram,
    compile_recorded_fused_tape,
)
from ..common.tensors.accelerator_backends.glsl_backend import (
    InstalledGLSLControlShell,
    build_control_shader_artifact,
    compile_captured_fused_program,
    compose_control_shader,
    dispatch_stats,
    emit_multi_output_program_source,
    execute_captured_fused_program,
)
from ..common.tensors.accelerator_backends.glsl_tensor_backend import (
    GLSLTensorOperations,
)
from ..common.tensors.accelerator_backends.precompile_observer_backend import (
    PrecompileObserverTensorOperations,
)
from ..common.tensors.autograd import GradTape, autograd
from ..common.tensors.fused_ir import (
    FusedProgram,
    Meta,
    OpStep,
    canonical_elementwise_op,
    ordered_feed_ids,
)
from ..common.tensors.operator_catalog import (
    ACCESSOR_OPERATORS,
    CREATION_OPERATORS,
)
from ..transmogrifier.graph.graph_deep_compiler import GraphDeepCompiler
from ..transmogrifier.operator_defs import (
    abstract_tensor_funcs,
    abstract_tensor_sigs,
)


def _dependency_order(graph: Any) -> tuple[int, ...]:
    """Return DAG order, or stable condensation order for retained loops.

    The result is cached on the graph, keyed by a cheap (node count, edge
    count) fingerprint. During deployment planning the same shell graph is
    dependency-ordered repeatedly (once per ``_build_shell_hierarchy_plan``,
    which runs on shell construction and on every ``refresh_hierarchy_plan``),
    yet it is only read there -- so re-running the topological sort each time
    was pure repeated work. The fingerprint invalidates the cache the moment
    the graph gains or loses a node/edge, so a genuinely mutated graph is
    re-ordered.
    """

    G = graph.G
    fingerprint = (G.number_of_nodes(), G.number_of_edges())
    cached = G.graph.get("_dependency_order_cache")
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    try:
        order = tuple(nx.lexicographical_topological_sort(
            G, key=lambda value_id: int(value_id)
        ))
    except nx.NetworkXUnfeasible:
        recursive = G.graph.get("recursion_table")
        if not recursive or set(graph.levels) != set(G):
            raise
        order = tuple(sorted(
            G,
            key=lambda node_id: (
                int(graph.levels[node_id]),
                int(node_id),
            ),
        ))
    G.graph["_dependency_order_cache"] = (fingerprint, order)
    return order


_scheduled_capture_backend: ContextVar[str] = ContextVar(
    "scheduled_capture_backend",
    default="glsl",
)


@contextmanager
def _use_scheduled_capture_backend(name: str):
    """Select the numerical observer used by one discovery traversal."""

    token = _scheduled_capture_backend.set(str(name))
    try:
        yield
    finally:
        _scheduled_capture_backend.reset(token)


_planned_capture_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "planned_process_graph_capture",
    default=None,
)


def _observe_process_graph_node(
    node_id: int,
    parent_ids: tuple[int, ...],
    parent_values: tuple[Any, ...],
    result: Any,
) -> Any:
    """Correlate one planned node with its immediate primitive occurrence.

    The ProcessGraph node ID is supplied by generated region code at the exact
    operation invocation.  The observer records only integer IDs and never
    compares payloads, wrappers, storage, shapes, or values.  It therefore
    cannot infer an alias or dependency; it only connects the already-planned
    operation to the primitive implementation occurrence revealed by the one
    forward capture.
    """

    context = _planned_capture_context.get()
    if context is None:
        return result
    if not isinstance(result, AbstractTensor):
        # This must be checked before the identity-aliasing logic below:
        # ``result is parent_value`` is only a meaningful alias signal for a
        # tensor, where object identity means shared storage.  For a plain
        # Python value it can be a pure CPython implementation detail --
        # small integers are interned, so ``0 + 1`` and the literal ``1``
        # operand satisfy ``is`` by cache coincidence, not because they are
        # the same computation.  Treating that as an alias would silently
        # drop the operation instead of recording it.
        #
        # Generated numeric-kernel code runs real Python operators, and this
        # observer is its only per-operation hook -- but everything below is
        # tape-primitive correlation, which exists to resolve which of
        # several possible dynamically-dispatched tensor primitives this
        # call produced.  A plain value has no such ambiguity: its own
        # graph node id already is its unambiguous identity, and it will
        # never enter ``tape._nodes`` no matter how long this waits.  Record
        # it directly, in exact execution order, the same way a reference
        # operator (SetAttr/GetAttr) already is.
        shell = context.get("shell")
        if shell is not None:
            shell.reference_operator_sequence.append(int(node_id))
        return result
    tape = context["tape"]
    capture_id = tensor_identity(result)
    aliased_parents = tuple(
        int(parent_id)
        for parent_id, parent_value in zip(parent_ids, parent_values)
        if result is parent_value
    )
    if len(aliased_parents) == 1 and int(node_id) != aliased_parents[0]:
        # The result is literally an existing explicit operand.  Its tape
        # entry therefore belongs to that pre-existing value; this operation
        # did not publish a new primitive result.  Record the ProcessGraph
        # endpoint as an SSA alias before primitive ownership is considered,
        # otherwise the same tape identity is incorrectly assigned both to
        # the call result and to its input socket.
        context["value_aliases"][int(node_id)] = aliased_parents[0]
        return result
    if capture_id in tape._nodes:
        primitive = tape._nodes[capture_id]
        self_parent_slots = tuple(
            int(slot)
            for primitive_id, slot in primitive.parents
            if int(primitive_id) == int(capture_id)
        )
        graph = context.get("graph")
        graph_node = (
            graph.G.nodes[int(node_id)]
            if graph is not None and int(node_id) in graph.G
            else {}
        )
        if (
            self_parent_slots
            and len(parent_ids) == 1
            and graph_node.get("type") not in {"Store", "IndexedStore"}
        ):
            # An identity-preserving tensor adapter can be recorded by the
            # tape under the same object ID as its operand.  Such a node is a
            # self-edge in object-keyed tape space, not a numerical kernel.
            # The ProcessGraph supplies the unambiguous SSA relationship.
            context["value_aliases"][int(node_id)] = int(parent_ids[0])
            return result
        existing_owner = primitive.ctx.get("process_graph_node_id")
        if existing_owner is not None:
            # A structural Store, call return, or other pass-through node may
            # observe the exact same tensor produced by an earlier operation.
            # That does not create a new primitive or a new value identity.
            # Producer ownership is single-assignment; the ProcessGraph's
            # explicit call/store binding carries the alias relationship.
            return result
        context["node_capture_ids"].setdefault(
            int(node_id), []
        ).append(int(capture_id))
        primitive.ctx["process_graph_node_id"] = int(node_id)
        tensor_op = (graph_node.get("attributes") or {}).get("tensor")
        if tensor_op in CREATION_OPERATORS and graph is not None:
            # The first positional argument is the size. Its role varies by
            # which ingestion path resolved this call -- "args" (generic
            # schema descent, taken here since a static class reference like
            # ``AbstractTensor.zeros`` doesn't go through the ordinary
            # receiver-argument rewiring), "arg0", or "arg:0" (other call
            # shapes) -- so match any arg-prefixed role instead of guessing
            # one spelling, and take the first by parent id for determinism.
            size_parent_id = min(
                (
                    int(parent)
                    for parent, role in (graph_node.get("parents") or ())
                    if str(role).lower().startswith("arg")
                ),
                default=None,
            )
            if (
                size_parent_id is not None
                and size_parent_id in graph.G
                and not _source_static_value(graph, size_parent_id)
            ):
                # The size argument is itself computed (not a literal), so
                # this result's shape is only what this one discovery trace
                # happened to observe, not a compile-time fact.  Record the
                # real origin (see Meta.shape_source_ids) instead of letting
                # it silently freeze into a fixed number downstream.
                dimensions = len(tuple(getattr(result, "shape", ()) or ()))
                primitive.ctx["shape_source_ids"] = tuple(
                    size_parent_id for _ in range(dimensions)
                )
        collection_owners = context.get("collection_owner_ids", frozenset())

        def collection_source(value_id: int) -> int | None:
            current = int(value_id)
            seen = set()
            while current not in seen:
                seen.add(current)
                if current in collection_owners:
                    return current
                if graph is None or current not in graph.G:
                    return None
                attributes = graph.G.nodes[current].get("attributes") or {}
                if attributes.get("producer_kind") not in {
                    "aggregate_materialization",
                    "loop_materialization",
                }:
                    return None
                sources = tuple(map(
                    int,
                    attributes.get("materialized_source_value_ids", ()),
                ))
                if len(sources) != 1:
                    return None
                current = sources[0]
            return None

        collection_parents = tuple(dict.fromkeys(
            owner
            for parent_id in parent_ids
            if (owner := collection_source(int(parent_id))) is not None
        ))
        params = primitive.ctx.get("params", {})
        raw_materialization_dim = params.get("dim", 0)
        materialization_dim = int(
            0
            if raw_materialization_dim is None
            else raw_materialization_dim
        )
        if (
            len(collection_parents) == 1
            and str(primitive.op) in {"stack", "cat", "concat"}
            and materialization_dim == 0
        ):
            # Only a planner-owned resident collection may cross this
            # producer boundary as a zero-copy view.  A Python list/tuple
            # observed during discovery proves nothing about storage
            # ownership.  Leading-axis stack/cat preserves the resident
            # collection's contiguous publication order; other axes require
            # a real numerical materialization producer.
            context["collection_materializations"][int(capture_id)] = (
                collection_parents[0]
            )
            return result
        # The generated ProcessGraph callsite supplies each parent ID beside
        # the exact Python value used for that operand.  Use those pairs
        # directly.  GradTape's parent list/slot convention is backend
        # bookkeeping and is not the ProcessGraph ABI.
        primitive_parents = tuple(sorted(
            primitive.parents,
            key=lambda parent: int(parent[1]),
        ))
        if primitive_parents:
            primitive_ids = {
                int(primitive_id)
                for primitive_id, _slot in primitive_parents
            }
            exact = []
            matched_graph_ids = set()
            matched_primitive_ids = set()
            for graph_id, parent_value in zip(
                parent_ids, parent_values
            ):
                primitive_id = tensor_identity(parent_value)
                if primitive_id not in primitive_ids:
                    continue
                exact.append((int(primitive_id), int(graph_id)))
                matched_graph_ids.add(int(graph_id))
                matched_primitive_ids.add(int(primitive_id))
            remaining_primitives = tuple(
                int(primitive_id)
                for primitive_id, _slot in primitive_parents
                if int(primitive_id) not in matched_primitive_ids
            )
            remaining_graph_ids = tuple(
                int(graph_id)
                for graph_id in parent_ids
                if int(graph_id) not in matched_graph_ids
            )
            # A scalar literal can be tensorized inside the AbstractTensor
            # operator, so its source Python value has no tape ID.  Once all
            # exact tensor operands are removed, the remaining one-to-one
            # socket is unambiguous.
            if len(remaining_primitives) == len(remaining_graph_ids):
                exact.extend(zip(
                    remaining_primitives,
                    remaining_graph_ids,
                ))
            if exact:
                context["step_input_ids"][int(capture_id)] = tuple(
                    exact
                )
    return result


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

    def __init__(
        self,
        enabled: bool = False,
        *,
        history: int = 240,
        verbose: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.verbose = bool(verbose)
        self.history = deque(maxlen=max(1, int(history)))
        self.trace_history = deque(maxlen=max(1, int(history)) * 64)
        self.device_trace_history = deque(
            maxlen=max(1, int(history)) * 256
        )
        self.depth = 0
        self.sequence = 0
        self._events: list[dict[str, Any]] = []
        self._root_started_ns = 0
        self._gpu_query_depth = 0
        self._runtime_suppression = 0
        self.error_buffer = DeploymentErrorBuffer(history)

    def trace(
        self,
        *,
        path: str,
        section: str,
        label: str,
        fields: dict[str, Any] | None = None,
    ) -> None:
        if not self.verbose:
            return
        self.sequence += 1
        record = {
            "sequence": self.sequence,
            "path": str(path),
            "section": str(section),
            "label": str(label),
            "fields": dict(fields or {}),
        }
        self.trace_history.append(record)
        details = " | ".join(
            f"{name}={value}"
            for name, value in record["fields"].items()
        )
        print(
            f"[shell-trace {record['sequence']}] {record['path']} | "
            f"{record['section']} | {record['label']}"
            + (f" | {details}" if details else ""),
            flush=True,
        )

    @property
    def exceptions(self):
        """Compatibility view over the root deployment's error buffer."""

        return self.error_buffer.records

    def begin_shell(self, path: str) -> tuple[int, bool] | None:
        if not self.enabled or self._runtime_suppression:
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
        gpu_ms: float | None = None,
    ) -> None:
        if not self.enabled or self._runtime_suppression:
            return
        self._events.append({
            "path": path,
            "section": section,
            "label": label,
            "cpu_ms": float(cpu_ms),
            "gpu_query": gpu_query,
            "gpu_ms": 0.0 if gpu_ms is None else float(gpu_ms),
            "dispatches": int(dispatches),
        })

    def record_device_trace(
        self,
        *,
        path: str,
        records,
        header,
    ) -> None:
        """Ingest records written by a compiled shell's logging SSBO."""

        if not self.enabled or self._runtime_suppression:
            return
        # ``_finish_root`` assigns this number to the invocation after the
        # device records have been read.  Keeping it on every SSBO record lets
        # summary(window=...) apply exactly the same window to host timings and
        # shader-written profiling data.
        invocation = self.sequence + 1
        labels = {
            1: "closure-enter",
            2: "loop-enter",
            3: "region-execute",
            4: "state-commit",
            5: "output-publish",
            8: "snippet-output",
            9: "closure-source-lifetime",
            255: "error",
        }
        for code, subject, payload0, payload1 in records:
            self.device_trace_history.append({
                "invocation": invocation,
                "path": str(path),
                "code": int(code),
                "label": labels.get(int(code), f"event-{int(code)}"),
                "subject": int(subject),
                "payload0": int(payload0),
                "payload1": int(payload1),
            })
        dropped = int(header[1]) if len(header) > 1 else 0
        if dropped:
            self.device_trace_history.append({
                "invocation": invocation,
                "path": str(path),
                "code": 255,
                "label": "ssbo-overflow",
                "subject": 0,
                "payload0": dropped,
                "payload1": int(header[0]),
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
            gpu_ms = float(event.pop("gpu_ms", 0.0))
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
                "device_events": tuple(self.device_trace_history),
            }
        return {
            **self.history[-1],
            "exceptions": self.error_buffer.snapshot(),
            "device_events": tuple(self.device_trace_history),
        }

    def summary(self, *, window: int = 60) -> dict[str, Any]:
        reports = list(self.history)[-max(1, int(window)):]
        if not reports:
            return {
                "frames": 0,
                "total_mean_ms": 0.0,
                "total_p95_ms": 0.0,
                "rows": (),
                "device_rows": (),
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
        report_sequences = {
            int(report["sequence"]) for report in reports
        }
        device_groups: dict[tuple[str, int, str], dict[str, int]] = {}
        for event in self.device_trace_history:
            if int(event.get("invocation", -1)) not in report_sequences:
                continue
            key = (
                str(event["path"]),
                int(event["code"]),
                str(event["label"]),
            )
            group = device_groups.setdefault(key, {
                "events": 0,
                "payload0": 0,
                "payload1": 0,
            })
            group["events"] += 1
            group["payload0"] += int(event["payload0"])
            group["payload1"] += int(event["payload1"])
        device_rows = tuple({
            "path": path,
            "code": code,
            "label": label,
            "events_mean": values["events"] / len(reports),
            "payload0_mean": values["payload0"] / len(reports),
            "payload1_mean": values["payload1"] / len(reports),
        } for (path, code, label), values in sorted(
            device_groups.items(),
            key=lambda item: (item[0][0], item[0][1]),
        ))
        return {
            "frames": len(reports),
            "total_mean_ms": sum(totals) / len(totals),
            "total_p95_ms": percentile95(totals),
            "rows": tuple(rows),
            "device_rows": device_rows,
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
    children = (
        *getattr(shell, "function_shells", {}).items(),
        *(
            (f"callsite-{node_id}", child)
            for node_id, child in getattr(
                shell, "callsite_function_shells", {}
            ).items()
        ),
    )
    for reference, child in children:
        if id(child) in visited:
            continue
        _attach_profiler(
            child,
            profiler,
            f"{path}/{_shell_profile_name(child)}@{reference}",
            visited=visited,
        )


def _walk_planned_shells(
    shell: Any,
    *,
    include_function_registry: bool = True,
):
    """Yield planner-created shell instances exactly once.

    ``function_shells`` is the definition catalogue;
    ``callsite_function_shells`` is one selected program's activation tree.
    Compilation/capture of one entrypoint passes
    ``include_function_registry=False`` so catalogue entries are not mistaken
    for executed children.  Administrative callers retain the broad default.
    """
    pending = [shell]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if include_function_registry:
            pending.extend(
                getattr(current, "function_shells", {}).values()
            )
        elif (
            getattr(current, "runtime_closure_only", False)
            and getattr(current, "_owns_function_shells", False)
        ):
            # In runtime-root mode this shell is an administrative owner of
            # the definition catalogue.  Its executable children are exactly
            # the submitted activation roots, not every catalogued definition
            # and not zero children.  Each selected root then exposes its full
            # explicit callsite tree below.
            function_shells = getattr(current, "function_shells", {})
            pending.extend(
                function_shells[int(reference)]
                for reference in reversed(tuple(
                    getattr(current, "activation_root_references", ())
                ))
                if int(reference) in function_shells
            )
        pending.extend(
            getattr(current, "callsite_function_shells", {}).values()
        )


def _deployment_program_table_lines(shell: Any) -> tuple[str, ...]:
    """Render the planner-owned shell hierarchy and region compartments."""

    def shell_name(current):
        return str(
            current.process_graph.G.graph.get("function_name")
            or type(current).__name__
        )

    def render_table(headers, rows):
        rows = [tuple(str(cell) for cell in row) for row in rows]
        widths = [
            max(len(str(header)), *(len(row[index]) for row in rows))
            for index, header in enumerate(headers)
        ]
        yield " | ".join(
            str(header).ljust(widths[index])
            for index, header in enumerate(headers)
        )
        yield "-+-".join("-" * width for width in widths)
        for row in rows:
            yield " | ".join(
                cell.ljust(widths[index])
                for index, cell in enumerate(row)
            )

    hierarchy = []
    ordered_shells = []
    seen = set()

    def visit(current, depth, compartment):
        if id(current) in seen:
            return
        seen.add(id(current))
        shell_index = len(ordered_shells)
        ordered_shells.append(current)
        compiled = {
            key[-2] if isinstance(key, tuple) else key
            for key in current.captured_region_programs
        }
        hierarchy.append((
            shell_index,
            depth,
            compartment,
            ("  " * depth) + shell_name(current),
            current.source_node_count,
            current.primitive_count,
            current.dispatch_count,
            len(compiled),
            len(current.coordinator_region_indices),
            current.planned_invocation_slots,
        ))
        for node_id, child in sorted(
            current.callsite_function_shells.items(),
            key=lambda item: item[0],
        ):
            visit(child, depth + 1, f"callsite-{node_id}")

    visit(shell, 0, "root")
    lines = [
        f"runtime selection: {shell.control_runtime}",
        "compiled program shell hierarchy",
    ]
    lines.extend(render_table(
        (
            "id", "depth", "compartment", "shell", "graph", "scheduled",
            "regions", "shaders", "coord", "slots",
        ),
        hierarchy,
    ))

    regions = []
    for shell_index, current in enumerate(ordered_shells):
        compiled = {
            key[-2] if isinstance(key, tuple) else key
            for key in current.captured_region_programs
        }
        for region_index, subgraph in enumerate(current.dispatch_subgraphs):
            if region_index in current.coordinator_region_indices:
                kind = "coordinator"
            elif region_index in compiled:
                kind = "shader"
            else:
                kind = "uncaptured"
            operations = " -> ".join(
                str(
                    subgraph.G.nodes[node_id].get("op")
                    or subgraph.G.nodes[node_id].get("type")
                )
                for node_id in subgraph.G.graph.get("deployment_nodes", ())
            )
            regions.append((
                shell_index,
                region_index,
                kind,
                len(subgraph.G.graph.get("deployment_inputs", ())),
                len(subgraph.G.graph.get("deployment_outputs", ())),
                len(subgraph.G.graph.get("compartment_schedule", ())),
                ",".join(subgraph.G.graph.get("rewrite_history", ())) or "-",
                operations or "-",
            ))
    lines.extend(("", "compiled program region compartments"))
    lines.extend(render_table(
        (
            "shell", "region", "kind", "in", "out", "waves", "identities",
            "operations",
        ),
        regions,
    ))
    loop_reductions = []
    for shell_index, current in enumerate(ordered_shells):
        for reduction in current.loop_shader_reductions:
            loop_reductions.append((
                shell_index,
                reduction.loop_node_id,
                ",".join(map(str, reduction.region_indices)) or "-",
                "collapse" if reduction.collapsible else "coordinator",
                (
                    "dynamic"
                    if reduction.estimated_dispatches_removed is None
                    else reduction.estimated_dispatches_removed
                ),
                ",".join(reduction.blockers) or "-",
                ",".join(
                    name for name, _initial, _updated
                    in reduction.carried_bindings
                ) or "-",
            ))
    if loop_reductions:
        lines.extend(("", "planner loop-to-shader reduction analysis"))
        lines.extend(render_table(
            (
                "shell", "loop", "regions", "verdict",
                "dispatches removed", "blockers", "carried",
            ),
            loop_reductions,
        ))
    return tuple(lines)


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


def _validation_control_blocks(shell: Any) -> tuple[ValidationBlock, ...]:
    """Translate ordinary ``if predicate: raise`` guards into typed control."""

    graph = shell.process_graph.G
    expression_nodes = {
        id(data.get("expr_obj")): int(node_id)
        for node_id, data in graph.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.AST)
    }
    blocks = []
    for node_id, data in graph.nodes(data=True):
        statement = data.get("expr_obj")
        if not (
            isinstance(statement, ast.If)
            and statement.body
            and all(isinstance(item, ast.Raise) for item in statement.body)
            and not statement.orelse
        ):
            continue
        test = statement.test
        raises_when_true = True
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            test = test.operand
            raises_when_true = False
        predicate_id = expression_nodes.get(id(test))
        if predicate_id is None:
            continue
        child = shell.callsite_function_shells.get(predicate_id)
        if child is not None:
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
        else:
            # Static scalar/shape guards are resolved during planning; only a
            # tensor scalar projection becomes runtime validation control.
            continue
        blocks.append(ValidationBlock(
            predicate_id,
            int(node_id),
            expect_true=not raises_when_true,
        ))
    return tuple(blocks)


def _positional_argument_index(role: str) -> int | None:
    """Normalize the two ProcessGraph spellings used for positional edges."""

    role = str(role)
    if role.startswith("arg:") and role[4:].isdigit():
        return int(role[4:])
    if role.startswith("arg") and role[3:].isdigit():
        return int(role[3:])
    return None


def _method_parameter_layout(graph: Any) -> tuple[
    str | None, tuple[str, ...], tuple[str, ...]
]:
    """Return receiver, positional-call parameters, and every parameter.

    ProcessGraph parameter identities are SSA-oriented and their combined
    ``function_parameters`` tuple is not an ABI guarantee: in particular an
    instance receiver can appear after another parameter.  Call binding must
    identify the receiver as a receiver, not manufacture an offset into that
    tuple.  The source spellings ``self`` and ``cls`` are authoritative when
    retained; a source positional list is the fallback for unconventional
    receiver names.
    """

    metadata = graph.graph
    all_parameters = tuple(metadata.get("function_parameters", ()))
    positional = tuple(
        metadata.get("positional_parameters", ())
        or all_parameters
    )
    binding = metadata.get("method_binding")
    receiver = None
    if binding == "instance":
        receiver = next(
            (name for name in all_parameters if name == "self"),
            positional[0] if positional else None,
        )
    elif binding == "class":
        receiver = next(
            (name for name in all_parameters if name == "cls"),
            positional[0] if positional else None,
        )
    call_positional = tuple(
        name for name in positional if name != receiver
    )
    return receiver, call_positional, all_parameters


def _declared_output_terminals(
    graph: Any,
    *,
    produced_values: set[int] | None = None,
) -> dict[str, int]:
    """Expand structural return aggregates into their numerical SSA leaves."""

    identities = graph.G.graph.get("identity_table") or {}
    terminals: dict[str, int] = {}
    visiting: set[int] = set()

    def expand(name: str, node_id: int) -> None:
        node_id = int(node_id)
        if node_id in visiting or node_id not in graph.G:
            return
        visiting.add(node_id)
        data = graph.G.nodes[node_id]
        expression = data.get("expr_obj")
        attributes = data.get("attributes") or {}
        class_name = attributes.get("class_ref")
        if data.get("type") in {"LoopExit", "LoopResult"}:
            value_parent = next(
                (
                    int(parent)
                    for parent, role in (data.get("parents") or ())
                    if str(role) == "value"
                ),
                None,
            )
            if value_parent is not None:
                expand(name, value_parent)
        elif isinstance(expression, ast.Call) and class_name is not None:
            descriptor = (
                graph.G.graph.get("class_table", {}).get(class_name)
                or {}
            )
            fields = tuple(descriptor.get("fields") or ())
            parents = tuple(
                (int(parent), str(role))
                for parent, role in (data.get("parents") or ())
                if str(role) not in {"callee", "func", "definition"}
            )
            positional = {
                position: parent
                for parent, role in parents
                if (
                    position := _positional_argument_index(role)
                ) is not None
            }
            keywords = {
                role[3:]: parent
                for parent, role in parents
                if role.startswith("kw:")
            }
            for index, field in enumerate(fields):
                parent = keywords.get(field, positional.get(index))
                if parent is not None:
                    expand(f"{name}.{field}", parent)
        elif isinstance(
            expression, (ast.Tuple, ast.List)
        ):
            elements = [
                int(parent)
                for parent, role in (data.get("parents") or ())
                if str(role) in {"elts", "element", "item"}
            ]
            for index, parent in enumerate(elements):
                expand(f"{name}.{index}", parent)
        elif produced_values is None or node_id in produced_values:
            terminals[str(name)] = node_id
        visiting.remove(node_id)

    for name in graph.G.graph.get("function_outputs", ()):
        values = identities.get(name, ())
        if values:
            expand(str(name), int(values[-1]))
    return terminals


def _control_dependency_value_ids(control: Any) -> frozenset[int]:
    from .control_source import control_dependency_value_ids

    return control_dependency_value_ids(control)

def _build_shell_hierarchy_plan(shell: Any) -> PlanClosure:
    """Freeze call/region ownership before backend source composition."""

    graph = shell.process_graph
    region_by_node = {
        int(node_id): int(region_index)
        for region_index, subgraph in enumerate(shell.dispatch_subgraphs)
        for node_id in subgraph.G.graph.get("deployment_nodes", ())
    }
    emitted_regions = set()
    items = []
    topological_nodes = _dependency_order(graph)
    order_index = {
        int(node_id): index
        for index, node_id in enumerate(topological_nodes)
    }
    def source_position(node_id: int) -> tuple[int, int] | None:
        expression = graph.G.nodes[int(node_id)].get("expr_obj")
        line = getattr(expression, "lineno", None)
        column = getattr(expression, "col_offset", None)
        if line is None:
            return None
        return int(line), int(column or 0)

    caller_identities = graph.G.graph.get("identity_table") or {}
    for node_id in topological_nodes:
        child = shell.callsite_function_shells.get(node_id)
        if child is not None:
            call_attributes = graph.G.nodes[node_id].get("attributes") or {}
            constructor_call = call_attributes.get("constructor_ref") is not None
            call_parents = tuple(
                (int(parent), str(role))
                for parent, role in (
                    graph.G.nodes[node_id].get("parents") or ()
                )
                if str(role) not in {"callee", "func", "definition"}
            )
            parents = tuple(parent for parent, _role in call_parents)
            child_graph = child.process_graph.G
            child_identities = (
                child_graph.graph.get("identity_table") or {}
            )
            receiver_parameter, positional_parameters, _parameters = (
                _method_parameter_layout(child_graph)
            )
            argument_bindings = []
            if constructor_call and receiver_parameter is not None:
                receiver_identities = child_identities.get(
                    receiver_parameter, ()
                )
                if receiver_identities:
                    # The class-call result is the caller-owned record identity
                    # initialized by __init__/__new__.  It is both the
                    # constructor callsite and the concrete ``self`` binding;
                    # its fields remain ordinary caller-owned arena values.
                    argument_bindings.append((
                        int(node_id), int(receiver_identities[0])
                    ))
            for parent, role in call_parents:
                position = _positional_argument_index(role)
                if position is not None:
                    name = (
                        positional_parameters[position]
                        if position < len(positional_parameters)
                        else None
                    )
                elif (
                    role == "operand"
                    and receiver_parameter is not None
                ):
                    name = receiver_parameter
                elif role.startswith("kw:"):
                    name = role.split(":", 1)[1]
                else:
                    name = None
                identities = child_identities.get(name, ())
                if identities:
                    argument_bindings.append(
                        (parent, int(identities[0]))
                    )
            bound_child_inputs = {
                int(child_input)
                for _caller, child_input in argument_bindings
            }
            for child_input, child_data in child_graph.nodes(data=True):
                child_attributes = child_data.get("attributes") or {}
                if (
                    child_data.get("type") != "Input"
                    or child_attributes.get("binding_kind") != "external"
                    or int(child_input) in bound_child_inputs
                ):
                    continue
                name = child_attributes.get("binding_name")
                call_position = source_position(int(node_id))
                candidates = [
                    int(definition)
                    for definition in caller_identities.get(name, ())
                    if (
                        int(definition) in order_index
                        and (
                            (
                                call_position is not None
                                and source_position(int(definition))
                                is not None
                                and source_position(int(definition))
                                < call_position
                            )
                            or (
                                (
                                    call_position is None
                                    or source_position(int(definition))
                                    is None
                                )
                                and order_index[int(definition)]
                                < order_index[int(node_id)]
                            )
                        )
                    )
                ]
                if candidates:
                    argument_bindings.append((
                        max(
                            candidates,
                            key=lambda definition: (
                                source_position(definition)
                                or (-1, order_index[definition])
                            ),
                        ),
                        int(child_input),
                    ))
            child_outputs = tuple(
                int(child_identities[name][-1])
                for name in child_graph.graph.get(
                    "function_outputs", ()
                )
                if child_identities.get(name)
            )
            unpacked = {}
            for successor in graph.G.successors(node_id):
                successor_data = graph.G.nodes[successor]
                if str(successor_data.get("op", "")).lower() != "indexed":
                    continue
                index_parents = tuple(
                    int(parent)
                    for parent, role in (
                        successor_data.get("parents") or ()
                    )
                    if str(role) == "index"
                )
                if not index_parents:
                    continue
                index_data = graph.G.nodes[index_parents[0]]
                index_value = index_data.get("constant")
                if index_value is None:
                    index_value = (
                        index_data.get("attributes") or {}
                    ).get("value")
                if isinstance(index_value, int):
                    unpacked[int(index_value)] = int(successor)
            if len(child_outputs) == 1:
                result_bindings = ((child_outputs[0], int(node_id)),)
            else:
                result_bindings = tuple(
                    (child_output, unpacked[index])
                    for index, child_output in enumerate(child_outputs)
                    if index in unpacked
                )
            items.append(PlanCall(
                int(node_id),
                _build_shell_hierarchy_plan(child),
                parents,
                tuple(caller for _callee, caller in result_bindings),
                tuple(argument_bindings),
                tuple(result_bindings),
                tuple(
                    int(plan.loop.node_id)
                    for plan in sorted(
                        (
                            plan
                            for plan in shell.loop_plans
                            if int(node_id) in plan.loop.body_nodes
                        ),
                        key=lambda plan: -len(plan.loop.body_nodes),
                    )
                ),
            ))
            continue
        region_index = region_by_node.get(int(node_id))
        if (
            region_index is None
            or region_index in emitted_regions
        ):
            continue
        emitted_regions.add(region_index)
        subgraph = shell.dispatch_subgraphs[region_index]
        region_nodes = tuple(
            int(value)
            for value in subgraph.G.graph.get("deployment_nodes", ())
        )
        original_region_node_set = set(region_nodes)
        carried_initial_boundaries = {
            int(initial)
            for loop_plan in getattr(shell, "loop_plans", ())
            for _name, initial, updated in loop_plan.loop.carried_bindings
            if int(updated) in original_region_node_set
            and int(initial) != int(updated)
        }
        # A carried initializer is the entry arm of the coordinator-owned Phi.
        # If the numerical compartment also owns that initializer, its body
        # recomputes from the iteration-zero value on every trip.  Cut it from
        # this region so the ordinary capture calculation below binds the
        # current Phi value into the update operation.
        region_nodes = tuple(
            value_id for value_id in region_nodes
            if value_id not in carried_initial_boundaries
        )
        region_node_set = set(region_nodes)
        # Loop-port materialization can rewrite a region's live parents after
        # its dispatch subgraph was first carved. Recompute region captures
        # from the authoritative current graph instead of
        # reusing stale deployment_inputs from the numeric view.
        region_captures = tuple(dict.fromkeys((
            *(
                int(parent)
                for node_id in region_nodes
                for parent, _role in (
                    graph.G.nodes[node_id].get("parents") or ()
                )
                if int(parent) not in region_node_set
            ),
        )))

        # A region capture is not necessarily a runtime input. Structural
        # arguments such as ``transpose(-2, -1)`` often sit just outside the
        # tensor region as a tiny constant-expression chain (Const -> neg, or
        # several constants -> Tuple). Preserve that source closure in the
        # region instead of promoting its terminal to an unexplained ABI
        # parameter. This is graph-source pursuit, not value specialization:
        # only expressions whose complete producer closure is literal are
        # admitted here.
        # Keep evaluator state separate from its values.  A Python ``object()``
        # sentinel used to be written into this runtime-indexed cache for an
        # unavailable producer.  When this planner itself is ingested, that is
        # a static Python reference flowing through dynamic table storage --
        # something complete SSA correctly refuses to pretend is data.
        _CONSTANT_KNOWN = 1
        _CONSTANT_UNAVAILABLE = 2
        _constant_status: dict[int, int] = {}
        _constant_values: dict[int, Any] = {}

        def _constant_expression(value_id: int):
            """Evaluate a literal producer closure without Python recursion.

            A retained loop can expose feedback in the ProcessGraph, and a
            large whole-object compile can contain a perfectly acyclic closure
            deeper than Python's call-stack limit. Neither makes the capture a
            constant. Use an explicit post-order walk, marking every active
            frame nonconstant if a feedback edge is encountered.
            """

            root = int(value_id)
            if root in _constant_status:
                return (
                    _constant_status[root] == _CONSTANT_KNOWN,
                    _constant_values.get(root),
                )

            # 0/absent = unseen, 1 = active DFS frame, 2 = evaluated.
            states: dict[int, int] = {}
            stack: list[tuple[int, bool]] = [(root, False)]
            while stack:
                current, expanded = stack.pop()
                if current in _constant_status:
                    states[current] = 2
                    continue
                node = graph.G.nodes.get(current, {})
                opcode = str(node.get("op") or node.get("type") or "")
                attributes = dict(node.get("attributes") or {})
                parents = tuple(
                    int(parent)
                    for parent, _role in (node.get("parents") or ())
                )

                if not expanded:
                    if states.get(current) == 1:
                        # This node is already on the active producer path: a
                        # feedback closure cannot be a literal expression.
                        for active, state in tuple(states.items()):
                            if state == 1:
                                _constant_status[active] = (
                                    _CONSTANT_UNAVAILABLE
                                )
                                states[active] = 2
                        continue
                    states[current] = 1
                    if opcode in {"const", "Const", "Constant"}:
                        expression = node.get("expr_obj")
                        if "value" in attributes:
                            value = attributes["value"]
                            known = True
                        elif "constant" in node:
                            value = node["constant"]
                            known = True
                        elif isinstance(expression, ast.Constant):
                            value = expression.value
                            known = True
                        else:
                            value = None
                            known = False
                        if known:
                            _constant_values[current] = value
                            _constant_status[current] = _CONSTANT_KNOWN
                        else:
                            _constant_status[current] = (
                                _CONSTANT_UNAVAILABLE
                            )
                        states[current] = 2
                        continue
                    stack.append((current, True))
                    for parent in reversed(parents):
                        if parent in _constant_status:
                            continue
                        if states.get(parent) == 1:
                            for active, state in tuple(states.items()):
                                if state == 1:
                                    _constant_status[active] = (
                                        _CONSTANT_UNAVAILABLE
                                    )
                                    states[active] = 2
                            break
                        stack.append((parent, False))
                    continue

                if states.get(current) == 2:
                    continue
                if any(
                    _constant_status.get(parent) != _CONSTANT_KNOWN
                    for parent in parents
                ):
                    known = False
                elif opcode in {"neg", "USub"} and len(parents) == 1:
                    values = tuple(_constant_values[parent] for parent in parents)
                    try:
                        value = -values[0]
                    except (ArithmeticError, TypeError, ValueError):
                        known = False
                    else:
                        known = True
                elif opcode in {"pos", "UAdd"} and len(parents) == 1:
                    values = tuple(_constant_values[parent] for parent in parents)
                    try:
                        value = +values[0]
                    except (ArithmeticError, TypeError, ValueError):
                        known = False
                    else:
                        known = True
                elif opcode in {"Tuple", "tuple"}:
                    values = tuple(_constant_values[parent] for parent in parents)
                    value = tuple(values)
                    known = True
                elif opcode in {"List", "list"}:
                    values = tuple(_constant_values[parent] for parent in parents)
                    value = list(values)
                    known = True
                elif opcode in {"add", "Add"} and len(parents) == 2:
                    values = tuple(_constant_values[parent] for parent in parents)
                    try:
                        value = values[0] + values[1]
                    except (ArithmeticError, TypeError, ValueError):
                        known = False
                    else:
                        known = True
                elif opcode in {"sub", "Sub"} and len(parents) == 2:
                    values = tuple(_constant_values[parent] for parent in parents)
                    try:
                        value = values[0] - values[1]
                    except (ArithmeticError, TypeError, ValueError):
                        known = False
                    else:
                        known = True
                elif opcode in {"mul", "Mul"} and len(parents) == 2:
                    values = tuple(_constant_values[parent] for parent in parents)
                    try:
                        value = values[0] * values[1]
                    except (ArithmeticError, TypeError, ValueError):
                        known = False
                    else:
                        known = True
                else:
                    known = False
                if known:
                    _constant_values[current] = value
                    _constant_status[current] = _CONSTANT_KNOWN
                else:
                    _constant_status[current] = _CONSTANT_UNAVAILABLE
                states[current] = 2
            return (
                _constant_status.get(root) == _CONSTANT_KNOWN,
                _constant_values.get(root),
            )

        capture_constants = {}
        for value_id in region_captures:
            if value_id in carried_initial_boundaries:
                continue
            known, value = _constant_expression(value_id)
            if known:
                capture_constants[value_id] = value
        region_captures = tuple(
            value_id
            for value_id in region_captures
            if value_id not in capture_constants
        )
        compute_lines = []
        for value in region_nodes:
            node_data = graph.G.nodes[value]
            expression = node_data.get("expr_obj")
            parents = tuple(
                (int(parent), str(role))
                for parent, role in (node_data.get("parents") or ())
            )
            opcode = str(node_data.get("op") or node_data.get("type"))
            line_attributes = dict(node_data.get("attributes") or {})
            if isinstance(expression, ast.Attribute):
                # The graph may retain the frontend spelling ``Attribute``;
                # repository SSA uses the semantic memory operation and must
                # retain the selected field name in its record ABI.
                opcode = "GetAttr"
                line_attributes.setdefault("attribute", expression.attr)
            compute_lines.append(PlanLine.create(
                opcode,
                inputs=tuple(parent for parent, _role in parents),
                outputs=(int(value),),
                attributes={
                    **line_attributes,
                    "region": region_index,
                },
                input_roles=tuple(role for _parent, role in parents),
            ))
        compute_lines = tuple(compute_lines)
        # A constant operand (``self.n + 1``) is a leaf the region consumes but
        # neither produces nor captures -- it is not a runtime input, so it never
        # entered ``deployment_inputs``. The fused capture folds it into a scalar
        # attribute; the plan lines reference it by value id, so without a
        # defining line it is a dangling SSA value the backend emits as an
        # undeclared variable. Materialise each such constant as its own ``const``
        # line, ahead of the consumers, so the region is self-contained.
        produced = set(region_nodes)
        captured = set(region_captures)
        const_lines = []
        materialised: set[int] = set()
        for value_id, value in capture_constants.items():
            materialised.add(int(value_id))
            const_lines.append(PlanLine.create(
                "Const",
                inputs=(),
                outputs=(int(value_id),),
                attributes={"value": value, "region": region_index},
            ))
        for value in region_nodes:
            for parent, _role in (graph.G.nodes[value].get("parents") or ()):
                parent = int(parent)
                if (
                    parent in produced
                    or parent in captured
                    or parent in materialised
                ):
                    continue
                parent_data = graph.G.nodes.get(parent, {})
                if (
                    parent_data.get("op") or parent_data.get("type")
                ) not in ("const", "Constant"):
                    continue
                materialised.add(parent)
                const_lines.append(PlanLine.create(
                    "Const",
                    inputs=(),
                    outputs=(parent,),
                    attributes={
                        **dict(parent_data.get("attributes") or {}),
                        "region": region_index,
                    },
                ))
        # Carry each region value's shape/dtype from its process-graph node's
        # domain so the lowered SSA is shaped: a value that is an array lowers
        # as an array, not a scalar. A domain padded with size-1 dims (a scalar
        # is (1, 1, 1)) squeezes to its logical shape -- () for a scalar.
        _shape_dtype_cache: dict[int, tuple[tuple[int, ...], str]] = {}

        def _value_shape_dtype(value_id):
            """Infer one value record with a bounded explicit producer stack.

            Feedback is a retained-loop boundary.  When the work stack sees a
            parent already active, it seeds that loop-carried record from the
            parent's declared tensor/domain metadata and does not expand the
            edge again.  This is finite even for irreducible recursion and does
            not turn a loop into an unbounded source expansion.
            """

            root = int(value_id)

            def declared(current: int):
                node = graph.G.nodes.get(int(current), {})
                tensor = node.get("tensor") or {}
                domain = node.get("domain_node")
                attributes = dict(node.get("attributes") or {})
                binding_name = attributes.get("binding_name")
                specialization = (
                    (graph.G.graph.get("planner_specializations") or {}).get(
                        str(binding_name)
                    )
                    if binding_name is not None else None
                )
                specialized_shape = tuple(
                    int(extent)
                    for extent in (
                        getattr(specialization, "shape", ()) or ()
                    )
                )
                shape = tuple(
                    tensor.get("shape") or specialized_shape or ()
                )
                if not shape:
                    shape = tuple(getattr(domain, "shape", ()) or ())
                logical = (
                    tuple(map(int, shape))
                    if tensor.get("shape") is not None or specialized_shape
                    else tuple(int(dim) for dim in shape if int(dim) != 1)
                )
                dtype = str(
                    tensor.get("dtype")
                    or getattr(specialization, "dtype", None)
                    or node.get("dtype")
                    or getattr(domain, "dtype", None)
                    or "float64"
                )
                operation = str(
                    attributes.get("tensor")
                    or attributes.get("tensor_operation")
                    or node.get("op")
                    or node.get("type")
                    or ""
                ).casefold()
                if operation in {"const", "constant"}:
                    literal = attributes.get("value", node.get("constant"))
                    if isinstance(literal, bool):
                        dtype = "bool"
                    elif isinstance(literal, int):
                        dtype = "int"
                    elif isinstance(literal, float):
                        dtype = "float64"
                parents = tuple(
                    int(parent)
                    for parent, _role in (node.get("parents") or ())
                    if int(parent) in graph.G
                )
                return node, tensor, attributes, logical, dtype, operation, parents

            states: dict[int, int] = {}
            stack: list[tuple[int, bool]] = [(root, False)]
            while stack:
                current, expanded = stack.pop()
                if current in _shape_dtype_cache:
                    states[current] = 2
                    continue
                (
                    _node,
                    tensor,
                    attributes,
                    logical,
                    dtype,
                    operation,
                    parents,
                ) = declared(current)
                if not expanded:
                    states[current] = 1
                    stack.append((current, True))
                    for parent in reversed(parents):
                        if parent in _shape_dtype_cache:
                            continue
                        if states.get(parent) == 1:
                            # The edge closes a retained loop. Its declared
                            # record is the finite boundary condition.
                            (_n, _t, _a, parent_shape, parent_dtype, _o, _p) = (
                                declared(parent)
                            )
                            _shape_dtype_cache[parent] = (
                                parent_shape, parent_dtype
                            )
                            states[parent] = 2
                            continue
                        stack.append((parent, False))
                    continue
                if states.get(current) == 2:
                    continue
                parent_records = [
                    (parent, *_shape_dtype_cache[parent])
                    for parent in parents
                    if parent in _shape_dtype_cache
                ]
                parent_shapes = [record[1] for record in parent_records]
                parent_dtypes = [record[2] for record in parent_records]
                # DomainNode historically pads scalars with unit axes. If it
                # says scalar but a canonical tensor op consumes a shaped
                # tensor, carry that shape through.
                if operation in {
                    "neg", "abs", "sqrt", "exp", "log", "round", "trunc",
                    "floor", "ceil", "isfinite", "isnan", "isinf",
                    "logical_not", "tanh", "sin", "cos", "tan", "asin",
                    "acos", "atan", "sinh", "cosh", "asinh", "acosh",
                    "atanh", "sign", "add", "sub", "mul", "truediv",
                    "floordiv", "mod", "pow", "less", "less_equal",
                    "greater", "greater_equal", "equal", "not_equal",
                    "maximum", "minimum", "where", "float", "double",
                    "long", "int", "to", "to_dtype", "astype", "cbrt",
                }:
                    shaped = [candidate for candidate in parent_shapes if candidate]
                    if shaped:
                        logical = max(
                            shaped,
                            key=lambda candidate: (len(candidate), candidate),
                        )
                    if parent_dtypes and not tensor.get("dtype"):
                        dtype = parent_dtypes[0]
                if operation in {"sum", "prod", "min", "max", "any", "all"}:
                    axis = attributes.get("axis", attributes.get("dim"))
                    source_shape = next(
                        (candidate for candidate in parent_shapes if candidate),
                        (),
                    )
                    if axis is None:
                        logical = ()
                    elif source_shape:
                        normalized = int(axis) % len(source_shape)
                        logical = (
                            source_shape[:normalized]
                            + source_shape[normalized + 1:]
                        )
                if operation == "matmul" and len(parent_shapes) >= 2:
                    left_shape, right_shape = parent_shapes[:2]
                    if (
                        len(left_shape) == 2
                        and len(right_shape) == 2
                        and left_shape[1] == right_shape[0]
                    ):
                        logical = (left_shape[0], right_shape[1])
                _shape_dtype_cache[current] = (logical, dtype)
                states[current] = 2
            logical, dtype = _shape_dtype_cache.get(
                root, declared(root)[3:5]
            )
            return (root, logical, dtype)

        value_shapes = tuple(
            _value_shape_dtype(value_id)
            for value_id in (*region_nodes, *region_captures)
        )
        items.append(PlanClosure(
            name=f"region_{region_index}",
            captures=region_captures,
            items=(*const_lines, *compute_lines),
            value_shapes=value_shapes,
        ))
    control_values = set(_control_dependency_value_ids(
        getattr(shell, "shell_control_program", None)
    ))

    return PlanClosure(
        name=str(
            graph.G.graph.get("function_name")
            or type(shell).__name__
        ),
        captures=tuple(dict.fromkeys((
            *sorted(control_values),
            *(
            int(node_id)
            for node_id, data in graph.G.nodes(data=True)
            if data.get("type") == "Input"
            and graph.G.out_degree(node_id)
            ),
        ))),
        items=tuple(items),
    )


def _loop_induction_name(loop_node_id: int) -> str:
    return f"iteration_{loop_node_id}"


def _find_nested_loop_node_ids_in_block(
    block: Any,
) -> "frozenset[int]":
    """Collect the node_ids of all LoopBlock/WhileBlock induction names in a
    block tree by parsing the ``iteration_{node_id}`` induction convention."""

    found: set[int] = set()

    def walk(b: Any) -> None:
        if isinstance(b, LoopBlock):
            name = str(b.induction)
            if name.startswith("iteration_"):
                try:
                    found.add(int(name[len("iteration_"):]))
                except ValueError:
                    pass
            walk(b.body)
        elif isinstance(b, WhileBlock):
            walk(b.condition)
            walk(b.body)
        elif isinstance(b, SequenceBlock):
            for child in b.blocks:
                walk(child)
        elif isinstance(b, StateMachineTick):
            for _value, body in b.cases:
                walk(body)
            if b.default is not None:
                walk(b.default)
        elif isinstance(b, ParallelDeployment):
            for lane in b.lanes:
                walk(lane)
        elif isinstance(b, CallBlock):
            walk(b.callee)

    walk(block)
    return frozenset(found)


def _loop_reduction_nesting_hints(
    reductions: "Sequence[LoopShaderReduction]",
    loop_plans: "Iterable[LoopPlan]",
    graph: Any | None = None,
) -> dict[int, frozenset[int]]:
    """Direct-child hints for ``overlay_scheduled_control``'s
    ``known_nesting``, keyed by index into ``reductions``.

    Region-set containment is only a proxy for "this loop is lexically
    nested inside that one" -- it fails when the outer loop's entire body
    is the inner loop (``while a: while b: ...`` with nothing of the
    outer's own between them), since then both compute the identical
    region set rather than the outer being a superset. The real signal is
    the same one ``analyze_shader_loop_reductions`` already used to
    compute each reduction's own ``region_indices`` in the first place:
    ``LoopDescriptor.body_nodes``, the set of graph nodes lexically inside
    a loop's body. A loop is a direct concern of this hint only if its own
    node id is a member of another loop's ``body_nodes`` -- real AST
    nesting, not inferred from schedule position or edge traversal.

    Fallback: when ``loop_plans`` does not provide ``body_nodes`` for a
    candidate outer loop (e.g. the plan was evaporated or belongs to a
    different shell), equal-region-set pairs are disambiguated by
    inspecting the ``control_program.root`` block tree.  The outer loop's
    un-projected ``LoopBlock`` body must contain the inner loop's
    ``LoopBlock`` as a direct or indirect child; this is always true for
    genuinely nested reductions because the planner embeds the inner loop's
    ``LoopBlock`` inside the outer one before any region projection occurs.
    """

    loop_plans = tuple(loop_plans or ())
    body_nodes_by_loop_id = {
        int(plan.loop.node_id): frozenset(map(int, plan.loop.body_nodes))
        for plan in loop_plans
    }
    loop_ids = [int(reduction.loop_node_id) for reduction in reductions]
    hints: dict[int, frozenset[int]] = {}
    for parent_index, parent_loop_id in enumerate(loop_ids):
        body = body_nodes_by_loop_id.get(parent_loop_id)
        if not body:
            continue
        children = frozenset(
            child_index
            for child_index, child_loop_id in enumerate(loop_ids)
            if child_index != parent_index and child_loop_id in body
        )
        if children:
            hints[parent_index] = children

    # Fallback for equal-region-set pairs missed by the body_nodes pass.
    # Two reductions whose projected region sets are equal but whose loop
    # plans did not provide a nesting signal are resolved structurally: the
    # outer reduction's un-projected control_program.root should embed the
    # inner reduction's LoopBlock (identifiable by its ``iteration_{id}``
    # induction name) somewhere in its tree.
    already_covered_as_child = frozenset(
        child for children in hints.values() for child in children
    )
    for parent_index, parent_reduction in enumerate(reductions):
        if parent_reduction.control_program is None:
            continue
        parent_region_set = frozenset(parent_reduction.region_indices)
        if not parent_region_set:
            continue
        parent_nested_ids = _find_nested_loop_node_ids_in_block(
            parent_reduction.control_program.root
        )
        if not parent_nested_ids:
            continue
        fallback_children = frozenset(
            child_index
            for child_index, child_reduction in enumerate(reductions)
            if child_index != parent_index
            and child_index not in already_covered_as_child
            and frozenset(child_reduction.region_indices) == parent_region_set
            and int(child_reduction.loop_node_id) in parent_nested_ids
        )
        if fallback_children:
            merged = hints.get(parent_index, frozenset()) | fallback_children
            hints[parent_index] = merged

    # Comprehension clauses are represented as sibling ``generators`` parents
    # of one aggregate materializer, not as one clause's body containing the
    # next clause's node. Their ordered parent list is nevertheless the exact
    # Python lexical nesting order: ``for outer ... for inner ...``. Preserve
    # that order explicitly when multiple clauses reduce to the same region.
    if graph is not None:
        reduction_index_by_loop_id = {
            int(reduction.loop_node_id): index
            for index, reduction in enumerate(reductions)
        }
        for _node_id, data in graph.G.nodes(data=True):
            generator_indices = tuple(
                reduction_index_by_loop_id[int(parent_id)]
                for parent_id, role in data.get("parents", ())
                if str(role) == "generators"
                and int(parent_id) in reduction_index_by_loop_id
            )
            for parent_index, child_index in zip(
                generator_indices, generator_indices[1:]
            ):
                parent_regions = frozenset(
                    reductions[parent_index].region_indices
                )
                child_regions = frozenset(
                    reductions[child_index].region_indices
                )
                if parent_regions and child_regions == parent_regions:
                    hints[parent_index] = (
                        hints.get(parent_index, frozenset()) | {child_index}
                    )

    return hints


def _planned_operator_node_ids(hierarchy: PlanClosure) -> tuple[int, ...]:
    """Return this closure's operators in planner-owned segment order."""

    return tuple(
        int(output_id)
        for item in hierarchy.items
        if isinstance(item, PlanClosure)
        and item.name.startswith("region_")
        for line in item.items
        if isinstance(line, PlanLine)
        for output_id in line.outputs
    )


@dataclass(frozen=True)
class PlannedOperatorImplementation:
    node_id: int
    kind: str
    fused_step_ids: tuple[int, ...] = ()


def _build_planned_operator_implementations(
    hierarchy: PlanClosure,
    captured_regions: Mapping[int, CapturedFusedProgram],
    observed_plain_node_ids: Iterable[int],
) -> dict[int, tuple[PlannedOperatorImplementation, ...]]:
    """Attach lowering evidence to planner-owned operators by node ID."""

    observed_plain = set(map(int, observed_plain_node_ids))
    implementations = {}
    for item in hierarchy.items:
        if not (
            isinstance(item, PlanClosure)
            and item.name.startswith("region_")
        ):
            continue
        region_index = int(item.name.split("_", 1)[1])
        captured = captured_regions.get(region_index)
        fused_steps: dict[int, list[int]] = {}
        if captured is not None:
            for program in captured.execution_programs:
                for step in program.steps:
                    fused_steps.setdefault(
                        int(step.result_id), []
                    ).append(int(step.step_id))
        region_implementations = []
        for line in item.items:
            if not isinstance(line, PlanLine):
                continue
            for node_id in line.outputs:
                node_id = int(node_id)
                step_ids = tuple(fused_steps.get(node_id, ()))
                region_implementations.append(
                    PlannedOperatorImplementation(
                        node_id,
                        "plain"
                        if node_id in observed_plain
                        else "fused"
                        if step_ids
                        else "structural",
                        step_ids,
                    )
                )
        implementations[region_index] = tuple(
            region_implementations
        )
    return implementations


def _refresh_hierarchy_control_captures(
    closure: PlanClosure,
    shell: Any,
) -> PlanClosure:
    """Update final control captures without rediscovering source structure.

    The pre-capture hierarchy already freezes every callsite, argument/result
    binding and region closure.  Numerical lowering only adds control ABI
    values.  Re-running graph topological discovery for the entire call tree
    at that point is redundant and becomes quadratic on deeply composed
    programs, so preserve the structural plan and refresh capture identities
    directly from each shell's final ControlProgram.
    """

    from .control_source import (
        CallBlock,
        LoopControlBlock,
        LoopBlock,
        ParallelDeployment,
        SequenceBlock,
        StateMachineTick,
        WhileBlock,
    )

    values = set(_control_dependency_value_ids(shell.shell_control_program))

    refreshed_items = []
    for item in closure.items:
        if not isinstance(item, PlanCall):
            refreshed_items.append(item)
            continue
        child = shell.callsite_function_shells.get(item.callsite_id)
        if child is None:
            refreshed_items.append(item)
            continue
        refreshed_items.append(PlanCall(
            item.callsite_id,
            _refresh_hierarchy_control_captures(item.callee, child),
            item.argument_value_ids,
            item.result_value_ids,
            item.argument_bindings,
            item.result_bindings,
            item.enclosing_loop_ids,
        ))
    return PlanClosure(
        closure.name,
        tuple(dict.fromkeys((*closure.captures, *sorted(values)))),
        tuple(refreshed_items),
        closure.closure_id,
    )



def _build_hierarchical_glsl_artifact(shell: Any):
    """Lower a captured call tree into one globally-namespaced shell."""

    compose_started = time.perf_counter()
    shell._profiler.trace(
        path=shell.profile_path,
        section="hierarchical-artifact",
        label="begin",
        fields={},
    )
    shell.hierarchical_compose_failure = None
    hierarchy = shell.hierarchy_plan
    value_table = shell.hierarchy_value_table

    identity_closures: dict[int, tuple[int, str]] = {}
    identity_call_sources: dict[
        tuple[int, int], tuple[int, int]
    ] = {}

    def find_identities(closure: PlanClosure, owner: Any) -> None:
        graph = owner.process_graph.G
        identity = _identity_parameter_projection(graph)
        if identity is not None:
            identity_closures[int(closure.closure_id)] = identity
        for item in closure.items:
            if not isinstance(item, PlanCall):
                continue
            child = owner.callsite_function_shells.get(item.callsite_id)
            if child is not None:
                find_identities(item.callee, child)
                identity = identity_closures.get(
                    int(item.callee.closure_id)
                )
                if identity is not None:
                    source_parameter, _kind = identity
                    caller_source = next((
                        int(caller)
                        for caller, callee in item.argument_bindings
                        if int(callee) == int(source_parameter)
                    ), None)
                    if caller_source is not None:
                        identity_call_sources[(
                            int(closure.closure_id),
                            int(item.callsite_id),
                        )] = (
                            int(closure.closure_id),
                            caller_source,
                        )

    find_identities(hierarchy, shell)
    identity_reduction = reduce_hierarchy_identities(
        hierarchy, set(identity_closures)
    )
    hierarchy = identity_reduction.root
    shell._profiler.trace(
        path=shell.profile_path,
        section="hierarchical-artifact",
        label="identity reduction complete",
        fields={
            "elapsed_ms": round(
                (time.perf_counter() - compose_started) * 1e3,
                3,
            ),
        },
    )
    shell.hierarchy_identity_collapses = (
        identity_reduction.collapsed_callsites
    )
    shell.hierarchy_identity_rounds = identity_reduction.rounds
    shell.hierarchy_remaining_callsite_ids = tuple(
        int(item.callsite_id)
        for item in hierarchy.items
        if isinstance(item, PlanCall)
    )
    controls: dict[int, ControlProgram] = {}
    shells: dict[int, Any] = {}
    closures: dict[int, PlanClosure] = {}
    calls: dict[tuple[int, int], PlanCall] = {}
    argument_sources: dict[tuple[int, int], tuple[int, int]] = {}

    def gather(closure: PlanClosure, owner: Any) -> bool:
        closure_id = int(closure.closure_id)
        closures[closure_id] = closure
        shells[closure_id] = owner
        control = owner.shell_control_program
        if control is None:
            if owner.dispatch_subgraphs:
                shell.hierarchical_compose_failure = {
                    "reason": "missing-control-program",
                    "closure_id": closure_id,
                    "function": owner.process_graph.G.graph.get(
                        "function_name"
                    ),
                    "dispatch_regions": len(owner.dispatch_subgraphs),
                }
                return False
            control = ControlProgram(SequenceBlock(()), ())
        controls[closure_id] = control
        for item in closure.items:
            if not isinstance(item, PlanCall):
                continue
            calls[(closure_id, int(item.callsite_id))] = item
            for caller_local, callee_local in item.argument_bindings:
                argument_sources[(
                    int(item.callee.closure_id),
                    int(callee_local),
                )] = (closure_id, int(caller_local))
            child = owner.callsite_function_shells.get(item.callsite_id)
            if child is None or not gather(item.callee, child):
                return False
        return True

    if not gather(hierarchy, shell):
        return None
    # The loop planner has already performed the control-flow/SSA analysis.
    # Its aliases are therefore authoritative value edges, not hints for a
    # later runtime reconstruction.  Carry them into hierarchy identity
    # resolution before numerical programs are globally namespaced.  In
    # particular, a LoopExit names the planner-selected carried value; mapping
    # it back to the syntactic list initializer would sever the real producer
    # edge and manufacture an external shader input.
    collection_owner_endpoints = {
        (int(closure_id), int(collection))
        for closure_id, control in controls.items()
        for _source, collection, _induction, _start
        in control.collection_bindings
    }
    loop_carried_update_endpoints: set[tuple[int, int]] = set()

    def collect_loop_carried_updates(
        closure_id: int,
        block: Any,
    ) -> None:
        carried = (
            block.carried_aliases
            if isinstance(block, (LoopBlock, WhileBlock))
            else ()
        )
        loop_carried_update_endpoints.update(
            (int(closure_id), int(updated))
            for updated, _initial in carried
        )
        if isinstance(block, SequenceBlock):
            for child in block.blocks:
                collect_loop_carried_updates(closure_id, child)
        elif isinstance(block, LoopBlock):
            collect_loop_carried_updates(closure_id, block.body)
        elif isinstance(block, WhileBlock):
            collect_loop_carried_updates(closure_id, block.condition)
            collect_loop_carried_updates(closure_id, block.body)

    for closure_id, control in controls.items():
        collect_loop_carried_updates(int(closure_id), control.root)
    control_alias_sources = {
        (int(closure_id), int(updated)): (
            int(closure_id),
            int(initial),
        )
        for closure_id, control in controls.items()
        for updated, initial in control.value_aliases
        if (int(closure_id), int(updated))
        not in collection_owner_endpoints
        and (int(closure_id), int(updated))
        not in loop_carried_update_endpoints
    }
    for closure_id, owner in shells.items():
        selected_returns = tuple(getattr(
            owner, "_captured_return_value_ids", ()
        ))
        output_names = tuple(
            owner.process_graph.G.graph.get("function_outputs", ())
        )
        if len(selected_returns) != 1 or len(output_names) != 1:
            continue
        selected = int(selected_returns[0])
        identities = owner.process_graph.G.graph.get(
            "identity_table", {}
        ) or {}
        for output_id in identities.get(output_names[0], ()):
            output_id = int(output_id)
            if output_id == selected:
                continue
            # A statically selected early return is the sole producer of a
            # single-output invocation.  Normalization may still correlate
            # the function output with a later, unexecuted return expression;
            # collapse that stale endpoint before global value projection so
            # it cannot veto the caller-argument redirect.
            control_alias_sources[(int(closure_id), output_id)] = (
                int(closure_id), selected
            )
    for call in calls.values():
        child_id = int(call.callee.closure_id)
        child_owner = shells[child_id]
        selected_returns = tuple(getattr(
            child_owner, "_captured_return_value_ids", ()
        ))
        if len(selected_returns) != 1 or len(call.result_bindings) != 1:
            continue
        callee_result, _caller_result = call.result_bindings[0]
        selected = int(selected_returns[0])
        if int(callee_result) != selected:
            control_alias_sources[(child_id, int(callee_result))] = (
                child_id, selected
            )
    for closure_id, owner in shells.items():
        parameter_inputs: dict[str, list[int]] = {}
        for local_id, data in owner.process_graph.G.nodes(data=True):
            attributes = data.get("attributes") or {}
            if (
                data.get("type") != "Input"
                or attributes.get("binding_kind") != "parameter"
            ):
                continue
            name = attributes.get("binding_name") or data.get("label")
            if name is not None:
                parameter_inputs.setdefault(str(name), []).append(
                    int(local_id)
                )
        for local_ids in parameter_inputs.values():
            bound = [
                local_id
                for local_id in local_ids
                if (int(closure_id), int(local_id)) in argument_sources
            ]
            if len(bound) != 1:
                continue
            canonical = int(bound[0])
            for local_id in local_ids:
                if int(local_id) == canonical:
                    continue
                # AST normalization may retain multiple Input occurrences for
                # one source parameter while PlanCall binds only its canonical
                # occurrence.  They are the same lexical SSA source; route
                # every duplicate through the explicit argument edge.
                control_alias_sources[(
                    int(closure_id), int(local_id)
                )] = (int(closure_id), canonical)
    shell._profiler.trace(
        path=shell.profile_path,
        section="hierarchical-artifact",
        label="control gather complete",
        fields={
            "closures": len(closures),
            "elapsed_ms": round(
                (time.perf_counter() - compose_started) * 1e3,
                3,
            ),
        },
    )
    global_redirects = {}
    for identity_endpoint, source_endpoint in identity_call_sources.items():
        try:
            identity_global = value_table.global_id(*identity_endpoint)
            source_global = value_table.global_id(*source_endpoint)
        except KeyError:
            continue
        if identity_global != source_global:
            global_redirects[int(identity_global)] = int(source_global)

    def redirected(global_id: int) -> int:
        seen = set()
        current = int(global_id)
        while current in global_redirects and current not in seen:
            seen.add(current)
            current = global_redirects[current]
        return current

    if global_redirects:
        value_table = type(value_table)(tuple(
            (closure_id, local_id, redirected(global_id))
            for closure_id, local_id, global_id
            in value_table.correlations
        ))
    shell.hierarchical_effective_value_table = value_table
    hierarchical = compose_hierarchical_control(
        hierarchy, controls, value_table
    )
    shell._profiler.trace(
        path=shell.profile_path,
        section="hierarchical-artifact",
        label="control composition complete",
        fields={
            "regions": len(hierarchical.program.region_indices),
            "elapsed_ms": round(
                (time.perf_counter() - compose_started) * 1e3,
                3,
            ),
        },
    )
    region_lookup = {
        (closure_id, local_region): global_region
        for closure_id, local_region, global_region
        in hierarchical.region_correlations
    }
    next_value = 1 + max(
        (
            global_id
            for _closure, _local, global_id
            in value_table.correlations
        ),
        default=-1,
    )
    private_values: dict[tuple[int, int], int] = {}
    capture_values: dict[int, int] = {}
    endpoint_meta: dict[tuple[int, int], Any] = {}
    for closure_id, owner in shells.items():
        for value_id, meta in owner.compiled_feed_meta.items():
            endpoint_meta.setdefault(
                (int(closure_id), int(value_id)),
                meta,
            )
        for captured in owner.captured_region_programs.values():
            for program in (captured.program, *captured.stages):
                for value_id, meta in (program.meta or {}).items():
                    endpoint_meta.setdefault(
                        (int(closure_id), int(value_id)),
                        meta,
                    )

    def global_value(closure_id: int, local_id: int) -> int:
        nonlocal next_value
        try:
            return value_table.global_id(closure_id, local_id)
        except KeyError:
            endpoint = (int(closure_id), int(local_id))
            owner = shells.get(int(closure_id))
            is_synthetic = endpoint in synthetic_field_paths
            if (
                owner is not None
                and int(local_id) not in owner.process_graph.G
                and not is_synthetic
            ):
                # The discovery tape is one global program observation.  A
                # transient result identity therefore means the same value
                # when it crosses function compartments; closure-namespacing
                # it severs the producer/consumer edge and fabricates a shell
                # input.  Graph-local IDs and compiler-created field
                # endpoints remain scoped, but raw tape identities are
                # canonical across the hierarchy.
                capture_id = int(local_id)
                if capture_id not in capture_values:
                    capture_values[capture_id] = next_value
                    next_value += 1
                return capture_values[capture_id]
            key = endpoint
            if key not in private_values:
                private_values[key] = next_value
                next_value += 1
            return private_values[key]

    # Python classes crossing function boundaries are logical aggregate SSA
    # values, not runtime buffers and never host callables.  Resolve only the
    # field that an ordinary Attribute expression requests, following call
    # arguments and returns through the hierarchy.  This keeps source programs
    # normal Python while making aggregate erasure a general compiler rule.
    resolving: set[tuple[int, int]] = set()
    resolved_leaves: dict[
        tuple[int, int],
        dict[tuple[object, ...], tuple[int, int]],
    ] = {}
    synthetic_field_endpoints: dict[
        tuple[int, int, tuple[str, ...]], tuple[int, int]
    ] = {}
    synthetic_field_paths: dict[
        tuple[int, int], tuple[tuple[int, int], tuple[str, ...]]
    ] = {}
    synthetic_static_values: dict[tuple[int, int], Any] = {}
    next_synthetic_local = -1
    unresolved_static = object()
    metadata_links = {
        **identity_call_sources,
        **argument_sources,
    }
    changed = True
    while changed:
        changed = False
        for target, source in metadata_links.items():
            target_meta = endpoint_meta.get(target)
            source_meta = endpoint_meta.get(source)
            if target_meta is None and source_meta is not None:
                endpoint_meta[target] = source_meta
                changed = True
            elif source_meta is None and target_meta is not None:
                endpoint_meta[source] = target_meta
                changed = True
        by_global: dict[int, list[tuple[int, int]]] = {}
        for closure_id, local_id, global_id in value_table.correlations:
            by_global.setdefault(int(global_id), []).append(
                (int(closure_id), int(local_id))
            )
        for endpoints in by_global.values():
            known = next(
                (
                    endpoint_meta[endpoint]
                    for endpoint in endpoints
                    if endpoint in endpoint_meta
                ),
                None,
            )
            if known is None:
                continue
            for endpoint in endpoints:
                if endpoint not in endpoint_meta:
                    endpoint_meta[endpoint] = known
                    changed = True
    global_meta: dict[int, Any] = {}
    for closure_id, local_id, global_id in value_table.correlations:
        meta = endpoint_meta.get((int(closure_id), int(local_id)))
        if meta is not None:
            global_meta.setdefault(int(global_id), meta)

    def value_meta(
        endpoint: tuple[int, int],
        visiting: set[tuple[int, int]] | None = None,
    ) -> Any | None:
        endpoint = (int(endpoint[0]), int(endpoint[1]))
        visiting = set() if visiting is None else set(visiting)
        if endpoint in visiting:
            return None
        visiting.add(endpoint)
        direct = endpoint_meta.get(endpoint)
        if direct is not None:
            return direct
        source = metadata_links.get(endpoint)
        if source is not None:
            inherited = value_meta(source, visiting)
            if inherited is not None:
                return inherited
        try:
            correlated = value_table.global_id(*endpoint)
        except KeyError:
            correlated = None
        if correlated is not None:
            inherited = global_meta.get(int(correlated))
            if inherited is not None:
                return inherited
        owner = shells.get(endpoint[0])
        graph = None if owner is None else owner.process_graph.G
        if graph is None or endpoint[1] not in graph:
            return None
        data = graph.nodes[endpoint[1]]
        parents = tuple(
            (int(parent), str(role))
            for parent, role in (data.get("parents") or ())
            if str(role) not in {"callee", "func", "definition"}
        )
        if data.get("type") in {"Indexed", "indexed"}:
            base = next((
                parent for parent, role in parents if role == "base"
            ), None)
            index_node = next((
                parent for parent, role in parents if role == "index"
            ), None)
            index = None
            if index_node is not None and index_node in graph:
                try:
                    index = _constant_value(graph.nodes[index_node])
                except KeyError:
                    pass
            if base is not None and isinstance(index, int):
                call = calls.get((endpoint[0], base))
                if call is not None:
                    child_id = int(call.callee.closure_id)
                    child_graph = shells[child_id].process_graph.G
                    output_names = tuple(
                        child_graph.graph.get("function_outputs", ())
                    )
                    if -len(output_names) <= index < len(output_names):
                        output_name = output_names[index]
                        identities = (
                            child_graph.graph.get("identity_table") or {}
                        )
                        output_ids = identities.get(output_name, ())
                        if output_ids:
                            inherited = value_meta(
                                (child_id, int(output_ids[-1])),
                                visiting,
                            )
                            if inherited is not None:
                                return inherited
        call = calls.get(endpoint)
        if call is not None:
            child_id = int(call.callee.closure_id)
            child_graph = shells[child_id].process_graph.G
            output_names = tuple(
                child_graph.graph.get("function_outputs", ())
            )
            if len(output_names) == 1:
                identities = child_graph.graph.get("identity_table") or {}
                output_ids = identities.get(output_names[0], ())
                if output_ids:
                    return value_meta(
                        (child_id, int(output_ids[-1])),
                        visiting,
                    )
        return None

    def field_endpoint(
        source: tuple[int, int],
        field: str,
    ) -> tuple[int, int]:
        nonlocal next_synthetic_local
        rooted = synthetic_field_paths.get(source)
        if rooted is None:
            root, path = source, ()
        else:
            root, path = rooted
        full_path = (*path, str(field))
        key = (int(root[0]), int(root[1]), full_path)
        endpoint = synthetic_field_endpoints.get(key)
        if endpoint is None:
            endpoint = (int(root[0]), next_synthetic_local)
            next_synthetic_local -= 1
            synthetic_field_endpoints[key] = endpoint
            synthetic_field_paths[endpoint] = (root, full_path)
        return endpoint

    def static_endpoint_value(
        endpoint: tuple[int, int],
        visiting: set[tuple[int, int]] | None = None,
    ) -> Any:
        visiting = set() if visiting is None else set(visiting)
        endpoint = (int(endpoint[0]), int(endpoint[1]))
        if endpoint in visiting:
            return unresolved_static
        visiting.add(endpoint)
        rooted = synthetic_field_paths.get(endpoint)
        if rooted is not None:
            if endpoint in synthetic_static_values:
                return synthetic_static_values[endpoint]
            root, path = rooted
            root_owner = shells.get(int(root[0]))
            root_graph = (
                None if root_owner is None
                else root_owner.process_graph.G
            )
            root_data = (
                None
                if root_graph is None or int(root[1]) not in root_graph
                else root_graph.nodes[int(root[1])]
            )
            root_name = (
                None
                if root_data is None
                else (root_data.get("attributes") or {}).get(
                    "binding_name"
                )
            )
            static_fields = (
                {}
                if root_owner is None
                else getattr(
                    root_owner, "_capture_input_static_fields", {}
                )
            )
            static_key = (str(root_name), tuple(map(str, path)))
            if root_name is not None and static_key in static_fields:
                return static_fields[static_key]
            projected = leaves(*root)
            target = projected.get(tuple(path))
            if target is not None and target != endpoint:
                return static_endpoint_value(target, visiting)
            return unresolved_static
        source = identity_call_sources.get(endpoint)
        if source is None:
            source = argument_sources.get(endpoint)
        if source is not None:
            return static_endpoint_value(source, visiting)
        owner = shells.get(endpoint[0])
        if owner is None or endpoint[1] not in owner.process_graph.G:
            return unresolved_static
        graph = owner.process_graph.G
        data = graph.nodes[endpoint[1]]
        if data.get("type") in {
            "Constant", "Const", "const", "StaticReference"
        }:
            try:
                return _constant_value(data)
            except KeyError:
                return unresolved_static
        if data.get("type") == "Input":
            name = (data.get("attributes") or {}).get("binding_name")
            specializations = (
                graph.graph.get("planner_specializations") or {}
            )
            if name in specializations:
                return specializations[name]
            defaults = graph.graph.get("parameter_defaults") or {}
            if name in defaults:
                return defaults[name]
        parents = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
            if str(role) not in {"callee", "func", "definition"}
        }
        expression = data.get("expr_obj")
        if data.get("type") == "Phi":
            test = parents.get("test")
            body = parents.get("body")
            orelse = parents.get("orelse")
            predicate = (
                unresolved_static
                if test is None
                else static_endpoint_value((endpoint[0], test), visiting)
            )
            selected = (
                body
                if predicate is not unresolved_static and bool(predicate)
                else (
                    orelse
                    if predicate is not unresolved_static
                    else None
                )
            )
            if selected is not None:
                return static_endpoint_value(
                    (endpoint[0], selected), visiting
                )
            if body is not None and orelse is not None:
                body_value = static_endpoint_value(
                    (endpoint[0], body), visiting
                )
                else_value = static_endpoint_value(
                    (endpoint[0], orelse), visiting
                )
                if (
                    body_value is not unresolved_static
                    and else_value is not unresolved_static
                    and type(body_value) is type(else_value)
                    and body_value == else_value
                ):
                    return body_value
        if isinstance(expression, ast.Attribute) and expression.attr in {
            "shape", "ndim", "ndims", "dtype", "device",
        }:
            base = next((
                parent
                for role, parent in parents.items()
                if role in {"value", "base", "operand", "object"}
            ), None)
            if base is not None:
                meta = value_meta((endpoint[0], base))
                if meta is None:
                    projected = leaves(endpoint[0], base)
                    if len(projected) == 1 and () in projected:
                        meta = value_meta(projected[()])
                if meta is not None:
                    if expression.attr == "shape" and meta.shape is not None:
                        return tuple(int(size) for size in meta.shape)
                    if expression.attr in {"ndim", "ndims"} and (
                        meta.shape is not None
                    ):
                        return len(meta.shape)
                    if expression.attr == "dtype" and meta.dtype is not None:
                        return str(meta.dtype)
                    if expression.attr == "device":
                        device = getattr(meta, "device", None)
                        if device is not None:
                            return device
        if (
            isinstance(expression, ast.Subscript)
            or data.get("type") in {"Indexed", "indexed"}
        ):
            base = next((
                parent
                for role, parent in parents.items()
                if role in {"value", "base", "operand", "object"}
            ), None)
            index = next((
                parent
                for role, parent in parents.items()
                if role in {"slice", "index", "subscript"}
            ), None)
            if base is not None and index is not None:
                container = static_endpoint_value(
                    (endpoint[0], base), visiting
                )
                position = static_endpoint_value(
                    (endpoint[0], index), visiting
                )
                if (
                    container is not unresolved_static
                    and position is not unresolved_static
                    and isinstance(position, (int, slice))
                ):
                    try:
                        return container[position]
                    except (IndexError, KeyError, TypeError):
                        pass
        if isinstance(expression, ast.UnaryOp):
            operand = next((
                parent
                for role, parent in parents.items()
                if role in {"operand", "value"}
            ), None)
            fixed = (
                unresolved_static
                if operand is None
                else static_endpoint_value((endpoint[0], operand), visiting)
            )
            if fixed is not unresolved_static:
                if isinstance(expression.op, ast.USub):
                    return -fixed
                if isinstance(expression.op, ast.UAdd):
                    return +fixed
                if isinstance(expression.op, ast.Not):
                    return not fixed
        if isinstance(expression, ast.BinOp):
            left = parents.get("lhs", parents.get("left"))
            right = parents.get("rhs", parents.get("right"))
            lhs = (
                unresolved_static
                if left is None
                else static_endpoint_value((endpoint[0], left), visiting)
            )
            rhs = (
                unresolved_static
                if right is None
                else static_endpoint_value((endpoint[0], right), visiting)
            )
            if lhs is not unresolved_static and rhs is not unresolved_static:
                operation = expression.op
                try:
                    if isinstance(operation, ast.Add):
                        return lhs + rhs
                    if isinstance(operation, ast.Sub):
                        return lhs - rhs
                    if isinstance(operation, ast.Mult):
                        return lhs * rhs
                    if isinstance(operation, ast.Div):
                        return lhs / rhs
                    if isinstance(operation, ast.FloorDiv):
                        return lhs // rhs
                    if isinstance(operation, ast.Mod):
                        return lhs % rhs
                    if isinstance(operation, ast.Pow):
                        return lhs ** rhs
                except (ArithmeticError, TypeError, ValueError):
                    pass
        if isinstance(expression, ast.Compare) and len(expression.ops) == 1:
            left = parents.get("lhs", parents.get("left"))
            right = parents.get("rhs", parents.get("right"))
            lhs = (
                unresolved_static
                if left is None
                else static_endpoint_value((endpoint[0], left), visiting)
            )
            rhs = (
                unresolved_static
                if right is None
                else static_endpoint_value((endpoint[0], right), visiting)
            )
            if lhs is not unresolved_static and rhs is not unresolved_static:
                operation = expression.ops[0]
                if isinstance(operation, ast.Eq):
                    return lhs == rhs
                if isinstance(operation, ast.NotEq):
                    return lhs != rhs
                if isinstance(operation, ast.Lt):
                    return lhs < rhs
                if isinstance(operation, ast.LtE):
                    return lhs <= rhs
                if isinstance(operation, ast.Gt):
                    return lhs > rhs
                if isinstance(operation, ast.GtE):
                    return lhs >= rhs
                if isinstance(operation, ast.Is):
                    return lhs is rhs
                if isinstance(operation, ast.IsNot):
                    return lhs is not rhs
        if isinstance(expression, ast.BoolOp):
            operands = [
                parent
                for role, parent in sorted(parents.items())
                if role.startswith("value")
            ]
            if operands:
                result = (
                    False if isinstance(expression.op, ast.Or) else True
                )
                for operand in operands:
                    fixed = static_endpoint_value(
                        (endpoint[0], operand), visiting
                    )
                    if fixed is unresolved_static:
                        return unresolved_static
                    result = bool(fixed)
                    if (
                        isinstance(expression.op, ast.Or) and result
                    ) or (
                        isinstance(expression.op, ast.And) and not result
                    ):
                        return result
                return result
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and expression.func.attr in {"numel", "ndim", "ndims"}
        ):
            base = next((
                parent
                for role, parent in parents.items()
                if role in {"operand", "value", "base", "object"}
            ), None)
            if base is not None:
                meta = value_meta((endpoint[0], base))
                if meta is None:
                    projected = leaves(endpoint[0], base)
                    if len(projected) == 1 and () in projected:
                        meta = value_meta(projected[()])
                shape = None if meta is None else meta.shape
                if shape is not None:
                    if expression.func.attr in {"ndim", "ndims"}:
                        return len(shape)
                    count = 1
                    for extent in shape:
                        count *= int(extent)
                    return count
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
        ):
            builtin_name = expression.func.id
            positional = tuple(
                parent
                for role, parent in sorted(
                    parents.items(),
                    key=lambda item: (
                        _positional_argument_index(item[0])
                        if _positional_argument_index(item[0]) is not None
                        else 1 << 30
                    ),
                )
                if _positional_argument_index(role) is not None
            )
            fixed_arguments = tuple(
                static_endpoint_value((endpoint[0], argument), visiting)
                for argument in positional
            )
            if builtin_name == "getattr" and len(fixed_arguments) >= 2:
                attribute_name = fixed_arguments[1]
                base_id = positional[0]
                if (
                    isinstance(attribute_name, str)
                    and not attribute_name.startswith("_")
                ):
                    meta = value_meta((endpoint[0], base_id))
                    if meta is not None:
                        if attribute_name == "shape" and meta.shape is not None:
                            return tuple(int(size) for size in meta.shape)
                        if attribute_name in {"ndim", "ndims"} and (
                            meta.shape is not None
                        ):
                            return len(meta.shape)
                        if attribute_name == "dtype" and meta.dtype is not None:
                            return str(meta.dtype)
                        if attribute_name == "device":
                            device = getattr(meta, "device", None)
                            if device is not None:
                                return device
                    owner_value = fixed_arguments[0]
                    if owner_value is not unresolved_static and isinstance(
                        owner_value,
                        (tuple, list, dict, range, str, bytes),
                    ):
                        try:
                            return getattr(owner_value, attribute_name)
                        except AttributeError:
                            pass
                if (
                    len(fixed_arguments) >= 3
                    and fixed_arguments[2] is not unresolved_static
                ):
                    return fixed_arguments[2]
            if builtin_name == "isinstance" and positional:
                candidate = fixed_arguments[0]
                type_expression = (
                    expression.args[1] if len(expression.args) > 1 else None
                )

                def safe_types(node: ast.AST | None):
                    if isinstance(node, ast.Name):
                        return {
                            "bool": (bool,), "int": (int,),
                            "float": (float,), "str": (str,),
                            "bytes": (bytes,), "tuple": (tuple,),
                            "list": (list,), "dict": (dict,),
                            "set": (set,), "range": (range,),
                        }.get(node.id)
                    if isinstance(node, ast.Tuple):
                        collected = tuple(
                            item
                            for element in node.elts
                            for item in (safe_types(element) or ())
                        )
                        return collected or None
                    return None

                accepted = safe_types(type_expression)
                if candidate is not unresolved_static and accepted is not None:
                    return isinstance(candidate, accepted)
            if all(
                argument is not unresolved_static
                for argument in fixed_arguments
            ):
                try:
                    if builtin_name in {"bool", "int", "float"}:
                        return {"bool": bool, "int": int, "float": float}[
                            builtin_name
                        ](*fixed_arguments)
                    if builtin_name in {"tuple", "list", "set"}:
                        constructor = {
                            "tuple": tuple, "list": list, "set": set,
                        }[builtin_name]
                        return constructor(*fixed_arguments)
                    if builtin_name == "len":
                        return len(*fixed_arguments)
                    if builtin_name == "range":
                        return range(*fixed_arguments)
                    if builtin_name == "enumerate":
                        return tuple(enumerate(*fixed_arguments))
                    if builtin_name == "zip":
                        return tuple(zip(*fixed_arguments))
                    if builtin_name == "sorted":
                        return sorted(*fixed_arguments)
                    if builtin_name == "max":
                        return max(*fixed_arguments)
                    if builtin_name == "min":
                        return min(*fixed_arguments)
                    if builtin_name == "all":
                        return all(*fixed_arguments)
                    if builtin_name == "any":
                        return any(*fixed_arguments)
                    if builtin_name == "slice":
                        return slice(*fixed_arguments)
                except (TypeError, ValueError, OverflowError):
                    pass
        if isinstance(data.get("expr_obj"), ast.Attribute):
            projected = leaves(*endpoint)
            if len(projected) == 1 and () in projected:
                target = projected[()]
                if target != endpoint:
                    return static_endpoint_value(target, visiting)
        return unresolved_static

    def static_predicate(
        closure_id: int,
        local_id: int,
    ) -> bool | None:
        owner = shells[int(closure_id)]
        graph = owner.process_graph.G
        data = graph.nodes[int(local_id)]
        expression = data.get("expr_obj")
        # Ordinary Python selection such as ``cached or build()`` becomes a
        # Phi whose test is the cached SSA value itself.  If that endpoint is
        # a source-static scalar (most importantly an optional parameter with
        # default None), resolve its truth value before considering comparison
        # syntax.  Tensor truth remains deliberately unresolved.
        direct_leaves = leaves(int(closure_id), int(local_id))
        if len(direct_leaves) == 1 and () in direct_leaves:
            direct = static_endpoint_value(direct_leaves[()])
            if direct is not unresolved_static and (
                direct is None
                or isinstance(direct, (bool, int, float, str, bytes))
            ):
                return bool(direct)
        if isinstance(expression, ast.UnaryOp) and isinstance(
            expression.op, ast.Not
        ):
            operand = next(
                (
                    int(parent)
                    for parent, role in (data.get("parents") or ())
                    if str(role) in {"operand", "value"}
                ),
                None,
            )
            if operand is not None:
                resolved = static_predicate(int(closure_id), operand)
                if resolved is not None:
                    return not resolved
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id == "isinstance"
            and len(expression.args) == 2
            and isinstance(expression.args[1], ast.Name)
            and expression.args[1].id == "AbstractTensor"
        ):
            argument = next(
                (
                    int(parent)
                    for parent, role in (data.get("parents") or ())
                    if _positional_argument_index(str(role)) == 0
                ),
                None,
            )
            if argument is not None:
                projected = leaves(int(closure_id), argument)
                if len(projected) == 1 and () in projected:
                    # Tensor metadata is the compiler's structural type fact;
                    # this does not inspect or specialize the captured value.
                    if value_meta(projected[()]) is not None:
                        return True
                argument_data = graph.nodes[int(argument)]
                argument_name = (
                    argument_data.get("attributes") or {}
                ).get("binding_name")
                if (
                    argument_data.get("type") == "Input"
                    and argument_name in getattr(
                        owner, "_capture_tensor_input_names", ()
                    )
                ):
                    return True
        if not (
            isinstance(expression, ast.Compare)
            and len(expression.ops) == 1
            and len(expression.comparators) == 1
        ):
            return None
        if (
            isinstance(expression.left, ast.Name)
            and isinstance(expression.comparators[0], ast.Constant)
            and expression.comparators[0].value is None
            and isinstance(expression.ops[0], (ast.Is, ast.IsNot))
        ):
            identities = graph.graph.get("identity_table") or {}
            parameter_ids = identities.get(expression.left.id, ())
            if parameter_ids and (
                int(closure_id), int(parameter_ids[0])
            ) in argument_sources:
                # An explicitly bound argument is a value edge, regardless of
                # whether its runtime payload is a tensor or aggregate.  It
                # cannot be the omitted/default None branch.
                return isinstance(expression.ops[0], ast.IsNot)
        parents = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        left = parents.get("lhs")
        right = parents.get("rhs")
        if left is None or right is None:
            return None
        left_leaves = leaves(int(closure_id), left)
        right_leaves = leaves(int(closure_id), right)
        if not (
            len(left_leaves) == len(right_leaves) == 1
            and () in left_leaves
            and () in right_leaves
        ):
            if isinstance(expression.ops[0], (ast.Is, ast.IsNot)):
                shell._profiler.trace(
                    path=owner.profile_path,
                    section="hierarchy-static-predicate",
                    label="non-scalar-identity-comparison",
                    fields={
                        "node": int(local_id),
                        "lhs": tuple(left_leaves.items()),
                        "rhs": tuple(right_leaves.items()),
                    },
                )
            return None
        lhs = static_endpoint_value(left_leaves[()])
        rhs = static_endpoint_value(right_leaves[()])
        if (
            lhs is unresolved_static or rhs is unresolved_static
        ) and isinstance(operation := expression.ops[0], (ast.Is, ast.IsNot)):
            shell._profiler.trace(
                path=owner.profile_path,
                section="hierarchy-static-predicate",
                label="unresolved-identity-comparison",
                fields={
                    "node": int(local_id),
                    "lhs_endpoint": left_leaves[()],
                    "rhs_endpoint": right_leaves[()],
                    "lhs": (
                        "<unresolved>"
                        if lhs is unresolved_static
                        else repr(lhs)
                    ),
                    "rhs": (
                        "<unresolved>"
                        if rhs is unresolved_static
                        else repr(rhs)
                    ),
                },
            )
        if lhs is unresolved_static or rhs is unresolved_static:
            return None
        operation = expression.ops[0]
        if isinstance(operation, ast.Is):
            return lhs is rhs
        if isinstance(operation, ast.IsNot):
            return lhs is not rhs
        if isinstance(operation, ast.Eq):
            return lhs == rhs
        if isinstance(operation, ast.NotEq):
            return lhs != rhs
        return None

    def leaves(
        closure_id: int,
        local_id: int,
    ) -> dict[tuple[object, ...], tuple[int, int]]:
        key = (int(closure_id), int(local_id))
        cached = resolved_leaves.get(key)
        if cached is not None:
            return cached
        if key in resolving:
            return {(): key}
        resolving.add(key)
        owner = shells[key[0]]
        graph = owner.process_graph.G
        result: dict[tuple[object, ...], tuple[int, int]] = {}
        source = identity_call_sources.get(key)
        if source is None:
            source = argument_sources.get(key)
        if source is None:
            source = control_alias_sources.get(key)
        if key in collection_owner_endpoints:
            # Planner collection/publication is a producer boundary.  In
            # particular, a LoopExit may retain the source initializer on its
            # syntactic ``value`` edge, but that initializer neither owns nor
            # produces the resident collection.
            result = {(): key}
        elif source is not None:
            result = leaves(*source)
        elif key[1] not in graph:
            result = {(): key}
        else:
            data = graph.nodes[key[1]]
            expression = data.get("expr_obj")
            attributes = data.get("attributes") or {}
            parents = tuple(
                (int(parent), str(role))
                for parent, role in (data.get("parents") or ())
                if str(role) not in {"callee", "func", "definition"}
            )
            if data.get("type") in {"LoopExit", "LoopResult"}:
                # LoopExit is a control-qualified SSA identity, not a new
                # numerical allocation.  The planner owns the control edge;
                # hierarchy/value reduction must carry the value edge through
                # so callers, returns, stream publications, and numerical
                # regions all name the same resident range.  Leaving the
                # wrapper as a distinct value fabricates an unproduced shell
                # slot and loses the producer's storage contract.
                value_parent = next(
                    (
                        parent
                        for parent, role in parents
                        if role == "value"
                    ),
                    None,
                )
                control_parent = next((
                    parent
                    for parent, role in parents
                    if role == "control"
                ), None)
                if (
                    data.get("type") == "LoopResult"
                    and control_parent is not None
                    and getattr(
                        owner, "_captured_loop_iterations", {}
                    ).get(int(control_parent)) == 0
                ):
                    binding_name = attributes.get("binding_name")
                    control_attributes = (
                        graph.nodes[int(control_parent)].get("attributes")
                        or {}
                    )
                    carried = (
                        control_attributes.get("loop_carried_bindings")
                        or {}
                    ).get(binding_name)
                    if carried is not None:
                        value_parent = int(carried[0])
                if (
                    value_parent is not None
                    and key not in collection_owner_endpoints
                ):
                    result = leaves(key[0], value_parent)
            elif data.get("type") == "Phi":
                test_id = next((
                    parent for parent, role in parents if role == "test"
                ), None)
                predicate = (
                    None
                    if test_id is None
                    else static_predicate(key[0], test_id)
                )
                selected_role = (
                    None
                    if predicate is None
                    else ("body" if predicate else "orelse")
                )
                selected = next((
                    parent
                    for parent, role in parents
                    if role == selected_role
                ), None)
                if selected is None:
                    phi_meta = value_meta(key)
                    if phi_meta is not None:
                        candidates = [
                            parent
                            for parent, role in parents
                            if role in {"body", "orelse"}
                            and (
                                candidate_meta := value_meta(
                                    (key[0], parent)
                                )
                            ) is not None
                            and tuple(candidate_meta.shape or ())
                            == tuple(phi_meta.shape or ())
                            and str(candidate_meta.dtype)
                            == str(phi_meta.dtype)
                        ]
                        if len(candidates) == 1:
                            selected = candidates[0]
                if selected is not None:
                    result = leaves(key[0], selected)
            elif isinstance(expression, ast.BoolOp):
                # Python ``a or b`` / ``a and b`` returns one operand; it is
                # not a numerical operation.  When operand truth is a source
                # fact (for example an optional parameter defaulting to
                # None), preserve the selected operand's SSA identity so
                # aggregate fields and tensor dependencies remain visible.
                operands = [
                    parent
                    for parent, role in parents
                    if (
                        str(role).startswith("value")
                        or str(role).startswith("arg")
                        or str(role) in {"operand", "values"}
                    )
                ]
                if not operands:
                    operands = [parent for parent, _role in parents]
                selected = None
                for index, operand in enumerate(operands):
                    projected = leaves(key[0], operand)
                    fixed = (
                        unresolved_static
                        if len(projected) != 1 or () not in projected
                        else static_endpoint_value(projected[()])
                    )
                    last = index == len(operands) - 1
                    if last:
                        selected = operand
                        break
                    if fixed is unresolved_static:
                        break
                    truth = bool(fixed)
                    if (
                        isinstance(expression.op, ast.Or) and truth
                    ) or (
                        isinstance(expression.op, ast.And) and not truth
                    ):
                        selected = operand
                        break
                if selected is not None:
                    result = leaves(key[0], selected)
            elif isinstance(expression, ast.IfExp):
                # Optional-parameter selection is a hierarchy fact when the
                # caller explicitly supplied that parameter.  Collapse
                # ``x if x is not None else ...`` to the supplied SSA
                # reference without consulting the captured tensor value.
                test = expression.test
                selected_role = None
                if (
                    isinstance(test, ast.Compare)
                    and len(test.ops) == 1
                    and len(test.comparators) == 1
                    and isinstance(test.left, ast.Name)
                    and isinstance(test.comparators[0], ast.Constant)
                    and test.comparators[0].value is None
                    and isinstance(test.ops[0], (ast.Is, ast.IsNot))
                ):
                    identities = graph.graph.get("identity_table") or {}
                    parameter_ids = identities.get(test.left.id, ())
                    if parameter_ids:
                        parameter_id = int(parameter_ids[0])
                        explicitly_supplied = (
                            (key[0], parameter_id) in argument_sources
                        )
                        if explicitly_supplied:
                            selected_role = (
                                "body"
                                if isinstance(test.ops[0], ast.IsNot)
                                else "orelse"
                            )
                        elif (
                            test.left.id
                            in (
                                graph.graph.get("parameter_defaults")
                                or {}
                            )
                        ):
                            default_is_none = (
                                graph.graph["parameter_defaults"][
                                    test.left.id
                                ]
                                is None
                            )
                            predicate = (
                                not default_is_none
                                if isinstance(test.ops[0], ast.IsNot)
                                else default_is_none
                            )
                            selected_role = (
                                "body" if predicate else "orelse"
                            )
                if selected_role is None:
                    test_id = next((
                        parent
                        for parent, role in parents
                        if role == "test"
                    ), None)
                    predicate = (
                        None
                        if test_id is None
                        else static_predicate(key[0], test_id)
                    )
                    if predicate is not None:
                        selected_role = (
                            "body" if predicate else "orelse"
                        )
                selected = next((
                    parent
                    for parent, role in parents
                    if role == selected_role
                ), None)
                if selected is not None:
                    result = leaves(key[0], selected)
            elif (
                attributes.get("materialization_kind")
                in {"unrolled_loop", "retained_loop_aggregate"}
            ):
                materialized = tuple(
                    int(value_id)
                    for value_id in attributes.get(
                        "materialized_value_ids", ()
                    )
                )
                for index, parent in enumerate(materialized):
                    for path, endpoint in leaves(
                        key[0], parent
                    ).items():
                        result[(int(index), *path)] = endpoint
            elif attributes.get("producer_kind") == "aggregate":
                elements = tuple(
                    int(value_id)
                    for value_id in attributes.get(
                        "aggregate_leaf_value_ids",
                        (),
                    )
                )
                for index, parent in enumerate(elements):
                    for path, endpoint in leaves(
                        key[0], parent
                    ).items():
                        result[(int(index), *path)] = endpoint
            elif data.get("type") in {"Indexed", "indexed"}:
                base = next((
                    parent for parent, role in parents
                    if role in {"base", "value", "object"}
                ), None)
                index = next((
                    parent for parent, role in parents
                    if role in {"index", "slice", "subscript"}
                ), None)
                fixed_index = (
                    unresolved_static
                    if index is None
                    else static_endpoint_value((key[0], index))
                )
                if (
                    base is not None
                    and isinstance(fixed_index, int)
                ):
                    projected = leaves(key[0], base)
                    result = {
                        path[1:]: endpoint
                        for path, endpoint in projected.items()
                        if path and path[0] == fixed_index
                    }
            elif isinstance(expression, ast.Attribute):
                base = next((
                    parent for parent, role in parents
                    if role in {"value", "base", "operand", "object"}
                ), None)
                projected = {} if base is None else leaves(key[0], base)
                field = str(expression.attr)
                result = {
                    path[1:]: endpoint
                    for path, endpoint in projected.items()
                    if path and str(path[0]) == field
                }
                if (
                    not result
                    and len(projected) == 1
                    and () in projected
                ):
                    # The aggregate may be a public/captured parameter whose
                    # Python class is intentionally absent.  A field reference
                    # is still a precise SSA identity: aggregate endpoint plus
                    # field path.  Canonicalize that identity instead of
                    # assigning every repeated Attribute AST node a new fake
                    # buffer ID.
                    result = {
                        (): field_endpoint(projected[()], field)
                    }
            elif attributes.get("producer_kind") in {
                "aggregate_materialization",
                "loop_materialization",
            }:
                # Materializing an iterable container does not create new
                # numerical values.  Preserve the source aggregate's ordered
                # leaves so downstream stack/cat operations can correlate the
                # one tape's transient leaf IDs with planner-owned identities.
                sources = tuple(
                    int(value_id)
                    for value_id in attributes.get(
                        "materialized_source_value_ids",
                        (),
                    )
                )
                if len(sources) == 1:
                    result = leaves(key[0], sources[0])
                else:
                    for index, source_id in enumerate(sources):
                        for path, endpoint in leaves(
                            key[0], source_id
                        ).items():
                            result[(int(index), *path)] = endpoint
            elif isinstance(expression, ast.Call) and (
                attributes.get("class_ref") is not None
                or (
                    isinstance(expression.func, ast.Name)
                    and expression.func.id
                    in (graph.graph.get("class_table") or {})
                )
                or (
                    graph.graph.get("method_binding") == "class"
                    and any(
                        role in {"callee", "func"}
                        and graph.nodes[parent].get("type") == "Input"
                        and (
                            graph.nodes[parent].get("attributes") or {}
                        ).get("binding_name") == "cls"
                        for parent, role in (
                            data.get("parents") or ()
                        )
                    )
                )
            ):
                # A classmethod commonly returns ``cls(...)``.  AST ingestion
                # cannot attach a concrete class_ref to that call because
                # ``cls`` is deliberately an SSA parameter, but the method's
                # lexical owner is still a compiler fact.  Treating the result
                # as opaque makes later ``result.field`` accesses escape as
                # fake shell inputs.  Resolve the aggregate schema from the
                # method owner without evaluating or retaining the Python
                # class object.
                class_name = (
                    attributes.get("class_ref")
                    or (
                        expression.func.id
                        if isinstance(expression.func, ast.Name)
                        and expression.func.id
                        in (graph.graph.get("class_table") or {})
                        else None
                    )
                    or graph.graph.get("method_owner")
                )
                descriptor = (
                    graph.graph.get("class_table", {}).get(
                        class_name
                    ) or {}
                )
                fields = tuple(descriptor.get("fields") or ())
                field_defaults = descriptor.get("field_defaults") or {}
                positional = {
                    position: parent
                    for parent, role in parents
                    if (
                        position := _positional_argument_index(role)
                    ) is not None
                }
                keywords = {
                    role.split(":", 1)[1]: parent
                    for parent, role in parents
                    if role.startswith("kw:")
                }
                for index, field in enumerate(fields):
                    parent = keywords.get(field, positional.get(index))
                    if parent is None:
                        if field in field_defaults:
                            endpoint = field_endpoint(key, str(field))
                            synthetic_static_values[endpoint] = (
                                field_defaults[field]
                            )
                            result[(str(field),)] = endpoint
                        continue
                    for path, endpoint in leaves(key[0], parent).items():
                        result[(str(field), *path)] = endpoint
            elif isinstance(expression, ast.Call) and (
                call := calls.get(key)
            ) is not None:
                child_id = int(call.callee.closure_id)
                child_owner = shells[child_id]
                child = child_owner.process_graph.G
                selected_returns = tuple(
                    getattr(
                        child_owner,
                        "_captured_return_value_ids",
                        (),
                    )
                )
                if len(selected_returns) == 1:
                    result = dict(leaves(
                        child_id, int(selected_returns[0])
                    ))
                    resolved_leaves[key] = result
                    resolving.remove(key)
                    return result
                output_names = tuple(
                    child.graph.get("function_outputs", ())
                )
                identities = child.graph.get("identity_table") or {}
                for output_index, name in enumerate(output_names):
                    values = identities.get(name, ())
                    if not values:
                        continue
                    prefix = (
                        () if len(output_names) == 1 else (output_index,)
                    )
                    for path, endpoint in leaves(
                        child_id, int(values[-1])
                    ).items():
                        result[(*prefix, *path)] = endpoint
            if not result:
                result = {(): key}
        resolving.remove(key)
        resolved_leaves[key] = result
        return result

    def canonical_global(closure_id: int, local_id: int) -> int:
        if (int(closure_id), int(local_id)) in loop_carried_update_endpoints:
            # A loop backedge is a cut in the projection graph.  Structural
            # leaf resolution walks value/alias/call edges as if the program
            # were acyclic, so it happily fuses the two endpoints of a cycle:
            # a LoopResult carries its value edge through to the very update
            # it publishes, and a caller result, local copy, or IndexedStore
            # correlated with the same source name resolves to that same leaf.
            # Both endpoints then land on one global and the header Phi
            # degenerates to Phi(x, x), leaving no body producer to find.
            # Storage reuse across the backedge stays an arena decision; the
            # update keeps its own SSA identity here so the pair survives to
            # Phi construction.  Same exclusion the alias table already
            # applies, extended to the projection that assigns identity.
            return global_value(closure_id, local_id)
        projected = leaves(closure_id, local_id)
        if len(projected) == 1 and () in projected:
            return global_value(*projected[()])
        return global_value(closure_id, local_id)

    # Canonicalization must apply to the hierarchy's shared value table, not
    # only to numerical region payloads.  A structural Phi/call/attribute and
    # a raw tape endpoint can already share one pre-reduction global ID.  If
    # only the numerical side follows the resolved source, composed control
    # retains the stale ID and fabricates a second, external shader input for
    # the same value.
    #
    # First discover an unambiguous redirect for every pre-reduction global
    # identity.  Then apply that redirect to *all* correlated endpoints and
    # compose control again from the unified table.  This is graph identity
    # reduction over immutable IR; it neither consults captured values nor
    # creates/replays a tape.
    projected_correlations = tuple(
        (
            int(closure_id),
            int(local_id),
            int(global_id),
            (
                canonical_global(int(closure_id), int(local_id))
                if int(closure_id) in shells
                else int(global_id)
            ),
        )
        for closure_id, local_id, global_id
        in value_table.correlations
    )
    projection_targets: dict[int, set[int]] = {}
    for _closure_id, _local_id, original, projected in (
        projected_correlations
    ):
        if projected != original:
            projection_targets.setdefault(original, set()).add(projected)
    aggregate_redirect_diagnostics = {}
    for closure_id, owner in shells.items():
        for aggregate_map in owner.compiled_aggregate_feed_paths:
            for capture_id, graph_input_id, path in aggregate_map:
                projected = leaves(
                    int(closure_id), int(graph_input_id)
                )
                target = projected.get(tuple(path))
                if target is None:
                    aggregate_redirect_diagnostics.setdefault(
                        int(closure_id), []
                    ).append({
                        "capture": int(capture_id),
                        "input": int(graph_input_id),
                        "path": tuple(path),
                        "projected": tuple(projected.items()),
                        "target": None,
                    })
                    continue
                original = global_value(
                    int(closure_id), int(capture_id)
                )
                destination = global_value(*target)
                aggregate_redirect_diagnostics.setdefault(
                    int(closure_id), []
                ).append({
                    "capture": int(capture_id),
                    "input": int(graph_input_id),
                    "path": tuple(path),
                    "projected": tuple(projected.items()),
                    "target": tuple(target),
                    "original": int(original),
                    "destination": int(destination),
                })
                if original != destination:
                    projection_targets.setdefault(
                        original, set()
                    ).add(destination)
    canonical_redirects = {
        original: next(iter(targets))
        for original, targets in projection_targets.items()
        if len(targets) == 1
    }
    shell.hierarchical_aggregate_redirect_diagnostics = {
        closure_id: tuple(rows)
        for closure_id, rows in aggregate_redirect_diagnostics.items()
    }

    def canonical_redirect(value_id: int) -> int:
        current = int(value_id)
        seen = set()
        while current in canonical_redirects and current not in seen:
            seen.add(current)
            current = int(canonical_redirects[current])
        return current

    def composed_global(closure_id: int, local_id: int) -> int:
        return canonical_redirect(canonical_global(closure_id, local_id))

    # A loop over an aggregate needs one resident binding per field actually
    # consumed by its body.  The local loop plan initially names the aggregate
    # target (for example ``packet``), while numerical regions name precise
    # field endpoints (``packet.octets`` and ``packet.byte_count``).  Expand
    # that one structural binding into field-wise source lists after hierarchy
    # leaves are known.  This is identity projection only: it neither reads a
    # captured object nor reconstructs the container at runtime.
    aggregate_binding_expansions: dict[
        tuple[int, int, str],
        tuple[tuple[int, int, str, tuple[int, ...]], ...],
    ] = {}
    aggregate_resident_bindings: set[tuple[int, int, str]] = set()
    aggregate_loop_bounds: dict[
        tuple[str, str, str],
        int,
    ] = {}
    aggregate_candidate_diagnostics: dict[int, list[dict[str, Any]]] = {}
    shell.hierarchical_control_binding_diagnostics = {
        int(closure_id): {
            "function": (
                shells[int(closure_id)].process_graph.G.graph.get(
                    "function_name"
                )
                if int(closure_id) in shells
                else None
            ),
            "iterable": tuple(control.iterable_bindings),
            "static": tuple(control.static_iterable_bindings),
            "closure": tuple(control.closure_iterable_bindings),
            "root": repr(control.root),
            "loop_plans": tuple(
                (
                    int(plan.loop.node_id),
                    plan.strategy.value,
                    int(plan.loop.iterable_node)
                    if plan.loop.iterable_node is not None
                    else None,
                    tuple(plan.loop.target_bindings),
                    tuple(plan.loop.publication_nodes),
                )
                for plan in shells[int(closure_id)].loop_plans
            ),
            "loop_reductions": tuple(
                (
                    int(reduction.loop_node_id),
                    bool(reduction.collapsible),
                    tuple(reduction.blockers),
                    reduction.control_program is not None,
                )
                for reduction
                in shells[int(closure_id)].loop_shader_reductions
            ),
        }
        for closure_id, control in controls.items()
        if int(closure_id) in shells
        and shells[int(closure_id)].process_graph.G.graph.get(
            "function_name"
        ) == "concatenate_resident_byte_packets"
    }
    for closure_id, local_control in controls.items():
        closure_id = int(closure_id)
        owner = shells.get(closure_id)
        if owner is None:
            continue
        owner_graph = owner.process_graph.G
        aggregate_candidates = (
            *tuple(
                (
                    iterable_id,
                    target_id,
                    induction,
                    source_ids,
                    False,
                )
                for iterable_id, target_id, induction, source_ids
                in local_control.closure_iterable_bindings
            ),
            *tuple(
                (
                    iterable_id,
                    target_id,
                    induction,
                    (iterable_id,),
                    True,
                )
                for iterable_id, target_id, induction
                in local_control.iterable_bindings
            ),
        )
        for (
            iterable_id,
            target_id,
            induction,
            _source_ids,
            was_resident,
        ) in aggregate_candidates:
            composed_induction = (
                f"{str(induction)}_closure_{closure_id}"
            )
            target_endpoint = (closure_id, int(target_id))
            # Demand only attribute nodes downstream of this lexical target so
            # field_endpoint() has established every field the loop body uses.
            if int(target_id) in owner_graph:
                for descendant in nx.descendants(
                    owner_graph, int(target_id)
                ):
                    if isinstance(
                        owner_graph.nodes[descendant].get("expr_obj"),
                        ast.Attribute,
                    ):
                        leaves(closure_id, int(descendant))
            field_targets = {
                tuple(str(part) for part in path): endpoint
                for endpoint, (root, path)
                in synthetic_field_paths.items()
                if root == target_endpoint and path
            }
            if not field_targets:
                aggregate_candidate_diagnostics.setdefault(
                    closure_id, []
                ).append({
                    "iterable": int(iterable_id),
                    "target": int(target_id),
                    "induction": str(induction),
                    "resident": bool(was_resident),
                    "argument_source": argument_sources.get(
                        (closure_id, int(iterable_id))
                    ),
                    "projected": (),
                    "fields": (),
                })
                continue
            projected_iterable = leaves(
                closure_id, int(iterable_id)
            )
            aggregate_candidate_diagnostics.setdefault(
                closure_id, []
            ).append({
                "iterable": int(iterable_id),
                "target": int(target_id),
                "induction": str(induction),
                "resident": bool(was_resident),
                "argument_source": argument_sources.get(
                    (closure_id, int(iterable_id))
                ),
                "projected": tuple(projected_iterable.items()),
                "fields": tuple(field_targets.items()),
            })
            expanded = []
            for field_path, field_target in sorted(
                field_targets.items()
            ):
                indexed_sources = sorted(
                    (
                        int(path[0]),
                        endpoint,
                    )
                    for path, endpoint in projected_iterable.items()
                    if (
                        path
                        and isinstance(path[0], int)
                        and tuple(str(part) for part in path[1:])
                        == field_path
                    )
                )
                if not indexed_sources:
                    continue
                indices = tuple(index for index, _ in indexed_sources)
                if indices != tuple(range(len(indices))):
                    continue
                expanded.append((
                    composed_global(closure_id, int(iterable_id)),
                    composed_global(*field_target),
                    composed_induction,
                    tuple(
                        composed_global(*source)
                        for _index, source in indexed_sources
                    ),
                ))
            if expanded:
                global_key = (
                    composed_global(closure_id, int(iterable_id)),
                    composed_global(closure_id, int(target_id)),
                    composed_induction,
                )
                aggregate_binding_expansions[global_key] = tuple(expanded)
                if was_resident:
                    aggregate_resident_bindings.add(global_key)
                aggregate_loop_bounds[(
                    composed_induction,
                    f"__iterable_extent_{int(iterable_id)}__",
                    f"__iterable_extent_{global_key[0]}__",
                )] = len(expanded[0][3])

    value_table = type(value_table)(tuple(
        (
            closure_id,
            local_id,
            canonical_redirect(projected),
        )
        for closure_id, local_id, _original, projected
        in projected_correlations
    ))
    shell.hierarchical_effective_value_table = value_table
    shell.hierarchical_aggregate_candidate_diagnostics = {
        closure_id: tuple(rows)
        for closure_id, rows in aggregate_candidate_diagnostics.items()
    }
    hierarchical = compose_hierarchical_control(
        hierarchy, controls, value_table
    )
    if aggregate_binding_expansions:
        expanded_bindings = []
        expanded_keys = set()
        for binding in hierarchical.program.closure_iterable_bindings:
            binding_key = (
                int(binding[0]),
                int(binding[1]),
                str(binding[2]),
            )
            replacement = aggregate_binding_expansions.get(binding_key)
            if replacement is None:
                expanded_bindings.append(binding)
            else:
                expanded_bindings.extend(replacement)
                expanded_keys.add(binding_key)
        for binding_key in aggregate_resident_bindings:
            if binding_key in expanded_keys:
                continue
            expanded_bindings.extend(
                aggregate_binding_expansions[binding_key]
            )

        def fix_aggregate_loop_bounds(block):
            from .control_source import (
                CallBlock,
                LoopControlBlock,
                LoopBlock,
                ParallelDeployment,
                StateMachineTick,
                WhileBlock,
            )

            if isinstance(block, LoopBlock):
                stop = str(block.stop)
                for (
                    induction,
                    local_marker,
                    global_marker,
                ), count in aggregate_loop_bounds.items():
                    if (
                        str(block.induction) == induction
                        and (
                            local_marker in stop
                            or global_marker in stop
                        )
                    ):
                        stop = str(int(count))
                return LoopBlock(
                    block.induction,
                    block.start,
                    stop,
                    block.step,
                    fix_aggregate_loop_bounds(block.body),
                    carried_aliases=block.carried_aliases,
                    result_ports=block.result_ports,
                    parallel_iterations=block.parallel_iterations,
                    dispatch_shell=block.dispatch_shell,
                    recursion_region_id=block.recursion_region_id,
                    schedule_preference=block.schedule_preference,
                )
            if isinstance(block, WhileBlock):
                return WhileBlock(
                    block.predicate_value_id,
                    fix_aggregate_loop_bounds(block.condition),
                    fix_aggregate_loop_bounds(block.body),
                    carried_aliases=block.carried_aliases,
                    result_ports=block.result_ports,
                    recursion_region_id=block.recursion_region_id,
                    predicate_expression=block.predicate_expression,
                    sequence_mutations=block.sequence_mutations,
                    source_loop_node_id=block.source_loop_node_id,
                )
            if isinstance(block, LoopControlBlock):
                return block
            if isinstance(block, SequenceBlock):
                return SequenceBlock(tuple(
                    fix_aggregate_loop_bounds(child)
                    for child in block.blocks
                ))
            if isinstance(block, StateMachineTick):
                return StateMachineTick(
                    block.state,
                    tuple(
                        (case, fix_aggregate_loop_bounds(body))
                        for case, body in block.cases
                    ),
                    None if block.default is None
                    else fix_aggregate_loop_bounds(block.default),
                )
            if isinstance(block, ParallelDeployment):
                return ParallelDeployment(
                    tuple(
                        fix_aggregate_loop_bounds(lane)
                        for lane in block.lanes
                    ),
                    block.schedule_preference,
                )
            if isinstance(block, CallBlock):
                return CallBlock(
                    block.callsite_id,
                    fix_aggregate_loop_bounds(block.callee),
                    block.argument_bindings,
                    block.result_bindings,
                )
            return block

        program = hierarchical.program
        hierarchical = type(hierarchical)(
            replace(
                program,
                root=fix_aggregate_loop_bounds(program.root),
                iterable_bindings=tuple(
                    binding
                    for binding in program.iterable_bindings
                    if (
                        int(binding[0]),
                        int(binding[1]),
                        str(binding[2]),
                    ) not in aggregate_resident_bindings
                ),
                closure_iterable_bindings=tuple(
                    dict.fromkeys(expanded_bindings)
                ),
            ),
            hierarchical.region_correlations,
        )
    shell.hierarchical_composed_closure_iterable_bindings = tuple(
        hierarchical.program.closure_iterable_bindings
    )
    shell.hierarchical_region_correlations = tuple(
        hierarchical.region_correlations
    )
    region_lookup = {
        (closure_id, local_region): global_region
        for closure_id, local_region, global_region
        in hierarchical.region_correlations
    }

    captured_regions = {}
    specialized_values = {}
    control_uniform_ids = {
        int(uniform.value_id)
        for uniform in hierarchical.program.uniforms
    }

    def add_specialization(global_id: int, fixed: Any) -> None:
        previous = specialized_values.get(global_id, fixed)
        if previous != fixed:
            raise ValueError(
                "hierarchical GLSL specialization conflict for "
                f"value {global_id}: {previous!r} != {fixed!r}"
            )
        specialized_values[global_id] = fixed

    for closure_id, owner in shells.items():
        for region_key, captured in owner.captured_region_programs.items():
            local_region = int(
                region_key
                if not isinstance(region_key, tuple)
                else region_key[-2]
            )
            global_region = region_lookup.get(
                (closure_id, local_region)
            )
            if global_region is None:
                continue
            programs = (captured.program, *captured.stages)
            all_ids = {
                int(value_id)
                for program in programs
                for value_id in (
                    *program.feeds,
                    *program.outputs.values(),
                    *(step.result_id for step in program.steps),
                    *(
                        input_id
                        for step in program.steps
                        for input_id in step.input_ids
                    ),
                )
            }
            id_map = {
                value_id: composed_global(closure_id, value_id)
                for value_id in all_ids
            }
            captured_regions[global_region] = (
                _remap_captured_all_ids(captured, id_map)
            )
        # Nested callsite shells are logical compartments of this artifact,
        # not separately finalized shaders.  Carry their typed specialization
        # facts directly from local lowering; requiring a throwaway child
        # shader merely to recover this map creates enormous compile-time CPU
        # churn and falsely treats every source call as a deployment.
        for value_id, fixed in (
            owner.composed_shell_specialized_values.items()
        ):
            global_id = composed_global(closure_id, int(value_id))
            add_specialization(global_id, fixed)
        owner_graph = owner.process_graph.G
        # Literal scalar nodes are already part of the source program.  When
        # hierarchy composition promotes one to control-uniform syntax, carry
        # the literal into the artifact specialization map instead of
        # inventing a host ABI input for it.
        for local_id, data in owner_graph.nodes(data=True):
            if data.get("type") not in {
                "Constant", "Const", "const", "StaticReference"
            }:
                continue
            try:
                fixed = _constant_value(data)
            except KeyError:
                continue
            if isinstance(fixed, (bool, int, float)):
                global_id = composed_global(
                    closure_id, int(local_id)
                )
                if global_id in control_uniform_ids:
                    add_specialization(global_id, fixed)
    # Defaulted scalar parameters and source-static predicate values are
    # compiler facts even when they are not literal nodes.  Resolve them
    # through explicit call bindings/defaults and specialize their canonical
    # identities; leaving them in the external set fabricates root ABI inputs
    # for nested optional arguments.
    inferred_static: dict[int, list[Any]] = {}
    for closure_id, local_id, _global_id in value_table.correlations:
        fixed = static_endpoint_value((closure_id, local_id))
        if isinstance(fixed, (bool, int, float)):
            inferred_static.setdefault(
                composed_global(closure_id, local_id), []
            ).append(fixed)
    for global_id, candidates in inferred_static.items():
        first = candidates[0]
        if not all(
            type(candidate) is type(first) and candidate == first
            for candidate in candidates[1:]
        ):
            continue
        previous = specialized_values.get(global_id, first)
        if type(previous) is type(first) and previous == first:
            specialized_values[global_id] = first
    shell._profiler.trace(
        path=shell.profile_path,
        section="hierarchical-artifact",
        label="numerical remap complete",
        fields={
            "regions": len(captured_regions),
            "elapsed_ms": round(
                (time.perf_counter() - compose_started) * 1e3,
                3,
            ),
        },
    )
    program_value_origins: dict[int, list[dict[str, Any]]] = {}
    region_owners = {
        int(global_region): {
            "closure": int(closure_id),
            "function": shells[int(closure_id)].process_graph.G.graph.get(
                "function_name"
            ),
            "local_region": int(local_region),
        }
        for closure_id, local_region, global_region
        in hierarchical.region_correlations
        if int(closure_id) in shells
    }
    for global_region, captured in captured_regions.items():
        programs = (
            tuple(captured.stages)
            if captured.stages
            else (captured.program,)
        )
        for stage_index, program in enumerate(programs):
            def note(value_id: int, role: str, **extra: Any) -> None:
                program_value_origins.setdefault(
                    int(value_id), []
                ).append({
                    "region": int(global_region),
                    **region_owners.get(int(global_region), {}),
                    "stage": int(stage_index),
                    "role": str(role),
                    **extra,
                })

            for value_id in ordered_feed_ids(program):
                note(value_id, "feed")
            for name, value_id in program.outputs.items():
                note(value_id, "output", name=str(name))
            for step_index, step in enumerate(program.steps):
                for input_index, value_id in enumerate(step.input_ids):
                    note(
                        value_id,
                        "step-input",
                        step=int(step_index),
                        input=int(input_index),
                        op=str(step.op_name),
                    )
                note(
                    step.result_id,
                    "step-result",
                    step=int(step_index),
                    op=str(step.op_name),
                )
    shell.hierarchical_program_value_origins = {
        value_id: tuple(origins)
        for value_id, origins in program_value_origins.items()
    }
    hierarchical_value_ids = {
        int(value_id)
        for captured in captured_regions.values()
        for program in (
            tuple(captured.stages)
            if captured.stages
            else (captured.program,)
        )
        for value_id in (
            *ordered_feed_ids(program),
            *tuple(program.outputs.values()),
        )
    }
    hierarchical = type(hierarchical)(
        project_control_regions(
            hierarchical.program,
            hierarchical.program.region_indices,
            retained_value_ids=hierarchical_value_ids,
        ),
        hierarchical.region_correlations,
    )
    shell.hierarchical_control_program = hierarchical.program
    shell.hierarchical_captured_region_programs = dict(captured_regions)
    if set(captured_regions) != set(
        hierarchical.program.region_indices
    ):
        shell.hierarchical_compose_failure = {
            "reason": "captured-control-region-mismatch",
            "captured_only": tuple(sorted(
                set(captured_regions)
                - set(hierarchical.program.region_indices)
            )),
            "control_only": tuple(sorted(
                set(hierarchical.program.region_indices)
                - set(captured_regions)
            )),
            "captured_only_origins": tuple(
                (
                    int(global_region),
                    int(closure_id),
                    int(local_region),
                    shells[int(closure_id)].process_graph.G.graph.get(
                        "function_name"
                    ),
                )
                for closure_id, local_region, global_region
                in hierarchical.region_correlations
                if global_region in (
                    set(captured_regions)
                    - set(hierarchical.program.region_indices)
                )
            )[:12],
        }
        return None
    root_id = int(hierarchy.closure_id)
    root_locals = {
        int(local_id)
        for closure_id, local_id, _global_id
        in value_table.correlations
        if int(closure_id) == root_id
    }
    # Identity-call reduction may erase a parameter's explicit correlation
    # after all of its uses move into a child closure.  It remains a public
    # root ABI value and must still receive its canonical global identity.
    # Derive root inputs from the source graph, not merely from the reduced
    # correlation table, or valid parameters surface as anonymous private
    # shader feeds.
    root_locals.update(
        int(local_id)
        for local_id, data in shell.process_graph.G.nodes(data=True)
        if data.get("type") == "Input"
        and shell.process_graph.G.out_degree(local_id)
    )
    shell.hierarchical_root_value_ids = {
        local_id: composed_global(root_id, local_id)
        for local_id in root_locals
    }
    # Preserve the compiler-generated portion of the global namespace as
    # identities only.  These are not captured values and carry no Python
    # objects; the table makes any accidentally exposed private operand
    # diagnosable as its owning closure/local SSA endpoint instead of an
    # inscrutable integer in the shell ABI.
    shell.hierarchical_private_value_ids = {
        int(global_id): (int(closure_id), int(local_id))
        for (closure_id, local_id), global_id in private_values.items()
    }
    shell.hierarchical_capture_value_ids = {
        int(global_id): int(capture_id)
        for capture_id, global_id in capture_values.items()
    }
    shell.hierarchical_synthetic_field_paths = {
        int(global_id): (
            (int(root[0]), int(root[1])),
            tuple(str(part) for part in path),
        )
        for endpoint, (root, path) in synthetic_field_paths.items()
        if (global_id := private_values.get(endpoint)) is not None
    }
    endpoint_details: dict[int, list[dict[str, Any]]] = {}

    def describe_endpoint(
        global_id: int,
        closure_id: int,
        local_id: int,
    ) -> None:
        owner = shells.get(int(closure_id))
        graph = None if owner is None else owner.process_graph.G
        data = (
            {}
            if graph is None or int(local_id) not in graph
            else graph.nodes[int(local_id)]
        )
        meta = value_meta((int(closure_id), int(local_id)))
        endpoint_details.setdefault(int(global_id), []).append({
            "closure": int(closure_id),
            "function": (
                None
                if graph is None
                else graph.graph.get("function_name")
            ),
            "local": int(local_id),
            "type": data.get("type"),
            "op": data.get("op"),
            "label": data.get("label"),
            "parents": tuple(
                (int(parent), str(role))
                for parent, role in (data.get("parents") or ())
            ),
            "attributes": {
                str(name): value
                for name, value in (data.get("attributes") or {}).items()
                if isinstance(value, (bool, int, float, str, type(None)))
            },
            "meta": (
                None
                if meta is None
                else {
                    "shape": tuple(meta.shape or ()),
                    "dtype": str(meta.dtype),
                }
            ),
            "compiled_feed_meta": (
                {}
                if owner is None
                else {
                    int(value_id): {
                        "shape": tuple(value_meta.shape or ()),
                        "dtype": str(value_meta.dtype),
                    }
                    for value_id, value_meta
                    in owner.compiled_feed_meta.items()
                }
            ),
            "captured_return_value_ids": (
                ()
                if owner is None
                else tuple(sorted(getattr(
                    owner, "_captured_return_value_ids", ()
                )))
            ),
            "hierarchy_call": (
                int(closure_id), int(local_id)
            ) in calls,
            "call_result_bindings": tuple(
                getattr(
                    calls.get((int(closure_id), int(local_id))),
                    "result_bindings",
                    (),
                )
            ),
            "argument_source": argument_sources.get((
                int(closure_id), int(local_id)
            )),
            "control_alias_source": control_alias_sources.get((
                int(closure_id), int(local_id)
            )),
            "parent_details": (
                ()
                if graph is None
                else tuple(
                    (
                        int(parent),
                        str(role),
                        graph.nodes[int(parent)].get("type"),
                        graph.nodes[int(parent)].get("label"),
                        dict(
                            graph.nodes[int(parent)].get("attributes") or {}
                        ),
                    )
                    for parent, role in (data.get("parents") or ())
                    if int(parent) in graph
                )
            ),
            "capture_input_types": (
                {}
                if owner is None
                else {
                    str(name): tuple(sorted(types))
                    for name, types in getattr(
                        owner, "_capture_input_type_names", {}
                    ).items()
                }
            ),
            "captured_source_branches": (
                ()
                if owner is None
                else tuple(getattr(
                    owner, "_captured_source_branches", ()
                ))
            ),
            "expr": (
                None
                if not isinstance(data.get("expr_obj"), ast.AST)
                else ast.dump(
                    data["expr_obj"],
                    annotate_fields=False,
                    include_attributes=False,
                )[:240]
            ),
        })

    for closure_id, local_id, global_id in value_table.correlations:
        describe_endpoint(global_id, closure_id, local_id)
    for (closure_id, local_id), global_id in private_values.items():
        describe_endpoint(global_id, closure_id, local_id)
        rooted = synthetic_field_paths.get((closure_id, local_id))
        if rooted is not None:
            (root_closure, root_local), field_path = rooted
            describe_endpoint(global_id, root_closure, root_local)
            endpoint_details[int(global_id)][-1]["field_path"] = tuple(
                str(part) for part in field_path
            )
    shell.hierarchical_endpoint_details = {
        global_id: tuple(details)
        for global_id, details in endpoint_details.items()
    }
    # Preserve source provenance through graph-inert tensor accommodation.
    # A common example is ``x if isinstance(x, AbstractTensor) else
    # AbstractTensor.tensor(x)``: both arms name the same semantic input, but
    # the IfExp owns a private ProcessGraph endpoint.  It must not become a
    # fabricated native ABI input merely because tensor discovery selected
    # one arm.  This is derived exclusively from scoped graph edges and the
    # hierarchy value table, never from observed values or wrapper identity.
    transparent_cache: dict[tuple[int, int], int | None] = {}

    def transparent_origin(
        closure_id: int,
        local_id: int,
        visiting: frozenset[tuple[int, int]] = frozenset(),
    ) -> int | None:
        endpoint = (int(closure_id), int(local_id))
        if endpoint in transparent_cache:
            return transparent_cache[endpoint]
        if endpoint in visiting:
            return None
        owner = shells.get(endpoint[0])
        graph = None if owner is None else owner.process_graph.G
        if graph is None or endpoint[1] not in graph:
            return None
        data = graph.nodes[endpoint[1]]
        parents = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        next_visiting = visiting | {endpoint}
        origin = None
        if data.get("type") == "Input":
            origin = composed_global(*endpoint)
        elif isinstance(data.get("expr_obj"), ast.IfExp):
            body = parents.get("body")
            orelse = parents.get("orelse")
            body_origin = (
                None if body is None else transparent_origin(
                    endpoint[0], body, next_visiting
                )
            )
            orelse_origin = (
                None if orelse is None else transparent_origin(
                    endpoint[0], orelse, next_visiting
                )
            )
            if body_origin is not None and body_origin == orelse_origin:
                origin = body_origin
        elif isinstance(data.get("expr_obj"), ast.Call):
            expression = data["expr_obj"]
            function = expression.func
            is_tensor_accommodation = (
                isinstance(function, ast.Attribute)
                and function.attr == "tensor"
                and isinstance(function.value, ast.Name)
                and function.value.id == "AbstractTensor"
            )
            if is_tensor_accommodation:
                argument = next((
                    int(parent)
                    for parent, role in (data.get("parents") or ())
                    if str(role) in {"arg", "args", "arg:0", "arg0"}
                ), None)
                if argument is not None:
                    origin = transparent_origin(
                        endpoint[0], argument, next_visiting
                    )
        transparent_cache[endpoint] = origin
        return origin

    hierarchical_value_aliases = {}
    for global_id, details in shell.hierarchical_endpoint_details.items():
        for detail in details:
            origin = transparent_origin(
                int(detail["closure"]), int(detail["local"])
            )
            if origin is not None and int(origin) != int(global_id):
                hierarchical_value_aliases[int(global_id)] = int(origin)
                break
    shell.hierarchical_value_aliases = hierarchical_value_aliases
    root_parameter_names = {
        str((data.get("attributes") or {}).get("binding_name"))
        for _node_id, data in shell.process_graph.G.nodes(data=True)
        if data.get("type") == "Input"
        and (data.get("attributes") or {}).get("binding_kind")
        == "parameter"
    }

    def attribute_path(local_id: int) -> str | None:
        if int(local_id) not in shell.process_graph.G:
            return None
        data = shell.process_graph.G.nodes[int(local_id)]
        expression = data.get("expr_obj")
        if not isinstance(expression, ast.Attribute):
            return None
        parts = [str(expression.attr)]
        current = expression.value
        while isinstance(current, ast.Attribute):
            parts.append(str(current.attr))
            current = current.value
        if (
            not isinstance(current, ast.Name)
            or current.id not in root_parameter_names
        ):
            return None
        return ".".join((current.id, *reversed(parts)))

    shell.hierarchical_root_field_value_ids = {
        path: int(global_id)
        for global_id, (closure_id, local_id)
        in shell.hierarchical_private_value_ids.items()
        if int(closure_id) == root_id
        and (path := attribute_path(local_id)) is not None
    }
    for global_id, endpoint in (
        shell.hierarchical_private_value_ids.items()
    ):
        rooted = synthetic_field_paths.get(endpoint)
        if rooted is None:
            continue
        (closure_id, local_id), field_path = rooted
        if int(closure_id) != root_id:
            continue
        root_graph = shell.process_graph.G
        if int(local_id) not in root_graph:
            continue
        root_data = root_graph.nodes[int(local_id)]
        root_attributes = root_data.get("attributes") or {}
        if (
            root_data.get("type") != "Input"
            or root_attributes.get("binding_kind") != "parameter"
        ):
            continue
        name = root_attributes.get("binding_name")
        if name is not None:
            shell.hierarchical_root_field_value_ids[
                ".".join((str(name), *field_path))
            ] = int(global_id)
    graph = shell.process_graph.G
    identities = graph.graph.get("identity_table") or {}
    terminals = {}
    root_output_leaves = set()
    for name in graph.graph.get("function_outputs", ()):
        if not identities.get(name):
            continue
        for path, endpoint in leaves(
            root_id, int(identities[name][-1])
        ).items():
            global_id = global_value(*endpoint)
            suffix = "".join(f".{part}" for part in path)
            terminals[f"{name}{suffix}"] = global_id
            root_output_leaves.add(global_id)
    resident_values = {
        int(value_id)
        for captured in captured_regions.values()
        for program in (
            tuple(captured.stages)
            if captured.stages
            else (captured.program,)
        )
        for value_id in (
            *ordered_feed_ids(program),
            *tuple(program.outputs.values()),
        )
    }
    resident_values.update(
        int(collection_id)
        for _source_id, collection_id, _induction, _start
        in hierarchical.program.collection_bindings
    )
    terminals = {
        name: value_id
        for name, value_id in terminals.items()
        if value_id in resident_values
    }
    shell.hierarchical_public_output_ids = set(terminals.values())
    shell.hierarchical_terminal_outputs = dict(terminals)
    shell.hierarchical_specialized_values = dict(specialized_values)
    stream_outputs = {}
    for plan in shell.loop_plans:
        for statement_id, published_value_id, _count_id in (
            plan.loop.publication_nodes
        ):
            for name in graph.graph.get("function_outputs", ()):
                output_ids = tuple(identities.get(name, ()))
                if not output_ids:
                    continue
                output_id = int(output_ids[-1])
                if (
                    statement_id in graph
                    and output_id in graph
                    and nx.has_path(graph, statement_id, output_id)
                ):
                    stream_outputs[str(name)] = int(
                        shell.hierarchical_root_value_ids[
                            int(published_value_id)
                        ]
                    )
    shell._profiler.trace(
        path=shell.profile_path,
        section="hierarchical-artifact",
        label="shader artifact begin",
        fields={
            "regions": len(captured_regions),
            "terminals": len(terminals),
        },
    )
    hierarchical_storage_meta = dict(global_meta)
    for endpoint, global_id in private_values.items():
        meta = value_meta(endpoint)
        if meta is not None:
            hierarchical_storage_meta.setdefault(int(global_id), meta)
    for closure_id, local_id, global_id in value_table.correlations:
        meta = value_meta((int(closure_id), int(local_id)))
        if meta is not None:
            hierarchical_storage_meta.setdefault(int(global_id), meta)
    for capture_id, global_id in capture_values.items():
        meta = next(
            (
                value_meta((int(closure_id), int(capture_id)))
                for closure_id in shells
                if value_meta((int(closure_id), int(capture_id))) is not None
            ),
            None,
        )
        if meta is not None:
            hierarchical_storage_meta.setdefault(int(global_id), meta)
    hierarchy_contract_diagnostics = {
        int(global_id): ({
            "kind": "hierarchy",
            "private_endpoint": (
                shell.hierarchical_private_value_ids.get(int(global_id))
            ),
            "capture_endpoint": (
                shell.hierarchical_capture_value_ids.get(int(global_id))
            ),
            "endpoints": tuple(details),
            "program_origins": tuple(
                shell.hierarchical_program_value_origins.get(
                    int(global_id), ()
                )
            ),
            "synthetic_field": (
                shell.hierarchical_synthetic_field_paths.get(
                    int(global_id)
                )
            ),
        },)
        for global_id, details in shell.hierarchical_endpoint_details.items()
    }
    try:
        # Retain the target-neutral whole-program form alongside the optional
        # GLSL artifact. AOT consumers (notably WebAssembly) need the same
        # hierarchy and region identities without reverse-engineering shader
        # source or selecting one nested shell as a surrogate root.
        shell.hierarchical_control_program = hierarchical.program
        shell.hierarchical_captured_region_programs = dict(captured_regions)
        artifact = build_control_shader_artifact(
            hierarchical.program,
            captured_regions,
            value_meta=hierarchical_storage_meta,
            value_contract_diagnostics=hierarchy_contract_diagnostics,
            instrumentation=shell._profiler.verbose,
            terminal_outputs=terminals or None,
            stream_outputs=stream_outputs,
            specialized_values=specialized_values,
            device_resident=(shell.shell_language == "glsl"),
        )
    except Exception as error:
        program = getattr(error, "program", None)
        value_ids = (
            ()
            if program is None
            else tuple(dict.fromkeys((
                *ordered_feed_ids(program),
                *program.outputs.values(),
                *(step.result_id for step in program.steps),
            )))
        )
        raise RuntimeError(
            f"{error}; endpoint_details="
            f"{ {int(value_id): shell.hierarchical_endpoint_details.get(int(value_id), ()) for value_id in value_ids}!r}"
        ) from error
    shell._profiler.trace(
        path=shell.profile_path,
        section="hierarchical-artifact",
        label="complete",
        fields={
            "elapsed_ms": round(
                (time.perf_counter() - compose_started) * 1e3,
                3,
            ),
        },
    )
    return artifact


def _diagnostic_value_summary(value: Any) -> str:
    """Describe one routed value without creating AbstractTensor operations."""

    if isinstance(value, type):
        return f"type:{value.__module__}.{value.__qualname__}"
    shape = getattr(value, "shape", None)
    if callable(shape):
        # Namespace objects may export an operation called ``shape``.  Mirror
        # the runtime tensorization rule so failure reporting cannot mask the
        # original exception while trying to iterate that function.
        shape = None
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    prefix = type(value).__name__
    if shape is not None:
        prefix += f"(shape={tuple(shape)},dtype={dtype},device={device})"
    data = getattr(value, "data", value)
    materialize = getattr(data, "numpy", None)
    if not callable(materialize):
        return prefix + f":{value!r}" if shape is None else prefix
    try:
        import numpy as np

        array = np.asarray(materialize())
        if not array.size:
            return prefix + "[empty]"
        finite = np.isfinite(array)
        finite_count = int(finite.sum())
        if finite_count:
            selected = array[finite]
            stats = (
                f"min={selected.min():.7g},max={selected.max():.7g},"
                f"mean={selected.mean():.7g}"
            )
        else:
            stats = "min=nan,max=nan,mean=nan"
        return (
            prefix
            + f"[{stats},finite={finite_count}/{array.size}]"
        )
    except Exception as error:
        return prefix + f"[diagnostic={type(error).__name__}:{error}]"


def _captured_storage_meta(values: dict[int, Any]) -> dict[int, Meta]:
    """Extract arena storage contracts without consuming numerical values."""

    result = {}
    for value_id, value in values.items():
        if isinstance(value, AbstractTensor):
            result[int(value_id)] = Meta(
                shape=tuple(int(size) for size in value.shape),
                dtype=str(value.dtype),
                device="glsl",
            )
        elif isinstance(value, bool):
            result[int(value_id)] = Meta((), "bool", "glsl")
        elif isinstance(value, int):
            result[int(value_id)] = Meta((), "int32", "glsl")
        elif isinstance(value, float):
            result[int(value_id)] = Meta((), "float32", "glsl")
    return result


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
    owns_gpu_query = bool(gpu and profiler._gpu_query_depth == 0)
    if owns_gpu_query:
        from OpenGL import GL

        generated = GL.glGenQueries(1)
        try:
            query = int(generated[0])
        except (IndexError, TypeError):
            query = int(generated)
        GL.glBeginQuery(GL.GL_TIME_ELAPSED, query)
    if gpu:
        profiler._gpu_query_depth += 1
    before_dispatches = dispatch_stats()["calls"]
    started_ns = time.perf_counter_ns()
    try:
        yield
    finally:
        if gpu:
            profiler._gpu_query_depth -= 1
        if owns_gpu_query:
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
    map_ir = graph.G.graph.get("map_ir") or {}
    return (
        node_id in set(map_ir.get("schema_node_ids", ()))
        or
        node_type in {"str", "NoneType"}
        or isinstance(
            data.get("expr_obj"),
            (
                ast.Module,
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.Import,
                ast.ImportFrom,
                ast.alias,
                ast.arguments,
                ast.arg,
                ast.keyword,
            ),
        )
    )


def _semantic_children(graph: Any) -> dict[int, set[int]]:
    """Union the edge and parent-reference views of who consumes each node."""

    children: dict[int, set[int]] = {
        node_id: set(graph.G.successors(node_id)) for node_id in graph.G
    }
    for node_id, data in graph.G.nodes(data=True):
        for parent, _role in data.get("parents", ()):
            if parent in children:
                children[parent].add(node_id)
    return children


def _inert_routing_nodes(graph: Any) -> frozenset[int]:
    """Find attribute lookups whose result nothing ever reads.

    Ingesting ``x.foo()`` records the attribute lookup beside the call even
    when the call already references ``x`` directly, so the lookup is left with
    no consumer.  Evaluating it anyway is not free: it demands a materialized
    value for ``x``, which forces an otherwise shader-local intermediate to
    become a published region output.  Naming these nodes lets both the region
    extractor and the coordinator ignore work no one observes.  Only the
    side-effect-free lookup itself qualifies: only a load, never an assignment
    or deletion target, and never a call or statement with no consumer.
    """

    fingerprint = (
        graph.G.number_of_nodes(),
        graph.G.number_of_edges(),
        tuple(map(int, getattr(graph, "roots", ()) or ())),
    )
    cached = graph.G.graph.get("_inert_routing_nodes_cache")
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    roots = set(getattr(graph, "roots", ()) or ())
    children = _semantic_children(graph)
    inert: set[int] = set()
    for node_id in reversed(_dependency_order(graph)):
        if node_id in roots:
            continue
        expression = graph.G.nodes[node_id].get("expr_obj")
        if not isinstance(expression, ast.Attribute):
            continue
        if not isinstance(getattr(expression, "ctx", None), ast.Load):
            # ``obj.field = value`` mutates the receiver.  Its node has no
            # consumer by construction, which is exactly why it must still run.
            continue
        if all(child in inert for child in children[node_id]):
            inert.add(node_id)
    result = frozenset(inert)
    graph.G.graph["_inert_routing_nodes_cache"] = (fingerprint, result)
    return result


def _is_dispatch_metadata_node(graph: Any, node_id: int) -> bool:
    """Whether a node routes syntax but performs no computation (cached).

    This ~200-line classifier dominated ``compile_process_graph`` because it is
    a pure function of the (planning-stable) graph yet is evaluated for every
    node more than once -- once building each shell's executable-node set, again
    validating dispatch coverage. Memoize it per graph, keyed by the same cheap
    (node count, edge count) fingerprint ``_dependency_order`` uses, so a graph
    that gains or loses structure is reclassified while a mere re-query is a
    dict hit.
    """

    G = graph.G
    fingerprint = (G.number_of_nodes(), G.number_of_edges())
    cache = G.graph.get("_dispatch_metadata_cache")
    if cache is None or cache.get("__fingerprint__") != fingerprint:
        cache = {"__fingerprint__": fingerprint}
        G.graph["_dispatch_metadata_cache"] = cache
    key = int(node_id)
    cached = cache.get(key)
    if cached is not None:
        return cached
    result = _is_dispatch_metadata_node_impl(graph, node_id)
    cache[key] = result
    return result


def _is_dispatch_metadata_node_impl(graph: Any, node_id: int) -> bool:
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
            in {"Tuple", "List", "Set", "Dict"}
            or any(
                parent in graph.G
                and graph.G.nodes[parent].get("type")
                in {"Const", "const", "Constant"}
                and isinstance(
                    _constant_value(graph.G.nodes[parent]),
                    (str, bytes),
                )
                for parent, role in parents
                if str(role) == "index"
            )
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
        candidate_op = str(
            candidate_data.get("op") or candidate_type
        )
        if candidate_op in ACCESSOR_OPERATORS:
            return True
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
    coordinator_accessor = (
        str(data.get("op") or node_type) in ACCESSOR_OPERATORS
    )
    # ``<<``/``>>`` are ordinary numeric tensor operators (bitfield extraction
    # in a byte decoder, for example), not inherently coordinator-side. A
    # genuine *scalar* shift -- address/index arithmetic -- is already caught
    # by ``static_scalar_expression`` below (every operand a scalar), exactly
    # as a scalar ``&`` is; there is deliberately no ``coordinator_bitand``.
    # Special-casing every shift as coordinator metadata additionally swallowed
    # tensor-parallel shifts, which then never reached a numeric region.
    coordinator_boolean_not = (
        isinstance(expression, ast.UnaryOp)
        and isinstance(expression.op, ast.Not)
    )
    chained_comparison = (
        isinstance(expression, ast.Compare)
        and len(expression.ops) > 1
    )
    python_shape_index = (
        isinstance(expression, ast.Subscript)
        and isinstance(expression.value, ast.Attribute)
        and expression.value.attr == "shape"
    )
    # Loop-target initializers are coordinator state setup.  Their values may
    # use ordinary tensor operations, but emitting them as an independent GPU
    # region duplicates work before the retained loop owns the binding.
    loop_target_initializers = getattr(
        graph,
        "_dispatch_loop_target_initializers",
        None,
    )
    if loop_target_initializers is None:
        loop_target_initializers = frozenset(
            int(initializer)
            for _candidate_id, candidate in graph.G.nodes(data=True)
            if isinstance(
                candidate.get("expr_obj"),
                (ast.For, ast.While),
            )
            for initializer in (
                (candidate.get("attributes") or {})
                .get("loop_target_initials", {})
                .values()
            )
        )
        graph._dispatch_loop_target_initializers = (
            loop_target_initializers
        )
    loop_target_initializer = node_id in loop_target_initializers
    attributes = data.get("attributes") or {}
    ungrounded_tensor_method = (
        attributes.get("tensor_candidate") is not None
        and attributes.get("tensor") is None
        and str(attributes.get("tensor_candidate")) in {"split"}
        and isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Attribute)
    )
    return (
        bool(
            (data.get("attributes") or {}).get(
                "coordinator_short_circuit"
            )
        )
        or any(
            (data.get("attributes") or {}).get(name) is not None
            for name in (
                "callee_ref",
                "method_ref",
                "class_ref",
            )
        )
        or (
            (data.get("attributes") or {}).get(
                "static_python_reference"
            ) is not None
            and (
                node_type not in abstract_tensor_funcs
                or (data.get("attributes") or {}).get(
                    "operator_reference_node"
                ) is None
            )
        )
        or
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
            "StaticReference",
            "SetAttr",
            "DelAttr",
            "DelItem",
            "Phi",
            "LoopExit",
            "LoopStateTransition",
            "LoopResult",
            "LoopStatePort",
            "LoopAggregateResult",
            "Yield",
            "YieldFrom",
            "no_grad",
        }
        or
        (
            isinstance(data.get("expr_obj"), ast.Call)
            and isinstance(data["expr_obj"].func, ast.Attribute)
            and (
                node_type not in abstract_tensor_funcs
                or ungrounded_tensor_method
            )
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
                ast.Assert,
                ast.Pass,
                ast.Break,
                ast.Continue,
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
                ast.Lambda,
                ast.Starred,
                ast.Tuple,
                ast.List,
                ast.Set,
                ast.Dict,
            ),
        )
        or isinstance(expression, ast.Attribute)
        or python_routing_index
        or python_shape_index
        or compares_none
        or static_scalar_expression
        or coordinator_accessor
        or coordinator_boolean_not
        or chained_comparison
        or loop_target_initializer
        or (
            data.get("type") == "Load"
            and (data.get("attributes") or {}).get("source_type") == "Name"
        )
    )


def _subgraph_reduction_digest(subgraph: Any) -> str:
    """Content key for one region's structural lowering.

    ``_structural_region_program_from_subgraph`` is a pure function of the
    dispatch subgraph, so hashing the subgraph identifies the region program it
    will produce -- letting the incremental backup reuse an already-lowered
    region whenever its subgraph is unchanged.
    """

    from joblib.externals import cloudpickle

    return hashlib.sha256(cloudpickle.dumps(subgraph)).hexdigest()


def _structural_region_program_from_subgraph(
    subgraph: Any,
) -> CapturedFusedProgram:
    """Build one region's numeric program from its dispatch subgraph structure.

    This is the value-free counterpart to a captured region program: the same
    ``dispatch_region_to_fused_program`` the JIT/tape path uses, driven by the
    subgraph the planner already isolated rather than by an observed tape. It
    is how the graph-only precompile (an unfed mutable parameter left symbolic,
    the exact case an autograd tape cannot observe) still produces real region
    programs -- transcribing the ops faithfully, no hand-synthesis. Layout/cast
    ops flow through under their own names for the backend to lower.
    """

    graph_data = subgraph.G.graph
    region = DispatchRegion(
        tuple(int(n) for n in graph_data.get("deployment_nodes", ())),
        tuple(int(n) for n in graph_data.get("deployment_inputs", ())),
        tuple(
            (f"value_{int(v)}", int(v))
            for v in graph_data.get("deployment_outputs", ())
        ),
        0.0,
    )
    return CapturedFusedProgram(
        dispatch_region_to_fused_program(subgraph, region), {}
    )


def _dispatch_subgraph(
    graph: Any,
    node_ids: tuple[int, ...],
    *,
    required_outputs: frozenset[int] = frozenset(),
    inert_nodes: frozenset[int] = frozenset(),
    schedule_preference: str = "asap",
) -> Any:
    """Return the planned dispatch as an independent ProcessGraph subgraph.

    ``required_outputs`` names values the coordinator observes by identity
    rather than through a dataflow edge.  Such a value is a compartment
    boundary even when every graph-visible consumer lives inside this region,
    so it must be published instead of staying shader-local.

    ``inert_nodes`` names lookups nobody reads.  They are not consumers, so
    they never turn a shader-local intermediate into a published output.
    """

    selected = set(node_ids)
    semantic_parents = {
        node_id: {
            *graph.G.predecessors(node_id),
            *(
                parent
                for parent, _role in graph.G.nodes[node_id].get(
                    "parents", ()
                )
                if parent in graph.G
            ),
        }
        for node_id in graph.G
    }
    semantic_children = {
        node_id: set(graph.G.successors(node_id))
        for node_id in graph.G
    }
    for child, parents in semantic_parents.items():
        for parent in parents:
            semantic_children[parent].add(child)
    # Container literals are routing, not dispatches of their own.  When one
    # feeds a numerical region, inline the container and expose its leaves as
    # the region boundary so a captured tensor operation retains each feed's
    # independent identity.
    pending = list(selected)
    while pending:
        child = pending.pop()
        for parent in semantic_parents[child]:
            if parent in selected:
                continue
            expression = graph.G.nodes[parent].get("expr_obj")
            if isinstance(expression, (ast.Tuple, ast.List)):
                selected.add(parent)
                pending.append(parent)
    boundary = {
        parent
        for node_id in selected
        for parent in semantic_parents[node_id]
        if parent not in selected
        and not _is_ast_metadata_node(graph, parent)
    }
    included = selected | boundary
    subgraph = extract_clean_process_subgraph(graph, included)
    subgraph.python_bindings = dict(
        getattr(graph, "python_bindings", {}) or {}
    )
    for child in included:
        roles = {
            parent: role
            for parent, role in graph.G.nodes[child].get("parents", ())
            if parent in included
        }
        for parent in semantic_parents[child]:
            if parent in included and not subgraph.G.has_edge(parent, child):
                subgraph.G.add_edge(
                    parent,
                    child,
                    role=roles.get(parent, "dependency"),
                )

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

    observed_children = {
        node_id: children - inert_nodes
        for node_id, children in semantic_children.items()
    }
    subgraph.roots = [
        node_id
        for node_id in node_ids
        if (
            node_id in required_outputs
            or not observed_children[node_id]
            or any(
                child not in selected
                for child in observed_children[node_id]
            )
        )
    ]
    deployment_outputs = tuple(subgraph.roots)
    next_node_id = max(
        (node_id for node_id in graph.G if isinstance(node_id, int)),
        default=0,
    ) + 1
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
        store_nodes.append(store_id)
    subgraph.roots = store_nodes
    # Compartmentalization changes both the region boundary and its available
    # inputs, so inherited levels from the source graph are no longer an
    # execution schedule for this subgraph.  Schedule every compartment from
    # its reconstructed internal dependencies.  The source schedule guided
    # legal region formation; this local schedule governs emitted instruction
    # order inside the resulting shader.
    schedule_preference = str(schedule_preference).lower()
    if schedule_preference not in {"asap", "alap"}:
        raise ValueError(
            "dispatch schedule preference must be 'asap' or 'alap'"
        )
    computed_levels = subgraph.compute_levels(
        method=schedule_preference,
        order="dependency",
    )
    if computed_levels is not None:
        subgraph.levels = computed_levels
    subgraph.G.graph["compartment_schedule_preference"] = (
        schedule_preference
    )
    # One topological sort, then bucket by level -- not one full sort per
    # level. The order within a level is identical (it is the single sort's
    # order, filtered), so the schedule is unchanged; only the redundant
    # O(levels x sort) work is removed.
    _compartment_order = nx.lexicographical_topological_sort(
        subgraph.G, key=lambda value_id: int(value_id)
    )
    _nodes_by_level: dict[int, list[int]] = {}
    for node_id in _compartment_order:
        _nodes_by_level.setdefault(
            int(subgraph.levels.get(node_id, 0)), []
        ).append(node_id)
    subgraph.G.graph["compartment_schedule"] = tuple(
        (level, tuple(_nodes_by_level.get(level, ())))
        for level in sorted(set(subgraph.levels.values()))
    )
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


def _branch_compartments(graph: Any) -> dict[int, frozenset[tuple[int, str]]]:
    """Map every node to the conditional branches that guard it.

    A branch body only runs when its test selects it.  The coordinator honours
    that by withholding controlled nodes from the ordinary topological sweep,
    but a shader region is executed as a unit: pulling any of its members
    forces every member to run.  Recording branch membership lets region
    planning refuse to mix guarded work with work that runs unconditionally,
    or to mix two branches of the same test with each other.
    """

    expression_nodes = {
        id(data.get("expr_obj")): node_id
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.AST)
    }
    memberships: dict[int, set[tuple[int, str]]] = {}
    for control_id, data in graph.G.nodes(data=True):
        expression = data.get("expr_obj")
        if isinstance(expression, ast.If):
            branches = {
                "body": tuple(expression.body),
                "orelse": tuple(expression.orelse),
            }
        elif isinstance(expression, ast.IfExp):
            branches = {
                "body": (expression.body,),
                "orelse": (expression.orelse,),
            }
        elif isinstance(expression, ast.Try):
            branches = {
                "body": tuple(expression.body),
                "handlers": tuple(expression.handlers),
                "orelse": tuple(expression.orelse),
                "finalbody": tuple(expression.finalbody),
            }
        elif isinstance(expression, ast.BoolOp):
            # Only the first operand of a short-circuiting operator is
            # guaranteed to run; the rest are guarded by its outcome.
            branches = {
                f"operand:{index}": (value,)
                for index, value in enumerate(expression.values)
                if index
            }
        else:
            continue
        for role, statements in branches.items():
            for statement in statements:
                for member in ast.walk(statement):
                    guarded = expression_nodes.get(id(member))
                    if guarded is not None and guarded != control_id:
                        memberships.setdefault(guarded, set()).add(
                            (control_id, role)
                        )
    # Reducer-authored Phi values do not own an AST expression, but they run
    # at the continuation of their exact source conditional.  When that
    # conditional is itself nested in another branch, its merge is therefore
    # guarded by the enclosing branch and must be a candidate outgoing value
    # for the enclosing Phi.  Inherit only the source conditional's outer
    # memberships; never mark the merge as belonging to either of its own
    # mutually exclusive arms.
    for node_id, data in graph.G.nodes(data=True):
        source_conditional_id = int((data.get("attributes") or {}).get(
            "source_conditional_id", -1
        ))
        if source_conditional_id in memberships:
            memberships.setdefault(int(node_id), set()).update(
                memberships[source_conditional_id]
            )
    return {
        node_id: frozenset(roles)
        for node_id, roles in memberships.items()
    }


def _ordinary_conditional_control_programs(
    graph: Any,
    retained_regions: Iterable[int],
    dispatch_subgraphs: Iterable[Any],
) -> tuple[ControlProgram, ...]:
    """Preserve ordinary source ``if`` arms and their lexical SSA merges."""

    from .control_source import ConditionalBlock

    retained = frozenset(map(int, retained_regions))
    subgraphs = tuple(dispatch_subgraphs)
    memberships = _branch_compartments(graph)
    node_by_value = {
        int(data.get("value_id", node_id)): int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("value_id", node_id), int)
    }
    programs = []
    for control_id, data in graph.G.nodes(data=True):
        expression = data.get("expr_obj")
        if not isinstance(expression, ast.If):
            continue
        parents = tuple(data.get("parents") or ())
        predicate_id = next((
            int(parent) for parent, role in parents if str(role) == "test"
        ), None)
        if predicate_id is None:
            continue
        predicate_value_id = int(
            graph.G.nodes[predicate_id].get("value_id", predicate_id)
        )
        body_regions = tuple(
            region_index
            for region_index, subgraph in enumerate(subgraphs)
            if region_index in retained
            and any(
                (int(control_id), "body") in memberships.get(int(node_id), ())
                for node_id in subgraph.G.graph.get("deployment_nodes", ())
            )
        )
        else_regions = tuple(
            region_index
            for region_index, subgraph in enumerate(subgraphs)
            if region_index in retained
            and any(
                (int(control_id), "orelse") in memberships.get(int(node_id), ())
                for node_id in subgraph.G.graph.get("deployment_nodes", ())
            )
        )
        predicate_regions = tuple(
            region_index
            for region_index, subgraph in enumerate(subgraphs)
            if region_index in retained
            and int(predicate_id) in set(map(
                int, subgraph.G.graph.get("deployment_nodes", ())
            ))
        )
        if not body_regions and not else_regions:
            continue

        carried = []
        for history in (graph.G.graph.get("identity_table") or {}).values():
            ordered = tuple(
                int(value_id) for value_id in history
                if int(value_id) in node_by_value
            )
            if not ordered:
                continue
            body_values = tuple(
                value_id for value_id in ordered
                if (int(control_id), "body") in memberships.get(
                    node_by_value[value_id], ()
                )
            )
            else_values = tuple(
                value_id for value_id in ordered
                if (int(control_id), "orelse") in memberships.get(
                    node_by_value[value_id], ()
                )
            )
            if not body_values and not else_values:
                continue
            branch_positions = tuple(
                ordered.index(value_id)
                for value_id in (*body_values, *else_values)
            )
            first_branch = min(branch_positions)
            last_branch = max(branch_positions)
            initial = next((
                ordered[position]
                for position in range(first_branch - 1, -1, -1)
                if ordered[position] not in {*body_values, *else_values}
            ), ordered[0])
            merged = next((
                value_id for value_id in ordered
                if int((graph.G.nodes[node_by_value[value_id]].get(
                    "attributes"
                ) or {}).get("source_conditional_id", -1))
                == int(control_id)
            ), None)
            if merged is None:
                merged = next((
                    ordered[position]
                    for position in range(last_branch + 1, len(ordered))
                    if ordered[position] not in {*body_values, *else_values}
                ), None)
            if merged is None:
                continue
            carried.append((
                body_values[-1] if body_values else initial,
                else_values[-1] if else_values else initial,
                initial,
                merged,
            ))
        body = SequenceBlock(tuple(
            StatementBlock((f"__scheduled_region_{index}__",))
            for index in body_regions
        ))
        orelse = (
            SequenceBlock(tuple(
                StatementBlock((f"__scheduled_region_{index}__",))
                for index in else_regions
            ))
            if else_regions else None
        )
        root = SequenceBlock((
            *(
                StatementBlock((f"__scheduled_region_{index}__",))
                for index in predicate_regions
            ),
            ConditionalBlock(
                int(predicate_value_id), body, orelse,
                predicate_expression=ControlExpression(
                    "value", value_id=int(predicate_value_id)
                ),
                carried_aliases=tuple(carried),
                source_node_id=int(control_id),
            ),
        ))
        programs.append(ControlProgram(
            root=root,
            region_indices=tuple(dict.fromkeys((
                *predicate_regions, *body_regions, *else_regions,
            ))),
        ))
    return tuple(programs)


def _ordinary_conditional_nesting(
    graph: Any,
    programs: Iterable[ControlProgram],
    *,
    offset: int = 0,
    root_index: int | None = None,
) -> dict[int, tuple[int, ...]]:
    """Return exact lexical parentage for source-derived conditional IR."""

    from .control_source import ConditionalBlock

    controls = tuple(programs)
    blocks = tuple(
        next((
            block for block in program.root.blocks
            if isinstance(block, ConditionalBlock)
        ), None)
        if isinstance(program.root, SequenceBlock)
        else (
            program.root
            if isinstance(program.root, ConditionalBlock) else None
        )
        for program in controls
    )
    expressions = {
        index: graph.G.nodes[int(block.source_node_id)].get("expr_obj")
        for index, block in enumerate(blocks)
        if block is not None and block.source_node_id is not None
    }
    parent_by_child: dict[int, int] = {}
    for child_index, child_expression in expressions.items():
        candidates = []
        for parent_index, parent_expression in expressions.items():
            if parent_index == child_index:
                continue
            descendants = {
                id(member)
                for statement in (
                    *parent_expression.body, *parent_expression.orelse
                )
                for member in ast.walk(statement)
            }
            if id(child_expression) not in descendants:
                continue
            span = int(getattr(
                parent_expression, "end_lineno", parent_expression.lineno
            )) - int(parent_expression.lineno)
            candidates.append((span, parent_index))
        if candidates:
            parent_by_child[child_index] = min(candidates)[1]
    children: dict[int, list[int]] = {}
    for child_index in range(len(blocks)):
        parent = parent_by_child.get(child_index)
        if parent is not None:
            children.setdefault(offset + parent, []).append(
                offset + child_index
            )
        elif root_index is not None:
            children.setdefault(int(root_index), []).append(
                offset + child_index
            )
    return {parent: tuple(items) for parent, items in children.items()}


def _control_partition_keys(
    graph: Any,
    loop_plans: Iterable[Any],
    node_ids: Iterable[int],
) -> dict[int, tuple[Any, ...]]:
    """Key nodes by lexical control and nested-call execution frontiers.

    A parent numerical region cannot span a nested function call.  Hierarchy
    composition inserts the child at the callsite, while one region marker is
    indivisible; allowing a region to contain work from both sides causes its
    post-call stages to execute before the child has produced their inputs.
    Descendant membership supplies the exact semantic frontier without
    separating independent work merely because of an arbitrary topological
    ordering.
    """

    plans = tuple(loop_plans)
    branches = _branch_compartments(graph)
    expression_nodes = {
        id(data.get("expr_obj")): int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.AST)
    }
    ordinary_if_frontiers = tuple(
        (
            int(control_id),
            int(getattr(expression, "end_lineno", expression.lineno)),
        )
        for control_id, data in graph.G.nodes(data=True)
        for expression in (data.get("expr_obj"),)
        if isinstance(expression, ast.If)
    )
    branch_frontiers_by_node: dict[int, list[int]] = {}
    for control_id, end_line in ordinary_if_frontiers:
        for node_id, data in graph.G.nodes(data=True):
            expression = data.get("expr_obj")
            if not isinstance(expression, ast.AST):
                continue
            if int(getattr(expression, "lineno", -1)) > end_line:
                branch_frontiers_by_node.setdefault(
                    int(node_id), []
                ).append(int(control_id))
    callsites = tuple(
        int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        for attributes in ((data.get("attributes") or {}),)
        if (
            attributes.get("callee_ref") is not None
            or attributes.get("method_ref") is not None
        )
    )
    call_descendants = {
        callsite: nx.descendants(graph.G, callsite)
        for callsite in callsites
    }
    comprehension_owners = tuple(dict.fromkeys(
        int(node_id)
        for plan in plans
        if isinstance(
            graph.G.nodes[int(plan.loop.node_id)].get("expr_obj"),
            ast.comprehension,
        )
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(
            data.get("expr_obj"),
            (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
        )
        and any(
            int(parent) == int(plan.loop.node_id)
            and str(role) == "generators"
            for parent, role in data.get("parents", ())
        )
    ))
    comprehension_descendants = {
        owner: nx.descendants(graph.G, owner)
        for owner in comprehension_owners
    }
    expression_nodes = {
        id(data.get("expr_obj")): int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.AST)
    }
    comprehension_body_members = {}
    for owner in comprehension_owners:
        aggregate = graph.G.nodes[owner].get("expr_obj")
        body_roots = (
            (aggregate.key, aggregate.value)
            if isinstance(aggregate, ast.DictComp)
            else (aggregate.elt,)
        )
        comprehension_body_members[owner] = frozenset(
            expression_nodes[id(member)]
            for body_root in body_roots
            for member in ast.walk(body_root)
            if id(member) in expression_nodes
        )
    loop_control_frontiers = tuple(
        (
            int(plan.loop.node_id),
            int(control_id),
            int(getattr(expression, "end_lineno", expression.lineno)),
            frozenset(map(int, plan.loop.body_nodes)),
        )
        for plan in plans
        for control_id in plan.loop.body_nodes
        if control_id in graph.G
        for expression in (graph.G.nodes[control_id].get("expr_obj"),)
        if isinstance(expression, ast.If)
        and any(
            isinstance(member, (ast.Break, ast.Continue))
            for member in ast.walk(expression)
        )
    )
    # Invert every owner->members relation into member->owners once, in owner
    # order, so each node's key is a set of dict lookups rather than a rescan of
    # every plan/callsite/comprehension/frontier. The old per-node membership
    # scans were O(nodes x owners) and dominated the planning pass; this is
    # O(sum of membership sizes + nodes) with byte-identical output.
    def _invert(owners, member_sets):
        by_node: dict[int, list[Any]] = {}
        for owner in owners:
            for member in member_sets(owner):
                by_node.setdefault(int(member), []).append(owner)
        return by_node

    loops_by_node = _invert(
        plans, lambda plan: plan.loop.body_nodes,
    )
    loops_by_node = {
        node_id: tuple(plan.loop.node_id for plan in owners)
        for node_id, owners in loops_by_node.items()
    }
    comp_body_by_node = _invert(
        comprehension_owners, lambda owner: comprehension_body_members[owner],
    )
    callsites_by_node = _invert(
        callsites, lambda callsite: call_descendants[callsite],
    )
    comp_desc_by_node = _invert(
        comprehension_owners, lambda owner: comprehension_descendants[owner],
    )
    frontiers_by_node: dict[int, list[tuple[int, int]]] = {}
    for loop_id, control_id, end_line, body in loop_control_frontiers:
        for member in body:
            member = int(member)
            if member not in graph.G:
                continue
            if int(getattr(
                graph.G.nodes[member].get("expr_obj"), "lineno", -1,
            )) > end_line:
                frontiers_by_node.setdefault(member, []).append(
                    (loop_id, control_id)
                )

    return {
        node_id: (
            tuple(loops_by_node.get(int(node_id), ())),
            tuple(sorted(branches.get(node_id, ()))),
            tuple(comp_body_by_node.get(int(node_id), ())),
            tuple(callsites_by_node.get(int(node_id), ())),
            tuple(comp_desc_by_node.get(int(node_id), ())),
            tuple(frontiers_by_node.get(int(node_id), ())),
            tuple(branch_frontiers_by_node.get(int(node_id), ())),
        )
        for node_id in node_ids
    }


def _closure_routing_dependencies(
    graph: Any,
) -> tuple[frozenset[tuple[int, int]], frozenset[int]]:
    """Recover causal facts carried by closure routing rather than by edges.

    A call to a compiled function supplies that callee's free variables by
    name: the coordinator looks the name up in this graph's identity table and
    evaluates whichever definition precedes the call.  No parent reference or
    graph edge records that read, so region planning cannot see it.  Two
    consequences follow, and both are returned here.

    The routed definition is observed from outside the shader, so it is a
    compartment boundary value and must be published.  The call also depends
    on it, so a region may not internalize both the definition and any value
    produced by that call; doing so would make the region depend on its own
    unpublished result.  The returned edges are declared to the reducer as
    structural dependencies exactly like a path through a coordinator node.
    """

    identities = graph.G.graph.get("identity_table", {}) or {}
    if not identities:
        return frozenset(), frozenset()
    function_table = getattr(graph, "function_table", None)
    if function_table is None:
        return frozenset(), frozenset()

    def free_binding_names(reference: int) -> frozenset[str]:
        try:
            entry = function_table.entry(int(reference))
        except (KeyError, TypeError, ValueError):
            return frozenset()
        callee = getattr(entry, "graph", None)
        if callee is None:
            return frozenset()
        return frozenset(
            str(name)
            for _node_id, data in callee.G.nodes(data=True)
            if data.get("type") == "Input"
            for attributes in ((data.get("attributes") or {}),)
            if attributes.get("binding_kind") == "external"
            for name in (attributes.get("binding_name"),)
            if name is not None
        )

    order_index = {
        node_id: index
        for index, node_id in enumerate(_dependency_order(graph))
    }
    free_names: dict[int, frozenset[str]] = {}
    edges: set[tuple[int, int]] = set()
    routed: set[int] = set()
    for call_id, data in graph.G.nodes(data=True):
        attributes = data.get("attributes") or {}
        reference = attributes.get("callee_ref")
        if reference is None:
            reference = attributes.get("method_ref")
        if reference is None:
            continue
        names = free_names.get(int(reference))
        if names is None:
            names = free_binding_names(reference)
            free_names[int(reference)] = names
        call_position = order_index[call_id]
        for name in names:
            for definition in identities.get(name, ()):
                # Only a definition that can already have executed is a
                # candidate for this call site.  Declaring a later one would
                # invent a backward dependency the schedule cannot satisfy.
                if (
                    definition in order_index
                    and order_index[definition] < call_position
                ):
                    edges.add((definition, call_id))
                    routed.add(definition)
    return frozenset(edges), frozenset(routed)


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


def _graph_source_binding_name(graph: Any, node_id: int) -> str | None:
    """Recover a dotted public-input path for a numerical boundary node."""

    data = graph.G.nodes[int(node_id)]
    if data.get("type") == "Input":
        return str(data.get("label"))
    if data.get("type") != "Attribute":
        return None
    parent = next(
        (
            int(candidate)
            for candidate, role in (data.get("parents") or ())
            if str(role) == "value"
        ),
        None,
    )
    attribute = getattr(data.get("expr_obj"), "attr", None)
    if parent is None or attribute is None:
        return None
    root = _graph_source_binding_name(graph, parent)
    return None if root is None else f"{root}.{attribute}"


def _bind_capture_tape(
    value: Any,
    tape: Any,
    cache: dict[int, Any] | None = None,
) -> Any:
    """Bind tensor storage to the one discovery tape without copying values."""

    if isinstance(value, tuple):
        return tuple(_bind_capture_tape(item, tape, cache) for item in value)
    if isinstance(value, list):
        return [_bind_capture_tape(item, tape, cache) for item in value]
    if isinstance(value, dict):
        return {
            key: _bind_capture_tape(item, tape, cache)
            for key, item in value.items()
        }
    if not isinstance(value, AbstractTensor):
        return value
    if getattr(value, "_tape", None) is tape:
        return value
    identity = id(value)
    if cache is not None and identity in cache:
        return cache[identity]
    rebound = type(value)(
        track_time=value.track_time,
        requires_grad=value.requires_grad,
        tape=tape,
    )
    rebound.data = value.data
    if cache is not None:
        cache[identity] = rebound
    return rebound


def _capture_storage_identity(value: Any):
    """Return the exact resident range represented by a capture wrapper."""

    storage = (
        value
        if getattr(value, "_storage", None) is not None
        else getattr(value, "data", None)
    )
    if storage is None:
        return None
    physical = getattr(storage, "_storage", None)
    if physical is not None:
        return (
            id(physical),
            getattr(storage, "_offset", None),
            getattr(storage, "_count", None),
            str(getattr(storage, "_dtype", None)),
        )
    return ("object", id(storage))


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
    captured_feed_meta = _captured_storage_meta({
        int(value_id): value
        for value_id, value in captured.feeds.items()
    })
    def remap_program(source, *, public_outputs=False):
        target = FusedProgram(
            version=source.version,
            feeds={remap(value_id) for value_id in source.feeds},
            steps=[
                OpStep(
                    step_id=step.step_id,
                    op_name=step.op_name,
                    input_ids=[
                        remap(value_id) for value_id in step.input_ids
                    ],
                    attrs=dict(step.attrs),
                    result_id=remap(step.result_id),
                    mode_sensitive=step.mode_sensitive,
                    level=step.level,
                )
                for step in source.steps
            ],
            outputs=(
                {
                    f"value_{output_id}": output_id
                    for output_id in output_ids
                }
                if public_outputs
                else {
                    name: remap(value_id)
                    for name, value_id in source.outputs.items()
                }
            ),
            state_in=(
                None
                if source.state_in is None
                else {remap(value_id) for value_id in source.state_in}
            ),
            meta={
                **{
                    remap(value_id): meta
                    for value_id, meta in captured_feed_meta.items()
                    if value_id in source.feeds
                },
                **{
                    remap(value_id): meta
                    for value_id, meta in (source.meta or {}).items()
                },
            },
            extras=source.extras,
        )
        if hasattr(source, "glsl_linear_output_shape"):
            target.glsl_linear_output_shape = tuple(
                source.glsl_linear_output_shape
            )
        return target

    remapped = remap_program(program, public_outputs=True)
    return CapturedFusedProgram(
        remapped,
        # Captured tensors are evidence used to reveal the operation sequence;
        # they are not part of the compiled program.  Runtime data enters only
        # through the remapped feed IDs.  Retaining numeric payloads here would
        # compile an observed instance instead of the program.
        {},
        tuple(
            remap_program(stage)
            for stage in captured.stages
        ),
    )


def _remap_captured_all_ids(
    captured: CapturedFusedProgram,
    id_map: Mapping[int, int],
) -> CapturedFusedProgram:
    """Namespace every captured IR identity for hierarchical composition."""

    def remap_program(source):
        remap = lambda value_id: int(
            id_map.get(int(value_id), int(value_id))
        )
        target = FusedProgram(
            version=source.version,
            feeds={remap(value_id) for value_id in source.feeds},
            steps=[
                OpStep(
                    step.step_id,
                    step.op_name,
                    [remap(value_id) for value_id in step.input_ids],
                    dict(step.attrs),
                    remap(step.result_id),
                    step.mode_sensitive,
                    step.level,
                )
                for step in source.steps
            ],
            outputs={
                name: remap(value_id)
                for name, value_id in source.outputs.items()
            },
            state_in=(
                None
                if source.state_in is None
                else {remap(value_id) for value_id in source.state_in}
            ),
            meta={
                remap(value_id): meta
                for value_id, meta in (source.meta or {}).items()
            },
            extras=source.extras,
        )
        if hasattr(source, "glsl_linear_output_shape"):
            target.glsl_linear_output_shape = tuple(
                source.glsl_linear_output_shape
            )
        return target

    return CapturedFusedProgram(
        remap_program(captured.program),
        {},
        tuple(remap_program(stage) for stage in captured.stages),
    )


def _wire_planned_step_inputs(
    captured: CapturedFusedProgram,
    step_inputs: Mapping[int, tuple[tuple[int, int], ...]],
) -> CapturedFusedProgram:
    """Replace primitive operand IDs with their planned PG input positions.

    ``step_inputs`` is emitted synchronously by GraphDeepCompiler for one
    operation invocation.  Keys are primitive result occurrence IDs and
    values are the ProcessGraph parent IDs already present in the compartment.
    No payload, object, storage, or tape traversal participates here.
    """

    def rewrite(source: FusedProgram) -> FusedProgram:
        steps = []
        for step in source.steps:
            planned = step_inputs.get(int(step.result_id))
            inputs = (
                tuple(int(graph_id) for _primitive_id, graph_id in planned)
                if planned is not None
                and len(planned) == len(step.input_ids)
                else tuple(map(int, step.input_ids))
            )
            steps.append(OpStep(
                step.step_id,
                step.op_name,
                inputs,
                dict(step.attrs),
                int(step.result_id),
                step.mode_sensitive,
                step.level,
            ))
        produced = {int(step.result_id) for step in steps}
        feeds = {
            int(value_id)
            for step in steps
            for value_id in step.input_ids
            if int(value_id) not in produced
        }
        meta = dict(source.meta or {})
        for positional_inputs in step_inputs.values():
            for primitive_id, graph_id in positional_inputs:
                if int(primitive_id) in meta:
                    meta.setdefault(int(graph_id), meta[int(primitive_id)])
        return FusedProgram(
            version=source.version,
            feeds=feeds,
            steps=steps,
            outputs=dict(source.outputs),
            state_in=source.state_in,
            meta=meta,
            extras=source.extras,
        )

    return CapturedFusedProgram(
        rewrite(captured.program),
        {},
        tuple(rewrite(stage) for stage in captured.stages),
    )


def _collapse_planned_collection_materializations(
    captured: CapturedFusedProgram,
    aliases: Mapping[int, int],
) -> CapturedFusedProgram:
    """Replace observed aggregate leaves with one planner collection input.

    The materialization remains a real numerical producer with its own result
    ID.  Only its discovery-time variadic inputs are replaced; the loop plan's
    resident result is the single runtime input.
    """

    aliases = {
        int(result_id): int(collection_id)
        for result_id, collection_id in aliases.items()
    }
    if not aliases:
        return captured

    def rewrite(source: FusedProgram) -> FusedProgram:
        metadata = dict(source.meta or {})
        steps = [
            OpStep(
                step.step_id,
                step.op_name,
                (
                    [aliases[int(step.result_id)]]
                    if int(step.result_id) in aliases
                    else list(step.input_ids)
                ),
                dict(step.attrs),
                int(step.result_id),
                step.mode_sensitive,
                step.level,
            )
            for step in source.steps
        ]
        for step in source.steps:
            collection_id = aliases.get(int(step.result_id))
            result_meta = metadata.get(int(step.result_id))
            if collection_id is not None and result_meta is not None:
                metadata.setdefault(int(collection_id), result_meta)
        produced = {int(step.result_id) for step in steps}
        feeds = {
            int(value_id)
            for step in steps
            for value_id in step.input_ids
            if int(value_id) not in produced
        }
        target = FusedProgram(
            version=source.version,
            feeds=feeds,
            steps=steps,
            outputs={
                name: int(value_id)
                for name, value_id in source.outputs.items()
            },
            state_in=(
                None
                if source.state_in is None
                else {int(value_id) for value_id in source.state_in}
            ),
            meta=metadata,
            extras=source.extras,
        )
        if hasattr(source, "glsl_linear_output_shape"):
            target.glsl_linear_output_shape = tuple(
                source.glsl_linear_output_shape
            )
        return target

    full = rewrite(captured.program)
    stages = tuple(
        rewritten
        for stage in captured.stages
        if (rewritten := rewrite(stage)).steps
    )
    return CapturedFusedProgram(full, {}, stages)


def _project_captured_program(
    captured: CapturedFusedProgram,
    *,
    output_ids: tuple[int, ...],
    boundary_ids: tuple[int, ...],
    allowed_result_ids: tuple[int, ...] | None = None,
) -> CapturedFusedProgram:
    """Project a planner region from an already-lowered shell program.

    This function operates only on backend-neutral IR.  It deliberately has
    no access to a GradTape or captured tensor payload, so requesting another
    region cannot rediscover, replay, or recursively lower the Python
    execution.  ``boundary_ids`` are planner cut points: dependencies stop
    there even if another region in the complete shell produces that value.
    """

    outputs = tuple(int(value) for value in output_ids)
    boundaries = {int(value) for value in boundary_ids}
    allowed_results = (
        None
        if allowed_result_ids is None
        else {int(value) for value in allowed_result_ids}
    )
    synthetic_results = {
        int(value_id)
        for program in (captured.program, *captured.execution_programs)
        for value_id in (
            (program.extras or {}).get("synthetic_result_ids", ())
        )
    }
    all_steps = [
        step
        for program in captured.execution_programs
        for step in program.steps
    ]
    all_meta = {
        int(value_id): meta
        for program in (captured.program, *captured.execution_programs)
        for value_id, meta in (program.meta or {}).items()
    }
    producers = {int(step.result_id): step for step in all_steps}
    available_feeds = {
        int(value_id)
        for program in (captured.program, *captured.execution_programs)
        for value_id in program.feeds
    }
    missing_outputs = tuple(
        value_id for value_id in outputs
        if value_id not in producers and value_id not in available_feeds
    )
    if missing_outputs:
        raise ValueError(
            "planner region outputs are absent from the once-lowered shell "
            f"program: missing={missing_outputs!r}; "
            f"available={tuple(sorted(producers))!r}"
        )
    # A value may be consumed again inside the stage that creates it and also
    # cross a later stage/region boundary.  `produced - consumed` alone is
    # therefore not liveness: it drops such values from the stage ABI and the
    # later consumer is misclassified as an external shell input.  The
    # once-lowered program already states every cross-stage feed, while the
    # planner states every region output; retain both sets as explicit
    # materialization boundaries.
    cross_stage_feeds = {
        int(value_id)
        for source in captured.execution_programs
        for value_id in source.feeds
    }
    required_live_outs = set(outputs) | cross_stage_feeds
    needed = set(outputs)
    selected: set[int] = set()
    pending = [(value_id, True) for value_id in outputs]
    while pending:
        value_id, is_region_output = pending.pop()
        value_id = int(value_id)
        if (
            (value_id in boundaries and not is_region_output)
            or (
                allowed_results is not None
                and value_id not in allowed_results
                and value_id not in synthetic_results
                and not is_region_output
            )
            or value_id in selected
        ):
            continue
        step = producers.get(value_id)
        if step is None:
            continue
        selected.add(value_id)
        for input_id in step.input_ids:
            input_id = int(input_id)
            needed.add(input_id)
            if (
                input_id not in boundaries
                and (
                    allowed_results is None
                    or input_id in allowed_results
                    or input_id in synthetic_results
                )
            ):
                pending.append((input_id, False))

    def project(source: FusedProgram, *, public: bool = False):
        steps = [
            OpStep(
                step.step_id,
                step.op_name,
                list(step.input_ids),
                dict(step.attrs),
                step.result_id,
                step.mode_sensitive,
                step.level,
            )
            for step in source.steps
            if int(step.result_id) in selected
        ]
        produced = {int(step.result_id) for step in steps}
        consumed = {
            int(input_id)
            for step in steps
            for input_id in step.input_ids
        }
        stage_terminals = (
            (produced - consumed)
            | (produced & required_live_outs)
        )
        feeds = {
            int(input_id)
            for step in steps
            for input_id in step.input_ids
            if int(input_id) not in produced
        }
        if public:
            feeds.update(
                value_id
                for value_id in outputs
                if value_id not in produced
            )
        target = FusedProgram(
            version=source.version,
            feeds=feeds,
            steps=steps,
            outputs=(
                {f"value_{value_id}": value_id for value_id in outputs}
                if public
                else {
                    f"value_{value_id}": value_id
                    for value_id in sorted(stage_terminals)
                }
            ),
            state_in=(
                None
                if source.state_in is None
                else {
                    int(value_id)
                    for value_id in source.state_in
                    if int(value_id) in needed
                }
            ),
            meta={
                value_id: (
                    (source.meta or {}).get(value_id)
                    or all_meta[value_id]
                )
                for value_id in (
                    needed | produced | feeds | set(outputs)
                )
                if (
                    value_id in (source.meta or {})
                    or value_id in all_meta
                )
            },
            extras=source.extras,
        )
        if hasattr(source, "glsl_linear_output_shape"):
            target.glsl_linear_output_shape = tuple(
                source.glsl_linear_output_shape
            )
        return target

    manifest = project(captured.program, public=True)
    stages = tuple(
        projected
        for source in captured.execution_programs
        if (projected := project(source)).steps
    )
    return CapturedFusedProgram(manifest, {}, stages)


def _resolve_binding_name(shell: Any, captured: Any, feed_id: int):
    """Which source parameter a feed came from, if the capture knows.

    Object identity is tried first and is exact when it survives. When the
    value was rewrapped on its way to the tape -- which is the usual case --
    the resident range underneath is unchanged, so the storage recorded at
    binding time still identifies it.
    """

    # The shell that recorded the binding is not always the shell that
    # lowers the program: a function's inputs are bound in its own shell,
    # while the owner is the one that builds the origins. Search the whole
    # planned tree rather than assuming they are the same object.
    value = (getattr(captured, "feeds", None) or {}).get(feed_id)
    storage = _capture_storage_identity(value) if value is not None else None
    candidates = [shell]
    try:
        candidates.extend(_walk_planned_shells(
            shell, include_function_registry=False
        ))
    except Exception:
        pass
    for candidate in candidates:
        names = getattr(candidate, "_capture_input_names", None) or {}
        direct = names.get(feed_id)
        if direct:
            return direct
        for aggregate_map in (
            getattr(candidate, "compiled_aggregate_feed_paths", ()) or ()
        ):
            for capture_id, graph_input_id, path in aggregate_map:
                if int(capture_id) != int(feed_id):
                    continue
                graph = getattr(candidate, "process_graph", None)
                if graph is None or int(graph_input_id) not in graph.G:
                    continue
                root = _compiler_input_name(
                    graph.G.nodes[int(graph_input_id)]["label"]
                )
                suffix = "".join(
                    f".{part}" if isinstance(part, str)
                    and part.isidentifier()
                    else f"[{part!r}]"
                    for part in path
                )
                return root + suffix
    if storage is None:
        return None
    for candidate in candidates:
        by_storage = getattr(candidate, "_capture_input_storage", None) or {}
        found = by_storage.get(storage)
        if found:
            return found
    return None


def _capture_feed_aliases(
    captured: CapturedFusedProgram,
    feed_ids: dict[int, int],
) -> dict[int, int]:
    """Correlate capture wrappers that share one resident boundary value."""

    remapped = dict(feed_ids)

    anchors = {
        feed_id: (
            captured.feeds[feed_id],
            _capture_storage_identity(captured.feeds[feed_id]),
        )
        for feed_id in feed_ids
        if feed_id in captured.feeds
    }
    for candidate_id, candidate in captured.feeds.items():
        if candidate_id in remapped:
            continue
        candidate_storage = _capture_storage_identity(candidate)
        for anchor_id, (anchor, anchor_storage) in anchors.items():
            if (
                candidate is anchor
                or (
                    candidate_storage is not None
                    and candidate_storage == anchor_storage
                )
            ):
                remapped[candidate_id] = feed_ids[anchor_id]
                break
        if candidate_id in remapped:
            continue
        origin = (
            (captured.program.extras or {})
            .get("capture_feed_origins", {})
            .get(candidate_id)
        )
        if origin is not None and origin.get("op") is not None:
            # A recorded result that becomes a later stage feed is an
            # on-device program value.  Similar shape/dtype does not make it
            # an alias of a source boundary—doing so can rewrite a loop result
            # back to its initial input and erase the operation.
            continue
        metadata = captured.program.meta or {}
        candidate_meta = metadata.get(candidate_id)
        compatible = [
            anchor_id
            for anchor_id in anchors
            if metadata.get(anchor_id) == candidate_meta
        ]
        # A non-recorded capture root with one uniquely compatible boundary
        # descriptor is another wrapper for that boundary value.  Shape/dtype
        # matching is deliberately not used when two graph inputs are
        # compatible; that ambiguity must remain an error.
        if len(compatible) == 1:
            remapped[candidate_id] = feed_ids[compatible[0]]
    return remapped


def _unique_runtime_feed_aliases(
    program: FusedProgram,
    missing_feed_ids: Iterable[int],
    runtime_feeds: dict[int, Any],
) -> dict[int, AbstractTensor]:
    """Resolve an orphan feed only when one runtime tensor can match it."""

    leaves: list[AbstractTensor] = []

    def collect(value):
        if isinstance(value, AbstractTensor):
            leaves.append(value)
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)

    for value in runtime_feeds.values():
        collect(value)

    resolved = {}
    metadata = program.meta or {}
    for feed_id in missing_feed_ids:
        meta = metadata.get(feed_id)
        if meta is None:
            continue
        compatible = [
            value
            for value in leaves
            if tuple(value.shape) == tuple(meta.shape or ())
            and str(value.dtype).lower().rsplit(".", 1)[-1]
            == str(meta.dtype).lower().rsplit(".", 1)[-1]
        ]
        if len(compatible) == 1:
            resolved[feed_id] = compatible[0]
    return resolved


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


class _SourceReturnSignal(Exception):
    """Internal non-error transfer for a compiled Python ``return``."""

    def __init__(self, value: Any):
        super().__init__()
        self.value = value


def _call_arguments(
    parents: tuple[tuple[int, str], ...],
    values: dict[int, Any],
    static_arguments: dict[str, Any] | None = None,
    graph: Any | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Reconstruct positional and keyword arguments from ProcessGraph roles."""

    positional: dict[int, Any] = {}
    starred: set[int] = set()
    keywords: dict[str, Any] = {}
    fallback_index = 1 << 30
    for parent, role_value in parents:
        role = str(role_value)
        if role == "kwargs":
            keywords.update(dict(values[parent]))
            continue
        if role == "args":
            for value in values[parent]:
                positional[len(positional)] = value
            continue
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
        elif role == "arg":
            index = len(positional)
        else:
            continue
        positional[index] = values[parent]
        if (
            graph is not None
            and parent in graph.G
            and isinstance(
                graph.G.nodes[parent].get("expr_obj"), ast.Starred
            )
        ):
            starred.add(index)
        fallback_index += 1
    for role, value in (static_arguments or {}).items():
        if role.startswith("kw:"):
            keywords[role[3:]] = value
        elif (index := _positional_argument_index(role)) is not None:
            positional[index] = value
    arguments = []
    for index in sorted(positional):
        value = positional[index]
        if index in starred:
            arguments.extend(value)
        else:
            arguments.append(value)
    return arguments, keywords


def _static_python_value(bindings: dict[str, Any], path: str) -> Any:
    """Resolve one reducer-retained Python reference for coordination."""

    parts = str(path).split(".")
    try:
        value = bindings[parts[0]]
    except KeyError as exc:
        candidates = []
        for module in tuple(sys.modules.values()):
            namespace = getattr(module, "__dict__", None)
            if isinstance(namespace, dict) and parts[0] in namespace:
                candidates.append(namespace[parts[0]])
        identities = {id(candidate): candidate for candidate in candidates}
        if len(identities) != 1:
            raise KeyError(
                f"static Python reference {path!r} has no retained binding"
            ) from exc
        value = next(iter(identities.values()))
        bindings[parts[0]] = value
    for part in parts[1:]:
        value = getattr(value, part)
    return value


def _tensorize_graph_input(value: Any, *, device: Any) -> Any:
    """Move array-shaped public inputs onto the selected AbstractTensor backend."""

    def has_array_protocol(candidate: Any) -> bool:
        return (
            isinstance(candidate, np.ndarray)
            or callable(getattr(candidate, "__array__", None))
            or hasattr(candidate, "__cuda_array_interface__")
            or callable(getattr(candidate, "__dlpack__", None))
            or (
                getattr(candidate, "dtype", None) is not None
                and getattr(candidate, "shape", None) is not None
                and not callable(getattr(candidate, "shape", None))
            )
        )

    if isinstance(value, AbstractTensor):
        return value
    if isinstance(value, np.dtype):
        # NumPy dtype descriptors misleadingly expose ``shape == ()`` even
        # though they are type metadata, not zero-dimensional array payloads.
        # Dtype expressions belong to the AbstractTensor program and must
        # arrive at calls such as ``zeros(..., dtype=octets.dtype)`` unchanged.
        # Uploading the descriptor would manufacture an object-dtype tensor
        # and make the compiler fight the tensor operation's own type rules.
        return value
    if isinstance(value, _CompiledStructuralObject):
        # Dataclass/closure state can contain tensor fields and therefore
        # expose a derived ``shape`` through structural resolution.  The
        # aggregate itself is not an ndarray and must stay decomposable into
        # its named fields; coercing it produces an object-dtype array and
        # destroys the compiler-visible packet/closure schema.
        return value
    if isinstance(value, type):
        # A retained Python class is a structural reference the coordinator
        # calls through.  Its ``shape`` attribute is the unbound descriptor of
        # its instances, not the extent of an array-shaped input.
        return value
    if isinstance(value, ModuleType):
        return value
    fields = getattr(value, "__dict__", None)
    if isinstance(fields, dict) and not callable(value):
        replacements = {}
        for name, field_value in fields.items():
            if isinstance(field_value, (type, ModuleType)) or callable(
                field_value
            ):
                continue
            if isinstance(field_value, AbstractTensor):
                continue
            if isinstance(field_value, np.dtype):
                continue
            field_shape = getattr(field_value, "shape", None)
            if (
                field_shape is None
                or callable(field_shape)
                or not has_array_protocol(field_value)
            ):
                continue
            replacements[str(name)] = AbstractTensor.tensor(
                field_value,
                device=device,
            )
        if replacements:
            # Discovery must not rewrite the caller's live Python object. A
            # shallow structural copy retains its scalar configuration and
            # the preallocated storage beneath each tensor wrapper, while
            # method assignments describe arena state on the capture copy.
            resident = copy.copy(value)
            for name, field_value in replacements.items():
                setattr(resident, name, field_value)
            return resident
    shape = getattr(value, "shape", None)
    if shape is None or callable(shape):
        # Imported scientific modules can export a public function named
        # ``shape`` (SymPy is one example).  Attribute presence alone is not
        # an array protocol: a callable shape describes an operation supplied
        # by the namespace, not the resident extent of the namespace object.
        # Keep such coordinator values intact so nested closures can continue
        # to call module constructors such as ``sympy.Symbol``.
        return value
    if not has_array_protocol(value):
        # Domain objects may expose a structural ``shape`` through dynamic
        # attribute routing.  Shape alone is not an upload protocol.
        return value
    return AbstractTensor.tensor(value, device=device)


def _coordinate_scheduled_capture_impl(
    shell: Any,
    initial_values: dict[str | int, Any],
    *,
    device: Any = None,
    capture: bool = True,
    discovery_session: dict[str, Any] | None = None,
) -> Any:
    """Execute structural boundaries and capture each planned numeric region.

    This is the first-invocation coordinator for a function deployment shell.
    Python objects remain structural values; array-shaped public inputs become
    resident tensors.  External calls and control/container nodes run between
    topologically closed numerical regions, while those regions alone enter
    forward capture for backend compilation.
    """

    if capture:
        invocation_slot = shell._capture_invocations
        shell._capture_invocations += 1
    else:
        invocation_slot = (
            shell._execute_invocations
            % max(1, shell.planned_invocation_slots)
        )
        shell._execute_invocations += 1
    graph = shell.process_graph
    supplied = dict(initial_values)
    live_binding_unavailable = object()

    def live_function_module_binding(name: str) -> Any:
        """Read a callee global from its owning live Python module.

        Static bindings describe the namespace at planning time.  A program
        may deliberately populate module globals later (lazy imports are the
        usual case), so a compiled callee must consult its own module at the
        source-ordered point where the name is first read.  The function
        table's qualified identity keeps this lookup lexical and prevents
        globals from unrelated discovered modules from being merged.
        """

        function_table = getattr(graph, "function_table", None)
        function_reference = graph.G.graph.get("function_ref")
        if function_table is None or function_reference is None:
            return live_binding_unavailable
        try:
            qualified_name = str(
                function_table.entry(int(function_reference)).qualified_name
            )
        except (KeyError, TypeError, ValueError):
            return live_binding_unavailable
        qualified_candidates = [qualified_name]
        method_owner = graph.G.graph.get("method_owner")
        if method_owner is not None:
            map_ir = graph.G.graph.get("map_ir") or {}
            for object_schema in map_ir.get("objects", ()):
                if str(object_schema.get("class_name")) == str(method_owner):
                    qualified_candidates.append(
                        str(object_schema.get("class_identity", ""))
                    )
        for qualified_candidate in qualified_candidates:
            components = qualified_candidate.split(".")
            for stop in range(len(components), 0, -1):
                module = sys.modules.get(".".join(components[:stop]))
                if module is None:
                    continue
                namespace = vars(module)
                return namespace.get(name, live_binding_unavailable)
        return live_binding_unavailable

    values: dict[int, Any] = {
        int(key): value
        for key, value in supplied.items()
        if isinstance(key, int)
    }
    function_identities = graph.G.graph.get("identity_table", {}) or {}

    def record_static_input_fields(
        binding_name: str,
        value: Any,
        path: tuple[str, ...] = (),
        visiting: frozenset[int] = frozenset(),
    ) -> None:
        if value is None or isinstance(
            value, (bool, int, float, str, bytes)
        ):
            shell._capture_input_static_fields[
                (str(binding_name), tuple(path))
            ] = value
            return
        identity = id(value)
        if identity in visiting or len(path) >= 8:
            return
        fields = getattr(value, "__dict__", None)
        if not isinstance(fields, dict) or callable(value):
            return
        next_visiting = visiting | {identity}
        for field_name, field_value in fields.items():
            record_static_input_fields(
                binding_name,
                field_value,
                (*path, str(field_name)),
                next_visiting,
            )

    def bound_target_names(target: ast.AST | None) -> tuple[str, ...]:
        if target is None:
            return ()
        if isinstance(target, ast.Name):
            return (target.id,)
        if isinstance(target, ast.Starred):
            return bound_target_names(target.value)
        if isinstance(target, (ast.Tuple, ast.List)):
            return tuple(
                name
                for item in target.elts
                for name in bound_target_names(item)
            )
        return ()

    source_with_bindings = frozenset(
        name
        for statement in graph.G.graph.get("function_body", ())
        for member in ast.walk(statement)
        if isinstance(member, ast.With)
        for item in member.items
        for name in bound_target_names(item.optional_vars)
    )

    def has_deferred_local_definition(name: str, input_id: int) -> bool:
        """Whether source execution, rather than invocation, defines name."""

        return name in source_with_bindings or any(
            int(identity) != int(input_id)
            and int(identity) in graph.G
            and str(graph.G.nodes[int(identity)].get("type"))
            not in {"Input", "input"}
            for identity in function_identities.get(name, ())
        )

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
            if binding_kind in {"loop", "exception"}:
                continue
            live_binding = live_function_module_binding(name)
            if (
                binding_kind == "external"
                and live_binding is not live_binding_unavailable
            ):
                values[node_id] = live_binding
                shell.static_python_bindings[name] = live_binding
                continue
            class_descriptor = graph.G.graph.get(
                "class_table",
                {},
            ).get(name)
            if binding_kind == "external" and class_descriptor is not None:
                values[node_id] = _CompiledStructuralClass(
                    name,
                    class_descriptor,
                )
                continue
            if (
                binding_kind == "external"
                and name in shell.static_python_bindings
            ):
                values[node_id] = shell.static_python_bindings[name]
                continue
            if (
                binding_kind == "external"
                and name not in supplied
                and has_deferred_local_definition(name, node_id)
            ):
                # Control-flow normalization can retain an Input-shaped
                # occurrence before its assignment producer (notably a
                # local assigned inside ``try``).  It is a Python local, not
                # an invocation requirement; its use will resolve from the
                # source frame after the defining statement executes.
                continue
            if name not in supplied:
                raise KeyError(
                    f"missing ProcessGraph input {name!r} in "
                    f"{graph.G.graph.get('function_name', '?')} at node "
                    f"{node_id}; binding_kind={binding_kind!r}"
                )
            supplied_value = supplied[name]
            supplied_shape = getattr(supplied_value, "shape", None)
            if (
                isinstance(supplied_value, AbstractTensor)
                or isinstance(supplied_value, np.ndarray)
                or callable(getattr(supplied_value, "__array__", None))
                or (
                    supplied_shape is not None
                    and not callable(supplied_shape)
                    and getattr(supplied_value, "dtype", None) is not None
                )
            ):
                # Public array parameters enter the selected capture device
                # at the function boundary.  Deferring this until the first
                # numerical region lets an enclosing structural call observe
                # NumPy while its callee observes AbstractTensor, changing
                # source branches such as dt_system._restore_type and baking
                # an incidental host conversion into the graph.
                values[node_id] = _tensorize_graph_input(
                    supplied_value, device=device
                )
            else:
                values[node_id] = supplied_value
            shell._capture_input_type_names.setdefault(
                str(name), set()
            ).add(type(values[node_id]).__qualname__)
            if isinstance(values[node_id], AbstractTensor):
                shell._capture_tensor_input_names.add(str(name))
            record_static_input_fields(str(name), values[node_id])
            # Keep the name against both the object and the storage under
            # it. The object is wrapped into an AbstractTensor before it is
            # ever used, so object identity alone does not survive to the
            # feed; the resident range does, which is the same correlation
            # _capture_feed_aliases uses.
            try:
                shell._capture_input_names[id(values[node_id])] = str(name)
                storage = _capture_storage_identity(values[node_id])
                if storage is not None:
                    shell._capture_input_storage[storage] = str(name)
            except AttributeError:
                pass

    # Source execution can prove a graph-inert expression live at runtime
    # (for example the selected arm of an IfExp), so this coordinator view is
    # intentionally mutable even though the planner returns a frozenset.
    inert_nodes = set(_inert_routing_nodes(graph))
    regions = tuple(
        zip(
            shell.deep_compilers,
            shell.dispatch_subgraphs,
            shell.ephemeral_callables,
        )
    )
    region_for_node: dict[int, int] = {}
    coordinator_override_nodes: set[int] = set()
    for region_index, (_compiler, subgraph, _ephemeral) in enumerate(regions):
        if region_index in shell.coordinator_region_indices:
            continue
        for node_id in subgraph.G.graph.get("deployment_nodes", ()):
            if node_id in region_for_node:
                raise RuntimeError(
                    f"ProcessGraph node {node_id} belongs to two dispatches"
                )
            region_for_node[node_id] = region_index
        deployment_nodes = tuple(
            subgraph.G.graph.get("deployment_nodes", ())
        )
        if any(
            graph.G.nodes[node_id].get("type") == "Indexed"
            and any(
                parent in graph.G
                and graph.G.nodes[parent].get("type")
                in {"Const", "const", "Constant"}
                and isinstance(
                    _constant_value(graph.G.nodes[parent]),
                    (str, bytes),
                )
                for parent, role in (
                    graph.G.nodes[node_id].get("parents") or ()
                )
                if str(role) == "index"
            )
            for node_id in deployment_nodes
            if node_id in graph.G
        ):
            coordinator_override_nodes.update(deployment_nodes)
        if any(
            graph.G.nodes[node_id].get("type") == "IndexedStore"
            and any(
                isinstance(values.get(int(parent)), (dict, list, set))
                for parent, role in (
                    graph.G.nodes[node_id].get("parents") or ()
                )
                if str(role) == "base"
            )
            for node_id in deployment_nodes
            if node_id in graph.G
        ):
            # A fused region cannot clone or shader-store a Python container.
            # Runtime input type is authoritative during discovery; route the
            # whole coupled region through coordinator mutation so no sibling
            # dispatch claims the same structural store first.
            coordinator_override_nodes.update(deployment_nodes)

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
    function_body = tuple(graph.G.graph.get("function_body", ()))
    source_returns: list[ast.Return] = []
    nonlocal_names: set[str] = set()

    class _CurrentFunctionReturnVisitor(ast.NodeVisitor):
        def visit_Return(self, statement):
            if statement.value is not None:
                source_returns.append(statement)

        def visit_Nonlocal(self, statement):
            nonlocal_names.update(map(str, statement.names))

        def visit_FunctionDef(self, statement):
            return None

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, statement):
            return None

        def visit_ClassDef(self, statement):
            return None

    return_visitor = _CurrentFunctionReturnVisitor()
    for body_statement in function_body:
        return_visitor.visit(body_statement)
    ordered_return_values = tuple(
        (graph.G.graph.get("return_value_nodes", {}) or {}).values()
    )
    checkpoint_return_value_nodes = {
        id(statement): int(value_node)
        for statement, value_node in zip(
            source_returns, ordered_return_values
        )
        if int(value_node) in graph.G
    }
    for _control_id, data in graph.G.nodes(data=True):
        expression = data.get("expr_obj")
        if isinstance(expression, ast.If):
            controlled_statements = (
                *expression.body,
                *expression.orelse,
            )
        elif isinstance(expression, ast.IfExp):
            controlled_statements = (
                expression.body,
                expression.orelse,
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
        ).get("binding_kind") in {"loop", "exception"}
    )
    for loop_binding in tuple(controlled_nodes):
        if (
            loop_binding in graph.G
            and graph.G.nodes[loop_binding].get("type") == "Input"
            and (
                graph.G.nodes[loop_binding].get("attributes") or {}
            ).get("binding_kind") == "loop"
        ):
            controlled_nodes.update(
                nx.descendants(graph.G, loop_binding)
            )
    for boolean_node, boolean_data in graph.G.nodes(data=True):
        if isinstance(boolean_data.get("expr_obj"), ast.BoolOp):
            controlled_nodes.add(boolean_node)
            controlled_nodes.update(
                nx.ancestors(graph.G, boolean_node)
            )
    loop_plans_by_node = {
        plan.loop.node_id: plan for plan in shell.loop_plans
    }
    nested_body_nodes_by_loop = {
        int(owner.loop.node_id): frozenset(
            int(body_node)
            for nested in shell.loop_plans
            if (
                int(nested.loop.node_id) != int(owner.loop.node_id)
                and int(nested.loop.node_id) in owner.loop.body_nodes
            )
            for body_node in nested.loop.body_nodes
        )
        for owner in shell.loop_plans
    }
    comprehension_owner_by_binding: dict[int, int] = {}
    comprehension_owner_by_generator: dict[int, int] = {}
    loop_owner_by_binding = {
        int(binding): int(plan.loop.node_id)
        for plan in shell.loop_plans
        for _name, binding in plan.loop.target_bindings
    }
    for plan in shell.loop_plans:
        loop_expression = graph.G.nodes[
            plan.loop.node_id
        ].get("expr_obj")
        if not isinstance(loop_expression, ast.comprehension):
            continue
        owner = next(
            (
                successor
                for successor in graph.G.successors(plan.loop.node_id)
                if isinstance(
                    graph.G.nodes[successor].get("expr_obj"),
                    (
                        ast.ListComp,
                        ast.SetComp,
                        ast.DictComp,
                        ast.GeneratorExp,
                    ),
                )
            ),
            None,
        )
        if owner is not None:
            comprehension_owner_by_generator[plan.loop.node_id] = owner
            for _name, binding in plan.loop.target_bindings:
                comprehension_owner_by_binding[binding] = owner
    loop_invalidated_nodes: dict[int, set[int]] = {}
    for plan in shell.loop_plans:
        controlled_nodes.add(plan.loop.node_id)
        # Lexical normalization can fold or remove a body node after the loop
        # plan records its source identity.  The surviving graph is the
        # executable authority; stale source ids must not be handed to
        # NetworkX as traversal roots.
        live_body_nodes = {
            int(body_node)
            for body_node in plan.loop.body_nodes
            if int(body_node) in graph.G
        }
        controlled_nodes.update(live_body_nodes)
        invalidated = set(live_body_nodes)
        for body_node in live_body_nodes:
            invalidated.update(nx.descendants(graph.G, body_node))
        # Retained-loop backedges make the graph cyclic.  A raw descendants
        # walk can therefore wrap through the loop and reach invariant
        # producers that dominate the control node.  Clearing those values
        # replays effectful work performed before the loop (for example the
        # dt system's completed superstep while iterating its attempt log).
        # Body identities remain iteration-owned; other ancestors of the
        # loop control are already-computed invariants.
        invariant_ancestors = (
            set(nx.ancestors(graph.G, plan.loop.node_id))
            - live_body_nodes
        )
        invalidated.difference_update(invariant_ancestors)
        loop_invalidated_nodes[plan.loop.node_id] = invalidated
        # The planner loop owns the complete work cone fed by its body,
        # including numerical regions that consume the induction binding.
        # Leaving descendants in the ordinary topological sweep evaluates
        # them before the loop has supplied that binding.
        controlled_nodes.update(invalidated)
        controlled_nodes.update(
            binding for _name, binding in plan.loop.target_bindings
        )
        for successor in graph.G.successors(plan.loop.node_id):
            if isinstance(
                graph.G.nodes[successor].get("expr_obj"),
                (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
            ):
                controlled_nodes.update(nx.ancestors(graph.G, successor))
    loop_runtime_region_indices = {
        loop_node: tuple(
            region_index
            for region_index, (_compiler, subgraph, _ephemeral)
            in enumerate(regions)
            if set(
                subgraph.G.graph.get("deployment_nodes", ())
            ).intersection(invalidated)
            or set(
                subgraph.G.graph.get("deployment_outputs", ())
            ).intersection(invalidated)
        )
        for loop_node, invalidated in loop_invalidated_nodes.items()
    }
    downstream_loop_nodes = {
        candidate
        for candidate in loop_plans_by_node
        if any(
            owner != candidate
            and owner in nx.ancestors(graph.G, candidate)
            for owner in loop_plans_by_node
        )
    }
    has_comprehension_owner = any(
        isinstance(
            graph.G.nodes[loop_node].get("expr_obj"),
            ast.comprehension,
        )
        for loop_node in loop_plans_by_node
    )
    completed_regions: set[int] = set()
    active_nodes: set[int] = set()
    active_exceptions: dict[int, BaseException] = {}
    # One source function owns one discovery tape, and only the explicit
    # compilation/discovery invocation may create it.  Planned regions retain
    # separate output/feed cut points into that tape, but loops, frames and
    # repeated calls never create new tapes or capture observed values as new
    # constants.  Compilation consumes these cut points and then destroys the
    # tape before any installed shell can execute.
    #
    # In particular, do not "helpfully" create a tape for capture=False.
    # That mode is an ordinary coordinator execution and cannot rediscover,
    # extend, or rebuild the program.  Runtime must enter the installed shell;
    # retaining this non-capture path for diagnostics does not grant it any
    # compilation authority.
    discovery_tape = None
    if capture:
        if discovery_session is not None:
            # Nested source shells are compartments in the one source program,
            # not independently observed programs.  They record into the
            # owner's already-existing tape and may only contribute planner
            # cut points.  Creating or lowering another tape here would turn
            # function boundaries into executions and make the captured
            # example, rather than the planner, determine program count.
            discovery_tape = discovery_session["tape"]
            shell._discovery_tape = discovery_tape
            shell._discovery_tape_owner = False
        elif shell._discovery_tape is not None:
            discovery_tape = shell._discovery_tape
            discovery_session = shell._discovery_session
        else:
            if shell._discovery_tape_complete:
                raise RuntimeError(
                    "compiled source program attempted to create another "
                    "discovery tape"
                )
            discovery_tape = GradTape()
            shell._discovery_tape = discovery_tape
            shell._discovery_tape_owner = True
            shell._discovery_tape_creations += 1
            if shell._discovery_tape_creations != 1:
                raise RuntimeError(
                    "one source program created more than one discovery tape"
                )
            discovery_session = {
                "owner": shell,
                "tape": discovery_tape,
                "bindings": {},
                "lowered_program": None,
            }
            shell._discovery_session = discovery_session
    tape_bindings: dict[int, Any] = (
        discovery_session["bindings"]
        if capture and discovery_session is not None
        else {}
    )
    shell._discovery_tape_bindings = tape_bindings
    feed_maps = list(shell.forward_feed_ids)
    aggregate_feed_maps = list(shell.forward_aggregate_feed_paths)
    captured_subgraphs = list(shell.forward_subgraphs)
    captured_compilers = list(shell.forward_compilers)
    captured_region_indices = list(shell.forward_region_indices)
    captured_output_values = list(shell.forward_output_values)
    captured_region_node_ids = list(
        shell.forward_region_capture_node_ids
    )
    captured_region_planned_ids = list(
        shell.forward_region_planned_capture_ids
    )
    captured_region_planned_input_ids = list(
        shell.forward_region_planned_input_ids
    )
    captured_region_slots: dict[int, int] = {
        int(
            region_index
            if not isinstance(region_index, tuple)
            else region_index[-2]
        ): slot
        for slot, region_index in enumerate(captured_region_indices)
    }
    region_occurrences: dict[int, int] = {}

    def is_shader_internal_node(node_id: int) -> bool:
        region_index = region_for_node.get(node_id)
        if region_index is None:
            return False
        subgraph = regions[region_index][1]
        return (
            node_id in subgraph.G.graph.get("deployment_nodes", ())
            and node_id
            not in subgraph.G.graph.get("deployment_outputs", ())
        )

    def evaluate_region(region_index: int) -> None:
        if region_index in completed_regions:
            return
        occurrence = region_occurrences.get(region_index, 0)
        region_occurrences[region_index] = occurrence + 1
        region_key = (invocation_slot, region_index, occurrence)
        compiler, subgraph, ephemeral = regions[region_index]
        inputs: dict[str, Any] = {}
        for input_id in subgraph.G.graph["deployment_inputs"]:
            input_value = evaluate_node(input_id)
            tensorized_input = _tensorize_graph_input(
                input_value,
                device=device,
            )
            if tensorized_input is not input_value:
                # Object-field projections become numerical region feeds only
                # after structural attribute evaluation.  Public inputs were
                # tensorized earlier, but these late projections must join the
                # same resident/tape path or ndarray methods execute invisibly
                # in Python during discovery.
                values[input_id] = tensorized_input
                input_value = tensorized_input
            source_binding = _graph_source_binding_name(graph, input_id)
            if source_binding is not None:
                shell._capture_input_names[id(input_value)] = source_binding
                storage = _capture_storage_identity(input_value)
                if storage is not None:
                    shell._capture_input_storage[storage] = source_binding
            name = _compiler_input_name(
                subgraph.G.nodes[input_id]["label"]
            )
            inputs[name] = input_value

        deployment_nodes = tuple(
            subgraph.G.graph.get("deployment_nodes", ())
        )
        has_structural_store = any(
            str(
                graph.G.nodes[candidate].get("op")
                or graph.G.nodes[candidate].get("type")
            ) == "IndexedStore"
            and any(
                str(role) == "base"
                and isinstance(values.get(int(parent)), (dict, list, set))
                for parent, role in (
                    graph.G.nodes[candidate].get("parents") or ()
                )
            )
            for candidate in deployment_nodes
            if candidate in graph.G
        )
        has_structural_container_call = any(
            isinstance(expression := graph.G.nodes[candidate].get("expr_obj"), ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and (
                any(
                    str(role) in {"operand", "receiver"}
                    and isinstance(values.get(int(parent)), (dict, list, set))
                    for parent, role in (
                        graph.G.nodes[candidate].get("parents") or ()
                    )
                )
                or (
                    isinstance(expression.func.value, ast.Name)
                    and isinstance(
                        source_binding_values.get(expression.func.value.id),
                        (dict, list, set),
                    )
                )
            )
            for candidate in deployment_nodes
            if candidate in graph.G
        )
        has_structural_indexed_read = any(
            str(
                graph.G.nodes[candidate].get("op")
                or graph.G.nodes[candidate].get("type")
            ) == "Indexed"
            and any(
                str(role) == "base"
                and isinstance(
                    values.get(int(parent)), (dict, list, tuple, set)
                )
                for parent, role in (
                    graph.G.nodes[candidate].get("parents") or ()
                )
            )
            for candidate in deployment_nodes
            if candidate in graph.G
        )
        if (
            has_structural_store
            or has_structural_container_call
            or has_structural_indexed_read
        ):
            shell.coordinator_region_indices.add(region_index)
            coordinator_override_nodes.update(deployment_nodes)
            active_nodes.difference_update(map(int, deployment_nodes))
            for deployment_node in deployment_nodes:
                evaluate_node(int(deployment_node))
            completed_regions.add(region_index)
            return

        operations = [
            str(subgraph.G.nodes[node_id].get("op") or
                subgraph.G.nodes[node_id].get("type"))
            for node_id in subgraph.G.graph.get("deployment_nodes", ())
        ]
        label = f"region {region_index}: " + " -> ".join(operations)
        if shell._profiler.verbose:
            shell._profiler.trace(
                path=shell.profile_path,
                section="region-input",
                label=label,
                fields={
                    name: _diagnostic_value_summary(value)
                    for name, value in inputs.items()
                },
            )

        def run_region() -> None:
            captured_program = (
                shell.captured_region_programs.get(region_index)
                or shell.captured_region_programs.get(region_key)
            )
            if captured_program is not None:
                runtime_region_feeds = {
                    input_id: evaluate_node(input_id)
                    for input_id in subgraph.G.graph["deployment_inputs"]
                }
                missing_feed_ids = [
                    feed_id
                    for feed_id in ordered_feed_ids(captured_program.program)
                    if feed_id not in runtime_region_feeds
                ]
                unique_aliases = _unique_runtime_feed_aliases(
                    captured_program.program,
                    missing_feed_ids,
                    runtime_region_feeds,
                )
                runtime_region_feeds.update(unique_aliases)
                missing_feed_ids = [
                    feed_id
                    for feed_id in missing_feed_ids
                    if feed_id not in unique_aliases
                ]
                if shell._profiler.verbose:
                    shell._profiler.trace(
                        path=shell.profile_path,
                        section="feed-resolution",
                        label=label,
                        fields={
                            "program": tuple(
                                ordered_feed_ids(captured_program.program)
                            ),
                            "captured": tuple(
                                sorted(captured_program.feeds)
                            ),
                            "runtime": tuple(
                                sorted(runtime_region_feeds)
                            ),
                            "unresolved": tuple(missing_feed_ids),
                        },
                )
                if missing_feed_ids:
                    tensor_leaves = []
                    exactly_routed_tensors = {
                        id(value)
                        for input_id, value in runtime_region_feeds.items()
                        if (
                            input_id in captured_program.program.feeds
                            and isinstance(value, AbstractTensor)
                        )
                    }

                    def collect_tensor_leaves(value):
                        if isinstance(value, AbstractTensor):
                            if id(value) not in exactly_routed_tensors:
                                tensor_leaves.append(value)
                        elif isinstance(value, (tuple, list)):
                            for item in value:
                                collect_tensor_leaves(item)
                        elif isinstance(value, dict):
                            for item in value.values():
                                collect_tensor_leaves(item)

                    for input_id, value in runtime_region_feeds.items():
                        if input_id not in captured_program.program.feeds:
                            collect_tensor_leaves(value)
                    if len(tensor_leaves) != len(missing_feed_ids):
                        runtime_tensor_meta = []

                        def collect_runtime_meta(input_id, value):
                            if isinstance(value, AbstractTensor):
                                runtime_tensor_meta.append((
                                    input_id,
                                    tuple(value.shape),
                                    str(value.dtype),
                                ))
                            elif isinstance(value, (tuple, list)):
                                for item in value:
                                    collect_runtime_meta(input_id, item)
                            elif isinstance(value, dict):
                                for item in value.values():
                                    collect_runtime_meta(input_id, item)

                        for input_id, value in runtime_region_feeds.items():
                            collect_runtime_meta(input_id, value)
                        raise RuntimeError(
                            f"compiled region {region_index} ({tuple(operations)!r}) "
                            "cannot project structured tensor feeds exactly; "
                            f"missing_feeds={len(missing_feed_ids)} "
                            f"tensor_leaves={len(tensor_leaves)}; "
                            f"program_feed_ids={tuple(ordered_feed_ids(captured_program.program))!r}; "
                            f"runtime_feed_ids={tuple(sorted(runtime_region_feeds))!r}; "
                            f"unresolved_feed_ids={tuple(missing_feed_ids)!r}; "
                            "unresolved_meta="
                            f"{tuple((feed_id, captured_program.program.meta.get(feed_id)) for feed_id in missing_feed_ids)!r}; "
                            "unresolved_origins="
                            f"{tuple((feed_id, (captured_program.program.extras or {}).get('capture_feed_origins', {}).get(feed_id)) for feed_id in missing_feed_ids)!r}; "
                            f"runtime_tensor_meta={tuple(runtime_tensor_meta)!r}"
                        )
                    runtime_region_feeds.update(
                        zip(missing_feed_ids, tensor_leaves)
                    )
                try:
                    chunks = execute_captured_fused_program(
                        captured_program,
                        runtime_region_feeds,
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise RuntimeError(
                        f"compiled shell {_shell_profile_name(shell)!r} "
                        f"id={id(shell)} region {region_index} "
                        f"({tuple(operations)!r}) "
                        "failed strict resident routing; "
                        f"cause={str(error)!r}; "
                        f"program_feeds={tuple(sorted(captured_program.program.feeds))!r}; "
                        f"runtime_feeds={tuple(sorted(runtime_region_feeds))!r}; "
                        f"step_inputs={tuple(tuple(step.input_ids) for step in captured_program.program.steps)!r}; "
                        f"stages={len(captured_program.stages)}"
                    ) from error
                for output_id in subgraph.G.graph["deployment_outputs"]:
                    output_name = f"value_{output_id}"
                    if output_name not in chunks:
                        def structural_value(structural_id):
                            if structural_id in values:
                                return values[structural_id]
                            data = subgraph.G.nodes[structural_id]
                            expression = data.get("expr_obj")
                            structural_parents = data.get("parents", ())
                            if isinstance(expression, ast.Constant):
                                result = expression.value
                            elif isinstance(expression, ast.Attribute):
                                parent = structural_value(
                                    structural_parents[0][0]
                                )
                                result = getattr(parent, expression.attr)
                            elif isinstance(expression, ast.Subscript):
                                base_id = next(
                                    parent
                                    for parent, role in structural_parents
                                    if str(role) == "base"
                                )
                                base = structural_value(base_id)
                                if isinstance(base, AbstractTensor):
                                    raise RuntimeError(
                                        "tensor indexing output was omitted "
                                        "from its compiled region"
                                    )
                                indices = [
                                    structural_value(parent)
                                    for parent, role in structural_parents
                                    if str(role) == "index"
                                ]
                                index = (
                                    indices[0]
                                    if len(indices) == 1
                                    else tuple(indices)
                                )
                                if isinstance(
                                    base, (dict, list, tuple, set)
                                ):
                                    index = coordinator_index(index)
                                result = base[index]
                            elif isinstance(expression, ast.BinOp):
                                operands = [
                                    structural_value(parent)
                                    for parent, _role in structural_parents
                                ]
                                if any(
                                    isinstance(value, AbstractTensor)
                                    for value in operands
                                ):
                                    raise RuntimeError(
                                        "tensor arithmetic output was omitted "
                                        "from its compiled region"
                                    )
                                binary = {
                                    ast.Add: operator.add,
                                    ast.Sub: operator.sub,
                                    ast.Mult: operator.mul,
                                    ast.Div: operator.truediv,
                                    ast.FloorDiv: operator.floordiv,
                                    ast.Mod: operator.mod,
                                }.get(type(expression.op))
                                if binary is None:
                                    raise KeyError(output_name)
                                result = binary(*operands)
                            else:
                                result = evaluate_node(structural_id)
                            values[structural_id] = result
                            return result

                        values[output_id] = structural_value(output_id)
                        continue
                    output = GLSLTensorOperations()
                    output.data = chunks[output_name]
                    values[output_id] = output
                completed_regions.add(region_index)
                return

            function = (
                shell.compiled_dispatch_functions[region_index]
                if shell.compiled_dispatch_functions
                else ephemeral
            )
            record_region = capture and region_index not in captured_region_slots
            if record_region:
                # The ProcessGraph planner already owns repetition,
                # cardinality, carried values and back-pressure.  Discovery
                # records exactly one primitive implementation exemplar for
                # this planned region.  Recording later loop occurrences
                # would unroll one observed execution into the program,
                # inflate the tape, and make runtime values decide program
                # structure.  Later occurrences may execute only to complete
                # this one discovery invocation; they cannot add tape nodes.
                capture_context = autograd.forward_capture(discovery_tape)
            else:
                capture_context = autograd.forward_observation()
            region_nodes_before = (
                set(discovery_tape._nodes)
                if capture and discovery_tape is not None
                else set()
            )
            with capture_context as tape:
                bound_inputs = (
                    {
                        name: _bind_capture_tape(value, tape, tape_bindings)
                        for name, value in inputs.items()
                    }
                    if capture
                    else inputs
                )
                planned_capture = {
                    "tape": discovery_tape,
                    "graph": graph,
                    "shell": shell,
                    "node_capture_ids": {},
                    "step_input_ids": {},
                    "collection_materializations": (
                        shell.forward_planned_collection_materializations
                    ),
                    "value_aliases": (
                        shell.compiled_process_graph_aliases
                    ),
                    "collection_owner_ids": frozenset(
                        int(collection_id)
                        for reduction in shell.loop_shader_reductions
                        if reduction.control_program is not None
                        for (
                            _source_id,
                            collection_id,
                            _induction,
                            _start,
                        ) in reduction.control_program.collection_bindings
                    ),
                }
                observer_token = (
                    _planned_capture_context.set(planned_capture)
                    if record_region
                    else None
                )
                try:
                    results = function(**bound_inputs)
                except Exception as error:
                    raise RuntimeError(
                        "ProcessGraph numerical region failed in "
                        f"{graph.G.graph.get('function_name', '?')} "
                        f"region {region_index}; "
                        f"nodes={subgraph.G.graph.get('deployment_nodes', ())!r}; "
                        f"inputs={subgraph.G.graph.get('deployment_inputs', ())!r}; "
                        f"outputs={subgraph.G.graph.get('deployment_outputs', ())!r}; "
                        f"bound={tuple((name, _diagnostic_value_summary(value)) for name, value in bound_inputs.items())!r}"
                    ) from error
                finally:
                    if observer_token is not None:
                        _planned_capture_context.reset(observer_token)
            if not isinstance(results, tuple):
                results = (results,)
            returned = dict(zip(compiler._outs, results))
            for output_id, store_id in zip(
                subgraph.G.graph["deployment_outputs"],
                subgraph.G.graph["deployment_store_nodes"],
            ):
                values[output_id] = returned[store_id]

            completed_regions.add(region_index)
            tensor_outputs_captured = any(
                isinstance(values.get(output_id), AbstractTensor)
                for output_id in subgraph.G.graph["deployment_outputs"]
            )
            if shell._profiler.verbose:
                shell._profiler.trace(
                    path=shell.profile_path,
                    section="capture-state",
                    label=label,
                    fields={
                        "record_region": record_region,
                        "tape_nodes": (
                            len(tape._nodes) if tape is not None else 0
                        ),
                        "tensor_outputs": tensor_outputs_captured,
                        "output_tape_state": tuple(
                            (
                                int(output_id),
                                id(values.get(output_id)),
                                (
                                    id(values.get(output_id))
                                    in discovery_tape._nodes
                                    if isinstance(
                                        values.get(output_id),
                                        AbstractTensor,
                                    )
                                    else None
                                ),
                                (
                                    getattr(
                                        values.get(output_id),
                                        "_tape",
                                        None,
                                    )
                                    is discovery_tape
                                    if isinstance(
                                        values.get(output_id),
                                        AbstractTensor,
                                    )
                                    else None
                                ),
                            )
                            for output_id in subgraph.G.graph[
                                "deployment_outputs"
                            ]
                        ),
                        "shell_id": id(shell),
                    },
                )
            occurrence_node_ids = tuple(
                int(node_id)
                for node_id in discovery_tape._nodes
                if int(node_id) not in region_nodes_before
            ) if record_region else ()
            # Repeated executions of one planned region are implementation
            # observations of the same ProcessGraph program.  Keep the first
            # occurrence as the primitive lowering exemplar; the planner's
            # LoopBlock owns repetition and cardinality.  Appending later tape
            # occurrences here would unroll one discovery execution into the
            # compiled program and create duplicate SSA writers.
            if (
                record_region
                and discovery_tape._nodes
                and occurrence_node_ids
            ):
                # Region ownership comes from the AST/ProcessGraph schedule,
                # not from the Python shape of its return value.  A region
                # may return a tuple or an ordinary object whose tensor fields
                # are consumed later.  Requiring the immediate graph output
                # itself to be an AbstractTensor silently drops every SSA
                # definition hidden behind that OOP boundary and forces a
                # later consumer to rediscover the producer by walking the
                # tape backward.  Record the region's owned forward nodes now;
                # lowering below computes only the live cross-region values.
                captured_region_slots[region_index] = len(
                    captured_region_indices
                )
                captured_region_node_ids.append(occurrence_node_ids)
                captured_region_planned_ids.append(tuple(
                    (
                        int(graph_node_id),
                        tuple(map(int, capture_ids)),
                    )
                    for graph_node_id, capture_ids
                    in planned_capture["node_capture_ids"].items()
                ))
                captured_region_planned_input_ids.append(tuple(
                    sorted(
                        (
                            int(result_capture_id),
                            tuple(
                                (
                                    int(primitive_id),
                                    int(graph_id),
                                )
                                for primitive_id, graph_id
                                in positional_inputs
                            ),
                        )
                        for result_capture_id, positional_inputs
                        in planned_capture["step_input_ids"].items()
                    )
                ))
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
                aggregate_paths = {}
                rebound_originals = {}
                for original_id, rebound in tape_bindings.items():
                    rebound_originals.setdefault(
                        id(rebound), []
                    ).append(int(original_id))

                def record_aggregate_paths(
                    graph_input_id: int,
                    value: Any,
                    path: tuple[object, ...] = (),
                ) -> None:
                    if isinstance(value, AbstractTensor):
                        if path:
                            leaf = (
                                int(graph_input_id),
                                tuple(path),
                                _capture_storage_identity(value),
                            )
                            aggregate_paths[id(value)] = leaf
                            rebound_aliases = {id(value)}
                            pending_aliases = [id(value)]
                            while pending_aliases:
                                rebound_id = pending_aliases.pop()
                                for original_id in (
                                    rebound_originals.get(
                                        rebound_id, ()
                                    )
                                ):
                                    if original_id in rebound_aliases:
                                        continue
                                    rebound_aliases.add(original_id)
                                    pending_aliases.append(original_id)
                            aggregate_paths.update({
                                int(alias_id): leaf
                                for alias_id in rebound_aliases
                            })
                            # Some primitive hooks retain the tensor's resident
                            # storage wrapper as their operand.  It is a direct
                            # alias of this aggregate leaf, not a new value.
                            # Record that identity now so lowering can erase
                            # the Python wrapper without fabricating a shell
                            # input later.
                            storage_value = getattr(value, "data", None)
                            if storage_value is not None:
                                aggregate_paths[id(storage_value)] = leaf
                        return
                    if isinstance(value, (tuple, list)):
                        for index, item in enumerate(value):
                            record_aggregate_paths(
                                graph_input_id,
                                item,
                                (*path, int(index)),
                            )
                    elif isinstance(value, Mapping):
                        for name, item in value.items():
                            record_aggregate_paths(
                                graph_input_id,
                                item,
                                (*path, name),
                            )
                    elif isinstance(getattr(value, "__dict__", None), dict):
                        for name, item in vars(value).items():
                            record_aggregate_paths(
                                graph_input_id,
                                item,
                                (*path, str(name)),
                            )

                for input_id in subgraph.G.graph["deployment_inputs"]:
                    name = _compiler_input_name(
                        subgraph.G.nodes[input_id]["label"]
                    )
                    record_aggregate_paths(
                        int(input_id), bound_inputs[name]
                    )
                aggregate_by_storage = {
                    storage: (graph_input_id, path, storage)
                    for graph_input_id, path, storage
                    in aggregate_paths.values()
                    if storage is not None
                }
                for value in values.values():
                    if not isinstance(value, AbstractTensor):
                        continue
                    storage = _capture_storage_identity(value)
                    leaf = aggregate_by_storage.get(storage)
                    if leaf is None:
                        continue
                    aggregate_paths[id(value)] = leaf
                    storage_value = getattr(value, "data", None)
                    if storage_value is not None:
                        aggregate_paths[id(storage_value)] = leaf
                aggregate_feed_maps.append(aggregate_paths)
                captured_subgraphs.append(subgraph)
                captured_compilers.append(compiler)
                captured_region_indices.append(region_index)
                captured_output_values.append({
                    f"result_{index}": values[output_id]
                    for index, output_id in enumerate(
                        subgraph.G.graph["deployment_outputs"]
                    )
                    if isinstance(values.get(output_id), AbstractTensor)
                })

        with _profile_event(
            shell,
            "capture" if capture else "dispatch",
            label,
            gpu=True,
        ):
            run_region()
        if shell._profiler.verbose:
            shell._profiler.trace(
                path=shell.profile_path,
                section="region-output",
                label=label,
                fields={
                    f"value_{output_id}": _diagnostic_value_summary(
                        values.get(output_id)
                    )
                    for output_id in subgraph.G.graph[
                        "deployment_outputs"
                    ]
                },
            )

    class _LoopBreakSignal(Exception):
        pass

    class _LoopContinueSignal(Exception):
        pass

    def evaluate_static_expression(expression: ast.AST) -> Any:
        if isinstance(expression, ast.Constant):
            return expression.value
        if isinstance(expression, ast.Name):
            if expression.id in shell.static_python_bindings:
                return shell.static_python_bindings[expression.id]
            defaults = graph.G.graph.get("parameter_defaults") or {}
            if expression.id in defaults:
                return defaults[expression.id]
            if expression.id in supplied:
                value = supplied[expression.id]
                if value is None or isinstance(
                    value,
                    (list, tuple, dict, set),
                ):
                    return value
        if isinstance(expression, ast.Attribute):
            return getattr(
                evaluate_static_expression(expression.value),
                expression.attr,
            )
        if (
            isinstance(expression, ast.Call)
            and not expression.args
            and not expression.keywords
        ):
            function = evaluate_static_expression(expression.func)
            if callable(function):
                return function()
        if (
            isinstance(expression, ast.Compare)
            and len(expression.ops) == 1
            and len(expression.comparators) == 1
            and isinstance(expression.ops[0], (ast.Is, ast.IsNot))
        ):
            left = evaluate_static_expression(expression.left)
            right = evaluate_static_expression(expression.comparators[0])
            result = left is right
            return not result if isinstance(expression.ops[0], ast.IsNot) else result
        raise KeyError(ast.unparse(expression))

    def evaluate_source_container_item(
        source: ast.AST,
        fallback_node: int | None = None,
    ) -> Any:
        """Evaluate a container element at its exact lexical source point."""

        if isinstance(source, ast.Name):
            if source.id in source_binding_values:
                return source_binding_values[source.id]
            if source.id in supplied:
                return supplied[source.id]
        if isinstance(source, ast.Dict):
            result = {}
            for key, value in zip(source.keys, source.values):
                item = evaluate_source_container_item(
                    value, expression_nodes.get(id(value))
                )
                if key is None:
                    result.update(dict(item))
                else:
                    result[evaluate_source_container_item(
                        key, expression_nodes.get(id(key))
                    )] = item
            return result
        if isinstance(source, (ast.Tuple, ast.List, ast.Set)):
            items = [
                evaluate_source_container_item(
                    item, expression_nodes.get(id(item))
                )
                for item in source.elts
            ]
            if isinstance(source, ast.Tuple):
                return tuple(items)
            if isinstance(source, ast.Set):
                return set(items)
            return items
        if isinstance(source, (ast.IfExp, ast.BoolOp)) or (
            isinstance(source, ast.Compare)
            and len(source.ops) == 1
            and isinstance(source.ops[0], (ast.Is, ast.IsNot))
        ):
            return evaluate_reduced_control_expression(source)
        node = (
            fallback_node
            if fallback_node is not None
            else expression_nodes.get(id(source))
        )
        if node is not None:
            return evaluate_node(int(node))
        return evaluate_reduced_control_expression(source)

    def coordinator_index(value: Any) -> Any:
        """Project tensor scalars into Python container index semantics."""

        if isinstance(value, AbstractTensor):
            if tuple(value.shape) == ():
                return value.item()
            return value
        if isinstance(value, tuple):
            return tuple(coordinator_index(item) for item in value)
        if isinstance(value, slice):
            return slice(
                coordinator_index(value.start),
                coordinator_index(value.stop),
                coordinator_index(value.step),
            )
        return value

    def _resolve_reference_node(expression: ast.expr) -> int | None:
        """Find the live graph node a reference-typed expression resolves to.

        A bound-method call's receiver (``machine``, ``machine.runner``, ...)
        is a reference value, not a tensor -- it has no numeric identity to
        fall back on. The existing, narrower check this replaces
        (``id(expression) in graph.G``) asks only "does this exact AST node
        still happen to be a node in the graph", which silently fails
        whenever an earlier reduction pass legitimately rebuilt or pruned
        that specific node while the value it represents is still perfectly
        live under its name. ``identity_table`` already tracks every SSA
        version of every name for exactly this reason (it is the same
        mechanism ``live_function_module_binding`` above uses); grounding
        receiver resolution in it instead of a raw ``id()`` presence check
        means a receiver resolves by what it *is*, not by whether one
        specific AST node object happened to survive unchanged.
        """

        if isinstance(expression, ast.Name):
            candidates = graph.G.graph.get("identity_table", {}).get(
                expression.id, ()
            )
            for candidate in reversed(candidates):
                if int(candidate) in graph.G:
                    return int(candidate)
            return None
        if isinstance(expression, ast.Attribute):
            if id(expression) in graph.G:
                return id(expression)
            base_node = _resolve_reference_node(expression.value)
            if base_node is None:
                return None
            for successor in graph.G.successors(base_node):
                successor_expr = graph.G.nodes[successor].get("expr_obj")
                if (
                    isinstance(successor_expr, ast.Attribute)
                    and successor_expr.attr == expression.attr
                ):
                    return successor
            return None
        return None

    def evaluate_node(node_id: int) -> Any:
        if node_id in values:
            cached_data = graph.G.nodes[node_id]
            if str(cached_data.get("type")) in {"Input", "input"}:
                if node_id in generator_binding_values:
                    values[node_id] = generator_binding_values[node_id]
                    return values[node_id]
                cached_attributes = cached_data.get("attributes") or {}
                cached_name = str(
                    cached_attributes.get(
                        "binding_name",
                        cached_data.get("label", node_id),
                    )
                )
                if cached_name in source_binding_values:
                    values[node_id] = source_binding_values[cached_name]
                    return values[node_id]
            return values[node_id]
        if node_id in inert_nodes:
            # A name or attribute lookup nobody reads.  Producing it would
            # demand a materialized receiver and force an otherwise
            # shader-local intermediate to become a published region output.
            return None
        if node_id in active_nodes:
            raise RuntimeError(
                f"recursive runtime dependency at ProcessGraph node {node_id}; "
                f"shell={_shell_profile_name(shell)!r}; "
                f"active_nodes={tuple(sorted(active_nodes))!r}; "
                f"region={region_for_node.get(node_id)!r}"
            )
        active_nodes.add(node_id)
        try:
            data = graph.G.nodes[node_id]
            node_type = str(data.get("type"))
            parents = tuple(data.get("parents") or ())
            if node_type == "Indexed":
                base_parent = next(
                    (
                        parent
                        for parent, role in parents
                        if str(role) == "base"
                    ),
                    None,
                )
                available_base = values.get(base_parent)
                available_indices = tuple(
                    values[parent]
                    for parent, role in parents
                    if str(role) == "index" and parent in values
                )
                if isinstance(
                    available_base,
                    (dict, list, tuple, set),
                ) or any(
                    isinstance(index, (str, bytes))
                    for index in available_indices
                ):
                    # Runtime loop bindings can carry structural keys that
                    # were not literals during planning. Container routing is
                    # a type fact, not value specialization, and must be
                    # decided before a numerical region claims the node.
                    coordinator_override_nodes.add(node_id)
            region_index = (
                None
                if node_id in coordinator_override_nodes
                else region_for_node.get(node_id)
            )
            if region_index is not None:
                evaluate_region(region_index)
                if node_id not in values:
                    subgraph = regions[region_index][1]
                    if is_shader_internal_node(node_id):
                        raise RuntimeError(
                            "shader-local ProcessGraph node was requested as "
                            "a compartment boundary value; "
                            f"node={node_id}; region={region_index}; "
                            f"shell={_shell_profile_name(shell)!r}. "
                            "The region extractor must export every value "
                            "observed by structural execution."
                        )
                    completed_regions.discard(region_index)
                    evaluate_region(region_index)
                if node_id not in values:
                    raise RuntimeError(
                        f"ProcessGraph region {region_index} did not publish "
                        f"requested boundary output node {node_id}; "
                        f"shell={_shell_profile_name(shell)!r}; "
                        f"deployment_outputs="
                        f"{subgraph.G.graph.get('deployment_outputs', ())!r}"
                    )
                return values[node_id]

            expression = data.get("expr_obj")
            attributes = data.get("attributes") or {}

            if node_type in {"Input", "input"}:
                if node_id in generator_binding_values:
                    result = generator_binding_values[node_id]
                    values[node_id] = result
                    return result
                # Retained stores and attribute writes may still point at an
                # Input-shaped identity for a local whose value is selected
                # by source control.  Once that statement has executed, the
                # active source frame is the authority for the spelling.
                name = str(
                    attributes.get(
                        "binding_name",
                        data.get("label", node_id),
                    )
                )
                if name in source_binding_values:
                    result = source_binding_values[name]
                    values[node_id] = result
                    return result
                comprehension_owner = (
                    comprehension_owner_by_binding.get(node_id)
                )
                if comprehension_owner is not None:
                    evaluate_node(comprehension_owner)
                    if node_id in generator_binding_values:
                        result = generator_binding_values[node_id]
                        values[node_id] = result
                        return result
                loop_owner = loop_owner_by_binding.get(node_id)
                if (
                    loop_owner is not None
                    and loop_owner not in active_nodes
                ):
                    # Normalized LoopExit/value dependencies can request an
                    # inner loop's iterable before the flat evaluator has
                    # visited the outer ``for`` statement.  A target is a
                    # definition produced by its owner loop, never a public
                    # function input.  Execute that owner first, then resolve
                    # the binding from the exact iteration/source frame.
                    evaluate_node(loop_owner)
                    if node_id in generator_binding_values:
                        result = generator_binding_values[node_id]
                        values[node_id] = result
                        return result
                    if node_id in values:
                        return values[node_id]
                    if name in source_binding_values:
                        result = source_binding_values[name]
                        values[node_id] = result
                        return result
                raise KeyError(
                    f"missing ProcessGraph input {name!r} in "
                    f"{graph.G.graph.get('function_name', '?')} at node "
                    f"{node_id}; loop targets="
                    f"{tuple(plan.loop.target_bindings for plan in shell.loop_plans)!r}; "
                    f"active generator bindings={tuple(generator_binding_values)!r}"
                )
            if node_type in {"Const", "const", "Constant"}:
                result = _constant_value(data)
            elif node_type in {"LoopExit", "LoopResult"}:
                by_role = {str(role): parent for parent, role in parents}
                control_id = int(by_role["control"])
                evaluate_node(control_id)
                if attributes.get("result_kind") == "carried":
                    binding_name = str(attributes.get("binding_name", ""))
                    carried_initial = next(
                        (
                            int(initial)
                            for name, initial, _updated in (
                                loop_plans_by_node[control_id]
                                .loop.carried_bindings
                            )
                            if str(name) == binding_name
                        ),
                        None,
                    )
                    if carried_initial is not None and carried_initial in values:
                        result = values[carried_initial]
                    elif (
                        carried_initial is not None
                        and carried_initial in graph.G
                    ):
                        result = evaluate_node(carried_initial)
                    elif binding_name in source_binding_values:
                        result = source_binding_values[binding_name]
                    else:
                        result = evaluate_node(by_role["value"])
                else:
                    result = evaluate_node(by_role["value"])
            elif node_type in {"LoopStateTransition", "LoopStatePort"}:
                by_role = {str(role): parent for parent, role in parents}
                # Retained-loop effects execute as members of the loop body.
                # A state port is only a publication boundary for the state
                # after that control has completed; replaying the effect here
                # both applies it twice and attempts to read loop-local target
                # bindings after their owning loop has gone out of scope.
                loop_id = (attributes or {}).get("loop_id")
                if loop_id is not None:
                    evaluate_node(int(loop_id))
                elif node_type == "LoopStateTransition":
                    evaluate_node(by_role["effect"])
                result = evaluate_node(by_role["state"])
            elif node_type == "LoopAggregateResult":
                result = tuple(
                    evaluate_node(parent)
                    for parent, role in parents
                    if str(role).startswith("arg")
                )
            elif isinstance(expression, (ast.Tuple, ast.List, ast.Set)):
                element_parents = [parent for parent, _role in parents]
                items = [
                    evaluate_source_container_item(
                        item,
                        element_parents[index]
                        if index < len(element_parents)
                        else None,
                    )
                    for index, item in enumerate(expression.elts)
                ]
                if isinstance(expression, ast.Tuple):
                    result = tuple(items)
                elif isinstance(expression, ast.Set):
                    result = set(items)
                else:
                    result = items
            elif isinstance(expression, ast.Dict):
                result = evaluate_source_container_item(expression, node_id)
            elif (
                isinstance(expression, ast.Attribute)
                and node_type != "SetAttr"
            ):
                parent = next(
                    (
                        parent
                        for parent, role in parents
                        if str(role) == "value"
                    ),
                    None,
                )
                if parent is None:
                    raise RuntimeError(
                        "attribute node has no receiver in "
                        f"{graph.G.graph.get('function_name', '?')} at "
                        f"node {node_id}: "
                        f"{ast.dump(expression, include_attributes=False)}; "
                        f"parents={parents!r}; attributes="
                        f"{(data.get('attributes') or {})!r}"
                    )
                receiver = evaluate_node(parent)
                if isinstance(receiver, _CompiledStructuralObject):
                    # A field read on a compiler-owned structural object is
                    # not a static Python fact to freeze into ``values``:
                    # the field is genuine per-call state (see
                    # ``_CompiledStructuralObject.state``), and a plain
                    # ``getattr`` here would capture whatever this one
                    # discovery trace happened to observe as if it were a
                    # constant -- the exact one-value trap that makes
                    # ``counter.value`` invisible to region capture. Keep
                    # the field's own identity live instead of collapsing
                    # it to a bare Python value.
                    result = receiver.state.get(expression.attr)
                else:
                    try:
                        result = getattr(receiver, expression.attr)
                    except AttributeError as exc:
                        raise AttributeError(
                            f"{graph.G.graph.get('function_name', '?')} "
                            f"attribute {expression.attr!r} at node "
                            f"{node_id} has receiver "
                            f"{type(receiver).__name__}; parent={parent}, "
                            f"parents={parents!r}"
                        ) from exc
                if node_type == "GetAttr":
                    shell.reference_operator_sequence.append(int(node_id))
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
            elif node_type == "IndexedStore":
                by_role: dict[str, list[int]] = {}
                for parent, role in parents:
                    by_role.setdefault(str(role), []).append(parent)
                base = evaluate_node(by_role["base"][0])
                if isinstance(expression, ast.Subscript):
                    index = evaluate_reduced_control_expression(
                        expression.slice
                    )
                else:
                    indices = [
                        evaluate_node(parent)
                        for parent in by_role.get("index", ())
                    ]
                    index = (
                        indices[0]
                        if len(indices) == 1
                        else tuple(indices)
                    )
                value = evaluate_node(by_role["value"][0])
                if isinstance(base, (dict, list, tuple, set)):
                    index = coordinator_index(index)
                base[index] = value
                result = base
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
                if isinstance(base, (dict, list, tuple, set)):
                    index = coordinator_index(index)
                result = base[index]
            elif isinstance(expression, ast.BoolOp):
                if isinstance(expression.op, ast.And):
                    result = True
                    for parent, role in parents:
                        result = evaluate_node(parent)
                        if shell._profiler.verbose:
                            shell._profiler.trace(
                                path=shell.profile_path,
                                section="boolop",
                                label=f"node {node_id} {role}",
                                fields={
                                    "value": _diagnostic_value_summary(
                                        result
                                    )
                                },
                            )
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
            elif (
                attributes.get("materialization_kind")
                in {"unrolled_loop", "retained_loop_aggregate"}
                and isinstance(
                    expression,
                    (
                        ast.ListComp,
                        ast.SetComp,
                        ast.DictComp,
                        ast.GeneratorExp,
                    ),
                )
            ):
                try:
                    materialized = [
                        evaluate_node(parent)
                        for parent, role in parents
                        if str(role).startswith("arg")
                    ]
                except KeyError:
                    # An evaporated structural comprehension can retain a
                    # cloned target-shaped Input after its loop binding was
                    # removed. Reconstruct from source semantics, whose
                    # comprehension frame publishes every destructured name.
                    result = evaluate_reduced_control_expression(expression)
                else:
                    if not materialized:
                        result = evaluate_reduced_control_expression(
                            expression
                        )
                    elif isinstance(expression, ast.SetComp):
                        result = set(materialized)
                    elif isinstance(expression, ast.DictComp):
                        result = dict(materialized)
                    elif isinstance(expression, ast.GeneratorExp):
                        # Keep a replayable finite sequence during discovery.
                        # Consumers such as any/all/sum and starred calls preserve
                        # their Python argument semantics without re-entering the
                        # evaporated generator control node.
                        result = tuple(materialized)
                    else:
                        result = materialized
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
                    try:
                        plan = loop_plans_by_node[generator_node]
                    except KeyError as error:
                        raise RuntimeError(
                            "aggregate retained an evaporated generator "
                            f"in {graph.G.graph.get('function_name', '?')}: "
                            f"aggregate={node_id}, generator={generator_node}, "
                            f"planned={tuple(loop_plans_by_node)!r}, "
                            f"specializations="
                            f"{graph.G.graph.get('planner_specializations')!r}, "
                            f"attributes={attributes!r}"
                        ) from error
                    loop = plan.loop
                    iterable = evaluate_node(loop.iterable_node)
                    if shell._profiler.verbose:
                        shell._profiler.trace(
                            path=shell.profile_path,
                            section="comprehension",
                            label=f"generator {generator_node}",
                            fields={
                                "iterable": _diagnostic_value_summary(iterable),
                                "targets": loop.target_bindings,
                            },
                        )
                    iterator = iter(iterable)
                    for item in iterator:
                        targets = loop.target_bindings
                        items_by_name = dict(_destructure_loop_target(
                            expression.generators[index].target,
                            item,
                        ))
                        # A later generator in the same comprehension may
                        # consume a destructured name that never enters a
                        # numerical target binding.  Publish the complete
                        # Python target frame, just as retained ``for`` source
                        # execution does.
                        source_binding_values.update(items_by_name)
                        invalidated = set()
                        assignments = tuple(
                            (
                                (name, binding),
                                items_by_name[str(name)],
                            )
                            for name, binding in targets
                        )
                        for (_name, binding), _value in assignments:
                            invalidated.update(
                                nx.descendants(graph.G, binding)
                            )
                        invalidated.discard(node_id)
                        for dependent in invalidated:
                            values.pop(dependent, None)
                        for (_name, binding), value in assignments:
                            generator_binding_values[binding] = value
                        values.update(generator_binding_values)
                        for region_index in loop_runtime_region_indices.get(
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
                if capture and isinstance(
                    expression, (ast.ListComp, ast.SetComp)
                ):
                    # Retain the actual wrappers until the one lowering.
                    # Integer-only identity tracking is invalid here because
                    # temporary wrappers can die during a long discovery run
                    # and CPython may reuse their addresses for unrelated
                    # values before the collection is materialized.
                    shell.collection_observations[int(node_id)] = tuple(
                        item
                        for item in result
                        if isinstance(item, AbstractTensor)
                    )
            elif isinstance(expression, ast.comprehension):
                owner = comprehension_owner_by_generator.get(node_id)
                if owner is None:
                    raise RuntimeError(
                        "comprehension control node has no planner-owned "
                        f"expression in {graph.G.graph.get('function_name', '?')} "
                        f"at node {node_id}"
                    )
                result = evaluate_node(owner)
            elif isinstance(expression, ast.Compare):
                by_role = {
                    str(role): int(parent) for parent, role in parents
                }
                source_operands = (
                    expression.left,
                    *expression.comparators,
                )
                operands = []
                for index, source_operand in enumerate(source_operands):
                    role = "lhs" if index == 0 else "rhs"
                    parent = by_role.get(role)
                    if parent is not None and (
                        index == 0 or len(expression.ops) == 1
                    ):
                        operands.append(evaluate_node(parent))
                    else:
                        operands.append(
                            evaluate_static_expression(source_operand)
                        )
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
                        ast.In: lambda left, right: left in right,
                        ast.NotIn: lambda left, right: left not in right,
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
            elif isinstance(expression, (ast.BinOp, ast.AugAssign)):
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
                if len(operands) != 2:
                    raise RuntimeError(
                        "structural binary expression requires two operands "
                        f"in {graph.G.graph.get('function_name', '?')} at "
                        f"node {node_id}: "
                        f"{ast.dump(expression, include_attributes=False)}; "
                        f"parents={parents!r}"
                    )
                result = binary(operands[0], operands[1])
                if not isinstance(result, AbstractTensor):
                    # A tensor-bearing structural op still enters the tape
                    # (see _observe_process_graph_node); this branch runs for
                    # plain values too, and a plain value never touches the
                    # tape at all.  Record it the same way a reference
                    # operator is recorded, in exact order, so an arithmetic
                    # step feeding a field write is a real internal step
                    # instead of looking like a phantom external input.
                    shell.reference_operator_sequence.append(int(node_id))
            elif isinstance(expression, ast.Call):
                if (
                    isinstance(expression.func, ast.Attribute)
                    and isinstance(expression.func.value, ast.Name)
                    and expression.func.value.id in source_binding_values
                ):
                    receiver_parent = next(
                        (
                            int(parent)
                            for parent, role in parents
                            if str(role) in {"operand", "receiver"}
                        ),
                        None,
                    )
                    if receiver_parent is not None:
                        # The receiver visible at this lexical callsite is
                        # authoritative.  A normalized receiver edge can run
                        # through a later LoopExit carrying the same spelling,
                        # creating a false dependency cycle such as an outer
                        # ``mapping.items()`` requiring its inner loop before
                        # the outer target has been assigned.  Seed the exact
                        # receiver identity before traversing call parents;
                        # the ordinary callable/method routing below remains
                        # responsible for compiling or invoking the method.
                        values[receiver_parent] = source_binding_values[
                            expression.func.value.id
                        ]
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
                    graph,
                )
                # Exact source-name arguments observe the active lexical
                # frame. A reduced edge may otherwise alias a saved value to
                # a later write bearing the same spelling.
                if not any(
                    isinstance(argument, ast.Starred)
                    for argument in expression.args
                ):
                    # Normalization can retain multiple SSA parents for one
                    # source spelling after loop-carried/container rebinding.
                    # Those are candidate identities, not additional Python
                    # arguments.  Source syntax owns call arity; keep exactly
                    # one planned slot per written positional argument before
                    # applying lexical/live-value corrections below.
                    source_args = list(args[: len(expression.args)])
                    for index, argument in enumerate(expression.args):
                        if (
                            isinstance(argument, ast.Name)
                            and argument.id in source_binding_values
                            and index < len(source_args)
                        ):
                            source_args[index] = source_binding_values[
                                argument.id
                            ]
                        elif (
                            isinstance(argument, ast.Name)
                            and argument.id not in supplied
                            and index < len(source_args)
                        ):
                            live_argument = live_function_module_binding(
                                argument.id
                            )
                            if live_argument is not live_binding_unavailable:
                                source_args[index] = live_argument
                        elif (
                            index < len(source_args)
                            and (
                                isinstance(argument, (ast.IfExp, ast.BoolOp))
                                or (
                                    isinstance(argument, ast.Compare)
                                    and len(argument.ops) == 1
                                    and isinstance(
                                        argument.ops[0],
                                        (ast.Is, ast.IsNot),
                                    )
                                )
                            )
                        ):
                            source_args[index] = (
                                evaluate_reduced_control_expression(argument)
                            )
                    args = tuple(source_args)
                for keyword in expression.keywords:
                    if (
                        keyword.arg is not None
                        and isinstance(keyword.value, ast.Name)
                        and keyword.value.id in source_binding_values
                    ):
                        kwargs[keyword.arg] = source_binding_values[
                            keyword.value.id
                        ]
                    elif (
                        keyword.arg is not None
                        and isinstance(keyword.value, ast.Name)
                        and keyword.value.id not in supplied
                    ):
                        live_argument = live_function_module_binding(
                            keyword.value.id
                        )
                        if live_argument is not live_binding_unavailable:
                            kwargs[keyword.arg] = live_argument
                external_ref = attributes.get("external_callee_ref")
                callee_ref = attributes.get(
                    "callee_ref", attributes.get("method_ref")
                )
                direct_self_recursion = (
                    isinstance(expression.func, ast.Name)
                    and expression.func.id
                    == graph.G.graph.get("function_name")
                )
                if direct_self_recursion:
                    callee_ref = graph.G.graph.get("function_ref", callee_ref)
                class_ref = attributes.get("class_ref")
                static_callable = None
                runtime_bound_callable = None
                static_reference = attributes.get(
                    "static_python_reference"
                )
                if isinstance(expression.func, ast.Name):
                    source_name = expression.func.id
                    exact_callable = None
                    if source_name in source_binding_values:
                        candidate = source_binding_values[source_name]
                        if callable(candidate):
                            exact_callable = candidate
                    elif source_name in supplied:
                        candidate = supplied[source_name]
                        if callable(candidate):
                            exact_callable = candidate
                    else:
                        candidate = shell.static_python_bindings.get(
                            source_name,
                            getattr(builtins, source_name, None),
                        )
                        if callable(candidate):
                            exact_callable = candidate
                    if exact_callable is not None:
                        # A bare source spelling has Python lexical authority.
                        # Normalization may merge call edges whose constructors
                        # have equal-looking scalar results (notably int(1)
                        # and bool(1)); it may not change which callable the
                        # source named.
                        static_callable = exact_callable
                        static_reference = source_name
                        external_ref = None
                        callable_name = getattr(
                            exact_callable, "__name__", None
                        )
                        internal_ref = callee_ref
                        if internal_ref is None:
                            for reference_name in (
                                source_name,
                                callable_name,
                            ):
                                if reference_name is None:
                                    continue
                                reference = graph.function_table.reference(
                                    reference_name
                                )
                                if reference is not None:
                                    internal_ref = reference.address
                                    if internal_ref in shell.function_shells:
                                        break
                        internal_shell = None
                        if internal_ref is not None:
                            internal_shell = shell.function_shells.get(
                                int(internal_ref)
                            )
                        internal_name = (
                            internal_shell.process_graph.G.graph.get(
                                "function_name"
                            )
                            if internal_shell is not None
                            else None
                        )
                        # Keep a source-ingested function linked to its
                        # planner shell.  Only discard a spelling-derived
                        # reference when it actually names a different
                        # callable (the int(1)/bool(1) collision described
                        # above).  Calling host Python with a compiled
                        # callback argument would escape the compiler and is
                        # invalid.
                        callee_ref = (
                            internal_ref
                            if internal_shell is not None
                            and str(internal_name) == str(callable_name)
                            else None
                        )
                        class_ref = None
                if static_reference is not None:
                    try:
                        candidate = _static_python_value(
                            shell.static_python_bindings,
                            static_reference,
                        )
                    except (AttributeError, KeyError):
                        candidate = None
                    if callable(candidate):
                        static_callable = candidate
                        if getattr(candidate, "__self__", None) is not None:
                            # Exact bound-call authority outranks a bare-name
                            # function-table collision and retains its receiver.
                            callee_ref = None
                if isinstance(expression.func, ast.Attribute):
                    receiver_parent = next(
                        (
                            int(parent)
                            for parent, role in parents
                            if str(role) in {"operand", "receiver"}
                        ),
                        None,
                    )
                    if (
                        receiver_parent is None
                        and id(expression.func.value) in graph.G
                    ):
                        # Older compiled-plan checkpoints may have already
                        # consumed the explicit operand edge while retaining
                        # the original Attribute AST.  Its value node remains
                        # the exact receiver identity and is safe to recover.
                        receiver_parent = id(expression.func.value)
                    source_receiver_name = (
                        expression.func.value.id
                        if isinstance(expression.func.value, ast.Name)
                        else None
                    )
                    receiver_value = (
                        source_binding_values[source_receiver_name]
                        if source_receiver_name in source_binding_values
                        else evaluate_node(receiver_parent)
                        if receiver_parent is not None
                        else None
                    )
                    if isinstance(
                        receiver_value,
                        (_CompiledStructuralObject, _CompiledStructuralClass),
                    ):
                        bound_method = getattr(
                            receiver_value, expression.func.attr, None
                        )
                        if isinstance(bound_method, _CompiledStructuralMethod):
                            # Runtime structural type identity is stronger than
                            # a spelling-only method-table guess.
                            callee_ref = bound_method.method_ref
                    elif receiver_value is not None:
                        candidate = getattr(
                            receiver_value, expression.func.attr, None
                        )
                        if callable(candidate):
                            candidate_name = getattr(
                                getattr(candidate, "__func__", candidate),
                                "__name__",
                                None,
                            )
                            internal_ref = callee_ref
                            if internal_ref is None and candidate_name:
                                reference = graph.function_table.reference(
                                    candidate_name
                                )
                                if reference is not None:
                                    internal_ref = reference.address
                            internal_shell = (
                                shell.function_shells.get(int(internal_ref))
                                if internal_ref is not None
                                else None
                            )
                            internal_graph = (
                                internal_shell.process_graph.G.graph
                                if internal_shell is not None
                                else {}
                            )
                            method_owner = internal_graph.get("method_owner")
                            receiver_type = type(receiver_value)
                            receiver_owner_names = {
                                receiver_type.__name__,
                                receiver_type.__qualname__,
                                f"{receiver_type.__module__}."
                                f"{receiver_type.__qualname__}",
                            }
                            if (
                                internal_shell is not None
                                and internal_graph.get("method_binding")
                                == "instance"
                                and str(internal_graph.get("function_name"))
                                == str(candidate_name)
                                and str(method_owner) in receiver_owner_names
                            ):
                                # The exact runtime bound method validates the
                                # planner's class-method reference.  Keep the
                                # call inside the source hierarchy so its
                                # numerical regions enter the one discovery
                                # tape.  Spelling alone remains insufficient:
                                # unrelated same-named methods still take the
                                # host-bound path below.
                                callee_ref = internal_ref
                                runtime_bound_callable = None
                            else:
                                runtime_bound_callable = candidate
                                callee_ref = None
                                external_ref = None
                if class_ref is not None:
                    descriptor = (
                        graph.G.graph.get("class_table", {}).get(class_ref)
                    )
                    if descriptor is None:
                        raise RuntimeError(
                            f"unknown compiled class reference {class_ref!r}"
                        )
                    host_factory = (
                        static_callable
                        if isinstance(static_callable, type)
                        and getattr(static_callable, "__new__", object.__new__)
                        is not object.__new__
                        else None
                    )
                    if host_factory is not None:
                        # A custom ``__new__`` is an allocation/factory
                        # protocol, not an initializer.  Until allocation IR
                        # exists, replacing it with a field-only structural
                        # object is observably wrong (Path -> WindowsPath is
                        # the canonical example). Preserve the exact available
                        # factory and let subsequent bound calls retain their
                        # real receiver.
                        result = host_factory(*args, **kwargs)
                        values[node_id] = result
                        return result
                    result = _CompiledStructuralObject(
                        class_ref,
                        descriptor,
                        args,
                        kwargs,
                    )
                    initializer_ref = (
                        descriptor.get("methods") or {}
                    ).get("__init__")
                    initializer = (
                        shell.function_shells.get(int(initializer_ref))
                        if initializer_ref is not None
                        else None
                    )
                    if initializer is not None:
                        initializer.static_python_bindings = {
                            **shell.static_python_bindings,
                            **initializer.static_python_bindings,
                        }
                        receiver_parameter, positional_parameters, _ = (
                            _method_parameter_layout(
                                initializer.process_graph.G
                            )
                        )
                        initializer_inputs = dict(
                            zip(positional_parameters, args)
                        ) | kwargs
                        if receiver_parameter is not None:
                            initializer_inputs[receiver_parameter] = result
                        for name, default in (
                            initializer.process_graph.G.graph.get(
                                "parameter_defaults",
                                {},
                            ).items()
                        ):
                            initializer_inputs.setdefault(name, default)
                        if capture:
                            _coordinate_scheduled_capture(
                                initializer,
                                initializer_inputs,
                                device=device,
                                capture=True,
                                discovery_session=discovery_session,
                            )
                        elif initializer.whole_program_compiled:
                            initializer.execute_process_graph(
                                initializer_inputs
                            )
                        else:
                            initializer.coordinate_first_invocation(
                                initializer_inputs,
                                device=device,
                            )
                elif runtime_bound_callable is not None:
                    try:
                        result = runtime_bound_callable(*args, **kwargs)
                    except Exception as error:
                        if hasattr(error, "add_note"):
                            error.add_note(
                                "runtime-bound Python call failed; "
                                f"shell={shell.profile_path!r}; "
                                f"node={node_id}; "
                                f"call={ast.dump(expression, include_attributes=False)!r}; "
                                f"callable={_diagnostic_value_summary(runtime_bound_callable)!r}; "
                                f"args={tuple(_diagnostic_value_summary(value) for value in args)!r}; "
                                f"kwargs={{{', '.join(f'{name!r}: {_diagnostic_value_summary(value)!r}' for name, value in kwargs.items())}}}; "
                                f"parents={parents!r}; "
                                f"static_arguments={static_arguments!r}"
                            )
                        raise
                elif external_ref is not None:
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
                ):
                    nested = (
                        shell
                        if direct_self_recursion
                        else getattr(
                            shell, "callsite_function_shells", {}
                        ).get(node_id)
                        or shell.function_shells[int(callee_ref)]
                    )
                    nested.static_python_bindings = {
                        **shell.static_python_bindings,
                        **nested.static_python_bindings,
                    }
                    if shell._profiler.verbose:
                        shell._profiler.trace(
                            path=shell.profile_path,
                            section="function-call",
                            label=(
                                f"{graph.G.graph.get('function_name', '?')} "
                                f"node {node_id} -> "
                                f"{nested.process_graph.G.graph.get('function_name', callee_ref)}"
                            ),
                            fields={
                                "positional": tuple(
                                    _diagnostic_value_summary(value)
                                    for value in args
                                ),
                                "keywords": {
                                    name: _diagnostic_value_summary(value)
                                    for name, value in kwargs.items()
                                },
                            },
                        )
                    if nested.process_graph_boundary:
                        if nested.python_callable is None:
                            raise RuntimeError(
                                "host-boundary ProcessGraph function has no "
                                f"Python implementation: {callee_ref}"
                            )
                        result = nested.python_callable(*args, **kwargs)
                        values[node_id] = result
                        return result
                    nested_graph_metadata = nested.process_graph.G.graph
                    receiver_parameter, positional_parameters, _ = (
                        _method_parameter_layout(
                            nested.process_graph.G
                        )
                    )
                    receiver_value = None
                    call_args = args
                    if nested_graph_metadata.get("method_binding") == "class":
                        owner = nested_graph_metadata.get("method_owner")
                        descriptor = (
                            nested_graph_metadata.get("class_table", {})
                            .get(owner)
                        )
                        if descriptor is None:
                            raise RuntimeError(
                                "compiled classmethod has no class descriptor: "
                                f"{owner!r}"
                            )
                        receiver_value = _CompiledStructuralClass(
                            owner, descriptor
                        )
                    elif nested_graph_metadata.get("method_binding") == "instance":
                        receiver_parent = next(
                            (
                                int(parent)
                                for parent, role in parents
                                if str(role) in {"operand", "receiver"}
                            ),
                            None,
                        )
                        if (
                            receiver_parent is None
                            and isinstance(expression.func, ast.Attribute)
                        ):
                            receiver_parent = _resolve_reference_node(
                                expression.func.value
                            )
                        if receiver_parent is not None:
                            receiver_value = evaluate_node(receiver_parent)
                        elif args:
                            # A method discovered through a named Python
                            # binding is an unbound function call: its first
                            # explicit positional argument is the instance.
                            receiver_value = args[0]
                            call_args = args[1:]
                    nested_inputs = dict(
                        zip(positional_parameters, call_args)
                    ) | kwargs
                    if receiver_parameter is not None and receiver_value is not None:
                        nested_inputs[receiver_parameter] = receiver_value
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
                        resolution_errors = []
                        if (
                            name not in nested_inputs
                            and discovery_session is not None
                        ):
                            for lexical_frame in reversed(
                                discovery_session.get(
                                    "lexical_frames", ()
                                )
                            ):
                                try:
                                    resolved, lexical_value = (
                                        lexical_frame["resolve"](name)
                                    )
                                except (KeyError, RuntimeError):
                                    continue
                                if resolved:
                                    nested_inputs[name] = lexical_value
                                    break
                        if name not in nested_inputs and name in supplied:
                            nested_inputs[name] = supplied[name]
                        if name not in nested_inputs:
                            for identity in reversed(
                                caller_identities.get(name, ())
                            ):
                                try:
                                    nested_inputs[name] = evaluate_node(
                                        identity
                                    )
                                except (KeyError, RuntimeError) as error:
                                    resolution_errors.append(
                                        (identity, str(error))
                                    )
                                    continue
                                break
                        if (
                            name not in nested_inputs
                            and caller_identities.get(name)
                        ):
                            raise RuntimeError(
                                f"failed to route closure {name!r} into "
                                f"{nested.process_graph.G.graph.get('function_name', '?')}; "
                                f"caller identities={caller_identities[name]!r}; "
                                f"errors={tuple(resolution_errors)!r}"
                            )
                    for name, value in (
                        nested.process_graph.G.graph.get(
                            "parameter_defaults",
                            {},
                        ).items()
                    ):
                        nested_inputs.setdefault(name, value)
                    # Literal callsite specializations were propagated across
                    # the function table before deployment planning.  Values
                    # observed here are runtime inputs, not authority to
                    # rebuild loop IR or invalidate the region schedule.
                    if capture:
                        _coordinate_scheduled_capture(
                            nested,
                            nested_inputs,
                            device=device,
                            capture=True,
                            discovery_session=discovery_session,
                        )
                    elif nested.whole_program_compiled:
                        nested.execute_process_graph(nested_inputs)
                    else:
                        nested.coordinate_first_invocation(
                            nested_inputs,
                            device=device,
                        )
                    result = nested.last_result
                    if discovery_session is not None:
                        history = discovery_session.setdefault(
                            "call_return_history", []
                        )
                        history.append((
                            nested.profile_path,
                            _diagnostic_value_summary(result),
                        ))
                        del history[:-64]
                elif static_reference:
                    # Exact lexical/builtin resolution can establish a static
                    # source reference even when the normalized call node did
                    # not originally carry metadata for one.
                    reference = static_reference
                    structural_constructors = {
                        "tuple": tuple,
                        "list": list,
                        "set": set,
                        "dict": dict,
                    }
                    constructor = structural_constructors.get(reference)
                    if constructor is not None:
                        if (
                            attributes.get("materialization_kind")
                            in {
                                "unrolled_loop",
                                "retained_loop_aggregate",
                            }
                        ):
                            if kwargs:
                                raise TypeError(
                                    "loop materialization cannot have "
                                    "keyword arguments"
                                )
                            result = constructor(args)
                        else:
                            try:
                                result = constructor(*args, **kwargs)
                            except Exception as error:
                                if hasattr(error, "add_note"):
                                    error.add_note(
                                        "static structural constructor failed; "
                                        f"shell={shell.profile_path!r}; "
                                        f"node={node_id}; "
                                        f"call={ast.dump(expression, include_attributes=False)!r}; "
                                        f"args={tuple(_diagnostic_value_summary(value) for value in args)!r}; "
                                        f"kwargs={{{', '.join(f'{name!r}: {_diagnostic_value_summary(value)!r}' for name, value in kwargs.items())}}}"
                                    )
                                raise
                    else:
                        callable_value = static_callable or _static_python_value(
                            shell.static_python_bindings, reference
                        )
                        if any(
                            isinstance(value, _CompiledStructuralFunction)
                            for value in (*args, *kwargs.values())
                        ):
                            available_shells = tuple(
                                (
                                    int(address),
                                    child.process_graph.G.graph.get(
                                        "function_name"
                                    ),
                                )
                                for address, child in sorted(
                                    shell.function_shells.items()
                                )
                            )
                            raise RuntimeError(
                                "source callable with compiled callback was "
                                "not linked to a planner shell; "
                                f"reference={reference!r}; "
                                f"callable={getattr(callable_value, '__name__', callable_value)!r}; "
                                f"callee_ref={attributes.get('callee_ref')!r}; "
                                f"available_shells={available_shells!r}"
                            )
                        if (
                            reference in {"max", "min"}
                            and not kwargs
                            and len(args) >= 2
                            and any(
                                isinstance(value, AbstractTensor)
                                for value in args
                            )
                        ):
                            # Python's scalar max/min returns one of its
                            # operands by identity.  During tensor discovery
                            # that would collapse the Call result onto the
                            # selected input from this one sample, assigning
                            # one primitive occurrence to two ProcessGraph
                            # endpoints and (worse) specializing away the
                            # runtime comparison.  Preserve the source scalar
                            # semantics as an elementwise tensor operation;
                            # scalar and singleton arenas are its degenerate
                            # case and therefore retain a distinct SSA result.
                            result = args[0]
                            method_name = (
                                "maximum" if reference == "max" else "minimum"
                            )
                            for operand in args[1:]:
                                if not isinstance(result, AbstractTensor):
                                    result = AbstractTensor.tensor(result)
                                result = getattr(result, method_name)(operand)
                        else:
                            call_args = args
                            if (
                                reference == "next"
                                and call_args
                                and isinstance(call_args[0], (list, tuple))
                            ):
                                # A GeneratorExp/comprehension argument is
                                # traced as an already-materialized list here
                                # (this simulator has no lazy generator
                                # value), but the real builtin `next()` --
                                # reached as a static host call because it
                                # has no compiled lowering -- requires an
                                # actual iterator, not a list, and raises
                                # TypeError otherwise. `next(genexpr, default)`
                                # is an ordinary, common idiom (used by
                                # MachineExecutionOrchestrator._relative_target,
                                # among others); wrap the materialized
                                # sequence in iter() so it behaves exactly
                                # like the real generator would have.
                                call_args = (iter(call_args[0]), *call_args[1:])
                            try:
                                result = callable_value(*call_args, **kwargs)
                            except Exception as error:
                                if hasattr(error, "add_note"):
                                    error.add_note(
                                        "static Python host call failed; "
                                        f"shell={shell.profile_path!r}; "
                                        f"node={node_id}; "
                                        f"reference={reference!r}; "
                                        f"call={ast.dump(expression, include_attributes=False)!r}; "
                                        f"callable={_diagnostic_value_summary(callable_value)!r}; "
                                        f"args={tuple(_diagnostic_value_summary(value) for value in call_args)!r}; "
                                        f"kwargs={{{', '.join(f'{name!r}: {_diagnostic_value_summary(value)!r}' for name, value in kwargs.items())}}}"
                                    )
                                raise
                elif any(
                    str(role) == "callee" for _parent, role in parents
                ):
                    callee_parent = next(
                        parent
                        for parent, role in parents
                        if str(role) == "callee"
                    )
                    callee_value = evaluate_node(callee_parent)
                    if isinstance(callee_value, _CompiledStructuralFunction):
                        nested = (
                            getattr(shell, "callsite_function_shells", {}).get(
                                node_id
                            )
                            or shell.function_shells[
                                int(callee_value.function_ref)
                            ]
                        )
                        _receiver, positional_parameters, _all_parameters = (
                            _method_parameter_layout(nested.process_graph.G)
                        )
                        nested_inputs = dict(
                            zip(positional_parameters, args)
                        ) | kwargs
                        caller_identities = graph.G.graph.get(
                            "identity_table", {}
                        )
                        for _input_id, input_data in (
                            nested.process_graph.G.nodes(data=True)
                        ):
                            input_attributes = input_data.get(
                                "attributes"
                            ) or {}
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
                                "parameter_defaults", {}
                            ).items()
                        ):
                            nested_inputs.setdefault(name, value)
                        if capture:
                            _coordinate_scheduled_capture(
                                nested,
                                nested_inputs,
                                device=device,
                                capture=True,
                                discovery_session=discovery_session,
                            )
                        elif nested.whole_program_compiled:
                            nested.execute_process_graph(nested_inputs)
                        else:
                            nested.coordinate_first_invocation(
                                nested_inputs,
                                device=device,
                            )
                        result = nested.last_result
                        values[node_id] = result
                        return result
                    if callable(callee_value) and not isinstance(
                        callee_value,
                        (
                            _CompiledStructuralClass,
                            _CompiledStructuralMethod,
                        ),
                    ):
                        if not capture:
                            raise RuntimeError(
                                "an uncaptured runtime callable reached "
                                f"compiled coordinator execution at node {node_id}"
                            )
                        # A callback parameter is an invocation boundary, not
                        # a name-based function-table reference.  During the
                        # one authorized discovery pass, invoke the supplied
                        # callable under the active AbstractTensor backend so
                        # its canonical tensor work is appended to the same
                        # program tape.  Installed execution may never fall
                        # back to the Python callable: its work must already
                        # have been absorbed by this capture.
                        result = callee_value(*args, **kwargs)
                        values[node_id] = result
                        return result
                    if not isinstance(
                        callee_value,
                        (
                            _CompiledStructuralClass,
                            _CompiledStructuralMethod,
                        ),
                    ):
                        raise RuntimeError(
                            "runtime callee edge is not a compiled "
                            f"structural class at node {node_id}: "
                            f"{type(callee_value).__name__}"
                        )
                    if isinstance(
                        callee_value,
                        _CompiledStructuralClass,
                    ):
                        result = _CompiledStructuralObject(
                            callee_value.class_ref,
                            callee_value.descriptor,
                            args,
                            kwargs,
                        )
                        initializer_ref = (
                            callee_value.descriptor.get("methods") or {}
                        ).get("__init__")
                        if initializer_ref is not None:
                            nested = shell.function_shells[int(initializer_ref)]
                            (
                                receiver_parameter,
                                positional_parameters,
                                _,
                            ) = _method_parameter_layout(
                                nested.process_graph.G
                            )
                            nested_inputs = dict(
                                zip(positional_parameters, args)
                            ) | kwargs
                            if receiver_parameter is not None:
                                nested_inputs[receiver_parameter] = result
                            for name, value in (
                                nested.process_graph.G.graph.get(
                                    "parameter_defaults", {}
                                ).items()
                            ):
                                nested_inputs.setdefault(name, value)
                            if capture:
                                _coordinate_scheduled_capture(
                                    nested,
                                    nested_inputs,
                                    device=device,
                                    capture=True,
                                    discovery_session=discovery_session,
                                )
                            elif nested.whole_program_compiled:
                                nested.execute_process_graph(nested_inputs)
                            else:
                                nested.coordinate_first_invocation(
                                    nested_inputs,
                                    device=device,
                                )
                    else:
                        nested = (
                            getattr(shell, "callsite_function_shells", {}).get(
                                node_id
                            )
                            or shell.function_shells[
                                int(callee_value.method_ref)
                            ]
                        )
                        receiver_parameter, positional_parameters, _ = (
                            _method_parameter_layout(
                                nested.process_graph.G
                            )
                        )
                        nested_inputs = dict(
                            zip(positional_parameters, args)
                        ) | kwargs
                        if receiver_parameter is not None:
                            nested_inputs[receiver_parameter] = (
                                callee_value.receiver
                            )
                        for name, value in (
                            nested.process_graph.G.graph.get(
                                "parameter_defaults",
                                {},
                            ).items()
                        ):
                            nested_inputs.setdefault(name, value)
                        if capture:
                            _coordinate_scheduled_capture(
                                nested,
                                nested_inputs,
                                device=device,
                                capture=True,
                                discovery_session=discovery_session,
                            )
                        elif nested.whole_program_compiled:
                            nested.execute_process_graph(nested_inputs)
                        else:
                            nested.coordinate_first_invocation(
                                nested_inputs,
                                device=device,
                            )
                        result = nested.last_result
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
                            live_callable = (
                                evaluate_reduced_control_expression(
                                    expression.func
                                )
                            )
                            if callable(live_callable):
                                result = live_callable(*args, **kwargs)
                                values[node_id] = result
                                return result
                            raise RuntimeError(
                                "attribute call has neither a callable nor "
                                f"receiver edge at ProcessGraph node {node_id}: "
                                f"{ast.dump(expression, include_attributes=False)}; "
                                f"parents={parents!r}"
                            )
                        receiver = evaluate_node(receiver_parent)
                        if isinstance(
                            receiver,
                            (_CompiledStructuralObject, _CompiledStructuralClass),
                        ):
                            methods = (
                                receiver.methods
                                if isinstance(receiver, _CompiledStructuralObject)
                                else receiver.descriptor.get("methods") or {}
                            )
                            method_ref = methods.get(expression.func.attr)
                            if method_ref is None:
                                raise RuntimeError(
                                    f"compiled class {receiver.class_ref!r} "
                                    "has no retained method "
                                    f"{expression.func.attr!r}"
                                )
                            nested = (
                                getattr(
                                    shell,
                                    "callsite_function_shells",
                                    {},
                                ).get(node_id)
                                or shell.function_shells[int(method_ref)]
                            )
                            (
                                receiver_parameter,
                                positional_parameters,
                                _,
                            ) = _method_parameter_layout(
                                nested.process_graph.G
                            )
                            nested_inputs = dict(
                                zip(positional_parameters, args)
                            ) | kwargs
                            if receiver_parameter is not None:
                                nested_inputs[receiver_parameter] = receiver
                            for name, value in (
                                nested.process_graph.G.graph.get(
                                    "parameter_defaults",
                                    {},
                                ).items()
                            ):
                                nested_inputs.setdefault(name, value)
                            if capture:
                                _coordinate_scheduled_capture(
                                    nested,
                                    nested_inputs,
                                    device=device,
                                    capture=True,
                                    discovery_session=discovery_session,
                                )
                            elif nested.whole_program_compiled:
                                nested.execute_process_graph(nested_inputs)
                            else:
                                nested.coordinate_first_invocation(
                                    nested_inputs,
                                    device=device,
                                )
                            result = nested.last_result
                        else:
                            result = getattr(
                                receiver,
                                expression.func.attr,
                            )(*args, **kwargs)
                else:
                    raise RuntimeError(
                        f"unresolved ProcessGraph Call node {node_id} in "
                        f"{graph.G.graph.get('function_name', '?')}: "
                        f"{ast.dump(expression, include_attributes=False)}; "
                        f"parents={parents!r}; attributes={attributes!r}"
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
            elif node_type == "Phi":
                by_role = {
                    str(role): parent for parent, role in parents
                }
                selected = (
                    by_role["body"]
                    if bool(evaluate_node(by_role["test"]))
                    else by_role["orelse"]
                )
                result = evaluate_node(selected)
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
                target_initials = (
                    (graph.G.nodes[node_id].get("attributes") or {})
                    .get("loop_target_initials") or {}
                )
                for name, binding in loop.target_bindings:
                    initial = target_initials.get(name)
                    if initial is not None:
                        values[binding] = evaluate_node(int(initial))
                if shell._profiler.verbose:
                    shell._profiler.trace(
                        path=shell.profile_path,
                        section="loop",
                        label=f"node {node_id} begin",
                        fields={
                            "strategy": plan.strategy.value,
                            "trip_count": loop.trip_count,
                            "carried": tuple(
                                name
                                for name, _initial, _updated
                                in loop.carried_bindings
                            ),
                        },
                    )
                while True:
                    if isinstance(expression, ast.For):
                        try:
                            item = next(iterator)
                        except StopIteration:
                            break
                        targets = loop.target_bindings
                        items_by_name = dict(_destructure_loop_target(
                            expression.target, item
                        ))
                        target_assignments = tuple(
                            (
                                (name, binding),
                                items_by_name[str(name)],
                            )
                            for name, binding in targets
                        )
                    else:
                        target_assignments = ()
                        test_parent = next(
                            parent
                            for parent, role in parents
                            if str(role) == "test"
                        )
                        if not bool(evaluate_node(test_parent)):
                            break

                    # Generated binding nodes (for example tuple-unpack
                    # ``Indexed`` nodes) can sit just beyond the lexical AST
                    # body while still depending on this iteration.  Clear
                    # the full downstream value cone so those bindings and
                    # any prematurely cached post-loop values cannot retain
                    # the preceding iteration.
                    for dependent in loop_invalidated_nodes[node_id]:
                        values.pop(dependent, None)
                    for (_name, binding), value in target_assignments:
                        values[binding] = value
                        generator_binding_values[binding] = value
                    # A flat graph dependency can demand a nested loop before
                    # the source-statement walker reaches that loop.  Its
                    # destructured targets are still ordinary Python lexical
                    # assignments, so publish every leaf into both runtime
                    # identity tables and the source frame.  Keeping only the
                    # planner-selected bindings makes a sibling target (for
                    # example ``value_ids`` in ``for name, value_ids in ...``)
                    # look like a missing function input when the inner loop
                    # consumes it.
                    source_binding_values.update(items_by_name)
                    for region_index in loop_runtime_region_indices.get(
                        node_id,
                        (),
                    ):
                        completed_regions.discard(region_index)
                    loop_signal = None
                    try:
                        for body_node in (
                            candidate
                            for candidate in loop.body_nodes
                            if (
                                candidate not in nested_body_nodes_by_loop[
                                    int(node_id)
                                ]
                            )
                        ):
                            body_region = region_for_node.get(body_node)
                            if is_shader_internal_node(body_node):
                                evaluate_region(body_region)
                            else:
                                result = evaluate_node(body_node)
                    except _LoopBreakSignal:
                        loop_signal = "break"
                    except _LoopContinueSignal:
                        loop_signal = "continue"
                    if loop_signal == "break":
                        iterations_completed += 1
                        break
                    for carried_name, initial, updated in loop.carried_bindings:
                        if (
                            loop_signal == "continue"
                            and carried_name in source_binding_values
                        ):
                            values[initial] = source_binding_values[
                                carried_name
                            ]
                        elif loop_signal != "continue":
                            values[initial] = evaluate_node(updated)
                    iterations_completed += 1
                    if loop_signal == "continue":
                        continue
                    if shell._profiler.verbose:
                        shell._profiler.trace(
                            path=shell.profile_path,
                            section="loop-iteration",
                            label=(
                                f"node {node_id} iteration "
                                f"{iterations_completed}"
                            ),
                            fields={
                                name: _diagnostic_value_summary(
                                    values.get(initial)
                                )
                                for name, initial, _updated
                                in loop.carried_bindings
                            },
                        )
                    if (
                        isinstance(expression, ast.While)
                        and iterations_completed > 1_000_000
                    ):
                        raise RuntimeError(
                            "ProcessGraph while loop exceeded safety limit"
                        )
                if shell._profiler.verbose:
                    shell._profiler.trace(
                        path=shell.profile_path,
                        section="loop",
                        label=f"node {node_id} end",
                        fields={"iterations": iterations_completed},
                    )
            elif isinstance(expression, ast.Raise):
                exception = (
                    evaluate_node(parents[0][0])
                    if parents
                    else RuntimeError("ProcessGraph bare raise")
                )
                if hasattr(exception, "add_note"):
                    input_state = tuple(
                        (
                            int(input_id),
                            (input_data.get("attributes") or {}).get(
                                "binding_name",
                                input_data.get("label"),
                            ),
                            _diagnostic_value_summary(values.get(input_id)),
                        )
                        for input_id, input_data in graph.G.nodes(data=True)
                        if input_data.get("type") == "Input"
                        and input_id in values
                    )
                    exception.add_note(
                        "raised by ProcessGraph "
                        f"{graph.G.graph.get('function_name', '?')} at node "
                        f"{node_id}; inputs={input_state!r}"
                    )
                raise exception
            elif isinstance(expression, ast.Assert):
                by_role = {
                    str(role): parent for parent, role in parents
                }
                if not bool(evaluate_node(by_role["test"])):
                    message_id = by_role.get("msg")
                    message = (
                        evaluate_node(message_id)
                        if message_id is not None
                        else None
                    )
                    raise AssertionError(message)
                result = None
            elif isinstance(expression, ast.Pass):
                result = None
            elif isinstance(expression, ast.Break):
                raise _LoopBreakSignal()
            elif isinstance(expression, ast.Continue):
                raise _LoopContinueSignal()
            elif node_type == "SetAttr":
                by_role = {
                    str(role): parent for parent, role in parents
                }
                receiver = evaluate_node(by_role["object"])
                value = evaluate_node(by_role["value"])
                setattr(
                    receiver,
                    (data.get("attributes") or {})["attribute"],
                    value,
                )
                result = value
                shell.reference_operator_sequence.append(int(node_id))
            elif node_type == "DelAttr":
                by_role = {
                    str(role): parent for parent, role in parents
                }
                receiver = evaluate_node(by_role["object"])
                delattr(
                    receiver,
                    (data.get("attributes") or {})["attribute"],
                )
                result = None
            elif node_type == "DelItem":
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
                del base[index]
                result = None
            elif node_type == "StaticReference":
                # This is a compiler-owned link to a function subgraph or
                # structural symbol.  It is deliberately not a Python object
                # and performs no runtime work.
                attributes = data.get("attributes") or {}
                first_class_ref = attributes.get(
                    "first_class_function_ref"
                )
                static_reference = attributes.get("static_python_reference")
                static_value = None
                if static_reference is not None:
                    try:
                        static_value = _static_python_value(
                            shell.static_python_bindings,
                            static_reference,
                        )
                    except (AttributeError, KeyError):
                        pass
                if isinstance(static_value, type):
                    # Classes used as first-class values (most importantly in
                    # isinstance/issubclass tuples) retain Python type
                    # identity. Constructor calls remain governed by the
                    # class_ref branch above.
                    result = static_value
                else:
                    result = (
                        _CompiledStructuralFunction(first_class_ref)
                        if first_class_ref is not None
                        else attributes.get(
                            "function_ref",
                            static_reference,
                        )
                    )
            elif isinstance(expression, ast.Lambda):
                lambda_expression = expression
                positional_parameters = tuple(
                    (*lambda_expression.args.posonlyargs,
                     *lambda_expression.args.args)
                )
                positional_defaults = {
                    parameter.arg: evaluate_reduced_control_expression(default)
                    for parameter, default in zip(
                        positional_parameters[
                            len(positional_parameters)
                            - len(lambda_expression.args.defaults):
                        ],
                        lambda_expression.args.defaults,
                    )
                }
                keyword_defaults = {
                    parameter.arg: evaluate_reduced_control_expression(default)
                    for parameter, default in zip(
                        lambda_expression.args.kwonlyargs,
                        lambda_expression.args.kw_defaults,
                    )
                    if default is not None
                }

                def structural_lambda(*call_args, **call_kwargs):
                    remaining_kwargs = dict(call_kwargs)
                    bound = {}
                    if (
                        len(call_args) > len(positional_parameters)
                        and lambda_expression.args.vararg is None
                    ):
                        raise TypeError("lambda received too many arguments")
                    for index, parameter in enumerate(positional_parameters):
                        if index < len(call_args):
                            if parameter.arg in remaining_kwargs:
                                raise TypeError(
                                    f"lambda got multiple values for "
                                    f"{parameter.arg!r}"
                                )
                            bound[parameter.arg] = call_args[index]
                        elif parameter.arg in remaining_kwargs:
                            bound[parameter.arg] = remaining_kwargs.pop(
                                parameter.arg
                            )
                        elif parameter.arg in positional_defaults:
                            bound[parameter.arg] = positional_defaults[
                                parameter.arg
                            ]
                        else:
                            raise TypeError(
                                f"lambda missing argument {parameter.arg!r}"
                            )
                    if lambda_expression.args.vararg is not None:
                        bound[lambda_expression.args.vararg.arg] = tuple(
                            call_args[len(positional_parameters):]
                        )
                    for parameter in lambda_expression.args.kwonlyargs:
                        if parameter.arg in remaining_kwargs:
                            bound[parameter.arg] = remaining_kwargs.pop(
                                parameter.arg
                            )
                        elif parameter.arg in keyword_defaults:
                            bound[parameter.arg] = keyword_defaults[
                                parameter.arg
                            ]
                        else:
                            raise TypeError(
                                "lambda missing keyword-only argument "
                                f"{parameter.arg!r}"
                            )
                    if lambda_expression.args.kwarg is not None:
                        bound[lambda_expression.args.kwarg.arg] = (
                            remaining_kwargs
                        )
                        remaining_kwargs = {}
                    if remaining_kwargs:
                        unexpected = next(iter(remaining_kwargs))
                        raise TypeError(
                            f"lambda got unexpected keyword {unexpected!r}"
                        )
                    missing = object()
                    previous = {
                        name: source_binding_values.get(name, missing)
                        for name in bound
                    }
                    source_binding_values.update(bound)
                    try:
                        return evaluate_reduced_control_expression(
                            lambda_expression.body
                        )
                    finally:
                        for name, prior in previous.items():
                            if prior is missing:
                                source_binding_values.pop(name, None)
                            else:
                                source_binding_values[name] = prior

                result = structural_lambda
            elif isinstance(expression, ast.Return) or node_type in {
                "Return",
                "return",
            }:
                result = (
                    evaluate_node(parents[0][0])
                    if parents
                    else None
                )
                values[node_id] = result
                if capture:
                    shell.forward_feed_ids = tuple(feed_maps)
                    shell.forward_aggregate_feed_paths = tuple(
                        aggregate_feed_maps
                    )
                    shell.forward_subgraphs = tuple(captured_subgraphs)
                    shell.forward_compilers = tuple(captured_compilers)
                    shell.forward_region_indices = tuple(
                        captured_region_indices
                    )
                    shell.forward_output_values = tuple(
                        captured_output_values
                    )
                    shell.forward_region_capture_node_ids = tuple(
                        captured_region_node_ids
                    )
                    shell.forward_region_planned_capture_ids = tuple(
                        captured_region_planned_ids
                    )
                    shell.forward_region_planned_input_ids = tuple(
                        captured_region_planned_input_ids
                    )
                shell.captured_values = values
                shell.last_result = result
                raise _SourceReturnSignal(result)
            elif node_type in {"Store", "store", "Output", "output"}:
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
            # Coordinator region arbitration can deliberately clear a
            # region's active markers before replaying it structurally.
            active_nodes.discard(node_id)

    generator_binding_values: dict[int, Any] = {}
    source_binding_values: dict[str, Any] = {
        str((data.get("attributes") or {}).get(
            "binding_name", data.get("label", "")
        )): values[int(node_id)]
        for node_id, data in graph.G.nodes(data=True)
        if str(data.get("type")) in {"Input", "input"}
        and (data.get("attributes") or {}).get("binding_kind")
        == "parameter"
        and int(node_id) in values
    }
    source_binding_node_ids: dict[str, int] = {
        str((data.get("attributes") or {}).get(
            "binding_name", data.get("label", "")
        )): int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        if str(data.get("type")) in {"Input", "input"}
        and (data.get("attributes") or {}).get("binding_kind")
        == "parameter"
        and int(node_id) in values
    }

    def has_live_source_root(expression: ast.AST) -> bool:
        current = expression
        while isinstance(current, (ast.Attribute, ast.Subscript)):
            current = current.value
        return (
            isinstance(current, ast.Name)
            and (
                current.id in source_binding_values
                or current.id in supplied
                or live_function_module_binding(current.id)
                is not live_binding_unavailable
            )
        )

    def evaluate_reduced_control_expression(expression: ast.AST) -> Any:
        if isinstance(
            expression,
            (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp),
        ):
            def target_names(target: ast.AST) -> tuple[str, ...]:
                if isinstance(target, ast.Name):
                    return (target.id,)
                if isinstance(target, (ast.Tuple, ast.List)):
                    return tuple(
                        name
                        for item in target.elts
                        for name in target_names(item)
                    )
                raise NotImplementedError(
                    "reduced comprehension target is not supported: "
                    f"{ast.unparse(target)}"
                )

            def bind_target(target: ast.AST, value: Any) -> None:
                if isinstance(target, ast.Name):
                    source_binding_values[target.id] = value
                    return
                if isinstance(target, (ast.Tuple, ast.List)):
                    unpacked = tuple(value)
                    if len(unpacked) != len(target.elts):
                        raise ValueError(
                            "comprehension target/value arity mismatch"
                        )
                    for item, item_value in zip(target.elts, unpacked):
                        bind_target(item, item_value)
                    return
                raise NotImplementedError(
                    "reduced comprehension target is not supported: "
                    f"{ast.unparse(target)}"
                )

            missing = object()

            def iterate(generators, index=0):
                if index >= len(generators):
                    if isinstance(expression, ast.DictComp):
                        yield (
                            evaluate_reduced_control_expression(
                                expression.key
                            ),
                            evaluate_reduced_control_expression(
                                expression.value
                            ),
                        )
                    else:
                        yield evaluate_reduced_control_expression(
                            expression.elt
                        )
                    return
                generator = generators[index]
                if generator.is_async:
                    raise NotImplementedError(
                        "async reduced comprehensions are not supported"
                    )
                iterable = evaluate_reduced_control_expression(
                    generator.iter
                )
                names = target_names(generator.target)
                previous = {
                    name: source_binding_values.get(name, missing)
                    for name in names
                }
                try:
                    for item in iterable:
                        bind_target(generator.target, item)
                        if all(
                            bool(evaluate_reduced_control_expression(test))
                            for test in generator.ifs
                        ):
                            yield from iterate(generators, index + 1)
                finally:
                    for name, value in previous.items():
                        if value is missing:
                            source_binding_values.pop(name, None)
                        else:
                            source_binding_values[name] = value

            items = iterate(expression.generators)
            if isinstance(expression, ast.GeneratorExp):
                return items
            if isinstance(expression, ast.ListComp):
                return list(items)
            if isinstance(expression, ast.SetComp):
                return set(items)
            return dict(items)
        if isinstance(expression, ast.Name):
            if expression.id in source_binding_values:
                return source_binding_values[expression.id]
            if expression.id in supplied:
                return supplied[expression.id]
            live_binding = live_function_module_binding(expression.id)
            if live_binding is not live_binding_unavailable:
                return live_binding
            identities = graph.G.graph.get("identity_table", {}) or {}
            for identity in reversed(identities.get(expression.id, ())):
                identity = int(identity)
                if identity in values or identity in graph.G:
                    return evaluate_node(identity)
            return evaluate_static_expression(expression)
        if isinstance(expression, ast.IfExp):
            branch = (
                expression.body
                if bool(evaluate_reduced_control_expression(expression.test))
                else expression.orelse
            )
            return evaluate_reduced_control_expression(branch)
        if isinstance(expression, ast.BoolOp):
            if isinstance(expression.op, ast.And):
                result = True
                for operand in expression.values:
                    result = evaluate_reduced_control_expression(operand)
                    if not result:
                        break
                return result
            result = False
            for operand in expression.values:
                result = evaluate_reduced_control_expression(operand)
                if result:
                    break
            return result
        if isinstance(expression, ast.UnaryOp):
            operand = evaluate_reduced_control_expression(
                expression.operand
            )
            unary = {
                ast.Not: operator.not_,
                ast.UAdd: operator.pos,
                ast.USub: operator.neg,
                ast.Invert: operator.invert,
            }.get(type(expression.op))
            if unary is None:
                raise NotImplementedError(
                    "reduced unary operator is not supported: "
                    f"{ast.unparse(expression)}"
                )
            return unary(operand)
        if isinstance(expression, ast.BinOp):
            left = evaluate_reduced_control_expression(expression.left)
            right = evaluate_reduced_control_expression(expression.right)
            binary = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.MatMult: operator.matmul,
                ast.Div: operator.truediv,
                ast.FloorDiv: operator.floordiv,
                ast.Mod: operator.mod,
                ast.Pow: operator.pow,
                ast.LShift: operator.lshift,
                ast.RShift: operator.rshift,
                ast.BitOr: operator.or_,
                ast.BitXor: operator.xor,
                ast.BitAnd: operator.and_,
            }.get(type(expression.op))
            if binary is None:
                raise NotImplementedError(
                    "reduced binary operator is not supported: "
                    f"{ast.unparse(expression)}"
                )
            return binary(left, right)
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
        ):
            source_callable = shell.static_python_bindings.get(
                expression.func.id
            )
            if discovery_session is not None:
                history = discovery_session.setdefault(
                    "structural_predicate_history", []
                )
                history.append((
                    "source-call",
                    expression.func.id,
                    repr(source_callable),
                    bool(
                        isinstance(source_callable, type)
                        and getattr(
                            source_callable, "__new__", object.__new__
                        ) is not object.__new__
                    ),
                ))
                del history[:-32]
            if (
                isinstance(source_callable, type)
                and getattr(source_callable, "__new__", object.__new__)
                is not object.__new__
            ):
                positional = []
                for argument in expression.args:
                    if isinstance(argument, ast.Starred):
                        positional.extend(
                            evaluate_reduced_control_expression(
                                argument.value
                            )
                        )
                    else:
                        positional.append(
                            evaluate_reduced_control_expression(argument)
                        )
                keywords = {}
                for keyword in expression.keywords:
                    value = evaluate_reduced_control_expression(
                        keyword.value
                    )
                    if keyword.arg is None:
                        keywords.update(value)
                    else:
                        keywords[keyword.arg] = value
                return source_callable(*positional, **keywords)
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id in {"isinstance", "issubclass"}
            and len(expression.args) == 2
            and not expression.keywords
        ):
            def evaluate_type_spec(type_expression: ast.AST) -> Any:
                if isinstance(type_expression, ast.Tuple):
                    return tuple(
                        evaluate_type_spec(item)
                        for item in type_expression.elts
                    )
                if isinstance(type_expression, ast.Name):
                    if type_expression.id in shell.static_python_bindings:
                        return shell.static_python_bindings[type_expression.id]
                    builtin_type = getattr(
                        builtins, type_expression.id, None
                    )
                    if isinstance(builtin_type, type):
                        return builtin_type
                if isinstance(type_expression, ast.Attribute):
                    return getattr(
                        evaluate_type_spec(type_expression.value),
                        type_expression.attr,
                    )
                return evaluate_reduced_control_expression(type_expression)

            predicate = (
                isinstance
                if expression.func.id == "isinstance"
                else issubclass
            )
            subject = evaluate_reduced_control_expression(expression.args[0])
            type_spec = evaluate_type_spec(expression.args[1])
            result = predicate(subject, type_spec)
            if discovery_session is not None:
                history = discovery_session.setdefault(
                    "structural_predicate_history", []
                )
                history.append((
                    expression.func.id,
                    _diagnostic_value_summary(subject),
                    repr(type_spec),
                    bool(result),
                ))
                del history[:-32]
            return result
        if (
            isinstance(expression, ast.Compare)
            and len(expression.ops) == 1
            and len(expression.comparators) == 1
            and isinstance(expression.ops[0], (ast.Is, ast.IsNot))
        ):
            def identity_operand(operand: ast.AST) -> Any:
                # Identity tests are coordinator/source semantics all the way
                # through an attribute path.  Do not let a retained Attribute
                # node substitute an equal normalized value: SymPy's ``One``
                # and ``true`` are especially sensitive to singleton identity.
                if isinstance(operand, ast.Attribute):
                    return getattr(
                        identity_operand(operand.value), operand.attr
                    )
                return evaluate_reduced_control_expression(operand)

            left = identity_operand(expression.left)
            right = identity_operand(expression.comparators[0])
            result = left is right
            return (
                not result
                if isinstance(expression.ops[0], ast.IsNot)
                else result
            )
        if isinstance(expression, ast.Compare):
            operands = [
                evaluate_reduced_control_expression(expression.left),
                *(
                    evaluate_reduced_control_expression(comparator)
                    for comparator in expression.comparators
                ),
            ]
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
                    ast.In: lambda left, right: left in right,
                    ast.NotIn: lambda left, right: left not in right,
                }.get(type(comparison))
                if comparator is None:
                    break
                if not bool(comparator(operands[index], operands[index + 1])):
                    return False
            else:
                return True
        if isinstance(expression, ast.Subscript):
            base = evaluate_reduced_control_expression(expression.value)
            slice_expression = expression.slice
            if isinstance(slice_expression, ast.Slice):
                index = slice(
                    evaluate_reduced_control_expression(
                        slice_expression.lower
                    ) if slice_expression.lower is not None else None,
                    evaluate_reduced_control_expression(
                        slice_expression.upper
                    ) if slice_expression.upper is not None else None,
                    evaluate_reduced_control_expression(
                        slice_expression.step
                    ) if slice_expression.step is not None else None,
                )
            elif isinstance(slice_expression, ast.Tuple):
                index = tuple(
                    evaluate_reduced_control_expression(item)
                    for item in slice_expression.elts
                )
            else:
                index = evaluate_reduced_control_expression(
                    slice_expression
                )
            if isinstance(base, (dict, list, tuple, set)):
                index = coordinator_index(index)
            return base[index]
        if (
            isinstance(expression, ast.Attribute)
            and has_live_source_root(expression)
        ):
            return getattr(
                evaluate_reduced_control_expression(expression.value),
                expression.attr,
            )
        if isinstance(expression, (ast.Tuple, ast.List, ast.Set)):
            items = [
                evaluate_reduced_control_expression(item)
                for item in expression.elts
            ]
            if isinstance(expression, ast.Tuple):
                return tuple(items)
            if isinstance(expression, ast.Set):
                return set(items)
            return items
        if isinstance(expression, ast.Dict):
            result = {}
            for key, value in zip(expression.keys, expression.values):
                item = evaluate_reduced_control_expression(value)
                if key is None:
                    result.update(dict(item))
                else:
                    result[
                        evaluate_reduced_control_expression(key)
                    ] = item
            return result
        retained_node = expression_nodes.get(id(expression))
        if retained_node is not None:
            # Reduction can classify a value as graph-inert when its only
            # consumer is a source assignment/control expression. Source
            # execution is itself a real consumer: once it demands the AST
            # expression, suppressing the node would replace the value with
            # None and violate Python branch semantics.
            inert_nodes.discard(retained_node)
            return evaluate_node(retained_node)
        if isinstance(expression, ast.Call):
            # A structural method call can disappear from the numerical
            # graph while its result is still consumed by source control.
            # Resolve it against the live lexical bindings (not only the
            # restricted static-binding table) and preserve Python's
            # starred argument/keyword semantics.  State snapshots such as
            # ``saved = state.copy_shallow()`` in the dt controller are the
            # canonical case.
            function = evaluate_reduced_control_expression(expression.func)
            positional = []
            for argument in expression.args:
                if isinstance(argument, ast.Starred):
                    positional.extend(
                        evaluate_reduced_control_expression(argument.value)
                    )
                else:
                    positional.append(
                        evaluate_reduced_control_expression(argument)
                    )
            keywords = {}
            for keyword in expression.keywords:
                value = evaluate_reduced_control_expression(keyword.value)
                if keyword.arg is None:
                    keywords.update(value)
                else:
                    keywords[keyword.arg] = value
            return function(*positional, **keywords)
        if isinstance(expression, ast.Constant):
            return expression.value
        if isinstance(expression, ast.Attribute):
            return getattr(
                evaluate_reduced_control_expression(expression.value),
                expression.attr,
            )
        return evaluate_static_expression(expression)

    def execute_generator_statements(statements):
        source_value_unavailable = object()

        def publish_nonlocal(name: str, value: Any) -> None:
            if discovery_session is None:
                return
            for lexical_frame in reversed(
                discovery_session.get("lexical_frames", ())
            ):
                if lexical_frame.get("shell") is shell:
                    continue
                if lexical_frame["assign"](name, value):
                    return

        def assign_source_target(target: ast.AST, value: Any) -> None:
            if isinstance(target, ast.Name):
                source_binding_values[target.id] = value
                if discovery_session is not None:
                    history = discovery_session.setdefault(
                        "source_assignment_history", []
                    )
                    history.append((
                        shell.profile_path,
                        target.id,
                        _diagnostic_value_summary(value),
                    ))
                    del history[:-64]
                if target.id in nonlocal_names:
                    publish_nonlocal(target.id, value)
                return
            if isinstance(target, (ast.Tuple, ast.List)):
                unpacked = tuple(value)
                starred = tuple(
                    index
                    for index, item in enumerate(target.elts)
                    if isinstance(item, ast.Starred)
                )
                if len(starred) > 1:
                    raise ValueError(
                        "multiple starred assignment targets are invalid"
                    )
                if not starred:
                    if len(unpacked) != len(target.elts):
                        raise ValueError(
                            "assignment target/value arity mismatch"
                        )
                    assigned = tuple(zip(target.elts, unpacked))
                else:
                    star_index = starred[0]
                    trailing = len(target.elts) - star_index - 1
                    minimum = len(target.elts) - 1
                    if len(unpacked) < minimum:
                        raise ValueError(
                            "not enough values to unpack assignment"
                        )
                    assigned = (
                        tuple(zip(target.elts[:star_index], unpacked[:star_index]))
                        + ((
                            target.elts[star_index].value,
                            list(
                                unpacked[
                                    star_index:
                                    len(unpacked) - trailing
                                    if trailing
                                    else len(unpacked)
                                ]
                            ),
                        ),)
                        + tuple(zip(
                            target.elts[star_index + 1 :],
                            unpacked[len(unpacked) - trailing :]
                            if trailing
                            else (),
                        ))
                    )
                for item_target, item_value in assigned:
                    assign_source_target(item_target, item_value)
                return

        for statement in statements:
            if isinstance(statement, ast.Break):
                # Source control remains authoritative even when reduction
                # removes the marker node because it has no numerical value.
                raise _LoopBreakSignal()
            if isinstance(statement, ast.Continue):
                raise _LoopContinueSignal()
            if isinstance(statement, ast.Expr) and isinstance(
                statement.value,
                (ast.Yield, ast.YieldFrom),
            ):
                yielded = statement.value
                value_node = (
                    expression_nodes.get(id(yielded.value))
                    if yielded.value is not None
                    else None
                )
                if yielded.value is not None and value_node is None:
                    yield_node = expression_nodes.get(id(yielded))
                    if yield_node is not None:
                        value_node = next(
                            (
                                parent
                                for parent, role in (
                                    graph.G.nodes[yield_node].get(
                                        "parents",
                                        (),
                                    )
                                )
                                if str(role) in {"value", "operand"}
                            ),
                            None,
                        )
                if yielded.value is not None and value_node is None:
                    raise RuntimeError(
                        "generator yield value was not retained in "
                        f"{graph.G.graph.get('function_name', '?')}: "
                        f"{ast.dump(yielded, include_attributes=False)}"
                    )
                value = (
                    evaluate_node(value_node)
                    if value_node is not None
                    else None
                )
                if isinstance(yielded, ast.YieldFrom):
                    yield from value
                else:
                    yield value
                continue
            if isinstance(statement, ast.If):
                try:
                    test = evaluate_reduced_control_expression(
                        statement.test
                    )
                except (KeyError, TypeError) as error:
                    raise RuntimeError(
                        "generator branch test was not retained in "
                        f"{graph.G.graph.get('function_name', '?')}: "
                        f"{ast.dump(statement.test, include_attributes=False)}"
                    ) from error
                if discovery_session is not None:
                    history = discovery_session.setdefault(
                        "source_branch_history", []
                    )
                    history.append((
                        shell.profile_path,
                        ast.dump(statement.test, include_attributes=False),
                        _diagnostic_value_summary(test),
                    ))
                    del history[:-512]
                shell._captured_source_branches.append((
                    ast.dump(statement.test, include_attributes=False),
                    _diagnostic_value_summary(test),
                ))
                del shell._captured_source_branches[:-64]
                if shell._profiler.verbose:
                    shell._profiler.trace(
                        path=shell.profile_path,
                        section="generator-branch",
                        label=ast.dump(
                            statement.test,
                            include_attributes=False,
                        ),
                        fields={
                            "value": _diagnostic_value_summary(test),
                            "selected": (
                                "body" if bool(test) else "orelse"
                            ),
                        },
                    )
                branch = statement.body if bool(test) else statement.orelse
                yield from execute_generator_statements(branch)
                continue
            if isinstance(statement, ast.With):
                # ``with`` is source-ordered Python control, including for
                # context managers whose enter/exit effects do not produce a
                # numerical graph value (locks are a common example).  Let
                # ExitStack implement Python's ordered entry, reverse exit,
                # and exception-suppression protocol while the coordinator
                # remains responsible for evaluating expressions and binding
                # optional ``as`` targets in the active lexical frame.
                with ExitStack() as context_stack:
                    for item in statement.items:
                        context_manager = (
                            evaluate_reduced_control_expression(
                                item.context_expr
                            )
                        )
                        entered_value = context_stack.enter_context(
                            context_manager
                        )
                        if item.optional_vars is not None:
                            assign_source_target(
                                item.optional_vars, entered_value
                            )
                    yield from execute_generator_statements(statement.body)
                continue
            if isinstance(statement, ast.Try):
                try:
                    try:
                        yield from execute_generator_statements(statement.body)
                    except (
                        _SourceReturnSignal,
                        _LoopBreakSignal,
                        _LoopContinueSignal,
                    ):
                        raise
                    except BaseException as error:
                        selected_handler = None
                        for handler in statement.handlers:
                            if handler.type is None:
                                selected_handler = handler
                                break
                            exception_type = (
                                evaluate_reduced_control_expression(
                                    handler.type
                                )
                            )
                            if isinstance(error, exception_type):
                                selected_handler = handler
                                break
                        if selected_handler is None:
                            raise
                        exception_name = selected_handler.name
                        if exception_name is not None:
                            source_binding_values[exception_name] = error
                        try:
                            yield from execute_generator_statements(
                                selected_handler.body
                            )
                        finally:
                            # CPython clears an ``except ... as name`` target
                            # when the handler suite exits.
                            if exception_name is not None:
                                source_binding_values.pop(exception_name, None)
                    else:
                        yield from execute_generator_statements(
                            statement.orelse
                        )
                finally:
                    yield from execute_generator_statements(
                        statement.finalbody
                    )
                continue
            if isinstance(statement, ast.For):
                loop_node = expression_nodes.get(id(statement))
                if loop_node is None:
                    # A statically bounded loop may already have evaporated
                    # into cloned straight-line graph nodes.  Those nodes are
                    # evaluated by the ordinary dependency sweep; replaying
                    # the removed source loop would duplicate the body.
                    continue
                plan = loop_plans_by_node[loop_node]
                loop = plan.loop
                target_initials = (
                    (graph.G.nodes[loop_node].get("attributes") or {})
                    .get("loop_target_initials") or {}
                )
                for name, binding in loop.target_bindings:
                    initial = target_initials.get(name)
                    if initial is not None:
                        values[binding] = evaluate_node(int(initial))
                if loop.iterator_kind == "arithmetic_sequence":
                    by_role = {
                        str(role): parent
                        for parent, role in (
                            graph.G.nodes[loop_node].get("parents") or ()
                        )
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
                loop_broken = False
                # Source execution is now the authoritative owner of this
                # retained loop.  Publish that fact before entering the body:
                # a LoopExit demanded by a body expression must not recursively
                # replay the generic flat-loop evaluator with an empty target
                # frame while the source loop is already active.
                values[loop_node] = None
                iteration_count = 0
                for item in iterator:
                    iteration_count += 1
                    items_by_name = dict(_destructure_loop_target(
                        statement.target, item
                    ))
                    # Numerical reduction is allowed to omit loop targets
                    # that never enter a tensor region, but Python source
                    # control may still read any destructured target.  Keep
                    # the complete source binding environment in step with
                    # the iterator rather than treating target_bindings as a
                    # complete account of Python assignment semantics.
                    source_binding_values.update(items_by_name)
                    assignments = tuple(
                        (
                            (name, binding),
                            items_by_name[str(name)],
                        )
                        for name, binding in loop.target_bindings
                    )
                    for dependent in loop_invalidated_nodes[loop_node]:
                        values.pop(dependent, None)
                    for (_name, binding), bound_value in assignments:
                        generator_binding_values[binding] = bound_value
                    values.update(generator_binding_values)
                    for region_index in loop_runtime_region_indices.get(
                        loop_node,
                        (),
                    ):
                        completed_regions.discard(region_index)
                    try:
                        yield from execute_generator_statements(statement.body)
                    except _LoopBreakSignal:
                        loop_broken = True
                        break
                    except _LoopContinueSignal:
                        loop_continued = True
                    else:
                        loop_continued = False
                    for carried_name, initial, updated in loop.carried_bindings:
                        if (
                            loop_continued
                            and carried_name in source_binding_values
                        ):
                            values[initial] = source_binding_values[
                                carried_name
                            ]
                        elif not loop_continued:
                            values[initial] = evaluate_node(updated)
                # The generator interpreter is the planner-owned execution of
                # this loop.  Publish completion so a downstream LoopExit
                # cannot replay the same body through the generic flat loop
                # evaluator (which would also lose nested branch semantics).
                values[loop_node] = None
                shell._captured_loop_iterations[int(loop_node)] = (
                    int(iteration_count)
                )
                if not loop_broken:
                    yield from execute_generator_statements(statement.orelse)
                continue
            if isinstance(statement, ast.While):
                loop_node = expression_nodes.get(id(statement))
                if loop_node is None:
                    continue
                source_loop_nodes = frozenset(
                    int(member_node)
                    for member in ast.walk(statement)
                    if (
                        member_node := expression_nodes.get(id(member))
                    ) is not None
                    and int(member_node) != int(loop_node)
                )
                iterations = 0
                loop_broken = False
                plan = loop_plans_by_node[loop_node]
                if plan.strategy is LoopStrategy.NATIVE_SOURCE:
                    # Discovery traces the body of a retained native loop; it
                    # must not execute the runtime trip count.  This is vital
                    # for event/render loops whose configured bound is
                    # intentionally infinite.  The semantic loop/control IR
                    # owns repetition after capture.
                    for dependent in (
                        *loop_invalidated_nodes[loop_node],
                        *source_loop_nodes,
                    ):
                        values.pop(dependent, None)
                    for region_index in loop_runtime_region_indices.get(
                        loop_node, ()
                    ):
                        completed_regions.discard(region_index)
                    test = evaluate_reduced_control_expression(statement.test)
                    if bool(test):
                        try:
                            yield from execute_generator_statements(
                                statement.body
                            )
                        except (_LoopBreakSignal, _LoopContinueSignal):
                            pass
                    values[loop_node] = None
                    # The runtime loop, not discovery, decides whether it
                    # exhausts normally and therefore enters ``else``.
                    continue
                while True:
                    for dependent in (
                        *loop_invalidated_nodes[loop_node],
                        *source_loop_nodes,
                    ):
                        values.pop(dependent, None)
                    for region_index in loop_runtime_region_indices.get(
                        loop_node,
                        (),
                    ):
                        completed_regions.discard(region_index)
                    test = evaluate_reduced_control_expression(
                        statement.test
                    )
                    if not bool(test):
                        break
                    try:
                        yield from execute_generator_statements(statement.body)
                    except _LoopBreakSignal:
                        loop_broken = True
                        break
                    except _LoopContinueSignal:
                        pass
                    iterations += 1
                    if iterations > 1_000_000:
                        raise RuntimeError(
                            "ProcessGraph generator while loop exceeded "
                            "safety limit"
                        )
                values[loop_node] = None
                if not loop_broken:
                    yield from execute_generator_statements(statement.orelse)
                continue
            if isinstance(statement, ast.Return):
                source_return = object()
                value = source_return
                if isinstance(statement.value, ast.Name):
                    if statement.value.id in source_binding_values:
                        value = source_binding_values[statement.value.id]
                    elif statement.value.id in supplied:
                        value = supplied[statement.value.id]
                elif (
                    statement.value is not None
                    and has_live_source_root(statement.value)
                ):
                    value = evaluate_reduced_control_expression(
                        statement.value
                    )
                # The executing Return statement and its value expression are
                # the exact source-order authority.  Normalized/checkpointed
                # ``return_value_nodes`` can collapse multiple returns onto a
                # function's syntactically last output identity; consulting
                # it first made an executed early tensor return appear to be
                # the later ``float(value.item())`` branch.  Prefer the live
                # expression node and retain both metadata tables solely as
                # compatibility fallbacks when AST identity was serialized.
                value_node = (
                    source_binding_node_ids.get(statement.value.id)
                    if isinstance(statement.value, ast.Name)
                    else expression_nodes.get(id(statement.value))
                    if statement.value is not None
                    else None
                )
                if value_node is None and statement.value is not None:
                    value_node = graph.G.graph.get(
                        "return_value_nodes", {}
                    ).get(id(statement))
                if value_node is None and statement.value is not None:
                    value_node = checkpoint_return_value_nodes.get(
                        id(statement)
                    )
                if value_node is not None and value_node not in graph.G:
                    output_names = tuple(
                        graph.G.graph.get("function_outputs", ())
                    )
                    identities = graph.G.graph.get("identity_table", {})
                    candidates = tuple(
                        int(identities[name][-1])
                        for name in output_names
                        if identities.get(name)
                        and int(identities[name][-1]) in graph.G
                    )
                    value_node = (
                        candidates[0]
                        if len(candidates) == 1
                        else graph.roots[0]
                        if len(graph.roots) == 1
                        else None
                    )
                if capture and value_node is not None:
                    shell._captured_return_value_ids.add(int(value_node))
                if value is source_return:
                    value = (
                        values[value_node]
                        if value_node in values
                        else evaluate_node(value_node)
                        if value_node is not None
                        else None
                    )
                if capture:
                    shell.forward_feed_ids = tuple(feed_maps)
                    shell.forward_aggregate_feed_paths = tuple(
                        aggregate_feed_maps
                    )
                    shell.forward_subgraphs = tuple(captured_subgraphs)
                    shell.forward_compilers = tuple(captured_compilers)
                    shell.forward_region_indices = tuple(
                        captured_region_indices
                    )
                    shell.forward_output_values = tuple(
                        captured_output_values
                    )
                    shell.forward_region_capture_node_ids = tuple(
                        captured_region_node_ids
                    )
                    shell.forward_region_planned_capture_ids = tuple(
                        captured_region_planned_ids
                    )
                    shell.forward_region_planned_input_ids = tuple(
                        captured_region_planned_input_ids
                    )
                shell.captured_values = values
                shell.last_result = value
                raise _SourceReturnSignal(value)
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                value_expression = statement.value
                assigned_value = None
                if value_expression is not None:
                    value_node = expression_nodes.get(id(value_expression))
                    # A bare source name denotes the value visible at this
                    # statement, not necessarily the last normalized SSA
                    # identity bearing that spelling.  The distinction is
                    # observable whenever a value is saved before a later
                    # write, for example ``saved = counter; counter += 1``.
                    # Consult the active lexical/source frame first; retained
                    # compound expressions remain owned by the planned graph.
                    if isinstance(
                        value_expression,
                        (ast.Name, ast.IfExp, ast.BoolOp),
                    ):
                        assigned_value = evaluate_reduced_control_expression(
                            value_expression
                        )
                    elif value_node is not None:
                        if is_shader_internal_node(value_node):
                            evaluate_region(region_for_node[value_node])
                            assigned_value = values.get(
                                value_node,
                                source_value_unavailable,
                            )
                        else:
                            assigned_value = evaluate_node(value_node)
                    else:
                        assigned_value = evaluate_reduced_control_expression(
                            value_expression
                        )
                targets = (
                    tuple(statement.targets)
                    if isinstance(statement, ast.Assign)
                    else (statement.target,)
                )
                for target in targets:
                    if (
                        assigned_value is not source_value_unavailable
                        and isinstance(target, (ast.Name, ast.Tuple, ast.List))
                    ):
                        assign_source_target(target, assigned_value)
                    target_node = expression_nodes.get(id(target))
                    if (
                        target_node is not None
                        and target_node in graph.G
                        and graph.G.nodes[target_node].get("type")
                        in {"SetAttr", "IndexedStore"}
                    ):
                        if is_shader_internal_node(target_node):
                            evaluate_region(region_for_node[target_node])
                            continue
                        # The source statement has already evaluated its RHS
                        # at this exact program point.  A normalized store can
                        # still refer to the final SSA identity with the same
                        # spelling (for example ``result`` assigned in many
                        # arms of a long if/elif chain).  Publish the live RHS
                        # into the store's value port so structural mutation
                        # observes Python's source-order value, not a later
                        # predicate or branch identity.
                        for parent, role in graph.G.nodes[target_node].get(
                            "parents", ()
                        ):
                            if (
                                str(role) == "value"
                                and assigned_value
                                is not source_value_unavailable
                            ):
                                values[int(parent)] = assigned_value
                                inert_nodes.discard(int(parent))
                        evaluate_node(target_node)
                continue
            if isinstance(statement, ast.AugAssign):
                statement_node = expression_nodes.get(id(statement))
                if (
                    statement_node is not None
                    and is_shader_internal_node(statement_node)
                ):
                    evaluate_region(region_for_node[statement_node])
                    if isinstance(statement.target, ast.Name):
                        source_binding_values.pop(
                            statement.target.id,
                            None,
                        )
                    continue
                assigned_value = None
                scalar_source_update = False
                if (
                    isinstance(statement.target, ast.Name)
                    and statement.target.id in source_binding_values
                    and not isinstance(
                        source_binding_values[statement.target.id],
                        AbstractTensor,
                    )
                ):
                    try:
                        right_value = evaluate_reduced_control_expression(
                            statement.value
                        )
                    except (KeyError, TypeError):
                        right_value = None
                    if not isinstance(right_value, AbstractTensor):
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
                        }.get(type(statement.op))
                        if binary is not None:
                            assigned_value = binary(
                                source_binding_values[statement.target.id],
                                right_value,
                            )
                            scalar_source_update = True
                if statement_node is not None:
                    values.pop(statement_node, None)
                    statement_region = region_for_node.get(statement_node)
                    if statement_region is not None:
                        completed_regions.discard(statement_region)
                if not scalar_source_update:
                    assigned_value = (
                        evaluate_node(statement_node)
                        if statement_node is not None
                        else evaluate_reduced_control_expression(statement)
                    )
                if isinstance(statement.target, ast.Name):
                    source_binding_values[
                        statement.target.id
                    ] = assigned_value
                    if statement.target.id in nonlocal_names:
                        publish_nonlocal(
                            statement.target.id, assigned_value
                        )
                continue
            statement_node = expression_nodes.get(id(statement))
            if statement_node is not None and statement_node in graph.G:
                if is_shader_internal_node(statement_node):
                    evaluate_region(region_for_node[statement_node])
                else:
                    evaluate_node(statement_node)
                continue
            for member in ast.iter_child_nodes(statement):
                member_node = expression_nodes.get(id(member))
                if member_node is not None and member_node in graph.G:
                    if is_shader_internal_node(member_node):
                        evaluate_region(region_for_node[member_node])
                    else:
                        evaluate_node(member_node)

    if graph.G.graph.get("generator_stream"):
        for input_id in tuple(values):
            previous = values[input_id]
            values[input_id] = _tensorize_graph_input(
                previous,
                device=device,
            )
            # Tensorising makes a new object; carry the name across or it is
            # lost exactly when it starts to matter.
            named = getattr(shell, "_capture_input_names", None)
            if named is not None and id(previous) in named:
                named[id(values[input_id])] = named[id(previous)]

        def generator_producer():
            try:
                with AbstractTensor.use_backend(
                    _scheduled_capture_backend.get(),
                    device,
                ):
                    yield from execute_generator_statements(
                        graph.G.graph.get("function_body", ())
                    )
            finally:
                if capture:
                    shell.forward_feed_ids = tuple(feed_maps)
                    shell.forward_aggregate_feed_paths = tuple(
                        aggregate_feed_maps
                    )
                    shell.forward_subgraphs = tuple(captured_subgraphs)
                    shell.forward_compilers = tuple(captured_compilers)
                    shell.forward_region_indices = tuple(
                        captured_region_indices
                    )
                    shell.forward_output_values = tuple(
                        captured_output_values
                    )
                    shell.forward_region_capture_node_ids = tuple(
                        captured_region_node_ids
                    )
                    shell.forward_region_planned_capture_ids = tuple(
                        captured_region_planned_ids
                    )
                    shell.forward_region_planned_input_ids = tuple(
                        captured_region_planned_input_ids
                    )
                shell.captured_values = values

        shell.last_result = _GreedyGeneratorStream(
            generator_producer(),
            fifo_slots=2,
        )
        return shell.last_result

    if discovery_session is not None:
        def resolve_lexical_name(name: str) -> tuple[bool, Any]:
            if name in source_binding_values:
                return True, source_binding_values[name]
            if name in supplied:
                return True, supplied[name]
            for identity in reversed(
                graph.G.graph.get("identity_table", {}).get(name, ())
            ):
                try:
                    return True, evaluate_node(identity)
                except (KeyError, RuntimeError):
                    continue
            return False, None

        def assign_lexical_name(name: str, value: Any) -> bool:
            if (
                name not in source_binding_values
                and name not in supplied
                and name not in graph.G.graph.get("identity_table", {})
            ):
                return False
            source_binding_values[name] = value
            return True

        discovery_session.setdefault("lexical_frames", []).append({
            "shell": shell,
            "resolve": resolve_lexical_name,
            "assign": assign_lexical_name,
        })

    with AbstractTensor.use_backend(
        _scheduled_capture_backend.get(),
        device,
    ):
        # Public array inputs are tensor values even when their first consumer
        # is structural control (for example a defensive ``isinstance``
        # guard).  Tensorise under the scheduled backend before control
        # ownership removes those inputs from the ordinary sweep.
        for input_id, previous in tuple(values.items()):
            resident = _tensorize_graph_input(previous, device=device)
            if resident is previous:
                continue
            values[input_id] = resident
            named = getattr(shell, "_capture_input_names", None)
            if named is not None and (carried := named.get(id(previous))):
                named[id(resident)] = carried
                storage = _capture_storage_identity(resident)
                if storage is not None:
                    shell._capture_input_storage[storage] = carried

        # Runtime control operands are outputs of the compiled program just as
        # surely as returned tensors are.  A predicate used only by
        # ``if tensor_scalar: raise`` may otherwise remain hidden beneath the
        # planner-owned statement branch: ordinary topological demand never
        # reaches it, then control-source composition discovers the predicate
        # ID after numerical capture has already ended.
        #
        # Demand every retained validation predicate here, while the source
        # function's one discovery tape is active.  This does not execute the
        # validation on the host, replay a region, or create a second tape. It
        # merely makes the predicate-producing region a terminal of the same
        # program observation so its resident slot can be consumed by the
        # compiled shell's ValidationBlock.
        if capture:
            for validation in _validation_control_blocks(shell):
                evaluate_node(int(validation.predicate_value_id))

        # Structural Python control is ordered by its retained source body,
        # not by numerical dependency levels.  In particular, a later branch
        # predicate must not run after an earlier branch returned.  Statement
        # execution still demands each planned numerical region through
        # evaluate_node; it does not replace or reinterpret numeric lowering.
        if function_body:
            for _yielded in execute_generator_statements(function_body):
                raise RuntimeError(
                    "non-generator ProcessGraph produced a yielded value"
                )

        for node_id in _dependency_order(graph):
            if node_id in inert_nodes:
                continue
            if node_id in controlled_nodes:
                # The planner owns statement loops and comprehension loops.
                # A statement loop with no controlling loop ancestor may
                # execute here; downstream loops defer until their owner,
                # LoopExit, returned value, or graph root demands them.
                if (
                    isinstance(
                        graph.G.nodes[node_id].get("expr_obj"),
                        (ast.For, ast.While),
                    )
                    and node_id not in downstream_loop_nodes
                    and not has_comprehension_owner
                ):
                    evaluate_node(node_id)
                continue
            if node_id in values:
                try:
                    previous = values[node_id]
                    values[node_id] = _tensorize_graph_input(
                        previous,
                        device=device,
                    )
                    # This is where a supplied array becomes the tensor the
                    # tape sees, so it is the identity a feed will carry.
                    # Recording the name only at binding time meant a
                    # parameter used directly by its own function -- rather
                    # than passed on to a callee, whose shell records it
                    # again -- lost its name here and could only fall back to
                    # a positional one.
                    named = getattr(shell, "_capture_input_names", None)
                    if named is not None:
                        carried = named.get(id(previous))
                        if carried:
                            named[id(values[node_id])] = carried
                            storage = _capture_storage_identity(values[node_id])
                            if storage is not None:
                                shell._capture_input_storage[storage] = carried
                except Exception as error:
                    value = values[node_id]
                    node = graph.G.nodes[node_id]
                    value_dtype = getattr(value, "dtype", None)
                    value_shape = getattr(value, "shape", None)
                    raise TypeError(
                        "failed to tensorize a scheduled graph input: "
                        f"shell={shell.profile_path!r}, node={node_id!r}, "
                        f"label={node.get('label')!r}, "
                        f"op={node.get('op')!r}, "
                        f"value_type={type(value).__module__}."
                        f"{type(value).__qualname__}, "
                        f"shape={value_shape!r}, dtype={value_dtype!r}, "
                        f"value={value!r}"
                    ) from error
            elif is_shader_internal_node(node_id):
                # Topological traversal schedules its containing region, but
                # does not ask the region to materialize this shader-local
                # intermediate as a coordinator-visible value.
                evaluate_region(region_for_node[node_id])
            else:
                evaluate_node(node_id)

    roots = tuple(evaluate_node(node_id) for node_id in graph.roots)
    if capture:
        shell.forward_feed_ids = tuple(feed_maps)
        shell.forward_aggregate_feed_paths = tuple(aggregate_feed_maps)
        shell.forward_subgraphs = tuple(captured_subgraphs)
        shell.forward_compilers = tuple(captured_compilers)
        shell.forward_region_indices = tuple(captured_region_indices)
        shell.forward_output_values = tuple(captured_output_values)
        shell.forward_region_capture_node_ids = tuple(
            captured_region_node_ids
        )
        shell.forward_region_planned_capture_ids = tuple(
            captured_region_planned_ids
        )
        shell.forward_region_planned_input_ids = tuple(
            captured_region_planned_input_ids
        )
    shell.captured_values = values
    shell.last_result = roots[0] if len(roots) == 1 else roots
    return shell.last_result


def _coordinate_scheduled_capture(
    shell: Any,
    initial_values: dict[str | int, Any],
    *,
    device: Any = None,
    capture: bool = True,
    discovery_session: dict[str, Any] | None = None,
) -> Any:
    """Profile-aware boundary around one scheduled shell invocation."""

    if capture:
        # Discovery/compilation diagnostics remain available through verbose
        # trace and the error buffer, but they are not runtime frames.  Mixing
        # their eager region dispatches into steady-state shell statistics made
        # one execution report as two invocations and obscured CPU churn.
        shell._profiler._runtime_suppression += 1
    token = shell._profiler.begin_shell(shell.profile_path)
    try:
        return _coordinate_scheduled_capture_impl(
            shell,
            initial_values,
            device=device,
            capture=capture,
            discovery_session=discovery_session,
        )
    except _SourceReturnSignal as signal:
        shell.last_result = signal.value
        return signal.value
    except Exception as error:
        if hasattr(error, "add_note"):
            structural_state = tuple(
                (
                    str(name),
                    {
                        str(field): _diagnostic_value_summary(field_value)
                        for field, field_value in list(value.state.items())[:24]
                    },
                )
                for name, value in initial_values.items()
                if isinstance(value, _CompiledStructuralObject)
            )
            active_session = discovery_session or getattr(
                shell, "_discovery_session", None
            )
            call_returns = tuple(
                (active_session or {}).get("call_return_history", ())
            )
            structural_predicates = tuple(
                (active_session or {}).get(
                    "structural_predicate_history", ()
                )
            )
            source_assignments = tuple(
                (active_session or {}).get(
                    "source_assignment_history", ()
                )
            )
            source_branches = tuple(
                (active_session or {}).get("source_branch_history", ())
            )
            error.add_note(
                "while coordinating ProcessGraph shell "
                f"{shell.profile_path!r} with inputs="
                f"{tuple((str(name), _diagnostic_value_summary(value)) for name, value in initial_values.items())!r}; "
                f"structural_state={structural_state!r}; "
                f"recent_call_returns={call_returns!r}; "
                f"recent_structural_predicates={structural_predicates!r}; "
                f"recent_source_assignments={source_assignments!r}; "
                f"recent_source_branches={source_branches!r}"
            )
        shell._profiler.record_exception(
            error,
            path=shell.profile_path,
            phase="execution",
        )
        raise
    finally:
        active_session = discovery_session or getattr(
            shell, "_discovery_session", None
        )
        if active_session is not None:
            frames = active_session.get("lexical_frames", [])
            for index in range(len(frames) - 1, -1, -1):
                if frames[index].get("shell") is shell:
                    del frames[index]
                    break
        shell._profiler.end_shell(shell.profile_path, token)
        if capture:
            shell._profiler._runtime_suppression -= 1


def _compile_whole_process_graph(
    shell: Any,
    *,
    device: Any = None,
    prepare_ephemerals: bool = False,
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

    if (
        getattr(shell, "runtime_closure_only", False)
        and getattr(shell, "_owns_function_shells", False)
    ):
        # The outer shell owns the complete source catalogue and map records;
        # it is not a second execution of the module containing the requested
        # entrypoint.  Compile the proven activation roots.  Their complete
        # call topology is represented by callsite_function_shells and is
        # recursively validated by the ordinary path below.
        activation_roots = tuple(
            getattr(shell, "activation_root_references", ())
        )
        if not activation_roots:
            raise RuntimeError(
                "runtime-root ProcessGraph has no activation root"
            )
        for reference in activation_roots:
            function_shell = shell.function_shells.get(int(reference))
            if function_shell is None:
                raise RuntimeError(
                    "ProcessGraph activation root has no deployment shell: "
                    f"{reference}"
                )
            _compile_whole_process_graph(
                function_shell,
                device=device,
                prepare_ephemerals=prepare_ephemerals,
                _visited=visited,
            )
        shell.compiled_dispatch_functions = ()
        shell.compiled_dispatch_sources = ()
        shell.whole_program_compiled = True
        shell.whole_program_device = device
        return shell

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
    # An ephemeral is a GraphDeepCompiler-built PYTHON callable for a numeric
    # region -- only useful when a numeric region has to stay executable as
    # Python (runtime execution). Deriving the dual IR (the whole-program,
    # no-bake precompile) does not execute anything, so it does not need them,
    # and eagerly building them here forces the deep compiler over the whole
    # graph -- including control/binding constructs (a walrus, an annotation)
    # that are not tensor operators and have no op-table entry -- which is the
    # wrong question to ask of them. So prepare them only when explicitly asked.
    # The runtime consumer falls back to the ephemeral itself (prepared lazily
    # on first call) when compiled_dispatch_functions is empty, so leaving them
    # unprepared here is safe.
    if prepare_ephemerals:
        for ephemeral in shell.ephemeral_callables:
            ephemeral.prepare(device=device)
            functions.append(ephemeral)
            sources.append(ephemeral.generated_source)
    direct_callees = [
        (node_id, int(reference))
        for node_id, data in shell.process_graph.G.nodes(data=True)
        for reference in (
            (data.get("attributes") or {}).get("callee_ref")
            or (data.get("attributes") or {}).get("method_ref"),
        )
        if reference is not None
    ]
    for node_id, reference in direct_callees:
        function_shell = (
            getattr(shell, "callsite_function_shells", {}).get(node_id)
            or shell.function_shells.get(reference)
        )
        if function_shell is None:
            try:
                referenced_entry = shell.process_graph.function_table.entry(
                    reference
                )
            except (AttributeError, KeyError):
                referenced_entry = None
            if (
                referenced_entry is not None
                and referenced_entry.metadata.get("host_ssa_module") is not None
            ):
                # The callee is already fully represented as repository SSA;
                # it does not need or admit a source-planning shell.
                continue
            function_name = shell.process_graph.G.graph.get(
                "function_name", "?"
            )
            try:
                referenced_name = (
                    shell.process_graph.function_table.entry(reference)
                    .qualified_name
                )
            except (AttributeError, KeyError):
                referenced_name = "?"
            node_data = shell.process_graph.G.nodes[node_id]
            expression = node_data.get("expr_obj")
            raise RuntimeError(
                "ProcessGraph references a function with no deployment "
                f"shell: {reference}; owner={function_name!r}; "
                f"referenced={referenced_name!r}; node={node_id}; "
                f"type={node_data.get('type')!r}; "
                f"source={ast.dump(expression, include_attributes=False) if isinstance(expression, ast.AST) else repr(expression)}; "
                "planned_references="
                f"{tuple(sorted(map(int, shell.function_shells)))}"
            )
        _compile_whole_process_graph(
            function_shell,
            device=device,
            prepare_ephemerals=prepare_ephemerals,
            _visited=visited,
        )
    shell.compiled_dispatch_functions = tuple(functions)
    shell.compiled_dispatch_sources = tuple(sources)
    shell.whole_program_compiled = True
    shell.whole_program_device = device
    return shell


def _source_static_value(graph: Any, node_id: int, visiting=None) -> bool:
    """Whether a call argument is structural source data, not runtime data."""

    visiting = set() if visiting is None else set(visiting)
    node_id = int(node_id)
    if node_id in visiting or node_id not in graph.G:
        return False
    visiting.add(node_id)
    data = graph.G.nodes[node_id]
    if data.get("type") in {"Constant", "Const", "const", "StaticReference"}:
        return True
    if data.get("type") == "Input":
        name = (data.get("attributes") or {}).get("binding_name")
        return name in (graph.G.graph.get("planner_specializations") or {})
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
    candidates: dict[tuple[int, str], list[Any]] = {}
    dynamic_argument = object()
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
            _receiver, positional, _all = _method_parameter_layout(callee.G)
            bound_parameters: set[str] = set()
            for parent, role_value in data.get("parents") or ():
                role = str(role_value)
                position = _positional_argument_index(role)
                parameter = (
                    positional[position]
                    if (
                        position is not None
                        and position < len(positional)
                    )
                    else role[3:]
                    if role.startswith("kw:")
                    else None
                )
                if parameter is None:
                    continue
                parameter = str(parameter)
                bound_parameters.add(parameter)
                if _source_static_value(caller, int(parent)):
                    try:
                        value = _source_static_literal(caller, int(parent))
                    except ValueError:
                        value = dynamic_argument
                else:
                    value = dynamic_argument
                candidates.setdefault(
                    (int(reference), parameter), []
                ).append(value)
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
                    ).append(copy.deepcopy(default))
    for (reference, parameter), values in candidates.items():
        if not values or any(value is dynamic_argument for value in values):
            continue
        first = values[0]
        try:
            consistent = all(value == first for value in values[1:])
        except (TypeError, ValueError):
            consistent = False
        if not consistent:
            continue
        callee = function_table.entry(reference).graph
        if callee is not None:
            callee.G.graph.setdefault(
                "planner_specializations", {}
            )[parameter] = first


def _propagate_callsite_tensor_specializations(graph: Any) -> None:
    """Carry consistent tensor descriptors through the function table.

    This is the descriptor analogue of literal planner specialization.  It
    settles to a fixed point so a root call can shape ``bw_matmul`` and that
    shaped wrapper can in turn shape ``matmul_vjp`` before either function's
    logical control is partitioned. Conflicting callsite shapes remain
    parametric and are left for per-callsite specialization.
    """

    function_table = getattr(graph, "function_table", None)
    if function_table is None:
        return
    graphs = [graph]
    graphs.extend(
        entry.graph for entry in function_table if entry.graph is not None
    )
    changed = True
    while changed:
        changed = False
        candidates: dict[tuple[int, str], list[dict[str, Any]]] = {}
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
                _receiver, positional, _all = _method_parameter_layout(callee.G)
                for parent, role_value in data.get("parents") or ():
                    role = str(role_value)
                    position = _positional_argument_index(role)
                    parameter = (
                        positional[position]
                        if position is not None and position < len(positional)
                        else role[3:] if role.startswith("kw:") else None
                    )
                    descriptor = _tensor_descriptor(caller, int(parent))
                    if parameter is not None and descriptor is not None:
                        candidates.setdefault(
                            (int(reference), str(parameter)), []
                        ).append(copy.deepcopy(dict(descriptor)))
        by_reference: dict[int, dict[str, dict[str, Any]]] = {}
        for (reference, parameter), descriptors in candidates.items():
            if not descriptors or any(
                descriptor != descriptors[0] for descriptor in descriptors[1:]
            ):
                continue
            by_reference.setdefault(reference, {})[parameter] = descriptors[0]
        for reference, descriptors in by_reference.items():
            callee = function_table.entry(reference).graph
            if callee is None:
                continue
            existing = callee.G.graph.setdefault(
                "planner_tensor_descriptors", {}
            )
            additions = {
                name: descriptor
                for name, descriptor in descriptors.items()
                if existing.get(name) != descriptor
            }
            if not additions:
                continue
            existing.update(copy.deepcopy(additions))
            _apply_callsite_tensor_descriptors(callee, additions)
            # The catalogue graph is shared by every callsite.  A tensor
            # descriptor alone is not a complete structural specialization:
            # helpers such as ``unbroadcast(G, target_shape)`` also require a
            # callsite-local shape value before their source control can be
            # selected.  Folding the shared graph here permanently discarded
            # branches before that value was known.  The copied graph is
            # folded below by ``_callsite_specialized_shell_type`` after both
            # descriptor and literal arguments have been collected.
            changed = True


def _resolve_bound_function_references(graph: Any) -> None:
    """Turn a specialized callable parameter into an ordinary call edge.

    A first-class source function is represented by its opaque function-table
    address.  When that address crosses a function parameter (for example the
    dt system's ``advance`` callback), the call remains parametric, but this
    specialized card invocation can still link it to the selected callee.
    No Python callable is retained or executed to establish the edge.
    """

    specializations = graph.G.graph.get("planner_specializations") or {}
    for _node_id, data in graph.G.nodes(data=True):
        attributes = data.get("attributes") or {}
        if (
            attributes.get("callee_ref") is not None
            or attributes.get("method_ref") is not None
        ):
            continue
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
        ):
            continue
        bound = specializations.get(expression.func.id)
        if not isinstance(bound, FunctionReference):
            continue
        attributes = dict(attributes)
        attributes["callee_ref"] = int(bound.address)
        attributes["callee_resolution"] = "bound-function-parameter"
        data["attributes"] = attributes


def propagate_bound_planner_specializations(
    graph: Any,
    entrypoint: str,
    bindings: Mapping[str, Any],
    mutable_parameters: Iterable[str] = (),
) -> None:
    """Carry safe structural feed values through the linked call graph.

    This is planning metadata, not constant folding of numerical tensors.
    Only syntax that reads an already-supplied object without invoking
    arbitrary user code is resolved: names, public attributes, subscripts,
    literal containers, and standard structural views such as ``items``.
    """

    function_table = getattr(graph, "function_table", None)
    if function_table is None:
        return
    reference = function_table.reference(entrypoint)
    if reference is None:
        return

    def resolve(node: ast.AST, environment: Mapping[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id in environment:
            return environment[node.id]
        if isinstance(node, ast.Attribute) and not node.attr.startswith("_"):
            return getattr(resolve(node.value, environment), node.attr)
        if isinstance(node, ast.Subscript):
            return resolve(node.value, environment)[
                resolve(node.slice, environment)
            ]
        if isinstance(node, ast.Tuple):
            return tuple(resolve(item, environment) for item in node.elts)
        if isinstance(node, ast.List):
            return [resolve(item, environment) for item in node.elts]
        if isinstance(node, ast.Dict):
            return {
                resolve(key, environment): resolve(value, environment)
                for key, value in zip(node.keys, node.values)
            }
        if isinstance(node, ast.Call):
            args = [resolve(argument, environment) for argument in node.args]
            if isinstance(node.func, ast.Name) and node.func.id in {
                "tuple", "list", "range", "enumerate", "zip"
            }:
                return {
                    "tuple": tuple, "list": list, "range": range,
                    "enumerate": enumerate, "zip": zip,
                }[node.func.id](*args)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"items", "keys", "values"}
                and not node.keywords
                and not args
            ):
                owner = resolve(node.func.value, environment)
                if isinstance(owner, Mapping):
                    return getattr(owner, node.func.attr)()
        raise ValueError("expression is not a safe structural binding")

    mutable = frozenset(map(str, mutable_parameters))
    entry = function_table.entry(reference.address).graph
    if entry is None:
        return
    queue: list[tuple[Any, dict[str, Any]]] = [(
        entry,
        {
            str(name): value
            for name, value in bindings.items()
            if str(name) not in mutable
        },
    )]
    visited: set[tuple[int, tuple[str, ...]]] = set()
    while queue:
        current, environment = queue.pop(0)
        key = (id(current), tuple(sorted(environment)))
        if key in visited:
            continue
        visited.add(key)
        current.G.graph.setdefault("planner_specializations", {}).update(
            environment
        )
        for _node_id, data in current.G.nodes(data=True):
            attributes = data.get("attributes") or {}
            callee_reference = attributes.get(
                "callee_ref", attributes.get("method_ref")
            )
            expression = data.get("expr_obj")
            if callee_reference is None or not isinstance(expression, ast.Call):
                continue
            try:
                callee = function_table.entry(int(callee_reference)).graph
            except (KeyError, TypeError, ValueError):
                continue
            if callee is None:
                continue
            receiver, positional, _all = _method_parameter_layout(callee.G)
            resolved: dict[str, Any] = {}
            if receiver is not None and isinstance(expression.func, ast.Attribute):
                try:
                    resolved[receiver] = resolve(
                        expression.func.value, environment
                    )
                except (AttributeError, KeyError, TypeError, ValueError):
                    pass
            for position, argument in enumerate(expression.args):
                if position >= len(positional):
                    break
                try:
                    resolved[positional[position]] = resolve(
                        argument, environment
                    )
                except (AttributeError, KeyError, TypeError, ValueError):
                    continue
            for keyword in expression.keywords:
                if keyword.arg is None:
                    continue
                try:
                    resolved[str(keyword.arg)] = resolve(
                        keyword.value, environment
                    )
                except (AttributeError, KeyError, TypeError, ValueError):
                    continue
            if resolved:
                queue.append((callee, resolved))


_CALLSITE_SHELL_TYPE_CACHE: dict[tuple[object, ...], type] = {}
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


def _tensor_descriptor(graph: Any, node_id: int) -> dict[str, Any] | None:
    """Return compiler-owned tensor facts without inspecting a runtime value."""

    if int(node_id) not in graph.G:
        return None
    tensor = dict(graph.G.nodes[int(node_id)].get("tensor") or {})
    if "shape" not in tensor:
        return None
    return {
        "shape": tuple(tensor.get("shape") or ()),
        "dtype": str(tensor.get("dtype") or "float64"),
        **(
            {"device": tensor["device"]}
            if tensor.get("device") is not None else {}
        ),
    }


def _fold_callsite_structural_values(graph: Any) -> None:
    """Fold only whitelisted structural Python identities from graph facts.

    This pass is deliberately smaller than Python evaluation.  It consumes
    constants and compiler-owned tensor descriptors, and recognizes the
    structural builtins used to express shapes and source control.  It never
    invokes a retained Python callable or reads a runtime tensor payload.
    """

    unresolved = _UNRESOLVED_STRUCTURAL_VALUE
    known: dict[int, Any] = {}
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

    def descriptor_attribute(parent: int, attribute: str) -> Any:
        descriptor = _tensor_descriptor(graph, parent)
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

    def evaluate(node_id: int, data: Mapping[str, Any]) -> Any:
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
            defaults = graph.G.graph.get("parameter_defaults") or {}
            if str(binding_name) in defaults:
                default = defaults[str(binding_name)]

                def structural_default(value: Any) -> bool:
                    if value is None or isinstance(
                        value, (bool, int, float, str, bytes)
                    ):
                        return True
                    return isinstance(value, tuple) and all(
                        structural_default(item) for item in value
                    )

                if structural_default(default):
                    return copy.deepcopy(default)
            value_abi = (
                graph.G.graph.get("parameter_value_abi") or {}
            ).get(str(binding_name))
            if value_abi is not None:
                return _ProgramABIValueFact(
                    str(value_abi["python_type"]),
                    str(value_abi["storage"]),
                    value_abi.get("dtype"),
                )
            record_abi = (
                graph.G.graph.get("parameter_record_abi") or {}
            ).get(str(binding_name))
            if record_abi is not None:
                return _ProgramABIValueFact(
                    str(record_abi["identity"]), "record", None
                )
            return unresolved
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
                fixed = descriptor_attribute(parent, attribute)
                if fixed is not unresolved:
                    return fixed
                owner_fact = known.get(parent, unresolved)
                if (
                    isinstance(owner_fact, _ProgramABIValueFact)
                    and owner_fact.storage == "record"
                ):
                    matching_records = tuple(
                        record
                        for record in (
                            graph.G.graph.get("parameter_record_abi") or {}
                        ).values()
                        if str(record.get("identity") or "")
                        == owner_fact.python_type
                    )
                    if len(matching_records) == 1:
                        field = dict(
                            matching_records[0].get("fields") or {}
                        ).get(attribute)
                        if field is not None:
                            dtype = field.get("dtype")
                            python_type = {
                                "bool": "builtins.bool",
                                "int": "builtins.int",
                                "int32": "builtins.int",
                                "int64": "builtins.int",
                                "float": "builtins.float",
                                "float32": "builtins.float",
                                "float64": "builtins.float",
                            }.get(str(dtype), str(dtype or "unknown"))
                            return _ProgramABIValueFact(
                                python_type,
                                str(field.get("storage") or "unknown"),
                                None if dtype is None else str(dtype),
                            )
        if operation in {"numel", "ndim", "ndims"}:
            parent = next((
                int(parent)
                for parent, role in (data.get("parents") or ())
                if str(role) in {"operand", "value", "base", "object"}
            ), None)
            descriptor = None if parent is None else _tensor_descriptor(
                graph, parent
            )
            if descriptor is not None:
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
            left = known.get(parents.get("lhs", parents.get("left")), unresolved)
            right = known.get(parents.get("rhs", parents.get("right")), unresolved)
            if left is unresolved or right is unresolved:
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
            comparison = {
                ast.Eq: operator.eq, ast.NotEq: operator.ne,
                ast.Lt: operator.lt, ast.LtE: operator.le,
                ast.Gt: operator.gt, ast.GtE: operator.ge,
                ast.Is: operator.is_, ast.IsNot: operator.is_not,
            }.get(type(expression.ops[0]))
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
            if not values or any(
                value is unresolved
                or isinstance(value, _ProgramABIValueFact)
                for value in values
            ):
                return unresolved
            return all(values) if isinstance(expression.op, ast.And) else any(values)
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
        if name == "getattr" and len(arguments) >= 2:
            attribute = values[1]
            if isinstance(attribute, str) and not attribute.startswith("_"):
                fixed = descriptor_attribute(arguments[0], attribute)
                if fixed is not unresolved:
                    return fixed
            if len(values) >= 3 and values[2] is not unresolved:
                return values[2]
        if name == "isinstance" and values:
            if isinstance(values[0], _ProgramABIValueFact):
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
        if any(value is unresolved for value in values):
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
        data = graph.G.nodes[int(node_id)]
        for parent, _role in tuple(data.get("parents") or ()):
            if graph.G.has_edge(int(parent), int(node_id)):
                graph.G.remove_edge(int(parent), int(node_id))
            if int(parent) in graph.G:
                graph.G.nodes[int(parent)]["children"] = [
                    (child, role)
                    for child, role in graph.G.nodes[int(parent)].get(
                        "children", ()
                    )
                    if int(child) != int(node_id)
                ]
        attributes = {
            "value": copy.deepcopy(value),
            "structural_specialization": True,
        }
        data.update({
            "type": "Constant", "op": "const", "label": repr(value),
            "parents": [], "attributes": attributes,
            "constant": copy.deepcopy(value), "expr_obj": None,
        })

    def remove_node(node_id: int) -> None:
        node_id = int(node_id)
        if node_id not in graph.G:
            return
        for successor in tuple(graph.G.successors(node_id)):
            successor_data = graph.G.nodes[int(successor)]
            successor_data["parents"] = [
                (int(parent), str(role))
                for parent, role in successor_data.get("parents") or ()
                if int(parent) != node_id
            ]
        for predecessor in tuple(graph.G.predecessors(node_id)):
            graph.G.nodes[int(predecessor)]["children"] = [
                (child, role)
                for child, role in graph.G.nodes[int(predecessor)].get(
                    "children", ()
                )
                if int(child) != node_id
            ]
        graph.G.remove_node(node_id)

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
            if graph.G.has_edge(node_id, int(successor)):
                graph.G.remove_edge(node_id, int(successor))
            successor_data["parents"] = list(replacement)
            for parent, role in replacement:
                if not graph.G.has_edge(int(parent), int(successor)):
                    graph.G.add_edge(int(parent), int(successor), role=str(role))
                children = graph.G.nodes[int(parent)].setdefault("children", [])
                if (int(successor), str(role)) not in children:
                    children.append((int(successor), str(role)))
        identities = graph.G.graph.get("identity_table") or {}
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

    changed = True
    while changed:
        changed = False
        for node_id in _dependency_order(graph):
            node_id = int(node_id)
            data = graph.G.nodes[node_id]
            operation = str(data.get("op") or data.get("type") or "").casefold()
            if _tensor_descriptor(graph, node_id) is None and operation in {
                "add", "sub", "mult", "mul", "div", "truediv",
                "floordiv", "mod", "pow", "maximum", "minimum",
            }:
                parent_descriptors = tuple(
                    descriptor
                    for parent, role in (data.get("parents") or ())
                    if str(role) not in {"callee", "func", "definition"}
                    and (
                        descriptor := _tensor_descriptor(graph, int(parent))
                    ) is not None
                )
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
                            "dtype": parent_descriptors[0]["dtype"],
                        }
            attributes = data.get("attributes") or {}
            tensor_candidate = str(
                attributes.get("tensor_candidate") or operation
            ).casefold()
            if (
                _tensor_descriptor(graph, node_id) is None
                and tensor_candidate in {
                    "transpose", "swapaxes", "transpose_last2", "t"
                }
            ):
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
            if (
                _tensor_descriptor(graph, node_id) is None
                and tensor_candidate in {"matmul", "mm"}
            ):
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
            if (
                _tensor_descriptor(graph, node_id) is None
                and tensor_candidate in {"reshape", "view"}
            ):
                source_descriptor = next((
                    _tensor_descriptor(graph, int(parent))
                    for parent, role in (data.get("parents") or ())
                    if str(role) in {"operand", "arg:0", "value"}
                    and _tensor_descriptor(graph, int(parent)) is not None
                ), None)
                shape_parent = next((
                    int(parent)
                    for parent, role in (data.get("parents") or ())
                    if str(role) in {"shape", "new_shape", "arg:1", "arg:0"}
                    and str(role) not in {"operand", "value"}
                    and known.get(int(parent), unresolved) is not unresolved
                ), None)
                if source_descriptor is not None and shape_parent is not None:
                    result_shape = known[int(shape_parent)]
                    if isinstance(result_shape, (tuple, list)):
                        data["tensor"] = {
                            "shape": tuple(int(extent) for extent in result_shape),
                            "dtype": source_descriptor["dtype"],
                        }
            if (
                _tensor_descriptor(graph, node_id) is None
                and tensor_candidate in {
                    "sum", "mean", "prod", "min", "max", "any", "all"
                }
            ):
                source_descriptor = next((
                    _tensor_descriptor(graph, int(parent))
                    for parent, role in (data.get("parents") or ())
                    if str(role) in {"operand", "arg:0", "value"}
                    and _tensor_descriptor(graph, int(parent)) is not None
                ), None)
                structural_arguments = {
                    str(role): known.get(int(parent), unresolved)
                    for parent, role in (data.get("parents") or ())
                }
                axis = next((
                    value for role, value in structural_arguments.items()
                    if role in {"dim", "axis", "kw:dim", "kw:axis", "arg:1"}
                    and value is not unresolved
                ), unresolved)
                keepdim = next((
                    value for role, value in structural_arguments.items()
                    if role in {"keepdim", "kw:keepdim", "arg:2"}
                    and value is not unresolved
                ), False)
                if source_descriptor is not None and axis is not unresolved:
                    source_shape = tuple(source_descriptor["shape"])
                    raw_axes = (
                        tuple(axis) if isinstance(axis, (tuple, list))
                        else (int(axis),)
                    )
                    axes = tuple(sorted({
                        int(item) % len(source_shape) for item in raw_axes
                    })) if source_shape else ()
                    if bool(keepdim):
                        result_shape = tuple(
                            1 if index in axes else extent
                            for index, extent in enumerate(source_shape)
                        )
                    else:
                        result_shape = tuple(
                            extent for index, extent in enumerate(source_shape)
                            if index not in axes
                        )
                    data["tensor"] = {
                        "shape": result_shape,
                        "dtype": source_descriptor["dtype"],
                    }
            if str(data.get("type")) in {"Constant", "Const", "const"}:
                value = constant(data)
            else:
                value = evaluate(node_id, data)
            if value is unresolved:
                continue
            if node_id in loop_carried_initial_ids:
                # This literal is the entry arm of a Phi, not an invariant
                # fact for the loop's condition/body.  Keeping it out of the
                # structural-known table prevents the fixed point from
                # replacing expressions such as ``iters < max_iters`` with
                # their iteration-zero value.
                continue
            if isinstance(value, _StructuralValueAlias):
                replace_alias(node_id, value.source_id)
                known.pop(node_id, None)
                changed = True
                break
            if (
                node_id not in known
                or not same_structural_value(known[node_id], value)
            ):
                known[node_id] = value
                changed = True
            # ABI facts prove source type and physical presence; they are not
            # runtime literals.  Keep the authored SSA producer intact so a
            # record field remains a normal dataflow value while comparisons
            # such as ``field is not None`` can still be decided here.
            if isinstance(value, _ProgramABIValueFact):
                continue
            if str(data.get("type")) not in {"Constant", "Const", "const"}:
                replace(node_id, value)

    # A callsite specialization may make source control decidable even when
    # the selected value remains runtime numerical data.  Remove only the
    # unselected lexical compartment and replace its merge Phi with an alias
    # to the selected producer.  The proof is the graph constant above; no
    # runtime tensor value or Python callable participates.
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
                if phi_parents.get("test") != predicate_id:
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
            rejected_nodes = {
                int(node_id)
                for node_id, data in graph.G.nodes(data=True)
                if id(data.get("expr_obj")) in rejected_ast_ids
            }
            for node_id in sorted(rejected_nodes, reverse=True):
                remove_node(node_id)
            remove_node(int(control_id))
            identities = graph.G.graph.get("identity_table") or {}
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
            if (
                int(node_id) not in protected_values
                and graph.G.out_degree(int(node_id)) == 0
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
        and int(node_id) not in set(map(int, graph.roots))
    }
    for node_id in sorted(unused_parameters, reverse=True):
        remove_node(node_id)
    if unused_parameters:
        identities = graph.G.graph.get("identity_table") or {}
        graph.G.graph["identity_table"] = {
            str(name): tuple(
                int(value_id) for value_id in history
                if int(value_id) in graph.G
            )
            for name, history in identities.items()
        }


def _apply_callsite_tensor_descriptors(
    graph: Any,
    descriptors: Mapping[str, Mapping[str, Any]],
) -> None:
    identities = graph.G.graph.get("identity_table") or {}
    for name, descriptor in descriptors.items():
        candidates = tuple(dict.fromkeys((
            *tuple(identities.get(str(name), ())),
            *(
                int(node_id)
                for node_id, data in graph.G.nodes(data=True)
                if data.get("type") == "Input"
                and (data.get("attributes") or {}).get("binding_name") == name
            ),
        )))
        for node_id in candidates:
            if int(node_id) not in graph.G:
                continue
            data = graph.G.nodes[int(node_id)]
            if data.get("type") != "Input":
                continue
            data["tensor"] = copy.deepcopy(dict(descriptor))


def _expand_specialized_unbroadcast_identity(graph: Any) -> bool:
    """Lower the authored ``unbroadcast`` helper to canonical tensor nodes.

    The helper's Python loop is only structural shape logic.  Once a callsite
    supplies both shapes, retain its exact BACKWARD_RULES meaning as a finite
    sequence of existing ``sum`` and ``reshape`` operators instead of asking
    the control planner to carry a tensor through a Python ``for`` loop.
    """

    if str(graph.G.graph.get("function_name") or "") != "unbroadcast":
        return False
    descriptors = graph.G.graph.get("planner_tensor_descriptors") or {}
    specializations = graph.G.graph.get("planner_specializations") or {}
    source_descriptor = descriptors.get("G")
    target_shape = specializations.get("target_shape")
    if source_descriptor is None or not isinstance(target_shape, (tuple, list)):
        return False
    source_shape = tuple(map(int, source_descriptor.get("shape") or ()))
    target_shape = tuple(map(int, target_shape))
    input_id = next((
        int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        if data.get("type") == "Input"
        and (data.get("attributes") or {}).get("binding_name") == "G"
    ), None)
    if input_id is None:
        return False
    metadata = copy.deepcopy(dict(graph.G.graph))
    input_data = copy.deepcopy(dict(graph.G.nodes[input_id]))
    graph.G.clear()
    graph.G.graph.update(metadata)
    input_data["parents"] = []
    input_data["children"] = []
    input_data["tensor"] = copy.deepcopy(dict(source_descriptor))
    graph.G.add_node(input_id, **input_data)
    next_id = input_id + 1
    current = input_id
    current_shape = list(source_shape)

    def append(op: str, attributes: Mapping[str, Any], shape: tuple[int, ...]):
        nonlocal next_id, current
        node_id = next_id
        next_id += 1
        role = "operand"
        graph.G.add_node(
            node_id, op=op, type=op, label=op,
            parents=[(current, role)], children=[],
            attributes={
                **dict(attributes),
                "tensor_candidate": op,
                "authored_identity": "backward_registry.unbroadcast",
            },
            extra_args=dict(attributes),
            tensor={
                "shape": tuple(shape),
                "dtype": str(source_descriptor.get("dtype") or "float64"),
            },
            control={}, constant=None, expr_obj=None, store_id=None,
        )
        graph.G.add_edge(current, node_id, role=role)
        graph.G.nodes[current]["children"].append((node_id, role))
        current = node_id

    while len(current_shape) > len(target_shape):
        current_shape.pop(0)
        append(
            "sum", {"dim": 0, "keepdim": False}, tuple(current_shape),
        )
    for axis, (actual, target) in enumerate(zip(current_shape, target_shape)):
        if target == 1 and actual != 1:
            current_shape[axis] = 1
            append(
                "sum", {"dim": axis, "keepdim": True},
                tuple(current_shape),
            )
    append("reshape", {"shape": target_shape}, target_shape)
    graph.roots = [current]
    identities = copy.deepcopy(metadata.get("identity_table") or {})
    identities["G"] = (input_id,)
    identities["result_0"] = (current,)
    graph.G.graph.update({
        "identity_table": identities,
        "function_outputs": ("result_0",),
        "function_parameters": ("G",),
        "positional_parameters": ("G",),
        "keyword_only_parameters": (),
        "authored_identity_expansion": "backward_registry.unbroadcast",
    })
    return True


def _callsite_specialized_shell_type(
    owner: Any,
    node_id: int,
    reference: int,
    fallback: type,
    max_nodes_per_dispatch: int,
) -> type:
    """Plan one literal callsite once, before its shell is instantiated."""

    caller = owner.process_graph
    if int(node_id) not in caller.G:
        return fallback
    function_table = getattr(caller, "function_table", None)
    if function_table is None:
        return fallback
    try:
        original = function_table.entry(int(reference)).graph
    except (KeyError, TypeError, ValueError):
        return fallback
    if original is None:
        return fallback
    _receiver, positional, _all = _method_parameter_layout(original.G)
    specializations = {}
    tensor_descriptors: dict[str, dict[str, Any]] = {}
    bound_parameters: set[str] = set()
    data = caller.G.nodes[int(node_id)]
    for parent, role_value in data.get("parents") or ():
        role = str(role_value)
        position = _positional_argument_index(role)
        parameter = (
            positional[position]
            if position is not None and position < len(positional)
            else role[3:]
            if role.startswith("kw:")
            else None
        )
        if parameter is None:
            continue
        parameter = str(parameter)
        bound_parameters.add(parameter)
        descriptor = _tensor_descriptor(caller, int(parent))
        if descriptor is not None:
            tensor_descriptors[parameter] = descriptor
        if _source_static_value(caller, int(parent)):
            try:
                specializations[parameter] = _source_static_literal(
                    caller, int(parent)
                )
            except ValueError:
                pass
    for parameter, default in (
        original.G.graph.get("parameter_defaults") or {}
    ).items():
        parameter = str(parameter)
        if parameter in bound_parameters:
            continue
        if default is None or isinstance(
            default, (bool, int, float, str, bytes)
        ) or (
            isinstance(default, tuple)
            and all(
                item is None or isinstance(
                    item, (bool, int, float, str, bytes)
                )
                for item in default
            )
        ):
            specializations[parameter] = copy.deepcopy(default)
    if not specializations and not tensor_descriptors:
        return fallback

    def stable(value: Any) -> object:
        if isinstance(value, dict):
            return tuple(sorted(
                (stable(key), stable(item))
                for key, item in value.items()
            ))
        if isinstance(value, (tuple, list, set, frozenset)):
            return tuple(stable(item) for item in value)
        try:
            hash(value)
        except TypeError:
            return repr(value)
        return value

    key = (
        id(function_table),
        int(reference),
        tuple(sorted(
            (name, stable(value))
            for name, value in specializations.items()
        )),
        tuple(sorted(
            (name, stable(value))
            for name, value in tensor_descriptors.items()
        )),
        int(max_nodes_per_dispatch),
    )
    cached = _CALLSITE_SHELL_TYPE_CACHE.get(key)
    if cached is not None:
        return cached
    specialized = extract_clean_process_subgraph(
        original,
        original.G,
    )
    # These are callsite facts, not accumulators.  A shared function-table
    # graph may already carry metadata from an earlier specialization; using
    # ``setdefault().update()`` allowed a scalar literal from one occurrence
    # to survive into a later tensor-valued occurrence of the same parameter.
    specialized.G.graph["planner_specializations"] = copy.deepcopy(
        specializations
    )
    specialized.G.graph["planner_tensor_descriptors"] = copy.deepcopy(
        tensor_descriptors
    )
    _apply_callsite_tensor_descriptors(specialized, tensor_descriptors)
    if not _expand_specialized_unbroadcast_identity(specialized):
        _fold_callsite_structural_values(specialized)
    planned = strategize_shell_deployment(
        specialized,
        max_nodes_per_dispatch=int(max_nodes_per_dispatch),
        _function_table_stack=(id(function_table),),
    )
    _CALLSITE_SHELL_TYPE_CACHE[key] = planned
    return planned


def _resolve_grounded_method_references(graph: Any) -> None:
    """Attach internal methods only through a proven receiver class.

    A method name being unique in the ingested class catalogue says nothing
    about an unrelated receiver.  In particular, ``os.environ.get`` must not
    become ``_RandomFloatQueue.get`` merely because that is the sole authored
    class method named ``get``.  Follow explicit value producers to a
    ``class_ref`` instead; ambiguity or absence leaves the call external/static
    and fully visible rather than fabricating an internal topology edge.
    """

    class_table = graph.G.graph.get("class_table") or {}
    specializations = graph.G.graph.get("planner_specializations") or {}

    def specialized_class(binding_name: str | None) -> str | None:
        if binding_name is None or str(binding_name) not in specializations:
            return None
        receiver_type = type(specializations[str(binding_name)])
        identities = {
            receiver_type.__name__,
            receiver_type.__qualname__,
            f"{receiver_type.__module__}.{receiver_type.__qualname__}",
        }
        matches = {
            str(identity)
            for identity in class_table
            if str(identity) in identities
            or str(identity).rsplit(".", 1)[-1] == receiver_type.__name__
        }
        return next(iter(matches)) if len(matches) == 1 else None

    def specialized_method_reference(
        binding_name: str | None,
        method_name: str,
    ) -> int | None:
        if binding_name is None or str(binding_name) not in specializations:
            return None
        receiver_type = type(specializations[str(binding_name)])
        owner_names = {
            receiver_type.__name__,
            receiver_type.__qualname__,
            f"{receiver_type.__module__}.{receiver_type.__qualname__}",
        }
        table = getattr(graph, "function_table", None)
        if table is None:
            return None
        matches = {
            int(entry.reference.address)
            for entry in table
            if str(entry.name) == str(method_name)
            and entry.graph is not None
            and str(entry.graph.G.graph.get("method_owner")) in owner_names
        }
        return next(iter(matches)) if len(matches) == 1 else None

    def receiver_class(value_id: int) -> str | None:
        pending = [int(value_id)]
        visited = set()
        candidates = set()
        while pending:
            current = pending.pop()
            if current in visited or current not in graph.G:
                continue
            visited.add(current)
            node = graph.G.nodes[current]
            attributes = node.get("attributes") or {}
            class_ref = attributes.get("class_ref")
            if class_ref is not None:
                candidates.add(str(class_ref))
                continue
            if str(node.get("type")) in {"Input", "input"}:
                class_ref = specialized_class(attributes.get("binding_name"))
                if class_ref is not None:
                    candidates.add(class_ref)
                    continue
            # Only identity-routing nodes may transmit a receiver class. An
            # arbitrary operation consuming an object does not return it.
            if str(node.get("type")) not in {
                "Input", "Phi", "LoopResult", "LoopExit", "Identity",
            }:
                continue
            pending.extend(
                int(parent)
                for parent, role in (node.get("parents") or ())
                if str(role) in {
                    "value", "body", "orelse", "initial", "updated",
                    "result", "operand",
                }
            )
        return next(iter(candidates)) if len(candidates) == 1 else None

    for _node_id, data in graph.G.nodes(data=True):
        attributes = data.get("attributes") or {}
        if (
            attributes.get("callee_ref") is not None
            or attributes.get("method_ref") is not None
        ):
            continue
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
        ):
            continue
        receiver_id = next((
            int(parent)
            for parent, role in (data.get("parents") or ())
            if str(role) in {"operand", "receiver"}
        ), None)
        if receiver_id is None:
            continue
        class_identity = receiver_class(receiver_id)
        reference = None
        if class_identity is not None:
            reference = (
                class_table.get(class_identity, {}).get("methods", {})
                .get(str(expression.func.attr))
            )
        if (
            reference is None
            and isinstance(expression.func.value, ast.Name)
        ):
            reference = specialized_method_reference(
                expression.func.value.id,
                expression.func.attr,
            )
        if reference is None:
            continue
        attributes = dict(attributes)
        attributes["method_ref"] = int(reference)
        attributes["method_resolution"] = "receiver-class-ref"
        if class_identity is not None:
            attributes["receiver_class_ref"] = class_identity
        data["attributes"] = attributes


def _resolve_grounded_tensor_operations(graph: Any) -> None:
    """Promote method-name candidates only along tensor-valued SSA edges."""

    specializations = graph.G.graph.get("planner_specializations") or {}

    def is_tensor_value(value: Any) -> bool:
        return (
            not isinstance(value, (str, bytes, bytearray, list, tuple, dict, set))
            and hasattr(value, "shape")
            and hasattr(value, "dtype")
        )

    tensor_values = {
        int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        if (data.get("attributes") or {}).get("tensor") is not None
        or (
            data.get("type") == "Input"
            and is_tensor_value(specializations.get(str(
                (data.get("attributes") or {}).get("binding_name", "")
            )))
        )
    }
    # These ordinary ProcessGraph operators preserve tensor-valuedness when
    # at least one data operand is a tensor.  Keep this as provenance only:
    # the later SSA shape pass and tensor repository lowering still decide
    # the concrete result layout and implementation.
    tensor_value_operators = {
        "Add", "Sub", "Mult", "Mul", "Div", "FloorDiv", "Mod", "Pow",
    }
    changed = True
    while changed:
        changed = False
        for node_id, data in graph.G.nodes(data=True):
            attributes = data.get("attributes") or {}
            candidate = attributes.get("tensor_candidate")
            if int(node_id) in tensor_values:
                continue
            if (
                str(data.get("type") or data.get("op"))
                in tensor_value_operators
                and any(
                    int(parent) in tensor_values
                    for parent, role in (data.get("parents") or ())
                    if str(role) not in {"callee", "func"}
                )
            ):
                tensor_values.add(int(node_id))
                changed = True
                continue
            if candidate is None:
                continue
            expression = data.get("expr_obj")
            if not (
                isinstance(expression, ast.Call)
                and isinstance(expression.func, ast.Attribute)
            ):
                continue
            receiver = next((
                int(parent)
                for parent, role in (data.get("parents") or ())
                if str(role) in {"operand", "receiver"}
            ), None)
            if receiver not in tensor_values:
                continue
            attributes = dict(attributes)
            attributes["tensor"] = str(candidate)
            attributes["tensor_resolution"] = "receiver-tensor-value"
            data["attributes"] = attributes
            tensor_values.add(int(node_id))
            changed = True


# Above this an unroll is refused rather than attempted. It is a guard
# against a runaway trip count turning into an unbounded program, not a
# judgement about what is reasonable: a caller who wants this much detail is
# spending their own compute, and the emitted program is theirs to hold.
MAX_UNROLL_LIMIT = 4096


class ProcessGraphGLSLDeployment:
    # Per-compile values. A fresh subclass is created for each compiled
    # ProcessGraph (see strategize_shell_deployment) with these overridden
    # via ordinary class-attribute assignment -- no source-text templating.
    process_graph = None
    external_function_table = None
    static_python_bindings = None
    dispatch_plan = None
    loop_plans = None
    loop_region_indices = None
    loop_shader_reductions = None
    control_deployment_regions = None
    process_graph_boundary = None
    python_callable = None
    dispatch_subgraphs = None
    deep_compilers = None
    ephemeral_callables = None
    reference_table_template = None
    function_shell_types = {}
    runtime_closure_only = False
    planned_function_references = ()
    activation_root_references = ()
    catalogued_function_references = ()
    catalogue_only_function_references = ()
    source_node_count = None
    primitive_count = None
    loop_count = None
    dispatch_count = None
    deployment_batches = None
    max_nodes_per_dispatch = None

    def __init__(
        self,
        *,
        batch_size=None,
        profiling=False,
        verbose_profile=False,
        input_slots=64,
        output_slots=64,
        legacy_fused_network=False,
        shell_language="glsl",
        **tuning,
    ):
        tuning = dict(tuning)
        tuning["legacy_fused_network"] = bool(legacy_fused_network)
        tuning["shell_language"] = str(shell_language)
        self.state = {}
        self.tuning = dict(tuning)
        self.legacy_fused_network = bool(legacy_fused_network)
        self.shell_language = str(shell_language)
        if self.shell_language not in {"python", "c", "glsl"}:
            raise ValueError(
                "shell_language must be python, c, or glsl"
            )
        self.control_runtime = (
            "legacy_fused_network"
            if self.legacy_fused_network
            else "composed_control"
        )
        self.batch_size = dict(
            self.deployment_batches
            if batch_size is None
            else batch_size
        )
        self.profiling = bool(profiling or verbose_profile)
        self.verbose_profile = bool(verbose_profile)
        self._profiler = DeploymentProfiler(
            self.profiling,
            verbose=self.verbose_profile,
        )
        self.error_buffer = self._profiler.error_buffer
        self.profile = self._profiler
        self.profile_path = _shell_profile_name(self)
        self.inputs = DeploymentFIFO(input_slots)
        self.outputs = DeploymentFIFO(output_slots)
        self.reference_tables = self.reference_table_template.copy()
        self.function_references = self.reference_tables.functions
        self.constant_references = self.reference_tables.constants
        self.memory_references = self.reference_tables.memory
        self.recursion_references = self.reference_tables.recursion
        self.reference_correlations = self.reference_tables.correlations
        self.function_shells = {
            reference: shell_type(**tuning)
            for reference, shell_type in self.function_shell_types.items()
        }
        self._owns_function_shells = bool(self.function_shells)
        for function_shell in self.function_shells.values():
            function_shell.function_shells = self.function_shells
            function_shell._owns_function_shells = False
        def plan_callsites(owner, active_references=()):
            owner.callsite_function_shells = {}
            for node_id, data in owner.process_graph.G.nodes(data=True):
                attributes = data.get("attributes") or {}
                reference = attributes.get("callee_ref")
                if reference is None:
                    reference = attributes.get("method_ref")
                if reference is None and attributes.get("class_ref") is not None:
                    class_record = (
                        owner.process_graph.G.graph.get("class_table") or {}
                    ).get(str(attributes["class_ref"]), {})
                    methods = class_record.get("methods") or {}
                    reference = methods.get(
                        "__new__", methods.get("__init__")
                    )
                    if reference is not None:
                        attributes = dict(attributes)
                        attributes["constructor_ref"] = int(reference)
                        data["attributes"] = attributes
                if reference is None:
                    continue
                reference = int(reference)
                shell_type = self.function_shell_types.get(reference)
                if shell_type is None or reference in active_references:
                    continue
                shell_type = _callsite_specialized_shell_type(
                    owner,
                    node_id,
                    reference,
                    shell_type,
                    self.max_nodes_per_dispatch,
                )
                planned = shell_type(**tuning)
                planned.function_shells = self.function_shells
                planned._owns_function_shells = False
                owner.callsite_function_shells[node_id] = planned
                plan_callsites(
                    planned,
                    (*active_references, reference),
                )
        activation_roots = (
            tuple(self.activation_root_references)
            if self.runtime_closure_only
            else tuple(self.function_shells)
        )
        for reference in activation_roots:
            function_shell = self.function_shells.get(int(reference))
            if function_shell is None:
                continue
            plan_callsites(function_shell)
            (
                function_shell.hierarchy_plan,
                function_shell.hierarchy_value_table,
            ) = assign_hierarchy_ids(
                _build_shell_hierarchy_plan(function_shell)
            )
        if self.runtime_closure_only:
            # The module shell owns the definition catalogue, not another
            # execution of every function body in the source file.  Submitted
            # target shells above are the activation roots in this mode.
            self.callsite_function_shells = {}
        else:
            plan_callsites(self)
        (
            self.hierarchy_plan,
            self.hierarchy_value_table,
        ) = assign_hierarchy_ids(_build_shell_hierarchy_plan(self))
        _attach_profiler(
            self,
            self._profiler,
            self.profile_path,
        )
        self.fused_network = None
        self.compiled_shell_callable = None
        self.whole_program_compiled = False
        self.whole_program_device = None
        self.compiled_dispatch_functions = ()
        self.compiled_dispatch_sources = ()
        self.forward_tapes = ()
        self.forward_feed_ids = ()
        self.forward_aggregate_feed_paths = ()
        self.compiled_aggregate_feed_paths = ()
        self.compiled_feed_meta = {}
        self.forward_subgraphs = ()
        self.forward_compilers = ()
        self.forward_region_indices = ()
        self.forward_output_values = ()
        self.forward_region_capture_node_ids = ()
        self.forward_region_planned_capture_ids = ()
        self.forward_region_planned_input_ids = ()
        self.forward_planned_collection_materializations = {}
        self.compiled_process_graph_aliases = {}
        self.compiled_tapes = ()
        self.compiled_shell_program = None
        self.compiled_region_indices = ()
        self.captured_region_programs = {}
        self.planned_operator_implementations = {}
        # A reference operator (SetAttr/GetAttr) has an unambiguous identity
        # from its own graph node id -- no tensor primitive to correlate,
        # no ambiguity the tape machinery above exists to resolve.  Recorded
        # here, in exact execution order, as each one runs: not a strategy,
        # not a region to score for launch profitability, just what happened,
        # in the order it happened -- the sequential memory operations this
        # represents have real causality that reordering or fusing would
        # break.
        self.reference_operator_sequence = []
        self._capture_invocations = 0
        # Which source parameter each captured input tensor came from, by
        # object identity. Feeds are identified downstream by id(), and
        # without this the name is lost at the boundary -- leaving a
        # descriptor that can say "feed0" but not "cx", so a caller has to
        # guess which array goes where.
        self._capture_input_names = {}
        # The same names, keyed by the resident range instead of by object
        # identity, so a value that was rewrapped can still be recognised.
        self._capture_input_storage = {}
        # Scalar configuration fields carried by structural input objects are
        # compile-time control facts.  Their scoped field paths let hierarchy
        # composition resolve predicates such as ``ctrl.dt_min is not None``
        # without retaining or inspecting the object later.
        self._capture_input_static_fields = {}
        self._captured_return_value_ids = set()
        self._capture_input_type_names = {}
        self._capture_tensor_input_names = set()
        self._captured_source_branches = []
        self._captured_loop_iterations = {}
        self._execute_invocations = 0
        self._discovery_tape_creations = 0
        self._discovery_tape_lowerings = 0
        self._discovery_tape = None
        self._discovery_tape_owner = False
        self._discovery_session = None
        self._discovery_tape_bindings = {}
        self._discovery_tape_complete = False
        self.collection_observations = {}
        self.collection_collapse_report = {}
        self.planned_invocation_slots = 1
        self.coordinator_region_indices = set()
        self.compile_time_region_indices = set()
        self.compile_failures = ()
        self.glsl_sources = ()
        self.composed_loop_sources = {}
        self.composed_loop_artifacts = {}
        self.installed_control_shell = None
        self.composed_shell_artifact = None
        self.composed_shell_specialized_values = {}
        self.shell_control_program = None
        self.hierarchical_control_program = None
        self.hierarchical_captured_region_programs = {}
        self.hierarchical_shell_composed = False
        self.hierarchical_root_value_ids = {}
        self.hierarchical_private_value_ids = {}
        self.hierarchical_capture_value_ids = {}
        self.hierarchical_synthetic_field_paths = {}
        self.hierarchical_program_value_origins = {}
        self.hierarchical_aggregate_redirect_diagnostics = {}
        self.hierarchical_aggregate_candidate_diagnostics = {}
        self.hierarchical_control_binding_diagnostics = {}
        self.hierarchical_composed_closure_iterable_bindings = ()
        self.hierarchical_region_correlations = ()
        self.hierarchical_endpoint_details = {}
        self.hierarchical_value_aliases = {}
        self.hierarchical_root_field_value_ids = {}
        self.hierarchical_effective_value_table = None
        self.hierarchical_compose_failure = None
        self.hierarchical_public_output_ids = set()
        self.hierarchical_terminal_outputs = {}
        self.hierarchical_specialized_values = {}
        self.hierarchy_identity_collapses = ()
        self.hierarchy_identity_rounds = 0
        self.hierarchy_remaining_callsite_ids = None
        self.composed_shell_complete = False
        self.composed_shell_blockers = ()
        self.input_bindings = {}
        self.output_bindings = {}
        self.output_aggregate_bindings = {}
        self.output_loop_aggregate_bindings = {}

    @property
    def ready(self):
        if not self.legacy_fused_network:
            return self.installed_control_shell is not None
        return (
            self.installed_control_shell is not None
            or self.whole_program_compiled
            or self.fused_network is not None
        )

    @property
    def programs(self):
        if self.fused_network is None:
            return ()
        return self.fused_network.programs

    def enable_profiling(self, enabled=True):
        self._profiler.enabled = bool(enabled)
        self.profiling = bool(enabled)
        return self

    def enable_verbose_profiling(self, enabled=True):
        self._profiler.verbose = bool(enabled)
        if enabled:
            self.enable_profiling(True)
        self.verbose_profile = bool(enabled)
        return self

    def trace_report(self, *, limit=None):
        records = tuple(self._profiler.trace_history)
        if limit is not None:
            records = records[-max(0, int(limit)):]
        return records

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
        for row in report["device_rows"]:
            lines.append(
                f"  {row['path']} | logging-ssbo | "
                f"{row['label']}[{row['code']}] | "
                f"events {row['events_mean']:.1f}/invocation | "
                f"payload {row['payload0_mean']:.1f},"
                f"{row['payload1_mean']:.1f}"
            )
        return tuple(lines)

    def program_table_lines(self):
        return _deployment_program_table_lines(self)

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
        raise RuntimeError(
            "per-region tape capture has been retired: call "
            "capture_fused_programs() once to discover the complete source "
            "program, compile its shell, and destroy the discovery tape"
        )

    def capture_scheduled_forward_tapes(
        self,
        initial_values,
        *,
        device=None,
    ):
        raise RuntimeError(
            "per-region scheduled tape capture has been retired: call "
            "capture_fused_programs() once; the planner's region boundaries "
            "are cut points within that single discovery tape, not separate "
            "executions"
        )

    def coordinate_first_invocation(self, initial_values, *, device=None):
        return _coordinate_scheduled_capture(
            self,
            initial_values,
            device=device,
            capture=True,
        )

    def refresh_hierarchy_plan(self):
        # Freeze hierarchy only after all callsite shells are attached.

        (
            self.hierarchy_plan,
            self.hierarchy_value_table,
        ) = assign_hierarchy_ids(_build_shell_hierarchy_plan(self))
        return self.hierarchy_plan

    def prepare_graph_precompile(
        self, *, device=None, reduction_cache=None, progress=None,
        structural_ssa_only: bool = False,
    ):
        # Prepare graph/control IR without observing runtime parameter values.
        #
        # ``reduction_cache`` (a ``ReductionArtifactStore``), when given,
        # persists each region program under its own content key as it is
        # lowered -- an incremental backup of this portion, so an interrupted
        # prepare reloads the regions it already built instead of redoing all
        # of them, and each region becomes an independently addressable
        # catalogue entry. ``progress``, if given, is called with a message
        # string per shell and per region so this otherwise-silent phase is
        # visible.

        def _note(message):
            if progress is not None:
                progress(message)

        self.refresh_hierarchy_plan()
        if not self.whole_program_compiled:
            self.compile_process_graph(device=device)
        # A selected-entrypoint compile prepares only its proven activation
        # tree.  The canonical whole-source caller explicitly requests the
        # complete definition catalogue: class identity is retained by
        # ClassNavigationTable, while every constructor/method shell supplies
        # the executable contents of that one class record.  Do not infer this
        # from ``runtime_closure_only``: older selected-entrypoint adapters also
        # use broad planning but still have one executable activation tree.
        for target in _walk_planned_shells(
            self,
            include_function_registry=bool(getattr(
                self, "prepare_complete_catalogue", False
            )),
        ):
            target.refresh_hierarchy_plan()
            complete_regions = tuple(range(len(target.dispatch_subgraphs)))
            retained_value_ids = {
                int(value_id)
                for item in target.hierarchy_plan.items
                if isinstance(item, PlanClosure)
                and item.name.startswith("region_")
                for line in item.items
                if isinstance(line, PlanLine)
                for value_id in (*line.inputs, *line.outputs)
            }
            considered_reductions = tuple(
                reduction
                for reduction in target.loop_shader_reductions
                if reduction.control_program is not None
            )
            controls = tuple(
                project_control_regions(
                    reduction.control_program,
                    complete_regions,
                    retained_value_ids=retained_value_ids,
                )
                for reduction in considered_reductions
            )
            shell_control = overlay_scheduled_control(
                complete_regions,
                controls,
                known_nesting=_loop_reduction_nesting_hints(
                    considered_reductions,
                    target.loop_plans,
                    target.process_graph,
                ),
            )
            target.shell_control_program = replace(
                shell_control,
                deployment_regions=tuple(dict.fromkeys((
                    *shell_control.deployment_regions,
                    *target.control_deployment_regions,
                ))),
            )
            # The complete-source SSA path consumes ``hierarchy_plan`` and
            # ``shell_control_program`` directly.  It must not manufacture a
            # nominal FusedProgram merely to satisfy the older dual-IR shape:
            # doing so makes a non-projecting compile indistinguishable from
            # numerical capture.  Keep the legacy sentinel only for callers
            # that explicitly request the dual-IR precompile product.
            target.compiled_shell_program = (
                None
                if structural_ssa_only
                else CapturedFusedProgram(
                    FusedProgram(
                        version=1,
                        feeds=set(),
                        steps=[],
                        outputs={},
                        meta={},
                        extras={"kernel_kind": "graph-precompile"},
                    ),
                    {},
                )
            )
            target.compiled_tapes = ()
            target.compiled_region_indices = ()
            # An unfed mutable parameter is left a symbolic SSA input; its
            # region programs come from the planner-isolated dispatch subgraphs
            # via the same structural builder the tape path uses, not from a
            # bespoke plan transcription. Layout/cast ops ride through under
            # their own names for each backend to lower.
            subgraphs = tuple(enumerate(target.dispatch_subgraphs))
            fn_name = str(
                target.process_graph.G.graph.get("function_name") or "?"
            )
            _note(
                f"aot: lowering {len(subgraphs)} region(s) for shell {fn_name}"
            )
            # NOTE: regions are NOT deduplicated by sharing program objects
            # here. Structurally-identical regions are distinct *invocations*
            # with their own value-ids and field bindings; sharing one object
            # collides their value-ids and destroys the coordinator's per-region
            # wiring (producer[value_id] would overwrite). The correct IR-level
            # reduction is the kernel/invocation split -- one shared kernel
            # structure plus a per-region binding table -- not object sharing.
            # Byte-identical *kernel files* are still deduped later in the bake.
            region_programs = (
                {}
                if structural_ssa_only
                else {
                    region_index: _structural_region_program_from_subgraph(
                        subgraph
                    )
                    for region_index, subgraph in subgraphs
                }
            )
            target.captured_region_programs = region_programs
            target.compile_time_region_indices = set()
            target.planned_operator_implementations = (
                _build_planned_operator_implementations(
                    target.hierarchy_plan,
                    {},
                    (),
                )
            )
        (
            self.hierarchy_plan,
            self.hierarchy_value_table,
        ) = assign_hierarchy_ids(
            _refresh_hierarchy_control_captures(
                self.hierarchy_plan,
                self,
            ),
            self.hierarchy_value_table,
        )
        return self

    def capture_fused_programs(
        self,
        initial_values,
        *,
        device=None,
        precompile_only=False,
    ):
        if self._discovery_tape_complete:
            raise RuntimeError(
                "this source program has already consumed its one discovery "
                "tape; execute the installed compiled shell"
            )
        self.refresh_hierarchy_plan()
        if not self.whole_program_compiled:
            self.compile_process_graph(device=device)
        observer_backend = (
            "precompile_observer" if precompile_only else "glsl"
        )
        with _use_scheduled_capture_backend(observer_backend):
            result = _coordinate_scheduled_capture(
                self,
                initial_values,
                device=device,
                capture=True,
            )
        targets = []
        seen = set()
        for target in _walk_planned_shells(
            self, include_function_registry=False
        ):
            identity = id(target)
            if identity in seen or target._discovery_tape is None:
                continue
            seen.add(identity)
            targets.append(target)
        observed_target_ids = {id(target) for target in targets}
        for target in _walk_planned_shells(
            self, include_function_registry=False
        ):
            if id(target) in observed_target_ids:
                continue
            # The program's one discovery sequence did not enter this
            # callsite (for example the grayscale side of a statically color
            # branch).  Record reachability as an explicit empty structural
            # compartment.  Do not execute it, create a tape for it, or infer
            # numerical constants.  If pruning it removes required numerical
            # output, the root artifact's public-output completeness check
            # still rejects the compiled shell.
            target.compiled_shell_program = CapturedFusedProgram(
                FusedProgram(
                    version=1,
                    feeds=set(),
                    steps=[],
                    outputs={},
                    meta={},
                    extras={"kernel_kind": "structural-unreached"},
                ),
                {},
            )
            target.compiled_tapes = ()
            target.compiled_region_indices = ()
            target.compile_time_region_indices = set(
                range(len(target.dispatch_subgraphs))
            )
            target.shell_control_program = ControlProgram(
                SequenceBlock(()),
                (),
            )
        incomplete_shells = []
        # Freeze the one discovery run's wrapper identities before any child
        # shell is lowered and discarded.  This table exists only during
        # compilation so a nested structured argument can be correlated with
        # its caller's SSA result; it is cleared with ``_discovery_session``
        # before the installed shell can execute.
        discovery_capture_objects = {}

        def collect_discovery_capture_objects(value):
            if isinstance(value, dict):
                for item in value.values():
                    collect_discovery_capture_objects(item)
                return
            if isinstance(value, (tuple, list)):
                for item in value:
                    collect_discovery_capture_objects(item)
                return
            if isinstance(value, AbstractTensor):
                discovery_capture_objects[id(value)] = value
                storage_value = getattr(value, "data", None)
                if storage_value is not None:
                    discovery_capture_objects[id(storage_value)] = (
                        storage_value
                    )

        for target in _walk_planned_shells(
            self, include_function_registry=False
        ):
            collect_discovery_capture_objects(
                getattr(target, "captured_values", {})
            )
        if self._discovery_session is not None:
            self._discovery_session["capture_objects"] = (
                discovery_capture_objects
            )
        # Lower the compilation tree's single observation exactly once at its
        # owner.  Child shells below only project their planner cut points from
        # this immutable IR; they never receive the tape and cannot lower it.
        lowered_program = self.compile_discovery_program(
            strict=True,
            emit_glsl=not precompile_only,
        )
        if lowered_program is None:
            raise RuntimeError(
                "the compilation owner did not produce lowered program IR"
            )

        def discard_discovery_payloads(target):
            # Remove observed values while retaining only compiled IR.  This
            # class is generated from an outer triple-quoted source string, so
            # keep this as a comment rather than an embedded docstring.

            target.forward_tapes = ()
            target.forward_feed_ids = ()
            target.forward_aggregate_feed_paths = ()
            target.forward_subgraphs = ()
            target.forward_compilers = ()
            target.forward_region_indices = ()
            target.forward_output_values = ()
            target.forward_region_capture_node_ids = ()
            target.forward_region_planned_capture_ids = ()
            target.forward_region_planned_input_ids = ()
            target.forward_planned_collection_materializations = {}
            target.captured_values = {}
            target.collection_observations = {}
            target._discovery_tape_bindings = {}
            target.last_result = None
            assert not target.forward_tapes
            assert not target.forward_aggregate_feed_paths
            assert not target.captured_values
            assert not target.collection_observations
            assert not target._discovery_tape_bindings
            assert target._discovery_tape_creations == (
                1 if target is self else 0
            )
            assert target._discovery_tape_lowerings == (
                1 if target is self else 0
            )
            assert target.compiled_shell_program is not None

        for target in reversed(targets):
            if target is not self:
                target.compile_discovery_program(
                    strict=True,
                    lowered_program=lowered_program,
                    emit_glsl=not precompile_only,
                )
            # Numerical lowering establishes the final control ABI (uniform
            # bounds, carried aliases, iterable bindings and validations).
            # The hierarchy is a property of the complete source program, not
            # of every suffix subtree.  Rebuilding and re-numbering it once
            # per nested shell repeatedly traverses the same descendants and
            # turns deep ordinary-Python call trees into quadratic compiler
            # work.  Rebuild it exactly once at the discovery owner, which is
            # visited after its children in this reversed traversal and can
            # therefore see every child's final control ABI.
            if target is self:
                hierarchy_refresh_started = time.perf_counter()
                target._profiler.trace(
                    path=target.profile_path,
                    section="hierarchy-refresh",
                    label="begin",
                    fields={},
                )
                (
                    target.hierarchy_plan,
                    target.hierarchy_value_table,
                ) = assign_hierarchy_ids(
                    _refresh_hierarchy_control_captures(
                        target.hierarchy_plan,
                        target,
                    ),
                    target.hierarchy_value_table,
                )
                target._profiler.trace(
                    path=target.profile_path,
                    section="hierarchy-refresh",
                    label="complete",
                    fields={
                        "elapsed_ms": round(
                            (
                                time.perf_counter()
                                - hierarchy_refresh_started
                            ) * 1e3,
                            3,
                        ),
                    },
                )
            target.planned_invocation_slots = max(
                1, target._capture_invocations
            )
            target._execute_invocations = 0
            captured = {
                (
                    region_key
                    if not isinstance(region_key, tuple)
                    else region_key[-2]
                )
                for region_key in target.compiled_region_indices
            }
            missing = (
                set(range(len(target.ephemeral_callables))) - captured
            )
            unexplained_missing = missing - set(
                target.coordinator_region_indices
            )
            for region_index in unexplained_missing:
                subgraph = target.dispatch_subgraphs[region_index]
                tensor_outputs = [
                    output_id
                    for output_id in subgraph.G.graph[
                        "deployment_outputs"
                    ]
                    if isinstance(
                        target.captured_values.get(output_id),
                        AbstractTensor,
                    )
                    and int(output_id) not in (
                        target.compiled_process_graph_aliases
                    )
                ]
                if tensor_outputs:
                    raise RuntimeError(
                        "every tensor-producing ProcessGraph callable must "
                        "become one CapturedFusedProgram; "
                        f"{target.process_graph.G.graph.get('function_name', '?')} "
                        f"region {region_index} left tensor outputs "
                        f"{tensor_outputs}; operations="
                        f"{tuple(str(subgraph.G.nodes[node].get('op') or subgraph.G.nodes[node].get('type')) for node in subgraph.G.graph.get('deployment_nodes', ()))!r}; "
                        f"recorded_regions={target.forward_region_indices!r}; "
                        f"compiled_regions={target.compiled_region_indices!r}; "
                        f"shell_id={id(target)}"
                    )
            target.coordinator_region_indices = missing
            target.captured_region_programs = dict(zip(
                target.compiled_region_indices,
                target.compiled_tapes,
            ))
            target.planned_operator_implementations = (
                _build_planned_operator_implementations(
                    target.hierarchy_plan,
                    target.captured_region_programs,
                    target.reference_operator_sequence,
                )
            )
            if target.reference_operator_sequence and not precompile_only:
                # Observation classifies the lowering selected at runtime;
                # it does not define segment membership or order.  Those are
                # already fixed by the ProcessGraph hierarchy plan.
                observed_node_ids = set(
                    map(int, target.reference_operator_sequence)
                )
                ordered_node_ids = [
                    node_id
                    for node_id in _planned_operator_node_ids(
                        target.hierarchy_plan
                    )
                    if node_id in observed_node_ids
                    if node_id in target.process_graph.G
                ]
                missing_planned_nodes = observed_node_ids - set(
                    ordered_node_ids
                )
                if missing_planned_nodes:
                    raise RuntimeError(
                        "observed plain operators are absent from the "
                        "ProcessGraph segment plan: "
                        f"{tuple(sorted(missing_planned_nodes))!r}"
                    )
                def _reference_step_input_ids(node_id: int) -> list[int]:
                    # A field's real identity is ``attribute_slot`` (see
                    # ``bind_target``/``resolve_expression``) -- a
                    # (class_identity, slot) pair grounded in the class's
                    # own declared layout, not a node id.  It carries
                    # through automatically via ``attrs`` below (a full copy
                    # of this node's attributes), so nothing needs pulling in
                    # here beyond the ordinary ``parents`` wiring.
                    node_data = target.process_graph.G.nodes[node_id]
                    return [
                        int(parent)
                        for parent, _role in (node_data.get("parents") or ())
                    ]

                reference_steps = [
                    OpStep(
                        step_id=index,
                        op_name=str(
                            target.process_graph.G.nodes[node_id].get("op")
                            or target.process_graph.G.nodes[node_id].get(
                                "type"
                            )
                        ),
                        input_ids=_reference_step_input_ids(node_id),
                        attrs=dict(
                            target.process_graph.G.nodes[node_id].get(
                                "attributes"
                            )
                            or {}
                        ),
                        result_id=int(node_id),
                    )
                    for index, node_id in enumerate(ordered_node_ids)
                ]
                reference_result_ids = {
                    step.result_id for step in reference_steps
                }
                reference_feeds = {
                    input_id
                    for step in reference_steps
                    for input_id in step.input_ids
                    if input_id not in reference_result_ids
                }
                reference_feed_origins = {
                    feed_id: {"binding_name": binding_name}
                    for feed_id in reference_feeds
                    if (
                        binding_name := (
                            target.process_graph.G.nodes.get(feed_id, {})
                            .get("attributes", {})
                            .get("binding_name")
                        )
                    )
                    is not None
                }
                reference_program = FusedProgram(
                    version=1,
                    feeds=reference_feeds,
                    steps=reference_steps,
                    outputs={},
                    extras={"capture_feed_origins": reference_feed_origins},
                )
                next_index = (
                    max(target.compiled_region_indices, default=-1) + 1
                )
                target.captured_region_programs[next_index] = (
                    CapturedFusedProgram(
                        program=reference_program,
                        feeds={},
                    )
                )
                target.compiled_region_indices = (
                    *target.compiled_region_indices, next_index,
                )
            if target is self and precompile_only:
                # SSA and other IR consumers stop at the Turing precompile
                # boundary.  They need the complete numerical manifest,
                # projected regions, and planned control, but must not build,
                # cache, emit, install, or driver-compile a GLSL artifact.
                target.composed_shell_blockers = (
                    "precompile-only",
                )
                target.composed_shell_complete = False
                discard_discovery_payloads(target)
                continue
            if target is not self:
                # A nested source function is a planner compartment, not a
                # separately deployed shell.  Its local control, projected
                # numerical IR and specialization map are all the root
                # hierarchy consumes.  Auditing a synthetic child ABI walks
                # large source subgraphs, invents "missing shell" blockers,
                # and repeats deployment work that has no runtime meaning.
                target.composed_shell_blockers = (
                    "planner-compartment-awaits-root-composition",
                )
                target.composed_shell_complete = False
                cleanup_started = time.perf_counter()
                target._profiler.trace(
                    path=target.profile_path,
                    section="discovery-cleanup",
                    label="begin",
                    fields={},
                )
                discard_discovery_payloads(target)
                target._profiler.trace(
                    path=target.profile_path,
                    section="discovery-cleanup",
                    label="complete",
                    fields={
                        "elapsed_ms": round(
                            (
                                time.perf_counter() - cleanup_started
                            ) * 1e3,
                            3,
                        ),
                    },
                )
                continue
            if target is self and target.callsite_function_shells:
                hierarchical_artifact = (
                    _build_hierarchical_glsl_artifact(target)
                )
                if hierarchical_artifact is not None:
                    target.composed_shell_artifact = (
                        hierarchical_artifact
                    )
                    target.hierarchical_shell_composed = True
            artifact = target.composed_shell_artifact
            required_inputs = (
                set()
                if artifact is None
                else set(artifact.external_value_ids)
                | set(artifact.uniform_value_ids.values())
            )
            public_inputs = {
                int(node_id)
                for node_id, data in target.process_graph.G.nodes(data=True)
                if data.get("type") == "Input"
                and (
                    data.get("attributes") or {}
                ).get("binding_kind") == "parameter"
                and target.process_graph.G.out_degree(node_id)
            }
            if target.hierarchical_shell_composed:
                public_inputs = {
                    target.hierarchical_root_value_ids.get(
                        node_id, node_id
                    )
                    for node_id in public_inputs
                }
            if artifact is not None:
                # Only specializations originating at the shell's public
                # parameter boundary are runtime constraints.  Specialized
                # lexical constants inside nested closures are compiled facts,
                # not fabricated root inputs.
                required_inputs.update(
                    set(artifact.specialized_values) & public_inputs
                )
            if artifact is not None:
                # A source-level parameter is not necessarily a runtime shader
                # operand.  Parameters such as AbstractTensor's `like=` carry
                # construction/type context and can legitimately disappear
                # during typed lowering.  Requiring an SSBO merely because the
                # source graph retains that informational edge invents runtime
                # work and makes metadata look like executable data.  Only
                # parameters referenced by the emitted artifact belong to its
                # runtime ABI; every such operand must still be covered below.
                artifact_references = set(artifact.slot_value_ids) | set(
                    artifact.uniform_value_ids.values()
                ) | set(artifact.specialized_values)
                public_inputs &= artifact_references
            locally_produced_values = {
                int(value_id)
                for captured in target.captured_region_programs.values()
                for program in (
                    tuple(captured.stages)
                    if captured.stages
                    else (captured.program,)
                )
                for value_id in program.outputs.values()
            }
            public_outputs = set(_declared_output_terminals(
                target.process_graph,
                produced_values=locally_produced_values,
            ).values())
            if target.hierarchical_shell_composed:
                public_outputs = set(
                    target.hierarchical_public_output_ids
                )
            artifact_outputs = (
                set()
                if artifact is None
                else set(artifact.terminal_outputs.values())
            )
            hierarchy_inputs = None
            if (
                artifact is not None
                and target.hierarchical_shell_composed
            ):
                # Nested lexical specializations are compiler-owned constants
                # and are filled from the artifact.  A specialization at the
                # public parameter boundary remains part of the ABI so execute
                # can reject a caller that violates the compiled contract.
                root_runtime_inputs = (
                    required_inputs
                    - set(artifact.specialized_values)
                    | (
                        set(artifact.specialized_values)
                        & public_inputs
                    )
                )
                hierarchy_inputs = {
                    str(data.get("label")): global_id
                    for local_id, data in (
                        target.process_graph.G.nodes(data=True)
                    )
                    if data.get("type") == "Input"
                    and (
                        global_id := (
                            target.hierarchical_root_value_ids.get(
                                int(local_id)
                            )
                        )
                    ) in root_runtime_inputs
                }
                hierarchy_inputs.update({
                    str(path): int(global_id)
                    for path, global_id in (
                        target.hierarchical_root_field_value_ids.items()
                    )
                    if int(global_id) in root_runtime_inputs
                })
            blockers = []
            compile_time_only = (
                target.compiled_shell_program is not None
                and not target.compiled_shell_program.program.steps
                and not target.compiled_shell_program.program.feeds
                and not target.compiled_shell_program.program.outputs
                and (
                    target.compiled_shell_program.program.extras or {}
                ).get("kernel_kind") == "structural"
            )
            if artifact is None and not compile_time_only:
                blockers.append("no-composed-artifact")
            if (
                target.callsite_function_shells
                and not target.hierarchical_shell_composed
                and not compile_time_only
            ):
                remaining_calls = (
                    tuple(sorted(target.callsite_function_shells))
                    if target.hierarchy_remaining_callsite_ids is None
                    else target.hierarchy_remaining_callsite_ids
                )
                blockers.append(
                    "callee-artifacts-not-absorbed:"
                    + ",".join(map(str, remaining_calls))
                    + (
                        ""
                        if target.hierarchical_compose_failure is None
                        else "[compose-failure="
                        f"{target.hierarchical_compose_failure!r}]"
                    )
                )
            if (
                artifact is not None
                and not compile_time_only
                and not public_inputs.issubset(
                required_inputs
                )
            ):
                blockers.append(
                    "public-inputs-not-covered:"
                    + ",".join(map(str, sorted(
                        public_inputs - required_inputs
                    )))
                )
            if (
                artifact is not None
                and not compile_time_only
                and not public_outputs.issubset(
                artifact_outputs
                )
            ):
                blockers.append(
                    "public-outputs-not-covered:"
                    + ",".join(map(str, sorted(
                        public_outputs - artifact_outputs
                    )))
                )
            if (
                hierarchy_inputs is not None
                and not compile_time_only
                and set(hierarchy_inputs.values()) != root_runtime_inputs
            ):
                missing_hierarchy_inputs = (
                    root_runtime_inputs
                    - set(hierarchy_inputs.values())
                )
                parameter_id_map = {
                    int(local_id): int(global_id)
                    for local_id, global_id in (
                        target.hierarchical_root_value_ids.items()
                    )
                    if local_id in target.process_graph.G
                    and target.process_graph.G.nodes[local_id].get("type")
                    == "Input"
                }
                target.hierarchical_unbound_diagnostics = {
                    "missing": tuple(sorted(missing_hierarchy_inputs)),
                    "parameter_ids": parameter_id_map,
                    "field_ids": dict(
                        target.hierarchical_root_field_value_ids
                    ),
                    "private_endpoints": {
                        int(value_id): (
                            target.hierarchical_private_value_ids.get(
                                int(value_id)
                            )
                        )
                        for value_id in missing_hierarchy_inputs
                    },
                    "capture_endpoints": {
                        int(value_id): (
                            target.hierarchical_capture_value_ids.get(
                                int(value_id)
                            )
                        )
                        for value_id in missing_hierarchy_inputs
                    },
                    "endpoint_details": {
                        int(value_id): (
                            target.hierarchical_endpoint_details.get(
                                int(value_id), ()
                            )
                        )
                        for value_id in missing_hierarchy_inputs
                    },
                    "synthetic_fields": {
                        int(value_id): (
                            target.hierarchical_synthetic_field_paths.get(
                                int(value_id)
                            )
                        )
                        for value_id in missing_hierarchy_inputs
                    },
                    "correlated_endpoints": {
                        int(value_id): tuple(
                            (int(closure_id), int(local_id))
                            for closure_id, local_id, global_id
                            in (
                                target.hierarchical_effective_value_table
                                .correlations
                            )
                            if int(global_id) == int(value_id)
                        )
                        for value_id in missing_hierarchy_inputs
                    },
                    "program_origins": {
                        int(value_id): (
                            target.hierarchical_program_value_origins.get(
                                int(value_id), ()
                            )
                        )
                        for value_id in missing_hierarchy_inputs
                    },
                    "aggregate_redirects": {
                        int(value_id): tuple(
                            row
                            for rows in (
                                target
                                .hierarchical_aggregate_redirect_diagnostics
                                .values()
                            )
                            for row in rows
                            if row.get("original") == int(value_id)
                        )
                        for value_id in missing_hierarchy_inputs
                    },
                    "aggregate_closure_rows": {
                        int(closure_id): tuple(
                            target
                            .hierarchical_aggregate_redirect_diagnostics
                            .get(int(closure_id), ())
                        )
                        for closure_id in {
                            int(origin["closure"])
                            for value_id in missing_hierarchy_inputs
                            for origin in (
                                target
                                .hierarchical_program_value_origins.get(
                                    int(value_id), ()
                                )
                            )
                            if "closure" in origin
                        }
                    },
                }
                blockers.append(
                    "hierarchical-inputs-not-root-bound:"
                    + ",".join(map(str, sorted(
                        missing_hierarchy_inputs
                    )))
                    + f"[parameter-map={parameter_id_map!r},"
                    f"field-map={target.hierarchical_root_field_value_ids!r},"
                    "private="
                    f"{target.hierarchical_unbound_diagnostics['private_endpoints']!r},"
                    "capture="
                    f"{target.hierarchical_unbound_diagnostics['capture_endpoints']!r},"
                    "details="
                    f"{target.hierarchical_unbound_diagnostics['endpoint_details']!r},"
                    "fields="
                    f"{target.hierarchical_unbound_diagnostics['synthetic_fields']!r},"
                    "correlated="
                    f"{target.hierarchical_unbound_diagnostics['correlated_endpoints']!r},"
                    "program-origins="
                    f"{target.hierarchical_unbound_diagnostics['program_origins']!r},"
                    "aggregate-redirects="
                    f"{target.hierarchical_unbound_diagnostics['aggregate_redirects']!r},"
                    "aggregate-closure-rows="
                    f"{target.hierarchical_unbound_diagnostics['aggregate_closure_rows']!r},"
                    "aggregate-candidates="
                    f"{target.hierarchical_aggregate_candidate_diagnostics!r},"
                    "control-bindings="
                    f"{target.hierarchical_control_binding_diagnostics!r}]"
                )
            target.composed_shell_blockers = tuple(blockers)
            target.composed_shell_complete = not blockers
            if not target.legacy_fused_network:
                if (
                    target.composed_shell_complete
                    and artifact is not None
                    and (
                        target is self
                        or not target.hierarchical_shell_composed
                    )
                ):
                    # A hierarchically composed child is source owned by its
                    # enclosing shell, not another runtime deployment.
                    # Installing every complete subtree allocates competing
                    # resident lifetimes and turns composition back into
                    # orchestration.  Leaf artifacts remain installable for
                    # differential tests while the parent is incomplete.
                    target.install_composed_control(
                        target.composed_shell_artifact,
                        input_bindings=hierarchy_inputs,
                    )
                elif (
                    target is self
                    and not target.composed_shell_complete
                ):
                    # Nested callsite shells are planner compartments inside
                    # the one source program, not separately deployable
                    # programs.  A method subtree may legitimately expose an
                    # aggregate field or unresolved branch which its caller
                    # later binds/collapses.  Requiring every subtree to have
                    # a standalone public ABI recreates function calls and
                    # defeats whole-shell hierarchy reduction.  Preserve each
                    # child's blockers for diagnostics, but gate installation
                    # only on the compilation root after all descendants have
                    # been absorbed.
                    incomplete_shells.append((
                        target.profile_path,
                        target.composed_shell_blockers,
                    ))

            # A tape is compile-time evidence of operation order, not retained
            # runtime state.  Its nodes hold every concrete tensor produced
            # while revealing the sequence; keeping them after lowering pins
            # all corresponding GL buffers and causes GPU memory to grow until
            # the driver terminates the process.  The compiled programs above
            # contain identities, metadata, and operations only—their captured
            # feed payloads have already been discarded.
            discard_discovery_payloads(target)

        # Structural helpers may legitimately produce no tensor region and
        # therefore never enter ``targets`` above.  They still participated in
        # the one discovery execution and must lose their tape here.  Runtime
        # ownership begins only after every function shell's discovery object
        # has been destroyed.
        for target in _walk_planned_shells(
            self, include_function_registry=False
        ):
            if target._discovery_tape_creations:
                target._discovery_tape = None
                target._discovery_tape_complete = True
                assert target._discovery_tape is None
            elif target._discovery_tape is not None:
                target._discovery_tape = None
                target._discovery_tape_complete = True
                assert target._discovery_tape is None
        if self._discovery_session is not None:
            self._discovery_session.clear()
            self._discovery_session = None

        # Drop the root capture result as well.  Returning it would keep the
        # observed frame tensors alive even though capture exists solely to
        # produce the reusable programs.
        result = None
        gc.collect()
        if incomplete_shells and not self.legacy_fused_network:
            details = "; ".join(
                f"{path}: {', '.join(blockers)}"
                for path, blockers in incomplete_shells[:12]
            )
            if len(incomplete_shells) > 12:
                details += (
                    f"; ... {len(incomplete_shells) - 12} more shells"
                )
            error = RuntimeError(
                "cannot install a truthful complete GLSL shell; "
                f"{len(incomplete_shells)} function shells retain uncovered "
                f"public ABI or control: {details}"
            )
            self._profiler.record_exception(
                error,
                path=self.profile_path,
                phase="shell-completeness",
            )
            raise error
        return None

    def compile_process_graph(self, *, device=None, prepare_ephemerals=False):
        try:
            return _compile_whole_process_graph(
                self, device=device, prepare_ephemerals=prepare_ephemerals
            )
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

    def compile_discovery_program(
        self,
        *,
        dynamic_scalar_ids=(),
        strict=False,
        lowered_program=None,
        emit_glsl=True,
    ):
        projection_started = time.perf_counter()
        self._profiler.trace(
            path=self.profile_path,
            section="numerical-projection",
            label="begin",
            fields={
                "regions": len(self.dispatch_subgraphs),
                "owner": bool(self._discovery_tape_owner),
                "shared_ir": lowered_program is not None,
                "emit_glsl": bool(emit_glsl),
            },
        )
        if self._discovery_tape is None:
            raise RuntimeError(
                "capture the source shell's discovery tape before compiling"
            )
        if dynamic_scalar_ids and len(dynamic_scalar_ids) != 1:
            raise ValueError(
                "dynamic_scalar_ids must provide one sequence for the one "
                "discovery tape"
            )
        tape = self._discovery_tape
        if lowered_program is None:
            if not self._discovery_tape_owner:
                raise RuntimeError(
                    "a nested shell attempted to lower the compilation "
                    "owner's discovery tape"
                )
            if self._discovery_tape_lowerings:
                raise RuntimeError(
                    "the source program's discovery tape has already been "
                    "lowered; planner work must use compiled_shell_program"
                )
            self._discovery_tape_lowerings += 1
            if self._discovery_tape_lowerings != 1:
                raise RuntimeError(
                    "one source program lowered its discovery tape more than "
                    "once"
                )
        if not tape._nodes:
            # A structural-only shell still consumed its single discovery
            # observation, but there is no numerical program to lower or
            # fabricate.  Preserve an explicit empty IR manifest so cleanup
            # can prove the tape was consumed exactly once.
            self.compiled_shell_program = CapturedFusedProgram(
                FusedProgram(
                    version=1,
                    feeds=set(),
                    steps=[],
                    outputs={},
                    meta={},
                    extras={"kernel_kind": "structural"},
                ),
                {},
            )
            self.compiled_tapes = ()
            self.compiled_region_indices = ()
            self.compile_failures = ()
            self.glsl_sources = ()
            self.compile_time_region_indices = set(
                range(len(self.dispatch_subgraphs))
            )
            self.shell_control_program = ControlProgram(
                SequenceBlock(()),
                (),
            )
            return self
        compiled = []
        compiled_region_indices = []
        failures = []
        sources = []
        # Projection is a cut through the one observed program IR.  Never
        # substitute every planned region when a shell contributed no cut
        # points: that fabricates executions (especially beneath lazy
        # generator boundaries) and attempts to infer feeds from an example
        # that never ran.  The planner still owns those regions and the
        # completeness audit must report them until generalized control
        # lowering makes them reachable.
        subgraphs = tuple(self.forward_subgraphs)
        region_indices = tuple(self.forward_region_indices)
        captured_outputs = tuple(
            self.forward_output_values
            or (None,) * len(subgraphs)
        )
        if not (
            len(self.forward_feed_ids)
            == len(subgraphs)
            == len(region_indices)
            == len(captured_outputs)
        ):
            raise RuntimeError(
                "discovery cut-point tables have inconsistent lengths; "
                "the tape is singular while region metadata is per cut point; "
                f"shell={self.profile_path!r}; "
                f"feeds={len(self.forward_feed_ids)} "
                f"subgraphs={len(subgraphs)} "
                f"regions={len(region_indices)} "
                f"outputs={len(captured_outputs)}"
            )

        if lowered_program is not None and not subgraphs:
            # This child contributed no numerical cut point to the program's
            # one observation.  It is a structural compartment, not an owner
            # of the root numerical manifest.  Preserve an explicit empty
            # artifact so hierarchy composition can erase it without
            # pretending it lowered or executed another program.
            self.compiled_shell_program = CapturedFusedProgram(
                FusedProgram(
                    version=1,
                    feeds=set(),
                    steps=[],
                    outputs={},
                    meta={},
                    extras={"kernel_kind": "structural"},
                ),
                {},
            )
            self.compiled_tapes = ()
            self.compiled_region_indices = ()
            self.compile_failures = ()
            self.glsl_sources = ()
            self.compile_time_region_indices = set(
                range(len(self.dispatch_subgraphs))
            )
            self.shell_control_program = ControlProgram(
                SequenceBlock(()),
                (),
            )
            self._profiler.trace(
                path=self.profile_path,
                section="numerical-projection",
                label="structural complete",
                fields={
                    "elapsed_ms": round(
                        (time.perf_counter() - projection_started) * 1e3,
                        3,
                    ),
                },
            )
            return lowered_program

        # Lower the complete observed numerical program exactly once.  Nothing
        # below this call receives ``tape``: region programs are dependency
        # projections of this backend-neutral IR, never repeated tape solves.
        try:
            whole = lowered_program
            if whole is None:
                whole = compile_recorded_fused_tape(
                    tape,
                    dynamic_scalar_ids=(
                        tuple(dynamic_scalar_ids[0])
                        if dynamic_scalar_ids
                        else ()
                    ),
                )
            whole = _collapse_planned_collection_materializations(
                whole,
                self.forward_planned_collection_materializations,
            )
            # Discovery observations are diagnostic samples, not compiler
            # identity.  In particular, matching a stack's transient operand
            # IDs to observed loop items must never erase the stack producer
            # and alias its tensor result to the resident collection.  The
            # planner-owned mapping above rewires only the producer's inputs
            # and deliberately preserves its distinct result ID.
            self.collection_collapse_report = {
                int(result_id): {
                    "collection_id": int(collection_id),
                    "producer_preserved": True,
                }
                for result_id, collection_id in (
                    self.forward_planned_collection_materializations.items()
                )
            }
        except ValueError as error:
            if strict:
                raise RuntimeError(
                    "the source shell's one discovery tape did not lower to "
                    "one complete FusedProgram"
                ) from error
            self.compile_failures = ({
                "region_index": None,
                "outputs": (),
                "unsupported_ops": (),
                "reason": str(error),
            },)
            return self

        whole.program.extras = {
            **(whole.program.extras or {}),
            "capture_feed_origins": {
                feed_id: {
                    "op": (
                        None
                        if tape._nodes.get(feed_id) is None
                        else str(tape._nodes[feed_id].op)
                    ),
                    # The source parameter this feed came from, recorded
                    # when the graph's Input nodes were bound. A feed is a
                    # trace root and has no tape node, so the old lookup
                    # through tape params could only ever return empty.
                    "binding_name": _resolve_binding_name(self, whole, feed_id),
                    "parameter_names": (
                        tuple(sorted(
                            tape._nodes[feed_id].ctx.get("params") or {}
                        ))
                        if tape._nodes.get(feed_id) is not None
                        else ()
                    ),
                }
                for feed_id in whole.program.feeds
            },
        }
        boundary_aliases = {}
        for feed_map in self.forward_feed_ids:
            boundary_aliases.update(feed_map)
        self.compiled_feed_meta = {
            int(graph_id): meta
            for feed_map in self.forward_feed_ids
            for transient_id, graph_id in feed_map.items()
            if (
                meta := (whole.program.meta or {}).get(
                    int(transient_id)
                )
            ) is not None
        }
        boundary_aliases = _capture_feed_aliases(
            whole,
            boundary_aliases,
        )
        # Keep occurrence identities unique in the whole-shell IR.  A loop
        # revisits the same ProcessGraph node, so globally replacing every
        # observed result with that graph ID would create several producers
        # for one value and corrupt dependency/shape selection.  Stable graph
        # IDs are assigned only after each planner cut point is projected.
        self.compiled_shell_program = _remap_captured_all_ids(whole, {})
        # Aggregate leaf paths are identities derived while lowering the one
        # discovery tape.  Freeze only those integer/path facts into compiled
        # IR before capture payload disposal.  The tape, tensor wrappers and
        # observed values remain forbidden after this point; hierarchical
        # composition must not rediscover them or run Python again.
        compiled_feed_values = {
            int(capture_id): value
            for capture_id, value in whole.feeds.items()
        }
        capture_objects = {}

        def collect_capture_objects(value):
            if isinstance(value, dict):
                for item in value.values():
                    collect_capture_objects(item)
                return
            if isinstance(value, (tuple, list)):
                for item in value:
                    collect_capture_objects(item)
                return
            if (
                isinstance(value, AbstractTensor)
                or _capture_storage_identity(value) is not None
            ):
                capture_objects[id(value)] = value
                if isinstance(value, AbstractTensor):
                    collect_capture_objects(getattr(value, "data", None))

        for node in tape._nodes.values():
            collect_capture_objects(node.ctx.get("result"))
            collect_capture_objects(node.ctx.get("inputs", ()))
            collect_capture_objects(node.ctx.get("params") or {})
        discovery_owner = (
            None
            if self._discovery_session is None
            else self._discovery_session.get("owner")
        )
        if discovery_owner is not None:
            for planned_shell in _walk_planned_shells(
                discovery_owner, include_function_registry=False
            ):
                collect_capture_objects(
                    getattr(planned_shell, "captured_values", {})
                )
        capture_objects.update(compiled_feed_values)
        if self._discovery_session is not None:
            capture_objects.update(
                self._discovery_session.get("capture_objects") or {}
            )
        self.compiled_aggregate_feed_paths = tuple(
            tuple(dict.fromkeys(
                (
                    int(alias_id),
                    int(graph_input_id),
                    tuple(path),
                )
                for capture_id, (
                    graph_input_id,
                    path,
                    storage,
                ) in aggregate_map.items()
                for alias_id in (
                    int(capture_id),
                    *(
                        int(candidate_id)
                        for candidate_id, candidate
                        in capture_objects.items()
                        if (
                            storage is not None
                            and _capture_storage_identity(candidate) == storage
                        )
                    ),
                )
            ))
            for aggregate_map in self.forward_aggregate_feed_paths
        )

        def is_parameter_aggregate_origin(
            graph_input_id: int,
            visiting=None,
        ) -> bool:
            visiting = set() if visiting is None else set(visiting)
            graph_input_id = int(graph_input_id)
            if graph_input_id in visiting or graph_input_id not in self.process_graph.G:
                return False
            visiting.add(graph_input_id)
            data = self.process_graph.G.nodes[graph_input_id]
            attributes = data.get("attributes") or {}
            if data.get("type") == "Input":
                return attributes.get("binding_kind") == "parameter"
            expression = data.get("expr_obj")
            function_expression = getattr(expression, "func", None)
            if not (
                type(expression).__name__ == "Call"
                and type(function_expression).__name__ == "Name"
                and getattr(function_expression, "id", None)
                in {"tuple", "list"}
            ):
                return False
            source = next((
                int(parent)
                for parent, role in (data.get("parents") or ())
                if str(role) in {"arg", "args", "arg:0", "arg0"}
            ), None)
            return (
                source is not None
                and is_parameter_aggregate_origin(source, visiting)
            )

        # Convert the once-lowered program into forward SSA ownership facts
        # before region projection.  The AST planner has already decided the
        # logic and the region boundaries; the lowered IR now supplies the
        # exact numerical edges between those regions.
        #
        # Do not reconstruct these edges by inspecting captured Python
        # wrappers, object identity, storage identity, values, or autograd
        # ancestry.  A backend may legitimately replace a wrapper while
        # retaining the same SSA value, and storage may be shared or reused.
        # Both cases made real producers appear terminal and then left their
        # consumers as unbound shell inputs.  Once FusedProgram exists, its
        # integer input/result IDs are the sole numerical dependency
        # authority.  Forward capture has finished revealing primitive
        # operations at this point and is never asked to rediscover logic.
        ssa_consumers: dict[int, set[int]] = {}
        whole_result_ids: set[int] = set()
        for execution_program in whole.execution_programs:
            for step in execution_program.steps:
                consumer_id = int(step.result_id)
                whole_result_ids.add(consumer_id)
                ssa_consumers.setdefault(consumer_id, set())
                for producer_id in step.input_ids:
                    ssa_consumers.setdefault(
                        int(producer_id), set()
                    ).add(consumer_id)

        # Equal typed literals are a mathematical identity already stated by
        # the ProcessGraph itself.  AbstractTensor may materialize one backend
        # constant primitive and reuse it at several source occurrences.  Use
        # the lowest existing ProcessGraph node ID as their canonical wire;
        # never invent an ID and never inspect a captured runtime value.
        endpoint_identity: dict[int, int] = {}
        literal_ids: dict[tuple[type, Any], int] = {}
        attribute_ids: dict[tuple[int, str], int] = {}
        for node_id, data in sorted(
            self.process_graph.G.nodes(data=True),
            key=lambda item: int(item[0]),
        ):
            node_id = int(node_id)
            if data.get("type") == "Attribute":
                parents = tuple(
                    int(parent)
                    for parent, role in (data.get("parents") or ())
                    if str(role) == "value"
                )
                expression = data.get("expr_obj")
                attribute = getattr(expression, "attr", None)
                if len(parents) == 1 and attribute is not None:
                    parent = endpoint_identity.get(
                        parents[0], parents[0]
                    )
                    key = (parent, str(attribute))
                    endpoint_identity[node_id] = (
                        attribute_ids.setdefault(key, node_id)
                    )
                continue
            if data.get("type") not in {"Constant", "Const", "const"}:
                continue
            try:
                literal = _constant_value(data)
            except KeyError:
                continue
            if not isinstance(
                literal, (type(None), bool, int, float, str, bytes)
            ):
                continue
            key = (type(literal), literal)
            canonical = literal_ids.setdefault(key, node_id)
            endpoint_identity[node_id] = canonical

        for (
            feed_ids,
            aggregate_feed_paths,
            region_capture_node_ids,
            region_planned_capture_ids,
            region_planned_input_ids,
            subgraph,
            region_index,
            captured_output_values,
        ) in zip(
            self.forward_feed_ids,
            self.compiled_aggregate_feed_paths,
            self.forward_region_capture_node_ids,
            self.forward_region_planned_capture_ids,
            self.forward_region_planned_input_ids,
            subgraphs,
            region_indices,
            captured_outputs,
        ):
            output_ids = tuple(
                subgraph.G.graph["deployment_outputs"]
            )
            parameter_aggregate_capture_ids = tuple(
                int(capture_id)
                for capture_id, input_id, _path in aggregate_feed_paths
                if is_parameter_aggregate_origin(input_id)
            )
            owned_result_ids = tuple(dict.fromkeys(
                int(value_id)
                for value_id in region_capture_node_ids
                if (
                    int(value_id) in tape._nodes
                    and int(value_id) in whole_result_ids
                )
            ))
            if not owned_result_ids:
                continue
            owned_set = set(owned_result_ids)
            # Generated region code reports this correspondence at the exact
            # planned operation invocation.  It is tab-A-to-slot-B wiring:
            # no wrapper/storage/value comparison and no reconstruction from
            # tape topology is involved.
            direct_output_ids = {
                int(capture_id): int(graph_node_id)
                for graph_node_id, capture_ids
                in region_planned_capture_ids
                if int(graph_node_id) in output_ids
                for capture_id in capture_ids
            }
            # Apply ProcessGraph identities only after the planner-owned
            # region has been projected from the complete primitive program.
            # Local node IDs are scoped to a function/closure and therefore
            # cannot safely be substituted into the whole discovery capture:
            # unrelated compartments routinely reuse the same integers.
            #
            # The positional rows and result rows below are emitted at the
            # exact generated ProcessGraph operation.  They are the complete
            # tab-A-to-slot-B correspondence for this compartment.  Remapping
            # both sides together also keeps literal tensor constructors and
            # intermediate results as real producers instead of fabricating
            # their ProcessGraph IDs as external shell inputs.
            planned_region_id_map: dict[int, int] = {}

            def compiled_alias_identity(graph_id: int) -> int:
                current = int(graph_id)
                seen = set()
                while (
                    current not in seen
                    and current in self.compiled_process_graph_aliases
                ):
                    seen.add(current)
                    current = int(
                        self.compiled_process_graph_aliases[current]
                    )
                return current

            def add_planned_id(primitive_id: int, graph_id: int) -> None:
                primitive_id = int(primitive_id)
                graph_id = compiled_alias_identity(graph_id)
                graph_id = endpoint_identity.get(
                    int(graph_id), int(graph_id)
                )
                previous = planned_region_id_map.get(
                    primitive_id, graph_id
                )
                if previous != graph_id:
                    raise RuntimeError(
                        "one primitive occurrence was assigned to two "
                        "ProcessGraph endpoints in region "
                        f"{region_index}: primitive={primitive_id}, "
                        f"first={previous}, second={graph_id}, "
                        f"function={self.process_graph.G.graph.get('function_name', '?')!r}, "
                        f"primitive_node={tape._nodes.get(primitive_id)!r}, "
                        f"capture_rows={region_planned_capture_ids!r}, "
                        f"input_rows={region_planned_input_ids!r}, "
                        f"first_node={dict(self.process_graph.G.nodes[previous]) if previous in self.process_graph.G else None!r}, "
                        f"second_node={dict(self.process_graph.G.nodes[graph_id]) if graph_id in self.process_graph.G else None!r}"
                    )
                planned_region_id_map[primitive_id] = graph_id

            for graph_node_id, capture_ids in region_planned_capture_ids:
                for capture_id in capture_ids:
                    add_planned_id(capture_id, graph_node_id)
            for _result_capture_id, positional_inputs in (
                region_planned_input_ids
            ):
                for primitive_id, graph_id in positional_inputs:
                    previous = planned_region_id_map.get(int(primitive_id))
                    previous_node = (
                        self.process_graph.G.nodes[int(previous)]
                        if previous is not None
                        and int(previous) in self.process_graph.G
                        else {}
                    )
                    if (
                        int(primitive_id) == int(_result_capture_id)
                        and previous is not None
                        and int(previous) != int(graph_id)
                        and previous_node.get("type")
                        not in {"Store", "IndexedStore"}
                    ):
                        # Object-keyed capture represents an identity adapter
                        # with the same transient ID at its input and result.
                        # Keep that transient ID wired to the real input and
                        # publish the planned result endpoint as an SSA alias.
                        self.compiled_process_graph_aliases[
                            int(previous)
                        ] = int(graph_id)
                        planned_region_id_map[int(primitive_id)] = int(
                            graph_id
                        )
                        continue
                    add_planned_id(primitive_id, graph_id)
            live_result_ids = tuple(
                value_id
                for value_id in owned_result_ids
                if (
                    value_id in direct_output_ids
                    or not ssa_consumers.get(value_id)
                    or any(
                        consumer_id not in owned_set
                        for consumer_id in ssa_consumers.get(value_id, ())
                    )
                )
            )
            if not live_result_ids:
                # This region's numerical definitions are dead within its
                # own AST compartment.  It needs no shader region, and no
                # observed payload or host fallback is retained.
                continue
            try:
                captured = _project_captured_program(
                    whole,
                    output_ids=live_result_ids,
                    boundary_ids=tuple(dict.fromkeys((
                        *feed_ids,
                        *parameter_aggregate_capture_ids,
                    ))),
                    # AST/ProcessGraph planning has already assigned logic and
                    # scope before the tape is lowered.  The tape contributes
                    # only the tensor operations observed while this region
                    # executed; it is not allowed to walk backward into an
                    # operation owned by a caller, callee, sibling, or loop
                    # compartment merely because lazy materialization joined
                    # their backend expressions.  Repeated occurrences are
                    # accumulated into this one region-owned set above, so
                    # planner-owned loops remain complete without allowing
                    # forward capture to rediscover control flow.
                    allowed_result_ids=tuple(region_capture_node_ids),
                )
                captured = _remap_captured_program(
                    captured,
                    feed_ids={
                        **{
                            transient_id: graph_id
                            for transient_id, graph_id
                            in boundary_aliases.items()
                            if (
                                transient_id in captured.program.feeds
                                or transient_id in feed_ids
                            )
                        },
                        # Exact planned correspondences override legacy
                        # boundary aliases.  The latter remain temporarily
                        # below for untouched aggregate paths, but may never
                        # override an explicit ProcessGraph endpoint.
                        **planned_region_id_map,
                    },
                    output_ids=tuple(
                        direct_output_ids.get(value_id, value_id)
                        for value_id in live_result_ids
                    ),
                )
                canonical_literals = set(literal_ids.values())
                for program in captured.execution_programs:
                    metadata = program.meta or {}
                    for literal_id in canonical_literals:
                        meta = metadata.get(int(literal_id))
                        if meta is None:
                            continue
                        metadata[int(literal_id)] = Meta(
                            (),
                            meta.dtype,
                            meta.device,
                        )
            except ValueError as error:
                if strict:
                    raise RuntimeError(
                        "ProcessGraph numerical region projection "
                        f"{self.process_graph.G.graph.get('function_name', '?')} "
                        f"region {region_index} did not project to one "
                        f"CapturedFusedProgram; operations="
                        f"{tuple(str(subgraph.G.nodes[node].get('op') or subgraph.G.nodes[node].get('type')) for node in subgraph.G.graph.get('deployment_nodes', ()))!r}; "
                        "owned_results="
                        f"{owned_result_ids!r}; live_results="
                        f"{live_result_ids!r}; direct_outputs="
                        f"{tuple(sorted(direct_output_ids.items()))!r}"
                    ) from error
                failures.append({
                    "region_index": region_index,
                    "outputs": tuple(
                        output_ids
                    ),
                    "unsupported_ops": (),
                    "reason": str(error),
                })
                continue
            compiled.append(captured)
            compiled_region_indices.append(region_index)
            if not (
                self._discovery_tape_owner
                and not self.callsite_function_shells
            ):
                # This projected region belongs to a compartment inside a
                # larger source shell.  Keep the CapturedFusedProgram IR for
                # hierarchical composition, but do not emit a throwaway
                # standalone shader.  The one root artifact emits every
                # snippet in its real control/dependency context and is the
                # authoritative GLSL validation point.
                continue
            if not emit_glsl:
                continue
            try:
                sources.append(compile_captured_fused_program(captured))
            except Exception as error:
                stage_shapes = tuple(
                    {
                        "steps": tuple(
                            (
                                step.op_name,
                                tuple(step.input_ids),
                                int(step.result_id),
                            )
                            for step in stage.steps
                        ),
                        "feeds": {
                            int(value_id): (
                                None
                                if (stage.meta or {}).get(value_id) is None
                                else tuple(
                                    (stage.meta or {})[value_id].shape or ()
                                )
                            )
                            for value_id in stage.feeds
                        },
                        "outputs": {
                            name: (
                                int(value_id),
                                None
                                if (stage.meta or {}).get(value_id) is None
                                else tuple(
                                    (stage.meta or {})[value_id].shape or ()
                                ),
                            )
                            for name, value_id in stage.outputs.items()
                        },
                    }
                    for stage in captured.execution_programs
                )
                raise RuntimeError(
                    "failed to compile captured GLSL source for ProcessGraph "
                    f"{self.process_graph.G.graph.get('function_name', '?')} "
                    f"region {region_index}; operations="
                    f"{tuple(str(subgraph.G.nodes[node].get('op') or subgraph.G.nodes[node].get('type')) for node in subgraph.G.graph.get('deployment_nodes', ()))!r}; "
                    f"projected_stages={stage_shapes!r}"
                ) from error
        self.compiled_tapes = tuple(compiled)
        self.compiled_region_indices = tuple(compiled_region_indices)
        self.compile_failures = tuple(failures)
        self.glsl_sources = tuple(sources)
        self._profiler.trace(
            path=self.profile_path,
            section="numerical-projection",
            label="complete",
            fields={
                "compiled_regions": len(compiled_region_indices),
                "elapsed_ms": round(
                    (time.perf_counter() - projection_started) * 1e3,
                    3,
                ),
            },
        )
        if self._discovery_tape_owner:
            self._discovery_session["lowered_program"] = whole
        by_region = {
            int(
                region_index
                if not isinstance(region_index, tuple)
                else region_index[-2]
            ): captured
            for region_index, captured in zip(
                compiled_region_indices,
                compiled,
            )
        }
        composed_loops = {}
        for reduction in self.loop_shader_reductions:
            if not emit_glsl:
                continue
            if not reduction.collapsible or reduction.control_program is None:
                continue
            missing = tuple(
                region_index
                for region_index in reduction.region_indices
                if region_index not in by_region
            )
            if missing:
                self._profiler.trace(
                    path=self.profile_path,
                    section="control-lowering",
                    label=f"loop {reduction.loop_node_id} not composed",
                    fields={
                        "reason": "regions have no captured GLSL program",
                        "regions": missing,
                    },
                )
                continue
            loop_regions = {
                region_index: by_region[region_index]
                for region_index in reduction.region_indices
            }
            available_loop_values = {
                int(value_id)
                for captured in loop_regions.values()
                for program in (
                    captured.program,
                    *tuple(captured.stages),
                )
                for value_id in (
                    *tuple(program.feeds),
                    *tuple(program.outputs.values()),
                    *(step.result_id for step in program.steps),
                )
            }
            deferred_sources = tuple(
                int(source_id)
                for source_id, _collection_id, _induction, _start
                in reduction.control_program.collection_bindings
                if int(source_id) not in available_loop_values
            )
            if deferred_sources:
                self._profiler.trace(
                    path=self.profile_path,
                    section="control-lowering",
                    label=(
                        f"loop {reduction.loop_node_id} deferred "
                        "to hierarchy"
                    ),
                    fields={
                        "collection_sources": deferred_sources,
                    },
                )
                continue
            artifact = build_control_shader_artifact(
                reduction.control_program,
                loop_regions,
                value_meta=_captured_storage_meta(self.captured_values),
                instrumentation=self._profiler.verbose,
            )
            composed_loops[reduction.loop_node_id] = artifact.source
            self.composed_loop_artifacts[reduction.loop_node_id] = artifact
        self.composed_loop_sources = composed_loops
        # A ProcessGraph deployment region is not automatically a runtime
        # shader.  Shape/type predicates and other structural calculations can
        # be fully resolved while discovering the program and produce no
        # tensor output.  Requiring such a region to fabricate an SSBO result
        # prevents the real numerical regions from composing and resurrects a
        # host coordinator for work that no longer exists.
        #
        # Only tensor-producing regions enter the shader schedule.  The
        # capture_fused_programs completeness audit below independently rejects
        # any omitted region that actually has a tensor output, so this cannot
        # hide failed numerical lowering.
        complete_regions = tuple(
            region_index
            for region_index in range(len(self.dispatch_subgraphs))
            if region_index in by_region
        )
        self.compile_time_region_indices = (
            set(range(len(self.dispatch_subgraphs))) - set(complete_regions)
        )
        retained_value_ids = {
            int(value_id)
            for captured in by_region.values()
            for program in (
                tuple(captured.stages)
                if captured.stages
                else (captured.program,)
            )
            for value_id in (
                *ordered_feed_ids(program),
                *tuple(program.outputs.values()),
            )
        }
        all_loops_lowered = all(
            not reduction.region_indices
            or (
                reduction.control_program is not None
                and (
                    not emit_glsl
                    or reduction.collapsible
                )
            )
            for reduction in self.loop_shader_reductions
        )
        if by_region and all_loops_lowered:
            control_compose_started = time.perf_counter()
            self._profiler.trace(
                path=self.profile_path,
                section="control-compose",
                label="overlay begin",
                fields={
                    "regions": len(complete_regions),
                    "loops": len(self.loop_shader_reductions),
                },
            )
            considered_reductions = tuple(
                reduction
                for reduction in self.loop_shader_reductions
                if reduction.control_program is not None
                and (
                    not emit_glsl
                    or reduction.collapsible
                )
            )
            conditional_controls = _ordinary_conditional_control_programs(
                self.process_graph,
                complete_regions,
                self.dispatch_subgraphs,
            )
            if len(conditional_controls) > 1:
                conditional_controls = ()
            shell_control = overlay_scheduled_control(
                complete_regions,
                tuple(
                    project_control_regions(
                        reduction.control_program,
                        complete_regions,
                        retained_value_ids=retained_value_ids,
                    )
                    for reduction in considered_reductions
                ) + conditional_controls,
                known_nesting=_loop_reduction_nesting_hints(
                    considered_reductions,
                    self.loop_plans,
                    self.process_graph,
                ),
            )
            shell_control = replace(
                shell_control,
                deployment_regions=tuple(dict.fromkeys((
                    *shell_control.deployment_regions,
                    *self.control_deployment_regions,
                ))),
            )
            validations = _validation_control_blocks(self)
            if validations:
                shell_control = replace(
                    shell_control,
                    root=SequenceBlock((
                        shell_control.root,
                        *validations,
                    )),
                )
            shell_control = project_control_regions(
                shell_control,
                complete_regions,
                retained_value_ids=retained_value_ids,
            )
            if self.compiled_process_graph_aliases:
                shell_control = replace(
                    shell_control,
                    value_aliases=tuple(dict.fromkeys((
                        *shell_control.value_aliases,
                        *(
                            (int(alias), int(owner))
                            for alias, owner in
                            self.compiled_process_graph_aliases.items()
                        ),
                    ))),
                )
            self.shell_control_program = shell_control
            self._profiler.trace(
                path=self.profile_path,
                section="control-compose",
                label="overlay complete",
                fields={
                    "elapsed_ms": round(
                        (
                            time.perf_counter()
                            - control_compose_started
                        ) * 1e3,
                        3,
                    ),
                },
            )
            produced_values = {
                int(value_id)
                for captured in by_region.values()
                for program in (
                    tuple(captured.stages)
                    if captured.stages
                    else (captured.program,)
                )
                for value_id in program.outputs.values()
            }
            declared_terminals = _declared_output_terminals(
                self.process_graph,
                produced_values=produced_values,
            )
            used_value_ids = {
                int(value_id)
                for captured in by_region.values()
                for program in (
                    tuple(captured.stages)
                    if captured.stages
                    else (captured.program,)
                )
                for value_id in (
                    *program.feeds,
                    *program.outputs.values(),
                )
            }
            specialized_values = {
                int(node_id): self.captured_values[node_id]
                for node_id, data in self.process_graph.G.nodes(data=True)
                if data.get("type") == "Input"
                and str(data.get("label")) not in (
                    self.process_graph.G.graph.get(
                        "planner_specializations", {}
                    )
                )
                and (
                    data.get("attributes") or {}
                ).get("value_kind") == "scalar"
                and node_id not in used_value_ids
                and node_id not in {
                    int(uniform.value_id)
                    for uniform in shell_control.uniforms
                }
                and node_id in self.captured_values
            }
            self.composed_shell_specialized_values = dict(
                specialized_values
            )
            # A child callsite is an IR compartment awaiting absorption into
            # its source program's root hierarchy.  Finalizing it here would
            # generate a shader that is never deployed, then repeat that work
            # for every ancestor.  Only a standalone leaf source is complete
            # at this level; a root with callees is finalized once by
            # _build_hierarchical_glsl_artifact after every compartment has
            # supplied control and numerical IR.
            if (
                emit_glsl
                and self._discovery_tape_owner
                and not self.callsite_function_shells
            ):
                artifact_started = time.perf_counter()
                self._profiler.trace(
                    path=self.profile_path,
                    section="control-compose",
                    label="artifact begin",
                    fields={
                        "regions": len(complete_regions),
                        "terminals": len(declared_terminals),
                    },
                )
                self.composed_shell_artifact = build_control_shader_artifact(
                    shell_control,
                    {
                        region_index: by_region[region_index]
                        for region_index in complete_regions
                    },
                    value_meta=_captured_storage_meta(self.captured_values),
                    instrumentation=self._profiler.verbose,
                    terminal_outputs=(
                        declared_terminals or None
                    ),
                    specialized_values=specialized_values,
                    device_resident=(self.shell_language == "glsl"),
                )
                self._profiler.trace(
                    path=self.profile_path,
                    section="control-compose",
                    label="artifact complete",
                    fields={
                        "elapsed_ms": round(
                            (
                                time.perf_counter() - artifact_started
                            ) * 1e3,
                            3,
                        ),
                    },
                )
        return whole

    def install_composed_control(
        self,
        artifact,
        *,
        input_bindings=None,
        output_bindings=None,
    ):
        if self.installed_control_shell is not None:
            self.installed_control_shell.release()
        try:
            installed = InstalledGLSLControlShell(artifact)
        except Exception as error:
            raise RuntimeError(
                "failed to install composed GLSL closure "
                f"{self.profile_path}"
            ) from error
        public_parameter_ids = {
            int(node_id)
            for node_id, data in self.process_graph.G.nodes(data=True)
            if data.get("type") == "Input"
            and (
                data.get("attributes") or {}
            ).get("binding_kind") == "parameter"
            and self.process_graph.G.out_degree(node_id)
        }
        if self.hierarchical_shell_composed:
            public_parameter_ids = {
                self.hierarchical_root_value_ids.get(node_id, node_id)
                for node_id in public_parameter_ids
            }
        specialized_inputs = (
            set(artifact.specialized_values) & public_parameter_ids
        )
        inferred_inputs = {
            (
                str(self.process_graph.G.nodes[value_id].get("label"))
                if value_id in self.process_graph.G
                and self.process_graph.G.nodes[value_id].get("label")
                else f"value_{value_id}"
            ): value_id
            for value_id in (
                *artifact.external_value_ids,
                *artifact.uniform_value_ids.values(),
                *specialized_inputs,
            )
        }
        self.input_bindings = dict(
            inferred_inputs if input_bindings is None else input_bindings
        )
        required_inputs = (
            set(artifact.external_value_ids)
            | (
                set(artifact.uniform_value_ids.values())
                - set(artifact.specialized_values)
            )
            | specialized_inputs
        )
        if set(self.input_bindings.values()) != required_inputs:
            installed.release()
            raise ValueError(
                "composed-control input bindings must cover every external "
                "buffer and dynamic control uniform exactly once; "
                f"declared={tuple(sorted(self.input_bindings.values()))!r}; "
                f"required={tuple(sorted(required_inputs))!r}; "
                f"missing={tuple(sorted(required_inputs - set(self.input_bindings.values())))!r}; "
                f"extra={tuple(sorted(set(self.input_bindings.values()) - required_inputs))!r}"
            )
        terminal_by_value = {
            int(value_id): terminal_name
            for terminal_name, value_id
            in artifact.terminal_outputs.items()
        }
        identity_table = (
            self.process_graph.G.graph.get("identity_table") or {}
        )
        inferred_outputs = {}
        for public_name in (
            self.process_graph.G.graph.get("function_outputs") or ()
        ):
            identities = tuple(identity_table.get(public_name, ()))
            if not identities:
                continue
            candidate_values = tuple(
                int(
                    self.hierarchical_root_value_ids.get(
                        int(value_id),
                        int(value_id),
                    )
                )
                if self.hierarchical_shell_composed
                else int(value_id)
                for value_id in reversed(identities)
            )
            # A retained loop can append a structural result port after the
            # numerical value actually published by the artifact.  Select the
            # newest identity that is a real terminal instead of discarding
            # the public name merely because the final structural alias is
            # not itself a shader slot.
            terminal_name = next(
                (
                    terminal_by_value[value_id]
                    for value_id in candidate_values
                    if value_id in terminal_by_value
                ),
                None,
            )
            if terminal_name is not None:
                inferred_outputs[str(public_name)] = terminal_name
        if not inferred_outputs:
            inferred_outputs = {
                name: name for name in artifact.terminal_outputs
            }
        self.output_bindings = dict(
            inferred_outputs if output_bindings is None else output_bindings
        )
        self.output_aggregate_bindings = {}
        self.output_loop_aggregate_bindings = {}
        for public_name in (
            self.process_graph.G.graph.get("function_outputs") or ()
        ):
            identities = tuple(identity_table.get(public_name, ()))
            output_id = int(identities[-1]) if identities else None
            output_attributes = (
                self.process_graph.G.nodes[output_id].get("attributes") or {}
                if output_id is not None
                and output_id in self.process_graph.G
                else {}
            )
            if (
                output_attributes.get("materialization_kind")
                == "retained_loop_aggregate"
            ):
                terminal_names = []
                for leaf_id in output_attributes.get(
                    "materialized_value_ids", ()
                ):
                    value_id = int(leaf_id)
                    if self.hierarchical_shell_composed:
                        value_id = int(
                            self.hierarchical_root_value_ids.get(
                                value_id,
                                value_id,
                            )
                        )
                    terminal_name = terminal_by_value.get(value_id)
                    if terminal_name is None:
                        break
                    terminal_names.append(terminal_name)
                else:
                    if terminal_names:
                        self.output_loop_aggregate_bindings[
                            str(public_name)
                        ] = (
                            int(output_attributes.get(
                                "loop_aggregate_axis", 0
                            )),
                            tuple(terminal_names),
                        )
                        continue
            prefix = f"{public_name}."
            leaves = []
            for terminal_name in artifact.terminal_outputs:
                if not str(terminal_name).startswith(prefix):
                    continue
                path = tuple(
                    int(part) if part.isdigit() else part
                    for part in str(terminal_name)[len(prefix):].split(".")
                )
                leaves.append((path, str(terminal_name)))
            if leaves:
                self.output_aggregate_bindings[str(public_name)] = tuple(
                    leaves
                )
        unknown = (
            set(self.output_bindings.values())
            - set(artifact.terminal_outputs)
        )
        if unknown:
            installed.release()
            raise ValueError(
                "composed-control output bindings name unknown terminals: "
                + ", ".join(sorted(map(str, unknown)))
            )
        self.installed_control_shell = installed
        transient_callables = tuple(self.ephemeral_callables)
        for transient in transient_callables:
            transient.discard_python_callable()
            assert not hasattr(transient, "_callable")
        self.ephemeral_callables = ()
        self.compiled_dispatch_functions = ()
        self.compiled_shell_callable = installed.execute
        assert callable(self.compiled_shell_callable)
        return self

    def compile_forward_tapes(self, *args, **kwargs):
        raise RuntimeError(
            "compile_forward_tapes is retired: a source shell owns one "
            "discovery tape, which compile_discovery_program lowers once"
        )

    def install_fused_programs(
        self,
        programs,
        *,
        fifo_slots=2,
        input_bindings=None,
        output_bindings=None,
    ):
        if not self.legacy_fused_network:
            raise RuntimeError(
                "GLSLFusedProgramNetwork is retired by the "
                "composed-control runtime flag. Install the planner-composed "
                "control artifact; pass legacy_fused_network=True only for "
                "explicit transition diagnostics."
            )
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
        # ------------------------------------------------------------------
        # NON-NEGOTIABLE COMPILED-SHELL INVARIANT
        #
        # ProcessGraph-derived Python callables exist only to *make* the
        # compiled program.  They are compile-time lowering instruments.  They
        # are not the deployed program, they are not an acceptable fallback,
        # and executing them after compilation is not compiled-shell behavior.
        #
        # Agents have repeatedly violated this invariant by retaining these
        # Python objects, running them through a Python coordinator, setting a
        # flag named ``whole_program_compiled``, and then claiming that the
        # shell was compiled.  That claim is false.  Producing correct output,
        # or dispatching compiled numerical fragments from Python, does not
        # make the controlling shell a compiled program.
        #
        # Anyone who changes this code to bypass, weaken, mock, or remove the
        # deletion and assertions below while continuing to claim
        # compiled-shell execution is lying about what the program does and
        # is ruining the compiler boundary this code exists to enforce.  This
        # has happened repeatedly and it must stop now.
        #
        # The proof is deliberately mechanical:
        #
        #   1. Every transient Python callable is physically discarded.
        #   2. The deployment retains no ephemeral callable collection.
        #   3. A real installed compiled-shell callable exists.
        #
        # Do not replace deletion with ``None`` while retaining another
        # reference.  Do not rename a Python coordinator and store it in
        # ``compiled_shell_callable``.  Do not make the assertions conditional.
        # Fix the compiler if these assertions fail.
        #
        # Likewise, do not evade these assertions earlier in the pipeline by
        # wrapping a tensor algorithm in an opaque host-boundary call.  A
        # Python codec that internally performs thousands of tensor/GLSL
        # operations is not one compiled operation.  Marking it external only
        # blinds the planner, leaves the work eager, and falsely makes the
        # remaining graph look compiled.  The numerical topology must remain
        # visible through planning and lowering; only the true terminal host
        # effect may cross the boundary.
        # ------------------------------------------------------------------
        transient_callables = tuple(self.ephemeral_callables)
        for transient in transient_callables:
            transient.discard_python_callable()
            assert not hasattr(transient, "_callable")
        self.ephemeral_callables = ()
        self.compiled_dispatch_functions = ()
        self.compiled_shell_callable = self.fused_network.execute

        # These assertions are the executable definition of the deployment
        # boundary described above.  A failure means compilation is incomplete;
        # it must never be hidden by falling back to ProcessGraph execution.
        assert not self.ephemeral_callables
        assert callable(self.compiled_shell_callable)
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
        if (
            not self.legacy_fused_network
            and self.installed_control_shell is None
        ):
            raise RuntimeError(
                "composed GLSL execution requires the installed compiled "
                "shell; consume the program's one discovery tape with "
                "capture_fused_programs() before runtime"
            )
        if not self.ready:
            self.compile_process_graph()
        self.require_ready()
        if self.installed_control_shell is not None:
            return self.installed_control_shell.execute(feeds)
        if self.whole_program_compiled:
            return self.execute_process_graph(feeds)
        return self.fused_network.execute(feeds)

    def execute_named(self, feeds):
        if (
            not self.legacy_fused_network
            and self.installed_control_shell is None
        ):
            raise RuntimeError(
                "composed GLSL execution requires the installed compiled "
                "shell; consume the program's one discovery tape with "
                "capture_fused_programs() before runtime"
            )
        if not self.ready:
            self.compile_process_graph()
        self.require_ready()
        if self.installed_control_shell is not None:
            unknown = set(feeds) - set(self.input_bindings)
            if unknown:
                raise KeyError(
                    "unknown deployment inputs: "
                    + ", ".join(sorted(map(str, unknown)))
                )
            token = self._profiler.begin_shell(self.profile_path)
            started_ns = time.perf_counter_ns()
            try:
                installed = self.installed_control_shell
                outputs = installed.execute({
                    self.input_bindings[name]: value
                    for name, value in feeds.items()
                })
                gpu_ms = installed.last_gpu_ms
                dispatches = installed.last_dispatches
                self._profiler.record_device_trace(
                    path=self.profile_path,
                    records=installed.last_debug_records,
                    header=installed.last_debug_header,
                )
                streamed = {}
                if installed.artifact.stream_outputs:
                    while True:
                        drained = installed.drain_stream()
                        for item in drained:
                            streamed.setdefault(
                                int(item["stream_id"]),
                                [],
                            ).append(
                                item["values"]
                                if getattr(
                                    item["values"], "nbytes", 0
                                )
                                or not item["words"].size
                                else item["words"].astype(
                                    "uint8",
                                    copy=False,
                                )
                            )
                        if installed.last_stream_status != 1:
                            break
                        if not drained:
                            raise RuntimeError(
                                "compiled GLSL shell suspended for "
                                "downstream capacity without publishing a "
                                "drainable resident prefix"
                            )
                        outputs = installed.resume()
                        gpu_ms += installed.last_gpu_ms
                        dispatches += installed.last_dispatches
                        self._profiler.record_device_trace(
                            path=self.profile_path,
                            records=installed.last_debug_records,
                            header=installed.last_debug_header,
                        )
                self._profiler.record(
                    path=self.profile_path,
                    section="compiled-glsl",
                    label="composed-shell-dispatch",
                    cpu_ms=(
                        time.perf_counter_ns() - started_ns
                    ) / 1e6,
                    gpu_ms=gpu_ms,
                    dispatches=dispatches,
                )
                named_outputs = {
                    public_name: outputs[terminal_name]
                    for public_name, terminal_name
                    in self.output_bindings.items()
                }
                for public_name, leaves in (
                    self.output_aggregate_bindings.items()
                ):
                    tree = {}
                    for path, terminal_name in leaves:
                        current = tree
                        for part in path[:-1]:
                            current = current.setdefault(part, {})
                        current[path[-1]] = outputs[terminal_name]

                    def freeze_aggregate(value):
                        if not isinstance(value, dict):
                            return value
                        if value and all(
                            isinstance(key, int) for key in value
                        ):
                            return tuple(
                                freeze_aggregate(value[index])
                                for index in range(max(value) + 1)
                            )
                        return {
                            key: freeze_aggregate(item)
                            for key, item in value.items()
                        }

                    named_outputs[public_name] = freeze_aggregate(tree)
                for public_name, (
                    axis,
                    terminal_names,
                ) in self.output_loop_aggregate_bindings.items():
                    leaf_values = [
                        outputs[terminal_name]
                        for terminal_name in terminal_names
                    ]
                    leaf_arrays = [
                        (
                            value.numpy()
                            if callable(getattr(value, "numpy", None))
                            else np.asarray(value)
                        )
                        for value in leaf_values
                    ]
                    extents = {
                        int(array.shape[axis])
                        for array in leaf_arrays
                    }
                    if len(extents) != 1:
                        raise RuntimeError(
                            "retained loop aggregate leaves disagree on "
                            f"axis {axis} extent: {tuple(sorted(extents))!r}"
                        )
                    named_outputs[public_name] = tuple(
                        tuple(
                            np.take(array, iteration, axis=axis)
                            for array in leaf_arrays
                        )
                        for iteration in range(next(iter(extents)))
                    )
                if installed.artifact.stream_outputs:
                    for public_name, stream_id in (
                        installed.artifact.stream_outputs.items()
                    ):
                        named_outputs[str(public_name)] = tuple(
                            streamed.get(int(stream_id), ())
                        )
                return named_outputs
            finally:
                self._profiler.end_shell(self.profile_path, token)
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
        if self.installed_control_shell is not None:
            self.installed_control_shell.release()
            self.installed_control_shell = None
        if self.fused_network is not None:
            self.fused_network.release()
            self.fused_network = None
        if self._owns_function_shells:
            for shell in _walk_planned_shells(self):
                if shell is self:
                    continue
                shell.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.release()
        return False


def _is_runtime_value_id(value_id: Any) -> bool:
    """Whether ``value_id`` names a runtime value node (an integer id or its
    digit spelling). A compile-time reference (a resolved type/module left as a
    _StaticPythonReference) is not a runtime value id -- it is a compile-time
    constant, and must be excluded from runtime value-id sets rather than forced
    through ``int()``."""

    if isinstance(value_id, bool):
        return False
    if isinstance(value_id, int):
        return True
    if isinstance(value_id, str):
        return value_id.lstrip("-").isdigit()
    return False


def strategize_shell_deployment(
    graph: Any,
    *,
    # Was 256, which was a GLSL dispatch-sizing constraint: a shader had to
    # be split into chunks a driver would accept. That is no longer what the
    # number means now that emission goes SPIR-V -> GLSL, and for a target
    # that wants one flat program it was actively harmful -- a program past
    # the bound was split into regions, so a caller asking for a flat
    # unrolled loop silently got a retained one instead.
    max_nodes_per_dispatch: int = 65536,
    backend: str | None = None,
    remove_loops: bool | None = None,
    unroll_limit: int | None = None,
    schedule_preference: str | None = None,
    runtime_closure_only: bool = False,
    _function_table_stack: tuple[int, ...] = (),
) -> type:
    """Build a stateful shell around the graph's flat dispatch schedule.

    This is the compilation choke point: every backend -- c, python, glsl,
    fortran, webgl, webgpu -- funnels its ProcessGraph through this one
    control-planning stage before any backend-specific emission happens.
    The name is generic on purpose; nothing below is GLSL-specific.

    ``backend`` only tags the loop-composition capability profile below --
    it does not change GLSL emission, which stays gated behind
    ``capture_fused_programs(precompile_only=...)`` regardless of this
    argument. Callers that only want the FusedProgram/ControlProgram/SSA
    stages (any backend) should pass ``precompile_only=True`` there and
    never reach GLSL-specific emission at all.

    ``remove_loops`` disables native loop retention, so
    ``evaporate_unrolled_loops`` unrolls every discovered loop into a flat
    instruction sequence instead of a real ``LoopBlock``.

    ``unroll_limit`` is the largest trip count that may be unrolled.  It is a
    representation-choice threshold, never an execution bound.  A loop above
    the threshold is retained with its complete source domain; if a target
    cannot lower retained control, compilation must fail visibly rather than
    dropping iterations.
    """

    # Inherit whatever the compilation asked for, then record it so the
    # function shells planned from subgraphs of this one see the same thing.
    # Without this a nested function -- which is where a loop usually lives,
    # since the entrypoint is a thin wrapper -- silently planned with the
    # defaults, and a caller asking for a flat program got a retained loop.
    inherited = dict(graph.G.graph.get("loop_settings") or {})
    backend = inherited.get("backend", "glsl") if backend is None else backend
    remove_loops = (
        inherited.get("remove_loops", False) if remove_loops is None
        else remove_loops
    )
    unroll_limit = (
        inherited.get("unroll_limit", 8) if unroll_limit is None
        else unroll_limit
    )
    schedule_preference = (
        inherited.get("schedule_preference", "alap")
        if schedule_preference is None else schedule_preference
    )
    schedule_preference = str(schedule_preference).lower()
    if schedule_preference not in {"asap", "alap"}:
        raise ValueError(
            "deployment schedule preference must be 'asap' or 'alap'"
        )
    unroll_limit = max(1, min(int(unroll_limit), MAX_UNROLL_LIMIT))
    graph.G.graph["loop_settings"] = {
        "backend": backend,
        "remove_loops": bool(remove_loops),
        "unroll_limit": int(unroll_limit),
        "schedule_preference": schedule_preference,
    }
    graph.G.graph["deployment_schedule_preference"] = schedule_preference

    _resolve_bound_function_references(graph)
    _propagate_callsite_planner_specializations(graph)
    _propagate_callsite_tensor_specializations(graph)
    if graph.G.graph.get("planner_specializations"):
        _fold_callsite_structural_values(graph)
    canonical_value_ids = bool(
        graph.G.graph.get("canonical_value_ids")
    )
    loop_composer = LoopComposer(
        LoopBackendCapabilities(
            backend=backend,
            native_for=not remove_loops,
            native_while=not remove_loops,
            dynamic_bounds=not remove_loops,
            kpn=False,
            unroll_limit=int(unroll_limit),
        )
    )
    discovered_loop_plans = (
        loop_composer.discover(graph)
        if canonical_value_ids
        else ()
    )
    evaporated_loop_plans = (
        evaporate_unrolled_loops(graph, discovered_loop_plans)
        if discovered_loop_plans
        else ()
    )
    if evaporated_loop_plans and (
        graph.G.graph.get("planner_specializations")
        or graph.G.graph.get("planner_tensor_descriptors")
    ):
        # Unrolling turns each induction/tuple component into a literal node.
        # Re-run the same structural fixed point so predicates inside the
        # cloned body select aliases to their real carried producer instead of
        # escaping as invented function arguments.
        _fold_callsite_structural_values(graph)
    evaporated_loop_ids = {
        int(plan.loop.node_id) for plan in evaporated_loop_plans
    }
    retained_loop_plans = tuple(
        (
            replace(
                plan,
                strategy=LoopStrategy.NATIVE_SOURCE,
                reason=(
                    "resident sequence mutation requires iterative SSA "
                    "memory effects"
                ),
            )
            if (
                plan.strategy in {LoopStrategy.UNROLL, LoopStrategy.CONSTANT}
                and any(
                    effect.mode is LoopStateEffectMode.SEQUENCE_MUTATION
                    for effect in plan.loop.state_effects
                )
            )
            else plan
        )
        for plan in discovered_loop_plans
        if int(plan.loop.node_id) not in evaporated_loop_ids
        and int(plan.loop.node_id) in graph.G
    )
    semantic_loop_plans = (
        loop_composer.materialize_semantic_ir(
            graph,
            retained_loop_plans,
        )
        if retained_loop_plans
        else ()
    )
    loop_plans = (
        materialize_retained_loop_ports(graph, semantic_loop_plans)
        if semantic_loop_plans
        else ()
    )
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
    _resolve_grounded_method_references(graph)
    _resolve_grounded_tensor_operations(graph)
    reference_tables = build_shell_reference_tables(graph)
    ordered_executable_nodes = _dependency_order(graph)
    executable_nodes = tuple(
        node_id
        for node_id in ordered_executable_nodes
        if not _is_dispatch_metadata_node(graph, node_id)
    )
    # Control membership is a semantic partition: numerical work may be
    # reduced freely inside one planner-owned loop body or conditional branch,
    # but never fused across that control boundary.  All other dispatch
    # boundaries are produced by the fixed-point shader-region identities
    # rather than inherited from schedule levels or operator spelling.
    partition_keys = _control_partition_keys(
        graph,
        loop_plans,
        executable_nodes,
    )
    inert_nodes = _inert_routing_nodes(graph)
    closure_edges, closure_outputs = _closure_routing_dependencies(graph)
    # A method may ``return`` a compile-time reference (a type or module, e.g.
    # ``return torch.float32``). That resolves to a _StaticPythonReference, not a
    # runtime value node id -- it is a compile-time constant, not a runtime
    # output -- so it is excluded from the runtime return-liveness set here
    # (int() would fail on it), the same way compile-time references are not
    # runtime roots.
    return_outputs = frozenset(
        int(value_id)
        for value_id in (
            *(
                parent
                for _node_id, data in graph.G.nodes(data=True)
                if isinstance(data.get("expr_obj"), ast.Return)
                for parent, role in (data.get("parents") or ())
                if str(role) in {"result", "value", "operand"}
            ),
            *(graph.G.graph.get("return_value_nodes", {}) or {}).values(),
        )
        if _is_runtime_value_id(value_id) and int(value_id) in graph.G
    )
    recursion_control_nodes = frozenset(
        int(node_id)
        for record in (
            graph.G.graph.get("recursion_table") or {}
        ).values()
        if record.get("control_ir", True)
        for node_id in record.get("control_members", ())
    )
    dispatch_plan = reduce_scheduled_shader_regions(
        graph,
        executable_nodes,
        max_nodes_per_region=max_nodes_per_dispatch,
        partition_keys=partition_keys,
        extra_dependency_edges=closure_edges,
        control_node_ids=recursion_control_nodes,
    )
    executable_dispatch_nodes = tuple(
        dispatch.node_ids
        for dispatch in dispatch_plan.dispatches
    )
    control_deployment_regions = bind_control_deployments_to_regions(
        graph.G.graph.get("control_deployment_regions", ()),
        executable_dispatch_nodes,
    )
    deployment_region_preferences: dict[int, str] = {}
    for deployment in control_deployment_regions:
        for lane in deployment.lanes:
            for region_index in lane.region_indices:
                previous = deployment_region_preferences.setdefault(
                    int(region_index), deployment.schedule_preference
                )
                if previous != deployment.schedule_preference:
                    raise ValueError(
                        "scheduled region has conflicting deployment "
                        f"preferences: region={region_index}, "
                        f"preferences={(previous, deployment.schedule_preference)!r}"
                    )
    dispatch_subgraphs = tuple(
        _dispatch_subgraph(
            graph,
            node_ids,
            required_outputs=frozenset((*closure_outputs, *return_outputs)),
            inert_nodes=inert_nodes,
            schedule_preference=deployment_region_preferences.get(
                region_index, "asap"
            ),
        )
        for region_index, node_ids in enumerate(executable_dispatch_nodes)
        if node_ids
    )
    for subgraph, dispatch in zip(
        dispatch_subgraphs,
        dispatch_plan.dispatches,
    ):
        subgraph.G.graph["rewrite_history"] = dispatch.rewrite_history
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
    loop_shader_reductions = analyze_shader_loop_reductions(
        graph,
        loop_plans,
        executable_dispatch_nodes,
    )
    deep_compilers = tuple(
        GraphDeepCompiler(
            subgraph,
            dict(abstract_tensor_funcs),
            abstract_tensor_sigs,
            node_observer=_observe_process_graph_node,
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
    deployment_class = type(
        "ProcessGraphGLSLDeployment",
        (ProcessGraphGLSLDeployment,),
        {
            "process_graph": graph,
            "external_function_table": getattr(
                graph, "external_function_table", None
            ),
            "static_python_bindings": dict(
                getattr(graph, "python_bindings", {}) or {}
            ),
            "dispatch_plan": dispatch_plan,
            "loop_plans": loop_plans,
            "loop_shader_reductions": loop_shader_reductions,
            "control_deployment_regions": control_deployment_regions,
            "loop_region_indices": loop_region_indices,
            "process_graph_boundary": process_graph_boundary,
            "python_callable": staticmethod(python_callable),
            "dispatch_subgraphs": dispatch_subgraphs,
            "deep_compilers": deep_compilers,
            "ephemeral_callables": ephemeral_callables,
            "reference_table_template": reference_tables,
            "source_node_count": graph.G.number_of_nodes(),
            "primitive_count": sum(
                len(node_ids) for node_ids in executable_dispatch_nodes
            ),
            "loop_count": sum(
                1
                for _node_id, data in graph.G.nodes(data=True)
                if str(data.get("type")) in {"For", "AsyncFor", "While"}
            ),
            "dispatch_count": len(dispatch_subgraphs),
            "max_nodes_per_dispatch": int(max_nodes_per_dispatch),
            "deployment_batches": node_locations,
        },
    )
    function_shell_types = {}
    function_table = getattr(graph, "function_table", None)
    table_identity = id(function_table)
    dependency_regions = (
        (graph.G.graph.get("map_ir") or {}).get("dependency_regions") or {}
    )
    runtime_references = (
        frozenset(map(int, dependency_regions["runtime"]))
        if runtime_closure_only and "runtime" in dependency_regions
        else None
    )
    if (
        function_table is not None
        and table_identity not in _function_table_stack
    ):
        nested_stack = (*_function_table_stack, table_identity)
        for entry in function_table:
            if entry.graph is None:
                continue
            reference = int(entry.reference.address)
            if (
                runtime_references is not None
                and reference not in runtime_references
            ):
                # ``build_map_dependency_regions`` has already proved the
                # strict call closure of every submitted compile target.
                # Map-only definitions stay in FunctionTable/map/class
                # navigation records, but they are not executable members of
                # this program and must not receive deployment shells.  The
                # old all-table expansion turned a 27-function program into
                # 103 function types and thousands of callsite activations,
                # lowering unrelated class methods as though they ran.
                continue
            function_graph = extract_clean_process_subgraph(
                entry.graph,
                entry.graph.G,
            )
            # A function's graph is built independently, not carved out of
            # the caller's, so it does not inherit the compilation's loop
            # settings the way an extracted subgraph does. Seeding them here
            # is what makes unroll_limit/remove_loops mean anything for a
            # callee -- and the callee is where loops usually live, since the
            # entrypoint is a thin wrapper by convention.
            function_graph.G.graph.setdefault(
                "loop_settings",
                dict(graph.G.graph.get("loop_settings") or {}),
            )
            function_shell_types[entry.reference.address] = (
                strategize_shell_deployment(
                    function_graph,
                    max_nodes_per_dispatch=max_nodes_per_dispatch,
                    _function_table_stack=nested_stack,
                )
            )
    activation_root_references = []
    if runtime_closure_only and function_table is not None:
        for target in graph.G.graph.get("compile_targets", ()):
            try:
                reference = int(
                    function_table.entry(str(target)).reference.address
                )
            except (KeyError, TypeError, ValueError):
                continue
            if reference in function_shell_types:
                activation_root_references.append(reference)
        if not activation_root_references:
            # A caller may provide a precomputed runtime closure without the
            # source-level target names (for example an older checkpoint).
            # In that case retain the complete proven runtime set rather than
            # guessing one privileged root.
            activation_root_references.extend(sorted(function_shell_types))
    deployment_class.function_shell_types = function_shell_types
    deployment_class.runtime_closure_only = bool(runtime_closure_only)
    deployment_class.planned_function_references = tuple(sorted(
        function_shell_types
    ))
    deployment_class.activation_root_references = tuple(dict.fromkeys(
        activation_root_references
    ))
    deployment_class.catalogued_function_references = tuple(sorted(
        int(entry.reference.address)
        for entry in function_table or ()
        if entry.graph is not None
    ))
    deployment_class.catalogue_only_function_references = tuple(sorted(
        set(deployment_class.catalogued_function_references)
        - set(deployment_class.planned_function_references)
    ))
    return deployment_class


__all__ = [
    "DeploymentErrorBuffer",
    "DeploymentProfiler",
    "strategize_shell_deployment",
]
class _CompiledStructuralObject:
    """Shell-owned state whose methods are compiled function subgraphs."""

    __slots__ = ("class_ref", "methods", "state")

    def __init__(self, class_ref, descriptor, args, kwargs):
        self.class_ref = str(class_ref)
        self.methods = dict(descriptor.get("methods") or {})
        fields = tuple(descriptor.get("fields") or ())
        defaults = descriptor.get("field_defaults") or {}
        self.state = {
            field: copy.deepcopy(defaults[field])
            for field in fields
            if field in defaults
        }
        self.state.update(zip(fields, args))
        self.state.update(kwargs)
        for field in fields:
            self.state.setdefault(field, None)

    def __getattr__(self, name):
        method_ref = self.methods.get(name)
        if method_ref is not None:
            return _CompiledStructuralMethod(self, method_ref)
        try:
            return self.state[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        if name in {"class_ref", "methods", "state"}:
            object.__setattr__(self, name, value)
        else:
            self.state[name] = value


class _CompiledStructuralMethod:
    """A method edge into a compiled function shell."""

    __slots__ = ("receiver", "method_ref")

    def __init__(self, receiver, method_ref):
        self.receiver = receiver
        self.method_ref = int(method_ref)


class _CompiledStructuralFunction:
    """First-class edge to a source function already in the function table."""

    __slots__ = ("function_ref",)

    def __init__(self, function_ref):
        self.function_ref = int(function_ref)


class _CompiledStructuralClass:
    """Shell primitive for a compiled class, never a retained Python class."""

    __slots__ = ("class_ref", "descriptor")

    def __init__(self, class_ref, descriptor):
        self.class_ref = str(class_ref)
        self.descriptor = descriptor

    def __call__(self, *args, **kwargs):
        return _CompiledStructuralObject(
            self.class_ref,
            self.descriptor,
            args,
            kwargs,
        )

    def __getattr__(self, name):
        method_ref = (self.descriptor.get("methods") or {}).get(name)
        if method_ref is None:
            raise AttributeError(name)
        return _CompiledStructuralMethod(self, method_ref)


class _GreedyGeneratorStream:
    """Pull-driven shell stream that fills available FIFO capacity per pull."""

    def __init__(self, producer, *, fifo_slots=2):
        self._producer = iter(producer)
        self._fifo = DeploymentFIFO(slots=fifo_slots)
        self._exhausted = False

    def __iter__(self):
        return self

    def __next__(self):
        while not self._exhausted and self._fifo.unread < self._fifo.slots:
            try:
                item = next(self._producer)
            except StopIteration:
                self._exhausted = True
                break
            if not self._fifo.publish(item):
                break
        available, value = self._fifo.consume()
        if not available:
            raise StopIteration
        return value
