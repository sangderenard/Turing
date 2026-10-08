"""Deployment profiler: the deployment error buffer, profiler and shell profile-event helpers."""

from __future__ import annotations

import time
import traceback
from collections import deque
from contextlib import contextmanager
from typing import Any

from ..common.tensors.accelerator_backends.glsl_backend import dispatch_stats


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
