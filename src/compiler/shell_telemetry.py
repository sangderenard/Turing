"""One channel for everything a shell has to say: logs, errors, profiling,
and progress.

These already existed, separately and in different shapes.
``DeploymentErrorBuffer`` keeps a traceback FIFO, ``DeploymentProfiler``
keeps hierarchical timings, ``CLaunchProfile`` carries one launch's
durations, and the HTML shell had grown a fourth log of its own. Four
channels means a caller correlates them by hand, and it means the one thing
none of them had -- progress -- would have become a fifth.

So this is a single record stream with a ``kind``, not four streams. A
consumer that wants only errors filters; a consumer that wants a timeline
gets one already in order. Nothing here replaces those classes: they keep
their own storage and behaviour, and ``attach_*`` wraps them so what they
already record also flows here. Wrapping rather than rewriting matters --
they are load-bearing in the deployment path, and their existing consumers
must not notice.

Progress arrives on the same channel deliberately. A progress indicator that
reads a different source than the log will disagree with it eventually,
usually while something is going wrong and the disagreement is least
affordable. A ``progress`` record is just a record with ``done`` and
``total``, so the log and the bar cannot drift apart.

The same schema is emitted by Python at build time and by JavaScript at run
time, so a shell page shows the compilation and the execution in one
timeline.
"""

from __future__ import annotations

import contextvars
import ctypes
import json
import re
import sys
import threading
import time
import traceback
from ctypes import wintypes
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

SCHEMA = "turing-shell-telemetry-v1"

LOG = "log"
ERROR = "error"
PROFILE = "profile"
PROGRESS = "progress"
# What executed, when. A ``trace`` record says a region of the compiled
# artifact entered or left, carrying the identity that survives recomposition,
# so a consumer can follow the running program rather than replay its
# structure on a clock of its own. Distinct from ``profile``, which reports
# how long something took after the fact -- a trace is the event itself.
TRACE = "trace"
KINDS = (LOG, ERROR, PROFILE, PROGRESS, TRACE)


@dataclass(frozen=True)
class Record:
    """One thing that happened, whatever kind of thing it was."""

    sequence: int
    at_ns: int
    kind: str
    message: str
    # Where it happened: a shell path, a compilation phase, a function name.
    path: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "at_ns": self.at_ns,
            "kind": self.kind,
            "message": self.message,
            "path": self.path,
            "detail": dict(self.detail),
        }


class TelemetryChannel:
    """An ordered record stream with optional live subscribers."""

    def __init__(
        self,
        *,
        capacity: int = 4096,
        name: str = "shell",
        kinds: Iterable[str] | None = None,
    ):
        self.name = str(name)
        self.records: deque[Record] = deque(maxlen=max(1, int(capacity)))
        self._sequence = 0
        self._started_ns = time.perf_counter_ns()
        self._subscribers: list[Callable[[Record], None]] = []
        # Aggregate timings, not the unbounded log or any compiler graph.
        # A long compile can evict its first stage from records while that
        # stage must still appear in the final table/DataFrame.
        self._profiles: dict[tuple[str, str, str], dict[str, Any]] = {}
        # Which kinds this channel actually carries. ``None`` keeps every kind,
        # which is what an ordinary build wants; naming a subset makes the rest
        # cost nothing at all -- a disabled kind is rejected before its record
        # is constructed, so an instrumented call site left in the source does
        # not allocate, timestamp, or notify when nobody asked for it.
        if kinds is None:
            self._enabled = frozenset(KINDS)
        else:
            requested = frozenset(str(kind) for kind in kinds)
            unknown = sorted(requested - set(KINDS))
            if unknown:
                raise ValueError(
                    f"unknown record kinds {unknown}; one of {KINDS}"
                )
            self._enabled = requested

    def carries(self, kind: str) -> bool:
        """Whether this channel is configured to carry ``kind`` at all."""

        return kind in self._enabled

    # -- production --------------------------------------------------------

    def emit(
        self,
        kind: str,
        message: str,
        *,
        path: str = "",
        **detail: Any,
    ) -> Record | None:
        if kind not in KINDS:
            raise ValueError(f"unknown record kind {kind!r}; one of {KINDS}")
        if kind not in self._enabled:
            # Nothing is built: no record, no sequence number, no subscriber
            # notification. The caller gets ``None`` rather than a record it
            # would have to check anyway.
            return None
        self._sequence += 1
        record = Record(
            sequence=self._sequence,
            at_ns=time.perf_counter_ns() - self._started_ns,
            kind=kind,
            message=str(message),
            path=str(path),
            detail=detail,
        )
        self.records.append(record)
        if kind == PROFILE:
            key = (record.path, str(detail.get("section", "")), record.message)
            elapsed = int(detail.get("nanoseconds", 0))
            row = self._profiles.setdefault(key, {
                "path": key[0], "section": key[1], "label": key[2],
                "calls": 0, "nanoseconds": 0, "max_ns": 0,
                "failed_calls": 0, "rss_bytes": 0, "slowest_function": "",
            })
            row["calls"] += 1
            row["nanoseconds"] += elapsed
            row["failed_calls"] += int(bool(detail.get("failed")))
            row["rss_bytes"] = max(row["rss_bytes"], int(detail.get("rss_bytes", 0)))
            if elapsed >= row["max_ns"]:
                row["max_ns"] = elapsed
                row["slowest_function"] = str(detail.get("function", ""))
        for subscriber in tuple(self._subscribers):
            # A broken subscriber must not take the channel down with it --
            # telemetry that can fail the thing it observes is worse than no
            # telemetry.
            try:
                subscriber(record)
            except Exception:
                pass
        return record

    def log(self, message: str, **detail: Any) -> Record | None:
        return self.emit(LOG, message, **detail)

    def error(self, message: str, **detail: Any) -> Record | None:
        return self.emit(ERROR, message, **detail)

    def profile(self, message: str, *, nanoseconds: int = 0, **detail: Any) -> Record | None:
        return self.emit(PROFILE, message, nanoseconds=int(nanoseconds), **detail)

    def progress(
        self, message: str, *, done: int, total: int, **detail: Any
    ) -> Record | None:
        return self.emit(
            PROGRESS, message, done=int(done), total=int(total), **detail
        )

    def trace(
        self,
        message: str,
        *,
        region: int,
        phase: str = "enter",
        path: str = "",
        **detail: Any,
    ) -> Record | None:
        """Record that a region of the running artifact entered or left.

        ``region`` is the identity that survives recomposition -- the same
        index the control shell dispatches through -- so a consumer can attach
        to it without knowing how the planner grouped or nested anything.
        """

        return self.emit(
            TRACE,
            message,
            path=path,
            region=int(region),
            phase=str(phase),
            **detail,
        )

    def exception(
        self, error: BaseException, *, path: str = "", phase: str = ""
    ) -> Record | None:
        return self.emit(
            ERROR,
            f"{type(error).__name__}: {error}",
            path=path,
            phase=phase,
            traceback=traceback.format_exc(limit=8),
        )

    # -- scopes ------------------------------------------------------------

    @contextmanager
    def timed(self, message: str, *, path: str = "", **detail: Any) -> Iterator[None]:
        """Time a block and record it, including when it raises.

        A phase that failed still took time, and losing that is how a slow
        failure looks like a fast one.
        """

        started = time.perf_counter_ns()
        try:
            yield
        except BaseException as error:
            self.emit(
                PROFILE,
                message,
                path=path,
                nanoseconds=time.perf_counter_ns() - started,
                failed=True,
                **detail,
            )
            self.exception(error, path=path, phase=message)
            raise
        else:
            self.emit(
                PROFILE,
                message,
                path=path,
                nanoseconds=time.perf_counter_ns() - started,
                **detail,
            )

    @contextmanager
    def stepped(
        self, message: str, total: int, *, path: str = ""
    ) -> Iterator[Callable[[str], None]]:
        """Report progress through a known number of steps.

        Yields ``advance(label)``. The final record is emitted even when the
        block raises, so a bar cannot be left stuck at an arbitrary fraction
        with no explanation beside it.
        """

        total = max(0, int(total))
        state = {"done": 0}
        self.progress(message, done=0, total=total, path=path)

        def advance(label: str = "") -> None:
            state["done"] += 1
            self.progress(
                label or message, done=state["done"], total=total, path=path
            )

        try:
            yield advance
        finally:
            if state["done"] != total:
                self.progress(
                    f"{message} (stopped)",
                    done=state["done"],
                    total=total,
                    path=path,
                    incomplete=True,
                )

    # -- consumption -------------------------------------------------------

    def subscribe(self, callback: Callable[[Record], None]) -> Callable[[], None]:
        self._subscribers.append(callback)

        def unsubscribe() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        return unsubscribe

    def of_kind(self, kind: str) -> tuple[Record, ...]:
        return tuple(r for r in self.records if r.kind == kind)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "name": self.name,
            "records": [r.to_mapping() for r in self.records],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), default=str)

    def profile_summary(self) -> tuple[dict[str, Any], ...]:
        """Inclusive wall timings by existing path/section/label.

        Parent and child rows must NOT be added to estimate total wall time.
        No per-function graph copies, stack tracing or DataFrames on dispatch.
        """
        return tuple({
            **row, "seconds": row["nanoseconds"] / 1e9,
            "mean_seconds": row["nanoseconds"] / row["calls"] / 1e9,
            "max_seconds": row["max_ns"] / 1e9,
        } for row in sorted(self._profiles.values(),
                            key=lambda row: (-row["nanoseconds"], row["path"], row["label"])))

    def profile_dataframe(self):
        """Materialize pandas only when requested, after collecting records."""
        import pandas as pd
        result = pd.DataFrame.from_records(self.profile_summary(), columns=(
            "path", "section", "label", "calls", "seconds", "mean_seconds",
            "max_seconds", "failed_calls", "rss_bytes", "slowest_function",
        ))
        result.attrs["timing_semantics"] = "inclusive wall time; nested rows are not additive"
        return result


# --- adapters over what already exists -------------------------------------
#
# Each of these wraps an object that already records something, so its
# records also reach the channel. None of them changes what the wrapped
# object stores or returns: existing consumers must not be able to tell.


def attach_error_buffer(buffer: Any, channel: TelemetryChannel) -> Callable[[], None]:
    """Mirror a ``DeploymentErrorBuffer``'s pushes onto the channel."""

    original = buffer.push

    def push(exception, *, path="", phase="", node_id=None, handled=False):
        record = original(
            exception, path=path, phase=phase, node_id=node_id, handled=handled
        )
        channel.emit(
            ERROR,
            f"{type(exception).__name__}: {exception}",
            path=str(path),
            phase=str(phase),
            node_id=node_id,
            handled=bool(handled),
        )
        return record

    buffer.push = push

    def detach() -> None:
        buffer.push = original

    return detach


def attach_profiler(profiler: Any, channel: TelemetryChannel) -> Callable[[], None]:
    """Mirror a ``DeploymentProfiler``'s records onto the channel."""

    original = profiler.record

    def record(*args: Any, **kwargs: Any):
        result = original(*args, **kwargs)
        if not getattr(profiler, "enabled", True):
            return result
        path = kwargs.get("path") or (args[0] if args else "")
        nanoseconds = (
            kwargs.get("elapsed_ns")
            or kwargs.get("duration_ns")
            or kwargs.get("nanoseconds")
            or float(kwargs.get("cpu_ms", 0)) * 1e6
            or 0
        )
        channel.emit(
            PROFILE,
            str(kwargs.get("label") or kwargs.get("phase") or "record"),
            path=str(path),
            nanoseconds=int(nanoseconds or 0),
            section=str(kwargs.get("section", "")),
            gpu_ms=kwargs.get("gpu_ms"), dispatches=kwargs.get("dispatches", 0),
        )
        return result

    profiler.record = record

    def detach() -> None:
        profiler.record = original

    return detach


def record_launch_profile(
    channel: TelemetryChannel, profile: Any, *, path: str = ""
) -> Record:
    """Put one ``CLaunchProfile`` (or ``DualIRShell.rollup_profile()``) on
    the channel."""

    return channel.emit(
        PROFILE,
        "launch",
        path=path,
        nanoseconds=int(getattr(profile, "shell_ns", 0)),
        device_ns=int(getattr(profile, "device_ns", 0)),
        host_ns=int(getattr(profile, "host_ns", 0)),
        status=int(getattr(profile, "status", 0)),
        language=str(getattr(profile, "language", "")),
    )


def record_shortfalls(
    channel: TelemetryChannel, shortfalls: Iterable[Any], *, path: str = ""
) -> tuple[Record, ...]:
    """Put a backend's named shortfalls on the channel as errors.

    A shortfall is the honest form of "this backend cannot do that", and it
    belongs in the same timeline as everything else rather than only in a
    report a caller has to remember to print.
    """

    out = []
    for shortfall in shortfalls:
        text = shortfall.format() if hasattr(shortfall, "format") else str(shortfall)
        out.append(channel.emit(ERROR, text, path=path, shortfall=True))
    return tuple(out)


# --- process graph summary -------------------------------------------------


def summarize_process_graph(graph: Any, *, limit: int = 400) -> dict[str, Any]:
    """A JSON-able view of a ``ProcessGraph``, for display beside a program.

    Deliberately a summary. The whole graph of a real program is far larger
    than anything worth putting in a page, and the questions a person
    actually asks of it here -- how big is it, what kinds of node are in it,
    what does this node connect to -- are answered by the shape and a capped
    node table.
    """

    nx_graph = getattr(graph, "G", graph)
    nodes = []
    histogram: dict[str, int] = {}
    for node_id, data in nx_graph.nodes(data=True):
        node_type = str(data.get("type") or data.get("op") or "?")
        histogram[node_type] = histogram.get(node_type, 0) + 1
        if len(nodes) < limit:
            nodes.append({
                "id": int(node_id) if isinstance(node_id, int) else str(node_id),
                "type": node_type,
                "label": str(data.get("label") or "")[:80],
                "parents": [
                    int(p) if isinstance(p, int) else str(p)
                    for p, _role in (data.get("parents") or ())
                ][:8],
            })
    return {
        "nodes": nx_graph.number_of_nodes(),
        "edges": nx_graph.number_of_edges(),
        "truncated": nx_graph.number_of_nodes() > limit,
        "histogram": dict(sorted(histogram.items(), key=lambda kv: -kv[1])),
        "table": nodes,
    }


_WINDOWS_RSS_BINDING = None
_WINDOWS_RSS_LOCK = threading.Lock()


class _WindowsProcessMemoryCounters(ctypes.Structure):
    """The one import-owned Windows RSS record type used by telemetry."""

    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _windows_rss_binding():
    # The 16-rule log retained ~67k PyCStructType/PyCPointerType objects.
    # A 1000-read microprobe here (also on untouched HEAD8eb40eae) retained
    # 1000 _Counters classes: POINTER caches every newly declared type.
    # Declare/bind once, not at the telemetry sampling frequency.
    global _WINDOWS_RSS_BINDING
    with _WINDOWS_RSS_LOCK:
        if _WINDOWS_RSS_BINDING is None:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            current = kernel32.GetCurrentProcess
            current.restype = wintypes.HANDLE
            query = kernel32.K32GetProcessMemoryInfo
            query.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_WindowsProcessMemoryCounters),
                wintypes.DWORD,
            ]
            query.restype = wintypes.BOOL
            _WINDOWS_RSS_BINDING = (_WindowsProcessMemoryCounters, current, query)
        return _WINDOWS_RSS_BINDING


def process_rss_bytes() -> int:
    """This process's resident set (working set on Windows), in bytes.

    The one RSS reading the compiler uses (stage-boundary progress lines and
    the ``memory_budget_bytes`` regulation read it; nothing else should
    re-derive it).  Returns 0 when the platform offers no reading.
    """

    import sys

    if sys.platform == "win32":
        import ctypes
        _Counters, current, query = _windows_rss_binding()
        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        if not query(current(), ctypes.byref(counters),
                     counters.cb):
            return 0
        return int(counters.WorkingSetSize)
    try:
        with open("/proc/self/statm", "rb") as stream:
            import os

            return int(stream.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def system_available_memory_bytes() -> int | None:
    """Return immediately available physical memory without another package."""

    import os

    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.WinDLL(
            "kernel32", use_last_error=True,
        ).GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullAvailPhys)
    try:
        with open("/proc/meminfo", "rb") as stream:
            for line in stream:
                if line.startswith(b"MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


# --- compile stage progress bars -------------------------------------------
#
# The compiler already says everything it is doing, as lines, through one
# ``progress(message)`` callback.  What a person watching a 30-minute compile
# lacks is the shape of it: which stage, how far through its loop, how much
# memory.  This layer adds that and nothing else.
#
# It is a consumer of the channel above, not a second mechanism.  Stage and
# loop progress are ``progress`` records (``done``/``total``), every line the
# compiler prints is a ``log`` record, and ``_CompileBars`` is a subscriber
# that draws the first with tqdm and writes the second with ``tqdm.write`` --
# so lines scroll above the bars and the bars stay at the bottom, and the
# bar and the log read one stream and cannot disagree.
#
# When the layer is off (no TTY, ``progress_bars=False``, tests) nothing here
# is built: ``progress_callback`` hands back the caller's callback, or the
# plain ``[compiler] message`` line to stderr, exactly as before, and every
# ``compile_*`` helper is a contextvar read that returns.

@dataclass(frozen=True)
class CompileStage:
    """One declared stage of a compile: a stable key and the label shown."""

    key: str
    label: str


#: The stages of ``lower_ast_source_to_ssa``, in the order its
#: ``report``/``progress`` call sites run them (``_lower_ast_source_to_ssa_impl``
#: then ``_lower_resolved_process_graph_deployment`` then
#: ``_class_surface_ssa_program``).
LOWERING_STAGES: tuple[CompileStage, ...] = (
    CompileStage("source-normalisation", "source normalisation"),
    CompileStage("source-closure", "source closure"),
    CompileStage("topology-reduction", "topology reduction"),
    CompileStage("compilation-units", "compilation units"),
    CompileStage("deployment-select", "deployment select"),
    CompileStage("deployment-instantiate", "deployment instantiate"),
    CompileStage("call-topology", "call-topology planning"),
    CompileStage("graph-planning", "graph planning"),
    CompileStage("ssa-lowering", "SSA lowering"),
    CompileStage("pre-native-repairs", "pre-native repairs"),
    CompileStage("compiler-finalization", "compiler finalization"),
)
#: A caller that owns the whole flow (SSA, then emission, then the native
#: build) opens the layer with these, so the bar does not claim completion
#: when only the SSA half is done.
COMPILE_STAGES: tuple[CompileStage, ...] = (
    *LOWERING_STAGES,
    CompileStage("emission", "emission"),
    CompileStage("build", "build"),
)

COMPILE_LINE_PREFIX = "[compiler] "

#: Selection events that are bookkeeping around other events, not phases.
_SELECTION_NOT_A_PHASE = frozenset(("deployment", "function-shell"))
#: The nested bars a selection phase owns; they close when it ends.
_SELECTION_PHASE_BARS: Mapping[str, tuple[str, ...]] = {
    "reduce-scheduled-shader-regions": ("region-reduction",),
    "propagate-callsite-tensor-specializations": (
        "callsite-callers", "callsites", "fold-iteration",
    ),
    "propagate-callsite-tensor-specializations-specialized-copy": (
        "callsite-callers", "callsites", "fold-iteration",
    ),
}
_GIB = float(1024 ** 3)


def _plain_compiler_line(message: str) -> None:
    """Today's default compiler line, unchanged."""

    print(f"{COMPILE_LINE_PREFIX}{message}", file=sys.stderr, flush=True)


def progress_bars_default(stream: Any = None) -> bool:
    """On when stdout is a terminal (and so is the stream the bars draw on)."""

    for candidate in (sys.stdout, stream if stream is not None else sys.stderr):
        try:
            if not candidate.isatty():
                return False
        except Exception:
            return False
    return True


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.0f}s"
    minutes, rest = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


_ACTIVE_RULE_BATCH: contextvars.ContextVar = contextvars.ContextVar(
    "turing_compile_rule_batch", default=None,
)


class RuleBatchProgress:
    """Wall-clock accounting shared by a sequence of rule compiles.

    The individual compiler live displays are opened and closed inside this
    scope. They read this one batch clock, so each replacement frame carries
    the same completed/total rule status without owning another renderer.
    """

    def __init__(self, total: int, *, clock: Callable[[], float] = time.monotonic,
                 names: Sequence[str] | None = None):
        self.total = int(total)
        if self.total < 0:
            raise ValueError("rule batch total cannot be negative")
        self.clock = clock
        self.started = self.clock()
        self.completed = 0
        self.completed_seconds = 0.0
        self.current = ""
        self._token = None
        self._lock = threading.RLock()
        self.names = None if names is None else frozenset(names)
        self._completed_names = set()
        self.worker_jobs = 1

    def __enter__(self):
        self.started = self.clock()
        self._token = _ACTIVE_RULE_BATCH.set(self)
        layer = active_compile_progress()
        if layer is not None:
            # Heartbeats and queued log writes run in separate contexts. The
            # one terminal owner must retain this batch independently of them.
            getattr(layer, "owner_display", layer).rule_batch = self
        return self

    def __exit__(self, exc_type, exc, traceback_):
        if self._token is not None:
            _ACTIVE_RULE_BATCH.reset(self._token)
            self._token = None

    @contextmanager
    def rule(self, name: str = ""):
        started = self.clock()
        with self._lock:
            self.current = str(name)
        try:
            yield self
        except BaseException:
            raise
        else:
            with self._lock:
                self.completed += 1
                self.completed_seconds += max(0.0, self.clock() - started)
        finally:
            with self._lock:
                self.current = ""

    def summary(self) -> dict[str, Any]:
        with self._lock:
            wall = max(0.0, self.clock() - self.started)
            average = (self.completed_seconds / self.completed
                       if self.completed else None)
            projected = (wall + average * (self.total - self.completed) / self.worker_jobs
                         if average is not None else None)
            return {
                "completed": self.completed,
                "total": self.total,
                "wall_seconds": wall,
                "average_rule_seconds": average,
                "projected_total_seconds": projected,
                "current": self.current,
            }

    def complete(self, name: str, seconds: float) -> None:
        """Count a named terminal compiler receipt once, including cache hits."""
        with self._lock:
            if (self.names is not None and name not in self.names) or name in self._completed_names:
                return
            self._completed_names.add(name)
            self.completed += 1
            self.completed_seconds += max(0.0, float(seconds))


def compile_rule_batch(total: int, *, clock: Callable[[], float] = time.monotonic,
                       names: Sequence[str] | None = None):
    """Return the batch clock consumed by compiler terminal renderers."""

    return RuleBatchProgress(total, clock=clock, names=names)


def active_rule_batch_progress() -> RuleBatchProgress | None:
    return _ACTIVE_RULE_BATCH.get()


class _TqdmLineFile:
    """A file-like whose complete lines go through ``tqdm_class.write``.

    Routes what a caller-supplied progress callback prints (it usually calls
    ``print``) above the bars instead of through them.
    """

    def __init__(self, tqdm_class: Any, stream: Any):
        self._tqdm = tqdm_class
        self._stream = stream
        self._buffer: list[str] = []

    def write(self, text: str) -> int:
        head, newline, tail = str(text).rpartition("\n")
        if newline:
            self._tqdm.write(
                "".join(self._buffer) + head, file=self._stream
            )
            self._buffer = [tail] if tail else []
        else:
            self._buffer.append(tail)
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


class _CompileBars:
    """Draws a channel's ``progress`` records as tqdm bars and writes its
    ``log`` records with ``tqdm.write``.  Subscribe it to a channel."""

    def __init__(self, owner: "CompileProgress"):
        self.owner = owner
        self.lock = threading.RLock()
        self.main: Any = None
        self.stage_index = 0
        self.stage_started = owner.clock()
        self.substage = ""
        self.bars: dict[str, Any] = {}
        self.created: dict[str, float] = {}
        self.details: dict[str, str] = {}
        self.positions: dict[str, int] = {}
        self.peak_rss = 0
        self.last_rss = 0
        self._last_rss_at = float("-inf")
        self.failure: str | None = None
        self.table: list[Any] = []

    # -- helpers -------------------------------------------------------------

    def _rss(self) -> int:
        now = self.owner.clock()
        if now - self._last_rss_at >= 0.25:
            self._last_rss_at = now
            try:
                self.last_rss = int(self.owner.rss())
            except Exception:
                self.last_rss = 0
            self.peak_rss = max(self.peak_rss, self.last_rss)
        return self.last_rss

    def _rss_text(self) -> str:
        return f"rss {self._rss() / _GIB:.2f}GB"

    def _stage_desc(self) -> str:
        stages = self.owner.stages
        if self.owner._workers:
            statuses = [
                str(worker.get("status") or "unknown")
                for worker in self.owner._workers.values()
            ]
            if self.owner._stage_index > 0:
                statuses.append("running" if self.owner._depth else "complete")
            running = sum(
                status in {"running", "starting"} for status in statuses
            )
            failed = sum(status == "failed" for status in statuses)
            complete = sum(
                status in {"complete", "done", "partial", "verified"}
                for status in statuses
            )
            return (
                f"compiler workers: {running} running, "
                f"{complete} complete, {failed} failed"
            )
        return (
            f"compile {self.stage_index + 1}/{len(stages)} "
            f"{stages[self.stage_index].label}"
        )

    def _main_postfix(self) -> str:
        parts = []
        if self.substage:
            parts.append(self.substage)
        parts.append(
            f"stage {_format_duration(self.owner.clock() - self.stage_started)}"
        )
        parts.append(self._rss_text())
        return ", ".join(parts)

    def _nested_postfix(self, key: str) -> str:
        detail = self.details.get(key, "")
        return f"{detail}, {self._rss_text()}" if detail else self._rss_text()

    def _write(self, text: str) -> None:
        line = f"{self.owner.prefix}{text}"
        if self.failure is not None:
            print(line, file=self.owner.stream, flush=True)
        else:
            self.owner.tqdm_class.write(line, file=self.owner.stream)

    def _close_nested(self, keep: Iterable[str] = ()) -> None:
        keep = frozenset(keep)
        for key in tuple(self.bars):
            if key not in keep:
                self.details.pop(key, None)
                self.positions.pop(key, None)
                self.created.pop(key, None)
                self.bars.pop(key).close()

    def heartbeat(self) -> None:
        """Refresh the clocks and memory without any event arriving."""

        # The event producer already holds owner._lock. Keep the same lock
        # order for heartbeat -> table -> summary to avoid a render deadlock.
        with self.owner._lock, self.lock:
            if self.main is None or self.failure is not None:
                return
            self.main.set_postfix_str(self._main_postfix(), refresh=False)
            self.main.refresh()
            self._refresh_table()
            now = self.owner.clock()
            for key, bar in self.bars.items():
                bar.set_postfix_str(self._nested_postfix(key), refresh=False)
                # ``delay`` is only honoured by ``update``; a bar for a loop
                # that has not yet run long enough must stay hidden here too.
                if now - self.created[key] >= self.owner.nested_delay:
                    bar.refresh()

    # -- the subscriber ------------------------------------------------------

    def __call__(self, record: Record) -> None:
        with self.owner._lock, self.lock:
            if self.failure is not None and record.kind != LOG:
                return
            try:
                self._dispatch(record)
            except Exception as error:
                # A drawing fault must not take the compile with it, and must
                # not be silent either: say so once, then fall back to the
                # plain lines.
                self.failure = f"{type(error).__name__}: {error}"
                try:
                    self._close_nested()
                    if self.main is not None:
                        self.main.close()
                    for row in self.table:
                        row.close()
                    self.table.clear()
                except Exception:
                    pass
                print(
                    f"{self.owner.prefix}progress bars disabled: {self.failure}",
                    file=self.owner.stream, flush=True,
                )
                if record.kind == LOG:
                    self._write(record.message)

    def _dispatch(self, record: Record) -> None:
        if record.kind == LOG:
            self._write(record.message)
            return
        if record.kind != PROGRESS:
            return
        detail = record.detail
        scope = detail.get("scope")
        if scope == "start":
            self._start()
        elif scope == "stage":
            self._stage(record)
        elif scope == "substage":
            self._close_nested(detail.get("keep", ()))
            self.substage = record.message
            self.heartbeat()
        elif scope == "count":
            self._count(record)
        elif scope == "end":
            self._end(record)

    def _start(self) -> None:
        owner = self.owner
        self.main = owner.tqdm_class(
            # Stage order has no relationship to remaining work/time.
            total=None,
            desc=self._stage_desc(),
            unit="stage",
            file=owner.stream,
            leave=True,
            dynamic_ncols=True,
            mininterval=0.25,
            position=len(owner.stages) + 1 if owner.timing_table else 0,
            bar_format="{desc} [{elapsed}]{postfix}",
        )
        if owner.timing_table:
            self._rss()
            for position in range(len(owner.stages) + 1):
                self.table.append(owner.tqdm_class(
                    total=None, desc="", file=owner.stream, leave=True,
                    position=position, dynamic_ncols=True, bar_format="{desc}",
                ))
            self._refresh_table()
        self.main.set_postfix_str(self._main_postfix(), refresh=True)

    def _refresh_table(self) -> None:
        if not self.table:
            return
        workers = list(self.owner._workers.values())
        if workers:
            stage_rows = next(
                (worker["stages"] for worker in workers if worker.get("stages")),
                (),
            )
            self.table[0].set_description_str(
                "Compiler stage | " + " | ".join(
                    str(worker["label"]) for worker in workers
                ),
                refresh=False,
            )
            self.table[0].refresh()
            for index, bar in enumerate(self.table[1:]):
                if index >= len(stage_rows):
                    bar.set_description_str("", refresh=False)
                    bar.refresh()
                    continue
                states = []
                for worker in workers:
                    worker_stages = worker.get("stages") or ()
                    state = (
                        str(worker_stages[index].get("state") or "pending")
                        if index < len(worker_stages) else "pending"
                    )
                    if index == int(worker["stage_number"]) - 1:
                        state = str(worker["status"])
                    states.append(state)
                bar.set_description_str(
                    f"{str(stage_rows[index].get('label') or '')} | "
                    + " | ".join(states),
                    refresh=False,
                )
                bar.refresh()
            return
        rows = self.owner.summary()["stages"]
        self.table[0].set_description_str(
            "Stage                              State       Wall(s)    RSS(GB)", refresh=False)
        self.table[0].refresh()
        for bar, row in zip(self.table[1:], rows, strict=True):
            bar.set_description_str(
                f"{row['label'][:34]:34} {row['state']:10} "
                f"{row['seconds']:9.2f} {row['rss_bytes'] / _GIB:10.2f}", refresh=False)
            bar.refresh()

    def _stage(self, record: Record) -> None:
        self._close_nested()
        self.stage_index = int(record.detail["index"])
        self.stage_started = float(record.detail["started"])
        self.substage = ""
        self.main.n = self.stage_index
        self.main.set_description_str(self._stage_desc(), refresh=False)
        self.main.set_postfix_str(self._main_postfix(), refresh=False)
        self.main.refresh()
        self._refresh_table()

    def _count(self, record: Record) -> None:
        detail = record.detail
        key = str(detail["key"])
        if detail.get("close"):
            self.details.pop(key, None)
            self.positions.pop(key, None)
            self.created.pop(key, None)
            bar = self.bars.pop(key, None)
            if bar is not None:
                bar.close()
            return
        total = detail.get("bar_total")
        bar = self.bars.get(key)
        if bar is None:
            self.created[key] = self.owner.clock()
            position = len(self.owner.stages) + 2 if self.owner.timing_table else 1
            while position in self.positions.values():
                position += 1
            self.positions[key] = position
            bar = self.bars[key] = self.owner.tqdm_class(
                total=total,
                desc=record.message,
                unit="it",
                file=self.owner.stream,
                leave=False,
                dynamic_ncols=True,
                mininterval=0.25,
                delay=self.owner.nested_delay,
                position=position,
                # A known total is a work-item fraction, not a time forecast.
                bar_format=("{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                            "[{elapsed}]{postfix}" if total is not None else
                            "{desc} {n_fmt} completed [{elapsed}]{postfix}"),
            )
        else:
            bar.set_description_str(record.message, refresh=False)
        if total is not None:
            bar.total = total
        self.details[key] = str(detail.get("detail") or "")
        bar.set_postfix_str(self._nested_postfix(key), refresh=False)
        step = int(detail["done"]) - bar.n
        if step:
            bar.update(step)

    def _end(self, record: Record) -> None:
        self._close_nested()
        failed = bool(record.detail.get("failed"))
        main = self.main
        if main is None:
            return
        total = len(self.owner.stages)
        if failed:
            main.set_description_str(
                "compiler workers FAILED" if self.owner._workers else
                f"compile FAILED in {self.owner.stages[self.stage_index].label}",
                refresh=False,
            )
        else:
            main.n = len(self.owner._stage_profiles)
            main.set_description_str(
                "compiler workers done" if self.owner._workers else
                f"compile done ({main.n} timed stages)",
                refresh=False,
            )
        main.set_postfix_str(
            f"peak {max(self.peak_rss, self._rss()) / _GIB:.2f}GB, "
            f"stage {_format_duration(self.owner.clock() - self.stage_started)}",
            refresh=False,
        )
        main.refresh()
        self._refresh_table()
        for row in self.table:
            row.close()
        self.table.clear()
        main.close()
        self.main = None


class _CompileLive(_CompileBars):
    """One Rich Live owns cursor movement for timing rows and work bars.

    Reuses the channel dispatcher and RSS sampler; never starts Progress's
    own Live or another refresh thread. Logs (including ordinary print)
    scroll above the complete frame, not between independently placed bars.
    """

    def __init__(self, owner):
        super().__init__(owner)
        from rich.console import Console, Group
        from rich.live import Live
        from rich.progress import BarColumn, Progress, ProgressColumn, TextColumn
        from rich.table import Column, Table
        from rich.text import Text

        class MeasuredBar(ProgressColumn):
            def render(self, task):
                return Text("") if task.total is None else BarColumn(bar_width=None).render(task)

        self.console = Console(file=owner.stream, markup=False, highlight=False)
        self.Group, self.Table, self.Text, self.Live = Group, Table, Text, Live
        self.progress = Progress(
            TextColumn("{task.description}", markup=False,
                       table_column=Column(ratio=2, overflow="ellipsis", no_wrap=True)),
            MeasuredBar(),
            TextColumn("{task.fields[count]}", markup=False),
            console=self.console, auto_refresh=False, expand=True,
            get_time=owner.clock,
        )
        self.tasks: dict[str, Any] = {}
        self.live = None
        self.finished = False
        self.failed = False
        self._last_render_at = float("-inf")

    def _frame(self):
        self._rss()
        summary = self.owner.summary()
        batch = self.owner.rule_batch or active_rule_batch_progress()
        batch_summary = None if batch is None else batch.summary()
        batch_status = None
        if batch_summary is not None:
            average = batch_summary["average_rule_seconds"]
            projected = batch_summary["projected_total_seconds"]
            batch_status = (
                f"Rules {batch_summary['completed']}/{batch_summary['total']} | "
                f"wall {_format_duration(batch_summary['wall_seconds'])} | "
                f"average {'--' if average is None else _format_duration(average)} | "
                f"projected total {'--' if projected is None else _format_duration(projected)}"
            )
        workers = list(self.owner._workers.values())
        has_worker_snapshots = bool(workers)
        hidden_workers = 0
        # The coordinator is always a compiler lane.  The provenance-first
        # outer stages have no child snapshots, but they must retain the same
        # stage-by-lane display as compiles that have admitted workers.
        current_stage = summary["stages"][self.owner._stage_index]
        workers.insert(0, {"key": "coordinator", "label": "source compiler",
            "status": "failed" if self.failed else "complete" if self.finished else "running",
            "stage_number": self.owner._stage_index + 1, "stage_count": len(summary["stages"]),
            "stage_label": current_stage["label"], "substage": self.substage,
            "elapsed_seconds": current_stage["seconds"], "rss_bytes": self.last_rss,
            "stages": summary["stages"]})
        if workers:
            # Keep useful column widths on real terminals. All lanes remain in
            # the owner; prefer active/failed lanes, then later-stage completions.
            column_budget = max(1, (self.console.width - 30) // 24)
            if len(workers) > column_budget:
                priority_workers = [lane for lane in workers if lane["status"] in {"running", "failed", "partial"}]
                # A completed supplied source has reached stage ten; its
                # region workers stopped at stage nine. Recent region lanes
                # used to displace that source and show pre-native pending
                # even after its real source coordinator had finished.
                priority_workers += sorted(
                    reversed([lane for lane in workers if lane not in priority_workers]),
                    key=lambda lane: int(lane["stage_number"]), reverse=True)
                hidden_workers = len(workers) - column_budget
                workers = priority_workers[:column_budget]
            status = (
                "compiler workers FAILED" if self.failed
                else "compiler workers done" if self.finished
                else self._stage_desc()
            ) if has_worker_snapshots else (
                f"compile FAILED in {self.owner.stages[self.stage_index].label}"
                if self.failed
                else f"compile done ({len(self.owner._stage_profiles)} timed stages)"
                if self.finished
                else self._stage_desc()
            )
        status += ": " + self._main_postfix()
        if self.finished:
            status += f", peak {self.peak_rss / _GIB:.2f}GB"
        available = max(2, self.console.height - 4)  # reserve room for scrolling logs
        tasks = list(self.tasks.values())
        if self.console.width < 55 or available < 6:
            active = summary["stages"][self.stage_index]
            compact_status = ("FAILED " if self.failed else "done " if self.finished else "")
            compact_status += active["label"]
            if has_worker_snapshots:
                lane = workers[-1]
                compact_status = (
                    f"{lane['label']} {lane['status']} "
                    f"{lane['stage_number']}/{lane['stage_count']} "
                    f"{lane['stage_label']}"
                )
            # At 20 columns the stage-order prefix used to hide the actual
            # active stage. Give the operation's name first claim on width.
            lines = ([] if batch_status is None else [
                         self.Text(batch_status, overflow="ellipsis", no_wrap=True)])
            lines += [self.Text(compact_status, overflow="ellipsis", no_wrap=True),
                     self.Text(f"{active['state']} | wall {active['seconds']:.2f}s | "
                               f"rss {active['rss_bytes'] / _GIB:.2f}GB", no_wrap=True)]
            if tasks and available > len(lines):
                task = self.progress.tasks[-1]
                lines.append(self.Text(f"{task.description}: {task.fields['count']}",
                                       overflow="ellipsis", no_wrap=True))
            return self.Group(*lines)

        # The timing table never consumes the bottom four lines. Active work
        # bars are always appended after the stationary status/table region.
        progress_reserve = min(4, max(1, available - 6))
        work_budget = min(len(tasks), progress_reserve)
        visible_tasks = set(tasks[-work_budget:]) if work_budget else set()
        for task in self.progress.tasks:
            task.visible = task.id in visible_tasks
        hidden_work = len(tasks) - work_budget
        rows = summary["stages"]
        header_lines = int(batch_status is not None)
        budget = max(1, available - progress_reserve - header_lines
                     - 4 - int(bool(hidden_work)))
        if self.finished:
            budget = len(rows)  # final frame is ordinary scrollback, not a redraw region
        priority = [self.stage_index] + [i for i in reversed(range(len(rows)))
                                        if rows[i]["state"] in {"done", "failed"}]
        priority += list(range(len(rows)))
        selected = set(list(dict.fromkeys(priority))[:budget])
        table = self.Table(
            title=("Compiler workers" if workers else "Compiler timings (wall time)"),
            box=None, expand=True, pad_edge=False,
        )
        if workers:
            stage_rows = next(
                (lane["stages"] for lane in workers if lane.get("stages")), ()
            )
            if stage_rows:
                table.add_column(
                    "Compiler stage", ratio=2, overflow="ellipsis", no_wrap=True,
                )
                for lane in workers:
                    table.add_column(
                        lane["label"].rsplit(".<locals>.", 1)[-1], ratio=1, overflow="ellipsis", no_wrap=True,
                    )
                row_budget = max(
                    1, available - progress_reserve - header_lines - 4
                )
                active_indices = {
                    int(lane["stage_number"]) - 1
                    for lane in workers if int(lane["stage_number"]) > 0
                }
                order = list(active_indices)
                order += list(reversed(range(len(stage_rows))))
                order += list(range(len(stage_rows)))
                visible = set(list(dict.fromkeys(order))[:row_budget])
                for index, stage_row in enumerate(stage_rows):
                    if index not in visible:
                        continue
                    cells = []
                    for lane in workers:
                        lane_stages = lane.get("stages") or ()
                        state = (
                            str(lane_stages[index].get("state") or "pending")
                            if index < len(lane_stages) else "pending"
                        )
                        if index == int(lane["stage_number"]) - 1:
                            state = (
                                f"{lane['status']} {lane['elapsed_seconds']:.0f}s "
                                f"{lane['rss_bytes'] / _GIB:.2f}GB"
                            )
                        style = ("bold red" if "failed" in state or "partial" in state
                                 else "green" if "done" in state or "complete" in state
                                 else "cyan" if "running" in state else "dim")
                        cells.append(self.Text(state, style=style))
                    table.add_row(str(stage_row.get("label") or ""), *cells)
                if len(visible) < len(stage_rows):
                    table.caption = (
                        f"{len(stage_rows) - len(visible)} stages hidden; "
                        "active and failed stages retained"
                    )
            else:
                table.add_column("Unit", ratio=2, overflow="ellipsis", no_wrap=True)
                table.add_column("Compiler stage", ratio=3, overflow="ellipsis", no_wrap=True)
                table.add_column("State", no_wrap=True)
                table.add_column("Wall(s)", justify="right", no_wrap=True)
                table.add_column("RSS(GB)", justify="right", no_wrap=True)
                for lane in workers:
                    stage = (
                        f"{lane['stage_number']}/{lane['stage_count']} "
                        f"{lane['stage_label']}"
                        + (f" · {lane['substage']}" if lane['substage'] else "")
                    )
                    table.add_row(
                        self.Text(lane["label"]), stage, lane["status"],
                        f"{lane['elapsed_seconds']:.2f}",
                        f"{lane['rss_bytes'] / _GIB:.2f}",
                    )
        else:
            table.add_column("Stage", ratio=3, overflow="ellipsis", no_wrap=True)
            table.add_column("State", no_wrap=True)
            table.add_column("Wall(s)", justify="right", no_wrap=True)
            table.add_column("RSS(GB)", justify="right", no_wrap=True)
            for index, row in enumerate(rows):
                if index in selected:
                    table.add_row(self.Text(row["label"]), row["state"],
                                  f"{row['seconds']:.2f}", f"{row['rss_bytes'] / _GIB:.2f}")
            if len(selected) < len(rows):
                table.caption = f"{len(rows) - len(selected)} stages hidden; full summary retained"
        parts = ([] if batch_status is None else [
                     self.Text(batch_status, overflow="ellipsis", no_wrap=True)])
        parts += [table, self.Text(status, overflow="ellipsis", no_wrap=True)]
        if hidden_workers:
            parts.append(self.Text(f"+{hidden_workers} worker lanes retained; active/failed lanes shown first", no_wrap=True))
        if tasks:
            parts.append(self.progress)
        if hidden_work:
            parts.append(self.Text(f"+{hidden_work} other active counters; summary retains all"))
        return self.Group(*parts)

    def _start(self):
        self.finished = self.failed = False
        self.stage_index = 0
        self.stage_started = self.owner._stage_started
        self.live = self.Live(self._frame(), console=self.console,
                              auto_refresh=False, redirect_stdout=True,
                              redirect_stderr=True, vertical_overflow="crop")
        self.live.start(refresh=True)
        self._last_render_at = self.owner.clock()

    def _write(self, text):
        if self.failure is None:
            lower = str(text).casefold()
            style = ("bold red" if any(word in lower for word in ("failed", "error", "refused"))
                     else "yellow" if "memory" in lower or "rss" in lower or "compress" in lower
                     else "green" if "done" in lower or "complete" in lower or "verified" in lower
                     else "cyan" if "stage" in lower else "magenta" if "deployment" in lower else None)
            self.console.print(self.Text(f"{self.owner.prefix}{text}", style=style))
        else:
            print(f"{self.owner.prefix}{text}", file=self.owner.stream, flush=True)

    def _close_nested(self, keep=()):
        for key in tuple(self.tasks):
            if key not in keep:
                self.progress.remove_task(self.tasks.pop(key))

    def _stage(self, record):
        self._close_nested()
        self.stage_index = int(record.detail["index"])
        self.stage_started = float(record.detail["started"])
        self.substage = ""
        self.heartbeat(force=True)

    def _count(self, record):
        detail = record.detail
        key = str(detail["key"])
        if detail.get("close"):
            if key in self.tasks:
                self.progress.remove_task(self.tasks.pop(key))
        else:
            total, done = detail.get("bar_total"), int(detail["done"])
            count = (f"{done}/{total} ({done / total:.0%})" if total else
                     f"{done} completed" if total is None else f"{done}/0")
            label = record.message + (f" — {detail['detail']}" if detail.get("detail") else "")
            if key not in self.tasks:
                self.tasks[key] = self.progress.add_task(label, total=total, completed=done, count=count)
            else:
                self.progress.update(self.tasks[key], description=label,
                                     total=total, completed=done, count=count)
        self.heartbeat()

    def heartbeat(self, *, force=False):
        # Same owner -> renderer lock order as event handling; Rich has no
        # refresh thread holding a Console lock while requesting owner data.
        with self.owner._lock, self.lock:
            if self.live is None or self.failure is not None:
                return
            now = self.owner.clock()
            if force or now - self._last_render_at >= 0.1:
                try:
                    self.live.update(self._frame(), refresh=True)
                    self._last_render_at = now
                except Exception as error:
                    self._disable(error)

    def _disable(self, error):
        self.failure = f"{type(error).__name__}: {error}"
        try:
            if self.live is not None:
                self.live.stop()
        except Exception:
            pass
        finally:
            self.live = None
        print(f"{self.owner.prefix}live display disabled: {self.failure}",
              file=self.owner.stream, flush=True)

    def _end(self, record):
        self._close_nested()
        self.finished = True
        self.failed = bool(record.detail.get("failed"))
        if self.live is not None:
            try:
                self.heartbeat(force=True)
            finally:
                if self.live is not None:
                    try:
                        self.live.stop()
                    finally:
                        self.live = None

    def __call__(self, record):
        with self.owner._lock, self.lock:
            if self.failure is not None:
                if record.kind == LOG:
                    self._write(record.message)
                return
            try:
                self._dispatch(record)
            except Exception as error:
                self._disable(error)
                if record.kind == LOG:
                    self._write(record.message)


_ACTIVE_COMPILE_PROGRESS: contextvars.ContextVar = contextvars.ContextVar(
    "turing_compile_progress", default=None
)


class CompileProgress:
    """The stage/loop progress layer for one compile.

    ``enabled=False, collect_profile=False`` builds nothing. With
    ``collect_profile=True`` it retains timings without a renderer/thread.
    Enabled, it owns a ``TelemetryChannel`` (log/progress/profile records), subscribes the tqdm
    renderer to it, and exposes the calls compile code makes:
    ``stage``/``substage`` (the persistent bar), ``count``/``iterate`` (nested
    bars, ``leave=False``, closed with their stage), and ``log``.

    Reentrant: a nested ``open()`` joins the layer already running, so a
    compile started inside another shares its bars.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        stages: Sequence[CompileStage] = LOWERING_STAGES,
        stream: Any = None,
        channel: TelemetryChannel | None = None,
        tqdm_class: Any = None,
        rss: Callable[[], int] | None = None,
        clock: Callable[[], float] = time.monotonic,
        heartbeat: float | None = 1.0,
        min_interval: float = 0.1,
        nested_delay: float = 0.5,
        prefix: str = COMPILE_LINE_PREFIX,
        timing_table: bool = True,
        collect_profile: bool = False,
        renderer: str | None = None,
        state_sink: Callable[[Mapping[str, Any]], None] | None = None,
    ):
        self.stages = tuple(stages)
        self.enabled = bool(enabled) and bool(self.stages)
        import os
        requested_renderer = renderer or os.environ.get("TURING_COMPILE_RENDERER", "auto")
        if requested_renderer not in {"auto", "rich", "tqdm"}:
            raise ValueError("compile renderer must be auto, rich or tqdm")
        self.renderer = "tqdm"
        if self.enabled and (requested_renderer == "rich" or
                             (requested_renderer == "auto" and tqdm_class is None)):
            try:
                from rich.live import Live
                self.renderer = "rich"
            except ImportError:
                pass
        if self.enabled and self.renderer == "tqdm" and tqdm_class is None:
            try:
                from tqdm import tqdm as tqdm_class
            except ImportError:
                self.enabled = False
        self.tqdm_class = tqdm_class
        self.stream = stream if stream is not None else sys.stderr
        self.rss = rss if rss is not None else process_rss_bytes
        self.clock = clock
        self.heartbeat_interval = heartbeat
        self.min_interval = float(min_interval)
        self.nested_delay = float(nested_delay)
        self.prefix = prefix
        self.timing_table = bool(timing_table)
        self.collect_profile = bool(collect_profile or enabled) and bool(self.stages)
        self.state_sink = state_sink
        self.channel: TelemetryChannel | None = None
        self.bars: _CompileBars | None = None
        self._depth = 0
        self._token: Any = None
        self._lock = threading.RLock()
        self._counts: dict[str, dict[str, Any]] = {}
        self._stage_index = 0
        self._stage_started = clock()
        self._substage = ""
        self._substage_started = self._stage_started
        self._stage_profiles: dict[str, Mapping[str, Any]] = {}
        self._entered_compiler_stage = False
        self._stage_seconds: dict[str, float] = {}
        self._work: dict[str, dict[str, dict[str, Any]]] = {}
        self._workers: dict[str, dict[str, Any]] = {}
        self.rule_batch = active_rule_batch_progress()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._log_condition = threading.Condition()
        self._log_queue = deque()
        self._log_stopping = False
        self._log_thread = None
        self._log_file = None
        self.log_path = None
        self._console_log_omissions = 0
        if self.enabled or self.collect_profile:
            self.channel = channel or TelemetryChannel(
                name="compile-progress", capacity=1024, kinds=(LOG, PROGRESS, PROFILE)
            )
        if self.enabled:
            self.bars = _CompileLive(self) if self.renderer == "rich" else _CompileBars(self)
            self.channel.subscribe(self.bars)

    # -- lifecycle -------------------------------------------------------------

    def open(self) -> "CompileProgress":
        if self.channel is None:
            return self
        with self._lock:
            self._depth += 1
            if self._depth == 1:
                self._token = _ACTIVE_COMPILE_PROGRESS.set(self)
                self._stage_index = 0
                self._stage_started = self.clock()
                self._substage = ""
                self._substage_started = self._stage_started
                self._stage_profiles.clear()
                self._entered_compiler_stage = False
                self._stage_seconds.clear()
                self._work.clear()
                if self.enabled:
                    import tempfile
                    from pathlib import Path
                    directory = Path.cwd() / "build" / "compiler-logs"
                    directory.mkdir(parents=True, exist_ok=True)
                    self._log_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                        prefix="compile_", suffix=".log", dir=directory, delete=False)
                    self.log_path = self._log_file.name
                    self._log_stopping = False
                    self._log_thread = threading.Thread(target=self._consume_logs,
                        name="compiler-terminal-log", daemon=True)
                    self._log_thread.start()
                self.channel.progress(
                    self.stages[0].label, done=0, total=len(self.stages),
                    path="compile/start", scope="start",
                    metric="stage-order", fraction_is_progress=False,
                )
                self._publish_state("running")
                if self.enabled and self.heartbeat_interval:
                    self._heartbeat_stop.clear()
                    self._heartbeat_thread = threading.Thread(
                        target=self._heartbeat_loop,
                        name="compile-progress-heartbeat",
                        daemon=True,
                    )
                    self._heartbeat_thread.start()
        return self

    def close(self, failed: bool = False) -> None:
        if self.channel is None:
            return
        with self._lock:
            if self._depth == 0:
                return
            self._depth -= 1
            if self._depth:
                return
            self._finish_substage(failed=failed)
            log_thread, self._log_thread = self._log_thread, None
            self._heartbeat_stop.set()
            thread, self._heartbeat_thread = self._heartbeat_thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        if log_thread is not None:
            with self._log_condition:
                self._log_stopping = True
                self._log_condition.notify()
            log_thread.join()
            if self._console_log_omissions:
                self.channel.log(f"{self._console_log_omissions} additional diagnostic lines retained in {self.log_path}")
            self._log_file.close()
            self._log_file = None
            self.channel.log(f"compiler diagnostic log: {self.log_path}")
        with self._lock:
            stage = self.stages[self._stage_index]
            if self._entered_compiler_stage or not self._workers:
                self._log_stage_finished(stage, failed=failed)
            elif self.enabled:
                # The supplied-source coordinator never entered stage one.
                # profile_memory_n2's post-native stdout error used to mint a
                # spurious "source normalisation FAILED" timing here.
                self.channel.log(
                    f"source schedule {'FAILED after' if failed else 'done in'} "
                    f"{_format_duration(self.clock() - self._stage_started)}")
            for key in tuple(self._counts):
                self.count_close(key)
            self._counts.clear()
            self.channel.progress(
                "compile failed" if failed else "compile done",
                done=len(self._stage_profiles),
                total=len(self.stages), path="compile/end",
                scope="end", failed=bool(failed),
                metric="stage-order", fraction_is_progress=False,
            )
            self._publish_state("failed" if failed else "complete")
            _ACTIVE_COMPILE_PROGRESS.reset(self._token)
            self._token = None

    def __enter__(self) -> "CompileProgress":
        return self.open()

    def __exit__(self, exc_type, exc, traceback_) -> None:
        self.close(failed=exc_type is not None)

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self.heartbeat_interval):
            try:
                self.bars.heartbeat()
            except Exception:
                pass

    # -- the logger, inside the bars ---------------------------------------------

    def log(self, message: str) -> None:
        """One compiler line.  Through ``tqdm.write`` when enabled."""

        if not self.enabled:
            owner = getattr(self, "owner_display", None)
            if owner is not None:
                label = getattr(self, "owner_label", "")
                owner.log(f"[{label}] {message}" if label else message)
                return
            _plain_compiler_line(message)
            return
        text = str(message)
        with self._log_condition:
            if self._log_file is None:
                self.channel.log(text)
                return
            # Preserve every byte independently of console rendering. The
            # producer only appends buffered text and updates a bounded queue;
            # terminal drawing never runs on a compiler worker's call stack.
            self._log_file.write(text + "\n")
            urgent = any(word in text.casefold() for word in ("failed", "error", "traceback", "refused"))
            shape = text if urgent else re.sub(r"\b\d+(?:\.\d+)?\b", "#", text)
            if self._log_queue and self._log_queue[-1][0] == shape:
                group = self._log_queue[-1]
                group[2], group[3] = text, group[3] + 1
            elif len(self._log_queue) < 1024 or urgent:
                self._log_queue.append([shape, text, text, 1])
            else:
                self._console_log_omissions += 1
            self._log_condition.notify()

    def _consume_logs(self):
        while True:
            with self._log_condition:
                self._log_condition.wait_for(lambda: self._log_queue or self._log_stopping)
                if not self._log_queue and self._log_stopping:
                    return
                # One console write for a batch; identical/range-shaped runs
                # keep their endpoints and exact repeat count. Full text stays
                # in the durable log, including diagnostics not drawn live.
                batch = []
                console_bytes = 0
                while self._log_queue and len(batch) < 128 and console_bytes < 8192:
                    group = self._log_queue.popleft()
                    batch.append(group)
                    console_bytes += min(len(group[1]), 4096) + min(len(group[2]), 4096)
                self._log_file.flush()
            lines = []
            def console_line(line):
                if len(line) <= 4096:
                    return line
                return line[:4096] + f" … full diagnostic in {self.log_path}"
            for _shape, first, last, count in batch:
                lines.append(console_line(first))
                if count > 1:
                    lines.append(f"  … {count} related lines; last: {console_line(last)}")
            with self._lock:
                self.channel.log("\n".join(lines))

    def progress_callback(
        self, inner: Callable[[str], None] | None = None
    ) -> Callable[[str], None]:
        """The ``progress`` callback the compile should be handed.

        Off: the caller's own callback, or today's plain ``[compiler]`` line.
        On: this layer's ``log``; a caller-supplied callback is kept, and
        anything it prints is routed above the bars.
        """

        if not self.enabled:
            self._diagnostic_callback = inner if inner is not None else _plain_compiler_line
            return self._diagnostic_callback
        if inner is None:
            return self.log
        if self.renderer == "rich":
            # Live owns stdout/stderr redirection for the whole compile.
            return inner

        def routed(message: str) -> None:
            sink = _TqdmLineFile(self.tqdm_class, self.stream)
            previous = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = sink
            try:
                inner(message)
            finally:
                sys.stdout, sys.stderr = previous
                sink.flush()

        return routed

    # -- stages ---------------------------------------------------------------

    def _phase_log(self, message: str) -> None:
        # Phase observations use the same caller-selected diagnostics route.
        # A profiling-only worker whose progress callback explicitly requests
        # silence must not print hundreds of new lines through owner_display.
        callback = getattr(self, "_diagnostic_callback", None)
        if not self.enabled and callback is not None:
            callback(message)
        elif self.enabled or getattr(self, "owner_display", None) is not None:
            self.log(message)

    def _publish_state(self, status: str) -> None:
        """Publish exact stage state without adding a per-item callback."""

        from .graph_work import raise_if_compiler_work_cancelled
        raise_if_compiler_work_cancelled()
        if self.state_sink is None or not self.stages:
            return
        current = self.stages[self._stage_index]
        rows = []
        for index, stage in enumerate(self.stages):
            profile = self._stage_profiles.get(stage.key)
            rows.append({
                "key": stage.key,
                "label": stage.label,
                "state": (
                    "running" if status == "running" and index == self._stage_index
                    else "failed" if profile and profile.get("failed")
                    else "done" if profile is not None
                    else "pending"
                ),
            })
        payload = {
            "status": str(status),
            "stage_index": self._stage_index,
            "stage_number": self._stage_index + 1,
            "stage_count": len(self.stages),
            "stage_key": current.key,
            "stage_label": current.label,
            "substage": self._substage,
            "stage_elapsed_seconds": max(
                0.0, self.clock() - self._stage_started
            ),
            "rss_bytes": int(self.rss()),
            "stages": rows,
            "workers": [dict(worker) for worker in self._workers.values()],
        }
        try:
            self.state_sink(payload)
        except Exception:
            # Compilation is authoritative; cross-process observation is not.
            pass

    def _log_stage_finished(self, stage: CompileStage, *, failed: bool) -> None:
        seconds = max(0.0, self.clock() - self._stage_started)
        try:
            rss_bytes = self.bars._rss() if self.bars is not None else int(self.rss())
        except Exception:
            rss_bytes = 0
        self._stage_profiles[stage.key] = {"failed": failed, "rss_bytes": rss_bytes}
        self.channel.profile(
            stage.label, path=f"compile/stage/{stage.key}", section="stage",
            nanoseconds=round(seconds * 1e9), failed=failed, rss_bytes=rss_bytes,
        )
        self._stage_seconds[stage.key] = self._stage_seconds.get(stage.key, 0.0) + seconds
        if self.enabled:
            self.channel.log(
                f"stage {self._stage_index + 1}/{len(self.stages)} "
                f"{stage.label}: {'FAILED after' if failed else 'done in'} "
                f"{_format_duration(seconds)} (rss {rss_bytes / _GIB:.2f}GB)"
            )

    def summary(self) -> dict[str, Any]:
        """Small serializable summary, including the active stage and real work."""
        with self._lock:
            rows = []
            for index, stage in enumerate(self.stages):
                record = self._stage_profiles.get(stage.key)
                detail = {} if record is None else record
                running = self._depth > 0 and index == self._stage_index
                rows.append({
                    "key": stage.key, "label": stage.label,
                    "state": "running" if running else "failed" if detail.get("failed")
                             else "done" if record is not None else "pending",
                    "seconds": self._stage_seconds.get(stage.key, 0.0) +
                               (max(0.0, self.clock() - self._stage_started) if running else 0),
                    "rss_bytes": (self.bars.last_rss if running and self.bars else
                                  int(detail.get("rss_bytes", 0))),
                })
            work = {stage: {key: dict(value) for key, value in counters.items()}
                    for stage, counters in self._work.items()}
            active = work.setdefault(self.stages[self._stage_index].key, {}) if self.stages else {}
            for key, value in self._counts.items():
                prior = active.get(key, {"done": 0, "total": 0})
                active[key] = {"done": prior["done"] + value["n"],
                               "total": None if prior["total"] is None or value["total"] is None
                                        else prior["total"] + value["total"],
                               "label": value["label"]}
            return {"stages": rows, "work": work, "log_path": self.log_path,
                    "workers": [dict(worker) for worker in self._workers.values()],
                    "renderer": self.renderer if self.enabled else "none",
                    "profiles": () if self.channel is None else self.channel.profile_summary(),
                    "timing_semantics": "inclusive wall time; stage order is not percent progress"}

    def profile_dataframe(self):
        return compile_profile_dataframe(self.summary())

    def inherit_completed_stages(self, keys: Sequence[str]) -> None:
        """Retain the coordinator-completed prefix of a frozen graph worker.

        These rows have no worker wall time: source ingestion/reduction already
        happened once before the cut. All later transitions use stage() and the
        ordinary compiler state sink.
        """
        if self.channel is None:
            return
        keys = tuple(keys)
        if keys != tuple(stage.key for stage in self.stages[:len(keys)]):
            raise ValueError("inherited compiler stages must be a declared prefix")
        with self._lock:
            self._entered_compiler_stage = bool(keys)
            for key in keys:
                self._stage_profiles[key] = {"failed": False, "rss_bytes": 0,
                                             "inherited": True}
                self._stage_seconds[key] = 0.0
            if len(keys) < len(self.stages):
                self._stage_index = len(keys)
                self._stage_started = self.clock()
                stage = self.stages[self._stage_index]
                self.channel.progress(
                    stage.label, done=len(keys), total=len(self.stages),
                    path="compile/stage", scope="stage", key=stage.key,
                    index=self._stage_index, started=self._stage_started,
                    metric="stage-order", fraction_is_progress=False,
                )
            self._publish_state("running")

    def stage(self, key: str) -> None:
        """Enter a declared stage; the previous stage's nested bars close."""

        if self.channel is None:
            return
        with self._lock:
            index = next(
                (i for i, item in enumerate(self.stages) if item.key == key),
                None,
            )
            if index is None:
                raise ValueError(
                    f"undeclared compile stage {key!r}; declared: "
                    f"{[item.key for item in self.stages]}"
                )
            entered = self._entered_compiler_stage
            self._entered_compiler_stage = True
            if index == self._stage_index:
                if not entered:
                    self._phase_log(f"stage {index + 1}/{len(self.stages)} {self.stages[index].label}: begin")
                return
            self._finish_substage()
            self._log_stage_finished(
                self.stages[self._stage_index], failed=False
            )
            for count_key in tuple(self._counts):
                self.count_close(count_key)
            self._counts.clear()
            self._stage_index = index
            self._stage_started = self.clock()
            self._substage = ""
            self._substage_started = self._stage_started
            self._phase_log(f"stage {index + 1}/{len(self.stages)} {self.stages[index].label}: begin")
            self.channel.progress(
                self.stages[index].label, done=index, total=len(self.stages),
                path="compile/stage", scope="stage", key=key, index=index,
                started=self._stage_started,
                metric="stage-order", fraction_is_progress=False,
            )
            self._publish_state("running")

    def _finish_substage(self, *, failed: bool = False) -> None:
        if self._substage:
            self._phase_log(f"compiler phase: {self._substage}: {'FAILED' if failed else 'end'} "
                     f"wall={self.clock() - self._substage_started:.3f}s")
        self._substage = ""

    def substage(self, name: str, *, keep: Iterable[str] = ()) -> None:
        """Name the part of the stage now running (the bar's postfix).

        Nested bars close with it, except those named in ``keep``.
        """

        if self.channel is None:
            return
        with self._lock:
            if str(name) != self._substage:
                self._finish_substage()
                self._substage_started = self.clock()
                self._phase_log(f"compiler phase: {name}: begin")
            keep = tuple(keep)
            for key in tuple(self._counts):
                if key not in keep:
                    self.count_close(key)
            self._substage = str(name)
            self.channel.progress(
                str(name), done=self._stage_index, total=len(self.stages),
                path="compile/substage", scope="substage", keep=keep,
                metric="stage-order", fraction_is_progress=False,
            )
            self._publish_state("running")

    # -- nested bars ------------------------------------------------------------

    def count(
        self,
        key: str,
        n: int | None = None,
        *,
        total: int | None = None,
        label: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Set (or, with ``n=None``, advance by one) the nested bar ``key``.

        Opened on first use; closed by ``count_close``, by ``iterate`` ending,
        or with the next stage.  ``total=None`` is an open-ended count.
        """

        from .graph_work import raise_if_compiler_work_cancelled
        raise_if_compiler_work_cancelled()
        if self.channel is None:
            return
        with self._lock:
            state = self._counts.get(key)
            fresh = state is None
            if fresh:
                state = self._counts[key] = {
                    "n": 0, "total": None, "label": key, "detail": "",
                    "emitted": float("-inf"),
                    "logged": float("-inf"),
                }
            state["n"] = state["n"] + 1 if n is None else int(n)
            if total is not None:
                state["total"] = int(total)
            if label:
                state["label"] = str(label)
            if detail is not None:
                state["detail"] = str(detail)
            now = self.clock()
            finished = (
                state["total"] is not None and state["n"] >= state["total"]
            )
            if not (fresh or finished) and (
                now - state["emitted"] < self.min_interval
            ):
                return
            state["emitted"] = now
            if fresh or finished or now - state["logged"] >= 10.0:
                state["logged"] = now
                self._phase_log(f"compiler work: {state['label']} {state['n']}/{state['total'] or '?'}"
                         + (f" {state['detail']}" if state["detail"] else ""))
                self._publish_state("running")
            self.channel.progress(
                state["label"], done=state["n"], total=state["total"] or 0,
                path=f"compile/count/{key}", scope="count", key=key,
                detail=state["detail"], bar_total=state["total"],
            )

    def _set_detail(self, key: str, detail: str) -> None:
        """Re-word an open bar without moving it."""

        with self._lock:
            state = self._counts.get(key)
        if state is not None:
            self.count(key, state["n"], detail=detail)

    def count_close(self, key: str) -> None:
        if self.channel is None:
            return
        with self._lock:
            state = self._counts.pop(key, None)
            if state is None:
                return
            stage = self.stages[self._stage_index].key
            prior = self._work.setdefault(stage, {}).setdefault(
                key, {"done": 0, "total": 0, "label": state["label"]})
            prior["done"] += state["n"]
            prior["total"] = (None if prior["total"] is None or state["total"] is None
                              else prior["total"] + state["total"])
            self.channel.progress(
                key, done=0, total=0, path=f"compile/count/{key}",
                scope="count", key=key, close=True,
            )

    def worker(
        self,
        key: str,
        *,
        label: str,
        status: str,
        stage_number: int = 0,
        stage_count: int = 0,
        stage_label: str = "",
        substage: str = "",
        elapsed_seconds: float = 0.0,
        rss_bytes: int = 0,
        stages: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        """Update one compiler worker lane in this display's single frame."""

        if self.channel is None:
            return
        with self._lock:
            prior = self._workers.get(str(key), {})
            if prior.get("status") in {"failed", "partial"} and prior.get("label") != label:
                # The thread may accept another source, but its failed receipt
                # must remain visible until the outer compilation ends.
                self._workers[f"{key}:failed:{prior['label']}"] = dict(prior)
            self._workers[str(key)] = {
                "key": str(key), "label": str(label), "status": str(status),
                "stage_number": int(stage_number or prior.get("stage_number") or 0),
                "stage_count": int(stage_count or prior.get("stage_count") or 0),
                "stage_label": str(stage_label or prior.get("stage_label") or ""),
                "substage": str(substage or prior.get("substage") or ""),
                "elapsed_seconds": float(
                    elapsed_seconds or prior.get("elapsed_seconds") or 0.0
                ),
                "rss_bytes": int(rss_bytes or prior.get("rss_bytes") or 0),
                "stages": [dict(row) for row in stages]
                or list(prior.get("stages") or ()),
            }
            self._publish_state("running")
            if self.bars is not None:
                if isinstance(self.bars, _CompileLive):
                    self.bars.heartbeat()
                else:
                    self.bars.heartbeat()

    def worker_clear(self) -> None:
        with self._lock:
            self._workers.clear()
            if self.bars is not None:
                if isinstance(self.bars, _CompileLive):
                    self.bars.heartbeat(force=True)
                else:
                    self.bars.heartbeat()

    def compilation_event(self, event: Mapping[str, Any]) -> None:
        """Consume scheduler events from the universal source frontend."""
        kind = event.get("stage")
        if kind in {"worker_output", "unit_failed", "native_failed"}:
            self.log(f"[{event.get('qualified_name', '')}] {event.get('message', event.get('error', ''))}")
        elif kind == "worker_snapshot":
            for record in event.get("running", ()):
                current = record.get("stage") or {}
                compiler = current.get("compiler") or {}
                self.worker(f"process-{record['process_id']}",
                    label=record["qualified_name"], status="running",
                    stage_number=compiler.get("stage_number", 0), stage_count=compiler.get("stage_count", len(LOWERING_STAGES)),
                    stage_label=compiler.get("stage_label", "starting"), substage=compiler.get("substage", ""),
                    stages=compiler.get("stages", ()), elapsed_seconds=record.get("elapsed_seconds", 0),
                    rss_bytes=record.get("resident_bytes", 0))
                for lane in compiler.get("workers", ()):
                    self.worker(f"process-{record['process_id']}:{lane['key']}",
                        **{key: value for key, value in lane.items() if key != "key"})
        elif kind == "unit_complete":
            compiler = (event.get("compiler_stage") or {}).get("compiler") or {}
            for key, record in tuple(self._workers.items()):
                if record["label"] == event.get("qualified_name"):
                    self.worker(key, label=record["label"], status=event.get("status", "complete"),
                        stage_number=compiler.get("stage_number", 0), stage_count=compiler.get("stage_count", 0),
                        stages=compiler.get("stages", ()))
        elif kind == "resolved_worker_snapshot":
            units = event.get("units") or ()
            batch = active_rule_batch_progress()
            if batch is not None:
                batch.worker_jobs = max(1, int(event.get("worker_jobs", 1)))
            for record in (*event.get("running", ()), *event.get("completed", ())):
                index = int(record["unit_index"])
                unit = units[index] if index < len(units) else record.get("unit", {})
                current = record.get("stage") or {}
                compiler = current.get("compiler") or {}
                self.worker(
                    str(index), label=", ".join(unit.get("qualified_names") or (str(index),)),
                    status=str(record.get("status") or "running"),
                    stage_number=compiler.get("stage_number", 0),
                    stage_count=compiler.get("stage_count", 0),
                    stage_label=compiler.get("stage_label", ""),
                    substage=compiler.get("substage", ""),
                    stages=compiler.get("stages", ()),
                    elapsed_seconds=record.get("elapsed_seconds", 0),
                    rss_bytes=record.get("resident_bytes", record.get("peak_resident_bytes", 0)),
                )
                if batch is not None and record.get("status") in {"verified", "compiled-unverified"}:
                    for name in unit.get("qualified_names", ()):
                        batch.complete(name, record.get("elapsed_seconds", 0))
            self.count("compiler-units", int(event.get("completed_count", 0)),
                       total=int(event.get("count", 0)), label="Compilation units")
        elif kind == "resolved_worker_log":
            label = ", ".join(event.get("qualified_names") or ())
            self.log(f"[{label}] {event.get('message', '')}")

    def iterate(
        self,
        iterable: Iterable[Any],
        key: str,
        *,
        label: str | None = None,
        total: int | None = None,
    ) -> Iterator[Any]:
        """Yield ``iterable`` unchanged, driving the nested bar ``key``."""

        if total is None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except TypeError:
                total = None
        self.count(key, 0, total=total, label=label)
        try:
            for item in iterable:
                yield item
                # Resuming the iterator means the caller finished the item.
                # Yielding it is not completion; an exception/break must not
                # count an unfinished item as completed.
                self.count(key)
        finally:
            self.count_close(key)

    # -- the deployment planner's structured events --------------------------------

    def selection(
        self,
        stage: str,
        state: str,
        depth: int,
        function: str,
        facts: Mapping[str, Any],
    ) -> None:
        """One ``deployment-select`` event (``strategize_shell_deployment``'s
        ``selection_event``), as the same facts the text line carries.

        The root's phases name the sub-stage; a child shell's events name what
        the function-shell bar is on; the callsite-specialization,
        fold and region-reduction facts drive their own nested bars.
        """

        if self.channel is None:
            return
        if state in ("end", "failed") and "duration_seconds" in facts:
            with self._lock:
                self.channel.profile(
                    stage, path=f"compile/planner/{stage}", section="planner",
                    nanoseconds=round(float(facts["duration_seconds"]) * 1e9),
                    failed=state == "failed", function=function,
                )
        inside_function_shells = "function-shells" in self._counts
        if depth == 0:
            if (
                state == "begin"
                and stage not in _SELECTION_NOT_A_PHASE
                and not inside_function_shells
            ):
                self.substage(stage, keep=("function-shells",))
            if stage == "function-shell" and state == "begin":
                self._set_detail(
                    "function-shells", str(facts.get("child_function", function))
                )
        elif state == "begin":
            self._set_detail("function-shells", f"{function} {stage}")
        if state in ("end", "failed"):
            for key in _SELECTION_PHASE_BARS.get(stage, ()):
                self.count_close(key)
        specialization = facts.get("specialization_state")
        if specialization == "round-begin":
            self.count(
                "callsite-callers", 0, total=int(facts.get("graphs", 0)),
                label=f"callsite specialization round {facts.get('round', '?')}: callers",
            )
        elif specialization == "prefold-end":
            self.count_close("fold-iteration")
            self.count(
                "callsite-callers", int(facts.get("caller_index", 0)) + 1,
                detail=str(facts.get("caller", "")),
            )
        elif specialization == "prefold-fixed-point":
            self.count(
                "fold-iteration", int(facts.get("fold_iteration", 0)),
                label="structural fold iterations",
                detail=f"{facts.get('caller', '')} changed={facts.get('changed')}",
            )
        elif specialization == "callsite-progress":
            self.count(
                "callsites", int(facts.get("callsites", 0)),
                label="callsites this round",
                detail=f"{facts.get('caller', '')} -> {facts.get('callee', '')}",
            )
        elif specialization == "round-end":
            self.count_close("callsites")
            self.count_close("callsite-callers")
            self.count_close("fold-iteration")
        if facts.get("reduction_state") is not None:
            self.count(
                "region-reduction", int(facts.get("iteration", 0)),
                label="shader region reduction iterations",
                detail=(
                    f"{facts.get('reduction_state')} "
                    f"regions={facts.get('regions')} merges={facts.get('merges')}"
                ),
            )


def compile_progress(
    progress_bars: bool | None = None,
    *,
    stages: Sequence[CompileStage] = LOWERING_STAGES,
    **options: Any,
) -> CompileProgress:
    """The layer for a compile.  ``progress_bars=None`` is on exactly when
    stdout (and the stream the bars draw on) is a terminal.

    If a layer is already running the call joins it: a nested compile shares
    the outer compile's bars, whatever it asked for.
    """

    active = _ACTIVE_COMPILE_PROGRESS.get()
    if active is not None:
        return active
    if progress_bars is None:
        progress_bars = progress_bars_default(options.get("stream"))
    return CompileProgress(enabled=bool(progress_bars), stages=stages, **options)


def active_compile_progress() -> CompileProgress | None:
    return _ACTIVE_COMPILE_PROGRESS.get()


def compile_stage(key: str) -> None:
    """Enter a declared compile stage.  No-op when no layer is running."""

    layer = _ACTIVE_COMPILE_PROGRESS.get()
    if layer is not None:
        layer.stage(key)


def compile_substage(name: str) -> None:
    layer = _ACTIVE_COMPILE_PROGRESS.get()
    if layer is not None:
        layer.substage(name)


@contextmanager
def compile_work(key: str, *, label: str):
    """An admitted thread working within its owner's current compiler stage."""
    from .graph_work import raise_if_compiler_work_cancelled
    raise_if_compiler_work_cancelled()
    layer = active_compile_progress()
    if layer is None:
        yield
        return
    identity = f"thread-{threading.get_ident()}"
    started = time.monotonic()
    stage = layer.stages[layer._stage_index]
    rows = layer.summary()["stages"]
    states = [{"key": row["key"], "label": row["label"], "state": row["state"]}
              for row in rows]
    def publish(status):
        layer.worker(identity, label=label, status=status,
                     stage_number=layer._stage_index + 1, stage_count=len(layer.stages),
                     stage_label=stage.label, substage=key, stages=states,
                     elapsed_seconds=time.monotonic() - started,
                     rss_bytes=layer.rss())
    publish("running")
    try:
        yield
    except BaseException:
        publish("failed")
        raise
    else:
        publish("done")


def compile_count(
    key: str,
    n: int | None = None,
    *,
    total: int | None = None,
    label: str | None = None,
    detail: str | None = None,
) -> None:
    layer = _ACTIVE_COMPILE_PROGRESS.get()
    if layer is not None:
        layer.count(key, n, total=total, label=label, detail=detail)


def compile_count_close(key: str) -> None:
    layer = _ACTIVE_COMPILE_PROGRESS.get()
    if layer is not None:
        layer.count_close(key)


def compile_iter(
    iterable: Iterable[Any],
    key: str,
    *,
    label: str | None = None,
    total: int | None = None,
) -> Iterable[Any]:
    """``iterable`` itself when no layer is running; otherwise the same items
    with a nested bar driven as they are consumed."""

    layer = _ACTIVE_COMPILE_PROGRESS.get()
    if layer is None:
        return iterable
    return layer.iterate(iterable, key, label=label, total=total)


def compile_selection_event(
    stage: str,
    state: str,
    depth: int,
    function: str,
    facts: Mapping[str, Any],
) -> None:
    layer = _ACTIVE_COMPILE_PROGRESS.get()
    if layer is not None:
        layer.selection(stage, state, depth, function, facts)


def declare_progress_bars_policy(enabled: bool, book: Any = None) -> None:
    """Record the progress-bar choice on the active book's ``compile_policy``
    page.

    One root row, NOVEL(POLICY_DECLARATION) at the ``compile_entry`` stage,
    the way the work contract's other compile policies are declared.  A book
    resumed from an earlier compile already holds its first choice; that row
    is left alone rather than contradicted.
    """

    from .concordance_declarations import (
        COMPILE_ENTRY, COMPILE_POLICY, POLICY_DECLARATION,
    )
    from .identity_concordance import Mode, Novel, current_identity_book

    if book is None:
        book = current_identity_book()
    row = ("progress_bars",)
    if book.page(COMPILE_POLICY).latest(row) is not None:
        return
    book.post(
        COMPILE_POLICY, row, "on" if enabled else "off",
        stage=COMPILE_ENTRY, provenance=Novel(POLICY_DECLARATION, ()),
        mode=Mode.CONCORD,
    )


def compile_profile_dataframe(summary_or_module):
    """View an accumulated lowering summary (or SSA module) without re-profiling.

    Pandas is optional and imported only here; normal compilation needs none.
    The module form reads ``module.metadata['compile_profile']``.
    """
    import pandas as pd
    summary = (summary_or_module if isinstance(summary_or_module, Mapping)
               else summary_or_module.metadata["compile_profile"])
    result = pd.DataFrame.from_records(summary["stages"], columns=(
        "key", "label", "state", "seconds", "rss_bytes"))
    result.attrs["work"] = summary["work"]
    result.attrs["profiles"] = summary["profiles"]
    result.attrs["timing_semantics"] = "stage wall time; no remaining-time estimate"
    return result


__all__ = [
    "COMPILE_STAGES",
    "CompileProgress",
    "CompileStage",
    "ERROR",
    "LOWERING_STAGES",
    "KINDS",
    "LOG",
    "PROFILE",
    "PROGRESS",
    "Record",
    "SCHEMA",
    "TelemetryChannel",
    "active_compile_progress",
    "active_rule_batch_progress",
    "attach_error_buffer",
    "compile_count",
    "compile_count_close",
    "compile_iter",
    "compile_progress",
    "compile_rule_batch",
    "compile_profile_dataframe",
    "compile_selection_event",
    "compile_stage",
    "compile_substage",
    "declare_progress_bars_policy",
    "attach_profiler",
    "process_rss_bytes",
    "system_available_memory_bytes",
    "progress_bars_default",
    "record_launch_profile",
    "record_shortfalls",
    "summarize_process_graph",
]
