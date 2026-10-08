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
import json
import sys
import threading
import time
import traceback
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
        path = kwargs.get("path") or (args[0] if args else "")
        nanoseconds = (
            kwargs.get("elapsed_ns")
            or kwargs.get("duration_ns")
            or kwargs.get("nanoseconds")
            or 0
        )
        channel.emit(
            PROFILE,
            str(kwargs.get("label") or kwargs.get("phase") or "record"),
            path=str(path),
            nanoseconds=int(nanoseconds or 0),
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


def process_rss_bytes() -> int:
    """This process's resident set (working set on Windows), in bytes.

    The one RSS reading the compiler uses (stage-boundary progress lines and
    the ``memory_budget_bytes`` regulation read it; nothing else should
    re-derive it).  Returns 0 when the platform offers no reading.
    """

    import sys

    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
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

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        # K32GetProcessMemoryInfo lives in kernel32 on Windows 7 and later.
        query = kernel32.K32GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters),
                          wintypes.DWORD]
        query.restype = wintypes.BOOL
        if not query(kernel32.GetCurrentProcess(), ctypes.byref(counters),
                     counters.cb):
            return 0
        return int(counters.WorkingSetSize)
    try:
        with open("/proc/self/statm", "rb") as stream:
            import os

            return int(stream.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


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
        self.peak_rss = 0
        self.last_rss = 0
        self._last_rss_at = float("-inf")
        self.failure: str | None = None

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
                self.created.pop(key, None)
                self.bars.pop(key).close()

    def heartbeat(self) -> None:
        """Refresh the clocks and memory without any event arriving."""

        with self.lock:
            if self.main is None or self.failure is not None:
                return
            self.main.set_postfix_str(self._main_postfix(), refresh=False)
            self.main.refresh()
            now = self.owner.clock()
            for key, bar in self.bars.items():
                bar.set_postfix_str(self._nested_postfix(key), refresh=False)
                # ``delay`` is only honoured by ``update``; a bar for a loop
                # that has not yet run long enough must stay hidden here too.
                if now - self.created[key] >= self.owner.nested_delay:
                    bar.refresh()

    # -- the subscriber ------------------------------------------------------

    def __call__(self, record: Record) -> None:
        with self.lock:
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
            total=len(owner.stages),
            desc=self._stage_desc(),
            unit="stage",
            file=owner.stream,
            leave=True,
            dynamic_ncols=True,
            mininterval=0.25,
            bar_format=(
                "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}]{postfix}"
            ),
        )
        self.main.set_postfix_str(self._main_postfix(), refresh=True)

    def _stage(self, record: Record) -> None:
        self._close_nested()
        self.stage_index = int(record.detail["index"])
        self.stage_started = float(record.detail["started"])
        self.substage = ""
        self.main.n = self.stage_index
        self.main.set_description_str(self._stage_desc(), refresh=False)
        self.main.set_postfix_str(self._main_postfix(), refresh=False)
        self.main.refresh()

    def _count(self, record: Record) -> None:
        detail = record.detail
        key = str(detail["key"])
        if detail.get("close"):
            self.details.pop(key, None)
            self.created.pop(key, None)
            bar = self.bars.pop(key, None)
            if bar is not None:
                bar.close()
            return
        total = detail.get("bar_total")
        bar = self.bars.get(key)
        if bar is None:
            self.created[key] = self.owner.clock()
            bar = self.bars[key] = self.owner.tqdm_class(
                total=total,
                desc=record.message,
                unit="it",
                file=self.owner.stream,
                leave=False,
                dynamic_ncols=True,
                mininterval=0.25,
                delay=self.owner.nested_delay,
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
                f"compile FAILED in {self.owner.stages[self.stage_index].label}",
                refresh=False,
            )
        else:
            main.n = total
            main.set_description_str(f"compile {total}/{total} done", refresh=False)
        main.set_postfix_str(
            f"peak {max(self.peak_rss, self._rss()) / _GIB:.2f}GB, "
            f"stage {_format_duration(self.owner.clock() - self.stage_started)}",
            refresh=False,
        )
        main.refresh()
        main.close()
        self.main = None


_ACTIVE_COMPILE_PROGRESS: contextvars.ContextVar = contextvars.ContextVar(
    "turing_compile_progress", default=None
)


class CompileProgress:
    """The stage/loop progress layer for one compile.

    ``enabled=False`` builds nothing and changes nothing.  Enabled, it owns a
    ``TelemetryChannel`` (``log`` + ``progress`` records), subscribes the tqdm
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
    ):
        self.stages = tuple(stages)
        self.enabled = bool(enabled) and bool(self.stages)
        if self.enabled and tqdm_class is None:
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
        self.channel: TelemetryChannel | None = None
        self.bars: _CompileBars | None = None
        self._depth = 0
        self._token: Any = None
        self._lock = threading.RLock()
        self._counts: dict[str, dict[str, Any]] = {}
        self._stage_index = 0
        self._stage_started = clock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        if self.enabled:
            self.channel = channel or TelemetryChannel(
                name="compile-progress", capacity=1024, kinds=(LOG, PROGRESS)
            )
            self.bars = _CompileBars(self)
            self.channel.subscribe(self.bars)

    # -- lifecycle -------------------------------------------------------------

    def open(self) -> "CompileProgress":
        if not self.enabled:
            return self
        with self._lock:
            self._depth += 1
            if self._depth == 1:
                self._token = _ACTIVE_COMPILE_PROGRESS.set(self)
                self._stage_index = 0
                self._stage_started = self.clock()
                self.channel.progress(
                    self.stages[0].label, done=0, total=len(self.stages),
                    path="compile/start", scope="start",
                )
                if self.heartbeat_interval:
                    self._heartbeat_stop.clear()
                    self._heartbeat_thread = threading.Thread(
                        target=self._heartbeat_loop,
                        name="compile-progress-heartbeat",
                        daemon=True,
                    )
                    self._heartbeat_thread.start()
        return self

    def close(self, failed: bool = False) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._depth == 0:
                return
            self._depth -= 1
            if self._depth:
                return
            self._heartbeat_stop.set()
            thread, self._heartbeat_thread = self._heartbeat_thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        with self._lock:
            stage = self.stages[self._stage_index]
            self._log_stage_finished(stage, failed=failed)
            self._counts.clear()
            self.channel.progress(
                "compile failed" if failed else "compile done",
                done=len(self.stages) if not failed else self._stage_index,
                total=len(self.stages), path="compile/end",
                scope="end", failed=bool(failed),
            )
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
            _plain_compiler_line(message)
            return
        self.channel.log(message)

    def progress_callback(
        self, inner: Callable[[str], None] | None = None
    ) -> Callable[[str], None]:
        """The ``progress`` callback the compile should be handed.

        Off: the caller's own callback, or today's plain ``[compiler]`` line.
        On: this layer's ``log``; a caller-supplied callback is kept, and
        anything it prints is routed above the bars.
        """

        if not self.enabled:
            return inner if inner is not None else _plain_compiler_line
        if inner is None:
            return self.log

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

    def _log_stage_finished(self, stage: CompileStage, *, failed: bool) -> None:
        elapsed = _format_duration(self.clock() - self._stage_started)
        rss = self.bars._rss() / _GIB
        self.channel.log(
            f"stage {self._stage_index + 1}/{len(self.stages)} "
            f"{stage.label}: {'FAILED after' if failed else 'done in'} "
            f"{elapsed} (rss {rss:.2f}GB)"
        )

    def stage(self, key: str) -> None:
        """Enter a declared stage; the previous stage's nested bars close."""

        if not self.enabled:
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
            if index == self._stage_index:
                return
            self._log_stage_finished(
                self.stages[self._stage_index], failed=False
            )
            self._counts.clear()
            self._stage_index = index
            self._stage_started = self.clock()
            self.channel.progress(
                self.stages[index].label, done=index, total=len(self.stages),
                path="compile/stage", scope="stage", key=key, index=index,
                started=self._stage_started,
            )

    def substage(self, name: str, *, keep: Iterable[str] = ()) -> None:
        """Name the part of the stage now running (the bar's postfix).

        Nested bars close with it, except those named in ``keep``.
        """

        if not self.enabled:
            return
        with self._lock:
            keep = tuple(keep)
            for key in tuple(self._counts):
                if key not in keep:
                    del self._counts[key]
            self.channel.progress(
                str(name), done=self._stage_index, total=len(self.stages),
                path="compile/substage", scope="substage", keep=keep,
            )

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

        if not self.enabled:
            return
        with self._lock:
            state = self._counts.get(key)
            fresh = state is None
            if fresh:
                state = self._counts[key] = {
                    "n": 0, "total": None, "label": key, "detail": "",
                    "emitted": float("-inf"),
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
        if not self.enabled:
            return
        with self._lock:
            if self._counts.pop(key, None) is None:
                return
            self.channel.progress(
                key, done=0, total=0, path=f"compile/count/{key}",
                scope="count", key=key, close=True,
            )

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
                self.count(key)
                yield item
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

        if not self.enabled:
            return
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
    "attach_error_buffer",
    "compile_count",
    "compile_count_close",
    "compile_iter",
    "compile_progress",
    "compile_selection_event",
    "compile_stage",
    "compile_substage",
    "declare_progress_bars_policy",
    "attach_profiler",
    "process_rss_bytes",
    "progress_bars_default",
    "record_launch_profile",
    "record_shortfalls",
    "summarize_process_graph",
]
