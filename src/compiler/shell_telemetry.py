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

import json
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


__all__ = [
    "ERROR",
    "KINDS",
    "LOG",
    "PROFILE",
    "PROGRESS",
    "Record",
    "SCHEMA",
    "TelemetryChannel",
    "attach_error_buffer",
    "attach_profiler",
    "record_launch_profile",
    "record_shortfalls",
    "summarize_process_graph",
]
