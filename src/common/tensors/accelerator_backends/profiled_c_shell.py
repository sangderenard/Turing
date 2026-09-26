"""One profiled C launch boundary shared by accelerated compute targets.

The shell does not know whether its callback enters a C kernel, LLVM machine
code, or an OpenGL dispatch.  It owns the common control contract:

* call exactly one already-compiled compute closure;
* measure host launch/control duration in C;
* retain a separately supplied device duration when the closure is a GPU
  dispatch;
* return an explicit status instead of swallowing backend failure.

Target adapters are responsible for exposing ``turing_compute_closure``.  LLVM
can emit this ABI directly; C can provide a native wrapper around its prepared
program; GLSL writes the resolved timer-query duration through ``device_ns``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Callable


_C_TRACE_DECLARATIONS = r"""
typedef struct TuringTraceRecord {
    unsigned long long sequence;
    unsigned long long shell_ns;
    unsigned long long device_ns;
    int region;
    int status;
} TuringTraceRecord;

typedef struct TuringTraceRing {
    TuringTraceRecord *records;
    unsigned long long capacity;
    unsigned long long written;
    unsigned long long drained;
} TuringTraceRing;

typedef struct TuringTraceSite {
    TuringTraceRing *ring;
    int region;
    int reserved;
} TuringTraceSite;

void turing_trace_ring_reset(
    TuringTraceRing *ring,
    TuringTraceRecord *storage,
    unsigned long long capacity
);
void turing_trace_logger(void *user, const TuringLaunchProfile *profile);
turing_launch_logger turing_trace_logger_address(void);
unsigned long long turing_trace_available(const TuringTraceRing *ring);
unsigned long long turing_trace_lost(const TuringTraceRing *ring);
unsigned long long turing_trace_drain(
    TuringTraceRing *ring,
    TuringTraceRecord *out,
    unsigned long long limit
);
"""


_C_DECLARATIONS = r"""
typedef int (*turing_compute_closure)(
    void *context,
    unsigned long long *device_ns
);

typedef struct TuringLaunchProfile {
    unsigned long long shell_ns;
    unsigned long long device_ns;
    int status;
    int language;
} TuringLaunchProfile;

typedef struct TuringLaunchStats {
    unsigned long long calls;
    unsigned long long failures;
    unsigned long long shell_ns_total;
    unsigned long long shell_ns_min;
    unsigned long long shell_ns_max;
    unsigned long long device_ns_total;
    unsigned long long overhead_ns;
} TuringLaunchStats;

typedef void (*turing_launch_logger)(
    void *user,
    const TuringLaunchProfile *profile
);

int turing_profiled_launch(
    turing_compute_closure compute,
    void *context,
    TuringLaunchProfile *profile
);

void turing_launch_stats_reset(TuringLaunchStats *stats);
int turing_null_closure(void *context, unsigned long long *device_ns);
unsigned long long turing_measure_launch_overhead(int repeats);
unsigned long long turing_measure_launch_overhead_ps(int repeats);

int turing_profiled_launch_ex(
    turing_compute_closure compute,
    void *context,
    TuringLaunchProfile *profile,
    TuringLaunchStats *stats,
    turing_launch_logger logger,
    void *logger_user,
    int language
);

"""


_C_TRACE_SOURCE = r"""
/* Trace digest.

   The logger hook fires once per launch, which is the right granularity to
   watch a program run -- but calling into Python there would cost more than
   the launch being measured, which is the same reason TuringLaunchStats
   accumulates in C rather than round-tripping. So the default logger writes a
   fixed-size record into a ring the artifact owns, and a reader drains it
   whenever it likes. The launch pays four stores; nothing crosses a language
   boundary until someone asks.

   ``written`` and ``drained`` are monotonic totals rather than wrapped
   indices, so a reader can tell the difference between an empty ring and one
   that lapped it: if more than ``capacity`` records accumulated since the last
   drain, the oldest are simply gone and the reader is told how many. Losing
   the tail of a burst is the correct failure -- the alternative is stalling
   the program to keep its own telemetry. */

typedef struct TuringTraceRecord {
    unsigned long long sequence;
    unsigned long long shell_ns;
    unsigned long long device_ns;
    int region;
    int status;
} TuringTraceRecord;

typedef struct TuringTraceRing {
    TuringTraceRecord *records;
    unsigned long long capacity;
    unsigned long long written;
    unsigned long long drained;
} TuringTraceRing;

typedef struct TuringTraceSite {
    TuringTraceRing *ring;
    int region;
    int reserved;
} TuringTraceSite;

void turing_trace_ring_reset(
    TuringTraceRing *ring,
    TuringTraceRecord *storage,
    unsigned long long capacity
) {
    if (ring == 0) {
        return;
    }
    ring->records = storage;
    ring->capacity = storage == 0 ? 0 : capacity;
    ring->written = 0;
    ring->drained = 0;
}

void turing_trace_logger(void *user, const TuringLaunchProfile *profile) {
    TuringTraceSite *site = (TuringTraceSite *)user;
    if (site == 0 || profile == 0) {
        return;
    }
    TuringTraceRing *ring = site->ring;
    if (ring == 0 || ring->capacity == 0 || ring->records == 0) {
        return;
    }
    TuringTraceRecord *record =
        &ring->records[ring->written % ring->capacity];
    record->sequence = ring->written;
    record->shell_ns = profile->shell_ns;
    record->device_ns = profile->device_ns;
    record->region = site->region;
    record->status = profile->status;
    ring->written += 1;
}

/* The launch wants a function pointer. Handing back C's own address avoids
   depending on how the binding happens to expose library symbols. */
turing_launch_logger turing_trace_logger_address(void) {
    return turing_trace_logger;
}

unsigned long long turing_trace_available(const TuringTraceRing *ring) {
    if (ring == 0 || ring->capacity == 0) {
        return 0;
    }
    unsigned long long pending = ring->written - ring->drained;
    return pending > ring->capacity ? ring->capacity : pending;
}

unsigned long long turing_trace_lost(const TuringTraceRing *ring) {
    if (ring == 0 || ring->capacity == 0) {
        return 0;
    }
    unsigned long long pending = ring->written - ring->drained;
    return pending > ring->capacity ? pending - ring->capacity : 0;
}

unsigned long long turing_trace_drain(
    TuringTraceRing *ring,
    TuringTraceRecord *out,
    unsigned long long limit
) {
    if (ring == 0 || out == 0 || ring->capacity == 0 || ring->records == 0) {
        return 0;
    }
    unsigned long long written = ring->written;
    unsigned long long drained = ring->drained;
    unsigned long long pending = written - drained;
    if (pending > ring->capacity) {
        /* The writer lapped the reader; resume at the oldest survivor. */
        drained = written - ring->capacity;
        pending = ring->capacity;
    }
    unsigned long long count = pending < limit ? pending : limit;
    for (unsigned long long index = 0; index < count; ++index) {
        out[index] = ring->records[(drained + index) % ring->capacity];
    }
    ring->drained = drained + count;
    return count;
}
"""


_C_SOURCE = r"""
#include <stdint.h>

#if defined(_WIN32)
#include <windows.h>
static uint64_t turing_monotonic_ns(void) {
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (uint64_t)(
        ((double)counter.QuadPart * 1000000000.0)
        / (double)frequency.QuadPart
    );
}
#else
#include <time.h>
static uint64_t turing_monotonic_ns(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (
        (uint64_t)value.tv_sec * UINT64_C(1000000000)
        + (uint64_t)value.tv_nsec
    );
}
#endif

typedef int (*turing_compute_closure)(
    void *context,
    unsigned long long *device_ns
);

typedef struct TuringLaunchProfile {
    unsigned long long shell_ns;
    unsigned long long device_ns;
    int status;
    int language;
} TuringLaunchProfile;

/* Accumulated across launches so a caller can characterise a dispatch without
   round-tripping every individual call into Python, which would cost more than
   the launch being measured. */
typedef struct TuringLaunchStats {
    unsigned long long calls;
    unsigned long long failures;
    unsigned long long shell_ns_total;
    unsigned long long shell_ns_min;
    unsigned long long shell_ns_max;
    unsigned long long device_ns_total;
    unsigned long long overhead_ns;
} TuringLaunchStats;

typedef void (*turing_launch_logger)(
    void *user,
    const TuringLaunchProfile *profile
);

void turing_launch_stats_reset(TuringLaunchStats *stats) {
    if (stats == 0) {
        return;
    }
    stats->calls = 0;
    stats->failures = 0;
    stats->shell_ns_total = 0;
    stats->shell_ns_min = ~(unsigned long long)0;
    stats->shell_ns_max = 0;
    stats->device_ns_total = 0;
    stats->overhead_ns = 0;
}

/* A closure that does nothing.  Timing the shell around it measures what the
   launch boundary itself costs, so that overhead can be reported separately
   from -- or subtracted from -- real dispatch timings. */
int turing_null_closure(void *context, unsigned long long *device_ns) {
    (void)context;
    if (device_ns != 0) {
        *device_ns = 0;
    }
    return 1;
}

int turing_profiled_launch_ex(
    turing_compute_closure compute,
    void *context,
    TuringLaunchProfile *profile,
    TuringLaunchStats *stats,
    turing_launch_logger logger,
    void *logger_user,
    int language
) {
    uint64_t started;
    uint64_t finished;
    uint64_t device_ns = 0;
    uint64_t shell_ns;
    int status;

    if (compute == 0 || profile == 0) {
        return 0;
    }
    started = turing_monotonic_ns();
    status = compute(context, &device_ns);
    finished = turing_monotonic_ns();
    shell_ns = finished - started;

    profile->shell_ns = shell_ns;
    profile->device_ns = device_ns;
    profile->status = status;
    profile->language = language;

    if (stats != 0) {
        if (stats->calls == 0 && stats->shell_ns_min == 0) {
            stats->shell_ns_min = ~(unsigned long long)0;
        }
        stats->calls += 1;
        if (!status) {
            stats->failures += 1;
        }
        stats->shell_ns_total += shell_ns;
        stats->device_ns_total += device_ns;
        if (shell_ns < stats->shell_ns_min) {
            stats->shell_ns_min = shell_ns;
        }
        if (shell_ns > stats->shell_ns_max) {
            stats->shell_ns_max = shell_ns;
        }
    }
#if TURING_TRACE
    if (logger != 0) {
        logger(logger_user, profile);
    }
#else
    /* Diagnostics were not requested, so the hook is not compiled in at all
       -- not merely skipped. A launch pays nothing for a facility it was not
       built with, which is the only honest meaning of opt-in here. */
    (void)logger;
    (void)logger_user;
#endif
    return status;
}

int turing_profiled_launch(
    turing_compute_closure compute,
    void *context,
    TuringLaunchProfile *profile
) {
    return turing_profiled_launch_ex(
        compute, context, profile, 0, 0, 0, 0
    );
}

/* Cost of the launch boundary itself, in picoseconds.

   Timing one empty call and taking the minimum does not work: the platform
   clock resolves to roughly 100ns on Windows, an empty launch is far below
   that, and every sample floors to zero.  Instead time the whole batch and
   divide, which resolves well below one clock tick.  Picoseconds are returned
   because the answer is normally a small number of nanoseconds and integer
   nanoseconds would quantise it away again. */
unsigned long long turing_measure_launch_overhead_ps(int repeats) {
    uint64_t started;
    uint64_t finished;
    uint64_t device_ns = 0;
    int index;
    volatile int sink = 0;

    if (repeats < 1) {
        repeats = 1;
    }
    /* Warm instruction cache and branch predictors first. */
    for (index = 0; index < 64; ++index) {
        sink += turing_null_closure(0, &device_ns);
    }
    started = turing_monotonic_ns();
    for (index = 0; index < repeats; ++index) {
        sink += turing_null_closure(0, &device_ns);
    }
    finished = turing_monotonic_ns();
    (void)sink;
    return ((finished - started) * UINT64_C(1000)) / (uint64_t)repeats;
}

unsigned long long turing_measure_launch_overhead(int repeats) {
    return turing_measure_launch_overhead_ps(repeats) / UINT64_C(1000);
}
"""


class ShellLanguage(int, Enum):
    """Which implementation language served a launch.

    Carried through the C struct as a plain int so the shell can report what
    actually ran without the caller having to track it separately.  This is what
    makes a hot swap observable rather than silent.
    """

    UNKNOWN = 0
    C = 1
    LLVM = 2
    GLSL = 3
    # Explicit dispatcher spelling. ``GLSL`` remains the compatibility name;
    # this lane is desktop OpenGL compute, not browser WebGL/WGSL.
    NATIVE_GLSL = 3
    FORTRAN = 4
    PYTHON = 5


@dataclass(frozen=True)
class CLaunchProfile:
    shell_ns: int
    device_ns: int
    status: int
    language: ShellLanguage = ShellLanguage.UNKNOWN

    @property
    def shell_ms(self) -> float:
        return self.shell_ns / 1.0e6

    @property
    def device_ms(self) -> float:
        return self.device_ns / 1.0e6

    @property
    def host_ns(self) -> int:
        """Host-side time, excluding any reported device duration."""

        return max(0, self.shell_ns - self.device_ns)


@dataclass(frozen=True)
class LaunchStatistics:
    """Aggregate launch behaviour, accumulated inside C."""

    calls: int
    failures: int
    shell_ns_total: int
    shell_ns_min: int
    shell_ns_max: int
    device_ns_total: int
    overhead_ns: int = 0

    @property
    def mean_shell_ns(self) -> float:
        return self.shell_ns_total / self.calls if self.calls else 0.0

    @property
    def net_mean_shell_ns(self) -> float:
        """Mean launch cost with the measured boundary overhead removed."""

        return max(0.0, self.mean_shell_ns - self.overhead_ns)

    @property
    def overhead_fraction(self) -> float:
        """How much of an average launch is boundary cost rather than work."""

        mean = self.mean_shell_ns
        return 0.0 if mean <= 0 else min(1.0, self.overhead_ns / mean)

    def format(self) -> str:
        return (
            f"calls={self.calls} failures={self.failures} "
            f"mean={self.mean_shell_ns / 1e3:.2f}us "
            f"min={self.shell_ns_min / 1e3:.2f}us "
            f"max={self.shell_ns_max / 1e3:.2f}us "
            f"overhead={self.overhead_ns / 1e3:.2f}us "
            f"({self.overhead_fraction:.1%} of mean)"
        )


@dataclass(frozen=True)
class ProfiledCShell:
    ffi: Any
    library: Any
    # Whether the digest was compiled in. Reading it beats calling a symbol
    # that does not exist in this build.
    trace: bool = False

    def callback(
        self,
        function: Callable[[Any, Any], int],
    ):
        """Create a kept-alive callback for adapters not yet emitting the ABI."""

        return self.ffi.callback(
            "int(void *, unsigned long long *)",
            function,
        )

    def launch(
        self,
        compute: Any,
        context: Any = None,
        *,
        language: "ShellLanguage" = None,
        stats: Any = None,
        logger: Any = None,
        logger_user: Any = None,
    ) -> CLaunchProfile:
        if isinstance(compute, int):
            compute = self.ffi.cast("turing_compute_closure", compute)
        if context is None:
            context = self.ffi.NULL
        elif isinstance(context, int):
            context = self.ffi.cast("void *", context)
        profile = self.ffi.new("TuringLaunchProfile *")
        tag = int(language if language is not None else ShellLanguage.UNKNOWN)
        status = int(
            self.library.turing_profiled_launch_ex(
                compute,
                context,
                profile,
                self.ffi.NULL if stats is None else stats,
                self.ffi.NULL if logger is None else logger,
                # The profile struct carries no identity, so this is the only
                # channel by which a logger learns *what* just ran. Pinning it
                # to NULL meant a logger could fire but never say for whom.
                self.ffi.NULL if logger_user is None else (
                    self.ffi.cast("void *", logger_user)
                    if isinstance(logger_user, int) else logger_user
                ),
                tag,
            )
        )
        return CLaunchProfile(
            shell_ns=int(profile.shell_ns),
            device_ns=int(profile.device_ns),
            status=status,
            language=ShellLanguage(int(profile.language)),
        )

    def new_trace_ring(self, capacity: int = 4096):
        """Allocate a trace ring the artifact writes into without stopping.

        Returns ``(ring, keepalive)``. The keepalive holds the record storage;
        dropping it while a launch still points at the ring frees memory the C
        side would keep writing to.
        """

        capacity = max(1, int(capacity))
        storage = self.ffi.new("TuringTraceRecord[]", capacity)
        ring = self.ffi.new("TuringTraceRing *")
        self.library.turing_trace_ring_reset(ring, storage, capacity)
        return ring, storage

    def trace_site(self, ring: Any, region: int):
        """Bind a region identity to a ring, for one launch site.

        This is what goes in ``logger_user``: the profile the logger receives
        says how long the launch took but not what it was, so identity has to
        arrive alongside it.
        """

        site = self.ffi.new("TuringTraceSite *")
        site.ring = ring
        site.region = int(region)
        site.reserved = 0
        return site

    @property
    def trace_logger(self):
        """The built-in C logger. Writes to the ring; never enters Python.

        Taken by address rather than as the bound library function: the launch
        wants a ``turing_launch_logger`` function pointer, and the attribute
        itself is a Python callable wrapper, which is exactly the round trip
        this exists to avoid.
        """

        return self.library.turing_trace_logger_address()

    def trace_pending(self, ring: Any) -> tuple[int, int]:
        """``(available, lost)`` without consuming anything."""

        return (
            int(self.library.turing_trace_available(ring)),
            int(self.library.turing_trace_lost(ring)),
        )

    def drain_trace(self, ring: Any, *, limit: int = 4096) -> list[dict]:
        """Skim whatever the artifact has written since the last skim."""

        limit = max(1, int(limit))
        out = self.ffi.new("TuringTraceRecord[]", limit)
        lost = int(self.library.turing_trace_lost(ring))
        count = int(self.library.turing_trace_drain(ring, out, limit))
        drained = []
        for index in range(count):
            record = out[index]
            drained.append({
                "sequence": int(record.sequence),
                "region": int(record.region),
                "shell_ns": int(record.shell_ns),
                "device_ns": int(record.device_ns),
                "status": int(record.status),
            })
        if lost and drained:
            drained[0]["lost_before"] = lost
        return drained

    def new_statistics(self):
        """Allocate a zeroed stats block the shell can accumulate into."""

        stats = self.ffi.new("TuringLaunchStats *")
        self.library.turing_launch_stats_reset(stats)
        return stats

    def read_statistics(self, stats, *, overhead_ns: int = 0) -> LaunchStatistics:
        calls = int(stats.calls)
        return LaunchStatistics(
            calls=calls,
            failures=int(stats.failures),
            shell_ns_total=int(stats.shell_ns_total),
            # The C sentinel for "no minimum yet" is all-ones; report zero.
            shell_ns_min=int(stats.shell_ns_min) if calls else 0,
            shell_ns_max=int(stats.shell_ns_max),
            device_ns_total=int(stats.device_ns_total),
            overhead_ns=overhead_ns,
        )

    def measure_launch_overhead_ps(self, repeats: int = 200000) -> int:
        """Cost of the launch boundary itself, in picoseconds.

        Timed over a batch rather than per call: an empty launch costs far less
        than one tick of the platform clock, so per-call samples all floor to
        zero.  Picoseconds keep the answer from being quantised away.
        """

        return int(
            self.library.turing_measure_launch_overhead_ps(int(repeats))
        )

    def measure_launch_overhead(self, repeats: int = 200000) -> float:
        """Launch boundary cost in nanoseconds, as a float."""

        return self.measure_launch_overhead_ps(repeats) / 1000.0

    def record(
        self,
        profiler: Any,
        profile: CLaunchProfile,
        *,
        path: str,
        label: str,
        dispatches: int = 1,
    ) -> None:
        """Publish the common measurements through DeploymentProfiler."""

        profiler.record(
            path=path,
            section="compiled-c-shell",
            label=label,
            cpu_ms=profile.shell_ms,
            gpu_ms=profile.device_ms,
            dispatches=dispatches,
        )


@lru_cache(maxsize=2)
def profiled_c_shell(*, trace: bool = False) -> ProfiledCShell:
    """Build the launch boundary, with diagnostics only if asked for.

    ``trace=False`` is not a runtime flag that skips the digest -- the digest
    is not in the binary. The ring, its logger, and the hook that would call
    it are all behind ``TURING_TRACE``, so an artifact built without
    diagnostics has no trace code to execute, no branch to predict, and no
    symbol to resolve. Turning it on is a compile-time decision, cached
    separately so both shells can coexist in one process.
    """

    from cffi import FFI

    ffi = FFI()
    declarations = _C_DECLARATIONS
    source = "#define TURING_TRACE {}\n".format(1 if trace else 0) + _C_SOURCE
    if trace:
        declarations = declarations + _C_TRACE_DECLARATIONS
        source = source + _C_TRACE_SOURCE
    ffi.cdef(declarations)
    library = ffi.verify(source)
    return ProfiledCShell(ffi=ffi, library=library, trace=bool(trace))


__all__ = [
    "CLaunchProfile",
    "LaunchStatistics",
    "ProfiledCShell",
    "ShellLanguage",
    "profiled_c_shell",
]
