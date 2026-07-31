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
    if (logger != 0) {
        logger(logger_user, profile);
    }
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
                self.ffi.NULL,
                tag,
            )
        )
        return CLaunchProfile(
            shell_ns=int(profile.shell_ns),
            device_ns=int(profile.device_ns),
            status=status,
            language=ShellLanguage(int(profile.language)),
        )

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


@lru_cache(maxsize=1)
def profiled_c_shell() -> ProfiledCShell:
    from cffi import FFI

    ffi = FFI()
    ffi.cdef(_C_DECLARATIONS)
    library = ffi.verify(_C_SOURCE)
    return ProfiledCShell(ffi=ffi, library=library)


__all__ = [
    "CLaunchProfile",
    "LaunchStatistics",
    "ProfiledCShell",
    "ShellLanguage",
    "profiled_c_shell",
]
