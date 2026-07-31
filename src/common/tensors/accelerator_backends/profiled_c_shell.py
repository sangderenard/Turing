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
} TuringLaunchProfile;

int turing_profiled_launch(
    turing_compute_closure compute,
    void *context,
    TuringLaunchProfile *profile
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
} TuringLaunchProfile;

int turing_profiled_launch(
    turing_compute_closure compute,
    void *context,
    TuringLaunchProfile *profile
) {
    uint64_t started;
    uint64_t finished;
    uint64_t device_ns = 0;
    int status;

    if (compute == 0 || profile == 0) {
        return 0;
    }
    started = turing_monotonic_ns();
    status = compute(context, &device_ns);
    finished = turing_monotonic_ns();
    profile->shell_ns = finished - started;
    profile->device_ns = device_ns;
    profile->status = status;
    return status;
}
"""


@dataclass(frozen=True)
class CLaunchProfile:
    shell_ns: int
    device_ns: int
    status: int

    @property
    def shell_ms(self) -> float:
        return self.shell_ns / 1.0e6

    @property
    def device_ms(self) -> float:
        return self.device_ns / 1.0e6


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

    def launch(self, compute: Any, context: Any = None) -> CLaunchProfile:
        if isinstance(compute, int):
            compute = self.ffi.cast("turing_compute_closure", compute)
        if context is None:
            context = self.ffi.NULL
        elif isinstance(context, int):
            context = self.ffi.cast("void *", context)
        profile = self.ffi.new("TuringLaunchProfile *")
        status = int(
            self.library.turing_profiled_launch(compute, context, profile)
        )
        return CLaunchProfile(
            shell_ns=int(profile.shell_ns),
            device_ns=int(profile.device_ns),
            status=status,
        )

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
    "ProfiledCShell",
    "profiled_c_shell",
]
