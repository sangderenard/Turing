"""A C++ dispatch-decision shell for backend and reduction-stage selection.

This is deliberately a **separate type** from :mod:`profiled_c_shell`.  That
shell is a two-nanosecond launch boundary and must stay that way for casual
callers; this one carries a decision table, an observation history and a graph,
and no ordinary dispatch should pay for it.

What it holds
-------------

A program is a set of *modules*.  Each module can be executed by one of several
*backends* (C, LLVM, GLSL, Fortran, ...) after being lowered to one of several
*reduction stages* -- the compartmentalisation choices produced while lowering,
including the SymPy and back-lowering routes.  The plan is therefore a cube:

    observations[module][backend][stage]

Every launch through the shell records into that cube, so the decision surface
is measured rather than assumed.

Why the layout is what it is
----------------------------

The accessors hand back exactly the arrays a graph neural network consumes, in
the shapes it already expects:

* ``features``    -- ``float32[modules, feature_dim]``  (node features)
* ``edge_index``  -- ``int32[2, edges]``                (COO, PyTorch Geometric)
* ``observations``-- ``float64[modules, backends, stages]``
* ``decisions``   -- ``int32[modules, 2]``              (backend, stage)

Python owns policy completely: it may read the cube, run whatever model it
likes, and implant a decision vector.  Nothing here chooses on its own unless
asked, and :meth:`select_best` exists only as a cheap measured baseline to
compare a learned policy against.  Python remains free to compile any reduction,
any subset, and observe the result -- exploration is the point of the cube.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


_CPP_DECLARATIONS = r"""
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

typedef struct TuringDispatchPlan TuringDispatchPlan;

TuringDispatchPlan *turing_plan_create(int, int, int, int);
void turing_plan_destroy(TuringDispatchPlan *);

int turing_plan_modules(const TuringDispatchPlan *);
int turing_plan_backends(const TuringDispatchPlan *);
int turing_plan_stages(const TuringDispatchPlan *);
int turing_plan_feature_dim(const TuringDispatchPlan *);
int turing_plan_edge_count(const TuringDispatchPlan *);

int turing_plan_set_features(TuringDispatchPlan *, int, const float *);
int turing_plan_copy_features(const TuringDispatchPlan *, float *);
int turing_plan_set_work(TuringDispatchPlan *, int, double);
int turing_plan_copy_work(const TuringDispatchPlan *, double *);

int turing_plan_set_edges(TuringDispatchPlan *, const int *, const int *, int);
int turing_plan_copy_edge_index(const TuringDispatchPlan *, int *);

int turing_plan_observe(TuringDispatchPlan *, int, int, int, double, double);
int turing_plan_copy_observations(const TuringDispatchPlan *, double *);
int turing_plan_copy_counts(const TuringDispatchPlan *, long long *);
int turing_plan_copy_device(const TuringDispatchPlan *, double *);

int turing_plan_set_decision(TuringDispatchPlan *, int, int, int);
int turing_plan_backend_for(const TuringDispatchPlan *, int);
int turing_plan_stage_for(const TuringDispatchPlan *, int);
int turing_plan_copy_decisions(const TuringDispatchPlan *, int *);

int turing_plan_set_backend_weights(TuringDispatchPlan *, const double *);
double turing_plan_score(const TuringDispatchPlan *, int, int, int);
int turing_plan_select_best(TuringDispatchPlan *, int);
int turing_plan_select_all(TuringDispatchPlan *);

int turing_plan_launch(
    TuringDispatchPlan *,
    int,
    turing_compute_closure,
    void *,
    TuringLaunchProfile *
);
"""


# The generated cffi wrapper is compiled as C; only these declarations are
# embedded in it.  The implementation lives in dispatch_shell.cpp and is linked
# alongside, because cffi's wrapper is not valid C++.
_C_GLUE = r"""
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

typedef struct TuringDispatchPlan TuringDispatchPlan;

extern TuringDispatchPlan *turing_plan_create(int, int, int, int);
extern void turing_plan_destroy(TuringDispatchPlan *);
extern int turing_plan_modules(const TuringDispatchPlan *);
extern int turing_plan_backends(const TuringDispatchPlan *);
extern int turing_plan_stages(const TuringDispatchPlan *);
extern int turing_plan_feature_dim(const TuringDispatchPlan *);
extern int turing_plan_edge_count(const TuringDispatchPlan *);
extern int turing_plan_set_features(TuringDispatchPlan *, int, const float *);
extern int turing_plan_copy_features(const TuringDispatchPlan *, float *);
extern int turing_plan_set_work(TuringDispatchPlan *, int, double);
extern int turing_plan_copy_work(const TuringDispatchPlan *, double *);
extern int turing_plan_set_edges(
    TuringDispatchPlan *, const int *, const int *, int);
extern int turing_plan_copy_edge_index(const TuringDispatchPlan *, int *);
extern int turing_plan_observe(
    TuringDispatchPlan *, int, int, int, double, double);
extern int turing_plan_copy_observations(const TuringDispatchPlan *, double *);
extern int turing_plan_copy_counts(const TuringDispatchPlan *, long long *);
extern int turing_plan_copy_device(const TuringDispatchPlan *, double *);
extern int turing_plan_set_decision(TuringDispatchPlan *, int, int, int);
extern int turing_plan_backend_for(const TuringDispatchPlan *, int);
extern int turing_plan_stage_for(const TuringDispatchPlan *, int);
extern int turing_plan_copy_decisions(const TuringDispatchPlan *, int *);
extern int turing_plan_set_backend_weights(
    TuringDispatchPlan *, const double *);
extern double turing_plan_score(const TuringDispatchPlan *, int, int, int);
extern int turing_plan_select_best(TuringDispatchPlan *, int);
extern int turing_plan_select_all(TuringDispatchPlan *);
extern int turing_plan_launch(
    TuringDispatchPlan *, int, turing_compute_closure, void *,
    TuringLaunchProfile *);
"""


_IMPLEMENTATION = Path(__file__).with_name("dispatch_shell.cpp")


@lru_cache(maxsize=1)
def _library():
    from cffi import FFI

    ffi = FFI()
    ffi.cdef(_CPP_DECLARATIONS)
    library = ffi.verify(
        _C_GLUE,
        sources=[str(_IMPLEMENTATION)],
        source_extension=".c",
    )
    return ffi, library


@dataclass
class DispatchPlan:
    """Modules × backends × reduction-stages, with measured observations.

    Python owns the policy.  This object is the shared, typed surface the policy
    reads from and writes decisions into.
    """

    modules: int
    backends: int
    stages: int
    feature_dim: int
    _ffi: Any
    _lib: Any
    _plan: Any

    # -- graph ---------------------------------------------------------
    def set_edges(self, edge_index) -> None:
        """Set the module graph from a ``[2, edges]`` COO index."""

        edges = np.ascontiguousarray(edge_index, dtype=np.int32)
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges]")
        count = edges.shape[1]
        src = self._ffi.cast("const int *", edges[0].ctypes.data)
        dst = self._ffi.cast("const int *", edges[1].ctypes.data)
        if not self._lib.turing_plan_set_edges(self._plan, src, dst, count):
            raise ValueError("edge endpoints out of range")

    @property
    def edge_index(self) -> np.ndarray:
        count = int(self._lib.turing_plan_edge_count(self._plan))
        out = np.zeros((2, count), dtype=np.int32)
        if count:
            self._lib.turing_plan_copy_edge_index(
                self._plan, self._ffi.cast("int *", out.ctypes.data)
            )
        return out

    # -- node features -------------------------------------------------
    def set_features(self, module: int, values) -> None:
        row = np.ascontiguousarray(values, dtype=np.float32)
        if row.size != self.feature_dim:
            raise ValueError(
                f"expected {self.feature_dim} features, got {row.size}"
            )
        self._lib.turing_plan_set_features(
            self._plan, module, self._ffi.cast("const float *", row.ctypes.data)
        )

    @property
    def features(self) -> np.ndarray:
        out = np.zeros((self.modules, self.feature_dim), dtype=np.float32)
        self._lib.turing_plan_copy_features(
            self._plan, self._ffi.cast("float *", out.ctypes.data)
        )
        return out

    def set_work(self, module: int, apparent_work: float) -> None:
        self._lib.turing_plan_set_work(self._plan, module, float(apparent_work))

    @property
    def work(self) -> np.ndarray:
        out = np.zeros(self.modules, dtype=np.float64)
        self._lib.turing_plan_copy_work(
            self._plan, self._ffi.cast("double *", out.ctypes.data)
        )
        return out

    # -- observations --------------------------------------------------
    def observe(
        self,
        module: int,
        backend: int,
        stage: int,
        seconds: float,
        device_seconds: float = 0.0,
    ) -> None:
        if not self._lib.turing_plan_observe(
            self._plan, module, backend, stage, float(seconds),
            float(device_seconds),
        ):
            raise IndexError("observation coordinates out of range")

    @property
    def observations(self) -> np.ndarray:
        """Mean seconds per cell; NaN where unmeasured."""

        out = np.zeros(
            (self.modules, self.backends, self.stages), dtype=np.float64
        )
        self._lib.turing_plan_copy_observations(
            self._plan, self._ffi.cast("double *", out.ctypes.data)
        )
        return out

    @property
    def device_observations(self) -> np.ndarray:
        out = np.zeros(
            (self.modules, self.backends, self.stages), dtype=np.float64
        )
        self._lib.turing_plan_copy_device(
            self._plan, self._ffi.cast("double *", out.ctypes.data)
        )
        return out

    @property
    def counts(self) -> np.ndarray:
        out = np.zeros(
            (self.modules, self.backends, self.stages), dtype=np.int64
        )
        self._lib.turing_plan_copy_counts(
            self._plan, self._ffi.cast("long long *", out.ctypes.data)
        )
        return out

    # -- decisions -----------------------------------------------------
    def set_decision(self, module: int, backend: int, stage: int) -> None:
        if not self._lib.turing_plan_set_decision(
            self._plan, module, backend, stage
        ):
            raise IndexError("decision coordinates out of range")

    def implant(self, decisions) -> None:
        """Install a whole ``[modules, 2]`` decision vector at once."""

        table = np.ascontiguousarray(decisions, dtype=np.int32)
        if table.shape != (self.modules, 2):
            raise ValueError(f"expected shape ({self.modules}, 2)")
        for module, (backend, stage) in enumerate(table):
            self.set_decision(module, int(backend), int(stage))

    @property
    def decisions(self) -> np.ndarray:
        out = np.zeros((self.modules, 2), dtype=np.int32)
        self._lib.turing_plan_copy_decisions(
            self._plan, self._ffi.cast("int *", out.ctypes.data)
        )
        return out

    # -- weighting -----------------------------------------------------
    def set_backend_weights(self, weights) -> None:
        row = np.ascontiguousarray(weights, dtype=np.float64)
        if row.size != self.backends:
            raise ValueError(f"expected {self.backends} weights")
        self._lib.turing_plan_set_backend_weights(
            self._plan, self._ffi.cast("const double *", row.ctypes.data)
        )

    def score(self, module: int, backend: int, stage: int) -> float:
        return float(
            self._lib.turing_plan_score(self._plan, module, backend, stage)
        )

    def select_best(self, module: int) -> int:
        return int(self._lib.turing_plan_select_best(self._plan, module))

    def select_all(self) -> int:
        """Greedy measured baseline; a learned policy replaces this."""

        return int(self._lib.turing_plan_select_all(self._plan))

    # -- execution -----------------------------------------------------
    def callback(self, function):
        return self._ffi.callback(
            "int(void *, unsigned long long *)", function
        )

    def launch(self, module: int, compute, context=None):
        if isinstance(compute, int):
            compute = self._ffi.cast("turing_compute_closure", compute)
        context = self._ffi.NULL if context is None else context
        profile = self._ffi.new("TuringLaunchProfile *")
        status = int(
            self._lib.turing_plan_launch(
                self._plan, module, compute, context, profile
            )
        )
        return {
            "shell_ns": int(profile.shell_ns),
            "device_ns": int(profile.device_ns),
            "status": status,
            "backend": int(profile.language),
        }

    def __del__(self):
        plan = getattr(self, "_plan", None)
        if plan is not None and self._lib is not None:
            self._lib.turing_plan_destroy(plan)
            self._plan = None


def dispatch_plan(
    *,
    modules: int,
    backends: int,
    stages: int,
    feature_dim: int = 8,
) -> DispatchPlan:
    """Allocate a dispatch plan cube."""

    ffi, lib = _library()
    handle = lib.turing_plan_create(modules, backends, stages, feature_dim)
    if handle == ffi.NULL:
        raise ValueError("invalid dispatch plan dimensions")
    return DispatchPlan(
        modules=modules,
        backends=backends,
        stages=stages,
        feature_dim=feature_dim,
        _ffi=ffi,
        _lib=lib,
        _plan=handle,
    )


__all__ = ["DispatchPlan", "dispatch_plan"]
