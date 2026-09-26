"""Join the chamber laws to the tensor dt system (``run_superstep``).

SymPy is the truth.  What runs here is the compiler's own AbstractTensor
materialization of each law -- stage 2 of tools/compile_symbolic_source.py,
the very program the native lane is lowered from -- so the dt system steps
the same program in tensor form, on whatever backend the tensor core
selects, with one column per voxel / particle / surface / pool.

    law = CompiledLaw.from_module(symbolic_chamber_solvers, "voxel_species_step")
    engine = LawEngine(law, state={"m_v": ..., "m_l": ..., ...}, params={...})
    total, dt_next, metrics = run_superstep(
        engine.state, round_max, dt_init, dx, targets, ctrl, engine.advance)

``advance(state, dt)`` runs the law on the state's columns plus the fixed
parameters, feeds every ``x_next`` output back into ``x``, and lifts the
law's published diagnostics into ``Metrics`` by name -- ``max_vel``,
``max_flux``, ``div_inf``, ``mass_err``, ``dt_limit`` and the ``energy_j`` /
``power_w`` channels -- reducing over the columns (max for rates and
errors, min for the stability limit).  Any parameter may itself be a column.
"""

from __future__ import annotations

from src.common.dt_system.error_channels import DT_CHANNEL_NAMES

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from src.common.dt_system.dt_scaler import Metrics
from src.common.dt_system.engine_api import DtCompatibleEngine
from src.common.tensors import AbstractTensor
from src.compiler.symbolic_equation_compiler import compile_sympy_equations
from src.compiler.vehicle_python_compilation import _abstract_tensor_stage_callable

METRIC_FIELDS = ("max_vel", "max_flux", "div_inf", "mass_err")
CHANNEL_FIELDS = ("energy_j", "power_w")


@dataclass
class CompiledLaw:
    name: str
    argument_names: tuple[str, ...]
    output_names: tuple[str, ...]
    stage: Callable[..., tuple]
    compilation: Any

    @classmethod
    def from_equations(cls, name, equations, *, publications=(), dtype="float64"):
        compilation = compile_sympy_equations(
            equations, name=name, publications=publications, dtype=dtype)
        meta = compilation.function.metadata
        return cls(
            name, tuple(meta["argument_names"]), tuple(meta["output_names"]),
            _abstract_tensor_stage_callable(compilation, name), compilation)

    @classmethod
    def from_module(cls, module, name):
        return cls.from_equations(
            name, module.LAWS[name],
            publications=getattr(module, "LAW_PUBLICATIONS", {}).get(name, ()),
            dtype=getattr(module, "DTYPE", "float64"))

    @classmethod
    def from_piece(cls, piece):
        """Use an existing LLVMPiece as this engine's compiled law."""
        import numpy as np

        def stage(*columns):
            arrays = tuple(
                np.asarray(
                    column.tolist() if isinstance(column, AbstractTensor) else column,
                    dtype=np.float64,
                )
                for column in columns
            )
            return tuple(
                AbstractTensor.tensor(np.asarray(value, dtype=np.float64).tolist())
                for value in piece(*arrays)
            )

        return cls(
            str(piece.entry), tuple(piece.argument_names),
            tuple(piece.output_names), stage, piece,
        )

    def __call__(self, **columns) -> dict[str, Any]:
        missing = [name for name in self.argument_names if name not in columns]
        if missing:
            raise KeyError(f"{self.name}: missing inputs {missing}")
        results = self.stage(*(columns[name] for name in self.argument_names))
        return dict(zip(self.output_names, results))


def _as_tensor(value):
    return value if isinstance(value, AbstractTensor) else AbstractTensor.tensor(value)


def _reduce(value, how: str) -> float:
    if isinstance(value, AbstractTensor):
        value = value.max() if how == "max" else value.min()
        return float(value.item())
    return float(value)


#: Bump when the payload shape below changes.
_LAW_MODULE_CACHE_SCHEMA = 1


def load_law_module_cached(path, *, name: str | None = None, cache_dir=None, force: bool = False):
    """Import a law module, or restore its constructed ``LAWS`` from disk.

    Executing ``symbolic_chamber_solvers.py`` builds every law's SymPy
    expression trees at import, and that construction -- not compilation,
    which the dual-IR cache already covers, and not stepping -- is the
    ~7 minutes every run of the raincloud demo was paying (measured: 435 s
    for ``module constructed`` before a single step).  The trees are a
    pure function of the source text, so they are cached as a pickle keyed
    on the source digest (plus the SymPy and Python versions, since a
    pickle from another interpreter is not one to trust).  A hit is
    seconds.  Any edit to the law file misses once and re-caches.

    What comes back is either the real module (miss) or a namespace
    carrying exactly what a caller needs to compile and run the laws --
    ``LAWS``, ``LAW_PUBLICATIONS``, ``DTYPE``, ``SCHEDULE``, ``BATCH`` --
    which is all ``CompiledLaw.from_module`` reads.  The helper FUNCTIONS
    of the law module (``negotiate``, ``relax``, ...) are not in the cache;
    a caller that wants those must import the module itself.

    Flags: ``TURING_DISABLE_LAW_MODULE_CACHE=1`` always executes the
    module; ``TURING_LAW_MODULE_CACHE_DIR`` relocates the cache (default:
    ``__lawcache__`` beside the law file).  ``force=True`` re-constructs
    and overwrites.
    """
    import hashlib
    import importlib.util
    import os
    import pickle
    import sys
    import time
    import types
    from pathlib import Path

    import sympy

    path = Path(path).resolve()
    name = name or path.stem
    source = path.read_bytes()
    key = hashlib.sha256(b"\0".join([
        str(_LAW_MODULE_CACHE_SCHEMA).encode(), source,
        sympy.__version__.encode(), sys.version.encode(),
    ])).hexdigest()
    disabled = os.environ.get("TURING_DISABLE_LAW_MODULE_CACHE", "").casefold() in {"1", "true", "yes", "on"}
    root = Path(cache_dir or os.environ.get("TURING_LAW_MODULE_CACHE_DIR") or (path.parent / "__lawcache__"))
    cache_file = root / f"{name}-{key[:16]}.pkl"

    if not force and not disabled and cache_file.exists():
        t0 = time.time()
        # Unpickling a SymPy tree rebuilds it through the constructors, and
        # Min/Max/Add canonicalise on construction -- which IS the expensive
        # part of building the laws, so a plain load re-paid most of the
        # 435 s (measured: >141 CPU-s and still going).  Inside evaluate(False)
        # the constructors skip canonicalisation and the same load is 8.8 s.
        # The tree is the same tree; the lowering walks args and never relied
        # on canonical order.
        with open(cache_file, "rb") as fh, sympy.evaluate(False):
            payload = pickle.load(fh)
        ns = types.SimpleNamespace(**payload)
        ns.__law_cache__ = str(cache_file)
        ns.__construct_seconds__ = time.time() - t0
        print(f"laws: cache hit ({ns.__construct_seconds__:.1f} s) {cache_file.name}", flush=True)
        return ns

    t0 = time.time()
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    mod.__construct_seconds__ = time.time() - t0
    mod.__law_cache__ = None
    print(f"laws: constructed ({mod.__construct_seconds__:.1f} s)"
          + ("" if disabled else f", caching to {cache_file.name}"), flush=True)
    if not disabled:
        payload = {
            "LAWS": mod.LAWS,
            "LAW_PUBLICATIONS": getattr(mod, "LAW_PUBLICATIONS", {}),
            "DTYPE": getattr(mod, "DTYPE", "float64"),
            "SCHEDULE": getattr(mod, "SCHEDULE", "asap"),
            "BATCH": getattr(mod, "BATCH", 1),
        }
        root.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_file)          # atomic: a concurrent reader never sees a half file
        mod.__law_cache__ = str(cache_file)
    return mod


class LawState:
    """The columns one law advances; ``run_superstep`` reads ``dt_limit_hint``."""

    def __init__(self, columns: Mapping[str, Any]):
        self.columns = dict(columns)
        self.outputs: dict[str, Any] = {}
        self.dt_limit_hint: float | None = None

    # advance() rebinds columns rather than mutating them, so a shallow
    # snapshot is a complete one for the controller's rollback.
    def copy_shallow(self) -> "LawState":
        snapshot = LawState(self.columns)
        snapshot.outputs = dict(self.outputs)
        snapshot.dt_limit_hint = self.dt_limit_hint
        return snapshot

    def restore(self, snapshot: "LawState") -> None:
        self.columns = dict(snapshot.columns)
        self.outputs = dict(snapshot.outputs)
        self.dt_limit_hint = snapshot.dt_limit_hint


class LawEngine:
    def __init__(self, law: CompiledLaw, state: Mapping[str, Any], params: Mapping[str, Any],
                 recurrence: Mapping[str, str] | None = None):
        self.law = law
        self.state = LawState(state)
        # The materialized program dispatches on its operands; a Min/Max of
        # plain floats has nothing to dispatch on, so every input is a tensor.
        self.params = {name: _as_tensor(value) for name, value in params.items()}
        self.recurrence = dict(recurrence) if recurrence is not None else {
            name: name[:-len("_next")] for name in law.output_names if name.endswith("_next")}
        # Inputs another law supplies each tick (face flows, neighbour
        # columns, a coupled temperature); set by the package before advance.
        self.feeds: dict[str, Any] = {}
        self.frame = 0

    def advance(self, state: LawState, dt) -> tuple[bool, Metrics]:
        out = self.law(**self.params, **self.feeds, **state.columns, dt=_as_tensor(dt))
        finite = all(math.isfinite(_reduce(value, "max")) and math.isfinite(_reduce(value, "min"))
                     for value in out.values())
        if finite:
            for out_name, state_name in self.recurrence.items():
                state.columns[state_name] = out[out_name]
        state.outputs = out
        self.frame += 1
        metrics = Metrics(
            **{name: _reduce(out.get(name, 0.0), "max") for name in METRIC_FIELDS},
            sim_frame=self.frame,
            dt_limit=_reduce(out["dt_limit"], "min") if "dt_limit" in out else None,
            error_channels=AbstractTensor.tensor([
                _reduce(out[name], "max") if name in out and name in CHANNEL_FIELDS else 0.0
                for name in DT_CHANNEL_NAMES]),
            error_present=AbstractTensor.tensor([
                float(name in out and name in CHANNEL_FIELDS) for name in DT_CHANNEL_NAMES]),
            hard_failure=not finite,
        )
        state.dt_limit_hint = metrics.dt_limit
        return finite, metrics


def validate_law(module, name: str, inputs: Mapping[str, float], *, rtol: float = 1e-9) -> dict[str, tuple[float, float, float]]:
    """Compare the compiled program against the SymPy truth on one input row.

    The truth is ``sympy.lambdify`` of the authored equations (CSE'd, plain
    math); the product is the same law through ``CompiledLaw``.  Returns
    ``{output: (truth, compiled, relative_error)}`` and raises on any output
    outside ``rtol`` or non-finite on either side, so a lowering defect shows
    up as a named output, not as a wrong simulation.
    """
    import math
    import sympy as sp

    equations = module.LAWS[name]
    symbols = sorted(set().union(*[eq.rhs.free_symbols for eq in equations]), key=str)
    truth_fn = sp.lambdify(symbols, [eq.rhs for eq in equations], modules="math", cse=True)
    truth = truth_fn(*[float(inputs[str(s)]) for s in symbols])
    law = CompiledLaw.from_module(module, name)
    product = law(**{k: AbstractTensor.tensor(float(inputs[k])) for k in law.argument_names})
    report: dict[str, tuple[float, float, float]] = {}
    bad = []
    for eq, t in zip(equations, truth):
        out = str(eq.lhs)
        c = _reduce(product[out], "max")
        t = float(t)
        err = abs(c - t) / (abs(t) + 1e-300)
        report[out] = (t, c, err)
        if not (math.isfinite(t) and math.isfinite(c)) or err > rtol:
            bad.append(f"{out}: truth={t!r} compiled={c!r} rel_err={err:.3g}")
    if bad:
        raise AssertionError(f"{name}: compiled program disagrees with SymPy truth\n  " + "\n  ".join(bad))
    return report


class LawDtEngine(DtCompatibleEngine):
    """One law as a ``DtCompatibleEngine``, so ``dt_graph`` can schedule it
    beside the fluid engines in ``dt_system.fluid_mechanics``.

    ``step`` returns ``(ok, metrics, state)`` like ``VoxelFluidEngine``; the
    state is the law's column dict, which another engine may read (a pool's
    ``overflow_rate`` into a fluid engine's inflow, a fluid engine's free
    surface into ``phi_in``).  ``preferred_dt`` is the law's own published
    stability limit from the last step.
    """

    def __init__(self, law: CompiledLaw, state: Mapping[str, Any], params: Mapping[str, Any],
                 recurrence: Mapping[str, str] | None = None, name: str | None = None):
        self.engine = LawEngine(law, state, params, recurrence)
        self.name = name or law.name
        self._last_metrics: Metrics | None = None

    def get_state(self, state=None):
        out = state if isinstance(state, dict) else {}
        out.update(self.engine.state.columns)
        return out

    def snapshot(self):
        return self.engine.state.copy_shallow()

    def restore(self, snap) -> None:
        if snap is not None:
            self.engine.state.restore(snap)

    def step(self, dt: float, state=None, state_table=None):
        if isinstance(state, dict):
            for name in self.engine.state.columns:
                if name in state:
                    self.engine.state.columns[name] = _as_tensor(state[name])
        ok, metrics = self.engine.advance(self.engine.state, dt)
        self._last_metrics = metrics
        return ok, metrics, self.get_state()

    def step_with_state(self, state: object, dt: float, *, realtime: bool = False):
        return self.step(float(dt), state=state)

    def preferred_dt(self) -> float | None:
        return self.engine.state.dt_limit_hint

    def get_metrics(self) -> Metrics | None:
        return self._last_metrics
