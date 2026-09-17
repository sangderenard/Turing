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
            error_channels={name: _reduce(out[name], "max") for name in CHANNEL_FIELDS if name in out},
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
