"""Critical explicit step of the balloon tyre, measured on its own force update.

The tyre is stepped explicitly (the bead is the one implicit part).  An
explicit update is stable only while the step stays under ``2 / omega_max``
for the stiffest mode of the skin, and explicit damping lowers that limit
further; past it the integrator's own difference equation grows a spatial
pattern that no material damping can bleed, because the damping is evaluated
one step late inside the same update.  Measured on the 16 x 8 skin: the
undamped estimate is 187 us, the skin is stable at 60 us and blows up at
100 us, so the usable fraction of the estimate is about 0.3.  On the 32 x 24
dually skin the estimate is 28.7 us; the run was stepping at 36 us.

``estimate_tire_critical_dt`` finds ``omega_max`` by power iteration on the
tyre program's acceleration at its rest state: perturb by a unit vector,
take one step with a negligible dt, read the acceleration difference, which
is ``-(K / M) u``.  Thirty iterations converge to better than a percent.
The enclosing vehicle recurrence then divides each outer step into
``ceil(dt / (fraction * dt_critical))`` tyre microsteps.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


def estimate_tire_critical_dt(
    program: Any,
    *,
    iterations: int = 30,
    probe_dt: float = 1.0e-9,
    perturbation: float = 1.0e-7,
) -> dict[str, float]:
    """Power-iterate the tyre program's stiffest mode at rest.

    Returns ``omega_max_rad_s``, ``dt_critical_s`` (= 2 / omega_max, the
    undamped explicit limit) and the fraction of the mode's energy on the
    tread, sidewall and bead rows.  Uses the eager AbstractTensor stages of
    the laws (never the native stand-ins) so it costs no kernel lowering.
    """

    from src.common.tensors import AbstractTensor
    from . import vehicle_balloon_tire_program as program_module
    from .vehicle_balloon_tire_program import BALLOON_TIRE_VECTOR_SOURCE
    from .vehicle_balloon_tire import balloon_tire_python_bindings
    from .vehicle_python_compilation import (
        _abstract_tensor_stage_callable, symbolic_law_compilations)

    namespace: dict[str, Any] = {"AbstractTensor": AbstractTensor, "np": np}
    for name in ("vector_cross", "vector_norm", "MAX_PLANES_PER_WHEEL"):
        if hasattr(program_module, name):
            namespace[name] = getattr(program_module, name)
    namespace.update(balloon_tire_python_bindings())
    for name, compilation in symbolic_law_compilations(False).items():
        namespace[name] = _abstract_tensor_stage_callable(compilation, name)
    exec(BALLOON_TIRE_VECTOR_SOURCE, namespace)
    step = namespace["balloon_tire_vector_step"]
    initialize = namespace["balloon_tire_vector_initialize"]

    constants: Mapping[str, Any] = program.constants
    index = {name: i for i, name in enumerate(program.input_names)}
    default = np.asarray(constants["default_input"], dtype=np.float64).copy()
    default[index["gravity_y"]] = 0.0
    default[index["dt"]] = probe_dt

    def tensor(value: Any) -> Any:
        return AbstractTensor.tensor(np.asarray(value, dtype=np.float64))

    rest = np.asarray(constants["rest"], dtype=np.float64)
    vertex_count = rest.shape[0]
    inputs = tensor(default.reshape(1, -1))
    wheel_input_indices = AbstractTensor.tensor(
        np.asarray(constants["wheel_input_indices"], dtype=np.int64))
    bead = np.asarray(constants["bead_mask"], dtype=bool)
    arguments = (
        wheel_input_indices, tensor(rest),
        AbstractTensor.tensor(np.asarray(constants["face_vertices"], dtype=np.int64)),
        tensor(constants["face_rest"]), tensor(constants["face_scatter"]),
        tensor(constants["bending_incidence"]), tensor(constants["bending_scatter"]),
        tensor(constants["bending_weight"]), tensor(constants["vertex_area"]),
        AbstractTensor.tensor(bead), tensor(constants["face_material"]),
    )
    wheel_count = int(np.asarray(constants["wheel_input_indices"]).shape[0])
    state0 = np.asarray(initialize(
        inputs, tensor(np.zeros((1, wheel_count, vertex_count, 6))),
        wheel_input_indices, arguments[1]).data).copy()
    output = tensor(np.zeros((1, wheel_count, len(program.output_names))))

    def acceleration(positions: np.ndarray) -> np.ndarray:
        state = state0.copy()
        state[0, 0, :, 0:3] = positions
        state[0, :, :, 3:6] = 0.0
        advanced, _ = step(inputs, tensor(state), output, *arguments)
        return np.asarray(advanced.data)[0, 0, :, 3:6] / probe_dt

    x0 = state0[0, 0, :, 0:3].copy()
    a0 = acceleration(x0)
    rng = np.random.default_rng(0)
    u = rng.normal(size=(vertex_count, 3))
    u[bead] = 0.0
    u /= np.linalg.norm(u)
    rayleigh = 0.0
    for _ in range(int(iterations)):
        ku = -(acceleration(x0 + perturbation * u) - a0) / perturbation
        ku[bead] = 0.0
        rayleigh = float(np.dot(u.reshape(-1), ku.reshape(-1)))
        norm = float(np.linalg.norm(ku))
        if norm <= 0.0:
            break
        u = ku / norm
    omega = math.sqrt(max(rayleigh, 0.0))
    bead_rows = np.nonzero(bead)[0]
    rows = (int(bead_rows[1] - bead_rows[0]) if bead_rows[1] - bead_rows[0] > 1
            else int(np.nonzero(bead[1:])[0][0] + 1))
    q = -1.0 + 2.0 * (np.arange(vertex_count) % rows) / max(rows - 1, 1)
    weight = np.linalg.norm(u, axis=1) ** 2
    tread = float(weight[(~bead) & (np.abs(q) <= 0.45)].sum())
    sidewall = float(weight[(~bead) & (np.abs(q) > 0.45)].sum())
    return {
        "omega_max_rad_s": omega,
        "dt_critical_s": (2.0 / omega) if omega > 0.0 else math.inf,
        "mode_tread_fraction": tread,
        "mode_sidewall_fraction": sidewall,
        "iterations": float(iterations),
    }


def tire_microstep_count(outer_dt: float, dt_critical: float, fraction: float) -> int:
    """Microsteps that keep the tyre's step at or under ``fraction * dt_critical``."""

    safe = float(fraction) * float(dt_critical)
    if not (safe > 0.0) or not math.isfinite(safe):
        return 1
    return max(1, int(math.ceil(float(outer_dt) / safe - 1.0e-12)))


__all__ = ["estimate_tire_critical_dt", "tire_microstep_count"]
