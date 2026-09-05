"""Phase 1 of docs/PLAN_SPECTRAL_GREENS_FUNCTION_TIRE_MODEL.md: measure one
circumferential mode's impulse response on the real tire, using the real,
already-compiled native law kernels -- no new SSA lowering here at all.

Method (standard "impact hammer" experimental modal analysis, adapted to a
spatial mode instead of a single point): settle the tire to a real
equilibrium, give the tread ring a velocity impulse shaped as
cos(m*theta) in the radial direction, then record the FREE response (no
further forcing) at every ring.  A velocity impulse at t=0 is flat in
frequency, so the driven ring's own displacement spectrum IS (up to a
known scale) that ring's point transfer function H_m(omega) at mode m --
this is the measurement the plan's Phase 1/2 need, not yet the full
cross-ring G(ring', ring, m, omega) (that is a straightforward extension
once this one is validated).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.common.tensors import AbstractTensor  # noqa: E402
from src.compiler import vehicle_balloon_tire_program as program_module  # noqa: E402
from src.compiler.vehicle_balloon_tire_program import (  # noqa: E402
    balloon_tire_python_program, BALLOON_TIRE_VECTOR_SOURCE,
)
from src.compiler.vehicle_python_compilation import (  # noqa: E402
    vehicle_python_runtime_bindings,
)
from src.compiler.vehicle_balloon_tire_stability import (  # noqa: E402
    estimate_tire_critical_dt,
)


def build_tire():
    program = balloon_tire_python_program(("w",))
    namespace = {"AbstractTensor": AbstractTensor, "np": np}
    for name in ("vector_cross", "vector_norm", "MAX_PLANES_PER_WHEEL"):
        if hasattr(program_module, name):
            namespace[name] = getattr(program_module, name)
    namespace.update(vehicle_python_runtime_bindings(include_configured_vehicle=False))
    exec(BALLOON_TIRE_VECTOR_SOURCE, namespace)
    return program, namespace["balloon_tire_vector_step"], namespace["balloon_tire_vector_initialize"]


def tensor(array):
    return AbstractTensor.tensor(np.asarray(array, dtype=np.float64))


def main() -> int:
    mode = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    amplitude = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1  # m/s
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 4000

    program, step, initialize = build_tire()
    C = program.constants
    index = {name: i for i, name in enumerate(program.input_names)}
    default = np.asarray(C["default_input"], dtype=np.float64).copy()
    default[index["gravity_y"]] = 0.0
    rest = np.asarray(C["rest"], dtype=np.float64)
    vertex_count = rest.shape[0]

    bead_mask = np.asarray(C["bead_mask"], dtype=bool)
    bead_rows = np.nonzero(bead_mask)[0]
    rows = (int(bead_rows[1] - bead_rows[0]) if bead_rows[1] - bead_rows[0] > 1
            else int(np.nonzero(bead_mask[1:])[0][0] + 1))
    circumferential_segments = vertex_count // rows
    if circumferential_segments * rows != vertex_count:
        raise RuntimeError(f"vertex layout not iu*rows: {vertex_count} / {rows}")

    def vertex(iu: int, iv: int) -> int:
        return (iu % circumferential_segments) * rows + iv

    tread_row = rows // 2  # the crown, |q| smallest
    theta = 2.0 * np.pi * np.arange(circumferential_segments) / circumferential_segments
    tread_indices = np.array([vertex(iu, tread_row) for iu in range(circumferential_segments)])

    stability = estimate_tire_critical_dt(program)
    dt = 0.25 * float(stability["dt_critical_s"])  # comfortably under the limit
    print(f"mode={mode} amplitude={amplitude} m/s steps={steps} dt={dt:.3e} s "
          f"(critical={stability['dt_critical_s']:.3e} s) rows={rows} "
          f"circumferential_segments={circumferential_segments} tread_row={tread_row}",
          flush=True)

    consts = dict(
        wheel_input_indices=AbstractTensor.tensor(np.asarray(C["wheel_input_indices"], dtype=np.int64)),
        rest=tensor(rest), face_vertices=AbstractTensor.tensor(np.asarray(C["face_vertices"], dtype=np.int64)),
        face_rest=tensor(C["face_rest"]), face_scatter=tensor(C["face_scatter"]),
        bending_incidence=tensor(C["bending_incidence"]), bending_scatter=tensor(C["bending_scatter"]),
        bending_weight=tensor(C["bending_weight"]), vertex_area=tensor(C["vertex_area"]),
        bead_mask=AbstractTensor.tensor(bead_mask),
        face_material=tensor(C["face_material"]),
    )
    default[index["dt"]] = dt
    inputs = tensor(default.reshape(1, -1))
    state = initialize(inputs, tensor(np.zeros((1, 1, vertex_count, 6))),
                       consts["wheel_input_indices"], consts["rest"])
    output = tensor(np.zeros((1, 1, len(program.output_names))))

    # Settle at rest (no gravity, no contact) before measuring -- confirmed
    # dead-still behavior for this same construction earlier in the
    # session (tire_alone.py), so 200 steps is ample margin, not a guess.
    for _ in range(200):
        state, output = step(inputs, state, output, consts["wheel_input_indices"], consts["rest"],
                             consts["face_vertices"], consts["face_rest"], consts["face_scatter"],
                             consts["bending_incidence"], consts["bending_scatter"], consts["bending_weight"],
                             consts["vertex_area"], consts["bead_mask"], consts["face_material"])
    settled = np.asarray(state.data).copy()
    max_speed_at_rest = float(np.max(np.linalg.norm(settled[0, 0, :, 3:6], axis=-1)))
    print(f"settled: max speed = {max_speed_at_rest:.3e} m/s (should be ~0)", flush=True)

    # The velocity impulse: radial direction at each tread-ring vertex,
    # shaped cos(m*theta).  Radial direction from the tire's own rest
    # geometry (r-hat in the x/y plane), not assumed.
    radial_direction = rest[tread_indices, :2] / np.maximum(
        np.linalg.norm(rest[tread_indices, :2], axis=-1, keepdims=True), 1e-12)
    perturbed = settled.copy()
    kick = amplitude * np.cos(mode * theta)
    perturbed[0, 0, tread_indices, 3] += kick * radial_direction[:, 0]
    perturbed[0, 0, tread_indices, 4] += kick * radial_direction[:, 1]
    state = AbstractTensor.tensor(perturbed)

    # Record radial displacement of the driven ring at every step.
    recorded = np.zeros((steps, circumferential_segments), dtype=np.float64)
    rest_tread = rest[tread_indices]
    rest_radius = np.linalg.norm(rest_tread[:, :2], axis=-1)
    t0 = time.time()
    for k in range(steps):
        position = np.asarray(state.data)[0, 0, tread_indices, :2]
        recorded[k] = np.linalg.norm(position, axis=-1) - rest_radius
        state, output = step(inputs, state, output, consts["wheel_input_indices"], consts["rest"],
                             consts["face_vertices"], consts["face_rest"], consts["face_scatter"],
                             consts["bending_incidence"], consts["bending_scatter"], consts["bending_weight"],
                             consts["vertex_area"], consts["bead_mask"], consts["face_material"])
    print(f"{steps} steps in {time.time()-t0:.1f}s", flush=True)

    # Spatial FFT at every timestep -> the mode-m component's time series.
    spatial = np.fft.rfft(recorded, axis=1) / circumferential_segments
    mode_series = spatial[:, mode] if mode < spatial.shape[1] else None
    if mode_series is None:
        print("mode index out of range for this circumferential resolution", flush=True)
        return 1
    leaked = np.delete(spatial, mode, axis=1)
    leak_fraction = float(np.max(np.abs(leaked)) / max(np.max(np.abs(mode_series)), 1e-30))
    print(f"peak |mode {mode}| amplitude = {np.max(np.abs(mode_series)):.4e} m, "
          f"worst other-mode leakage fraction = {leak_fraction:.3e} "
          f"({'diagonal holds' if leak_fraction < 0.05 else 'REAL cross-mode coupling'})",
          flush=True)

    # Temporal FFT of the mode's own time series -> H_m(omega), up to the
    # known flat-spectrum scale of a velocity impulse.
    spectrum = np.fft.rfft(mode_series.real)
    frequencies = np.fft.rfftfreq(steps, d=dt)
    peak_index = int(np.argmax(np.abs(spectrum[1:]))) + 1  # skip DC
    print(f"H_{mode}(omega) peak at f = {frequencies[peak_index]:.2f} Hz "
          f"(omega = {2*np.pi*frequencies[peak_index]:.1f} rad/s), "
          f"|H| there = {np.abs(spectrum[peak_index]):.4e}", flush=True)

    out_path = Path(__file__).resolve().parents[1] / "build" / f"greens_mode{mode}_impulse.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, mode=mode, dt=dt, theta=theta, recorded=recorded,
             mode_series=mode_series, spectrum=spectrum, frequencies=frequencies)
    print(f"saved: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
