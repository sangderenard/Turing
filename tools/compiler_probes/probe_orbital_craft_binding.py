"""The orbital benchmark and the craft are one system.

The whole orbital transfer set (``orbital_transfer_raw``: the equation of
motion, the total energy, the force-cost integral and both boundary
conditions, as written) is compiled once through the sanctioned lane
(``compile_sympy_equations`` -> ``piece_from_law`` -> LLVM, and C).  Its
externals are then bound, at load, to a LIVE ``OrbitalJumper``
(``engine_toy/orbital_jumper.py``) through the craft's own seam:

    r_i(s)  the craft's position  at recorded instant s   (``r()[0]``)
    v_i(s)  its velocity          (dr/ds, ``r()[1]``)
    a_i(s)  its acceleration      (dv/ds: the craft's compiled N4.1 piece
                                   at that position plus the applied force,
                                   over the mass)
    F_i(s)  the applied force per unit mass the craft integrates (set
            through ``F()``, read back with ``applied_force()``)

Arc length is time (design decision 1).  The host functions are the
craft's trajectory on its recorded instants and refuse any other ``s``
(no interpolation).  Evaluated against the craft state:

  * equation of motion residual  a - (F_grav1 + F_grav2 + F): the set's
    gravity against the craft's catalogue N4.1 (decision 5) -- zero to
    rounding;
  * total energy at each instant vs the same expression in numpy; its drift
    over the run is the integrator's (reported, not asserted);
  * force cost  m * Integral(|F|, (s, 0, L))  vs the craft's own
    ``fuel_impulse`` (the craft imports the same integrand);
  * r(0) - r_start and r(L) - r_end against the recorded endpoints: zero.

Both native lanes are checked.  Exit 1 on any disagreement.

    python -u tools/compiler_probes/probe_orbital_craft_binding.py
"""
from __future__ import annotations

import faulthandler
import math
import pathlib
import sys
import warnings

import numpy as np

faulthandler.dump_traceback_later(900, repeat=True, file=sys.stderr)
REPO = pathlib.Path(__file__).resolve().parents[2]
ENGINE_TOY = REPO.parent / "engine_toy"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "compiler_probes"))
sys.path.insert(0, str(ENGINE_TOY))

import probe_orbital_transfer as orbital  # noqa: E402

BATCH = 4
LAW = "orbital_transfer_raw"
BUILD = REPO / "build" / "orbital_craft_binding"
MU_EARTH = 3.986004418e14
MU_MOON = 4.9048695e12
MOON_M = (3.844e8, 0.0, 0.0)
R_ORBIT = 7.0e6
MASS_KG = 1000.0
RAW_FORCE_N = (0.0, 0.0, 2.0)
TOLERANCE = 1e-12


class CraftTrajectory:
    """The live craft's state at each instant it has reached."""

    def __init__(self, jumper):
        self.jumper = jumper
        self.times: list[float] = []
        self.rows: dict[str, list] = {"r": [], "v": [], "a": [], "F": []}
        self.record()

    def record(self):
        jumper = self.jumper
        position, velocity = jumper.r()
        mass = float(np.asarray(jumper.mass_kg).reshape(-1)[0])
        gravity_piece = jumper.pieces[jumper.piece_labels.index("N4.1 gravity")]
        columns = {name: jumper._span(name) for name in gravity_piece.argument_names
                   if not name.startswith("applied_force_")}
        gravity = np.asarray(jumper._gravity_force(columns))[0]
        # the applied force the craft integrates: what the seam set, read
        # back once a substep has run
        applied = (np.asarray(RAW_FORCE_N, dtype=float) if not self.times
                   else np.asarray(jumper.applied_force(), dtype=float).reshape(3))
        self.times.append(float(jumper.time_s))
        self.rows["r"].append(np.asarray(position, dtype=float).reshape(3))
        self.rows["v"].append(np.asarray(velocity, dtype=float).reshape(3))
        self.rows["a"].append((gravity + applied) / mass)
        self.rows["F"].append(applied / mass)

    def lookup(self, kind: str, axis: int):
        times = np.asarray(self.times)

        def at(s):
            s = np.asarray(s, dtype=float)
            out = np.empty(s.shape)
            for index, value in np.ndenumerate(s):
                hit = np.flatnonzero(np.abs(times - value) <= 1e-9 * max(1.0, abs(value)))
                if hit.size != 1:
                    if kind == "F" and np.allclose(self.rows["F"], self.rows["F"][0],
                                                   rtol=0, atol=0):
                        # the commanded force is constant over the run: the
                        # craft's F(s) at a quadrature node inside [0, L]
                        out[index] = self.rows["F"][0][axis]
                        continue
                    raise ValueError(
                        f"craft {kind}{axis + 1}({value}): not a recorded instant "
                        f"{self.times}")
                out[index] = self.rows[kind][int(hit[0])][axis]
            return out
        return at

    def externals(self) -> dict:
        return {f"{kind}{axis + 1}": self.lookup(kind, axis)
                for kind in ("r", "v", "a", "F") for axis in range(3)}


def main() -> int:
    from orbital_jumper import GravityCenter, OrbitalJumper

    from src.compiler.external_functions import bind_external_slots
    from src.compiler.native_package import piece_from_law
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    from src.compiler.symbolic_equation_compiler import (
        compile_sympy_equations, declared_symbolic_outputs,
    )

    sys.stdout.reconfigure(errors="backslashreplace")
    program = orbital.build_program()
    laws = orbital.benchmark_laws(program)
    # the whole set: its three Equalities as written (equation of motion,
    # initial and terminal condition) and its two expressions, named
    import sympy as sp

    equations = [
        *laws["orbital_transfer_raw"],
        sp.Eq(sp.Symbol("total_energy"), program["total_energy_expression"], evaluate=False),
        sp.Eq(sp.Symbol("force_cost"), program["force_cost_integral"], evaluate=False),
    ]
    kinds = ("motion", "initial", "terminal", "energy", "cost")
    compilation = compile_sympy_equations(
        equations, name=LAW, external_derivatives=orbital.external_derivatives())
    arguments = tuple(compilation.function.metadata["argument_names"])
    rows = declared_symbolic_outputs(compilation.equations, LAW)
    print("outputs:", [(row.name, row.equation_index, row.component) for row in rows])
    directory = BUILD / LAW
    directory.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        piece = piece_from_law(compilation, LAW, BATCH, directory=directory)
    c_artifact = emit_ssa_module_to_c(piece.module, piece.entry)
    if not c_artifact.complete:
        print("FAIL C emission:", c_artifact.shortfalls[:3])
        return 1
    c_artifact = c_artifact.compile(directory / "c")
    print(f"ok   compiled {LAW}: slots="
          f"{sorted({row['external'] for row in piece.artifact.external_slots})}")

    # the live craft: Earth + Moon, low orbit, a constant raw applied force
    speed = math.sqrt(MU_EARTH / R_ORBIT)
    period = 2.0 * math.pi * math.sqrt(R_ORBIT**3 / MU_EARTH)
    jumper = OrbitalJumper(
        [GravityCenter((0.0, 0.0, 0.0), MU_EARTH), GravityCenter(MOON_M, MU_MOON)],
        mass_kg=MASS_KG, position_m=(R_ORBIT, 0.0, 0.0),
        velocity_m_s=(0.0, speed, 0.0), length_scale_m=5.0e4,
        window_s=period / 24)
    jumper.F(RAW_FORCE_N)
    trajectory = CraftTrajectory(jumper)
    for _ in range(6):
        jumper.advance()
        trajectory.record()
    times = trajectory.times
    print(f"ok   craft advanced to t={jumper.time_s:.3f} s over {len(times) - 1} rounds")

    lanes = np.array([times[0], times[2], times[4], times[-1]])
    centers = [(0.0, 0.0, 0.0), MOON_M]
    columns = {
        "s": lanes, "L": np.full(BATCH, times[-1]),
        "μ_1": np.full(BATCH, MU_EARTH), "μ_2": np.full(BATCH, MU_MOON),
    }
    for index, center in enumerate(centers, start=1):
        for row in range(3):
            columns[f"c_{index}_{row}_0"] = np.full(BATCH, center[row])
    for row in range(3):
        columns[f"r_start_{row}_0"] = np.full(BATCH, trajectory.rows["r"][0][row])
        columns[f"r_end_{row}_0"] = np.full(BATCH, trajectory.rows["r"][-1][row])
    missing = [name for name in arguments if name not in columns]
    if missing:
        print("FAIL no column for", missing)
        return 1

    externals = trajectory.externals()
    bind_external_slots(piece.artifact, externals)
    bind_external_slots(c_artifact, externals)
    llvm = dict(zip(piece.output_names, piece(*(columns[name] for name in arguments))))
    execution = c_artifact.prepare_execution(
        {value_id: columns[name] for name, value_id in zip(arguments, piece.argument_ids)})
    execution.run()
    c_out = {name: (execution.buffers[piece.output_ids[name]] if name in piece.output_ids
                    else np.full(BATCH, piece.constant_outputs[name]))
             for name in piece.output_names}

    # the set evaluated in numpy at the craft's recorded states
    r = np.array([[trajectory.lookup("r", k)(lanes)[i] for k in range(3)] for i in range(BATCH)])
    v = np.array([[trajectory.lookup("v", k)(lanes)[i] for k in range(3)] for i in range(BATCH)])
    a = np.array([[trajectory.lookup("a", k)(lanes)[i] for k in range(3)] for i in range(BATCH)])
    f = np.array([[trajectory.lookup("F", k)(lanes)[i] for k in range(3)] for i in range(BATCH)])
    grav = sum(-mu * (r - np.asarray(c)) / np.linalg.norm(r - np.asarray(c), axis=1)[:, None]**3
               for mu, c in ((MU_EARTH, centers[0]), (MU_MOON, centers[1])))
    energy = (0.5 * np.sum(v * v, axis=1)
              - sum(mu / np.linalg.norm(r - np.asarray(c), axis=1)
                    for mu, c in ((MU_EARTH, centers[0]), (MU_MOON, centers[1]))))

    failures = 0
    fuel = float(np.asarray(jumper.fuel_impulse_n_s).reshape(-1)[0])
    for lane, out in (("llvm", llvm), ("c", c_out)):
        print(f"--- {lane}")
        for row in rows:
            got = np.asarray(out[row.name], dtype=float)
            kind = kinds[row.equation_index]
            if kind == "motion":
                k = row.component[0]
                # relative to the acceleration's magnitude (a component can
                # be exactly zero: the orbit's first instant has a_y = 0)
                error = float(np.max(np.abs(got) / np.linalg.norm(a, axis=1)))
                text = "a - (F_grav1 + F_grav2 + F), over |a|"
            elif kind in {"initial", "terminal"}:
                error = float(np.max(np.abs(got)))
                text = "r(endpoint) - recorded endpoint"
            elif kind == "energy":
                error = float(np.max(np.abs(got - energy) / np.abs(energy)))
                text = (f"vs numpy at the craft states; craft drift over the run "
                        f"{abs(energy[-1] - energy[0]) / abs(energy[0]):.3e}")
            else:
                error = float(np.max(np.abs(got * MASS_KG - fuel)) / abs(fuel))
                text = f"m * cost vs the craft's fuel_impulse {fuel:.6e} N s"
            passed = error <= TOLERANCE      # NaN fails
            verdict = "ok  " if passed else "FAIL"
            failures += not passed
            print(f"{verdict} {row.name:34} max rel err {error:.3e}  ({text})")
    print(f"failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
